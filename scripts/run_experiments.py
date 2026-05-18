"""End-to-end experiment driver: data → encodings → central/local DP → eval metrics.

Run from the repository root, e.g.:
``python scripts/run_experiments.py`` or
``python -m scripts.run_experiments`` (if ``pythonpath`` is set).
"""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/experiment.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# Allow ``python scripts/run_experiments.py`` without a prior PYTHONPATH=.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.data_loader import ChatDoctorLoader
from src.evaluation.inversion import EmbeddingInversion
from src.evaluation.inversion_probe import LinearProbeInversion
from src.evaluation.mia_lira import LiRAMembershipInference
from src.evaluation.utility import UtilityEvaluator
from src.models.central_dp import CentralDPMechanism
from src.models.local_dp import LocalDPProjector
from src.models.metric_dp import MetricDPMechanism
from src.models.rag_baseline import RAGBaseline

# Sprint 1: repeat the stochastic DP mechanisms to estimate mean ± std curves.
N_RUNS = 5
EPSILONS = [0.1, 1.0, 5.0, 10.0]


def _seed_everything(seed: int) -> None:
    """Reset all RNGs used by this script for one reproducible realization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _encode_corpus(model: RAGBaseline, texts: list[str], label: str) -> np.ndarray:
    """Helper: ST encoder, L2-norm, float32, shape ``(n, 384)``."""
    log.info("Encoding start | phase=%s | texts=%d", label, len(texts))
    arr = model.model.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    log.info("Encoding finished | phase=%s | texts=%d", label, len(texts))
    return np.asarray(arr, dtype=np.float32)


def _eval_mechanism(
    embs: np.ndarray,
    target_labels: np.ndarray,
    orig_by_row: list[str],
    sample_i: np.ndarray,
    lira: LiRAMembershipInference,
    inv: EmbeddingInversion,
    probe: LinearProbeInversion,
    util: UtilityEvaluator,
    label: str,
) -> dict[str, float]:
    """Run MIA, inversion, and BERTScore evaluation on a batch of embeddings.

    Args:
        embs: ``(N, 384)`` possibly privatised embeddings to evaluate.
        target_labels: ``(N,)`` member (1) / non-member (0) labels.
        orig_by_row: Gold text strings aligned with ``embs`` rows.
        sample_i: Row indices to use for ROUGE-L inversion (fixed 100-sample subset).
        lira: Trained ``LiRAMembershipInference`` instance.
        inv: Built ``EmbeddingInversion`` index.
        probe: Fitted ``LinearProbeInversion`` attack.
        util: ``UtilityEvaluator`` instance.

    Returns:
        Dict with keys ``tpr_mia``, ``inversion_rouge_l_mean``,
        ``probe_rouge_l_mean``, ``bert_precision``, ``bert_recall``, ``bert_f1``.
    """
    tpr = lira.evaluate_tpr_at_fpr(
        np.asarray(embs, dtype=np.float64), target_labels, 0.001
    )

    rouge_scores = []
    probe_scores = []
    for j in sample_i:
        rc = inv.nearest_neighbor_lookup(embs[j], orig_by_row[j])
        rouge_scores.append(float(rc["rouge_l_fmeasure"]))

        ps = probe.score(embs[j], orig_by_row[j])
        probe_scores.append(float(ps["rouge_l_fmeasure"]))

    mean_rouge = float(np.mean(rouge_scores)) if rouge_scores else 0.0
    mean_probe_rouge = float(np.mean(probe_scores)) if probe_scores else 0.0

    cands, refs = [], []
    for j in range(embs.shape[0]):
        cands.append(
            inv.nearest_neighbor_lookup(embs[j], orig_by_row[j])["retrieved_text"]
        )
        refs.append(orig_by_row[j])
    log.info("BERTScore starting — this may take several minutes on CPU | %s", label)
    bs = util.compute_bertscore(references=refs, candidates=cands)
    log.info("BERTScore finished | %s", label)

    return {
        "tpr_mia": tpr,
        "inversion_rouge_l_mean": mean_rouge,
        "probe_rouge_l_mean": mean_probe_rouge,
        "bert_precision": bs["precision"],
        "bert_recall": bs["recall"],
        "bert_f1": bs["f1"],
    }


def main() -> None:
    """70/30 target vs. shadow, LiRA, inversion ROUGE, and BERTScore utility by ε."""
    start_time = time.time()
    # Global seeds for deterministic setup before per-run stochastic DP draws.
    _seed_everything(0)

    log.info("Experiment script starting")
    loader = ChatDoctorLoader("data")
    all_texts = loader.load_data()
    initial_corpus_size = len(all_texts)
    log.info("Corpus loaded | texts_before_cap=%d", initial_corpus_size)
    # Cap corpus size for CPU-feasible runtime; sufficient for stable LiRA statistics
    MAX_CORPUS = 5000
    if len(all_texts) > MAX_CORPUS:
        rng = random.Random(42)
        all_texts = rng.sample(all_texts, MAX_CORPUS)
    log.info(
        "Corpus ready | texts_after_cap=%d | max_corpus=%d",
        len(all_texts),
        MAX_CORPUS,
    )
    n = len(all_texts)
    if n < 4:
        msg = f"Need at least 4 lines for MIA splits; got {n}."
        raise RuntimeError(msg)

    # 70% target (attack evaluation), 30% shadow (fit shadow LR + null moments).
    target_texts, shadow_texts = train_test_split(
        all_texts,
        test_size=0.3,
        random_state=42,
        shuffle=True,
    )
    # Within each side, 50/50 member vs. non-member labels (text-only proxy split).
    sm_text, snm_text = train_test_split(
        shadow_texts, test_size=0.5, random_state=11, shuffle=True
    )
    tm_text, tnm_text = train_test_split(
        target_texts, test_size=0.5, random_state=22, shuffle=True
    )

    rag = RAGBaseline()
    e_sm = _encode_corpus(rag, sm_text, "shadow members")
    e_snm = _encode_corpus(rag, snm_text, "shadow non-members")
    e_tm = _encode_corpus(rag, tm_text, "target members")
    e_tnm = _encode_corpus(rag, tnm_text, "target non-members")

    target_embs = np.vstack([e_tm, e_tnm])
    target_labels = np.concatenate(
        [
            np.ones(e_tm.shape[0], dtype=np.int64),
            np.zeros(e_tnm.shape[0], dtype=np.int64),
        ]
    )
    all_clean = _encode_corpus(rag, all_texts, "clean reference corpus")
    # Row-aligned texts for the global retrieval index.
    line_texts: list[str] = list(all_texts)

    lira = LiRAMembershipInference(n_shadow_models=16)
    log.info("LiRA shadow training starting | shadow_models=%d", lira.n_shadow_models)
    lira.train_shadow_models(e_sm, e_snm)
    log.info("LiRA shadow training finished | shadow_models=%d", lira.n_shadow_models)

    # Fixed 100 target rows to compare ROUGE across ε without resampling noise.
    rng = np.random.default_rng(42)
    sample_i = rng.choice(
        target_embs.shape[0], size=min(100, target_embs.shape[0]), replace=False
    )
    # Original (clean) string per target row, same vstack order as ``target_embs``.
    text_tm = list(tm_text)
    text_tnm = list(tnm_text)
    orig_by_row = text_tm + text_tnm

    inv = EmbeddingInversion(
        line_texts, np.ascontiguousarray(all_clean, dtype=np.float32)
    )
    log.info("Fitting linear probe inversion attack on clean corpus")
    probe = LinearProbeInversion(max_features=2000, top_k_tokens=20)
    probe.fit(line_texts, np.ascontiguousarray(all_clean, dtype=np.float32))
    log.info("Linear probe fitting complete")
    log.info(
        "Loading BERTScore model (roberta-large) — "
        "one-time load, this may take a few minutes"
    )
    util = UtilityEvaluator()

    rows: list[dict[str, float | str]] = []

    # --- Baseline: unperturbed embeddings (epsilon = inf) ---
    log.info("Baseline evaluation starting")
    baseline_embs = np.copy(target_embs)
    baseline_metrics = _eval_mechanism(
        baseline_embs,
        target_labels,
        orig_by_row,
        sample_i,
        lira,
        inv,
        probe,
        util,
        "Baseline",
    )
    rows.append(
        {
            "run_id": 0,
            "epsilon": float("inf"),
            "mechanism": "Baseline",
            **baseline_metrics,
        }
    )
    log.info("Baseline evaluation finished")

    # --- Privacy sweep: multiple stochastic realizations per mechanism/epsilon. ---
    for run_id in range(N_RUNS):
        _seed_everything(run_id)
        log.info("Starting DP realization | run=%d/%d", run_id + 1, N_RUNS)

        for eps in EPSILONS:
            # Central DP
            log.info(
                "Evaluation starting | run_id=%d | epsilon=%s | mechanism=Central",
                run_id,
                eps,
            )
            central = CentralDPMechanism(epsilon=float(eps), delta=1e-5)
            noisy_c = np.asarray(
                central.apply_noise(np.asarray(target_embs, dtype=np.float64)),
                dtype=np.float32,
            )
            central_metrics = _eval_mechanism(
                noisy_c,
                target_labels,
                orig_by_row,
                sample_i,
                lira,
                inv,
                probe,
                util,
                f"run_id={run_id} epsilon={eps} mechanism=Central",
            )
            rows.append(
                {
                    "run_id": run_id,
                    "epsilon": eps,
                    "mechanism": "Central",
                    **central_metrics,
                }
            )
            log.info(
                "Evaluation finished | run_id=%d | epsilon=%s | mechanism=Central",
                run_id,
                eps,
            )

            # Local DP
            log.info(
                "Evaluation starting | run_id=%d | epsilon=%s | mechanism=Local",
                run_id,
                eps,
            )
            local = LocalDPProjector(
                input_dim=384, bottleneck_dim=16, epsilon=float(eps), delta=1e-5
            )
            local.eval()
            with torch.no_grad():
                t_in = torch.from_numpy(
                    np.ascontiguousarray(target_embs, dtype=np.float32)
                )
                noisy_l = local(t_in).numpy().astype(np.float32)
            local_metrics = _eval_mechanism(
                noisy_l,
                target_labels,
                orig_by_row,
                sample_i,
                lira,
                inv,
                probe,
                util,
                f"run_id={run_id} epsilon={eps} mechanism=Local",
            )
            rows.append(
                {
                    "run_id": run_id,
                    "epsilon": eps,
                    "mechanism": "Local",
                    **local_metrics,
                }
            )
            log.info(
                "Evaluation finished | run_id=%d | epsilon=%s | mechanism=Local",
                run_id,
                eps,
            )

            # Metric DP
            log.info(
                "Evaluation starting | run_id=%d | epsilon=%s | mechanism=Metric",
                run_id,
                eps,
            )
            metric = MetricDPMechanism(epsilon=float(eps), delta=1e-5, k=20)
            noisy_m = np.asarray(
                metric.apply_noise(
                    np.asarray(target_embs, dtype=np.float64),
                    np.asarray(all_clean, dtype=np.float64),
                ),
                dtype=np.float32,
            )
            metric_metrics = _eval_mechanism(
                noisy_m,
                target_labels,
                orig_by_row,
                sample_i,
                lira,
                inv,
                probe,
                util,
                f"run_id={run_id} epsilon={eps} mechanism=Metric",
            )
            rows.append(
                {
                    "run_id": run_id,
                    "epsilon": eps,
                    "mechanism": "Metric",
                    **metric_metrics,
                }
            )
            log.info(
                "Evaluation finished | run_id=%d | epsilon=%s | mechanism=Metric",
                run_id,
                eps,
            )

    out_dir = _ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # Canonical column order.
    df = df[
        [
            "run_id",
            "epsilon",
            "mechanism",
            "tpr_mia",
            "inversion_rouge_l_mean",
            "probe_rouge_l_mean",
            "bert_precision",
            "bert_recall",
            "bert_f1",
        ]
    ]
    out_path = out_dir / "results.csv"
    df.to_csv(out_path, index=False)
    log.info("Results written | path=%s | rows=%d", out_path.resolve(), len(df))
    elapsed = time.time() - start_time
    elapsed_minutes = int(elapsed // 60)
    elapsed_seconds = int(elapsed % 60)
    log.info(
        "Experiment script finished | elapsed=%dm %02ds",
        elapsed_minutes,
        elapsed_seconds,
    )


if __name__ == "__main__":
    main()
