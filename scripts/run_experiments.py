"""End-to-end experiment driver: data → encodings → central/local DP → eval metrics.

Run from the repository root, e.g.:
``python scripts/run_experiments.py`` or
``python -m scripts.run_experiments`` (if ``pythonpath`` is set).
"""

from __future__ import annotations

import sys
from pathlib import Path

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
from src.evaluation.mia_lira import LiRAMembershipInference
from src.evaluation.utility import UtilityEvaluator
from src.models.central_dp import CentralDPMechanism
from src.models.local_dp import LocalDPProjector
from src.models.rag_baseline import RAGBaseline


def _encode_corpus(model: RAGBaseline, texts: list[str]) -> np.ndarray:
    """Helper: ST encoder, L2-norm, float32, shape ``(n, 384)``."""
    arr = model.model.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(arr, dtype=np.float32)


def main() -> None:
    """70/30 target vs. shadow, LiRA, inversion ROUGE, and BERTScore utility by ε."""
    loader = ChatDoctorLoader("data")
    all_texts = loader.load_data()
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
    e_sm = _encode_corpus(rag, sm_text)
    e_snm = _encode_corpus(rag, snm_text)
    e_tm = _encode_corpus(rag, tm_text)
    e_tnm = _encode_corpus(rag, tnm_text)

    target_embs = np.vstack([e_tm, e_tnm])
    target_labels = np.concatenate(
        [
            np.ones(e_tm.shape[0], dtype=np.int64),
            np.zeros(e_tnm.shape[0], dtype=np.int64),
        ]
    )
    all_clean = _encode_corpus(rag, all_texts)
    # Row-aligned texts for the global retrieval index.
    line_texts: list[str] = list(all_texts)

    lira = LiRAMembershipInference()
    lira.train_shadow_models(e_sm, e_snm)

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
    util = UtilityEvaluator()

    rows: list[dict[str, float | int | str]] = []
    epsilons = [0.1, 1.0, 5.0, 10.0]

    for eps in epsilons:
        central = CentralDPMechanism(epsilon=float(eps), delta=1e-5)
        noisy_c = np.asarray(
            central.apply_noise(np.asarray(target_embs, dtype=np.float64)),
            dtype=np.float32,
        )

        local = LocalDPProjector(
            input_dim=384, bottleneck_dim=16, epsilon=float(eps), delta=1e-5
        )
        local.eval()
        with torch.no_grad():
            t_in = torch.from_numpy(np.ascontiguousarray(target_embs, dtype=np.float32))
            noisy_l = local(t_in).numpy().astype(np.float32)

        tpr_c = lira.evaluate_tpr_at_fpr(
            np.asarray(noisy_c, dtype=np.float64), target_labels, 0.001
        )
        tpr_l = lira.evaluate_tpr_at_fpr(
            np.asarray(noisy_l, dtype=np.float64), target_labels, 0.001
        )

        rouge_c, rouge_l = [], []
        for j in sample_i:
            otxt = orig_by_row[j]
            rc = inv.nearest_neighbor_lookup(noisy_c[j], otxt)
            rl = inv.nearest_neighbor_lookup(noisy_l[j], otxt)
            rouge_c.append(float(rc["rouge_l_fmeasure"]))
            rouge_l.append(float(rl["rouge_l_fmeasure"]))
        mean_rouge_c = float(np.mean(rouge_c)) if rouge_c else 0.0
        mean_rouge_l = float(np.mean(rouge_l)) if rouge_l else 0.0

        # RAG sim: NNs from noisy query vectors to clean corpus, vs. same-line golds.
        cand_c, cand_l, ref_b = [], [], []
        for j in range(target_embs.shape[0]):
            otxt = orig_by_row[j]
            cand_c.append(
                inv.nearest_neighbor_lookup(noisy_c[j], otxt)["retrieved_text"]
            )
            cand_l.append(
                inv.nearest_neighbor_lookup(noisy_l[j], otxt)["retrieved_text"]
            )
            ref_b.append(otxt)
        b_c = util.compute_bertscore(references=ref_b, candidates=cand_c)
        b_l = util.compute_bertscore(references=ref_b, candidates=cand_l)

        rows.append(
            {
                "epsilon": eps,
                "tpr_mia_central": tpr_c,
                "tpr_mia_local": tpr_l,
                "inversion_rouge_l_mean_central": mean_rouge_c,
                "inversion_rouge_l_mean_local": mean_rouge_l,
                "utility_bert_precision_central": b_c["precision"],
                "utility_bert_recall_central": b_c["recall"],
                "utility_bert_f1_central": b_c["f1"],
                "utility_bert_precision_local": b_l["precision"],
                "utility_bert_recall_local": b_l["recall"],
                "utility_bert_f1_local": b_l["f1"],
            }
        )

    out_dir = _ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_path = out_dir / "results.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(df)} rows.")


if __name__ == "__main__":
    main()
