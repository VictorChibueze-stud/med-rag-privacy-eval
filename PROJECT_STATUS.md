# Project Status — med-rag-privacy-eval
**Course:** Privacy-Preserving Methods for Data Science
**Project:** Evaluating Central vs. Local Differential Privacy for Vector Embeddings in Medical RAG Systems
**Team:** Victor Okoroafor, Ifeanyi Omonigho Odugo, Gopal Krishna, Niramay Roopesh Kolalle
**Last updated:** 2026-05-11

---

## Overview

This document is the single source of truth for the project. Read it fully before
touching any file. It describes the exact current state of the codebase, what has
been completed and verified, and the full sprint plan through final submission.

Work is organised into **tickets**. Each ticket belongs to a sprint. Teammates can
self-assign tickets by adding their name to the ticket header. Do not start a ticket
without reading the guardrails at the bottom of this document.

---

## Current Completion: ~55%

| Component | Status |
|---|---|
| Central DP mechanism | ✅ Complete and verified |
| Local DP mechanism | ✅ Complete and verified |
| MIA (LiRA, 16 models) | ✅ Complete and verified |
| Embedding inversion (nearest-neighbour) | ✅ Complete |
| BERTScore utility | ✅ Complete |
| Experiment orchestration | ✅ Complete with baseline, seeds, logging, corpus cap |
| Plotting engine | ✅ Complete, reads long-format CSV |
| Test suite (5 tests) | ✅ All passing |
| Manuscript skeleton | ✅ Scaffolded — Methodology and Evaluation Framework have real content |
| Data loader (ChatDoctor field mapping) | ✅ Fixed and verified — 112,165 rows load correctly |
| **Single-realisation results.csv** | ✅ Generated — 9 rows, real data |
| **Three baseline figures** | ✅ Generated — utility, MIA, inversion |
| **Multiple realisations + error bars** | ❌ Sprint 1 |
| **Metric DP (third mechanism)** | ❌ Sprint 2 |
| **Stronger inversion attack** | ❌ Sprint 3 |
| **Results section** | ❌ Sprint 4 |
| **Discussion section** | ❌ Sprint 4 |
| **Abstract (final)** | ❌ Sprint 4 |
| **Introduction** | ❌ Sprint 4 |
| **Bibliography** | ❌ Sprint 4 |
| **Final PDF compilation** | ❌ Sprint 5 |

---

## What the Experiment Produced (Sprint 0 Results)

The single-realisation experiment ran successfully on 5,000 rows of the
ChatDoctor-HealthCareMagic-100k dataset. Results are in `data/results.csv`.

| epsilon | mechanism | tpr_mia | inversion_rouge_l_mean | bert_f1 |
|---|---|---|---|---|
| inf | Baseline | 0.002286 | 1.000000 | 1.000000 |
| 0.1 | Central | 0.000000 | 0.213207 | 0.831245 |
| 0.1 | Local | 0.002286 | 0.207698 | 0.831108 |
| 1.0 | Central | 0.000000 | 0.204211 | 0.831120 |
| 1.0 | Local | 0.001714 | 0.210376 | 0.831017 |
| 5.0 | Central | 0.000000 | 0.214474 | 0.831684 |
| 5.0 | Local | 0.001143 | 0.213283 | 0.830659 |
| 10.0 | Central | 0.000000 | 0.210860 | 0.832781 |
| 10.0 | Local | 0.001714 | 0.204862 | 0.829386 |

**Key findings from these numbers:**

- **Inversion:** Both mechanisms drop ROUGE-L from 1.0 to ~0.21 at every epsilon
  value. The cliff is immediate and epsilon-invariant — DP noise completely destroys
  nearest-neighbour reconstruction regardless of privacy budget.
- **MIA:** Central DP achieves TPR = 0.000 at every epsilon. Local DP shows small
  non-zero TPR (0.001–0.002) with non-monotonic behaviour. Both are near-zero.
- **Utility:** BERTScore F1 stays between 0.829–0.833 across all conditions. Less
  than 2% degradation even at epsilon=0.1.
- **Flat curves:** No epsilon-dependent tradeoff is visible in any metric. This is
  a novel finding — no paper in our reference corpus reports epsilon-invariant curves.
  The hypothesis is encoder-level saturation: once noise exceeds the local
  neighbourhood radius, all three metrics saturate simultaneously.

**Important caveat (Carlini et al. 2022):** Near-zero TPR does not prove privacy.
It may indicate the attack is underpowered. This must be addressed explicitly in
the Discussion section.

---

## Sprint Plan

Sprints must be executed in dependency order. Sprint 4 writing can begin partially
in parallel with Sprints 1–3. Sprint 5 requires all prior sprints complete.

```
Sprint 0 (done) → Sprint 1 → Sprint 2 ─┐
                           → Sprint 3 ─┤→ Sprint 4 → Sprint 5
```

---

## SPRINT 1 — Multiple Realisations + Error Bars
**Goal:** Replace single-point estimates with statistically robust curves.
**Depends on:** Sprint 0 (complete)
**Blocks:** Sprint 4 Results section (needs mean ± std numbers)
**Estimated runtime:** ~5 hours on CPU — run overnight
**Files touched:** `scripts/run_experiments.py`, `scripts/plot_results.py`

### Why this is needed
Each current data point is a single noise draw. The non-monotonic behaviour in
Local DP MIA (TPR dips at epsilon=5 then rises at epsilon=10) may be noise
variance, not a real signal. Carlini et al. (2022) note that near-zero TPR
can indicate an underpowered attack — multiple realisations with error bars
is the standard response. Bollegala et al. (2025) explicitly run multiple
realisations for this reason.

### TICKET S1-A — Multi-realisation experiment loop
**Assignee:** _______________
**Effort:** Medium — 1–2 hours of coding, ~5 hours of compute

**What to implement in `scripts/run_experiments.py`:**

Add a constant near the top of `main()`:
```python
N_RUNS = 5
```

Wrap the entire existing epsilon loop (from `for eps in epsilons:` to the
end of the Local DP block) in an outer loop:
```python
for run_id in range(N_RUNS):
    np.random.seed(run_id)
    torch.manual_seed(run_id)
    log.info("Starting run %d of %d", run_id + 1, N_RUNS)
    for eps in epsilons:
        ... (existing Central and Local DP blocks, unchanged) ...
        rows.append({"run_id": run_id, "epsilon": eps, "mechanism": "Central", **central_metrics})
        rows.append({"run_id": run_id, "epsilon": eps, "mechanism": "Local", **local_metrics})
```

Also add `"run_id": 0` to the baseline row (baseline is deterministic, run once).

The results.csv schema gains one column: `run_id`. Rows grow from 9 to 45
(1 baseline + 4 epsilon × 2 mechanisms × 5 runs).

**Verification:** After running, confirm `data/results.csv` has 45 rows and
that the `run_id` column contains values 0–4.

---

### TICKET S1-B — Error band plots
**Assignee:** _______________
**Effort:** Small — 1 hour

**What to implement in `scripts/plot_results.py`:**

In `_lineplot_save`, change the seaborn call to aggregate across run_ids:
```python
sns.lineplot(
    data=d,
    x="epsilon",
    y=ycol,
    hue="mechanism",
    style="mechanism",
    markers=True,
    dashes=False,
    err_style="band",       # was None
    errorbar="sd",          # show ±1 std dev band
    ax=ax,
)
```

This works automatically because seaborn's `lineplot` groups by x-value and
computes mean ± std when multiple rows share the same (epsilon, mechanism).
No other changes needed — the plotter already handles arbitrary row counts.

**Verification:** Re-run `python scripts/plot_results.py` after S1-A completes.
Each curve should show a shaded band around the line. If the band is
invisible (all realisations identical), something is wrong with the seed
resetting in S1-A.

**Dependency:** S1-B cannot be run until S1-A produces the new results.csv.

---

## SPRINT 2 — Metric DP (Third Mechanism)
**Goal:** Add Mahalanobis-noise Metric DP as a third comparison curve.
**Depends on:** Sprint 1 (needs stable multi-run infrastructure)
**Blocks:** Sprint 4 Discussion (needs three-way comparison)
**Files touched:** `src/models/metric_dp.py` (new), `scripts/run_experiments.py`

### Why this is needed
Bollegala et al. (2025) define Metric DP using the Mahalanobis distance and
the Analytical Gaussian Mechanism, applying noise scaled to the local
covariance structure of the embedding neighbourhood rather than a global
worst-case sensitivity. We are the first paper to compare Central, Local, and
Metric DP on the same RAG embedding evaluation. This is the primary
contribution that differentiates this work from existing literature.

Bollegala et al. do **not** compare against Central DP in their plots —
meaning our three-way comparison is a novel result regardless of what the
numbers show.

### TICKET S2-A — Implement MetricDPMechanism
**Assignee:** _______________
**Effort:** Large — 3–4 hours

**Create `src/models/metric_dp.py`** with the following class:

```python
"""Metric DP via Mahalanobis-scaled Analytical Gaussian Mechanism.

Based on Bollegala et al. (2025) CMAG: noise is scaled to the local
covariance of the embedding neighbourhood rather than a global L2 sensitivity.

Privacy guarantee: (epsilon, delta)-Metric DP where the distance function
is the Mahalanobis distance computed from the local neighbourhood covariance.
"""
from __future__ import annotations
import numpy as np


class MetricDPMechanism:
    """Apply Metric DP noise scaled to local Mahalanobis geometry.

    For each embedding x, estimates the local covariance from its k nearest
    neighbours in the corpus, then applies Analytical Gaussian noise scaled
    to the Mahalanobis sensitivity in that local geometry.

    Attributes:
        epsilon: Privacy budget.
        delta: Privacy parameter delta, default 1e-5.
        k: Number of neighbours for local covariance estimation, default 50.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float = 1e-5,
        k: int = 50,
    ) -> None:
        if epsilon <= 0.0 or delta <= 0.0 or delta >= 1.0:
            raise ValueError("Need epsilon > 0 and 0 < delta < 1.")
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.k = int(k)

    def apply_noise(
        self,
        embeddings: np.ndarray,
        corpus_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Add Mahalanobis-calibrated Gaussian noise to each embedding.

        Args:
            embeddings: (n, d) target embeddings to privatise.
            corpus_embeddings: (N, d) reference corpus for neighbourhood
                covariance estimation. Typically the full clean corpus.

        Returns:
            Noised embeddings of shape (n, d), same dtype as input.
        """
        X = np.asarray(embeddings, dtype=np.float64)
        C = np.asarray(corpus_embeddings, dtype=np.float64)
        n, d = X.shape
        sigma_scalar = (
            2.0 * np.sqrt(2.0 * np.log(1.25 / self.delta)) / self.epsilon
        )
        result = np.empty_like(X)
        for i in range(n):
            # Compute L2 distances to all corpus embeddings.
            diffs = C - X[i]
            dists = np.einsum("nd,nd->n", diffs, diffs)
            # k nearest neighbours (excluding self if present).
            idx = np.argpartition(dists, min(self.k, len(dists) - 1))[: self.k]
            neighbours = C[idx]
            # Local covariance from neighbourhood.
            cov = np.cov(neighbours.T) + 1e-6 * np.eye(d)
            # Cholesky factor for correlated noise draw.
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                # Fallback to isotropic if covariance is not PD.
                L = np.eye(d)
            # Correlated Gaussian noise scaled by sigma_scalar.
            z = np.random.normal(size=d)
            noise = sigma_scalar * L @ z
            result[i] = X[i] + noise
        return result
```

**Unit test** — add to `tests/test_privacy_math.py`:
```python
def test_metric_dp_output_shape():
    from src.models.metric_dp import MetricDPMechanism
    mech = MetricDPMechanism(epsilon=1.0, delta=1e-5, k=10)
    corpus = np.random.randn(100, 384).astype(np.float64)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    targets = corpus[:5]
    out = mech.apply_noise(targets, corpus)
    assert out.shape == targets.shape
```

**Warning:** The per-embedding neighbourhood search is O(n × N). With
n=3,500 target embeddings and N=5,000 corpus embeddings it will be slow
on CPU — expect 10–20 minutes per epsilon value. Use `k=20` if too slow.

---

### TICKET S2-B — Integrate MetricDP into experiment loop
**Assignee:** _______________
**Effort:** Small — 30 minutes
**Dependency:** S2-A must be complete first.

In `scripts/run_experiments.py`, inside the `for run_id` and `for eps` loops,
add a Metric DP block after the existing Local DP block:

```python
# Metric DP
log.info("Evaluation starting | epsilon=%s | mechanism=Metric", eps)
from src.models.metric_dp import MetricDPMechanism
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
    util,
    f"epsilon={eps} mechanism=Metric",
)
rows.append({"run_id": run_id, "epsilon": eps, "mechanism": "Metric", **metric_metrics})
log.info("Evaluation finished | epsilon=%s | mechanism=Metric", eps)
```

results.csv grows from 45 rows to 65 rows
(1 baseline + 4 epsilon × 3 mechanisms × 5 runs).

The plotting script requires no changes — it handles arbitrary mechanisms.

---

## SPRINT 3 — Stronger Inversion Attack
**Goal:** Replace nearest-neighbour ROUGE-L with a trained linear probe,
closing the gap to Vec2Text and strengthening privacy claims.
**Depends on:** Sprint 0 (can run in parallel with Sprint 2)
**Blocks:** Sprint 4 Discussion
**Files touched:** `src/evaluation/inversion_probe.py` (new),
`scripts/run_experiments.py`

### Why this is needed
Morris et al. (2023) show that Vec2Text reconstructs 89% of patient names and
26% of full clinical documents from MIMIC-III embeddings. Our current
nearest-neighbour baseline is explicitly weaker than Vec2Text — meaning our
ROUGE-L ~0.21 **underestimates** true inversion risk. A linear probe is a
reasonable middle ground: stronger than nearest-neighbour, computationally
feasible, and honest about the threat level.

From the Limitations section already in main.tex:
> "This is a nearest-neighbour approximation of a full inversion attack.
> It underestimates the true inversion risk but is computationally feasible."

Sprint 3 partially addresses this acknowledged limitation.

### TICKET S3-A — Implement linear probe inversion
**Assignee:** _______________
**Effort:** Medium — 2–3 hours

**Create `src/evaluation/inversion_probe.py`**:

```python
"""Linear probe embedding inversion attack.

Trains a linear decoder from embedding space to a bag-of-words token
representation, then decodes noisy embeddings and measures reconstruction
quality via ROUGE-L. Stronger than nearest-neighbour but weaker than Vec2Text.
"""
from __future__ import annotations
import numpy as np
from rouge_score import rouge_scorer
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer


class LinearProbeInversion:
    """Train a linear map from embeddings to BoW, decode, score with ROUGE-L.

    Attributes:
        vectorizer: Fitted CountVectorizer over the training corpus.
        probe: Fitted Ridge regression from embeddings to BoW vectors.
        rouge: ROUGE-L scorer.
    """

    def __init__(self, max_features: int = 2000) -> None:
        self.max_features = max_features
        self.vectorizer = CountVectorizer(
            max_features=max_features, binary=True
        )
        self.probe = Ridge(alpha=1.0)
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self._corpus: list[str] = []

    def fit(
        self,
        corpus_texts: list[str],
        corpus_embeddings: np.ndarray,
    ) -> None:
        """Fit the vectorizer and probe on the clean reference corpus.

        Args:
            corpus_texts: Raw text strings aligned with corpus_embeddings.
            corpus_embeddings: (N, d) clean embeddings for training.
        """
        self._corpus = list(corpus_texts)
        bow = self.vectorizer.fit_transform(corpus_texts).toarray().astype(np.float32)
        X = np.asarray(corpus_embeddings, dtype=np.float32)
        self.probe.fit(X, bow)

    def reconstruct(self, embedding: np.ndarray) -> str:
        """Decode one embedding to text via BoW prediction + vocabulary lookup.

        Args:
            embedding: (d,) possibly noisy embedding vector.

        Returns:
            Reconstructed text string (space-joined top predicted tokens).
        """
        e = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        bow_pred = self.probe.predict(e)[0]
        vocab = self.vectorizer.get_feature_names_out()
        # Take top-20 predicted tokens as the reconstruction.
        top_idx = np.argsort(bow_pred)[::-1][:20]
        return " ".join(vocab[i] for i in top_idx)

    def score(
        self, embedding: np.ndarray, original_text: str
    ) -> dict[str, float]:
        """Reconstruct and compute ROUGE-L against original_text.

        Args:
            embedding: (d,) possibly noisy embedding to invert.
            original_text: Gold text for this sample.

        Returns:
            Dict with key rouge_l_fmeasure in [0, 1].
        """
        recon = self.reconstruct(embedding)
        s = self.rouge.score(original_text, recon)
        return {"rouge_l_fmeasure": float(s["rougeL"].fmeasure)}
```

### TICKET S3-B — Integrate probe into experiment loop
**Assignee:** _______________
**Effort:** Small — 30 minutes
**Dependency:** S3-A must be complete first.

In `scripts/run_experiments.py`, after the `EmbeddingInversion` is built, add:

```python
from src.evaluation.inversion_probe import LinearProbeInversion
log.info("Fitting linear probe inversion attack on clean corpus")
probe = LinearProbeInversion(max_features=2000)
probe.fit(line_texts, all_clean)
log.info("Linear probe fitting complete")
```

In `_eval_mechanism`, add a probe score loop alongside the existing ROUGE loop:
```python
probe_scores = []
for j in sample_i:
    ps = probe.score(embs[j], orig_by_row[j])
    probe_scores.append(float(ps["rouge_l_fmeasure"]))
mean_probe_rouge = float(np.mean(probe_scores)) if probe_scores else 0.0
```

Add `"probe_rouge_l_mean": mean_probe_rouge` to the returned dict and to
the results.csv column list in the orchestration script.

The plot script will need one new call in `main()` for the probe figure:
```python
_lineplot_save(
    df,
    ycol="probe_rouge_l_mean",
    ylabel="ROUGE-L (linear probe reconstruction)",
    title="Linear Probe Inversion Fidelity under DP",
    out_name="probe_inversion_vs_epsilon.png",
)
```

---

## SPRINT 4 — Paper Writing
**Goal:** Complete all empty sections of main.tex with real numbers.
**Depends on:** Sprint 1 (for mean ± std), Sprint 2 and 3 (for Discussion)
**Blocks:** Sprint 5
**Files touched:** `docs/main.tex` only

**Introduction and Methodology polish can begin NOW in parallel with Sprints 1–3.**
Results and Discussion must wait for the data.

### TICKET S4-A — Introduction
**Assignee:** Niramay
**Effort:** Medium — 2–3 hours
**Can start:** Immediately

Write the Introduction section in `docs/main.tex`. Cover:

1. Why medical RAG systems handle sensitive data — patient consultation logs,
   clinical records, diagnosis text. Cite Zeng et al. (2024) for the threat
   landscape and Morris et al. (2023) for the Vec2Text inversion result
   (89% of patient names reconstructed from embeddings on MIMIC-III).

2. Why embedding-level privacy is the right intervention point — the
   embedding is the surface exposed to the retrieval pipeline, and DP at
   this layer protects without requiring architectural changes to the LLM.

3. What we contribute: the first three-way comparison of Central, Local,
   and Metric DP on a medical RAG embedding benchmark, evaluated on three
   complementary axes (MIA resistance, inversion fidelity, semantic utility).

4. One sentence noting the surprising flat-curve finding to motivate the paper.

Keep to ~400 words.

---

### TICKET S4-B — Results Section
**Assignee:** Niramay
**Effort:** Medium — 2 hours
**Dependency:** Sprint 1 must be complete (needs mean ± std numbers)

Write the Results section in `docs/main.tex`. Structure:

**Paragraph 1 — BERTScore Utility:**
Report baseline F1 = 1.000, then the mean ± std F1 at each epsilon for
both Central and Local DP (and Metric DP if Sprint 2 is complete).
Reference `\ref{fig:util}`. Note the epsilon-invariance.

**Paragraph 2 — MIA TPR:**
Report baseline TPR = 0.002, then mean ± std TPR at each epsilon for
both mechanisms. Central DP: TPR = 0.000 at all epsilon. Local DP:
small non-zero values with non-monotonic pattern. Reference `\ref{fig:mia}`.

**Paragraph 3 — Inversion ROUGE-L:**
Report baseline ROUGE-L = 1.000, then mean ± std at each epsilon.
Both mechanisms: ~0.21 at all epsilon. Reference `\ref{fig:inv}`.
If Sprint 3 is complete, also report linear probe scores and compare.

Use actual numbers from results.csv — do not invent or estimate.

---

### TICKET S4-C — Discussion and Analysis
**Assignee:** Victor + Niramay
**Effort:** Large — 3–4 hours
**Dependency:** Sprints 1, 2, and 3 should be complete

Write the Discussion section in `docs/main.tex`. Cover these points in order:

1. **The flat curves.** Interpret the epsilon-invariance finding. The
   hypothesis: once Gaussian noise magnitude exceeds the local neighbourhood
   radius of the all-MiniLM-L6-v2 encoder, retrieval degrades to a fixed
   floor and privacy saturates simultaneously. This is consistent with the
   geometry of unit-sphere embeddings where most mass is concentrated in a
   thin shell. Connect to Du et al. (2023) who note that 16-d projection
   already suffices for privacy — our results suggest even the 384-d
   uncompressed space is similarly threshold-governed.

2. **Central vs Local DP.** Central DP achieves strictly lower TPR at
   every epsilon. Explain why: Central DP applies isotropic noise in the
   full 384-d space, while Local DP's bottleneck projection introduces
   variance not fully controlled by epsilon. The non-monotonic Local DP
   MIA curve is consistent with this — it is single-realisation noise,
   not a real privacy property. (Sprint 1 error bars will confirm or deny
   this interpretation.)

3. **The underpowered attack caveat.** Explicitly cite Carlini et al. (2022):
   near-zero TPR does not prove privacy — it may indicate the LiRA
   adaptation to embedding space is underpowered. Our LiRA uses logistic
   regression shadow classifiers on raw embeddings, which is not the
   native setting for LiRA. Report this honestly and note it as a direction
   for future work.

4. **Clinical translation.** At epsilon=1.0, Central DP TPR@0.1%FPR = 0.000.
   Translate: an adversary issuing 1,000 membership queries against a
   deployed medical RAG system protected by Central DP at epsilon=1.0 would
   correctly identify zero patient records with near-certainty of no false
   alarm. Compare this to the baseline (TPR=0.002): at baseline, 2 out of
   1,000 queries succeed.

5. **Metric DP comparison** (if Sprint 2 complete): Does local covariance
   structure improve the tradeoff? Where does Metric DP sit relative to
   Central and Local DP on all three axes?

---

### TICKET S4-D — Abstract, Limitations updates, Bibliography
**Assignee:** Niramay
**Effort:** Small-Medium — 1–2 hours
**Dependency:** S4-B and S4-C must be drafted first

1. **Abstract:** 150 words maximum. Three quantitative claims:
   (a) inversion ROUGE-L drops from 1.0 to ~0.21 under both mechanisms,
   (b) Central DP achieves TPR@0.1%FPR = 0.000 at all epsilon,
   (c) BERTScore F1 remains above 0.83 — less than 2% degradation.
   End with one sentence on the flat-curve finding.

2. **Limitations updates:** Fill the two remaining TODO stubs in the
   Evaluation Framework section (Embedding Inversion and BERTScore Utility
   subsections). Keep each to 2–3 sentences matching the existing style.

3. **Bibliography:** Add BibTeX entries in `docs/main.tex` for all 9 papers:
   - Dwork & Roth (2014) — Algorithmic Foundations of Differential Privacy
   - Carlini et al. (2022) — Membership Inference Attacks From First Principles
   - Du et al. (2023) — Sanitizing Sentence Embeddings for LDP
   - Bollegala et al. (2025) — CMAG
   - Morris et al. (2023) — Text Embeddings Reveal Almost As Much As Text
   - Zeng et al. (2024) — The Good and The Bad: Privacy Issues in RAG
   - Anderson et al. (2025) — MIA Against RAG
   - Shokri et al. (2017) — Membership Inference Attacks Against ML Models
   - Zhang et al. (2020) — BERTScore

---

## SPRINT 5 — Final Assembly and Submission
**Goal:** Compile clean PDF, verify all figures render, check page limit.
**Depends on:** All prior sprints complete
**Files touched:** `docs/main.tex`, `docs/figures/`

### TICKET S5-A — Final compilation and review
**Assignee:** All team members
**Effort:** Small — 1 hour

Checklist:
- [ ] Compile `docs/main.tex` with `pdflatex` — zero errors, zero missing references
- [ ] All five figures render correctly in the compiled PDF:
      `utility_vs_epsilon.png`, `mia_vs_epsilon.png`, `inversion_vs_epsilon.png`,
      `probe_inversion_vs_epsilon.png` (Sprint 3), and any Metric DP figure
- [ ] Abstract contains actual numbers, not placeholders
- [ ] Results section numbers match figures exactly
- [ ] Bibliography resolves all 9 citations
- [ ] Page limit checked against course requirements
- [ ] `docs/proposal.pdf` still present in repository

---

## Guardrails — Read Before Touching Any File

**1. No isolated changes.**
Every change must be tied to a ticket. If you find a bug not covered by a
ticket, open a discussion before fixing it. Isolated patches create merge
conflicts and untraceable regressions.

**2. Investigate before modifying.**
Before editing any file, read it fully. Use the coding agent's Ask mode to
search for how functions are called, what they return, and what depends on
them. The agent prompt format is: describe what you need to know, list the
files to read, and ask specific questions. Do not guess.

**3. One ticket per branch.**
Create a git branch named `sprint-N-ticket-ID` (e.g. `sprint1-s1a`) before
starting any ticket. Open a pull request to `main` when done. Do not commit
directly to `main`.

**4. Test before pushing.**
Run `pytest tests/ -v` after every change. All 5 tests must pass. If your
ticket adds new code, add a corresponding test in `tests/`.

**5. Do not modify protected files without discussion.**
The following files contain verified mathematical implementations and must
not be changed without team agreement:
- `src/models/central_dp.py`
- `src/models/local_dp.py`
- `src/evaluation/mia_lira.py`
- `tests/test_privacy_math.py`

**6. results.csv is generated, not edited.**
Never manually edit `data/results.csv`. It is always produced by running
`python scripts/run_experiments.py`. If you need to rerun, do so fully.

**7. Large compute jobs run overnight.**
Sprint 1 (5 realisations) takes ~5 hours. Sprint 2 with Metric DP adds
significant compute per epsilon due to neighbourhood search. Schedule these
to run when you are not using the machine.

---

## Environment Setup

```bash
git clone <repo-url>
cd med-rag-privacy-eval
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -v                # All 5 tests must pass before proceeding
python scripts/run_experiments.py
python scripts/plot_results.py
```

---

## Reference Papers

All 9 papers are in `papers/`. Read at minimum the abstract and results
of each before writing any section that cites them.

1. Dwork & Roth (2014) — Chapter 3 (Gaussian mechanism), Prop 2.1 (post-processing)
2. Carlini et al. (2022) — Sections 3–5 (LiRA, TPR@FPR justification)
3. Du et al. (2023) — Section 3 (bottleneck architecture), Section 4 (privacy proof)
4. Bollegala et al. (2025) — Section 3 (Metric DP definition and CMAG mechanism)
5. Morris et al. (2023) — Vec2Text method and MIMIC-III clinical reconstruction results
6. Zeng et al. (2024) — RAG privacy threat taxonomy
7. Anderson et al. (2025) — RAG-specific MIA threat model
8. Shokri et al. (2017) — Original shadow model MIA foundation
9. Zhang et al. (2020) — BERTScore justification
