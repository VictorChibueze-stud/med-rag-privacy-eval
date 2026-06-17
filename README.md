# Med-RAG Privacy Evaluation

**Evaluating Differential Privacy Mechanisms for Vector Embeddings in Medical RAG Systems: A Three-Way Comparison of Central, Local, and Metric DP**

Project report for *Privacy-Preserving Methods for Data Science and Distributed Systems*
Department of Mathematics and Computer Science, University of Basel, Spring 2026.

---

## Team

| Name | Role |
|------|------|
| Victor Okoroafor | Project lead, system architecture, adversarial evaluation |
| Ifeanyi Omonigho Odugo | Privacy mathematics, Local DP architecture |
| Gopal Krishna | Utility evaluation, plotting engine, k-NN MIA baseline |
| Niramay Roopesh Kolalle | Academic documentation, manuscript synthesis |

---

## Overview

This repository implements and evaluates three differential privacy mechanisms applied to sentence embeddings in a medical Retrieval-Augmented Generation (RAG) pipeline:

- **Central DP** — Analytical Gaussian mechanism with global L2 sensitivity
- **Local DP** — Fixed random orthogonal projection to 16-d bottleneck with per-document noise
- **Metric DP** — Mahalanobis-geometry-aware noise following Bollegala et al. (2025)

Privacy is evaluated using membership inference (LiRA + k-NN distance ratio baseline) and embedding inversion (nearest-neighbour ROUGE-L + linear probe). Utility is measured via BERTScore F1.

---

## Key Finding

Central and Local DP exhibit epsilon-invariant protection across ε ∈ [0.1, 10]: inversion ROUGE-L drops from 1.00 to ~0.21 immediately and stays flat, with less than 2% utility degradation. Metric DP collapses at ε ≥ 5, reaching inversion ROUGE-L of 0.950 at ε = 10. **Central DP is recommended for medical RAG deployments.**

---

## Repository Structure

```
src/
  data_loader.py              # ChatDoctor dataset loading and splitting
  models/
    central_dp.py             # Central DP Gaussian mechanism
    local_dp.py               # Local DP bottleneck projection
    metric_dp.py              # Metric DP Mahalanobis mechanism
    rag_baseline.py           # Unperturbed RAG baseline
  evaluation/
    mia_lira.py               # LiRA membership inference attack
    mia_knn_ratio.py          # k-NN distance ratio MIA baseline
    inversion.py              # Nearest-neighbour embedding inversion
    inversion_probe.py        # Linear probe inversion attack
    utility.py                # BERTScore utility evaluation
scripts/
  run_experiments.py          # Full experiment pipeline
  plot_results.py             # Figure generation
tests/                        # Unit and integration tests (14 tests)
docs/
  main.tex                    # Paper manuscript
  references.bib              # Bibliography
  figures/                    # Generated experiment figures
  proposal.pdf                # Original project proposal
data/
  .gitkeep                    # Directory placeholder
  results.csv                 # Generated — run run_experiments.py
```

---

## Setup

```bash
git clone https://github.com/VictorChibueze-stud/med-rag-privacy-eval.git
cd med-rag-privacy-eval
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -v              # All 14 tests must pass
```

---

## Data

The ChatDoctor-HealthCareMagic-100k dataset is not included in this repository due to size. Download it with:

```bash
python download_chatdoctor.py
```

This creates `data/chatdoctor.json` (required by `run_experiments.py`).

---

## Reproducing Results

```bash
python scripts/run_experiments.py   # ~4-8 hours on CPU; produces data/results.csv
python scripts/plot_results.py      # Produces docs/figures/*.png
```

The experiment runs 5 independent noise realisations per (ε, mechanism) combination across ε ∈ {0.1, 1.0, 5.0, 10.0} with δ = 1e-5.

---

## Reference

Paper manuscript: `docs/main.tex`
Dataset: [ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k)
