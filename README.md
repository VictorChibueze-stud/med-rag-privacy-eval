# 🛡️ Med-RAG Privacy Evaluation

**Evaluating Central vs. Local Differential Privacy for Vector Embeddings in Medical RAG Systems**

This repository contains the core evaluation pipeline for measuring the privacy-utility tradeoff of applying Differential Privacy (DP) mechanisms directly to the embedding layer of a Retrieval-Augmented Generation (RAG) system. We evaluate these protections against Membership Inference Attacks (LiRA plus a simple k-NN distance-ratio baseline) and Embedding Inversion.

## 👥 Team

* **Victor Okoroafor** (Project Lead, AI Infrastructure & Adversarial Evaluation)
* **Ifeanyi Omonigho Odugo** (Privacy Mathematics & Local DP Architect)
* **Gopal Krishna** (Utility Evaluation & Plotting Engine)
* **Niramay Roopesh Kolalle** (Academic Documentation & Synthesis)

## 🚀 Quickstart & Environment Setup

We strictly use Python 3.10+ and manage dependencies via `requirements.txt`.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/VictorChibueze-stud/med-rag-privacy-eval.git
   cd med-rag-privacy-eval
   ```

2. **Create and activate a virtual environment:**

   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

   ```bash
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## 📜 Git Workflow & Contribution Rules

To prevent merge conflicts and broken code, direct pushes to the `main` branch are strictly forbidden.

**Branch Naming Convention:** Always branch off `main` before starting your ticket.

* **Features:** `feat/ticket-name` (e.g., `feat/local-dp-projection`)
* **Bug Fixes:** `fix/ticket-name` (e.g., `fix/faiss-index-bug`)
* **Documentation:** `docs/ticket-name` (e.g., `docs/latex-draft`)

**Code Quality (Ruff):**

We use ruff to enforce PEP-8 standards. Before committing your code, you must run:

```bash
ruff check .
ruff format .
```

If the GitHub Actions CI pipeline fails because of formatting, your PR will not be reviewed.

**Pull Requests (PRs):**

When your feature is complete, open a Pull Request against `main`. Tag Victor for architectural review before merging.

## 📂 Project Architecture

* `data/`: Contains the ChatDoctor dataset and generated `.csv` results (ignored in git).
* `src/models/`: Contains the RAGBaseline, CentralDPMechanism, LocalDPProjector, and MetricDPMechanism.
* `src/evaluation/`: Contains LiRA MIA, k-NN distance-ratio MIA, embedding inversion, and utility evaluation.
* `scripts/`: Contains the execution loops and plotting engines.
* `tests/`: Mathematical variance and dimensional unit tests.

***


## Running the k-NN MIA baseline

`python scripts/run_experiments.py` generates `data/results.csv` with both MIA columns:

```text
tpr_mia
tpr_mia_knn_ratio
```

The `tpr_mia_knn_ratio` value is produced for every evaluated mechanism: Baseline, Central DP, Local DP, and Metric DP. Then run:

```bash
python scripts/plot_results.py
```

to create `docs/figures/mia_knn_ratio_vs_epsilon.png`. If you still have an older `data/results.csv` from before this feature, rerun `scripts/run_experiments.py`; the old CSV will not contain the new k-NN column.
