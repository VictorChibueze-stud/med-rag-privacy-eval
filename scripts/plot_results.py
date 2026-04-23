"""Generate academic figures from `data/results.csv` (long or wide layout)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Project root: ``python scripts/plot_results.py`` from the repository root.
_ROOT = Path(__file__).resolve().parents[1]

RESULTS_CSV = _ROOT / "data" / "results.csv"
FIG_DIR = _ROOT / "docs" / "figures"

# Sprint 5 "tidy" column names (one row per &epsilon; & mechanism).
LONG_COLUMNS = {
    "epsilon",
    "mechanism",
    "bertscore_f1",
    "lira_tpr_at_fpr_0.001",
    "inversion_rouge_l",
}

# Wide layout emitted by `scripts/run_experiments.py` (Sprint 4).
WIDE_CENTRAL_POSTFIXES = {
    "bertscore_f1": "utility_bert_f1_central",
    "lira_tpr_at_fpr_0.001": "tpr_mia_central",
    "inversion_rouge_l": "inversion_rouge_l_mean_central",
}
WIDE_LOCAL_POSTFIXES = {
    "bertscore_f1": "utility_bert_f1_local",
    "lira_tpr_at_fpr_0.001": "tpr_mia_local",
    "inversion_rouge_l": "inversion_rouge_l_mean_local",
}


def _load_demo_long() -> pd.DataFrame:
    """Plausible rows when `data/results.csv` is absent (e.g. clean clone)."""
    eps = [0.1, 1.0, 5.0, 10.0]
    # Synthetic curves: more ε → better utility / MIA; central vs. local differ.
    rows: list[dict[str, str | float]] = []
    for i, e in enumerate(eps):
        rows += [
            {
                "epsilon": e,
                "mechanism": "Central",
                "bertscore_f1": 0.72 + 0.12 * (i + 1) / len(eps),
                "lira_tpr_at_fpr_0.001": max(0.02, 0.45 - 0.05 * (i + 1)),
                "inversion_rouge_l": 0.5 + 0.08 * (i + 1) / len(eps),
            },
            {
                "epsilon": e,
                "mechanism": "Local",
                "bertscore_f1": 0.68 + 0.1 * (i + 1) / len(eps),
                "lira_tpr_at_fpr_0.001": max(0.05, 0.38 - 0.04 * (i + 1)),
                "inversion_rouge_l": 0.45 + 0.07 * (i + 1) / len(eps),
            },
        ]
    return pd.DataFrame(rows)


def _wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Melt a wide `run_experiments` frame into the tidy format."""
    if "epsilon" not in df.columns:
        msg = "Wide table must contain an 'epsilon' column."
        raise ValueError(msg)
    for col in WIDE_CENTRAL_POSTFIXES.values():
        if col not in df.columns:
            msg = f"Missing expected column {col!r} for wide-format results."
            raise ValueError(msg)
    parts: list[pd.DataFrame] = []
    for mname, suffixes in [
        ("Central", WIDE_CENTRAL_POSTFIXES),
        ("Local", WIDE_LOCAL_POSTFIXES),
    ]:
        sub = df[["epsilon"]].copy()
        for tidy, col in suffixes.items():
            sub[tidy] = df[col]
        sub["mechanism"] = mname
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def load_results() -> pd.DataFrame:
    """Read CSV and normalize to a tidy long table."""
    if RESULTS_CSV.is_file():
        raw = pd.read_csv(RESULTS_CSV)
        cols = set(str(c) for c in raw.columns)
        if LONG_COLUMNS.issubset(cols):
            out = raw[list(LONG_COLUMNS)].copy()
        elif "tpr_mia_central" in cols and "mechanism" not in cols:
            out = _wide_to_long(raw)
        else:
            msg = (
                f"Unrecognized columns in {RESULTS_CSV}.\n"
                f"Need either the tidy set {sorted(LONG_COLUMNS)} or a wide run output."
            )
            raise ValueError(msg)
    else:
        print(
            f"Note: {RESULTS_CSV} not found. Using a built-in demo long-format table.",
            file=sys.stderr,
        )
        out = _load_demo_long()
    if out["mechanism"].dtype == object:
        out["mechanism"] = out["mechanism"].str.strip()
    for c in [
        "epsilon",
        "bertscore_f1",
        "lira_tpr_at_fpr_0.001",
        "inversion_rouge_l",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _lineplot_save(
    df: pd.DataFrame,
    ycol: str,
    ylabel: str,
    title: str,
    out_name: str,
) -> None:
    """One dual-mechanism line plot with log-scaled &epsilon; when spread is large."""
    sns.set_theme(style="whitegrid")
    d = df.sort_values(["mechanism", "epsilon"]).copy()
    d["mechanism"] = (
        d["mechanism"].str.strip().map({"Central": "Central DP", "Local": "Local DP"})
    )

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    sns.lineplot(
        data=d,
        x="epsilon",
        y=ycol,
        hue="mechanism",
        style="mechanism",
        markers=True,
        dashes=False,
        err_style=None,
        ax=ax,
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r"$\epsilon$ (privacy budget)")

    eps = np.sort(d["epsilon"].unique())
    if eps.size >= 2 and (eps.max() / max(eps.min(), 1e-12) >= 5.0):
        ax.set_xscale("log")
    else:
        # Treat ε as a categorical *scale* of ticks when nearly uniform (visual clarity)
        pass

    ax.set_title(title)
    ax.legend(title=None)
    fig.tight_layout()
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    """Load `results`, emit three publication PNGs into ``docs/figures/``."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()

    _lineplot_save(
        df,
        ycol="bertscore_f1",
        ylabel="BERTScore F1 (utility)",
        title="RAG Utility Degradation under Differential Privacy",
        out_name="utility_vs_epsilon.png",
    )
    _lineplot_save(
        df,
        ycol="lira_tpr_at_fpr_0.001",
        ylabel="TPR @ 0.1% FPR",
        title="Membership Inference Vulnerability (TPR @ 0.1% FPR)",
        out_name="mia_vs_epsilon.png",
    )
    _lineplot_save(
        df,
        ycol="inversion_rouge_l",
        ylabel="ROUGE-L (reconstruction F-measure)",
        title="Embedding Inversion Reconstruction Fidelity",
        out_name="inversion_vs_epsilon.png",
    )


if __name__ == "__main__":
    main()
