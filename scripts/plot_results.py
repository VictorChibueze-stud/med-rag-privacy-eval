from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = _ROOT / 'data' / 'results.csv'
FIG_DIR = _ROOT / 'docs' / 'figures'
LONG_REQUIRED_COLUMNS = ['epsilon', 'mechanism', 'tpr_mia', 'inversion_rouge_l_mean', 'bert_f1']
LONG_OPTIONAL_COLUMNS = ['run_id', 'probe_rouge_l_mean', 'tpr_mia_knn_ratio', 'bert_precision', 'bert_recall']
WIDE_CENTRAL_POSTFIXES = {'bert_f1': 'utility_bert_f1_central', 'tpr_mia': 'tpr_mia_central', 'inversion_rouge_l_mean': 'inversion_rouge_l_mean_central'}
WIDE_LOCAL_POSTFIXES = {'bert_f1': 'utility_bert_f1_local', 'tpr_mia': 'tpr_mia_local', 'inversion_rouge_l_mean': 'inversion_rouge_l_mean_local'}

def _load_demo_long() -> pd.DataFrame:
    eps = [0.1, 1.0, 5.0, 10.0]
    rows: list[dict[str, str | float | int]] = [{'run_id': 0, 'epsilon': float('inf'), 'mechanism': 'Baseline', 'bert_f1': 0.86, 'tpr_mia': 0.5, 'tpr_mia_knn_ratio': 0.42, 'inversion_rouge_l_mean': 0.62, 'probe_rouge_l_mean': 0.48}]
    offsets = [-0.015, -0.0075, 0.0, 0.0075, 0.015]
    for run_id, offset in enumerate(offsets):
        for i, e in enumerate(eps):
            rows += [{'run_id': run_id, 'epsilon': e, 'mechanism': 'Central', 'bert_f1': 0.72 + 0.12 * (i + 1) / len(eps) + offset, 'tpr_mia': max(0.02, 0.45 - 0.05 * (i + 1) - offset), 'tpr_mia_knn_ratio': max(0.01, 0.36 - 0.045 * (i + 1) - offset), 'inversion_rouge_l_mean': 0.5 + 0.08 * (i + 1) / len(eps) + offset, 'probe_rouge_l_mean': 0.38 + 0.06 * (i + 1) / len(eps) + offset}, {'run_id': run_id, 'epsilon': e, 'mechanism': 'Local', 'bert_f1': 0.68 + 0.1 * (i + 1) / len(eps) + offset, 'tpr_mia': max(0.05, 0.38 - 0.04 * (i + 1) - offset), 'tpr_mia_knn_ratio': max(0.01, 0.3 - 0.035 * (i + 1) - offset), 'inversion_rouge_l_mean': 0.45 + 0.07 * (i + 1) / len(eps) + offset, 'probe_rouge_l_mean': 0.34 + 0.055 * (i + 1) / len(eps) + offset}, {'run_id': run_id, 'epsilon': e, 'mechanism': 'Metric', 'bert_f1': 0.74 + 0.11 * (i + 1) / len(eps) + offset, 'tpr_mia': max(0.02, 0.41 - 0.045 * (i + 1) - offset), 'tpr_mia_knn_ratio': max(0.01, 0.33 - 0.04 * (i + 1) - offset), 'inversion_rouge_l_mean': 0.52 + 0.075 * (i + 1) / len(eps) + offset, 'probe_rouge_l_mean': 0.4 + 0.058 * (i + 1) / len(eps) + offset}]
    return pd.DataFrame(rows)

def _wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    if 'epsilon' not in df.columns:
        msg = "Wide table must contain an 'epsilon' column."
        raise ValueError(msg)
    for col in WIDE_CENTRAL_POSTFIXES.values():
        if col not in df.columns:
            msg = f'Missing expected column {col!r} for wide-format results.'
            raise ValueError(msg)
    parts: list[pd.DataFrame] = []
    for mname, suffixes in [('Central', WIDE_CENTRAL_POSTFIXES), ('Local', WIDE_LOCAL_POSTFIXES)]:
        sub = df[['epsilon']].copy()
        for tidy, col in suffixes.items():
            sub[tidy] = df[col]
        sub['mechanism'] = mname
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)

def load_results() -> pd.DataFrame:
    if RESULTS_CSV.is_file():
        raw = pd.read_csv(RESULTS_CSV)
        cols = set((str(c) for c in raw.columns))
        if set(LONG_REQUIRED_COLUMNS).issubset(cols):
            keep = LONG_REQUIRED_COLUMNS + [c for c in LONG_OPTIONAL_COLUMNS if c in cols]
            out = raw[keep].copy()
        elif 'tpr_mia_central' in cols and 'mechanism' not in cols:
            out = _wide_to_long(raw)
        else:
            msg = f'Unrecognized columns in {RESULTS_CSV}.\nNeed either the tidy set {LONG_REQUIRED_COLUMNS} or a wide run output.'
            raise ValueError(msg)
    else:
        print(f'Note: {RESULTS_CSV} not found. Using a built-in demo long-format table.', file=sys.stderr)
        out = _load_demo_long()
    if out['mechanism'].dtype == object:
        out['mechanism'] = out['mechanism'].str.strip()
    numeric_cols = ['epsilon', 'tpr_mia', 'inversion_rouge_l_mean', 'probe_rouge_l_mean', 'tpr_mia_knn_ratio', 'bert_f1']
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

def _lineplot_save(df: pd.DataFrame, ycol: str, ylabel: str, title: str, out_name: str) -> None:
    if ycol not in df.columns:
        print(f'Skipping {out_name}: column {ycol!r} is not present.')
        return
    sns.set_theme(style='whitegrid')
    d = df[df['mechanism'] != 'Baseline'].sort_values(['mechanism', 'epsilon']).copy()
    d['mechanism'] = d['mechanism'].str.strip()
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    sns.lineplot(data=d, x='epsilon', y=ycol, hue='mechanism', style='mechanism', markers=True, dashes=False, err_style='band', errorbar='sd', estimator='mean', ax=ax)
    baseline_rows = df[df['mechanism'] == 'Baseline']
    if not baseline_rows.empty and ycol in baseline_rows.columns:
        baseline_val = float(baseline_rows[ycol].mean())
        ax.axhline(baseline_val, linestyle=':', color='grey', linewidth=1.2, label='Baseline (no noise)')
        ax.legend(title=None)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('$\\epsilon$ (privacy budget)')
    eps = np.sort(d['epsilon'].dropna().unique())
    if eps.size >= 2 and eps.max() / max(float(eps.min()), 1e-12) >= 5.0:
        ax.set_xscale('log')
    ax.set_title(title)
    fig.tight_layout()
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Wrote {out}')

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    _lineplot_save(df, ycol='bert_f1', ylabel='BERTScore F1 (utility)', title='RAG Utility Degradation under Differential Privacy', out_name='utility_vs_epsilon.png')
    _lineplot_save(df, ycol='tpr_mia', ylabel='TPR @ 0.1% FPR', title='Membership Inference Vulnerability (TPR @ 0.1% FPR)', out_name='mia_vs_epsilon.png')
    _lineplot_save(df, ycol='inversion_rouge_l_mean', ylabel='ROUGE-L (nearest-neighbour reconstruction)', title='Nearest-Neighbor Embedding Inversion Fidelity', out_name='inversion_vs_epsilon.png')
    _lineplot_save(df, ycol='probe_rouge_l_mean', ylabel='ROUGE-L (linear probe reconstruction)', title='Linear Probe Inversion Fidelity under DP', out_name='probe_inversion_vs_epsilon.png')
    _lineplot_save(df, ycol='tpr_mia_knn_ratio', ylabel='k-NN ratio TPR @ 0.1% FPR', title='k-NN Distance-Ratio MIA under DP Mechanisms', out_name='mia_knn_ratio_vs_epsilon.png')

if __name__ == '__main__':
    main()
