#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:
    stats = None

try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config
from viz.feature import viz_combined_weighted as vcw


def main() -> None:
    if stats is None:
        raise RuntimeError("scipy is required")
    if multipletests is None:
        raise RuntimeError("statsmodels is required")

    res = vcw.load_combined_results()
    per = res["dnn"]["per_layer"]
    bins = [0, 20, 40, 60, 80, 100]
    bin_h = [[] for _ in range(len(bins) - 1)]
    bin_m = [[] for _ in range(len(bins) - 1)]

    for model in vcw._ordered_models(list(per.keys())) if per else []:
        layers = per[model]
        if not layers:
            continue
        depths = [float(v["depth"]) for v in layers.values()]
        dmin, dmax = min(depths), max(depths)
        for layer, vals in layers.items():
            d = float(vals["depth"])
            dp = (d - dmin) / (dmax - dmin + 1e-9) * 100.0
            rh = vcw.metric_stats(vals["human"])[0]
            rm = vcw.metric_stats(vals["macaque"])[0]
            for bi in range(len(bins) - 1):
                a, b = bins[bi], bins[bi + 1]
                if (dp >= a) and (dp < b if b < 100 else dp <= b):
                    if np.isfinite(rh):
                        bin_h[bi].append(float(rh))
                    if np.isfinite(rm):
                        bin_m[bi].append(float(rm))

    rows = []
    pvals = []
    valid_idx = []
    for i in range(len(bins) - 1):
        a, b = bins[i], bins[i + 1]
        h_vals = np.array(bin_h[i], float)
        m_vals = np.array(bin_m[i], float)
        row = {
            "bin_start": a,
            "bin_end": b,
            "human_n": int(h_vals.size),
            "human_mean": float(np.nanmean(h_vals)) if h_vals.size else np.nan,
            "human_std": float(np.nanstd(h_vals)) if h_vals.size else np.nan,
            "macaque_n": int(m_vals.size),
            "macaque_mean": float(np.nanmean(m_vals)) if m_vals.size else np.nan,
            "macaque_std": float(np.nanstd(m_vals)) if m_vals.size else np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "q_value": np.nan,
            "significant_fdr": False,
        }
        if h_vals.size >= 2 and m_vals.size >= 2:
            t_stat, p_val = stats.ttest_ind(h_vals, m_vals, equal_var=False)
            row["t_stat"] = float(t_stat)
            row["p_value"] = float(p_val)
            pvals.append(float(p_val))
            valid_idx.append(i)
        rows.append(row)

    if pvals:
        _, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        for q, idx in zip(qvals, valid_idx):
            rows[idx]["q_value"] = float(q)
            rows[idx]["significant_fdr"] = bool(q < 0.05)

    out_path = config.results_dir / "supp" / "dnn" / "depth_bin_stats.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"[supp.dnn_depth_bins] saved {out_path}")


if __name__ == "__main__":
    main()
