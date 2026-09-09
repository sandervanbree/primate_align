#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config
from viz.feature import viz_cca_axes as vf


def main() -> None:
    fig_dir = config.fig_dir / "supp" / "cross_species"
    res_dir = config.results_dir / "supp" / "cross_species"
    fmt = config.plotting.get("savefig_format", "pdf")
    out_fig = fig_dir / f"spikiness_alexnet_fc6_pc1.{fmt}"
    out_stats = res_dir / "spikiness_alexnet_fc6_pc1_stats.csv"

    X = vf.load_alexnet_fc6()
    comp_scores, _, _ = vf.load_cca_components()
    results = vf.compute_ridge_cv(X, comp_scores)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    vf.plot_variances_with_stats(results, out_path=out_fig, stats_path=out_stats)

    print(f"[supp.spikiness_alexnet] saved {out_fig}")
    print(f"[supp.spikiness_alexnet] saved {out_stats}")


if __name__ == "__main__":
    main()
