#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config
from viz.feature import viz_cca_axes as vf


def _aberrant_indices(comps: np.ndarray, animacy: np.ndarray, n: int) -> np.ndarray:
    comp1_vals = comps[:, 0]
    lives_median = np.nanmedian(animacy)
    nonliving_mask = animacy <= lives_median
    nonliving_idxs = np.where(nonliving_mask)[0]
    comp1_nonliving = comp1_vals[nonliving_idxs]
    sorted_order = np.argsort(comp1_nonliving)
    return nonliving_idxs[sorted_order[:n]]


def _assemble(left_path: Path, right_path: Path, out_path: Path) -> None:
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")

    fig = plt.figure(
        figsize=((left.width + right.width) / 300.0, max(left.height, right.height) / 300.0),
        dpi=300,
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[left.width, right.width], wspace=0.04)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    ax_left.imshow(left)
    ax_right.imshow(right)
    ax_left.axis("off")
    ax_right.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    fig_dir = config.fig_dir / "supp" / "cross_species"
    res_dir = config.results_dir / "supp" / "cross_species"
    fmt = config.plotting.get("savefig_format", "pdf")
    out_fig = fig_dir / f"animacy_exceptions.{fmt}"
    out_sel = res_dir / "animacy_exceptions_selected_items.tsv"

    X = vf.load_alexnet_fc6()
    _, comps_all, all_stims = vf.load_cca_components()
    features = vf.load_features(all_stims, X)
    all_cats = [s.rsplit("_", 1)[0] for s in all_stims]
    aberrant_idxs = _aberrant_indices(comps_all, features["animacy"], n=25)

    sel_df = pd.DataFrame(
        {
            "stimulus": [all_stims[i] for i in aberrant_idxs],
            "category": [all_cats[i] for i in aberrant_idxs],
            "component_1": comps_all[aberrant_idxs, 0],
            "component_2": comps_all[aberrant_idxs, 1],
            "animacy": features["animacy"][aberrant_idxs],
        }
    )
    out_sel.parent.mkdir(parents=True, exist_ok=True)
    sel_df.to_csv(out_sel, sep="\t", index=False)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        grid_path = tmp_dir / "nonliving_negative_comp1_grid.png"
        scatter_path = tmp_dir / "scatter_animacy_aberrant.png"
        vf.plot_nonliving_negative_comp1_grid(
            comps_all,
            all_stims,
            all_cats,
            features["animacy"],
            n=25,
            n_cols=5,
            out_path=grid_path,
        )
        vf.plot_scatter(
            comps_all,
            features["animacy"],
            "animacy_aberrant",
            "PRGn",
            c1=0,
            c2=1,
            highlight_idxs=aberrant_idxs,
            out_path=scatter_path,
        )
        _assemble(grid_path, scatter_path, out_fig)

    print(f"[supp.animacy_exceptions] saved {out_fig}")
    print(f"[supp.animacy_exceptions] saved {out_sel}")


if __name__ == "__main__":
    main()
