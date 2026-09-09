#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config
from analysis.nmf import gallery as gal


def build(which: list[str] | None = None) -> None:
    fig_dir = config.fig_dir / "supp" / "nmf"
    res_dir = config.results_dir / "supp" / "nmf"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    results, stims = gal.load_srf_results()
    img_paths = gal.build_img_paths(stims)
    cats = gal.derive_categories(stims)
    cat_counts = Counter(cats)

    targets = which or list(gal.GALLERY_SPECS.keys())
    for key in targets:
        spec = dict(gal.GALLERY_SPECS[key])
        spec.pop("top_n", None)
        family = spec["family"]
        guard_df = gal.load_guard_table(family)
        rank = gal.pick_rank(family, results[family])
        W = gal.load_component_matrix(results, family, rank)
        entries = gal.build_entries(key, spec, guard_df, W, cats, cat_counts)
        fig_fp = fig_dir / spec["filename"]
        gal.render_gallery(key, entries, img_paths, spec["title"], spec["filename"], out_path=fig_fp)
        summary_fp = res_dir / spec["summary"]
        gal.save_summary(entries, summary_fp)
        if entries:
            print(f"[supp.nmf_galleries] {key}: saved {fig_fp}")
            print(f"[supp.nmf_galleries] {key}: saved {summary_fp}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", nargs="+", choices=list(gal.GALLERY_SPECS.keys()), default=None)
    args = parser.parse_args()
    build(args.which)


if __name__ == "__main__":
    main()
