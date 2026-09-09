#!/usr/bin/env python3
"""
Compute ridge-guard R² scores for sNMF components across families.

Outputs per-family CSVs under analysis/nmf/results/guard/guard_<family>.csv
Including per-view scores plus human/monkey aggregates.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from model.feature.core.ridge_utils import ridge_cv_fold
from config.paths import config
from analysis.nmf.utils import (
    FAMILIES,
    load_srf_results,
    load_cca_state,
    pick_rank,
    guard_views_for_family,
    ensure_dir,
)

HERE = Path(__file__).resolve().parent
RESULT_DIR = ensure_dir(HERE / 'results')
GUARD_DIR = ensure_dir(RESULT_DIR / 'guard')
GUARD_FOLDS = 5


def _component_iter(total, family):
    base = range(total)
    if tqdm is None:
        return base
    return tqdm(base, desc=f"{family} guard", dynamic_ncols=True, unit='comp')


def compute_view_scores(W, comps, view_keys, n_jobs, family, n_components=None, verbose=0):
    mats = {v: np.asarray(comps[v], float) for v in view_keys}
    total = n_components if n_components is not None else W.shape[1]

    def _fit(ci):
        y = W[:, ci]
        out = {}
        for v in view_keys:
            mean_r2, std_r2 = ridge_cv_fold(mats[v], y, n_folds=GUARD_FOLDS)
            out[f"{v}__mean"] = mean_r2
            out[f"{v}__std"] = std_r2
        return out

    rows = Parallel(n_jobs=n_jobs, verbose=verbose, prefer='processes')(
        delayed(_fit)(ci) for ci in _component_iter(total, family)
    )
    return rows


def summarise_guard(rows, family, rank, human_views, monkey_views):
    df = pd.DataFrame(rows)
    df.insert(0, 'component', np.arange(1, len(rows) + 1))
    df.insert(1, 'family', family)
    df.insert(2, 'rank', rank)

    def _agg(view_list, suffix):
        cols = [f"{v}__{suffix}" for v in view_list if f"{v}__{suffix}" in df.columns]
        if not cols:
            return pd.Series(np.nan, index=df.index)
        return df[cols].mean(axis=1)

    if human_views:
        df['human_r2'] = _agg(human_views, 'mean')
        df['human_r2_std'] = _agg(human_views, 'std')
    else:
        df['human_r2'] = np.nan
        df['human_r2_std'] = np.nan
    if monkey_views:
        df['monkey_r2'] = _agg(monkey_views, 'mean')
        df['monkey_r2_std'] = _agg(monkey_views, 'std')
    else:
        df['monkey_r2'] = np.nan
        df['monkey_r2_std'] = np.nan
    return df


def main():
    parser = argparse.ArgumentParser(description="Compute ridge guard scores for sNMF components.")
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=FAMILIES,
                        help='Which NMF families to process.')
    parser.add_argument('--jobs', type=int, default=None,
                        help='Override number of parallel jobs (default: config.analysis.n_jobs).')
    parser.add_argument('--force', action='store_true',
                        help='Recompute even if CSV already exists.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Debug-only: limit components per family (processes first N).')
    args = parser.parse_args()

    n_jobs_cfg = int(config.analysis.get('n_jobs', 4))
    n_jobs = args.jobs if args.jobs and args.jobs > 0 else n_jobs_cfg

    results, _ = load_srf_results()
    cca_state = load_cca_state()

    summary_rows = []

    for family in args.families:
        if family not in results:
            print(f"[ridge_guard] Family '{family}' missing in SRF results, skipping.")
            continue

        out_fp = GUARD_DIR / f"guard_{family}.csv"
        if out_fp.exists() and not args.force:
            print(f"[ridge_guard] {family}: found existing {out_fp}, skipping. Use --force to recompute.")
            continue

        rank = pick_rank(family, results[family])
        W = np.asarray(results[family]['components'][rank], float)
        comps, human_views, monkey_views = guard_views_for_family(family, cca_state)
        view_order = human_views + [v for v in monkey_views if v not in human_views]
        if not view_order:
            print(f"[ridge_guard] {family}: no views to evaluate, skipping.")
            continue

        print(f"[ridge_guard] {family}: rank={rank}, components={W.shape[1]}, views={len(view_order)}, jobs={n_jobs}")
        rows = compute_view_scores(W, comps, view_order, n_jobs=n_jobs, family=family,
                                   n_components=args.limit, verbose=5)
        df = summarise_guard(rows, family, rank, human_views, monkey_views)
        df.to_csv(out_fp, index=False)
        print(f"[ridge_guard] {family}: wrote {out_fp}")

        summary_rows.append({
            'family': family,
            'rank': rank,
            'n_components': W.shape[1],
            'n_views': len(view_order),
            'csv': str(out_fp.relative_to(HERE))
        })

    if summary_rows:
        summary_fp = RESULT_DIR / 'guard_summary.csv'
        pd.DataFrame(summary_rows).to_csv(summary_fp, index=False)
        print(f"[ridge_guard] Summary written to {summary_fp}")


if __name__ == '__main__':
    main()
