#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config
from functions.recoverability import retained_balance_values

DEFAULT_N_BOOT = 100000
DEFAULT_N_PERM = 200000
DEFAULT_SEED = 12345


def _load_family_stats():
    with open(config.results_dir / "ridge_results.pkl", "rb") as f:
        return pickle.load(f)["family_stats"]


def _bootstrap_ci(vals, rng, n_boot):
    boot = rng.choice(vals, size=(n_boot, vals.size), replace=True).mean(axis=1)
    return np.percentile(boot, [2.5, 97.5])


def _permutation_test(reference, comparison, rng, n_perm):
    obs = float(comparison.mean() - reference.mean())
    pooled = np.concatenate([reference, comparison])
    n_ref = reference.size
    one_sided = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = float(perm[n_ref:].mean() - perm[:n_ref].mean())
        if diff >= obs - 1e-15:
            one_sided += 1

    two_sided = 0
    obs_abs = abs(obs)
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = float(perm[n_ref:].mean() - perm[:n_ref].mean())
        if abs(diff) >= obs_abs - 1e-15:
            two_sided += 1
    return obs, (one_sided + 1) / (n_perm + 1), (two_sided + 1) / (n_perm + 1)


def _write_tsv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize cross-species recoverability balance.")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    values, ref_m, ref_h = retained_balance_values(_load_family_stats())

    summary_rows = []
    for family, vals in values.items():
        ci_lo, ci_hi = _bootstrap_ci(vals, rng, args.n_boot)
        summary_rows.append({
            "family": family,
            "n_components": int(vals.size),
            "mean_balance": float(vals.mean()),
            "median_balance": float(np.median(vals)),
            "ci_2.5": float(ci_lo),
            "ci_97.5": float(ci_hi),
            "reference_macaque": ref_m,
            "reference_human": ref_h,
            "n_boot": int(args.n_boot),
        })

    reference = values["Cross-species"]
    test_rows = []
    for family in ("Human-only", "Monkey-only"):
        delta, p_one, p_two = _permutation_test(reference, values[family], rng, args.n_perm)
        test_rows.append({
            "comparison": f"{family} - Cross-species",
            "delta_mean_balance": delta,
            "p_one_sided": p_one,
            "p_two_sided": p_two,
            "n_perm": int(args.n_perm),
        })

    out_dir = config.results_dir / "supp" / "cross_species"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "recoverability_balance_summary.tsv"
    tests_path = out_dir / "recoverability_balance_tests.tsv"
    _write_tsv(summary_path, summary_rows)
    _write_tsv(tests_path, test_rows)
    print(f"[recoverability_stats] saved {summary_path}")
    print(f"[recoverability_stats] saved {tests_path}")


if __name__ == "__main__":
    main()
