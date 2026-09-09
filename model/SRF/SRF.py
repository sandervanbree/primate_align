#!/usr/bin/env python3
"""SRF factorization over significant CCA spaces."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import check_random_state
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.paths import config
from functions.srf._common import (
    n_jobs_for_tasks,
    observation_mask,
    replace_missing_with_nan,
    symmetrize_observations,
)
from functions.srf.cross_validation import (
    CVResult,
    _CVSplit,
    _best_rank,
    _checked_srf_kwargs,
    _cv_seeds,
    _entrywise_splits,
    _observed_bounds,
    _prepare_similarity,
    _score_job,
    _summarize,
    _validate_fraction,
    _validate_ranks,
    get_fold_fraction,
)
from functions.srf.coherence import calibrate_cross_validation
from functions.srf.model import SRF, _w_solver_backend
from model.CCA.significance import get_significant_components


FAMILIES = ("all", "human", "monkey")
DEFAULT_FAMILY_ORDER = ("monkey", "human", "all")


def compute_mean_rsm(cca_results: dict, fisher: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return a normalized Fisher-z averaged cosine RSM across views."""
    views = sorted(cca_results["components"])
    sims = []
    print(f"Computing RSMs for {len(views)} views: {views}")

    for view in views:
        comps = np.asarray(cca_results["components"][view], dtype=np.float64)
        sim = cosine_similarity(comps)
        if fisher:
            np.clip(sim, -0.999999, 0.999999, out=sim)
            sim = np.arctanh(sim)
        sims.append(sim)
        print(f"  {view}: {sim.shape}")

    stacked = np.stack(sims)
    mean_sim = stacked.mean(axis=0)
    if fisher:
        mean_sim = np.tanh(mean_sim)
        print("Used Fisher-z transform")
    else:
        print("Used simple averaging")

    mean_sim = symmetrize_observations(replace_missing_with_nan(mean_sim, np.nan))
    finite = np.isfinite(mean_sim)
    vmin = float(mean_sim[finite].min())
    vmax = float(mean_sim[finite].max())
    if vmax <= vmin:
        raise ValueError(f"Degenerate RSM: min={vmin}, max={vmax}")
    rsm = (mean_sim - vmin) / (vmax - vmin)
    rsm = np.asarray(rsm, dtype=np.float64)
    rsm = 0.5 * (rsm + rsm.T)
    np.fill_diagonal(rsm, 1.0)
    print(f"Final RSM range: [{rsm.min():.3f}, {rsm.max():.3f}]")
    return rsm, stacked


def output_dir() -> Path:
    return Path(config.results_dir) / "srf_results"


def results_file() -> Path:
    return output_dir() / "srf_all_results.pkl"


def check_srf_status() -> tuple[set[str], list[str]]:
    """Check which SRF families have been completed."""
    fp = results_file()
    if not fp.exists():
        print("No SRF results found. All families need to be processed.")
        return set(), list(FAMILIES)

    try:
        with open(fp, "rb") as f:
            data = pickle.load(f)
        completed = set(data["results"].keys())
        pending = [f for f in FAMILIES if f not in completed]
        print("SRF Status:")
        print(f"  Completed: {sorted(completed) if completed else 'None'}")
        print(f"  Pending: {pending if pending else 'None'}")
        return completed, pending
    except Exception as exc:
        print(f"Error reading results file: {exc}")
        return set(), list(FAMILIES)


def config_section() -> dict:
    return config.hyperparameters.get("srf", {})


def rank_ranges() -> dict[str, list[int]]:
    ranges = config_section().get("rank_ranges", {})
    default = list(range(20, 201, 20))
    return {family: [int(r) for r in ranges.get(family, default)] for family in FAMILIES}


def srf_kwargs(seed: int, verbose: int = 0) -> dict:
    cfg = config_section()
    return {
        "rho": float(cfg.get("rho", 3.0)),
        "max_outer": int(cfg.get("max_outer", 30)),
        "max_inner": int(cfg.get("max_inner", 20)),
        "tol": float(cfg.get("tol", 1e-4)),
        "init": str(cfg.get("init", "random_sqrt")),
        "verbose": int(verbose),
        "random_state": int(seed),
        "missing_values": np.nan,
        "bounds": (0.0, 1.0),
        "check_input": False,
    }


def fit_srf(rsm: np.ndarray, rank: int, seed: int, verbose: int = 0) -> tuple[np.ndarray, dict]:
    model = SRF(rank=rank, **srf_kwargs(seed, verbose=verbose))
    t0 = time.time()
    w = model.fit_transform(rsm)
    elapsed = time.time() - t0
    return np.asarray(w, dtype=np.float64), {
        "rank": int(rank),
        "elapsed_sec": float(elapsed),
        "n_iter": int(model.n_iter_),
        "history": {k: [float(x) for x in v] for k, v in model.history_.items()},
        "backend": _w_solver_backend,
    }


def best_score_from_rank_scores(rank_scores, rank: int) -> float:
    row = rank_scores[rank_scores["candidate_rank"] == rank]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["val_mse_mean"])


def load_state() -> dict:
    state_fp = Path(config.cache_file)
    if not state_fp.exists():
        raise FileNotFoundError(f"CCA state file not found at {state_fp}. Run CCA_families.py first.")
    print(f"Loading CCA state from {state_fp}...")
    with open(state_fp, "rb") as f:
        return pickle.load(f)


def load_existing(force: bool) -> dict:
    fp = results_file()
    if force or not fp.exists():
        return {}
    print(f"Loading existing SRF results from {fp}...")
    with open(fp, "rb") as f:
        return pickle.load(f).get("results", {})


def save_results(results: dict, all_stims: list[str], metadata: dict) -> None:
    output_dir().mkdir(parents=True, exist_ok=True)
    with open(results_file(), "wb") as f:
        pickle.dump(
            {
                "results": results,
                "all_stims": all_stims,
                "config": metadata,
            },
            f,
        )


def write_rank_tables(family_dir: Path, cv_result) -> None:
    family_dir.mkdir(parents=True, exist_ok=True)
    cv_result.fold_scores.to_csv(family_dir / "fold_scores.csv", index=False)
    cv_result.rank_scores.to_csv(family_dir / "rank_scores.csv", index=False)
    meta = {
        "model_rank": int(cv_result.model_rank),
        "sampling_fraction": float(cv_result.sampling_fraction),
        "spectral_cutoff": None if cv_result.spectral_cutoff is None else int(cv_result.spectral_cutoff),
        "candidate_ranks": [int(r) for r in cv_result.candidate_ranks],
    }
    with open(family_dir / "cv_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def clean_family_outputs(family_dir: Path) -> None:
    for pattern in (
        "fold_scores.csv",
        "rank_scores.csv",
        "cv_meta.json",
        "cv_result.pkl",
        "rank_*_components.npz",
        "rank_*_fit_meta.json",
    ):
        for path in family_dir.glob(pattern):
            path.unlink()


def cv_fit_kwargs(cfg: dict, fit_verbose: int) -> dict:
    return _checked_srf_kwargs(
        {
            "rho": float(cfg.get("rho", 3.0)),
            "max_outer": int(cfg.get("max_outer", 30)),
            "max_inner": int(cfg.get("max_inner", 20)),
            "tol": float(cfg.get("tol", 1e-4)),
            "init": str(cfg.get("init", "random_sqrt")),
            "verbose": fit_verbose,
            "check_input": False,
        }
    )


def calibration_kwargs(cfg: dict) -> dict:
    kwargs: dict[str, object] = {}
    sampling_grid = cfg.get("calibration_sampling_grid", None)
    n_bootstrap = cfg.get("calibration_n_bootstrap", None)
    max_eigenpairs = cfg.get("calibration_max_eigenpairs", None)
    if sampling_grid is not None:
        kwargs["sampling_grid"] = np.asarray(
            sampling_grid, dtype=np.float64
        )
    if n_bootstrap is not None:
        kwargs["n_bootstrap"] = int(n_bootstrap)
    if max_eigenpairs is not None:
        kwargs["max_eigenpairs"] = int(max_eigenpairs)
    return kwargs


def _checkpointed_score_job(s, job, bounds, fit_kwargs, pin_threads):
    split, rank, _ = job
    t0 = time.time()
    score = _score_job(s, job, bounds, fit_kwargs, pin_threads)
    return {
        "repeat": int(split.repeat),
        "fold": int(split.fold),
        "candidate_rank": int(rank),
        "val_mse": float(score),
        "elapsed_sec": float(time.time() - t0),
    }


def _observed_offdiag_pairs(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triu_i, triu_j = np.triu_indices_from(observed, k=1)
    valid = np.flatnonzero(observed[triu_i, triu_j])
    if valid.size == 0:
        raise ValueError("cross-validation needs observed off-diagonal entries")
    return triu_i, triu_j, valid


def _pairs_to_symmetric_mask(pairs: np.ndarray, triu_i: np.ndarray, triu_j: np.ndarray, n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    mask[triu_i[pairs], triu_j[pairs]] = True
    mask[triu_j[pairs], triu_i[pairs]] = True
    return mask


def _legacy_holdout_splits(
    observed: np.ndarray,
    observed_fraction: float,
    n_repeats: int,
    split_seeds: np.ndarray,
) -> list[_CVSplit]:
    """Legacy matrix completion: train on p, validate on the held-out complement."""
    observed_fraction = _validate_fraction(observed_fraction)
    triu_i, triu_j, pairs = _observed_offdiag_pairs(observed)
    n_train = max(1, int(observed_fraction * pairs.size))
    splits = []
    for repeat, seed in enumerate(split_seeds):
        rng = check_random_state(int(seed))
        train_pairs = rng.choice(pairs, size=n_train, replace=False)
        train = _pairs_to_symmetric_mask(train_pairs, triu_i, triu_j, observed.shape[0])
        diag = np.diag_indices_from(train)
        train[diag] = observed[diag]

        validation = observed.copy()
        validation[diag] = False
        validation[train] = False
        splits.append(_CVSplit(repeat=repeat, fold=0, train_mask=train, validation_mask=validation))
    return splits


def _read_fold_scores(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["repeat", "fold", "candidate_rank", "val_mse", "elapsed_sec"])
    return pd.read_csv(path)


def _completed_cv_keys(df: pd.DataFrame) -> set[tuple[int, int, int]]:
    if df.empty:
        return set()
    return {
        (int(row.repeat), int(row.fold), int(row.candidate_rank))
        for row in df.itertuples(index=False)
    }


def _write_cv_progress(family_dir: Path, df: pd.DataFrame, meta: dict) -> None:
    family_dir.mkdir(parents=True, exist_ok=True)
    df = (
        df.sort_values(["candidate_rank", "repeat", "fold"], kind="stable")
        .reset_index(drop=True)
    )
    df.to_csv(family_dir / "fold_scores.csv", index=False)
    if not df.empty:
        _summarize(df).to_csv(family_dir / "rank_scores.csv", index=False)
    with open(family_dir / "cv_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def checkpointed_cross_val_score(
    similarity_matrix: np.ndarray,
    ranks: list[int],
    family_dir: Path,
    n_folds: int,
    n_repeats: int,
    random_state: int,
    n_jobs: int,
    sampling_fraction: float | None,
    observed_fraction: float | None,
    legacy_holdout: bool,
    fit_kwargs: dict,
    force: bool = False,
) -> CVResult:
    family_dir.mkdir(parents=True, exist_ok=True)
    if force:
        clean_family_outputs(family_dir)

    fold_scores_path = family_dir / "fold_scores.csv"
    meta_path = family_dir / "cv_meta.json"

    s = _prepare_similarity(similarity_matrix, np.nan)
    ranks_tuple = _validate_ranks(ranks, s.shape[0])
    bounds = _observed_bounds(s)

    calibration = None
    cached_spectral_cutoff = None
    if not legacy_holdout and sampling_fraction is None and meta_path.exists() and not force:
        try:
            with open(meta_path) as f:
                prior_meta = json.load(f)
            prior_fraction = prior_meta.get("sampling_fraction")
            if prior_fraction is not None:
                sampling_fraction = float(prior_fraction)
                prior_cutoff = prior_meta.get("spectral_cutoff")
                cached_spectral_cutoff = None if prior_cutoff is None else int(prior_cutoff)
                print(f"Reusing cached sampling_fraction={sampling_fraction:.4f}")
        except Exception:
            sampling_fraction = None

    if legacy_holdout:
        if observed_fraction is None:
            raise ValueError("legacy_holdout=True requires observed_fraction")
        effective_fraction = _validate_fraction(observed_fraction)
        split_seeds, fit_seeds = _cv_seeds(random_state, n_repeats, 1, len(ranks_tuple))
        splits = _legacy_holdout_splits(
            observation_mask(s),
            effective_fraction,
            n_repeats,
            split_seeds,
        )
    elif sampling_fraction is None:
        cfg = config_section()
        calibration_n_jobs = int(cfg.get("calibration_n_jobs", n_jobs))
        calib_kwargs = calibration_kwargs(cfg)
        calib_desc = ", ".join(
            f"{key}={value if not isinstance(value, np.ndarray) else value.tolist()}"
            for key, value in calib_kwargs.items()
        )
        if calib_desc:
            calib_desc = f" ({calib_desc})"
        print(
            f"Estimating SRF CV sampling fraction with n_jobs={calibration_n_jobs}{calib_desc}...",
            flush=True,
        )
        calibration = calibrate_cross_validation(
            s,
            random_state=random_state,
            n_jobs=calibration_n_jobs,
            **calib_kwargs,
        )
        sampling_fraction = _validate_fraction(calibration.sampling_fraction)
    else:
        sampling_fraction = _validate_fraction(sampling_fraction)
    if calibration is not None:
        print(
            f"Estimated sampling_fraction={sampling_fraction:.4f}, "
            f"spectral_cutoff={calibration.spectral_cutoff}",
            flush=True,
        )
    if not legacy_holdout:
        fold_fraction = get_fold_fraction(sampling_fraction, n_folds, s.shape[0])
        effective_fraction = float(fold_fraction * (n_folds - 1) / n_folds)
        split_seeds, fit_seeds = _cv_seeds(random_state, n_repeats, n_folds, len(ranks_tuple))
        splits = _entrywise_splits(observation_mask(s), fold_fraction, n_folds, split_seeds)

    all_jobs = [
        (split, rank, int(fit_seeds[split.repeat, split.fold, i]))
        for split in splits
        for i, rank in enumerate(ranks_tuple)
    ]

    df = _read_fold_scores(fold_scores_path)
    completed = _completed_cv_keys(df)
    pending = [
        job
        for job in all_jobs
        if (int(job[0].repeat), int(job[0].fold), int(job[1])) not in completed
    ]

    meta = {
        "model_rank": None,
        "sampling_fraction": effective_fraction,
        "spectral_cutoff": (
            cached_spectral_cutoff
            if calibration is None
            else int(calibration.spectral_cutoff)
        ),
        "candidate_ranks": [int(r) for r in ranks_tuple],
        "n_expected_scores": len(all_jobs),
        "n_completed_scores": int(len(df)),
        "cv_mode": "legacy_holdout" if legacy_holdout else "pysrf_fold_sampling",
        "observed_fraction": effective_fraction if legacy_holdout else None,
        "complete": False,
    }
    _write_cv_progress(family_dir, df, meta)

    if pending:
        n_jobs_eff = n_jobs_for_tasks(n_jobs, len(pending))
        pin_threads = bool(config_section().get("pin_cv_threads", n_jobs_eff > 1))
        print(
            f"Running checkpointed SRF CV: {len(pending)}/{len(all_jobs)} pending "
            f"jobs, n_jobs={n_jobs_eff}, "
            f"{'observed_fraction' if legacy_holdout else 'sampling_fraction'}={effective_fraction:.4f}, "
            f"pin_threads={pin_threads}"
        )
        generator = Parallel(
            n_jobs=n_jobs_eff,
            prefer="processes",
            return_as="generator_unordered",
            batch_size=1,
        )(
            delayed(_checkpointed_score_job)(s, job, bounds, fit_kwargs, pin_threads)
            for job in pending
        )
        rows = [] if df.empty else df.to_dict("records")
        completed_count = len(rows)
        for row in generator:
            rows.append(row)
            completed_count += 1
            df = pd.DataFrame(rows)
            meta["n_completed_scores"] = int(completed_count)
            _write_cv_progress(family_dir, df, meta)
            print(
                f"[cv] rank={row['candidate_rank']} repeat={row['repeat']} "
                f"fold={row['fold']} mse={row['val_mse']:.6g} "
                f"elapsed={row['elapsed_sec']:.1f}s "
                f"({completed_count}/{len(all_jobs)})",
                flush=True,
            )
    else:
        print("All CV fold scores already present; summarizing cached results.")

    df = _read_fold_scores(fold_scores_path)
    expected = len(all_jobs)
    if len(df) != expected:
        raise RuntimeError(f"CV incomplete: {len(df)}/{expected} fold scores present")

    rank_scores = _summarize(df)
    model_rank = _best_rank(rank_scores)
    meta["model_rank"] = int(model_rank)
    meta["n_completed_scores"] = int(len(df))
    meta["complete"] = True
    _write_cv_progress(family_dir, df, meta)

    return CVResult(
        model_rank=int(model_rank),
        fold_scores=df,
        rank_scores=rank_scores,
        spectral_cutoff=meta["spectral_cutoff"],
        sampling_fraction=effective_fraction,
        candidate_ranks=ranks_tuple,
        calibration=calibration,
    )


def component_paths(family_dir: Path, rank: int) -> tuple[Path, Path]:
    return (
        family_dir / f"rank_{int(rank)}_components.npz",
        family_dir / f"rank_{int(rank)}_fit_meta.json",
    )


def _fit_and_save_rank(rsm: np.ndarray, rank: int, seed: int, fit_verbose: int, family_dir: Path):
    comp_path, meta_path = component_paths(family_dir, rank)
    print(f"[fit] rank={rank}: starting", flush=True)
    w, meta = fit_srf(rsm, rank, seed + rank, verbose=fit_verbose)
    np.savez(comp_path, components=w)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[fit] rank={rank}: done in {meta['elapsed_sec']:.1f}s", flush=True)
    return int(rank), w, meta


def load_or_fit_components(
    rsm: np.ndarray,
    ranks: list[int],
    family_dir: Path,
    seed: int,
    fit_verbose: int,
    n_jobs: int = 1,
) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
    components = {}
    fit_meta = {}
    print("Generating components for all ranks...")
    missing = []
    for rank in tqdm(ranks, desc=f"{family_dir.name} ranks", leave=False):
        comp_path, meta_path = component_paths(family_dir, rank)
        if comp_path.exists() and meta_path.exists():
            with np.load(comp_path) as data:
                components[int(rank)] = np.asarray(data["components"], dtype=np.float64)
            with open(meta_path) as f:
                fit_meta[int(rank)] = json.load(f)
            print(f"[fit] rank={rank}: cached", flush=True)
            continue
        missing.append(int(rank))

    if missing:
        n_jobs_eff = n_jobs_for_tasks(n_jobs, len(missing))
        print(f"[fit] fitting {len(missing)} missing ranks with n_jobs={n_jobs_eff}", flush=True)
        fitted = Parallel(n_jobs=n_jobs_eff, prefer="processes", batch_size=1)(
            delayed(_fit_and_save_rank)(rsm, rank, seed, fit_verbose, family_dir)
            for rank in missing
        )
        for rank, w, meta in fitted:
            components[int(rank)] = np.asarray(w, dtype=np.float64)
            fit_meta[int(rank)] = meta
    return components, fit_meta


def process_family(family: str, ranks: list[int], seed: int, fit_verbose: int, force: bool = False):
    cfg = config_section()
    fisher_z = bool(cfg.get("use_fisher_z", True))
    n_folds = int(cfg.get("n_folds", 3))
    n_repeats = int(cfg.get("n_repeats", 1))
    n_jobs = int(cfg.get("n_jobs", 36))
    sampling_fraction = cfg.get("sampling_fraction", None)
    legacy_holdout = bool(cfg.get("legacy_holdout", False))
    observed_fraction = cfg.get("observed_fraction", None)

    cca_results = {"components": get_significant_components(family)}
    rsm, _ = compute_mean_rsm(cca_results, fisher=fisher_z)

    family_dir = output_dir() / family
    cv_cache = family_dir / "cv_result.pkl"
    if cv_cache.exists() and not force:
        print(f"Loading cached SRF CV results for {family}...")
        with open(cv_cache, "rb") as f:
            cv_result = pickle.load(f)
    else:
        print(f"Running SRF rank search for {family}...")
        cv_result = checkpointed_cross_val_score(
            rsm,
            ranks=ranks,
            family_dir=family_dir,
            n_folds=n_folds,
            n_repeats=n_repeats,
            random_state=seed,
            n_jobs=n_jobs,
            sampling_fraction=sampling_fraction,
            observed_fraction=observed_fraction,
            legacy_holdout=legacy_holdout,
            fit_kwargs=cv_fit_kwargs(cfg, fit_verbose),
            force=force,
        )
        family_dir.mkdir(parents=True, exist_ok=True)
        with open(cv_cache, "wb") as f:
            pickle.dump(cv_result, f)
    write_rank_tables(family_dir, cv_result)
    print(f"Best rank={cv_result.model_rank}, CV MSE={best_score_from_rank_scores(cv_result.rank_scores, cv_result.model_rank):.6g}")

    components, fit_meta = load_or_fit_components(
        rsm,
        [int(cv_result.model_rank)],
        family_dir,
        seed,
        fit_verbose,
        n_jobs=1,
    )

    return {
        "best_rank": int(cv_result.model_rank),
        "best_score": best_score_from_rank_scores(cv_result.rank_scores, cv_result.model_rank),
        "cv_results": cv_result.fold_scores.rename(columns={"candidate_rank": "rank", "val_mse": "score"}),
        "rank_scores": cv_result.rank_scores,
        "components": components,
        "fit_meta": fit_meta,
        "backend": _w_solver_backend,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SRF factorization over significant CCA spaces.")
    parser.add_argument("--status-only", action="store_true", help="Only check status of existing results.")
    parser.add_argument("--force", action="store_true", help="Recompute all requested families.")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(DEFAULT_FAMILY_ORDER))
    parser.add_argument("--fit-verbose", type=int, default=0)
    args = parser.parse_args(argv)

    if args.status_only:
        check_srf_status()
        return 0

    if _w_solver_backend != "cython":
        raise RuntimeError(f"SRF backend is {_w_solver_backend!r}; expected cython.")
    print(f"SRF backend: {_w_solver_backend}")

    state = load_state()
    all_stims = [str(s) for s in state["all_stims"]]
    print(f"CCA state loaded. Common stimuli: {len(all_stims)}")

    output_dir().mkdir(parents=True, exist_ok=True)
    ranks_by_family = rank_ranges()
    print("Rank ranges:")
    for family in args.families:
        print(f"  {family}: {ranks_by_family[family]}")

    results = load_existing(args.force)
    seed = int(config_section().get("seed", 42))
    metadata = {
        "method": "SRF",
        "backend": _w_solver_backend,
        "rank_ranges": ranks_by_family,
        "hyperparameters": dict(config_section()),
    }

    for family in tqdm(args.families, desc="Processing families"):
        if family in results and not args.force:
            print(f"\n{family.upper()} already processed, skipping...")
            continue
        print(f"\n{'=' * 50}")
        print(f"Processing {family.upper()}")
        print(f"{'=' * 50}")
        results[family] = process_family(
            family,
            ranks_by_family[family],
            seed,
            args.fit_verbose,
            force=args.force,
        )
        save_results(results, all_stims, metadata)
        print(f"Saved SRF results with {len(results)} families complete")

    print(f"\n{'=' * 50}")
    print("SRF ANALYSIS COMPLETE")
    print(f"{'=' * 50}")
    print(f"Results: {output_dir()}")
    for family, result in results.items():
        print(f"  {family}: best_rank={result['best_rank']}, best_score={result['best_score']:.6g}")
    print(f"All results: {results_file()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
