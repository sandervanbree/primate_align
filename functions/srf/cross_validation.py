from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils import check_random_state
from threadpoolctl import threadpool_limits

from ._common import (
    RandomStateLike,
    n_jobs_for_tasks,
    observation_mask,
    replace_missing_with_nan,
    seed_stream,
    symmetrize_observations,
    validate_n_jobs,
)
from .coherence import CVCalibration, calibrate_cross_validation
from .model import SRF

_RESERVED_SRF_KWARGS = frozenset({"rank", "bounds", "missing_values", "random_state"})


@dataclass(frozen=True, slots=True)
class CVResult:
    model_rank: int
    fold_scores: pd.DataFrame
    rank_scores: pd.DataFrame
    spectral_cutoff: int | None
    sampling_fraction: float
    candidate_ranks: tuple[int, ...]
    calibration: CVCalibration | None = None


@dataclass(frozen=True, slots=True)
class _CVSplit:
    repeat: int
    fold: int
    train_mask: np.ndarray
    validation_mask: np.ndarray


def cross_val_score(
    similarity_matrix: np.ndarray,
    ranks: Sequence[int],
    sampling_fraction: float | None = None,
    n_folds: int = 5,
    n_repeats: int = 1,
    random_state: RandomStateLike = 0,
    n_jobs: int | None = -1,
    missing_values: float | None = np.nan,
    srf_kwargs: Mapping[str, object] | None = None,
) -> CVResult:
    _check_args(n_folds, n_repeats, n_jobs)
    fit_kwargs = _checked_srf_kwargs(srf_kwargs)

    s = _prepare_similarity(similarity_matrix, missing_values)
    n = s.shape[0]
    ranks = _validate_ranks(ranks, n)
    bounds = _observed_bounds(s)
    sampling_fraction, calibration = _cv_sampling_fraction(
        s, sampling_fraction, random_state, n_jobs
    )

    fold_fraction = get_fold_fraction(sampling_fraction, n_folds, n)
    fold_scores = _cross_validate_ranks(
        s,
        ranks,
        fold_fraction,
        n_folds,
        n_repeats,
        random_state,
        n_jobs,
        bounds,
        fit_kwargs,
    )
    rank_scores = _summarize(fold_scores)

    return CVResult(
        model_rank=_best_rank(rank_scores),
        fold_scores=fold_scores,
        rank_scores=rank_scores,
        spectral_cutoff=_spectral_cutoff(calibration),
        sampling_fraction=_training_fraction(fold_fraction, n_folds),
        candidate_ranks=ranks,
        calibration=calibration,
    )


def _prepare_similarity(
    similarity_matrix: np.ndarray,
    missing_values: float | None,
) -> np.ndarray:
    return symmetrize_observations(
        replace_missing_with_nan(similarity_matrix, missing_values)
    )


def _check_args(n_folds: int, n_repeats: int, n_jobs: int | None) -> None:
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be at least 1, got {n_repeats}")
    validate_n_jobs(n_jobs)


def _checked_srf_kwargs(srf_kwargs: Mapping[str, object] | None) -> dict:
    kwargs = dict(srf_kwargs) if srf_kwargs else {}
    reserved = _RESERVED_SRF_KWARGS & kwargs.keys()
    if reserved:
        raise ValueError(
            f"srf_kwargs must not contain {sorted(reserved)}; these are set per fit"
        )
    return kwargs


def _validate_ranks(ranks: Sequence[int], n: int) -> tuple[int, ...]:
    ranks = tuple(dict.fromkeys(int(r) for r in ranks))
    if not ranks:
        raise ValueError("ranks must contain at least one value")
    if any(r < 1 for r in ranks):
        raise ValueError("ranks must be positive")
    if any(r > n for r in ranks):
        raise ValueError(f"ranks must be at most n_features={n}")
    return ranks


def _validate_fraction(fraction: float) -> float:
    fraction = float(fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("sampling_fraction must be in (0, 1)")
    return fraction


def _observed_bounds(s: np.ndarray) -> tuple[float, float]:
    observed = observation_mask(s)
    values = s[observed & np.isfinite(s)]
    if values.size == 0:
        raise ValueError("similarity_matrix has no finite observed entries")
    return float(values.min()), float(values.max())


def _cv_sampling_fraction(
    s: np.ndarray,
    sampling_fraction: float | None,
    random_state: RandomStateLike,
    n_jobs: int | None,
) -> tuple[float, CVCalibration | None]:
    if sampling_fraction is not None:
        return _validate_fraction(sampling_fraction), None
    calibration = calibrate_cross_validation(
        s, random_state=random_state, n_jobs=n_jobs
    )
    return _validate_fraction(calibration.sampling_fraction), calibration


def _spectral_cutoff(calibration: CVCalibration | None) -> int | None:
    return None if calibration is None else int(calibration.spectral_cutoff)


def _cv_seeds(
    random_state: RandomStateLike, n_repeats: int, n_folds: int, n_ranks: int
) -> tuple[np.ndarray, np.ndarray]:
    seeds = seed_stream(random_state)
    split_seeds = np.fromiter(seeds, dtype=np.uint32, count=n_repeats)
    fit_seeds = np.fromiter(seeds, dtype=np.uint32, count=n_repeats * n_folds * n_ranks)
    return split_seeds, fit_seeds.reshape(n_repeats, n_folds, n_ranks)


def _entrywise_splits(
    observed: np.ndarray,
    fold_fraction: float,
    n_folds: int,
    split_seeds: np.ndarray,
) -> list[_CVSplit]:
    splits: list[_CVSplit] = []
    for repeat, seed in enumerate(split_seeds):
        rng = check_random_state(int(seed))
        fold_sample = _sample_fold_entries(observed, fold_fraction, rng)
        for fold, validation in enumerate(_fold_masks(fold_sample, n_folds, rng)):
            train = fold_sample & ~validation
            diag = np.diag_indices_from(train)
            train[diag] = observed[diag]
            splits.append(_CVSplit(repeat, fold, train, validation))
    return splits


def _sample_fold_entries(
    observed: np.ndarray, fold_fraction: float, rng: np.random.RandomState
) -> np.ndarray:
    pairs = _observed_pairs(observed)
    if pairs.size == 0:
        raise ValueError("cross-validation needs observed off-diagonal entries")
    n_keep = max(1, int(fold_fraction * pairs.size))
    kept = rng.choice(pairs, size=n_keep, replace=False)
    return _pairs_to_mask(kept, observed.shape[0])


def _fold_masks(
    fold_sample: np.ndarray, n_folds: int, rng: np.random.RandomState
) -> list[np.ndarray]:
    pairs = _observed_pairs(fold_sample)
    if pairs.size < n_folds:
        raise ValueError(
            f"cross-validation needs at least {n_folds} fold-sampled "
            f"off-diagonal entries; got {pairs.size}"
        )
    shuffled = np.array_split(rng.permutation(pairs), n_folds)
    return [_pairs_to_mask(fold, fold_sample.shape[0]) for fold in shuffled]


def _observed_pairs(mask: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(mask.shape[0], k=1)
    return np.flatnonzero(mask[i, j])


def _pairs_to_mask(pairs: np.ndarray, n: int) -> np.ndarray:
    i, j = np.triu_indices(n, k=1)
    mask = np.zeros((n, n), dtype=bool)
    mask[i[pairs], j[pairs]] = True
    mask[j[pairs], i[pairs]] = True
    return mask


def get_fold_fraction(sampling_fraction: float, n_folds: int, n: int) -> float:
    cap = _max_fold_fraction(n)
    requested = sampling_fraction * n_folds / (n_folds - 1)
    if requested > cap:
        warnings.warn(
            f"Requested fold fraction {requested:.3f} exceeds the cap {cap:.3f}. "
            f"Each fold will train at {_training_fraction(cap, n_folds):.3f} "
            f"instead of {sampling_fraction:.3f}.",
            RuntimeWarning,
            stacklevel=3,
        )
    return min(requested, cap)


def _training_fraction(fold_fraction: float, n_folds: int) -> float:
    return float(fold_fraction * (n_folds - 1) / n_folds)


def _max_fold_fraction(
    n: int, floor: float = 0.95, holdout_budget: int = 2000
) -> float:
    n_pairs = n * (n - 1) / 2
    if n_pairs <= 0:
        return floor
    return float(max(floor, 1.0 - holdout_budget / n_pairs))


def _cross_validate_ranks(
    s: np.ndarray,
    ranks: tuple[int, ...],
    fold_fraction: float,
    n_folds: int,
    n_repeats: int,
    random_state: RandomStateLike,
    n_jobs: int | None,
    bounds: tuple[float, float],
    fit_kwargs: dict,
) -> pd.DataFrame:
    split_seeds, fit_seeds = _cv_seeds(random_state, n_repeats, n_folds, len(ranks))
    splits = _entrywise_splits(observation_mask(s), fold_fraction, n_folds, split_seeds)
    return _score_splits(s, splits, ranks, fit_seeds, n_jobs, bounds, fit_kwargs)


def _score_splits(
    s: np.ndarray,
    splits: list[_CVSplit],
    ranks: tuple[int, ...],
    fit_seeds: np.ndarray,
    n_jobs: int | None,
    bounds: tuple[float, float],
    fit_kwargs: dict,
) -> pd.DataFrame:
    jobs = [
        (split, rank, int(fit_seeds[split.repeat, split.fold, i]))
        for split in splits
        for i, rank in enumerate(ranks)
    ]
    n_jobs = n_jobs_for_tasks(n_jobs, len(jobs))
    pin = n_jobs > 1

    if n_jobs == 1:
        scores = [_score_job(s, job, bounds, fit_kwargs, pin) for job in jobs]
    else:
        scores = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_score_job)(s, job, bounds, fit_kwargs, pin) for job in jobs
        )

    return pd.DataFrame(
        [
            {
                "repeat": split.repeat,
                "fold": split.fold,
                "candidate_rank": rank,
                "val_mse": score,
            }
            for (split, rank, _), score in zip(jobs, scores)
        ]
    )


def _score_job(
    s: np.ndarray,
    job: tuple[_CVSplit, int, int],
    bounds: tuple[float, float],
    fit_kwargs: dict,
    pin_threads: bool,
) -> float:
    split, rank, seed = job
    return _fit_and_score(
        s,
        split.train_mask,
        split.validation_mask,
        rank,
        seed,
        bounds,
        fit_kwargs,
        pin_threads,
    )


def _fit_and_score(
    s: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    rank: int,
    seed: int,
    bounds: tuple[float, float],
    fit_kwargs: dict,
    pin_threads: bool,
) -> float:
    limit = threadpool_limits(limits=1) if pin_threads else nullcontext()
    with limit:
        train = np.full(s.shape, np.nan, dtype=np.float64)
        train[train_mask] = s[train_mask]
        est = SRF(
            rank=rank,
            bounds=bounds,
            missing_values=np.nan,
            random_state=seed,
            **fit_kwargs,
        )
        est.fit(train)
        s_hat = est.reconstruct()
        return _validation_mse(s, s_hat, validation_mask)


def _validation_mse(
    s: np.ndarray, s_hat: np.ndarray, validation_mask: np.ndarray
) -> float:
    triu = np.triu_indices(s.shape[0], k=1)
    scored = validation_mask[triu] & np.isfinite(s[triu]) & np.isfinite(s_hat[triu])
    if not scored.any():
        return float("nan")
    residual = s[triu][scored] - s_hat[triu][scored]
    return float(np.mean(residual * residual))


def _summarize(fold_scores: pd.DataFrame) -> pd.DataFrame:
    rank_scores = (
        fold_scores.groupby("candidate_rank", as_index=False)
        .agg(
            val_mse_mean=("val_mse", "mean"),
            val_mse_std=("val_mse", "std"),
            n_fold_scores=("val_mse", "count"),
        )
        .sort_values("candidate_rank", kind="stable")
        .reset_index(drop=True)
    )
    rank_scores["val_mse_sem"] = rank_scores["val_mse_std"] / np.sqrt(
        rank_scores["n_fold_scores"]
    )
    return rank_scores


def _best_rank(rank_scores: pd.DataFrame) -> int:
    valid = rank_scores["val_mse_mean"].notna()
    if not valid.any():
        raise ValueError("all cross-validation scores are NaN")
    best = rank_scores.loc[valid, "val_mse_mean"].idxmin()
    return int(rank_scores.loc[best, "candidate_rank"])
