from __future__ import annotations

from collections.abc import Iterator
import warnings
from typing import Union

import numpy as np
from joblib import cpu_count as joblib_cpu_count

RandomStateLike = Union[
    None,
    int,
    np.integer,
    np.random.SeedSequence,
    np.random.Generator,
    np.random.RandomState,
]


def as_seed_sequence(random_state: RandomStateLike) -> np.random.SeedSequence:
    if isinstance(random_state, np.random.SeedSequence):
        return random_state
    if random_state is None or isinstance(random_state, (int, np.integer)):
        return np.random.SeedSequence(random_state)
    if isinstance(random_state, np.random.Generator):
        seed_seq = getattr(random_state.bit_generator, "seed_seq", None)
        if isinstance(seed_seq, np.random.SeedSequence):
            return seed_seq
        return np.random.SeedSequence(int(random_state.integers(0, 2**31 - 1)))
    if isinstance(random_state, np.random.RandomState):
        return np.random.SeedSequence(int(random_state.randint(0, 2**31 - 1)))
    raise TypeError(
        f"random_state must be None, int, SeedSequence, Generator, or "
        f"RandomState; got {type(random_state).__name__}"
    )


def make_rng(random_state: RandomStateLike) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        seed = int(random_state.randint(0, 2**32 - 1, dtype=np.uint32))
        return np.random.default_rng(seed)
    return np.random.default_rng(as_seed_sequence(random_state))


def seed_stream(random_state: RandomStateLike) -> Iterator[int]:
    seed_sequence = as_seed_sequence(random_state)
    while True:
        child = seed_sequence.spawn(1)[0]
        yield int(child.generate_state(1)[0])


def spawn_ints(random_state: RandomStateLike, n: int) -> np.ndarray:
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return np.fromiter(seed_stream(random_state), dtype=np.uint32, count=n)


def is_nan_marker(value: float | None) -> bool:
    return value is np.nan or (isinstance(value, float) and np.isnan(value))


def observation_mask(
    x: np.ndarray, missing_values: float | None = np.nan
) -> np.ndarray:
    if missing_values is None or is_nan_marker(missing_values):
        return np.isfinite(x)
    return np.not_equal(x, missing_values)


def replace_missing_with_nan(
    x: np.ndarray,
    missing_values: float | None = np.nan,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if missing_values is not None and not is_nan_marker(missing_values):
        x = x.copy()
        x[np.equal(x, missing_values)] = np.nan
    return x


def validate_n_jobs(n_jobs: int | None) -> None:
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")
    if n_jobs is not None:
        int(n_jobs)


def n_jobs_for_tasks(n_jobs: int | None, n_tasks: int) -> int:
    validate_n_jobs(n_jobs)
    if n_jobs is None:
        return 1
    n_tasks = max(1, int(n_tasks))
    n_jobs = int(n_jobs)
    if n_jobs < 0:
        n_jobs = max(1, joblib_cpu_count() + 1 + n_jobs)
    return max(1, min(n_jobs, n_tasks))


def symmetrize_observations(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(np.stack([x, x.T]), axis=0)
    diagonal = np.diag(out).copy()
    diagonal[~np.isfinite(diagonal)] = 0.0
    np.fill_diagonal(out, diagonal)
    return out
