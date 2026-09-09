"""Similarity-based Representation Factorization (SRF).

Reference
---------
Mahner, F.P.*, Lam, K.C.*, Pereira, F. & Hebart, M.N. (2026). Revealing the
core dimensions underlying representations in brains, behavior and AI.
arXiv:2605.26921. (* equal contribution). https://arxiv.org/abs/2605.26921
"""

# Author: Florian P. Mahner
# License: MIT

from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.linalg.blas import dsymm
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import trange
from sklearn.utils.validation import (
    check_is_fitted,
    check_symmetric,
    check_array,
)

try:
    from sklearn.utils.validation import validate_data
except ImportError:
    def validate_data(estimator, x, reset=True, **kwargs):
        if "ensure_all_finite" in kwargs:
            kwargs["force_all_finite"] = kwargs.pop("ensure_all_finite")
        out = check_array(x, **kwargs)
        if reset:
            estimator.n_features_in_ = out.shape[1]
        return out

try:
    from sklearn.utils._param_validation import Interval, StrOptions, Integral, Real
except ImportError:
    Interval = StrOptions = Integral = Real = None

from ._common import is_nan_marker, observation_mask


logger = logging.getLogger(__name__)


def _frobenius_residual(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Compute ||X - WW'||_F and ||WW'||_F without forming the n x n product.

    Parameters
    ----------
    x : ndarray of shape (n, n)
        Symmetric matrix
    w : ndarray of shape (n, r)
        Factor matrix

    Returns
    -------
    reconstruction_err : float
        ||X - WW'||_F, Frobenius distance between X and its approximation
    approx_norm : float
        ||WW'||_F, Frobenius norm of the low-rank approximation
    """
    xw = dsymm(1.0, x, w)
    wtw = w.T @ w
    ss_x = np.einsum("ij,ij->", x, x)
    ss_xw = np.einsum("ij,ij->", xw, w)
    ss_wwt = np.einsum("ij,ij->", wtw, wtw)
    reconstruction_err = np.sqrt(max(0.0, ss_x - 2.0 * ss_xw + ss_wwt))
    approx_norm = np.sqrt(ss_wwt)
    return reconstruction_err, approx_norm


def _initialize_w(
    x: np.ndarray,
    rank: int,
    method: str = "random_sqrt",
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Initialize the embedding matrix W.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_samples)
        Symmetric similarity matrix (used for scaling)
    rank : int
        Number of dimensions
    method : {'random', 'random_sqrt'}, default='random_sqrt'
        Initialization strategy:

        - 'random' : Uniform [0, 0.1], simple baseline
        - 'random_sqrt' : Uniform scaled by sqrt(X.mean() / rank)
    random_state : int, RandomState or None
        Controls random number generation

    Returns
    -------
    w : ndarray of shape (n_samples, rank)
        Non-negative embedding matrix
    """
    rng = np.random.RandomState(random_state)
    n_samples = x.shape[0]

    if method == "random":
        w = 0.1 * rng.rand(n_samples, rank)
    elif method == "random_sqrt":
        avg = np.sqrt(x.mean() / rank)
        w = rng.rand(n_samples, rank) * avg
    else:
        raise ValueError(
            f"Invalid initialization method '{method}'. "
            f"Choose from: 'random', 'random_sqrt'"
        )

    return w


def _validate_bounds(val) -> None:
    """Raise ValueError if lower bound exceeds upper bound."""
    if val is None:
        return
    lo, hi = val
    if lo is not None and hi is not None and lo > hi:
        raise ValueError("Lower bound cannot be greater than upper bound")


def _get_w_solver() -> tuple[callable, str]:
    """Return the best available solver for the W update step.

    Tries the compiled Cython implementation first, falls back to the
    pure-Python solver with a warning.

    Returns
    -------
    solver : callable
        The update_w function to use
    source : str
        ``'cython'`` or ``'python'``
    """
    try:
        from ._bsum import update_w_blas_blocked

        return update_w_blas_blocked, "cython"
    except ImportError:
        import warnings

        warnings.warn(
            "Cython implementation not available. Using Python implementation "
            "which is significantly slower. Compile Cython to improve performance.",
            RuntimeWarning,
            stacklevel=2,
        )
        from ._bsum import update_w

        return update_w, "python"


_w_solver, _w_solver_backend = _get_w_solver()


def update_v_(
    observed_mask: np.ndarray,
    x: np.ndarray,
    x_hat: np.ndarray,
    lam: np.ndarray,
    rho: float,
    bound_min: float,
    bound_max: float,
    v: np.ndarray,
) -> None:
    """Update the auxiliary variable V for missing data handling.

    V is an auxiliary variable that decouples the data-fitting term
    from the factorization constraint.

    For unobserved entries: v_ij = x_hat_ij - lam_ij / rho.
    For observed entries: v_ij = (x_ij + rho * x_hat_ij - lam_ij) / (1 + rho).

    Parameters
    ----------
    observed_mask : ndarray of shape (n, n)
        Binary mask, True where similarities are observed
    x : ndarray of shape (n, n)
        Similarity matrix
    x_hat : ndarray of shape (n, n)
        Current reconstruction WW'
    lam : ndarray of shape (n, n)
        Dual variables enforcing agreement between V and WW'
    rho : float
        Penalty weight for the constraint V = WW'
    bound_min : float or None
        Lower bound on V entries
    bound_max : float or None
        Upper bound on V entries
    v : ndarray of shape (n, n)
        Auxiliary variable, updated in place
    """
    v[:] = lam
    v /= -rho
    v += x_hat

    v[observed_mask] = (
        x[observed_mask] + rho * x_hat[observed_mask] - lam[observed_mask]
    ) / (1.0 + rho)

    if bound_min is not None or bound_max is not None:
        np.clip(v, bound_min, bound_max, out=v)

    # Enforce symmetry
    v += v.T
    v *= 0.5


def update_lambda_(
    lam: np.ndarray, v: np.ndarray, x_hat: np.ndarray, rho: float
) -> None:
    """Update dual variables via dual ascent: lam += rho * (V - WW').

    Parameters
    ----------
    lam : ndarray of shape (n, n)
        Dual variables, updated in place
    v : ndarray of shape (n, n)
        Auxiliary variable
    x_hat : ndarray of shape (n, n)
        Current reconstruction WW'
    rho : float
        Penalty weight for the constraint V = WW'
    """
    lam += rho * (v - x_hat)


class SRF(TransformerMixin, BaseEstimator):
    """Similarity-based Representation Factorization.

    Factorizes a symmetric similarity matrix S into WW' where W >= 0.
    Handles missing entries and can enforce bounds on the reconstructed
    similarities.

    The objective is:
        min_{W>=0, V} ||M * (S - V)||^2_F + rho/2 * ||V - WW'||^2_F
    where M is the observation mask.

    Parameters
    ----------
    rank : int, default=10
        Number of dimensions in the embedding
    rho : float, default=3.0
        Penalty weight controlling how closely WW' must match V
    max_outer : int, default=30
        Maximum number of outer optimization iterations
    max_inner : int, default=20
        Maximum iterations for the W update per outer iteration
    tol : float, default=1e-4
        Convergence tolerance
    verbose : int, default=0
        Whether to print optimization progress
    init : str, default='random_sqrt'
        Embedding initialization method ('random', 'random_sqrt')
    random_state : int or None, default=None
        Random seed for reproducible initialization
    missing_values : float or None, default=np.nan
        Sentinel value marking missing entries in the similarity matrix
    bounds : tuple of (float, float) or None, default=(None, None)
        (lower, upper) bounds on the reconstructed similarities.
        For example, (0, 1) for similarities normalized to [0, 1].
    check_input : bool, default=True
        Whether to verify that the input matrix and missing mask are
        symmetric. Set to False for large matrices where the input is
        known to be valid, to avoid allocating temporary arrays during
        the symmetry check.

    Attributes
    ----------
    w_ : ndarray of shape (n_samples, rank)
        Learned embedding matrix
    components_ : ndarray of shape (n_samples, rank)
        Alias for w_ (sklearn compatibility)
    n_iter_ : int
        Number of outer iterations performed
    history_ : dict
        Optimization metrics per iteration

    Examples
    --------
    >>> from pysrf import SRF
    >>> model = SRF(rank=10, random_state=42)
    >>> w = model.fit_transform(similarity_matrix)
    >>> reconstruction = w @ w.T

    >>> # With missing data
    >>> similarity_matrix[mask] = np.nan
    >>> model = SRF(rank=10, missing_values=np.nan)
    >>> w = model.fit_transform(similarity_matrix)
    """

    _parameter_constraints = {
        "rank": [Interval(Integral, 1, None, closed="left")],
        "rho": [Interval(Real, 0.0, None, closed="left")],
        "max_outer": [Interval(Integral, 1, None, closed="left")],
        "max_inner": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "init": [StrOptions({"random", "random_sqrt"})],
        "verbose": [Interval(Integral, 0, None, closed="left")],
        "random_state": ["random_state"],  # sklearn's special validator
        "missing_values": [None, Real, np.nan],
        "bounds": [None, tuple],
        "check_input": ["boolean"],
    }
    _solver = staticmethod(_w_solver)

    def _validate_params(self):
        return None

    def __init__(
        self,
        rank: int = 10,
        rho: float = 3.0,
        max_outer: int = 30,
        max_inner: int = 20,
        tol: float = 1e-4,
        verbose: int = 0,
        init: str = "random_sqrt",
        random_state: int | None = None,
        missing_values: float | None = np.nan,
        bounds: tuple[float, float] | None = (None, None),
        check_input: bool = True,
    ) -> None:
        self.rank = rank
        self.rho = rho
        self.max_outer = max_outer
        self.max_inner = max_inner
        self.tol = tol
        self.verbose = verbose
        self.init = init
        self.random_state = random_state
        self.missing_values = missing_values
        self.bounds = bounds
        self.check_input = check_input

    def _tolerance(self, *norms):
        """Adaptive convergence tolerance (Boyd et al., 2010, Sec. 3.3.1)."""
        return self.n_features_in_ * self.tol + self.tol * max(norms)

    def _compute_metrics(
        self,
        x: np.ndarray,
        v: np.ndarray,
        x_hat: np.ndarray,
        lam: np.ndarray,
        primal_residual: np.ndarray,
        v_old: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute optimization metrics for monitoring and convergence.

        Convergence follows Boyd et al. (2010), Section 3.3.1, using
        adaptive primal/dual tolerances that scale with the problem.

        Parameters
        ----------
        x : ndarray of shape (n, n)
            Similarity matrix
        v : ndarray of shape (n, n)
            Auxiliary variable
        x_hat : ndarray of shape (n, n)
            Current reconstruction WW'
        lam : ndarray of shape (n, n)
            Dual variables
        primal_residual : ndarray of shape (n, n)
            V - WW'
        v_old : ndarray of shape (n, n) or None
            Previous V, used to compute the dual residual

        Returns
        -------
        metrics : dict[str, float]
            Keys: total_objective, data_fit, penalty, lagrangian,
            rec_error, evar, primal_residual, dual_residual, converged
        """
        observed_mask = self._observed_mask

        data_fit = np.sum((x[observed_mask] - v[observed_mask]) ** 2)

        penalty_term = np.einsum("ij,ij->", primal_residual, primal_residual)
        penalty = (self.rho / 2.0) * penalty_term

        lagrangian = np.einsum("ij,ij->", lam, primal_residual)
        total_obj = data_fit + penalty + lagrangian

        rec_ss = np.sum((x[observed_mask] - x_hat[observed_mask]) ** 2)
        rec_error = np.sqrt(rec_ss)

        if self._total_var > 0:
            evar = 1.0 - rec_ss / self._total_var
        else:
            evar = 0.0

        primal_res = np.sqrt(penalty_term)
        if v_old is not None:
            dual_res = self.rho * np.sqrt(np.einsum("ij,ij->", v - v_old, v - v_old))
        else:
            dual_res = np.inf

        eps_pri = self._tolerance(np.linalg.norm(v), np.linalg.norm(x_hat))
        eps_dual = self._tolerance(np.linalg.norm(lam))
        converged = primal_res <= eps_pri and dual_res <= eps_dual

        return {
            "total_objective": total_obj,
            "data_fit": data_fit,
            "penalty": penalty,
            "lagrangian": lagrangian,
            "rec_error": rec_error,
            "evar": evar,
            "primal_residual": primal_res,
            "dual_residual": dual_res,
            "converged": converged,
        }

    def _fit_complete_data(self, x: np.ndarray) -> SRF:
        """Fit with complete data (no missing values).

        When all similarities are observed, no auxiliary variables are
        needed. Convergence is checked directly on ||X - WW'||_F.
        """
        w = _initialize_w(x, self.rank, self.init, self.random_state)
        history = defaultdict(list)

        n = x.shape[0]
        x_norm = np.linalg.norm(x, "fro")
        total_var = np.var(x)

        pbar = trange(
            1,
            self.max_outer + 1,
            disable=not self.verbose,
            desc="SRF",
            mininterval=10.0 if not sys.stderr.isatty() else 0.1,
        )
        for i in pbar:
            w = self._solver(x, w, max_iter=self.max_inner, tol=self.tol)

            reconstruction_err, approx_norm = _frobenius_residual(x, w)

            mse = reconstruction_err**2 / (n * n)
            evar = 1.0 - mse / total_var if total_var > 0 else 0.0

            history["rec_error"].append(reconstruction_err)
            history["evar"].append(evar)
            history["primal_residual"].append(reconstruction_err)

            pbar.set_postfix(
                rec_error=f"{reconstruction_err:.3f}",
                evar=f"{evar:.3f}",
            )

            if reconstruction_err <= self._tolerance(x_norm, approx_norm):
                logger.info("Converged at iteration %d", i)
                break

        self._store_results(w, i, history)

        return self

    def _fit_missing_data(self, x: np.ndarray) -> SRF:
        """Fit with missing data.

        Introduces auxiliary variable V and dual variables to decouple
        the data-fitting term from the factorization, iterating between
        updating W, V, and the dual variables until convergence.
        """
        bound_min, bound_max = self.bounds
        history = defaultdict(list)

        w = _initialize_w(x, self.rank, self.init, self.random_state)
        lam = np.zeros_like(x)

        v = x.copy()
        x_hat = np.empty_like(x)
        np.dot(w, w.T, out=x_hat)

        v_old = np.empty_like(x)
        target = np.empty_like(x)
        primal_residual = np.empty_like(x)

        pbar = trange(
            1,
            self.max_outer + 1,
            disable=not self.verbose,
            desc="SRF",
            mininterval=10.0 if not sys.stderr.isatty() else 0.1,
        )
        for i in pbar:
            v_old[:] = v

            target[:] = lam
            target /= self.rho
            target += v
            w = self._solver(target, w, max_iter=self.max_inner, tol=self.tol)
            # Avoid temporary by writing directly into pre-allocated buffer
            np.dot(w, w.T, out=x_hat)

            update_v_(
                self._observed_mask, x, x_hat, lam, self.rho, bound_min, bound_max, v
            )
            primal_residual[:] = v
            primal_residual -= x_hat
            target[:] = primal_residual
            target *= self.rho
            lam += target

            metrics = self._compute_metrics(x, v, x_hat, lam, primal_residual, v_old)
            for key, value in metrics.items():
                history[key].append(value)

            pbar.set_postfix(
                obj=f"{metrics['total_objective']:.3f}",
                rec_error=f"{metrics['rec_error']:.3f}",
                evar=f"{metrics['evar']:.3f}",
            )

            if i > 1 and metrics["converged"]:
                logger.info("Converged at iteration %d", i)
                break

        self._store_results(w, i, history)

        return self

    def _store_results(self, w: np.ndarray, n_iter: int, history: dict) -> None:
        """Set fitted attributes after optimization."""
        self.w_ = w
        self.components_ = w
        self.n_iter_ = n_iter
        self.history_ = history

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> SRF:
        """Fit the model to a similarity matrix.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_samples)
            Symmetric similarity matrix. Missing values should be marked
            according to the ``missing_values`` parameter.
        y : Ignored
            Not used, present for sklearn API compatibility.

        Returns
        -------
        self : SRF
            Fitted estimator.
        """
        self._validate_params()
        _validate_bounds(self.bounds)

        x = validate_data(
            self,
            x,
            reset=True,
            ensure_all_finite=(
                "allow-nan" if is_nan_marker(self.missing_values) else True
            ),
            ensure_2d=True,
            dtype=np.float64,
            copy=True,
        )

        self._observed_mask = observation_mask(x, self.missing_values)
        if not np.any(self._observed_mask):
            raise ValueError(
                "No observed entries found in the data. All values are missing."
            )

        x[~self._observed_mask] = 0.0
        if self.check_input:
            check_symmetric(self._observed_mask, raise_exception=True)
            x = check_symmetric(x, raise_exception=True, tol=1e-10)

        n_observed = np.count_nonzero(self._observed_mask)
        if n_observed > 0:
            observed_mean = np.sum(x[self._observed_mask]) / n_observed
            self._total_var = np.sum((x[self._observed_mask] - observed_mean) ** 2)
        else:
            self._total_var = 0.0

        if self.verbose:
            n_threads = os.environ.get("OMP_NUM_THREADS", "1")
            logger.info(
                "Using %s BLAS thread(s). For faster fitting, set "
                "OMP_NUM_THREADS before importing pysrf "
                "(e.g. export OMP_NUM_THREADS=4)",
                n_threads,
            )

        if n_observed == self._observed_mask.size:
            return self._fit_complete_data(x)
        else:
            return self._fit_missing_data(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Return the learned embedding matrix W.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_samples)
            Ignored, present for sklearn API compatibility.

        Returns
        -------
        w : ndarray of shape (n_samples, rank)
            Embedding matrix
        """
        check_is_fitted(self)
        return self.w_

    def fit_transform(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit the model and return the learned embedding matrix W.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_samples)
            Symmetric similarity matrix
        y : Ignored
            Not used, present for sklearn API compatibility.

        Returns
        -------
        w : ndarray of shape (n_samples, rank)
            Embedding matrix
        """
        return self.fit(x, y).transform(x)

    def reconstruct(self, w: np.ndarray | None = None) -> np.ndarray:
        """Reconstruct the similarity matrix as WW'.

        Parameters
        ----------
        w : ndarray of shape (n_samples, rank) or None
            Embedding matrix. If None, uses the fitted embedding.

        Returns
        -------
        s_hat : ndarray of shape (n_samples, n_samples)
            Reconstructed similarity matrix
        """
        if w is None:
            check_is_fitted(self)
            w = self.w_

        return w @ w.T

    def score(self, x: np.ndarray, y: np.ndarray | None = None) -> float:
        """Negative MSE on observed entries (higher is better).

        Parameters
        ----------
        x : array-like of shape (n_samples, n_samples)
            Symmetric similarity matrix. Missing values should be marked
            according to the ``missing_values`` parameter.
        y : Ignored
            Not used, present for sklearn API compatibility.

        Returns
        -------
        score : float
            Negative mean squared reconstruction error on observed entries.
        """
        check_is_fitted(self)

        x = validate_data(
            self,
            x,
            reset=False,
            ensure_2d=True,
            dtype=np.float64,
            ensure_all_finite="allow-nan"
            if is_nan_marker(self.missing_values)
            else True,
        )
        observed_mask = observation_mask(x, self.missing_values)
        reconstruction = self.reconstruct()
        mse = np.mean((x[observed_mask] - reconstruction[observed_mask]) ** 2)
        return -mse
