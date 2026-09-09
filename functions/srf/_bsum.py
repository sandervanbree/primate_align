"""
This module implements the BSUM algorithm from
Shi et al. (2016). "Inexact Block Coordinate Descent Methods For Symmetric Nonnegative Matrix Factorization"
"""

import numpy as np
import math


def _solve_quartic_minimization(a: float, b: float, c: float, d: float) -> float:
    """Find x >= 0 minimizing g(x) = a/4 x^4 + b/3 x^3 + c/2 x^2 + d*x.

    Key subroutine in the BSUM algorithm for updating individual
    elements of the factor matrix W. Equation (11) in paper.

    Parameters
    ----------
    a, b, c, d : float
        Polynomial coefficients

    Returns
    -------
    root : float
        Non-negative minimizer of g(x)
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    bb = b * b
    a2 = a * a
    p = (3.0 * a * c - bb) / (3.0 * a2)

    q = (9.0 * a * b * c - 27.0 * a2 * d - 2.0 * b * bb) / (27.0 * a2 * a)

    if c > bb / (3.0 * a):
        delta = math.sqrt(q * q * 0.25 + p * p * p / 27.0)
        root = math.cbrt(q * 0.5 - delta) + math.cbrt(q * 0.5 + delta)
    else:
        tmp = b * bb / (27.0 * a2 * a) - d / a
        root = math.cbrt(tmp)

    return max(0.0, root)


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product via scalar accumulation matching the Cython summation order.

    Parameters
    ----------
    a, b : ndarray of shape (n,)
        Input vectors

    Returns
    -------
    result : float
        Dot product of a and b
    """
    return sum(x * y for x, y in zip(a, b))


def update_w(
    m: np.ndarray,
    w0: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Block successive upper-bound minimization (BSUM) for SymNMF.

    Solves the W-subproblem by minimizing ||M - WW'||^2_F element-wise
    via quartic surrogate functions. See TABLE 1 in Shi et al. (2016),
    "Inexact Block Coordinate Descent Methods For Symmetric Nonnegative
    Matrix Factorization".

    We adopt `m` instead of `x` here for the similarity to align with the
    notation of the original derivations in the paper.

    Parameters
    ----------
    m : ndarray of shape (n, n)
        Target symmetric similarity matrix to factorize
    w0 : ndarray of shape (n, r)
        Initial embedding matrix
    max_iter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-6
        Convergence tolerance

    Returns
    -------
    w : ndarray of shape (n, r)
        Optimized embedding matrix
    """
    w = w0.copy()
    n, r = w.shape
    xtx = w.T @ w
    diag = np.einsum("ij,ij->i", w, w)
    a = 4.0

    for _ in range(max_iter):
        max_delta = 0.0
        for i in range(n):
            for j in range(r):
                old = w[i, j]
                b = 12.0 * old
                c = 4.0 * ((diag[i] - m[i, i]) + xtx[j, j] + old * old)
                d = 4.0 * _dot(w[i], xtx[:, j]) - 4.0 * _dot(m[i], w[:, j])
                new = _solve_quartic_minimization(a, b, c, d)

                delta = new - old
                if abs(delta) > max_delta:
                    max_delta = abs(delta)

                # steps 7 - 13 in TABLE 1
                diag[i] += new * new - old * old
                update_row = delta * w[i]
                xtx[j, :] += update_row
                xtx[:, j] += update_row
                xtx[j, j] += delta * delta
                w[i, j] = new

        if max_delta < tol and tol > 0.0:
            break

    return w
