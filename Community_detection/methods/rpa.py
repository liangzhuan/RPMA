"""Drop-in RPMA method using the corrected Cayley--SMW algorithm.

Place this file at ``Community_detection/methods/rpa.py``.

It exposes the same ``rpa(...)`` interface expected by
``experiments.image_all_methods`` and by the RPMA->SymNMF method.


The optimization problem is

    min_X  -2 <A, X> + lam * sum_ij huber_delta(X_ij)
    s.t.   X = X.T, X^2 = X, rank(X) = K.

The variable is represented as X = U U.T with U.T U = I_K.  Each update
uses the low-rank Sherman--Morrison--Woodbury form of the Cayley transform,
so the rank-K projection constraint is preserved without an eigendecomposition
at every iteration.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


__all__ = [
    "rpa",
    "rpma",
    "objective",
    "huber_value",
    "huber_grad",
]


def _check_delta(delta: float) -> float:
    delta = float(delta)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be a finite positive number.")
    return delta


def huber_value(X: ArrayLike, delta: float) -> float:
    """Return the entrywise Huber penalty summed over all entries of ``X``."""
    delta = _check_delta(delta)
    X_arr = np.asarray(X, dtype=float)
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X contains NaN or infinite values.")

    abs_x = np.abs(X_arr)
    small = abs_x <= delta
    values = np.empty_like(abs_x)
    values[small] = X_arr[small] ** 2 / (2.0 * delta)
    values[~small] = abs_x[~small] - 0.5 * delta
    return float(np.sum(values))


def huber_grad(X: ArrayLike, delta: float) -> FloatArray:
    """Return the entrywise gradient of the Huber penalty."""
    delta = _check_delta(delta)
    X_arr = np.asarray(X, dtype=float)
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X contains NaN or infinite values.")

    grad = np.empty_like(X_arr)
    small = np.abs(X_arr) <= delta
    grad[small] = X_arr[small] / delta
    grad[~small] = np.sign(X_arr[~small])
    return grad


def _validate_square_matrix(A: ArrayLike) -> FloatArray:
    A_arr = np.asarray(A, dtype=float)
    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
        raise ValueError("A must be a square two-dimensional array.")
    if A_arr.shape[0] == 0:
        raise ValueError("A must be non-empty.")
    if not np.all(np.isfinite(A_arr)):
        raise ValueError("A contains NaN or infinite values.")
    return A_arr


def objective(X: ArrayLike, A: ArrayLike, lam: float, delta: float) -> float:
    """Evaluate the RPMA objective.

    Only the symmetric part of ``A`` contributes on the feasible set because
    every feasible ``X`` is symmetric.  Symmetrizing here also makes this
    standalone helper consistent with the manifold-gradient implementation.
    """
    X_arr = _validate_square_matrix(X)
    A_arr = _validate_square_matrix(A)
    if X_arr.shape != A_arr.shape:
        raise ValueError("X and A must have the same shape.")

    lam = float(lam)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError("lam must be a finite non-negative number.")

    A_sym = 0.5 * (A_arr + A_arr.T)
    return float(-2.0 * np.sum(A_sym * X_arr) + lam * huber_value(X_arr, delta))


def _objective_validated(X: FloatArray, A_sym: FloatArray, lam: float, delta: float) -> float:
    """Fast internal objective evaluation after inputs have been validated."""
    return float(-2.0 * np.sum(A_sym * X) + lam * huber_value(X, delta))


def _orthonormalize(U: FloatArray) -> FloatArray:
    """Return an orthonormal basis spanning the columns of ``U``."""
    Q, R = np.linalg.qr(U, mode="reduced")
    if np.min(np.abs(np.diag(R))) <= np.finfo(float).eps:
        raise ValueError("The supplied initial basis is rank deficient.")
    return Q


def rpa(
    A: ArrayLike,
    K: int,
    lam: float = 0.04,
    delta: float = 2e-3,
    tau_max: float = 1.0,
    beta: float = 0.5,
    sigma: float = 1e-4,
    tol: float = 1e-8,
    max_iter: int = 1000,
    eig_init: bool = True,
    return_history: bool = False,
    verbose: bool = False,
    *,
    U0: Optional[ArrayLike] = None,
    tau_min: float = 1e-14,
    max_backtracking: int = 80,
    reorth_tol: float = 1e-10,
) -> Union[FloatArray, Tuple[FloatArray, FloatArray, list[float]]]:
    """Solve RPMA by the Cayley--SMW Riemannian-gradient method.

    Parameters
    ----------
    A : array_like, shape (n, n)
        Similarity matrix.  The symmetric part ``(A + A.T)/2`` is used.
    K : int
        Target rank / number of communities, with ``1 <= K <= n``.
    lam : float, default=0.04
        Non-negative regularization parameter.
    delta : float, default=2e-3
        Positive Huber threshold.
    tau_max : float, default=1.0
        Initial trial step size in each Armijo line search.
    beta : float, default=0.5
        Backtracking contraction factor in ``(0, 1)``.
    sigma : float, default=1e-4
        Armijo parameter in ``(0, 1)``.
    tol : float, default=1e-8
        Stopping tolerance for ``||(I-UU.T) gradF(X) U||_F``.
    max_iter : int, default=1000
        Maximum number of outer iterations.
    eig_init : bool, default=True
        Use the leading-K eigenvectors of ``A`` when ``U0`` is not supplied.
        If False, use the first K canonical basis vectors.  Unlike the original
        code, no unnecessary eigendecomposition is performed in this branch.
    return_history : bool, default=False
        If True, return ``(X, U, gradient_norm_history)``.
    verbose : bool, default=False
        Print iteration diagnostics.
    U0 : array_like, shape (n, K), optional
        User-supplied initial basis.  It is orthonormalized before use.
    tau_min : float, default=1e-14
        Smallest permitted trial step size.
    max_backtracking : int, default=80
        Maximum number of backtracking reductions per outer iteration.
    reorth_tol : float, default=1e-10
        Reorthonormalize a candidate only when its orthogonality error exceeds
        this tolerance.  The exact Cayley update is already orthogonal; this is
        only a floating-point safeguard.

    Returns
    -------
    X : ndarray, shape (n, n)
        Rank-K orthogonal projection matrix.
    U : ndarray, shape (n, K), optional
        Orthonormal basis, returned only when ``return_history=True``.
    history : list of float, optional
        Norms of the scaled Grassmann gradient, returned only when
        ``return_history=True``.

    Notes
    -----
    The quantity

        Xi = (I - U U.T) gradF(U U.T) U

    is one half of the usual Grassmann gradient of ``F(U U.T)`` under a common
    metric convention.  This constant factor is absorbed into the step size.
    Along the Cayley curve used below,

        d/dtau F(X(tau))|_{tau=0} = -2 ||Xi||_F^2,

    so the Armijo test in this implementation is consistent with the actual
    descent derivative.
    """
    A_arr = _validate_square_matrix(A)
    n = A_arr.shape[0]

    if not isinstance(K, (int, np.integer)) or not (1 <= int(K) <= n):
        raise ValueError(f"K must be an integer satisfying 1 <= K <= {n}.")
    K = int(K)

    lam = float(lam)
    delta = _check_delta(delta)
    tau_max = float(tau_max)
    beta = float(beta)
    sigma = float(sigma)
    tol = float(tol)
    tau_min = float(tau_min)
    reorth_tol = float(reorth_tol)

    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError("lam must be a finite non-negative number.")
    if not np.isfinite(tau_max) or tau_max <= 0.0:
        raise ValueError("tau_max must be finite and positive.")
    if not np.isfinite(tau_min) or tau_min <= 0.0 or tau_min >= tau_max:
        raise ValueError("tau_min must satisfy 0 < tau_min < tau_max.")
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must lie in (0, 1).")
    if not (0.0 < sigma < 1.0):
        raise ValueError("sigma must lie in (0, 1).")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive.")
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) <= 0:
        raise ValueError("max_iter must be a positive integer.")
    if not isinstance(max_backtracking, (int, np.integer)) or int(max_backtracking) <= 0:
        raise ValueError("max_backtracking must be a positive integer.")
    if not np.isfinite(reorth_tol) or reorth_tol <= 0.0:
        raise ValueError("reorth_tol must be finite and positive.")

    max_iter = int(max_iter)
    max_backtracking = int(max_backtracking)

    # The feasible variable X is symmetric, so the antisymmetric part of A is
    # irrelevant to the objective.  Using A directly when it is nonsymmetric
    # would make gradF nonsymmetric and would give an incorrect manifold step.
    A_sym = 0.5 * (A_arr + A_arr.T)

    # Initialization.  The original code always called eigh(A), even when
    # eig_init=False; that unnecessary O(n^3) operation has been removed.
    if U0 is not None:
        U_init = np.asarray(U0, dtype=float)
        if U_init.shape != (n, K):
            raise ValueError(f"U0 must have shape {(n, K)}, got {U_init.shape}.")
        if not np.all(np.isfinite(U_init)):
            raise ValueError("U0 contains NaN or infinite values.")
        U = _orthonormalize(U_init)
    elif eig_init:
        _, eigvec = np.linalg.eigh(A_sym)
        U = eigvec[:, -K:]
    else:
        U = np.zeros((n, K), dtype=float)
        U[:K, :] = np.eye(K, dtype=float)

    history: list[float] = []

    J = np.block(
        [
            [np.zeros((K, K)), np.eye(K)],
            [-np.eye(K), np.zeros((K, K))],
        ]
    )
    identity_2k = np.eye(2 * K)
    identity_k = np.eye(K)

    converged = False
    line_search_failed = False

    for iteration in range(max_iter):
        X = U @ U.T
        f_old = _objective_validated(X, A_sym, lam, delta)

        # Euclidean gradient of the selected ambient extension
        # F(X) = -2 <A, X> + lam * R(X).
        grad_f = -2.0 * A_sym + lam * huber_grad(X, delta)

        # Scaled Grassmann gradient / first-order stationarity residual.
        grad_u = grad_f @ U
        Xi = grad_u - U @ (U.T @ grad_u)
        grad_norm = float(np.linalg.norm(Xi, ord="fro"))
        history.append(grad_norm)

        if verbose:
            orth_error = np.linalg.norm(U.T @ U - identity_k, ord="fro")
            print(
                f"Iter {iteration:4d} | f = {f_old:.10e} | "
                f"grad = {grad_norm:.3e} | orth = {orth_error:.3e}"
            )

        if grad_norm <= tol:
            converged = True
            break

        Y = np.hstack((Xi, U))

        # These quantities are independent of the trial step and should not be
        # recomputed inside the backtracking loop.
        yty = Y.T @ Y
        rhs = J @ (Y.T @ U)

        tau = tau_max
        accepted = False
        U_candidate: Optional[FloatArray] = None

        for _ in range(max_backtracking):
            if tau < tau_min:
                break

            # Low-rank SMW form of the exact Cayley update.
            M = identity_2k + 0.5 * tau * (J @ yty)
            try:
                Z = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                tau *= beta
                continue

            trial_u = U - tau * (Y @ Z)

            # Exact arithmetic preserves orthogonality.  Reorthonormalize only
            # if floating-point drift is larger than the requested tolerance.
            orth_error = np.linalg.norm(trial_u.T @ trial_u - identity_k, ord="fro")
            if not np.isfinite(orth_error):
                tau *= beta
                continue
            if orth_error > reorth_tol:
                trial_u = _orthonormalize(trial_u)

            trial_x = trial_u @ trial_u.T
            f_new = _objective_validated(trial_x, A_sym, lam, delta)

            # Since the derivative at zero is -2 ||Xi||_F^2, this Armijo test
            # is valid for sigma in (0, 1).
            if f_new <= f_old - sigma * tau * grad_norm**2:
                accepted = True
                U_candidate = trial_u
                break

            tau *= beta

        if not accepted or U_candidate is None:
            line_search_failed = True
            warnings.warn(
                "RPMA line search failed before reaching the stationarity "
                "tolerance; the last feasible iterate is being returned.",
                RuntimeWarning,
                stacklevel=2,
            )
            if verbose:
                print("Line search failed; returning the last feasible iterate.")
            break

        U = U_candidate

    if verbose and not converged and not line_search_failed and len(history) >= max_iter:
        print("Maximum iteration count reached before convergence.")

    # Final safeguard against accumulated roundoff.
    final_orth_error = np.linalg.norm(U.T @ U - identity_k, ord="fro")
    if final_orth_error > reorth_tol:
        U = _orthonormalize(U)

    X = U @ U.T

    if return_history:
        return X, U, history
    return X


# The paper calls the method RPMA; keep both names for backward compatibility.
rpma = rpa
