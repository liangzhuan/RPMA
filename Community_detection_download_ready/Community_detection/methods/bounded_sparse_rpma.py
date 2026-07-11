"""
Bounded sparse RPMA (BS-RPMA).

Place this file at:
    Community_detection/methods/bounded_sparse_rpma.py

Model
-----
For a rank-K orthogonal projector X = U U^T, U^T U = I_K, solve

    min_X  -2 <A, X>
           + lam * sum_ij huber_delta(X_ij)
           + mu  * sum_ij dist(X_ij, [alpha, upper])^2

with the recommended fixed bounds

    alpha = 0,
    upper = 1 / n_k = K / n

for balanced K-class data.

The Huber term is retained to encourage sparsity.  The box penalty jointly
penalizes negative entries and entries larger than the ideal within-class
projector value 1/n_k.  The box coefficient mu is independent of lam and is
intended to be fixed at a large value rather than tuned together with lam.

The optimization uses the corrected Cayley--SMW update, preserving the
rank-K projection constraint throughout the iterations.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "bounded_sparse_rpa",
    "bs_rpma",
    "box_penalty",
    "huber_value",
    "huber_grad",
    "objective",
]


def _validate_square_matrix(A: ArrayLike) -> FloatArray:
    A_arr = np.asarray(A, dtype=np.float64)
    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
        raise ValueError("A must be a square two-dimensional array.")
    if A_arr.shape[0] == 0:
        raise ValueError("A must be non-empty.")
    if not np.all(np.isfinite(A_arr)):
        raise ValueError("A contains NaN or infinite values.")
    return A_arr


def _check_positive(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def huber_value(X: ArrayLike, delta: float) -> float:
    """Entrywise Huber penalty summed over all entries."""
    delta = _check_positive("delta", delta)
    X_arr = np.asarray(X, dtype=np.float64)
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X contains NaN or infinite values.")

    abs_x = np.abs(X_arr)
    small = abs_x <= delta
    values = np.empty_like(X_arr)
    values[small] = X_arr[small] ** 2 / (2.0 * delta)
    values[~small] = abs_x[~small] - 0.5 * delta
    return float(np.sum(values))


def huber_grad(X: ArrayLike, delta: float) -> FloatArray:
    """Entrywise gradient of the Huber penalty."""
    delta = _check_positive("delta", delta)
    X_arr = np.asarray(X, dtype=np.float64)
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X contains NaN or infinite values.")

    grad = np.empty_like(X_arr)
    small = np.abs(X_arr) <= delta
    grad[small] = X_arr[small] / delta
    grad[~small] = np.sign(X_arr[~small])
    return grad


def box_penalty(
    X: ArrayLike,
    alpha: float,
    upper: float,
    mu: float,
    *,
    off_diagonal_only: bool = False,
) -> Tuple[float, FloatArray]:
    """
    Squared-distance penalty to the interval [alpha, upper].

    Penalty:
        mu * sum_ij [min(X_ij-alpha, 0)^2 + max(X_ij-upper, 0)^2]

    If off_diagonal_only=True, diagonal entries are excluded from this
    penalty.  The recommended rank-K balanced model uses all entries.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    alpha = float(alpha)
    upper = float(upper)
    mu = float(mu)

    if not np.isfinite(alpha) or not np.isfinite(upper) or alpha >= upper:
        raise ValueError("Bounds must be finite and satisfy alpha < upper.")
    if not np.isfinite(mu) or mu < 0.0:
        raise ValueError("mu must be a finite non-negative number.")

    lower_violation = np.minimum(X_arr - alpha, 0.0)
    upper_violation = np.maximum(X_arr - upper, 0.0)

    if off_diagonal_only:
        lower_violation = lower_violation.copy()
        upper_violation = upper_violation.copy()
        np.fill_diagonal(lower_violation, 0.0)
        np.fill_diagonal(upper_violation, 0.0)

    loss = mu * float(
        np.sum(lower_violation * lower_violation)
        + np.sum(upper_violation * upper_violation)
    )
    grad = 2.0 * mu * (lower_violation + upper_violation)
    return loss, grad


def objective(
    X: ArrayLike,
    A: ArrayLike,
    lam: float,
    delta: float,
    box_mu: float,
    alpha: float,
    upper: float,
    *,
    off_diagonal_only: bool = False,
) -> float:
    """Evaluate the bounded sparse RPMA objective."""
    X_arr = _validate_square_matrix(X)
    A_arr = _validate_square_matrix(A)
    if X_arr.shape != A_arr.shape:
        raise ValueError("X and A must have the same shape.")

    lam = float(lam)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError("lam must be finite and non-negative.")

    A_sym = 0.5 * (A_arr + A_arr.T)
    box_loss, _ = box_penalty(
        X_arr,
        alpha,
        upper,
        box_mu,
        off_diagonal_only=off_diagonal_only,
    )
    return float(
        -2.0 * np.sum(A_sym * X_arr)
        + lam * huber_value(X_arr, delta)
        + box_loss
    )


def _orthonormalize(U: FloatArray) -> FloatArray:
    Q, R = np.linalg.qr(U, mode="reduced")
    if np.min(np.abs(np.diag(R))) <= np.finfo(float).eps:
        raise ValueError("The supplied initial basis is rank deficient.")
    return Q


def _constraint_diagnostics(
    X: FloatArray,
    alpha: float,
    upper: float,
    *,
    off_diagonal_only: bool,
) -> Dict[str, float]:
    mask = np.ones_like(X, dtype=bool)
    if off_diagonal_only:
        np.fill_diagonal(mask, False)

    values = X[mask]
    lower = np.maximum(alpha - values, 0.0)
    upper_v = np.maximum(values - upper, 0.0)

    return {
        "x_min": float(np.min(X)),
        "x_max": float(np.max(X)),
        "lower_violation_max": float(np.max(lower)) if lower.size else 0.0,
        "upper_violation_max": float(np.max(upper_v)) if upper_v.size else 0.0,
        "lower_violation_ratio": float(np.mean(lower > 1e-12)) if lower.size else 0.0,
        "upper_violation_ratio": float(np.mean(upper_v > 1e-12)) if upper_v.size else 0.0,
        "box_violation_fro": float(np.sqrt(np.sum(lower**2) + np.sum(upper_v**2))),
    }


def bounded_sparse_rpa(
    A: ArrayLike,
    K: int,
    lam: float = 0.07,
    delta: float = 1e-3,
    box_mu: float = 1e5,
    alpha: float = 0.0,
    upper: Optional[float] = None,
    tau_max: float = 1.0,
    backtrack_beta: float = 0.5,
    armijo_sigma: float = 1e-4,
    tol: float = 1e-8,
    max_iter: int = 500,
    eig_init: bool = True,
    return_info: bool = False,
    verbose: bool = False,
    *,
    U0: Optional[ArrayLike] = None,
    tau_min: float = 1e-14,
    max_backtracking: int = 80,
    reorth_tol: float = 1e-10,
    off_diagonal_only: bool = False,
) -> Union[FloatArray, Tuple[FloatArray, FloatArray, Dict[str, object]]]:
    """
    Solve bounded sparse RPMA by the Cayley--SMW method.

    Recommended balanced setting:
        projection rank = K
        alpha = 0
        upper = K / n = 1 / n_k
        box_mu = 1e5 (fixed)

    Notes
    -----
    With an all-entry upper bound upper=K/n, using projection rank K+1 is
    infeasible because tr(X)=K+1 but sum_i X_ii <= n*(K/n)=K.  Therefore this
    implementation is intended for rank K unless off_diagonal_only=True or a
    compatible larger upper bound is explicitly supplied.
    """
    A_arr = _validate_square_matrix(A)
    n = A_arr.shape[0]

    if not isinstance(K, (int, np.integer)) or not (1 <= int(K) <= n):
        raise ValueError(f"K must be an integer satisfying 1 <= K <= {n}.")
    K = int(K)

    lam = float(lam)
    delta = _check_positive("delta", delta)
    box_mu = float(box_mu)
    alpha = float(alpha)
    upper = float(K / n if upper is None else upper)
    tau_max = _check_positive("tau_max", tau_max)
    tau_min = _check_positive("tau_min", tau_min)
    tol = _check_positive("tol", tol)
    reorth_tol = _check_positive("reorth_tol", reorth_tol)
    backtrack_beta = float(backtrack_beta)
    armijo_sigma = float(armijo_sigma)

    if lam < 0.0 or not np.isfinite(lam):
        raise ValueError("lam must be finite and non-negative.")
    if box_mu < 0.0 or not np.isfinite(box_mu):
        raise ValueError("box_mu must be finite and non-negative.")
    if not np.isfinite(alpha) or not np.isfinite(upper) or alpha >= upper:
        raise ValueError("Bounds must satisfy finite alpha < upper.")
    if not (0.0 < backtrack_beta < 1.0):
        raise ValueError("backtrack_beta must lie in (0, 1).")
    if not (0.0 < armijo_sigma < 1.0):
        raise ValueError("armijo_sigma must lie in (0, 1).")
    if tau_min >= tau_max:
        raise ValueError("tau_min must be smaller than tau_max.")
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) <= 0:
        raise ValueError("max_iter must be a positive integer.")
    if not isinstance(max_backtracking, (int, np.integer)) or int(max_backtracking) <= 0:
        raise ValueError("max_backtracking must be a positive integer.")

    max_iter = int(max_iter)
    max_backtracking = int(max_backtracking)
    A_sym = 0.5 * (A_arr + A_arr.T)

    if U0 is not None:
        U_init = np.asarray(U0, dtype=np.float64)
        if U_init.shape != (n, K):
            raise ValueError(f"U0 must have shape {(n, K)}, got {U_init.shape}.")
        if not np.all(np.isfinite(U_init)):
            raise ValueError("U0 contains NaN or infinite values.")
        U = _orthonormalize(U_init)
    elif eig_init:
        _, eigvec = np.linalg.eigh(A_sym)
        U = eigvec[:, -K:]
    else:
        U = np.zeros((n, K), dtype=np.float64)
        U[:K, :] = np.eye(K, dtype=np.float64)

    J = np.block(
        [
            [np.zeros((K, K)), np.eye(K)],
            [-np.eye(K), np.zeros((K, K))],
        ]
    )
    identity_2k = np.eye(2 * K)
    identity_k = np.eye(K)

    grad_history: list[float] = []
    objective_history: list[float] = []
    step_history: list[float] = []

    converged = False
    line_search_failed = False

    for iteration in range(max_iter):
        X = U @ U.T

        box_loss, box_grad = box_penalty(
            X,
            alpha,
            upper,
            box_mu,
            off_diagonal_only=off_diagonal_only,
        )
        sparse_value = huber_value(X, delta)
        f_old = float(
            -2.0 * np.sum(A_sym * X)
            + lam * sparse_value
            + box_loss
        )

        grad_f = (
            -2.0 * A_sym
            + lam * huber_grad(X, delta)
            + box_grad
        )

        grad_u = grad_f @ U
        Xi = grad_u - U @ (U.T @ grad_u)
        grad_norm = float(np.linalg.norm(Xi, ord="fro"))

        grad_history.append(grad_norm)
        objective_history.append(f_old)

        if verbose:
            diagnostics = _constraint_diagnostics(
                X,
                alpha,
                upper,
                off_diagonal_only=off_diagonal_only,
            )
            print(
                f"Iter {iteration:4d} | f={f_old:.10e} | "
                f"grad={grad_norm:.3e} | "
                f"min={diagnostics['x_min']:.3e} | "
                f"max={diagnostics['x_max']:.3e} | "
                f"box_v={diagnostics['box_violation_fro']:.3e}"
            )

        if grad_norm <= tol:
            converged = True
            step_history.append(0.0)
            break

        Y = np.hstack((Xi, U))
        yty = Y.T @ Y
        rhs = J @ (Y.T @ U)

        tau = tau_max
        accepted = False
        U_candidate: Optional[FloatArray] = None

        for _ in range(max_backtracking):
            if tau < tau_min:
                break

            M = identity_2k + 0.5 * tau * (J @ yty)
            try:
                Z = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                tau *= backtrack_beta
                continue

            trial_u = U - tau * (Y @ Z)
            orth_error = np.linalg.norm(
                trial_u.T @ trial_u - identity_k,
                ord="fro",
            )
            if not np.isfinite(orth_error):
                tau *= backtrack_beta
                continue
            if orth_error > reorth_tol:
                trial_u = _orthonormalize(trial_u)

            trial_x = trial_u @ trial_u.T
            trial_box_loss, _ = box_penalty(
                trial_x,
                alpha,
                upper,
                box_mu,
                off_diagonal_only=off_diagonal_only,
            )
            f_new = float(
                -2.0 * np.sum(A_sym * trial_x)
                + lam * huber_value(trial_x, delta)
                + trial_box_loss
            )

            if f_new <= f_old - armijo_sigma * tau * grad_norm**2:
                accepted = True
                U_candidate = trial_u
                break

            tau *= backtrack_beta

        step_history.append(float(tau if accepted else 0.0))

        if not accepted or U_candidate is None:
            line_search_failed = True
            warnings.warn(
                "Bounded sparse RPMA line search failed before reaching the "
                "stationarity tolerance; the last feasible projection iterate "
                "is being returned.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

        U = U_candidate

    final_orth_error = np.linalg.norm(U.T @ U - identity_k, ord="fro")
    if final_orth_error > reorth_tol:
        U = _orthonormalize(U)

    X = U @ U.T
    final_box_loss, _ = box_penalty(
        X,
        alpha,
        upper,
        box_mu,
        off_diagonal_only=off_diagonal_only,
    )
    final_huber = huber_value(X, delta)
    diagnostics = _constraint_diagnostics(
        X,
        alpha,
        upper,
        off_diagonal_only=off_diagonal_only,
    )

    info: Dict[str, object] = {
        "method": "Bounded-Sparse-RPMA",
        "n": int(n),
        "rank": int(K),
        "lam": float(lam),
        "delta": float(delta),
        "box_mu": float(box_mu),
        "alpha": float(alpha),
        "upper": float(upper),
        "off_diagonal_only": bool(off_diagonal_only),
        "n_iter": int(len(grad_history)),
        "converged": bool(converged),
        "line_search_failed": bool(line_search_failed),
        "final_grad_norm": float(grad_history[-1]) if grad_history else np.nan,
        "final_objective": float(
            -2.0 * np.sum(A_sym * X)
            + lam * final_huber
            + final_box_loss
        ),
        "final_huber_value": float(final_huber),
        "final_box_loss": float(final_box_loss),
        "orthogonality_error": float(
            np.linalg.norm(U.T @ U - identity_k, ord="fro")
        ),
        "idempotence_error": float(
            np.linalg.norm(X @ X - X, ord="fro")
        ),
        "grad_history": grad_history,
        "objective_history": objective_history,
        "step_history": step_history,
    }
    info.update(diagnostics)

    if return_info:
        return X, U, info
    return X


bs_rpma = bounded_sparse_rpa
