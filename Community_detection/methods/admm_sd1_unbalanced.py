"""
Unbalanced ADMM-SD1 / Projection-SDP.

This module replaces the equal-size SDP-1 constraints

    diag(M) = 1,
    M 1 = (n / K) 1,

with projection-matrix-scale constraints suitable for unequal cluster sizes:

    X >= 0 elementwise,
    X is positive semidefinite,
    X 1 = 1,
    trace(X) = K.

The ideal matrix for an arbitrary partition is

    X_* = H (H^T H)^{-1} H^T,

so unequal class sizes are feasible.

Place this file at:
    Community_detection/methods/admm_sd1_unbalanced.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import eigh


def _symmetrize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return 0.5 * (X + X.T)


def _validate_problem(A: np.ndarray, K: int) -> tuple[np.ndarray, int, int]:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if not np.all(np.isfinite(A)):
        raise ValueError("A contains NaN or infinite values.")

    n = A.shape[0]
    K = int(K)
    if not 1 <= K <= n:
        raise ValueError(f"K must satisfy 1 <= K <= n; got K={K}, n={n}.")

    return _symmetrize(A), n, K


def project_psd(X: np.ndarray) -> np.ndarray:
    """Frobenius projection onto the positive-semidefinite cone."""
    X = _symmetrize(X)
    eigenvalues, eigenvectors = eigh(
        X,
        overwrite_a=False,
        check_finite=False,
    )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = (eigenvectors * eigenvalues) @ eigenvectors.T
    return _symmetrize(projected)


def project_affine_unbalanced(Y: np.ndarray, K: int) -> np.ndarray:
    """Project a symmetric matrix onto

        A = {X = X^T : X 1 = 1, trace(X) = K}.

    The projection solves

        min_X 0.5 ||X - Y||_F^2
        s.t.  X 1 = 1, trace(X) = K, X = X^T.

    Parameters
    ----------
    Y:
        Square input matrix.
    K:
        Desired trace / target number of clusters.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[0] != Y.shape[1]:
        raise ValueError("Y must be a square matrix.")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y contains NaN or infinite values.")

    Y = _symmetrize(Y)
    n = Y.shape[0]
    K = int(K)
    if not 1 <= K <= n:
        raise ValueError(f"K must satisfy 1 <= K <= n; got K={K}, n={n}.")

    if n == 1:
        return np.ones((1, 1), dtype=np.float64)

    one = np.ones(n, dtype=np.float64)

    # Residuals of the two affine constraints.
    row_residual = Y @ one - one
    trace_residual = float(np.trace(Y) - K)

    row_residual_mean = float(one @ row_residual) / n

    # Lagrange-multiplier solution in the symmetric matrix space.
    scalar_a = (
        row_residual_mean - trace_residual / n
    ) / (n - 1)

    u = (
        (2.0 / n) * (row_residual - row_residual_mean * one)
        + scalar_a * one
    )
    tau = trace_residual / n - scalar_a

    X = (
        Y
        - 0.5 * (np.outer(u, one) + np.outer(one, u))
        - tau * np.eye(n, dtype=np.float64)
    )
    return _symmetrize(X)


def feasible_initialization(n: int, K: int) -> np.ndarray:
    """Return a PSD, nonnegative, row-stochastic matrix with trace K."""
    n = int(n)
    K = int(K)
    if not 1 <= K <= n:
        raise ValueError(f"K must satisfy 1 <= K <= n; got K={K}, n={n}.")
    if n == 1:
        return np.ones((1, 1), dtype=np.float64)

    one = np.ones((n, 1), dtype=np.float64)
    J = (one @ one.T) / n
    alpha = (K - 1.0) / (n - 1.0)
    X0 = J + alpha * (np.eye(n, dtype=np.float64) - J)
    return _symmetrize(X0)


def _diagnostics(X: np.ndarray, K: int) -> dict[str, float | int]:
    X = _symmetrize(X)
    n = X.shape[0]
    one = np.ones(n, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(X)

    negative_part = np.minimum(X, 0.0)
    return {
        "row_sum_residual": float(np.linalg.norm(X @ one - one)),
        "trace_residual": float(abs(np.trace(X) - K)),
        "symmetry_residual": float(np.linalg.norm(X - X.T)),
        "negative_violation_fro": float(np.linalg.norm(negative_part)),
        "minimum_entry": float(np.min(X)),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "effective_rank_1e-6": int(np.sum(eigenvalues > 1e-6)),
    }


def admm_sd1_unbalanced(
    A: np.ndarray,
    K: int,
    *,
    rho: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 300,
    adaptive_rho: bool = True,
    rho_balance: float = 10.0,
    rho_scale: float = 2.0,
    rho_update_interval: int = 10,
    verbose: bool = False,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Solve the unequal-size Projection-SDP with two-block consensus ADMM.

    Model
    -----
        maximize    <A, X>
        subject to  X >= 0,
                    X is PSD,
                    X 1 = 1,
                    trace(X) = K.

    Splitting
    ---------
    X handles the affine constraints, Z handles elementwise nonnegativity,
    and Y handles positive semidefiniteness.

    Notes
    -----
    A full n-by-n eigendecomposition is required at each iteration. For image
    data, first test with a moderate sampling percentage.
    """
    A, n, K = _validate_problem(A, K)

    rho = float(rho)
    tol = float(tol)
    max_iter = int(max_iter)
    if rho <= 0.0:
        raise ValueError("rho must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if rho_balance <= 1.0:
        raise ValueError("rho_balance must be greater than 1.")
    if rho_scale <= 1.0:
        raise ValueError("rho_scale must be greater than 1.")

    # Start from a point feasible for every constraint set.
    X = feasible_initialization(n, K)
    Z = X.copy()
    Y = X.copy()

    # Unscaled Lagrange multipliers for X=Z and X=Y.
    U = np.zeros((n, n), dtype=np.float64)
    V = np.zeros((n, n), dtype=np.float64)

    history: list[dict[str, float | int]] = []
    converged = False
    primal_residual = float("inf")
    dual_residual = float("inf")

    # Standard ADMM absolute and relative tolerances.
    abs_tol = tol
    rel_tol = tol

    for iteration in range(1, max_iter + 1):
        Z_old = Z.copy()
        Y_old = Y.copy()

        center = (
            rho * Z - U
            + rho * Y - V
            + A
        ) / (2.0 * rho)

        X = project_affine_unbalanced(center, K)

        Z = np.maximum(_symmetrize(X + U / rho), 0.0)
        Z = _symmetrize(Z)

        Y = project_psd(X + V / rho)

        R_z = X - Z
        R_y = X - Y
        U = _symmetrize(U + rho * R_z)
        V = _symmetrize(V + rho * R_y)

        primal_residual = float(
            np.sqrt(
                np.linalg.norm(R_z, "fro") ** 2
                + np.linalg.norm(R_y, "fro") ** 2
            )
        )
        dual_residual = float(
            rho
            * np.sqrt(
                np.linalg.norm(Z - Z_old, "fro") ** 2
                + np.linalg.norm(Y - Y_old, "fro") ** 2
            )
        )

        norm_x_pair = np.sqrt(2.0) * np.linalg.norm(X, "fro")
        norm_zy = np.sqrt(
            np.linalg.norm(Z, "fro") ** 2
            + np.linalg.norm(Y, "fro") ** 2
        )
        eps_primal = float(
            np.sqrt(2.0) * n * abs_tol
            + rel_tol * max(norm_x_pair, norm_zy)
        )
        eps_dual = float(
            np.sqrt(2.0) * n * abs_tol
            + rel_tol
            * np.sqrt(
                np.linalg.norm(U, "fro") ** 2
                + np.linalg.norm(V, "fro") ** 2
            )
        )

        objective = float(np.sum(A * X))
        history.append(
            {
                "iteration": iteration,
                "objective": objective,
                "primal_residual": primal_residual,
                "dual_residual": dual_residual,
                "eps_primal": eps_primal,
                "eps_dual": eps_dual,
                "rho": rho,
            }
        )

        if verbose and (
            iteration == 1
            or iteration % 10 == 0
            or iteration == max_iter
        ):
            print(
                "[ADMM-SD1-Unbalanced] "
                f"iter={iteration:4d}, "
                f"score={objective:.6e}, "
                f"r={primal_residual:.3e}/{eps_primal:.3e}, "
                f"s={dual_residual:.3e}/{eps_dual:.3e}, "
                f"rho={rho:.3e}"
            )

        if primal_residual <= eps_primal and dual_residual <= eps_dual:
            converged = True
            break

        if (
            adaptive_rho
            and iteration % int(rho_update_interval) == 0
            and primal_residual > 0.0
            and dual_residual > 0.0
        ):
            if primal_residual > rho_balance * dual_residual:
                rho *= rho_scale
            elif dual_residual > rho_balance * primal_residual:
                rho /= rho_scale

    # X exactly satisfies the affine constraints at every iteration.
    X = _symmetrize(X)

    info: dict[str, Any] = {
        "model": "Projection-SDP / unbalanced ADMM-SD1",
        "n_iter": iteration,
        "converged": converged,
        "rho_final": float(rho),
        "primal_residual": primal_residual,
        "dual_residual": dual_residual,
        "objective_similarity": float(np.sum(A * X)),
        "history": history,
    }
    info.update(_diagnostics(X, K))

    if return_info:
        return X, info
    return X


# A concise alias for experiment scripts.
projection_sdp = admm_sd1_unbalanced
