"""Nonnegative stochastic RPMA (NS-RPMA).

Place at::

    Community_detection/methods/ns_rpma.py

NS-RPMA keeps the RPMA projection-manifold model and adds two structures of an
ideal class-membership projector:

1. exact row sums ``X @ 1 = 1``;
2. a soft nonnegativity penalty.

The row-sum constraint is imposed exactly through

    q = 1 / sqrt(n) * 1,
    X = q q.T + V V.T,
    V.T V = I_(K-1),  V.T q = 0.

Hence ``X`` is always a rank-K orthogonal projector and always satisfies
``X @ 1 = 1``.  The target objective is

    -2 <A, X>
    + lam * sum_{i != j} huber_delta(X_ij)
    + mu  * sum_ij min(X_ij, 0)^2.

Unlike the earlier bounded model, NS-RPMA does not force a common upper bound
``K/n`` and therefore does not force all leverage scores to be equal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from methods.rpma_advanced_common import (
    cayley_smw_solve,
    geometric_delta_schedule,
    huber_grad,
    huber_value,
    linear_schedule,
    nonnegative_penalty,
    orthonormalize,
    validate_nonnegative,
    validate_positive,
    validate_square_matrix,
)


def _initial_complement_basis(
    A_sym: np.ndarray,
    K: int,
    q: np.ndarray,
    U0: Optional[ArrayLike],
) -> np.ndarray:
    n = A_sym.shape[0]
    rank = K - 1
    if rank == 0:
        return np.empty((n, 0), dtype=np.float64)

    if U0 is not None:
        candidate = np.asarray(U0, dtype=np.float64)
        if candidate.ndim != 2 or candidate.shape[0] != n:
            raise ValueError(
                f"U0 must have n={n} rows; got shape {candidate.shape}."
            )
        if candidate.shape[1] not in {rank, K}:
            raise ValueError(
                f"U0 must have {rank} or {K} columns; got {candidate.shape[1]}."
            )
        candidate = candidate - q @ (q.T @ candidate)
        # SVD gives the dominant independent directions after removing q.
        left, singular_values, _ = np.linalg.svd(candidate, full_matrices=False)
        if singular_values.size < rank or singular_values[rank - 1] <= 1e-10:
            raise ValueError("U0 does not contain enough directions orthogonal to q.")
        return orthonormalize(left[:, :rank], expected_rank=rank)

    # Spectral initialization in q-perp.  Project A on both sides before
    # extracting the leading K-1 eigenvectors.
    P_A_P = A_sym.copy()
    Aq = A_sym @ q
    qTAq = float(q.T @ Aq)
    P_A_P -= q @ Aq.T
    P_A_P -= Aq @ q.T
    P_A_P += qTAq * (q @ q.T)
    values, vectors = np.linalg.eigh(0.5 * (P_A_P + P_A_P.T))
    order = np.argsort(values)[::-1]
    V = vectors[:, order[:rank]]
    V -= q @ (q.T @ V)
    return orthonormalize(V, expected_rank=rank)


def ns_rpma(
    A: ArrayLike,
    K: int,
    *,
    lam: float = 0.005,
    delta: float = 1e-3,
    nonnegative_mu: float = 1.0,
    start_delta: float = 1e-2,
    continuation_steps: int = 4,
    max_iter_per_stage: int = 150,
    U0: Optional[ArrayLike] = None,
    tol: float = 1e-5,
    tau_max: float = 1.0,
    tau_min: float = 1e-14,
    backtrack_beta: float = 0.5,
    armijo_sigma: float = 1e-4,
    nonmonotone_window: int = 5,
    verbose: bool = False,
    return_info: bool = False,
):
    """Solve NS-RPMA with exact row sums and soft nonnegativity."""
    A_arr = validate_square_matrix(A)
    A_sym = 0.5 * (A_arr + A_arr.T)
    n = A_sym.shape[0]
    if not isinstance(K, (int, np.integer)) or not (1 <= int(K) <= n):
        raise ValueError(f"K must be an integer satisfying 1 <= K <= {n}.")
    K = int(K)
    lam = validate_nonnegative("lam", lam)
    delta = validate_positive("delta", delta)
    nonnegative_mu = validate_nonnegative("nonnegative_mu", nonnegative_mu)
    start_delta = validate_positive("start_delta", start_delta)
    if not isinstance(continuation_steps, (int, np.integer)) or int(continuation_steps) <= 0:
        raise ValueError("continuation_steps must be a positive integer.")
    if not isinstance(max_iter_per_stage, (int, np.integer)) or int(max_iter_per_stage) <= 0:
        raise ValueError("max_iter_per_stage must be a positive integer.")

    q = np.ones((n, 1), dtype=np.float64) / np.sqrt(float(n))
    if K == 1:
        X = q @ q.T
        U = q.copy()
        neg_loss, _ = nonnegative_penalty(X, nonnegative_mu)
        info = {
            "n_iter": 0,
            "final_grad_norm": 0.0,
            "converged": True,
            "line_search_failed": False,
            "final_objective": float(
                -2.0 * np.sum(A_sym * X)
                + lam * huber_value(X, delta, off_diagonal_only=True)
                + neg_loss
            ),
            "orthogonality_error": 0.0,
            "idempotence_error": float(np.linalg.norm(X @ X - X, ord="fro")),
            "row_sum_residual": float(np.linalg.norm(X @ np.ones(n) - np.ones(n))),
            "negative_violation_fro": 0.0,
            "negative_entry_ratio": 0.0,
            "x_min": float(np.min(X)),
            "x_max": float(np.max(X)),
        }
        return (X, U, info) if return_info else X

    V = _initial_complement_basis(A_sym, K, q, U0)
    lam_schedule = linear_schedule(lam, int(continuation_steps), include_zero=True)
    mu_schedule = linear_schedule(
        nonnegative_mu,
        int(continuation_steps),
        include_zero=True,
    )
    delta_schedule = geometric_delta_schedule(
        max(start_delta, delta),
        delta,
        int(continuation_steps),
    )

    def projector(V_basis: np.ndarray) -> np.ndarray:
        return q @ q.T + V_basis @ V_basis.T

    all_objectives: list[float] = []
    all_gradients: list[float] = []
    all_steps: list[float] = []
    stage_records: list[dict[str, object]] = []
    line_search_failed = False
    total_cg_restarts = 0

    for stage, (lam_stage, mu_stage, delta_stage) in enumerate(
        zip(lam_schedule, mu_schedule, delta_schedule),
        start=1,
    ):
        if verbose:
            print(
                f"  [NS-RPMA stage {stage}/{continuation_steps}] "
                f"lam={lam_stage:.6g}, mu={mu_stage:.6g}, "
                f"delta={delta_stage:.6g}"
            )

        def objective(
            X: np.ndarray,
            ls=lam_stage,
            ms=mu_stage,
            ds=delta_stage,
        ) -> float:
            nonneg_loss, _ = nonnegative_penalty(X, ms)
            return float(
                -2.0 * np.sum(A_sym * X)
                + ls * huber_value(X, ds, off_diagonal_only=True)
                + nonneg_loss
            )

        def gradient(
            X: np.ndarray,
            ls=lam_stage,
            ms=mu_stage,
            ds=delta_stage,
        ) -> np.ndarray:
            _, nonneg_grad = nonnegative_penalty(X, ms)
            return (
                -2.0 * A_sym
                + ls * huber_grad(X, ds, off_diagonal_only=True)
                + nonneg_grad
            )

        X_stage, V, stage_info = cayley_smw_solve(
            V,
            objective,
            gradient,
            projector=projector,
            fixed_basis=q,
            max_iter=int(max_iter_per_stage),
            tol=tol,
            tau_max=tau_max,
            tau_min=tau_min,
            backtrack_beta=backtrack_beta,
            armijo_sigma=armijo_sigma,
            reuse_step=True,
            step_growth=1.25,
            use_cg=True,
            nonmonotone_window=nonmonotone_window,
            verbose=verbose,
        )
        record = stage_info.as_dict()
        record.update(
            {
                "stage": stage,
                "stage_lam": float(lam_stage),
                "stage_mu": float(mu_stage),
                "stage_delta": float(delta_stage),
            }
        )
        stage_records.append(record)
        all_objectives.extend(stage_info.objective_history)
        all_gradients.extend(stage_info.grad_history)
        all_steps.extend(stage_info.step_history)
        line_search_failed = line_search_failed or stage_info.line_search_failed
        total_cg_restarts += stage_info.cg_restart_count
        if stage_info.line_search_failed:
            break

    X_final = projector(V)
    U_final = np.hstack((q, V))
    final_nonnegative_loss, final_nonnegative_grad = nonnegative_penalty(
        X_final,
        nonnegative_mu,
    )
    final_objective = float(
        -2.0 * np.sum(A_sym * X_final)
        + lam * huber_value(X_final, delta, off_diagonal_only=True)
        + final_nonnegative_loss
    )
    G_final = (
        -2.0 * A_sym
        + lam * huber_grad(X_final, delta, off_diagonal_only=True)
        + final_nonnegative_grad
    )
    residual = G_final @ V
    residual -= V @ (V.T @ residual)
    residual -= q @ (q.T @ residual)
    final_grad_norm = float(np.linalg.norm(residual, ord="fro"))
    negative = np.maximum(-X_final, 0.0)

    info = {
        "n_iter": len(all_gradients),
        "final_grad_norm": final_grad_norm,
        "converged": bool(final_grad_norm <= tol),
        "line_search_failed": line_search_failed,
        "final_objective": final_objective,
        "orthogonality_error": float(
            np.linalg.norm(U_final.T @ U_final - np.eye(K), ord="fro")
        ),
        "idempotence_error": float(
            np.linalg.norm(X_final @ X_final - X_final, ord="fro")
        ),
        "row_sum_residual": float(
            np.linalg.norm(X_final @ np.ones(n) - np.ones(n))
        ),
        "negative_violation_fro": float(np.linalg.norm(negative, ord="fro")),
        "negative_entry_ratio": float(np.mean(X_final < -1e-12)),
        "final_nonnegative_loss": float(final_nonnegative_loss),
        "objective_history": all_objectives,
        "grad_history": all_gradients,
        "step_history": all_steps,
        "cg_restart_count": total_cg_restarts,
        "continuation_steps_completed": len(stage_records),
        "stage_records": stage_records,
        "lam_schedule": lam_schedule,
        "mu_schedule": mu_schedule,
        "delta_schedule": delta_schedule,
        "lam": lam,
        "delta": delta,
        "nonnegative_mu": nonnegative_mu,
        "x_min": float(np.min(X_final)),
        "x_max": float(np.max(X_final)),
    }

    if return_info:
        return X_final, U_final, info
    return X_final


__all__ = ["ns_rpma"]
