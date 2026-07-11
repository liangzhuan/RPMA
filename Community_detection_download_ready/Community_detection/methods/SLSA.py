"""
SLSA: Simultaneously Low-Rank and Sparse Approximation for graph refinement.

Replacement / new file path:
    Community_detection/methods/SLSA.py

Based on:
    Zhang, Zhai, Li, "Graph Refinement via Simultaneously Low-Rank and
    Sparse Approximation", SIAM J. Sci. Comput., 2022.

This implementation follows Algorithm 3.1 in the paper:
    1. rescale A <- sqrt(K) A / ||A||_F
    2. initialize Z = A
    3. repeat:
       - compute top-K eigenvectors U of Z
       - P = U U^T
       - update Z by the closed form solution for l1 or Frobenius loss
       - truncate Z by keeping eta sparse off-diagonal entries
       - stop when ||Z - Z_old||_F < tau

Compared with the old ssl2.py, this version is more explicit and robust:
    - uses the paper's closed-form update directly;
    - enforces symmetry and nonnegativity after each update;
    - truncates only upper-triangle off-diagonal candidates;
    - records n_iter, convergence flag, final_diff, and nnz;
    - includes eigen-solver fallbacks to reduce random LinAlgError failures.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def _symmetrize(A: np.ndarray, zero_diagonal: bool = False) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    A = 0.5 * (A + A.T)
    if zero_diagonal:
        np.fill_diagonal(A, 0.0)
    return A


def _top_k_eigenvectors_symmetric(Z: np.ndarray, K: int) -> np.ndarray:
    """Return top-K eigenvectors of a real symmetric matrix with fallbacks."""
    Z = _symmetrize(Z)
    n = Z.shape[0]
    if K <= 0 or K > n:
        raise ValueError(f"K must satisfy 1 <= K <= n, got K={K}, n={n}.")

    # Try partial symmetric eigensolver first. For n around 1000-2000 this is
    # usually faster than full decomposition and often more stable in scipy.
    try:
        vals, vecs = eigh(
            Z,
            subset_by_index=[n - K, n - 1],
            check_finite=False,
            overwrite_a=False,
            driver="evr",
        )
        order = np.argsort(vals)[::-1]
        U = vecs[:, order]
    except Exception:
        # Fallback 1: full scipy.linalg.eigh.
        try:
            vals, vecs = eigh(Z, check_finite=False, overwrite_a=False)
            idx = np.argsort(vals)[::-1][:K]
            U = vecs[:, idx]
        except Exception:
            # Fallback 2: numpy.linalg.eigh.
            vals, vecs = np.linalg.eigh(Z)
            idx = np.argsort(vals)[::-1][:K]
            U = vecs[:, idx]

    U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
    return U


def _eta_to_num_upper_edges(eta: int, n: int, eta_mode: str = "total") -> int:
    """
    Convert eta into number of upper-triangle undirected edges to keep.

    eta_mode='paper': eta is paper's off-diagonal sparsity ||Z_off||_0 <= eta.
                      Since Z is symmetric, keep eta/2 upper-triangle edges.
    eta_mode='total': eta means total nonzero budget including diagonal, matching
                      the previous ssl2.py / image_all_methods convention:
                          eta_total = n + 2 * (#upper edges).
    """
    eta = int(eta)
    max_edges = n * (n - 1) // 2

    if eta_mode == "paper":
        m = eta // 2
    elif eta_mode == "total":
        m = (eta - n) // 2
    else:
        raise ValueError("eta_mode must be either 'total' or 'paper'.")

    m = int(max(0, min(m, max_edges)))
    return m


def trunc_matrix_slsa(
    Z_tilde: np.ndarray,
    Delta: np.ndarray,
    eta: int,
    eta_mode: str = "total",
    keep_diagonal: bool = True,
) -> np.ndarray:
    """
    Truncation T_eta from the SLSA paper.

    Keep the upper-triangle off-diagonal entries with the largest Delta scores,
    mirror them to preserve symmetry, and optionally preserve the diagonal.
    """
    Z_tilde = _symmetrize(Z_tilde)
    Delta = _symmetrize(Delta)
    n = Z_tilde.shape[0]
    m = _eta_to_num_upper_edges(eta, n, eta_mode=eta_mode)

    mask = np.zeros((n, n), dtype=bool)
    if keep_diagonal:
        np.fill_diagonal(mask, True)

    if m > 0:
        iu = np.triu_indices(n, k=1)
        scores = np.asarray(Delta[iu], dtype=np.float64)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)

        if m >= scores.size:
            chosen = np.arange(scores.size)
        else:
            # argpartition avoids sorting all n(n-1)/2 entries. Then sort only
            # selected entries for deterministic order.
            chosen_unsorted = np.argpartition(scores, -m)[-m:]
            chosen = chosen_unsorted[np.argsort(scores[chosen_unsorted])[::-1]]

        rows = iu[0][chosen]
        cols = iu[1][chosen]
        mask[rows, cols] = True
        mask[cols, rows] = True

    Z = np.where(mask, Z_tilde, 0.0)
    Z = _symmetrize(Z)
    Z = np.maximum(Z, 0.0)
    return Z


def slsa(
    A: np.ndarray,
    K: int,
    eta: int,
    theta: float = 1.0,
    tau: float = 1e-6,
    loss: str = "fro",
    max_iter: int = 200,
    eta_mode: str = "total",
    return_info: bool = False,
    verbose: bool = False,
):
    """
    Run SLSA graph refinement.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric nonnegative input affinity / graph matrix.
    K : int
        Number of clusters / target rank.
    eta : int
        Sparse budget. By default eta_mode='total', so eta follows the existing
        pipeline convention: eta = n + 2*n*eta_k. Set eta_mode='paper' if eta is
        the paper's off-diagonal budget ||Z_off||_0 <= eta.
    theta : float
        Low-rank penalty strength.
    tau : float
        Stop when ||Z_new - Z_old||_F < tau.
    loss : {'fro', 'l1'}
        Loss f(Z; A). 'fro' corresponds to ||Z-A||_F^2; 'l1' to ||Z-A||_1.
    max_iter : int
        Maximum iterations.
    return_info : bool
        If True, return (Z, U, info). Otherwise return Z.

    Returns
    -------
    Z : ndarray, shape (n, n)
        Refined sparse graph matrix.
    U : ndarray, shape (n, K), optional
        Top-K eigenvectors from the last iteration, returned when return_info=True.
    info : dict, optional
        n_iter, converged, final_diff, nnz, eta, theta, tau, loss, max_iter.
    """
    if theta <= 0:
        raise ValueError("theta must be positive.")
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if loss not in {"fro", "l1"}:
        raise ValueError("loss must be 'fro' or 'l1'.")

    A0 = _symmetrize(A)
    A0 = np.maximum(A0, 0.0)
    n = A0.shape[0]
    fro_norm = np.linalg.norm(A0, "fro")
    if not np.isfinite(fro_norm) or fro_norm <= 0:
        raise ValueError(f"Invalid input graph Frobenius norm: {fro_norm}")

    # Paper Algorithm 3.1, Step 1.
    A_scaled = A0 * (np.sqrt(K) / fro_norm)
    A_scaled = _symmetrize(np.maximum(A_scaled, 0.0))
    A_abs = np.abs(A_scaled)

    # Paper Algorithm 3.1, Step 2.
    Z = A_scaled.copy()
    U = None
    converged = False
    final_diff = np.nan

    for it in range(int(max_iter)):
        Z_old = Z.copy()
        Z = _symmetrize(Z)

        # Paper Algorithm 3.1, Step 4.
        U = _top_k_eigenvectors_symmetric(Z, K)
        P = _symmetrize(U @ U.T)

        # Paper Eq. (3.5): closed-form update without sparsity.
        if loss == "fro":
            Z_tilde = (A_scaled + theta * P) / (1.0 + theta)
            Z_tilde = np.maximum(Z_tilde, 0.0)
            # Paper Algorithm 3.1 uses Delta = Z for Frobenius loss.
            Delta = Z_tilde
        else:
            nu = 1.0 / (2.0 * theta)
            # z_ij = ( min{ max{a_ij, p_ij - nu}, p_ij + nu } )_+
            Z_tilde = np.minimum(np.maximum(A_scaled, P - nu), P + nu)
            Z_tilde = np.maximum(Z_tilde, 0.0)
            # Paper Eq. (3.6): delta_ij = phi_ij(0) - phi_ij(z_tilde).
            phi0 = A_abs + theta * (P ** 2)
            phiz = np.abs(A_scaled - Z_tilde) + theta * ((P - Z_tilde) ** 2)
            Delta = phi0 - phiz

        Z_tilde = _symmetrize(Z_tilde)
        Delta = _symmetrize(Delta)

        # Paper Algorithm 3.1, Step 7.
        Z = trunc_matrix_slsa(
            Z_tilde,
            Delta,
            eta=eta,
            eta_mode=eta_mode,
            keep_diagonal=True,
        )

        final_diff = float(np.linalg.norm(Z - Z_old, "fro"))
        if verbose:
            print(f"SLSA iter={it + 1:04d}, diff={final_diff:.6e}, nnz={np.count_nonzero(Z)}")

        # Paper Algorithm 3.1, Step 8.
        if final_diff < tau:
            converged = True
            break

    if U is None:
        U = _top_k_eigenvectors_symmetric(Z, K)

    info = {
        "n_iter": int(it + 1) if "it" in locals() else 0,
        "converged": bool(converged),
        "final_diff": float(final_diff) if np.isfinite(final_diff) else np.nan,
        "nnz": int(np.count_nonzero(Z)),
        "eta": int(eta),
        "eta_mode": eta_mode,
        "theta": float(theta),
        "tau": float(tau),
        "loss": loss,
        "max_iter": int(max_iter),
    }

    if return_info:
        return Z, U, info
    return Z
