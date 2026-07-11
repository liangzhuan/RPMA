"""
Unified affinity-matrix construction for image clustering experiments.

Place at:
    Community_detection/methods/affinity.py

Supported graph types
---------------------
full_gaussian:
    Dense Gaussian graph with one global bandwidth.

knn_gaussian:
    k-nearest-neighbor Gaussian graph with one global bandwidth.

self_tuning:
    Locally scaled Gaussian graph:
        A_ij = exp(-d_ij^2 / (sigma_i sigma_j))
    where sigma_i is the distance to the k-th nearest neighbor.

cosine:
    Nonnegative cosine-similarity graph.

binary_knn:
    Unweighted symmetric k-nearest-neighbor graph.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity


def _validate_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X must have shape (n_samples, n_features), with n_samples >= 2.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")
    return X


def _symmetrize(A: np.ndarray, rule: str = "max") -> np.ndarray:
    if rule == "max":
        return np.maximum(A, A.T)
    if rule == "mean":
        return 0.5 * (A + A.T)
    raise ValueError("symmetrize_rule must be 'max' or 'mean'.")


def _knn_mask(D2: np.ndarray, k: int) -> np.ndarray:
    n = D2.shape[0]
    if not 1 <= int(k) < n:
        raise ValueError(f"k must satisfy 1 <= k < n; got k={k}, n={n}.")
    k = int(k)

    # Exclude each sample itself before selecting neighbors.
    work = D2.copy()
    np.fill_diagonal(work, np.inf)
    indices = np.argpartition(work, kth=k - 1, axis=1)[:, :k]

    mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    mask[rows, indices] = True
    return mask


def _global_sigma2(D2: np.ndarray, mode: str, scale: float) -> float:
    n = D2.shape[0]
    upper = D2[np.triu_indices(n, k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        raise ValueError("All pairwise distances are zero; cannot construct a Gaussian graph.")

    if mode == "mean":
        sigma2 = float(np.mean(positive))
    elif mode == "median":
        sigma2 = float(np.median(positive))
    else:
        raise ValueError("bandwidth must be 'mean' or 'median'.")

    sigma2 *= float(scale)
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        raise ValueError(f"Invalid sigma^2={sigma2}.")
    return sigma2


def build_affinity(
    X: np.ndarray,
    graph: str = "full_gaussian",
    *,
    k: int = 10,
    bandwidth: str = "mean",
    sigma2_scale: float = 1.0,
    symmetrize_rule: str = "max",
    zero_diagonal: bool = False,
    cosine_knn: bool = False,
) -> Tuple[np.ndarray, Dict[str, float | int | str | bool]]:
    """
    Construct an affinity matrix A from feature matrix X.

    Returns
    -------
    A:
        Symmetric nonnegative affinity matrix, shape (n, n).
    info:
        Graph-construction metadata for logging and result files.
    """
    X = _validate_features(X)
    n = X.shape[0]
    graph = str(graph).lower()

    info: Dict[str, float | int | str | bool] = {
        "affinity_graph": graph,
        "affinity_k": int(k),
        "affinity_bandwidth": str(bandwidth),
        "affinity_sigma2_scale": float(sigma2_scale),
        "affinity_symmetrize_rule": str(symmetrize_rule),
        "zero_diagonal": bool(zero_diagonal),
    }

    if graph in {"full_gaussian", "knn_gaussian", "self_tuning", "binary_knn"}:
        D2 = pairwise_distances(X, metric="sqeuclidean", n_jobs=1)
        D2 = np.maximum(D2, 0.0)
        np.fill_diagonal(D2, 0.0)

    if graph == "full_gaussian":
        sigma2 = _global_sigma2(D2, bandwidth, sigma2_scale)
        A = np.exp(-D2 / sigma2)
        A = 0.5 * (A + A.T)
        info["sigma2"] = sigma2

    elif graph == "knn_gaussian":
        sigma2 = _global_sigma2(D2, bandwidth, sigma2_scale)
        mask = _knn_mask(D2, k)
        A = np.zeros((n, n), dtype=np.float64)
        A[mask] = np.exp(-D2[mask] / sigma2)
        A = _symmetrize(A, symmetrize_rule)
        info["sigma2"] = sigma2

    elif graph == "self_tuning":
        # sigma_i is the Euclidean distance to the k-th nearest neighbor.
        work = D2.copy()
        np.fill_diagonal(work, np.inf)
        kth_d2 = np.partition(work, kth=int(k) - 1, axis=1)[:, int(k) - 1]
        sigma = np.sqrt(np.maximum(kth_d2, np.finfo(float).eps))

        denom = np.outer(sigma, sigma)
        A = np.exp(-D2 / np.maximum(denom, np.finfo(float).eps))

        # Keep a sparse neighborhood graph, then symmetrize it.
        mask = _knn_mask(D2, k)
        A = np.where(mask, A, 0.0)
        A = _symmetrize(A, symmetrize_rule)

        info["sigma2"] = float("nan")
        info["local_sigma_min"] = float(np.min(sigma))
        info["local_sigma_median"] = float(np.median(sigma))
        info["local_sigma_max"] = float(np.max(sigma))

    elif graph == "cosine":
        A = cosine_similarity(X)
        # Most downstream graph methods assume nonnegative affinities.
        A = np.clip(A, 0.0, None)

        if cosine_knn:
            # Larger cosine similarity means closer neighbor.
            work = A.copy()
            np.fill_diagonal(work, -np.inf)
            if not 1 <= int(k) < n:
                raise ValueError(f"k must satisfy 1 <= k < n; got k={k}, n={n}.")
            indices = np.argpartition(-work, kth=int(k) - 1, axis=1)[:, : int(k)]
            mask = np.zeros((n, n), dtype=bool)
            mask[np.arange(n)[:, None], indices] = True
            A = np.where(mask, A, 0.0)
            A = _symmetrize(A, symmetrize_rule)
        else:
            A = 0.5 * (A + A.T)

        info["sigma2"] = float("nan")
        info["cosine_knn"] = bool(cosine_knn)

    elif graph == "binary_knn":
        mask = _knn_mask(D2, k)
        A = mask.astype(np.float64)
        A = _symmetrize(A, symmetrize_rule)
        A = (A > 0.0).astype(np.float64)
        info["sigma2"] = float("nan")

    else:
        raise ValueError(
            "Unknown graph={!r}. Choose from full_gaussian, knn_gaussian, "
            "self_tuning, cosine, binary_knn.".format(graph)
        )

    A = np.asarray(A, dtype=np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.maximum(A, 0.0)
    A = 0.5 * (A + A.T)

    if zero_diagonal:
        np.fill_diagonal(A, 0.0)
    elif graph in {"full_gaussian", "self_tuning", "cosine"}:
        np.fill_diagonal(A, 1.0)

    info["affinity_density"] = float(np.mean(A > 1e-12))
    info["affinity_min"] = float(np.min(A))
    info["affinity_max"] = float(np.max(A))
    return A, info
