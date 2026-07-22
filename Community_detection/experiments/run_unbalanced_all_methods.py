"""
Centralized unequal-class image clustering benchmark.

Requested methods
-----------------
1. SDP-1:
   the modified unequal-size Projection-SDP / Peng--Wei constraint model
   implemented in ``methods/admm_sd1_unbalanced.py``;

2. SDP-2:
   the aggregated unequal-size Peng--Wei relaxation in
   ``methods/admm_sd2_unbalanced.py``;

3. RPMA:
   ``methods/rpa.py``;

4. NS-RPMA:
   ``methods/ns_rpma.py``;

5. Spectral Projection:
   top-K eigenvectors of the common affinity matrix;

6. Normalized Cut:
   ``methods/normalized_cut.py``;

7. Regularized Spectral Clustering:
   ``methods/normalized_cut.py``;

8. Regularized Spectral Clustering:
   ``methods/regularized_spectral_clustering.py``;

9. Kernel K-means:
   ``methods/kernel_kmeans.py``;

10. SymNMF:
    ``methods/symnmf.py``;

11. CLR:
    ``methods/clr.py``;

11. SLSA:
    ``methods/SLSA.py``.

All methods receive exactly the same sampled dataset and exactly the same
affinity matrix in each run. The final rounding always uses ordinary K-means;
no balanced-capacity rounding is used.

Unequal-class benchmark construction
------------------------------------
COIL20 and AT&T/ORL Faces are originally balanced. This script creates an
unequal-size benchmark by retaining an exact fixed percentage of all images
and assigning unequal retained counts to the classes.

Ground-truth labels are used ONLY to construct this controlled benchmark and
to calculate ACC/NMI/ARI. Labels are never passed to any clustering method.

Recommended location
--------------------
    Community_detection/experiments/run_unbalanced_all_methods.py

Required additional files
-------------------------
    Community_detection/methods/admm_sd1_unbalanced.py
    Community_detection/methods/admm_sd2_unbalanced.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from datasets.image_datasets import load_att_faces, load_coil20
from evaluation.metrics import evaluate
from methods.admm_sd1_unbalanced import admm_sd1_unbalanced
from methods.admm_sd2_unbalanced import admm_sd2_unbalanced
from methods.affinity import build_affinity
from methods.clr import clr
from methods.kernel_kmeans import kernel_kmeans
from methods.normalized_cut import normalized_cut
from methods.ns_rpma import ns_rpma
from methods.regularized_spectral_clustering import (
    regularized_spectral_clustering,
)
from methods.rpa import rpa
from methods.SLSA import slsa
from methods.spectral_utils import spectral_rounding
from methods.symnmf import symnmf


METHOD_ORDER = [
    "sdp1",
    "sdp2",
    "rpma",
    "ns_rpma",
    "spectral_projection",
    "normalized_cut",
    "regularized_spectral_clustering",
    "kernel_kmeans",
    "symnmf",
    "clr",
    "slsa",
]

NEW_METHOD_ORDER = [
    "normalized_cut",
    "regularized_spectral_clustering",
    "kernel_kmeans",
    "symnmf",
]


# ---------------------------------------------------------------------------
# Parsing and basic utilities
# ---------------------------------------------------------------------------

def parse_image_size(value: str | None):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"original", "orig", "none"}:
        return None
    if "x" in text:
        width, height = text.split("x", 1)
        return int(width), int(height)
    side = int(text)
    return side, side


def parse_seeds(value: str) -> list[int]:
    seeds = [
        int(item.strip())
        for item in str(value).split(",")
        if item.strip()
    ]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def parse_methods(value: str) -> list[str]:
    aliases = {
        "sdp-1": "sdp1",
        "admm_sd1": "sdp1",
        "admm-sd1": "sdp1",
        "projection_sdp": "sdp1",
        "sdp-2": "sdp2",
        "admm_sd2": "sdp2",
        "admm-sd2": "sdp2",
        "rpa": "rpma",
        "ns-rpma": "ns_rpma",
        "nsrpma": "ns_rpma",
        "spectral": "spectral_projection",
        "spectral-projection": "spectral_projection",
        "spectral_projection": "spectral_projection",
        "ncut": "normalized_cut",
        "normalized-cut": "normalized_cut",
        "normalizedcut": "normalized_cut",
        "rsc": "regularized_spectral_clustering",
        "regularized-spectral-clustering": (
            "regularized_spectral_clustering"
        ),
        "regularized_spectral": "regularized_spectral_clustering",
        "kkm": "kernel_kmeans",
        "kernel-kmeans": "kernel_kmeans",
        "kernelkmeans": "kernel_kmeans",
        "sym-nmf": "symnmf",
        "symmetric_nmf": "symnmf",
        "symmetric-nmf": "symnmf",
    }

    raw = [
        item.strip().lower()
        for item in str(value).split(",")
        if item.strip()
    ]
    if not raw:
        raise ValueError("At least one method is required.")
    if "all" in raw:
        return METHOD_ORDER.copy()

    parsed: list[str] = []
    for item in raw:
        canonical = aliases.get(item, item)
        if canonical not in METHOD_ORDER:
            raise ValueError(
                f"Unknown method {item!r}. Available methods: "
                + ", ".join(METHOD_ORDER)
            )
        if canonical not in parsed:
            parsed.append(canonical)
    return parsed


def canonical_dataset_name(value: str) -> str:
    name = str(value).strip().lower()
    if name == "coil20":
        return "coil20"
    if name in {"att", "att_face", "att_faces", "attfaces", "orl"}:
        return "att_faces"
    raise ValueError("dataset must be coil20 or att_faces.")


def default_data_root(dataset: str) -> str:
    if dataset == "coil20":
        return "datasets/data/coil20"
    return "datasets/data/att_faces"


def load_dataset(
    dataset: str,
    data_root: str,
    image_size,
) -> tuple[np.ndarray, np.ndarray, int]:
    if dataset == "coil20":
        return load_coil20(
            data_root,
            image_size=image_size,
            max_per_class=None,
            random_state=0,
        )

    return load_att_faces(
        data_root,
        image_size=image_size,
        max_per_class=None,
        random_state=0,
    )


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = np.nan_to_num(
        matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return 0.5 * (matrix + matrix.T)


def standardize_features(
    X: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    centered = X - np.mean(X, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, keepdims=True)
    return centered / np.maximum(scale, eps)


# ---------------------------------------------------------------------------
# Exact fixed-percentage unequal sampling
# ---------------------------------------------------------------------------

def _validate_count_request(
    capacities: np.ndarray,
    target_total: int,
    min_per_class: int,
) -> None:
    capacities = np.asarray(capacities, dtype=int)
    if min_per_class < 1:
        raise ValueError("min_per_class must be at least 1.")
    if np.any(capacities < min_per_class):
        raise ValueError(
            "At least one class contains fewer images than min_per_class."
        )

    minimum_total = int(capacities.size * min_per_class)
    maximum_total = int(np.sum(capacities))
    if target_total < minimum_total:
        raise ValueError(
            f"The requested sample has {target_total} images, but at least "
            f"{minimum_total} are required to keep all classes."
        )
    if target_total > maximum_total:
        raise ValueError(
            f"The requested sample has {target_total} images, but only "
            f"{maximum_total} are available."
        )


def allocate_dirichlet_counts(
    capacities: np.ndarray,
    target_total: int,
    min_per_class: int,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Allocate an exact total with unequal capacity-constrained counts."""
    capacities = np.asarray(capacities, dtype=int)
    _validate_count_request(
        capacities,
        target_total,
        min_per_class,
    )
    if alpha <= 0.0:
        raise ValueError("imbalance_alpha must be positive.")

    K = capacities.size
    counts = np.full(K, min_per_class, dtype=int)
    remaining = int(target_total - np.sum(counts))
    weights = rng.dirichlet(np.full(K, alpha, dtype=float))

    for _ in range(remaining):
        available = counts < capacities
        probabilities = np.where(available, weights, 0.0)
        normalizer = float(np.sum(probabilities))
        if normalizer <= 0.0:
            raise RuntimeError(
                "No capacity remains during Dirichlet allocation."
            )
        probabilities /= normalizer
        selected_class = int(rng.choice(K, p=probabilities))
        counts[selected_class] += 1

    return counts


def allocate_global_random_counts(
    capacities: np.ndarray,
    target_total: int,
    min_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniformly sample remaining image slots after retaining every class."""
    capacities = np.asarray(capacities, dtype=int)
    _validate_count_request(
        capacities,
        target_total,
        min_per_class,
    )

    K = capacities.size
    counts = np.full(K, min_per_class, dtype=int)
    remaining = int(target_total - np.sum(counts))

    remaining_slots = np.repeat(
        np.arange(K, dtype=int),
        capacities - counts,
    )
    chosen_slots = rng.choice(
        remaining_slots.size,
        size=remaining,
        replace=False,
    )
    counts += np.bincount(
        remaining_slots[chosen_slots],
        minlength=K,
    )
    return counts


def allocate_balanced_counts(
    capacities: np.ndarray,
    target_total: int,
    min_per_class: int,
) -> np.ndarray:
    """Near-equal allocation used only as a control experiment."""
    capacities = np.asarray(capacities, dtype=int)
    _validate_count_request(
        capacities,
        target_total,
        min_per_class,
    )

    K = capacities.size
    counts = np.full(K, min_per_class, dtype=int)
    remaining = int(target_total - np.sum(counts))
    cursor = 0

    while remaining > 0:
        class_index = cursor % K
        if counts[class_index] < capacities[class_index]:
            counts[class_index] += 1
            remaining -= 1
        cursor += 1

    return counts


def sample_unequal_fixed_percentage(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_percent: float,
    sampling_mode: str,
    imbalance_alpha: float,
    min_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    """Create one controlled benchmark shared by every clustering method."""
    X = np.asarray(X)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("X and y have inconsistent shapes.")
    if not 0.0 < sample_percent <= 100.0:
        raise ValueError("sample_percent must lie in (0, 100].")

    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    capacities = np.array(
        [np.sum(y == class_label) for class_label in classes],
        dtype=int,
    )

    target_total = int(round(y.size * sample_percent / 100.0))
    target_total = min(target_total, y.size)

    mode = str(sampling_mode).strip().lower()
    if mode == "dirichlet":
        counts = allocate_dirichlet_counts(
            capacities,
            target_total,
            min_per_class,
            imbalance_alpha,
            rng,
        )
    elif mode == "global":
        counts = allocate_global_random_counts(
            capacities,
            target_total,
            min_per_class,
            rng,
        )
    elif mode == "balanced":
        counts = allocate_balanced_counts(
            capacities,
            target_total,
            min_per_class,
        )
    else:
        raise ValueError(
            "sampling_mode must be dirichlet, global, or balanced."
        )

    selected_indices: list[int] = []
    count_map: dict[int, int] = {}

    for class_position, class_label in enumerate(classes):
        class_indices = np.flatnonzero(y == class_label)
        chosen = rng.choice(
            class_indices,
            size=int(counts[class_position]),
            replace=False,
        )
        selected_indices.extend(chosen.tolist())
        count_map[int(class_label)] = int(counts[class_position])

    selected = np.asarray(selected_indices, dtype=int)
    rng.shuffle(selected)

    return (
        np.asarray(X[selected], dtype=np.float64),
        y[selected],
        selected,
        count_map,
    )


def class_count_statistics(
    count_map: dict[int, int],
) -> dict[str, float | int]:
    values = np.asarray(list(count_map.values()), dtype=float)
    return {
        "class_count_min": int(np.min(values)),
        "class_count_max": int(np.max(values)),
        "class_count_mean": float(np.mean(values)),
        "class_count_std": float(np.std(values)),
        "class_count_cv": float(np.std(values) / np.mean(values)),
        "class_count_max_min_ratio": float(
            np.max(values) / max(np.min(values), 1.0)
        ),
    }


def format_count_map(count_map: dict[int, int]) -> str:
    return ", ".join(
        f"{class_label}:{count_map[class_label]}"
        for class_label in sorted(count_map)
    )


# ---------------------------------------------------------------------------
# Common affinity and ordinary rounding
# ---------------------------------------------------------------------------

def build_common_affinity(
    X: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    if args.affinity == "linear_gram":
        A = np.asarray(X, dtype=np.float64) @ np.asarray(
            X,
            dtype=np.float64,
        ).T
        A = symmetrize(A)

        # A common positive scaling improves numerical conditioning but does
        # not change a linear objective's optimizer.
        scale = float(np.max(np.abs(A)))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        A = A / scale

        if np.min(A) < -1e-10:
            raise ValueError(
                "linear_gram contains negative entries. Do not combine "
                "--standardize with --affinity linear_gram when CLR/SLSA "
                "are included."
            )
        A = np.maximum(A, 0.0)
        if args.zero_diagonal:
            np.fill_diagonal(A, 0.0)

        return A, {
            "affinity_graph": "linear_gram",
            "sigma2": np.nan,
            "gram_scale": scale,
            "affinity_density": float(np.mean(A > 1e-12)),
            "affinity_min": float(np.min(A)),
            "affinity_max": float(np.max(A)),
        }

    A, info = build_affinity(
        X,
        graph=args.affinity,
        k=args.affinity_k,
        bandwidth=args.bandwidth,
        sigma2_scale=args.bandwidth_scale,
        symmetrize_rule=args.affinity_symmetrize,
        zero_diagonal=args.zero_diagonal,
        cosine_knn=args.cosine_knn,
    )
    return symmetrize(A), dict(info)


def ordinary_kmeans(
    embedding: np.ndarray,
    K: int,
    *,
    seed: int,
    n_init: int,
    row_normalize: bool,
) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float64)
    embedding = np.nan_to_num(
        embedding,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if row_normalize:
        embedding = normalize(embedding, norm="l2")

    return KMeans(
        n_clusters=K,
        n_init=n_init,
        random_state=seed,
    ).fit_predict(embedding)


def top_k_embedding(
    matrix: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = symmetrize(matrix)
    n = matrix.shape[0]

    if K == n:
        eigenvalues, eigenvectors = eigh(
            matrix,
            check_finite=False,
        )
    else:
        eigenvalues, eigenvectors = eigh(
            matrix,
            subset_by_index=[n - K, n - 1],
            check_finite=False,
        )

    order = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, order], eigenvalues[order]


def matrix_diagnostics(
    matrix: np.ndarray,
    K: int,
) -> dict[str, float | int]:
    matrix = symmetrize(matrix)
    n = matrix.shape[0]
    one = np.ones(n)
    eigenvalues = np.linalg.eigvalsh(matrix)
    return {
        "matrix_trace": float(np.trace(matrix)),
        "trace_minus_K": float(np.trace(matrix) - K),
        "row_sum_to_one_residual": float(
            np.linalg.norm(matrix @ one - one)
        ),
        "negative_violation_fro": float(
            np.linalg.norm(np.minimum(matrix, 0.0))
        ),
        "minimum_entry": float(np.min(matrix)),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "effective_rank_1e-6": int(
            np.sum(eigenvalues > 1e-6)
        ),
    }


# ---------------------------------------------------------------------------
# Partition-projector decoding for the added methods
# ---------------------------------------------------------------------------

def partition_projector_labels(
    projector: np.ndarray,
    K: int,
    edge_tolerance: float = 1e-12,
) -> tuple[np.ndarray, int]:
    """Recover hard labels from X = H(H^T H)^(-1)H^T.

    The four added method files return an exact normalized partition
    projector. Entries are positive within a cluster and zero between
    clusters, so connected components recover the method's own partition
    without adding another K-means stage.
    """
    projector = symmetrize(projector)
    if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
        raise ValueError("The returned projector must be square.")
    if not np.all(np.isfinite(projector)):
        raise ValueError("The returned projector contains nonfinite values.")

    scale = max(1.0, float(np.max(np.abs(projector))))
    threshold = float(edge_tolerance) * scale

    graph = np.where(projector > threshold, projector, 0.0)
    np.fill_diagonal(graph, 0.0)

    component_count, labels = connected_components(
        csr_matrix(graph),
        directed=False,
        return_labels=True,
    )

    if int(component_count) != int(K):
        raise RuntimeError(
            "The method output was expected to encode exactly "
            f"K={K} clusters, but {component_count} connected "
            "components were recovered."
        )

    return np.asarray(labels, dtype=int), int(component_count)


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def run_one_method(
    method: str,
    A: np.ndarray,
    y: np.ndarray,
    K: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one requested method and return ordinary-k-means labels."""
    n = A.shape[0]
    extra: dict[str, Any] = {}

    if method == "sdp1":
        X_sdp1, info = admm_sd1_unbalanced(
            A,
            K,
            rho=args.admm_rho,
            tol=args.admm_tol,
            max_iter=args.admm_max_iter,
            adaptive_rho=not args.no_adaptive_rho,
            verbose=args.verbose,
            return_info=True,
        )
        embedding, top_values = top_k_embedding(X_sdp1, K)
        labels = ordinary_kmeans(
            embedding,
            K,
            seed=seed,
            n_init=args.kmeans_n_init,
            row_normalize=args.row_normalize,
        )
        extra.update(
            {
                "display_name": "SDP-1-Unbalanced",
                "solver_converged": info.get("converged", False),
                "solver_n_iter": info.get("n_iter", np.nan),
                "solver_primal_residual": info.get(
                    "primal_residual",
                    np.nan,
                ),
                "solver_dual_residual": info.get(
                    "dual_residual",
                    np.nan,
                ),
                "solver_objective": info.get(
                    "objective_similarity",
                    np.nan,
                ),
                "top_eigenvalue": float(top_values[0]),
                "kth_eigenvalue": float(top_values[-1]),
            }
        )
        extra.update(matrix_diagnostics(X_sdp1, K))
        return labels, extra

    if method == "sdp2":
        X_sdp2, info = admm_sd2_unbalanced(
            A,
            K,
            rho=args.admm_rho,
            tol=args.admm_tol,
            max_iter=args.admm_max_iter,
            adaptive_rho=not args.no_adaptive_rho,
            verbose=args.verbose,
            return_info=True,
        )
        embedding, top_values = top_k_embedding(X_sdp2, K)
        labels = ordinary_kmeans(
            embedding,
            K,
            seed=seed,
            n_init=args.kmeans_n_init,
            row_normalize=args.row_normalize,
        )
        extra.update(
            {
                "display_name": "SDP-2-Unbalanced-Aggregated",
                "solver_converged": info.get("converged", False),
                "solver_n_iter": info.get("n_iter", np.nan),
                "solver_primal_residual": info.get(
                    "primal_residual",
                    np.nan,
                ),
                "solver_dual_residual": info.get(
                    "dual_residual",
                    np.nan,
                ),
                "solver_objective": info.get(
                    "objective_similarity",
                    np.nan,
                ),
                "trace_residual": info.get(
                    "trace_residual",
                    np.nan,
                ),
                "total_sum_residual": info.get(
                    "total_sum_residual",
                    np.nan,
                ),
                "row_sum_to_one_residual": info.get(
                    "row_sum_to_one_residual",
                    np.nan,
                ),
                "row_sum_mean": info.get(
                    "row_sum_mean",
                    np.nan,
                ),
                "row_sum_std": info.get(
                    "row_sum_std",
                    np.nan,
                ),
                "negative_violation_fro": info.get(
                    "negative_violation_fro",
                    np.nan,
                ),
                "minimum_entry": info.get(
                    "minimum_entry",
                    np.nan,
                ),
                "minimum_eigenvalue": info.get(
                    "minimum_eigenvalue",
                    np.nan,
                ),
                "maximum_eigenvalue": info.get(
                    "maximum_eigenvalue",
                    np.nan,
                ),
                "effective_rank_1e-6": info.get(
                    "effective_rank_1e-6",
                    np.nan,
                ),
                "top_eigenvalue": float(top_values[0]),
                "kth_eigenvalue": float(top_values[-1]),
            }
        )
        return labels, extra

    if method == "rpma":
        X_rpma, U_rpma, history = rpa(
            A,
            K,
            lam=args.rpma_lam,
            delta=args.rpma_delta,
            tau_max=args.rpma_tau_max,
            beta=args.rpma_backtrack_beta,
            sigma=args.rpma_armijo_sigma,
            tol=args.rpma_tol,
            max_iter=args.rpma_max_iter,
            eig_init=True,
            return_history=True,
            verbose=args.verbose,
        )
        labels = ordinary_kmeans(
            U_rpma,
            K,
            seed=seed,
            n_init=args.kmeans_n_init,
            row_normalize=args.row_normalize,
        )
        extra.update(
            {
                "display_name": "RPMA",
                "solver_n_iter": len(history),
                "solver_final_grad": (
                    float(history[-1]) if history else np.nan
                ),
                "idempotence_residual": float(
                    np.linalg.norm(X_rpma @ X_rpma - X_rpma)
                ),
                "row_sum_to_one_residual": float(
                    np.linalg.norm(
                        X_rpma @ np.ones(n) - np.ones(n)
                    )
                ),
                "negative_violation_fro": float(
                    np.linalg.norm(np.minimum(X_rpma, 0.0))
                ),
            }
        )
        return labels, extra

    if method == "ns_rpma":
        X_ns, U_ns, info = ns_rpma(
            A,
            K,
            lam=args.ns_lam,
            delta=args.ns_delta,
            nonnegative_mu=args.ns_mu,
            start_delta=args.ns_start_delta,
            continuation_steps=args.ns_continuation_steps,
            max_iter_per_stage=args.ns_stage_max_iter,
            tol=args.ns_tol,
            tau_max=args.ns_tau_max,
            tau_min=args.ns_tau_min,
            backtrack_beta=args.ns_backtrack_beta,
            armijo_sigma=args.ns_armijo_sigma,
            nonmonotone_window=args.ns_nonmonotone_window,
            verbose=args.verbose,
            return_info=True,
        )
        labels = ordinary_kmeans(
            U_ns,
            K,
            seed=seed,
            n_init=args.kmeans_n_init,
            row_normalize=args.row_normalize,
        )
        extra.update(
            {
                "display_name": "NS-RPMA",
                "solver_converged": info.get("converged", False),
                "solver_n_iter": info.get("n_iter", np.nan),
                "solver_final_grad": info.get(
                    "final_grad_norm",
                    np.nan,
                ),
                "solver_objective": info.get(
                    "final_objective",
                    np.nan,
                ),
                "idempotence_residual": info.get(
                    "idempotence_error",
                    np.nan,
                ),
                "row_sum_to_one_residual": info.get(
                    "row_sum_residual",
                    np.nan,
                ),
                "negative_violation_fro": info.get(
                    "negative_violation_fro",
                    np.nan,
                ),
                "minimum_entry": info.get("x_min", np.nan),
                "maximum_entry": info.get("x_max", np.nan),
            }
        )
        return labels, extra

    if method == "spectral_projection":
        embedding, top_values = top_k_embedding(A, K)
        labels = ordinary_kmeans(
            embedding,
            K,
            seed=seed,
            n_init=args.kmeans_n_init,
            row_normalize=args.row_normalize,
        )
        extra.update(
            {
                "display_name": "Spectral-Projection",
                "top_eigenvalue": float(top_values[0]),
                "kth_eigenvalue": float(top_values[-1]),
            }
        )
        return labels, extra

    if method == "normalized_cut":
        X_ncut, info = normalized_cut(
            A,
            K,
            random_state=seed,
            n_init=args.ncut_n_init,
            return_info=True,
        )
        labels, component_count = partition_projector_labels(
            X_ncut,
            K,
        )
        extra.update(
            {
                "display_name": "Normalized-Cut",
                "rounding": "internal_ncut_kmeans",
                "solver_converged": info.get("converged", True),
                "solver_n_iter": info.get("n_iter", 1),
                "ncut_n_init": args.ncut_n_init,
                "projector_components": component_count,
                "minimum_degree": info.get(
                    "minimum_degree",
                    np.nan,
                ),
                "maximum_degree": info.get(
                    "maximum_degree",
                    np.nan,
                ),
                "mean_degree": info.get(
                    "mean_degree",
                    np.nan,
                ),
                "top_eigenvalue": info.get(
                    "top_eigenvalue",
                    np.nan,
                ),
                "kth_eigenvalue": info.get(
                    "kth_eigenvalue",
                    np.nan,
                ),
            }
        )
        extra.update(matrix_diagnostics(X_ncut, K))
        return labels, extra

    if method == "regularized_spectral_clustering":
        tau = None if args.rsc_tau < 0.0 else args.rsc_tau
        X_rsc, info = regularized_spectral_clustering(
            A,
            K,
            tau=tau,
            random_state=seed,
            n_init=args.rsc_n_init,
            return_info=True,
        )
        labels, component_count = partition_projector_labels(
            X_rsc,
            K,
        )
        extra.update(
            {
                "display_name": "Regularized-Spectral-Clustering",
                "rounding": "internal_rsc_kmeans",
                "solver_converged": info.get("converged", True),
                "solver_n_iter": info.get("n_iter", 1),
                "rsc_tau": info.get("tau", np.nan),
                "rsc_n_init": args.rsc_n_init,
                "projector_components": component_count,
                "minimum_degree": info.get(
                    "minimum_degree",
                    np.nan,
                ),
                "maximum_degree": info.get(
                    "maximum_degree",
                    np.nan,
                ),
                "mean_degree": info.get(
                    "mean_degree",
                    np.nan,
                ),
                "top_eigenvalue": info.get(
                    "top_eigenvalue",
                    np.nan,
                ),
                "kth_eigenvalue": info.get(
                    "kth_eigenvalue",
                    np.nan,
                ),
            }
        )
        extra.update(matrix_diagnostics(X_rsc, K))
        return labels, extra

    if method == "kernel_kmeans":
        X_kkm, info = kernel_kmeans(
            A,
            K,
            random_state=seed,
            n_init=args.kkm_n_init,
            max_iter=args.kkm_max_iter,
            tol=args.kkm_tol,
            return_info=True,
        )
        labels, component_count = partition_projector_labels(
            X_kkm,
            K,
        )
        extra.update(
            {
                "display_name": "Kernel-KMeans",
                "rounding": "kernel_kmeans_assignments",
                "solver_converged": info.get("converged", False),
                "solver_n_iter": info.get("n_iter", np.nan),
                "solver_objective": info.get(
                    "objective",
                    np.nan,
                ),
                "kkm_n_init": info.get(
                    "n_init",
                    args.kkm_n_init,
                ),
                "kkm_max_iter": args.kkm_max_iter,
                "kkm_tol": args.kkm_tol,
                "projector_components": component_count,
                "top_eigenvalue": info.get(
                    "top_kernel_eigenvalue",
                    np.nan,
                ),
                "kth_eigenvalue": info.get(
                    "kth_kernel_eigenvalue",
                    np.nan,
                ),
            }
        )
        extra.update(matrix_diagnostics(X_kkm, K))
        return labels, extra

    if method == "symnmf":
        X_symnmf, info = symnmf(
            A,
            K,
            random_state=seed,
            n_init=args.symnmf_n_init,
            max_iter=args.symnmf_max_iter,
            tol=args.symnmf_tol,
            armijo_sigma=args.symnmf_armijo_sigma,
            backtrack_beta=args.symnmf_backtrack_beta,
            min_step=args.symnmf_min_step,
            return_info=True,
        )
        labels, component_count = partition_projector_labels(
            X_symnmf,
            K,
        )
        extra.update(
            {
                "display_name": "SymNMF",
                "rounding": "symnmf_row_argmax",
                "solver_converged": info.get("converged", False),
                "solver_n_iter": info.get("n_iter", np.nan),
                "solver_objective": info.get(
                    "objective",
                    np.nan,
                ),
                "solver_final_grad": info.get(
                    "projected_gradient_norm",
                    np.nan,
                ),
                "symnmf_n_init": info.get(
                    "n_init",
                    args.symnmf_n_init,
                ),
                "symnmf_max_iter": args.symnmf_max_iter,
                "symnmf_tol": args.symnmf_tol,
                "symnmf_factor_minimum": info.get(
                    "factor_minimum",
                    np.nan,
                ),
                "symnmf_factor_maximum": info.get(
                    "factor_maximum",
                    np.nan,
                ),
                "projector_components": component_count,
            }
        )
        extra.update(matrix_diagnostics(X_symnmf, K))
        return labels, extra

    if method == "clr":
        S_clr = clr(
            A,
            lam=args.clr_lam,
            K=K,
            max_iter=args.clr_max_iter,
        )
        labels = spectral_rounding(
            S_clr,
            K,
            random_state=seed,
            laplacian=True,
            row_normalize=args.row_normalize,
        )
        extra.update(
            {
                "display_name": "CLR",
                "clr_row_sum_residual": float(
                    np.linalg.norm(
                        np.sum(S_clr, axis=1) - 1.0
                    )
                ),
                "clr_minimum_entry": float(np.min(S_clr)),
                "clr_maximum_entry": float(np.max(S_clr)),
            }
        )
        return labels, extra

    if method == "slsa":
        if args.slsa_eta > 0:
            eta = int(args.slsa_eta)
        else:
            # Existing package convention: roughly eta_k undirected edges
            # retained per sample, plus the diagonal.
            eta = int(n + 2 * n * args.slsa_eta_k)

        eta = max(eta, n + 2)
        eta = min(eta, n * n)

        Z_slsa, U_slsa, info = slsa(
            A,
            K=K,
            eta=eta,
            theta=args.slsa_theta,
            tau=args.slsa_tau,
            loss=args.slsa_loss,
            max_iter=args.slsa_max_iter,
            eta_mode="total",
            return_info=True,
            verbose=args.verbose,
        )

        if args.slsa_rounding == "U":
            labels = ordinary_kmeans(
                U_slsa,
                K,
                seed=seed,
                n_init=args.kmeans_n_init,
                row_normalize=args.row_normalize,
            )
        elif args.slsa_rounding == "top_eigen":
            embedding, _ = top_k_embedding(Z_slsa, K)
            labels = ordinary_kmeans(
                embedding,
                K,
                seed=seed,
                n_init=args.kmeans_n_init,
                row_normalize=args.row_normalize,
            )
        elif args.slsa_rounding == "laplacian":
            labels = spectral_rounding(
                Z_slsa,
                K,
                random_state=seed,
                laplacian=True,
                row_normalize=args.row_normalize,
            )
        else:
            raise ValueError(
                f"Unknown slsa_rounding={args.slsa_rounding!r}."
            )

        extra.update(
            {
                "display_name": "SLSA",
                "solver_converged": info.get("converged", False),
                "solver_n_iter": info.get("n_iter", np.nan),
                "solver_final_diff": info.get(
                    "final_diff",
                    np.nan,
                ),
                "slsa_eta": eta,
                "slsa_nnz": info.get("nnz", np.nan),
                "slsa_rounding": args.slsa_rounding,
            }
        )
        return labels, extra

    raise ValueError(f"Unhandled method {method!r}.")


# ---------------------------------------------------------------------------
# Experiment loop and result saving
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    dataset = canonical_dataset_name(args.dataset)
    data_root = args.data_root or default_data_root(dataset)
    image_size = parse_image_size(args.image_size)
    methods = parse_methods(args.methods)
    seeds = parse_seeds(args.seeds)

    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    X_full, y_full, original_K = load_dataset(
        dataset,
        data_root,
        image_size,
    )
    original_n, feature_dimension = X_full.shape

    print("=" * 100)
    print("Centralized unequal-class experiment")
    print(f"dataset              = {dataset}")
    print(f"data_root            = {data_root}")
    print(f"original n / K       = {original_n} / {original_K}")
    print(f"feature dimension    = {feature_dimension}")
    print(f"image_size           = {args.image_size}")
    print(f"sample_percent       = {args.sample_percent}%")
    print(f"sampling_mode        = {args.sampling_mode}")
    print(f"imbalance_alpha      = {args.imbalance_alpha}")
    print(f"min_per_class        = {args.min_per_class}")
    print(f"methods              = {','.join(methods)}")
    print(f"common affinity      = {args.affinity}")
    print(f"bandwidth / scale    = {args.bandwidth} / {args.bandwidth_scale}")
    print(f"standardize          = {args.standardize}")
    print(f"row_normalize        = {args.row_normalize}")
    print("rounding             = ordinary K-means; never balanced")
    print("=" * 100)

    result_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []

    for seed in seeds:
        (
            X_sample,
            y_sample,
            selected_indices,
            count_map,
        ) = sample_unequal_fixed_percentage(
            X_full,
            y_full,
            sample_percent=args.sample_percent,
            sampling_mode=args.sampling_mode,
            imbalance_alpha=args.imbalance_alpha,
            min_per_class=args.min_per_class,
            seed=seed,
        )

        n = int(y_sample.size)
        K = int(np.unique(y_sample).size)
        count_stats = class_count_statistics(count_map)

        X_used = (
            standardize_features(X_sample)
            if args.standardize
            else np.asarray(X_sample, dtype=np.float64)
        )

        affinity_start = time.perf_counter()
        A, affinity_info = build_common_affinity(
            X_used,
            args,
        )
        affinity_time = time.perf_counter() - affinity_start

        print("\n" + "-" * 100)
        print(f"seed                 = {seed}")
        print(f"sample n / K         = {n} / {K}")
        print(
            "class imbalance      = "
            f"min {count_stats['class_count_min']}, "
            f"max {count_stats['class_count_max']}, "
            f"mean {count_stats['class_count_mean']:.2f}, "
            f"CV {count_stats['class_count_cv']:.4f}, "
            f"max/min {count_stats['class_count_max_min_ratio']:.3f}"
        )
        print("class counts         = " + format_count_map(count_map))
        print(
            f"affinity time        = {affinity_time:.2f}s, "
            f"density={affinity_info.get('affinity_density', np.nan):.4f}"
        )

        np.savez_compressed(
            output_directory / f"{dataset}_seed{seed}_sample.npz",
            selected_indices=selected_indices,
            y_true=y_sample,
            class_labels=np.asarray(sorted(count_map), dtype=int),
            class_counts=np.asarray(
                [count_map[key] for key in sorted(count_map)],
                dtype=int,
            ),
        )

        for class_label, count in sorted(count_map.items()):
            count_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "class_label": class_label,
                    "retained_count": count,
                }
            )

        for method_index, method in enumerate(methods):
            # Separate but reproducible K-means seed for every method.
            method_seed = seed + 100_003 * (method_index + 1)
            method_start = time.perf_counter()

            try:
                labels, extra = run_one_method(
                    method,
                    A,
                    y_sample,
                    K,
                    method_seed,
                    args,
                )
                method_time = time.perf_counter() - method_start
                metrics = evaluate(y_sample, labels)
                error_text = ""

                print(
                    f"{extra.get('display_name', method):<24} "
                    f"ACC={metrics['ACC']:.6f}  "
                    f"NMI={metrics['NMI']:.6f}  "
                    f"ARI={metrics['ARI']:.6f}  "
                    f"time={method_time:.2f}s"
                )

                np.savez_compressed(
                    output_directory
                    / f"{dataset}_seed{seed}_{method}_labels.npz",
                    y_true=y_sample,
                    y_pred=labels,
                )

            except Exception as error:
                method_time = time.perf_counter() - method_start
                metrics = {
                    "ACC": np.nan,
                    "NMI": np.nan,
                    "ARI": np.nan,
                }
                extra = {
                    "display_name": method,
                    "traceback": traceback.format_exc(),
                }
                error_text = repr(error)
                print(
                    f"{method:<24} FAILED after {method_time:.2f}s: "
                    f"{error_text}"
                )

            row: dict[str, Any] = {
                "dataset": dataset,
                "method": method,
                "display_name": extra.get(
                    "display_name",
                    method,
                ),
                "seed": seed,
                "original_n": original_n,
                "n": n,
                "K": K,
                "feature_dimension": feature_dimension,
                "image_size": args.image_size,
                "sample_percent": args.sample_percent,
                "sampling_mode": args.sampling_mode,
                "imbalance_alpha": args.imbalance_alpha,
                "min_per_class": args.min_per_class,
                "standardize": args.standardize,
                "affinity": args.affinity,
                "affinity_k": args.affinity_k,
                "bandwidth": args.bandwidth,
                "bandwidth_scale": args.bandwidth_scale,
                "affinity_symmetrize": args.affinity_symmetrize,
                "zero_diagonal": args.zero_diagonal,
                "row_normalize": args.row_normalize,
                "rounding": "ordinary_kmeans",
                "ACC": metrics["ACC"],
                "NMI": metrics["NMI"],
                "ARI": metrics["ARI"],
                "affinity_time_sec": affinity_time,
                "method_time_sec": method_time,
                "total_time_sec": affinity_time + method_time,
                "error": error_text,
                **count_stats,
                **{
                    f"affinity_{key}": value
                    for key, value in affinity_info.items()
                },
            }

            # Keep scalar diagnostics in CSV. Long histories and tracebacks are
            # excluded unless an error occurs.
            for key, value in extra.items():
                if key == "display_name":
                    continue
                if key == "traceback":
                    if error_text:
                        row["traceback"] = value
                    continue
                if np.isscalar(value) or value is None:
                    row[key] = value

            result_rows.append(row)

    results = pd.DataFrame(result_rows)
    results_path = (
        output_directory / "unbalanced_all_methods_results.csv"
    )
    results.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    successful = results[
        results["error"].fillna("").eq("")
    ].copy()

    if successful.empty:
        summary = pd.DataFrame(
            columns=[
                "dataset",
                "method",
                "display_name",
                "runs",
                "ACC_mean",
                "ACC_std",
                "NMI_mean",
                "NMI_std",
                "ARI_mean",
                "ARI_std",
                "time_mean_sec",
            ]
        )
    else:
        summary = (
            successful.groupby(
                ["dataset", "method", "display_name"],
                as_index=False,
            )
            .agg(
                runs=("seed", "count"),
                ACC_mean=("ACC", "mean"),
                ACC_std=("ACC", lambda x: x.std(ddof=0)),
                NMI_mean=("NMI", "mean"),
                NMI_std=("NMI", lambda x: x.std(ddof=0)),
                ARI_mean=("ARI", "mean"),
                ARI_std=("ARI", lambda x: x.std(ddof=0)),
                time_mean_sec=("method_time_sec", "mean"),
                class_count_cv_mean=("class_count_cv", "mean"),
            )
        )

    summary_path = (
        output_directory / "unbalanced_all_methods_summary.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    counts_path = (
        output_directory / "unbalanced_sample_counts.csv"
    )
    pd.DataFrame(count_rows).to_csv(
        counts_path,
        index=False,
        encoding="utf-8-sig",
    )

    config_path = output_directory / "run_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            vars(args),
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 100)
    print("Final summary")
    if summary.empty:
        print("No method completed successfully.")
    else:
        columns = [
            "display_name",
            "runs",
            "ACC_mean",
            "ACC_std",
            "NMI_mean",
            "NMI_std",
            "ARI_mean",
            "ARI_std",
            "time_mean_sec",
        ]
        print(summary[columns].to_string(index=False))
    print(f"Raw results   : {results_path}")
    print(f"Summary       : {summary_path}")
    print(f"Class counts  : {counts_path}")
    print(f"Configuration : {config_path}")
    print("=" * 100)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the original methods plus Normalized Cut, Regularized "
            "Spectral Clustering, Kernel K-means and SymNMF on exactly "
            "the same unequal-size image subset."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "coil20",
            "att",
            "att_face",
            "att_faces",
            "attfaces",
            "orl",
        ],
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--image-size",
        default="original",
        help="'original', one integer, or WIDTHxHEIGHT.",
    )

    parser.add_argument(
        "--sample-percent",
        type=float,
        required=True,
        help="Exact percentage of all original images to retain.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["dirichlet", "global", "balanced"],
        default="dirichlet",
        help=(
            "dirichlet explicitly creates unequal class counts; "
            "global performs milder random sampling; balanced is a control."
        ),
    )
    parser.add_argument(
        "--imbalance-alpha",
        type=float,
        default=0.7,
        help=(
            "Dirichlet concentration. Smaller values mean stronger "
            "class imbalance."
        ),
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated experiment seeds, e.g. 0,1,2,3,4.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help=(
            "all, or a comma-separated subset of "
            "sdp1,sdp2,rpma,ns_rpma,spectral_projection,normalized_cut,"
            "regularized_spectral_clustering,kernel_kmeans,symnmf,"
            "clr,slsa."
        ),
    )

    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Feature-wise standardization; no PCA is used.",
    )
    parser.add_argument(
        "--affinity",
        choices=[
            "full_gaussian",
            "knn_gaussian",
            "self_tuning",
            "cosine",
            "binary_knn",
            "linear_gram",
        ],
        default="full_gaussian",
    )
    parser.add_argument("--affinity-k", type=int, default=10)
    parser.add_argument(
        "--bandwidth",
        choices=["mean", "median"],
        default="mean",
    )
    parser.add_argument(
        "--bandwidth-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--affinity-symmetrize",
        choices=["max", "mean"],
        default="max",
    )
    parser.add_argument(
        "--zero-diagonal",
        action="store_true",
    )
    parser.add_argument(
        "--cosine-knn",
        action="store_true",
    )

    parser.add_argument(
        "--row-normalize",
        action="store_true",
        help=(
            "L2-normalize spectral embedding rows before ordinary K-means. "
            "Disabled by default."
        ),
    )
    parser.add_argument("--kmeans-n-init", type=int, default=50)

    parser.add_argument("--admm-rho", type=float, default=1.0)
    parser.add_argument("--admm-tol", type=float, default=1e-3)
    parser.add_argument("--admm-max-iter", type=int, default=200)
    parser.add_argument(
        "--no-adaptive-rho",
        action="store_true",
        help="Applies only to the modified SDP-1 solver.",
    )

    parser.add_argument("--rpma-lam", type=float, default=0.1)
    parser.add_argument("--rpma-delta", type=float, default=0.1)
    parser.add_argument("--rpma-tol", type=float, default=1e-5)
    parser.add_argument("--rpma-max-iter", type=int, default=200)
    parser.add_argument("--rpma-tau-max", type=float, default=1.0)
    parser.add_argument(
        "--rpma-backtrack-beta",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--rpma-armijo-sigma",
        type=float,
        default=1e-4,
    )

    parser.add_argument("--ns-lam", type=float, default=0.1)
    parser.add_argument("--ns-delta", type=float, default=0.1)
    parser.add_argument("--ns-mu", type=float, default=1.0)
    parser.add_argument(
        "--ns-start-delta",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--ns-continuation-steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--ns-stage-max-iter",
        type=int,
        default=100,
    )
    parser.add_argument("--ns-tol", type=float, default=1e-5)
    parser.add_argument("--ns-tau-max", type=float, default=1.0)
    parser.add_argument("--ns-tau-min", type=float, default=1e-14)
    parser.add_argument(
        "--ns-backtrack-beta",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--ns-armijo-sigma",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--ns-nonmonotone-window",
        type=int,
        default=5,
    )

    # Added mainstream graph-clustering baselines.
    parser.add_argument(
        "--ncut-n-init",
        type=int,
        default=50,
        help="K-means restarts inside Normalized Cut.",
    )

    parser.add_argument(
        "--rsc-tau",
        type=float,
        default=-1.0,
        help=(
            "RSC degree regularization. A negative value selects the "
            "average weighted degree automatically."
        ),
    )
    parser.add_argument(
        "--rsc-n-init",
        type=int,
        default=50,
        help="K-means restarts inside RSC.",
    )

    parser.add_argument("--kkm-n-init", type=int, default=5)
    parser.add_argument("--kkm-max-iter", type=int, default=100)
    parser.add_argument("--kkm-tol", type=float, default=1e-6)

    parser.add_argument("--symnmf-n-init", type=int, default=3)
    parser.add_argument("--symnmf-max-iter", type=int, default=200)
    parser.add_argument("--symnmf-tol", type=float, default=1e-5)
    parser.add_argument(
        "--symnmf-armijo-sigma",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--symnmf-backtrack-beta",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--symnmf-min-step",
        type=float,
        default=1e-12,
    )

    parser.add_argument("--clr-lam", type=float, default=1.0)
    parser.add_argument("--clr-max-iter", type=int, default=100)

    parser.add_argument(
        "--slsa-eta",
        type=int,
        default=0,
        help=(
            "Explicit total nonzero budget. Set 0 to use n+2*n*eta_k."
        ),
    )
    parser.add_argument(
        "--slsa-eta-k",
        type=int,
        default=10,
    )
    parser.add_argument("--slsa-theta", type=float, default=1.0)
    parser.add_argument("--slsa-tau", type=float, default=1e-6)
    parser.add_argument(
        "--slsa-loss",
        choices=["fro", "l1"],
        default="fro",
    )
    parser.add_argument("--slsa-max-iter", type=int, default=100)
    parser.add_argument(
        "--slsa-rounding",
        choices=["U", "top_eigen", "laplacian"],
        default="U",
    )

    parser.add_argument(
        "--out-dir",
        default="results/unbalanced_all_methods",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print solver iteration diagnostics.",
    )

    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
