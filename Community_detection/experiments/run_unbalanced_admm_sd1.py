"""
Unequal-size image clustering experiment for modified ADMM-SD1.

The experiment:
  1. Loads COIL20 or AT&T/ORL faces.
  2. Selects an exact fixed percentage of all images.
  3. Randomly allocates the retained sample count across classes to create
     unequal class sizes (Dirichlet mode), while retaining every class.
  4. Builds a dense Gaussian affinity matrix A.
  5. Solves the unbalanced Projection-SDP.
  6. Uses ordinary k-means (never balanced rounding).
  7. Reports ACC, NMI and ARI.

Place this file at:
    Community_detection/experiments/run_unbalanced_admm_sd1.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from datasets.image_datasets import (
    load_att_faces,
    load_coil20,
)
from evaluation.metrics import evaluate
from methods.admm_sd1_unbalanced import admm_sd1_unbalanced
from methods.affinity import build_affinity


def parse_size(value: str | None):
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
    seeds = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def resolve_dataset_name(name: str) -> str:
    name = str(name).strip().lower()
    if name == "coil20":
        return "coil20"
    if name in {"att", "att_face", "att_faces", "orl"}:
        return "att_faces"
    raise ValueError("dataset must be coil20 or att_faces.")


def default_data_root(dataset: str) -> str:
    if dataset == "coil20":
        return "datasets/data/coil20"
    return "datasets/data/att_faces"


def load_full_dataset(
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


def _allocate_dirichlet_counts(
    capacities: np.ndarray,
    target_total: int,
    min_per_class: int,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Allocate an exact total across classes with unequal random counts."""
    capacities = np.asarray(capacities, dtype=int)
    K = capacities.size

    if alpha <= 0.0:
        raise ValueError("imbalance_alpha must be positive.")
    if min_per_class < 1:
        raise ValueError("min_per_class must be at least 1.")
    if np.any(capacities < min_per_class):
        raise ValueError(
            "At least one class has fewer samples than min_per_class."
        )

    minimum_total = K * min_per_class
    maximum_total = int(np.sum(capacities))
    if not minimum_total <= target_total <= maximum_total:
        raise ValueError(
            f"Requested target_total={target_total} is infeasible; "
            f"must lie in [{minimum_total}, {maximum_total}]."
        )

    counts = np.full(K, min_per_class, dtype=int)
    remaining = target_total - minimum_total
    weights = rng.dirichlet(np.full(K, alpha, dtype=float))

    # Exact capacity-constrained allocation. Dataset sizes are small enough
    # that the simple loop is both clear and fast.
    for _ in range(remaining):
        available = counts < capacities
        probabilities = np.where(available, weights, 0.0)
        total_probability = float(np.sum(probabilities))
        if total_probability <= 0.0:
            raise RuntimeError("No capacity remains during sample allocation.")
        probabilities /= total_probability
        selected_class = int(rng.choice(K, p=probabilities))
        counts[selected_class] += 1

    return counts


def _allocate_balanced_counts(
    capacities: np.ndarray,
    target_total: int,
    min_per_class: int,
) -> np.ndarray:
    """Near-equal allocation, provided as a balanced control experiment."""
    capacities = np.asarray(capacities, dtype=int)
    K = capacities.size
    if np.any(capacities < min_per_class):
        raise ValueError("A class has fewer samples than min_per_class.")

    counts = np.full(K, min_per_class, dtype=int)
    remaining = target_total - int(np.sum(counts))
    class_cursor = 0
    while remaining > 0:
        class_index = class_cursor % K
        if counts[class_index] < capacities[class_index]:
            counts[class_index] += 1
            remaining -= 1
        class_cursor += 1
        if class_cursor > target_total * K * 2:
            raise RuntimeError("Could not allocate balanced sample counts.")
    return counts


def _allocate_global_counts(
    y: np.ndarray,
    target_total: int,
    min_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform global sampling after reserving a minimum for every class."""
    classes = np.unique(y)
    counts = np.full(classes.size, min_per_class, dtype=int)
    capacities = np.array([np.sum(y == c) for c in classes], dtype=int)

    remaining = target_total - int(np.sum(counts))
    if remaining < 0:
        raise ValueError("sample_percent is too small for min_per_class.")

    # Sampling uniformly from all remaining image slots gives random class
    # counts but less severe imbalance than Dirichlet mode.
    available_class_slots = np.repeat(
        np.arange(classes.size),
        capacities - counts,
    )
    chosen_slots = rng.choice(
        available_class_slots.size,
        size=remaining,
        replace=False,
    )
    extra_classes = available_class_slots[chosen_slots]
    counts += np.bincount(extra_classes, minlength=classes.size)
    return counts


def sample_fixed_percentage(
    X: np.ndarray,
    y: np.ndarray,
    *,
    percentage: float,
    mode: str,
    seed: int,
    imbalance_alpha: float,
    min_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    """Select an exact percentage while controlling class imbalance."""
    X = np.asarray(X)
    y = np.asarray(y, dtype=int)
    if X.shape[0] != y.size:
        raise ValueError("X and y have inconsistent sample counts.")
    if not 0.0 < percentage <= 100.0:
        raise ValueError("sample_percent must lie in (0, 100].")

    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    capacities = np.array([np.sum(y == c) for c in classes], dtype=int)

    target_total = int(round(y.size * percentage / 100.0))
    target_total = min(target_total, y.size)
    target_total = max(target_total, classes.size * min_per_class)

    if target_total > y.size:
        raise ValueError(
            "The requested percentage and min_per_class require more "
            "samples than the dataset contains."
        )

    mode = str(mode).lower()
    if mode == "dirichlet":
        class_counts = _allocate_dirichlet_counts(
            capacities,
            target_total,
            min_per_class,
            imbalance_alpha,
            rng,
        )
    elif mode == "global":
        class_counts = _allocate_global_counts(
            y,
            target_total,
            min_per_class,
            rng,
        )
    elif mode == "balanced":
        class_counts = _allocate_balanced_counts(
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
            size=int(class_counts[class_position]),
            replace=False,
        )
        selected_indices.extend(chosen.tolist())
        count_map[int(class_label)] = int(class_counts[class_position])

    selected_indices = np.asarray(selected_indices, dtype=int)
    rng.shuffle(selected_indices)

    return (
        X[selected_indices],
        y[selected_indices],
        selected_indices,
        count_map,
    )


def standardize_features(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    centered = X - np.mean(X, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, keepdims=True)
    return centered / np.maximum(scale, eps)


def round_projection_sdp(
    X: np.ndarray,
    K: int,
    *,
    seed: int,
    n_init: int,
    row_normalize: bool,
    eigenvalue_weighted: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-K spectral embedding followed by ordinary (unbalanced) k-means."""
    X = 0.5 * (np.asarray(X, dtype=np.float64) + np.asarray(X).T)
    n = X.shape[0]

    if K == n:
        eigenvalues, eigenvectors = eigh(X)
    else:
        eigenvalues, eigenvectors = eigh(
            X,
            subset_by_index=[n - K, n - 1],
            check_finite=False,
        )

    # scipy.linalg.eigh returns ascending eigenvalues.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    embedding = eigenvectors[:, order]

    if eigenvalue_weighted:
        embedding = embedding * np.sqrt(
            np.maximum(eigenvalues, 0.0)
        )[None, :]

    if row_normalize:
        embedding = normalize(embedding, norm="l2")

    labels = KMeans(
        n_clusters=K,
        n_init=n_init,
        random_state=seed,
    ).fit_predict(embedding)

    return labels, embedding, eigenvalues


def class_count_statistics(count_map: dict[int, int]) -> dict[str, float | int]:
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


def run(args: argparse.Namespace) -> None:
    dataset = resolve_dataset_name(args.dataset)
    image_size = parse_size(args.image_size)
    data_root = args.data_root or default_data_root(dataset)
    seeds = parse_seeds(args.seeds)

    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Unequal-size ADMM-SD1 image experiment")
    print("Model: X >= 0, X PSD, X1=1, trace(X)=K")
    print(f"dataset          = {dataset}")
    print(f"data_root        = {data_root}")
    print(f"image_size       = {args.image_size}")
    print(f"sample_percent   = {args.sample_percent}% (exact total after rounding)")
    print(f"sampling_mode    = {args.sampling_mode}")
    print(f"imbalance_alpha  = {args.imbalance_alpha}")
    print(f"min_per_class    = {args.min_per_class}")
    print("PCA              = none")
    print(
        "affinity         = full Gaussian, "
        f"bandwidth={args.bandwidth}, scale={args.bandwidth_scale}"
    )
    print(
        "ADMM             = "
        f"rho={args.rho}, tol={args.tol}, max_iter={args.max_iter}, "
        f"adaptive_rho={not args.no_adaptive_rho}"
    )
    print("rounding         = ordinary k-means (not balanced)")
    print("=" * 88)

    X_full, y_full, original_K = load_full_dataset(
        dataset,
        data_root,
        image_size,
    )
    original_n, feature_dimension = X_full.shape

    all_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []

    for seed in seeds:
        X_sample, y_sample, selected_indices, count_map = (
            sample_fixed_percentage(
                X_full,
                y_full,
                percentage=args.sample_percent,
                mode=args.sampling_mode,
                seed=seed,
                imbalance_alpha=args.imbalance_alpha,
                min_per_class=args.min_per_class,
            )
        )

        retained_classes = np.unique(y_sample)
        K = retained_classes.size
        n = y_sample.size
        count_stats = class_count_statistics(count_map)

        print("\n" + "-" * 88)
        print(f"seed={seed}")
        print(
            f"full n={original_n}, retained n={n}, K={K}, "
            f"features={feature_dimension}"
        )
        print(
            "class counts: "
            f"min={count_stats['class_count_min']}, "
            f"max={count_stats['class_count_max']}, "
            f"mean={count_stats['class_count_mean']:.2f}, "
            f"CV={count_stats['class_count_cv']:.3f}"
        )
        print(
            "counts by class: "
            + ", ".join(
                f"{class_id}:{count_map[class_id]}"
                for class_id in sorted(count_map)
            )
        )

        X_used = (
            standardize_features(X_sample)
            if args.standardize
            else np.asarray(X_sample, dtype=np.float64)
        )

        affinity_start = time.perf_counter()
        A, affinity_info = build_affinity(
            X_used,
            graph="full_gaussian",
            bandwidth=args.bandwidth,
            sigma2_scale=args.bandwidth_scale,
            zero_diagonal=args.zero_diagonal,
        )
        affinity_time = time.perf_counter() - affinity_start

        solver_start = time.perf_counter()
        X_sdp, solver_info = admm_sd1_unbalanced(
            A,
            K,
            rho=args.rho,
            tol=args.tol,
            max_iter=args.max_iter,
            adaptive_rho=not args.no_adaptive_rho,
            verbose=args.verbose,
            return_info=True,
        )
        solver_time = time.perf_counter() - solver_start

        labels, embedding, top_eigenvalues = round_projection_sdp(
            X_sdp,
            K,
            seed=seed,
            n_init=args.kmeans_n_init,
            row_normalize=not args.no_row_normalize,
            eigenvalue_weighted=args.eigenvalue_weighted,
        )

        metrics = evaluate(y_sample, labels)
        total_time = affinity_time + solver_time

        print("\nResult")
        print(f"ACC               = {metrics['ACC']:.6f}")
        print(f"NMI               = {metrics['NMI']:.6f}")
        print(f"ARI               = {metrics['ARI']:.6f}")
        print(f"ADMM converged    = {solver_info['converged']}")
        print(f"ADMM iterations   = {solver_info['n_iter']}")
        print(
            f"primal / dual     = "
            f"{solver_info['primal_residual']:.3e} / "
            f"{solver_info['dual_residual']:.3e}"
        )
        print(
            f"row / trace error = "
            f"{solver_info['row_sum_residual']:.3e} / "
            f"{solver_info['trace_residual']:.3e}"
        )
        print(
            f"min entry / eig   = "
            f"{solver_info['minimum_entry']:.3e} / "
            f"{solver_info['minimum_eigenvalue']:.3e}"
        )
        print(
            f"affinity / solver / total time = "
            f"{affinity_time:.2f}s / {solver_time:.2f}s / {total_time:.2f}s"
        )

        row: dict[str, object] = {
            "dataset": dataset,
            "seed": seed,
            "original_n": original_n,
            "n": n,
            "K": K,
            "feature_dimension": feature_dimension,
            "sample_percent": args.sample_percent,
            "sampling_mode": args.sampling_mode,
            "imbalance_alpha": args.imbalance_alpha,
            "min_per_class": args.min_per_class,
            "standardize": args.standardize,
            "bandwidth": args.bandwidth,
            "bandwidth_scale": args.bandwidth_scale,
            "sigma2": affinity_info.get("sigma2", np.nan),
            "zero_diagonal": args.zero_diagonal,
            "rho_initial": args.rho,
            "rho_final": solver_info["rho_final"],
            "tol": args.tol,
            "max_iter": args.max_iter,
            "adaptive_rho": not args.no_adaptive_rho,
            "rounding": "ordinary_kmeans",
            "row_normalize": not args.no_row_normalize,
            "eigenvalue_weighted": args.eigenvalue_weighted,
            "ACC": metrics["ACC"],
            "NMI": metrics["NMI"],
            "ARI": metrics["ARI"],
            "affinity_time_sec": affinity_time,
            "solver_time_sec": solver_time,
            "total_time_sec": total_time,
            "converged": solver_info["converged"],
            "n_iter": solver_info["n_iter"],
            "primal_residual": solver_info["primal_residual"],
            "dual_residual": solver_info["dual_residual"],
            "row_sum_residual": solver_info["row_sum_residual"],
            "trace_residual": solver_info["trace_residual"],
            "negative_violation_fro": solver_info[
                "negative_violation_fro"
            ],
            "minimum_entry": solver_info["minimum_entry"],
            "minimum_eigenvalue": solver_info["minimum_eigenvalue"],
            "maximum_eigenvalue": solver_info["maximum_eigenvalue"],
            "effective_rank_1e-6": solver_info[
                "effective_rank_1e-6"
            ],
            **count_stats,
        }
        all_rows.append(row)

        for class_id, count in sorted(count_map.items()):
            count_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "class_id": class_id,
                    "retained_count": count,
                }
            )

        seed_prefix = (
            output_directory
            / f"{dataset}_seed{seed}_pct{args.sample_percent:g}"
        )
        np.savez_compressed(
            str(seed_prefix) + "_sample.npz",
            selected_indices=selected_indices,
            y_true=y_sample,
            y_pred=labels,
            embedding=embedding,
            top_eigenvalues=top_eigenvalues,
        )

        if args.save_matrix:
            np.savez_compressed(
                str(seed_prefix) + "_matrix.npz",
                A=A,
                X_sdp=X_sdp,
            )

        # Keep solver history separate because it can contain many rows.
        pd.DataFrame(solver_info["history"]).to_csv(
            str(seed_prefix) + "_admm_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

    results = pd.DataFrame(all_rows)
    results_path = output_directory / "unbalanced_admm_sd1_results.csv"
    results.to_csv(results_path, index=False, encoding="utf-8-sig")

    count_table = pd.DataFrame(count_rows)
    count_path = output_directory / "unbalanced_sample_counts.csv"
    count_table.to_csv(count_path, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "runs": len(results),
                "ACC_mean": float(results["ACC"].mean()),
                "ACC_std": float(results["ACC"].std(ddof=0)),
                "NMI_mean": float(results["NMI"].mean()),
                "NMI_std": float(results["NMI"].std(ddof=0)),
                "ARI_mean": float(results["ARI"].mean()),
                "ARI_std": float(results["ARI"].std(ddof=0)),
                "n_mean": float(results["n"].mean()),
                "class_count_cv_mean": float(
                    results["class_count_cv"].mean()
                ),
                "solver_time_mean_sec": float(
                    results["solver_time_sec"].mean()
                ),
                "convergence_rate": float(
                    results["converged"].astype(float).mean()
                ),
            }
        ]
    )
    summary_path = output_directory / "unbalanced_admm_sd1_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    config_path = output_directory / "run_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 88)
    print("Summary")
    print(summary.to_string(index=False))
    print(f"Raw results   : {results_path}")
    print(f"Class counts  : {count_path}")
    print(f"Summary       : {summary_path}")
    print(f"Configuration : {config_path}")
    print("=" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run unequal-size Projection-SDP on a fixed random percentage "
            "of COIL20 or AT&T faces."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["coil20", "att", "att_face", "att_faces", "orl"],
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Defaults to datasets/data/coil20 or "
            "datasets/data/att_faces."
        ),
    )
    parser.add_argument(
        "--image-size",
        default="original",
        help="'original' or a size such as 32x32.",
    )

    parser.add_argument(
        "--sample-percent",
        type=float,
        required=True,
        help="Exact percentage of the complete dataset to retain.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["dirichlet", "global", "balanced"],
        default="dirichlet",
        help=(
            "dirichlet creates explicit unequal class sizes; global gives "
            "milder random imbalance; balanced is a control."
        ),
    )
    parser.add_argument(
        "--imbalance-alpha",
        type=float,
        default=0.5,
        help=(
            "Dirichlet concentration. Smaller values create stronger "
            "class imbalance. Used only in dirichlet mode."
        ),
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=2,
        help="Minimum retained images in every class.",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated sampling and k-means seeds, e.g. 0,1,2.",
    )

    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Feature-wise standardization; no PCA is ever applied.",
    )
    parser.add_argument(
        "--bandwidth",
        choices=["mean", "median"],
        default="mean",
    )
    parser.add_argument(
        "--bandwidth-scale",
        type=float,
        default=1.0,
        help="Multiplier of the ordinary Gaussian sigma^2.",
    )
    parser.add_argument(
        "--zero-diagonal",
        action="store_true",
    )

    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument(
        "--no-adaptive-rho",
        action="store_true",
    )

    parser.add_argument("--kmeans-n-init", type=int, default=50)
    parser.add_argument(
        "--no-row-normalize",
        action="store_true",
    )
    parser.add_argument(
        "--eigenvalue-weighted",
        action="store_true",
        help="Weight top eigenvectors by sqrt(eigenvalue).",
    )

    parser.add_argument(
        "--out-dir",
        default="results/unbalanced_admm_sd1",
    )
    parser.add_argument(
        "--save-matrix",
        action="store_true",
        help="Also save A and the SDP solution X; these can be large.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print ADMM iteration diagnostics.",
    )

    run(parser.parse_args())


if __name__ == "__main__":
    main()
