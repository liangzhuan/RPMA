"""Run and compare NS-RPMA with Adaptive Graph Refinement NS-RPMA.

Place at:
    Community_detection/experiments/run_agr_ns_rpma.py

The script uses the same initial dense Gaussian affinity for both methods,
reports ACC/NMI/ARI, saves matrices, and plots true-label-ordered heatmaps of
A0, the learned graph S and the two projection matrices.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    pairwise_distances,
)

from datasets import image_datasets as image_ds
from methods.agr_ns_rpma import agr_ns_rpma
from methods.ns_rpma import ns_rpma


def parse_image_size(value: str):
    value = str(value).strip().lower()
    if value in {"original", "orig", "none"}:
        return None
    if "x" in value:
        h, w = value.split("x", 1)
        return int(h), int(w)
    side = int(value)
    return side, side


def load_dataset(args: argparse.Namespace):
    kwargs = {
        "image_size": parse_image_size(args.image_size),
        "max_per_class": args.max_per_class,
        "random_state": args.data_seed,
    }
    if args.dataset == "coil20":
        return image_ds.load_coil20(args.data_root, **kwargs)
    if args.dataset == "att_faces":
        return image_ds.load_att_faces(args.data_root, **kwargs)
    if args.dataset == "yaleB":
        return image_ds.load_extended_yale_b(args.data_root, **kwargs)
    if args.dataset == "synthetic":
        return make_synthetic(args)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def make_synthetic(args: argparse.Namespace):
    rng = np.random.default_rng(args.data_seed)
    K = args.synthetic_k
    n_per = args.synthetic_per_class
    dim = args.synthetic_dim
    centers = rng.normal(size=(K, dim)) * args.synthetic_separation
    X = []
    y = []
    for k in range(K):
        X.append(centers[k] + rng.normal(scale=1.0, size=(n_per, dim)))
        y.extend([k] * n_per)
    return np.vstack(X).astype(np.float64), np.asarray(y), K


def dense_gaussian_affinity(features: np.ndarray, scale: float, keep_diagonal: bool):
    t0 = time.perf_counter()
    d2 = pairwise_distances(features, metric="sqeuclidean", n_jobs=1)
    d2 = np.maximum(d2, 0.0)
    n = d2.shape[0]
    upper = np.sqrt(d2[np.triu_indices(n, k=1)])
    positive = upper[upper > 0.0]
    if positive.size == 0:
        raise ValueError("All pairwise distances are zero.")
    sigma0 = float(np.mean(positive))
    sigma = float(scale) * sigma0
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("bandwidth-scale must be positive.")
    A = np.exp(-d2 / (2.0 * sigma * sigma))
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 1.0 if keep_diagonal else 0.0)
    off = A[np.triu_indices(n, k=1)]
    return A, {
        "sigma0": sigma0,
        "sigma": sigma,
        "bandwidth_scale": float(scale),
        "affinity_offdiag_mean": float(np.mean(off)),
        "affinity_offdiag_median": float(np.median(off)),
        "affinity_time_sec": float(time.perf_counter() - t0),
    }


def leading_basis(A: np.ndarray, K: int) -> np.ndarray:
    n = A.shape[0]
    values, vectors = eigh(
        0.5 * (A + A.T),
        subset_by_index=[n - K, n - 1],
        check_finite=False,
        driver="evr",
    )
    return vectors[:, np.argsort(values)[::-1]]


def row_normalize(U: np.ndarray, eps: float = 1e-12):
    return U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), eps)


def balanced_kmeans(Z: np.ndarray, K: int, seed: int, n_init: int, max_iter: int):
    n = Z.shape[0]
    if n % K != 0:
        raise ValueError(f"balanced rounding requires n divisible by K; n={n}, K={K}")
    capacity = n // K
    rng = np.random.default_rng(seed)
    best_labels = None
    best_objective = np.inf
    for _ in range(max(1, n_init)):
        km_seed = int(rng.integers(0, 2**31 - 1))
        centers = KMeans(n_clusters=K, n_init=1, random_state=km_seed).fit(Z).cluster_centers_
        labels = None
        for _ in range(max(1, max_iter)):
            distances = pairwise_distances(Z, centers, metric="sqeuclidean", n_jobs=1)
            cost = np.repeat(distances, repeats=capacity, axis=1)
            row_ind, slots = linear_sum_assignment(cost)
            new_labels = np.empty(n, dtype=np.int64)
            new_labels[row_ind] = slots // capacity
            if labels is not None and np.array_equal(labels, new_labels):
                labels = new_labels
                break
            labels = new_labels
            for cluster in range(K):
                members = Z[labels == cluster]
                if members.size:
                    centers[cluster] = np.mean(members, axis=0)
        objective = float(np.sum((Z - centers[labels]) ** 2))
        if objective < best_objective:
            best_objective = objective
            best_labels = labels.copy()
    if best_labels is None:
        raise RuntimeError("balanced k-means failed")
    return best_labels


def round_embedding(U: np.ndarray, K: int, args: argparse.Namespace):
    Z = row_normalize(U)
    if args.rounding == "balanced":
        return balanced_kmeans(
            Z,
            K,
            args.seed,
            args.kmeans_n_init,
            args.balanced_max_iter,
        )
    return KMeans(
        n_clusters=K,
        n_init=args.kmeans_n_init,
        random_state=args.seed,
    ).fit_predict(Z)


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    true_values = np.unique(y_true)
    pred_values = np.unique(y_pred)
    contingency = np.zeros((pred_values.size, true_values.size), dtype=np.int64)
    for i, pred in enumerate(pred_values):
        for j, true in enumerate(true_values):
            contingency[i, j] = int(np.sum((y_pred == pred) & (y_true == true)))
    rows, cols = linear_sum_assignment(contingency.max() - contingency)
    return float(contingency[rows, cols].sum() / y_true.size)


def evaluate(method: str, U: np.ndarray, y: np.ndarray, K: int, args, elapsed: float, info):
    labels = round_embedding(U, K, args)
    return {
        "method": method,
        "ACC": clustering_accuracy(y, labels),
        "NMI": float(normalized_mutual_info_score(y, labels)),
        "ARI": float(adjusted_rand_score(y, labels)),
        "time_sec": float(elapsed),
        "final_grad_norm": float(info.get("final_grad_norm", np.nan)),
        "converged": bool(info.get("converged", False)),
        "row_sum_residual": float(info.get("row_sum_residual", np.nan)),
        "idempotence_error": float(info.get("idempotence_error", np.nan)),
        "negative_violation_fro": float(info.get("negative_violation_fro", np.nan)),
        "negative_entry_ratio": float(info.get("negative_entry_ratio", np.nan)),
    }, labels


def ideal_projector(y: np.ndarray):
    classes, inverse = np.unique(y, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    U = np.zeros((y.size, classes.size), dtype=np.float64)
    U[np.arange(y.size), inverse] = 1.0 / np.sqrt(counts[inverse])
    return U @ U.T


def block_stats(M: np.ndarray, y: np.ndarray, prefix: str):
    n = y.size
    same = y[:, None] == y[None, :]
    offdiag = ~np.eye(n, dtype=bool)
    within = M[same & offdiag]
    between = M[~same]
    eps = np.finfo(float).eps
    return {
        f"{prefix}_within_mean": float(np.mean(within)),
        f"{prefix}_between_mean": float(np.mean(between)),
        f"{prefix}_between_to_within_abs_ratio": float(
            np.mean(np.abs(between)) / (np.mean(np.abs(within)) + eps)
        ),
    }


def save_heatmap(M: np.ndarray, y: np.ndarray, path: Path, title: str, signed: bool = False):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError as exc:
        raise ImportError("Install matplotlib with: python -m pip install matplotlib") from exc
    order = np.argsort(y, kind="stable")
    sorted_y = y[order]
    matrix = M[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(9, 8))
    if signed:
        max_abs = float(np.percentile(np.abs(matrix), 99.5))
        max_abs = max(max_abs, np.finfo(float).eps)
        image = ax.imshow(
            matrix,
            interpolation="nearest",
            aspect="equal",
            norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
        )
    else:
        vmin = float(np.percentile(matrix, 0.5))
        vmax = float(np.percentile(matrix, 99.5))
        image = ax.imshow(matrix, interpolation="nearest", aspect="equal", vmin=vmin, vmax=vmax)
    boundaries = np.flatnonzero(sorted_y[1:] != sorted_y[:-1]) + 1
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, linewidth=0.4)
        ax.axvline(boundary - 0.5, linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel("sample index")
    ax.set_ylabel("sample index")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_outer_curves(records: list[dict[str, Any]], out_dir: Path):
    if not records:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    frame = pd.DataFrame(records)
    frame.to_csv(out_dir / "agr_outer_history.csv", index=False)
    for column, filename, title in [
        ("graph_change", "07_graph_change.png", "AGR graph relative change"),
        ("projector_change", "08_projector_change.png", "AGR projector relative change"),
        ("ns_final_grad_norm", "09_outer_gradient.png", "NS-RPMA gradient per outer round"),
    ]:
        if column not in frame:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(frame["outer_iteration"], frame[column], marker="o")
        if np.all(frame[column].fillna(0).to_numpy() > 0):
            ax.set_yscale("log")
        ax.set_xlabel("outer iteration")
        ax.set_ylabel(column)
        ax.set_title(title)
        ax.grid(True, linewidth=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)


def json_ready(value: Any):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def run(args: argparse.Namespace):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features, y, K = load_dataset(args)
    features = np.asarray(features, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if args.pca_dim > 0 and args.pca_dim < features.shape[1]:
        pca = PCA(
            n_components=min(args.pca_dim, features.shape[0] - 1),
            svd_solver="randomized",
            random_state=args.data_seed,
        )
        features = pca.fit_transform(features)

    A0, affinity_info = dense_gaussian_affinity(
        features,
        args.bandwidth_scale,
        args.keep_diagonal,
    )
    U0 = leading_basis(A0, K)
    rows = []
    matrices: dict[str, np.ndarray] = {"A0": A0, "y": y, "K": np.asarray(K)}

    print("=" * 100)
    print(
        f"dataset={args.dataset}, n={len(y)}, K={K}, features={features.shape[1]}, "
        f"bandwidth={args.bandwidth_scale:g}"
    )
    print(
        f"NS parameters: lambda={args.lam:g}, delta={args.delta:g}, mu={args.mu:g}; "
        f"rounding={args.rounding}"
    )
    print("=" * 100)

    if args.methods in {"both", "ns"}:
        print("[run] baseline NS-RPMA")
        t0 = time.perf_counter()
        X_ns, U_ns, ns_info = ns_rpma(
            A0,
            K,
            lam=args.lam,
            delta=args.delta,
            nonnegative_mu=args.mu,
            start_delta=args.start_delta,
            continuation_steps=args.continuation_steps,
            max_iter_per_stage=args.stage_max_iter,
            U0=U0,
            tol=args.tol,
            tau_max=args.tau_max,
            tau_min=args.tau_min,
            backtrack_beta=args.backtrack_beta,
            armijo_sigma=args.armijo_sigma,
            nonmonotone_window=args.nonmonotone_window,
            verbose=args.verbose,
            return_info=True,
        )
        elapsed = time.perf_counter() - t0
        row, labels_ns = evaluate("NS-RPMA", U_ns, y, K, args, elapsed, ns_info)
        row.update(block_stats(X_ns, y, "X"))
        rows.append(row)
        matrices.update({"X_ns": X_ns, "U_ns": U_ns, "labels_ns": labels_ns})
        print(
            f"  ACC={row['ACC']:.4f}, NMI={row['NMI']:.4f}, ARI={row['ARI']:.4f}, "
            f"grad={row['final_grad_norm']:.3e}, time={elapsed:.2f}s"
        )

    if args.methods in {"both", "agr"}:
        print("[run] AGR-NS-RPMA")
        t0 = time.perf_counter()
        X_agr, U_agr, S_final, P_final, agr_info = agr_ns_rpma(
            A0,
            K,
            lam=args.lam,
            delta=args.delta,
            nonnegative_mu=args.mu,
            start_delta=args.start_delta,
            initial_continuation_steps=args.continuation_steps,
            refinement_continuation_steps=args.refinement_continuation_steps,
            initial_max_iter_per_stage=args.stage_max_iter,
            refinement_max_iter_per_stage=args.inner_stage_max_iter,
            n_neighbors=args.knn,
            outer_iterations=args.outer_iterations,
            graph_alpha=args.graph_alpha,
            graph_beta_max=args.graph_beta_max,
            graph_damping=args.graph_damping,
            graph_tol=args.graph_tol,
            projector_tol=args.projector_tol,
            sinkhorn_max_iter=args.sinkhorn_max_iter,
            sinkhorn_tol=args.sinkhorn_tol,
            U0=U0,
            tol=args.tol,
            tau_max=args.tau_max,
            tau_min=args.tau_min,
            backtrack_beta=args.backtrack_beta,
            armijo_sigma=args.armijo_sigma,
            nonmonotone_window=args.nonmonotone_window,
            verbose=args.verbose,
            return_info=True,
        )
        elapsed = time.perf_counter() - t0
        row, labels_agr = evaluate("AGR-NS-RPMA", U_agr, y, K, args, elapsed, agr_info)
        row.update(block_stats(X_agr, y, "X"))
        row.update(block_stats(S_final, y, "S"))
        row.update(
            {
                "outer_iterations_completed": agr_info["outer_iterations_completed"],
                "graph_converged": agr_info["graph_converged"],
                "graph_row_sum_residual": agr_info["graph_row_sum_residual"],
                "graph_symmetry_residual": agr_info["graph_symmetry_residual"],
            }
        )
        rows.append(row)
        matrices.update(
            {
                "X_agr": X_agr,
                "U_agr": U_agr,
                "S_final": S_final,
                "P_final": P_final,
                "labels_agr": labels_agr,
            }
        )
        save_outer_curves(agr_info["outer_records"], out_dir)
        (out_dir / "agr_info.json").write_text(
            json.dumps(json_ready(agr_info), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"  ACC={row['ACC']:.4f}, NMI={row['NMI']:.4f}, ARI={row['ARI']:.4f}, "
            f"grad={row['final_grad_norm']:.3e}, outer={row['outer_iterations_completed']}, "
            f"time={elapsed:.2f}s"
        )

    results = pd.DataFrame(rows)
    results["Composite"] = (results["ACC"] + results["NMI"] + results["ARI"]) / 3.0
    results.to_csv(out_dir / "agr_ns_rpma_results.csv", index=False)

    X_ideal = ideal_projector(y)
    matrices["X_ideal"] = X_ideal
    save_heatmap(A0, y, out_dir / "01_initial_A0_true_order.png", "Initial affinity A0")
    if "S_final" in matrices:
        save_heatmap(
            matrices["S_final"],
            y,
            out_dir / "02_refined_graph_S_true_order.png",
            "AGR learned effective affinity S",
        )
    if "X_ns" in matrices:
        save_heatmap(
            matrices["X_ns"],
            y,
            out_dir / "03_NS_RPMA_X_true_order.png",
            "Baseline NS-RPMA projector X",
            signed=True,
        )
    if "X_agr" in matrices:
        save_heatmap(
            matrices["X_agr"],
            y,
            out_dir / "04_AGR_NS_RPMA_X_true_order.png",
            "AGR-NS-RPMA projector X",
            signed=True,
        )
    save_heatmap(
        X_ideal,
        y,
        out_dir / "05_ideal_projector.png",
        "Ideal class-membership projector",
    )
    if "X_agr" in matrices:
        save_heatmap(
            np.abs(matrices["X_agr"] - X_ideal),
            y,
            out_dir / "06_AGR_absolute_error.png",
            "|X_AGR - X_ideal|",
        )

    np.savez_compressed(out_dir / "agr_ns_rpma_matrices.npz", **matrices)
    config = vars(args).copy()
    config.update(affinity_info)
    config.update({"n": int(len(y)), "K": int(K), "feature_dim": int(features.shape[1])})
    (out_dir / "run_config.json").write_text(
        json.dumps(json_ready(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nResults")
    display_columns = [
        "method", "ACC", "NMI", "ARI", "Composite", "final_grad_norm",
        "negative_violation_fro", "X_between_to_within_abs_ratio", "time_sec",
    ]
    print(results[[c for c in display_columns if c in results.columns]].to_string(index=False))
    print(f"\nSaved to: {out_dir.resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(description="Compare NS-RPMA and AGR-NS-RPMA.")
    parser.add_argument("--dataset", choices=["coil20", "att_faces", "yaleB", "synthetic"], default="coil20")
    parser.add_argument("--data-root", default="datasets/data/coil20")
    parser.add_argument("--image-size", default="original")
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--pca-dim", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--methods", choices=["both", "ns", "agr"], default="both")

    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--keep-diagonal", action="store_true")

    parser.add_argument("--lam", type=float, default=0.005)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--start-delta", type=float, default=1e-2)
    parser.add_argument("--continuation-steps", type=int, default=4)
    parser.add_argument("--stage-max-iter", type=int, default=100)
    parser.add_argument("--refinement-continuation-steps", type=int, default=1)
    parser.add_argument("--inner-stage-max-iter", type=int, default=30)

    parser.add_argument("--knn", type=int, default=20)
    parser.add_argument("--outer-iterations", type=int, default=5)
    parser.add_argument("--graph-alpha", type=float, default=1.0)
    parser.add_argument("--graph-beta-max", type=float, default=0.5)
    parser.add_argument("--graph-damping", type=float, default=0.3)
    parser.add_argument("--graph-tol", type=float, default=1e-4)
    parser.add_argument("--projector-tol", type=float, default=1e-4)
    parser.add_argument("--sinkhorn-max-iter", type=int, default=500)
    parser.add_argument("--sinkhorn-tol", type=float, default=1e-8)

    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--tau-max", type=float, default=1.0)
    parser.add_argument("--tau-min", type=float, default=1e-14)
    parser.add_argument("--backtrack-beta", type=float, default=0.5)
    parser.add_argument("--armijo-sigma", type=float, default=1e-4)
    parser.add_argument("--nonmonotone-window", type=int, default=5)

    parser.add_argument("--rounding", choices=["kmeans", "balanced"], default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmeans-n-init", type=int, default=20)
    parser.add_argument("--balanced-max-iter", type=int, default=30)
    parser.add_argument("--out-dir", default="results/coil20_agr_ns_rpma")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--synthetic-k", type=int, default=4)
    parser.add_argument("--synthetic-per-class", type=int, default=20)
    parser.add_argument("--synthetic-dim", type=int, default=12)
    parser.add_argument("--synthetic-separation", type=float, default=4.0)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
