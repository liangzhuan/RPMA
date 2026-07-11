"""
Image experiment for bounded sparse RPMA.

Place at:
    Community_detection/experiments/image_bounded_sparse_rpma.py

The script uses the project's original full Gaussian affinity by default and
compares:

    1. Spectral projection baseline
    2. Original RPMA-Huber
    3. Bounded-Sparse-RPMA:
           Huber sparsity
         + fixed large box penalty
         + alpha=0
         + upper=K/n=1/n_k

It reports clustering metrics and the distance between each method's spectral
space and the true class-indicator space.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.metrics import pairwise_distances

from datasets import image_datasets as image_ds
from methods.bounded_sparse_rpma import bounded_sparse_rpa

try:
    from methods.rpa import rpa as original_rpa
except Exception:
    original_rpa = None

try:
    from methods.affinity import build_affinity
except Exception:
    build_affinity = None


def parse_size(value: str | None):
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"original", "orig", "none"}:
        return None
    if "x" in value:
        h, w = value.split("x", 1)
        return int(h), int(w)
    v = int(value)
    return v, v


def parse_float_list(text: str) -> List[float]:
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one floating-point value.")
    return values


def parse_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def load_dataset(args):
    size = parse_size(args.image_size)
    kwargs = {
        "image_size": size,
        "max_per_class": args.max_per_class,
        "random_state": args.data_seed,
    }

    if args.dataset == "att_faces":
        loader = getattr(image_ds, "load_att_faces", None)
        if loader is None:
            raise ImportError(
                "datasets/image_datasets.py does not define load_att_faces."
            )
        return loader(args.data_root, **kwargs)

    if args.dataset == "coil20":
        return image_ds.load_coil20(args.data_root, **kwargs)

    if args.dataset == "yaleB":
        return image_ds.load_extended_yale_b(args.data_root, **kwargs)

    raise ValueError(f"Unknown dataset={args.dataset!r}")


def original_full_gaussian(X: np.ndarray, keep_diagonal: bool = True):
    D2 = pairwise_distances(X, metric="sqeuclidean", n_jobs=1)
    D2 = np.maximum(D2, 0.0)
    n = D2.shape[0]
    upper = D2[np.triu_indices(n, k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        raise ValueError("All pairwise distances are zero.")
    sigma2 = float(np.mean(positive))
    A = np.exp(-D2 / sigma2)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 1.0 if keep_diagonal else 0.0)
    return A, {"sigma2": sigma2, "affinity_density": float(np.mean(A > 1e-12))}


def build_original_affinity(X: np.ndarray, keep_diagonal: bool):
    if build_affinity is not None:
        A, info = build_affinity(
            X,
            graph="full_gaussian",
            bandwidth="mean",
            sigma2_scale=1.0,
            zero_diagonal=not keep_diagonal,
        )
        return A, info
    return original_full_gaussian(X, keep_diagonal=keep_diagonal)


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n_classes = max(int(y_true.max()), int(y_pred.max())) + 1
    contingency = np.zeros((n_classes, n_classes), dtype=np.int64)
    for yp, yt in zip(y_pred, y_true):
        contingency[yp, yt] += 1
    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    return float(contingency[row_ind, col_ind].sum() / y_true.size)


def row_normalize(U: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), eps)


def balanced_kmeans(
    Z: np.ndarray,
    K: int,
    seed: int,
    n_init: int = 10,
    max_iter: int = 30,
) -> np.ndarray:
    n = Z.shape[0]
    if n % K != 0:
        raise ValueError(f"Balanced k-means requires n divisible by K; n={n}, K={K}.")
    cap = n // K
    rng = np.random.default_rng(seed)

    best_labels = None
    best_obj = np.inf

    for _ in range(n_init):
        km_seed = int(rng.integers(0, 2**31 - 1))
        centers = KMeans(
            n_clusters=K,
            n_init=1,
            random_state=km_seed,
        ).fit(Z).cluster_centers_

        labels = None
        for _ in range(max_iter):
            D2 = pairwise_distances(Z, centers, metric="sqeuclidean")
            cost = np.repeat(D2, repeats=cap, axis=1)
            _, cols = linear_sum_assignment(cost)
            new_labels = cols // cap

            if labels is not None and np.array_equal(labels, new_labels):
                labels = new_labels
                break

            labels = new_labels
            for k in range(K):
                centers[k] = Z[labels == k].mean(axis=0)

        obj = float(np.sum((Z - centers[labels]) ** 2))
        if obj < best_obj:
            best_obj = obj
            best_labels = labels.copy()

    return np.asarray(best_labels, dtype=np.int64)


def labels_from_space(
    U: np.ndarray,
    K: int,
    seed: int,
    rounding: str,
    kmeans_n_init: int,
) -> np.ndarray:
    Z = row_normalize(U)
    if rounding == "balanced":
        return balanced_kmeans(
            Z,
            K,
            seed,
            n_init=max(1, min(kmeans_n_init, 20)),
        )
    return KMeans(
        n_clusters=K,
        n_init=kmeans_n_init,
        random_state=seed,
    ).fit_predict(Z)


def true_class_basis(y: np.ndarray):
    classes, inverse = np.unique(y, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    F = np.zeros((y.size, classes.size), dtype=np.float64)
    F[np.arange(y.size), inverse] = 1.0 / np.sqrt(counts[inverse])
    return F


def subspace_metrics(U: np.ndarray, F: np.ndarray) -> Dict[str, float]:
    Qu, _ = np.linalg.qr(U, mode="reduced")
    Qf, _ = np.linalg.qr(F, mode="reduced")
    s = np.linalg.svd(Qf.T @ Qu, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)

    k = Qf.shape[1]
    r = Qu.shape[1]
    overlap = float(np.sum(s**2))
    projection_fro = float(np.sqrt(max(0.0, k + r - 2.0 * overlap)))
    angles = np.arccos(s)

    return {
        "projection_fro_normalized": float(
            projection_fro / np.sqrt(k + r)
        ),
        "true_space_residual": float(
            np.sqrt(max(0.0, k - overlap) / k)
        ),
        "method_space_residual": float(
            np.sqrt(max(0.0, r - overlap) / r)
        ),
        "overlap_ratio": float(overlap / k),
        "principal_angle_mean_deg": float(np.degrees(np.mean(angles))),
        "principal_angle_max_deg": float(np.degrees(np.max(angles))),
    }


def evaluate(
    method: str,
    U: np.ndarray,
    y: np.ndarray,
    K: int,
    seed: int,
    rounding: str,
    kmeans_n_init: int,
    elapsed: float,
    extra: Dict | None = None,
) -> Dict:
    labels = labels_from_space(
        U,
        K,
        seed,
        rounding,
        kmeans_n_init,
    )
    F = true_class_basis(y)

    row = {
        "method": method,
        "seed": int(seed),
        "ACC": clustering_accuracy(y, labels),
        "NMI": float(normalized_mutual_info_score(y, labels)),
        "ARI": float(adjusted_rand_score(y, labels)),
        "time_sec": float(elapsed),
        "rounding": rounding,
    }
    row.update(subspace_metrics(U, F))
    if extra:
        row.update(extra)
    return row


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, K = load_dataset(args)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n = X.shape[0]

    print("=" * 88)
    print("Bounded sparse RPMA image experiment")
    print(f"dataset={args.dataset}, n={n}, K={K}, n_k={n / K:.6g}")
    print("affinity=original full Gaussian, bandwidth=mean")
    print(f"box: alpha=0, upper=K/n={K/n:.12g}, mu={args.box_mu:g}")
    print("=" * 88)

    t_aff = time.time()
    A, affinity_info = build_original_affinity(
        X,
        keep_diagonal=args.keep_diagonal,
    )
    affinity_time = time.time() - t_aff
    print(
        f"A built: shape={A.shape}, "
        f"sigma2={float(affinity_info.get('sigma2', np.nan)):.8e}, "
        f"time={affinity_time:.2f}s"
    )

    lambdas = parse_float_list(args.lam_list)
    deltas = parse_float_list(args.delta_list)
    seeds = parse_int_list(args.seeds)

    rows: List[Dict] = []

    if "spectral" in args.methods:
        t0 = time.time()
        vals, vecs = eigh(
            0.5 * (A + A.T),
            subset_by_index=[n - K, n - 1],
            check_finite=False,
            driver="evr",
        )
        U_spec = vecs[:, np.argsort(vals)[::-1]]
        elapsed = time.time() - t0
        for seed in seeds:
            rows.append(
                evaluate(
                    "Spectral-Projection",
                    U_spec,
                    y,
                    K,
                    seed,
                    args.rounding,
                    args.kmeans_n_init,
                    elapsed,
                    extra={
                        "lam": np.nan,
                        "delta": np.nan,
                        "box_mu": 0.0,
                        "box_alpha": np.nan,
                        "box_upper": np.nan,
                    },
                )
            )

    for lam in lambdas:
        for delta in deltas:
            if "rpma" in args.methods:
                if original_rpa is None:
                    print("[skip] methods.rpa.rpa is unavailable")
                else:
                    t0 = time.time()
                    try:
                        X_rpa, U_rpa, hist = original_rpa(
                            A,
                            K,
                            lam=lam,
                            delta=delta,
                            tol=args.tol,
                            max_iter=args.max_iter,
                            eig_init=True,
                            return_history=True,
                            verbose=args.verbose,
                        )
                        elapsed = time.time() - t0
                        for seed in seeds:
                            rows.append(
                                evaluate(
                                    "RPMA-Huber",
                                    U_rpa,
                                    y,
                                    K,
                                    seed,
                                    args.rounding,
                                    args.kmeans_n_init,
                                    elapsed,
                                    extra={
                                        "lam": lam,
                                        "delta": delta,
                                        "box_mu": 0.0,
                                        "box_alpha": np.nan,
                                        "box_upper": np.nan,
                                        "n_iter": len(hist),
                                        "final_grad_norm": float(hist[-1]) if hist else np.nan,
                                        "x_min": float(np.min(X_rpa)),
                                        "x_max": float(np.max(X_rpa)),
                                    },
                                )
                            )
                    except Exception as exc:
                        rows.append(
                            {
                                "method": "RPMA-Huber",
                                "lam": lam,
                                "delta": delta,
                                "error": repr(exc),
                            }
                        )

            if "bounded_rpma" in args.methods:
                t0 = time.time()
                try:
                    X_box, U_box, info = bounded_sparse_rpa(
                        A,
                        K,
                        lam=lam,
                        delta=delta,
                        box_mu=args.box_mu,
                        alpha=0.0,
                        upper=K / n,
                        tol=args.tol,
                        max_iter=args.max_iter,
                        eig_init=True,
                        return_info=True,
                        verbose=args.verbose,
                        off_diagonal_only=args.box_off_diagonal_only,
                    )
                    elapsed = time.time() - t0

                    extra = {
                        "lam": lam,
                        "delta": delta,
                        "box_mu": args.box_mu,
                        "box_alpha": 0.0,
                        "box_upper": K / n,
                        "box_off_diagonal_only": bool(args.box_off_diagonal_only),
                        "n_iter": info["n_iter"],
                        "converged": info["converged"],
                        "line_search_failed": info["line_search_failed"],
                        "final_grad_norm": info["final_grad_norm"],
                        "final_objective": info["final_objective"],
                        "final_huber_value": info["final_huber_value"],
                        "final_box_loss": info["final_box_loss"],
                        "x_min": info["x_min"],
                        "x_max": info["x_max"],
                        "lower_violation_max": info["lower_violation_max"],
                        "upper_violation_max": info["upper_violation_max"],
                        "lower_violation_ratio": info["lower_violation_ratio"],
                        "upper_violation_ratio": info["upper_violation_ratio"],
                        "box_violation_fro": info["box_violation_fro"],
                        "orthogonality_error": info["orthogonality_error"],
                        "idempotence_error": info["idempotence_error"],
                    }

                    for seed in seeds:
                        rows.append(
                            evaluate(
                                "Bounded-Sparse-RPMA",
                                U_box,
                                y,
                                K,
                                seed,
                                args.rounding,
                                args.kmeans_n_init,
                                elapsed,
                                extra=extra,
                            )
                        )

                    if args.save_matrices:
                        tag = (
                            f"lam{lam:g}_delta{delta:g}"
                            .replace(".", "p")
                            .replace("-", "m")
                        )
                        np.savez_compressed(
                            out_dir / f"bounded_rpma_{tag}.npz",
                            X=X_box,
                            U=U_box,
                            y=y,
                            A=A,
                        )

                except Exception as exc:
                    rows.append(
                        {
                            "method": "Bounded-Sparse-RPMA",
                            "lam": lam,
                            "delta": delta,
                            "box_mu": args.box_mu,
                            "box_alpha": 0.0,
                            "box_upper": K / n,
                            "error": repr(exc),
                        }
                    )

    df = pd.DataFrame(rows)
    df["dataset"] = args.dataset
    df["n"] = n
    df["K"] = K
    df["n_k"] = n / K
    df["affinity"] = "full_gaussian"
    df["affinity_bandwidth"] = "mean"
    df["keep_diagonal"] = bool(args.keep_diagonal)
    df["affinity_sigma2"] = float(affinity_info.get("sigma2", np.nan))
    df["affinity_time_sec"] = affinity_time
    if "error" not in df.columns:
        df["error"] = ""
    df["error"] = df["error"].fillna("")

    csv_path = out_dir / "bounded_sparse_rpma_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    valid = df[df["error"] == ""].copy()
    if not valid.empty:
        summary = (
            valid.groupby(
                [
                    "method",
                    "lam",
                    "delta",
                    "box_mu",
                    "box_alpha",
                    "box_upper",
                    "rounding",
                ],
                dropna=False,
            )
            .agg(
                ACC_mean=("ACC", "mean"),
                ACC_std=("ACC", "std"),
                NMI_mean=("NMI", "mean"),
                NMI_std=("NMI", "std"),
                ARI_mean=("ARI", "mean"),
                ARI_std=("ARI", "std"),
                projection_distance_mean=("projection_fro_normalized", "mean"),
                overlap_ratio_mean=("overlap_ratio", "mean"),
                time_sec_mean=("time_sec", "mean"),
                n_runs=("seed", "count"),
            )
            .reset_index()
            .sort_values(
                ["ACC_mean", "NMI_mean", "ARI_mean"],
                ascending=False,
            )
        )
    else:
        summary = pd.DataFrame()

    summary_path = out_dir / "bounded_sparse_rpma_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if args.save_xlsx:
        with pd.ExcelWriter(
            out_dir / "bounded_sparse_rpma_results.xlsx",
            engine="openpyxl",
        ) as writer:
            df.to_excel(writer, sheet_name="runs", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)

    config = vars(args).copy()

    # argparse stores --methods as a Python set.  Sets are not JSON serializable,
    # so convert it to a stable sorted list before writing run_config.json.
    if isinstance(config.get("methods"), set):
        config["methods"] = sorted(config["methods"])

    config.update(
        {
            "n": int(n),
            "K": int(K),
            "n_k": float(n / K),
            "box_alpha_effective": 0.0,
            "box_upper_effective": float(K / n),
            "sigma2": float(affinity_info.get("sigma2", np.nan)),
        }
    )
    (out_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nBest valid runs:")
    if summary.empty:
        print("No valid result.")
    else:
        cols = [
            "method",
            "lam",
            "delta",
            "box_mu",
            "box_upper",
            "ACC_mean",
            "NMI_mean",
            "ARI_mean",
            "projection_distance_mean",
            "overlap_ratio_mean",
            "time_sec_mean",
        ]
        print(summary[[c for c in cols if c in summary.columns]].head(20).to_string(index=False))

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {summary_path}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Compare original RPMA with bounded sparse RPMA on image datasets."
    )
    p.add_argument("--dataset", choices=["att_faces", "coil20", "yaleB"], default="att_faces")
    p.add_argument("--data-root", default=r"datasets\data\att_faces")
    p.add_argument("--image-size", default="original")
    p.add_argument("--max-per-class", type=int, default=10)
    p.add_argument("--data-seed", type=int, default=42)

    p.add_argument(
        "--methods",
        type=lambda s: {x.strip() for x in s.split(",") if x.strip()},
        default={"spectral", "rpma", "bounded_rpma"},
        help="Comma-separated: spectral,rpma,bounded_rpma",
    )

    p.add_argument("--lam-list", default="0.05,0.07,0.1")
    p.add_argument("--delta-list", default="0.0001,0.001,0.01")
    p.add_argument("--box-mu", type=float, default=100000.0)
    p.add_argument("--box-off-diagonal-only", action="store_true")

    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--seeds", default="42")
    p.add_argument("--rounding", choices=["kmeans", "balanced"], default="kmeans")
    p.add_argument("--kmeans-n-init", type=int, default=50)

    p.set_defaults(keep_diagonal=True)
    p.add_argument("--keep-diagonal", action="store_true", dest="keep_diagonal")
    p.add_argument("--zero-diagonal", action="store_false", dest="keep_diagonal")

    p.add_argument("--verbose", action="store_true")
    p.add_argument("--save-xlsx", action="store_true")
    p.add_argument("--save-matrices", action="store_true")
    p.add_argument(
        "--out-dir",
        default=r"results\att_faces_bounded_sparse_rpma",
    )
    return p


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
