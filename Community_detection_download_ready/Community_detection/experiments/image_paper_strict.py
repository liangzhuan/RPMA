"""
Paper-strict image experiment for RPMA.

Copy this file to:
    Community_detection/experiments/image_paper_strict.py

Run from the project root, for example:
    python -m experiments.image_paper_strict --dataset coil20 --data-root datasets/data/coil20 --image-size original --max-per-class 10 --lams 0.1 --deltas 1e-3

This script deliberately removes the extra experimental knobs that are not in
the paper-style image experiment:
    - no PCA preprocessing parameter pca_dim;
    - no k-nearest-neighbor graph sparsification k_neighbors.

It follows the paper-style pipeline:
    images -> vectorized features -> full Gaussian affinity matrix A
    Spectral-Projection: top-K eigenvectors of A
    RPMA-Huber: min_{X in P_K} -2<A,X> + lambda * sum_ij Huber_delta(X_ij)
    k-means on rows of U -> ACC/NMI/ARI
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from datasets.image_datasets import load_coil20, load_extended_yale_b
from methods.rpa import rpa
from evaluation.metrics import evaluate


def parse_size(s):
    """Parse image size. Use 'original' or 'none' to keep original image size."""
    if s is None:
        return None
    s = str(s).lower().strip()
    if s in {"none", "original", "orig"}:
        return None
    if "x" in s:
        a, b = s.split("x")
        return int(a), int(b)
    v = int(s)
    return v, v


def parse_float_list(s):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_int_list(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def symmetrize(A):
    A = np.asarray(A, dtype=float)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    return 0.5 * (A + A.T)


def load_dataset(dataset, data_root, image_size, max_per_class, random_state):
    if dataset == "coil20":
        return load_coil20(
            data_root,
            image_size=image_size,
            max_per_class=max_per_class,
            random_state=random_state,
        )
    if dataset == "yaleB":
        return load_extended_yale_b(
            data_root,
            image_size=image_size,
            max_per_class=max_per_class,
            random_state=random_state,
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def standardize_features(X, eps=1e-12):
    """
    Basic centering and scaling. This is not PCA and does not change the
    feature dimension. It prevents pixel scale from dominating distances.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    X = X / (std + eps)
    return X


def paper_gaussian_affinity(X, zero_diagonal=False):
    """
    Full Gaussian affinity used in the paper-style experiment:
        A_ij = exp(-||x_i - x_j||^2 / sigma^2)
        sigma^2 = 2 / (n (n - 1)) * sum_{i<j} ||x_i - x_j||^2

    No kNN sparsification is applied.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least two samples to build an affinity matrix")

    D2 = pairwise_distances(X, metric="sqeuclidean", n_jobs=1)
    upper = D2[np.triu_indices(n, k=1)]
    sigma2 = 2.0 * float(np.sum(upper)) / (n * (n - 1))
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError(f"Invalid Gaussian bandwidth sigma^2={sigma2}")

    A = np.exp(-D2 / sigma2)
    A = symmetrize(A)
    if zero_diagonal:
        np.fill_diagonal(A, 0.0)
    return A, sigma2


def kmeans_on_rows(U, K, random_state=0):
    U = np.asarray(U, dtype=float)
    return KMeans(n_clusters=K, n_init=20, random_state=random_state).fit_predict(U)


def spectral_projection(A, K, random_state=0):
    """Paper baseline: X_spe = U_K U_K^T, top-K eigenvectors of A."""
    A = symmetrize(A)
    eigvals, eigvecs = eigh(A)
    idx = np.argsort(eigvals)[::-1][:K]
    U = eigvecs[:, idx]
    labels = kmeans_on_rows(U, K, random_state=random_state)
    X = U @ U.T
    return labels, X, U


def append_rows_csv(rows, csv_path):
    df = pd.DataFrame(rows)
    csv_path = Path(csv_path)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False, encoding="utf-8-sig")


def save_excel(raw_csv, summary_csv, xlsx_path):
    raw = pd.read_csv(raw_csv)
    summary = pd.read_csv(summary_csv)
    with pd.ExcelWriter(xlsx_path) as writer:
        raw.to_excel(writer, sheet_name="raw_results", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    print(f"Excel saved to: {xlsx_path}")


def run(args):
    image_size = parse_size(args.image_size)
    max_per_class = None if args.max_per_class <= 0 else args.max_per_class
    lams = parse_float_list(args.lams)
    deltas = parse_float_list(args.deltas)
    seeds = parse_int_list(args.seeds)

    ensure_dir(args.out_dir)
    out_dir = Path(args.out_dir)
    raw_csv = out_dir / f"{args.dataset}_paper_strict_results.csv"
    summary_csv = out_dir / f"{args.dataset}_paper_strict_summary.csv"
    xlsx_path = out_dir / f"{args.dataset}_paper_strict_summary.xlsx"
    config_path = out_dir / f"{args.dataset}_paper_strict_config.json"

    if args.overwrite:
        for p in [raw_csv, summary_csv, xlsx_path, config_path]:
            if p.exists():
                p.unlink()

    config = vars(args).copy()
    config["pipeline"] = (
        "image vectors -> full Gaussian affinity A using paper bandwidth -> "
        "Spectral-Projection and RPMA-Huber on the same A"
    )
    config["removed_from_previous_code"] = ["pca_dim", "k_neighbors"]
    config["gaussian_bandwidth"] = "sigma^2 = 2/(n(n-1)) * sum_{i<j} ||x_i-x_j||^2"
    config["spectral_baseline"] = "top-K eigenvectors of full Gaussian affinity A"
    config["rpma_objective"] = "F(X) = -2<A,X> + lambda * sum_ij Huber_delta(X_ij), X in P_K"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Paper-strict RPMA image experiment")
    print("No PCA. No kNN graph sparsification.")
    print("Affinity: full Gaussian kernel with paper bandwidth.")
    print(f"dataset       = {args.dataset}")
    print(f"data_root     = {args.data_root}")
    print(f"image_size    = {args.image_size}")
    print(f"max_per_class = {args.max_per_class}  (0 means full dataset)")
    print(f"standardize   = {args.standardize}")
    print(f"zero_diagonal = {args.zero_diagonal}")
    print(f"lams          = {lams}")
    print(f"deltas        = {deltas}")
    print(f"seeds         = {seeds}")
    print(f"out_dir       = {out_dir}")
    print("=" * 80)

    rows = []
    for seed in seeds:
        print(f"\n[Load dataset] seed={seed}")
        X, y, K = load_dataset(
            args.dataset,
            args.data_root,
            image_size=image_size,
            max_per_class=max_per_class,
            random_state=seed,
        )
        n, d = X.shape
        print(f"Loaded {args.dataset}: X={X.shape}, classes={K}, n={n}")

        if args.standardize:
            print("[Feature] standardize raw vector features; no PCA is applied")
            X_used = standardize_features(X)
        else:
            print("[Feature] use raw vector features; no PCA is applied")
            X_used = np.asarray(X, dtype=np.float64)

        print("[Affinity] build full Gaussian affinity; no k_neighbors sparsification")
        t_aff = time.time()
        A, sigma2 = paper_gaussian_affinity(X_used, zero_diagonal=args.zero_diagonal)
        aff_time = time.time() - t_aff
        print(f"Gaussian sigma^2={sigma2:.6e}, affinity_time={aff_time:.2f}s")

        t0 = time.time()
        labels_sp, X_sp, U_sp = spectral_projection(A, K, random_state=seed)
        metric_sp = evaluate(y, labels_sp)
        time_sp = time.time() - t0
        row_sp = {
            "dataset": args.dataset,
            "method": "Spectral-Projection",
            "seed": seed,
            "n": n,
            "K": K,
            "image_size": args.image_size,
            "feature_dim": d,
            "standardize": args.standardize,
            "zero_diagonal": args.zero_diagonal,
            "sigma2": sigma2,
            "affinity_time_sec": aff_time,
            "lam": np.nan,
            "delta": np.nan,
            "rpa_max_iter": np.nan,
            "ACC": metric_sp["ACC"],
            "NMI": metric_sp["NMI"],
            "ARI": metric_sp["ARI"],
            "time_sec": time_sp,
            "final_grad": np.nan,
            "n_iter": np.nan,
        }
        append_rows_csv([row_sp], raw_csv)
        rows.append(row_sp)
        print(
            f"  Spectral-Projection | "
            f"ACC={metric_sp['ACC']:.4f}, NMI={metric_sp['NMI']:.4f}, "
            f"ARI={metric_sp['ARI']:.4f}, time={time_sp:.2f}s"
        )

        for lam in lams:
            for delta in deltas:
                t0 = time.time()
                try:
                    X_rpma, U_rpma, history = rpa(
                        A,
                        K,
                        lam=lam,
                        delta=delta,
                        max_iter=args.rpa_max_iter,
                        eig_init=True,
                        return_history=True,
                        verbose=False,
                    )
                    labels_rpma = kmeans_on_rows(U_rpma, K, random_state=seed)
                    metric_rpma = evaluate(y, labels_rpma)
                    time_rpma = time.time() - t0
                    final_grad = float(history[-1]) if history else np.nan
                    n_iter = len(history)
                    row_rpma = {
                        "dataset": args.dataset,
                        "method": "RPMA-Huber",
                        "seed": seed,
                        "n": n,
                        "K": K,
                        "image_size": args.image_size,
                        "feature_dim": d,
                        "standardize": args.standardize,
                        "zero_diagonal": args.zero_diagonal,
                        "sigma2": sigma2,
                        "affinity_time_sec": aff_time,
                        "lam": lam,
                        "delta": delta,
                        "rpa_max_iter": args.rpa_max_iter,
                        "ACC": metric_rpma["ACC"],
                        "NMI": metric_rpma["NMI"],
                        "ARI": metric_rpma["ARI"],
                        "time_sec": time_rpma,
                        "final_grad": final_grad,
                        "n_iter": n_iter,
                    }
                    append_rows_csv([row_rpma], raw_csv)
                    rows.append(row_rpma)
                    print(
                        f"  RPMA-Huber | lam={lam:g}, delta={delta:g} | "
                        f"ACC={metric_rpma['ACC']:.4f}, NMI={metric_rpma['NMI']:.4f}, "
                        f"ARI={metric_rpma['ARI']:.4f}, grad={final_grad:.3e}, "
                        f"iter={n_iter}, time={time_rpma:.2f}s"
                    )
                except Exception as exc:
                    time_rpma = time.time() - t0
                    row_fail = {
                        "dataset": args.dataset,
                        "method": "RPMA-Huber",
                        "seed": seed,
                        "n": n,
                        "K": K,
                        "image_size": args.image_size,
                        "feature_dim": d,
                        "standardize": args.standardize,
                        "zero_diagonal": args.zero_diagonal,
                        "sigma2": sigma2,
                        "affinity_time_sec": aff_time,
                        "lam": lam,
                        "delta": delta,
                        "rpa_max_iter": args.rpa_max_iter,
                        "ACC": np.nan,
                        "NMI": np.nan,
                        "ARI": np.nan,
                        "time_sec": time_rpma,
                        "final_grad": np.nan,
                        "n_iter": np.nan,
                        "error": repr(exc),
                    }
                    append_rows_csv([row_fail], raw_csv)
                    rows.append(row_fail)
                    print(f"  RPMA-Huber FAILED | lam={lam:g}, delta={delta:g} | error={repr(exc)}")

    df = pd.DataFrame(rows)
    group_cols = [
        "dataset",
        "method",
        "image_size",
        "feature_dim",
        "standardize",
        "zero_diagonal",
        "lam",
        "delta",
        "rpa_max_iter",
    ]
    summary = (
        df.groupby(group_cols, dropna=False)[["ACC", "NMI", "ARI", "time_sec", "final_grad", "n_iter"]]
        .mean()
        .reset_index()
    )
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    if args.save_xlsx:
        save_excel(raw_csv, summary_csv, xlsx_path)

    print("\n" + "=" * 80)
    print("Finished")
    print(f"Raw CSV:     {raw_csv}")
    print(f"Summary CSV: {summary_csv}")
    if args.save_xlsx:
        print(f"Excel:       {xlsx_path}")
    print("=" * 80)

    for metric in ["ACC", "NMI", "ARI"]:
        valid = summary.dropna(subset=[metric])
        if len(valid) == 0:
            continue
        best = valid.sort_values(metric, ascending=False).iloc[0]
        print(f"\nBest by {metric}:")
        print(best[["method", "lam", "delta", "ACC", "NMI", "ARI", "time_sec", "final_grad", "n_iter"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["coil20", "yaleB"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image-size", default="original", help="Use 'original' for original image size, or e.g. 32x32")
    parser.add_argument("--max-per-class", type=int, default=10, help="0 means full dataset")
    parser.add_argument("--lams", default="0.001,0.005,0.01,0.05,0.1")
    parser.add_argument("--deltas", default="1e-3")
    parser.add_argument("--rpa-max-iter", type=int, default=200)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--out-dir", default="results/image_paper_strict")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-xlsx", action="store_true")
    parser.add_argument("--standardize", action="store_true", help="Center and scale pixel features; no dimensionality reduction")
    parser.add_argument("--zero-diagonal", action="store_true", help="Set diagonal of A to zero; default keeps A_ii=1")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
