"""
Paper-consistent grid search for image clustering experiments.

This file replaces or creates: experiments/grid_image_params.py

This script fixes the previous inconsistency: Spectral is no longer the graph
Laplian spectral clustering baseline.  It is the paper's unregularized spectral
projection baseline on the same Gaussian affinity matrix A used by RPMA:

    Spectral: X_spe = U_K U_K^T, top-K eigenvectors of A.
    RPMA:     min_{X in P_K} -2<A,X> + lambda * sum_ij g_delta(X_ij).

For every pca_dim and k_neighbors setting, both methods use the same A.
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
from sklearn.preprocessing import normalize

from datasets.image_datasets import load_coil20, load_extended_yale_b
from methods.affinity import preprocess_features, gaussian_affinity
from methods.rpa import rpa
from evaluation.metrics import evaluate


def parse_size(s):
    if s is None:
        return None
    s = str(s).lower()
    if s in {"none", "original"}:
        return None
    if "x" in s:
        a, b = s.split("x")
        return int(a), int(b)
    v = int(s)
    return v, v


def parse_int_list(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_float_list(s):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _symmetrize(A):
    A = np.asarray(A, dtype=float)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    return 0.5 * (A + A.T)


def kmeans_on_rows(U, K, random_state=0, row_normalize=False):
    U = np.asarray(U, dtype=float)
    if row_normalize:
        U = normalize(U, norm="l2")
    return KMeans(n_clusters=K, n_init=20, random_state=random_state).fit_predict(U)


def paper_spectral_projection(A, K, random_state=0, row_normalize=False):
    """Unregularized spectral projection baseline used by the RPMA paper."""
    A = _symmetrize(A)
    eigvals, eigvecs = eigh(A)
    idx = np.argsort(eigvals)[::-1][:K]
    U = eigvecs[:, idx]
    labels = kmeans_on_rows(U, K, random_state=random_state, row_normalize=row_normalize)
    X = U @ U.T
    return labels, X, U


def append_rows_csv(rows, csv_path):
    df = pd.DataFrame(rows)
    csv_path = Path(csv_path)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False, encoding="utf-8-sig")


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


def write_xlsx_if_possible(raw_csv, summary_csv, xlsx_path):
    try:
        raw = pd.read_csv(raw_csv)
        summary = pd.read_csv(summary_csv)
        with pd.ExcelWriter(xlsx_path) as writer:
            raw.to_excel(writer, sheet_name="raw_results", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
        print(f"Excel saved to: {xlsx_path}")
    except Exception as exc:
        print(f"Excel export skipped: {repr(exc)}")


def run_grid(args):
    image_size = parse_size(args.image_size)
    max_per_class = None if args.max_per_class <= 0 else args.max_per_class

    pca_dims = parse_int_list(args.pca_dims)
    k_neighbors_list = parse_int_list(args.k_neighbors)
    lams = parse_float_list(args.lams)
    deltas = parse_float_list(args.deltas)
    seeds = parse_int_list(args.seeds)

    ensure_dir(args.out_dir)

    out_csv = Path(args.out_dir) / f"{args.dataset}_grid_results.csv"
    summary_path = Path(args.out_dir) / f"{args.dataset}_grid_summary.csv"
    xlsx_path = Path(args.out_dir) / f"{args.dataset}_grid_summary.xlsx"
    config_json = Path(args.out_dir) / f"{args.dataset}_grid_config.json"

    # Avoid appending to stale files from old experiments.
    if args.overwrite:
        for p in [out_csv, summary_path, xlsx_path, config_json]:
            if p.exists():
                p.unlink()

    with open(config_json, "w", encoding="utf-8") as f:
        config = vars(args).copy()
        config["spectral_definition"] = "paper raw-affinity spectral projection: top-K eigenvectors of A"
        config["rpma_definition"] = "min_{X in P_K} -2<A,X> + lambda * Huber_delta(X)"
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Paper-consistent RPMA grid search started")
    print("Spectral baseline: top-K eigenvectors of the same raw Gaussian affinity A")
    print("RPMA-Huber: same A, Huber regularization on projection matrix entries")
    print(f"dataset       = {args.dataset}")
    print(f"data_root     = {args.data_root}")
    print(f"image_size    = {args.image_size}")
    print(f"max_per_class = {args.max_per_class}  (0 means full dataset)")
    print(f"pca_dims      = {pca_dims}")
    print(f"k_neighbors   = {k_neighbors_list}")
    print(f"lams          = {lams}")
    print(f"deltas        = {deltas}")
    print(f"seeds         = {seeds}")
    print(f"row_norm_emb  = {args.row_normalize_embedding}")
    print(f"out_csv       = {out_csv}")
    print("=" * 80)

    all_rows = []

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

        for pca_dim in pca_dims:
            print(f"\n[Preprocess] pca_dim={pca_dim}")
            Xp = preprocess_features(X, pca_dim=pca_dim, random_state=seed)

            for knn in k_neighbors_list:
                print(f"\n[Affinity] pca_dim={pca_dim}, k_neighbors={knn}")
                A = gaussian_affinity(
                    Xp,
                    sigma="median",
                    k_neighbors=knn,
                    self_loop=False,
                )
                A = _symmetrize(A)

                # 1. Paper spectral projection baseline.
                t0 = time.time()
                labels_sp, X_sp, U_sp = paper_spectral_projection(
                    A,
                    K,
                    random_state=seed,
                    row_normalize=args.row_normalize_embedding,
                )
                metric_sp = evaluate(y, labels_sp)
                time_sp = time.time() - t0

                row_sp = {
                    "dataset": args.dataset,
                    "method": "Spectral-Projection",
                    "seed": seed,
                    "n": n,
                    "K": K,
                    "image_size": args.image_size,
                    "max_per_class": args.max_per_class,
                    "pca_dim": pca_dim,
                    "k_neighbors": knn,
                    "lam": np.nan,
                    "delta": np.nan,
                    "rpa_max_iter": np.nan,
                    "row_normalize_embedding": args.row_normalize_embedding,
                    "ACC": metric_sp["ACC"],
                    "NMI": metric_sp["NMI"],
                    "ARI": metric_sp["ARI"],
                    "time_sec": time_sp,
                    "final_grad": np.nan,
                    "n_iter": np.nan,
                    "objective": "max_<A,X>_over_P_K",
                }
                append_rows_csv([row_sp], out_csv)
                all_rows.append(row_sp)

                print(
                    f"  Spectral-Projection | "
                    f"ACC={metric_sp['ACC']:.4f}, "
                    f"NMI={metric_sp['NMI']:.4f}, "
                    f"ARI={metric_sp['ARI']:.4f}, "
                    f"time={time_sp:.2f}s"
                )

                # 2. RPMA-Huber grid on the same A.
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

                            labels_rpma = kmeans_on_rows(
                                U_rpma,
                                K,
                                random_state=seed,
                                row_normalize=args.row_normalize_embedding,
                            )
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
                                "max_per_class": args.max_per_class,
                                "pca_dim": pca_dim,
                                "k_neighbors": knn,
                                "lam": lam,
                                "delta": delta,
                                "rpa_max_iter": args.rpa_max_iter,
                                "row_normalize_embedding": args.row_normalize_embedding,
                                "ACC": metric_rpma["ACC"],
                                "NMI": metric_rpma["NMI"],
                                "ARI": metric_rpma["ARI"],
                                "time_sec": time_rpma,
                                "final_grad": final_grad,
                                "n_iter": n_iter,
                                "objective": "min_-2<A,X>_plus_lambda_Huber_over_P_K",
                            }

                            append_rows_csv([row_rpma], out_csv)
                            all_rows.append(row_rpma)

                            print(
                                f"  RPMA-Huber | lam={lam:g}, delta={delta:g} | "
                                f"ACC={metric_rpma['ACC']:.4f}, "
                                f"NMI={metric_rpma['NMI']:.4f}, "
                                f"ARI={metric_rpma['ARI']:.4f}, "
                                f"grad={final_grad:.3e}, "
                                f"iter={n_iter}, "
                                f"time={time_rpma:.2f}s"
                            )

                        except Exception as e:
                            time_rpma = time.time() - t0
                            row_fail = {
                                "dataset": args.dataset,
                                "method": "RPMA-Huber",
                                "seed": seed,
                                "n": n,
                                "K": K,
                                "image_size": args.image_size,
                                "max_per_class": args.max_per_class,
                                "pca_dim": pca_dim,
                                "k_neighbors": knn,
                                "lam": lam,
                                "delta": delta,
                                "rpa_max_iter": args.rpa_max_iter,
                                "row_normalize_embedding": args.row_normalize_embedding,
                                "ACC": np.nan,
                                "NMI": np.nan,
                                "ARI": np.nan,
                                "time_sec": time_rpma,
                                "final_grad": np.nan,
                                "n_iter": np.nan,
                                "objective": "min_-2<A,X>_plus_lambda_Huber_over_P_K",
                                "error": repr(e),
                            }
                            append_rows_csv([row_fail], out_csv)
                            all_rows.append(row_fail)
                            print(f"  RPMA FAILED | lam={lam:g}, delta={delta:g} | error={repr(e)}")

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No results.")
        return

    group_cols = [
        "dataset",
        "method",
        "image_size",
        "max_per_class",
        "pca_dim",
        "k_neighbors",
        "lam",
        "delta",
        "rpa_max_iter",
        "row_normalize_embedding",
        "objective",
    ]

    summary = (
        df.groupby(group_cols, dropna=False)[["ACC", "NMI", "ARI", "time_sec", "final_grad", "n_iter"]]
        .mean()
        .reset_index()
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if args.save_xlsx:
        write_xlsx_if_possible(out_csv, summary_path, xlsx_path)

    print("\n" + "=" * 80)
    print("Grid search finished.")
    print(f"Raw results saved to:     {out_csv}")
    print(f"Summary results saved to: {summary_path}")
    if args.save_xlsx:
        print(f"Excel summary target:     {xlsx_path}")
    print("=" * 80)

    for metric in ["ACC", "NMI", "ARI"]:
        valid = summary.dropna(subset=[metric])
        if valid.empty:
            continue
        best = valid.sort_values(metric, ascending=False).iloc[0]
        print(f"\nBest by {metric}:")
        print(best[[
            "method", "pca_dim", "k_neighbors", "lam", "delta", "ACC", "NMI", "ARI",
            "time_sec", "final_grad", "n_iter", "objective"
        ]])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["coil20", "yaleB"], required=True)
    parser.add_argument("--data-root", required=True)

    parser.add_argument("--image-size", default="32x32")
    parser.add_argument("--max-per-class", type=int, default=10,
                        help="0 means use all images; e.g. 10 means 10 images per class.")

    parser.add_argument("--pca-dims", default="30,50,80")
    parser.add_argument("--k-neighbors", default="5,10,15")
    parser.add_argument("--lams", default="0.001,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--deltas", default="1e-4,1e-3,1e-2")

    parser.add_argument("--rpa-max-iter", type=int, default=150)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--row-normalize-embedding", action="store_true",
                        help="Optional extension: L2-normalize rows of U before k-means. Off by default for paper consistency.")

    parser.add_argument("--out-dir", default="results/image_grid_paper")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete old output files in out-dir before running.")
    parser.add_argument("--save-xlsx", action="store_true",
                        help="Also export raw and summary results to an xlsx file if openpyxl/xlsxwriter is available.")

    args = parser.parse_args()
    run_grid(args)


if __name__ == "__main__":
    main()
