"""
Run all available methods on image clustering datasets.

Place this file at:
    Community_detection/experiments/image_all_methods.py

This version fixes the CSV/XLSX saving issue:
    It does NOT append rows with different columns into one CSV during the run.
    Instead, it collects all method results in memory and writes CSV/XLSX once at the end.

Recommended first test:
    python -m experiments.image_all_methods \
        --dataset coil20 \
        --data-root datasets/data/coil20 \
        --image-size original \
        --max-per-class 3 \
        --methods all \
        --rpma-lam 0.04 \
        --rpma-delta 1e-4 \
        --out-dir results/coil20_all_methods_3_per_class \
        --overwrite \
        --save-xlsx
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
from sklearn.preprocessing import normalize

from datasets.image_datasets import load_att_faces, load_coil20, load_extended_yale_b
from evaluation.metrics import evaluate

from methods.rpa import rpa
from methods.spectral_utils import spectral_rounding
from methods.admm_sd1 import admm_sd1
from methods.admm_sd2 import admm_sd2
from methods.clr import clr
from methods.ssl2 import ssl2
from methods.SLSA import slsa


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------
def parse_size(s):
    """Parse image size. Use 'original' / 'none' / 'orig' to keep original size."""
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


def parse_int_list(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def symmetrize(A):
    A = np.asarray(A, dtype=float)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    return 0.5 * (A + A.T)


def standardize_features(X, eps=1e-12):
    """
    Center and scale raw pixel features.
    This is not PCA and does not change the feature dimension.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return X / (std + eps)


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
    if dataset in {"att_faces", "attfaces", "orl"}:
        return load_att_faces(
            data_root,
            image_size=image_size,
            max_per_class=max_per_class,
            random_state=random_state,
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def paper_gaussian_affinity(X, zero_diagonal=False):
    """
    Full Gaussian affinity:
        A_ij = exp(-||x_i - x_j||^2 / sigma^2)
        sigma^2 = 2 / (n(n-1)) * sum_{i<j} ||x_i - x_j||^2

    No PCA and no kNN sparsification are used here.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least two samples to build an affinity matrix.")

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


def kmeans_on_rows(U, K, random_state=0, row_normalize=True):
    U = np.asarray(U, dtype=float)
    if row_normalize:
        U = normalize(U, norm="l2")
    return KMeans(n_clusters=K, n_init=20, random_state=random_state).fit_predict(U)


def spectral_projection_labels(A, K, random_state=0, row_normalize=True):
    """
    Paper-style Spectral-Projection:
        X_spe = U_K U_K^T
    where U_K contains top-K eigenvectors of A.
    """
    A = symmetrize(A)
    eigvals, eigvecs = eigh(A)
    idx = np.argsort(eigvals)[::-1][:K]
    U = eigvecs[:, idx]
    labels = kmeans_on_rows(U, K, random_state=random_state, row_normalize=row_normalize)
    return labels, U


def make_result_row(args, dataset, method, y_true, labels, elapsed, seed, n, K,
                    feature_dim, sigma2, extra=None, error=None):
    base = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "n": n,
        "K": K,
        "image_size": args.image_size,
        "feature_dim": feature_dim,
        "standardize": args.standardize,
        "zero_diagonal": args.zero_diagonal,
        "sigma2": sigma2,
        "time_sec": elapsed,
        "error": "",
    }

    if error is None:
        metric = evaluate(y_true, labels)
        base.update({
            "ACC": metric["ACC"],
            "NMI": metric["NMI"],
            "ARI": metric["ARI"],
        })
        if extra:
            base.update(extra)
        print(
            f"  {method:<22} | "
            f"ACC={metric['ACC']:.4f}, NMI={metric['NMI']:.4f}, ARI={metric['ARI']:.4f}, "
            f"time={elapsed:.2f}s"
        )
    else:
        base.update({
            "ACC": np.nan,
            "NMI": np.nan,
            "ARI": np.nan,
            "error": repr(error),
        })
        if extra:
            base.update(extra)
        print(f"  {method:<22} | FAILED | error={repr(error)}")

    return base


# ---------------------------------------------------------------------
# Method runner
# ---------------------------------------------------------------------
def run_methods_on_A(A, y, K, args, seed, n, feature_dim, sigma2):
    rows = []
    methods = [x.strip().lower() for x in args.methods.split(",") if x.strip()]
    A = symmetrize(A)

    def wants(name):
        return "all" in methods or name in methods

    # 1. Paper-style Spectral Projection
    if wants("spectral_projection"):
        method = "Spectral-Projection"
        t0 = time.time()
        try:
            labels, _ = spectral_projection_labels(
                A, K, random_state=seed, row_normalize=not args.no_row_normalize
            )
            rows.append(make_result_row(
                args, args.dataset, method, y, labels, time.time() - t0,
                seed, n, K, feature_dim, sigma2,
                extra={"laplacian": False}
            ))
        except Exception as exc:
            rows.append(make_result_row(
                args, args.dataset, method, y, None, time.time() - t0,
                seed, n, K, feature_dim, sigma2,
                extra={"laplacian": False}, error=exc
            ))

    # 2. Usual unnormalized-Laplacian spectral clustering
    if wants("spectral_laplacian"):
        method = "Spectral-Laplacian"
        t0 = time.time()
        try:
            labels = spectral_rounding(A, K, random_state=seed, laplacian=True)
            rows.append(make_result_row(
                args, args.dataset, method, y, labels, time.time() - t0,
                seed, n, K, feature_dim, sigma2,
                extra={"laplacian": True}
            ))
        except Exception as exc:
            rows.append(make_result_row(
                args, args.dataset, method, y, None, time.time() - t0,
                seed, n, K, feature_dim, sigma2,
                extra={"laplacian": True}, error=exc
            ))

    # 3. RPMA-Huber / RPA
    if wants("rpma") or wants("rpa"):
        method = "RPMA-Huber"
        t0 = time.time()
        try:
            X_rpa, U_rpa, history = rpa(
                A,
                K,
                lam=args.rpma_lam,
                delta=args.rpma_delta,
                max_iter=args.rpma_max_iter,
                eig_init=True,
                return_history=True,
                verbose=False,
            )
            labels = kmeans_on_rows(
                U_rpa, K, random_state=seed, row_normalize=not args.no_row_normalize
            )
            rows.append(make_result_row(
                args, args.dataset, method, y, labels, time.time() - t0,
                seed, n, K, feature_dim, sigma2,
                extra={
                    "lam": args.rpma_lam,
                    "delta": args.rpma_delta,
                    "max_iter": args.rpma_max_iter,
                    "final_grad": float(history[-1]) if history else np.nan,
                    "n_iter": len(history),
                }
            ))
        except Exception as exc:
            rows.append(make_result_row(
                args, args.dataset, method, y, None, time.time() - t0,
                seed, n, K, feature_dim, sigma2,
                extra={
                    "lam": args.rpma_lam,
                    "delta": args.rpma_delta,
                    "max_iter": args.rpma_max_iter,
                    "final_grad": np.nan,
                    "n_iter": np.nan,
                },
                error=exc,
            ))

    # 4. ADMM-SD1
    if wants("admm_sd1"):
        method = "ADMM-SD1"
        t0 = time.time()
        try:
            X = admm_sd1(A, K, rho=args.admm_rho, tol=args.admm_tol, max_iter=args.admm_max_iter)
            labels = spectral_rounding(X, K, random_state=seed, laplacian=False)
            rows.append(make_result_row(
                args, args.dataset, method, y, labels, time.time() - t0,
                seed, n, K, feature_dim, sigma2,
                extra={"rho": args.admm_rho, "tol": args.admm_tol, "max_iter": args.admm_max_iter}
            ))
        except Exception as exc:
            rows.append(make_result_row(
                args, args.dataset, method, y, None, time.time() - t0,
                seed, n, K, feature_dim, sigma2,
                extra={"rho": args.admm_rho, "tol": args.admm_tol, "max_iter": args.admm_max_iter},
                error=exc,
            ))

    # 5. ADMM-SD2
    if wants("admm_sd2"):
        method = "ADMM-SD2"
        t0 = time.time()
        try:
            X = admm_sd2(A, K, rho=args.admm_rho, tol=args.admm_tol, max_iter=args.admm_max_iter)
            labels = spectral_rounding(X, K, random_state=seed, laplacian=False)
            rows.append(make_result_row(
                args, args.dataset, method, y, labels, time.time() - t0,
                seed, n, K, feature_dim, sigma2,
                extra={"rho": args.admm_rho, "tol": args.admm_tol, "max_iter": args.admm_max_iter}
            ))
        except Exception as exc:
            rows.append(make_result_row(
                args, args.dataset, method, y, None, time.time() - t0,
                seed, n, K, feature_dim, sigma2,
                extra={"rho": args.admm_rho, "tol": args.admm_tol, "max_iter": args.admm_max_iter},
                error=exc,
            ))

    # 6. CLR
    if wants("clr"):
        method = "CLR"
        t0 = time.time()
        try:
            S = clr(A, lam=args.clr_lam, K=K, max_iter=args.clr_max_iter)
            labels = spectral_rounding(S, K, random_state=seed, laplacian=True)
            rows.append(make_result_row(
                args, args.dataset, method, y, labels, time.time() - t0,
                seed, n, K, feature_dim, sigma2,
                extra={"clr_lam": args.clr_lam, "max_iter": args.clr_max_iter}
            ))
        except Exception as exc:
            rows.append(make_result_row(
                args, args.dataset, method, y, None, time.time() - t0,
                seed, n, K, feature_dim, sigma2,
                extra={"clr_lam": args.clr_lam, "max_iter": args.clr_max_iter},
                error=exc,
            ))

    # 7. SSL2
    if wants("ssl2"):
        method = "SSL2"
        t0 = time.time()
        try:
            # ssl2.trunc_matrix internally transforms eta into the number of upper-triangle entries.
            # eta = n + 2*n*k means roughly k undirected retained edges per sample.
            if args.ssl_eta > 0:
                eta = int(args.ssl_eta)
            else:
                eta = int(n + 2 * n * args.ssl_eta_k)
            eta = max(eta, n + 2)
            eta = min(eta, n * n)

            Z = ssl2(
                A,
                c=K,
                eta=eta,
                theta=args.ssl_theta,
                tau=args.ssl_tau,
                loss=args.ssl_loss,
                max_iter=args.ssl_max_iter,
            )
            labels = spectral_rounding(Z, K, random_state=seed, laplacian=True)
            rows.append(make_result_row(
                args, args.dataset, method, y, labels, time.time() - t0,
                seed, n, K, feature_dim, sigma2,
                extra={
                    "eta": eta,
                    "eta_k": args.ssl_eta_k,
                    "theta": args.ssl_theta,
                    "tau": args.ssl_tau,
                    "loss": args.ssl_loss,
                    "max_iter": args.ssl_max_iter,
                }
            ))
        except Exception as exc:
            rows.append(make_result_row(
                args, args.dataset, method, y, None, time.time() - t0,
                seed, n, K, feature_dim, sigma2,
                extra={
                    "eta": args.ssl_eta,
                    "eta_k": args.ssl_eta_k,
                    "theta": args.ssl_theta,
                    "tau": args.ssl_tau,
                    "loss": args.ssl_loss,
                    "max_iter": args.ssl_max_iter,
                },
                error=exc,
            ))


    # 8. SLSA: Simultaneously Low-Rank and Sparse Approximation
    if wants("slsa"):
        method = "SLSA"
        t0 = time.time()
        try:
            # Keep the same eta convention as SSL2 for fair command-line use:
            # eta = n + 2*n*k means roughly k undirected retained edges per sample.
            if args.slsa_eta > 0:
                eta = int(args.slsa_eta)
            else:
                eta = int(n + 2 * n * args.slsa_eta_k)
            eta = max(eta, n + 2)
            eta = min(eta, n * n)

            Z, U_slsa, info = slsa(
                A,
                K=K,
                eta=eta,
                theta=args.slsa_theta,
                tau=args.slsa_tau,
                loss=args.slsa_loss,
                max_iter=args.slsa_max_iter,
                eta_mode="total",
                return_info=True,
                verbose=args.slsa_verbose,
            )

            if args.slsa_rounding == "laplacian":
                labels = spectral_rounding(Z, K, random_state=seed, laplacian=True)
            elif args.slsa_rounding == "top_eigen":
                labels = spectral_rounding(Z, K, random_state=seed, laplacian=False)
            elif args.slsa_rounding == "U":
                labels = kmeans_on_rows(
                    U_slsa,
                    K,
                    random_state=seed,
                    row_normalize=not args.no_row_normalize,
                )
            else:
                raise ValueError(f"Unknown slsa_rounding={args.slsa_rounding}")

            rows.append(make_result_row(
                args, args.dataset, method, y, labels, time.time() - t0,
                seed, n, K, feature_dim, sigma2,
                extra={
                    "eta": eta,
                    "eta_k": args.slsa_eta_k,
                    "theta": args.slsa_theta,
                    "tau": args.slsa_tau,
                    "loss": args.slsa_loss,
                    "max_iter": args.slsa_max_iter,
                    "rounding": args.slsa_rounding,
                    "n_iter": info.get("n_iter", np.nan),
                    "converged": info.get("converged", False),
                    "final_diff": info.get("final_diff", np.nan),
                    "nnz": info.get("nnz", np.nan),
                }
            ))
        except Exception as exc:
            rows.append(make_result_row(
                args, args.dataset, method, y, None, time.time() - t0,
                seed, n, K, feature_dim, sigma2,
                extra={
                    "eta": args.slsa_eta,
                    "eta_k": args.slsa_eta_k,
                    "theta": args.slsa_theta,
                    "tau": args.slsa_tau,
                    "loss": args.slsa_loss,
                    "max_iter": args.slsa_max_iter,
                    "rounding": args.slsa_rounding,
                    "n_iter": np.nan,
                    "converged": False,
                    "final_diff": np.nan,
                    "nnz": np.nan,
                },
                error=exc,
            ))

    return rows


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------
def run(args):
    image_size = parse_size(args.image_size)
    max_per_class = None if args.max_per_class <= 0 else args.max_per_class
    seeds = parse_int_list(args.seeds)

    ensure_dir(args.out_dir)
    out_dir = Path(args.out_dir)
    raw_csv = out_dir / f"{args.dataset}_all_methods_results.csv"
    summary_csv = out_dir / f"{args.dataset}_all_methods_summary.csv"
    xlsx_path = out_dir / f"{args.dataset}_all_methods_summary.xlsx"
    config_path = out_dir / f"{args.dataset}_all_methods_config.json"

    if args.overwrite:
        for p in [raw_csv, summary_csv, xlsx_path, config_path]:
            if p.exists():
                p.unlink()

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Image all-methods experiment")
    print("Pipeline: images -> full Gaussian affinity -> methods -> labels -> ACC/NMI/ARI")
    print(f"dataset       = {args.dataset}")
    print(f"data_root     = {args.data_root}")
    print(f"image_size    = {args.image_size}")
    print(f"max_per_class = {args.max_per_class}  (0 means full dataset)")
    print(f"methods       = {args.methods}")
    print(f"out_dir       = {out_dir}")
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
        n, feature_dim = X.shape
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

        rows = run_methods_on_A(A, y, K, args, seed, n, feature_dim, sigma2)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # Save raw results once. This avoids malformed CSV caused by variable row columns.
    df.to_csv(raw_csv, index=False, encoding="utf-8-sig")

    # Summarize mean metrics over seeds.
    metric_cols = ["ACC", "NMI", "ARI", "time_sec"]
    group_cols = [
        "dataset", "method", "image_size", "feature_dim", "standardize",
        "zero_diagonal"
    ]
    possible_param_cols = [
        "lam", "delta", "rho", "tol", "clr_lam", "eta", "eta_k",
        "theta", "tau", "loss", "max_iter", "laplacian", "rounding", "n_iter", "converged", "final_diff", "nnz"
    ]
    for c in possible_param_cols:
        if c in df.columns and not df[c].isna().all():
            group_cols.append(c)

    summary = (
        df.groupby(group_cols, dropna=False)[metric_cols]
          .mean()
          .reset_index()
          .sort_values(["ACC", "NMI", "ARI"], ascending=False)
    )
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    if args.save_xlsx:
        with pd.ExcelWriter(xlsx_path) as writer:
            df.to_excel(writer, sheet_name="raw_results", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
        print(f"Excel saved to: {xlsx_path}")

    print("\n" + "=" * 80)
    print("Finished")
    print(f"Raw CSV:     {raw_csv}")
    print(f"Summary CSV: {summary_csv}")
    if args.save_xlsx:
        print(f"Excel:       {xlsx_path}")
    print("=" * 80)

    print("\nBest methods by ACC:")
    cols = [c for c in ["method", "ACC", "NMI", "ARI", "time_sec", "lam", "delta", "eta", "clr_lam"] if c in summary.columns]
    print(summary[cols].head(20).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["coil20", "yaleB", "att_faces"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image-size", default="original", help="Use 'original' for original size, or e.g. 32x32")
    parser.add_argument("--max-per-class", type=int, default=3, help="0 means full dataset")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--out-dir", default="results/image_all_methods")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-xlsx", action="store_true")

    parser.add_argument(
        "--methods",
        default="all",
        help=(
            "Comma-separated methods. Use 'all' or any subset of: "
            "spectral_projection,spectral_laplacian,rpma,admm_sd1,admm_sd2,clr,ssl2,slsa"
        ),
    )

    parser.add_argument("--standardize", action="store_true", help="Center and scale pixel features; no dimensionality reduction")
    parser.add_argument("--zero-diagonal", action="store_true", help="Set diagonal of A to zero")
    parser.add_argument("--no-row-normalize", action="store_true", help="Disable row normalization before k-means for embedding methods")

    parser.add_argument("--rpma-lam", type=float, default=0.04)
    parser.add_argument("--rpma-delta", type=float, default=1e-4)
    parser.add_argument("--rpma-max-iter", type=int, default=200)

    parser.add_argument("--admm-rho", type=float, default=1.0)
    parser.add_argument("--admm-tol", type=float, default=1e-4)
    parser.add_argument("--admm-max-iter", type=int, default=200)

    parser.add_argument("--clr-lam", type=float, default=1.0)
    parser.add_argument("--clr-max-iter", type=int, default=100)

    parser.add_argument("--ssl-eta", type=int, default=-1, help="If >0, use this eta directly")
    parser.add_argument("--ssl-eta-k", type=int, default=10, help="If ssl-eta<=0, eta=n+2*n*ssl_eta_k")
    parser.add_argument("--ssl-theta", type=float, default=1.0)
    parser.add_argument("--ssl-tau", type=float, default=1e-6)
    parser.add_argument("--ssl-loss", choices=["l1", "fro"], default="l1")
    parser.add_argument("--ssl-max-iter", type=int, default=200)

    parser.add_argument("--slsa-eta", type=int, default=-1, help="If >0, use this total-nonzero eta directly")
    parser.add_argument("--slsa-eta-k", type=int, default=10, help="If slsa-eta<=0, eta=n+2*n*slsa_eta_k")
    parser.add_argument("--slsa-theta", type=float, default=1.0)
    parser.add_argument("--slsa-tau", type=float, default=1e-6)
    parser.add_argument("--slsa-loss", choices=["l1", "fro"], default="fro")
    parser.add_argument("--slsa-max-iter", type=int, default=200)
    parser.add_argument("--slsa-rounding", choices=["laplacian", "top_eigen", "U"], default="laplacian")
    parser.add_argument("--slsa-verbose", action="store_true", help="Print SLSA iteration diffs")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
