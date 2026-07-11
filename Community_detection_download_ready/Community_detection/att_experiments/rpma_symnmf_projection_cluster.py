"""
Strict pipeline required by the advisor:

    image features
    -> Gaussian affinity A
    -> RPMA/RPA produces sparse projection matrix X_rpa
    -> SymNMF directly decomposes that projection matrix: X_rpa ≈ H H^T
    -> cluster with the specified number of clusters K

Copy this file to:
    Community_detection/experiments/rpma_symnmf_projection_cluster.py

Run from the project root, for example:
    python -m experiments.rpma_symnmf_projection_cluster --dataset coil20 --data-root datasets/data/coil20 --image-size 32x32 --max-per-class 0 --clusters 20
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

from datasets.image_datasets import load_att_faces, load_coil20, load_extended_yale_b
from evaluation.metrics import evaluate
from methods.rpa import rpa
from methods.symnmf import symnmf_mu, symnmf_pgd, symnmf_cluster_features


SCRIPT_VERSION = "2026-07-08-strict-rpma-projection-symnmf-v1"


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


def parse_str_list(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def symmetrize(M):
    M = np.asarray(M, dtype=np.float64)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    return 0.5 * (M + M.T)


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


def standardize_features(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return X / (std + eps)


def gaussian_affinity(X, zero_diagonal=False):
    """
    Full Gaussian affinity:
        A_ij = exp(-||x_i-x_j||^2 / sigma^2)
        sigma^2 = 2/(n(n-1)) * sum_{i<j} ||x_i-x_j||^2
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


def spectral_projection_baseline(A, K, seed=0):
    """Only a baseline: cluster from top-K eigenvectors of the original A."""
    A = symmetrize(A)
    vals, vecs = eigh(A)
    idx = np.argsort(vals)[::-1][:K]
    U = vecs[:, idx]
    labels = KMeans(n_clusters=K, n_init=20, random_state=seed).fit_predict(U)
    X = U @ U.T
    return labels, X, U


def prepare_rpma_projection_for_symnmf(X_rpa, keep_diagonal=True):
    """
    Strictly use the RPMA/RPA projection matrix as the SymNMF target.

    SymNMF requires a symmetric nonnegative target. X_rpa is already symmetric
    up to numerical error, but it can contain tiny negative entries because RPMA
    imposes a projection constraint and Huber sparsity, not an explicit
    nonnegative constraint. Therefore this function only performs the minimum
    required SymNMF input processing:
        1) symmetrize X_rpa;
        2) clip negative entries to zero;
        3) keep the diagonal by default.

    It never falls back to the original affinity matrix A.
    """
    X = symmetrize(X_rpa)
    neg_ratio = float(np.mean(X < 0.0))
    min_before = float(np.min(X))
    max_before = float(np.max(X))

    S = np.maximum(X, 0.0)
    if not keep_diagonal:
        np.fill_diagonal(S, 0.0)

    density_nonzero = float(np.mean(np.abs(S) > 1e-12))
    return S, {
        "x_rpa_min_before_clip": min_before,
        "x_rpa_max_before_clip": max_before,
        "x_rpa_negative_ratio_before_clip": neg_ratio,
        "symnmf_target_density_nonzero": density_nonzero,
    }


def symnmf_objective(S, H):
    R = S - H @ H.T
    return 0.5 * float(np.sum(R * R))


def labels_from_H(H, mode, K, seed):
    if mode == "argmax":
        return np.asarray(np.argmax(H, axis=1), dtype=int)
    Z = symnmf_cluster_features(H, mode)
    return KMeans(n_clusters=K, n_init=20, random_state=seed).fit_predict(Z)


def append_rows_csv(rows, csv_path):
    df = pd.DataFrame(rows)
    csv_path = Path(csv_path)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False, encoding="utf-8-sig")


def save_summary(raw_csv, summary_csv, xlsx_path=None):
    df = pd.read_csv(raw_csv)
    group_cols = [
        "dataset", "method", "K", "image_size", "max_per_class", "standardize",
        "lam", "delta", "rpa_max_iter", "symnmf_solver", "symnmf_init",
        "symnmf_feature", "keep_diagonal",
    ]
    metric_cols = [
        "ACC", "NMI", "ARI", "time_sec", "rpa_time_sec", "symnmf_time_sec",
        "symnmf_obj_final", "symnmf_n_iter", "x_rpa_negative_ratio_before_clip",
        "symnmf_target_density_nonzero",
    ]
    group_cols = [c for c in group_cols if c in df.columns]
    metric_cols = [c for c in metric_cols if c in df.columns]
    summary = df.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    if xlsx_path is not None:
        with pd.ExcelWriter(xlsx_path) as writer:
            df.to_excel(writer, sheet_name="raw_results", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
    return summary


def run(args):
    image_size = parse_size(args.image_size)
    max_per_class = None if args.max_per_class <= 0 else args.max_per_class
    lams = parse_float_list(args.lams)
    deltas = parse_float_list(args.deltas)
    seeds = parse_int_list(args.seeds)
    symnmf_solvers = parse_str_list(args.symnmf_solvers)
    symnmf_inits = parse_str_list(args.symnmf_inits)
    symnmf_features = parse_str_list(args.symnmf_features)
    symnmf_seeds = parse_int_list(args.symnmf_seeds)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    raw_csv = out_dir / f"{args.dataset}_strict_rpma_symnmf_raw.csv"
    summary_csv = out_dir / f"{args.dataset}_strict_rpma_symnmf_summary.csv"
    xlsx_path = out_dir / f"{args.dataset}_strict_rpma_symnmf_summary.xlsx"
    config_path = out_dir / f"{args.dataset}_strict_rpma_symnmf_config.json"

    if args.overwrite:
        for p in [raw_csv, summary_csv, xlsx_path, config_path]:
            if p.exists():
                p.unlink()

    config = vars(args).copy()
    config.update({
        "script_version": SCRIPT_VERSION,
        "strict_pipeline": (
            "features -> Gaussian affinity A -> RPA/RPMA returns sparse projection X_rpa -> "
            "SymNMF decomposes X_rpa target S=max((X_rpa+X_rpa.T)/2,0) -> clustering from H"
        ),
        "important": "SymNMF is NOT applied to the original affinity A in the strict method.",
    })
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("=" * 88)
    print("STRICT advisor pipeline")
    print("RPA/RPMA sparse projection matrix X_rpa -> SymNMF X_rpa≈HH^T -> clustering")
    print("SymNMF target is NOT original affinity A.")
    print(f"dataset       = {args.dataset}")
    print(f"data_root     = {args.data_root}")
    print(f"image_size    = {args.image_size}")
    print(f"max_per_class = {args.max_per_class}  (0 means full dataset)")
    print(f"clusters      = {args.clusters}  (0 means use dataset class count; COIL20 full => 20)")
    print(f"lams          = {lams}")
    print(f"deltas        = {deltas}")
    print(f"symnmf        = solvers={symnmf_solvers}, inits={symnmf_inits}, features={symnmf_features}")
    print(f"out_dir       = {out_dir}")
    print("=" * 88)

    all_rows = []
    for seed in seeds:
        print(f"\n[Dataset] seed={seed}")
        X, y, K_data = load_dataset(
            args.dataset,
            args.data_root,
            image_size=image_size,
            max_per_class=max_per_class,
            random_state=seed,
        )
        K = int(K_data if args.clusters <= 0 else args.clusters)
        n, d = X.shape
        print(f"Loaded: X={X.shape}, dataset_classes={K_data}, used_K={K}")

        if args.dataset == "coil20" and args.max_per_class == 0 and K != 20:
            print(f"WARNING: COIL20 full experiment should use K=20, but used_K={K}.")

        X_used = standardize_features(X) if args.standardize else np.asarray(X, dtype=np.float64)

        print("[Affinity] build original Gaussian affinity A, only for RPA/RPMA input")
        t_aff = time.time()
        A, sigma2 = gaussian_affinity(X_used, zero_diagonal=args.zero_diagonal_affinity)
        aff_time = time.time() - t_aff
        print(f"sigma^2={sigma2:.6e}, affinity_time={aff_time:.2f}s")

        if args.include_spectral_baseline:
            t0 = time.time()
            labels_sp, X_sp, U_sp = spectral_projection_baseline(A, K, seed=seed)
            met_sp = evaluate(y, labels_sp)
            row = {
                "dataset": args.dataset,
                "method": "Spectral-Projection-baseline-on-A",
                "seed": seed,
                "n": n,
                "feature_dim": d,
                "K": K,
                "image_size": args.image_size,
                "max_per_class": args.max_per_class,
                "standardize": args.standardize,
                "sigma2": sigma2,
                "affinity_time_sec": aff_time,
                "lam": np.nan,
                "delta": np.nan,
                "rpa_max_iter": np.nan,
                "symnmf_solver": "",
                "symnmf_init": "",
                "symnmf_seed": np.nan,
                "symnmf_feature": "",
                "keep_diagonal": args.keep_diagonal,
                "ACC": met_sp["ACC"],
                "NMI": met_sp["NMI"],
                "ARI": met_sp["ARI"],
                "time_sec": time.time() - t0,
                "rpa_time_sec": np.nan,
                "symnmf_time_sec": np.nan,
                "symnmf_obj_final": np.nan,
                "symnmf_n_iter": np.nan,
                "x_rpa_negative_ratio_before_clip": np.nan,
                "symnmf_target_density_nonzero": np.nan,
            }
            append_rows_csv([row], raw_csv)
            all_rows.append(row)
            print(f"  Baseline Spectral on A | ACC={met_sp['ACC']:.4f}, NMI={met_sp['NMI']:.4f}, ARI={met_sp['ARI']:.4f}")

        for lam in lams:
            for delta in deltas:
                print(f"\n[RPA/RPMA] lam={lam:g}, delta={delta:g}")
                t_rpa = time.time()
                X_rpa, U_rpa, history = rpa(
                    A,
                    K,
                    lam=lam,
                    delta=delta,
                    max_iter=args.rpa_max_iter,
                    eig_init=True,
                    return_history=True,
                    verbose=args.verbose_rpa,
                )
                rpa_time = time.time() - t_rpa
                final_grad = float(history[-1]) if history else np.nan
                print(f"  RPA done: time={rpa_time:.2f}s, iter={len(history)}, final_grad={final_grad:.3e}")

                # This is the strict handoff: use X_rpa as the matrix to be decomposed by SymNMF.
                S_sym, xstats = prepare_rpma_projection_for_symnmf(
                    X_rpa,
                    keep_diagonal=args.keep_diagonal,
                )
                print(
                    "  SymNMF target = clipped/symmetrized X_rpa; "
                    f"negative_ratio_before_clip={xstats['x_rpa_negative_ratio_before_clip']:.4f}, "
                    f"density={xstats['symnmf_target_density_nonzero']:.4f}"
                )

                if args.save_matrices:
                    npz_path = out_dir / f"matrices_seed{seed}_K{K}_lam{lam:g}_delta{delta:g}.npz"
                    np.savez_compressed(
                        npz_path,
                        A=A.astype(np.float32),
                        X_rpa=X_rpa.astype(np.float32),
                        S_symnmf_target=S_sym.astype(np.float32),
                        U_rpa=U_rpa.astype(np.float32),
                        y_true=np.asarray(y, dtype=int),
                    )
                    print(f"  Saved matrices: {npz_path}")

                for solver in symnmf_solvers:
                    for init in symnmf_inits:
                        for sym_seed in symnmf_seeds:
                            real_seed = int(seed * 100000 + sym_seed)
                            t_sym = time.time()
                            if solver == "mu":
                                H, hinfo = symnmf_mu(
                                    S_sym,
                                    K,
                                    max_iter=args.symnmf_max_iter,
                                    tol=args.symnmf_tol,
                                    seed=real_seed,
                                    init=init,
                                    return_history=True,
                                )
                            elif solver == "pgd":
                                H, hinfo = symnmf_pgd(
                                    S_sym,
                                    K,
                                    max_iter=args.symnmf_max_iter,
                                    lr=args.symnmf_lr,
                                    tol=args.symnmf_tol,
                                    seed=real_seed,
                                    init=init,
                                    return_history=True,
                                )
                            else:
                                raise ValueError(f"Unknown symnmf solver: {solver}")
                            sym_time = time.time() - t_sym
                            obj_final = symnmf_objective(S_sym, H)
                            n_iter = int(hinfo.get("n_iter", np.nan))

                            for feat in symnmf_features:
                                labels = labels_from_H(H, feat, K, seed=real_seed)
                                met = evaluate(y, labels)
                                row = {
                                    "dataset": args.dataset,
                                    "method": "RPMA-projection-then-SymNMF",
                                    "seed": seed,
                                    "n": n,
                                    "feature_dim": d,
                                    "K": K,
                                    "dataset_classes": K_data,
                                    "image_size": args.image_size,
                                    "max_per_class": args.max_per_class,
                                    "standardize": args.standardize,
                                    "zero_diagonal_affinity": args.zero_diagonal_affinity,
                                    "sigma2": sigma2,
                                    "affinity_time_sec": aff_time,
                                    "lam": lam,
                                    "delta": delta,
                                    "rpa_max_iter": args.rpa_max_iter,
                                    "rpa_final_grad": final_grad,
                                    "rpa_n_iter": len(history),
                                    "symnmf_solver": solver,
                                    "symnmf_init": init,
                                    "symnmf_seed": sym_seed,
                                    "symnmf_feature": feat,
                                    "keep_diagonal": args.keep_diagonal,
                                    "ACC": met["ACC"],
                                    "NMI": met["NMI"],
                                    "ARI": met["ARI"],
                                    "time_sec": aff_time + rpa_time + sym_time,
                                    "rpa_time_sec": rpa_time,
                                    "symnmf_time_sec": sym_time,
                                    "symnmf_obj_final": obj_final,
                                    "symnmf_n_iter": n_iter,
                                    **xstats,
                                }
                                append_rows_csv([row], raw_csv)
                                all_rows.append(row)
                                print(
                                    f"  SymNMF({solver}, init={init}, sseed={sym_seed}) | feature={feat} | "
                                    f"ACC={met['ACC']:.4f}, NMI={met['NMI']:.4f}, ARI={met['ARI']:.4f}, "
                                    f"obj={obj_final:.3e}, sym_time={sym_time:.2f}s"
                                )

    summary = save_summary(raw_csv, summary_csv, xlsx_path if args.save_xlsx else None)

    print("\n" + "=" * 88)
    print("Finished strict RPMA-projection -> SymNMF experiment")
    print(f"Raw CSV:     {raw_csv}")
    print(f"Summary CSV: {summary_csv}")
    if args.save_xlsx:
        print(f"Excel:       {xlsx_path}")
    print(f"Config:      {config_path}")
    print("=" * 88)

    for metric in ["ACC", "NMI", "ARI"]:
        valid = summary.dropna(subset=[metric])
        if len(valid) == 0:
            continue
        best = valid.sort_values(metric, ascending=False).iloc[0]
        cols = [
            "method", "K", "lam", "delta", "symnmf_solver", "symnmf_init",
            "symnmf_feature", "ACC", "NMI", "ARI", "symnmf_obj_final",
        ]
        cols = [c for c in cols if c in best.index]
        print(f"\nBest by {metric}:")
        print(best[cols])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["coil20", "yaleB", "att_faces"], required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--image-size", default="32x32", help="Use 'original' or e.g. 32x32")
    p.add_argument("--max-per-class", type=int, default=0, help="0 means full dataset")
    p.add_argument("--clusters", type=int, default=0, help="0 means use dataset class count; use 20 for full COIL20")
    p.add_argument("--seeds", default="0")
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--zero-diagonal-affinity", action="store_true", help="Only affects original A before RPA; default keeps Gaussian diagonal")

    p.add_argument("--lams", default="0.001,0.005,0.01,0.05,0.1")
    p.add_argument("--deltas", default="1e-3")
    p.add_argument("--rpa-max-iter", type=int, default=200)
    p.add_argument("--verbose-rpa", action="store_true")

    p.add_argument("--symnmf-solvers", default="mu", help="mu or pgd, comma-separated")
    p.add_argument("--symnmf-inits", default="random,nndsvd_spectral")
    p.add_argument("--symnmf-seeds", default="0,1,2,3,4")
    p.add_argument("--symnmf-features", default="H_norm,H,argmax", help="H_norm,H,HHt,HHt_norm,argmax")
    p.add_argument("--symnmf-max-iter", type=int, default=1000)
    p.add_argument("--symnmf-tol", type=float, default=1e-5)
    p.add_argument("--symnmf-lr", type=float, default=1e-3)

    p.add_argument("--keep-diagonal", action="store_true", default=True, help="Keep diagonal of X_rpa target; default True")
    p.add_argument("--drop-diagonal", dest="keep_diagonal", action="store_false", help="Not recommended for projection matrices")
    p.add_argument("--include-spectral-baseline", action="store_true")
    p.add_argument("--save-matrices", action="store_true")
    p.add_argument("--out-dir", default="results/strict_rpma_projection_symnmf")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save-xlsx", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
