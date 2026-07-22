"""
Probe and post-process the H factor from SymNMF on the original full Gaussian affinity.

This script does NOT use kNN / mutual-NN sparsification. It builds
A_ij = exp(-||x_i - x_j||^2 / sigma^2) and then runs SymNMF:
    min_{H >= 0} 0.5 ||A - H H^T||_F^2.

It explores different H-based features and rounding rules:
- H, H_norm, H_l1, H_power_* and normalized variants
- kmeans, balanced kmeans, argmax
- H diagnostics: entropy, max-vs-second gap, column balance, sparsity
"""

SCRIPT_VERSION = "2026-07-08-symnmf-H-probe-v1"

import argparse
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix, pairwise_distances

from evaluation.metrics import evaluate
from methods.symnmf import symnmf_mu, symnmf_pgd, symnmf_cluster_features


def parse_csv(s, typ=str):
    if s is None or str(s).strip() == "":
        return []
    return [typ(x.strip()) for x in str(s).split(",") if x.strip() != ""]


def parse_bool(s):
    if isinstance(s, bool):
        return s
    s = str(s).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool: {s}")


def parse_image_size(s):
    if s is None or str(s).lower() == "original":
        return "original"
    if "x" in str(s):
        a, b = str(s).lower().split("x")
        return (int(a), int(b))
    v = int(s)
    return (v, v)


def _call_loader(fn, data_root, image_size, max_per_class, seed):
    sig = inspect.signature(fn)
    kwargs = {}
    params = sig.parameters
    if "data_root" in params:
        kwargs["data_root"] = data_root
    elif "root" in params:
        kwargs["root"] = data_root
    elif "path" in params:
        kwargs["path"] = data_root

    if "image_size" in params:
        kwargs["image_size"] = None if image_size == "original" else image_size
    elif "resize" in params:
        kwargs["resize"] = None if image_size == "original" else image_size

    if "max_per_class" in params:
        kwargs["max_per_class"] = max_per_class
    elif "n_per_class" in params:
        kwargs["n_per_class"] = max_per_class

    if "random_state" in params:
        kwargs["random_state"] = seed
    elif "seed" in params:
        kwargs["seed"] = seed

    try:
        return fn(**kwargs)
    except TypeError:
        kwargs.pop("data_root", None)
        kwargs.pop("root", None)
        kwargs.pop("path", None)
        return fn(data_root, **kwargs)


def load_dataset(dataset, data_root, image_size, max_per_class, seed):
    d = dataset.lower()
    if d == "coil20":
        from datasets.image_datasets import load_coil20
        out = _call_loader(load_coil20, data_root, image_size, max_per_class, seed)
    elif d in {"yaleb", "extended_yale_b", "extended-yale-b"}:
        from datasets.image_datasets import load_extended_yale_b
        out = _call_loader(load_extended_yale_b, data_root, image_size, max_per_class, seed)
    elif d in {"att_faces", "attfaces", "orl"}:
        from datasets.image_datasets import load_att_faces
        out = _call_loader(load_att_faces, data_root, image_size, max_per_class, seed)
    else:
        raise ValueError(f"Unknown dataset {dataset!r}")

    if len(out) == 3:
        X, y, K = out
    else:
        X, y = out
        K = len(np.unique(y))
    return np.asarray(X, dtype=np.float64), np.asarray(y), int(K)


def build_full_gaussian_affinity(X, sigma2_scale=1.0, zero_diagonal=False):
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    D2 = pairwise_distances(X, metric="sqeuclidean", n_jobs=1)
    off = D2[np.triu_indices(n, k=1)]
    sigma2 = float(2.0 * np.sum(off) / (n * (n - 1))) * float(sigma2_scale)
    sigma2 = max(sigma2, 1e-12)
    A = np.exp(-D2 / sigma2)
    A = 0.5 * (A + A.T)
    if zero_diagonal:
        np.fill_diagonal(A, 0.0)
    return A, {"sigma2": sigma2}


def spectral_embedding(A, K):
    vals, vecs = np.linalg.eigh(0.5 * (A + A.T))
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    U = vecs[:, idx[:K]]
    gap = float(vals[K - 1] - vals[K]) if len(vals) > K else np.nan
    return U, vals, gap


def row_l2_norm(Z, eps=1e-12):
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), eps)


def row_l1_norm(Z, eps=1e-12):
    return Z / np.maximum(np.sum(np.abs(Z), axis=1, keepdims=True), eps)


def col_l2_norm(Z, eps=1e-12):
    return Z / np.maximum(np.linalg.norm(Z, axis=0, keepdims=True), eps)


def col_sum_norm(Z, eps=1e-12):
    return Z / np.maximum(np.sum(Z, axis=0, keepdims=True), eps)


def label_stats(labels):
    _, counts = np.unique(labels, return_counts=True)
    return int(len(counts)), int(counts.min()), int(counts.max()), counts.tolist()


def kmeans_labels(Z, K, seed, n_init=50):
    return KMeans(n_clusters=K, n_init=n_init, random_state=seed).fit_predict(Z)


def balanced_kmeans_labels(Z, K, seed, max_iter=30, n_init=20):
    """Balanced k-means with exact equal sizes when n is divisible by K.

    It alternates between capacity-constrained assignment and center update.
    For n=600, K=20 this enforces 30 samples per cluster.
    """
    Z = np.asarray(Z, dtype=np.float64)
    n = Z.shape[0]
    if n % K != 0:
        raise ValueError(f"Exact balanced k-means requires n divisible by K; got n={n}, K={K}")
    cap = n // K

    best_labels = None
    best_obj = np.inf
    rng = np.random.default_rng(seed)

    for restart in range(n_init):
        km_seed = int(rng.integers(0, 2**31 - 1))
        centers = KMeans(n_clusters=K, n_init=1, random_state=km_seed).fit(Z).cluster_centers_
        labels = None

        for _ in range(max_iter):
            D2 = pairwise_distances(Z, centers, metric="sqeuclidean")
            # Repeat each center cap times. Hungarian then assigns each sample to one slot.
            C = np.repeat(D2, repeats=cap, axis=1)
            row_ind, col_ind = linear_sum_assignment(C)
            new_labels = col_ind // cap
            if labels is not None and np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            for k in range(K):
                mask = labels == k
                if np.any(mask):
                    centers[k] = Z[mask].mean(axis=0)

        obj = float(np.sum((Z - centers[labels]) ** 2))
        if obj < best_obj:
            best_obj = obj
            best_labels = labels.copy()

    return best_labels


def symnmf_objective(A, H):
    R = A - H @ H.T
    return 0.5 * float(np.sum(R * R))


def H_diagnostics(H, eps=1e-12):
    H = np.asarray(H, dtype=np.float64)
    n, K = H.shape
    row_sum = np.sum(H, axis=1, keepdims=True)
    P = H / np.maximum(row_sum, eps)
    entropy = -np.sum(P * np.log(np.maximum(P, eps)), axis=1) / np.log(K)
    sorted_vals = np.sort(P, axis=1)
    max1 = sorted_vals[:, -1]
    max2 = sorted_vals[:, -2] if K >= 2 else np.zeros(n)
    col_sums = np.sum(H, axis=0)
    return {
        "H_min": float(np.min(H)),
        "H_max": float(np.max(H)),
        "H_mean": float(np.mean(H)),
        "H_sparsity_1e_8": float(np.mean(H <= 1e-8)),
        "H_sparsity_1e_6": float(np.mean(H <= 1e-6)),
        "H_row_entropy_mean": float(np.mean(entropy)),
        "H_row_entropy_std": float(np.std(entropy)),
        "H_row_maxprob_mean": float(np.mean(max1)),
        "H_row_gap12_mean": float(np.mean(max1 - max2)),
        "H_colsum_min": float(np.min(col_sums)),
        "H_colsum_max": float(np.max(col_sums)),
        "H_colsum_cv": float(np.std(col_sums) / max(eps, np.mean(col_sums))),
    }


def make_H_feature(H, mode):
    """Build features from H. Supported modes are compositional."""
    H = np.asarray(H, dtype=np.float64)
    if mode == "H":
        return H
    if mode == "H_norm":
        return row_l2_norm(H)
    if mode == "H_l1":
        return row_l1_norm(H)
    if mode == "H_col_l2":
        return col_l2_norm(H)
    if mode == "H_colsum":
        return col_sum_norm(H)
    if mode == "H_col_l2_norm":
        return row_l2_norm(col_l2_norm(H))
    if mode == "H_colsum_norm":
        return row_l2_norm(col_sum_norm(H))
    if mode == "HHt_norm":
        return row_l2_norm(H @ H.T)
    if mode.startswith("H_pow"):
        # Examples: H_pow0.5, H_pow2, H_pow3
        power = float(mode.replace("H_pow", ""))
        return np.power(np.maximum(H, 0.0), power)
    if mode.startswith("H_norm_pow"):
        # Examples: H_norm_pow0.5, H_norm_pow2
        power = float(mode.replace("H_norm_pow", ""))
        return row_l2_norm(np.power(np.maximum(H, 0.0), power))
    if mode.startswith("H_l1_pow"):
        power = float(mode.replace("H_l1_pow", ""))
        return row_l1_norm(np.power(np.maximum(H, 0.0), power))
    raise ValueError(f"Unknown H feature mode {mode!r}")


def affinity_diagnostics(A, y, K, gapK):
    y = np.asarray(y)
    n = len(y)
    same = y[:, None] == y[None, :]
    off = ~np.eye(n, dtype=bool)
    within = A[same & off]
    between = A[(~same) & off]
    within_mean = float(within.mean()) if within.size else np.nan
    between_mean = float(between.mean()) if between.size else np.nan
    ratio = float(within_mean / max(between_mean, 1e-15))
    kk = min(10, n - 1)
    B = A.copy()
    np.fill_diagonal(B, -np.inf)
    nn = np.argpartition(-B, kth=kk - 1, axis=1)[:, :kk]
    prec = float(np.mean(y[nn] == y[:, None]))
    return {
        "within_mean": within_mean,
        "between_mean": between_mean,
        "within_between_ratio": ratio,
        "nn_precision_at_10": prec,
        "gapK": float(gapK),
        "normA2": float(np.linalg.norm(A, 2)),
        "normAF": float(np.linalg.norm(A, "fro")),
        "density_nonzero": float(np.count_nonzero(A) / A.size),
        "num_isolated": int(np.sum(A.sum(axis=1) <= 1e-15)),
    }


def maybe_save_plots(out_dir, prefix, H, y_true, labels, save_plots):
    if not save_plots:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable; skip plots: {e}")
        return

    out_dir = Path(out_dir)
    y_true = np.asarray(y_true)
    labels = np.asarray(labels)

    # H sorted by true labels.
    for order_name, order in [
        ("true", np.lexsort((np.arange(len(y_true)), y_true))),
        ("pred", np.lexsort((np.arange(len(labels)), labels))),
    ]:
        fig = plt.figure(figsize=(8, 7))
        plt.imshow(H[order], aspect="auto")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"H sorted by {order_name} labels")
        plt.xlabel("factor column")
        plt.ylabel("sample")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_H_sorted_by_{order_name}.png", dpi=180)
        plt.close(fig)

    cm = confusion_matrix(y_true, labels)
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(cm, aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Confusion matrix: true label x predicted cluster")
    plt.xlabel("predicted cluster")
    plt.ylabel("true label")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_confusion.png", dpi=180)
    plt.close(fig)


def run(args):
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_seeds = parse_csv(args.seeds, int)
    sym_seeds = parse_csv(args.symnmf_seeds, int)
    solvers = parse_csv(args.symnmf_solver_list, str)
    inits = parse_csv(args.symnmf_init_list, str)
    feature_modes = parse_csv(args.H_feature_list, str)
    rounding_methods = parse_csv(args.rounding_methods, str)

    all_rows = []

    for seed in dataset_seeds:
        Xdata, y, K = load_dataset(args.dataset, args.data_root, parse_image_size(args.image_size), args.max_per_class, seed)
        if args.K is not None:
            K = int(args.K)
        print(f"Loaded {args.dataset}: X={Xdata.shape}, K={K}, data_seed={seed}")

        A, Ainfo = build_full_gaussian_affinity(Xdata, args.sigma2_scale, args.zero_diagonal)
        U_spe, eigvals, gapK = spectral_embedding(A, K)
        spectral_labels = kmeans_labels(row_l2_norm(U_spe), K, seed)
        Adiag = affinity_diagnostics(A, y, K, gapK)
        Adiag.update(Ainfo)
        print(
            f"A=full_gaussian | zero_diag={args.zero_diagonal} | "
            f"||A||2={Adiag['normA2']:.6g} gapK={gapK:.6g} "
            f"within/between={Adiag['within_between_ratio']:.4g} NN@10={Adiag['nn_precision_at_10']:.4f}"
        )

        for solver in solvers:
            for init in inits:
                for sym_seed in sym_seeds:
                    real_seed = seed * 1000 + sym_seed
                    print(f"\n[SymNMF] solver={solver}, init={init}, sym_seed={sym_seed}, real_seed={real_seed}")
                    if solver == "mu":
                        H, hinfo = symnmf_mu(
                            A, K, max_iter=args.symnmf_max_iter, tol=args.symnmf_tol,
                            seed=real_seed, init=init, return_history=True
                        )
                    elif solver == "pgd":
                        H, hinfo = symnmf_pgd(
                            A, K, max_iter=args.symnmf_max_iter, lr=args.symnmf_lr,
                            tol=args.symnmf_tol, seed=real_seed, init=init, return_history=True
                        )
                    else:
                        raise ValueError(solver)

                    Hdiag = H_diagnostics(H)
                    obj_final = symnmf_objective(A, H)
                    prefix_base = f"seed{seed}_{solver}_{init}_symseed{sym_seed}"
                    if args.save_H:
                        np.save(out_dir / f"{prefix_base}_H.npy", H)
                        pd.DataFrame(H).to_csv(out_dir / f"{prefix_base}_H.csv", index=False)

                    for feat in feature_modes:
                        if feat == "argmax":
                            feature_description = "argmax_H"
                            labels_base = np.argmax(H, axis=1)
                            feature_for_balanced = H
                        else:
                            feature_description = feat
                            Z = make_H_feature(H, feat)
                            labels_base = None
                            feature_for_balanced = Z

                        for rounding in rounding_methods:
                            if feat == "argmax" and rounding != "argmax":
                                continue
                            if feat != "argmax" and rounding == "argmax":
                                continue

                            if rounding == "kmeans":
                                labels = kmeans_labels(feature_for_balanced, K, real_seed, n_init=args.kmeans_n_init)
                            elif rounding == "balanced":
                                labels = balanced_kmeans_labels(
                                    feature_for_balanced, K, real_seed,
                                    max_iter=args.balanced_max_iter, n_init=args.balanced_n_init
                                )
                            elif rounding == "argmax":
                                labels = labels_base
                            else:
                                raise ValueError(rounding)

                            met = evaluate(y, labels)
                            npc, mn, mx, counts = label_stats(labels)
                            row = {
                                "dataset_seed": seed,
                                "method": "SymNMF-H-Probe",
                                "affinity": f"full_gaussian_zeroDiag{args.zero_diagonal}_scale{args.sigma2_scale:g}",
                                "solver": solver,
                                "init": init,
                                "symnmf_seed": sym_seed,
                                "real_seed": real_seed,
                                "feature": feat,
                                "rounding": rounding,
                                "K": K,
                                "ACC": float(met["ACC"]),
                                "NMI": float(met["NMI"]),
                                "ARI": float(met["ARI"]),
                                "num_pred_clusters": npc,
                                "min_cluster_size": mn,
                                "max_cluster_size": mx,
                                "cluster_sizes": " ".join(str(int(c)) for c in counts),
                                "label_ari_vs_spectral": float(adjusted_rand_score(spectral_labels, labels)),
                                "symnmf_obj_final": float(obj_final),
                                "symnmf_n_iter": int(hinfo.get("n_iter", len(hinfo.get("objective", [])) - 1)),
                                **Adiag,
                                **Hdiag,
                            }
                            all_rows.append(row)
                            print(
                                f"  {feat:14s} | {rounding:8s} | ACC={row['ACC']:.4f} "
                                f"NMI={row['NMI']:.4f} ARI={row['ARI']:.4f} "
                                f"size=[{mn},{mx}] ARIvsSpe={row['label_ari_vs_spectral']:.4f}"
                            )

                            if args.save_labels:
                                pd.DataFrame({
                                    "y_true": y,
                                    "label_pred": labels,
                                }).to_csv(out_dir / f"{prefix_base}_{feat}_{rounding}_labels.csv", index=False)

                            if args.save_plots and feat in {"H", "H_norm", "H_l1"} and rounding in {"kmeans", "balanced", "argmax"}:
                                maybe_save_plots(out_dir, f"{prefix_base}_{feat}_{rounding}", H, y, labels, True)

    df = pd.DataFrame(all_rows)
    raw_path = out_dir / "symnmf_H_probe_raw.csv"
    df.to_csv(raw_path, index=False)

    group_cols = ["method", "affinity", "solver", "init", "feature", "rounding"]
    metric_cols = [c for c in [
        "ACC", "NMI", "ARI", "num_pred_clusters", "min_cluster_size", "max_cluster_size",
        "label_ari_vs_spectral", "symnmf_obj_final", "symnmf_n_iter",
        "H_sparsity_1e_8", "H_sparsity_1e_6", "H_row_entropy_mean", "H_row_maxprob_mean",
        "H_row_gap12_mean", "H_colsum_cv", "within_between_ratio", "nn_precision_at_10",
        "gapK", "normA2", "normAF",
    ] if c in df.columns]

    summary = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if str(x) != ""]).rstrip("_") for col in summary.columns.values]
    summary = summary.sort_values("ACC_mean", ascending=False)

    summary_path = out_dir / "symnmf_H_probe_summary.csv"
    best_path = out_dir / "symnmf_H_probe_best.csv"
    summary.to_csv(summary_path, index=False)
    summary.head(100).to_csv(best_path, index=False)

    if args.save_xlsx:
        with pd.ExcelWriter(out_dir / "symnmf_H_probe_summary.xlsx") as writer:
            df.to_excel(writer, sheet_name="raw", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
            summary.head(100).to_excel(writer, sheet_name="best", index=False)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("\nTop results:")
    show_cols = [c for c in [
        "solver", "init", "feature", "rounding", "ACC_mean", "NMI_mean", "ARI_mean",
        "num_pred_clusters_mean", "min_cluster_size_mean", "max_cluster_size_mean",
        "label_ari_vs_spectral_mean", "symnmf_obj_final_mean", "H_row_entropy_mean_mean",
        "H_row_gap12_mean_mean", "H_colsum_cv_mean",
    ] if c in summary.columns]
    print(summary[show_cols].head(40).to_string(index=False))
    print(f"\nSaved: {raw_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {best_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="coil20")
    p.add_argument("--data-root", default="datasets/data/coil20")
    p.add_argument("--image-size", default="original")
    p.add_argument("--max-per-class", type=int, default=30)
    p.add_argument("--seeds", default="42")
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--sigma2-scale", type=float, default=1.0)
    p.add_argument("--zero-diagonal", type=parse_bool, default=False)

    p.add_argument("--symnmf-solver-list", default="mu")
    p.add_argument("--symnmf-init-list", default="random,nndsvd_spectral")
    p.add_argument("--symnmf-seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--symnmf-max-iter", type=int, default=1000)
    p.add_argument("--symnmf-tol", type=float, default=1e-5)
    p.add_argument("--symnmf-lr", type=float, default=1e-3)

    p.add_argument(
        "--H-feature-list",
        default="H,H_norm,H_l1,H_col_l2_norm,H_colsum_norm,H_norm_pow0.5,H_norm_pow2,H_norm_pow3,HHt_norm,argmax",
    )
    p.add_argument("--rounding-methods", default="kmeans,balanced,argmax")
    p.add_argument("--kmeans-n-init", type=int, default=50)
    p.add_argument("--balanced-max-iter", type=int, default=30)
    p.add_argument("--balanced-n-init", type=int, default=10)

    p.add_argument("--out-dir", default="results/coil20_symnmf_H_probe")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save-xlsx", action="store_true")
    p.add_argument("--save-H", action="store_true")
    p.add_argument("--save-labels", action="store_true")
    p.add_argument("--save-plots", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
