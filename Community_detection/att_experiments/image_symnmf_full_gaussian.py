"""
Compare Spectral / RPMA / SymNMF on the original full Gaussian affinity A.

A_ij = exp(-||x_i-x_j||^2 / sigma^2),
where sigma^2 is the average squared pairwise distance.

This script does not use kNN sparsification and does not modify methods/rpa.py.
"""

SCRIPT_VERSION = "2026-07-08-image-symnmf-full-gaussian-v1"

import argparse
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import pairwise_distances

from evaluation.metrics import evaluate
from methods.rpa import rpa
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
    if dataset.lower() == "coil20":
        from datasets.image_datasets import load_coil20
        out = _call_loader(load_coil20, data_root, image_size, max_per_class, seed)
    elif dataset.lower() in {"yaleb", "extended_yale_b", "extended-yale-b"}:
        from datasets.image_datasets import load_extended_yale_b
        out = _call_loader(load_extended_yale_b, data_root, image_size, max_per_class, seed)
    elif dataset.lower() in {"att_faces", "attfaces", "orl"}:
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
    return A, {"sigma2": sigma2, "density_nonzero": float(np.count_nonzero(A) / A.size), "num_isolated": int(np.sum(A.sum(axis=1) <= 1e-15))}


def spectral_embedding(A, K):
    vals, vecs = np.linalg.eigh(0.5 * (A + A.T))
    idx = np.argsort(vals)[::-1]
    U = vecs[:, idx[:K]]
    eigvals = vals[idx]
    gap = float(eigvals[K - 1] - eigvals[K]) if len(eigvals) > K else np.nan
    return U, eigvals, gap


def row_norm(Z, eps=1e-12):
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), eps)


def embedding_from_UX(U, Xproj, mode):
    if mode == "U":
        return U
    if mode == "U_norm":
        return row_norm(U)
    if mode == "X":
        return Xproj
    if mode == "X_norm":
        return row_norm(Xproj)
    raise ValueError(mode)


def kmeans_labels(Z, K, seed):
    return KMeans(n_clusters=K, n_init=20, random_state=seed).fit_predict(Z)


def label_stats(labels):
    _, c = np.unique(labels, return_counts=True)
    return int(len(c)), int(c.min()), int(c.max())


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
    }


def run_one_seed(args, seed):
    X, y, K = load_dataset(args.dataset, args.data_root, parse_image_size(args.image_size), args.max_per_class, seed)
    print(f"Loaded {args.dataset}: X={X.shape}, K={K}, seed={seed}")

    A, Ainfo = build_full_gaussian_affinity(X, sigma2_scale=args.sigma2_scale, zero_diagonal=args.zero_diagonal)
    U_spe, eigvals, gapK = spectral_embedding(A, K)
    X_spe = U_spe @ U_spe.T
    diag = affinity_diagnostics(A, y, K, gapK)
    diag.update(Ainfo)
    affinity_name = f"full_gaussian_zeroDiag{args.zero_diagonal}_scale{args.sigma2_scale:g}"
    print(
        f"A={affinity_name} | ||A||2={diag['normA2']:.6g}, "
        f"within/between={diag['within_between_ratio']:.4g}, "
        f"NN@10={diag['nn_precision_at_10']:.4f}, gapK={diag['gapK']:.6g}"
    )

    rows = []
    methods = parse_csv(args.methods, str)
    embeddings = parse_csv(args.embedding_list, str)
    sym_features = parse_csv(args.symnmf_feature_list, str)
    sym_inits = parse_csv(args.symnmf_init_list, str)
    spectral_ref_labels = kmeans_labels(row_norm(U_spe), K, seed)

    if "spectral" in methods:
        for emb in embeddings:
            Z = embedding_from_UX(U_spe, X_spe, emb)
            labels = kmeans_labels(Z, K, seed)
            met = evaluate(y, labels)
            npc, mn, mx = label_stats(labels)
            rows.append({
                "seed": seed, "method": "Spectral-Projection", "affinity": affinity_name,
                "embedding": emb, "symnmf_solver": "", "symnmf_init": "", "symnmf_seed": np.nan,
                "lam": np.nan, "delta": np.nan,
                "ACC": float(met["ACC"]), "NMI": float(met["NMI"]), "ARI": float(met["ARI"]),
                "num_pred_clusters": npc, "min_cluster_size": mn, "max_cluster_size": mx,
                "label_ari_vs_spectral": float(adjusted_rand_score(spectral_ref_labels, labels)),
                **diag,
            })
            print(f"Spectral | {emb} | ACC={met['ACC']:.4f}, NMI={met['NMI']:.4f}, ARI={met['ARI']:.4f}")

    if "rpma" in methods:
        for lam in parse_csv(args.lam_list, float):
            for delta in parse_csv(args.delta_list, float):
                out = rpa(A, K, lam=lam, delta=delta, max_iter=args.rpma_max_iter, eig_init=True, return_history=True)
                if isinstance(out, tuple) and len(out) == 3:
                    X_r, U_r, hist = out
                else:
                    X_r = out
                    U_r, _, _ = spectral_embedding(X_r, K)
                    hist = []
                proj_fro = float(np.linalg.norm(X_r - X_spe, "fro"))
                for emb in embeddings:
                    Z = embedding_from_UX(U_r, X_r, emb)
                    labels = kmeans_labels(Z, K, seed)
                    met = evaluate(y, labels)
                    npc, mn, mx = label_stats(labels)
                    rows.append({
                        "seed": seed, "method": "RPMA-Huber", "affinity": affinity_name,
                        "embedding": emb, "symnmf_solver": "", "symnmf_init": "", "symnmf_seed": np.nan,
                        "lam": lam, "delta": delta,
                        "ACC": float(met["ACC"]), "NMI": float(met["NMI"]), "ARI": float(met["ARI"]),
                        "num_pred_clusters": npc, "min_cluster_size": mn, "max_cluster_size": mx,
                        "label_ari_vs_spectral": float(adjusted_rand_score(spectral_ref_labels, labels)),
                        "proj_fro_to_spectral": proj_fro,
                        "rpma_n_iter": len(hist) if hasattr(hist, "__len__") else np.nan,
                        **diag,
                    })
                    print(f"RPMA | lam={lam:g} | {emb} | ACC={met['ACC']:.4f}, proj_fro={proj_fro:.3e}")

    if "symnmf" in methods:
        for solver in parse_csv(args.symnmf_solver_list, str):
            for init in sym_inits:
                for sym_seed in parse_csv(args.symnmf_seeds, int):
                    real_seed = seed * 1000 + sym_seed
                    if solver == "mu":
                        H, hinfo = symnmf_mu(A, K, max_iter=args.symnmf_max_iter, tol=args.symnmf_tol, seed=real_seed, init=init, return_history=True)
                    elif solver == "pgd":
                        H, hinfo = symnmf_pgd(A, K, max_iter=args.symnmf_max_iter, lr=args.symnmf_lr, tol=args.symnmf_tol, seed=real_seed, init=init, return_history=True)
                    else:
                        raise ValueError(solver)
                    for feat in sym_features:
                        if feat == "argmax":
                            labels = np.argmax(H, axis=1)
                        else:
                            Z = symnmf_cluster_features(H, feat)
                            labels = kmeans_labels(Z, K, real_seed)
                        met = evaluate(y, labels)
                        npc, mn, mx = label_stats(labels)
                        rows.append({
                            "seed": seed, "method": "SymNMF", "affinity": affinity_name,
                            "embedding": feat, "symnmf_solver": solver, "symnmf_init": init, "symnmf_seed": sym_seed,
                            "symnmf_obj_final": float(hinfo["objective"][-1]),
                            "symnmf_n_iter": int(hinfo["n_iter"]),
                            "lam": np.nan, "delta": np.nan,
                            "ACC": float(met["ACC"]), "NMI": float(met["NMI"]), "ARI": float(met["ARI"]),
                            "num_pred_clusters": npc, "min_cluster_size": mn, "max_cluster_size": mx,
                            "label_ari_vs_spectral": float(adjusted_rand_score(spectral_ref_labels, labels)),
                            **diag,
                        })
                        print(f"SymNMF | {solver}/{init}/seed{sym_seed} | {feat} | ACC={met['ACC']:.4f}, obj={hinfo['objective'][-1]:.4e}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="coil20")
    p.add_argument("--data-root", default="datasets/data/coil20")
    p.add_argument("--image-size", default="original")
    p.add_argument("--max-per-class", type=int, default=30)
    p.add_argument("--seeds", default="42")
    p.add_argument("--sigma2-scale", type=float, default=1.0)
    p.add_argument("--zero-diagonal", type=parse_bool, default=False)
    p.add_argument("--methods", default="spectral,symnmf")
    p.add_argument("--embedding-list", default="U_norm,X_norm")
    p.add_argument("--lam-list", default="0,0.0005,0.001,0.005,0.01")
    p.add_argument("--delta-list", default="0.001")
    p.add_argument("--rpma-max-iter", type=int, default=500)
    p.add_argument("--symnmf-solver-list", default="mu")
    p.add_argument("--symnmf-init-list", default="random,nndsvd_spectral")
    p.add_argument("--symnmf-feature-list", default="H_norm,H,argmax,HHt_norm")
    p.add_argument("--symnmf-seeds", default="0,1,2,3,4")
    p.add_argument("--symnmf-max-iter", type=int, default=1000)
    p.add_argument("--symnmf-tol", type=float, default=1e-5)
    p.add_argument("--symnmf-lr", type=float, default=1e-3)
    p.add_argument("--out-dir", default="results/coil20_symnmf_full_gaussian")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save-xlsx", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed in parse_csv(args.seeds, int):
        all_rows.extend(run_one_seed(args, seed))

    df = pd.DataFrame(all_rows)
    raw_path = out_dir / "symnmf_full_gaussian_raw.csv"
    df.to_csv(raw_path, index=False)

    group_cols = ["method", "affinity", "embedding", "symnmf_solver", "symnmf_init", "lam", "delta"]
    metric_cols = [c for c in [
        "ACC", "NMI", "ARI", "num_pred_clusters", "min_cluster_size", "max_cluster_size",
        "label_ari_vs_spectral", "proj_fro_to_spectral", "symnmf_obj_final", "symnmf_n_iter",
        "within_between_ratio", "nn_precision_at_10", "gapK", "density_nonzero", "num_isolated",
        "normA2", "normAF",
    ] if c in df.columns]
    summary = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if str(x) != ""]).rstrip("_") for col in summary.columns.values]
    summary = summary.sort_values("ACC_mean", ascending=False)

    summary_path = out_dir / "symnmf_full_gaussian_summary.csv"
    best_path = out_dir / "symnmf_full_gaussian_best.csv"
    summary.to_csv(summary_path, index=False)
    summary.head(100).to_csv(best_path, index=False)

    if args.save_xlsx:
        with pd.ExcelWriter(out_dir / "symnmf_full_gaussian_summary.xlsx") as writer:
            df.to_excel(writer, sheet_name="raw", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
            summary.head(100).to_excel(writer, sheet_name="best", index=False)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("\nTop results:")
    cols = [c for c in [
        "method", "embedding", "symnmf_solver", "symnmf_init", "lam", "ACC_mean", "NMI_mean", "ARI_mean",
        "num_pred_clusters_mean", "min_cluster_size_mean", "max_cluster_size_mean",
        "label_ari_vs_spectral_mean", "symnmf_obj_final_mean", "gapK_mean", "normA2_mean",
    ] if c in summary.columns]
    print(summary[cols].head(30).to_string(index=False))
    print(f"\nSaved: {raw_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()
