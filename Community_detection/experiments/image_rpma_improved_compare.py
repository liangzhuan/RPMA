"""Compare Spectral Projection, RPMA, and three RPMA improvements.

Place at::

    Community_detection/experiments/image_rpma_improved_compare.py

Methods
-------
``spectral``
    Leading-K spectral projection.
``rpma``
    The project's original all-entry Huber RPMA implementation.
``rpma_c``
    Same final objective as RPMA, but solved by continuation, Riemannian
    conjugate-gradient directions, step reuse, and nonmonotone Armijo search.
``rpma_od``
    Huber regularization only on off-diagonal entries.
``ns_rpma``
    Off-diagonal Huber + exact X1=1 + soft nonnegativity.

All methods receive exactly the same dense Gaussian affinity matrix and use the
same rounding/evaluation code.  No label information is used by the methods.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.metrics import pairwise_distances

from experiments.image_bounded_sparse_rpma import (
    evaluate,
    load_dataset,
    parse_int_list,
)
from methods.rpa import rpa as ordinary_rpa, objective as ordinary_rpma_objective
from methods.rpma_continuation import rpma_continuation
from methods.rpma_offdiag import rpma_offdiag
from methods.ns_rpma import ns_rpma


ALLOWED_METHODS = {"spectral", "rpma", "rpma_c", "rpma_od", "ns_rpma"}


def dense_gaussian_affinity(
    X: np.ndarray,
    bandwidth_scale: float,
    *,
    keep_diagonal: bool,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Construct the same dense Gaussian graph used in the bandwidth script."""
    t0 = time.perf_counter()
    D2 = pairwise_distances(X, metric="sqeuclidean", n_jobs=1)
    D2 = np.maximum(D2, 0.0)
    n = D2.shape[0]
    upper_d2 = D2[np.triu_indices(n, k=1)]
    distances = np.sqrt(upper_d2)
    positive = distances[distances > 0.0]
    if positive.size == 0:
        raise ValueError("All pairwise distances are zero.")

    sigma0 = float(np.mean(positive))
    scale = float(bandwidth_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("bandwidth_scale must be finite and positive.")
    sigma = scale * sigma0
    denominator = 2.0 * sigma * sigma
    A = np.exp(-D2 / denominator)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 1.0 if keep_diagonal else 0.0)
    offdiag = A[np.triu_indices(n, k=1)]

    return A, {
        "bandwidth_scale": scale,
        "sigma0_mean_distance": sigma0,
        "sigma": sigma,
        "kernel_denominator": denominator,
        "affinity_offdiag_mean": float(np.mean(offdiag)),
        "affinity_offdiag_median": float(np.median(offdiag)),
        "affinity_offdiag_min": float(np.min(offdiag)),
        "affinity_offdiag_max": float(np.max(offdiag)),
        "affinity_density": float(np.mean(A > 1e-12)),
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


def add_seed_evaluations(
    rows: List[Dict],
    *,
    method: str,
    U: np.ndarray,
    y: np.ndarray,
    K: int,
    seeds: List[int],
    args: argparse.Namespace,
    elapsed: float,
    common: Dict[str, object],
    extra: Dict[str, object],
) -> None:
    for seed in seeds:
        row = evaluate(
            method,
            U,
            y,
            K,
            seed,
            args.rounding,
            args.kmeans_n_init,
            elapsed,
            extra=extra,
        )
        row.update(common)
        rows.append(row)


def method_failure_row(
    method: str,
    elapsed: float,
    common: Dict[str, object],
    exc: Exception,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "method": method,
        "time_sec": elapsed,
        "error": repr(exc),
    }
    row.update(common)
    return row


def run(args: argparse.Namespace) -> None:
    unknown = set(args.methods) - ALLOWED_METHODS
    if unknown:
        raise ValueError(
            f"Unknown methods {sorted(unknown)}. Allowed: {sorted(ALLOWED_METHODS)}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features, y, K = load_dataset(args)
    features = np.asarray(features, dtype=np.float64)
    y = np.asarray(y)
    n = features.shape[0]
    seeds = parse_int_list(args.seeds)

    print("=" * 108)
    print("RPMA improvement comparison")
    print(
        f"dataset={args.dataset}, n={n}, K={K}, n_k={n / K:.6g}, "
        f"features={features.shape[1]}"
    )
    print(
        "A_ij = exp(-||xi-xj||^2 / (2*(scale*mean_pairwise_distance)^2))"
    )
    print(f"bandwidth_scale={args.bandwidth_scale:g}")
    print(f"methods={sorted(args.methods)}")
    rpma_lam = args.lam if args.rpma_lam is None else args.rpma_lam
    rpma_c_lam = args.lam if args.rpma_c_lam is None else args.rpma_c_lam
    rpma_od_lam = args.lam if args.rpma_od_lam is None else args.rpma_od_lam
    ns_lam = args.lam if args.ns_lam is None else args.ns_lam
    print(
        f"lambdas: RPMA={rpma_lam:g}, RPMA-C={rpma_c_lam:g}, "
        f"RPMA-OD={rpma_od_lam:g}, NS-RPMA={ns_lam:g}"
    )
    print(
        f"delta={args.delta:g}, NS nonnegative_mu={args.ns_mu:g}"
    )
    print("=" * 108)

    A, affinity_info = dense_gaussian_affinity(
        features,
        args.bandwidth_scale,
        keep_diagonal=args.keep_diagonal,
    )
    print(
        f"A ready: sigma0={affinity_info['sigma0_mean_distance']:.8e}, "
        f"sigma={affinity_info['sigma']:.8e}, "
        f"offdiag_mean={affinity_info['affinity_offdiag_mean']:.6f}, "
        f"time={affinity_info['affinity_time_sec']:.2f}s"
    )

    t0 = time.perf_counter()
    U_spec = leading_basis(A, K)
    spectral_time = time.perf_counter() - t0
    X_spec = U_spec @ U_spec.T

    common: Dict[str, object] = dict(affinity_info)
    common.update(
        {
            "dataset": args.dataset,
            "n": n,
            "K": K,
            "n_k": n / K,
            "feature_dim": features.shape[1],
            "keep_diagonal": bool(args.keep_diagonal),
        }
    )

    rows: List[Dict] = []

    if "spectral" in args.methods:
        add_seed_evaluations(
            rows,
            method="Spectral-Projection",
            U=U_spec,
            y=y,
            K=K,
            seeds=seeds,
            args=args,
            elapsed=spectral_time,
            common=common,
            extra={
                "lam": np.nan,
                "delta": np.nan,
                "nonnegative_mu": 0.0,
                "n_iter": 0,
                "converged": True,
                "line_search_failed": False,
                "final_grad_norm": np.nan,
                "final_objective": float(-2.0 * np.sum(A * X_spec)),
                "row_sum_residual": float(
                    np.linalg.norm(X_spec @ np.ones(n) - np.ones(n))
                ),
                "negative_violation_fro": float(
                    np.linalg.norm(np.maximum(-X_spec, 0.0), ord="fro")
                ),
            },
        )

    if "rpma" in args.methods:
        print("\n[run] ordinary RPMA")
        t0 = time.perf_counter()
        try:
            X_rpma, U_rpma, history = ordinary_rpa(
                A,
                K,
                lam=rpma_lam,
                delta=args.delta,
                tau_max=args.tau_max,
                beta=args.backtrack_beta,
                sigma=args.armijo_sigma,
                tol=args.tol,
                max_iter=args.max_iter,
                eig_init=False,
                U0=U_spec,
                return_history=True,
                verbose=args.verbose,
            )
            elapsed = time.perf_counter() - t0
            final_grad = float(history[-1]) if history else 0.0
            add_seed_evaluations(
                rows,
                method="RPMA-Huber",
                U=U_rpma,
                y=y,
                K=K,
                seeds=seeds,
                args=args,
                elapsed=elapsed,
                common=common,
                extra={
                    "lam": rpma_lam,
                    "delta": args.delta,
                    "nonnegative_mu": 0.0,
                    "n_iter": len(history),
                    "converged": bool(final_grad <= args.tol),
                    "line_search_failed": np.nan,
                    "final_grad_norm": final_grad,
                    "final_objective": float(
                        ordinary_rpma_objective(
                            X_rpma, A, rpma_lam, args.delta
                        )
                    ),
                    "row_sum_residual": float(
                        np.linalg.norm(X_rpma @ np.ones(n) - np.ones(n))
                    ),
                    "negative_violation_fro": float(
                        np.linalg.norm(np.maximum(-X_rpma, 0.0), ord="fro")
                    ),
                },
            )
            print(
                f"  done: iter={len(history)}, grad={final_grad:.3e}, "
                f"time={elapsed:.2f}s"
            )
            if args.save_matrices:
                np.savez_compressed(
                    out_dir / "rpma.npz", A=A, X=X_rpma, U=U_rpma, y=y, K=K
                )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            rows.append(method_failure_row("RPMA-Huber", elapsed, common, exc))
            print(f"  failed: {exc!r}")

    if "rpma_c" in args.methods:
        print("\n[run] RPMA-C")
        t0 = time.perf_counter()
        try:
            X_c, U_c, info = rpma_continuation(
                A,
                K,
                lam=rpma_c_lam,
                delta=args.delta,
                start_delta=args.start_delta,
                continuation_steps=args.continuation_steps,
                max_iter_per_stage=args.stage_max_iter,
                U0=U_spec,
                tol=args.tol,
                tau_max=args.tau_max,
                backtrack_beta=args.backtrack_beta,
                armijo_sigma=args.armijo_sigma,
                nonmonotone_window=args.nonmonotone_window,
                n_starts=args.rpma_c_n_starts,
                perturb_scale=args.perturb_scale,
                random_state=args.data_seed,
                verbose=args.verbose,
                return_info=True,
            )
            elapsed = time.perf_counter() - t0
            add_seed_evaluations(
                rows,
                method="RPMA-C",
                U=U_c,
                y=y,
                K=K,
                seeds=seeds,
                args=args,
                elapsed=elapsed,
                common=common,
                extra={
                    "lam": rpma_c_lam,
                    "delta": args.delta,
                    "nonnegative_mu": 0.0,
                    "n_iter": info["n_iter"],
                    "converged": info["converged"],
                    "line_search_failed": info["line_search_failed"],
                    "final_grad_norm": info["final_grad_norm"],
                    "final_objective": info["final_objective"],
                    "cg_restart_count": info["cg_restart_count"],
                    "continuation_steps_completed": info[
                        "continuation_steps_completed"
                    ],
                    "selected_start": info["selected_start"],
                    "row_sum_residual": float(
                        np.linalg.norm(X_c @ np.ones(n) - np.ones(n))
                    ),
                    "negative_violation_fro": float(
                        np.linalg.norm(np.maximum(-X_c, 0.0), ord="fro")
                    ),
                },
            )
            print(
                f"  done: iter={info['n_iter']}, "
                f"grad={info['final_grad_norm']:.3e}, "
                f"stages={info['continuation_steps_completed']}, "
                f"time={elapsed:.2f}s"
            )
            if args.save_matrices:
                np.savez_compressed(
                    out_dir / "rpma_c.npz", A=A, X=X_c, U=U_c, y=y, K=K
                )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            rows.append(method_failure_row("RPMA-C", elapsed, common, exc))
            print(f"  failed: {exc!r}")

    if "rpma_od" in args.methods:
        print("\n[run] RPMA-OD")
        t0 = time.perf_counter()
        try:
            X_od, U_od, info = rpma_offdiag(
                A,
                K,
                lam=rpma_od_lam,
                delta=args.delta,
                U0=U_spec,
                max_iter=args.max_iter,
                tol=args.tol,
                tau_max=args.tau_max,
                backtrack_beta=args.backtrack_beta,
                armijo_sigma=args.armijo_sigma,
                verbose=args.verbose,
                return_info=True,
            )
            elapsed = time.perf_counter() - t0
            add_seed_evaluations(
                rows,
                method="RPMA-OD",
                U=U_od,
                y=y,
                K=K,
                seeds=seeds,
                args=args,
                elapsed=elapsed,
                common=common,
                extra={
                    "lam": rpma_od_lam,
                    "delta": args.delta,
                    "nonnegative_mu": 0.0,
                    "n_iter": info["n_iter"],
                    "converged": info["converged"],
                    "line_search_failed": info["line_search_failed"],
                    "final_grad_norm": info["final_grad_norm"],
                    "final_objective": info["final_objective"],
                    "row_sum_residual": float(
                        np.linalg.norm(X_od @ np.ones(n) - np.ones(n))
                    ),
                    "negative_violation_fro": float(
                        np.linalg.norm(np.maximum(-X_od, 0.0), ord="fro")
                    ),
                },
            )
            print(
                f"  done: iter={info['n_iter']}, "
                f"grad={info['final_grad_norm']:.3e}, time={elapsed:.2f}s"
            )
            if args.save_matrices:
                np.savez_compressed(
                    out_dir / "rpma_od.npz", A=A, X=X_od, U=U_od, y=y, K=K
                )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            rows.append(method_failure_row("RPMA-OD", elapsed, common, exc))
            print(f"  failed: {exc!r}")

    if "ns_rpma" in args.methods:
        print("\n[run] NS-RPMA")
        t0 = time.perf_counter()
        try:
            X_ns, U_ns, info = ns_rpma(
                A,
                K,
                lam=ns_lam,
                delta=args.delta,
                nonnegative_mu=args.ns_mu,
                start_delta=args.start_delta,
                continuation_steps=args.continuation_steps,
                max_iter_per_stage=args.stage_max_iter,
                U0=U_spec,
                tol=args.tol,
                tau_max=args.tau_max,
                backtrack_beta=args.backtrack_beta,
                armijo_sigma=args.armijo_sigma,
                nonmonotone_window=args.nonmonotone_window,
                verbose=args.verbose,
                return_info=True,
            )
            elapsed = time.perf_counter() - t0
            add_seed_evaluations(
                rows,
                method="NS-RPMA",
                U=U_ns,
                y=y,
                K=K,
                seeds=seeds,
                args=args,
                elapsed=elapsed,
                common=common,
                extra={
                    "lam": ns_lam,
                    "delta": args.delta,
                    "nonnegative_mu": args.ns_mu,
                    "n_iter": info["n_iter"],
                    "converged": info["converged"],
                    "line_search_failed": info["line_search_failed"],
                    "final_grad_norm": info["final_grad_norm"],
                    "final_objective": info["final_objective"],
                    "cg_restart_count": info["cg_restart_count"],
                    "continuation_steps_completed": info[
                        "continuation_steps_completed"
                    ],
                    "row_sum_residual": info["row_sum_residual"],
                    "negative_violation_fro": info["negative_violation_fro"],
                    "negative_entry_ratio": info["negative_entry_ratio"],
                },
            )
            print(
                f"  done: iter={info['n_iter']}, "
                f"grad={info['final_grad_norm']:.3e}, "
                f"row_v={info['row_sum_residual']:.3e}, "
                f"neg_v={info['negative_violation_fro']:.3e}, "
                f"time={elapsed:.2f}s"
            )
            if args.save_matrices:
                np.savez_compressed(
                    out_dir / "ns_rpma.npz", A=A, X=X_ns, U=U_ns, y=y, K=K
                )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            rows.append(method_failure_row("NS-RPMA", elapsed, common, exc))
            print(f"  failed: {exc!r}")

    df = pd.DataFrame(rows)
    if "error" not in df.columns:
        df["error"] = ""
    df["error"] = df["error"].fillna("")
    results_path = out_dir / "rpma_improved_results.csv"
    df.to_csv(results_path, index=False, encoding="utf-8-sig")

    valid = df[df["error"] == ""].copy()
    if valid.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            valid.groupby(
                [
                    "method",
                    "bandwidth_scale",
                    "lam",
                    "delta",
                    "nonnegative_mu",
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
                final_grad_norm_mean=("final_grad_norm", "mean"),
                row_sum_residual_mean=("row_sum_residual", "mean"),
                negative_violation_fro_mean=("negative_violation_fro", "mean"),
                time_sec_mean=("time_sec", "mean"),
                n_runs=("seed", "count"),
            )
            .reset_index()
            .sort_values(["ACC_mean", "NMI_mean", "ARI_mean"], ascending=False)
        )
    summary_path = out_dir / "rpma_improved_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    config = vars(args).copy()
    if isinstance(config.get("methods"), set):
        config["methods"] = sorted(config["methods"])
    config.update({"n": n, "K": K, "feature_dim": features.shape[1]})
    (out_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nBest valid runs:")
    if summary.empty:
        print("No valid result.")
    else:
        columns = [
            "method",
            "ACC_mean",
            "NMI_mean",
            "ARI_mean",
            "final_grad_norm_mean",
            "row_sum_residual_mean",
            "negative_violation_fro_mean",
            "projection_distance_mean",
            "time_sec_mean",
        ]
        print(summary[[c for c in columns if c in summary.columns]].to_string(index=False))

    print(f"\nSaved: {results_path}")
    print(f"Saved: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Spectral, RPMA, RPMA-C, RPMA-OD, and NS-RPMA."
    )
    parser.add_argument(
        "--dataset",
        choices=["att_faces", "coil20", "yaleB"],
        default="coil20",
    )
    parser.add_argument("--data-root", default="datasets/data/coil20")
    parser.add_argument("--image-size", default="original")
    parser.add_argument("--max-per-class", type=int, default=10)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        type=lambda text: {x.strip() for x in text.split(",") if x.strip()},
        default=set(ALLOWED_METHODS),
        help="Comma-separated: spectral,rpma,rpma_c,rpma_od,ns_rpma",
    )

    parser.add_argument("--bandwidth-scale", type=float, default=0.5)
    parser.add_argument(
        "--lam",
        type=float,
        default=0.005,
        help="Fallback lambda used when a method-specific lambda is omitted.",
    )
    parser.add_argument("--rpma-lam", type=float, default=None)
    parser.add_argument("--rpma-c-lam", type=float, default=None)
    parser.add_argument("--rpma-od-lam", type=float, default=None)
    parser.add_argument("--ns-lam", type=float, default=None)
    parser.add_argument("--delta", type=float, default=0.001)
    parser.add_argument("--ns-mu", type=float, default=1.0)

    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--continuation-steps", type=int, default=4)
    parser.add_argument("--stage-max-iter", type=int, default=150)
    parser.add_argument("--start-delta", type=float, default=0.01)
    parser.add_argument("--rpma-c-n-starts", type=int, default=1)
    parser.add_argument("--perturb-scale", type=float, default=0.02)
    parser.add_argument("--nonmonotone-window", type=int, default=5)

    parser.add_argument("--tau-max", type=float, default=1.0)
    parser.add_argument("--backtrack-beta", type=float, default=0.5)
    parser.add_argument("--armijo-sigma", type=float, default=1e-4)
    parser.add_argument("--tol", type=float, default=1e-5)

    parser.add_argument("--seeds", default="42")
    parser.add_argument("--rounding", choices=["kmeans", "balanced"], default="kmeans")
    parser.add_argument("--kmeans-n-init", type=int, default=50)

    parser.set_defaults(keep_diagonal=True)
    parser.add_argument("--keep-diagonal", action="store_true", dest="keep_diagonal")
    parser.add_argument("--zero-diagonal", action="store_false", dest="keep_diagonal")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-matrices", action="store_true")
    parser.add_argument("--out-dir", default="results/coil20_rpma_improved_compare")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
