"""
Run image clustering experiments on COIL-20 or Extended Yale B.

Examples:
  python -m experiments.run_image_experiment --dataset coil20 --data-root datasets/data/coil20
  python -m experiments.run_image_experiment --dataset yaleB --data-root datasets/data/CroppedYale

The script expects you to place the raw dataset locally. It does not download
COIL-20 or Extended Yale B automatically.
"""
import argparse
import json
import os
import sys

# Allow running as a script from project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from datasets.image_datasets import load_coil20, load_extended_yale_b
from experiments.image_compare import run_image_clustering


def parse_size(s):
    if s.lower() in {"none", "original"}:
        return None
    if "x" in s:
        a, b = s.lower().split("x")
        return int(a), int(b)
    v = int(s)
    return v, v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["coil20", "yaleB"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image-size", default="32x32")
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="0 means use all images. Use e.g. 20 for a quick smoke test.")
    parser.add_argument("--pca-dim", type=int, default=100)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--lam", type=float, default=0.02)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--rpa-max-iter", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--out-dir", default="results/image")
    args = parser.parse_args()

    image_size = parse_size(args.image_size)
    max_per_class = None if args.max_per_class <= 0 else args.max_per_class

    if args.dataset == "coil20":
        X, y, K = load_coil20(args.data_root, image_size=image_size,
                              max_per_class=max_per_class, random_state=args.random_state)
    else:
        X, y, K = load_extended_yale_b(args.data_root, image_size=image_size,
                                       max_per_class=max_per_class, random_state=args.random_state)

    print(f"Loaded {args.dataset}: X={X.shape}, classes={K}, n={len(y)}")

    results, A, X_rpa = run_image_clustering(
        X,
        y,
        K,
        pca_dim=args.pca_dim,
        k_neighbors=args.k_neighbors,
        lam=args.lam,
        delta=args.delta,
        rpa_max_iter=args.rpa_max_iter,
        random_state=args.random_state,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = f"{args.dataset}_n{len(y)}_k{K}"

    df = pd.DataFrame(results).T
    print(df)
    df.to_csv(os.path.join(args.out_dir, f"{prefix}_metrics.csv"))

    np.savez_compressed(
        os.path.join(args.out_dir, f"{prefix}_matrices.npz"),
        A=A.astype(np.float32),
        X_rpa=X_rpa.astype(np.float32),
        labels=y.astype(int),
        K=int(K),
    )

    with open(os.path.join(args.out_dir, f"{prefix}_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
