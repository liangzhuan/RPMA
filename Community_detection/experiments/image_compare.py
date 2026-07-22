import time
import numpy as np

from evaluation.metrics import evaluate
from methods.affinity import preprocess_features, gaussian_affinity
from methods.rpa import rpa
from methods.spectral_utils import spectral_rounding, kmeans_labels


def run_image_clustering(
    X,
    y_true,
    K,
    pca_dim=100,
    k_neighbors=10,
    lam=0.02,
    delta=1e-3,
    rpa_max_iter=200,
    random_state=0,
):
    """Run Spectral baseline and RPA/RPMA on image features."""
    out = {}

    Xp = preprocess_features(X, pca_dim=pca_dim, random_state=random_state)
    A = gaussian_affinity(Xp, sigma="median", k_neighbors=k_neighbors, self_loop=False)

    t0 = time.time()
    labels_sp = spectral_rounding(A, K, random_state=random_state, laplacian=True)
    out["Spectral"] = evaluate(y_true, labels_sp)
    out["Spectral"]["time_sec"] = time.time() - t0

    t0 = time.time()
    X_rpa, U_rpa, history = rpa(
        A,
        K,
        lam=lam,
        delta=delta,
        max_iter=rpa_max_iter,
        eig_init=True,
        return_history=True,
        verbose=False,
    )
    labels_rpa = kmeans_labels(U_rpa, K, random_state=random_state)
    out["RPA-Huber"] = evaluate(y_true, labels_rpa)
    out["RPA-Huber"]["time_sec"] = time.time() - t0
    out["RPA-Huber"]["final_grad"] = float(history[-1]) if history else np.nan
    out["RPA-Huber"]["n_iter"] = len(history)

    return out, A, X_rpa
