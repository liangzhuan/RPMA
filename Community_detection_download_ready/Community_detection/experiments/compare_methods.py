from methods.ssl2 import ssl2
from methods.admm_sd1 import admm_sd1
from methods.admm_sd2 import admm_sd2
from methods.clr import clr
from methods.rpa import rpa

from methods.spectral_utils import spectral_rounding, kmeans_labels
from evaluation.metrics import evaluate


def compare(A, y_true, K, lam=0.02, delta=1e-3, rpa_max_iter=200):
    """Compare graph/community-detection methods on an affinity matrix."""
    results = {}

    labels = spectral_rounding(A, K, laplacian=True)
    results["Spectral"] = evaluate(y_true, labels)

    X, U, _ = rpa(A, K, lam=lam, delta=delta, max_iter=rpa_max_iter,
                  eig_init=True, return_history=True)
    labels = kmeans_labels(U, K)
    results["RPA-Huber"] = evaluate(y_true, labels)

    S = ssl2(A, c=K, eta=min(2000, A.shape[0] * A.shape[0]))
    labels = spectral_rounding(S, K, laplacian=True)
    results["SSL2"] = evaluate(y_true, labels)

    X = admm_sd1(A, K)
    labels = spectral_rounding(X, K, laplacian=False)
    results["ADMM-SD1"] = evaluate(y_true, labels)

    X = admm_sd2(A, K)
    labels = spectral_rounding(X, K, laplacian=False)
    results["ADMM-SD2"] = evaluate(y_true, labels)

    S = clr(A, lam=1.0, K=K)
    labels = spectral_rounding(S, K, laplacian=True)
    results["CLR"] = evaluate(y_true, labels)

    return results
