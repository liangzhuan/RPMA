"""
Spectral utilities.

Replacement path:
    Community_detection/methods/spectral_utils.py

Important change:
    For projection matrices X = U U^T, spectral_rounding(..., laplacian=False)
    no longer zeroes the diagonal by default.

Default policy:
    laplacian=True:
        zero_diagonal=True
        row_normalize=False

    laplacian=False:
        zero_diagonal=False
        row_normalize=False

This is more consistent with rank-K projection matrix methods, where X itself
is the object optimized on the projection manifold.
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def _symmetrize(S, zero_diagonal=False):
    """
    Symmetrize a square matrix.

    Parameters
    ----------
    S : array-like, shape (n, n)
        Input matrix.
    zero_diagonal : bool, default=False
        Whether to set diag(S) to zero after symmetrization.

    Notes
    -----
    The old implementation always zeroed the diagonal. That is inappropriate
    when S is a projection matrix X = U U^T because diag(X) carries leverage
    information and X ceases to be the same projection after diagonal removal.
    """
    S = np.asarray(S, dtype=float)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square matrix.")

    S = 0.5 * (S + S.T)

    if zero_diagonal:
        np.fill_diagonal(S, 0.0)

    return S


def kmeans_labels(embedding, K, random_state=0, row_normalize=False, n_init=20):
    """
    Run k-means on rows of an embedding matrix.

    Parameters
    ----------
    embedding : array-like, shape (n, d)
        Row embedding.
    K : int
        Number of clusters.
    random_state : int, default=0
        KMeans random seed.
    row_normalize : bool, default=False
        Whether to l2-normalize embedding rows before k-means.
        For paper-strict RPMA / projection-matrix experiments, keep False.
    n_init : int, default=20
        KMeans n_init.
    """
    embedding = np.asarray(embedding, dtype=float)
    embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

    if row_normalize:
        embedding = normalize(embedding, norm="l2")

    return KMeans(
        n_clusters=K,
        n_init=n_init,
        random_state=random_state,
    ).fit_predict(embedding)


def spectral_rounding(
    S,
    K,
    random_state=0,
    laplacian=False,
    zero_diagonal=None,
    row_normalize=False,
):
    """
    Convert a similarity/projection matrix into cluster labels.

    Parameters
    ----------
    S : array-like, shape (n, n)
        Affinity matrix or projection matrix.
    K : int
        Number of clusters.
    random_state : int, default=0
        KMeans random seed.
    laplacian : bool, default=False
        If True, use bottom-K eigenvectors of the unnormalized graph Laplacian.
        If False, use top-K eigenvectors of S.
    zero_diagonal : bool or None, default=None
        If None:
            zero_diagonal = True  when laplacian=True
            zero_diagonal = False when laplacian=False

        For projection matrices X=UU^T, use zero_diagonal=False.
    row_normalize : bool, default=False
        Whether to l2-normalize embedding rows before k-means.
        For paper-strict RPMA / projection-matrix experiments, keep False.

    Returns
    -------
    labels : ndarray, shape (n,)
        KMeans labels.
    """
    if zero_diagonal is None:
        zero_diagonal = bool(laplacian)

    S = _symmetrize(S, zero_diagonal=zero_diagonal)

    if laplacian:
        degrees = np.sum(S, axis=1)
        L = np.diag(degrees) - S
        eigvals, eigvecs = eigh(L)
        U = eigvecs[:, :K]
    else:
        eigvals, eigvecs = eigh(S)
        idx = np.argsort(eigvals)[::-1][:K]
        U = eigvecs[:, idx]

    return kmeans_labels(
        U,
        K,
        random_state=random_state,
        row_normalize=row_normalize,
    )


def top_eigen_embedding(S, K, zero_diagonal=False):
    """
    Return top-K eigenvectors of a symmetric matrix.

    For a projection matrix X=UU^T, use zero_diagonal=False.
    """
    S = _symmetrize(S, zero_diagonal=zero_diagonal)
    eigvals, eigvecs = eigh(S)
    idx = np.argsort(eigvals)[::-1][:K]
    return eigvecs[:, idx]


def laplacian_embedding(S, K, zero_diagonal=True):
    """
    Return bottom-K eigenvectors of the unnormalized graph Laplacian.
    """
    S = _symmetrize(S, zero_diagonal=zero_diagonal)
    degrees = np.sum(S, axis=1)
    L = np.diag(degrees) - S
    eigvals, eigvecs = eigh(L)
    return eigvecs[:, :K]
