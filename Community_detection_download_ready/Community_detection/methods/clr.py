import numpy as np
from scipy.linalg import eigh


def simplex_projection(v):

    u = np.sort(v)[::-1]

    cssv = np.cumsum(u)

    rho = np.nonzero(
        u + (1 - cssv) / (np.arange(len(u)) + 1) > 0
    )[0][-1]

    theta = (cssv[rho] - 1) / (rho + 1)

    return np.maximum(v - theta, 0)


def clr(A,
        lam=1.0,
        K=3,
        max_iter=100):

    m = A.shape[0]

    S = np.zeros_like(A)

    for i in range(m):
        S[i] = simplex_projection(A[i])

    for _ in range(max_iter):

        T = (S + S.T) / 2

        L = np.diag(T.sum(axis=1)) - T

        eigvals, eigvecs = eigh(L)

        F = eigvecs[:, :K]

        normF = np.sum(F ** 2, axis=1)

        DIST = (
                normF[:, None]
                + normF[None, :]
                - 2 * F @ F.T
        )

        S_new = np.zeros_like(S)

        for i in range(m):

            S_new[i] = simplex_projection(
                A[i] - lam / 2 * DIST[i]
            )

        if np.linalg.norm(S_new - S) < 1e-8:
            break

        S = S_new

    return S