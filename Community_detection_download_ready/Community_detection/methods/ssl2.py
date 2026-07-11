import numpy as np
from scipy.linalg import eigh


def trunc_matrix(Z, D, eta):

    n = Z.shape[0]

    eta = int((eta - n) // 2)

    D = np.triu(D, 1)

    idx = np.argsort(D.ravel())[::-1]

    mask = np.zeros(n * n)

    mask[idx[:eta]] = 1

    mask = mask.reshape(n, n)

    mask = mask + mask.T + np.eye(n)

    return Z * mask


def ssl2(A,
         c,
         eta,
         theta=1.0,
         tau=1e-6,
         loss='l1',
         max_iter=200):

    n = A.shape[0]

    A = A * np.sqrt(c) / np.linalg.norm(A, 'fro')
    A_abs = np.abs(A)

    Z = A.copy()

    for _ in range(max_iter):

        Z_old = Z.copy()

        eigvals, eigvecs = eigh(Z)

        idx = np.argsort(eigvals)[-c:]

        U = eigvecs[:, idx]

        W = U @ U.T - A

        if loss == 'fro':

            H = theta / (1 + theta) * W

            Z = A + H

            D = np.abs(Z)

        else:

            H = np.sign(W) * np.maximum(
                np.abs(W) - 1 / (2 * theta), 0
            )

            Z = A + H

            D = (
                    A_abs
                    - np.abs(H)
                    + theta * (
                            (A + W) ** 2
                            - (np.abs(H) - np.abs(W)) ** 2
                    )
            )

        Z = trunc_matrix(Z, D, eta)

        if np.linalg.norm(Z - Z_old) < tau:
            break

    return Z