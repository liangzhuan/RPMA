import numpy as np
from scipy.linalg import eigh


def projection_sd(X):

    d, U = eigh(X)

    d[d < 0] = 0

    return U @ np.diag(d) @ U.T


def project_affine1(Y, K):

    n = Y.shape[0]

    b = np.concatenate([
        np.ones(n) * 2 * (n / K - 1),
        np.ones(n)
    ])

    LY = np.concatenate([
        2 * (np.sum(Y, axis=1) - np.diag(Y)),
        np.diag(Y)
    ])

    LL = np.zeros((2 * n, 2 * n))

    LL[:n, :n] = (
            1 / (2 * n - 4)
            * (np.eye(n) - np.ones((n, n)) / (2 * n - 2))
    )

    LL[n:, n:] = np.eye(n)

    ve = LL @ (LY - b)

    RE = Y - (
            ve[:n][:, None]
            + ve[:n][None, :]
            - 2 * np.diag(ve[:n])
            + np.diag(ve[n:])
    )

    return RE


def admm_sd1(A,
             K,
             rho=1.0,
             tol=1e-4,
             max_iter=1000):

    n = A.shape[0]

    Z = np.zeros((n, n))
    Y = np.zeros((n, n))

    U = np.zeros((n, n))
    V = np.zeros((n, n))

    for _ in range(max_iter):

        X = project_affine1(
            0.5 * (rho * Z - U + rho * Y - V + A) / rho,
            K
        )

        Z = np.maximum(X + U / rho, 0)

        Y = projection_sd(X + V / rho)

        U += rho * (X - Z)

        V += rho * (X - Y)

        if (
                np.linalg.norm(X - Z)
                + np.linalg.norm(X - Y)
        ) < tol:
            break

    return X