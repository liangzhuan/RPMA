import numpy as np
from scipy.linalg import eigh


def projection_sd(X):

    d, U = eigh(X)

    d[d < 0] = 0

    return U @ np.diag(d) @ U.T


def project_affine2(Y, K):

    n = Y.shape[0]

    b = np.array([
        n ** 2 / K - n,
        n
    ])

    LY = np.array([
        np.sum(Y) - np.trace(Y),
        np.trace(Y)
    ])

    LL = np.diag([
        1 / (n ** 2 - n),
        1 / n
    ])

    ve = LL @ (LY - b)

    RE = (
            Y
            - ve[0] * (np.ones((n, n)) - np.eye(n))
            - ve[1] * np.eye(n)
    )

    return RE


def admm_sd2(A,
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

        X = project_affine2(
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