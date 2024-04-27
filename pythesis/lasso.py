import time

import numpy as np


def project_simplex_2d(v, z):
    shape = v.shape
    if shape[1] == 1:
        w = np.array(v)
        w[:] = z
        return w

    mu = np.sort(v, axis=1)
    mu = np.flip(mu, axis=1)
    cum_sum = np.cumsum(mu, axis=1)
    j = np.expand_dims(np.arange(1, shape[1] + 1), 0)
    rho = np.sum(mu * j - cum_sum + z > 0.0, axis=1, keepdims=True) - 1
    max_nn = cum_sum[np.arange(shape[0]), rho[:, 0]]
    theta = (np.expand_dims(max_nn, -1) - z) / (rho + 1)
    w = (v - theta).clip(min=0.0)
    return w


def st(x, lambda_):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0.0)


def lasso_gd(A, b, lambda_, tau=None, method="proj", max_iter=100_000, tol=1e-6):
    m, n = A.shape

    x = np.zeros(n)

    x_hist = np.zeros((max_iter, n))
    f_hist = np.zeros(max_iter)
    t_hist = np.zeros(max_iter)
    # gap_hist = np.zeros(max_iter)

    L = np.linalg.norm(A, ord=2) ** 2

    t0 = time.perf_counter()

    it = 0

    while it < max_iter:
        r = A @ x - b

        p_obj = 0.5 * np.linalg.norm(r) ** 2 + lambda_ * np.linalg.norm(x, 1)

        x_hist[it] = x
        f_hist[it] = p_obj
        t_hist[it] = time.perf_counter() - t0

        if it % 10 == 0:
            scaling = max(1, np.linalg.norm(A.T @ r, ord=np.inf) / lambda_)
            d_obj = (
                0.5 * np.linalg.norm(b) ** 2
                - 0.5 * np.linalg.norm(b + r / scaling) ** 2
            )
            rel_gap = (p_obj - d_obj) / p_obj

            if rel_gap < tol:
                break

        g = A.T @ r
        x_update = x - g / L

        if method == "prox":
            x = st(x_update, lambda_ / L)
        elif method == "proj":
            s = np.sign(x_update)
            x = project_simplex_2d(np.abs(x_update)[None, :], tau)
            x = x[0] * s
        else:
            raise ValueError("Invalid method")

        it += 1

    return x_hist[:it], f_hist[:it], t_hist[:it]


def lasso_proj_2d(A, b, x_start, tau=None, max_iter=10):
    m, n = A.shape

    x = x_start

    # x_grad_hist = np.zeros((max_iter, n))
    # x_proj_hist = np.zeros((max_iter + 1, n))
    x_hist = np.zeros((2 * max_iter + 1, n))

    x_hist = [x]

    # x_hist[0] = x

    L = np.linalg.norm(A, ord=2) ** 2

    for it in range(max_iter):
        r = A @ x - b

        g = A.T @ r
        x_update = x - g / L

        x_hist = np.vstack([x_hist, x_update])

        s = np.sign(x_update)
        x = project_simplex_2d(np.abs(x_update)[None, :], tau)
        x = x[0] * s

        x_hist = np.vstack([x_hist, x])

    return x_hist


def lasso_ball(t0):
    t = np.linspace(-t0, t0, 101)
    return t, t0 - np.abs(t), np.abs(t) - t0
    # ax.hlines(0, -1.5, 1.5, color="lightgrey")
    # ax.vlines(0, -1.5, 1.5, color="lightgrey")
    # ax.fill_between(
    #     t, t0 - np.abs(t), np.abs(t) - t0, color=PAL[0], alpha=0.4, zorder=2
    # )
    # ax.plot(t, t0 - np.abs(t), color=PAL[0])  # upper half of diamond
    # ax.plot(t, np.abs(t) - t0, color=PAL[0])  # lower half of diamond
