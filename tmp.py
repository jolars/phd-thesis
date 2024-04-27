import time

import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets.simulated import make_correlated_data


def st(x, lambda_):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0.0)


def lasso_gd(A, b, lambda_, method="proj", max_iter=100):
    m, n = A.shape

    x = np.zeros(n)

    x_hist = np.zeros((max_iter, n))
    f_hist = np.zeros(max_iter)
    t_hist = np.zeros(max_iter)

    L = np.linalg.norm(A) ** 2

    t0 = time.perf_counter()

    for i in range(max_iter):
        x_hist[i] = x
        f_hist[i] = 0.5 * np.linalg.norm(A @ x - b) ** 2 + lambda_ * np.linalg.norm(
            x, 1
        )
        t_hist[i] = time.perf_counter() - t0

        g = A.T @ (A @ x - b)

        if method == "prox":
            x = st(x - g / L, lambda_ / L)
        elif method == "proj":
            x = euclidean_proj_l1ball(x - g / L, lambda_)
        else:
            raise ValueError("Invalid method")

    return x_hist, f_hist, t_hist


n = 10000
m = 1000

max_iter = 500

A, b, x_true = make_correlated_data(m, n, rho=0.5, snr=1.0, density=0.1)
rng = np.random.default_rng(5)

lambda_max = np.linalg.norm(A.T @ b, np.inf)

x_hist, f_hist, t_hist = lasso_gd(
    A, b, lambda_max * 0.1, method="prox", max_iter=max_iter
)

tau = np.linalg.norm(x_hist[-1], 1)

x_hist_proj, f_hist_proj, t_hist_proj = lasso_gd(
    A, b, tau, method="proj", max_iter=max_iter
)

fig, ax = plt.subplots(1, 1, figsize=(2, 2))

opt = np.min(f_hist)

ax.semilogy(f_hist - opt, label="Proximal GD")
ax.semilogy(f_hist_proj - opt, label="Projected GD")
ax.set_xlabel("Iteration")

# ax.semilogy(t_hist, f_hist - opt, label="Proximal GD")
# ax.semilogy(t_hist_proj, f_hist_proj - opt, label="Projected GD")
# ax.set_xlabel("Time (s)")

ax.set_ylabel("Suboptimality")

plt.legend()
