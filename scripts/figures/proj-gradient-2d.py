import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets.simulated import make_correlated_data

from pythesis.lasso import lasso_proj_2d
from pythesis.utils import PAL, save_fig, set_default_plot_settings


def f(beta, x, y):
    return 0.5 * np.linalg.norm(x @ beta - y) ** 2


set_default_plot_settings()

beta_star = np.array([1, -2.0])

n = 100

rng = np.random.default_rng(5)

mu = np.zeros(2)
sigma = np.array([[1, 0.8], [0.8, 1]])

A = rng.multivariate_normal(mu, sigma, n)

b = A @ beta_star

lambda_max = np.linalg.norm(A.T @ b, np.inf)
lam = lambda_max * 0.5

tau = 0.5

max_iter = 20

beta0 = np.array([-0.8, 1.2])
x = lasso_proj_2d(A, b, beta0, tau, max_iter=max_iter)

x = np.array(x)

fig, ax = plt.subplots(1, 1, figsize=(2.2, 2.2))

lim = (-1.6, 1.6)
beta_grid = np.linspace(lim[0], lim[1], 100)
B1, B2 = np.meshgrid(beta_grid, beta_grid)

F = np.array([f(beta_i, A, b) for beta_i in zip(B1.ravel(), B2.ravel())])

ax.set_aspect("equal")
ax.contour(B1, B2, F.reshape(B1.shape), levels=10, colors="lightgrey")

t = np.linspace(-tau, tau, 101)
ax.hlines(0, lim[0], lim[1], color="lightgrey", linestyle="dotted")
ax.vlines(0, lim[0], lim[1], color="lightgrey", linestyle="dotted")
ax.fill_between(t, tau - np.abs(t), np.abs(t) - tau, color=PAL[0], alpha=0.4, zorder=2)
ax.plot(t, tau - np.abs(t), color=PAL[0])  # upper half of diamond
ax.plot(t, np.abs(t) - tau, color=PAL[0])  # lower half of diamond

for i in range(8):
    linestyle = ":" if i % 2 == 0 else "-"
    label = "Gradient" if i % 2 == 0 else "Projection"
    ax.plot(x[0, 0], x[0, 1], "ok")
    ax.plot(
        x[i : i + 2, 0],
        x[i : i + 2, 1],
        "ko",
        linestyle=linestyle,
        label=label if i < 2 else None,
        markerfacecolor="white" if i % 2 == 0 else "black",
        markevery=(-1, 3),
        zorder=50 - i,
    )

ax.text(x[0, 0] - 0.1, x[0, 1], r"$\boldsymbol{\beta}^{(0)}$", ha="right")
ax.text(x[-1, 0] + 0.05, x[-1, 1], r"$\boldsymbol{\beta}^*$", ha="left", va="top")

ax.set_xlabel(r"$\beta_1$")
ax.set_ylabel(r"$\beta_2$")

ax.legend(loc="upper right")

save_fig("proj-gradient-2d.pdf")

plt.close("all")
