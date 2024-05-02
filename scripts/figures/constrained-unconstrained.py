import matplotlib.pyplot as plt
import numpy as np

from pythesis.pgd import pgd
from pythesis.utils import FULL_WIDTH, PAL, save_fig, set_default_plot_settings

set_default_plot_settings()


# Define the OLS objective function
def ols_objective(beta, X, y):
    return np.sum(0.5 * (y - np.dot(X, beta)) ** 2)


# Define the l1 norm constraint
def l1_norm(beta):
    return np.sum(np.abs(beta))


# Define the lasso objective function
def lasso_objective(beta, X, y, lmbd):
    return np.sum(0.5 * (y - np.dot(X, beta)) ** 2) + l1_norm(beta) * lmbd


fig_width = 2.2

# Generate some random data
np.random.seed(0)
X = np.random.randn(100, 2)
corr = np.array([[1, 0.5], [0.5, 1]])
X = np.dot(np.linalg.cholesky(corr), X.T).T
y = np.dot(X, [1.8, -1]) + np.random.randn(100)

lambd = 101
w0 = [0.9, 1.4]
n_it = 1000

w_hist, grad_hist, prox_hist = pgd(X, y, lambd, w0, n_it)
w_opt = w_hist[:, -1]
t0 = l1_norm(w_opt)

lim = (-1.3, 2.2)

# Generate a grid of beta values
beta_grid = np.linspace(lim[0], lim[1], 400)
B1, B2 = np.meshgrid(beta_grid, beta_grid)
Z = np.array([ols_objective(beta, X, y) for beta in zip(B1.ravel(), B2.ravel())])
Z = Z.reshape(B1.shape)

# Plot the level curves of the OLS objective
fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width), layout="constrained")

ax.set_xlabel(r"$\beta_1$")
ax.set_ylabel(r"$\beta_2$")
ax.set_aspect("equal")

ax.contour(B1, B2, Z, levels=np.logspace(0, 5, 50), colors="grey", zorder=1)

ax.hlines(0, lim[0], lim[1], "grey", ":", zorder=2)
ax.vlines(0, lim[0], lim[1], "grey", ":", zorder=2)

t = np.linspace(-t0, t0, 101)
ax.fill_between(t, t0 - np.abs(t), np.abs(t) - t0, color=PAL[0], alpha=0.4, zorder=2)
ax.plot(t, t0 - np.abs(t), c=PAL[0])  # upper half of diamond
ax.plot(t, np.abs(t) - t0, c=PAL[0])  # lower half of diamond

ax.plot(w_opt[0], w_opt[1], "ko")
ax.annotate(
    r"$\boldsymbol{\beta}^*$",
    w_opt,
    w_opt + np.array([0.0, -0.2]),
    horizontalalignment="left",
    verticalalignment="center",
    zorder=10,
)

save_fig("constrained.pdf")

fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width), layout="constrained")

Z2 = np.array(
    [lasso_objective(beta, X, y, lambd) for beta in zip(B1.ravel(), B2.ravel())]
)
Z2 = Z2.reshape(B1.shape)

ax.set_xlabel(r"$\beta_1$")
ax.set_ylabel(r"$\beta_2$")
ax.set_aspect("equal")

ax.contour(B1, B2, Z2, levels=np.geomspace(180, 1000, 10))

ax.hlines(0, lim[0], lim[1], "grey", ":", zorder=2)
ax.vlines(0, lim[0], lim[1], "grey", ":", zorder=2)

ax.plot(w_opt[0], w_opt[1], "ko")
ax.annotate(
    r"$\boldsymbol{\beta}^*$",
    w_opt,
    w_opt + np.array([0.0, -0.2]),
    horizontalalignment="left",
    verticalalignment="center",
    zorder=10,
)

save_fig("unconstrained.pdf")
