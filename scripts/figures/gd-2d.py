import matplotlib.pyplot as plt
import numpy as np

from pythesis.utils import FULL_WIDTH, PAL, save_fig, set_default_plot_settings


def f(beta, x, y):
    return 0.5 * np.linalg.norm(x @ beta - y) ** 2


def f_grad(beta, x, y):
    return x.T @ (x @ beta - y)


def approx2d(beta, beta0, x, y, t):
    return (
        f(beta0, x, y)
        + f_grad(beta0, x, y).T @ (beta - beta0)
        + 1 / (2 * t) * np.linalg.norm(beta - beta0) ** 2
    )


def gradient_descent2d(beta0, x, y, t, epochs):
    betas = np.zeros((epochs + 1, 2))
    beta = beta0.copy()
    betas[0] = beta
    for i in range(epochs):
        dx = f_grad(beta, x, y)
        beta -= t * dx
        betas[i + 1] = beta
    return betas


set_default_plot_settings()

beta_star = np.array([1, -2.0])
beta0 = np.array([0, 2.43])

n = 100

rng = np.random.default_rng(5)

mu = np.zeros(2)
sigma = np.array([[1, 0.5], [0.5, 1]])

x = rng.multivariate_normal(mu, sigma, n)

y = x @ beta_star

t = 1 / np.linalg.norm(x) ** 2

epochs = 500
x_path = gradient_descent2d(beta0, x, y, t, epochs)

lim = (-2.9, 3.9)

beta_grid = np.linspace(lim[0], lim[1], 100)
B1, B2 = np.meshgrid(beta_grid, beta_grid)

F = np.array([f(beta_i, x, y) for beta_i in zip(B1.ravel(), B2.ravel())])

A = np.array(
    [approx2d(beta_i, beta0, x, y, t) for beta_i in zip(B1.ravel(), B2.ravel())]
)

fig, ax = plt.subplots(figsize=(2.2, 2.2))

ax.contour(B1, B2, F.reshape(B1.shape), levels=10, colors="lightgrey")

f0 = f(beta0, x, y)
fapprox = approx2d(x_path[1], beta0, x, y, t)

lvls = np.geomspace(f(x_path[1], x, y), f(x_path[0], x, y), 6)

ax.contour(B1, B2, A.reshape(B1.shape), levels=lvls, colors=PAL[0], alpha=0.6)
# ax.contour(B1, B2, A.reshape(B1.shape), levels=lvls, colors="black", linestyles=":")

ax.plot(x_path[:, 0], x_path[:, 1], "-o", color="black")

ax.scatter(beta_star[0], beta_star[1], s=25, c=PAL[1], zorder=15, marker="X")
ax.scatter(x_path[1, 0], x_path[1, 1], s=14, c=PAL[0], zorder=15)

plt.text(
    x_path[0, 0],
    x_path[0, 1] + 0.2,
    r"$\boldsymbol{\beta}^{(0)}$",
    ha="left",
    va="bottom",
)
plt.text(
    beta_star[0] + 0.3,
    beta_star[1],
    r"$\boldsymbol{\beta}^*$",
    ha="left",
    va="center",
)

plt.xlabel(r"$\beta_1$")
plt.ylabel(r"$\beta_2$")

# plt.legend(loc="upper left")

save_fig("gd-2d.pdf")
