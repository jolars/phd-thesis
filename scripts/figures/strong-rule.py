import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.linear_model import lars_path

from pythesis.utils import FULL_WIDTH, PAL, save_fig


def linear_interpolation(x1, y1, x2, y2, x):
    """
    Perform linear interpolation between two points (x1, y1) and (x2, y2) for a given x.

    Parameters:
    x1, y1: coordinates of the first point.
    x2, y2: coordinates of the second point.
    x: the x-coordinate at which to interpolate.

    Returns:
    The interpolated y value.
    """
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


# Load the diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

n, p = X.shape

lambdas, _, coefs = lars_path(X, y, method="lasso", verbose=True)

lambdas *= n

c = np.zeros_like(coefs)

for j in range(p):
    c[:, j] = np.abs(X.T @ (y - X @ coefs[:, j]))

fig, ax = plt.subplots(1, 1, figsize=(3, 3))

cmap = ListedColormap(PAL)


for i in range(coefs.shape[0]):
    ax.plot(lambdas, c[i, :], color=PAL[i])
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$|c_j|$")

ylim = np.array(ax.get_ylim())

ax.set_ylim(ylim)

lambda_grid = np.geomspace(lambdas[0], lambdas[0] * 1e-4, 100)


def plot_rule(step, j):
    lam_diff = lambda_grid[step] - lambda_grid[step + 1]

    # interpolate between the two points
    l0, l1 = lambda_grid[step : step + 2]

    i = np.where(lambdas > l0)[0][-1]
    k = np.where(lambdas < l0)[0][0]

    y0 = linear_interpolation(lambdas[i], c[j, i], lambdas[k], c[j, k], l0)

    ax.plot(
        [l0, l1],
        [y0, y0 + lam_diff],
        "o--",
        color=PAL[j],
        markevery=[-1, 1],
        markerfacecolor="white",
    )

    ax.plot(l0, y0, "o", color=PAL[j])

    ax.text(
        l1 - 25,
        y0 + lam_diff,
        rf"$\big|\hat{{c}}^{{(k+1)}}_{j+1}\big|$",
        ha="left",
        va="center",
    )


step = 5
ylim = np.array(ax.get_ylim())
ax.set_ylim(ylim)
# ax.set_xlim(390, ax.get_xlim()[1])

ax.axline((0, 0), slope=1.0, color="lightgrey", linestyle="dotted", zorder=-2)

l0, l1 = lambda_grid[step : step + 2]

ax.vlines(l0, ylim[0], ylim[1], color="lightgrey", ls="dotted", zorder=-1)
ax.vlines(
    l1,
    ylim[0],
    ylim[1],
    color="lightgrey",
    ls="dotted",
    zorder=-1,
)

ax.text(l0 + 25, ylim[1] - 20, r"$\lambda_{k}$", ha="right", va="top")
ax.text(l1 - 25, ylim[1] - 20, r"$\lambda_{k + 1}$", ha="left", va="top")

plot_rule(step, 0)
plot_rule(step, 3)

ax.set_aspect("equal")

ax.get_xaxis().set_inverted(True)

save_fig("strong-rule.pdf")

plt.close("all")
