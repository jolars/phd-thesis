import matplotlib.pyplot as plt
import numpy as np

from pythesis.utils import FULL_WIDTH, PAL, save_fig, set_default_plot_settings

set_default_plot_settings()

np.random.seed(4)


def f1(x):
    return (np.array(x) - 1) ** 2


def g1(x):
    return 2 * (np.array(x) - 1), None


def f2(x):
    return (np.array(x) - 1) ** 3 - (np.array(x)) ** 2


def g2(x):
    return 3 * (np.array(x) - 1) ** 2 - 2 * np.array(x), None


def f3(x):
    return (np.array(x) - 1) ** 2 + 2 * np.abs(np.array(x))


def g3(x):
    v = 2 * (np.array(x) - 1) + 2 * np.sign(np.array(x))
    switch_ind = np.where(v >= 0)[0][0]
    v0 = v[:switch_ind]
    v1 = v[switch_ind:]

    return v0, v1


# obj = 0.5 * np.linalg.norm(y - x_grid[:, None] * x, axis=1) ** 2


def plot_convexity(f, g):
    fig, ax = plt.subplots(2, 1, figsize=(1.5, 1.9), layout="constrained", sharex=True)

    xmin = -1.0
    xmax = 3

    x = np.linspace(xmin, xmax, 100)

    fx = f(x)
    gx0, gx1 = g(x)

    ax[0].plot(x, fx, color="black")

    i = 15
    j = 60
    ax[0].plot([x[i], x[j]], [fx[i], fx[j]], "o:", color=PAL[0])

    # ax[0].plot(beta_hat, obj_hat, "o", color=PAL[1])
    # ax[0].text(beta_hat, obj_hat + 1, r"$\hat{\beta}$", va="bottom")

    # ax[0].set_xlabel(r"$\beta$")
    ax[0].set_ylabel(r"$f(\beta)$")

    # plot gradient

    # ax[0].plot(beta_hat, obj_hat, "o", color=PAL[1])
    # ax[0].text(beta_hat, obj_hat + 1, r"$\hat{\beta}$", va="bottom")
    #

    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].set_xlabel(r"$\beta$")
    ax[1].set_ylabel(r"$\nabla f(\beta)$")

    #
    ax[1].hlines(0, xmin, xmax, color="lightgray")
    # ax[1].vlines(beta_hat, grad.min(), grad.max(), color="lightgray", linestyle=":")
    # ax[1].plot(x, gx0, color="black")

    if gx1 is None:
        ax[1].plot(x, gx0, color="black")
    else:
        n = len(gx0)
        ax[1].plot(x[:n], gx0, color="black")
        ax[1].plot(x[n:], gx1, color="black")
        ax[1].scatter(x[n], gx0[-1], edgecolors="black", facecolors="white", zorder=10)
        ax[1].scatter(x[n], gx1[0], edgecolors="black", facecolors="white", zorder=10)

    ax[1].set_yticks([0])


plot_convexity(f1, g1)
save_fig("1d-convexity-convex.pdf")

plot_convexity(f2, g2)
save_fig("1d-convexity-nonconvex.pdf")

plot_convexity(f3, g3)
save_fig("1d-convexity-nonsmooth.pdf")

plt.close("all")
