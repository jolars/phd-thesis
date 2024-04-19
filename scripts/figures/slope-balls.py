import matplotlib.pyplot as plt
import numpy as np

from pythesis.utils import FULL_WIDTH, PAL, save_fig, set_default_plot_settings

set_default_plot_settings()

t0 = 1


def plot_slopeball(w, i):
    fig, ax = plt.subplots(figsize=(1.5, 1.2), layout="constrained")

    ax.set_xlabel(r"$\beta_1$")

    if i == 0:
        ax.set_ylabel(r"$\beta_2$")

    ax.set_aspect("equal")

    a = 1 / (w[0] + w[1])

    lim = 1.2

    x = np.array([-1, -a, 0, a, 1])
    y1 = np.array([0, a, 1, a, 0])
    y2 = np.array([0, -a, -1, -a, 0])

    ax.hlines(0, -lim, lim, color="lightgrey")
    ax.vlines(0, -lim, lim, color="lightgrey")
    ax.fill_between(x, y1, y2, color=PAL[0], alpha=0.4, zorder=2)
    ax.plot(x, y1, color=PAL[0])  # upper half of diamond
    ax.plot(x, y2, color=PAL[0])  # lower half of diamond

    if i == 1:
        pos = (-1, 0)
        ax.plot(pos[0], pos[1], "o", color="black")
        ax.text(
            pos[0],
            pos[1] + 0.1,
            r"$\boldsymbol{\beta}$",
            va="bottom",
            ha="center",
        )

    if i == 2:
        b = 1 / (w[0] + w[1])

        pos = (-b, b)
        ax.plot(pos[0], pos[1], "o", color="black")
        ax.text(
            pos[0],
            pos[1],
            r"$\boldsymbol{\beta}$",
            va="bottom",
            ha="right",
        )

    # ax.spines[["right", "top"]].set_visible(False)

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])


w2 = [1, 3 / 4, 1 / 4, 0]

for i, w in enumerate(w2):
    plot_slopeball([1, w], i)

    save_fig(f"slope-ball-{i}.pdf")

# plot_elasticnetball(0)
# save_fig("elasticnet-balls-ridge.pdf")
#
# plot_elasticnetball(0.5)
# save_fig("elasticnet-balls-elasticnet.pdf")
#
# plot_elasticnetball(1)
# save_fig("elasticnet-balls-lasso.pdf")

# plt.close("all")
