import matplotlib.pyplot as plt
import numpy as np
from pyprojroot.here import here

from pythesis.utils import FULL_WIDTH, PAL, save_fig, set_default_plot_settings

set_default_plot_settings()


def plot_lassoball(type="sparse"):
    t0 = 1

    fig, ax = plt.subplots(figsize=(1.5, 1.5), layout="constrained")

    ax.set_xlabel(r"$\beta_1$")
    ax.set_ylabel(r"$\beta_2$")
    ax.set_aspect("equal")

    t = np.linspace(-t0, t0, 101)
    ax.hlines(0, -1.5, 1.5, color="lightgrey")
    ax.vlines(0, -1.5, 1.5, color="lightgrey")
    ax.fill_between(
        t, t0 - np.abs(t), np.abs(t) - t0, color=PAL[0], alpha=0.4, zorder=2
    )
    ax.plot(t, t0 - np.abs(t), color=PAL[0])  # upper half of diamond
    ax.plot(t, np.abs(t) - t0, color=PAL[0])  # lower half of diamond

    # ax.spines[["right", "top"]].set_visible(False)

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])

    if type == "sparse":
        pos = (0, -1)
        v = "top"
        h = "left"
        xmod = 0.1
        ymod = -0.1
    elif type == "dense":
        pos = (1 / 2, 1 / 2)
        v = "bottom"
        h = "left"
        xmod = 0.1
        ymod = 0.1
    else:
        pos = (0.4, 0.3)
        v = "top"
        h = "center"
        xmod = 0
        ymod = -0.15

    ax.plot(pos[0], pos[1], "o", color="black")
    ax.text(
        pos[0] + xmod,
        pos[1] + ymod,
        r"$\boldsymbol{\beta}$",
        verticalalignment=v,
        horizontalalignment=h,
    )

    if type == "sparse":
        ax.annotate(
            "$t$",
            xy=(-0.5, 1.13),
            xycoords="data",
            xytext=(-0.5, 1.4),
            textcoords="data",
            ha="center",
            va="center",
            arrowprops=dict(
                arrowstyle="-[, widthB=1.4, lengthB=0.2",  # Bracket A is "{", Bracket B is "|-|"
                lw=0.75,
                shrinkA=4,
                # shrinkB=0.5,
            ),
        )


plot_lassoball("sparse")
path = here("figures") / "lasso-ball-sparse.pdf"
save_fig(path)

plot_lassoball("dense")
path = here("figures") / "lasso-ball-dense.pdf"
save_fig(path)

plot_lassoball("inactive")
path = here("figures") / "lasso-ball-inactive.pdf"
save_fig(path)

plt.close("all")
