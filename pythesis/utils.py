import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from pyprojroot import here

FULL_WIDTH = 4.6

PAL = [
    "#5778a4",
    "#e49444",
    "#d1615d",
    "#85b6b2",
    "#6a9f58",
    "#e7ca60",
    "#a87c9f",
    "#f1a2a9",
    "#967662",
    "#b8b0ac",
]


def extend_range(x, f=0.1):
    """Extend range of 1d array by f * (max - min) on both sides."""
    r = np.max(x) - np.min(x)
    return (np.min(x) - f * r, np.max(x) + f * r)


def set_default_plot_settings():
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 8
    plt.rcParams["lines.markersize"] = 2
    plt.rcParams["figure.labelsize"] = "medium"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage[ebgaramond,textscale=0,semibold,vvarbb,subscriptcorrection,amsthm]{newtx}\usepackage{mathtools}"
    )


def save_fig(name, pad_inches=0.01):
    set_default_plot_settings()

    path = here("figures") / name

    plt.savefig(path, bbox_inches="tight", pad_inches=pad_inches)
    os.system("pdfcrop %s %s >/dev/null 2>&1" % (path, path))
