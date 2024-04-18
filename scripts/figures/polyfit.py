import matplotlib.pyplot as plt
import numpy as np

from pythesis.utils import (
    FULL_WIDTH,
    default_palette,
    save_fig,
    set_default_plot_settings,
)

set_default_plot_settings()

pal = default_palette()

np.random.seed(4)

n = 30
n_new = 5

x = np.random.rand(n)
y = 2 * x + np.random.randn(n) * 0.5

x_new = np.random.rand(n_new)
y_new = 2 * x_new + np.random.randn(n_new) * 0.5

degrees = [1, 4, 8, 15]

n_degrees = len(degrees)

fig, axs = plt.subplots(
    1,
    n_degrees,
    figsize=(FULL_WIDTH, 1.5),
    sharex=True,
    sharey=True,
    layout="constrained",
)

for i, degree in enumerate(degrees):
    coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coefficients)

    x_fit = np.linspace(
        np.min(np.hstack([x, x_new])), np.max(np.hstack([x, x_new])), 100
    )
    y_fit = polynomial(x_fit)

    axs[i].scatter(x, y, color="black")
    axs[i].scatter(x_new, y_new, color=pal[0])
    axs[i].plot(x_fit, y_fit, color=pal[1])
    axs[i].set_title("Degree " + str(degree))
    axs[i].set_xlabel(r"$\boldsymbol{x}$")
    axs[i].set_yticklabels([])
    axs[i].set_xticklabels([])
    axs[i].set_yticks([])
    axs[i].set_xticks([])

axs[0].set_ylabel(r"$\boldsymbol{y}$")

plt.show(block=False)

save_fig("polyfit.pdf")
