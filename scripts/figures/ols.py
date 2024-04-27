import matplotlib.pyplot as plt
import numpy as np

from pythesis.utils import FULL_WIDTH, PAL, save_fig, set_default_plot_settings

set_default_plot_settings()

np.random.seed(4)

n = 30
n_new = 5

x = np.random.rand(n) * 2
y = x + np.random.randn(n) * 0.2

coefficients = np.polyfit(x, y, 1)
linear_regression = np.poly1d(coefficients)
x_fit = np.linspace(min(x), max(x), 100)
y_fit = linear_regression(x_fit)

y_pred = linear_regression(x)


def plot_ols(plot_squares=False):
    fig, ax = plt.subplots(1, 1, figsize=(2.2, 2.2), layout="constrained")

    if plot_squares:
        for i in range(n):
            residual = y[i] - y_pred[i]
            ax.add_patch(
                plt.Rectangle(
                    (x[i], y_pred[i]), residual, residual, color="gray", alpha=0.3
                )
            )
            ax.plot([x[i], x[i]], [y_pred[i], y[i]], ":", color="black")

    ax.scatter(x, y, color="black", zorder=10)

    ax.set_aspect("equal")

    ax.plot(x_fit, y_fit, color=PAL[1])
    ax.set_xlabel(r"$\boldsymbol{x}$")

    ax.set_yticks([0, 1, 2])
    if not plot_squares:
        ax.set_ylabel(r"$\bm{y}$")
    else:
        ax.set_yticklabels([])


# plt.show(block=False)
plot_ols()
save_fig("ols-clean.pdf")

plot_ols(True)
save_fig("ols-squares.pdf")
