import matplotlib.pyplot as plt
import numpy as np

from pythesis.utils import FULL_WIDTH, PAL, save_fig, set_default_plot_settings


def f(x):
    return 0.5 * (x - 1) ** 2


def df(x):
    return x - 1


def approx(x, x_start, t):
    return f(x_start) + df(x_start) * (x - x_start) + 1 / (2 * t) * (x - x_start) ** 2


def gradient_descent(x_start, df, learning_rate, epochs):
    xs = np.zeros(epochs + 1)
    x = x_start
    xs[0] = x
    for i in range(epochs):
        dx = df(x)
        x -= learning_rate * dx
        xs[i + 1] = x
    return xs


set_default_plot_settings()

t = 0.2

x_start = 4.0
epochs = 15
x_path = gradient_descent(x_start, df, t, epochs)
x = np.linspace(0.5, 5, 400)
y = f(x)

x_grad = np.linspace(-1.5, 1.2, 100)
y_grad_f = f(x_grad)
y_grad_df = f(x_start) + x_grad * df(x_start)

x_approx = np.linspace(x_path[1] - 0.9, x_path[1] + 1.2, 100)
f_approx = approx(x_approx, x_start, t)

plt.figure(figsize=(2.2, 2.2))

plt.plot(x, y, "k", zorder=10, color="black")
plt.scatter(x_path, f(x_path), zorder=12, color="black")

plt.plot(
    x_start + x_grad,
    y_grad_df,
    "k:",
    zorder=9,
    label=r"$\nabla f(\beta)$",
    color="grey",
)
plt.xlabel(r"$\beta$")
plt.ylabel(r"$f(\beta)$")

plt.plot(x_approx, f_approx, zorder=11, label="Approximation", color=PAL[0], alpha=0.6)
plt.plot(x_path[1], approx(x_path[1], x_start, t), "o", zorder=15, color=PAL[0])

plt.scatter(1, 0, s=25, c=PAL[1], zorder=15, marker="X")

plt.text(
    1,
    0 + 0.3,
    r"$\boldsymbol{\beta}^*$",
    ha="center",
    va="bottom",
)
plt.text(
    x_start + 0.15,
    f(x_start),
    r"$\boldsymbol{\beta}^{(0)}$",
    ha="left",
    va="top",
)

plt.legend(loc="upper left")

save_fig("gd-1d.pdf")
