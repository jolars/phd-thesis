import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from pythesis.utils import PAL, save_fig, set_default_plot_settings

set_default_plot_settings()


def extend_range(x, f=0.1):
    """Extend range of 1d array by f * (max - min) on both sides."""
    r = np.max(x) - np.min(x)
    return (np.min(x) - f * r, np.max(x) + f * r)


def get_limit(x, f=0.1):
    r = np.max(x) - np.min(x)
    return np.max(np.abs(x)) + f * r


def st(x, lmbd):
    return np.sign(x) * np.maximum(np.abs(x) - lmbd, 0)


def cd(x, y, lmbd, w0, max_it=100):
    p = x.shape[1]
    residual = np.copy(y) - x @ w0

    w = w0.copy()
    w_history = np.zeros((p, max_it))

    L = (x**2).sum(axis=0)

    it = 0

    while True:
        w_old = np.copy(w)
        for j in range(p):
            w_history[:, it] = w

            old = w[j]
            w[j] = st(w[j] + x[:, j] @ residual / L[j], lmbd / L[j])
            diff = old - w[j]
            if diff != 0:
                residual += diff * x[:, j]

            it += 1

        if np.linalg.norm(w - w_old) < 1e-4 or it >= max_it:
            break

    return w_history[:, :it]


# Generate some sparse data
np.random.seed(42)
n, p = 100, 2

x = np.random.randn(n, p)
corr = 0.8
x[:, 1] = corr * x[:, 0] + np.sqrt(1 - corr**2) * x[:, 1]
coef = 3 * np.random.randn(p)
inds = np.arange(p)
np.random.shuffle(inds)

coef[inds[1:]] = 0

y = np.dot(x, coef)
y += 0.01 * np.random.normal(size=n)

max_it = 50

w = np.zeros(p)

lmbd_max = np.max(np.abs(x.T @ y))
lmbd = lmbd_max * 0.5
w0 = [0.8, 0.0]

w_history = cd(x, y, lmbd, w0, max_it)

w_optim = w_history[:, -1]

# Add level curves
x_lim = [-0.4, 1.55]
y_lim = [-0.1, 1.55]
w1_vals = np.linspace(x_lim[0], x_lim[1], 100)
w2_vals = np.linspace(y_lim[0], y_lim[1], 100)

W1, W2 = np.meshgrid(w1_vals, w2_vals)

J = np.zeros_like(W1)
for i in range(len(w1_vals)):
    for j in range(len(w2_vals)):
        w = np.array([w1_vals[i], w2_vals[j]])
        J[j, i] = 0.5 * np.linalg.norm(y - x @ w) ** 2 + lmbd * np.linalg.norm(w, 1)


fig, ax = plt.subplots(figsize=(2.5, 2.2), layout="constrained")

ax.hlines(0, x_lim[0], x_lim[1], "grey", ":", zorder=2)
ax.vlines(0, y_lim[0], y_lim[1], "grey", ":", zorder=2)
ax.contour(W1, W2, J, levels=15)

ax.plot(w_history[0, :], w_history[1, :], "ko-")

ax.text(
    w_optim[0] + 0.05, w_optim[1], r"$\boldsymbol{\beta}^*$", va="center", ha="left"
)
ax.text(w0[0] + 0.05, w0[1], r"$\boldsymbol{\beta}^{(0)}$", va="bottom", ha="left")

ax.set_xlabel(r"$\beta_1$")
ax.set_ylabel(r"$\beta_2$")

save_fig("cd.pdf")

plt.close("all")
