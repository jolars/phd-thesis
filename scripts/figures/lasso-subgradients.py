import matplotlib.pyplot as plt
import numpy as np

from pythesis.utils import PAL, save_fig, set_default_plot_settings


def lasso(x):
    return 0.5 * (x - 1.5) ** 2 + np.abs(x)


def l1_subgradient(x):
    return np.where(x >= 0, 1, np.where(x < 0, -1, 0))


def gradient(x):
    return x - 1.5 + np.sign(x)


set_default_plot_settings()
plt.rcParams["lines.markersize"] = 3

# Generate x values
x = np.linspace(-1.5, -1e-6, 100)
x = np.concatenate((x, np.linspace(1e-6, 4, 100)))
# Compute L1 norm and its subgradient
y = lasso(x)
dy = l1_subgradient(x)
# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(3, 2), layout="constrained")
# plt.figure(figsize=(3.5, 2), layout="constrained")
# Plot L1 norm
plt.vlines(0, -2, 9, color="lightgray")
plt.ylabel(r"$f(\beta)$")
plt.xlabel(r"$\beta$")

eps = 1e-4

ts = np.linspace(-4, 4, 20)

for t in ts:
    dy = lasso(0) + t * x
    if np.all(dy < y):
        plt.plot(x, dy, ":", color="gray")

dx = np.linspace(-1.2, 1, 100)
dy = lasso(3) + gradient(3) * dx
plt.plot(dx + 3, dy, ":", color="gray")
plt.plot(3, lasso(3), "ok")
plt.plot(0, lasso(0), "ok", markerfacecolor="white", zorder=10)

plt.plot(x, y, "k")
plt.ylim(-0.2, 8.2)


save_fig("lasso-subgradients.pdf")
