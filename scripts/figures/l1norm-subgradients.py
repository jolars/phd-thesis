import matplotlib.pyplot as plt
import numpy as np

from pythesis.utils import PAL, save_fig, set_default_plot_settings

set_default_plot_settings()


# Define the L1 norm and its subgradient
def l1_norm(x):
    return np.abs(x)


def l1_subgradient(x):
    return np.where(x >= 0, 1, np.where(x < 0, -1, 0))


# Generate x values
x = np.linspace(-2, 0, 100)
x = np.concatenate((x, np.linspace(0, 2, 100)))
# Compute L1 norm and its subgradient
y = l1_norm(x)
dy = l1_subgradient(x)

fig, axs = plt.subplots(2, 1, figsize=(1.5, 2), constrained_layout=True, sharex=True)

axs[0].hlines(0, -2, 2, color="lightgray")
axs[0].plot(x, y, c="black")
axs[0].set_ylabel(r"$\lVert \beta\rVert_1$")

axs[1].hlines(0, -2, 2, color="lightgray")
axs[1].plot(x, dy, c="black")
axs[1].set_xlabel(r"$\beta$")
axs[1].set_ylabel(r"$\partial \lVert\beta\rVert_1$")

save_fig("l1norm-subgradients.pdf")
