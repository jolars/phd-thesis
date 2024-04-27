import numpy as np
from matplotlib import pyplot as plt

from pythesis.utils import PAL, save_fig, set_default_plot_settings

set_default_plot_settings()


def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


x = np.linspace(-2, 2, 100)

y = soft_threshold(x, 1)
y2 = soft_threshold(x, 0.5)
y3 = soft_threshold(x, 0)

fig, ax = plt.subplots(
    1, 3, figsize=(4.5, 1.7), layout="constrained", sharex=True, sharey=True
)

for a in ax:
    a.set_aspect("equal")

ax[0].hlines(0, -2, 2, "lightgrey")
ax[0].plot(x, y, "k-")
ax[0].set_ylabel(r"$\operatorname{S}_{\lambda}(u)$")
ax[0].set_title(r"$\lambda = 1$")

ax[1].hlines(0, -2, 2, "lightgrey")
ax[1].plot(x, y2, "k-")
ax[1].set_xlabel(r"$u$")
ax[1].set_title(r"$\lambda = 1/2$")

ax[2].hlines(0, -2, 2, "lightgrey")
ax[2].plot(x, y3, "k-")
ax[2].set_title(r"$\lambda = 0$")

save_fig("soft-thresholding.pdf")
