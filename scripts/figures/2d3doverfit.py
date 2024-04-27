import os

import matplotlib.pyplot as plt
import numpy as np
from pyprojroot import here

from pythesis.utils import FULL_WIDTH, save_fig, set_default_plot_settings

set_default_plot_settings()

x1_full = np.array([0, 1])
x2_full = np.array([0, 1])
y_full = np.array([0, 1])
x1 = np.array([0.2, 0.8])
x2 = np.array([0.2, 0.8])
y = np.array([0.2, 0.8])

fig = plt.figure(figsize=(2.1, 2.1), layout="compressed")
ax1 = fig.add_subplot(111, box_aspect=1)

ax1.plot(x1_full, y_full, color="black")
ax1.scatter(x1, y, color="black")
ax1.set_xlabel(r"$\boldsymbol{x}_1$")
ax1.set_ylabel(r"$\boldsymbol{y}$")

path = here("figures") / "overfit-2d.pdf"

save_fig(path)

fig = plt.figure(figsize=(2, 2), layout="constrained")
ax2 = plt.axes(projection="3d", computed_zorder=False)

for i in range(-1, 2):
    X1, X2 = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    Y = i * X1 + (1 - i) * X2
    ax2.plot_surface(X1, X2, Y, alpha=0.3, zorder=1)

ax2.plot(x1_full, x2_full, y_full, "-", color="black")
ax2.plot(x1, x2, y, "o", color="black")

ax2.set_xlabel(r"$\boldsymbol{x}_1$")
ax2.set_ylabel(r"$\boldsymbol{x}_2$")
ax2.set_zlabel(r"$\boldsymbol{y}$", labelpad=0)

path = here("figures") / "overfit-3d.pdf"

save_fig("overfit-3d.pdf", pad_inches=1)

plt.close("all")
