import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets, linear_model

from pythesis.utils import FULL_WIDTH, PAL, save_fig, set_default_plot_settings

set_default_plot_settings()

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

alphas, _, coefs = linear_model.lars_path(X, y, method="lasso", verbose=True)

features = [
    "Age",
    "Sex",
    "BMI",
    "Blood pressure",
    "Total serum cholesterol",
    "Low-density lipoproteins",
    "High-density lipoproteins (HDL)",
    "Total cholesterol / HDL",
    "Serum triglycerides",
    "Blood sugar",
]


xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

fig = plt.figure(figsize=(4.5, 5), layout="constrained")
ax = fig.gca()

ymin = np.min(coefs)
ymax = np.max(coefs)
plt.vlines(xx, ymin, ymax, linestyle="dotted", color="lightgrey")

cmap = ListedColormap(PAL)

for i in range(coefs.shape[0]):
    plt.plot(xx, coefs[i], color=PAL[i])

plt.xlabel(r"$t / \max t$")
plt.ylabel(r"$\hat{\boldsymbol{\beta}}$")

for i, label in enumerate(features):
    y_pos = coefs[i, -1]

    nudge = 0.0

    # if i == 9:
    #     nudge = -10
    # elif i == 6:
    #     nudge = 10
    # elif label == "BMI":
    #     nudge = 5
    # elif i == 5:
    #     nudge = -5

    plt.text(
        1.02,
        y_pos + nudge,
        label,
        fontsize="small",
        verticalalignment="center",
        color=PAL[i],
    )


ax.spines[["right", "top"]].set_visible(False)

save_fig("lasso-path.pdf")

plt.close("all")
