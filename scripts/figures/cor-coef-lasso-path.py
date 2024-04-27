import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.linear_model import lars_path

from pythesis.utils import FULL_WIDTH, PAL, save_fig

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

n, p = X.shape

lambdas, _, coefs = lars_path(X, y, method="lasso", verbose=True)

lambdas *= n

c = np.zeros_like(coefs)

for j in range(p):
    c[:, j] = X.T @ (y - X @ coefs[:, j])

fig, ax = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3), layout="constrained", sharex=True)

cmap = ListedColormap(PAL)

for i in range(coefs.shape[0]):
    ax[0].plot(lambdas, coefs[i, :], color=PAL[i])
ax[0].set_xlabel(r"$\lambda$")
ax[0].set_ylabel(r"$\hat{\bm{\beta}}$")

ax[1].axline((0, 0), slope=1.0, color="lightgrey", linestyle="dotted")
ax[1].axline((0, 0), slope=-1.0, color="lightgrey", linestyle="dotted")

for i in range(coefs.shape[0]):
    ax[1].plot(lambdas, c[i, :], color=PAL[i])
ax[1].set_xlabel(r"$\lambda$")
ax[1].set_ylabel(r"$\bm{c} = \bm{X}^\intercal(\bm{y} - \bm{X}\hat{\bm{\beta}})$")

ax[0].set_ylim((-820, 820))
ax[1].set_ylim((-1050, 1050))

for a in ax:
    a.get_xaxis().set_inverted(True)

save_fig("cor-coef-lasso-path.pdf")

plt.close("all")
