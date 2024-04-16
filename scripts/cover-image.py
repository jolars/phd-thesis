import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from pyprojroot import here
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet


def fit_lasso(x, y, alpha, lam):
    model = ElasticNet(alpha=lam, l1_ratio=alpha, max_iter=10_000)
    model.fit(x, y)
    return model.coef_


diabetes = load_diabetes()

x = diabetes.data
y = diabetes.target

n = x.shape[0]
p = x.shape[1]

lambda_max = np.max(np.abs(x.T @ y) / n)

n_lambda = 10
n_alpha = 10

lambda_seq = np.geomspace(lambda_max, lambda_max * 1e-5, n_lambda)
alpha_seq = np.linspace(0.01, 1, n_alpha)
lambda_grid, alpha_grid = np.meshgrid(lambda_seq, alpha_seq)

lambdas, alphas = lambda_grid.ravel(), alpha_grid.ravel()

coefs = np.array([fit_lasso(x, y, alpha, lam) for lam, alpha in zip(lambdas, alphas)])

coefs = np.zeros((p, n_alpha, n_lambda))

for i, j in np.ndindex(n_alpha, n_lambda):
    coefs[:, i, j] = fit_lasso(x, y, alpha_seq[i], lambda_seq[j])

j = 0

fig = plt.figure(figsize=(8, 3), layout="constrained")
ax = fig.add_subplot(projection="3d")

# cm = mpl.colormaps["Spectral"]
cm = mpl.colormaps["BrBG"]

ind = [1, 2, 3, 6, 8, 9]

coefs_selected = coefs[ind, :, :]
vmax = np.max(np.abs(coefs_selected.ravel()))
vmin = -vmax

for j in ind:
    ax.plot_surface(
        np.log(lambda_grid),
        alpha_grid,
        coefs[j, :, :],
        vmin=vmin,
        vmax=vmax,
        # color="orange",
        cmap=cm,
        alpha=0.6,
    )

plt.axis("off")

ax.set_proj_type("persp", focal_length=2)
ax.view_init(elev=20, azim=290, roll=0)

plt.savefig(here() / "figures" / "cover-image.pdf", bbox_inches="tight")
