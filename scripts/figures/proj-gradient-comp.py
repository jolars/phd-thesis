import matplotlib.pyplot as plt
import numpy as np
from benchopt.datasets.simulated import make_correlated_data
from labellines import labelLines

from pythesis.lasso import lasso_gd
from pythesis.utils import PAL, save_fig, set_default_plot_settings

set_default_plot_settings()

n = 1000
m = 100

A, b, x_true = make_correlated_data(
    m, n, rho=0.9, snr=1.0, density=0.01, random_state=1541
)
lambda_max = np.linalg.norm(A.T @ b, np.inf)
lam = lambda_max * 0.5

x_hist, f_hist, t_hist = lasso_gd(A, b, lam, method="prox", tol=1e-6)

tau = np.linalg.norm(x_hist[-1], 1)

x_hist_proj, f_hist_proj, t_hist_proj = lasso_gd(
    A, b, lam, tau, method="proj", tol=1e-6
)

opt = min(np.min(f_hist), np.min(f_hist_proj))

fig, ax = plt.subplots(1, 2, figsize=(4, 1.8), sharey=True, layout="constrained")

ax[0].semilogy(f_hist - opt, label="Proximal GD", color=PAL[0])
ax[0].semilogy(f_hist_proj - opt, label="Projected GD", color=PAL[1])

ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Suboptimality")

ax[1].semilogy(t_hist, f_hist - opt, label="Proximal", color=PAL[0])
ax[1].semilogy(t_hist_proj, f_hist_proj - opt, label="Projected", color=PAL[1])

ax[1].set_xlabel("Time (s)")

# ax[1].legend()


for a in ax:
    a.set_ylim(1e-7 * 0.9, None)

labelLines(ax[1].get_lines(), xvals=[0.1, 0.2], align=True)

save_fig("lasso-projgrad-comp.pdf")
