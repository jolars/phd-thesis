import matplotlib.pyplot as plt
import numpy as np
from libsvmdata import fetch_libsvm
from pyprojroot import here
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from pythesis.utils import FULL_WIDTH, save_fig, set_default_plot_settings

set_default_plot_settings()

X, y = fetch_libsvm("madelon", normalize=False)

X_train, _, y_train, _ = train_test_split(X, y, train_size=0.9, random_state=42)

mm = VarianceThreshold()

X_train = mm.fit_transform(X_train)

n, p = X_train.shape

cor = np.zeros(p)

for j in range(p):
    cor[j] = np.corrcoef(X_train[:, j], y_train)[0, 1]

fig, ax = plt.subplots(
    1, 1, sharex=True, sharey=True, figsize=(FULL_WIDTH, 1.1), layout="constrained"
)

colors = np.where(np.abs(cor) > 0.1, "black", "gray")

ax.hlines(0, 0, p, color="gray")
ax.vlines(np.arange(p), 0, cor, color=colors)
ax.set_ylabel("Correlation")
ax.set_xlabel("Feature")
ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
ax.set_ylim(-0.25, 0.25)
ax.get_xaxis().set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)

save_fig("sparse-signal.pdf")
