import numpy as np
from libsvmdata import fetch_libsvm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import norm, svds
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

news20, _ = fetch_libsvm("news20.binary", normalize=False)
rcv1, _ = fetch_libsvm("rcv1.binary", normalize=False)

pipe = make_pipeline(MaxAbsScaler(), VarianceThreshold())

res = pipe.fit_transform(news20)
X = csc_matrix(res)

L = svds(X, k=1)[1][0]

X_norms = np.zeros(X.shape[1])
for j in range(X.shape[1]):
    X_norms[j] = norm(X[:, j])

# svds(X, k=1)
