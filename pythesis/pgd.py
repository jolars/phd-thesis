import numpy as np


def get_limit(x, f=0.1):
    r = np.max(x) - np.min(x)
    return np.max(np.abs(x)) + f * r


def st(x, lmbd):
    return np.sign(x) * np.maximum(np.abs(x) - lmbd, 0)


def pgd(x, y, lmbd, w0, max_it=100):
    p = x.shape[1]

    w = w0.copy()

    w_history = np.zeros((p, max_it + 1))
    grad_history = np.zeros((p, max_it))
    prox_history = np.zeros((p, max_it))

    L = np.linalg.norm(x, ord=2) ** 2

    it = 0

    for it in range(max_it):
        w_old = np.copy(w)

        w_history[:, it] = w_old

        grad_step = w - x.T @ (x @ w - y) / L
        w = st(grad_step, lmbd / L)

        grad_history[:, it] = grad_step
        prox_history[:, it] = w

        if np.linalg.norm(w - w_old) < 1e-8:
            break

    return w_history[:, :it], grad_history[:, :it], prox_history[:, :it]
