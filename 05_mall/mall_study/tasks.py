import numpy as np
from scipy.linalg import solve as sp_solve
from .config import d, N_few


def case_setup(name, d_=5):
    if name == "identity":
        Sigma_x = np.eye(d_); Sigma_w = np.eye(d_)
    elif name == "diag":
        Sigma_x = np.diag([4.0, 2.0, 1.0, 0.5, 0.25])
        Sigma_w = np.linalg.inv(Sigma_x)
    elif name == "rotated":
        rng = np.random.RandomState(0)
        gaus = rng.uniform(-1, 1, size=(d_, d_))
        U_rot, _, _ = np.linalg.svd(gaus)
        D_rot = np.diag([1.0, 1.0, 0.5, 0.25, 1.0])
        UD = U_rot @ D_rot
        Sigma_x = UD @ UD.T
        Sigma_w = np.linalg.inv(Sigma_x)
    else:
        raise ValueError(name)
    L_x = np.linalg.cholesky(Sigma_x + 1e-12 * np.eye(d_))
    L_w = np.linalg.cholesky(Sigma_w + 1e-12 * np.eye(d_))
    return Sigma_x, Sigma_w, L_x, L_w


def make_tasks(seed, n_tasks, L_x, L_w):
    rng = np.random.RandomState(seed)
    tasks = []
    for _ in range(n_tasks):
        w_t  = L_w @ rng.randn(d)
        X_t  = rng.randn(N_few, d) @ L_x.T
        y_t  = X_t @ w_t
        xq_t = L_x @ rng.randn(d)
        yq_t = float(xq_t @ w_t)
        tasks.append((X_t, y_t, xq_t, yq_t, w_t))
    return tasks


def oracle_mse(tasks, lam=1e-2):
    return np.mean([
        (float(xq @ sp_solve(X.T @ X / N_few + lam * np.eye(d), X.T @ y / N_few)) - yq) ** 2
        for X, y, xq, yq, _ in tasks
    ])
