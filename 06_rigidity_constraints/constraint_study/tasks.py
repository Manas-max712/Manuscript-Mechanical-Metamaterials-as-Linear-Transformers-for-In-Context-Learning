import numpy as np
from scipy.linalg import solve as sp_solve
from .config import D, N_FEW

def case_setup(name: str, d: int = D):
    if name == "identity":
        Sigma_x = np.eye(d)
        Sigma_w = np.eye(d)
    elif name == "diag":
        Sigma_x = np.diag([4.0, 2.0, 1.0, 0.5, 0.25])
        Sigma_w = np.linalg.inv(Sigma_x)
    elif name == "rotated":
        rng = np.random.RandomState(0)
        gaus = rng.uniform(-1, 1, size=(d, d))
        U_rot, _, _ = np.linalg.svd(gaus)
        D_rot = np.diag([1.0, 1.0, 0.5, 0.25, 1.0])
        UD = U_rot @ D_rot
        Sigma_x = UD @ UD.T
        Sigma_w = np.linalg.inv(Sigma_x)
    else:
        raise ValueError(name)
    L_x = np.linalg.cholesky(Sigma_x + 1e-12 * np.eye(d))
    L_w = np.linalg.cholesky(Sigma_w + 1e-12 * np.eye(d))
    return Sigma_x, Sigma_w, L_x, L_w


def make_tasks(seed: int, n_tasks: int, L_x: np.ndarray, L_w: np.ndarray):
    rng = np.random.RandomState(seed)
    tasks = []
    for _ in range(n_tasks):
        w_t = L_w @ rng.randn(D)
        X_t = rng.randn(N_FEW, D) @ L_x.T
        y_t = X_t @ w_t
        xq_t = L_x @ rng.randn(D)
        yq_t = float(xq_t @ w_t)
        tasks.append((X_t, y_t, xq_t, yq_t, w_t))
    return tasks


def oracle_mse(tasks, lam: float = 1e-2):
    return float(np.mean([
        (float(xq @ sp_solve(X.T @ X / N_FEW + lam * np.eye(D),
                             X.T @ y / N_FEW)) - yq) ** 2
        for X, y, xq, yq, _ in tasks
    ]))
