import numpy as np
from scipy.linalg import solve as sp_solve
from .geometry import D

N_CONTEXT = 20
N_TEST = 300
ADAPTIVE_ALPHA_SAFETY = 0.95
ADAPTIVE_ALPHA_CAP = 1.0


def synthetic_tasks(Sigma, n_tasks=N_TEST, n_context=N_CONTEXT, seed=0):
    rng = np.random.RandomState(seed)
    Lx = np.linalg.cholesky(Sigma + 1e-12 * np.eye(D))
    Lw = np.linalg.cholesky(np.linalg.inv(Sigma) + 1e-12 * np.eye(D))
    tasks = []
    for _ in range(n_tasks):
        w_true = Lw @ rng.randn(D)
        X = rng.randn(n_context, D) @ Lx.T
        y = X @ w_true
        Sh = X.T @ X / n_context
        F = X.T @ y / n_context
        w_star = sp_solve(Sh + 1e-12 * np.eye(D), F)
        tasks.append((Sh, F, w_star))
    return tasks


def adaptive_alpha(G_phys, Sh):
    G_sym = 0.5 * (G_phys + G_phys.T)
    eigvals, eigvecs = np.linalg.eigh(G_sym)
    eigvals = np.clip(eigvals, 1e-14, None)
    G_half = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
    M = G_half @ Sh @ G_half
    lam_max = float(np.linalg.eigvalsh(0.5 * (M + M.T)).max())
    return min(ADAPTIVE_ALPHA_CAP, ADAPTIVE_ALPHA_SAFETY * 2.0 / max(lam_max, 1e-14))


def power_lambda_max(G: np.ndarray, Sh: np.ndarray, n_iter: int,
                     seed: int = 0) -> float:
    rng = np.random.RandomState(seed)
    v = rng.randn(G.shape[0])
    v /= max(np.linalg.norm(v), 1e-14)
    for _ in range(n_iter):
        q = G @ (Sh @ v)
        nq = np.linalg.norm(q)
        if nq < 1e-14 or not np.isfinite(nq):
            return 1e-14
        v = q / nq
    q = G @ (Sh @ v)
    lam = float(v @ q) / max(float(v @ v), 1e-14)
    return max(lam, 1e-14)


def power_alpha(G: np.ndarray, Sh: np.ndarray, n_iter: int, safety: float,
                cap: float, seed: int, safety_scale: float) -> float:
    lam = power_lambda_max(G, Sh, n_iter=n_iter, seed=seed)
    return min(cap, safety_scale * safety * 2.0 / lam)


def evaluate_curve(G_phys, tasks, alpha_fn, max_stage):
    errors = np.zeros(max_stage + 1, dtype=float)
    alphas = []
    capped = 0
    for task_index, (Sh, F, w_star) in enumerate(tasks):
        alpha = float(alpha_fn(Sh, task_index))
        alphas.append(alpha)
        w = np.zeros(D)
        denom = float(np.dot(w_star, w_star)) + 1e-14
        errors[0] += float(np.dot(w - w_star, w - w_star)) / denom
        for stage in range(1, max_stage + 1):
            residual = F - Sh @ w
            w = w + alpha * (G_phys @ residual)
            value = float(np.dot(w - w_star, w - w_star)) / denom
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite coefficient error during inference")
            capped += int(value > 1e6)
            errors[stage] += min(value, 1e6)
    return errors / len(tasks), np.asarray(alphas, dtype=float), capped
