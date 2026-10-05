import numpy as np
from scipy.linalg import solve as sp_solve
from scipy.optimize import minimize
from .geometry import B_STRAIN, EDGES, OFI, INT_IDX


def schur_from_stiffness(D_bar):
    K = B_STRAIN @ np.diag(D_bar) @ B_STRAIN.T
    K_oo = K[np.ix_(OFI, OFI)]
    K_oi = K[np.ix_(OFI, INT_IDX)]
    K_io = K[np.ix_(INT_IDX, OFI)]
    K_ii = K[np.ix_(INT_IDX, INT_IDX)]
    return K_oo - K_oi @ sp_solve(K_ii + 1e-12 * np.eye(len(INT_IDX)), K_io)


def schur_obj_and_grad(theta, K_tgt):
    D_bar = np.exp(theta)
    K = B_STRAIN @ np.diag(D_bar) @ B_STRAIN.T
    K_oo = K[np.ix_(OFI, OFI)]
    K_oi = K[np.ix_(OFI, INT_IDX)]
    K_io = K[np.ix_(INT_IDX, OFI)]
    K_ii = K[np.ix_(INT_IDX, INT_IDX)]
    W = sp_solve(K_ii + 1e-12 * np.eye(len(INT_IDX)), K_io)
    R = K_oo - K_oi @ W - K_tgt
    value = np.linalg.norm(R, "fro") ** 2

    b_o = B_STRAIN[OFI, :]
    b_i = B_STRAIN[INT_IDX, :]
    d_all = b_o - W.T @ b_i
    grad_D = 2.0 * np.sum(d_all * (R @ d_all), axis=0)
    return float(value), grad_D * D_bar


def fit_physical_lattice(K_tgt, seed=0, theta_bound=6.0, n_starts=10):
    rng = np.random.RandomState(seed)
    starts = [
        np.zeros(len(EDGES)),
        np.full(len(EDGES), 0.5),
        np.full(len(EDGES), -0.5),
        rng.randn(len(EDGES)) * 0.25,
        rng.randn(len(EDGES)) * 0.50,
    ]
    while len(starts) < n_starts:
        starts.append(rng.randn(len(EDGES)) * 0.75)

    best_fun = np.inf
    best_theta = starts[0]
    for th0 in starts:
        th0 = np.clip(th0, -theta_bound, theta_bound)
        res = minimize(
            lambda th: schur_obj_and_grad(th, K_tgt),
            th0,
            method="L-BFGS-B",
            jac=True,
            bounds=[(-theta_bound, theta_bound)] * len(EDGES),
            options={"maxiter": 6000, "ftol": 1e-14, "gtol": 1e-11},
        )
        if res.fun < best_fun:
            best_fun = float(res.fun)
            best_theta = res.x.copy()

    D_bar = np.exp(best_theta)
    K_fit = schur_from_stiffness(D_bar)
    rel_err = np.linalg.norm(K_fit - K_tgt, "fro") / np.linalg.norm(K_tgt, "fro")
    return D_bar, K_fit, rel_err
