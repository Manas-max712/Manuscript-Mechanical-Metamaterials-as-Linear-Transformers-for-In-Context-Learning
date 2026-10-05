import numpy as np
from scipy.linalg import solve as sp_solve
from .config import d, eta_pgd_safety, eta_pgd_cap
from .geometry import nf, ofi, int_idx, B_strain


def solve_truss(K, f_applied, reg=1e-10):
    return sp_solve(K + reg * np.eye(K.shape[0]), f_applied)


def apply_force_at_output(F_vec):
    f = np.zeros(nf)
    for k in range(d):
        f[ofi[k]] = F_vec[k]
    return f


def read_output(u):
    return u[ofi]


def measure_strains(u):
    return B_strain.T @ u


def schur_complement_from_K(K):
    K_oo = K[np.ix_(ofi, ofi)]
    K_oi = K[np.ix_(ofi, int_idx)]
    K_io = K[np.ix_(int_idx, ofi)]
    K_ii = K[np.ix_(int_idx, int_idx)]
    return K_oo - K_oi @ sp_solve(K_ii + 1e-10 * np.eye(K_ii.shape[0]), K_io)


def stable_pgd_eta(K, Sh, safety=eta_pgd_safety, eta_cap=eta_pgd_cap):
    K_schur = schur_complement_from_K(K)
    G = np.linalg.inv(0.5 * (K_schur + K_schur.T))
    G_evals, G_evecs = np.linalg.eigh(0.5 * (G + G.T))
    G_sqrt = G_evecs @ np.diag(np.sqrt(np.clip(G_evals, 1e-14, None))) @ G_evecs.T
    A = G_sqrt @ Sh @ G_sqrt
    lam_max = max(float(np.linalg.eigvalsh(0.5 * (A + A.T)).max()), 1e-14)
    return min(eta_cap, safety * 2.0 / lam_max)


def pgd_rho_for_task(D, X, eta=None):
    Sh = X.T @ X / len(X)
    K = B_strain @ np.diag(D) @ B_strain.T
    K_schur = schur_complement_from_K(K)
    G = np.linalg.inv(0.5 * (K_schur + K_schur.T))
    eta_use = stable_pgd_eta(K, Sh) if eta is None else eta
    return max(abs(np.linalg.eigvals(np.eye(d) - eta_use * G @ Sh))), eta_use


def pgd_stability_stats(D, tasks):
    rhos = []
    etas = []
    for X, _, _, _, _ in tasks:
        rho, eta = pgd_rho_for_task(D, X)
        rhos.append(rho)
        etas.append(eta)
    return np.asarray(rhos, dtype=float), np.asarray(etas, dtype=float)
