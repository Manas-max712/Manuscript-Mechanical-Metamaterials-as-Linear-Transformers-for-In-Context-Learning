import numpy as np
from .config import d
from .geometry import m, B_strain
from .mechanics import (stable_pgd_eta, apply_force_at_output, solve_truss,
                        measure_strains, read_output)


def forward_pgd(D, X, y, n_stages):
    n   = len(X)
    F   = X.T @ y / n
    Sh  = X.T @ X / n
    K   = B_strain @ np.diag(D) @ B_strain.T
    eta = stable_pgd_eta(K, Sh)
    w   = np.zeros(d)
    eps_fwd_all = []
    for s in range(n_stages):
        r     = F - Sh @ w if s > 0 else F
        f_app = apply_force_at_output(r)
        u     = solve_truss(K, f_app)
        eps_fwd_all.append(measure_strains(u))
        w    += eta * read_output(u)
    return w, eps_fwd_all, F, Sh, eta


def adjoint_pgd_query(D, eps_fwd_all, w_hat, Xq, yq, F, Sh, n_stages, eta):
    K = B_strain @ np.diag(D) @ B_strain.T
    preds = Xq @ w_hat
    errs  = preds - yq
    delta = Xq.T @ errs / len(yq)
    objective_for_gradient = 0.5 * np.mean(errs ** 2)
    reported_train_mse = np.mean(errs ** 2)
    grad_total = np.zeros(m)
    for s in range(n_stages - 1, -1, -1):
        f_adj = apply_force_at_output(-eta * delta)
        u_adj = solve_truss(K, f_adj)
        eps_adj = measure_strains(u_adj)
        grad_total += eps_fwd_all[s] * eps_adj
        if s > 0:
            delta = delta + Sh @ read_output(u_adj)
    return grad_total, objective_for_gradient, reported_train_mse


def evaluate_truss(D_eval, tasks, n_stages_eval):
    mses = []
    for X, y, xq, yq, _ in tasks:
        w_hat, _, _, _, _ = forward_pgd(D_eval, X, y, n_stages_eval)
        mses.append((float(xq @ w_hat) - yq) ** 2)
    return np.array(mses)
