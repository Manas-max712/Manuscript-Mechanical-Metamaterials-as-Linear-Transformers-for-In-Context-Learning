import numpy as np
from scipy.linalg import solve as sp_solve
from .config import D, NY, ETA_CAP, ETA_SAFETY
from .geometry import make_lattice, schur_from_D, sym

class MallSystem:
    def __init__(self):
        self.lat = make_lattice(D, NY)
        self.B = self.lat.B
        self.m = self.lat.m
        self.nf = self.lat.nf
        self.ofi = self.lat.ofi

    def K_from_D(self, D_vec):
        return self.B @ np.diag(D_vec) @ self.B.T

    def solve_truss(self, K, f_applied, reg: float = 1e-10):
        return sp_solve(K + reg * np.eye(K.shape[0]), f_applied)

    def apply_force_at_output(self, F_vec):
        f = np.zeros(self.nf)
        for k in range(D):
            f[self.ofi[k]] = F_vec[k]
        return f

    def read_output(self, u):
        return u[self.ofi]

    def measure_strains(self, u):
        return self.B.T @ u

    def stable_eta(self, K, Sh):
        K_schur = schur_from_D(self.lat, np.ones(self.m))
        K_schur = self.schur_from_K(K)
        G = np.linalg.inv(sym(K_schur))
        evals, evecs = np.linalg.eigh(sym(G))
        G_sqrt = evecs @ np.diag(np.sqrt(np.clip(evals, 1e-14, None))) @ evecs.T
        A = G_sqrt @ Sh @ G_sqrt
        lam_max = max(float(np.linalg.eigvalsh(sym(A)).max()), 1e-14)
        return min(ETA_CAP, ETA_SAFETY * 2.0 / lam_max)

    def schur_from_K(self, K):
        K_oo = K[np.ix_(self.lat.ofi, self.lat.ofi)]
        K_oi = K[np.ix_(self.lat.ofi, self.lat.int_idx)]
        K_io = K[np.ix_(self.lat.int_idx, self.lat.ofi)]
        K_ii = K[np.ix_(self.lat.int_idx, self.lat.int_idx)]
        return K_oo - K_oi @ sp_solve(
            K_ii + 1e-10 * np.eye(len(self.lat.int_idx)), K_io)

    def forward_pgd(self, D_vec, X, y, n_stages):
        n = len(X)
        F = X.T @ y / n
        Sh = X.T @ X / n
        K = self.K_from_D(D_vec)
        eta = self.stable_eta(K, Sh)
        w = np.zeros(D)
        eps_fwd_all = []
        for s in range(n_stages):
            r = F - Sh @ w if s > 0 else F
            u = self.solve_truss(K, self.apply_force_at_output(r))
            eps_fwd_all.append(self.measure_strains(u))
            w += eta * self.read_output(u)
        return w, eps_fwd_all, F, Sh, eta

    def adjoint_pgd_query(self, D_vec, eps_fwd_all, w_hat, Xq, yq,
                          F, Sh, n_stages, eta):
        K = self.K_from_D(D_vec)
        errs = Xq @ w_hat - yq
        delta = Xq.T @ errs / len(yq)
        mse = float(np.mean(errs ** 2))
        grad_total = np.zeros(self.m)
        for s in range(n_stages - 1, -1, -1):
            u_adj = self.solve_truss(
                K, self.apply_force_at_output(-eta * delta))
            eps_adj = self.measure_strains(u_adj)
            grad_total += eps_fwd_all[s] * eps_adj
            if s > 0:
                delta = delta + Sh @ self.read_output(u_adj)
        return grad_total, 0.5 * mse, mse

    def evaluate_truss(self, D_eval, tasks, n_stages_eval):
        mses = []
        for X, y, xq, yq, _ in tasks:
            w_hat, _, _, _, _ = self.forward_pgd(D_eval, X, y, n_stages_eval)
            mses.append((float(xq @ w_hat) - yq) ** 2)
        return np.asarray(mses, dtype=float)

    def learned_G(self, D_vec):
        Ks = schur_from_D(self.lat, D_vec)
        return np.linalg.inv(sym(Ks))
