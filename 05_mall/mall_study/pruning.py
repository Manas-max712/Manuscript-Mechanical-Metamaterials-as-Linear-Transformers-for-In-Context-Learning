import numpy as np
from .geometry import B_strain, ofi
from .learning import evaluate_truss
from .mechanics import schur_complement_from_K, stable_pgd_eta, solve_truss
from .tasks import case_setup, make_tasks

STAGES = [0, 1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]


def evaluate_pruning(case, result, test_seed=20001234, n_test=1000):
    _, _, Lx, Lw = case_setup(case)
    tasks = make_tasks(test_seed, n_test, Lx, Lw)
    D = np.where(result["D_best"] > 0.1 * (1.0 + 1e-9), result["D_best"], 0.0)
    K = B_strain @ np.diag(D) @ B_strain.T
    Kc = schur_complement_from_K(K)
    k_min = float(np.linalg.eigvalsh(0.5 * (K + K.T)).min())
    kc_min = float(np.linalg.eigvalsh(0.5 * (Kc + Kc.T)).min())
    gains = [stable_pgd_eta(K, X.T @ X / len(X)) for X, *_ in tasks]
    G = solve_truss(K, np.eye(len(K))[:, ofi])[ofi, :]
    rho_max = max(float(np.max(np.abs(np.linalg.eigvals(
        np.eye(len(ofi)) - eta * G @ (X.T @ X / len(X))))))
        for eta, (X, *_) in zip(gains, tasks))
    rows = []
    for stage in STAGES:
        rows.append(dict(case=case, threshold=0.1, removed_bars=int(np.sum(D == 0)),
                         total_bars=len(D), stage=stage,
                         mse=float(evaluate_truss(D, tasks, stage).mean()),
                         mean_eta=float(np.mean(gains)), train_stages=int(result["n_stages"]),
                         K_min_eig=k_min, Kcond_min_eig=kc_min, max_spectral_radius=rho_max))
    return rows
