import numpy as np
from .config import d
from .geometry import B_strain
from .mechanics import schur_complement_from_K, pgd_stability_stats
from .learning import evaluate_truss
from .tasks import oracle_mse


def sym(A):
    return 0.5 * (A + A.T)


def sorted_eigh(A, descending=False):
    vals, vecs = np.linalg.eigh(sym(A))
    order = np.argsort(vals)
    if descending:
        order = order[::-1]
    return vals[order], vecs[:, order]


def principal_angles_deg(U, V):
    svals = np.linalg.svd(U.T @ V, compute_uv=False)
    svals = np.clip(svals, -1.0, 1.0)
    return np.degrees(np.arccos(svals))


def spectral_evidence(G_learned, Sigma_inv, alpha, gap_tol=1e-8):
    G_target = alpha * Sigma_inv
    G_vals_asc, _ = sorted_eigh(G_learned, descending=False)
    T_vals_asc, _ = sorted_eigh(G_target, descending=False)
    eigval_rel_err = (
        np.linalg.norm(G_vals_asc - T_vals_asc) /
        max(np.linalg.norm(T_vals_asc), 1e-14)
    )

    _, G_vecs_desc = sorted_eigh(G_learned, descending=True)
    S_vals_desc, S_vecs_desc = sorted_eigh(Sigma_inv, descending=True)
    scale = max(float(np.max(np.abs(S_vals_desc))), 1e-14)

    ks = np.arange(1, d)
    max_angles = np.full(len(ks), np.nan)
    mean_angles = np.full(len(ks), np.nan)
    gap_safe = np.zeros(len(ks), dtype=bool)
    for i, k in enumerate(ks):
        gap = abs(S_vals_desc[k - 1] - S_vals_desc[k])
        if gap <= gap_tol * scale:
            continue
        angles = principal_angles_deg(G_vecs_desc[:, :k], S_vecs_desc[:, :k])
        max_angles[i] = float(np.max(angles))
        mean_angles[i] = float(np.mean(angles))
        gap_safe[i] = True

    return {
        'G_eigs_sorted': G_vals_asc,
        'target_eigs_sorted': T_vals_asc,
        'eigval_rel_err': float(eigval_rel_err),
        'subspace_ks': ks,
        'subspace_max_angles': max_angles,
        'subspace_mean_angles': mean_angles,
        'subspace_gap_safe': gap_safe,
    }


def analyze_model(D_best, Sigma_x, val_tasks, test_tasks, n_stages):
    K = B_strain @ np.diag(D_best) @ B_strain.T
    K_schur = schur_complement_from_K(K)
    G_learned = np.linalg.inv(K_schur)
    Sigma_inv = np.linalg.inv(Sigma_x)
    alpha = float(np.sum(G_learned * Sigma_inv)
                  / max(np.sum(Sigma_inv * Sigma_inv), 1e-14))
    target = alpha * Sigma_inv
    evidence = spectral_evidence(G_learned, Sigma_inv, alpha)
    angles = evidence["subspace_max_angles"]
    rho_val, eta_val = pgd_stability_stats(D_best, val_tasks)
    rho_test, eta_test = pgd_stability_stats(D_best, test_tasks)
    stages = sorted(set([1, 3, 5, 7, 10, 15, 20, n_stages, 30, 50, 100]))
    per_stage = {s: float(evaluate_truss(D_best, test_tasks, s).mean()) for s in stages}
    return {
        "D_best": D_best, "Sigma_x": Sigma_x, "K_schur": K_schur,
        "G_learned": G_learned, "G_eigs": np.linalg.eigvalsh(G_learned),
        "Sinv_eigs": np.linalg.eigvalsh(Sigma_inv), "alpha": alpha,
        "g_err": float(np.linalg.norm(G_learned - target, "fro")
                       / max(np.linalg.norm(target, "fro"), 1e-14)),
        **evidence,
        "angle_top1_deg": float(angles[0]), "angle_top2_deg": float(angles[1]),
        "mse_final_val": float(evaluate_truss(D_best, val_tasks, n_stages).mean()),
        "mse_final_test": per_stage[n_stages],
        "mse_oracle_val": float(oracle_mse(val_tasks)),
        "mse_oracle_test": float(oracle_mse(test_tasks)),
        "mse_ols_val": float(oracle_mse(val_tasks, lam=0.0)),
        "mse_ols_test": float(oracle_mse(test_tasks, lam=0.0)),
        "rho_val_mean": float(rho_val.mean()), "rho_test_mean": float(rho_test.mean()),
        "rho_test_max": float(rho_test.max()), "eta_test_mean": float(eta_test.mean()),
        "test_per_stage": per_stage, "n_stages": int(n_stages),
    }
