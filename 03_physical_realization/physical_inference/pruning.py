import numpy as np
from scipy.linalg import solve as sp_solve
from .geometry import B_STRAIN, OFI, INT_IDX


def stiffness_matrix(D_bar):
    return (B_STRAIN * D_bar[None, :]) @ B_STRAIN.T


def schur_from_stiffness(D_bar, tol=1e-10):
    K = stiffness_matrix(D_bar)
    eig_K_min = float(np.linalg.eigvalsh(0.5 * (K + K.T)).min())
    if eig_K_min <= tol:
        raise np.linalg.LinAlgError(f"free stiffness matrix is singular/indefinite; min eig={eig_K_min:.3e}")

    K_oo = K[np.ix_(OFI, OFI)]
    K_oi = K[np.ix_(OFI, INT_IDX)]
    K_io = K[np.ix_(INT_IDX, OFI)]
    K_ii = K[np.ix_(INT_IDX, INT_IDX)]
    eig_Kii_min = float(np.linalg.eigvalsh(0.5 * (K_ii + K_ii.T)).min())
    if eig_Kii_min <= tol:
        raise np.linalg.LinAlgError(f"interior stiffness block is singular/indefinite; min eig={eig_Kii_min:.3e}")

    K_schur = K_oo - K_oi @ sp_solve(K_ii, K_io, assume_a="sym")
    eig_schur_min = float(np.linalg.eigvalsh(0.5 * (K_schur + K_schur.T)).min())
    if eig_schur_min <= tol:
        raise np.linalg.LinAlgError(f"Schur stiffness is singular/indefinite; min eig={eig_schur_min:.3e}")
    return 0.5 * (K_schur + K_schur.T)


def compliance_from_stiffness(D_bar):
    K_schur = schur_from_stiffness(D_bar)
    return np.linalg.inv(K_schur), K_schur


def threshold_prune(D_bar, log10_threshold):
    threshold = 10.0 ** log10_threshold
    keep = D_bar > threshold
    return np.where(keep, D_bar, 0.0), int((~keep).sum()), threshold
