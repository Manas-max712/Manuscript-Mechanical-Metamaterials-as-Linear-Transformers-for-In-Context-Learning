import numpy as np


def sigma_from_Q(Q, Sigma_sqrt=None):
    return sigma_from_Q_details(Q, Sigma_sqrt)[1]


def sigma_from_Q_details(Q, Sigma_sqrt=None):
    Qs = 0.5 * (Q + Q.T)
    if Sigma_sqrt is not None:
        Ss = np.asarray(Sigma_sqrt)
        Qr = Ss.T @ Qs @ Ss
        a = -np.trace(Qr) / Q.shape[0]
    else:
        a = -np.trace(Qs) / Q.shape[0]
    Sigma_est = -a * np.linalg.inv(Qs)
    Sigma_ev = np.sort(np.linalg.eigvalsh(0.5 * (Sigma_est + Sigma_est.T)))
    return (a, Sigma_ev)


def sigma_from_G(G):
    Gs = 0.5 * (G + G.T)
    return np.sort(np.linalg.eigvalsh(np.linalg.inv(Gs)))


def dist_to_scaled_identity(M):
    alpha = np.trace(M) / M.shape[0]
    denom = np.linalg.norm(M, 'fro') + 1e-14
    return np.linalg.norm(M - alpha * np.eye(M.shape[0]), 'fro') / denom


def whiten_operator(M, Sigma_sqrt=None):
    Ms = 0.5 * (M + M.T)
    if Sigma_sqrt is None:
        S = np.eye(M.shape[0])
    else:
        S = np.asarray(Sigma_sqrt)
    return S.T @ Ms @ S


def whiten_full_operator(M, factor):
    return np.asarray(factor).T @ np.asarray(M) @ np.asarray(factor)


def symmetric_root(sigma):
    sigma = np.asarray(sigma, dtype=np.float64)
    values, vectors = np.linalg.eigh(0.5 * (sigma + sigma.T))
    if np.any(values <= 0):
        raise ValueError("Covariance must be positive definite")
    return (vectors * np.sqrt(values)) @ vectors.T
