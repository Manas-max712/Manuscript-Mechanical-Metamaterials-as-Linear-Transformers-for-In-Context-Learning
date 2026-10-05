from __future__ import annotations

import numpy as np


def case_sigma(case: str, d: int):
    if case == "identity":
        return np.eye(d)
    if case == "diag":
        vals = np.geomspace(4.0, 0.25, d)
        return np.diag(vals)
    if case == "rotated":
        rng = np.random.RandomState(17 + d)
        A = rng.randn(d, d)
        Q, _ = np.linalg.qr(A)
        vals = np.ones(d)
        k = min(4, d)
        vals[-k:] = np.geomspace(0.5, 0.0625, k)
        return Q @ np.diag(vals) @ Q.T
    if case == "random_spd":
        rng = np.random.RandomState(100 + d)
        A = rng.randn(d, d)
        return A @ A.T / d + 0.25 * np.eye(d)
    raise ValueError(f"unknown case: {case}")


def ideal_targets(case: str, d: int):
    Sigma = case_sigma(case, d)
    G_tgt = np.linalg.inv(Sigma)
    K_tgt = np.linalg.inv(G_tgt)
    return Sigma, G_tgt, K_tgt
