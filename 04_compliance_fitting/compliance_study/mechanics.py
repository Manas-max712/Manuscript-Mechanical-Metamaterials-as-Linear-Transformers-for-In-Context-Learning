from __future__ import annotations

import numpy as np
from scipy.linalg import solve as sp_solve

from .geometry import Lattice


def sym(A):
    return 0.5 * (A + A.T)


def schur_from_D(lat: Lattice, D: np.ndarray):
    K = lat.B @ np.diag(D) @ lat.B.T
    K_oo = K[np.ix_(lat.ofi, lat.ofi)]
    K_oi = K[np.ix_(lat.ofi, lat.int_idx)]
    K_io = K[np.ix_(lat.int_idx, lat.ofi)]
    K_ii = K[np.ix_(lat.int_idx, lat.int_idx)]
    return K_oo - K_oi @ sp_solve(
        K_ii + 1e-12 * np.eye(len(lat.int_idx)), K_io)


def schur_obj_and_grad(theta: np.ndarray, lat: Lattice, K_tgt: np.ndarray):
    D = np.exp(theta)
    K = lat.B @ np.diag(D) @ lat.B.T
    K_oo = K[np.ix_(lat.ofi, lat.ofi)]
    K_oi = K[np.ix_(lat.ofi, lat.int_idx)]
    K_io = K[np.ix_(lat.int_idx, lat.ofi)]
    K_ii = K[np.ix_(lat.int_idx, lat.int_idx)]
    W = sp_solve(K_ii + 1e-12 * np.eye(len(lat.int_idx)), K_io)
    R = K_oo - K_oi @ W - K_tgt
    value = float(np.linalg.norm(R, "fro") ** 2)
    b_o = lat.B[lat.ofi, :]
    b_i = lat.B[lat.int_idx, :]
    d_all = b_o - W.T @ b_i
    grad_D = 2.0 * np.sum(d_all * (R @ d_all), axis=0)
    return value, grad_D * D


def schur_compliance_obj_and_grad(theta: np.ndarray, lat: Lattice,
                                  G_tgt: np.ndarray):
    D = np.exp(theta)
    K = lat.B @ np.diag(D) @ lat.B.T
    K_oo = K[np.ix_(lat.ofi, lat.ofi)]
    K_oi = K[np.ix_(lat.ofi, lat.int_idx)]
    K_io = K[np.ix_(lat.int_idx, lat.ofi)]
    K_ii = K[np.ix_(lat.int_idx, lat.int_idx)]
    W = sp_solve(K_ii + 1e-12 * np.eye(len(lat.int_idx)), K_io)
    K_s = K_oo - K_oi @ W
    G_s = np.linalg.inv(sym(K_s))
    R = G_s - G_tgt
    value = float(np.linalg.norm(R, "fro") ** 2)

    grad_K = -2.0 * G_s @ sym(R) @ G_s
    b_o = lat.B[lat.ofi, :]
    b_i = lat.B[lat.int_idx, :]
    d_all = b_o - W.T @ b_i
    grad_D = np.sum(d_all * (grad_K @ d_all), axis=0)
    return value, grad_D * D
