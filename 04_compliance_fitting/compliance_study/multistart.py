from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from .mechanics import (sym, schur_from_D, schur_obj_and_grad,
                        schur_compliance_obj_and_grad)


def make_start(rng: np.random.RandomState, m: int, start_id: int) -> np.ndarray:
    if start_id == 0:
        return np.zeros(m)
    if start_id == 1:
        return np.full(m, 0.5)
    if start_id == 2:
        return np.full(m, -0.5)
    if start_id == 3:
        return rng.randn(m) * 0.25
    if start_id == 4:
        return rng.randn(m) * 0.50
    return rng.randn(m) * 0.75


def run_one_start(lat, target, objective: str, theta0: np.ndarray,
                  theta_bound: float, maxiter: int):
    if objective == "K":
        obj = lambda th: schur_obj_and_grad(th, lat, target)
        K_tgt = target
        G_tgt = np.linalg.inv(sym(K_tgt))
    else:
        obj = lambda th: schur_compliance_obj_and_grad(th, lat, target)
        G_tgt = target
        K_tgt = np.linalg.inv(sym(G_tgt))

    t0 = time.time()
    res = minimize(
        obj,
        np.clip(theta0, -theta_bound, theta_bound),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-theta_bound, theta_bound)] * lat.m,
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-9},
    )
    D = np.exp(res.x)
    K_fit = schur_from_D(lat, D)
    G_fit = np.linalg.inv(sym(K_fit))
    K_err = np.linalg.norm(K_fit - K_tgt, "fro") / max(np.linalg.norm(K_tgt, "fro"), 1e-14)
    G_err = np.linalg.norm(G_fit - G_tgt, "fro") / max(np.linalg.norm(G_tgt, "fro"), 1e-14)
    eigs = np.linalg.eigvalsh(sym(K_fit))
    return {
        "D": D, "K_fit": K_fit, "G_fit": G_fit,
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "nit": int(res.nit),
        "nfev": int(res.nfev),
        "final_obj": float(res.fun),
        "K_rel_err": float(K_err),
        "G_rel_err": float(G_err),
        "D_min": float(D.min()),
        "D_max": float(D.max()),
        "D_ratio": float(D.max() / max(D.min(), 1e-14)),
        "K_fit_min_eig": float(eigs[0]),
        "K_fit_cond": float(eigs[-1] / max(eigs[0], 1e-14)),
        "runtime_sec": float(time.time() - t0),
    }


def summarize_group(rows: list[dict], metric: str):
    vals = np.array([float(r[metric]) for r in rows], dtype=float)
    best = vals.min()
    return {
        f"{metric}_best": float(best),
        f"{metric}_median": float(np.median(vals)),
        f"{metric}_mean": float(vals.mean()),
        f"{metric}_std": float(vals.std(ddof=0)),
        f"{metric}_p10": float(np.quantile(vals, 0.10)),
        f"{metric}_p90": float(np.quantile(vals, 0.90)),
        f"{metric}_worst": float(vals.max()),
        f"{metric}_worst_over_best": float(vals.max() / max(best, 1e-300)),
        f"{metric}_median_over_best": float(np.median(vals) / max(best, 1e-300)),
    }
