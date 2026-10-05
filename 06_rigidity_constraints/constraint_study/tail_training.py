import os
import time
import numpy as np
from .config import D, CASE_TRAIN_CONFIG, clip_from_ratio
from .geometry import sym
from .mechanics import MallSystem
from .tasks import case_setup, make_tasks, oracle_mse
from .files import load_original_D, load_tail_masks
from .tail_checkpoints import checkpoint_path, save_checkpoint, load_checkpoint

def masks_from_tails(D_init: np.ndarray, low_fraction: float, high_fraction: float):
    n = len(D_init)
    order = np.argsort(D_init)
    n_low = int(np.floor(low_fraction * n))
    n_high = int(np.floor(high_fraction * n))
    low_mask = np.zeros(n, dtype=bool)
    high_mask = np.zeros(n, dtype=bool)
    if n_low > 0:
        low_mask[order[:n_low]] = True
    if n_high > 0:
        high_candidates = order[n_low:]
        high_mask[high_candidates[-min(n_high, len(high_candidates)):]] = True
    train_mask = ~(low_mask | high_mask)
    return low_mask, high_mask, train_mask


def enforce_quantized_values(D_vec, train_mask, low_mask, high_mask,
                             middle_range, soft_value, rigid_value):
    D_vec[train_mask] = np.clip(
        D_vec[train_mask], middle_range[0], middle_range[1])
    D_vec[low_mask] = soft_value
    D_vec[high_mask] = rigid_value
    return D_vec


def train_tail_quantized(case: str, args, stop_after=None):
    sysm = MallSystem()
    cfg = CASE_TRAIN_CONFIG[case].copy()
    cfg["N_epochs"] = int(args.fine_tune_epochs)
    if args.train_tasks is not None:
        cfg["N_train_tasks"] = args.train_tasks

    middle_range = clip_from_ratio(args.middle_stiffness_ratio)
    original_D = load_original_D(case, args.mall_root)
    if len(original_D) != sysm.m:
        raise ValueError(
            f"D length mismatch for {case}: got {len(original_D)}, expected {sysm.m}")

    low_mask, high_mask, train_mask = masks_from_tails(
        original_D, args.low_fraction, args.high_fraction)
    if getattr(args, "tail_mask_root", None) is not None:
        low_mask, high_mask, train_mask = load_tail_masks(
            args.tail_mask_root, case, args.middle_stiffness_ratio, original_D)
    D_vec = original_D.copy()
    D_vec = enforce_quantized_values(
        D_vec, train_mask, low_mask, high_mask,
        middle_range, args.soft_value, args.rigid_value)

    Sigma_x, _, L_x, L_w = case_setup(case)
    val_tasks = make_tasks(seed=9999, n_tasks=min(args.eval_tasks, 200), L_x=L_x, L_w=L_w)
    test_tasks = make_tasks(seed=20001234, n_tasks=args.eval_tasks, L_x=L_x, L_w=L_w)
    mse_oracle_test = oracle_mse(test_tasks)

    beta1 = 0.9
    beta2 = 0.999
    eps_adam = 1e-8
    m_adam = np.zeros(sysm.m)
    v_adam = np.zeros(sysm.m)
    t_adam = 0
    best_val = np.inf
    best_D = D_vec.copy()
    best_epoch = 0
    train_mses = []
    val_mses = []
    total_pushes = 0
    start_epoch = 1
    t0 = time.time()

    ckpt_path = checkpoint_path(args, case)
    if args.resume and os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path)
        D_vec = ckpt["D_vec"]
        train_mask = ckpt["train_mask"]
        low_mask = ckpt["low_mask"]
        high_mask = ckpt["high_mask"]
        m_adam = ckpt["m_adam"]
        v_adam = ckpt["v_adam"]
        t_adam = ckpt["t_adam"]
        best_val = ckpt["best_val"]
        best_D = ckpt["best_D"]
        best_epoch = ckpt["best_epoch"]
        train_mses = ckpt["train_mses"]
        val_mses = ckpt["val_mses"]
        total_pushes = ckpt["total_pushes"]
        start_epoch = ckpt["epoch"] + 1
        D_vec = enforce_quantized_values(
            D_vec, train_mask, low_mask, high_mask,
            middle_range, args.soft_value, args.rigid_value)
        print(
            f"[tail_quant {case}] resumed from {ckpt_path} "
            f"at epoch {ckpt['epoch']} / {cfg['N_epochs']} "
            f"best={best_val:.4e}")

    end_epoch = min(cfg["N_epochs"], stop_after) if stop_after is not None else cfg["N_epochs"]
    for epoch in range(start_epoch, end_epoch + 1):
        cos_decay = 0.5 * (1 + np.cos(np.pi * (epoch - 1) / cfg["N_epochs"]))
        lr = cfg["lr_min"] + (cfg["lr_max"] - cfg["lr_min"]) * cos_decay
        case_offset = {"identity": 0, "diag": 1_000_000, "rotated": 2_000_000}[case]
        rng_epoch = np.random.RandomState(epoch + case_offset)
        epoch_mse = 0.0

        for _ in range(cfg["N_train_tasks"]):
            w_star = L_w @ rng_epoch.randn(D)
            X = rng_epoch.randn(20, D) @ L_x.T
            y = X @ w_star
            Xq = rng_epoch.randn(16, D) @ L_x.T
            yq = Xq @ w_star

            w_hat, eps_fwd_all, F, Sh, eta = sysm.forward_pgd(
                D_vec, X, y, cfg["n_stages"])
            grad, _, task_mse = sysm.adjoint_pgd_query(
                D_vec, eps_fwd_all, w_hat, Xq, yq, F, Sh, cfg["n_stages"], eta)
            grad[~train_mask] = 0.0
            epoch_mse += task_mse
            total_pushes += 2 * cfg["n_stages"]

            t_adam += 1
            m_adam = beta1 * m_adam + (1 - beta1) * grad
            v_adam = beta2 * v_adam + (1 - beta2) * grad ** 2
            m_hat = m_adam / (1 - beta1 ** t_adam)
            v_hat = v_adam / (1 - beta2 ** t_adam)
            D_vec[train_mask] -= lr * m_hat[train_mask] / (
                np.sqrt(v_hat[train_mask]) + eps_adam)
            D_vec = enforce_quantized_values(
                D_vec, train_mask, low_mask, high_mask,
                middle_range, args.soft_value, args.rigid_value)

        epoch_mse /= cfg["N_train_tasks"]
        train_mses.append(epoch_mse)
        val_mse = float(sysm.evaluate_truss(D_vec, val_tasks, cfg["n_stages"]).mean())
        val_mses.append(val_mse)
        if val_mse < best_val:
            best_val = val_mse
            best_D = D_vec.copy()
            best_epoch = epoch

        if epoch <= 3 or epoch == cfg["N_epochs"] or epoch % 25 == 0:
            print(
                f"[tail_quant {case}] epoch={epoch:4d} train={epoch_mse:.4e} "
                f"val={val_mse:.4e} middle={train_mask.mean():.2f} "
                f"best={best_val:.4e} elapsed={time.time() - t0:.0f}s")

        if args.checkpoint_every > 0 and (
                epoch == end_epoch or epoch % args.checkpoint_every == 0):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            save_checkpoint(
                ckpt_path, epoch, D_vec, train_mask, low_mask, high_mask,
                m_adam, v_adam, t_adam, best_val, best_D, best_epoch,
                train_mses, val_mses, total_pushes)
            print(f"[tail_quant {case}] checkpoint saved: {ckpt_path}")

    test_mse = float(sysm.evaluate_truss(best_D, test_tasks, cfg["n_stages"]).mean())
    G = sysm.learned_G(best_D)
    Sigma_inv = np.linalg.inv(Sigma_x)
    alpha = float(np.sum(G * Sigma_inv) / max(np.sum(Sigma_inv * Sigma_inv), 1e-14))
    g_err = float(np.linalg.norm(G - alpha * Sigma_inv, "fro") /
                  max(np.linalg.norm(alpha * Sigma_inv, "fro"), 1e-14))
    middle_vals = best_D[train_mask]
    middle_ratio = float(middle_vals.max() / max(middle_vals.min(), 1e-14))
    return {
        "mode": "tail_quant",
        "case": case,
        "middle_stiffness_ratio": args.middle_stiffness_ratio,
        "middle_clip_min": middle_range[0],
        "middle_clip_max": middle_range[1],
        "low_fraction": float(low_mask.mean()),
        "high_fraction": float(high_mask.mean()),
        "middle_fraction": float(train_mask.mean()),
        "soft_value": args.soft_value,
        "rigid_value": args.rigid_value,
        "n_stages": cfg["n_stages"],
        "fine_tune_epochs": cfg["N_epochs"],
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "test_mse": test_mse,
        "oracle_test_mse": mse_oracle_test,
        "test_over_oracle": test_mse / max(mse_oracle_test, 1e-14),
        "g_err": g_err,
        "full_D_min": float(best_D.min()),
        "full_D_max": float(best_D.max()),
        "full_D_ratio": float(best_D.max() / max(best_D.min(), 1e-14)),
        "middle_D_min": float(middle_vals.min()),
        "middle_D_max": float(middle_vals.max()),
        "middle_D_ratio": middle_ratio,
        "total_pushes": total_pushes,
        "runtime_sec": float(time.time() - t0),
    }
