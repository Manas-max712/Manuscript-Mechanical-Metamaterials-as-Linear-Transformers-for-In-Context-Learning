import os
import time
import numpy as np
from .config import D, N_FEW, N_QUERIES, CASE_TRAIN_CONFIG, clip_from_ratio
from .geometry import sym
from .mechanics import MallSystem
from .tasks import case_setup, make_tasks, oracle_mse
from .bounded_checkpoints import checkpoint_path, save_checkpoint, load_checkpoint

def train_bounded(case: str, args, stop_after=None):
    init_D = active_mask = None
    max_epochs_override = args.epochs
    sys = MallSystem()
    cfg = CASE_TRAIN_CONFIG[case].copy()
    if max_epochs_override is not None:
        cfg["N_epochs"] = int(max_epochs_override)
    if args.train_tasks is not None:
        cfg["N_train_tasks"] = args.train_tasks

    clip_range = clip_from_ratio(args.stiffness_ratio)
    Sigma_x, _, L_x, L_w = case_setup(case)
    val_tasks = make_tasks(seed=9999, n_tasks=min(args.eval_tasks, 200), L_x=L_x, L_w=L_w)
    test_tasks = make_tasks(seed=20001234, n_tasks=args.eval_tasks, L_x=L_x, L_w=L_w)
    mse_oracle_test = oracle_mse(test_tasks)

    if init_D is None:
        D_vec = np.random.RandomState(0).rand(sys.m) * 2.0 + 0.5
    else:
        D_vec = np.asarray(init_D, dtype=float).copy()
    D_vec = np.clip(D_vec, clip_range[0], clip_range[1])

    if active_mask is None:
        active_mask = np.ones(sys.m, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).copy()
        D_vec[~active_mask] = clip_range[0]

    beta1 = 0.9
    beta2 = 0.999
    eps_adam = 1e-8
    m_adam = np.zeros(sys.m)
    v_adam = np.zeros(sys.m)
    t_adam = 0
    best_val = np.inf
    best_D = D_vec.copy()
    best_mask = active_mask.copy()
    best_epoch = 0
    train_mses = []
    val_mses = []
    total_pushes = 0
    t0 = time.time()
    start_epoch = 1

    ckpt_path = checkpoint_path(args, case)
    if args.resume and os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path)
        if len(ckpt["D_vec"]) != sys.m:
            raise ValueError(
                f"checkpoint D length mismatch: {ckpt_path} has "
                f"{len(ckpt['D_vec'])}, expected {sys.m}")
        D_vec = ckpt["D_vec"]
        active_mask = ckpt["active_mask"]
        m_adam = ckpt["m_adam"]
        v_adam = ckpt["v_adam"]
        t_adam = ckpt["t_adam"]
        best_val = ckpt["best_val"]
        best_D = ckpt["best_D"]
        best_mask = ckpt["best_mask"]
        best_epoch = ckpt["best_epoch"]
        train_mses = ckpt["train_mses"]
        val_mses = ckpt["val_mses"]
        total_pushes = ckpt["total_pushes"]
        start_epoch = ckpt["epoch"] + 1
        D_vec = np.clip(D_vec, clip_range[0], clip_range[1])
        D_vec[~active_mask] = clip_range[0]
        print(
            f"[{args.mode} {case}] resumed from {ckpt_path} "
            f"at epoch {ckpt['epoch']} / {cfg['N_epochs']} "
            f"best={best_val:.4e}"
        )

    end_epoch = min(cfg["N_epochs"], stop_after) if stop_after is not None else cfg["N_epochs"]
    for epoch in range(start_epoch, end_epoch + 1):
        cos_decay = 0.5 * (1 + np.cos(np.pi * (epoch - 1) / cfg["N_epochs"]))
        lr = cfg["lr_min"] + (cfg["lr_max"] - cfg["lr_min"]) * cos_decay
        case_offset = {"identity": 0, "diag": 1_000_000, "rotated": 2_000_000}[case]
        rng_epoch = np.random.RandomState(epoch + case_offset)
        epoch_mse = 0.0

        for _ in range(cfg["N_train_tasks"]):
            w_star = L_w @ rng_epoch.randn(D)
            X = rng_epoch.randn(N_FEW, D) @ L_x.T
            y = X @ w_star
            Xq = rng_epoch.randn(N_QUERIES, D) @ L_x.T
            yq = Xq @ w_star

            w_hat, eps_fwd_all, F, Sh, eta = sys.forward_pgd(
                D_vec, X, y, cfg["n_stages"])
            grad, _, task_mse = sys.adjoint_pgd_query(
                D_vec, eps_fwd_all, w_hat, Xq, yq, F, Sh, cfg["n_stages"], eta)
            grad[~active_mask] = 0.0
            epoch_mse += task_mse
            total_pushes += 2 * cfg["n_stages"]

            t_adam += 1
            m_adam = beta1 * m_adam + (1 - beta1) * grad
            v_adam = beta2 * v_adam + (1 - beta2) * grad ** 2
            m_hat = m_adam / (1 - beta1 ** t_adam)
            v_hat = v_adam / (1 - beta2 ** t_adam)
            D_vec -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)
            D_vec = np.clip(D_vec, clip_range[0], clip_range[1])
            D_vec[~active_mask] = clip_range[0]

        epoch_mse /= cfg["N_train_tasks"]
        train_mses.append(epoch_mse)
        val_mse = float(sys.evaluate_truss(D_vec, val_tasks, cfg["n_stages"]).mean())
        val_mses.append(val_mse)
        if val_mse < best_val:
            best_val = val_mse
            best_D = D_vec.copy()
            best_mask = active_mask.copy()
            best_epoch = epoch
        if epoch <= 3 or epoch == cfg["N_epochs"] or epoch % 50 == 0:
            print(
                f"[{args.mode} {case}] epoch={epoch:4d} train={epoch_mse:.4e} "
                f"val={val_mse:.4e} active={active_mask.mean():.2f} "
                f"best={best_val:.4e} elapsed={time.time() - t0:.0f}s"
            )
        if args.checkpoint_every > 0 and (
                epoch == end_epoch or epoch % args.checkpoint_every == 0):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            save_checkpoint(
                ckpt_path, epoch, D_vec, active_mask, m_adam, v_adam, t_adam,
                best_val, best_D, best_mask, best_epoch, train_mses, val_mses,
                total_pushes)
            print(f"[{args.mode} {case}] checkpoint saved: {ckpt_path}")

    test_mse = float(sys.evaluate_truss(best_D, test_tasks, cfg["n_stages"]).mean())
    G = sys.learned_G(best_D)
    Sigma_inv = np.linalg.inv(Sigma_x)
    alpha = float(np.sum(G * Sigma_inv) / max(np.sum(Sigma_inv * Sigma_inv), 1e-14))
    g_err = float(np.linalg.norm(G - alpha * Sigma_inv, "fro") /
                  max(np.linalg.norm(alpha * Sigma_inv, "fro"), 1e-14))
    return {
        "mode": args.mode,
        "case": case,
        "stiffness_ratio": args.stiffness_ratio,
        "clip_min": clip_range[0],
        "clip_max": clip_range[1],
        "n_stages": cfg["n_stages"],
        "N_epochs": cfg["N_epochs"],
        "N_train_tasks": cfg["N_train_tasks"],
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "test_mse": test_mse,
        "oracle_test_mse": mse_oracle_test,
        "test_over_oracle": test_mse / max(mse_oracle_test, 1e-14),
        "g_err": g_err,
        "D_min": float(best_D.min()),
        "D_max": float(best_D.max()),
        "D_ratio": float(best_D.max() / max(best_D.min(), 1e-14)),
        "active_fraction": float(best_mask.mean()),
        "total_pushes": total_pushes,
        "runtime_sec": float(time.time() - t0),
    }, best_D, active_mask
