import json
import os
import time
from pathlib import Path
import numpy as np
from .config import d, N_few, N_queries, CASE_TRAIN_CONFIG, SNAPSHOT_EPOCHS
from .geometry import m
from .tasks import case_setup, make_tasks, oracle_mse
from .learning import forward_pgd, adjoint_pgd_query, evaluate_truss
from .analysis import analyze_model
from .results import save_case_postprocess_data, add_plot_arrays


def train_case(CASE, out_dir, config=None, stop_after=None, test_seed=20001234, n_eval=1000):
    print(f"\n\n{'#'*72}\n#  CASE: {CASE}\n{'#'*72}")
    os.makedirs(out_dir, exist_ok=True)

    cfg = dict(CASE_TRAIN_CONFIG[CASE] if config is None else config)
    N_train_tasks = cfg["N_train_tasks"]
    N_epochs      = cfg["N_epochs"]
    lr_max        = cfg["lr_max"]
    lr_min        = cfg["lr_min"]
    n_stages_case = cfg["n_stages"]

    Sigma_x, Sigma_w, L_x, L_w = case_setup(CASE, d_=d)
    print(f"  Σ_x eigvals:   {np.linalg.eigvalsh(Sigma_x).round(4)}")
    print(f"  Σ_x⁻¹ eigvals: {np.linalg.eigvalsh(np.linalg.inv(Sigma_x)).round(4)}")
    print(f"  Training config: n_stages={n_stages_case}, N_train_tasks={N_train_tasks}, "
          f"N_epochs={N_epochs}, lr={lr_max:.1e}→{lr_min:.1e}")

    D = np.random.RandomState(0).rand(m) * 2.0 + 0.5
    print(f"  Initial D: [{D.min():.3f}, {D.max():.3f}]  (m={m} bars)")
    snapshot_targets = set(e for e in SNAPSHOT_EPOCHS if e <= N_epochs)
    D_snapshots = {0: D.copy()}

    val_tasks  = make_tasks(seed=9999, n_tasks=min(n_eval, 200),  L_x=L_x, L_w=L_w)
    test_tasks = make_tasks(seed=test_seed, n_tasks=n_eval, L_x=L_x, L_w=L_w)

    mse_oracle_val  = oracle_mse(val_tasks)
    mse_oracle_test = oracle_mse(test_tasks)
    mse_ols_val     = oracle_mse(val_tasks, lam=0.0)
    mse_ols_test    = oracle_mse(test_tasks, lam=0.0)
    print(f"  Oracle (ridge) val MSE:  {mse_oracle_val:.6f}")
    print(f"  Oracle (ridge) test MSE: {mse_oracle_test:.6f}")
    print(f"  Oracle (OLS)   val MSE:  {mse_ols_val:.6f}")
    print(f"  Oracle (OLS)   test MSE: {mse_ols_test:.6f}")

    mse_init = evaluate_truss(D, val_tasks, n_stages_case).mean()
    print(f"  Initial val MSE: {mse_init:.6f}\n  Training...")

    beta1 = 0.9; beta2 = 0.999; eps_adam = 1e-8
    clip_range    = (0.01, 200)

    m_adam = np.zeros(m); v_adam = np.zeros(m); t_adam = 0
    train_objectives = []; train_mses = []; val_mses = []

    best_val_mse = np.inf
    best_D       = D.copy()
    best_epoch   = 0

    t_start = time.time()
    total_pushes = 0

    completed_epochs = min(N_epochs, stop_after) if stop_after is not None else N_epochs
    for epoch in range(1, completed_epochs + 1):
        cos_decay = 0.5 * (1 + np.cos(np.pi * (epoch - 1) / N_epochs))
        lr        = lr_min + (lr_max - lr_min) * cos_decay

        rng_epoch = np.random.RandomState(epoch)
        epoch_loss = 0.0

        for task_i in range(N_train_tasks):
            w_star = L_w @ rng_epoch.randn(d)
            X      = rng_epoch.randn(N_few, d) @ L_x.T
            y      = X @ w_star
            Xq     = rng_epoch.randn(N_queries, d) @ L_x.T
            yq     = Xq @ w_star

            w_hat, eps_fwd_all, F, Sh, eta_pgd = forward_pgd(D, X, y, n_stages_case)
            total_pushes += n_stages_case

            grad, task_obj, task_mse = adjoint_pgd_query(
                D, eps_fwd_all, w_hat, Xq, yq, F, Sh, n_stages_case, eta_pgd)
            total_pushes += n_stages_case
            epoch_loss   += task_mse

            t_adam += 1
            m_adam = beta1 * m_adam + (1 - beta1) * grad
            v_adam = beta2 * v_adam + (1 - beta2) * grad ** 2
            m_hat  = m_adam / (1 - beta1 ** t_adam)
            v_hat  = v_adam / (1 - beta2 ** t_adam)
            D     -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)
            D      = np.clip(D, clip_range[0], clip_range[1])

        epoch_loss /= N_train_tasks
        train_objectives.append(0.5 * epoch_loss)
        train_mses.append(epoch_loss)

        mse_val = evaluate_truss(D, val_tasks, n_stages_case).mean()
        val_mses.append(mse_val)

        if mse_val < best_val_mse:
            best_val_mse = mse_val
            best_D       = D.copy()
            best_epoch   = epoch

        if epoch in snapshot_targets:
            D_snapshots[epoch] = D.copy()

        elapsed = time.time() - t_start
        if epoch <= 5 or epoch % 50 == 0 or epoch == N_epochs:
            marker = "  ★ best" if epoch == best_epoch else ""
            print(f"    Epoch {epoch:4d}  train_mse={epoch_loss:.6f}  "
                  f"val={mse_val:.6f}  lr={lr:.2e}  "
                  f"pushes={total_pushes:,}  ({elapsed:.0f}s){marker}")

    print(f"\n  Best val MSE: {best_val_mse:.6f}  at epoch {best_epoch}")

    result = analyze_model(best_D, Sigma_x, val_tasks, test_tasks, n_stages_case)
    D_snapshots[best_epoch] = best_D.copy()
    D_snapshots[completed_epochs] = D.copy()
    snapshot_epochs = np.asarray(sorted(D_snapshots), dtype=int)
    snapshot_values = np.stack([D_snapshots[e] for e in snapshot_epochs])
    result.update(case=CASE, best_epoch=best_epoch, best_val_mse=best_val_mse,
                  mse_init_val=mse_init, train_objectives=train_objectives,
                  train_mses=train_mses, val_mses=val_mses, total_pushes=total_pushes,
                  snapshot_epochs=snapshot_epochs, D_snapshots=snapshot_values)
    add_plot_arrays(result)
    save_case_postprocess_data(out_dir, result, snapshot_epochs, snapshot_values)
    metadata = dict(cfg, completed_epochs=completed_epochs, initialization_seed=0,
                    training_epoch_seed="epoch", validation_seed=9999,
                    test_seed=test_seed, evaluation_tasks=n_eval, validation_tasks=min(n_eval, 200),
                    query_count=N_queries, context_count=N_few,
                    fixed_gain_adjoint=True, beta1=0.9, beta2=0.999,
                    adam_epsilon=1e-8, rigidity_limits=[0.01, 200.0])
    (Path(out_dir) / "training_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return result
