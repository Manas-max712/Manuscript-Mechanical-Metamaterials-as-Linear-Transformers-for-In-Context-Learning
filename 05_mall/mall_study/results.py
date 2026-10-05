import os
from pathlib import Path
import numpy as np


def save_case_postprocess_data(out_dir, result, snapshot_epochs, D_snapshots):
    path = os.path.join(out_dir, "postprocess_data.npz")
    stages = np.asarray(sorted(result['test_per_stage'].keys()), dtype=int)
    stage_mses = np.asarray([result['test_per_stage'][s] for s in stages], dtype=float)
    np.savez_compressed(
        path,
        snapshot_epochs=np.asarray(snapshot_epochs, dtype=int),
        D_snapshots=np.asarray(D_snapshots, dtype=float),
        train_mses=np.asarray(result['train_mses'], dtype=float),
        val_mses=np.asarray(result['val_mses'], dtype=float),
        test_stage_grid=stages,
        test_stage_mses=stage_mses,
        G_eigs=np.asarray(result['G_eigs'], dtype=float),
        Sinv_eigs=np.asarray(result['Sinv_eigs'], dtype=float),
        G_eigs_sorted=np.asarray(result['G_eigs_sorted'], dtype=float),
        target_eigs_sorted=np.asarray(result['target_eigs_sorted'], dtype=float),
        subspace_ks=np.asarray(result['subspace_ks'], dtype=int),
        subspace_max_angles=np.asarray(result['subspace_max_angles'], dtype=float),
        subspace_mean_angles=np.asarray(result['subspace_mean_angles'], dtype=float),
        Sigma_x=np.asarray(result['Sigma_x'], dtype=float),
        D_best=np.asarray(result['D_best'], dtype=float),
        alpha=float(result['alpha']),
        g_err=float(result['g_err']),
        eigval_rel_err=float(result['eigval_rel_err']),
        best_val_mse=float(result['best_val_mse']),
        best_epoch=int(result['best_epoch']),
        mse_init_val=float(result['mse_init_val']),
        mse_final_val=float(result['mse_final_val']),
        mse_final_test=float(result['mse_final_test']),
        mse_oracle_val=float(result['mse_oracle_val']),
        mse_oracle_test=float(result['mse_oracle_test']),
        mse_ols_val=float(result['mse_ols_val']),
        mse_ols_test=float(result['mse_ols_test']),
        angle_top1_deg=float(result['angle_top1_deg']),
        angle_top2_deg=float(result['angle_top2_deg']),
        rho_val_mean=float(result['rho_val_mean']),
        rho_test_mean=float(result['rho_test_mean']),
        rho_test_max=float(result['rho_test_max']),
        eta_test_mean=float(result['eta_test_mean']),
        total_pushes=int(result['total_pushes']),
        n_stages=int(result['n_stages']),
    )
    print(f"  Saved postprocess data: {path}")


def load_case(path, case):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key].copy() for key in data.files}
    if result["D_best"].shape != (132,) or result["D_snapshots"].shape[1] != 132:
        raise ValueError(f"Unexpected MALL lattice size in {path}")
    result["case"] = case
    result["test_per_stage"] = {int(s): float(v) for s, v in zip(
        result["test_stage_grid"], result["test_stage_mses"])}
    return result


def add_plot_arrays(result):
    stages = sorted(result["test_per_stage"])
    result["test_stage_grid"] = np.asarray(stages, dtype=int)
    result["test_stage_mses"] = np.asarray([result["test_per_stage"][s] for s in stages])
    return result
