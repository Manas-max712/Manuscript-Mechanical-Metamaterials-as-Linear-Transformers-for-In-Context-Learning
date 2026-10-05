import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .analysis import sigma_from_G, sigma_from_Q_details, whiten_operator, whiten_full_operator, dist_to_scaled_identity, symmetric_root
from .config import DIMENSION, LAYERS, experiment_case
from .training import train_ahn, train_lattice, initialize_lattice_rng
from .ahn_protocol import COMMIT


def run_experiment(case, settings, device, output_dir):
    parameters = experiment_case(case)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sigma = parameters["Sigma"]
    factor = parameters["covariance_factor"]
    batch_size = settings.ahn_batch_size or parameters["ahn_batch_size"]

    ahn_model, ahn_losses, q_matrices, p_matrices, ahn_factor = train_ahn(
        case, max_iters=settings.ahn_iterations, batch_size=batch_size, device=device)
    torch.save({"state_dict": ahn_model.state_dict(), "dimension": DIMENSION,
                "layers": LAYERS, "case": case, "dtype": "float32",
                "covariance_factor": torch.from_numpy(ahn_factor),
                "completed_steps": len(ahn_losses), "upstream_commit": COMMIT}, output_dir / "ahn_final.pt")
    recovered = [sigma_from_Q_details(q, ahn_factor) for q in q_matrices]
    diagnostics = []
    for layer, q in enumerate(q_matrices):
        w = whiten_full_operator(q, ahn_factor)
        diagnostics.append(dict(layer=layer, scale=float(np.trace(w)/DIMENSION),
                                full_distance=float(dist_to_scaled_identity(w))))
    (output_dir / "ahn_diagnostics.json").write_text(json.dumps(
        dict(case=case, completed_steps=len(ahn_losses), dtype="float32", layers=diagnostics),
        indent=2) + "\n")
    initialize_lattice_rng(case, device, settings.lattice_rng_ahn_iterations,
                              settings.ahn_batch_size)

    (lattice_model, operator, output_indices, members, free_dofs, compatibility,
     train_losses, val_losses) = train_lattice(
        sigma, parameters["w_prior_cov"], n_ctx=settings.lattice_context,
        N_train=settings.lattice_train_tasks, N_val=settings.lattice_validation_tasks,
        n_epochs=settings.lattice_epochs, lr=settings.lattice_lr,
        bs=settings.lattice_batch_size, device=device)
    torch.save({"state_dict": lattice_model.state_dict(), "dimension": DIMENSION,
                "layers": LAYERS, "case": case, "members": members,
                "output_indices": output_indices,
                "compatibility": torch.from_numpy(compatibility),
                "selected_epoch": int(np.argmin(val_losses)) + 1,
                "validation_loss": min(val_losses)}, output_dir / "lattice_best.pt")
    np.savez_compressed(output_dir / "case_inputs.npz", Sigma=sigma,
                        w_prior_cov=parameters["w_prior_cov"], covariance_factor=factor,
                        compatibility=compatibility, output_indices=output_indices,
                        ahn_covariance_factor=ahn_factor)
    metadata = dict(case=case, dimension=DIMENSION, layers=LAYERS, dtype="float64",
                    device=str(device), settings=asdict(settings),
                    ahn_lr=parameters["ahn_lr"], ahn_clip=parameters["ahn_clip"],
                    ahn_batch_size=batch_size, ahn_betas=parameters["ahn_betas"],
                    ahn_var=parameters["ahn_var"], ahn_context=20,
                    ahn_seed=0, ahn_dtype="float32", ahn_iterations=len(ahn_losses),
                    ahn_batch_refresh=1 if case == "isotropic" else 100,
                    ahn_upstream_commit=COMMIT, ahn_heatmap="full S.T @ Q @ S",
                    lattice_train_data_seed=0,
                    lattice_validation_data_seed=settings.lattice_train_tasks,
                    lattice_shuffle_seed="Seed-0 float64 random stream after prescribed initialization draws",
                    lattice_weight_decay=0.01, lattice_betas=[0.9, 0.99],
                    lattice_clip_norm=1.0, lattice_validation_batch_size=64,
                    free_dofs=free_dofs, members=members)
    (output_dir / "training_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    eigenvalues = np.sort(np.linalg.eigvalsh(sigma))
    lattice_root = symmetric_root(sigma)
    ahn_root = symmetric_root(ahn_factor @ ahn_factor.T)
    return {
        "name": parameters["name"], "theorem": parameters["theorem"],
        "Sigma_ev_true": eigenvalues,
        "Sigma_ev_ahn": recovered[0][1], "Sigma_ev_lat": sigma_from_G(operator),
        "Sigma_ev_ahn_layers": [item[1] for item in recovered],
        "Ahn_layer_scales": [item[0] for item in recovered],
        "Ahn_whitened_Q": [whiten_full_operator(q, ahn_root) for q in q_matrices[:2]],
        "Lattice_whitened_G": whiten_full_operator(operator, lattice_root),
        "Lattice_whitened_G_target": whiten_operator(operator, lattice_root),
        "Sigma": sigma, "covariance_factor": factor,
        "Ahn_Sigma": ahn_factor @ ahn_factor.T,
        "G": operator,
        "G_ev": np.sort(np.linalg.eigvalsh(0.5 * (operator + operator.T))),
        "G_theory_ev": np.sort(1.0 / eigenvalues),
        "Q_list": q_matrices, "P_list": p_matrices, "loss_ahn": ahn_losses,
        "loss_lat_train": train_losses, "loss_lat_val": val_losses,
    }
