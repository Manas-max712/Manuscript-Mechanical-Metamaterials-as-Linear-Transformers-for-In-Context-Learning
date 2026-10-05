import time
import numpy as np
import torch
from torch import nn

from .config import DIMENSION as d, LAYERS as n_layer, AHN_CONTEXT as N_ahn
from .data import regression_tasks, encode_lattice_tasks
from .geometry import build_lattice, build_B
from .models import LatticeTF, lat_loss
from .extraction import extract_operator


def train_ahn(case, max_iters=None, batch_size=None, device='cpu'):
    from .ahn_training import run
    old_dtype = torch.get_default_dtype()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        result = run('paper-rotated' if case == 'rotated' else case, 0, device,
                     steps=max_iters, batch_size=batch_size)
    finally:
        torch.set_default_dtype(old_dtype)
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    model = result['model']
    params = model.allparam.detach().cpu().numpy().copy()
    factor = (np.eye(d, dtype=np.float32) if result['U'] is None else
              (result['U'] @ result['D']).detach().cpu().numpy())
    return model, result['losses'], [params[l,0,1] for l in range(n_layer)], [params[l,0,0] for l in range(n_layer)], factor


def initialize_lattice_rng(case, device, iterations=10000, batch_size=None):
    torch.manual_seed(0)
    np.random.seed(0)
    torch.zeros(n_layer, 1, 2, d, d, dtype=torch.float64).normal_(0, .0001)
    if torch.device(device).type != 'cpu':
        return
    batch = batch_size or (4000 if case == 'isotropic' else 20000)
    print('Initializing lattice shuffle stream', flush=True)
    for t in range(iterations):
        torch.randn(batch, d, dtype=torch.float64)
        torch.randn(batch, N_ahn, d, dtype=torch.float64)
        torch.randn(batch, 1, d, dtype=torch.float64)
        if case != 'isotropic':
            torch.randn(batch, d, dtype=torch.float64)
        if (t+1) % 2000 == 0:
            print(f'  RNG replay {t+1}/{iterations}', flush=True)


def train_lattice(Sigma, w_prior_cov, n_ctx=20000, N_train=4000, N_val=500,
                  n_epochs=500, lr=1e-4, bs=16, device="cpu"):
    nodes, edges, _, free, out_dofs, _, _ = build_lattice(d - 1, 4)
    members, free_dofs = len(edges), len(free)
    free_index = {dof: index for index, dof in enumerate(free)}
    output_indices = [free_index[dof] for dof in out_dofs]
    compatibility = build_B(nodes, edges, free)
    print(f"Lattice: {len(nodes)} nodes, {members} bars, {free_dofs} free DOFs")
    print(f"Training: {N_train} tasks, context={n_ctx}, epochs={n_epochs}")

    train_inputs, train_targets = encode_lattice_tasks(
        regression_tasks(Sigma, w_prior_cov, N_train, n_ctx, seed=0),
        compatibility, output_indices)
    val_inputs, val_targets = encode_lattice_tasks(
        regression_tasks(Sigma, w_prior_cov, N_val, n_ctx, seed=N_train),
        compatibility, output_indices)

    model = LatticeTF(n_layer, members, device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.99), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, n_epochs, eta_min=lr * 0.01)
    best_validation, best_state = float("inf"), None
    train_history, val_history = [], []

    for epoch in range(1, n_epochs + 1):
        model.train()
        permutation = torch.randperm(N_train)
        train_sum, train_count = 0.0, 0
        for start in range(0, N_train, bs):
            indices = permutation[start:start + bs]
            optimizer.zero_grad()
            loss = lat_loss(model(train_inputs[indices].to(device)),
                            train_targets[indices].to(device), members, output_indices)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += loss.item() * indices.numel()
            train_count += indices.numel()
        scheduler.step()

        model.eval()
        validation_sum, validation_count = 0.0, 0
        with torch.no_grad():
            for start in range(0, val_inputs.shape[0], 64):
                inputs = val_inputs[start:start + 64].to(device)
                targets = val_targets[start:start + 64].to(device)
                validation_sum += lat_loss(model(inputs), targets, members,
                                           output_indices).item() * inputs.shape[0]
                validation_count += inputs.shape[0]
        validation_loss = validation_sum / max(validation_count, 1)
        train_history.append(train_sum / max(train_count, 1))
        val_history.append(validation_loss)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
        if epoch <= 3 or epoch % 50 == 0 or epoch == n_epochs:
            print(f"Epoch {epoch:04d}: val={validation_loss:.3e}, best={best_validation:.3e}")

    model.load_state_dict(best_state)
    model.eval()
    operator = extract_operator(model, compatibility, output_indices, device)
    return (model, operator, output_indices, members, free_dofs, compatibility,
            train_history, val_history)
