#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

torch.set_default_dtype(torch.float64)
device = torch.device("cpu")
cur_dir = "runs/5x5"
E, A = 70e08, 1e-4
s = {"architecture": "stack", "obj": "electric", "data": "fc", "PB": "zero", "ff": "yes"}
PLOT_LABEL_FONTSIZE = 16
PLOT_TICK_FONTSIZE = 14
PLOT_LEGEND_FONTSIZE = 14
PLOT_TITLE_FONTSIZE = 15

def style_plot_axes(ax, xlabel="x", ylabel="y"):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=PLOT_LABEL_FONTSIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=PLOT_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=PLOT_TICK_FONTSIZE)

def tile_unit_cell(cell_coords, nx=1, ny=1, spacing=(1.0, 1.0), tol=1e-8):
    cell_coords = np.array(cell_coords, dtype=float)
    sx, sy = spacing
    Nc = cell_coords.shape[0]

    raw_coords = []
    for iy in range(ny):
        for ix in range(nx):
            dx, dy = ix * sx, iy * sy
            for (x, y) in cell_coords:
                raw_coords.append((x + dx, y + dy))
    raw_coords = np.array(raw_coords, dtype=float)

    quant = np.round(raw_coords / tol).astype(np.int64)
    key_to_index = {}
    unique_nodes = []
    raw_to_unique_idx = []
    for idx, q in enumerate(map(tuple, quant)):
        if q not in key_to_index:
            key_to_index[q] = len(unique_nodes)
            unique_nodes.append(raw_coords[idx])
        raw_to_unique_idx.append(key_to_index[q])
    nodes_np = np.array(unique_nodes, dtype=float)

    cell_edges_local = [(i, j) for i in range(Nc) for j in range(i + 1, Nc)]
    edges_set = set()
    for iy in range(ny):
        for ix in range(nx):
            tile_id = iy * nx + ix
            base = tile_id * Nc
            for (ia, jb) in cell_edges_local:
                g_i = raw_to_unique_idx[base + ia]
                g_j = raw_to_unique_idx[base + jb]
                if g_i == g_j:
                    continue
                a, b = (g_i, g_j) if g_i < g_j else (g_j, g_i)
                edges_set.add((a, b))

    def edge_key(pair):
        i, j = pair
        dist = j - i
        if dist == 1:
            return (0, i)
        return (1, -dist, i)

    edges = list(edges_set)
    edges.sort(key=edge_key)
    return nodes_np, edges

def gen_incidence_from_edges(n_nodes, edges_list):
    d_local = len(edges_list)
    B = torch.zeros(n_nodes, d_local, device=device)
    for e_id, (i, j) in enumerate(edges_list):
        B[i, e_id] = 1.0
        B[j, e_id] = -1.0
    return B

def B_strain(B, Coords):
    n_local, d_local = B.shape
    Coords_t = torch.tensor(Coords, dtype=torch.float64, device=device) if not isinstance(Coords, torch.Tensor) else Coords.to(device)
    B_new = torch.zeros(2 * n_local, d_local, device=device)
    eps = 1e-30
    for e in range(d_local):
        nodes = torch.nonzero(B[:, e]).squeeze(1).tolist()
        if len(nodes) != 2:
            raise RuntimeError(f"Edge {e} does not connect exactly 2 nodes: found {nodes}")
        i, j = nodes
        vec = Coords_t[j] - Coords_t[i]
        L = torch.norm(vec) + eps
        t = vec / L
        B_new[2 * i, e] = t[0] / L
        B_new[2 * i + 1, e] = t[1] / L
        B_new[2 * j, e] = -t[0] / L
        B_new[2 * j + 1, e] = -t[1] / L
    return B_new

def D_matrix(Coords, edges, E_local, A_local):
    Coords_t = torch.tensor(Coords, dtype=torch.float32, device=device) if not isinstance(Coords, torch.Tensor) else Coords.to(device)
    d_local = len(edges)
    D = torch.zeros(d_local, d_local, device=device)
    eps = 1e-12
    for k, (i, j) in enumerate(edges):
        vec = Coords_t[j] - Coords_t[i]
        L = torch.norm(vec) + eps
        D[k, k] = E_local * A_local * L
    return D

def permute_D_diag(D, perm):
    diag = torch.diag(D)
    return torch.diag(diag[perm])

def stiffness_matrix(B_tilde):
    return B_tilde @ B_tilde.transpose(-1, -2)

def choose_kinematically_stable_dofs(n_nodes, Coords=None, generator=None, tol=1e-8):
    if n_nodes < 1:
        raise ValueError("need at least 1 node")

    if Coords is None:
        raise ValueError("Coords must be provided for deterministic bottom-edge supports")

    if isinstance(Coords, torch.Tensor):
        C = Coords.to(device).double()
    else:
        C = torch.tensor(Coords, dtype=torch.float64, device=device)

    if C.shape[0] != n_nodes:
        raise ValueError(f"Coords has {C.shape[0]} nodes but n_nodes={n_nodes}")

    ys = C[:, 1]
    min_y = ys.min()
    bottom_nodes = torch.nonzero((ys - min_y).abs() <= tol, as_tuple=False).squeeze(1)

    if bottom_nodes.numel() == 0:
        raise RuntimeError("No bottom-edge nodes found (check tol / Coords).")

    xs_bottom = C[bottom_nodes, 0]
    node_bl = int(bottom_nodes[torch.argmin(xs_bottom)].item())

    fixed_dofs = []

    for n in bottom_nodes.tolist():
        fixed_dofs.append(2 * n + 1)

    fixed_dofs.append(2 * node_bl)

    fixed = torch.tensor(sorted(set(fixed_dofs)), dtype=torch.long, device=device)
    return fixed

def generate_B_inplace(Z, nv, incidence_B_input, Coords, mask_free, perm, D_local):
    batch_size = Z.shape[0]
    B_perm = incidence_B_input[:, perm]
    B_mat = B_strain(B_perm, Coords)
    if not isinstance(mask_free, torch.Tensor):
        mask_free = torch.as_tensor(mask_free, dtype=torch.bool, device=device)
    else:
        mask_free = mask_free.to(device).bool()
    diagD = torch.sqrt(torch.clamp(torch.diag(D_local), min=0.0))[perm]
    B_ff = B_mat[mask_free, :]
    B_tilde = B_ff * diagD.unsqueeze(0)
    d_local = B_tilde.shape[1]
    Z[:, :, 0:d_local] = 0.0
    free_idx = torch.nonzero(mask_free, as_tuple=False).squeeze(1).to(device)
    Z[:, free_idx, 0:d_local] = B_tilde.unsqueeze(0).repeat(batch_size, 1, 1)
    return Z

def generate_demand_inplace(Z, nv, d):
    Z[:, :, d] = 0.0
    return Z

def apply_demand_on_mask(Z, mask_free, d, sample_idx=0, generator=None,
                         mean_val=50000.0, std_val=10000.0,
                         low=10000.0, high=90000.0):
    if not isinstance(mask_free, torch.Tensor):
        mask_free = torch.as_tensor(mask_free, dtype=torch.bool, device=device)
    else:
        mask_free = mask_free.to(device).bool()
    B = Z.shape[0]
    n_dofs = mask_free.shape[0]
    for b in range(B):
        mags = torch.normal(mean=mean_val, std=std_val, size=(n_dofs,), device=device, generator=generator)
        mags = torch.clamp(mags, min=low, max=high)
        signs = torch.randint(0, 2, (n_dofs,), device=device, dtype=torch.float64)
        signs = 2.0 * signs - 1.0
        sample = mags * signs
        sample_masked = torch.zeros_like(sample)
        sample_masked[mask_free] = sample[mask_free]
        Z[b, :, d] = sample_masked
    return Z

def get_target(Z, D_local, Coords_local, edges, n_nodes, d, feat_dim):
    batch_size = Z.shape[0]
    n_dofs = 2 * n_nodes
    lap = torch.zeros(batch_size, n_dofs, n_dofs, device=device)
    B_block = Z[:, :n_dofs, 0:d]
    for b in range(batch_size):
        lap[b] = stiffness_matrix(B_block[b])
    target_disp = torch.zeros(batch_size, n_dofs, device=device)
    for b in range(batch_size):
        K_b = lap[b]
        f_b = Z[b, :, d].clone()
        mask_free = Z[b, :, d + 2].bool()
        mask_fixed = ~mask_free
        f_b[mask_fixed] = 0.0
        if mask_free.sum() == 0:
            full_u = torch.zeros(n_dofs, device=device)
        else:
            K_ff = K_b[mask_free][:, mask_free]
            f_f = f_b[mask_free]
            try:
                eps_reg = 1e-8 * (torch.trace(K_ff).abs() / max(1, K_ff.shape[0]))
                K_ff_reg = K_ff + eps_reg * torch.eye(K_ff.shape[0], device=device)
                uf = torch.linalg.solve(K_ff_reg, f_f)
            except Exception:
                uf = torch.linalg.pinv(K_ff) @ f_f
            full_u = torch.zeros(n_dofs, device=device)
            full_u[mask_free] = uf
        target_disp[b] = full_u
    return target_disp

def get_loss(output, target, Z, n_nodes, d):
    n_dofs = 2 * n_nodes
    pred_u = output[:, :, d + 1]
    tgt_scalar = target
    mask_free = (Z[:, :, d + 2] > 0)
    diff = (pred_u - tgt_scalar) * mask_free.float()
    per_sample_mse = []
    for b in range(diff.shape[0]):
        m = mask_free[b]
        if m.any():
            per_sample_mse.append((diff[b, m] ** 2).mean())
        else:
            per_sample_mse.append(torch.tensor(0.0, device=diff.device))
    loss = torch.stack(per_sample_mse).mean()
    return loss, loss, torch.tensor(0.0, device=loss.device)

def linear_attention(P, Q, K, Z, d_attn=None, mask=None):
    B, N, feat = Z.shape

    if d_attn is None:
        Z_attn = Z
        Q_use, K_use = Q, K
    else:
        Z_attn = Z[:, :, :d_attn]
        Q_use  = Q[:, :d_attn, :d_attn]
        K_use  = K[:, :d_attn, :d_attn]

    QK = torch.einsum("Hji,Hjk->Hik", Q_use, K_use)
    Attn = torch.einsum("BNi,Hij,BMj->HBNM", Z_attn, QK, Z_attn)

    PZ = torch.einsum("Hij,BNj->HBNi", P, Z)

    out = torch.einsum("HBNi,HBNM->BMi", PZ, Attn) / N

    if mask is not None:
        out = out * mask.unsqueeze(-1)
    return out

class Transformer_LG(nn.Module):
    def __init__(self, n_layer, n_head, n_nodes, d, feat_dim, var, eta_init=1e-3, use_layernorm=False):
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.n = n_nodes
        self.d = d
        self.feat_dim = feat_dim
        self.use_layernorm = use_layernorm

        self.allparam = nn.Parameter(
            torch.zeros(n_layer, n_head, 4, feat_dim, feat_dim, device=device)
        )

        force_idx = min(self.d, self.feat_dim - 1)
        pred_idx  = min(self.d + 1, self.feat_dim - 1)

        with torch.no_grad():
            self.allparam.zero_()

            eye_d = torch.eye(self.d, device=device)
            for L in range(n_layer):
                for H in range(n_head):
                    self.allparam[L, H, 1, :self.d, :self.d].copy_(eye_d)
                    self.allparam[L, H, 2, :self.d, :self.d].copy_(eye_d)

            for L in range(n_layer):
                for H in range(n_head):
                    self.allparam[L, H, 0, pred_idx, pred_idx] = -eta_init

            for L in range(n_layer):
                for H in range(n_head):
                    self.allparam[L, H, 3, pred_idx, force_idx] = eta_init

        self.layer_norms = nn.ModuleList([nn.LayerNorm(feat_dim) for _ in range(n_layer)])

    def forward(self, Z):
        Z_local = Z
        mask = (Z_local[:, :, self.d + 2] > 0).float()

        pred_col = self.d + 1
        force_col = self.d
        mask_col = self.d + 2

        for i in range(self.n_layer):
            P = self.allparam[i, :, 0]
            Q = self.allparam[i, :, 1]
            Kmat = self.allparam[i, :, 2]
            S_all = self.allparam[i, :, 3]
            S = S_all[0]

            resid = linear_attention(P, Q, Kmat, Z_local, d_attn=self.d, mask=mask)

            Z_local = Z_local + resid

            if s.get("ff", "yes") == "yes":
                Z_local = Z_local + torch.nn.functional.linear(Z_local, S)

            if self.use_layernorm:
                Z_local = self.layer_norms[i](Z_local)

            Z_local[:, :, mask_col] = mask
            Z_local[:, :, pred_col] *= mask
            Z_local[:, :, force_col] *= mask

        return Z_local * mask.unsqueeze(-1)

def make_sample_Z(B_inc, D_local, Coords_local, edges, n_nodes, d, feat_dim, p_fix=0.0, sample_seed=0):
    g = torch.Generator(device=device)
    g.manual_seed(int(sample_seed))
    n_dofs = 2 * n_nodes
    Z = torch.zeros(1, n_dofs, feat_dim, device=device)
    fixed = choose_kinematically_stable_dofs(n_nodes, Coords=Coords_local, generator=g)
    mask_free = torch.ones(n_dofs, dtype=torch.bool, device=device)
    mask_free[fixed] = False

    Z[0, :, d + 2] = mask_free.float()
    perm = torch.arange(B_inc.shape[1], device=device)
    Z = generate_B_inplace(Z, n_nodes, B_inc, Coords_local, mask_free, perm, D_local)
    Dp = permute_D_diag(D_local, perm)
    Z = generate_demand_inplace(Z, n_nodes, d)
    Z[:, :, d + 1] = 0.0
    Z = apply_demand_on_mask(Z, mask_free, d=d, sample_idx=sample_seed, generator=g)
    target = get_target(Z, Dp, Coords_local, edges, n_nodes, d, feat_dim)
    return Z.squeeze(0), target.squeeze(0)

def build_dataset_pool(N_total, B_inc, D_local, Coords_local, edges, n_nodes, d, feat_dim, p_fix=0.0):
    Z_list = []
    T_list = []
    for ii in range(N_total):
        Zi, Ti = make_sample_Z(B_inc, D_local, Coords_local, edges, n_nodes, d, feat_dim, p_fix=p_fix, sample_seed=ii)
        Z_list.append(Zi)
        T_list.append(Ti)
    return torch.stack(Z_list), torch.stack(T_list)

def compute_relative_errors_phys(true_u, pred_u, mask_free, eps=1e-12):
    m = mask_free
    if not m.any():
        return 0.0, 0.0, 0.0

    t = true_u[m]
    p = pred_u[m]
    diff = p - t

    num = np.sum(diff ** 2)
    den = np.sum(t ** 2)

    rel_mse = float(num / (den + eps))
    rel_l2  = float(np.sqrt(num) / (np.sqrt(den) + eps))
    rel_l1  = float(np.sum(np.abs(diff)) / (np.sum(np.abs(t)) + eps))

    return rel_mse, rel_l2, rel_l1

def plot_forces_with_arrows(save_path, Coords_np, Z_phys_sample, d, title_prefix="", dpi=150):
    n_dofs = Z_phys_sample.shape[0]
    n_nodes = n_dofs // 2

    fx = np.array(Z_phys_sample[0::2, d], dtype=float)
    fy = np.array(Z_phys_sample[1::2, d], dtype=float)

    xs = np.array(Coords_np[:, 0], dtype=float)
    ys = np.array(Coords_np[:, 1], dtype=float)

    mags = np.sqrt(fx**2 + fy**2)
    nonzero = mags[mags > 0]

    bbox = max(xs.max() - xs.min(), ys.max() - ys.min(), 1.0)
    if nonzero.size > 0:
        med = np.median(nonzero)
        scale = (bbox * 0.05) / (med + 1e-20)
    else:
        scale = 1.0

    fig, ax = plt.subplots(figsize=(6,6), dpi=dpi)
    ax.scatter(xs, ys, s=30, zorder=1)
    U = fx * scale
    V = fy * scale
    ax.quiver(xs, ys, U, V, angles='xy', scale_units='xy', scale=1.0, width=0.004, zorder=2)

    idx = 1
    for i in range(n_nodes):
        if abs(fx[i]) > 1e-12 or abs(fy[i]) > 1e-12:
            ax.annotate(f"P{idx}", (xs[i], ys[i]), textcoords="offset points", xytext=(5,5), fontsize=9)
            idx += 1

    ax.set_aspect("equal")
    ax.set_title(title_prefix + " — Force vectors (P1,P2,...)", fontsize=PLOT_TITLE_FONTSIZE)
    style_plot_axes(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print("Saved force-arrow plot:", save_path)

def save_forces_csv(save_path, Coords_np, Z_phys_sample, d):
    n_dofs = Z_phys_sample.shape[0]
    n_nodes = n_dofs // 2
    fx = np.array(Z_phys_sample[0::2, d], dtype=float)
    fy = np.array(Z_phys_sample[1::2, d], dtype=float)

    with open(save_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["P_label", "node_id", "x", "y", "fx", "fy", "magnitude", "sign_fx", "sign_fy"])
        idx = 1
        for node in range(n_nodes):
            if abs(fx[node]) > 1e-12 or abs(fy[node]) > 1e-12:
                mag = float(np.sqrt(fx[node]**2 + fy[node]**2))
                sfx = "pos" if fx[node] > 0 else ("neg" if fx[node] < 0 else "zero")
                sfy = "pos" if fy[node] > 0 else ("neg" if fy[node] < 0 else "zero")
                writer.writerow([f"P{idx}", node, float(Coords_np[node,0]), float(Coords_np[node,1]),
                                 float(fx[node]), float(fy[node]), mag, sfx, sfy])
                idx += 1
    print("Saved forces CSV:", save_path)

def save_sample_csv_with_rel(idx, tag, nodes_np, n_nodes, T_test, per_sample_preds, mask_fixed_test,
                             per_sample_rel_mse, per_sample_rel_l2, per_sample_rel_l1, per_sample_errors, out_dir):
    true_u = T_test[idx].cpu().numpy()
    pred_u = per_sample_preds[idx]
    if isinstance(pred_u, torch.Tensor):
        pred_u = pred_u.cpu().numpy()

    ux_true = true_u[0::2]
    uy_true = true_u[1::2]
    ux_pred = pred_u[0::2]
    uy_pred = pred_u[1::2]

    fixed_dofs = mask_fixed_test[idx].numpy()
    fixed_x = fixed_dofs[0::2].astype(int)
    fixed_y = fixed_dofs[1::2].astype(int)
    csv_path = os.path.join(out_dir, f"test_pred_{tag}.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "x", "y", "ux_true", "uy_true", "ux_pred", "uy_pred", "fixed_x", "fixed_y"])
        for node_id in range(n_nodes):
            x = float(nodes_np[node_id, 0])
            y = float(nodes_np[node_id, 1])
            row = [
                node_id,
                x,
                y,
                float(ux_true[node_id]),
                float(uy_true[node_id]),
                float(ux_pred[node_id]),
                float(uy_pred[node_id]),
                int(fixed_x[node_id]),
                int(fixed_y[node_id])
            ]
            writer.writerow(row)
    meta_path = os.path.join(out_dir, f"test_pred_{tag}_metrics.txt")
    with open(meta_path, "w") as mf:
        mf.write(f"index: {idx}\n")
        mf.write(f"rel_MSE: {per_sample_rel_mse[idx]:.6e}\n")
        mf.write(f"rel_L2: {per_sample_rel_l2[idx]:.6e}\n")
        mf.write(f"rel_L1: {per_sample_rel_l1[idx]:.6e}\n")
        mf.write(f"phys_MSE: {per_sample_errors[idx]:.6e}\n")
    print("Saved CSV and metrics:", csv_path, meta_path)

def plot_error_deformation_overlay(save_path, Coords_np, edges, true_u, pred_u, mask_free,
                                   scale=1.0, dpi=150, tol=1e-12):
    err_u = pred_u - true_u
    err_u[~mask_free] = 0.0

    xs, ys = Coords_np[:, 0], Coords_np[:, 1]
    ux_e = err_u[0::2]
    uy_e = err_u[1::2]

    xs_err = xs + scale * ux_e
    ys_err = ys + scale * uy_e

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

    for (i, j) in edges:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], "-", color="0.7", linewidth=2)

    if np.max(np.abs(err_u)) > tol:
        for (i, j) in edges:
            ax.plot([xs_err[i], xs_err[j]], [ys_err[i], ys_err[j]], "-", color="k", linewidth=2)

    ax.scatter(xs, ys, color="0.7", s=30)
    ax.set_aspect("equal")
    style_plot_axes(ax, xlabel=None, ylabel=None)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()

def select_max_avg_target_case(T_test, mask_free_test):
    scores = []
    for i in range(T_test.shape[0]):
        u = T_test[i].cpu().numpy()
        m = mask_free_test[i].numpy()
        if m.any():
            scores.append(np.mean(np.abs(u[m])))
        else:
            scores.append(0.0)
    return int(np.argmax(scores))

def run_training(n_layer, n_head, n_nodes, d, feat_dim, Coords, edges,
                 n_samples=42000, epochs=2000):
    N_total = n_samples
    frac_train = 0.8
    frac_val = 0.1
    frac_test = 0.1

    B_inc = gen_incidence_from_edges(n_nodes, edges)
    D_mat = D_matrix(Coords, edges, E, A)

    print("Generating full pool of samples (this may take a while)...")
    Z_pool, T_pool = build_dataset_pool(N_total, B_inc, D_mat, Coords, edges, n_nodes, d, feat_dim)
    print("Pool ready:", Z_pool.shape, T_pool.shape)

    split_seed = 12345
    g = torch.Generator()
    g.manual_seed(split_seed)
    perm = torch.randperm(N_total, generator=g)

    n_train = int(N_total * frac_train)
    n_val = int(N_total * frac_val)
    n_test = N_total - n_train - n_val

    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]

    Z_train = Z_pool[idx_train].clone()
    T_train = T_pool[idx_train].clone()
    Z_val = Z_pool[idx_val].clone()
    T_val = T_pool[idx_val].clone()
    Z_test = Z_pool[idx_test].clone()
    T_test = T_pool[idx_test].clone()

    print(f"Split sizes -> train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}")

    pred_col_idx = d + 1
    mask_col_idx = d + 2

    Z_train_predcol = Z_train[:, :, pred_col_idx].clone()
    Z_val_predcol = Z_val[:, :, pred_col_idx].clone()
    Z_test_predcol = Z_test[:, :, pred_col_idx].clone()

    Z_train_mask = Z_train[:, :, mask_col_idx].clone()
    Z_val_mask = Z_val[:, :, mask_col_idx].clone()
    Z_test_mask = Z_test[:, :, mask_col_idx].clone()

    Z_mean = Z_train.mean(dim=(0, 1), keepdim=True)
    Z_std = Z_train.std(dim=(0, 1), keepdim=True) + 1e-8
    Z_train = (Z_train - Z_mean) / Z_std
    Z_val = (Z_val - Z_mean) / Z_std
    Z_test = (Z_test - Z_mean) / Z_std

    Z_train[:, :, mask_col_idx] = Z_train_mask
    Z_val[:, :, mask_col_idx] = Z_val_mask
    Z_test[:, :, mask_col_idx] = Z_test_mask

    Z_train[:, :, pred_col_idx] = Z_train_predcol
    Z_val[:, :, pred_col_idx] = Z_val_predcol
    Z_test[:, :, pred_col_idx] = Z_test_predcol

    u_mean = T_train.mean(dim=0, keepdim=True)
    u_std = T_train.std(dim=0, keepdim=True) + 1e-8
    T_train_n = (T_train - u_mean) / u_std
    T_val_n = (T_val - u_mean) / u_std
    T_test_n = (T_test - u_mean) / u_std

    print("Normalized inputs & targets (BC mask and pred col preserved).")
    print("u_mean shape", u_mean.shape, "u_std shape", u_std.shape)

    model = Transformer_LG(n_layer=n_layer, n_head=1, n_nodes=n_nodes, d=d, feat_dim=feat_dim, var=0.1).to(device)
    allparam_params = [p for n, p in model.named_parameters() if 'allparam' in n]
    other_params = [p for n, p in model.named_parameters() if 'allparam' not in n]
    opt = torch.optim.AdamW([
        {'params': allparam_params, 'lr': 3e-5},
        {'params': other_params, 'lr': 1e-5}
    ], betas=(0.9, 0.99), weight_decay=0.0)

    clip_r = 1.0
    batch_size, n_epochs = 32, epochs

    def iterate(Z, T, shuffle=True):
        Nall = Z.shape[0]
        idx = torch.randperm(Nall) if shuffle else torch.arange(Nall)
        for i in range(0, Nall, batch_size):
            batch = idx[i: i + batch_size]
            yield Z[batch].to(device), T[batch].to(device)

    def compute_full_dataset_loss(model_local, Z_all, T_all, batch_eval=256):
        model_local.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for i0 in range(0, Z_all.shape[0], batch_eval):
                Zb = Z_all[i0:i0+batch_eval].to(device)
                Tb = T_all[i0:i0+batch_eval].to(device)
                out = model_local(Zb)
                l, _, _ = get_loss(out, Tb, Zb, n_nodes=n_nodes, d=d)
                total += l.item() * Zb.size(0)
                count += Zb.size(0)
        return (total / max(1, count))

    best_val = float("inf")
    history = {'train_loss': [], 'val_loss': []}
    best_ckpt_path = os.path.join(cur_dir, "best_by_val.pth")
    history_path = os.path.join(cur_dir, "history.pth")

    for epoch in range(1, n_epochs + 1):
        model.train()
        for Zb, Tb in iterate(Z_train, T_train_n):
            opt.zero_grad()
            out = model(Zb)
            loss, _, _ = get_loss(out, Tb, Zb, n_nodes=n_nodes, d=d)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_r)
            opt.step()

        model.eval()
        train_loss_full = compute_full_dataset_loss(model, Z_train, T_train_n)
        val_loss_full = compute_full_dataset_loss(model, Z_val, T_val_n)

        history['train_loss'].append(train_loss_full)
        history['val_loss'].append(val_loss_full)
        try:
            torch.save(history, history_path)
        except Exception as e:
            print("Could not save history:", e)

        if val_loss_full < best_val:
            best_val = val_loss_full
            print(f"Epoch {epoch:04d} | train(full) {train_loss_full:.3e} | val {val_loss_full:.3e}  <-- best")
            ckpt = {
                'model_state': model.state_dict(),
                'Z_mean': Z_mean.cpu(),
                'Z_std': Z_std.cpu(),
                'u_mean': u_mean.cpu(),
                'u_std': u_std.cpu(),
                'epoch': epoch,
                'best_val': best_val
            }
            try:
                torch.save(ckpt, best_ckpt_path)
            except Exception as e:
                print("Could not save best checkpoint:", e)
        else:
            if epoch == 1 or epoch % 20 == 0:
                print(f"Epoch {epoch:04d} | train(full) {train_loss_full:.3e} | val {val_loss_full:.3e}")

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f"Loaded best-by-val checkpoint from epoch {ckpt.get('epoch', 'N/A')} (val={ckpt.get('best_val', 'N/A'):.3e})")
    else:
        print("No checkpoint found; using final model weights for test evaluation.")

    model.eval()

    Z_test_phys_cpu = (Z_test.cpu() * Z_std.cpu()) + Z_mean.cpu()
    nodes_np_local = Coords.cpu().numpy()

    per_sample_errors = []
    per_sample_preds = []

    with torch.no_grad():
        B_test = Z_test.shape[0]
        bs_eval = 128
        for i0 in range(0, B_test, bs_eval):
            Zb = Z_test[i0: i0 + bs_eval].to(device)
            Tb = T_test[i0: i0 + bs_eval].to(device)
            out = model(Zb)
            pred_norm = out[:, :, d + 1]
            pred_phys = pred_norm * u_std.to(device) + u_mean.to(device)
            mask_free = Zb[:, :, d + 2].bool()
            for b in range(pred_phys.shape[0]):
                m = mask_free[b]
                if m.any():
                    tgt = Tb[b]
                    p = pred_phys[b]
                    mse = ((p[m] - tgt[m]) ** 2).mean().item()
                else:
                    mse = 0.0
                per_sample_errors.append(mse)
                pv = pred_phys[b].cpu().clone()
                per_sample_preds.append(pv)

    per_sample_errors = np.array(per_sample_errors)
    per_sample_preds = [p.cpu() for p in per_sample_preds]

    mask_free_test = Z_test[:, :, d + 2].bool().cpu()
    mask_fixed_test = ~mask_free_test

    for ii in range(len(per_sample_preds)):
        p = per_sample_preds[ii].clone()
        fixed_idx = mask_fixed_test[ii]
        if fixed_idx.any():
            p[fixed_idx] = 0.0
        per_sample_preds[ii] = p

    per_sample_rel_mse = []
    per_sample_rel_l2 = []
    per_sample_rel_l1 = []
    for ii in range(len(per_sample_preds)):
        true_u = T_test[ii].cpu().numpy()
        pred_u = per_sample_preds[ii].cpu().numpy()
        mask_free_np = mask_free_test[ii].numpy()
        rel_mse, rel_l2, rel_l1 = compute_relative_errors_phys(true_u, pred_u, mask_free_np, eps=1e-12)
        per_sample_rel_mse.append(rel_mse)
        per_sample_rel_l2.append(rel_l2)
        per_sample_rel_l1.append(rel_l1)

    per_sample_rel_mse = np.array(per_sample_rel_mse)
    per_sample_rel_l2  = np.array(per_sample_rel_l2)
    per_sample_rel_l1  = np.array(per_sample_rel_l1)

    idx_worst = int(np.argmax(per_sample_rel_l2))
    idx_median = int(np.argsort(per_sample_rel_l2)[len(per_sample_rel_l2) // 2])
    rng = np.random.RandomState(999)
    idx_random = int(rng.choice(len(per_sample_rel_l2)))

    idx_max_avg = select_max_avg_target_case(T_test, mask_free_test)

    N_test = len(per_sample_rel_mse)
    if N_test == 0:
        test_rel_MSE_percent = test_rel_L2_percent = test_rel_L1_percent = 0.0
    else:
        test_rel_MSE_percent = 100.0 * per_sample_rel_mse.sum() / N_test
        test_rel_L2_percent  = 100.0 * per_sample_rel_l2.sum()  / N_test
        test_rel_L1_percent  = 100.0 * per_sample_rel_l1.sum()  / N_test

    save_sample_csv_with_rel(idx_worst, "worst", nodes_np_local, n_nodes, T_test, per_sample_preds, mask_fixed_test,
                             per_sample_rel_mse, per_sample_rel_l2, per_sample_rel_l1, per_sample_errors, cur_dir)
    save_sample_csv_with_rel(idx_median, "median", nodes_np_local, n_nodes, T_test, per_sample_preds, mask_fixed_test,
                             per_sample_rel_mse, per_sample_rel_l2, per_sample_rel_l1, per_sample_errors, cur_dir)
    save_sample_csv_with_rel(idx_random, "random", nodes_np_local, n_nodes, T_test, per_sample_preds, mask_fixed_test,
                             per_sample_rel_mse, per_sample_rel_l2, per_sample_rel_l1, per_sample_errors, cur_dir)

    try:
        summary_path = os.path.join(cur_dir, "test_error_summary_percent.txt")
        with open(summary_path, "w") as f:
            f.write("GLOBAL TEST-SET ERRORS (PHYSICAL SPACE)\n")
            f.write(f"Relative MSE (%) : {test_rel_MSE_percent:.6e}\n")
            f.write(f"Relative L2  (%) : {test_rel_L2_percent:.6e}\n")
            f.write(f"Relative L1  (%) : {test_rel_L1_percent:.6e}\n")
        print("Saved global test error summary:", summary_path)
    except Exception as e:
        print("Could not save global test summary:", e)

    try:
        plt.figure(figsize=(6,4), dpi=150)
        epochs = np.arange(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], label="training")
        plt.plot(epochs, history['val_loss'], label="validation")
        plt.yscale("log")
        plt.xlim(1, len(history['train_loss']))
        plt.xlabel("Epoch", fontsize=PLOT_LABEL_FONTSIZE)
        plt.ylabel("Loss (log scale)", fontsize=PLOT_LABEL_FONTSIZE)
        plt.tick_params(axis="both", which="major", labelsize=PLOT_TICK_FONTSIZE)
        plt.legend(fontsize=PLOT_LEGEND_FONTSIZE)
        plt.tight_layout()
        loss_out_log = os.path.join(cur_dir, f"loss_curve_train_val_log.png")
        plt.savefig(loss_out_log, dpi=150)
        plt.close()
        print("Saved loss plot (log):", loss_out_log)
    except Exception as e:
        print("Could not save loss curves:", e)

    try:
        true_u_worst = T_test[idx_worst].cpu().numpy()
        pred_u_worst = per_sample_preds[idx_worst].cpu().numpy()
        mask_free_worst = mask_free_test[idx_worst].cpu().numpy()

        out_true_pred_worst = os.path.join(cur_dir, f"test_worst_true_pred.png")
        plot_true_pred_pair(out_true_pred_worst, nodes_np_local, edges, Z_test[idx_worst], T_test[idx_worst], per_sample_preds[idx_worst], title_prefix=f"Worst (idx={idx_worst})")
        print("Saved worst true/pred plot:", out_true_pred_worst)

        out_err_overlay = os.path.join(cur_dir, f"test_worst_error_overlay.png")
        plot_error_deformation_overlay(out_err_overlay, nodes_np_local, edges, true_u_worst, pred_u_worst, mask_free_worst)
        print("Saved worst error overlay:", out_err_overlay)

        Z_phys_worst = Z_test_phys_cpu[idx_worst].numpy()
        forces_plot_path = os.path.join(cur_dir, f"worst_forces_idx{idx_worst}.png")
        forces_csv_path = os.path.join(cur_dir, f"worst_forces_idx{idx_worst}_P_list.csv")
        plot_forces_with_arrows(forces_plot_path, nodes_np_local, Z_phys_worst, d, title_prefix=f"Worst sample idx={idx_worst}")
        save_forces_csv(forces_csv_path, nodes_np_local, Z_phys_worst, d)
    except Exception as e:
        print("Could not generate worst-case plots/CSVs:", e)

    try:
        true_u_max = T_test[idx_max_avg].cpu().numpy()
        pred_u_max = per_sample_preds[idx_max_avg].cpu().numpy()
        mask_free_max = mask_free_test[idx_max_avg].cpu().numpy()

        out_true_pred_max = os.path.join(cur_dir, f"test_maxavg_true_pred.png")
        plot_true_pred_pair(out_true_pred_max, nodes_np_local, edges, Z_test[idx_max_avg], T_test[idx_max_avg], per_sample_preds[idx_max_avg], title_prefix=f"MaxAvgTarget (idx={idx_max_avg})")
        print("Saved max-avg true/pred plot:", out_true_pred_max)

        out_err_overlay_max = os.path.join(cur_dir, f"test_maxavg_error_overlay.png")
        plot_error_deformation_overlay(out_err_overlay_max, nodes_np_local, edges, true_u_max, pred_u_max, mask_free_max)
        print("Saved max-avg error overlay:", out_err_overlay_max)

        Z_phys_max = Z_test_phys_cpu[idx_max_avg].numpy()
        forces_plot_path2 = os.path.join(cur_dir, f"maxavg_forces_idx{idx_max_avg}.png")
        forces_csv_path2 = os.path.join(cur_dir, f"maxavg_forces_idx{idx_max_avg}_P_list.csv")
        plot_forces_with_arrows(forces_plot_path2, nodes_np_local, Z_phys_max, d, title_prefix=f"MaxAvg sample idx={idx_max_avg}")
        save_forces_csv(forces_csv_path2, nodes_np_local, Z_phys_max, d)
    except Exception as e:
        print("Could not generate max-avg plots/CSVs:", e)

    print("Test errors summary (physical MSE over free DOFs):")
    print("worst idx:", idx_worst, "phys_MSE:", per_sample_errors[idx_worst] if len(per_sample_errors)>0 else None,
          "rel_L2:", per_sample_rel_l2[idx_worst], "rel_MSE:", per_sample_rel_mse[idx_worst], "rel_L1:", per_sample_rel_l1[idx_worst])
    print("max-avg idx:", idx_max_avg, "phys_MSE:", per_sample_errors[idx_max_avg] if len(per_sample_errors)>0 else None,
          "rel_L2:", per_sample_rel_l2[idx_max_avg], "rel_MSE:", per_sample_rel_mse[idx_max_avg], "rel_L1:", per_sample_rel_l1[idx_max_avg])
    print(f"\nGLOBAL (mean over {N_test} samples) Relative MSE (%): {test_rel_MSE_percent:.6e}")
    print(f"GLOBAL (mean over {N_test} samples) Relative L2  (%): {test_rel_L2_percent:.6e}")
    print(f"GLOBAL (mean over {N_test} samples) Relative L1  (%): {test_rel_L1_percent:.6e}")

    print("Saved best-by-val checkpoint path:", best_ckpt_path)

    return (model, Z_mean, Z_std, u_mean, u_std, B_inc, D_mat, Coords, edges,
            history, Z_train, T_train, Z_val, T_val, Z_test, T_test,
            idx_worst, idx_median, idx_random, per_sample_preds, per_sample_errors, best_ckpt_path)

def split_uv(uvec):
    ux = uvec[0::2]
    uy = uvec[1::2]
    return ux, uy

def plot_true_pred_pair(save_path, Coords_np, edges, Z_raw, T_raw, pred_u, title_prefix="", scale=1.0, dpi=150):
    true_u = T_raw.cpu().numpy()
    if isinstance(pred_u, torch.Tensor):
        pred_u = pred_u.cpu().numpy()
    xs, ys = Coords_np[:, 0], Coords_np[:, 1]
    ux_true, uy_true = split_uv(true_u)
    ux_pred, uy_pred = split_uv(pred_u)
    xs_true = xs + scale * ux_true
    ys_true = ys + scale * uy_true
    xs_pred = xs + scale * ux_pred
    ys_pred = ys + scale * uy_pred

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)
    for (i, j) in edges:
        axs[0].plot([xs[i], xs[j]], [ys[i], ys[j]], "--", color="0.7")
        axs[1].plot([xs[i], xs[j]], [ys[i], ys[j]], "--", color="0.7")
    for (i, j) in edges:
        axs[0].plot([xs_true[i], xs_true[j]], [ys_true[i], ys_true[j]], "-", color="orange", linewidth=2)
        axs[1].plot([xs_pred[i], xs_pred[j]], [ys_pred[i], ys_pred[j]], "-", color="green", linewidth=2)
    axs[0].scatter(xs, ys, color="k", s=30)
    axs[1].scatter(xs, ys, color="k", s=30)
    axs[0].scatter(xs_true, ys_true, color="orange", s=50, edgecolors="k")
    axs[1].scatter(xs_pred, ys_pred, color="green", s=50, edgecolors="k")
    axs[0].set_aspect("equal")
    axs[1].set_aspect("equal")
    for ax in axs:
        style_plot_axes(ax, xlabel=None, ylabel=None)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print("Saved:", save_path)


def main():
    global device, cur_dir
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/5x5"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=42000)
    parser.add_argument("--smoke", action="store_true", help="One epoch with 20 samples; execution check only")
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 10:
        parser.error("Use positive epochs and at least 10 samples")
    if args.smoke:
        args.epochs, args.samples = 1, 20
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    cur_dir = str(output)
    requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    nodes, edges = tile_unit_cell([(0.,0.), (1.,0.), (0.,1.), (1.,1.)], 5, 5)
    coords = torch.tensor(nodes, dtype=torch.float64, device=device)
    config = dict(case="5x5", samples=args.samples, epochs=args.epochs,
                  layers=7, heads=1, seed=42, split_seed=12345, device=str(device),
                  dtype=str(coords.dtype), smoke=args.smoke,
                  python=__import__("platform").python_version(),
                  torch=torch.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__)
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    run_training(7, 1, len(nodes), len(edges), len(edges)+3, coords, edges,
                 n_samples=args.samples, epochs=args.epochs)
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
