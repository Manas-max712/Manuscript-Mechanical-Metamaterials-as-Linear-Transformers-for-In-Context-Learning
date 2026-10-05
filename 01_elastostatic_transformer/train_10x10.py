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

torch.set_default_dtype(torch.float32)
device = torch.device("cpu")
cur_dir = "runs/10x10"
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

def gen_incidence_from_edges(n_nodes, edges_list, out_device=None):
    if out_device is None:
        out_device = torch.device('cpu')
    d_local = len(edges_list)
    B = torch.zeros(n_nodes, d_local, device=out_device)
    for e_id, (i, j) in enumerate(edges_list):
        B[i, e_id] = 1.0
        B[j, e_id] = -1.0
    return B

def B_strain(B, Coords):
    if isinstance(Coords, torch.Tensor):
        Coords_t = Coords.to(Coords.device).float()
    else:
        Coords_t = torch.tensor(Coords, dtype=torch.float32)
    if isinstance(B, torch.Tensor):
        B = B.to(Coords_t.device)
    n_local, d_local = B.shape
    B_new = torch.zeros(2 * n_local, d_local, device=Coords_t.device)
    eps = 1e-12
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
    if isinstance(Coords, torch.Tensor):
        Coords_t = Coords.to(Coords.device).float()
    else:
        Coords_t = torch.tensor(Coords, dtype=torch.float32)
    d_local = len(edges)
    D = torch.zeros(d_local, d_local, device=Coords_t.device)
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
    if Coords is None:
        raise ValueError("Coords required")
    C = torch.tensor(np.array(Coords, dtype=float), dtype=torch.float32).to("cpu")
    if C.ndim != 2 or C.shape[0] != int(n_nodes) or C.shape[1] != 2:
        raise ValueError("Coords shape mismatch")

    y = C[:, 1]
    x = C[:, 0]
    y_min = float(y.min().item())
    bottom_mask = (y - y_min).abs() <= tol
    bottom_nodes = torch.nonzero(bottom_mask, as_tuple=False).squeeze(1)
    if bottom_nodes.numel() == 0:
        raise RuntimeError("No bottom-edge nodes found")

    x_bottom = x[bottom_nodes]
    idx_min = torch.argmin(x_bottom)
    bottom_left_node = int(bottom_nodes[idx_min].item())

    fixed = set()
    for node in bottom_nodes.tolist():
        fixed.add(2 * int(node) + 1)
    fixed.add(2 * bottom_left_node)
    fixed.add(2 * bottom_left_node + 1)

    fixed_sorted = sorted(fixed)
    return torch.tensor(fixed_sorted, dtype=torch.long)

def generate_B_inplace(Z, nv, incidence_B_input, Coords, mask_free, perm, D_local):
    dev = incidence_B_input.device if isinstance(incidence_B_input, torch.Tensor) else torch.device('cpu')
    B_perm = incidence_B_input[:, perm].to(dev)
    B_mat = B_strain(B_perm, Coords)
    if not isinstance(mask_free, torch.Tensor):
        mask_free = torch.as_tensor(mask_free, dtype=torch.bool, device=dev)
    else:
        mask_free = mask_free.to(dev).bool()
    diagD = torch.sqrt(torch.clamp(torch.diag(D_local), min=0.0))[perm].to(dev)
    B_ff = B_mat[mask_free, :]
    B_tilde = B_ff * diagD.unsqueeze(0)
    d_local = B_tilde.shape[1]
    Z = Z.to(dev)
    Z[:, :, 0:d_local] = 0.0
    free_idx = torch.nonzero(mask_free, as_tuple=False).squeeze(1).to(dev)
    if free_idx.numel() > 0:
        Z[:, free_idx, 0:d_local] = B_tilde.unsqueeze(0).repeat(Z.shape[0], 1, 1)
    return Z

def generate_demand_inplace(Z, nv, d):
    Z[:, :, d] = 0.0
    return Z

def apply_demand_on_mask(Z, mask_free, d, sample_idx=0, generator=None,
                         mean_val=50000.0, std_val=10000.0,
                         low=10000.0, high=90000.0):
    dev = Z.device
    if not isinstance(mask_free, torch.Tensor):
        mask_free = torch.as_tensor(mask_free, dtype=torch.bool, device=dev)
    else:
        mask_free = mask_free.to(dev).bool()
    B = Z.shape[0]
    n_dofs = mask_free.shape[0]
    for b in range(B):
        if generator is None:
            mags = torch.normal(mean=mean_val, std=std_val, size=(n_dofs,), device=dev)
        else:
            if dev.type == 'cpu':
                mags = torch.normal(mean=mean_val, std=std_val, size=(n_dofs,), generator=generator, device=dev)
            else:
                mags = torch.normal(mean=mean_val, std=std_val, size=(n_dofs,)).to(dev)
        mags = torch.clamp(mags, min=low, max=high)
        signs = torch.randint(0, 2, (n_dofs,), dtype=torch.float32, device=dev)
        signs = 2.0 * signs - 1.0
        sample = mags * signs
        sample_masked = torch.zeros_like(sample)
        sample_masked[mask_free] = sample[mask_free]
        Z[b, :, d] = sample_masked
    return Z

def get_target(Z, D_local, Coords_local, edges, n_nodes, d, feat_dim):
    batch_size = Z.shape[0]
    n_dofs = 2 * n_nodes
    lap = torch.zeros(batch_size, n_dofs, n_dofs, device=Z.device)
    B_block = Z[:, :n_dofs, 0:d]
    for b in range(batch_size):
        lap[b] = stiffness_matrix(B_block[b])
    target_disp = torch.zeros(batch_size, n_dofs, device=Z.device)
    for b in range(batch_size):
        K_b = lap[b]
        f_b = Z[b, :, d].clone()
        mask_free = Z[b, :, d + 2].bool()
        mask_fixed = ~mask_free
        f_b[mask_fixed] = 0.0
        if mask_free.sum() == 0:
            full_u = torch.zeros(n_dofs, device=Z.device)
        else:
            K_ff = K_b[mask_free][:, mask_free]
            f_f = f_b[mask_free]
            try:
                eps_reg = 1e-8 * (torch.trace(K_ff).abs() / max(1, K_ff.shape[0]))
                K_ff_reg = K_ff + eps_reg * torch.eye(K_ff.shape[0], device=Z.device)
                uf = torch.linalg.solve(K_ff_reg, f_f)
            except Exception:
                uf = torch.linalg.pinv(K_ff) @ f_f
            full_u = torch.zeros(n_dofs, device=Z.device)
            full_u[mask_free] = uf
        target_disp[b] = full_u
    return target_disp

def get_loss(output, target, Z, n_nodes, d):
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
    QK = torch.einsum("Hji,Hjk->Hik", Q, K)
    Attn = torch.einsum("BNi,Hij,BMj->HBNM", Z, QK, Z)
    PZ = torch.einsum("Hij,BNj->HBNi", P, Z)
    out = torch.einsum("HBNi,HBNM->BMi", PZ, Attn) / max(1, N)
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
    g = torch.Generator()
    g.manual_seed(int(sample_seed))
    n_dofs = 2 * n_nodes
    Z = torch.zeros(1, n_dofs, feat_dim, device=torch.device('cpu'))
    fixed = choose_kinematically_stable_dofs(n_nodes, Coords=Coords_local, generator=g)
    mask_free = torch.ones(n_dofs, dtype=torch.bool, device=torch.device('cpu'))
    mask_free[fixed] = False

    rem = torch.nonzero(mask_free).squeeze(1)
    extra = max(0, int(p_fix * n_dofs) - (~mask_free).sum().item())
    if rem.numel() > 0 and extra > 0:
        chosen = rem[torch.randperm(rem.numel(), generator=g)[:extra]]
        mask_free[chosen] = False

    Z[0, :, d + 2] = mask_free.float()
    perm = torch.arange(B_inc.shape[1])
    Z = generate_B_inplace(Z, n_nodes, B_inc.cpu(), Coords_local.cpu(), mask_free.cpu(), perm, D_local.cpu())
    Dp = permute_D_diag(D_local.cpu(), perm)
    Z = generate_demand_inplace(Z, n_nodes, d)
    Z[:, :, d + 1] = 0.0
    Z = apply_demand_on_mask(Z, mask_free.cpu(), d=d, sample_idx=sample_seed, generator=g)
    target = get_target(Z, Dp, Coords_local.cpu(), edges, n_nodes, d, feat_dim)
    return Z.squeeze(0).cpu(), target.squeeze(0).cpu()

class OnTheFlyDataset(Dataset):
    def __init__(self, indices, B_inc, D_local, Coords_local, edges, n_nodes, d, feat_dim, p_fix=0.0):
        self.indices = indices
        self.B_inc = B_inc.cpu()
        self.D_local = D_local.cpu()
        self.Coords_local = Coords_local.cpu()
        self.edges = edges
        self.n_nodes = n_nodes
        self.d = d
        self.feat_dim = feat_dim
        self.p_fix = p_fix

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_seed = int(self.indices[idx].item())
        Z_sample, T_sample = make_sample_Z(
            self.B_inc, self.D_local, self.Coords_local,
            self.edges, self.n_nodes, self.d,
            self.feat_dim, p_fix=self.p_fix, sample_seed=global_seed
        )
        return Z_sample.to(torch.float32), T_sample.to(torch.float32)

def split_uv(uvec):
    ux = uvec[0::2]
    uy = uvec[1::2]
    return ux, uy

def plot_true_pred_pair(save_path, Coords_np, edges, T_raw, pred_u, title_prefix="", scale=1.0, dpi=150):
    true_u = T_raw.cpu().numpy()
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

def plot_error_deformation(save_path, Coords_np, edges, T_raw, pred_u, title="", scale=1.0, dpi=150):
    if isinstance(T_raw, torch.Tensor):
        T_arr = T_raw.detach().cpu().numpy()
    else:
        T_arr = np.array(T_raw)

    if isinstance(pred_u, torch.Tensor):
        P_arr = pred_u.detach().cpu().numpy()
    else:
        P_arr = np.array(pred_u)

    xs = Coords_np[:, 0]
    ys = Coords_np[:, 1]

    ux_true = T_arr[0::2]
    uy_true = T_arr[1::2]
    ux_pred = P_arr[0::2]
    uy_pred = P_arr[1::2]

    ux_err = ux_true - ux_pred
    uy_err = uy_true - uy_pred

    xs_err = xs + scale * ux_err
    ys_err = ys + scale * uy_err

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)

    for (i, j) in edges:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="0.85", linestyle="-", linewidth=1)
    ax.scatter(xs, ys, color="0.6", s=20, zorder=1)

    for (i, j) in edges:
        ax.plot([xs_err[i], xs_err[j]], [ys_err[i], ys_err[j]], color="orange", linestyle="-", linewidth=1.5, zorder=3)
    ax.scatter(xs_err, ys_err, color="orange", s=30, edgecolors="k", zorder=4)

    ax.set_aspect("equal")
    style_plot_axes(ax, xlabel=None, ylabel=None)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print("Saved:", save_path)

def save_sample_csv_global_generic(out_dir, prefix, nodes_np, T_raw, pred_raw, mask_fixed):
    if not isinstance(nodes_np, np.ndarray):
        nodes_np = np.array(nodes_np, dtype=float)

    t_raw = T_raw.detach().cpu().numpy() if isinstance(T_raw, torch.Tensor) else np.array(T_raw)
    p_raw = pred_raw.detach().cpu().numpy() if isinstance(pred_raw, torch.Tensor) else np.array(pred_raw)

    if isinstance(mask_fixed, torch.Tensor):
        fixed = mask_fixed.detach().cpu().numpy().reshape(-1).astype(bool)
    else:
        fixed = np.array(mask_fixed).reshape(-1).astype(bool)

    n_dofs = t_raw.shape[0]
    n_nodes = nodes_np.shape[0]
    if n_dofs != 2 * n_nodes:
        raise RuntimeError(f"Mismatch: n_dofs={n_dofs} vs 2*n_nodes={2*n_nodes}")

    ux_true = t_raw[0::2]
    uy_true = t_raw[1::2]
    ux_pred = p_raw[0::2]
    uy_pred = p_raw[1::2]
    ux_err = ux_true - ux_pred
    uy_err = uy_true - uy_pred

    fixed_x = fixed[0::2].astype(int)
    fixed_y = fixed[1::2].astype(int)

    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "x", "y",
                         "ux_true", "uy_true",
                         "ux_pred", "uy_pred",
                         "ux_err", "uy_err",
                         "fixed_x", "fixed_y"])
        for node_id in range(nodes_np.shape[0]):
            writer.writerow([
                node_id,
                float(nodes_np[node_id, 0]),
                float(nodes_np[node_id, 1]),
                float(ux_true[node_id]),
                float(uy_true[node_id]),
                float(ux_pred[node_id]),
                float(uy_pred[node_id]),
                float(ux_err[node_id]),
                float(uy_err[node_id]),
                int(fixed_x[node_id]),
                int(fixed_y[node_id]),
            ])
    print("Saved CSV:", csv_path)

def plot_force_vectors(save_path, nodes_np, fx, fy, title="Forces on undeformed grid",
                       dpi=150, label_nodes=False, label_every=10,
                       arrow_frac=0.06, cmap='viridis'):
    nodes = np.asarray(nodes_np)
    fx = np.asarray(fx).astype(float)
    fy = np.asarray(fy).astype(float)
    if nodes.shape[0] != fx.shape[0] or nodes.shape[0] != fy.shape[0]:
        raise ValueError("nodes_np and fx/fy must have matching lengths")

    xs = nodes[:, 0]
    ys = nodes[:, 1]

    vec_mags = np.sqrt(fx**2 + fy**2)
    max_mag = float(np.max(vec_mags)) if vec_mags.size > 0 else 0.0

    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()
    diag = math.hypot(x_range, y_range) if (x_range > 0 or y_range > 0) else 1.0

    desired_arrow_length = max(1e-6, arrow_frac * diag)

    if max_mag <= 0:
        multiplier = 1.0
    else:
        multiplier = desired_arrow_length / max_mag

    fx_plot = fx * multiplier
    fy_plot = fy * multiplier
    plot_mags = vec_mags * multiplier

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=dpi)
    ax.set_aspect("equal")

    try:
        for (i, j) in edges:
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="0.9", linestyle="-", linewidth=0.7, zorder=0)
    except Exception:
        ax.plot([xs.min(), xs.max(), xs.max(), xs.min(), xs.min()],
                [ys.min(), ys.min(), ys.max(), ys.max(), ys.min()],
                color="0.9", linewidth=1.0, zorder=0)

    ax.scatter(xs, ys, color="k", s=18, zorder=2)

    Q = ax.quiver(xs, ys, fx_plot, fy_plot,
                  plot_mags,
                  angles='xy', scale_units='xy', scale=1.0,
                  width=0.008, headwidth=6, headlength=6, headaxislength=5,
                  cmap=cmap, zorder=3)

    cbar = fig.colorbar(Q, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('force magnitude (scaled for visibility)', fontsize=PLOT_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=PLOT_TICK_FONTSIZE)

    if label_nodes:
        for i, (x, y) in enumerate(zip(xs, ys)):
            if (i % label_every) == 0:
                ax.text(x, y, f"P{i+1}", fontsize=6, ha='left', va='bottom', zorder=4)

    xpad = x_range * 0.03 if x_range > 0 else 0.5
    ypad = y_range * 0.03 if y_range > 0 else 0.5
    ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
    ax.set_ylim(ys.min() - ypad, ys.max() + ypad)

    ax.set_title(title, fontsize=PLOT_TITLE_FONTSIZE)
    style_plot_axes(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print("Saved force vector plot:", save_path)

def save_forces_csv(save_path, nodes_np, fx, fy):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "x", "y", "fx", "fy"])
        for i in range(nodes_np.shape[0]):
            writer.writerow([i, float(nodes_np[i, 0]), float(nodes_np[i, 1]),
                             float(fx[i]), float(fy[i])])
    print("Saved forces CSV:", save_path)

def run_training(n_layer, n_head, n_nodes, d, feat_dim, Coords, edges,
                 nodes_np,
                 resume_ckpt_path=None, n_samples=42000, epochs=200, workers=None):
    if resume_ckpt_path is None:
        resume_ckpt_path = os.path.join(cur_dir, "resume_checkpoint.pth")

    N_total = n_samples
    frac_train = 0.8
    frac_val = 0.1
    frac_test = 0.1

    B_inc = gen_incidence_from_edges(n_nodes, edges, out_device=torch.device('cpu')).cpu()
    D_mat = D_matrix(Coords.cpu(), edges, E, A).cpu()

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

    ds_train = OnTheFlyDataset(idx_train, B_inc, D_mat, Coords.cpu(), edges, n_nodes, d, feat_dim)
    ds_val   = OnTheFlyDataset(idx_val,   B_inc, D_mat, Coords.cpu(), edges, n_nodes, d, feat_dim)
    ds_test  = OnTheFlyDataset(idx_test,  B_inc, D_mat, Coords.cpu(), edges, n_nodes, d, feat_dim)

    def allocated_cpus():
        for v in ("SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE"):
            if v in os.environ:
                try:
                    return int(os.environ[v])
                except Exception:
                    pass
        try:
            return int(os.cpu_count() or 1)
        except Exception:
            return 1

    num_cpus_local = allocated_cpus()
    num_workers = min(8, max(1, num_cpus_local // 2)) if workers is None else workers
    print(f"Detected allocated_cpus={num_cpus_local}; using num_workers={num_workers}")

    eval_batch = 128
    train_batch = 16

    start_epoch = 1
    best_val = float("inf")
    history = {'train_loss': [], 'val_loss': []}

    model = Transformer_LG(n_layer=n_layer, n_head=n_head, n_nodes=n_nodes, d=d, feat_dim=feat_dim, var=0.1).to(device)
    allparam_params = [p for n, p in model.named_parameters() if 'allparam' in n]
    other_params = [p for n, p in model.named_parameters() if 'allparam' not in n]
    opt = torch.optim.AdamW([
        {'params': allparam_params, 'lr': 3e-5},
        {'params': other_params, 'lr': 1e-5}
    ], betas=(0.9, 0.99), weight_decay=0.0)

    Z_mean = None; Z_std = None; u_mean = None; u_std = None

    pred_col_idx = d + 1
    mask_col_idx = d + 2

    if os.path.exists(resume_ckpt_path):
        print("Resuming from checkpoint:", resume_ckpt_path)
        ckpt = torch.load(resume_ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        opt.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val = ckpt.get('best_val', float("inf"))
        history = ckpt.get('history', history)
        Z_mean = ckpt['Z_mean']
        Z_std = ckpt['Z_std']
        u_mean = ckpt['u_mean']
        u_std = ckpt['u_std']
        idx_train = ckpt.get('idx_train', idx_train)
        idx_val = ckpt.get('idx_val', idx_val)
        idx_test = ckpt.get('idx_test', idx_test)
        torch.set_rng_state(ckpt['torch_rng_state'])
        np.random.set_state(ckpt['numpy_rng_state'])
        random.setstate(ckpt['random_state'])
        print(f"Loaded checkpoint epoch {ckpt['epoch']}. Resuming at epoch {start_epoch}.")

    if Z_mean is None:
        print("Computing normalization statistics (streaming over training set)...")
        dl_stream = DataLoader(ds_train, batch_size=train_batch, shuffle=False, num_workers=num_workers, pin_memory=False)

        sum_Z = torch.zeros(1, 1, feat_dim, dtype=torch.float64)
        sumsq_Z = torch.zeros_like(sum_Z)
        count_Z = 0

        sum_u = torch.zeros(1, 2*n_nodes, dtype=torch.float64)
        sumsq_u = torch.zeros_like(sum_u)
        count_u = 0

        for Zb_cpu, Tb_cpu in dl_stream:
            Bbatch = Zb_cpu.shape[0]
            sum_Z += Zb_cpu.double().sum(dim=(0,1), keepdim=True)
            sumsq_Z += (Zb_cpu.double() ** 2).sum(dim=(0,1), keepdim=True)
            count_Z += Bbatch * (2 * n_nodes)

            sum_u += Tb_cpu.double().sum(dim=0, keepdim=True)
            sumsq_u += (Tb_cpu.double() ** 2).sum(dim=0, keepdim=True)
            count_u += Bbatch

        Z_mean = (sum_Z / float(count_Z)).to(torch.float32)
        Z_var = (sumsq_Z / float(count_Z) - (Z_mean.double() ** 2)).clamp(min=1e-12)
        Z_std = torch.sqrt(Z_var).to(torch.float32)

        u_mean = (sum_u / float(count_u)).to(torch.float32)
        u_var = (sumsq_u / float(count_u) - (u_mean.double() ** 2)).clamp(min=1e-12)
        u_std = torch.sqrt(u_var).to(torch.float32)
        print("Normalization stats computed.")

    def normalize_batch_Z(Zb_cpu):
        Zb = Zb_cpu.clone()
        pred_keep = Zb[:, :, pred_col_idx].clone()
        mask_keep = Zb[:, :, mask_col_idx].clone()

        Zb = (Zb - Z_mean) / Z_std

        Zb[:, :, mask_col_idx] = mask_keep
        return Zb

    def normalize_target(Tb_cpu):
        return (Tb_cpu - u_mean) / u_std

    def unnormalize_pred(pred_norm):
        return pred_norm * u_std.to(pred_norm.device) + u_mean.to(pred_norm.device)

    clip_r = 1.0
    n_epochs = epochs
    checkpoint_every = 10
    best_ckpt_path = os.path.join(cur_dir, "best_by_val.pth")

    def compute_full_dataset_loss(model, loader, dataset_len):
        model.eval()
        total_loss = 0.0
        n_seen = 0
        with torch.no_grad():
            for Zb_cpu, Tb_cpu in loader:
                Zb = normalize_batch_Z(Zb_cpu).to(device)
                Tb = normalize_target(Tb_cpu).to(device)
                out = model(Zb)
                l, _, _ = get_loss(out, Tb, Zb, n_nodes=n_nodes, d=d)
                B = Zb_cpu.shape[0]
                total_loss += l.item() * B
                n_seen += B
        if n_seen == 0:
            return float('nan')
        return total_loss / float(n_seen)

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()

        g_epoch = torch.Generator()
        g_epoch.manual_seed(split_seed + epoch)
        perm_epoch_local = torch.randperm(len(idx_train), generator=g_epoch)
        sampler = SubsetRandomSampler(perm_epoch_local.tolist())
        train_loader = DataLoader(ds_train, batch_size=train_batch, sampler=sampler,
                                  num_workers=num_workers, pin_memory=False)

        for Zb_cpu, Tb_cpu in train_loader:
            Zb = normalize_batch_Z(Zb_cpu).to(device)
            Tb = normalize_target(Tb_cpu).to(device)

            opt.zero_grad()
            out = model(Zb)
            loss, _, _ = get_loss(out, Tb, Zb, n_nodes=n_nodes, d=d)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_r)
            opt.step()

        train_full_loader = DataLoader(ds_train, batch_size=eval_batch, shuffle=False, num_workers=num_workers, pin_memory=False)
        val_loader = DataLoader(ds_val, batch_size=eval_batch, shuffle=False, num_workers=num_workers, pin_memory=False)

        train_full_loss = compute_full_dataset_loss(model, train_full_loader, len(ds_train))
        val_full_loss = compute_full_dataset_loss(model, val_loader, len(ds_val))

        history['train_loss'].append(train_full_loss)
        history['val_loss'].append(val_full_loss)

        is_best = False
        if val_full_loss < best_val:
            best_val = val_full_loss
            is_best = True
            ckpt = {
                'model_state': model.state_dict(),
                'Z_mean': Z_mean,
                'Z_std': Z_std,
                'u_mean': u_mean,
                'u_std': u_std,
                'epoch': epoch,
                'best_val': best_val,
                'history': history
            }
            torch.save(ckpt, best_ckpt_path)
            print(f"Epoch {epoch:04d} | train_full {train_full_loss:.3e} | val_full {val_full_loss:.3e}  <-- best")
        else:
            if epoch == 1 or epoch % 20 == 0:
                print(f"Epoch {epoch:04d} | train_full {train_full_loss:.3e} | val_full {val_full_loss:.3e}")

        if epoch % checkpoint_every == 0 or is_best:
            resume_ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': opt.state_dict(),
                'best_val': best_val,
                'history': history,
                'Z_mean': Z_mean,
                'Z_std': Z_std,
                'u_mean': u_mean,
                'u_std': u_std,
                'idx_train': idx_train,
                'idx_val': idx_val,
                'idx_test': idx_test,
                'torch_rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'random_state': random.getstate()
            }
            torch.save(resume_ckpt, resume_ckpt_path)
            print("Saved resume checkpoint at epoch", epoch)

    try:
        train_losses = np.array(history['train_loss'], dtype=float)
        val_losses = np.array(history['val_loss'], dtype=float)
        epochs = np.arange(1, len(train_losses) + 1)

        plt.figure(figsize=(7, 4.5), dpi=160)
        plt.plot(epochs, train_losses, label="training")
        plt.plot(epochs, val_losses, label="validation")
        plt.yscale("log")
        plt.xlim(1, len(epochs))
        plt.xlabel("Epoch", fontsize=PLOT_LABEL_FONTSIZE)
        plt.ylabel("Loss (log scale)", fontsize=PLOT_LABEL_FONTSIZE)
        plt.title("Train & Val Loss (log y) — per-epoch full-dataset evaluation", fontsize=PLOT_TITLE_FONTSIZE)
        plt.tick_params(axis="both", which="major", labelsize=PLOT_TICK_FONTSIZE)
        plt.legend(fontsize=PLOT_LEGEND_FONTSIZE)
        plt.tight_layout()
        out_loss = os.path.join(cur_dir, "loss_train_val_logy_full_dataset.png")
        plt.savefig(out_loss, dpi=160)
        plt.close()
        print("Saved loss plot ->", out_loss)
    except Exception as e:
        print("Could not save loss plot:", e)

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f"Loaded best-by-val checkpoint from epoch {ckpt.get('epoch', 'N/A')} (val={ckpt.get('best_val', 'N/A'):.3e})")
    else:
        print("No best checkpoint found; using final model weights for test evaluation.")

    model.eval()
    per_sample_abs_mse = np.zeros(len(idx_test), dtype=float)
    per_sample_rel_L2 = np.zeros(len(idx_test), dtype=float)
    per_sample_rel_L1 = np.zeros(len(idx_test), dtype=float)
    per_sample_rel_MSE = np.zeros(len(idx_test), dtype=float)
    per_sample_target_norm = np.zeros(len(idx_test), dtype=float)
    all_preds = {}
    all_targets = {}
    all_masks = {}
    all_forces = {}

    test_loader = DataLoader(ds_test, batch_size=eval_batch, shuffle=False,
                             num_workers=num_workers, pin_memory=False)

    sample_counter = 0
    with torch.no_grad():
        for Zb_cpu, Tb_cpu in test_loader:
            Bchunk = Zb_cpu.shape[0]
            Zb = normalize_batch_Z(Zb_cpu).to(device)
            Tb = Tb_cpu.to(device)

            out = model(Zb)
            pred_norm = out[:, :, d + 1]
            pred_phys = unnormalize_pred(pred_norm)

            mask_free_batch = Zb_cpu[:, :, mask_col_idx].bool()

            for i in range(Bchunk):
                m = mask_free_batch[i]
                t = Tb[i].cpu()
                p = pred_phys[i].cpu()

                fixed = ~m
                if fixed.any():
                    p[fixed] = 0.0

                if m.any():
                    mse = ((p[m] - t[m]) ** 2).mean().item()
                else:
                    mse = 0.0
                per_sample_abs_mse[sample_counter] = mse

                diff = (t - p).numpy()
                true_vec = t.numpy()
                l2_diff = np.linalg.norm(diff)
                l2_true = np.linalg.norm(true_vec)
                l1_diff = np.linalg.norm(diff, ord=1)
                l1_true = np.linalg.norm(true_vec, ord=1)
                sumsq_diff = float((diff ** 2).sum())
                sumsq_true = float((true_vec ** 2).sum())

                per_sample_target_norm[sample_counter] = l2_true
                if l2_true > 0:
                    per_sample_rel_L2[sample_counter] = l2_diff / l2_true
                else:
                    per_sample_rel_L2[sample_counter] = float('inf') if l2_diff > 0 else 0.0

                if l1_true > 0:
                    per_sample_rel_L1[sample_counter] = l1_diff / l1_true
                else:
                    per_sample_rel_L1[sample_counter] = float('inf') if l1_diff > 0 else 0.0

                if sumsq_true > 0:
                    per_sample_rel_MSE[sample_counter] = sumsq_diff / sumsq_true
                else:
                    per_sample_rel_MSE[sample_counter] = float('inf') if sumsq_diff > 0 else 0.0

                all_preds[sample_counter] = p.clone()
                all_targets[sample_counter] = t.clone()
                all_masks[sample_counter] = fixed.clone()

                forces = Zb_cpu[i, :, d].clone()
                all_forces[sample_counter] = forces

                sample_counter += 1
    idx_rel_l2_worst = int(np.nanargmax(per_sample_rel_L2))
    idx_high_norm = int(np.nanargmax(per_sample_target_norm))

    print("Selected test samples:")
    print("Worst (relative L2) idx:", idx_rel_l2_worst, "value:", per_sample_rel_L2[idx_rel_l2_worst])
    print("Highest target-norm idx:", idx_high_norm, "target ||u||2:", per_sample_target_norm[idx_high_norm])

    def make_and_save_plots_and_csv(sample_idx, tag):
        if sample_idx not in all_targets:
            raise KeyError(f"sample_idx {sample_idx} not found in all_targets")

        T_raw = all_targets[sample_idx]
        p_raw = all_preds[sample_idx]
        fixed_mask = all_masks[sample_idx]
        forces_vec = all_forces[sample_idx]

        if isinstance(T_raw, torch.Tensor):
            T_cpu = T_raw.detach().cpu().clone()
        else:
            T_cpu = torch.tensor(np.array(T_raw), dtype=torch.float32)

        if isinstance(p_raw, torch.Tensor):
            P_cpu = p_raw.detach().cpu().clone()
        else:
            P_cpu = torch.tensor(np.array(p_raw), dtype=torch.float32)

        if isinstance(fixed_mask, torch.Tensor):
            fixed_cpu = fixed_mask.detach().cpu().clone()
        else:
            fixed_cpu = torch.as_tensor(np.array(fixed_mask), dtype=torch.bool)

        if isinstance(forces_vec, torch.Tensor):
            forces_cpu = forces_vec.detach().cpu().clone()
        else:
            forces_cpu = torch.tensor(np.array(forces_vec), dtype=torch.float32)

        out_path = os.path.join(cur_dir, f"test_{tag}_idx{sample_idx}_true_pred.png")
        plot_true_pred_pair(out_path, nodes_np, edges, T_cpu, P_cpu,
                            title_prefix=f"Test {tag} (idx={sample_idx})", scale=1.0)

        err_out = os.path.join(cur_dir, f"test_{tag}_idx{sample_idx}_error_deformation.png")
        plot_error_deformation(
            err_out,
            nodes_np,
            edges,
            T_cpu,
            P_cpu,
            title=f"Test {tag} (idx={sample_idx}) — Error deformation (undeformed + (u_true - u_pred))",
            scale=1.0,
            dpi=150
        )

        prefix = f"test_{tag}_idx{sample_idx}"
        save_sample_csv_global_generic(cur_dir, prefix, nodes_np, T_cpu, P_cpu, fixed_cpu)

        forces_np = forces_cpu.numpy().reshape(-1)
        if forces_np.shape[0] != 2 * nodes_np.shape[0]:
            raise RuntimeError(f"forces length {forces_np.shape[0]} != 2*n_nodes ({2*nodes_np.shape[0]})")

        fx = forces_np[0::2]
        fy = forces_np[1::2]

        forces_csv = os.path.join(cur_dir, f"forces_test_{tag}_idx{sample_idx}.csv")
        save_forces_csv(forces_csv, nodes_np, fx, fy)

        forces_plot = os.path.join(cur_dir, f"forces_test_{tag}_idx{sample_idx}.png")
        plot_force_vectors(forces_plot, nodes_np, fx, fy,
                           title=f"Forces ({tag} idx={sample_idx})",
                           dpi=150, label_nodes=True, label_every=1)

        print(f"Completed saving artifacts for sample {sample_idx} (tag={tag})")

    make_and_save_plots_and_csv(idx_rel_l2_worst, "worst_relL2")
    if idx_high_norm != idx_rel_l2_worst:
        make_and_save_plots_and_csv(idx_high_norm, "highest_target_norm")
    else:
        print("Note: both criteria picked the same test sample; two sets will be identical.")

    mean_rel_L2 = np.nanmean(per_sample_rel_L2) * 100.0
    mean_rel_L1 = np.nanmean(per_sample_rel_L1) * 100.0
    mean_rel_MSE_percent = np.nanmean(per_sample_rel_MSE) * 100.0

    print("Aggregate test set relative errors (averaged over test samples):")
    print(f"Relative L2 (mean over samples): {mean_rel_L2:.4f} %")
    print(f"Relative L1 (mean over samples): {mean_rel_L1:.4f} %")
    print(f"Relative MSE (mean over samples): {mean_rel_MSE_percent:.6f} %")

    history_npz_path = os.path.join(cur_dir, "history_full.npz")
    np.savez_compressed(
        history_npz_path,
        train=np.array(history['train_loss'], dtype=float),
        val=np.array(history['val_loss'], dtype=float)
    )

    history_torch_path = os.path.join(cur_dir, "history_full.pth")
    torch.save(history, history_torch_path)

    print("Saved full history ->", history_npz_path, "and", history_torch_path)

    return {
        'model': model,
        'Z_mean': Z_mean, 'Z_std': Z_std, 'u_mean': u_mean, 'u_std': u_std,
        'history': history, 'best_val': best_val,
        'test_metrics': {
            'mean_rel_L2_percent': mean_rel_L2,
            'mean_rel_L1_percent': mean_rel_L1,
            'mean_rel_MSE_percent': mean_rel_MSE_percent
        }
    }


def main():
    global device, cur_dir
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/10x10"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--samples", type=int, default=42000)
    parser.add_argument("--smoke", action="store_true", help="One epoch with 20 samples; execution check only")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 10:
        parser.error("Use positive epochs and at least 10 samples")
    if args.smoke:
        args.epochs, args.samples = 1, 20
        args.workers = 0
    if args.workers is not None and args.workers < 0:
        parser.error("--workers must be nonnegative")
    output = args.output_dir.resolve()
    if args.resume:
        if not (output / "resume_checkpoint.pth").is_file():
            parser.error("No resume checkpoint in the output directory")
        config = json.loads((output / "run_config.json").read_text())
        if config["samples"] != args.samples:
            parser.error("Resume requires the same sample count")
    elif output.exists() and any(output.iterdir()):
        parser.error("Choose a new output directory or use --resume")
    output.mkdir(parents=True, exist_ok=True)
    cur_dir = str(output)
    requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    nodes, edges = tile_unit_cell([(0.,0.), (1.,0.), (0.,1.), (1.,1.)], 10, 10)
    coords = torch.tensor(nodes, dtype=torch.float32, device=device)
    config = dict(case="10x10", samples=args.samples, epochs=args.epochs,
                  layers=7, heads=1, seed=42, split_seed=12345, device=str(device),
                  dtype=str(coords.dtype), smoke=args.smoke,
                  python=__import__("platform").python_version(),
                  torch=torch.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__)
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    run_training(7, 1, len(nodes), len(edges), len(edges)+3, coords, edges,
                 n_samples=args.samples, epochs=args.epochs, nodes_np=nodes, workers=args.workers)
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
