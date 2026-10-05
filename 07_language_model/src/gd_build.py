from collections import namedtuple

import torch

from gd_utils import tokenizer_vocab_size
from gd_lattice_geom import build_lattice, build_grid_lattice, build_B
from gd_model import GDLattice2D

Geometry = namedtuple("Geometry",
                      "nodes edges free out_dofs B tok_idx int_idx")


def internal_node_train_mask(nodes, out_dofs):
    train = torch.ones((len(nodes), 2), dtype=torch.float32)
    y_min = float(nodes[:, 1].min())
    y_max = float(nodes[:, 1].max())
    boundary = (nodes[:, 1] == y_min) | (nodes[:, 1] == y_max)
    train[torch.as_tensor(boundary, dtype=torch.bool)] = 0.0
    for dof in out_dofs:
        train[int(dof) // 2] = 0.0
    return train.numpy()


def build_geometry(args):
    if args.lattice_connectivity == "grid_interleaved":
        nodes, edges, free, out_dofs = build_grid_lattice(
            grid_width=args.grid_width, n_ports=args.n_ctx + 1,
            long_range=args.lattice_long_range)
        assert len(out_dofs) == args.n_ctx + 1, "grid must yield n_ctx+1 ports."
    else:
        nodes, edges, free, out_dofs = build_lattice(
            nx=args.nx, ny=args.ny, connectivity=args.lattice_connectivity,
            long_range=args.lattice_long_range)
        assert len(out_dofs) >= args.n_ctx + 1, "Set nx == n_ctx."
    B = build_B(nodes, edges, free)
    free_map = {dof: i for i, dof in enumerate(free)}
    ofi = [free_map[dof] for dof in out_dofs]
    tok_idx = ofi[:args.n_ctx + 1]
    tok_set = set(tok_idx)
    int_idx = [i for i in range(len(free)) if i not in tok_set]
    return Geometry(nodes, edges, free, out_dofs, B, tok_idx, int_idx)


def build_model(args, data, device, verbose=False):
    vocab_size = tokenizer_vocab_size(data.tokenizer, args.vocab_limit)
    args.vocab_limit = vocab_size
    g = build_geometry(args)
    model = GDLattice2D(
        vocab_size, g.B, g.tok_idx, g.int_idx, args.d, args.n_ctx,
        args.n_layer, args.n_head, gd_step=args.gd_step, use_ff=args.use_ff,
        dropout=args.dropout, ridge=args.ridge,
        per_layer_kernel=args.per_layer_kernel, kernel_type=args.kernel_type,
        normalize_kernel=args.normalize_kernel,
        kernel_norm_mode=args.kernel_norm_mode,
        kernel_norm_eps=args.kernel_norm_eps,
        learn_dof_gains=args.learn_dof_gains,
        learn_node_coords=args.learn_node_coords,
        nodes=g.nodes, edges=g.edges, free=g.free,
        node_train_mask=internal_node_train_mask(g.nodes, g.out_dofs),
        min_bar_length=args.min_bar_length, attn_fn=args.attn_fn,
        ridge_mode=args.ridge_mode, n_phys=args.n_phys,
        hard_relu=args.hard_relu, geo_newton_steps=args.geo_newton_steps,
        geo_disp_scale=args.geo_disp_scale,
        geo_delta_scale=args.geo_delta_scale).to(device)

    if verbose:
        print(f"Truss: {len(g.nodes)} nodes, {len(g.edges)} bars, "
              f"{len(g.free)} free DOFs -> {len(g.tok_idx)} token/prediction + "
              f"{len(g.int_idx)} internal  connectivity={args.lattice_connectivity}",
              flush=True)
        print(f"Kernel: {args.kernel_type}"
              f"{' normalized' if args.normalize_kernel else ''}"
              f"{' per-layer' if args.per_layer_kernel else ' shared'}"
              f"{' + dof-gains' if args.learn_dof_gains else ''}"
              f"{' + learned-node-coords' if args.learn_node_coords else ''}",
              flush=True)
        print(f"Trainable parameters: "
              f"{sum(p.numel() for p in model.parameters()):,}", flush=True)
    return model
