import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GDLattice2D(nn.Module):
    def __init__(self, vocab_size, B, tok_idx, int_idx, d, n_ctx, n_layer, n_head,
                 gd_step=1.0, use_ff=False, dropout=0.0, ridge=1e-4,
                 per_layer_kernel=False, kernel_type="lattice",
                 normalize_kernel=False, kernel_norm_eps=1e-6,
                 learn_dof_gains=False, learn_node_coords=False,
                 nodes=None, edges=None, free=None, node_train_mask=None,
                 min_bar_length=1e-3, attn_fn="softmax", ridge_mode="relative",
                 n_phys=2, hard_relu=False, geo_newton_steps=4,
                 geo_disp_scale=0.05, geo_delta_scale=0.10,
                 kernel_norm_mode="causal_prefix"):
        super().__init__()
        self.n_phys = n_phys
        self.hard_relu = hard_relu
        self.geo_newton_steps = geo_newton_steps
        self.geo_disp_scale = geo_disp_scale
        self.geo_delta_scale = geo_delta_scale
        self.d, self.n_ctx, self.n_head = d, n_ctx, n_head
        self.gd_step, self.ridge = gd_step, ridge
        self.ridge_mode = ridge_mode
        self.per_layer_kernel = per_layer_kernel
        self.kernel_type = kernel_type
        self.attn_fn = attn_fn
        self.skip_n_reg = (kernel_type == "content_attn" and attn_fn == "softmax")
        self.normalize_kernel = normalize_kernel
        if kernel_norm_mode not in ("causal_prefix", "causal_row", "none"):
            raise ValueError("Unknown causal kernel normalization")
        self.kernel_norm_mode = kernel_norm_mode
        self.kernel_norm_eps = kernel_norm_eps
        self.learn_dof_gains = learn_dof_gains
        self.learn_node_coords = learn_node_coords
        self.min_bar_length = min_bar_length
        m = B.shape[1]

        self.wte = nn.Embedding(vocab_size, d)
        self.wte_ln = nn.LayerNorm(d, bias=False)
        self.ln_out = nn.LayerNorm(d, bias=False)

        if kernel_type == "lattice":
            shape = (n_layer, n_head, m) if per_layer_kernel else (n_head, m)
            self.Lambda = nn.Parameter(torch.ones(shape))
        elif kernel_type == "linear_gd":
            self.wpe = nn.Embedding(n_ctx + 1, d)
            self.wpe_ln = nn.LayerNorm(d, bias=False)
            self.W_q_diag = self.W_k_diag = nn.Parameter(torch.zeros(n_head, d))
        elif kernel_type == "dynamical_lattice":
            shape = (n_layer, n_head, m) if per_layer_kernel else (n_head, m)
            self.Lambda = nn.Parameter(torch.ones(shape))
            nt = len(tok_idx)
            self.dyn_alpha = nn.Parameter(torch.zeros(()))
            self.dyn_beta = nn.Parameter(torch.zeros(()))
            self.dyn_log_dt = nn.Parameter(torch.zeros(()))
            self.dyn_b = nn.Parameter(torch.randn(n_head, nt) * 0.05)
            self.dyn_c = nn.Parameter(torch.randn(n_head, nt) * 0.05)
        elif kernel_type == "content_attn":
            assert d % n_head == 0, "d must be divisible by n_head for content_attn"
            self.head_dim = d // n_head
            self.attn_q = nn.ModuleList(
                [nn.Linear(d, d, bias=False) for _ in range(n_layer)])
            self.attn_k = nn.ModuleList(
                [nn.Linear(d, d, bias=False) for _ in range(n_layer)])
        elif kernel_type == "lattice_content":
            shape = (n_layer, n_head, m) if per_layer_kernel else (n_head, m)
            self.Lambda = nn.Parameter(torch.ones(shape))
            assert d % n_head == 0, "d must be divisible by n_head for lattice_content"
            self.head_dim = d // n_head
            self.attn_q = nn.ModuleList(
                [nn.Linear(d, d, bias=False) for _ in range(n_layer)])
            self.attn_k = nn.ModuleList(
                [nn.Linear(d, d, bias=False) for _ in range(n_layer)])
        elif kernel_type == "lattice_prestretch_gate":
            shape = (n_layer, n_head, m) if per_layer_kernel else (n_head, m)
            self.Lambda = nn.Parameter(torch.ones(shape))
            self.delta0 = nn.Parameter(torch.zeros(shape))
            self.phi = nn.ModuleList(
                [nn.Linear(d, n_phys, bias=False) for _ in range(n_layer)])
            pshape = (n_layer, n_head, n_phys) if per_layer_kernel else (n_head, n_phys)
            self.phys_mix = nn.Parameter(torch.randn(pshape) * 0.1)
        elif kernel_type == "lattice_geo_prestress":
            if nodes is None or edges is None or free is None:
                raise ValueError("lattice_geo_prestress requires nodes, edges, free")
            shape = (n_layer, n_head, m) if per_layer_kernel else (n_head, m)
            self.Lambda = nn.Parameter(torch.ones(shape))
            self.delta0 = nn.Parameter(torch.zeros(shape))
            self.phi = nn.ModuleList(
                [nn.Linear(d, n_phys, bias=False) for _ in range(n_layer)])
            pshape = (n_layer, n_head, n_phys) if per_layer_kernel else (n_head, n_phys)
            self.phys_mix = nn.Parameter(torch.randn(pshape) * 0.1)
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")

        self.W_o = nn.ParameterList([
            nn.Parameter(torch.zeros(n_head * d, d)) for _ in range(n_layer)])

        self.use_ff = use_ff
        if use_ff:
            self.ff = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(d, bias=False),
                    nn.Linear(d, 4 * d, bias=False), nn.GELU(),
                    nn.Linear(4 * d, d, bias=False), nn.Dropout(dropout),
                ) for _ in range(n_layer)])
        self.drop_attn = nn.Dropout(dropout)
        self.drop_gd = nn.Dropout(dropout)

        self.register_buffer("B", torch.tensor(B, dtype=torch.float32))
        if learn_dof_gains:
            self.dof_log_gains = nn.Parameter(torch.zeros(B.shape[0]))
        if learn_node_coords:
            if nodes is None or edges is None or free is None:
                raise ValueError("nodes, edges, and free are required for "
                                 "learn_node_coords=True")
            self._init_learned_geometry(nodes, edges, free, node_train_mask)
        if kernel_type == "lattice_geo_prestress":
            self._init_geo_buffers(nodes, edges, free)
        self.register_buffer("tok", torch.tensor(tok_idx, dtype=torch.long))
        self.register_buffer("int_", torch.tensor(int_idx, dtype=torch.long))
        self.register_buffer("eye_ii",
                             torch.eye(len(int_idx), dtype=torch.float32))
        causal = torch.tril(torch.ones(n_ctx, n_ctx))
        self.register_buffer("causal", causal)
        n_reg = 1.0 / torch.arange(1, n_ctx + 1).float().unsqueeze(1)
        self.register_buffer("n_reg", n_reg)

        nn.init.normal_(self.wte.weight, std=0.02)
        if self.kernel_type == "linear_gd":
            nn.init.normal_(self.wpe.weight, std=0.02)
            nn.init.normal_(self.W_q_diag, std=0.02)
            nn.init.normal_(self.W_k_diag, std=0.02)
        for w in self.W_o:
            nn.init.normal_(w, std=0.02 / math.sqrt(2 * n_layer))

        self.collect_gate_stats = False
        self._gate_sum = self._gate_sumsq = None
        self._gate_count = 0.0

    def reset_gate_stats(self):
        m = self.B.shape[1]
        dev = self.B.device
        self._gate_sum = torch.zeros(self.n_head, m, device=dev)
        self._gate_sumsq = torch.zeros(self.n_head, m, device=dev)
        self._gate_count = 0.0

    def _init_learned_geometry(self, nodes, edges, free, node_train_mask):
        nodes_t = torch.tensor(nodes, dtype=torch.float32)
        edges_t = torch.tensor(edges, dtype=torch.long)
        free_map = {int(dof): i for i, dof in enumerate(free)}
        rows, cols, edge_ids, axes, signs = [], [], [], [], []
        for e, (i, j) in enumerate(edges):
            for dg, axis, sign in ((2 * i, 0, 1.0), (2 * i + 1, 1, 1.0),
                                   (2 * j, 0, -1.0), (2 * j + 1, 1, -1.0)):
                if dg in free_map:
                    rows.append(free_map[dg])
                    cols.append(e)
                    edge_ids.append(e)
                    axes.append(axis)
                    signs.append(sign)
        if node_train_mask is None:
            node_train_mask = torch.ones_like(nodes_t)
        else:
            node_train_mask = torch.tensor(node_train_mask, dtype=torch.float32)
        self.register_buffer("nodes0", nodes_t)
        self.register_buffer("edge_i", edges_t[:, 0])
        self.register_buffer("edge_j", edges_t[:, 1])
        self.register_buffer("B_rows", torch.tensor(rows, dtype=torch.long))
        self.register_buffer("B_cols", torch.tensor(cols, dtype=torch.long))
        self.register_buffer("B_edge_ids", torch.tensor(edge_ids, dtype=torch.long))
        self.register_buffer("B_axes", torch.tensor(axes, dtype=torch.long))
        self.register_buffer("B_signs", torch.tensor(signs, dtype=torch.float32))
        self.register_buffer("node_train_mask", node_train_mask)
        self.node_delta = nn.Parameter(torch.zeros_like(nodes_t))

    def _init_geo_buffers(self, nodes, edges, free):
        nodes_t = torch.tensor(nodes, dtype=torch.float32)
        edges_t = torch.tensor(edges, dtype=torch.long)
        if not hasattr(self, "nodes0"):
            self.register_buffer("nodes0", nodes_t)
        if not hasattr(self, "edge_i"):
            self.register_buffer("edge_i", edges_t[:, 0])
            self.register_buffer("edge_j", edges_t[:, 1])
        free_nodes = torch.tensor([int(d) // 2 for d in free], dtype=torch.long)
        free_axes = torch.tensor([int(d) % 2 for d in free], dtype=torch.long)
        self.register_buffer("geo_free_nodes", free_nodes)
        self.register_buffer("geo_free_axes", free_axes)

        free_map = {int(dof): r for r, dof in enumerate(free)}
        rows = []
        axes = []
        signs = []
        for i, j in edges:
            local_rows = []
            local_axes = []
            local_signs = []
            for node, sign in ((int(i), -1.0), (int(j), 1.0)):
                for axis in (0, 1):
                    local_rows.append(free_map.get(2 * node + axis, -1))
                    local_axes.append(axis)
                    local_signs.append(sign)
            rows.append(local_rows)
            axes.append(local_axes)
            signs.append(local_signs)
        self.register_buffer("geo_bar_rows", torch.tensor(rows, dtype=torch.long))
        self.register_buffer("geo_bar_axes", torch.tensor(axes, dtype=torch.long))
        self.register_buffer("geo_bar_signs", torch.tensor(signs, dtype=torch.float32))

        nf = len(free)
        rows_t = torch.tensor(rows, dtype=torch.long)
        axes_t = torch.tensor(axes, dtype=torch.long)
        signs_t = torch.tensor(signs, dtype=torch.float32)
        m = rows_t.shape[0]
        row_a = rows_t.unsqueeze(2).expand(m, 4, 4).reshape(-1)
        col_b = rows_t.unsqueeze(1).expand(m, 4, 4).reshape(-1)
        sgn = (signs_t.unsqueeze(2) * signs_t.unsqueeze(1)).reshape(-1)
        ax_a = axes_t.unsqueeze(2).expand(m, 4, 4).reshape(-1)
        ax_b = axes_t.unsqueeze(1).expand(m, 4, 4).reshape(-1)
        e_grid = torch.arange(m).view(m, 1, 1).expand(m, 4, 4).reshape(-1)
        valid = (row_a >= 0) & (col_b >= 0)
        self.register_buffer("geo_pair_e", e_grid[valid])
        self.register_buffer("geo_pair_flat", row_a[valid] * nf + col_b[valid])
        self.register_buffer("geo_pair_sign", sgn[valid])
        self.register_buffer("geo_pair_axa", ax_a[valid])
        self.register_buffer("geo_pair_axb", ax_b[valid])
        self.geo_nf = int(nf)

        ref = nodes_t[edges_t[:, 1]] - nodes_t[edges_t[:, 0]]
        L0 = ref.pow(2).sum(dim=-1).sqrt().clamp_min(self.min_bar_length)
        self.register_buffer("geo_L0", L0)

    def _current_geo_nodes(self):
        if self.learn_node_coords:
            return self.nodes0 + self.node_train_mask * self.node_delta
        return self.nodes0

    def _geometry_B(self):
        nodes = self.nodes0 + self.node_train_mask * self.node_delta
        vec = nodes[self.edge_j] - nodes[self.edge_i]
        L2 = vec.pow(2).sum(dim=-1).clamp_min(self.min_bar_length ** 2)
        vals = (self.B_signs * vec[self.B_edge_ids, self.B_axes] /
                L2[self.B_edge_ids])
        B = torch.zeros_like(self.B)
        B.index_put_((self.B_rows, self.B_cols), vals)
        return B

    def _effective_B(self):
        B = self._geometry_B() if self.learn_node_coords else self.B
        if self.learn_dof_gains:
            gains = torch.exp(self.dof_log_gains).clamp(1e-3, 1e3)
            B = B * gains.unsqueeze(1)
        return B

    def E_wte(self, f):
        g = torch.softmax(f @ self.wte.weight.t(), dim=-1)
        return g @ self.wte.weight

    def _normalize_causal_kernel(self, A):
        if not self.normalize_kernel or self.kernel_norm_mode == "none":
            return A
        S = A.shape[-2]
        A = A * self.causal[:S, :A.shape[-1]]
        row_energy = A.square().sum(dim=-1, keepdim=True)
        count = torch.arange(1, S + 1, device=A.device, dtype=A.dtype).view(S, 1)
        if self.kernel_norm_mode == "causal_prefix":
            energy = row_energy.cumsum(dim=-2)
            count = count * (count + 1) / 2
        else:
            energy = row_energy
        rms = (energy / count).clamp_min(self.kernel_norm_eps ** 2).sqrt()
        return A / rms

    def lattice_kernel(self, lam):
        B = self._effective_B()
        Bw = B.unsqueeze(0) * (lam ** 2).unsqueeze(1)
        K = Bw @ B.t()
        K_tt = K[:, self.tok][:, :, self.tok]
        K_ti = K[:, self.tok][:, :, self.int_]
        K_it = K[:, self.int_][:, :, self.tok]
        K_ii = self._add_ridge(K[:, self.int_][:, :, self.int_])
        X = torch.linalg.solve(K_ii, K_it)
        K_eff = K_tt - K_ti @ X
        A = K_eff[:, 1:, :-1]
        A = A * self.causal
        return self._normalize_causal_kernel(A)

    def linear_gd_kernel(self, device):
        p = self.wpe(torch.arange(self.n_ctx + 1, device=device))
        p = self.wpe_ln(p)
        x_i = p[:-1].unsqueeze(0).expand(self.n_head, -1, -1)
        x_j = p[1:].unsqueeze(0).expand(self.n_head, -1, -1)
        K = x_i * self.W_k_diag.unsqueeze(1)
        Q = x_j * self.W_q_diag.unsqueeze(1)
        A = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d)
        return A * self.causal

    def _condensed_stiffness(self, lam):
        B = self._effective_B()
        Bw = B.unsqueeze(0) * (lam ** 2).unsqueeze(1)
        K = Bw @ B.t()
        K_tt = K[:, self.tok][:, :, self.tok]
        K_ti = K[:, self.tok][:, :, self.int_]
        K_it = K[:, self.int_][:, :, self.tok]
        K_ii = self._add_ridge(K[:, self.int_][:, :, self.int_])
        return K_tt - K_ti @ torch.linalg.solve(K_ii, K_it)

    def _add_ridge(self, K_ii):
        if self.ridge_mode == "absolute":
            return K_ii + self.ridge * self.eye_ii
        diag = K_ii.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True
                                                    ).unsqueeze(-1).clamp_min(1e-6)
        return K_ii + self.ridge * diag * self.eye_ii

    def prestretch_gate_kernel(self, V, layer_idx):
        Bsz, S, _ = V.shape
        H = self.n_head
        lam = self.Lambda[layer_idx] if self.per_layer_kernel else self.Lambda
        d0 = self.delta0[layer_idx] if self.per_layer_kernel else self.delta0
        mix = self.phys_mix[layer_idx] if self.per_layer_kernel else self.phys_mix

        B_tok = self.B[self.tok]
        B_int = self.B[self.int_]
        nt = B_tok.shape[0]

        disp = self.phi[layer_idx](V)
        contrib = torch.einsum("bjp,je->bpje", disp, B_tok[:S])
        s_all = contrib.cumsum(dim=2)
        z = (d0.view(1, H, 1, -1)
             + torch.einsum("hp,bpie->bhie", mix, s_all))
        if self.hard_relu:
            soft = torch.sigmoid(z)
            gate = (z > 0).to(V.dtype) - soft.detach() + soft
        else:
            gate = torch.sigmoid(z)
        if self.collect_gate_stats:
            g = gate.detach()
            self._gate_sum += g.sum(dim=(0, 2))
            self._gate_sumsq += (g * g).sum(dim=(0, 2))
            self._gate_count += g.shape[0] * g.shape[2]
        k = (lam ** 2).view(1, H, 1, -1) * gate

        Bw_t = torch.einsum("te,bhse->bhste", B_tok, k)
        Bw_i = torch.einsum("ae,bhse->bhsae", B_int, k)
        K_tt = Bw_t @ B_tok.t()
        K_ti = Bw_t @ B_int.t()
        K_ii = self._add_ridge(Bw_i @ B_int.t())
        K_eff = K_tt - K_ti @ torch.linalg.solve(K_ii, K_ti.transpose(-1, -2))
        idx = torch.arange(S, device=V.device)
        A = K_eff[:, :, idx, idx + 1, :S]
        A = A * self.causal[:S, :S]
        A = self._normalize_causal_kernel(A)
        return A

    def _assemble_geo_force_tangent(self, u_free, k_bar, prestretch):
        dtype, device = u_free.dtype, u_free.device
        nodes = self._current_geo_nodes().to(device=device, dtype=dtype)
        pos = nodes.clone()
        pos[self.geo_free_nodes, self.geo_free_axes] = (
            pos[self.geo_free_nodes, self.geo_free_axes] + u_free)

        xi = pos[self.edge_i]
        xj = pos[self.edge_j]
        r = xj - xi
        L = r.pow(2).sum(dim=-1).sqrt().clamp_min(self.min_bar_length)
        n = r / L.unsqueeze(-1)
        L0 = self.geo_L0.to(device=device, dtype=dtype)
        L_rest = (L0 - prestretch).clamp_min(self.min_bar_length)
        t = k_bar * (L - L_rest)

        pair_force = t.unsqueeze(-1) * n
        f_nodes = torch.zeros_like(pos)
        f_nodes.index_add_(0, self.edge_i, -pair_force)
        f_nodes.index_add_(0, self.edge_j, pair_force)
        f_int = f_nodes[self.geo_free_nodes, self.geo_free_axes]

        I2 = torch.eye(2, device=device, dtype=dtype).unsqueeze(0)
        nn = n.unsqueeze(-1) * n.unsqueeze(-2)
        Kloc = (k_bar.view(-1, 1, 1) * nn
                + (t / L).view(-1, 1, 1) * (I2 - nn))

        nf = self.geo_nf
        val = (self.geo_pair_sign.to(dtype)
               * Kloc[self.geo_pair_e, self.geo_pair_axa, self.geo_pair_axb])
        K = torch.zeros(nf * nf, device=device, dtype=dtype)
        K.index_add_(0, self.geo_pair_flat, val)
        K = K.view(nf, nf)
        return f_int, K

    def _geo_case_kernel_row(self, tok_u, k_bar, prestretch, row, S):
        nf = self.B.shape[0]
        dtype, device = k_bar.dtype, k_bar.device
        u_free = torch.zeros(nf, device=device, dtype=dtype)
        u_free = u_free.index_copy(0, self.tok, tok_u)

        for _ in range(max(0, int(self.geo_newton_steps))):
            f_int, K = self._assemble_geo_force_tangent(u_free, k_bar, prestretch)
            K_ii = self._add_ridge(K[self.int_][:, self.int_])
            rhs = -f_int[self.int_]
            du = torch.linalg.solve(K_ii, rhs.unsqueeze(-1)).squeeze(-1)
            u_free = u_free.index_copy(0, self.int_, u_free[self.int_] + du)

        _f_int, K = self._assemble_geo_force_tangent(u_free, k_bar, prestretch)
        K_tt = K[self.tok][:, self.tok]
        K_ti = K[self.tok][:, self.int_]
        K_it = K[self.int_][:, self.tok]
        K_ii = self._add_ridge(K[self.int_][:, self.int_])
        K_eff = K_tt - K_ti @ torch.linalg.solve(K_ii, K_it)
        return K_eff[row + 1, :S]

    def _assemble_geo_batched(self, u_free, k_bar, prestretch):
        dtype, device = u_free.dtype, u_free.device
        N, nf = u_free.shape[0], self.geo_nf
        n_nodes = self.nodes0.shape[0]
        nodes = self._current_geo_nodes().to(device=device, dtype=dtype)
        disp = torch.zeros(N, n_nodes, 2, device=device, dtype=dtype)
        disp[:, self.geo_free_nodes, self.geo_free_axes] = u_free
        pos = nodes.unsqueeze(0) + disp

        r = pos[:, self.edge_j] - pos[:, self.edge_i]
        L = r.pow(2).sum(-1).sqrt().clamp_min(self.min_bar_length)
        n = r / L.unsqueeze(-1)
        L0 = self.geo_L0.to(device=device, dtype=dtype)
        L_rest = (L0.unsqueeze(0) - prestretch).clamp_min(self.min_bar_length)
        t = k_bar * (L - L_rest)

        pair = t.unsqueeze(-1) * n
        f_nodes = torch.zeros(N, n_nodes, 2, device=device, dtype=dtype)
        f_nodes.index_add_(1, self.edge_i, -pair)
        f_nodes.index_add_(1, self.edge_j, pair)
        f_int = f_nodes[:, self.geo_free_nodes, self.geo_free_axes]

        I2 = torch.eye(2, device=device, dtype=dtype)
        nn = n.unsqueeze(-1) * n.unsqueeze(-2)
        Kloc = (k_bar.view(N, -1, 1, 1) * nn
                + (t / L).view(N, -1, 1, 1) * (I2 - nn))
        val = (self.geo_pair_sign.to(dtype).view(1, -1)
               * Kloc[:, self.geo_pair_e, self.geo_pair_axa, self.geo_pair_axb])
        K = torch.zeros(N, nf * nf, device=device, dtype=dtype)
        K.index_add_(1, self.geo_pair_flat, val)
        return f_int, K.view(N, nf, nf)

    def _geo_batched_rows(self, u_tok, k_bar, prestretch, rows, S):
        device, dtype = k_bar.device, k_bar.dtype
        N, nf = u_tok.shape[0], self.B.shape[0]
        u_free = torch.zeros(N, nf, device=device, dtype=dtype)
        u_free = u_free.index_add(1, self.tok[:S], u_tok)

        for _ in range(max(0, int(self.geo_newton_steps))):
            f_int, K = self._assemble_geo_batched(u_free, k_bar, prestretch)
            K_ii = self._add_ridge(K[:, self.int_][:, :, self.int_])
            rhs = -f_int[:, self.int_]
            du = torch.linalg.solve(K_ii, rhs.unsqueeze(-1)).squeeze(-1)
            u_free = u_free.index_add(1, self.int_, du)

        _f, K = self._assemble_geo_batched(u_free, k_bar, prestretch)
        K_tt = K[:, self.tok][:, :, self.tok]
        K_ti = K[:, self.tok][:, :, self.int_]
        K_it = K[:, self.int_][:, :, self.tok]
        K_ii = self._add_ridge(K[:, self.int_][:, :, self.int_])
        K_eff = K_tt - K_ti @ torch.linalg.solve(K_ii, K_it)
        bidx = torch.arange(N, device=device)
        return K_eff[bidx, rows + 1, :S]

    def geo_prestress_kernel(self, V, layer_idx):
        Bsz, S, _ = V.shape
        H = self.n_head
        lam = self.Lambda[layer_idx] if self.per_layer_kernel else self.Lambda
        d0 = self.delta0[layer_idx] if self.per_layer_kernel else self.delta0
        mix = self.phys_mix[layer_idx] if self.per_layer_kernel else self.phys_mix

        disp = self.phi[layer_idx](V)
        u_heads = torch.einsum("hp,bsp->bhs", mix, disp)
        u_heads = self.geo_disp_scale * torch.tanh(u_heads)
        k_all = lam ** 2
        L0 = self.geo_L0.to(device=V.device, dtype=V.dtype)
        prestretch = self.geo_delta_scale * L0.view(1, -1) * torch.tanh(d0)
        m = k_all.shape[1]

        tri = torch.tril(torch.ones(S, S, device=V.device, dtype=V.dtype))
        u_tok = (u_heads.unsqueeze(2) * tri.view(1, 1, S, S)).reshape(Bsz * H * S, S)
        N = Bsz * H * S
        k_batch = k_all.view(1, H, 1, m).expand(Bsz, H, S, m).reshape(N, m)
        pre_batch = prestretch.view(1, H, 1, m).expand(Bsz, H, S, m).reshape(N, m)
        rows = torch.arange(S, device=V.device).view(1, 1, S).expand(Bsz, H, S).reshape(N)

        A = self._geo_batched_rows(u_tok, k_batch, pre_batch, rows, S)
        A = A.view(Bsz, H, S, S)
        A = A * self.causal[:S, :S]
        return self._normalize_causal_kernel(A)

    def _toeplitz(self, h):
        H, S = h.shape
        idx = torch.arange(S, device=h.device)
        lag = idx[:, None] - idx[None, :]
        A = h[:, lag.clamp(min=0)]
        return A * (lag >= 0)

    def dynamical_kernel(self, lam):
        H = lam.shape[0]
        K = self._condensed_stiffness(lam)
        nt = K.shape[-1]
        a = F.softplus(self.dyn_alpha)
        bray = F.softplus(self.dyn_beta)
        dt = F.softplus(self.dyn_log_dt) + 1e-3
        I = torch.eye(nt, device=K.device, dtype=K.dtype).expand(H, nt, nt)
        D = a * I + bray * K
        Z = torch.zeros_like(K)
        Ac = torch.cat([torch.cat([Z, I], dim=-1),
                        torch.cat([-K, -D], dim=-1)], dim=-2)
        Ad = torch.linalg.matrix_exp(Ac * dt)
        bvec = torch.cat([torch.zeros_like(self.dyn_b), self.dyn_b], dim=-1)
        cvec = torch.cat([self.dyn_c, torch.zeros_like(self.dyn_c)], dim=-1)
        s = bvec.unsqueeze(-1)
        hs = []
        for _ in range(self.n_ctx):
            hs.append((cvec.unsqueeze(1) @ s).reshape(H))
            s = Ad @ s
        h = torch.stack(hs, dim=1)
        return self._normalize_causal_kernel(self._toeplitz(h))

    def kernel(self, layer_idx, device):
        if self.kernel_type == "linear_gd":
            return self.linear_gd_kernel(device)
        lam = self.Lambda[layer_idx] if self.per_layer_kernel else self.Lambda
        if self.kernel_type == "dynamical_lattice":
            return self.dynamical_kernel(lam)
        return self.lattice_kernel(lam)

    def content_attn_kernel(self, e, layer_idx):
        B, S, _ = e.shape
        H, hd = self.n_head, self.head_dim
        q = self.attn_q[layer_idx](e).view(B, S, H, hd).transpose(1, 2)
        k = self.attn_k[layer_idx](e).view(B, S, H, hd).transpose(1, 2)
        A = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        mask = (self.causal[:S, :S] == 0)
        if self.attn_fn == "softmax":
            A = A.masked_fill(mask, float("-inf"))
            A = torch.softmax(A, dim=-1)
        else:
            A = A.masked_fill(mask, 0.0)
        return A

    def _content_gate(self, e, layer_idx):
        Bsz, S, _ = e.shape
        H, hd = self.n_head, self.head_dim
        q = self.attn_q[layer_idx](e).view(Bsz, S, H, hd).transpose(1, 2)
        k = self.attn_k[layer_idx](e).view(Bsz, S, H, hd).transpose(1, 2)
        score = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        return F.softplus(score)

    def forward(self, ids):
        B, S = ids.shape
        if S < 1 or S > self.n_ctx:
            raise ValueError(f"Expected 1..{self.n_ctx} input tokens; got {S}")
        e = self.wte_ln(self.wte(ids))
        f = torch.zeros(B, S + 1, self.d, device=ids.device, dtype=e.dtype)
        A_shared = None
        if self.kernel_type in ("lattice", "dynamical_lattice", "lattice_content") \
                and not self.per_layer_kernel:
            A_shared = self.kernel(0, ids.device)[:, :S, :S]
            A_shared = A_shared.unsqueeze(0).expand(B, -1, -1, -1)
            if self.kernel_type != "lattice_content":
                A_shared = self.drop_attn(A_shared)
        for l in range(len(self.W_o)):
            E = self.E_wte(f[:, :-1, :])
            V = e - E
            if self.kernel_type == "content_attn":
                A = self.drop_attn(self.content_attn_kernel(e, l))
            elif self.kernel_type == "lattice_prestretch_gate":
                A = self.drop_attn(self.prestretch_gate_kernel(V, l))
            elif self.kernel_type == "lattice_geo_prestress":
                A = self.drop_attn(self.geo_prestress_kernel(V, l))
            elif self.kernel_type == "lattice_content":
                base = A_shared if A_shared is not None else \
                    self.kernel(l, ids.device)[:, :S, :S].unsqueeze(0).expand(B, -1, -1, -1)
                A = self.drop_attn(base * self._content_gate(e, l))
            elif A_shared is None:
                A = self.kernel(l, ids.device)[:, :S, :S]
                A = A.unsqueeze(0).expand(B, -1, -1, -1)
                A = self.drop_attn(A)
            else:
                A = A_shared
            delta = torch.einsum("BHST,BTd->BHSd", A, V)
            if not self.skip_n_reg:
                delta = delta * self.n_reg[:S].view(1, 1, S, 1)
            delta = delta.permute(0, 2, 1, 3).reshape(B, S, -1)
            delta = self.drop_gd(delta @ self.W_o[l])
            f = f + self.gd_step * F.pad(delta, (0, 0, 1, 0))
            if self.use_ff:
                f = f + self.ff[l](f)
        return self.ln_out(f[:, 1:, :]) @ self.wte_ln(self.wte.weight).t()
