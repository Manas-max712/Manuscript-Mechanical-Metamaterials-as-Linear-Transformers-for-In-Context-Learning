import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_BLUE = "#1a73e8"

_RED = "#d62728"

_BLACK = "#111111"

_INT = "#2b6a7a"

_FUT = "#333333"

_LOCAL_MAX = 1.6

_MARKER_DRIVEN = 110

_MARKER_INTERNAL = 82

_MARKER_FUTURE = 104

_MARKER_SUPPORT = 112

_MARKER_READOUT = 280

_LEGEND_FONTSIZE = 16

_LEGEND_MARKERSIZE = 13

_CONFIG_ARROW_LEN = 0.48

_ZOOM_HALF_WIDTH = 3.9

_ZOOM_HALF_HEIGHT = 3.25

def _draw_bars(ax, nodes, edges, keep, cvals, norm, cmap, pos, wvals=None,
               wnorm=None):
    for e, (a, b) in enumerate(edges):
        L = float(np.linalg.norm(nodes[a] - nodes[b]))
        local = L <= _LOCAL_MAX
        if not keep[e]:
            ax.plot(pos[[a, b], 0], pos[[a, b], 1], color="#e6e6e6", lw=0.4,
                    ls=":", alpha=0.4, zorder=1)
            continue
        c = float(np.clip(float(norm(float(cvals[e]))), 0.0, 1.0))
        w = c
        if wvals is not None and wnorm is not None:
            w = float(np.clip(float(wnorm(float(abs(wvals[e])))), 0.0, 1.0))
        if local:
            ax.plot(pos[[a, b], 0], pos[[a, b], 1], color=cmap(c),
                    lw=1.2 + 1.8 * w, alpha=0.95, zorder=3)
        else:
            ax.plot(pos[[a, b], 0], pos[[a, b], 1], color=cmap(c),
                    lw=0.6 + 0.8 * w, alpha=0.30, zorder=2)

def _set_zoom_window(ax, nodes, center_node):
    cx, cy = nodes[center_node]
    ax.set_xlim(cx - _ZOOM_HALF_WIDTH, cx + _ZOOM_HALF_WIDTH)
    ax.set_ylim(cy - _ZOOM_HALF_HEIGHT, cy + _ZOOM_HALF_HEIGHT)

def _set_full_window(ax, nodes):
    pad_x = 0.8
    pad_y = 0.65
    ax.set_xlim(nodes[:, 0].min() - pad_x, nodes[:, 0].max() + pad_x)
    ax.set_ylim(nodes[:, 1].min() - pad_y, nodes[:, 1].max() + pad_y)

def _roles(nodes, token_ids, S):
    port_set = set(token_ids)
    arrived = np.array(sorted({token_ids[i] for i in range(S)}))
    future = (np.array(sorted({token_ids[i] for i in range(S, len(token_ids))}))
              if S < len(token_ids) else np.array([], dtype=int))
    ymin = nodes[:, 1].min()
    support = np.array([i for i in range(len(nodes)) if nodes[i, 1] == ymin])
    interior = np.array([i for i in range(len(nodes))
                         if i not in port_set and nodes[i, 1] != ymin])
    return arrived, future, support, interior

def _legend(ax, interior_marker, interior_label):
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_BLUE,
               markeredgecolor="white", label="driven ports",
               markersize=_LEGEND_MARKERSIZE),
        Line2D([0], [0], marker="o", color=_RED, markerfacecolor="white",
               markeredgewidth=2.2, label="readout port",
               markersize=_LEGEND_MARKERSIZE + 1),
        Line2D([0], [0], marker=interior_marker, color="w", markerfacecolor=_INT,
               label=interior_label, markersize=_LEGEND_MARKERSIZE),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=_BLACK,
               label="support", markersize=_LEGEND_MARKERSIZE),
    ]
    ax.legend(handles=handles, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.035), frameon=False,
              fontsize=_LEGEND_FONTSIZE, handletextpad=0.8,
              columnspacing=1.8, borderaxespad=0.0)

def _annotate_tokens(ax, pos, token_ids, S, token_texts, next_label, n_label=4,
                     y_off=0.95):
    if token_texts is not None:
        for i in range(max(0, S - n_label), S):
            if i >= len(token_texts):
                break
            txt = token_texts[i].strip()
            if not txt:
                continue
            x, y = pos[token_ids[i]]
            ax.annotate(txt, (x, y), xytext=(x, y + y_off), ha="center",
                        va="bottom", fontsize=9, color="#0b3d91",
                        arrowprops=dict(arrowstyle="-", color=_RED, lw=0.7,
                                        alpha=0.75), zorder=9)
    if next_label and S < len(token_ids):
        x, y = pos[token_ids[S]]
        ax.annotate(next_label.strip() or "·", (x, y), xytext=(x, y + y_off),
                    ha="center", va="bottom", fontsize=11, color=_RED,
                    fontweight="bold", zorder=9)

def _plot_configure(nodes, edges, keep, token_ids, S, disp_xy, cvals, norm, cmap,
                    path, arrow_len=_CONFIG_ARROW_LEN, zoom=True, wvals=None,
                    wnorm=None, token_texts=None, next_label=None, title=None):
    arrived, future, support, interior = _roles(nodes, token_ids, S)
    mag = np.linalg.norm(disp_xy, axis=1, keepdims=True)
    unit = np.divide(disp_xy, mag, out=np.zeros_like(disp_xy), where=mag > 1e-9)
    defo = nodes + arrow_len * unit
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_aspect("equal")
    ax.axis("off")
    _draw_bars(ax, nodes, edges, keep, cvals, norm, cmap, nodes,
               wvals=wvals, wnorm=wnorm)
    ax.scatter(nodes[interior, 0], nodes[interior, 1], s=_MARKER_INTERNAL, c=_INT,
               marker="o", zorder=4)
    ax.scatter(nodes[support, 0], nodes[support, 1], s=_MARKER_SUPPORT, marker="s",
               c=_BLACK, zorder=5)
    ax.scatter(nodes[arrived, 0], nodes[arrived, 1], s=_MARKER_DRIVEN, c=_BLUE,
               edgecolors="white", linewidths=1.0, zorder=6)
    if len(future):
        ax.scatter(nodes[future, 0], nodes[future, 1], s=_MARKER_FUTURE,
                   facecolors="white", edgecolors=_FUT, linewidths=1.8, zorder=6)
    if S < len(token_ids):
        pn = token_ids[S]
        ax.scatter([nodes[pn, 0]], [nodes[pn, 1]], s=_MARKER_READOUT,
                   facecolors="none", edgecolors=_RED, linewidths=3.0, zorder=7)
    for n in arrived:
        dx, dy = defo[n] - nodes[n]
        if abs(dx) + abs(dy) > 1e-9:
            ax.arrow(nodes[n, 0], nodes[n, 1], dx, dy, length_includes_head=True,
                     width=0.020, head_width=0.20, head_length=0.22,
                     color=_BLUE, alpha=0.85, zorder=7)
    _annotate_tokens(ax, nodes, token_ids, S, token_texts, next_label)
    if zoom:
        center = token_ids[S] if S < len(token_ids) else token_ids[S - 1]
        _set_zoom_window(ax, nodes, center)
    else:
        _set_full_window(ax, nodes)
    if title:
        ax.set_title(title, fontsize=11)
    _legend(ax, "o", "fixed internal (clamped)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def _plot_geo_configure(nodes, edges, keep, token_ids, S, disp_xy, cvals, norm,
                        cmap, path, dscale, zoom=True, wvals=None, wnorm=None,
                        token_texts=None, next_label=None, title=None):
    arrived, future, support, interior = _roles(nodes, token_ids, S)
    defo = nodes + dscale * disp_xy
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.scatter(nodes[:, 0], nodes[:, 1], s=5, c="#d9d9d9", alpha=0.35, zorder=0)
    _draw_bars(ax, nodes, edges, keep, cvals, norm, cmap, defo,
               wvals=wvals, wnorm=wnorm)
    ax.scatter(defo[interior, 0], defo[interior, 1], s=_MARKER_INTERNAL, c=_INT,
               marker="s", zorder=4)
    ax.scatter(defo[support, 0], defo[support, 1], s=_MARKER_SUPPORT, marker="s",
               c=_BLACK, zorder=5)
    ax.scatter(defo[arrived, 0], defo[arrived, 1], s=_MARKER_DRIVEN, c=_BLUE,
               edgecolors="white", linewidths=1.0, zorder=6)
    if len(future):
        ax.scatter(defo[future, 0], defo[future, 1], s=_MARKER_FUTURE,
                   facecolors="white", edgecolors=_FUT, linewidths=1.8, zorder=6)
    if S < len(token_ids):
        pn = token_ids[S]
        ax.scatter([defo[pn, 0]], [defo[pn, 1]], s=_MARKER_READOUT,
                   facecolors="none", edgecolors=_RED, linewidths=3.0, zorder=7)
    _annotate_tokens(ax, defo, token_ids, S, token_texts, next_label)
    if zoom:
        center = token_ids[S] if S < len(token_ids) else token_ids[S - 1]
        _set_zoom_window(ax, nodes, center)
    else:
        _set_full_window(ax, nodes)
    if title:
        ax.set_title(title, fontsize=11)
    _legend(ax, "s", "updated internal (Newton)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def _plot_read(nodes, edges, keep, token_ids, S, disp_xy, reaction, cvals, norm,
               cmap, path, dscale, rscale, zoom=True, wvals=None, wnorm=None,
               token_texts=None, next_label=None, title=None, cfg_disp=None,
               cfg_scale=0.0):
    arrived, future, support, interior = _roles(nodes, token_ids, S)
    base = nodes if cfg_disp is None else nodes + cfg_scale * cfg_disp
    defo = base + dscale * disp_xy
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_aspect("equal")
    ax.axis("off")
    _draw_bars(ax, nodes, edges, keep, cvals, norm, cmap, defo,
               wvals=wvals, wnorm=wnorm)
    ax.scatter(defo[interior, 0], defo[interior, 1], s=_MARKER_INTERNAL, c=_INT,
               marker="s", zorder=4)
    ax.scatter(defo[support, 0], defo[support, 1], s=_MARKER_SUPPORT, marker="s",
               c=_BLACK, zorder=5)
    ax.scatter(defo[arrived, 0], defo[arrived, 1], s=_MARKER_DRIVEN, c=_BLUE,
               edgecolors="white", linewidths=1.0, zorder=6)
    if len(future):
        ax.scatter(defo[future, 0], defo[future, 1], s=_MARKER_FUTURE,
                   facecolors="white", edgecolors=_FUT, linewidths=1.8, zorder=6)
    if S < len(token_ids):
        pn = token_ids[S]
        px, py = defo[pn]
        ax.scatter([px], [py], s=_MARKER_READOUT, facecolors="none", edgecolors=_RED,
                   linewidths=3.0, zorder=7)
        react = float(reaction[S]) if S < len(reaction) else 0.0
        if abs(react) > 1e-12:
            dx = rscale * react
            sign = 1.0 if dx >= 0.0 else -1.0
            dx = sign * max(abs(dx), 0.75)
            start_x = px + sign * 0.24
            end_x = px + dx
            if sign * (end_x - start_x) < 0.35:
                end_x = start_x + sign * 0.35
            ax.annotate("", xy=(end_x, py), xytext=(start_x, py),
                        arrowprops=dict(arrowstyle="-|>", color=_RED, lw=4.2,
                                        mutation_scale=28, shrinkA=0, shrinkB=0),
                        zorder=9)
    _annotate_tokens(ax, defo, token_ids, S, None, next_label)
    if zoom:
        center = token_ids[S] if S < len(token_ids) else token_ids[S - 1]
        _set_zoom_window(ax, nodes, center)
    else:
        _set_full_window(ax, nodes)
    if title:
        ax.set_title(title, fontsize=11)
    _legend(ax, "s", "interior tangent response")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def _save_colorbar(cmap, norm, path, label):
    fig, ax = plt.subplots(figsize=(6, 0.5))
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cb.set_label(label, fontsize=11)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

@torch.no_grad()
def _geo_frontier_config_fields(model, ids, layer_to_plot):
    Bsz, S = ids.shape
    if Bsz != 1:
        raise ValueError("Expected one prompt at a time.")

    e = model.wte_ln(model.wte(ids))
    f = torch.zeros(Bsz, S + 1, model.d, device=ids.device, dtype=e.dtype)
    n_layers = len(model.W_o)
    if layer_to_plot < 0:
        layer_to_plot = n_layers + layer_to_plot
    layer_to_plot = max(0, min(layer_to_plot, n_layers - 1))

    for l in range(n_layers):
        E = model.E_wte(f[:, :-1, :])
        V = e - E
        if l == layer_to_plot:
            lam = model.Lambda[l] if model.per_layer_kernel else model.Lambda
            d0 = model.delta0[l] if model.per_layer_kernel else model.delta0
            mix = model.phys_mix[l] if model.per_layer_kernel else model.phys_mix
            disp = model.phi[l](V)[0]
            u_heads = torch.einsum("hp,sp->hs", mix, disp)
            u_heads = model.geo_disp_scale * torch.tanh(u_heads)
            k = lam ** 2
            L0 = model.geo_L0.to(device=ids.device, dtype=e.dtype)
            prestretch = model.geo_delta_scale * L0.view(1, -1) * torch.tanh(d0)
            return {
                "u_heads": u_heads.detach().cpu(),
                "k": k.detach().cpu(),
                "prestretch": prestretch.detach().cpu(),
                "V": V[0].detach().cpu(),
                "layer": l,
                "row": S - 1,
            }

        A = model.geo_prestress_kernel(V, l)
        delta = torch.einsum("BHST,BTd->BHSd", A, V)
        if not model.skip_n_reg:
            delta = delta * model.n_reg[:S].view(1, 1, S, 1)
        delta = delta.permute(0, 2, 1, 3).reshape(Bsz, S, -1)
        delta = model.drop_gd(delta @ model.W_o[l])
        f = f + model.gd_step * torch.nn.functional.pad(delta, (0, 0, 1, 0))
        if model.use_ff:
            f = f + model.ff[l](f)

    raise RuntimeError("Layer selection failed.")

@torch.no_grad()
def _geo_settle_and_read(model, geom, nodes, V0, u_tok, k_bar, prestretch, S,
                         channel):
    dev, dt = model.B.device, model.B.dtype
    u_t = torch.as_tensor(np.asarray(u_tok[:S]), device=dev, dtype=dt)
    k = torch.as_tensor(np.asarray(k_bar), device=dev, dtype=dt).unsqueeze(0)
    pre = torch.as_tensor(np.asarray(prestretch), device=dev, dtype=dt).unsqueeze(0)

    nf = model.B.shape[0]
    u_free = torch.zeros(1, nf, device=dev, dtype=dt)
    u_free = u_free.index_add(1, model.tok[:S], u_t.unsqueeze(0))

    for _ in range(max(0, int(model.geo_newton_steps))):
        f_int, K = model._assemble_geo_batched(u_free, k, pre)
        K_ii = model._add_ridge(K[:, model.int_][:, :, model.int_])
        rhs = -f_int[:, model.int_]
        du = torch.linalg.solve(K_ii, rhs.unsqueeze(-1)).squeeze(-1)
        u_free = u_free.index_add(1, model.int_, du)

    _f, K = model._assemble_geo_batched(u_free, k, pre)
    K = K[0]

    base_nodes = model._current_geo_nodes().to(device=dev, dtype=dt)
    disp = torch.zeros(1, base_nodes.shape[0], 2, device=dev, dtype=dt)
    disp[:, model.geo_free_nodes, model.geo_free_axes] = u_free
    pos = base_nodes.unsqueeze(0) + disp
    r = pos[:, model.edge_j] - pos[:, model.edge_i]
    L = r.pow(2).sum(-1).sqrt().clamp_min(model.min_bar_length)
    L_rest = (model.geo_L0.to(device=dev, dtype=dt).unsqueeze(0) - pre
              ).clamp_min(model.min_bar_length)
    tension = k * (L - L_rest)
    geo = (tension / L)[0]

    tok, int_ = model.tok, model.int_
    K_tt = K[tok][:, tok]
    K_ti = K[tok][:, int_]
    K_it = K[int_][:, tok]
    K_ii = model._add_ridge(K[int_][:, int_])
    nt = tok.shape[0]
    Vt = V0[:S].to(dev)
    if channel is None or channel < 0:
        channel = int(Vt.pow(2).sum(0).argmax().item())
    read_t = torch.zeros(nt, device=dev, dtype=dt)
    read_t[:S] = Vt[:, channel].to(dt)
    read_i = -torch.linalg.solve(K_ii, K_it @ read_t)
    reaction_t = K_tt @ read_t + K_ti @ read_i

    read_free = torch.zeros(nf, device=dev, dtype=dt)
    read_free[tok] = read_t
    read_free[int_] = read_i
    read_full = np.zeros(2 * len(nodes))
    read_full[np.asarray(geom.free)] = read_free.cpu().numpy()
    return (read_full.reshape(len(nodes), 2), reaction_t.cpu().numpy(),
            int(channel), geo.detach().cpu().numpy(),
            u_free[0].detach().cpu().numpy())