import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from .geometry import nodes, edges, free, out_dofs, top_nodes, bot_nodes, B_strain

PAPER_EPOCHS = (0, 5, 25, 100, 500, 3000)


def plot_figure4(result, output_dir):
    epochs_all = np.asarray(result["snapshot_epochs"], dtype=int)
    epochs = [e for e in PAPER_EPOCHS if e in epochs_all]
    if len(epochs) < 2:
        epochs = epochs_all.tolist()
    indices = [int(np.flatnonzero(epochs_all == e)[0]) for e in epochs]
    selected = np.asarray(result["D_snapshots"], dtype=float)[indices]
    force = np.ones(len(out_dofs), dtype=float) / np.sqrt(len(out_dofs))
    deformations = np.asarray([solve_deformation(D, B_strain, free, out_dofs, force)
                               for D in selected])
    span = max(np.ptp(nodes[:, 0]), np.ptp(nodes[:, 1]), 1e-12)
    maximum = max(float(np.max(np.linalg.norm(deformations, axis=2))), 1e-14)
    scale = 0.80 * span / maximum
    positions = nodes[None, :, :] + scale * deformations
    logD = np.log10(np.clip(selected, 1e-12, None))
    norm = mpl.colors.Normalize(vmin=float(logD.min()), vmax=float(logD.max()))
    save_paper_exports(output_dir, epochs, nodes, edges, top_nodes, bot_nodes,
                       selected, positions, norm, plt.get_cmap("viridis"))
    np.savez_compressed(output_dir / "Figure_4_plot_data.npz", epochs=epochs,
                        nodes=nodes, edges=np.asarray(edges), free=np.asarray(free),
                        output_dofs=np.asarray(out_dofs), force_vec=force,
                        D_snapshots=selected, deformations=deformations,
                        deformed_nodes=positions, deform_scale=scale,
                        logD_limits=np.array([norm.vmin, norm.vmax]))
    metadata = dict(case="identity", epochs=epochs, numerical_ridge=1e-9,
                    force_vec=force.tolist(), deform_frac=0.80, deform_scale=scale,
                    geometry_order="training cell-corner first-visit order",
                    complete_paper_epochs=tuple(epochs) == PAPER_EPOCHS)
    (output_dir / "Figure_4_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def solve_deformation(D, B, free, output_dofs, force_vec):
    K = (B * D[None, :]) @ B.T
    K = K + 1e-9 * np.eye(K.shape[0])
    free_map = {dof: i for i, dof in enumerate(free)}
    f = np.zeros(len(free), dtype=float)
    for dof, val in zip(output_dofs, force_vec):
        if dof in free_map:
            f[free_map[dof]] = val
    u_free = np.linalg.solve(K, f)

    u_full = np.zeros(2 * ((max(free + output_dofs) // 2) + 1), dtype=float)
    for dof, idx in free_map.items():
        u_full[dof] = u_free[idx]
    return u_full.reshape(-1, 2)


def draw_boundary_arc(ax, pts, i, j, side, color, lw, alpha=1.0, zorder=1):
    p0, p1 = pts[i], pts[j]
    if p0[0] > p1[0]:
        p0, p1 = p1, p0
    span = p1[0] - p0[0]
    t = np.linspace(0.0, 1.0, 80)
    xs = p0[0] + span * t
    baseline = p0[1] + (p1[1] - p0[1]) * t
    height = 0.10 + 0.040 * abs(span)
    sign = -1.0 if side == "top" else 1.0
    ys = baseline + sign * (0.04 + height * 4.0 * t * (1.0 - t))
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha,
            solid_capstyle="round", zorder=zorder)


def draw_panel(ax, nodes, edges, top_nodes, bottom_nodes, D, deformed_nodes, norm, cmap, title):
    top_set = set(top_nodes)
    bottom_set = set(bottom_nodes)
    logD = np.log10(np.clip(D, 1e-12, None))

    for i, j in edges:
        is_boundary = (i in top_set and j in top_set) or (i in bottom_set and j in bottom_set)
        if is_boundary and abs(nodes[i, 0] - nodes[j, 0]) > 1.0:
            side = "top" if i in top_set and j in top_set else "bottom"
            draw_boundary_arc(ax, nodes, i, j, side, color="0.80", lw=0.65, alpha=0.55, zorder=1)
        else:
            ax.plot(
                [nodes[i, 0], nodes[j, 0]],
                [nodes[i, 1], nodes[j, 1]],
                color="0.82",
                lw=0.65,
                alpha=0.55,
                zorder=1,
            )

    for e, (i, j) in enumerate(edges):
        value = norm(logD[e])
        color = cmap(value)
        lw = 0.55 + 3.1 * value
        is_boundary = (i in top_set and j in top_set) or (i in bottom_set and j in bottom_set)
        if is_boundary and abs(nodes[i, 0] - nodes[j, 0]) > 1.0:
            side = "top" if i in top_set and j in top_set else "bottom"
            draw_boundary_arc(ax, deformed_nodes, i, j, side, color=color, lw=lw, alpha=0.94, zorder=2)
        else:
            ax.plot(
                [deformed_nodes[i, 0], deformed_nodes[j, 0]],
                [deformed_nodes[i, 1], deformed_nodes[j, 1]],
                color=color,
                lw=lw,
                alpha=0.94,
                solid_capstyle="round",
                zorder=2,
            )

    ax.scatter(nodes[:, 0], nodes[:, 1], s=9, color="0.70", zorder=3)
    ax.scatter(deformed_nodes[:, 0], deformed_nodes[:, 1], s=14, color="black", zorder=4)
    ax.scatter(
        deformed_nodes[top_nodes, 0],
        deformed_nodes[top_nodes, 1],
        s=34,
        color="#e74c3c",
        edgecolors="white",
        linewidths=0.5,
        zorder=5,
    )
    ax.scatter(
        nodes[bottom_nodes, 0],
        nodes[bottom_nodes, 1],
        s=28,
        color="#34495e",
        edgecolors="white",
        linewidths=0.5,
        zorder=5,
    )

    pts = np.vstack([nodes, deformed_nodes])
    xpad = 0.12 * max(np.ptp(pts[:, 0]), 1e-12)
    ypad = 0.18 * max(np.ptp(pts[:, 1]), 1e-12)
    ax.set_xlim(pts[:, 0].min() - xpad, pts[:, 0].max() + xpad)
    ax.set_ylim(pts[:, 1].min() - ypad, pts[:, 1].max() + ypad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_paper_exports(out_dir, epochs, nodes, edges, top_nodes, bottom_nodes,
                       D_sel, deformed_nodes, norm, cmap, save_separate=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ncols = min(3, len(epochs))
    nrows = int(np.ceil(len(epochs) / ncols))
    fig = plt.figure(figsize=(12.0, 4.0 * nrows + 0.6), dpi=300)
    grid = fig.add_gridspec(nrows + 1, ncols, height_ratios=[0.10] + [1] * nrows,
                           hspace=0.14, wspace=0.04)
    cax = fig.add_subplot(grid[0, :])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(r"$\log_{10} D_e$", fontsize=13)
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cax.tick_params(labelsize=11)
    for index, (epoch, D, pts) in enumerate(zip(epochs, D_sel, deformed_nodes)):
        slot = grid[1 + index // ncols, index % ncols]
        ax = fig.add_subplot(slot)
        draw_panel(ax, nodes, edges, top_nodes, bottom_nodes, D, pts, norm, cmap, "")
        box = slot.get_position(fig)
        fig.text((box.x0 + box.x1) / 2, box.y0 - 0.005,
                 f"({chr(97 + index)})  Epoch = {int(epoch)}",
                 ha="center", va="top", fontsize=13)
        if save_separate:
            panel, pax = plt.subplots(figsize=(4.6, 4.25), dpi=300)
            draw_panel(pax, nodes, edges, top_nodes, bottom_nodes, D, pts, norm, cmap, "")
            panel.tight_layout(pad=0.05)
            for ext in ("png", "pdf"):
                panel.savefig(out_dir / f"Figure_4{chr(97 + index)}.{ext}",
                              bbox_inches="tight", pad_inches=0.015)
            plt.close(panel)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"Figure_4.{ext}", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    barfig, barax = plt.subplots(figsize=(10.0, 0.45), dpi=300)
    colorbar = barfig.colorbar(sm, cax=barax, orientation="horizontal")
    colorbar.set_label(r"$\log_{10} D_e$", fontsize=13)
    barax.tick_params(labelsize=11)
    for ext in ("png", "pdf"):
        barfig.savefig(out_dir / f"Figure_4_colorbar.{ext}", bbox_inches="tight")
    plt.close(barfig)
