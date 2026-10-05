import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from .geometry import nodes, edges, top_nodes, bot_nodes

PRUNE_THRESHOLD, PRUNE_TOL = 0.1, 1e-9
VMIN, VMAX = -1.0, 3.0


def plot_figure_s5(results, output_dir):
    from .config import CASES, CASE_LABELS
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 18,
                         "axes.titlesize": 20, "xtick.labelsize": 15,
                         "ytick.labelsize": 15}):
        for label, case in zip("abc", CASES):
            if case not in results:
                continue
            fig, ax = plt.subplots(figsize=(4.1, 3.7), dpi=300)
            draw_lattice(ax, results[case]["D_best"], title=CASE_LABELS[case])
            for ext in ("png", "pdf"):
                fig.savefig(output_dir / f"Figure_S5{label}.{ext}", bbox_inches="tight", dpi=300)
            plt.close(fig)
        save_colorbar(output_dir)


def draw_boundary_arc(ax, pts, i, j, color, lw, alpha, zorder):
    p0, p1 = pts[i], pts[j]
    if p0[0] > p1[0]:
        p0, p1 = p1, p0
    top_set = set(top_nodes)
    dx = abs(p1[0] - p0[0])
    rad = (0.10 + 0.035 * dx) * (1 if (i in top_set and j in top_set) else -1)
    patch = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-",
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=1,
        linewidth=lw,
        color=color,
        alpha=alpha,
        zorder=zorder,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_patch(patch)


def draw_lattice(ax, D, title=None):
    cmap = plt.get_cmap("viridis")
    top_set = set(top_nodes)
    bot_set = set(bot_nodes)
    boundary_edges = {
        e
        for e, (i, j) in enumerate(edges)
        if (i in top_set and j in top_set) or (i in bot_set and j in bot_set)
    }

    keep = D > PRUNE_THRESHOLD * (1.0 + PRUNE_TOL)
    logD = np.log10(np.clip(D, 1e-30, None))

    for e, (i, j) in enumerate(edges):
        if not keep[e]:
            continue
        t = np.clip((logD[e] - VMIN) / (VMAX - VMIN), 0.0, 1.0)
        color = cmap(t)
        lw = 0.55 + 3.1 * t
        if e in boundary_edges:
            draw_boundary_arc(ax, nodes, i, j, color=color, lw=lw, alpha=0.95, zorder=2)
        else:
            ax.plot(
                [nodes[i, 0], nodes[j, 0]],
                [nodes[i, 1], nodes[j, 1]],
                color=color,
                lw=lw,
                alpha=0.95,
                solid_capstyle="round",
                zorder=2,
            )

    ax.scatter(nodes[:, 0], nodes[:, 1], s=18, color="black", zorder=5)
    ax.scatter(
        nodes[top_nodes, 0],
        nodes[top_nodes, 1],
        s=58,
        color="#e74c3c",
        edgecolors="white",
        linewidths=0.7,
        zorder=6,
    )
    ax.scatter(
        nodes[bot_nodes, 0],
        nodes[bot_nodes, 1],
        s=48,
        color="#34495e",
        edgecolors="white",
        linewidths=0.7,
        zorder=6,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    xpad = 0.10 * max(np.ptp(nodes[:, 0]), 1e-12)
    ypad = 0.18 * max(np.ptp(nodes[:, 1]), 1e-12)
    ax.set_xlim(nodes[:, 0].min() - xpad, nodes[:, 0].max() + xpad)
    ax.set_ylim(nodes[:, 1].min() - ypad, nodes[:, 1].max() + ypad)
    if title:
        ax.set_title(title, pad=8)
    return int(np.sum(~keep)), int(len(D))


def save_colorbar(out_dir):
    fig, ax = plt.subplots(figsize=(6.8, 0.52), dpi=300)
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis"), norm=plt.Normalize(VMIN, VMAX))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cb.set_label(r"$\log_{10}D_e$", fontsize=18, labelpad=4)
    cb.ax.tick_params(labelsize=14, length=4, pad=2)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"mall_final_pruned_colorbar_threshold_1e-1.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(path)
    plt.close(fig)
