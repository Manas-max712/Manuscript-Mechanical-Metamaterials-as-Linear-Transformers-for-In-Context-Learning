import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from .config import CASES
from .geometry import make_lattice


def plot_figure_s6(results, output_dir, cases, ratios):
    output_dir.mkdir(parents=True, exist_ok=True)
    lat = make_lattice(5, 4)
    values = np.concatenate([np.log10(item["D"][item["active"]]) for item in results.values()])
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(float(values.min()), float(values.max()))
    labels = dict(CASES)
    with plt.rc_context({"font.size":11, "axes.titlesize":15, "pdf.fonttype":42, "ps.fonttype":42}):
        fig, axes = plt.subplots(len(cases), len(ratios), figsize=(12.2, 8.2),
                                 constrained_layout=True, squeeze=False)
        for i, case in enumerate(cases):
            for j, ratio in enumerate(ratios):
                item = results[(case, ratio)]
                draw_lattice(axes[i,j], lat, item["D"], item["active"], cmap, norm,
                             labels[case], rf"$R={ratio:g}$" if ratio < 10000 else rf"$R={ratio:.0e}$",
                             item["summary"], j == 0, i == 0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), location="bottom",
                            fraction=0.045, pad=0.035, aspect=45)
        cbar.set_label(r"$\log_{10}$ bar stiffness", fontsize=15)
        cbar.ax.tick_params(labelsize=12)
        for extension in ("png", "pdf"):
            fig.savefig(output_dir/f"Figure_S6.{extension}", dpi=500, bbox_inches="tight")
        plt.close(fig)

def draw_boundary_arc(ax, pts, i, j, top_set, color, lw, alpha, zorder):
    p0, p1 = pts[i], pts[j]
    if p0[0] > p1[0]:
        p0, p1 = p1, p0
    is_top = i in top_set and j in top_set
    dx = abs(p1[0] - p0[0])
    rad = (0.10 + 0.035 * dx) * (1 if is_top else -1)
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


def format_sci(x: float) -> str:
    if not np.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10 ** exp)
    return rf"${mant:.2f}\times10^{{{exp}}}$"


def draw_lattice(ax, lat, D: np.ndarray, active: np.ndarray, cmap, norm,
                 row_label: str, col_label: str, metrics: dict[str, str],
                 show_row_label: bool, show_col_label: bool) -> None:
    nodes = lat.nodes
    top_set = set(lat.top_nodes)
    bot_set = set(lat.bot_nodes)
    boundary_edges = {
        e
        for e, (i, j) in enumerate(lat.edges)
        if (i in top_set and j in top_set) or (i in bot_set and j in bot_set)
    }
    logD = np.log10(np.clip(D, 1e-14, None))

    for e, (i, j) in enumerate(lat.edges):
        if not active[e]:
            continue
        val = logD[e]
        color = cmap(norm(val))
        lw = 0.55 + 2.2 * norm(val)
        if e in boundary_edges:
            draw_boundary_arc(ax, nodes, i, j, top_set, color="0.83", lw=0.55,
                              alpha=0.45, zorder=1)
            draw_boundary_arc(ax, nodes, i, j, top_set, color=color, lw=lw,
                              alpha=0.96, zorder=3)
        else:
            ax.plot(
                [nodes[i, 0], nodes[j, 0]],
                [nodes[i, 1], nodes[j, 1]],
                color=color,
                lw=lw,
                alpha=0.96,
                solid_capstyle="round",
                zorder=2,
            )

    ax.scatter(nodes[:, 0], nodes[:, 1], s=10, color="0.08", zorder=4)
    ax.scatter(nodes[lat.top_nodes, 0], nodes[lat.top_nodes, 1], s=26,
               color="#E64B35", edgecolors="white", linewidths=0.45, zorder=5)
    ax.scatter(nodes[lat.bot_nodes, 0], nodes[lat.bot_nodes, 1], s=22,
               color="#34495E", edgecolors="white", linewidths=0.45, zorder=5)

    test_mse = float(metrics.get("test_mse", "nan")) if metrics else np.nan
    info = rf"MSE {format_sci(test_mse)}"
    ax.text(0.5, -0.035, info, transform=ax.transAxes, ha="center", va="top",
            fontsize=12.5, clip_on=False)

    if show_col_label:
        ax.set_title(col_label, fontsize=15, pad=6)
    if show_row_label:
        ax.text(-0.12, 0.5, row_label, transform=ax.transAxes, rotation=90,
                ha="center", va="center", fontsize=15, fontweight="bold")

    ax.set_aspect("equal", adjustable="box")
    xpad = 0.10 * max(np.ptp(nodes[:, 0]), 1e-12)
    ypad = 0.22 * max(np.ptp(nodes[:, 1]), 1e-12)
    ax.set_xlim(nodes[:, 0].min() - xpad, nodes[:, 0].max() + xpad)
    ax.set_ylim(nodes[:, 1].min() - ypad, nodes[:, 1].max() + ypad)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
