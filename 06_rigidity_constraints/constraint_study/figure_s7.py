import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from .config import CASES
from .geometry import make_lattice


def plot_figure_s7(results, output_dir, ratios):
    output_dir.mkdir(parents=True, exist_ok=True)
    lat = make_lattice(5, 4)
    item = results.get(("rotated", 1000))
    cmap = plt.get_cmap("viridis")
    if item is not None:
        values = np.log10(item["D"][item["train_mask"]])
        norm = plt.Normalize(float(values.min()), float(values.max()))
    style = {"font.size":16, "axes.titlesize":18, "axes.labelsize":17,
             "xtick.labelsize":15, "ytick.labelsize":15, "pdf.fonttype":42, "ps.fonttype":42}
    with plt.rc_context(style):
        for label in ("a", "b", "c"):
            if label != "c" and item is None:
                continue
            fig, ax = plt.subplots(figsize=(3.35, 2.75), constrained_layout=True)
            if label == "a":
                plot_histogram(ax, None, item, 28, show_legend=False)
                ax.set_ylabel("count", fontsize=12)
                ax.set_xlabel(r"$\log_{10}$ bar stiffness", fontsize=12)
                ax.tick_params(labelsize=10)
            elif label == "b":
                plot_class_lattice_paper(ax, lat, item, cmap, norm)
            else:
                plot_mse_summary(ax, results, ratios)
                ax.set_ylabel("Test query-prediction MSE", fontsize=12)
                ax.set_xlabel("Imposed middle-band\nrigidity bound $R$", fontsize=12)
                ax.tick_params(labelsize=10)
                ax.legend(frameon=False, fontsize=9, loc="lower left")
            ax.set_title("")
            for extension in ("png", "pdf"):
                fig.savefig(output_dir/f"Figure_S7{label}.{extension}", dpi=500, bbox_inches="tight")
            plt.close(fig)

def draw_boundary_arc(ax, pts, i, j, top_set, color, lw, alpha, zorder, style="-"):
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
        linestyle=style,
        color=color,
        alpha=alpha,
        zorder=zorder,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_patch(patch)


def draw_edge(ax, lat, e, color, lw, alpha=1.0, zorder=2, style="-"):
    nodes = lat.nodes
    top_set = set(lat.top_nodes)
    bot_set = set(lat.bot_nodes)
    i, j = lat.edges[e]
    boundary = (i in top_set and j in top_set) or (i in bot_set and j in bot_set)
    if boundary:
        draw_boundary_arc(ax, nodes, i, j, top_set, color, lw, alpha, zorder, style)
    else:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            color=color,
            lw=lw,
            alpha=alpha,
            linestyle=style,
            solid_capstyle="round",
            zorder=zorder,
        )


def finish_lattice_axis(ax, lat):
    nodes = lat.nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], s=13, color="0.08", zorder=5)
    ax.scatter(nodes[lat.top_nodes, 0], nodes[lat.top_nodes, 1], s=32,
               color="#E64B35", edgecolors="white", linewidths=0.45, zorder=6)
    ax.scatter(nodes[lat.bot_nodes, 0], nodes[lat.bot_nodes, 1], s=26,
               color="#34495E", edgecolors="white", linewidths=0.45, zorder=6)
    ax.set_aspect("equal", adjustable="box")
    xpad = 0.10 * max(np.ptp(nodes[:, 0]), 1e-12)
    ypad = 0.18 * max(np.ptp(nodes[:, 1]), 1e-12)
    ax.set_xlim(nodes[:, 0].min() - xpad, nodes[:, 0].max() + xpad)
    ax.set_ylim(nodes[:, 1].min() - ypad, nodes[:, 1].max() + ypad)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_histogram(ax, original_D, item, bins: int, legend_outside: bool = False,
                   show_legend: bool = True):
    low_mask = item["low_mask"]
    high_mask = item["high_mask"]
    train_mask = item["train_mask"]
    implemented_D = item["D"]
    logD = np.log10(np.clip(implemented_D, 1e-14, None))

    lo = float(logD.min()) - 0.10
    hi = float(logD.max()) + 0.10
    bin_edges = np.linspace(lo, hi, bins + 1)

    ax.hist(logD[train_mask], bins=bin_edges, color="#64B5CD", alpha=0.85,
            label="middle tunable")

    low_x = float(np.median(logD[low_mask]))
    high_x = float(np.median(logD[high_mask]))
    low_n = int(low_mask.sum())
    high_n = int(high_mask.sum())
    ax.vlines(low_x, 0, low_n, color="0.70", lw=9.0, alpha=0.95,
              label="soft-frozen tail")
    ax.vlines(high_x, 0, high_n, color="0.12", lw=9.0, alpha=0.95,
              label="stiff-frozen tail")
    ax.text(low_x, low_n + 1.0, f"{low_n}", ha="center", va="bottom",
            fontsize=14, color="0.35")
    ax.text(high_x, high_n + 1.0, f"{high_n}", ha="center", va="bottom",
            fontsize=14, color="0.12")
    ax.set_xlabel(r"$\log_{10}$ bar stiffness")
    ax.set_ylabel("member count")
    ax.set_ylim(0, max(low_n, high_n, 1) * 1.18)
    ax.grid(True, color="0.86", linewidth=0.8, alpha=0.65)
    if show_legend:
        if legend_outside:
            ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.72),
                      fontsize=14)
        else:
            ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.58, 0.98),
                      fontsize=14)


def plot_class_lattice_paper(ax, lat, item, cmap, norm):
    D = item["D"]
    low_mask = item["low_mask"]
    high_mask = item["high_mask"]
    train_mask = item["train_mask"]
    logD = np.log10(np.clip(D, 1e-14, None))

    for e in np.where(low_mask)[0]:
        draw_edge(ax, lat, e, color="0.78", lw=0.75, alpha=0.45, zorder=1,
                  style=(0, (2.2, 2.2)))
    for e in np.where(train_mask)[0]:
        val = logD[e]
        color = cmap(norm(val))
        draw_edge(ax, lat, e, color=color, lw=0.9 + 1.9 * norm(val), alpha=0.96,
                  zorder=3)
    for e in np.where(high_mask)[0]:
        draw_edge(ax, lat, e, color="0.05", lw=2.7, alpha=0.95, zorder=4)

    finish_lattice_axis(ax, lat)


def plot_mse_summary(ax, results, ratios):
    colors = {"identity": "#1f4e79", "diag": "#b44e9d", "rotated": "#d47a00"}
    markers = {"identity": "o", "diag": "s", "rotated": "^"}
    for case, label in CASES:
        xs, ys = [], []
        for ratio in ratios:
            try:
                item = results[(case, ratio)]
            except KeyError:
                continue
            summary = item["summary"]
            if not summary:
                continue
            xs.append(float(ratio))
            ys.append(float(summary["test_mse"]))
        if xs:
            order = np.argsort(xs)
            xs = np.asarray(xs)[order]
            ys = np.asarray(ys)[order]
            ax.plot(xs, ys, marker=markers[case], color=colors[case], lw=1.9,
                    ms=6.2, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Imposed middle-band rigidity bound $R$")
    ax.set_ylabel("Test query-prediction MSE")
    ax.grid(True, which="both", color="0.86", linewidth=0.8, alpha=0.65)
    ax.legend(frameon=False, fontsize=14, loc="best")
