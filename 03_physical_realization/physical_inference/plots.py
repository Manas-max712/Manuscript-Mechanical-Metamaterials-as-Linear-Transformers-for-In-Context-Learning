import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .geometry import NODES, EDGES, OUT_NODES, BOT_NODES

CASE_COLORS = ("#2f4858", "#d1495b", "#2a9d8f")
PROTOCOL_STYLES = {
    "adaptive": dict(color="#1F4E79", ls="-", marker="o"),
    "power8": dict(color="#B24C9A", ls="-.", marker="s"),
    "fixed": dict(color="#D17B0F", ls="--", marker="x"),
}


def save_pair(fig, directory, stem, pad=0.02, dpi=None):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{extension}", bbox_inches="tight",
                    pad_inches=pad, dpi=dpi)
    plt.close(fig)


def draw_inside_arc(ax, pts, i, j, color, lw, alpha=0.92, zorder=2):
    p0, p1 = pts[i], pts[j]
    if p0[0] > p1[0]:
        p0, p1 = p1, p0
    is_top = abs(p0[1] - 4.0) < 1e-12 and abs(p1[1] - 4.0) < 1e-12
    dx = abs(p1[0] - p0[0])
    amp = min(0.12 + 0.055 * dx, 0.55)
    sign = -1.0 if is_top else 1.0
    t = np.linspace(0.0, 1.0, 60)
    xy = (1 - t)[:, None] * p0 + t[:, None] * p1
    xy[:, 1] += sign * amp * np.sin(np.pi * t)
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, alpha=alpha,
            solid_capstyle="round", zorder=zorder)


def draw_lattice(ax, d_bar, norm, title):
    keep = d_bar > 10.0 ** -2.0
    log_d = np.log10(np.clip(d_bar, 1e-12, None))
    top_set, bot_set = set(OUT_NODES), set(BOT_NODES)
    cmap = plt.get_cmap("viridis")
    for edge, (i, j) in enumerate(EDGES):
        if not keep[edge]:
            continue
        value = norm(log_d[edge])
        color, width = cmap(value), 0.45 + 3.5 * value
        boundary = (i in top_set and j in top_set) or (i in bot_set and j in bot_set)
        if boundary and abs(NODES[i, 0] - NODES[j, 0]) > 1.0:
            draw_inside_arc(ax, NODES, i, j, color, width)
        else:
            ax.plot([NODES[i, 0], NODES[j, 0]], [NODES[i, 1], NODES[j, 1]],
                    color=color, lw=width, alpha=0.92,
                    solid_capstyle="round", zorder=2)
    ax.scatter(NODES[:, 0], NODES[:, 1], s=18, color="black", zorder=4)
    ax.scatter(NODES[OUT_NODES, 0], NODES[OUT_NODES, 1], s=48,
               color="#e74c3c", edgecolors="white", linewidths=0.7, zorder=5)
    ax.set_title(title, fontsize=18, pad=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.35, 4.35)
    ax.set_ylim(-0.35, 4.35)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_stage_curves(ax, results, compact=False):
    for result, color in zip(results, CASE_COLORS):
        curve = result["figure3_curve"]
        ax.semilogy(np.arange(len(curve)), curve, lw=3.0, color=color,
                    label=result["name"] + rf"  ($\bar{{\alpha}}={result['figure3_gains'].mean():.3f}$)")
    ax.set_xlabel("Equilibrium stages", fontsize=16 if compact else 22, labelpad=8)
    ax.set_ylabel("Mean relative squared coefficient error",
                  fontsize=14 if compact else 18, labelpad=10)
    ax.tick_params(axis="both", labelsize=13 if compact else 18, width=1.0, length=5)
    ax.grid(True, axis="y", which="major", color="0.85", lw=0.9)
    ax.grid(False, which="minor")
    ax.set_xlim(0, 40)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=12 if compact else 14,
              loc="upper right", handlelength=2.8)


def plot_figure3(results, directory):
    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=220)
    draw_stage_curves(ax, results)
    fig.tight_layout(pad=0.7)
    save_pair(fig, directory, "Figure_3a", pad=0.1)

    log_all = np.concatenate([np.log10(np.clip(r["D_bar"], 1e-12, None)) for r in results])
    norm = matplotlib.colors.Normalize(vmin=float(log_all.min()), vmax=float(log_all.max()))
    for label, result in zip("bcd", results):
        fig, ax = plt.subplots(figsize=(4.6, 4.25), dpi=320)
        draw_lattice(ax, result["D_bar"], norm, result["name"])
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90)
        save_pair(fig, directory, f"Figure_3{label}")

    fig, ax = plt.subplots(figsize=(6.8, 0.55), dpi=320)
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.45, top=0.85)
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis")
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, cax=ax, orientation="horizontal")
    colorbar.set_label(r"$\log_{10}$ bar stiffness", fontsize=18, labelpad=4)
    colorbar.ax.tick_params(labelsize=14, length=4, width=1.0)
    save_pair(fig, directory, "Figure_3_colorbar")

    fig = plt.figure(figsize=(12, 10))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
    ax = fig.add_subplot(grid[0, :])
    draw_stage_curves(ax, results, compact=True)
    ax.text(0.5, -0.22, "(a)", transform=ax.transAxes, ha="center")
    for i, (label, result) in enumerate(zip("bcd", results)):
        ax = fig.add_subplot(grid[1, i])
        draw_lattice(ax, result["D_bar"], norm, result["name"])
        ax.set_title(result["name"], fontsize=14)
        ax.text(0.5, -0.04, f"({label})", transform=ax.transAxes, ha="center")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.14,
                        hspace=0.45, wspace=0.12)
    cax = fig.add_axes([0.33, 0.055, 0.38, 0.018])
    colorbar = fig.colorbar(scalar, cax=cax, orientation="horizontal")
    colorbar.set_label(r"$\log_{10}$ bar stiffness")
    save_pair(fig, directory, "Figure_3", dpi=240)


def draw_gain_curves(ax, result):
    labels = {"adaptive": "adaptive", "power8": "power 8", "fixed": "fixed"}
    for protocol, curve in result["supplement_curves"].items():
        ax.plot(np.arange(len(curve)), curve, lw=1.45, ms=3.8, mew=0.9,
                markevery=10 if protocol == "fixed" else 4,
                label=labels[protocol], **PROTOCOL_STYLES[protocol])
    ax.set_yscale("log")
    ax.set_xlabel("Equilibrium stages")
    ax.set_ylabel("Mean relative squared coefficient error")
    ax.grid(True, which="both", alpha=0.22, lw=0.55)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper right", handlelength=2.0)


def plot_figure_s4(results, directory):
    with plt.rc_context({"font.size": 18, "axes.labelsize": 20,
                         "xtick.labelsize": 17, "ytick.labelsize": 17,
                         "legend.fontsize": 17, "pdf.fonttype": 42, "ps.fonttype": 42}):
        for label, result in zip("abc", results):
            fig, ax = plt.subplots(figsize=(5.9, 4.6), constrained_layout=True)
            draw_gain_curves(ax, result)
            save_pair(fig, directory, f"Figure_S4{label}", pad=0.1, dpi=600)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.7), constrained_layout=True)
    for label, ax, result in zip("abc", axes, results):
        draw_gain_curves(ax, result)
        ax.set_title(result["name"])
        ax.text(0.5, -0.2, f"({label})", transform=ax.transAxes, ha="center")
    save_pair(fig, directory, "Figure_S4", pad=0.1, dpi=240)
