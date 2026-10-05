from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

PLOT_LABEL_FS = 15
PLOT_TITLE_FS = 16
PLOT_TICK_FS = 13
PLOT_ANNOT_FS = 11


def polish_axis(ax):
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FS)
    ax.xaxis.label.set_size(PLOT_LABEL_FS)
    ax.yaxis.label.set_size(PLOT_LABEL_FS)
    ax.title.set_size(PLOT_TITLE_FS)


def plot_annotated_matrix(ax, matrix, title):
    max_abs = max(float(np.max(np.abs(matrix))), 1e-12)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-max_abs,
                      vmax=max_abs, aspect="equal")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if abs(value) > 0.45 * max_abs else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color=color, fontsize=PLOT_ANNOT_FS, fontweight="bold")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_title(title, fontsize=PLOT_TITLE_FS)
    polish_axis(ax)
    return image


def export_panels(raw, whitened, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "font.size": PLOT_TICK_FS,
        "axes.labelsize": PLOT_LABEL_FS,
        "axes.titlesize": PLOT_TITLE_FS,
        "xtick.labelsize": PLOT_TICK_FS,
        "ytick.labelsize": PLOT_TICK_FS,
    }
    paths = []

    def save(fig, stem):
        for extension in ("png", "pdf"):
            path = output_dir / f"{stem}.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)

    with plt.rc_context(settings):
        specifications = [
            (raw, r"Rotated: raw $G_{\mathrm{LT}}$", "Figure_2b"),
            (whitened,
             r"Rotated: whitened $\Sigma_x^{1/2}G_{\mathrm{LT}}\Sigma_x^{1/2}$",
             "Figure_2c"),
        ]
        for matrix, title, stem in specifications:
            fig, ax = plt.subplots(figsize=(5.2, 4.8))
            image = plot_annotated_matrix(ax, matrix, title)
            colorbar = plt.colorbar(image, ax=ax, shrink=0.82)
            colorbar.ax.tick_params(labelsize=PLOT_TICK_FS)
            fig.tight_layout()
            save(fig, stem)

        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
        for ax, matrix, title in [
            (axes[0], raw, r"Raw $G_{\mathrm{LT}}$"),
            (axes[1], whitened,
             r"Whitened $\Sigma_x^{1/2}G_{\mathrm{LT}}\Sigma_x^{1/2}$"),
        ]:
            image = plot_annotated_matrix(ax, matrix, title)
            colorbar = plt.colorbar(image, ax=ax, shrink=0.82)
            colorbar.ax.tick_params(labelsize=PLOT_TICK_FS)
        fig.tight_layout()
        save(fig, "Figure_2bc")
    return paths
