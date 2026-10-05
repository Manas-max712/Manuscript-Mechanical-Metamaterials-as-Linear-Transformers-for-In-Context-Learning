from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from .figure2_plots import export_panels, plot_annotated_matrix
from .loss_plots import plot_case, plot_combined
from .tables import export_table_s1
from .extraction_plot import export_extraction_diagram


def case_label(name):
    if "Rotated" in name:
        return "Rotated"
    if "Diagonal" in name:
        return "Diagonal"
    return "Isotropic"


def export_figure_s3(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {"font.size": 13, "axes.labelsize": 15, "axes.titlesize": 16,
                "legend.fontsize": 12, "xtick.labelsize": 13, "ytick.labelsize": 13}
    with plt.rc_context(settings):
        fig_all, axes = plt.subplots(len(results), 3, figsize=(15, 5 * len(results)),
                                     squeeze=False)
        for row, result in enumerate(results):
            label = case_label(result["name"])
            suffix = label.lower()
            lattice_name = "G rotated" if label == "Rotated" else f"G_{suffix}"
            ahn_prefix = "Ahn_Diagonal" if label == "Diagonal" else f"Ahn_{suffix}"
            specs = [
                (result["Ahn_whitened_Q"][0], f"{label}: Ahn layer 0\nΣ^½ Q₀ Σ^½",
                 f"{ahn_prefix}_layer0"),
                (result["Ahn_whitened_Q"][1], f"{label}: Ahn layer 1\nΣ^½ Q₁ Σ^½",
                 f"{ahn_prefix}_layer1"),
                (result["Lattice_whitened_G"], f"{label}: Lattice\nΣ^½ G$_{{\\mathrm{{LT}}}}$ Σ^½",
                 lattice_name),
            ]
            for col, (matrix, title, stem) in enumerate(specs):
                image = plot_annotated_matrix(axes[row, col], matrix, title)
                cbar = fig_all.colorbar(image, ax=axes[row, col], shrink=0.8)
                cbar.ax.tick_params(labelsize=13)
                fig, ax = plt.subplots(figsize=(5.2, 4.8))
                image = plot_annotated_matrix(ax, matrix, title)
                cbar = fig.colorbar(image, ax=ax, shrink=0.82)
                cbar.ax.tick_params(labelsize=13)
                fig.tight_layout()
                for extension in ("png", "pdf"):
                    fig.savefig(output_dir / f"{stem}.{extension}", dpi=150,
                                bbox_inches="tight")
                plt.close(fig)
        fig_all.tight_layout()
        for extension in ("png", "pdf"):
            fig_all.savefig(output_dir / f"Figure_S3_combined.{extension}", dpi=150,
                            bbox_inches="tight")
        plt.close(fig_all)


def export_results(results, output_dir, target="all"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if target in ("all", "figure2"):
        export_extraction_diagram(output_dir / "Figure_2")
        rotated = next((r for r in results if "Rotated" in r["name"]), None)
        if rotated is None and target == "figure2":
            raise ValueError("Figure 2(b,c) requires the rotated experiment.")
        if rotated is not None:
            export_panels(rotated["G"], rotated["Lattice_whitened_G"], output_dir / "Figure_2")
    if target in ("all", "figure-s2"):
        directory = output_dir / "Figure_S2"
        directory.mkdir(exist_ok=True)
        losses = [dict(name=r["name"], train=np.asarray(r["loss_lat_train"]),
                       val=np.asarray(r["loss_lat_val"])) for r in results]
        with plt.rc_context():
            plt.rcdefaults()
            for result in losses:
                plot_case(result, directory)
            if len(losses) == 3:
                plot_combined(losses, directory)
    if target in ("all", "figure-s3"):
        export_figure_s3(results, output_dir / "Figure_S3")
    if target in ("all", "table-s1"):
        export_table_s1(results, output_dir / "Table_S1")
