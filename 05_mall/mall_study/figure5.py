import numpy as np
from matplotlib import pyplot as plt

TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 20, 24, 20, 24
COLORS = {"identity": "#2f4858", "diag": "#d1495b", "rotated": "#2a9d8f"}


def plot_figure5(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = [("a", "rotated", plot_train_val),
              ("b", "rotated", plot_relative_stage_mse),
              ("c", "diag", plot_eigen_bars),
              ("d", "diag", plot_eigen_scatter),
              ("e", "diag", plot_subspace_angles),
              ("f", "rotated", plot_eigen_bars),
              ("g", "rotated", plot_eigen_scatter),
              ("h", "rotated", plot_subspace_angles)]
    for letter, case, function in panels:
        if case in results:
            function(results[case], output_dir / f"Figure_5{letter}")


def finish_axis(ax, grid_axis="both"):
    ax.tick_params(axis="both", labelsize=TICK_FS, width=1.0, length=5)
    ax.tick_params(axis="both", which="minor", length=3)
    if grid_axis:
        ax.grid(True, which="major", axis=grid_axis, color="0.86", lw=0.9)
        ax.grid(False, which="minor")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save_fig(fig, path_base):
    png_path = path_base + ".png"
    pdf_path = path_base + ".pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_train_val(result, path_base):
    epochs = np.arange(1, len(result["train_mses"]) + 1)
    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=220)
    ax.semilogy(epochs, result["train_mses"], lw=2.0, color="#2f55ff", label="Train query MSE")
    ax.semilogy(epochs, result["val_mses"], lw=2.0, color="#d62728", label="Validation MSE")
    ax.set_xlabel("Epoch", fontsize=LABEL_FS)
    ax.set_ylabel("Query prediction MSE", fontsize=20)
    ax.legend(frameon=False, fontsize=LEGEND_FS, loc="upper right")
    finish_axis(ax)
    fig.tight_layout(pad=0.6)
    save_fig(fig, str(path_base))


def plot_eigen_scatter(result, path_base):
    target = np.asarray(result["target_eigs_sorted"], dtype=float)
    learned = np.asarray(result["G_eigs_sorted"], dtype=float)
    lo = min(float(target.min()), float(learned.min()))
    hi = max(float(target.max()), float(learned.max()))
    pad = 0.08 * max(hi - lo, 1e-12)
    fig, ax = plt.subplots(figsize=(6.3, 5.5), dpi=220)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            "k--", lw=1.4, alpha=0.65, label="Perfect match")
    ax.scatter(target, learned, s=90, color=COLORS[result["case"]],
               edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_xlabel(r"$\alpha \Sigma_x^{-1}$ eigenvalue", fontsize=LABEL_FS)
    ax.set_ylabel(r"Learned $G$ eigenvalue", fontsize=LABEL_FS)
    ax.legend(frameon=False, fontsize=LEGEND_FS, loc="upper left")
    finish_axis(ax)
    fig.tight_layout(pad=0.6)
    save_fig(fig, str(path_base))


def plot_eigen_bars(result, path_base):
    x = np.arange(len(result["G_eigs"]))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=220)
    ax.bar(x - width / 2, np.sort(result["G_eigs"]), width,
           color=COLORS[result["case"]], alpha=0.88, label=r"Learned $G$")
    ax.bar(x + width / 2, np.sort(result["alpha"] * result["Sinv_eigs"]), width,
           color="#f4a261", alpha=0.9, label=rf"$\alpha \Sigma_x^{{-1}}$ ($\alpha={float(result['alpha']):.3f}$)")
    ax.set_xlabel("Eigenvalue index", fontsize=LABEL_FS)
    ax.set_ylabel("Eigenvalue", fontsize=LABEL_FS)
    ax.legend(frameon=False, fontsize=LEGEND_FS, loc="upper left")
    finish_axis(ax, grid_axis="y")
    fig.tight_layout(pad=0.6)
    save_fig(fig, str(path_base))


def plot_relative_stage_mse(result, path_base):
    stages = np.asarray(result["test_stage_grid"], dtype=float)
    mses = np.asarray(result["test_stage_mses"], dtype=float)
    rel = mses / max(float(mses[0]), 1e-30)
    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=220)
    ax.semilogy(stages, rel, "o-", lw=2.6, ms=7, color="#2f4858")
    ax.set_xlabel("Equilibrium stages", fontsize=LABEL_FS)
    ax.set_ylabel("Test query MSE\n/ one-stage MSE", fontsize=20)
    finish_axis(ax)
    fig.tight_layout(pad=0.6)
    save_fig(fig, str(path_base))


def plot_subspace_angles(result, path_base):
    max_angles = np.asarray(result["subspace_max_angles"], dtype=float)
    mean_angles = np.asarray(result["subspace_mean_angles"], dtype=float)
    ks = np.asarray(result["subspace_ks"], dtype=float)
    valid = np.isfinite(max_angles)
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=220)
    if np.any(valid):
        ax.plot(ks[valid], max_angles[valid], "o-", lw=2.4, ms=7,
                color="#8e44ad", label="Max angle")
        ax.plot(ks[valid], mean_angles[valid], "s--", lw=2.1, ms=6,
                color="#16a085", label="Mean angle")
        ax.set_xticks(ks[valid])
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=LEGEND_FS, loc="upper right")
    else:
        ax.text(0.5, 0.5, "Target eigenspaces\ndegenerate",
                ha="center", va="center", transform=ax.transAxes, fontsize=LABEL_FS)
    ax.set_xlabel("Top-k eigenspace", fontsize=LABEL_FS)
    ax.set_ylabel("Principal angle (deg)", fontsize=LABEL_FS)
    finish_axis(ax)
    fig.tight_layout(pad=0.6)
    save_fig(fig, str(path_base))
