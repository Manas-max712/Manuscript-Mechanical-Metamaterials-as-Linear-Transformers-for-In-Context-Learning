import os
import re
import numpy as np
from matplotlib import pyplot as plt


TITLE_FS = 20
LABEL_FS = 18
TICK_FS = 15
LEGEND_FS = 15
LINE_W = 2.8
TRAIN_COLOR = '#263f4f'
VAL_COLOR = '#d1495b'


def slugify(name):
    return re.sub('[^a-z0-9]+', '_', name.lower()).strip('_')


def style_loss_axis(ax, epochs, ymin, ymax, show_ylabel=True):
    ax.set_xlabel('Epoch', fontsize=LABEL_FS, labelpad=6)
    if show_ylabel:
        ax.set_ylabel('Relative loss', fontsize=LABEL_FS, labelpad=8)
    ax.tick_params(axis='both', labelsize=TICK_FS, width=1.0, length=5)
    ax.tick_params(axis='both', which='minor', length=3)
    ax.grid(True, which='major', axis='y', color='0.86', lw=0.9)
    ax.grid(False, which='minor')
    ax.set_xlim(1, epochs[-1])
    ax.set_ylim(max(ymin * 0.55, 1e-06), ymax * 1.8)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)


def plot_case(result, out_dir):
    epochs = np.arange(1, len(result['train']) + 1)
    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=180)
    ax.semilogy(epochs, result['train'], color=TRAIN_COLOR, lw=LINE_W, label='Training')
    ax.semilogy(epochs, result['val'], color=VAL_COLOR, lw=LINE_W, label='Validation')
    ax.set_title(result['name'], fontsize=TITLE_FS, pad=12)
    ymin = min(float(result['train'].min()), float(result['val'].min()))
    ymax = max(float(result['train'].max()), float(result['val'].max()))
    style_loss_axis(ax, epochs, ymin, ymax)
    ax.legend(frameon=False, fontsize=LEGEND_FS, loc='upper right')
    fig.tight_layout(pad=0.6)
    base = f"lattice_tf_loss_{slugify(result['name'])}"
    png_path = os.path.join(out_dir, base + '.png')
    pdf_path = os.path.join(out_dir, base + '.pdf')
    fig.savefig(png_path, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return (png_path, pdf_path)


def plot_combined(results, out_dir):
    all_train = np.concatenate([r['train'] for r in results])
    all_val = np.concatenate([r['val'] for r in results])
    ymin = min(float(all_train.min()), float(all_val.min()))
    ymax = max(float(all_train.max()), float(all_val.max()))
    fig, axes = plt.subplots(1, len(results), figsize=(16.8, 5.0), dpi=220)
    for ax, result, panel in zip(axes, results, ('(a)', '(b)', '(c)')):
        epochs = np.arange(1, len(result['train']) + 1)
        ax.semilogy(epochs, result['train'], color=TRAIN_COLOR, lw=LINE_W, label='Training')
        ax.semilogy(epochs, result['val'], color=VAL_COLOR, lw=LINE_W, label='Validation')
        ax.set_title(result['name'], fontsize=TITLE_FS, pad=12)
        style_loss_axis(ax, epochs, ymin, ymax, show_ylabel=ax is axes[0])
        ax.text(0.5, -0.29, panel, transform=ax.transAxes, ha='center', va='top', fontsize=18, family='serif')
    axes[-1].legend(frameon=False, fontsize=LEGEND_FS, loc='upper right')
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.22, top=0.86, wspace=0.25)
    png_path = os.path.join(out_dir, 'lattice_tf_losses_combined.png')
    pdf_path = os.path.join(out_dir, 'lattice_tf_losses_combined.pdf')
    fig.savefig(png_path, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return (png_path, pdf_path)
