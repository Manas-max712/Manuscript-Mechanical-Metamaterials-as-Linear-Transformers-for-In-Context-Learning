#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def load_run(directory, side):
    history_path = directory / ('history.pth' if side == 5 else 'history_full.pth')
    history = torch.load(history_path, map_location='cpu', weights_only=True)
    matches = sorted(directory.glob('test_pred_worst.csv' if side == 5 else 'test_worst_relL2_idx*.csv'))
    if len(matches) != 1:
        raise ValueError(f'Expected one worst-case displacement CSV in {directory}, found {len(matches)}')
    data = np.genfromtxt(matches[0], delimiter=',', names=True)
    nodes = np.column_stack((data['x'], data['y']))
    true = np.column_stack((data['ux_true'], data['uy_true']))
    pred = np.column_stack((data['ux_pred'], data['uy_pred']))
    source = Path(__file__).with_name(f'train_{side}x{side}.py')
    spec = importlib.util.spec_from_file_location(f'train_{side}x{side}', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected, edges = module.tile_unit_cell([(0.,0.), (1.,0.), (0.,1.), (1.,1.)], side, side)
    if not np.allclose(nodes, expected):
        raise ValueError('CSV node ordering does not match the lattice geometry')
    return history, nodes, edges, true, pred


def draw_loss(ax, history):
    for key, label in [('train_loss', 'training'), ('val_loss', 'validation')]:
        y = history[key]
        ax.semilogy(np.arange(1, len(y)+1), y, label=label, linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(frameon=False)


def draw_deformation(ax, nodes, edges, displacement, color):
    deformed = nodes + displacement
    for i, j in edges:
        ax.plot(nodes[[i,j],0], nodes[[i,j],1], '--', color='0.7', linewidth=.8)
        ax.plot(deformed[[i,j],0], deformed[[i,j],1], color=color, linewidth=1)
    ax.scatter(nodes[:,0], nodes[:,1], c='black', s=10)
    ax.scatter(deformed[:,0], deformed[:,1], c=color, s=10)
    ax.set_aspect('equal')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-5x5', type=Path, default=Path('runs/5x5'))
    p.add_argument('--run-10x10', type=Path, default=Path('runs/10x10'))
    p.add_argument('--output-dir', type=Path, default=Path('figures'))
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('Choose a new output directory')
    runs = [load_run(args.run_5x5, 5), load_run(args.run_10x10, 10)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14})
    combined, axes = plt.subplots(2, 3, figsize=(13,8), constrained_layout=True)
    for row, (history, nodes, edges, true, pred) in enumerate(runs):
        for col in range(3):
            letter = 'abcdef'[3*row+col]
            single, ax = plt.subplots(figsize=(4.5,3.7), constrained_layout=True)
            for target in [ax, axes[row,col]]:
                if col == 0:
                    draw_loss(target, history)
                else:
                    draw_deformation(target, nodes, edges, true if col == 1 else pred,
                                     'orange' if col == 1 else 'green')
            axes[row,col].set_title(f'({letter})')
            for ext in ['png','pdf']:
                single.savefig(args.output_dir / f'Figure_1{letter}.{ext}', dpi=300)
            plt.close(single)
    for ext in ['png','pdf']:
        combined.savefig(args.output_dir / f'Figure_1.{ext}', dpi=300)
    plt.close(combined)


if __name__ == '__main__':
    main()
