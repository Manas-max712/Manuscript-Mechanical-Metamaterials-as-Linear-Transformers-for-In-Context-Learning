#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch


def compute_example(directory, side):
    source = Path(__file__).with_name(f'train_{side}x{side}.py')
    spec = importlib.util.spec_from_file_location(f'elastostatic_{side}', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    module.device = torch.device('cpu')
    nodes, edges = module.tile_unit_cell([(0.,0.), (1.,0.), (0.,1.), (1.,1.)], side, side)
    coords = torch.tensor(nodes, dtype=dtype)
    n, bars = len(nodes), len(edges)
    incidence = module.gen_incidence_from_edges(n, edges)
    rigidity = module.D_matrix(coords, edges, module.E, module.A)
    free = torch.ones(2*n, dtype=torch.bool)
    free[module.choose_kinematically_stable_dofs(n, coords)] = False
    loads = np.genfromtxt(directory / 'loads.csv', delimiter=',', names=True)
    if len(loads) != n or not np.array_equal(loads['node'], np.arange(n)):
        raise ValueError('Load file must contain one row per node in lattice order')
    forces = torch.tensor(np.column_stack((loads['fx'], loads['fy'])).ravel(), dtype=dtype)
    if not torch.isfinite(forces).all() or torch.any(forces[~free] != 0):
        raise ValueError('Invalid loads or nonzero loads at fixed coordinates')
    z = torch.zeros(1, 2*n, bars+3, dtype=dtype)
    z[0,:,bars+2] = free.to(dtype)
    z = module.generate_B_inplace(z, n, incidence, coords, free, torch.arange(bars), rigidity)
    z[0,:,bars] = forces
    target = module.get_target(z, rigidity, coords, edges, n, bars, bars+3)[0]
    checkpoint = torch.load(directory / 'best_by_val.pth', map_location='cpu', weights_only=True)
    model = module.Transformer_LG(7, 1, n, bars, bars+3, var=.1).eval()
    model.load_state_dict(checkpoint['model_state'], strict=True)
    normalized = (z-checkpoint['Z_mean'].to(dtype))/checkpoint['Z_std'].to(dtype)
    normalized[:,:,bars+1:] = z[:,:,bars+1:]
    with torch.no_grad():
        pred = (model(normalized)[:,:,bars+1]*checkpoint['u_std'].to(dtype)+checkpoint['u_mean'].to(dtype))[0]
    pred[~free] = 0
    return nodes, target.numpy().reshape(n,2), pred.numpy().reshape(n,2), free.numpy().reshape(n,2), int(checkpoint['epoch'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path(__file__).resolve().parent / 'examples')
    parser.add_argument('--output-dir', type=Path, default=Path('runs/reproduced_figure1'))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Choose a new output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for side in (5,10):
        inputs = args.inputs / f'{side}x{side}'
        nodes, target, pred, free, epoch = compute_example(inputs, side)
        output = args.output_dir / f'{side}x{side}'
        output.mkdir()
        history = 'history.pth' if side == 5 else 'history_full.pth'
        shutil.copy2(inputs / history, output / history)
        filename = 'test_pred_worst.csv' if side == 5 else 'test_worst_relL2_idx1357.csv'
        with (output / filename).open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['node','x','y','ux_true','uy_true','ux_pred','uy_pred','fixed_x','fixed_y'])
            for i in range(len(nodes)):
                writer.writerow([i,*nodes[i],*target[i],*pred[i],*map(int,~free[i])])
        metrics[f'{side}x{side}'] = dict(checkpoint_epoch=epoch,
            relative_L2=float(np.linalg.norm(pred-target)/np.linalg.norm(target)))
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    subprocess.run([sys.executable, str(Path(__file__).with_name('plot_figure1.py')),
                    '--run-5x5', str(args.output_dir/'5x5'),
                    '--run-10x10', str(args.output_dir/'10x10'),
                    '--output-dir', str(args.output_dir/'figures')], check=True)
    print(json.dumps(metrics, indent=2))
    print(f'Figure 1 saved to {args.output_dir / "figures"}')


if __name__ == '__main__':
    main()
