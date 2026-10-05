#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, default=Path('runs/newton_study'))
    a = p.parse_args()
    rows = []
    configs = []
    for n in (1, 2):
        run = a.study / f'newton{n}'
        if not (run / 'run_config.json').exists():
            print(f'Newton {n}: has not initialized yet')
            continue
        config = json.loads((run / 'run_config.json').read_text())
        configs.append(config)
        summary = json.loads((run / 'summary.json').read_text()) if (run / 'summary.json').exists() else {}
        history = list(csv.DictReader((run / 'metrics.csv').open())) if (run / 'metrics.csv').exists() else []
        row = dict(newton_steps=n, complete=summary.get('training_complete', False),
                   baseline_val_ce=float(history[0]['val_ce']) if history else None,
                   latest_validated_step=int(history[-1]['training_step']) if history else None,
                   latest_val_ce=float(history[-1]['val_ce']) if history else None,
                   best_val_ce=min(float(r['val_ce']) for r in history) if history else None,
                   mean_training_seconds_per_step=summary.get('mean_training_seconds_per_step'),
                   peak_allocated_GB=summary.get('peak_allocated_GB'))
        rows.append(row)
    if len(configs) == 2:
        for key in ['initialization_sha256','dataset_id','lr','weight_decay','batch_size','epochs','seed','data_seed','validate_every']:
            if configs[0][key] != configs[1][key]:
                raise RuntimeError(f'Unmatched study configuration: {key}')
        for key, value in configs[0]['model'].items():
            if key != 'geo_newton_steps' and configs[1]['model'][key] != value:
                raise RuntimeError(f'Unmatched model setting: {key}')
    print(json.dumps(rows, indent=2))
    if len(rows) == 2 and all(r['complete'] for r in rows):
        if rows[0]['latest_validated_step'] != rows[1]['latest_validated_step']:
            raise RuntimeError('Finished runs have different update budgets')
        print('Both arms completed equal update counts. Compare final CE and separately selected best CE.')
        print('Best validation CE difference (two minus one):', rows[1]['best_val_ce'] - rows[0]['best_val_ce'])
        print('Measured training-step time ratio (two / one):',
              rows[1]['mean_training_seconds_per_step'] / rows[0]['mean_training_seconds_per_step'])
    else:
        print('Study incomplete: do not interpret partial values as an equal-budget comparison.')


if __name__ == '__main__':
    main()
