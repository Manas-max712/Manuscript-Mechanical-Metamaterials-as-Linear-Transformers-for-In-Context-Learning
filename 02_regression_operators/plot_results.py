#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('runs/paper/results.npz'))
    parser.add_argument('--output-dir', type=Path, default=Path('figures'))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Choose a new output directory')
    os.environ.setdefault('MPLCONFIGDIR', str(args.output_dir/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    from operator_study.cache import load_results_cache
    from operator_study.plots import export_results
    export_results(load_results_cache(args.results), args.output_dir)


if __name__ == '__main__':
    main()
