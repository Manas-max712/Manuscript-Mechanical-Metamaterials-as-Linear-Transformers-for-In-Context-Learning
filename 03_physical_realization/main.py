#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", required=True, type=Path,
                        help="All-case results.npz from the Figure 2 training package")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/figure3"),
                        help="New or empty destination; existing results are protected")
    parser.add_argument("--fit-cache", type=Path,
                        help="Optional matching physical_lattice_fit_data.npz; default fits anew")
    parser.add_argument("--covariances", type=Path,
                        help="Optional exact r0/r1/r2 Sigma/factor NPZ; otherwise use the operator archive")
    args = parser.parse_args()
    for key in ("operators", "fit_cache", "covariances"):
        value = getattr(args, key)
        if value is not None:
            value = value.expanduser().resolve()
            if not value.is_file():
                parser.error(f"Missing --{key.replace('_', '-')} file: {value}")
            setattr(args, key, value)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output is not empty: {output}. Choose a new --output-dir.")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output / ".cache"))

    from physical_inference.inputs import load_cases
    from physical_inference.experiments import physical_fits, run_inference
    from physical_inference.outputs import save_fits, save_results
    from physical_inference.plots import plot_figure3, plot_figure_s4

    results = load_cases(args.operators, args.covariances)
    physical_fits(results, args.fit_cache)
    save_fits(results, output)
    for result in results:
        run_inference(result)
    save_results(results, output, args)
    plot_figure3(results, output / "figures/Figure_3")
    plot_figure_s4(results, output / "figures/Figure_S4")
    print(f"Saved Figure 3(a-d), Figure S4(a-c), Tables S2/S3 to {output}", flush=True)


if __name__ == "__main__":
    main()
