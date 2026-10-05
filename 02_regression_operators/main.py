#!/usr/bin/env python3
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def environment_record():
    versions = {}
    for package in ("numpy", "scipy", "torch", "matplotlib"):
        versions[package] = importlib.metadata.version(package)
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions}


def source_hashes():
    paths = [ROOT / "main.py", *sorted((ROOT / "operator_study").glob("*.py"))]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "isotropic", "diagonal", "rotated"),
                        default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, help="New directory for this run")
    parser.add_argument("--epochs", type=int, help="Lattice epochs (paper: 500)")
    parser.add_argument("--ahn-iterations", type=int,
                        help="Override Ahn updates (defaults: isotropic 10000, others 100000)")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny execution check; these settings are not paper results")
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import torch
    from operator_study.cache import save_results_cache
    from operator_study.config import CASE_NAMES, TrainingSettings
    from operator_study.experiments import run_experiment
    from operator_study.plots import export_results

    torch.set_default_dtype(torch.float64)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable.")
    settings = TrainingSettings.smoke() if args.smoke else TrainingSettings()
    if args.smoke:
        torch.set_num_threads(1)
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        settings = replace(settings, lattice_epochs=args.epochs)
    if args.ahn_iterations is not None:
        if args.ahn_iterations < 1:
            parser.error("--ahn-iterations must be positive")
        settings = replace(settings, ahn_iterations=args.ahn_iterations)

    output = args.output_dir or ROOT / "runs" / ("smoke" if args.smoke else "paper")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"{output} is not an empty directory. Choose a new run directory.")
    output.mkdir(parents=True, exist_ok=True)
    cases = CASE_NAMES if args.case == "all" else (args.case,)
    record = {
        "mode": "training", "paper_settings": settings == TrainingSettings(),
        "smoke": args.smoke, "cases": list(cases), "device": device,
        "settings": asdict(settings), "environment": environment_record(),
        "source_sha256": source_hashes(),
        "checkpoints": "Model states for extraction; optimizer resume is not implemented.",
        "ahn_protocol": "FP32 Ahn schedule; full-Q heatmaps; seeded lattice initialization",
    }
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")

    results = []
    for case in cases:
        result = run_experiment(case, settings, device, output / case)
        save_results_cache([result], output / case / "results.npz")
        results.append(result)
        save_results_cache(results, output / "results.npz")
    export_results(results, output / "figures_and_tables")
    print(f"Training and exports complete: {output.resolve()}")


if __name__ == "__main__":
    main()
