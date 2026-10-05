#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", "identity", "diag", "rotated"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/paper"))
    parser.add_argument("--from-results", type=Path,
                        help="Optional existing case folders containing postprocess_data.npz")
    parser.add_argument("--study", choices=["all", "core", "pruning", "tolerance"], default="all")
    parser.add_argument("--epochs", type=int, help="Override the paper training duration")
    parser.add_argument("--train-tasks", type=int, help="Override the 1000 tasks per epoch")
    parser.add_argument("--test-seed", type=int, default=20001234,
                        help="Seed for independent test tasks")
    parser.add_argument("--smoke", action="store_true",
                        help="Two epochs, four training tasks/epoch, five evaluation tasks; not paper results")
    args = parser.parse_args()
    if args.from_results and (args.epochs or args.train_tasks or args.smoke):
        parser.error("Training overrides apply to fresh training, not --from-results")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.train_tasks is not None and args.train_tasks <= 0:
        parser.error("--train-tasks must be positive")
    if args.test_seed < 0 or args.test_seed >= 2**32:
        parser.error("--test-seed must be in [0, 2**32)")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output is not empty: {output}. Choose a new --output-dir.")
    if args.from_results:
        args.from_results = args.from_results.expanduser().resolve()
        if not args.from_results.is_dir():
            parser.error(f"Missing results directory: {args.from_results}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output / ".cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")

    import numpy as np
    import scipy
    import matplotlib
    matplotlib.use("Agg")
    from mall_study.config import CASES, CASE_TRAIN_CONFIG
    from mall_study.training import train_case
    from mall_study.results import load_case, add_plot_arrays, save_case_postprocess_data
    from mall_study.analysis import analyze_model
    from mall_study.tasks import case_setup, make_tasks
    from mall_study.tables import export_tables, write_csv, write_table
    from mall_study.figure4 import plot_figure4
    from mall_study.figure5 import plot_figure5
    from mall_study.figure_s5 import plot_figure_s5
    from mall_study.pruning import evaluate_pruning
    from mall_study.tolerance import run_tolerance

    cases = CASES if args.case == "all" else (args.case,)
    n_eval = 5 if args.smoke else 1000
    results, inputs = {}, {}
    for case in cases:
        if args.from_results:
            path = args.from_results / case / "postprocess_data.npz"
            result = load_case(path, case)
            inputs[case] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            config_path = path.with_name("training_config.json")
            if config_path.exists():
                result["training_config"] = json.loads(config_path.read_text())
            sigma, _, Lx, Lw = case_setup(case)
            if not np.allclose(result["Sigma_x"], sigma, rtol=1e-12, atol=1e-12):
                parser.error(f"Covariance in {path} does not match {case}")
            result.update(analyze_model(result["D_best"], sigma,
                make_tasks(9999, min(n_eval, 200), Lx, Lw),
                make_tasks(args.test_seed, n_eval, Lx, Lw), int(result["n_stages"])))
            add_plot_arrays(result)
            case_dir = output / case
            case_dir.mkdir()
            save_case_postprocess_data(case_dir, result, result["snapshot_epochs"],
                                       result["D_snapshots"])
            metadata = dict(result.get("training_config", CASE_TRAIN_CONFIG[case]),
                            test_seed=args.test_seed, evaluation_tasks=n_eval,
                            source_checkpoint=str(path))
            (case_dir / "training_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"Loaded {case}: {path}", flush=True)
        else:
            config = dict(CASE_TRAIN_CONFIG[case])
            if args.epochs is not None:
                config["N_epochs"] = args.epochs
            if args.train_tasks is not None:
                config["N_train_tasks"] = args.train_tasks
            if args.smoke:
                config.update(N_epochs=2, N_train_tasks=4)
            result = train_case(case, output / case, config=config,
                                test_seed=args.test_seed, n_eval=n_eval)
            result["training_config"] = config
        results[case] = result

    pruning_rows = []
    if args.study in ("all", "core", "pruning"):
        for case, result in results.items():
            print(f"Evaluating pruning for {case}", flush=True)
            pruning_rows.extend(evaluate_pruning(case, result, test_seed=args.test_seed, n_test=n_eval))
        write_csv(output / "tables/S4_1_postpruning_inference.csv", pruning_rows)

    if args.study in ("all", "core"):
        export_tables(results, output / "tables", test_seed=args.test_seed, n_test=n_eval, pruning_rows=pruning_rows)
        if "identity" in results:
            plot_figure4(results["identity"], output / "figures/Figure_4")
        plot_figure5(results, output / "figures/Figure_5")
    if args.study in ("all", "pruning"):
        plot_figure_s5(results, output / "figures/Figure_S5")
    if args.study in ("all", "tolerance"):
        print("Evaluating fabrication tolerance", flush=True)
        rows = run_tolerance(results, test_seed=args.test_seed, n_test=n_eval)
        write_csv(output / "tables/Section_6_tolerance.csv", rows)
        table_rows = [{k: r[k] for k in ("case", "sigma", "clean15", "noisy15", "noisy50", "noisy100")} for r in rows]
        write_table(output / "tables/Table_S11", table_rows,
                    ["Task family", r"$\sigma_{\log D}$", "Unperturbed MSE at 15", "Perturbed MSE at 15", "Perturbed MSE at 50", "Perturbed MSE at 100"],
                    f"Rigidity perturbations and additional inference stages. Unnormalized query MSE on {n_eval} noiseless test tasks, averaged over ten independently perturbed lattices. Equilibrium solves are numerical.")

    source_dir = Path(__file__).resolve().parent
    sources = [source_dir / "main.py", *sorted((source_dir / "mall_study").glob("*.py")),
               source_dir / "requirements.txt"]
    manifest = dict(mode="existing_results" if args.from_results else "training",
                    smoke=args.smoke, cases=list(cases), study=args.study,
                    test_seed=args.test_seed, evaluation_tasks=n_eval,
                    training_seed_convention="epoch for all families",
                    full_gain_derivative=False, inputs=inputs,
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  scipy=scipy.__version__, matplotlib=matplotlib.__version__),
                    source_sha256={str(p.relative_to(source_dir)):
                                   hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (output / "run_summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved results to {output}", flush=True)


if __name__ == "__main__":
    main()
