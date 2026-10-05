#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["all", "bounds", "tails"], default="all")
    parser.add_argument("--case", choices=["all", "identity", "diag", "rotated"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/paper"))
    parser.add_argument("--mall-root", type=Path, default=Path("../05_mall/runs/paper"),
                        help="Main MALL results, used to initialize three-class fine-tuning")
    parser.add_argument("--from-results", type=Path,
                        help="Recompute figures/tables from existing checkpoints instead of training")
    parser.add_argument("--tail-mask-root", type=Path,
                        help="Optional checkpoint root: reuse only tail class masks while retraining from the main MALL model")
    parser.add_argument("--bound-ratios", nargs="+", type=float, default=[100, 1000, 10000, 20000])
    parser.add_argument("--tail-ratios", nargs="+", type=float, default=[100, 1000])
    parser.add_argument("--epochs", type=int, help="Override bounded-training epoch count")
    parser.add_argument("--fine-tune-epochs", type=int, default=300)
    parser.add_argument("--train-tasks", type=int, help="Override the 1000 training tasks/epoch")
    parser.add_argument("--resume", action="store_true", help="Resume this output directory's unchanged training configuration")
    parser.add_argument("--smoke", action="store_true", help="Two epochs, four training tasks/epoch, five evaluation tasks")
    args = parser.parse_args()
    for name in ("epochs", "fine_tune_epochs", "train_tasks"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for ratios in (args.bound_ratios, args.tail_ratios):
        if len(set(ratios)) != len(ratios) or any(not 1 <= r < float("inf") for r in ratios):
            parser.error("Ratios must be distinct finite values >= 1")
    if args.from_results and (args.resume or args.smoke or args.epochs or args.train_tasks):
        parser.error("Training/resume overrides do not apply to --from-results")
    output = args.output_dir.expanduser().resolve()
    args.mall_root = args.mall_root.expanduser().resolve()
    if args.tail_mask_root:
        args.tail_mask_root = args.tail_mask_root.expanduser().resolve()
        if args.from_results or args.study == "bounds":
            parser.error("--tail-mask-root applies only to fresh/resumed tail fine-tuning")
    cases = ["identity", "diag", "rotated"] if args.case == "all" else [args.case]
    studies = ["bounds", "tails"] if args.study == "all" else [args.study]
    if args.smoke:
        args.epochs = args.fine_tune_epochs = 2
        args.train_tasks = 4
    eval_tasks = 5 if args.smoke else 1000
    if args.from_results:
        args.from_results = args.from_results.expanduser().resolve()
        if not args.from_results.is_dir():
            parser.error(f"Missing checkpoint root: {args.from_results}")
    source = Path(__file__).resolve().parent
    files = [source/"main.py", *sorted((source/"constraint_study").glob("*.py")), source/"requirements.txt"]
    source_hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    initial_hashes, mask_hashes = {}, {}
    if "tails" in studies and not args.from_results:
        for case in cases:
            path = args.mall_root/case/"postprocess_data.npz"
            if not path.is_file():
                parser.error(f"Missing main MALL result: {path}. Supply --mall-root or first train Folder 5.")
            initial_hashes[case] = hashlib.sha256(path.read_bytes()).hexdigest()
            if args.tail_mask_root:
                for ratio in args.tail_ratios:
                    ratio_tag = f"{ratio:g}".replace(".", "p")
                    mask_path = args.tail_mask_root/"checkpoints/H_tail_quantization"/f"tail_quant_{case}_R{ratio_tag}_low0p3_high0p1.npz"
                    if not mask_path.is_file():
                        parser.error(f"Missing class-assignment checkpoint: {mask_path}")
                    mask_hashes[f"{case}/{ratio:g}"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    config = dict(studies=studies, cases=cases, bound_ratios=args.bound_ratios,
                  tail_ratios=args.tail_ratios, epochs=args.epochs,
                  fine_tune_epochs=args.fine_tune_epochs, train_tasks=args.train_tasks,
                  eval_tasks=eval_tasks, smoke=args.smoke, source_sha256=source_hashes,
                  initialization_sha256=initial_hashes,
                  tail_mask_checkpoint_sha256=mask_hashes,
                  mode="existing_checkpoints" if args.from_results else "training")
    config_file = output/"run_config.json"
    if args.resume:
        if not config_file.is_file() or json.loads(config_file.read_text()) != config:
            parser.error("Resume requires the same output directory, configuration, source files and initialization data.")
    elif output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output is not empty: {output}. Choose a new --output-dir or use --resume.")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output/".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output/".cache"))

    import numpy as np
    import scipy
    import matplotlib
    matplotlib.use("Agg")
    from constraint_study.bounded_training import train_bounded
    from constraint_study.tail_training import train_tail_quantized
    from constraint_study.files import checkpoint_file, write_csv, load_original_D, load_tail_masks, tag_float
    from constraint_study.evaluation import evaluate_checkpoint
    from constraint_study.tables import export_tables
    from constraint_study.figure_s6 import plot_figure_s6
    from constraint_study.figure_s7 import plot_figure_s7
    if initial_hashes:
        for case in cases:
            initial_D = load_original_D(case, args.mall_root)
            if args.tail_mask_root:
                for ratio in args.tail_ratios:
                    load_tail_masks(args.tail_mask_root, case, ratio, initial_D)
    if not args.resume:
        config_file.write_text(json.dumps(config, indent=2)+"\n")
    results_root = args.from_results or output
    bounded, tails, inputs = {}, {}, {}
    for study in studies:
        ratios = args.bound_ratios if study == "bounds" else args.tail_ratios
        directory = "G_mall_bounds_pruning" if study == "bounds" else "H_tail_quantization"
        checkpoint_dir = output/"checkpoints"/directory
        if not args.from_results:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for case in cases:
            for ratio in ratios:
                if not args.from_results:
                    options = SimpleNamespace(mode="bound_train", stiffness_ratio=ratio,
                        middle_stiffness_ratio=ratio, prune_fraction=0.3,
                        epochs=args.epochs, fine_tune_epochs=args.fine_tune_epochs,
                        train_tasks=args.train_tasks, eval_tasks=eval_tasks,
                        low_fraction=0.30, high_fraction=0.10, soft_value=0.01, rigid_value=200.0,
                        mall_root=str(args.mall_root), checkpoint_dir=str(checkpoint_dir),
                        tail_mask_root=args.tail_mask_root,
                        checkpoint_every=1, resume=args.resume)
                    if study == "bounds":
                        row, _, _ = train_bounded(case, options)
                        stem = f"bound_train_{case}_R{tag_float(ratio)}_p0p3"
                    else:
                        row = train_tail_quantized(case, options)
                        stem = f"tail_quant_{case}_R{tag_float(ratio)}_low0p3_high0p1"
                    write_csv(output/"outputs"/directory/f"{stem}.csv", [row])
                checkpoint = checkpoint_file(results_root, study, case, ratio)
                print(f"Evaluating {study}: {case}, R={ratio:g}", flush=True)
                item = evaluate_checkpoint(results_root, study, case, ratio, eval_tasks)
                (bounded if study == "bounds" else tails)[(case, ratio)] = item
                inputs[f"{study}/{case}/R{ratio:g}"] = dict(path=str(checkpoint),
                    sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    export_tables(bounded, tails, output/"tables", eval_tasks)
    if bounded:
        plot_figure_s6(bounded, output/"figures/Figure_S6", cases, args.bound_ratios)
    if tails:
        plot_figure_s7(tails, output/"figures/Figure_S7", args.tail_ratios)
    summary = dict(config=config, input_checkpoints=inputs,
                   versions=dict(python=platform.python_version(), numpy=np.__version__,
                                 scipy=scipy.__version__, matplotlib=matplotlib.__version__),
                   numerical_conventions=dict(fixed_gain_adjoint=True,
                       training_case_seed_offsets={"identity":0,"diag":1000000,"rotated":2000000},
                       validation_seed=9999, test_seed=20001234, noiseless_labels=True,
                       tail_topology="all bars retained with finite fixed tail rigidities"))
    (output/"run_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"Saved figures and tables to {output}", flush=True)


if __name__ == "__main__":
    main()
