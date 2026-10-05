import csv
import math
import numpy as np
from .config import CASE_TRAIN_CONFIG


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display(value, latex=False):
    if isinstance(value, str):
        return value
    if not math.isfinite(float(value)):
        return "--"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if latex:
        mantissa, exponent = f"{float(value):.3e}".split("e")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"
    return f"{float(value):.6g}"


def write_table(path, rows, headers, caption):
    write_csv(path.with_suffix(".csv"), rows)
    fields = list(rows[0])
    text = caption + "\n\n| " + " | ".join(headers) + " |\n"
    text += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        text += "| " + " | ".join(display(row[k]) for k in fields) + " |\n"
    path.with_suffix(".md").write_text(text)
    tex = "\\begin{table}[ht]\n\\centering\n\\small\n\\caption{" + caption + "}\n"
    tex += "\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}\n\\hline\n"
    tex += " & ".join(headers) + r" \\" + "\n\\hline\n"
    for row in rows:
        tex += " & ".join(display(row[k], latex=True) for k in fields) + r" \\" + "\n"
    tex += "\\hline\n\\end{tabular}\n\\end{table}\n"
    path.with_suffix(".tex").write_text(tex)


def export_tables(results, output_dir, test_seed=20001234, n_test=1000, pruning_rows=None):
    settings, alignment, inference = [], [], []
    for case, result in results.items():
        config = result.get("training_config", CASE_TRAIN_CONFIG[case])
        settings.append(dict(case=case, stages=int(result["n_stages"]),
                             epochs=len(result["train_mses"]),
                             train_tasks_per_epoch=int(config["N_train_tasks"]),
                             context_examples=20, training_queries=16,
                             learning_rate_max=float(config["lr_max"]),
                             learning_rate_min=float(config["lr_min"])))
        alignment.append(dict(case=case, alpha=float(result["alpha"]),
                              relative_operator_error=float(result["g_err"]),
                              relative_eigenvalue_error=float(result["eigval_rel_err"]),
                              top1_angle_degrees=float(result["angle_top1_deg"])))
        inference.append(dict(configuration="Unpruned", case=case, training_stages=int(result["n_stages"]),
                              query_mse_training_stages=float(result["mse_final_test"]),
                              query_mse_50_stages=float(result["test_per_stage"][50]),
                              query_mse_100_stages=float(result["test_per_stage"][100]),
                              max_spectral_radius=float(result["rho_test_max"])))
    write_table(output_dir / "Table_S7", settings,
                ["Task family", "Stages", "Epochs", "Training tasks/epoch", "Context size",
                 "Training queries", "Maximum learning rate", "Minimum learning rate"],
                "MALL training settings. The adjoint differentiates half the query MSE "
                "while holding the task-adaptive gain fixed. Reported losses are full MSE.")
    write_table(output_dir / "Table_S8", alignment,
                ["Task family", r"$\alpha$", "Relative operator error",
                 "Relative eigenvalue error", "Top-1 angle (degrees)"],
                r"Learned compliance alignment with $\alpha\Sigma_x^{-1}$. The scalar "
                r"$\alpha$ is fitted by Frobenius projection; it differs from the inference "
                r"gain $\eta$. The isotropic eigenspace angle is undefined.")
    if pruning_rows:
        for case, result in results.items():
            by_stage = {r['stage']: r for r in pruning_rows if r['case'] == case}
            depth = int(result['n_stages'])
            inference.append(dict(configuration="Pruned", case=case, training_stages=depth,
                                  query_mse_training_stages=by_stage[depth]['mse'],
                                  query_mse_50_stages=by_stage[50]['mse'],
                                  query_mse_100_stages=by_stage[100]['mse'],
                                  max_spectral_radius=by_stage[100]['max_spectral_radius']))
    caption = (f"Unnormalized query prediction MSE on {n_test} noiseless tasks "
               f"(test seed {test_seed}), using numerical equilibrium solves. "
               "The best-validation lattice is evaluated before and after deleting members with rigidity at or below 0.1, without retraining.")
    write_table(output_dir / "Table_S9", inference,
                ["Configuration", "Task family", "Training stages", "MSE at training stages",
                 "MSE at 50 stages", "MSE at 100 stages", r"$\rho_{\max}$"], caption)
