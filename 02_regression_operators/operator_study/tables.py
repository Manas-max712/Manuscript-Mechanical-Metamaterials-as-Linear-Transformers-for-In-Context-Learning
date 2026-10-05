import csv
from pathlib import Path

import numpy as np

from .analysis import dist_to_scaled_identity


def operator_diagnostics(result):
    operator = result["G"]
    symmetric = 0.5 * (operator + operator.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    return {
        "task_family": result["name"],
        "asymmetry": float(np.linalg.norm(operator - operator.T, "fro") /
                           max(np.linalg.norm(operator, "fro"), 1e-14)),
        "lambda_min": float(eigenvalues[0]),
        "lambda_max": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / max(eigenvalues[0], 1e-14)),
        "whitened_distance_raw": float(dist_to_scaled_identity(result["Lattice_whitened_G"])),
        "whitened_distance_target": float(dist_to_scaled_identity(result["Lattice_whitened_G_target"])),
    }


def export_table_s1(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [operator_diagnostics(result) for result in results]
    csv_path = output_dir / "Table_S1.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    caption = ("Operator diagnostics. Asymmetry uses raw G_LT; eigenvalues and condition "
               "number use G_target=(G_LT+G_LT.T)/2. The last two columns give relative "
               "Frobenius distances of the whitened raw and target operators to their "
               "closest scaled identities, using the symmetric covariance square root.")
    header = ["Task family", "Asymmetry", "Minimum eigenvalue", "Maximum eigenvalue",
              "Condition number", "Whitened distance (raw)", "Whitened distance (target)"]
    markdown = [caption, "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    tex = [r"\begin{tabular}{lrrrrrr}", r"\hline",
           r"Task family & Asymmetry & $\lambda_{\min}$ & $\lambda_{\max}$ & $\kappa$ & Raw whitened dist. & Target whitened dist. \\",
           r"\hline"]
    for row in rows:
        formatted = [row["task_family"], f'{row["asymmetry"]:.2e}',
                     f'{row["lambda_min"]:.4f}', f'{row["lambda_max"]:.4f}',
                     f'{row["condition_number"]:.4f}', f'{row["whitened_distance_raw"]:.2e}', f'{row["whitened_distance_target"]:.2e}']
        markdown.append("| " + " | ".join(formatted) + " |")
        tex.append(" & ".join(formatted) + r" \\")
    tex.extend([r"\hline", r"\end{tabular}"])
    (output_dir / "Table_S1.md").write_text("\n".join(markdown) + "\n")
    (output_dir / "Table_S1.tex").write_text("\n".join(tex) + "\n")
    return rows
