import csv
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import scipy
import matplotlib


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_table(directory, stem, rows, headers, caption):
    write_csv(directory / f"{stem}.csv", rows)
    columns = list(rows[0])
    records = [[row[columns[0]]] + [f"{row[k]:.3e}" for k in columns[1:]] for row in rows]
    markdown = caption + "\n\n| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    markdown += "".join("| " + " | ".join(row) + " |\n" for row in records)
    (directory / f"{stem}.md").write_text(markdown)
    tex = "\\begin{table}[ht]\n\\centering\n\\small\n\\caption{" + caption + "}\n"
    tex += "\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}\n\\hline\n"
    tex += " & ".join(headers) + r" \\" + "\n\\hline\n"
    for row in rows:
        formatted = [row[columns[0]]]
        for key in columns[1:]:
            mantissa, exponent = f"{row[key]:.3e}".split("e")
            formatted.append(rf"${mantissa}\times10^{{{int(exponent)}}}$")
        tex += " & ".join(formatted) + r" \\" + "\n"
    tex += "\\hline\n\\end{tabular}\n\\end{table}\n"
    (directory / f"{stem}.tex").write_text(tex)


def save_fits(results, output):
    arrays = {"n_cases": np.array([len(results)])}
    for r in results:
        prefix = f"r{r['index']}_"
        arrays.update({prefix + "name": np.array(r["name"]),
                       prefix + "D_bar": r["D_bar"], prefix + "K_fit": r["K_fit"],
                       prefix + "schur_rel_err": np.array([r["schur_rel_err"]])})
    np.savez_compressed(output / "physical_lattice_fit_data.npz", **arrays)


def save_results(results, output, args):
    arrays, figure3_rows, supplement_rows, summaries = {}, [], [], []
    for result in results:
        prefix = f"r{result['index']}_"
        for key in ("Sigma", "factor", "G", "D_pruned", "K_pruned", "G_pruned",
                    "figure3_curve", "figure3_gains", "calibration_gains"):
            arrays[prefix + key] = result[key]
        for stage, error in enumerate(result["figure3_curve"]):
            figure3_rows.append(dict(case=result["name"], stage=stage,
                                     relative_squared_coefficient_error=float(error)))
        for protocol, curve in result["supplement_curves"].items():
            arrays[prefix + protocol + "_curve"] = curve
            arrays[prefix + protocol + "_gains"] = result["supplement_gains"][protocol]
            for stage, error in enumerate(curve):
                supplement_rows.append(dict(case=result["name"], protocol=protocol,
                                             stage=stage, relative_squared_coefficient_error=float(error)))
        summaries.append({"case": result["name"], "removed_bars": result["removed"],
                          "mean_figure3_gain": float(result["figure3_gains"].mean()),
                          "figure3_error_at_40": float(result["figure3_curve"][-1]),
                          "covariance_source": result["covariance_source"],
                          "figure3_error_cap_activations": result["figure3_cap_count"],
                          "supplement_error_cap_activations": result["supplement_cap_counts"],
                          "supplement_max_spectral_radii": result["supplement_max_spectral_radii"]})
    np.savez_compressed(output / "computed_results.npz", **arrays)
    write_csv(output / "Figure_3_curves.csv", figure3_rows)
    write_csv(output / "Figure_S4_curves.csv", supplement_rows)
    s2_caption = (r"Static-condensation realization errors before and after pruning. "
                  r"The optimizer fits stiffness to the inverse of the symmetric target. "
                  r"Errors are relative Frobenius norms with respect to the corresponding "
                  r"target matrices. Members with $D_e \leq 10^{-2}$ are deleted.")
    s3_caption = (r"Gain calibration using the matching digital-training covariance families. "
                  r"Errors are mean $\|w_{100}-w_\star\|_2^2/(\|w_\star\|_2^2+10^{-14})$ "
                  r"over 300 test tasks. Fixed gains use the minimum adaptive gain over 1000 "
                  r"independent calibration tasks. Power-8 uses eight iterations and an extra "
                  r"safety multiplier of 0.5. All protocols use the unpruned fitted lattice. "
                  r"Data are noiseless and equilibrium responses are computed numerically.")
    write_table(output / "tables", "Table_S2", [r["table_s2"] for r in results],
                ["Task family", "K-error before", "G-error before", "K-error after", "G-error after"], s2_caption)
    write_table(output / "tables", "Table_S3", [r["table_s3"] for r in results],
                ["Task family", r"Mean adaptive $\alpha$", r"Mean power-8 $\alpha$",
                 r"Fixed $\alpha$", "Adaptive error", "Power-8 error", "Fixed error"], s3_caption)
    source_root = Path(__file__).resolve().parents[1]
    sources = [source_root / "main.py", *sorted((source_root / "physical_inference").glob("*.py"))]
    def fingerprint(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    inputs = {key: {"path": str(Path(value).resolve()), "sha256": fingerprint(value)}
              for key, value in vars(args).items()
              if key in ("operators", "fit_cache", "covariances") and value is not None}
    report = {"inputs": inputs, "fit_mode": "existing" if args.fit_cache else "fresh",
              "settings": {"test_tasks": 300, "context": 20, "calibration_tasks": 1000,
                           "figure3_stages": 40, "supplement_stages": 100,
                           "pruning_threshold": 0.01, "fit_starts": 10,
                           "fit_maxiter": 6000, "fit_log_rigidity_bound": 6},
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
              "source_sha256": {str(p.relative_to(source_root)): fingerprint(p) for p in sources},
              "cases": summaries}
    (output / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n")
