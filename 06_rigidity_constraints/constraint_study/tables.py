from .files import write_csv


def format_value(value, latex=False):
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if latex:
        mantissa, exponent = f"{value:.3e}".split("e")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"
    return f"{value:.8g}"


def write_table(path, rows, headers, caption):
    write_csv(path.with_suffix(".csv"), rows)
    fields = list(rows[0])
    md = caption + "\n\n| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"]*len(headers)) + " |\n"
    md += "".join("| " + " | ".join(format_value(row[k]) for k in fields) + " |\n" for row in rows)
    path.with_suffix(".md").write_text(md)
    tex = "\\begin{table}[ht]\n\\centering\n\\caption{" + caption + "}\n"
    tex += "\\begin{tabular}{l" + "r"*(len(headers)-1) + "}\n\\hline\n"
    tex += " & ".join(headers) + r" \\" + "\n\\hline\n"
    tex += "".join(" & ".join(format_value(row[k], True) for k in fields) + r" \\" + "\n" for row in rows)
    tex += "\\hline\n\\end{tabular}\n\\end{table}\n"
    path.with_suffix(".tex").write_text(tex)


def export_tables(bounds, tails, output_dir, n_test):
    common = (f" Test MSE is unnormalized query prediction error on {n_test} noiseless tasks "
              "with independent test seed 20001234 and numerical equilibrium solves. "
              "Inference uses 15/15/25 stages for isotropic/diagonal/rotated cases.")
    if bounds:
        rows = [dict(case=case, stiffness_ratio=ratio, test_mse=item["summary"]["test_mse"],
                     relative_operator_error=item["summary"]["g_err"],
                     actual_stiffness_ratio=item["summary"]["D_ratio"])
                for (case, ratio), item in bounds.items()]
        write_table(output_dir/"Table_S10", rows,
                    ["Task family", "$R$", "Test MSE", "Relative operator error", "Actual rigidity ratio"],
                    "Bounded-stiffness MALL retraining." + common)
        write_csv(output_dir/"bounded_diagnostics.csv", [x["summary"] for x in bounds.values()])
    if tails:
        rows = [dict(case=case, middle_stiffness_ratio=ratio, test_mse=item["summary"]["test_mse"])
                for (case, ratio), item in sorted(tails.items(), key=lambda kv: (kv[0][1], {"identity":0,"diag":1,"rotated":2}[kv[0][0]]))]
        write_table(output_dir/"Table_S12", rows,
                    ["Task family", "Middle-band $R$", "Test MSE"],
                    "Three-class member simplification: soft and stiff tails retain finite "
                    "rigidities 0.01 and 200; only the middle band is trained." + common)
        write_csv(output_dir/"tail_diagnostics.csv", [x["summary"] for x in tails.values()])
