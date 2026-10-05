import numpy as np

from . import fitting, pruning
from .geometry import EDGES
from .inference import adaptive_alpha, evaluate_curve, power_alpha, synthetic_tasks


def physical_fits(results, fit_cache=None):
    if fit_cache is None:
        for result in results:
            print(f"Fitting {result['name']}: 10 starts, up to 6000 iterations each",
                  flush=True)
            d_bar, k_fit, error = fitting.fit_physical_lattice(
                result["K_target"], seed=100 + result["index"])
            result.update(D_bar=d_bar, K_fit=k_fit, schur_rel_err=error,
                          G_phys=np.linalg.inv(k_fit))
            print(f"  Relative stiffness fit error: {error:.8e}", flush=True)
    else:
        with np.load(fit_cache, allow_pickle=False) as data:
            if int(data["n_cases"][0]) != len(results):
                raise ValueError("Fit archive must contain all three cases")
            for result in results:
                prefix = f"r{result['index']}_"
                if prefix + "name" in data and str(data[prefix + "name"]) != result["name"]:
                    raise ValueError("Fit archive case order mismatch")
                d_bar = data[prefix + "D_bar"].astype(float)
                k_fit = data[prefix + "K_fit"].astype(float)
                error = float(data[prefix + "schur_rel_err"][0])
                if d_bar.shape != (len(EDGES),) or not np.isfinite(d_bar).all() or np.any(d_bar <= 0):
                    raise ValueError("Fit archive must contain positive unpruned rigidities")
                if k_fit.shape != (5, 5) or not np.isfinite(k_fit).all():
                    raise ValueError("Invalid fitted stiffness matrix")
                reconstructed = fitting.schur_from_stiffness(d_bar)
                if not np.allclose(reconstructed, k_fit, rtol=1e-8, atol=1e-10):
                    raise ValueError("Fit archive does not match the lattice geometry")
                relative = np.linalg.norm(k_fit - result["K_target"]) / np.linalg.norm(result["K_target"])
                if not np.isclose(relative, error, rtol=1e-5, atol=1e-11):
                    raise ValueError(f"Fit archive does not match {result['name']}'s target")
                result.update(D_bar=d_bar, K_fit=k_fit, schur_rel_err=error,
                              G_phys=np.linalg.inv(k_fit))
        print(f"Loaded matching fits: {fit_cache}", flush=True)


def run_inference(result):
    index, sigma = result["index"], result["Sigma"]
    tasks = synthetic_tasks(sigma, n_tasks=300, n_context=20, seed=3000 + index)

    d_pruned, removed, threshold = pruning.threshold_prune(result["D_bar"], -2.0)
    g_pruned, k_pruned = pruning.compliance_from_stiffness(d_pruned)
    curve3, gains3, capped3 = evaluate_curve(
        g_pruned, tasks, lambda sh, _: adaptive_alpha(g_pruned, sh), 40)

    g = result["G_phys"]
    calibration = synthetic_tasks(sigma, n_tasks=1000, n_context=20, seed=6000 + index)
    calibration_gains = np.asarray([adaptive_alpha(g, sh) for sh, _, _ in calibration])
    fixed_gain = float(np.percentile(calibration_gains, 0.0))
    protocols = {
        "adaptive": lambda sh, _: adaptive_alpha(g, sh),
        "power8": lambda sh, task: power_alpha(
            g, sh, n_iter=8, safety=0.95, cap=1.0,
            seed=9000 + 1000 * index + task, safety_scale=0.5),
        "fixed": lambda sh, _: fixed_gain,
    }
    curves, gains, cap_counts, radii = {}, {}, {}, {}
    for name, alpha_fn in protocols.items():
        curves[name], gains[name], cap_counts[name] = evaluate_curve(g, tasks, alpha_fn, 100)
        radii[name] = float(max(np.abs(np.linalg.eigvals(np.eye(5) - a * (g @ sh))).max()
                               for a, (sh, _, _) in zip(gains[name], tasks)))

    table2 = {
        "case": result["name"],
        "relative_K_error": float(np.linalg.norm(result["K_fit"] - result["K_target"])
                                  / np.linalg.norm(result["K_target"])),
        "relative_G_error": float(np.linalg.norm(g - result["G_target"])
                                  / np.linalg.norm(result["G_target"])),
    }
    table2.update(
        post_pruning_relative_K_error=float(np.linalg.norm(k_pruned - result["K_target"]) / np.linalg.norm(result["K_target"])),
        post_pruning_relative_G_error=float(np.linalg.norm(g_pruned - result["G_target"]) / np.linalg.norm(result["G_target"])),
    )
    table3 = {
        "case": result["name"],
        "mean_adaptive_alpha": float(gains["adaptive"].mean()),
        "mean_power8_alpha": float(gains["power8"].mean()),
        "fixed_alpha": fixed_gain,
        "adaptive_relative_coefficient_error": float(curves["adaptive"][-1]),
        "power8_relative_coefficient_error": float(curves["power8"][-1]),
        "fixed_relative_coefficient_error": float(curves["fixed"][-1]),
    }
    result.update(D_pruned=d_pruned, K_pruned=k_pruned, G_pruned=g_pruned,
                  removed=removed, pruning_threshold=threshold,
                  figure3_curve=curve3, figure3_gains=gains3,
                  figure3_cap_count=capped3, supplement_curves=curves,
                  supplement_gains=gains, supplement_cap_counts=cap_counts,
                  calibration_gains=calibration_gains,
                  supplement_max_spectral_radii=radii,
                  table_s2=table2, table_s3=table3)
    print(f"{result['name']}: removed {removed}/{len(EDGES)} bars; "
          f"Figure 3 error at stage 40 = {curve3[-1]:.8e}", flush=True)
