import numpy as np

ARRAY_FIELDS = (
    "Sigma_ev_true", "Sigma_ev_ahn", "Sigma_ev_lat", "Sigma_ev_ahn_layers",
    "Ahn_layer_scales", "Ahn_whitened_Q", "Lattice_whitened_G",
    "Lattice_whitened_G_target", "Sigma", "covariance_factor", "Ahn_Sigma",
    "G", "G_ev", "G_theory_ev", "Q_list", "P_list",
    "loss_ahn", "loss_lat_train", "loss_lat_val",
)
TEXT_FIELDS = ("name", "theorem")


def save_results_cache(results, path):
    payload = {
        "n_results": np.array([len(results)], dtype=int),
        "d": np.array([results[0]["G"].shape[0]], dtype=int),
        "n_layer": np.array([len(results[0]["Q_list"])], dtype=int),
    }
    for index, result in enumerate(results):
        prefix = f"r{index}_"
        for field in TEXT_FIELDS:
            payload[prefix + field] = np.array(result[field])
        for field in ARRAY_FIELDS:
            payload[prefix + field] = np.asarray(result[field], dtype=float)
    np.savez_compressed(path, **payload)
    print(f"Saved result arrays: {path}")


def load_results_cache(path):
    results = []
    with np.load(path, allow_pickle=False) as data:
        for index in range(int(data["n_results"][0])):
            prefix = f"r{index}_"
            result = {field: str(data[prefix + field]) for field in TEXT_FIELDS}
            result.update({field: data[prefix + field].copy() for field in ARRAY_FIELDS})
            results.append(result)
    return results
