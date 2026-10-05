import numpy as np

D, NY, N_FEW, N_QUERIES, N_VAL, N_TEST = 5, 4, 20, 16, 200, 1000
ETA_SAFETY, ETA_CAP = 0.95, 1.0
ORIGINAL_CLIP = (0.01, 200.0)
CASES = [("identity", "Isotropic"), ("diag", "Diagonal"), ("rotated", "Rotated")]
BOUND_RATIOS = [100, 1000, 10000, 20000]
TAIL_RATIOS = [100, 1000]
CASE_TRAIN_CONFIG = {
    "identity": {"N_train_tasks": 1000, "N_epochs": 3000, "lr_max": 1e-2, "lr_min": 1e-5, "n_stages": 15},
    "diag": {"N_train_tasks": 1000, "N_epochs": 2000, "lr_max": 1e-2, "lr_min": 1e-4, "n_stages": 15},
    "rotated": {"N_train_tasks": 1000, "N_epochs": 2000, "lr_max": 1e-2, "lr_min": 1e-4, "n_stages": 25},
}

def clip_from_ratio(ratio: float):
    center = np.sqrt(ORIGINAL_CLIP[0] * ORIGINAL_CLIP[1])
    half = np.sqrt(float(ratio))
    return center / half, center * half
