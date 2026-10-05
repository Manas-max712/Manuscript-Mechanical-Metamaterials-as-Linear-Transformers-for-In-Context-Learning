d = 5
N_few = 20
N_queries = 16
N_val = 200
N_test = 1000
eta_pgd_safety = 0.95
eta_pgd_cap = 1.0
nx, ny = 4, 4
extra_offsets = [(2, 1), (1, 2), (-2, 1), (-1, 2)]

CASES = ("identity", "diag", "rotated")
CASE_LABELS = {"identity": "Isotropic Gaussian", "diag": "Diagonal Non-Isotropic",
               "rotated": "Rotated Non-Isotropic"}
CASE_TRAIN_CONFIG = {
    "identity": {"N_train_tasks": 1000, "N_epochs": 3000,
                 "lr_max": 1e-2, "lr_min": 1e-5, "n_stages": 15},
    "diag": {"N_train_tasks": 1000, "N_epochs": 2000,
             "lr_max": 1e-2, "lr_min": 1e-4, "n_stages": 15},
    "rotated": {"N_train_tasks": 1000, "N_epochs": 2000,
                "lr_max": 1e-2, "lr_min": 1e-4, "n_stages": 25},
}
SNAPSHOT_EPOCHS = [0, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 1500, 2000, 3000]

VALIDATION_SEED = 9999
PAPER_TEST_SEED = 20001234
