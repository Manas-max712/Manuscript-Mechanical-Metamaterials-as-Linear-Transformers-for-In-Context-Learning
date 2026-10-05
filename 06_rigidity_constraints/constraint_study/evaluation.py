import numpy as np
from .mechanics import MallSystem
from .tasks import case_setup, make_tasks, oracle_mse
from .config import CASE_TRAIN_CONFIG, clip_from_ratio
from .files import read_npz, checkpoint_file


def evaluate_checkpoint(root, study, case, ratio, n_test=1000):
    path = checkpoint_file(root, study, case, ratio)
    data = read_npz(path)
    rigidities = np.asarray(data["best_D"], dtype=float)
    if rigidities.shape != (132,) or not np.isfinite(rigidities).all() or (rigidities <= 0).any():
        raise ValueError(f"Invalid best_D in {path}")
    system = MallSystem()
    sigma, _, Lx, Lw = case_setup(case)
    tasks = make_tasks(20001234, n_test, Lx, Lw)
    stages = CASE_TRAIN_CONFIG[case]["n_stages"]
    mse = float(system.evaluate_truss(rigidities, tasks, stages).mean())
    G = system.learned_G(rigidities)
    sigma_inverse = np.linalg.inv(sigma)
    alpha = float(np.sum(G * sigma_inverse) / max(np.sum(sigma_inverse**2), 1e-14))
    relative_error = float(np.linalg.norm(G - alpha * sigma_inverse, "fro") /
                           max(np.linalg.norm(alpha * sigma_inverse, "fro"), 1e-14))
    row = dict(case=case, test_mse=mse, g_err=relative_error,
               best_epoch=int(data["best_epoch"]), completed_epochs=int(data["epoch"]),
               best_val_mse=float(data["best_val"]), n_stages=stages,
               test_tasks=n_test, test_seed=20001234, oracle_test_mse=oracle_mse(tasks))
    item = dict(D=rigidities, summary=row, ckpt=path)
    low, high = clip_from_ratio(ratio)
    if study == "bounds":
        mask = np.asarray(data["best_mask"], dtype=bool)
        if not mask.all():
            raise ValueError(f"Expected unpruned bound_train checkpoint: {path}")
        item["active"] = mask
        row.update(stiffness_ratio=ratio, clip_min=low, clip_max=high,
                   D_min=float(rigidities.min()), D_max=float(rigidities.max()),
                   D_ratio=float(rigidities.max()/rigidities.min()))
    else:
        for key in ("train_mask", "low_mask", "high_mask"):
            item[key] = np.asarray(data[key], dtype=bool)
        memberships = sum(item[key].astype(int) for key in ("train_mask", "low_mask", "high_mask"))
        if memberships.shape != (132,) or not np.all(memberships == 1):
            raise ValueError(f"Invalid class masks in {path}")
        middle = rigidities[item["train_mask"]]
        if not np.all(rigidities[item["low_mask"]] == 0.01) or not np.all(rigidities[item["high_mask"]] == 200):
            raise ValueError(f"Checkpoint does not use the paper's finite tail rigidities: {path}")
        row.update(middle_stiffness_ratio=ratio, middle_clip_min=low, middle_clip_max=high,
                   middle_D_min=float(middle.min()), middle_D_max=float(middle.max()),
                   middle_D_ratio=float(middle.max()/middle.min()),
                   low_count=int(item["low_mask"].sum()), high_count=int(item["high_mask"].sum()),
                   trainable_count=int(item["train_mask"].sum()), soft_value=0.01, rigid_value=200.0)
    return item
