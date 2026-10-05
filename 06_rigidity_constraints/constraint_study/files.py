import csv
import os
from pathlib import Path
import numpy as np


def atomic_save_npz(path, **arrays):
    temporary = f"{path}.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def load_original_D(case, mall_root):
    path = Path(mall_root) / case / "postprocess_data.npz"
    data = read_npz(path)
    rigidities = np.asarray(data["D_best"], dtype=float)
    if rigidities.shape != (132,) or not np.isfinite(rigidities).all() or (rigidities <= 0).any():
        raise ValueError(f"Invalid MALL rigidities in {path}")
    return rigidities


def tag_float(value):
    return f"{value:g}".replace(".", "p")


def checkpoint_file(root, study, case, ratio):
    if study == "bounds":
        return Path(root) / "checkpoints/G_mall_bounds_pruning" / f"bound_train_{case}_R{tag_float(ratio)}_p0p3.npz"
    return Path(root) / "checkpoints/H_tail_quantization" / f"tail_quant_{case}_R{tag_float(ratio)}_low0p3_high0p1.npz"


def load_tail_masks(root, case, ratio, original_D):
    path = checkpoint_file(root, "tails", case, ratio)
    data = read_npz(path)
    low, high, middle = (np.asarray(data[key], dtype=bool)
                         for key in ("low_mask", "high_mask", "train_mask"))
    if any(mask.shape != original_D.shape for mask in (low, high, middle)):
        raise ValueError(f"Invalid class-mask shape in {path}")
    if not np.all(low.astype(int)+high.astype(int)+middle.astype(int) == 1):
        raise ValueError(f"Class masks must partition all members: {path}")
    if low.sum() != int(.30*len(original_D)) or high.sum() != int(.10*len(original_D)):
        raise ValueError(f"Class masks must use the paper's 30%/10% tail counts: {path}")
    if original_D[low].max() > original_D[~low].min() or original_D[high].min() < original_D[~high].max():
        raise ValueError(f"Class assignment is incompatible with the supplied initial rigidities: {path}")
    return low, high, middle


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
