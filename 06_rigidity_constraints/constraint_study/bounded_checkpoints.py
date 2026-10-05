import os
import numpy as np
from .files import atomic_save_npz

def checkpoint_path(args, case: str):
    ratio_tag = f"R{args.stiffness_ratio:g}".replace(".", "p")
    prune_tag = f"p{args.prune_fraction:g}".replace(".", "p")
    fname = f"{args.mode}_{case}_{ratio_tag}_{prune_tag}.npz"
    return os.path.join(args.checkpoint_dir, fname)


def save_checkpoint(path: str, epoch: int, D_vec: np.ndarray,
                    active_mask: np.ndarray, m_adam: np.ndarray,
                    v_adam: np.ndarray, t_adam: int, best_val: float,
                    best_D: np.ndarray, best_mask: np.ndarray,
                    best_epoch: int, train_mses, val_mses,
                    total_pushes: int):
    atomic_save_npz(
        path,
        epoch=np.array(epoch, dtype=np.int64),
        D_vec=D_vec,
        active_mask=active_mask.astype(np.uint8),
        m_adam=m_adam,
        v_adam=v_adam,
        t_adam=np.array(t_adam, dtype=np.int64),
        best_val=np.array(best_val, dtype=float),
        best_D=best_D,
        best_mask=best_mask.astype(np.uint8),
        best_epoch=np.array(best_epoch, dtype=np.int64),
        train_mses=np.asarray(train_mses, dtype=float),
        val_mses=np.asarray(val_mses, dtype=float),
        total_pushes=np.array(total_pushes, dtype=np.int64),
    )


def load_checkpoint(path: str):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    return {
        "epoch": int(data["epoch"]),
        "D_vec": np.asarray(data["D_vec"], dtype=float),
        "active_mask": np.asarray(data["active_mask"], dtype=bool),
        "m_adam": np.asarray(data["m_adam"], dtype=float),
        "v_adam": np.asarray(data["v_adam"], dtype=float),
        "t_adam": int(data["t_adam"]),
        "best_val": float(data["best_val"]),
        "best_D": np.asarray(data["best_D"], dtype=float),
        "best_mask": np.asarray(data["best_mask"], dtype=bool),
        "best_epoch": int(data["best_epoch"]),
        "train_mses": list(np.asarray(data["train_mses"], dtype=float)),
        "val_mses": list(np.asarray(data["val_mses"], dtype=float)),
        "total_pushes": int(data["total_pushes"]),
    }
