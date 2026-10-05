import os
import numpy as np
from .files import atomic_save_npz

def checkpoint_path(args, case: str):
    ratio_tag = f"R{args.middle_stiffness_ratio:g}".replace(".", "p")
    low_tag = f"low{args.low_fraction:g}".replace(".", "p")
    high_tag = f"high{args.high_fraction:g}".replace(".", "p")
    fname = f"tail_quant_{case}_{ratio_tag}_{low_tag}_{high_tag}.npz"
    return os.path.join(args.checkpoint_dir, fname)


def save_checkpoint(path, epoch, D_vec, train_mask, low_mask, high_mask,
                    m_adam, v_adam, t_adam, best_val, best_D, best_epoch,
                    train_mses, val_mses, total_pushes):
    atomic_save_npz(
        path,
        epoch=np.array(epoch, dtype=np.int64),
        D_vec=D_vec,
        train_mask=train_mask.astype(np.uint8),
        low_mask=low_mask.astype(np.uint8),
        high_mask=high_mask.astype(np.uint8),
        m_adam=m_adam,
        v_adam=v_adam,
        t_adam=np.array(t_adam, dtype=np.int64),
        best_val=np.array(best_val, dtype=float),
        best_D=best_D,
        best_epoch=np.array(best_epoch, dtype=np.int64),
        train_mses=np.asarray(train_mses, dtype=float),
        val_mses=np.asarray(val_mses, dtype=float),
        total_pushes=np.array(total_pushes, dtype=np.int64),
    )


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    return {
        "epoch": int(data["epoch"]),
        "D_vec": np.asarray(data["D_vec"], dtype=float),
        "train_mask": np.asarray(data["train_mask"], dtype=bool),
        "low_mask": np.asarray(data["low_mask"], dtype=bool),
        "high_mask": np.asarray(data["high_mask"], dtype=bool),
        "m_adam": np.asarray(data["m_adam"], dtype=float),
        "v_adam": np.asarray(data["v_adam"], dtype=float),
        "t_adam": int(data["t_adam"]),
        "best_val": float(data["best_val"]),
        "best_D": np.asarray(data["best_D"], dtype=float),
        "best_epoch": int(data["best_epoch"]),
        "train_mses": list(np.asarray(data["train_mses"], dtype=float)),
        "val_mses": list(np.asarray(data["val_mses"], dtype=float)),
        "total_pushes": int(data["total_pushes"]),
    }
