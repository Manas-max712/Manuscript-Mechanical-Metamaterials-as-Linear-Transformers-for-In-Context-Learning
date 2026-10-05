from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import random
import numpy as np
import torch

SCHEMA = "causal_language_revision_v1"


def model_config(mode="causal_prefix", **changes):
    cfg = dict(d=512, n_ctx=64, nx=64, ny=2, n_layer=2, n_head=4,
               gd_step=1.0, use_ff=False, dropout=.1, ridge=.05,
               ridge_mode="relative", per_layer_kernel=False,
               kernel_type="lattice_geo_prestress", normalize_kernel=mode != "none",
               kernel_norm_mode=mode, kernel_norm_eps=1e-6,
               learn_dof_gains=False, learn_node_coords=False,
               min_bar_length=1e-3, attn_fn="softmax", n_phys=2,
               hard_relu=False, geo_newton_steps=1, geo_disp_scale=.12,
               geo_delta_scale=.10, lattice_connectivity="grid_interleaved",
               grid_width=16, lattice_long_range="1,2,3,4,8,16,32", vocab_limit=50258)
    cfg.update(changes)
    return cfg


class VocabSize:
    def __init__(self, n): self.n = n
    def __len__(self): return self.n


def make_model(cfg, device, verbose=False):
    from gd_build import build_model
    return build_model(SimpleNamespace(**cfg),
                       SimpleNamespace(tokenizer=VocabSize(cfg["vocab_limit"])),
                       device, verbose=verbose)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def code_hashes():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary); os.replace(temporary, path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
