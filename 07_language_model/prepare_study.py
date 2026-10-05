#!/usr/bin/env python3
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-dir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--study', type=Path, default=Path('runs/newton_study'))
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--small-model', action='store_true', help='Integration tests only')
    a = p.parse_args()
    package = a.package_dir.resolve()
    study, data = [(package / path).resolve() for path in (a.study, a.data)]
    if study == package / 'src' or package / 'src' in study.parents:
        raise ValueError('Do not write inside src')
    sys.path.insert(0, str(package / 'src'))
    import torch
    from gd_data import load_manifest
    from common import (SCHEMA, model_config, make_model, seed_all,
                                 rng_state, code_hashes, sha256, atomic_json, atomic_save)
    metadata = load_manifest(data)
    cfg = model_config('causal_prefix', n_ctx=metadata['n_ctx'], nx=metadata['n_ctx'],
                       vocab_limit=metadata['vocab_size'], geo_newton_steps=1)
    if a.small_model:
        cfg.update(d=16, n_head=2, n_layer=2, grid_width=3, lattice_long_range='1,2')
    requested = dict(kind='fresh_random_initialization_v1', seed=a.seed,
                     dataset_id=metadata['dataset_id'], model_config=cfg,
                     source_code=code_hashes(), small_model=a.small_model)
    study.mkdir(parents=True, exist_ok=True)
    with (study / '.initialization.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        destination = study / 'initialization'
        if destination.exists():
            record = json.loads((destination / 'initialization.json').read_text())
            for key, value in requested.items():
                if record[key] != value:
                    raise RuntimeError(f'Existing random initialization differs: {key}')
            if sha256(destination / 'checkpoint.pt') != record['checkpoint_sha256']:
                raise RuntimeError('Shared initialization changed')
            expected_config = dict(requested, data_seed=0, weight_decay=.01)
            if json.loads((destination / 'run_config.json').read_text()) != expected_config:
                raise RuntimeError('Shared initialization configuration changed')
            print(f'Using verified random initialization: {destination}', flush=True)
            return
        seed_all(a.seed)
        torch.set_default_dtype(torch.float32)
        model = make_model(cfg, torch.device('cpu'))
        initial = dict(schema=SCHEMA, dataset_id=metadata['dataset_id'], model_config=cfg,
                       model=model.state_dict(), epoch=0, global_step=0, seed=a.seed,
                       rng=rng_state(), initialization='random; no pretrained checkpoint')
        staging = Path(tempfile.mkdtemp(prefix='initialization_staging_', dir=study))
        try:
            atomic_save(staging / 'checkpoint.pt', initial)
            atomic_json(staging / 'run_config.json', dict(requested, data_seed=0, weight_decay=.01))
            atomic_json(staging / 'initialization.json', dict(requested,
                        checkpoint_sha256=sha256(staging / 'checkpoint.pt')))
            os.rename(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        print(f'Created fresh random initialization: {destination}', flush=True)


if __name__ == '__main__':
    main()
