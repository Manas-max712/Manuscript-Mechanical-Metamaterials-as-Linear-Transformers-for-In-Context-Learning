#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-dir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--newton-steps', type=int, choices=(1, 2), required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stories', type=int, default=8)
    p.add_argument('--require-cuda', action='store_true')
    a = p.parse_args()
    if a.stories < 1:
        p.error('stories must be positive')
    package = a.package_dir.resolve()
    sys.path.insert(0, str(package / 'src'))
    import torch
    from gd_data import TokenDataset, load_manifest
    from common import make_model, atomic_json, sha256, SCHEMA, code_hashes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if a.require_cuda and device.type != 'cuda':
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(8)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    checkpoint, data, output = [(package / path).resolve() for path in (a.checkpoint, a.data, a.output)]
    if output.exists():
        raise FileExistsError(output)
    if output == package / 'src' or package / 'src' in output.parents:
        raise ValueError('Do not write inside src')
    metadata = load_manifest(data)
    ckpt = torch.load(checkpoint, weights_only=False, map_location='cpu')
    if ckpt.get('schema') != SCHEMA or ckpt['dataset_id'] != metadata['dataset_id']:
        raise RuntimeError('Checkpoint/data mismatch')
    cfg = dict(ckpt['model_config'], geo_newton_steps=a.newton_steps)
    model = make_model(cfg, device).eval()
    model.load_state_dict(ckpt['model'], strict=True)
    dataset = TokenDataset(data, 'valid')
    records = []
    sample_index = [-1]
    stages = []
    original_assemble = model._assemble_geo_batched
    original_rows = model._geo_batched_rows

    def observe_assemble(*args, **kwargs):
        force, tangent = original_assemble(*args, **kwargs)
        residual = torch.linalg.vector_norm(force[:, model.int_].detach().double(), dim=-1).cpu()
        if not bool(torch.isfinite(residual).all()):
            raise FloatingPointError('Nonfinite interior force residual')
        stages.append(residual)
        return force, tangent

    def observe_rows(*args, **kwargs):
        stages.clear()
        answer = original_rows(*args, **kwargs)
        if len(stages) != a.newton_steps + 1:
            raise RuntimeError('Unexpected assembly sequence; residual attribution is invalid')
        records.append(dict(story_index=sample_index[0],
                            layer_call=sum(r['story_index'] == sample_index[0] for r in records),
                            force_norms_by_update=[x.tolist() for x in stages]))
        return answer

    model._assemble_geo_batched = observe_assemble
    model._geo_batched_rows = observe_rows
    try:
        with torch.no_grad():
            for i in range(min(a.stories, len(dataset))):
                sample_index[0] = i
                ids, length = dataset[i]
                logits = model(ids[:length-1][None].to(device))
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError('Nonfinite diagnostic logits')
    finally:
        model._assemble_geo_batched = original_assemble
        model._geo_batched_rows = original_rows
    if not records:
        raise RuntimeError('No mechanical cases observed')
    initial = torch.tensor([x for r in records for x in r['force_norms_by_update'][0]], dtype=torch.float64)
    final = torch.tensor([x for r in records for x in r['force_norms_by_update'][-1]], dtype=torch.float64)
    nonzero = initial > 1e-12
    ratios = final[nonzero] / initial[nonzero]
    def stats(values):
        return dict(mean=float(values.mean()), median=float(values.median()),
                    p95=float(torch.quantile(values, .95)), maximum=float(values.max())) if values.numel() else None
    report = dict(checkpoint_sha256=sha256(checkpoint), checkpoint=str(checkpoint),
                  epoch=ckpt['epoch'], global_step=ckpt['global_step'],
                  dataset_id=metadata['dataset_id'], newton_steps=a.newton_steps,
                  stories=min(a.stories, len(dataset)), cases=len(initial),
                  initial_force_norm=stats(initial), final_force_norm=stats(final),
                  final_over_initial=stats(ratios), excluded_near_zero_initial=int((~nonzero).sum()),
                  aggregate_final_over_initial=float(final.norm()/initial.norm()) if initial.norm() > 0 else None,
                  model_config=cfg, source_sha256=code_hashes(), records=records,
                  note='Interior force Euclidean norms in model units. Ratio excludes initial norms <=1e-12. '
                       'Sampled validation prefixes, not a convergence or causality certification.')
    atomic_json(output, report)
    print(json.dumps({k: report[k] for k in ['cases','newton_steps','final_over_initial','aggregate_final_over_initial']}, indent=2))


if __name__ == '__main__':
    main()
