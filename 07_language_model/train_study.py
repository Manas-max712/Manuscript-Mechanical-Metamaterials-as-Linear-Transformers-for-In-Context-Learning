#!/usr/bin/env python3
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-dir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--initialization', type=Path, default=Path('runs/newton_study/initialization'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--newton-steps', type=int, choices=(1, 2), required=True)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=15, help='Total epochs from random initialization')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--validate-every', type=int, default=2500)
    p.add_argument('--checkpoint-every', type=int, default=250)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--cpu-threads', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-hours', type=float, default=10, help='Limit for this invocation; resubmission resumes')
    p.add_argument('--stop-after-steps', type=int, default=0, help='Integration-test stop; continuation steps')
    p.add_argument('--require-cuda', action='store_true')
    a = p.parse_args()
    if min(a.lr, a.epochs, a.batch_size, a.validate_every, a.checkpoint_every,
           a.log_every, a.cpu_threads, a.max_hours) <= 0:
        p.error('Learning rate and count/time limits must be positive')
    package = a.package_dir.resolve()
    sys.path.insert(0, str(package / 'src'))
    import torch
    from torch.utils.data import DataLoader
    from gd_data import TokenDataset, EpochBatches, collate_tokens, input_targets, load_manifest
    from gd_eval import evaluate, loss_sum
    from common import (SCHEMA, make_model, sha256, code_hashes, object_hash,
                                 atomic_json, atomic_save, seed_all, rng_state, restore_rng)

    def absolute(path):
        return (path if path.is_absolute() else package / path).resolve()

    source, output, data = map(absolute, (a.initialization, a.output, a.data))
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Training output must be separate from the initialization directory')
    if output == package / 'src' or package / 'src' in output.parents:
        raise ValueError('Do not write inside src')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.training.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if a.require_cuda and device.type != 'cuda':
            raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(a.cpu_threads)
        if device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        seed_all(a.seed)
        metadata = load_manifest(data)
        source_path = source / 'checkpoint.pt'
        initial = torch.load(source_path, map_location='cpu', weights_only=False)
        if initial.get('schema') != SCHEMA or initial.get('dataset_id') != metadata['dataset_id']:
            raise RuntimeError('Random initialization and dataset do not match')
        for name, value in initial['model'].items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise FloatingPointError(f'Nonfinite initial model tensor: {name}')
        source_config = json.loads((source / 'run_config.json').read_text())
        if source_config['source_code'] != code_hashes():
            raise RuntimeError('Model/data sources differ from the shared random initialization')
        data_seed = source_config.get('data_seed', 0)
        weight_decay = source_config.get('weight_decay', .01)
        if initial['global_step'] != 0 or initial['epoch'] != 0 or initial['seed'] != a.seed:
            raise RuntimeError('Study requires matching seed and a fresh, zero-step initialization')
        cfg = dict(initial['model_config'], geo_newton_steps=a.newton_steps)
        identity = dict(kind='fresh_matched_newton_study_v1', initialization_sha256=sha256(source_path),
                        dataset_id=metadata['dataset_id'], model=cfg, lr=a.lr,
                        weight_decay=weight_decay, batch_size=a.batch_size, epochs=a.epochs,
                        seed=a.seed, data_seed=data_seed, validate_every=a.validate_every,
                        source_code=code_hashes(), trainer_sha256=sha256(__file__))
        fingerprint = object_hash(identity)
        config_file = output / 'run_config.json'
        if config_file.exists():
            if json.loads(config_file.read_text())['fingerprint'] != fingerprint:
                raise RuntimeError('Training configuration changed; use a new output directory')
        else:
            if any(x.name != '.training.lock' for x in output.iterdir()):
                raise RuntimeError('Refusing to use a nonempty output directory without matching config')
            atomic_json(config_file, dict(**identity, fingerprint=fingerprint,
                        initialization_checkpoint=str(source_path), python=sys.version, torch=torch.__version__,
                        device=str(device), fresh_optimizer=True,
                        gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None))
        train, valid = TokenDataset(data, 'train'), TokenDataset(data, 'valid')
        model = make_model(cfg, device)
        model.load_state_dict(initial['model'], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=weight_decay)
        checkpoints = output / 'checkpoints'
        checkpoints.mkdir(exist_ok=True)
        last = checkpoints / 'latest.pt'
        best = checkpoints / 'best.pt'
        origin_epoch, origin_step = 0, 0
        restore_rng(initial['rng'])
        epoch, next_batch, step = origin_epoch + 1, 0, 0
        train_sum, train_tokens, elapsed_before = 0., 0, 0.
        history = []
        training_seconds = 0.
        benchmark_seconds = []
        best_val = float('inf')
        baseline_ce = None
        if last.exists():
            state = torch.load(last, map_location='cpu', weights_only=False)
            if state['fingerprint'] != fingerprint:
                raise RuntimeError('Resume checkpoint fingerprint mismatch')
            model.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            epoch, next_batch, step = state['epoch'], state['next_batch'], state['training_step']
            train_sum, train_tokens = state['train_sum'], state['train_tokens']
            best_val, baseline_ce, history = state['best_val'], state['baseline_ce'], state['history']
            elapsed_before = state['elapsed_seconds']
            training_seconds = state['training_seconds']
            restore_rng(state['rng'])
            print(f'Resumed training step {step}, epoch {epoch}, next batch {next_batch}', flush=True)
        start = time.monotonic()
        stop = [False]
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, lambda *_: stop.__setitem__(0, True))

        def payload():
            return dict(schema=SCHEMA, fingerprint=fingerprint, model=model.state_dict(),
                        model_config=cfg, dataset_id=metadata['dataset_id'],
                        epoch=epoch, next_batch=next_batch, global_step=origin_step + step,
                        training_step=step, optimizer=optimizer.state_dict(), rng=rng_state(),
                        train_sum=train_sum, train_tokens=train_tokens, best_val=best_val,
                        baseline_ce=baseline_ce, history=history, training_seconds=training_seconds,
                        elapsed_seconds=elapsed_before + time.monotonic() - start)

        def save():
            atomic_save(last, payload())

        def write_history():
            if history:
                temporary = output / 'metrics.csv.tmp'
                with temporary.open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(history[0]))
                    w.writeheader(); w.writerows(history)
                temporary.replace(output / 'metrics.csv')

        def validate(kind):
            nonlocal best_val, baseline_ce
            was_training = model.training
            metrics = evaluate(model, valid, a.batch_size, device)
            if kind == 'baseline':
                baseline_ce = metrics['ce']
            if metrics['ce'] < best_val:
                best_val = metrics['ce']
                atomic_save(best, dict(schema=SCHEMA, fingerprint=fingerprint,
                    model=model.state_dict(), model_config=cfg, dataset_id=metadata['dataset_id'],
                    epoch=origin_epoch if step == 0 else epoch, global_step=origin_step + step,
                    training_step=step, val_ce=best_val, initialization_sha256=identity['initialization_sha256']))
            history.append(dict(kind=kind, epoch=origin_epoch if step == 0 else epoch,
                                next_batch=next_batch, training_step=step,
                                global_step=origin_step + step,
                                train_ce=train_sum / train_tokens if train_tokens else None,
                                val_ce=metrics['ce'], val_ppl=metrics['perplexity'],
                                elapsed_seconds=elapsed_before + time.monotonic() - start))
            atomic_json(output / 'validation_latest.json', metrics)
            write_history()
            model.train(was_training)
            print(f'VALIDATION {kind} step={step} CE={metrics["ce"]:.6f} best={best_val:.6f}', flush=True)

        def failure_diagnostics(exc, ids, lengths, batch_rng):
            details = []
            nonfinite_parameters = []
            for name, parameter in model.named_parameters():
                if not bool(torch.isfinite(parameter).all()):
                    nonfinite_parameters.append(name)
                if parameter.grad is None:
                    continue
                g = parameter.grad.detach()
                finite = torch.isfinite(g)
                all_finite = bool(finite.all())
                details.append(dict(parameter=name, shape=list(g.shape),
                                    nonfinite_elements=int((~finite).sum()),
                                    norm_float64=float(torch.linalg.vector_norm(g.double())) if all_finite else None,
                                    max_abs=float(g.abs().max()) if all_finite else None))
            bad_gradients = any(r['nonfinite_elements'] for r in details)
            total_norm64 = math.hypot(*(r['norm_float64'] for r in details)) if not bad_gradients else None
            report = dict(error=repr(exc), epoch=epoch, next_batch=next_batch,
                          training_step=step, global_step=origin_step + step,
                          nonfinite_parameters=nonfinite_parameters,
                          nonfinite_gradient_elements=bad_gradients, gradients=details,
                          total_gradient_norm_float64=total_norm64,
                          note='Failure replay contains the failing state and batch; use latest.pt for continuation.')
            atomic_json(output / 'failure.json', report)
            state = payload()
            state.update(failing_input_ids=None if ids is None else ids.cpu(),
                         failing_lengths=None if lengths is None else lengths.cpu(),
                         rng_before_batch=batch_rng, error=repr(exc))
            atomic_save(output / 'failure_replay.pt', state)

        ids = lengths = batch_rng = None
        try:
            if baseline_ce is None:
                validate('baseline')
                save()
            write_history()
            while epoch <= origin_epoch + a.epochs:
                model.train()
                sampler = EpochBatches(len(train), a.batch_size, data_seed, epoch, next_batch)
                loader = DataLoader(train, batch_sampler=sampler, collate_fn=collate_tokens,
                                    num_workers=0, pin_memory=device.type == 'cuda',
                                    generator=torch.Generator().manual_seed(data_seed + epoch))
                for batch_index, (ids, lengths) in enumerate(loader, start=next_batch):
                    tick = time.monotonic()
                    batch_rng = rng_state()
                    ids, lengths = ids.to(device), lengths.to(device)
                    x, y = input_targets(ids, lengths)
                    optimizer.zero_grad(set_to_none=True)
                    summed, count = loss_sum(model(x), y)
                    loss = summed / count
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite training loss')
                    loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    train_sum += float(summed.detach()); train_tokens += count
                    step += 1; next_batch = batch_index + 1
                    duration = time.monotonic() - tick
                    training_seconds += duration
                    benchmark_seconds.append(duration)
                    if len(benchmark_seconds) == 105:
                        seconds_per_step = sum(benchmark_seconds[5:]) / 100
                        timing = dict(newton_steps=a.newton_steps, seconds_per_step=seconds_per_step,
                                      measured_steps=100, warmup_steps=5,
                                      estimated_remaining_training_hours=seconds_per_step * (
                                          a.epochs * math.ceil(len(train)/a.batch_size) - step) / 3600,
                                      excludes='validation, diagnostics, data loading, checkpoint I/O, queueing, preemption')
                        atomic_json(output / 'throughput.json', timing)
                        print('THROUGHPUT ' + json.dumps(timing), flush=True)
                    if step % a.log_every == 0:
                        print(f'epoch={epoch} batch={next_batch}/{math.ceil(len(train)/a.batch_size)} '
                              f'training_step={step} CE={train_sum/train_tokens:.5f} '
                              f'grad_norm={float(grad):.5g} lr={a.lr}', flush=True)
                    end_epoch = next_batch == math.ceil(len(train) / a.batch_size)
                    if step % a.validate_every == 0 or end_epoch:
                        validate('epoch_end' if end_epoch else 'periodic')
                        save()
                    elif step % a.checkpoint_every == 0:
                        save()
                    if stop[0] or time.monotonic() - start >= a.max_hours * 3600 or (
                            a.stop_after_steps and step >= a.stop_after_steps):
                        save()
                        print('Paused with resume state saved. Submit the same command to continue.', flush=True)
                        return
                epoch += 1; next_batch = 0; train_sum = 0.; train_tokens = 0
                save()
            atomic_json(output / 'summary.json', dict(training_complete=True,
                baseline_val_ce=baseline_ce, best_val_ce=best_val,
                improved=best_val < baseline_ce, training_steps=step,
                initialization_checkpoint=str(source_path), fresh_optimizer=True,
                newton_steps=a.newton_steps, training_seconds=training_seconds,
                mean_training_seconds_per_step=training_seconds / step if step else None,
                peak_allocated_GB=torch.cuda.max_memory_allocated()/1e9 if device.type == 'cuda' else None,
                causality=dict(status='not_run', passed=None),
                note='Selected on validation only. No test evaluation or causality diagnostic was run.'))
            print(f'COMPLETE baseline CE={baseline_ce:.6f}; best CE={best_val:.6f}', flush=True)
        except Exception as exc:
            failure_diagnostics(exc, ids, lengths, batch_rng)
            raise


if __name__ == '__main__':
    main()
