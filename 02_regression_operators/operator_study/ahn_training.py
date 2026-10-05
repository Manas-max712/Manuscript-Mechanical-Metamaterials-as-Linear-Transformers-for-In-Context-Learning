import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from .ahn_protocol import ahn, covariance, clip_function


def settings(case):
    iso = case == 'isotropic'
    return dict(steps=10000 if iso else 100000, batch_size=4000 if iso else 20000,
                lr=.001 if iso else .1, betas=(.9,.9) if iso else (.99,.9),
                clip=1000. if iso else .01)


@torch.no_grad()
def device_clip(grad, radius, isotropic):
    if isotropic:
        norms = grad.norm().reshape(1)
    else:
        norms = torch.stack([g.norm() for g in grad.reshape(-1,5,5)])
    scales = torch.where(norms > radius,
        (radius / norms.double().clamp_min(torch.finfo(torch.float64).tiny)).to(grad.dtype),
        torch.ones_like(norms))
    grad.mul_(scales[0] if isotropic else scales.reshape(*grad.shape[:-2],1,1))


def atomic_save(value, path):
    path = Path(path)
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temp.open('wb') as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def run(case, seed, device, *, steps=None, batch_size=None, checkpoint=None,
        resume=None, identity=None, checkpoint_every=1000, stop_after=None,
        efficient=True, progress=True, benchmark_warmup=0):
    torch.set_default_dtype(torch.float32)
    ahn.device = torch.device(device)
    iso = case == 'isotropic'
    cfg = settings(case)
    total = steps or cfg['steps']
    batch = batch_size or cfg['batch_size']
    torch.manual_seed(seed)
    model = ahn.Transformer_F(3,1,5,.0001).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                                 betas=cfg['betas'], weight_decay=0)
    reference_clip = clip_function(iso)
    losses, snapshots, snapshot_steps = [], [], []
    z = y = None
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Checkpoint settings/source/environment differ; refusing unsafe resume')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        u = None if saved['U'] is None else saved['U'].to(device)
        d = None if saved['D'] is None else saved['D'].to(device)
        if saved['Z'] is not None:
            z, y = saved['Z'].to(device), saved['y'].to(device)
        losses = saved['losses']
        snapshots = list(saved['snapshots'].to(device).unbind(0))
        snapshot_steps = saved['snapshot_steps']
        begin = saved['completed_steps']
        torch.set_rng_state(saved['cpu_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        if torch.device(device).type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        if progress: print(f'Resuming {case} after {begin}/{total} updates',flush=True)
    else:
        begin = 0
        np.random.seed(seed)
        torch.manual_seed(seed)
        u,d = covariance(case,seed,device)
        if not iso:
            z,y = ahn.generate_data('normal',20,5,batch,.1,u,d)
    stop = [False]
    def handler(signum, frame):
        stop[0] = True
    previous = {}
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
        previous[sig] = signal.signal(sig,handler)
    pending = []
    finite = torch.ones((),dtype=torch.bool,device=device)
    completed = begin
    start = time.perf_counter()
    timed_start = None
    timed_updates = 0

    def sync():
        if torch.device(device).type == 'cuda': torch.cuda.synchronize()

    def flush():
        nonlocal pending,finite
        if not bool(finite.item()) or not bool(torch.isfinite(model.allparam).all().item()):
            raise FloatingPointError('Nonfinite training state; last good checkpoint preserved')
        if pending:
            losses.extend(torch.stack(pending).cpu().tolist())
            pending = []
        finite = torch.ones((),dtype=torch.bool,device=device)

    def save():
        if checkpoint is None: return
        flush()
        atomic_save(dict(identity=identity, model=model.state_dict(),
            optimizer=optimizer.state_dict(), U=u,D=d,
            Z=z if not iso else None,y=y if not iso else None,
            losses=losses,snapshots=torch.stack(snapshots).cpu(),snapshot_steps=snapshot_steps,
            completed_steps=completed,cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.device(device).type=='cuda' else [],
            numpy_rng=np.random.get_state(),python_rng=random.getstate()),checkpoint)

    try:
        sync()
        start = time.perf_counter()
        for t in range(begin,total):
            if t-begin == benchmark_warmup:
                sync()
                timed_start = time.perf_counter()
            if not iso and t%4000==0 and t>1:
                optimizer.param_groups[0]['lr'] *= .5
            if not iso and t%100==0:
                z,y = ahn.generate_data_inplace(z,U=u,D=d)
            if t%(1 if iso else 100)==0:
                snapshots.append(model.allparam.detach().clone())
                snapshot_steps.append(t)
            if iso:
                z,y = ahn.generate_data('normal',20,5,batch,.1)
            loss = ahn.in_context_loss(model,z,y)
            loss.backward()
            finite.logical_and_(torch.isfinite(loss.detach()) & torch.isfinite(model.allparam.grad).all())
            if efficient:
                device_clip(model.allparam.grad,cfg['clip'],iso)
                optimizer.step()
            else:
                reference_clip(model.allparam,optimizer,clip_r=cfg['clip'])
            optimizer.zero_grad()
            pending.append(loss.detach())
            completed = t+1
            if timed_start is not None: timed_updates += 1
            boundary = completed%100==0 or completed==total
            if boundary:
                flush()
                if progress:
                    print(f'{case} {completed}/{total} loss={losses[-1]:.8g} '
                          f'lr={optimizer.param_groups[0]["lr"]:.6g} '
                          f'elapsed={time.perf_counter()-start:.1f}s',flush=True)
            if checkpoint is not None and completed%checkpoint_every==0:
                save()
            if stop[0] or (stop_after is not None and completed-begin>=stop_after):
                break
        flush()
        sync()
        timing = dict(measured_updates=timed_updates,
                      seconds_per_update=(time.perf_counter()-timed_start)/timed_updates
                      if timed_start is not None and timed_updates else None,
                      elapsed_seconds=time.perf_counter()-start)
        save()
    finally:
        for sig,old in previous.items():signal.signal(sig,old)
    return dict(model=model,optimizer=optimizer,U=u,D=d,losses=losses,
                snapshots=torch.stack(snapshots).cpu(),snapshot_steps=snapshot_steps,
                completed_steps=completed,complete=completed==total,
                steps=total,batch_size=batch,timing=timing)
