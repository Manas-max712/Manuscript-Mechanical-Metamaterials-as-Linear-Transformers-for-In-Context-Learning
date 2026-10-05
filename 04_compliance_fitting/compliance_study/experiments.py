import csv
import hashlib
import json
import numpy as np
from .topologies import lattice
from .targets import ideal_targets
from .multistart import make_start, run_one_start

FAMILIES = ('identity', 'diag', 'rotated')
TOPOLOGIES = ('none', 'baseline', 'medium', 'long', 'wide', 'complete')


def requests(study):
    result = {}
    def add(d, family, preset, counts):
        result.setdefault((d, family, preset), set()).update(counts)
    if study in ('all', 's4'):
        for d in (5, 10, 15, 20):
            for family in FAMILIES:
                add(d, family, 'baseline', [50])
    if study in ('all', 's5'):
        for preset in TOPOLOGIES:
            add(10, 'rotated', preset, [50, 150] if preset == 'complete' else [50])
    if study in ('all', 's6'):
        for d in (5, 10):
            for family in FAMILIES:
                add(d, family, 'baseline', [50, 150] if family == 'rotated' else [50])
        add(10, 'rotated', 'complete', [50, 100, 150, 200])
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(study, output, smoke=False):
    records = {}
    for (d, family, preset), counts in requests(study).items():
        counts = [2] if smoke else sorted(counts)
        lat = lattice(d, preset)
        sigma, target, _ = ideal_targets(family, d)
        rng = np.random.RandomState(0)
        directory = output / 'fits' / f'd{d}_{family}_{preset}'
        directory.mkdir(parents=True)
        rows, best, best_obj = [], None, float('inf')
        print(f'd={d}, {family}, {preset}: {lat.m} bars, {max(counts)} starts', flush=True)
        for i in range(max(counts)):
            theta = make_start(rng, lat.m, i)
            fit = run_one_start(lat, target, 'G', theta, 6., 5 if smoke else 3000)
            arrays = {key: fit.pop(key) for key in ('D', 'K_fit', 'G_fit')}
            if fit['final_obj'] < best_obj:
                best_obj, best = fit['final_obj'], arrays
            fit.update(start_id=i, initialization_sha256=hashlib.sha256(theta.tobytes()).hexdigest())
            rows.append(fit)
            write_csv(directory/'starts.csv', rows)
            n = i+1
            if n in counts:
                errors = np.array([r['G_rel_err'] for r in rows])
                summary = dict(d=d, family=family, topology=preset, bars=lat.m,
                               initializations=n, best=float(errors.min()),
                               median=float(np.median(errors)), worst=float(errors.max()))
                records[(d,family,preset,n)] = summary
                np.savez_compressed(directory/f'best_{n}.npz', **best, G_ideal=target,
                                    Sigma=sigma, nodes=lat.nodes, edges=lat.edges,
                                    free=lat.free, output_indices=lat.ofi, seed=0)
                (directory/f'summary_{n}.json').write_text(json.dumps(summary,indent=2)+'\n')
                print(f'  {n} starts: best={summary["best"]:.8e}',flush=True)
    return records
