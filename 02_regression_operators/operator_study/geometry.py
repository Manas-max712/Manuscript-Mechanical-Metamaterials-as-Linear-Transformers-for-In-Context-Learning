import numpy as np

from .config import EXTRA_OFFSETS


def build_lattice(nx=4, ny=4, extra_offsets=EXTRA_OFFSETS):
    unit = [(0, 0), (1, 0), (0, 1), (1, 1)]
    raw = []
    for iy in range(ny):
        for ix in range(nx):
            for x, y in unit:
                raw.append((x + ix, y + iy))
    raw = np.array(raw, dtype=float)
    q = np.round(raw / 1e-08).astype(np.int64)
    k2i = {}
    un = []
    r2u = []
    for i, qq in enumerate(map(tuple, q)):
        if qq not in k2i:
            k2i[qq] = len(un)
            un.append(raw[i])
        r2u.append(k2i[qq])
    nodes = np.array(un)
    Nc = 4
    ce = [(i, j) for i in range(Nc) for j in range(i + 1, Nc)]
    es = set()
    for iy in range(ny):
        for ix in range(nx):
            base = (iy * nx + ix) * Nc
            for ia, jb in ce:
                gi, gj = (r2u[base + ia], r2u[base + jb])
                if gi != gj:
                    es.add((min(gi, gj), max(gi, gj)))
    nn_ = len(nodes)
    yt = float(ny)
    top = sorted([i for i in range(nn_) if abs(nodes[i, 1] - yt) < 1e-08], key=lambda i: nodes[i, 0])
    bot = sorted([i for i in range(nn_) if abs(nodes[i, 1]) < 1e-08], key=lambda i: nodes[i, 0])
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = (top[i], top[j])
            es.add((min(a, b), max(a, b)))
    for i in range(len(bot)):
        for j in range(i + 1, len(bot)):
            a, b = (bot[i], bot[j])
            es.add((min(a, b), max(a, b)))
    coord_to_idx = {(int(round(n[0])), int(round(n[1]))): i for i, n in enumerate(nodes)}
    for i, n_ in enumerate(nodes):
        ix0, iy0 = (int(round(n_[0])), int(round(n_[1])))
        for dx, dy in extra_offsets:
            target = (ix0 + dx, iy0 + dy)
            if target in coord_to_idx:
                j = coord_to_idx[target]
                if i != j:
                    es.add((min(i, j), max(i, j)))
    edges = sorted(es)
    fixed = [2 * bot[0], 2 * bot[0] + 1]
    for n in bot[1:]:
        fixed.append(2 * n + 1)
    fixed = sorted(fixed)
    free = [i for i in range(2 * nn_) if i not in set(fixed)]
    return (nodes, edges, fixed, free, [2 * i for i in top], top, bot)


def build_B(nodes, edges, free):
    nf = len(free)
    m = len(edges)
    fs = set(free)
    fm = {dd: i for i, dd in enumerate(free)}
    B = np.zeros((nf, m), dtype=np.float64)
    for e, (i, j) in enumerate(edges):
        v = nodes[j] - nodes[i]
        L = np.linalg.norm(v) + 1e-30
        tx, ty = (v[0] / L, v[1] / L)
        for dg, c in [(2 * i, tx / L), (2 * i + 1, ty / L), (2 * j, -tx / L), (2 * j + 1, -ty / L)]:
            if dg in fs:
                B[fm[dg], e] = c
    return B
