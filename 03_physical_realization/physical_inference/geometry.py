from itertools import combinations
import numpy as np

D = 5
EXTRA_OFFSETS = ((2, 1), (1, 2), (-2, 1), (-1, 2))


def build_lattice(nx=4, ny=4, extra_offsets=EXTRA_OFFSETS):
    unit = [(0, 0), (1, 0), (0, 1), (1, 1)]
    raw = []
    for iy in range(ny):
        for ix in range(nx):
            for x, y in unit:
                raw.append((x + ix, y + iy))
    raw = np.asarray(raw, dtype=float)
    q = np.round(raw / 1e-8).astype(np.int64)

    k2i, nodes_list, r2u = {}, [], []
    for i, qq in enumerate(map(tuple, q)):
        if qq not in k2i:
            k2i[qq] = len(nodes_list)
            nodes_list.append(raw[i])
        r2u.append(k2i[qq])
    nodes = np.asarray(nodes_list, dtype=float)

    es = set()
    cell_edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    for iy in range(ny):
        for ix in range(nx):
            base = (iy * nx + ix) * 4
            for ia, jb in cell_edges:
                gi, gj = r2u[base + ia], r2u[base + jb]
                if gi != gj:
                    es.add((min(gi, gj), max(gi, gj)))

    top = sorted(
        [i for i in range(len(nodes)) if abs(nodes[i, 1] - float(ny)) < 1e-8],
        key=lambda i: nodes[i, 0],
    )
    bot = sorted(
        [i for i in range(len(nodes)) if abs(nodes[i, 1]) < 1e-8],
        key=lambda i: nodes[i, 0],
    )
    for row in (top, bot):
        for i, j in combinations(row, 2):
            es.add((min(i, j), max(i, j)))

    coord_to_idx = {
        (int(round(x)), int(round(y))): i for i, (x, y) in enumerate(nodes)
    }
    for i, n in enumerate(nodes):
        x0, y0 = int(round(n[0])), int(round(n[1]))
        for dx, dy in extra_offsets:
            target = (x0 + dx, y0 + dy)
            if target in coord_to_idx:
                j = coord_to_idx[target]
                if i != j:
                    es.add((min(i, j), max(i, j)))

    edges = sorted(es)
    fixed = [2 * bot[0], 2 * bot[0] + 1]
    for n in bot[1:]:
        fixed.append(2 * n + 1)
    fixed = sorted(fixed)
    free = [i for i in range(2 * len(nodes)) if i not in set(fixed)]
    out_dofs = [2 * i for i in top]
    return nodes, edges, fixed, free, out_dofs, top, bot


def build_B(nodes, edges, free):
    nf = len(free)
    B = np.zeros((nf, len(edges)), dtype=float)
    free_set = set(free)
    free_map = {dof: i for i, dof in enumerate(free)}
    for e, (i, j) in enumerate(edges):
        v = nodes[j] - nodes[i]
        L = np.linalg.norm(v) + 1e-30
        tx, ty = v[0] / L, v[1] / L
        for dof, c in (
            (2 * i, tx / L),
            (2 * i + 1, ty / L),
            (2 * j, -tx / L),
            (2 * j + 1, -ty / L),
        ):
            if dof in free_set:
                B[free_map[dof], e] = c
    return B


NODES, EDGES, FIXED, FREE, OUT_DOFS, OUT_NODES, BOT_NODES = build_lattice(D - 1, 4)
B_STRAIN = build_B(NODES, EDGES, FREE)
FREE_MAP = {dof: i for i, dof in enumerate(FREE)}
OFI = [FREE_MAP[dof] for dof in OUT_DOFS]
INT_IDX = [i for i in range(len(FREE)) if i not in set(OFI)]
