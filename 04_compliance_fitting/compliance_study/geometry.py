from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EXTRA_OFFSETS = [(2, 1), (1, 2), (-2, 1), (-1, 2)]


@dataclass
class Lattice:
    d: int
    nx: int
    ny: int
    nodes: np.ndarray
    edges: list[tuple[int, int]]
    fixed: list[int]
    free: list[int]
    out_dofs: list[int]
    top_nodes: list[int]
    bot_nodes: list[int]
    B: np.ndarray
    ofi: list[int]
    int_idx: list[int]

    @property
    def m(self) -> int:
        return len(self.edges)

    @property
    def nf(self) -> int:
        return len(self.free)


def build_lattice(nx: int, ny: int = 4, extra_offsets=EXTRA_OFFSETS):
    unit = [(0, 0), (1, 0), (0, 1), (1, 1)]
    raw = []
    for iy in range(ny):
        for ix in range(nx):
            for x, y in unit:
                raw.append((x + ix, y + iy))
    raw = np.array(raw, dtype=float)
    q = np.round(raw / 1e-8).astype(np.int64)
    k2i = {}
    un = []
    r2u = []
    for i, qq in enumerate(map(tuple, q)):
        if qq not in k2i:
            k2i[qq] = len(un)
            un.append(raw[i])
        r2u.append(k2i[qq])
    nodes = np.array(un)
    ce = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    es = set()
    for iy in range(ny):
        for ix in range(nx):
            base = (iy * nx + ix) * 4
            for ia, jb in ce:
                gi, gj = r2u[base + ia], r2u[base + jb]
                if gi != gj:
                    es.add((min(gi, gj), max(gi, gj)))
    yt = float(ny)
    top = sorted([i for i, n in enumerate(nodes) if abs(n[1] - yt) < 1e-8],
                 key=lambda i: nodes[i, 0])
    bot = sorted([i for i, n in enumerate(nodes) if abs(n[1]) < 1e-8],
                 key=lambda i: nodes[i, 0])
    for boundary in (top, bot):
        for i in range(len(boundary)):
            for j in range(i + 1, len(boundary)):
                a, b = boundary[i], boundary[j]
                es.add((min(a, b), max(a, b)))
    coord_to_idx = {(int(round(n[0])), int(round(n[1]))): i
                    for i, n in enumerate(nodes)}
    for i, n in enumerate(nodes):
        ix0, iy0 = int(round(n[0])), int(round(n[1]))
        for dx, dy in extra_offsets:
            target = (ix0 + dx, iy0 + dy)
            if target in coord_to_idx:
                j = coord_to_idx[target]
                if i != j:
                    es.add((min(i, j), max(i, j)))
    fixed = [2 * bot[0], 2 * bot[0] + 1]
    for n in bot[1:]:
        fixed.append(2 * n + 1)
    fixed = sorted(fixed)
    free = [i for i in range(2 * len(nodes)) if i not in set(fixed)]
    return nodes, sorted(es), fixed, free, [2 * i for i in top], top, bot


def build_B(nodes: np.ndarray, edges: list[tuple[int, int]], free: list[int]):
    fs = set(free)
    fm = {dof: i for i, dof in enumerate(free)}
    B = np.zeros((len(free), len(edges)), dtype=np.float64)
    for e, (i, j) in enumerate(edges):
        v = nodes[j] - nodes[i]
        L = np.linalg.norm(v) + 1e-30
        tx, ty = v[0] / L, v[1] / L
        for dg, c in [(2 * i, tx / L), (2 * i + 1, ty / L),
                      (2 * j, -tx / L), (2 * j + 1, -ty / L)]:
            if dg in fs:
                B[fm[dg], e] = c
    return B


def make_lattice(d: int, ny: int = 4, extra_offsets=EXTRA_OFFSETS) -> Lattice:
    nx = d - 1
    nodes, edges, fixed, free, out_dofs, top, bot = build_lattice(
        nx, ny, extra_offsets)
    B = build_B(nodes, edges, free)
    fm = {dof: i for i, dof in enumerate(free)}
    ofi = [fm[dof] for dof in out_dofs]
    int_idx = [i for i in range(len(free)) if i not in set(ofi)]
    return Lattice(d, nx, ny, nodes, edges, fixed, free, out_dofs, top, bot,
                   B, ofi, int_idx)
