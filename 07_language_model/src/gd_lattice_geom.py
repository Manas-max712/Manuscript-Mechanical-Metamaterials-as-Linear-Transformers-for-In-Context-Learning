import numpy as np


def _parse_long_range(long_range):
    if long_range is None:
        return ()
    if isinstance(long_range, str):
        if not long_range.strip():
            return ()
        return tuple(int(x) for x in long_range.split(",") if x.strip())
    return tuple(int(x) for x in long_range)


def _add_edge(edges, i, j):
    if i == j:
        return
    a, b = (i, j) if i < j else (j, i)
    edges.add((a, b))


def _fully_connect(edges, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            _add_edge(edges, row[i], row[j])


def _add_long_range_cross_bars(edges, top, bot, distances):
    n = min(len(top), len(bot))
    for d in distances:
        if d <= 0 or d >= n:
            continue
        for i in range(n - d):
            _add_edge(edges, top[i], bot[i + d])
            _add_edge(edges, bot[i], top[i + d])


def build_lattice(nx=128, ny=2, connectivity="local", long_range=None):
    if connectivity not in ("local", "row_dense", "row_dense_longrange",
                            "local_longrange"):
        raise ValueError(f"Unknown lattice connectivity: {connectivity}")
    cc = np.array([(0, 0), (1, 0), (0, 1), (1, 1)], dtype=float)
    nc = 4
    raw = []
    for iy in range(ny):
        for ix in range(nx):
            for x, y in cc:
                raw.append((x + ix, y + iy))
    raw = np.array(raw)
    q = np.round(raw / 1e-8).astype(np.int64)
    key_to_idx, uniq, raw_to_uniq = {}, [], []
    for i, qq in enumerate(map(tuple, q)):
        if qq not in key_to_idx:
            key_to_idx[qq] = len(uniq)
            uniq.append(raw[i])
        raw_to_uniq.append(key_to_idx[qq])
    nodes = np.array(uniq)

    ce = [(i, j) for i in range(nc) for j in range(i + 1, nc)]
    edges = set()
    for iy in range(ny):
        for ix in range(nx):
            base = (iy * nx + ix) * nc
            for ia, jb in ce:
                gi = raw_to_uniq[base + ia]
                gj = raw_to_uniq[base + jb]
                _add_edge(edges, gi, gj)
    nnodes = len(nodes)
    y_top = float(ny)
    top = sorted([i for i in range(nnodes) if abs(nodes[i, 1] - y_top) < 1e-8],
                 key=lambda i: nodes[i, 0])
    bot = sorted([i for i in range(nnodes) if abs(nodes[i, 1]) < 1e-8],
                 key=lambda i: nodes[i, 0])
    for arr in [top, bot]:
        for i in range(len(arr) - 1):
            _add_edge(edges, arr[i], arr[i + 1])
    if connectivity in ("row_dense", "row_dense_longrange"):
        _fully_connect(edges, top)
        _fully_connect(edges, bot)
    if connectivity in ("row_dense_longrange", "local_longrange"):
        _add_long_range_cross_bars(edges, top, bot, _parse_long_range(long_range))
    edges = sorted(edges)
    fixed = [2 * bot[0], 2 * bot[0] + 1] + [2 * n + 1 for n in bot[1:]]
    free = [i for i in range(2 * nnodes) if i not in set(fixed)]
    out_dofs = [2 * i for i in top]
    return nodes, edges, free, out_dofs


def _cell_fully_connect(edges, idx, ix, iy, W):
    corners = [idx(ix, iy), idx(ix + 1, iy), idx(ix, iy + 1), idx(ix + 1, iy + 1)]
    for a in range(4):
        for b in range(a + 1, 4):
            _add_edge(edges, corners[a], corners[b])


def build_grid_lattice(grid_width=12, n_ports=65, long_range=None):
    W = int(grid_width)
    if W < 2:
        raise ValueError("grid_width must be >= 2")
    port_rows = -(-int(n_ports) // W)
    H = 2 * port_rows
    nodes = np.array([(x, y) for y in range(H) for x in range(W)], dtype=float)

    def idx(x, y):
        return y * W + x

    port_row_ys = list(range(1, H, 2))
    port_nodes = []
    for y in sorted(port_row_ys, reverse=True):
        for x in range(W):
            port_nodes.append(idx(x, y))
    port_nodes = port_nodes[:int(n_ports)]

    edges = set()
    for iy in range(H - 1):
        for ix in range(W - 1):
            _cell_fully_connect(edges, idx, ix, iy, W)
    for d in _parse_long_range(long_range):
        if d <= 0:
            continue
        for i in range(len(port_nodes) - d):
            _add_edge(edges, port_nodes[i], port_nodes[i + d])
    edges = sorted(edges)

    bottom = [idx(x, 0) for x in range(W)]
    fixed = [2 * bottom[0], 2 * bottom[0] + 1] + [2 * n + 1 for n in bottom[1:]]
    free = [i for i in range(2 * len(nodes)) if i not in set(fixed)]
    out_dofs = [2 * n for n in port_nodes]
    return nodes, edges, free, out_dofs


def build_B(nodes, edges, free):
    nf, m = len(free), len(edges)
    free_set = set(free)
    fmap = {dof: i for i, dof in enumerate(free)}
    B = np.zeros((nf, m), dtype=np.float32)
    for e, (i, j) in enumerate(edges):
        v = nodes[j] - nodes[i]
        L = np.linalg.norm(v) + 1e-30
        tx, ty = v[0] / L, v[1] / L
        for dg, c in [(2 * i, tx / L), (2 * i + 1, ty / L),
                      (2 * j, -tx / L), (2 * j + 1, -ty / L)]:
            if dg in free_set:
                B[fmap[dg], e] = c
    return B
