from .geometry import build_lattice, build_B, Lattice, make_lattice

OFFSET_PRESETS = {
    "none": [],
    "baseline": [
        (2, 1), (1, 2), (-2, 1), (-1, 2),
    ],
    "medium": [
        (2, 1), (1, 2), (-2, 1), (-1, 2),
        (3, 1), (1, 3), (-3, 1), (-1, 3),
        (3, 2), (2, 3), (-3, 2), (-2, 3),
    ],
    "long": [
        (2, 1), (1, 2), (-2, 1), (-1, 2),
        (3, 1), (1, 3), (-3, 1), (-1, 3),
        (3, 2), (2, 3), (-3, 2), (-2, 3),
        (4, 1), (1, 4), (-4, 1), (-1, 4),
        (4, 2), (2, 4), (-4, 2), (-2, 4),
        (4, 3), (3, 4), (-4, 3), (-3, 4),
    ],
    "wide": [
        (2, 1), (1, 2), (-2, 1), (-1, 2),
        (3, 1), (1, 3), (-3, 1), (-1, 3),
        (3, 2), (2, 3), (-3, 2), (-2, 3),
        (4, 1), (1, 4), (-4, 1), (-1, 4),
        (4, 2), (2, 4), (-4, 2), (-2, 4),
        (4, 3), (3, 4), (-4, 3), (-3, 4),
        (5, 1), (1, 5), (-5, 1), (-1, 5),
        (5, 2), (2, 5), (-5, 2), (-2, 5),
        (5, 3), (3, 5), (-5, 3), (-3, 5),
    ],
    "complete": [],
}

def complete_lattice(d, ny):
    nx = d - 1
    nodes, _, fixed, free, out_dofs, top, bot = build_lattice(nx, ny, extra_offsets=[])
    edges = [(i, j) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]
    B = build_B(nodes, edges, free)
    fm = {dof: i for i, dof in enumerate(free)}
    ofi = [fm[dof] for dof in out_dofs]
    int_idx = [i for i in range(len(free)) if i not in set(ofi)]
    return Lattice(d, nx, ny, nodes, edges, fixed, free, out_dofs, top, bot,
                   B, ofi, int_idx)

def lattice(d, preset):
    return complete_lattice(d, 4) if preset == "complete" else make_lattice(d, 4, OFFSET_PRESETS[preset])
