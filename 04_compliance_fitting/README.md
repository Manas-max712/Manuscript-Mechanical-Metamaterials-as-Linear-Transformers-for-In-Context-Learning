# Compliance fitting

Code for Tables S4–S6. These experiments construct ideal covariance-derived targets and fit lattice compliance directly; no trained operator or dataset is required.

| File | Purpose |
|---|---|
| `main.py` | Run the studies and export tables. |
| `compliance_study/geometry.py`, `topologies.py` | Construct lattice nodes, bars, supports, and connectivity presets. |
| `compliance_study/targets.py` | Construct ideal target operators. |
| `compliance_study/mechanics.py` | Compute static condensation, compliance, and analytic gradients. |
| `compliance_study/multistart.py` | Generate initializations and optimize member rigidities. |
| `compliance_study/experiments.py` | Define dimension, connectivity, and initialization studies. |
| `compliance_study/tables.py` | Export Tables S4–S6. |

## Environment and run

```bash
conda env create -f environment.yml
conda activate compliance-fitting
python main.py --output-dir runs/paper
```

Alternatively, use Python 3.11.9 and `python -m pip install -r requirements.txt`. Computations use the CPU.

Run an individual table with `--study s4`, `--study s5`, or `--study s6`. Use `--smoke` for a short workflow check.

- **Table S4:** Dimensions \(d=5,10,15,20\), with three covariance families and 50 initializations per case.
- **Table S5:** Rotated target at \(d=10\), with 50 nodes and 229–1225 bars. Each topology uses 50 initializations; an additional complete-graph result uses 150 initializations.
- **Table S6:** Initialization sensitivity for baseline lattices at \(d=5\) and \(d=10\), using 50 starts for all families and 150 starts for the rotated cases. The fully connected \(d=10\) lattice uses 50, 100, 150, and 200 starts.

All groups use seed 0 and height 4. The first three initializations are constant log rigidities; later starts are seeded Gaussian draws. Larger initialization budgets retain the same initial sequence. Shared cases are computed once when running all studies.

The ideal compliance is `G_ideal = inverse(Sigma)`. Optimization minimizes squared Frobenius compliance error, with log rigidities bounded between −6 and 6. Reported errors are relative Frobenius norms.

Tables are saved as CSV and LaTeX under `tables/`. The `fits/` directory contains per-start diagnostics and the best fitted rigidities and operators for each initialization effort.
