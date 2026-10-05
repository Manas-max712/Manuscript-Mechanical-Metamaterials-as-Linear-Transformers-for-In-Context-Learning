# MALL

Code for Figures 4–5, Figure S5, Tables S7–S9, and the fabrication-tolerance study in Table S11.

| File | Purpose |
|---|---|
| `main.py` | Train all three task families and export results. |
| `mall_study/config.py`, `tasks.py` | Experiment settings and Gaussian regression tasks. |
| `mall_study/geometry.py`, `mechanics.py` | Lattice geometry, equilibrium solves, and inference gains. |
| `mall_study/learning.py`, `training.py` | Multistage forward inference, adjoint gradients, and Adam training. |
| `mall_study/analysis.py` | Operator alignment and inference beyond training depth. |
| `mall_study/figure4.py`, `figure5.py`, `figure_s5.py` | Export Figures 4, 5, and S5. |
| `mall_study/pruning.py` | Evaluate lattices after deleting weak members. |
| `mall_study/tolerance.py` | Evaluate rigidity perturbations at 15, 50, and 100 stages. |
| `mall_study/results.py`, `tables.py` | Save trained parameters and export tables. |

## Environment and run

```bash
conda env create -f environment.yml
conda activate mall
python main.py --output-dir runs/paper
```

Alternatively, use Python 3.13.7 and `python -m pip install -r requirements.txt`. Training uses the CPU.

Use `--case identity`, `--case diag`, or `--case rotated` for a single family. Use `--smoke` for a short workflow check.

| Family | Epochs | Training stages |
|---|---:|---:|
| Isotropic | 3000 | 15 |
| Diagonal | 2000 | 15 |
| Rotated | 2000 | 25 |

Training uses 1000 tasks per epoch, with 20 context examples and 16 query examples per task. The validation set contains 200 tasks (seed 9999); testing uses 1000 tasks (seed 20001234), each with one query. Training uses the epoch number as its seed. The adjoint holds the task-adaptive gain fixed.

Figure 5(b) reports test query MSE divided by its one-stage value. Table S9 reports unnormalized query MSE. Table S8 fits an alignment scalar to the learned operator; no ideal-target scaling is imposed during MALL training.

Pruning deletes members with rigidity at or below 0.1, using a relative threshold tolerance of 1e-9. The tolerance study uses ten perturbation realizations at each log-rigidity standard deviation, 0.2 and 0.3, evaluated on the same test tasks.

To evaluate an existing completed run:

```bash
python main.py --from-results runs/paper --output-dir runs/evaluation
```

The input directory must contain a `postprocess_data.npz` in each selected case folder. Figures and tables are exported under `figures/` and `tables/`. Select `--study core`, `--study pruning`, or `--study tolerance` to export only that study.
