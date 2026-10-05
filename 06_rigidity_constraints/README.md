# Rigidity constraints

Code for Figures S6–S7 and Tables S10 and S12.

| File | Purpose |
|---|---|
| `main.py` | Run bounded-rigidity training and three-class fine-tuning. |
| `constraint_study/config.py`, `tasks.py` | Experiment settings and regression task generation. |
| `constraint_study/geometry.py`, `mechanics.py` | Lattice geometry, equilibrium solves, forward inference, and adjoint gradients. |
| `constraint_study/bounded_training.py` | Train with prescribed rigidity contrasts for Figure S6 and Table S10. |
| `constraint_study/tail_training.py` | Fix low/high-rigidity members and train the middle band for Figure S7 and Table S12. |
| `constraint_study/bounded_checkpoints.py`, `tail_checkpoints.py` | Save and restore training and optimizer states. |
| `constraint_study/evaluation.py` | Evaluate test MSE, operator error, and achieved rigidity ratios. |
| `constraint_study/figure_s6.py`, `figure_s7.py`, `tables.py` | Export figures and tables. |
| `constraint_study/files.py` | Load inputs and save results. |

## Environment and run

```bash
conda env create -f environment.yml
conda activate rigidity-constraints
python main.py --study bounds --output-dir runs/bounds
python main.py --study tails --mall-root ../05_mall/runs/paper --output-dir runs/tails
```

Alternatively, use Python 3.13.7 and `python -m pip install -r requirements.txt`. Computations use the CPU.

Bounded training starts from random rigidities at imposed contrasts 100, 1000, 10000, and 20000. The isotropic, diagonal, and rotated cases use 3000, 2000, and 2000 epochs and 15, 15, and 25 stages, respectively.

Three-class fine-tuning starts from Folder 5's learned rigidities. The lowest 30% and highest 10% of members are fixed at 0.01 and 200; the remaining members are trained for 300 epochs at middle-band bounds 100 and 1000. Figure S7(c) plots the imposed bound, while diagnostics also report the achieved ratio. `--tail-mask-root` optionally reuses class assignments from supplied constraint checkpoints.

Both studies use 1000 training tasks per epoch, 200 validation tasks, and 1000 independent test tasks (seed 20001234). Test MSE is unnormalized query prediction error. Training holds the task-adaptive gain fixed when computing adjoint gradients.

Use `--case identity`, `--case diag`, or `--case rotated` to select a family. Use `--smoke` for a short workflow check. Repeat an unchanged training command with `--resume` to restore the latest saved epoch and Adam state.

To regenerate figures and tables from completed checkpoints:

```bash
python main.py --study bounds --from-results runs/bounds --output-dir runs/bounds_evaluation
python main.py --study tails --from-results runs/tails --output-dir runs/tails_evaluation
```

Outputs are saved under `checkpoints/`, `figures/`, and `tables/`.
