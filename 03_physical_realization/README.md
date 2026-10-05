# Physical realization

Code for Figure 3, Figure S4, and Tables S2–S3.

| File | Purpose |
|---|---|
| `main.py` | Fit lattices to the learned operators, evaluate inference, and export figures and tables. |
| `physical_inference/inputs.py` | Load and validate operators and covariances from Folder 2. |
| `physical_inference/geometry.py` | Lattice geometry, supports, and strain compatibility matrix. |
| `physical_inference/fitting.py` | Optimize member rigidities to fit the target stiffness. |
| `physical_inference/pruning.py` | Delete weak members and calculate the pruned compliance. |
| `physical_inference/inference.py` | Generate regression tasks and evaluate adaptive, power-iteration, and fixed gains. |
| `physical_inference/experiments.py` | Connect fitting, pruning, and inference experiments. |
| `physical_inference/plots.py`, `outputs.py` | Export figures, Tables S2–S3, fitted parameters, and numerical curves. |

## Environment

```bash
conda env create -f environment.yml
conda activate physical-realization
```

Alternatively, use Python 3.13.7 and `python -m pip install -r requirements.txt`.
The computations use the CPU.

## Run

First complete the all-case operator training in `02_regression_operators`, then:

```bash
python main.py --operators ../02_regression_operators/runs/paper/results.npz --output-dir runs/paper
```

The target compliance is the symmetric part of the learned operator. Fitting
minimizes stiffness error using ten initializations. Figure 3 uses the lattice
after deleting members with rigidity at or below 0.01. Table S2 reports
relative stiffness and compliance errors before and after pruning.

Figure 3 and Figure S4 use 300 noiseless test tasks per family. Fixed gains
are calibrated on 1000 separate tasks. Figure S4 and Table S3 use unpruned
lattices. Inference error is relative squared coefficient error; equilibrium
responses are computed numerically.

Outputs are saved under `figures/` and `tables/` within the run directory.
To regenerate the results without repeating optimization, supply a completed fit:

```bash
python main.py --operators ../02_regression_operators/runs/paper/results.npz --fit-cache runs/paper/physical_lattice_fit_data.npz --output-dir runs/evaluation
```
