# Regression operators

Code for Figure 2, Figures S2–S3, and Table S1.

| File | Purpose |
|---|---|
| `main.py` | Train the lattice and Ahn linear Transformers for the three covariance families and export the results. |
| `plot_results.py` | Regenerate figures and Table S1 from a completed run. |
| `operator_study/config.py` | Covariances and training settings. |
| `operator_study/geometry.py`, `data.py` | Lattice geometry and synthetic regression tasks. |
| `operator_study/models.py`, `training.py` | Lattice Transformer and training routines. |
| `operator_study/ahn_upstream.py`, `ahn_protocol.py`, `ahn_training.py` | Ahn model, task settings, and training. |
| `operator_study/extraction.py`, `analysis.py` | Unit-force operator extraction, whitening, and spectral diagnostics. |
| `operator_study/extraction_plot.py`, `figure2_plots.py` | Figure 2(a) extraction diagram and Figure 2(b,c) heatmaps. |
| `operator_study/loss_plots.py`, `plots.py`, `tables.py` | Figures S2–S3 and Table S1. |
| `operator_study/experiments.py`, `cache.py` | Run orchestration and numerical output storage. |

## Environment

```bash
conda env create -f environment.yml
conda activate regression-operators
```

Alternatively, use Python 3.14 and `python -m pip install -r requirements.txt`.

## Run

```bash
python main.py --case all --device cpu --output-dir runs/paper
```

Use `--device cuda` for GPU execution, or `--case rotated` for one covariance
family. Both models have three layers. The lattice Transformer trains for
500 epochs; the Ahn model uses 10,000 updates for the isotropic case and
100,000 for the diagonal and rotated cases.

Outputs include checkpoints, covariance inputs, `results.npz`, and
`figures_and_tables/`. Figure 2 and Figure S3 whiten the raw learned operators
using the symmetric covariance square root. Table S1 reports whitened distances
for both the raw lattice operator and its symmetric target.

To regenerate plots without training:

```bash
python plot_results.py --results runs/paper/results.npz --output-dir figures
```

The combined `results.npz` and case covariance files are also inputs to the
physical-realization experiment. A short execution check is available with
`python main.py --smoke --output-dir runs/smoke`.

The Ahn implementation is adapted from
[chengxiang/LinearTransformer](https://github.com/chengxiang/LinearTransformer),
commit `b6ddc35f0f83e3f8f26e45548e94bac2adafdd50`.
