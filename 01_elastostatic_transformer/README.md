# Elastostatic lattice Transformer

Code for Figure 1 and the model described in Supplementary Section S1.

| File | Purpose |
|---|---|
| `train_5x5.py` | Train and evaluate the 5×5 lattice model; generate losses and displacement fields for Figure 1(a–c). |
| `train_10x10.py` | Train and evaluate the 10×10 lattice model; generate losses and displacement fields for Figure 1(d–f). |
| `reproduce_figure1.py` | Load the supplied checkpoints and loads, compute equilibrium targets and model predictions, and export Figure 1. |
| `examples/` | Best-validation checkpoints, applied-load inputs, and training histories for both lattice sizes. |
| `plot_figure1.py` | Export the six panels and combined Figure 1. |

## Environment

```bash
conda env create -f environment.yml
conda activate elastostatic-transformer

## To Reproduce Figure 1 

```bash
python reproduce_figure1.py
```

The loss curves (a, d) are plotted from the per-epoch training and validation
losses saved in `examples/<size>/history*.pth`. For the deformation panels, the
nodal forces in `examples/<size>/loads.csv` are applied to the lattice: solving
the equilibrium equations gives the true displacements (b, e), and the trained
checkpoint `best_by_val.pth` gives the predicted displacements (c, f). Output
goes to `runs/reproduced_figure1/`.

## Run

```bash
python train_5x5.py
python train_10x10.py
python plot_figure1.py
```

The models use seven layers, one head, and 42,000 load samples with an
80/10/10 split. The 5×5 case uses float64 and 2,000 epochs; the 10×10 case
uses float32 and 200 epochs. Checkpoints are selected by validation loss.
Loss curves report MSE on normalized displacements at free degrees of freedom.
The deformation panels show the test sample with the largest relative L2 error.

The 5×5 dataset is generated once. The 10×10 dataset is generated on the fly.

