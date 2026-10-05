# Mechanical language model

Code for Figure 6, Figure S8, and Tables S13–S14.

| File | Purpose |
|---|---|
| `prepare_data.py` | Prepare deduplicated TinyStories and Children-Stories-Collection splits and the GPT-2 tokenizer. |
| `prepare_study.py` | Create a shared random initialization for the one-step and two-step models. |
| `train_study.py` | Train either model and resume its model, AdamW, RNG, and batch state. |
| `src/gd_model.py`, `gd_build.py`, `gd_lattice_geom.py` | Mechanical language model, lattice construction, and model assembly. |
| `src/gd_data.py`, `gd_eval.py`, `gd_generation.py` | Token batches, cross entropy, and autoregressive decoding. |
| `src/common.py`, `gd_utils.py`, `causality.py` | Configuration, checkpoint utilities, and causal-invariance diagnostics. |
| `measure_residuals.py` | Measure interior force residuals before and after Newton updates. |
| `generate_greedy.py` | Generate ten-token continuations, including the three Table S14 prompts. |
| `generate_figure6.py`, `compose_figure6.py`, `paper_plot_helpers.py` | Compute mechanical fields and export Figure 6 without token labels on its mechanical panels. |
| `plot_loss.py` | Export epoch training/validation loss curves for Figure S8. |
| `export_tables.py` | Export Tables S13–S14 from measured results. |
| `evaluate.py`, `summarize.py` | Evaluate held-out data and compare the two training runs. |

## Environment

```bash
conda env create -f environment.yml
conda activate mechanical-language
```

The training runs used Python 3.10.19, PyTorch 2.5.1, and an NVIDIA H200. Full training requires substantial GPU time. Dataset preparation downloads the source stories and tokenizer from Hugging Face.

## Prepare and train

Run from this folder:

```bash
python prepare_data.py --output data/stories
python prepare_study.py --data data/stories --study runs/newton_study
python train_study.py --data data/stories --newton-steps 1 --output runs/newton_study/newton1 --require-cuda
python train_study.py --data data/stories --newton-steps 2 --output runs/newton_study/newton2 --require-cuda
```

The models share initialization, data order, and training settings. Each uses two layers, four heads, embedding dimension 512, context length 64, and relative regularization 0.05. Internal displacements start at zero for each prompt configuration; one or two Newton updates provide an approximate equilibrium state. Kernel normalization uses only the causal prefix.

Data preparation uses 300000 training, 2000 validation, and 2000 test stories. Each story supplies up to 65 tokens, yielding at most 64 input–target pairs. Duplicate normalized stories and retained token windows are excluded across splits. The saved data manifest records source revisions and file hashes.

Training uses 15 epochs, batch size 8, learning rate 0.0001, and AdamW weight decay 0.01. Validation occurs every 2500 updates and at epoch boundaries. Each invocation has a default ten-hour limit; repeat the same command to resume until training completes. Best-validation and latest checkpoints are saved separately.

## Figures and tables

For each arm, measure the selected best checkpoint (use `2` and `newton2` for the second arm):

```bash
python measure_residuals.py --data data/stories --checkpoint runs/newton_study/newton1/checkpoints/best.pt --newton-steps 1 --output runs/newton_study/diagnostics/best_newton1.json
python generate_greedy.py --data data/stories --run runs/newton_study/newton1 --output runs/generations
python plot_loss.py --metrics runs/newton_study/newton1/metrics.csv --output runs/figures
python generate_figure6.py --data data/stories --checkpoint runs/newton_study/newton1/checkpoints/best.pt --output runs/figure6
python compose_figure6.py --results runs/figure6
```

After both residual measurements are available:

```bash
python export_tables.py --study runs/newton_study --generations runs/generations/generations.json --output runs/tables
```

Figure S8 uses completed-epoch losses; training cross entropy is averaged over valid target tokens processed during the epoch. Residual diagnostics use eight validation stories. Story examples use greedy decoding with up to ten new tokens and no repetition ban.

Optional held-out evaluation and sampled causality checks:

```bash
python evaluate.py --checkpoint runs/newton_study/newton1/checkpoints/best.pt --data data/stories --output runs/newton_study/newton1/evaluation.json
```
