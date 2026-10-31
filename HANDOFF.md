# FT-Transformer: hand-off for the full-dataset run

Christian, this is everything needed to make the official FT-Transformer run on `final_dataset.parquet` (980,964 rows). It uses the same temporal folds, Optuna tuning and `collab_sol` risk report as the team's other models, so the outputs drop straight into the comparison. More detail is in the README section [Running the FT-Transformer on the full dataset](README.md#running-the-ft-transformer-on-the-full-dataset).

## What you need
- A machine with an NVIDIA GPU and [uv](https://docs.astral.sh/uv/).
- A local copy of `final_dataset.parquet`. It is not in the repo because it's over GitHub's size limit.

## Steps

```sh
git clone https://github.com/lucascampa/crossings.git
cd crossings
git clone https://github.com/helloimfrei/collab-sol.git collab-sol
uv sync

# 1. Two-minute check on the bundled 50k sample. Must finish with "Saved results to: ..."
uv run python scripts/train_ft_transformer.py --smoke --require-cuda

# 2. The real run
uv run python scripts/train_ft_transformer.py --dataset /path/to/final_dataset.parquet --require-cuda > run.log 2>&1
```

Step 1 matters: the GPU code path (CUDA and mixed precision) follows the team's MLP code but has never run on a GPU, because it was developed on a CPU-only laptop. If anything is going to break, it should break there, in two minutes, not hours into the real run.

## While it runs
`tail -f run.log` shows one line per epoch:

```
trial 3/20 | fold 2/4 | 327,000 train rows | n_blocks=2 d_block=128
trial 3/20 | fold 2/4 | epoch 7/50 | loss 0.41234 | val PR-AUC 0.0312 (best 0.0350 @ 5) | 42.1 s/epoch
```

Nobody knows yet how long the full run takes. It's 20 trials × 4 folds, each up to 50 epochs (early stopping usually ends them sooner), on about 163k, 327k, 490k and 654k training rows, then one refit on about 817k rows. The `s/epoch` figure from the first trial gives a rough estimate. If it's too slow, `--trials 10` roughly halves the time. Please mention any change from the defaults when you send the results back. `run.json` also records the exact command.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `--require-cuda was given but torch.cuda.is_available() is False` | torch is the CPU build. Install the CUDA build for your machine from [pytorch.org](https://pytorch.org/get-started/locally/) into `.venv`, then rerun. |
| `CUDA out of memory` | Add `--batch-size 512` (or 256). |
| `collab-sol not found` | Run the `git clone ... collab-sol` line from the repo root. |
| `... is not empty; earlier runs are never overwritten` | Leave out `--output-dir`, or point it to a new folder. |

## What to send back
- The whole output folder, `results/ft_transformer_<timestamp>/` (metrics, risk scores, plots, Optuna trials, `run.json`). `model.pt` is optional.
- `run.log`.

## One thing to know when comparing models
The FT-Transformer drops five columns that the team's MLP still uses: `latitude`, `longitude` (ID-like), `latlongsourcecode` (a coordinate-provenance / record-quality proxy), and `B01001_E001` and `B03002_E001` (duplicates of `B01003_E001`). That leaves 224 of the 233 columns. The full list and the reasons are in `EXCLUDED_FEATURES` in `ft_transformer/preprocess.py`, and are written to every `run.json`.
