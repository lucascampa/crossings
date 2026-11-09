# FT-Transformer: hand-off for the full-dataset runs

Christian, this is everything needed to make the official FT-Transformer runs on `final_dataset.parquet` (980,964 rows). There are two scripts:

- **Likelihood** (`scripts/train_ft_transformer.py`): will this crossing have an incident next year? Same temporal folds, Optuna tuning and `collab_sol` risk report as the team's other likelihood models. Details: README section [Running the FT-Transformer on the full dataset](README.md#running-the-ft-transformer-on-the-full-dataset).
- **Severity** (`scripts/train_severity_ft_transformer.py`): given an incident, is the most severe outcome property damage, injury or fatal? Same protocol as your `train_severity_mlp.py`: incident rows only, the three `SEVERITY_CLASSES`, Optuna on mean validation log loss, unweighted cross-entropy, `generate_severity_report`. Details: README section [Severity model](README.md#severity-model).

## What you need
- A machine with an NVIDIA GPU and [uv](https://docs.astral.sh/uv/).
- A local copy of `final_dataset.parquet`. It is not in the repo because it's over GitHub's size limit.

## Steps

```sh
git clone https://github.com/lucascampa/crossings.git
cd crossings
git clone https://github.com/helloimfrei/collab-sol.git collab-sol
uv sync

# 1. Quick checks on the bundled 50k sample. Each must finish with "Saved results to: ..."
uv run python scripts/train_ft_transformer.py --smoke --require-cuda            # about two minutes
uv run python scripts/train_severity_ft_transformer.py --smoke --require-cuda   # under a minute

# 2. The real runs
uv run python scripts/train_severity_ft_transformer.py --dataset /path/to/final_dataset.parquet --require-cuda > severity_run.log 2>&1
uv run python scripts/train_ft_transformer.py --dataset /path/to/final_dataset.parquet --require-cuda > run.log 2>&1
```

Step 1 matters: the GPU code path (CUDA and mixed precision) follows the team's MLP code but has never run on a GPU, because it was developed on a CPU-only laptop. If anything is going to break, it should break there, in a few minutes, not hours into the real run.

The severity run is listed first because it is much shorter: it trains only on incident rows (about 1% of the data).

## While it runs
`tail -f run.log` (or `severity_run.log`) shows one line per epoch:

```
trial 3/20 | fold 2/4 | 327,000 train rows | n_blocks=2 d_block=128
trial 3/20 | fold 2/4 | epoch 7/50 | loss 0.41234 | val PR-AUC 0.0312 (best 0.0350 @ 5) | 42.1 s/epoch
```

The severity run prints `val log loss` instead of `val PR-AUC` (lower is better).

Nobody knows yet how long the likelihood run takes. It's 20 trials × 4 folds, each up to 50 epochs (early stopping usually ends them sooner), on about 163k, 327k, 490k and 654k training rows, then one refit on about 817k rows. The `s/epoch` figure from the first trial gives a rough estimate. If it's too slow, `--trials 10` roughly halves the time. Please mention any change from the defaults when you send the results back. `run.json` also records the exact command.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `--require-cuda was given but torch.cuda.is_available() is False` | torch is the CPU build. Install the CUDA build for your machine from [pytorch.org](https://pytorch.org/get-started/locally/) into `.venv`, then rerun. |
| `CUDA out of memory` | Add `--batch-size 512` (or 256). |
| `collab-sol not found` | The `collab-sol` folder must sit inside `crossings/`. Run the `git clone ... collab-sol` line from the repo root. |
| `cannot import name 'SEVERITY_CLASSES'` (or another severity name) | The `collab-sol` clone predates your severity changes. Run `git -C collab-sol pull`. |
| `... is not empty; earlier runs are never overwritten` | Leave out `--output-dir`, or point it to a new folder. |

## What to send back
- The whole output folders, `results/ft_transformer_<timestamp>/` and `results/severity_ft_transformer_<timestamp>/` (metrics, scores, plots, Optuna trials, `run.json`). `model.pt` is optional.
- `run.log` and `severity_run.log`.

## Severity scores for crossings with no incident
Besides the standard severity report, the severity run writes `severity_scores_all_crossings.csv`: the class probabilities and severity score for **every** 2024 crossing, including the ones with no incident (`actual_class` = `none`). Read them as "if a crash happens here, how severe it would likely be". This is meant for the open question of how to use the severity models on crossings with no crash: multiplying by P(incident) from a likelihood model gives unconditional risk, e.g. P(fatal crash) = P(incident) × P(fatal | incident). Two caveats: the model never trained on rows without an incident, so those scores are an extrapolation, and the likelihood scores are inflated by `pos_weight`, so they need recalibrating before the multiplication.

## Features
Both scripts use the same 224 of the 233 columns as `collab_sol.split_risk_feature_columns`, i.e. the same feature set as the team's other models. The five dropped predictors (`latitude`, `longitude`, `latlongsourcecode`, `B01001_E001`, `B03002_E001`) and the reasons are in `EXCLUDED_FEATURES` in `ft_transformer/preprocess.py`, and are written to every `run.json`.
