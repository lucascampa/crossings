# crossings

FT-Transformer for US highway-rail grade crossing incident risk (Collab Solutions practicum). This repo is the **run pipeline** for the team's FT-Transformer: it is developed and smoke-tested here on a 50k-row sample, and the official run on the full dataset (980k rows) is made on a GPU machine with one command (see [Running the FT-Transformer on the full dataset](#running-the-ft-transformer-on-the-full-dataset)). It uses the same temporal folds, tuning and risk report as the team's other models, via the shared [`collab-sol`](https://github.com/helloimfrei/collab-sol) utilities, so the results compare directly.

## Setup

Requires [uv](https://docs.astral.sh/uv/). From the repo root, clone the team utilities into this folder first (it is gitignored here and has its own `.git`), then sync:

```sh
git clone https://github.com/helloimfrei/collab-sol.git collab-sol
uv sync --system-certs   # --system-certs only if your network intercepts TLS
```

`uv sync` creates `.venv` with `collab-sol` (editable, from `collab-sol/`), PyTorch (CPU build on Windows), `rtdl-revisiting-models`, pyarrow and Jupyter support. Select `.venv` as the kernel in VS Code.

## Data

- `final_dataset_sample_50k.parquet`: the training sample (49,927 rows, 8,458 whole crossings, 1.02% positive). Committed.
- `final_dataset.parquet`: the full file (980,964 rows). Not in git; copy it into the repo root when needed.
- `make_sample.py` rebuilds the sample: `uv run python make_sample.py final_dataset.parquet out.parquet`.
- `sample_notes.md` profiles the full file and shows how the sample was checked.

One row per crossing per year (`crossingid` x `asof_year`, 2019-2024). The target is `incident`. Never use `severity_target`, `incident_count`, `total_killed`, `total_injured` or `severity_ordinal` as features.

## Features not used

The dataset has 233 columns. The FT-Transformer trains on 224 of them. The 9 it doesn't use are below. The code is the source of truth (`EXCLUDED_FEATURES` in `ft_transformer/preprocess.py`), and every run also records the list in `run.json`. **If you add or remove an exclusion, update that dict and this table.**

**Dropped from the predictors (on the team's advice):**

| Column | Why it was dropped |
|--------|--------------------|
| `latitude` | Near-unique per crossing, so it acts like an ID and invites memorization. Many crossings also share identical placeholder coordinates. |
| `longitude` | Same as `latitude`. |
| `latlongsourcecode` | The Form 6180.71 (FRA crossing inventory) field recording how the coordinates were obtained: 1 = Actual (surveyed, e.g. GPS), 2 = Estimated (e.g. read off a map). It says how far to trust the coordinates, not anything about crash risk. It is null in ~23% of sample rows, and those rows have ~40x lower incident rate (0.03% vs 1.0-1.4%). That looks like a record-quality or record-age proxy, not physical risk. The text label `latlongsource` is not in the dataset. |
| `B01001_E001` | ACS total population from the sex-by-age table. Identical to `B01003_E001` in the 50k sample; dropped as a duplicate on the team's advice. |
| `B03002_E001` | ACS total population from the race/Hispanic table. Same as above. |

**Never predictors, by design:**

| Column | Role |
|--------|------|
| `incident` | Target. |
| `severity_target` | Leaks the label (`none` exactly when `incident` = 0). |
| `crossingid` | ID; used only to group and report rows. |
| `asof_year` | Used for the temporal split, not as a feature. |

**For comparisons:** `collab-sol/scripts/train_mlp.py` uses all 229 non-ID, non-target columns, including the three above. The FT-Transformer's feature set therefore differs from the MLP's by those columns unless the other models drop them too.

**Open questions (not yet decided):**
- `B01001_E001` and `B03002_E001` were dropped as duplicates of `B01003_E001`, which is identical to them in the 50k sample. Confirm they are also identical in the full file.
- Whether the as-of join can pick up Form 71 revisions made after an incident in the same year (inventories are updated after accidents). If it can, other record-completeness fields may leak the same way as `latlongsourcecode`.

## Team utilities

No inheritance is required. Define one function for the parameters Optuna searches and one that builds the model; the model only needs `fit(X, y)` and `predict_proba(X)` returning two columns.

```py
from collab_sol import make_expanding_temporal_split, tune_model, generate_risk_report

split = make_expanding_temporal_split(frame, year_column="asof_year")
study = tune_model(model_factory=build_model, suggest_params=suggest_params,
                   X=X, y=y, split=split, preprocessor_factory=make_preprocessor, n_trials=20)
```

### Folds

| Fold | Train | Validate |
|------|-------|----------|
| 1 | 2019 | 2020 |
| 2 | 2019-2020 | 2021 |
| 3 | 2019-2021 | 2022 |
| 4 | 2019-2022 | 2023 |
| Test | 2019-2023 | 2024 |

Optuna maximizes mean validation average precision across the four folds and never sees 2024. After tuning, refit on 2019-2023 and evaluate 2024 once. Preprocessing is fit independently inside each fold. Models that need validation data while fitting (early stopping, custom training loops) can pass the optional `fit_model` callback; see the `tune_model` docstring.

`tune_model` returns an ordinary Optuna study, held in memory unless `storage=` is given. Save the trials with `study.trials_dataframe().to_csv(...)`. `generate_risk_report` saves average precision, prevalence, a precision-recall curve and one continuous risk score per crossing.

## Running the FT-Transformer on the full dataset

One command runs the whole pipeline, with the same protocol as `collab-sol/scripts/train_mlp.py`: Optuna over the four expanding folds, one refit on 2019-2023, one evaluation on 2024.

```sh
git clone https://github.com/helloimfrei/collab-sol.git collab-sol   # if not already present
uv sync                                                              # on Linux this installs the CUDA build of torch
uv run python scripts/train_ft_transformer.py     --dataset /path/to/final_dataset.parquet     --require-cuda
```

Flags: `--trials` (default 20), `--max-epochs` (50), `--patience` (5), `--batch-size` (1024), `--seed` (42), `--output-dir` (default `results/ft_transformer_<UTC timestamp>/`; a non-empty folder is refused, so earlier runs are never overwritten). `--require-cuda` fails fast if no GPU is visible. If torch doesn't see the GPU, install the CUDA build for that machine from [pytorch.org](https://pytorch.org/get-started/locally/). `--smoke` runs a tiny CPU test on the 50k sample (20% of crossings, 1 trial, 2 epochs, a 1-block model) to check that everything works before the long run; its numbers mean nothing.

Each run writes to its output folder:

| File | Contents |
|------|----------|
| `metrics.csv`, `risk_scores.csv`, `inspection_budget.csv`, `precision_recall_curve.png` | the standard `collab_sol` report, same as the team's other models |
| `calibration.json`, `reliability_plot.png` | ROC-AUC, Brier score (and the constant-prevalence Brier baseline), reliability plot |
| `optuna_trials.csv` | every trial with its per-fold validation scores |
| `class_balance_by_year.csv`, `class_balance_by_fold.csv` | class balance of every split |
| `final_training_history.csv` | loss per epoch of the final refit |
| `run.json` | best parameters, validation scores, test metrics, seed, device, versions, the exact command |
| `model.pt`, `preprocessor.joblib` | weights and the fitted preprocessing (`model.pt` is gitignored) |

How it differs from the MLP:
- **Preprocessing:** numeric columns are median-imputed and quantile-transformed (normal output), with missing indicators. Categoricals become integer codes for the FT-Transformer's tokenizer, with missing and unseen values as their own levels. All of it is fit on the training rows of each fold only (`ft_transformer/preprocess.py`).
- **Loss and stopping:** weighted BCE (`pos_weight` is tuned, 1-50). Each fold early-stops on validation PR-AUC. The final refit has no validation data, so it trains for the median best epoch count from the tuned folds.
- **Calibration:** because of `pos_weight` the scores are over-confident, so the Brier score will look worse than the prevalence baseline. Rankings (PR-AUC, inspection budgets) are unaffected.
- **Search space** (`suggest_params` in the script): blocks 1-4, width 64/128/192/256 with 8 heads, attention/FFN/residual dropout, learning rate 1e-5 to 1e-3, weight decay 1e-6 to 1e-3.

The code is in `ft_transformer/` (`model.py`, `preprocess.py`, `metrics.py`) and `scripts/train_ft_transformer.py`.

Status: the pipeline has been run end to end with `--smoke` on CPU only. The CUDA path (`--require-cuda`, mixed precision) follows the team's MLP code but has not been run on a GPU, so run `--smoke` once on the target machine before the long run.

## Notebooks

- `notebooks/utils_demo_local.ipynb`: the team's logistic regression demo adapted to local paths on the 50k sample. It shows the `collab-sol` workflow that `scripts/train_ft_transformer.py` follows.
- `collab-sol/notebooks/utils_demo.ipynb`: the team's original Colab version (Google Drive paths), in the clone.

## Results

Run outputs go in `results/<run_name>/`. Don't overwrite earlier runs.
