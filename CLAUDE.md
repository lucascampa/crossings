# Crossings: FT-Transformer for grade crossing incident risk

## What this is
Practicum project (Collab Solutions externship): predict which US highway-rail grade crossings will have an incident in a given year. My assignment is the **FT-Transformer** (Gorishniy et al. 2021). The rest of the team has already run the other models (logistic regression, MLP, etc.), so I only run the FT-Transformer, and only on the sample. **My real deliverable is scaffolding**: code, a run script and instructions so Christian Tapp can make the official run on the full dataset (980k rows) on his own GPU machine by pressing one button. It must produce outputs in the same format as the team's other models so the results compare fairly. Everything here is tested on the 50k sample, on CPU, with tiny settings; the sample numbers are smoke tests, not results.

**Status:** the scaffolding is built (`ft_transformer/`, `scripts/train_ft_transformer.py`) and runs end to end with `--smoke` on CPU. Not yet done: a run on a GPU (the CUDA / `--require-cuda` path is untested), a hand-off note for Christian, and unit tests for the preprocessing and model wrapper. Don't claim model performance from sample runs.

Background: a prior team (Patel & Buddenbaum, May 2026) built FRA grade-crossing risk models with ACS census features. FRA's current tool (GXAPS) plateaus around 70% accuracy. The data is a rare-event problem, and stakeholders want risk tiers / rankings rather than raw scores.

I work locally (VS Code + Claude Code), not in Colab. Setup: `uv sync --system-certs` (the flag is needed on this machine's network), then `uv run ...`. `pyproject.toml` is this repo's own and installs the team's `collab-sol` package as an editable install from the local clone at `collab-sol/` plus torch, rtdl-revisiting-models and pyarrow. No GPU here, so torch is the CPU build.

## Repo files
- `final_dataset_sample_50k.parquet`, `make_sample.py`, `sample_notes.md`: the sample, its script and its notes (see Data).
- `collab-sol/`: clone of `github.com/helloimfrei/collab-sol` (public, team's shared utilities, has its own `.git`, gitignored here). Don't edit it without telling the user; update with `git -C collab-sol pull`. Its `notebooks/utils_demo.ipynb` is the original Colab demo (Drive paths, don't run locally).
- `ft_transformer/` (`model.py`, `preprocess.py`, `metrics.py`) and `scripts/train_ft_transformer.py`: the scaffolding for Christian's run. The script mirrors `collab-sol/scripts/train_mlp.py` and its output files; see the README section "Running the FT-Transformer on the full dataset". Test with `uv run python scripts/train_ft_transformer.py --smoke`.
- `notebooks/utils_demo_local.ipynb`: that demo adapted to local paths on the 50k sample; use it as the template.
- `docs/FT-Transformer.pdf`: the FT-Transformer paper.
- `pyproject.toml`, `uv.lock`, `.python-version`: environment. `README.md`: setup, the team utilities, and the full-dataset run instructions. `LICENSE`.
- `results/`: run outputs, one folder per run (`ft_transformer_<timestamp>/`; `_smoke` suffix for smoke runs; `logistic_demo/` is from the demo notebook).

## Data
- `final_dataset_sample_50k.parquet`: what we train on. 49,927 rows, 8,458 whole crossings with all their years, 510 positives (1.02%). Committed to the repo.
- `final_dataset.parquet`: the full file, 980,964 rows × 233 columns, 155MB. **Not in git** (`.gitignore`, over GitHub's 100MB limit). Copy it in by hand when needed.
- `make_sample.py`: rebuilds the sample: `python make_sample.py final_dataset.parquet out.parquet`. Samples whole crossings, stratified by state × incident history (0 / 1 / 2+ incident-years), seed 42. Change `TARGET_ROWS` for another size.
- `sample_notes.md`: profile of the full file and how the sample was checked against it.

Grain: **one row per crossing per year** (`crossingid` × `asof_year`, 2019–2024, no duplicates).

### Columns
- **Target: `incident`** (int8, 0/1). About 1% positive, flat across years.
- **`severity_target`**: none / property_damage_only / injury / fatal. It's `incident` with more detail, so it **leaks the label. Never use it as a feature** for the binary model. (It could be a target for a later severity model.)
- **`crossingid`**: ID, not a feature. Use as the group key.
- **`asof_year`**: the year. Use for splitting; think before you use it as a feature.
- Weather, 5-year means: `pr_total_mm_5y_mean`, `wet_days_*`, `freeze_thaw_days_*`, `hot_days_*`, `tmin/tmax_mean_c_*`, `wind_*`, etc.
- FRA Form 71 crossing inventory: train counts and speeds, track counts, warning devices (gates, flashers, bells, signs), road traffic (AADT, % trucks, school buses), speed limit, lanes, lat/long.
- ACS 5-year tract tables, **raw counts**: B01003 total pop, B01001 sex by age, B03002 race/Hispanic, B19013 median income, B19001 income brackets, B15003 education, B08301 commute mode, B08303 commute time. Consider turning them into shares of the table total (`_E001`) before training.
- **Categorical**:
  - 25 string columns (Y/N flags plus `highwayspeedlimittypecode`, `reporting_railroad_class`, `reporting_railroad_smt_grouping`, `wdcode`), all 2–11 levels.
  - Integer-coded columns that are categories, not quantities: `statecode`, `reportingagencytypecode`, `innearcode`, `crossingpurposecode`, `typeoftrainserviceid1`, `developmenttypecode`, `whistlebancode`, `latlongsourcecode`, `trafficlanetypecode`, `crossingsurfaceid1`, `smallestcrossinganglecode`, `highwaysystemcode`, `roadatcrossingcode`, `roadatcrossingtypecode`.
  - These go to the FT-Transformer's categorical tokenizer.
- **Missing values**: 222 columns have some. 8 are over 50% missing (`trainsperweek` 81%, `numberlowgroundclearancesigns` 80%, `emergencyservicesroute` 76%, …). For categoricals, treat missing as its own category. For numerics, impute and add a missing-indicator where it's informative.

## Modeling rules
- **Split by time, never by random row.** The same crossing appears up to 6 times. Team protocol (use it so results compare across models), via `collab_sol.make_expanding_temporal_split`:
  - Fold 1: train 2019, validate 2020. Fold 2: train 2019–2020, validate 2021. Fold 3: train 2019–2021, validate 2022. Fold 4: train 2019–2022, validate 2023.
  - Test: train 2019–2023, predict 2024. The tuner never sees 2024. Refit once on 2019–2023 and evaluate 2024 once.
  - Tune with Optuna (`tune_model`, mean validation average precision across folds). Models that need validation data while fitting, like the FT-Transformer's early stopping, use the `fit_model` callback.
- **Fit preprocessing on train only** (scalers / quantile transforms, imputers, category vocabularies).
- **Class imbalance**: weighted BCE with `pos_weight` tuned by Optuna (1–50); focal loss is not implemented. Don't resample the test set. Weighting makes scores over-confident, so the Brier score will look worse than the prevalence baseline; rankings are unaffected.
- **Metrics**: PR-AUC (primary) and ROC-AUC, plus recall / precision at a fixed alert budget (e.g. top 1% / 5% of crossings), and calibration (Brier score, reliability plot). Accuracy is meaningless here, since all-zeros scores 99%.
- **Baselines**: not our job. The team has already run them, so don't train baselines here. Use the same split and the `collab_sol` report so the FT-Transformer's numbers compare directly with theirs.
- Report the class balance of every split, and fix random seeds.

## FT-Transformer notes
- Paper: Gorishniy et al. 2021, "Revisiting Deep Learning Models for Tabular Data" (NeurIPS), arXiv 2106.11959, saved at `docs/FT-Transformer.pdf`. Check it when an architecture or tuning detail matters.
- The model is `rtdl_revisiting_models.FTTransformer` (official, PyTorch), wrapped in `ft_transformer/model.py` as `FTTransformerClassifier` (`fit`, `predict_proba`, `fit_with_validation`). Preprocessing output is `[numeric | missing indicators | category codes]`; the last `n_cat_features` columns are codes, with -1 for unseen.
- Stopping: each tuning fold early-stops on validation PR-AUC. The final refit on 2019–2023 has no validation data, so it trains for the median best epoch count from the tuned folds.
- Paper defaults (d_token 192, 3 blocks, 8 heads, attention dropout 0.2, FFN dropout 0.1, AdamW lr 1e-4, weight decay 1e-5) are the model class's defaults. The Optuna search space is in `suggest_params` in `scripts/train_ft_transformer.py`: blocks 1–4, width 64/128/192/256 (8 heads), dropouts, lr 1e-5–1e-3, weight decay 1e-6–1e-3, `pos_weight` 1–50.
- Feature exclusions: `latitude`, `longitude` (ID-like) and `latlongsourcecode` (coordinate-provenance metadata, a record-quality proxy) are dropped on the team's advice, and the duplicate ACS totals `B01001_E001` and `B03002_E001` (identical to `B01003_E001`, which is kept) are also dropped, leaving 224 of 233 columns. Every excluded column and its reason lives in `EXCLUDED_FEATURES` in `ft_transformer/preprocess.py`, is shown in the README section "Features not used" and is written to each run's `run.json`. **Whenever you add or remove an exclusion, update the dict and the README table.** The team's MLP still uses these columns, so tell the team when the feature sets differ.
- Numerics: median-impute, quantile-transform (normal output), plus missing indicators; categoricals are integer-coded with missing as its own level. All fit on training rows only (`ft_transformer/preprocess.py`).
- Local machine: CPU only, so tests use the sample with tiny settings. The full run is on Christian's machine (a Linux GPU box per the team's `mlp_training.py`; confirm). Everything must work with `--require-cuda` there and fall back to CPU here.

## Conventions
- Python, pandas + pyarrow, PyTorch, scikit-learn.
- Keep results (metrics JSON, plots) in `results/`, and don't overwrite earlier runs.
- Don't commit the full parquet or model checkpoints over ~50MB.
