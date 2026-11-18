# Results index

One row per folder or file. Run folders are never overwritten; new runs get a new timestamped folder.

**Read this first**
- `_smoke` runs are CPU plumbing checks (1 Optuna trial, tiny settings). Their metrics mean nothing and must not be quoted.
- "Sample" is `final_dataset_sample_50k.parquet`. "Full" is the 980k-row `final_dataset.parquet`.
- Likelihood scores come from weighted-loss models, so they rank crossings but are not calibrated probabilities.
- Compare models only within a family (likelihood with likelihood, severity with severity) and only from non-smoke runs.

## Committed to git

| Path | What it is | Produced by | Real or smoke |
|---|---|---|---|
| `ft_transformer_20261028_193531_smoke/`, `..._204409_smoke/`, `ft_transformer_20261031_043423_smoke/`, `..._20261109_001803_smoke/` | FT-Transformer likelihood runs on the sample. Early ones are from earlier versions of the scaffolding (different machine paths, `model.pt` not committed because `*.pt` is gitignored). | `scripts/train_ft_transformer.py --smoke` | Smoke |
| `severity_ft_transformer_20261109_001741_smoke/` | Conditional-severity FT-Transformer (3-class) on the sample's 510 incident rows. Includes `severity_scores_all_crossings.csv`. | `scripts/train_severity_ft_transformer.py --smoke` | Smoke |
| `severity_panel_20261115_232351_smoke/` | Same 8-model severity panel as below, with 1 trial per model. | collab-sol severity scripts, run locally | Smoke |
| `severity_panel_20261115_233914/` | 8-model severity panel (4 classifiers, 4 regressors) on the incident-only cohort, 50 trials per model, CPU. `model_comparison.csv` is the summary; `panel_run.json` has each command. | collab-sol `train_severity_classification.py` / `train_severity_regression.py` | Real (50 trials) |
| `*.log` (`panel_smoke_...`, `severity_panel_..._233914.log`) | Console logs of the two panel runs above. | | |
| `logistic_demo/` | Output of `notebooks/utils_demo_local.ipynb` (team logistic regression demo on the sample). | notebook | Demo |
| `shap_likelihood_xgboost_full/`, `shap_severity_xgboost_full/` | SHAP beeswarm and importance plots for the team's final XGBoost models, on the **full data**: 20,000 of the 163,706 rows from 2024 (likelihood) and all 10,206 incident rows (severity). Findings in `docs/shap_findings.md`. | `scripts/shap_xgboost.py --dataset data/drive/final_dataset.parquet --max-rows 20000` | Real |

## Not in git (`google drive/`, gitignored, copied by hand from the team Drive)

| Path | What it is | Notes |
|---|---|---|
| `likelihood_20261031/` | Likelihood comparison on the full data: `xgboost`, `logistic_regression`, `isolation_forest`, `mlp`, `ft_transformer`, plus `model_comparison.csv`. | Run by Christian from collab-sol. XGBoost used 20 trials on GPU. Holds the final XGBoost likelihood model. |
| `severity_comparison_20261115_50trials/` | Severity comparison on the full data, 8 models, 50 trials each. | Run by Christian. Holds the final XGBoost severity regression. |
| `runs/severity_ft_transformer_2026-11-16/` | Severity FT-Transformer (classification and regression) on a Colab GPU, 50 trials. | Run from `severity_ft_20261115.ipynb`. |
| `severity_ft_20261115.ipynb` | The Colab notebook for the run above. | |

## Notes
- `model.joblib` and `model.pt` files are not committed (`*.joblib`, `*.pt` are gitignored). The Drive XGBoost pickles load on CPU only through `load_xgb` in `scripts/shap_xgboost.py`.
- Each run folder's `run.json` records the command, dataset path, seed, best parameters and excluded features. Check it before comparing two runs.
