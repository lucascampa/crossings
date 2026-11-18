# SHAP findings: final XGBoost models

What the SHAP analysis of the team's two final XGBoost models showed, how to read the plots, and how to rerun it. No model was retrained; SHAP reads the saved `model.joblib` files.

- Likelihood model: `results/google drive/likelihood_20261031/xgboost`
- Severity model (regression, 1 = property damage only, 2 = injury, 3 = fatal): `results/google drive/severity_comparison_20261115_50trials/regression_xgboost`
- Plots (full data): `results/shap_likelihood_xgboost_full/` (20,000 of the 163,706 rows from 2024) and `results/shap_severity_xgboost_full/` (all 10,206 incident rows). The numbers and findings below come from these.
- `results/shap_likelihood_xgboost/` and `results/shap_severity_xgboost/` are an earlier pass on the 50k sample. They give the same picture and are kept only for reference.

Check on the loading: with the saved model and preprocessor, average precision on all 2024 rows is 0.0957, equal to the value Christian reported, so the SHAP runs explain the same model.

## Reading the plots

SHAP splits one prediction into per-feature pushes from the model's average output. The pushes add up to the prediction.

- `beeswarm.png`: one row per feature (most important at the top), one dot per crossing. Position left or right is the push. Colour is the feature's value, red high and blue low. Red on the right means more of the feature gives a higher score.
- Units: **log-odds of an incident** for likelihood (+0.3 multiplies the odds by about 1.35, +1.0 by about 2.7), **severity points** for severity.
- The models use one-hot encoding, so a category such as `statecode` shows up as many yes/no rows (`statecode_6`). On those rows red means "this crossing is in the category" and blue means it isn't. Most crossings aren't, so the rows look like one tight blue blob; their direction is hard to read from the beeswarm.
- `importance_bar.png` and `importance_by_original_feature.csv` sum the one-hot levels and missing flags back to the original column, so you can rank features without that clutter. `importance_by_encoded_feature.csv` is the unsummed version.
- SHAP describes **what the model uses**, not what causes incidents. Correlated features share credit.

## Likelihood model

Top original features by mean |SHAP|: `numberoftransittracks` (0.31), `statecode` (0.30), `crossingpurposecode` (0.27), `annualaveragedailytrafficcount` (0.26), `developmenttypecode` (0.23), `typeoftrainserviceid1` (0.18), `whistlebancode` (0.16), `totaldaylightthrutrains` (0.14), `totalnighttimethrutrains` (0.14), `totaltransittrains` (0.14).

- **Exposure drives risk.** Higher traffic count (AADT), more daytime and nighttime through trains, and higher timetable speed all push risk up (red dots on the right). This is the clearest, most plausible result.
- **State and crossing-type codes matter** (`statecode`, `crossingpurposecode`, `developmenttypecode`, `typeoftrainserviceid1`, `whistlebancode`). The bars show they matter, but not their direction; decode the codes in the FRA data dictionary before interpreting them.
- **Missing-value flags are among the top features** (see the next section).
- Census columns account for about 28% of total importance, spread across 146 columns.

## Severity model

Top original features: `maximumspeedrangeovercrossing` (0.079), `statecode` (0.077), `B03002_E017` (0.030), `numberpassengertrainperday` (0.027), `B03002_E008` (0.023), `B19001_E016` (0.022), `B19001_E004` (0.021), `totaldaylightthrutrains` (0.020), `B08301_E008` (0.020), `maximumtimetablespeed` (0.018).

- **Physical drivers make sense.** A higher maximum speed range over the crossing, more passenger trains per day, more daytime through trains and a higher timetable speed push severity up.
- **Census columns carry about 62% of the total importance** (146 of 224 features; 28% for the likelihood model). The individual columns are narrow ACS cells, not broad demographics. Labels checked against the Census API (2023 ACS 5-year): `B03002_E017` is Hispanic or Latino, Native Hawaiian and Other Pacific Islander alone; `B03002_E008` is Not Hispanic or Latino, some other race alone; `B19001_E016` is households earning $150,000 to $199,999; `B19001_E004` is households earning $15,000 to $19,999; `B01001_E034` is females aged 22 to 24; `B08301_E008` is commuters in a 5- or 6-person carpool; `B15003_E011` is adults whose highest education is 7th grade. These are small raw counts per tract. They most likely act as **location fingerprints**: `statecode_6` (California, FIPS 06; the data's state codes run up to 56, which matches FIPS) and `B03002_E017` both push severity up, which fits a group concentrated on the Pacific coast. They are not evidence about those groups.
- **Weak model.** The full incident cohort has 10,206 rows. On the 2024 holdout the XGBoost regression reached MAE 0.598 against 0.546 for always predicting property damage only. SHAP values are small (mostly within ±0.2 points). Read this plot as a description of what the model leans on, not as a strong finding.
- **Use consideration.** If severity feeds an inspection ranking, demographic-count columns can end up influencing which areas are inspected first. Worth a line in any write-up, and a reason to run the ablation without the census columns (that needs retraining).

## Missing-value flags

Features named `missingindicator_X` are 0/1 columns added by the preprocessing: 1 if the crossing's value for `X` was blank (the blank is filled with the median, and the flag tells the model it was filled). In the beeswarm, red means blank.

- Several flags rank high in the likelihood plot (`numberoftransittracks`, `totaltransittrains`, `highwayspeedlimit`, `numberofindustrytracks`).
- **A blank is not the same as zero.** In `numberoftransittracks` (full data), 68.4% of rows report 0 explicitly and a separate 31.4% are blank. Blank rows have an incident rate of 0.20% against 1.43% for rows with a value, and none of them report transit trains. 99.3% of the transit-tracks blanks are also blank for transit trains, and 95.7% are blank for transit tracks, industry tracks and transit trains together. So whole sections of Part II of the FRA form (Form 6180.71, Box II.4 "Type and Count of Tracks") were left empty on those records.
- Likely reading: these flags mark crossings whose railroad section was never filled in (inactive, private or low-quality records), which have far fewer incidents. They are record-quality or crossing-type proxies, not causes.

## What this does not tell you

- Whether performance changes without a feature group. That requires retraining (feature ablation).
- Any causal effect. A feature that pushes risk up is associated with incidents in the data.
- Which crossings the blank sections belong to (inactive, private, never completed). That is inferred from the pattern, not checked against crossing status.

## Rerunning

```sh
uv run python scripts/shap_xgboost.py --task likelihood --run-dir "<likelihood run folder>/xgboost"
uv run python scripts/shap_xgboost.py --task severity   --run-dir "<severity run folder>/regression_xgboost"
```

- Add `--dataset /path/to/final_dataset.parquet` for the full file (defaults to the 50k sample). The full-data plots were made with `--dataset data/drive/final_dataset.parquet --max-rows 20000 --out-dir results/shap_<task>_xgboost_full` and took under a minute each on CPU. Likelihood explains 2024 rows, severity explains incident rows; `--max-rows` (default 5000) caps the rows explained.
- Output goes to `results/shap_<task>_xgboost/` by default; use `--out-dir` to keep an earlier run.
- The GPU-trained pickles fail to unpickle on CPU ("input stream corrupted") even with the same xgboost version, so the script loads only the model section (`load_xgb`). If you re-save the models with `booster.save_model()`, that workaround isn't needed.
- The script works for the team's XGBoost runs only, because it uses their saved preprocessor and `TreeExplainer`.

For the different analyses (ablation, risk scores, other models), `results/README.md` lists every run folder.
