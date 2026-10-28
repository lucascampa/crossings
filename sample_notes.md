# 50k sample of final_dataset.parquet

## Full file
- 980,964 rows, 233 columns. One row per crossing per year: 166,014 crossings, 2019–2024 (~163k rows per year).
- Target `incident` (int8, 0/1): 10,206 positives, **1.04%**. Rate is flat across years (0.9–1.1%).
- 8,457 crossings ever had an incident; most of them (7,124) only once.
- 49 state codes. 222 columns have some missing values, 8 are over 50% missing (e.g. `trainsperweek` 81%, `numberlowgroundclearancesigns` 80%, `emergencyservicesroute` 76%).
- 25 string columns, all low-cardinality (2–11 levels) apart from `crossingid`. Many integer-coded columns (`statecode`, `*code`, `*id1`) are categories stored as numbers.

## Sample: `final_dataset_sample_50k.parquet`
- 49,927 rows, 8,458 whole crossings with all their years, 510 positives (**1.02%**).
- Stratified by state × incident history (0 / 1 / 2+ incident-years), seed 42. Rebuild with `python make_sample.py final_dataset.parquet out.parquet`.
- Check against the full file: all 49 states present (max share gap 0.07 pts); no numeric column has a standardized mean difference above 0.03; missing-value rates within 1.1 pts; category shares within 1.3 pts; quartiles of traffic, speed and train counts match.

## Before you train
- **Drop `severity_target`.** It's the same thing as `incident` (`none` exactly when incident = 0), so it leaks the label.
- **Drop `crossingid`** as a feature; use it as the group key.
- **Split by crossing or by year**, not by random row. The same crossing appears up to 6 times, so a random split puts its other years into the test set. A clean option: train 2019–2022, test 2023–2024. Or use `GroupKFold` on `crossingid`.
- Treat the integer code columns as categorical for the FT-Transformer's categorical tokenizer, not as numbers.
- With ~1% positives, judge the model on PR-AUC / recall at a fixed alert budget, not accuracy (all-zeros gets 99%).
