"""Tune and evaluate the three-class conditional-severity FT-Transformer.

Same protocol and outputs as collab-sol's scripts/train_severity_mlp.py: train on
crossing-years that had an incident, predict the most severe outcome
(property_damage_only / injury / fatal), tune with Optuna on mean validation
multiclass log loss over the expanding temporal folds (validation years
2020-2023), refit once on 2019-2023 and score the untouched 2024 test year with
collab-sol's severity report.

Extra output: ``severity_scores_all_crossings.csv`` scores every test-year
crossing, including the ones with no incident, as "if a crash happens here, how
severe". Multiply by P(incident) from a likelihood model to get unconditional
severity risk (P(fatal) = P(incident) x P(fatal | incident)). Rows with no
incident were never seen in training, so read those scores as an extrapolation.

Full run:    uv run python scripts/train_severity_ft_transformer.py --dataset <path> --require-cuda
Smoke test:  uv run python scripts/train_severity_ft_transformer.py --smoke
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Print progress line by line even when output is redirected to a log file.
sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "collab-sol"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "final_dataset.parquet")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="default: results/severity_ft_transformer_<UTC timestamp>; an existing non-empty folder is refused",
)
parser.add_argument("--trials", type=int, default=20)
parser.add_argument("--max-epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--require-cuda", action="store_true")
parser.add_argument(
    "--smoke",
    action="store_true",
    help="tiny CPU-friendly run: whole severity cohort of the sample, 1 trial, 2 epochs, small model",
)
args = parser.parse_args()

if args.smoke:
    args.trials, args.max_epochs, args.patience = 1, 2, 2
    if args.dataset == REPO_ROOT / "final_dataset.parquet":
        args.dataset = REPO_ROOT / "final_dataset_sample_50k.parquet"

DATASET_PATH = args.dataset.resolve()
if args.output_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.output_dir = REPO_ROOT / "results" / f"severity_ft_transformer_{stamp}{'_smoke' if args.smoke else ''}"
RUN_DIR = args.output_dir.resolve()

if not PACKAGE_ROOT.is_dir():
    raise FileNotFoundError(
        f"collab-sol not found at {PACKAGE_ROOT}. "
        "Run: git clone https://github.com/helloimfrei/collab-sol.git collab-sol"
    )
if not DATASET_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH} (pass --dataset)")
if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
    raise FileExistsError(f"{RUN_DIR} is not empty; earlier runs are never overwritten.")

for path in (PACKAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
RUN_DIR.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load the severity cohort: crossing-years with an incident.
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from collab_sol import (
    SEVERITY_CLASSES,
    SEVERITY_CODES,
    SEVERITY_WEIGHTS,
    generate_severity_report,
    make_expanding_temporal_split,
    multiclass_log_loss,
    tune_model,
)
from collab_sol.evaluate import expected_severity
from ft_transformer import (
    EXCLUDED_FEATURES,
    FTTransformerClassifier,
    make_preprocessor,
    split_feature_columns,
)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
elif args.require_cuda:
    raise RuntimeError("--require-cuda was given but torch.cuda.is_available() is False.")
else:
    print("CUDA is unavailable; running on CPU.")

NUMERIC_FEATURES, CATEGORICAL_FEATURES = split_feature_columns(pq.read_schema(DATASET_PATH).names)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LOAD_COLUMNS = ["crossingid", "asof_year", "incident", "severity_target", *FEATURE_COLUMNS]


def prepare(frame):
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype(np.float32)
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("category")
    return frame


frame = prepare(pd.read_parquet(DATASET_PATH, columns=LOAD_COLUMNS, filters=[("incident", "=", 1)]))
frame = frame.reset_index(drop=True)
invalid = ~frame["severity_target"].isin(SEVERITY_CODES)
if invalid.any():
    raise ValueError(f"Unexpected severity targets: {sorted(frame.loc[invalid, 'severity_target'].unique())}")
severity_codes = frame["severity_target"].astype(str).map(SEVERITY_CODES).astype(int)

yearly_summary = (
    frame.groupby(["asof_year", "severity_target"], observed=True)
    .size()
    .unstack(fill_value=0)
    .reindex(columns=list(SEVERITY_CLASSES), fill_value=0)
)
yearly_summary.to_csv(RUN_DIR / "class_balance_by_year.csv")
print(yearly_summary)
print("Dataset:", DATASET_PATH)
print(f"Using {len(FEATURE_COLUMNS)} predictors: {len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical")

# %%
# 2. Temporal folds. The tuner never sees the holdout year.
X = frame[FEATURE_COLUMNS]
y = severity_codes
split = make_expanding_temporal_split(frame, year_column="asof_year")
crossing_ids = frame["crossingid"].to_numpy()
del frame


def class_counts(indices):
    counts = np.bincount(y.iloc[indices], minlength=len(SEVERITY_CLASSES))
    return {name: int(count) for name, count in zip(SEVERITY_CLASSES, counts)}


fold_balance = pd.DataFrame(
    [
        {"fold": f"validate_{fold.validation_year}", "part": part, **class_counts(rows)}
        for fold in split.folds
        for part, rows in (("train", fold.train_indices), ("validation", fold.validation_indices))
    ]
    + [
        {"fold": f"test_{split.holdout_year}", "part": part, **class_counts(rows)}
        for part, rows in (("train", split.development_indices), ("test", split.holdout_indices))
    ]
)
print(fold_balance.to_string(index=False))
fold_balance.to_csv(RUN_DIR / "class_balance_by_fold.csv", index=False)


def new_preprocessor():
    return make_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, random_state=args.seed)


# %%
# 3. Model: search space and construction. Unweighted cross-entropy, as in the
# team's severity MLP: the objective is log loss, which class weights would distort.
def suggest_params(trial):
    if args.smoke:
        # One-choice categoricals keep these in study.best_params like a real run.
        n_blocks = trial.suggest_categorical("n_blocks", [1])
        d_block = trial.suggest_categorical("d_block", [64])
    else:
        n_blocks = trial.suggest_int("n_blocks", 1, 4)
        d_block = trial.suggest_categorical("d_block", [64, 128, 192, 256])
    return {
        "n_blocks": n_blocks,
        "d_block": d_block,
        "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.5),
        "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.5),
        "residual_dropout": trial.suggest_float("residual_dropout", 0.0, 0.2),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }


def build_model(params, random_state):
    model = FTTransformerClassifier(
        n_cat_features=len(CATEGORICAL_FEATURES),
        n_classes=len(SEVERITY_CLASSES),
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        random_state=random_state,
        **params,
    )
    model.tuned_params_ = dict(params)
    return model


# Best epoch per fold for every parameter set, so the final refit (which has no
# validation data) can train for the epoch count the tuning found.
BEST_EPOCHS: dict[str, list[int]] = {}


def fit_model(model, X_train, y_train, X_valid, y_valid):
    key = json.dumps(model.tuned_params_, sort_keys=True)
    trial_number = len(BEST_EPOCHS) + (key not in BEST_EPOCHS)
    fold_number = len(BEST_EPOCHS.get(key, [])) + 1
    model.log_prefix = f"trial {trial_number}/{args.trials} | fold {fold_number}/{len(split.folds)}"
    print(
        f"{model.log_prefix} | {len(y_train):,} train rows | "
        f"n_blocks={model.n_blocks} d_block={model.d_block}"
    )
    model.fit_with_validation(X_train, y_train, X_valid, y_valid)
    BEST_EPOCHS.setdefault(key, []).append(model.best_epoch_)


# %%
# 4. Tune on the validation folds with early stopping on validation log loss.
study = tune_model(
    model_factory=build_model,
    suggest_params=suggest_params,
    X=X,
    y=y,
    split=split,
    preprocessor_factory=new_preprocessor,
    fit_model=fit_model,
    scorer=multiclass_log_loss,
    direction="minimize",
    metric_name="log loss",
    n_trials=args.trials,
    random_state=args.seed,
    study_name="conditional_severity_ft_transformer",
    cache_preprocessing=True,
    verbose=True,
)
study.trials_dataframe().to_csv(RUN_DIR / "optuna_trials.csv", index=False)
print("Mean validation log loss:", study.best_value)
print("Best parameters:", study.best_params)

# %%
# 5. Refit the winner on the development years and evaluate the holdout once.
best_epochs = BEST_EPOCHS[json.dumps(study.best_params, sort_keys=True)]
final_epochs = max(1, int(round(float(np.median(best_epochs)))))

development_rows = split.development_indices
test_rows = split.holdout_indices

preprocessor = new_preprocessor()
X_development = preprocessor.fit_transform(X.iloc[development_rows], y.iloc[development_rows])
X_test = preprocessor.transform(X.iloc[test_rows])

model = build_model(study.best_params, random_state=args.seed)
model.fixed_epochs = final_epochs
model.log_prefix = "final refit"
model.fit(X_development, y.iloc[development_rows].to_numpy())

metrics = generate_severity_report(
    model,
    X_test,
    y.iloc[test_rows].to_numpy(),
    output_dir=RUN_DIR,
    row_ids=crossing_ids[test_rows],
    class_names=SEVERITY_CLASSES,
    class_weights=SEVERITY_WEIGHTS,
)

# %%
# 6. Score every test-year crossing, crash or not, for the two-part combination.
all_test = prepare(
    pd.read_parquet(DATASET_PATH, columns=LOAD_COLUMNS, filters=[("asof_year", "=", split.holdout_year)])
)
all_probabilities = model.predict_proba(preprocessor.transform(all_test[FEATURE_COLUMNS]))
all_scores = pd.DataFrame(
    {
        "row_id": all_test["crossingid"].to_numpy(),
        "asof_year": all_test["asof_year"].to_numpy(),
        "incident": all_test["incident"].to_numpy(),
        "actual_class": all_test["severity_target"].astype(str).to_numpy(),
        "severity_score": expected_severity(all_probabilities, class_weights=SEVERITY_WEIGHTS),
        **{f"probability_{name}": all_probabilities[:, i] for i, name in enumerate(SEVERITY_CLASSES)},
    }
)
all_scores.to_csv(RUN_DIR / "severity_scores_all_crossings.csv", index=False)
print(f"Scored all {len(all_scores):,} crossings in {split.holdout_year} ({int(all_scores['incident'].sum()):,} with an incident)")
del all_test

pd.DataFrame(model.history_).to_csv(RUN_DIR / "final_training_history.csv", index=False)
joblib.dump(preprocessor, RUN_DIR / "preprocessor.joblib")
torch.save(model.network_.state_dict(), RUN_DIR / "model.pt")

run_metadata = {
    "task": "conditional_severity",
    "model": "FTTransformerClassifier",
    "dataset_path": str(DATASET_PATH),
    "smoke_run": args.smoke,
    "holdout_year": split.holdout_year,
    "feature_count": len(FEATURE_COLUMNS),
    "numeric_features": len(NUMERIC_FEATURES),
    "categorical_features": len(CATEGORICAL_FEATURES),
    "excluded_features": EXCLUDED_FEATURES,
    "severity_classes": SEVERITY_CLASSES,
    "severity_weights": SEVERITY_WEIGHTS,
    "severity_score_definition": "P(property_damage_only) + 2 * P(injury) + 3 * P(fatal)",
    "best_params": study.best_params,
    "mean_validation_multiclass_log_loss": study.best_value,
    "validation_fold_scores": study.best_trial.user_attrs.get("fold_scores"),
    "best_epoch_per_fold": best_epochs,
    "final_refit_epochs": final_epochs,
    "device": str(model.device),
    "n_trials": args.trials,
    "max_epochs": args.max_epochs,
    "patience": args.patience,
    "batch_size": args.batch_size,
    "seed": args.seed,
    "test_metrics": metrics.to_dict(),
    "torch": torch.__version__,
    "python": platform.python_version(),
    "command": " ".join(sys.argv),
}
(RUN_DIR / "run.json").write_text(json.dumps(run_metadata, indent=2))

print(pd.DataFrame([metrics.to_dict()]).T)
print("Saved results to:", RUN_DIR)
