"""Tune and evaluate the FT-Transformer grade-crossing risk model.

Same protocol and outputs as collab-sol's scripts/train_mlp.py: tune with Optuna
over the expanding temporal folds (validation years 2020-2023), refit once on
2019-2023, and score the untouched 2024 test year. Adds ROC-AUC, Brier score and
a reliability plot.

Full run:    uv run python scripts/train_ft_transformer.py --dataset <path> --require-cuda
Smoke test:  uv run python scripts/train_ft_transformer.py --smoke
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "collab-sol"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "final_dataset.parquet")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="default: results/ft_transformer_<UTC timestamp>; an existing non-empty folder is refused",
)
parser.add_argument("--trials", type=int, default=20)
parser.add_argument("--max-epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--require-cuda", action="store_true")
parser.add_argument(
    "--smoke",
    action="store_true",
    help="tiny CPU-friendly run: 20%% of crossings, 1 trial, 2 epochs, small model",
)
args = parser.parse_args()

if args.smoke:
    args.trials, args.max_epochs, args.patience = 1, 2, 2
    if args.dataset == REPO_ROOT / "final_dataset.parquet":
        args.dataset = REPO_ROOT / "final_dataset_sample_50k.parquet"

DATASET_PATH = args.dataset.resolve()
if args.output_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.output_dir = REPO_ROOT / "results" / f"ft_transformer_{stamp}{'_smoke' if args.smoke else ''}"
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
# 1. Load the modeling dataset.
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from collab_sol import generate_risk_report, make_expanding_temporal_split, tune_model
from ft_transformer import (
    EXCLUDED_FEATURES,
    FTTransformerClassifier,
    make_preprocessor,
    save_calibration_report,
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

frame = pd.read_parquet(DATASET_PATH, columns=["crossingid", "asof_year", "incident", *FEATURE_COLUMNS])
if args.smoke:
    ids = frame["crossingid"].drop_duplicates().sample(frac=0.2, random_state=args.seed)
    frame = frame[frame["crossingid"].isin(ids)].reset_index(drop=True)
frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype(np.float32)
for column in CATEGORICAL_FEATURES:
    frame[column] = frame[column].astype("category")

yearly_summary = frame.groupby("asof_year")["incident"].agg(rows="size", crashes="sum", prevalence="mean")
print(yearly_summary)
print("Dataset:", DATASET_PATH)
print(f"Using {len(FEATURE_COLUMNS)} predictors: {len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical")
yearly_summary.to_csv(RUN_DIR / "class_balance_by_year.csv")

# %%
# 2. Temporal folds. The tuner never sees the holdout year.
X = frame[FEATURE_COLUMNS]
y = frame["incident"].astype(int)
split = make_expanding_temporal_split(frame, year_column="asof_year")
crossing_ids = frame["crossingid"].to_numpy()
del frame

fold_balance = pd.DataFrame(
    [
        {
            "fold": f"validate_{fold.validation_year}",
            "train_rows": len(fold.train_indices),
            "train_prevalence": float(y.iloc[fold.train_indices].mean()),
            "validation_rows": len(fold.validation_indices),
            "validation_prevalence": float(y.iloc[fold.validation_indices].mean()),
        }
        for fold in split.folds
    ]
    + [
        {
            "fold": f"test_{split.holdout_year}",
            "train_rows": len(split.development_indices),
            "train_prevalence": float(y.iloc[split.development_indices].mean()),
            "validation_rows": len(split.holdout_indices),
            "validation_prevalence": float(y.iloc[split.holdout_indices].mean()),
        }
    ]
)
print(fold_balance.to_string(index=False))
fold_balance.to_csv(RUN_DIR / "class_balance_by_fold.csv", index=False)


def new_preprocessor():
    return make_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, random_state=args.seed)


# %%
# 3. Model: search space and construction.
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
        "pos_weight": trial.suggest_float("pos_weight", 1.0, 50.0, log=True),
    }


def build_model(params, random_state):
    model = FTTransformerClassifier(
        n_cat_features=len(CATEGORICAL_FEATURES),
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
    model.fit_with_validation(X_train, y_train, X_valid, y_valid)
    key = json.dumps(model.tuned_params_, sort_keys=True)
    BEST_EPOCHS.setdefault(key, []).append(model.best_epoch_)


# %%
# 4. Tune on the validation folds with early stopping on validation PR-AUC.
study = tune_model(
    model_factory=build_model,
    suggest_params=suggest_params,
    X=X,
    y=y,
    split=split,
    preprocessor_factory=new_preprocessor,
    fit_model=fit_model,
    n_trials=args.trials,
    random_state=args.seed,
    study_name="ft_transformer_risk",
)
study.trials_dataframe().to_csv(RUN_DIR / "optuna_trials.csv", index=False)
print("Mean validation AP:", study.best_value)
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
model.fit(X_development, y.iloc[development_rows].to_numpy())

metrics = generate_risk_report(
    model,
    X_test,
    y.iloc[test_rows].to_numpy(),
    output_dir=RUN_DIR,
    row_ids=crossing_ids[test_rows],
)
calibration = save_calibration_report(RUN_DIR)

pd.DataFrame(model.history_).to_csv(RUN_DIR / "final_training_history.csv", index=False)
joblib.dump(preprocessor, RUN_DIR / "preprocessor.joblib")
torch.save(model.network_.state_dict(), RUN_DIR / "model.pt")

run_metadata = {
    "model": "FTTransformerClassifier",
    "dataset_path": str(DATASET_PATH),
    "smoke_run": args.smoke,
    "holdout_year": split.holdout_year,
    "feature_count": len(FEATURE_COLUMNS),
    "numeric_features": len(NUMERIC_FEATURES),
    "categorical_features": len(CATEGORICAL_FEATURES),
    "excluded_features": EXCLUDED_FEATURES,
    "best_params": study.best_params,
    "mean_validation_average_precision": study.best_value,
    "validation_fold_scores": study.best_trial.user_attrs.get("fold_scores"),
    "best_epoch_per_fold": best_epochs,
    "final_refit_epochs": final_epochs,
    "device": str(model.device),
    "n_trials": args.trials,
    "max_epochs": args.max_epochs,
    "patience": args.patience,
    "batch_size": args.batch_size,
    "seed": args.seed,
    "test_metrics": {**metrics.to_dict(), **calibration},
    "torch": torch.__version__,
    "python": platform.python_version(),
    "command": " ".join(sys.argv),
}
(RUN_DIR / "run.json").write_text(json.dumps(run_metadata, indent=2))

print(pd.DataFrame([{**metrics.to_dict(), **calibration}]).T)
print("Saved results to:", RUN_DIR)
