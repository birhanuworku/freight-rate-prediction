from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "posted_rate"
ID_COL = "load_id"
CATEGORICAL = ["pickup", "delivery", "equipment", "route"]
NUMERIC_BASE = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "distance", "weight", "market_index", "quote_signal",
]
MODEL_COLUMNS = CATEGORICAL + NUMERIC_BASE + [
    "year", "month", "day", "dayofweek", "dayofyear"
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def build_features(frame: pd.DataFrame, numeric_defaults: dict[str, float] | None = None) -> pd.DataFrame:
    """Create a stable feature matrix for training and all inference files."""
    x = frame.copy()
    date = pd.to_datetime(x.get("date"), errors="coerce")
    if date.isna().any():
        raise ValueError("One or more rows have a missing or invalid date.")

    for col in ["pickup", "delivery", "equipment"]:
        if col not in x:
            x[col] = "Unknown"
        x[col] = x[col].fillna("Unknown").astype(str)

    x["route"] = x["pickup"] + "__TO__" + x["delivery"]

    for col in NUMERIC_BASE:
        if col not in x:
            x[col] = np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")
        # Zero/negative shipment weights are not physically plausible; treat them as missing.
        if col == "weight":
            x.loc[x[col] <= 0, col] = np.nan
        if numeric_defaults and col in numeric_defaults:
            x[col] = x[col].fillna(numeric_defaults[col])

    x["year"] = date.dt.year
    x["month"] = date.dt.month
    x["day"] = date.dt.day
    x["dayofweek"] = date.dt.dayofweek
    x["dayofyear"] = date.dt.dayofyear

    for col in CATEGORICAL:
        x[col] = x[col].fillna("Unknown").astype(str)
    return x[MODEL_COLUMNS]


def metric_dict(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": float(r2_score(y_true, pred)),
        "mean_absolute_percentage_error": float(
            np.mean(np.abs((np.asarray(y_true) - pred) / np.asarray(y_true)))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and validate freight-rate model, then generate submission files.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--iterations", type=int, default=700)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = read_csv(data_dir / "train_test.csv")
    validation = read_csv(data_dir / "validation.csv")
    template = read_csv(data_dir / "validation_predictions_template.csv")
    december = read_csv(data_dir / "december_chart_inputs.csv")

    required = {"date", TARGET, ID_COL}
    if not required.issubset(train.columns):
        raise ValueError(f"Training file must contain: {sorted(required)}")
    if train[TARGET].isna().any() or (train[TARGET] <= 0).any():
        raise ValueError("Target posted_rate has missing or non-positive values.")

    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    if train["date"].isna().any():
        raise ValueError("Training data contains invalid dates.")

    # Forward-in-time holdout: the latest calendar month is validation.
    latest_month = train["date"].max().to_period("M")
    holdout_mask = train["date"].dt.to_period("M") == latest_month
    dev = train.loc[~holdout_mask].copy()
    holdout = train.loc[holdout_mask].copy()
    if dev.empty or holdout.empty:
        raise ValueError("Could not create a chronological holdout split.")

    numeric_defaults = {}
    for col in NUMERIC_BASE:
        if col in dev.columns:
            numeric_defaults[col] = float(pd.to_numeric(dev[col], errors="coerce").median())
        else:
            numeric_defaults[col] = 0.0

    X_dev = build_features(dev, numeric_defaults)
    X_holdout = build_features(holdout, numeric_defaults)
    y_dev, y_holdout = dev[TARGET], holdout[TARGET]

    validation_model = CatBoostRegressor(
        iterations=args.iterations,
        depth=8,
        learning_rate=0.05,
        loss_function="MAE",
        eval_metric="MAE",
        l2_leaf_reg=5,
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
    )
    validation_model.fit(
        X_dev, y_dev,
        cat_features=CATEGORICAL,
        eval_set=(X_holdout, y_holdout),
        early_stopping_rounds=70,
        verbose=100,
    )
    holdout_pred = validation_model.predict(X_holdout)
    metrics = metric_dict(y_holdout, holdout_pred)
    metrics.update({
        "split_method": "chronological holdout: latest calendar month",
        "development_rows": int(len(dev)),
        "holdout_rows": int(len(holdout)),
        "holdout_month": str(latest_month),
        "best_iteration": int(max(validation_model.best_iteration_, 0) + 1),
    })
    (output_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"actual_rate": y_holdout, "predicted_rate": holdout_pred}).to_csv(
        output_dir / "holdout_predictions.csv", index=False
    )
    print("\nChronological holdout metrics:")
    print(json.dumps(metrics, indent=2))

    # Refit on all labeled data using the iteration count selected by the holdout.
    final_iterations = max(validation_model.best_iteration_, 0) + 1
    full_defaults = {}
    for col in NUMERIC_BASE:
        if col in train.columns:
            full_defaults[col] = float(pd.to_numeric(train[col], errors="coerce").median())
        else:
            full_defaults[col] = 0.0

    X_full = build_features(train, full_defaults)
    X_validation = build_features(validation, full_defaults)
    final_model = CatBoostRegressor(
        iterations=final_iterations,
        depth=8,
        learning_rate=0.05,
        loss_function="MAE",
        l2_leaf_reg=5,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    final_model.fit(X_full, train[TARGET], cat_features=CATEGORICAL, verbose=False)

    if "load_id" not in validation or "load_id" not in template:
        raise ValueError("Both validation and prediction template must contain load_id.")
    if validation[ID_COL].duplicated().any() or template[ID_COL].duplicated().any():
        raise ValueError("Duplicate load_id found.")
    if set(validation[ID_COL].astype(str)) != set(template[ID_COL].astype(str)):
        raise ValueError("Validation IDs and template IDs do not match exactly.")

    validation_rates = final_model.predict(X_validation)
    prediction_map = dict(zip(validation[ID_COL].astype(str), validation_rates))
    submission = template[["load_id"]].copy()
    submission["predicted_rate"] = submission["load_id"].astype(str).map(prediction_map)
    submission["predicted_rate"] = pd.to_numeric(submission["predicted_rate"], errors="coerce")
    if submission["predicted_rate"].isna().any() or (submission["predicted_rate"] <= 0).any():
        raise ValueError("Generated submission has missing or non-positive predictions.")
    submission.to_csv(output_dir / "validation_predictions.csv", index=False)

    X_december = build_features(december, full_defaults)
    december_result = december.copy()
    december_result["predicted_rate"] = final_model.predict(X_december)
    # Preserve the exact seven-column schema required by score.py.
    expected_december_columns = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    december_result[expected_december_columns].to_csv(output_dir / "december_chart_inputs.csv", index=False)

    final_model.save_model(str(output_dir / "freight_rate_model.cbm"))
    print(f"\nSaved {len(submission):,} predictions to {output_dir / 'validation_predictions.csv'}")
    print(f"Saved December predictions to {output_dir / 'december_chart_inputs.csv'}")
    print(f"Saved model to {output_dir / 'freight_rate_model.cbm'}")


if __name__ == "__main__":
    main()
