import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CONFIG_PATH = Path("data/processed/ml_ready/feature_config.json")
SPLIT_DIR = Path("data/processed/ml_ready/splits")

OUTPUT_DIR = Path("models/baselines")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = Path("reports/ml")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV_PATH = REPORT_DIR / "baseline_model_summary.csv"
SUMMARY_MD_PATH = REPORT_DIR / "baseline_model_report.md"
BEST_MODEL_PATH = OUTPUT_DIR / "best_baseline_model.joblib"

RANDOM_STATE = 42


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def load_split(split_type):
    if split_type == "random":
        train = pd.read_csv(SPLIT_DIR / "random_train.csv")
        val = pd.read_csv(SPLIT_DIR / "random_val.csv")
        test = pd.read_csv(SPLIT_DIR / "random_test.csv")

    elif split_type == "group_city":
        train = pd.read_csv(SPLIT_DIR / "group_city_train.csv")
        val = pd.read_csv(SPLIT_DIR / "group_city_val.csv")
        test = pd.read_csv(SPLIT_DIR / "group_city_test.csv")

    else:
        raise ValueError("split_type must be either 'random' or 'group_city'")

    return train, val, test


def build_preprocessor(categorical_features, numeric_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def get_models():
    return {
        "dummy_most_frequent": DummyClassifier(
            strategy="most_frequent"
        ),

        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_model(model, X, y):
    preds = model.predict(X)

    labels = ["low", "medium", "high"]

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro"),
        "weighted_f1": f1_score(y, preds, average="weighted"),
        "classification_report": classification_report(
            y,
            preds,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "classification_report_text": classification_report(
            y,
            preds,
            labels=labels,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y,
            preds,
            labels=labels,
        ).tolist(),
        "labels": labels,
    }

    return metrics


def train_single_model(
    split_type,
    model_name,
    estimator,
    preprocessor,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):
    print(f"\nTraining model: {split_type} / {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)

    val_metrics = evaluate_model(pipeline, X_val, y_val)
    test_metrics = evaluate_model(pipeline, X_test, y_test)

    print("Validation accuracy:", round(val_metrics["accuracy"], 4))
    print("Validation macro F1:", round(val_metrics["macro_f1"], 4))
    print("Validation weighted F1:", round(val_metrics["weighted_f1"], 4))

    print("Test accuracy:", round(test_metrics["accuracy"], 4))
    print("Test macro F1:", round(test_metrics["macro_f1"], 4))
    print("Test weighted F1:", round(test_metrics["weighted_f1"], 4))

    model_path = OUTPUT_DIR / f"{split_type}_{model_name}.joblib"
    joblib.dump(pipeline, model_path)

    row = {
        "split_type": split_type,
        "model": model_name,
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_weighted_f1": val_metrics["weighted_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "model_path": str(model_path),
        "val_confusion_matrix": val_metrics["confusion_matrix"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "val_classification_report_text": val_metrics["classification_report_text"],
        "test_classification_report_text": test_metrics["classification_report_text"],
    }

    return row, pipeline


def train_for_split(split_type):
    config = load_config()

    target_col = config["target_column"]
    categorical_features = config["categorical_features"]
    numeric_features = config["numeric_features"]

    train_df, val_df, test_df = load_split(split_type)

    X_train = train_df[categorical_features + numeric_features]
    y_train = train_df[target_col]

    X_val = val_df[categorical_features + numeric_features]
    y_val = val_df[target_col]

    X_test = test_df[categorical_features + numeric_features]
    y_test = test_df[target_col]

    print(f"\n================ TRAINING SPLIT: {split_type} ================\n")
    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    preprocessor = build_preprocessor(categorical_features, numeric_features)
    models = get_models()

    split_rows = []
    trained_models = {}

    for model_name, estimator in models.items():
        row, pipeline = train_single_model(
            split_type=split_type,
            model_name=model_name,
            estimator=estimator,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )

        split_rows.append(row)
        trained_models[f"{split_type}_{model_name}"] = pipeline

    return split_rows, trained_models


def write_single_markdown_report(summary_df):
    lines = []

    lines.append("# NariSafe Baseline Model Report")
    lines.append("")
    lines.append("## Important notes")
    lines.append("")
    lines.append("- `risk_level` is a rule-based proxy label, not real incident-level ground truth.")
    lines.append("- `risk_score` is not used as an input feature to avoid target leakage.")
    lines.append("- Random split is useful for quick baseline checking.")
    lines.append("- Group-city split is more realistic because it tests unseen cities.")
    lines.append("- Macro F1 is more important than plain accuracy because the `high` risk class is small.")
    lines.append("")

    lines.append("## Model summary")
    lines.append("")
    lines.append("| Split | Model | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Test Weighted F1 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['split_type']} "
            f"| {row['model']} "
            f"| {row['val_accuracy']:.4f} "
            f"| {row['val_macro_f1']:.4f} "
            f"| {row['test_accuracy']:.4f} "
            f"| {row['test_macro_f1']:.4f} "
            f"| {row['test_weighted_f1']:.4f} |"
        )

    lines.append("")

    lines.append("## Detailed results")
    lines.append("")

    for _, row in summary_df.iterrows():
        lines.append(f"### {row['split_type']} / {row['model']}")
        lines.append("")

        lines.append("Validation confusion matrix:")
        lines.append("")
        lines.append("Labels order: `low`, `medium`, `high`")
        lines.append("")
        lines.append("```text")
        lines.append(str(row["val_confusion_matrix"]))
        lines.append("```")
        lines.append("")

        lines.append("Validation classification report:")
        lines.append("")
        lines.append("```text")
        lines.append(row["val_classification_report_text"])
        lines.append("```")
        lines.append("")

        lines.append("Test confusion matrix:")
        lines.append("")
        lines.append("Labels order: `low`, `medium`, `high`")
        lines.append("")
        lines.append("```text")
        lines.append(str(row["test_confusion_matrix"]))
        lines.append("```")
        lines.append("")

        lines.append("Test classification report:")
        lines.append("")
        lines.append("```text")
        lines.append(row["test_classification_report_text"])
        lines.append("```")
        lines.append("")

    with open(SUMMARY_MD_PATH, "w") as f:
        f.write("\n".join(lines))


def main():
    all_rows = []
    all_trained_models = {}

    for split_type in ["random", "group_city"]:
        rows, trained_models = train_for_split(split_type)
        all_rows.extend(rows)
        all_trained_models.update(trained_models)

    summary_df = pd.DataFrame(all_rows)

    summary_df_for_csv = summary_df.drop(
        columns=[
            "val_confusion_matrix",
            "test_confusion_matrix",
            "val_classification_report_text",
            "test_classification_report_text",
        ]
    )

    summary_df_for_csv.to_csv(SUMMARY_CSV_PATH, index=False)

    summary_df_sorted = summary_df.sort_values(
        by=["split_type", "test_macro_f1"],
        ascending=[True, False]
    )

    write_single_markdown_report(summary_df_sorted)

    # Pick best model based on group-city test macro F1.
    # This is the honest benchmark.
    group_city_results = summary_df[summary_df["split_type"] == "group_city"].copy()

    best_row = group_city_results.sort_values(
        by="test_macro_f1",
        ascending=False
    ).iloc[0]

    best_model_key = f"{best_row['split_type']}_{best_row['model']}"
    best_model = all_trained_models[best_model_key]

    joblib.dump(best_model, BEST_MODEL_PATH)

    print("\n================ BASELINE TRAINING COMPLETE ================\n")

    print("CSV summary saved to:")
    print(SUMMARY_CSV_PATH)

    print("\nSingle markdown report saved to:")
    print(SUMMARY_MD_PATH)

    print("\nBest model saved to:")
    print(BEST_MODEL_PATH)

    print("\nBest model selected using group-city test macro F1:")
    print(best_row[[
        "split_type",
        "model",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "model_path",
    ]])

    print("\nModel summary:")
    print(
        summary_df_for_csv.sort_values(
            by="test_macro_f1",
            ascending=False
        )
    )


if __name__ == "__main__":
    main()