"""
Threshold tuning for the no-location HistGradientBoosting model.

Strategy
--------
Train the same HistGradientBoostingClassifier pipeline on group_city_train.
Get P(high) from predict_proba on the validation set.
For each candidate threshold t:
    predict "high"   if P(high) >= t
    predict argmax({low, medium})  otherwise
Evaluate accuracy / macro-F1 / weighted-F1 / high precision-recall-F1.
Select the threshold that maximises macro-F1 subject to:
  (a) high_recall > baseline high_recall (threshold = 0.50)
  (b) macro_F1 drop from baseline <= 0.05
  If no threshold satisfies (a)+(b), fall back to highest high_recall.

Outputs
-------
  reports/ml_no_location/high_threshold_tuning.csv
  reports/ml_no_location/high_threshold_tuning_report.md
  models/no_location_baselines/best_no_location_threshold_model.joblib
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "data/processed/ml_ready_no_location/feature_config_no_location.json"
TRAIN_PATH  = ROOT / "data/processed/ml_ready_no_location/splits/group_city_train.csv"
VAL_PATH    = ROOT / "data/processed/ml_ready_no_location/splits/group_city_val.csv"
TEST_PATH   = ROOT / "data/processed/ml_ready_no_location/splits/group_city_test.csv"
REPORT_DIR  = ROOT / "reports/ml_no_location"
MODEL_DIR   = ROOT / "models/no_location_baselines"
CSV_OUT     = REPORT_DIR / "high_threshold_tuning.csv"
MD_OUT      = REPORT_DIR / "high_threshold_tuning_report.md"
MODEL_OUT   = MODEL_DIR / "best_no_location_threshold_model.joblib"

THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
BASELINE_T = 0.50
MAX_F1_DROP = 0.05   # tolerated macro-F1 degradation from baseline


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_xy(path: Path, feature_cols: list[str], target: str):
    df = pd.read_csv(path)
    return df[feature_cols], df[target]


def build_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    """Replicate the exact pipeline used in the baseline training."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    clf = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=200,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", clf)])


def threshold_predict(proba: np.ndarray, classes: np.ndarray, threshold: float) -> np.ndarray:
    """
    Assign 'high' if P(high) >= threshold; otherwise take argmax of
    {low, medium} probabilities.
    """
    high_idx   = np.where(classes == "high")[0][0]
    other_mask = np.ones(len(classes), dtype=bool)
    other_mask[high_idx] = False
    other_classes = classes[other_mask]

    preds = []
    for row in proba:
        if row[high_idx] >= threshold:
            preds.append("high")
        else:
            other_probs = row[other_mask]
            preds.append(other_classes[np.argmax(other_probs)])
    return np.array(preds)


def evaluate(y_true, y_pred, classes):
    acc = accuracy_score(y_true, y_pred)
    mac = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    wt  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    high_idx = list(classes).index("high")
    return {
        "accuracy":        round(acc, 6),
        "macro_f1":        round(mac, 6),
        "weighted_f1":     round(wt,  6),
        "high_precision":  round(p[high_idx], 6),
        "high_recall":     round(r[high_idx], 6),
        "high_f1":         round(f[high_idx], 6),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load config ──────────────────────────────────────────────────────────
    with open(CONFIG_PATH) as fh:
        cfg = json.load(fh)
    target      = cfg["target_column"]
    num_cols    = cfg["numeric_features"]
    cat_cols    = cfg["categorical_features"]
    feature_cols = num_cols + cat_cols

    # ── Load data ────────────────────────────────────────────────────────────
    X_train, y_train = load_xy(TRAIN_PATH, feature_cols, target)
    X_val,   y_val   = load_xy(VAL_PATH,   feature_cols, target)
    X_test,  y_test  = load_xy(TEST_PATH,  feature_cols, target)

    print(f"Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")
    print(f"Train class dist:\n{y_train.value_counts().to_string()}")
    print(f"Val   class dist:\n{y_val.value_counts().to_string()}")
    print(f"Test  class dist:\n{y_test.value_counts().to_string()}\n")

    # ── Train pipeline ───────────────────────────────────────────────────────
    print("Training HistGradientBoostingClassifier pipeline on group_city_train …")
    pipeline = build_pipeline(num_cols, cat_cols)
    pipeline.fit(X_train, y_train)
    classes = pipeline.classes_
    print(f"Classes: {classes}\n")

    # ── Validation probabilities ─────────────────────────────────────────────
    val_proba = pipeline.predict_proba(X_val)

    # ── Threshold sweep on validation ────────────────────────────────────────
    print(f"{'Threshold':>10}  {'Accuracy':>10}  {'Macro-F1':>10}  "
          f"{'High-P':>8}  {'High-R':>8}  {'High-F1':>8}")
    print("-" * 68)

    rows = []
    for t in THRESHOLDS:
        y_pred = threshold_predict(val_proba, classes, t)
        m = evaluate(y_val, y_pred, list(classes))
        m["threshold"] = t
        rows.append(m)
        print(
            f"  t={t:.2f}     {m['accuracy']:.4f}      {m['macro_f1']:.4f}    "
            f"  {m['high_precision']:.4f}   {m['high_recall']:.4f}   {m['high_f1']:.4f}"
        )

    sweep_df = pd.DataFrame(rows)[[
        "threshold", "accuracy", "macro_f1", "weighted_f1",
        "high_precision", "high_recall", "high_f1",
    ]]

    # ── Select best threshold ────────────────────────────────────────────────
    baseline_row   = sweep_df[sweep_df.threshold == BASELINE_T].iloc[0]
    baseline_recall = baseline_row["high_recall"]
    baseline_macro  = baseline_row["macro_f1"]

    # Candidates: improved recall and acceptable macro-F1 drop
    candidates = sweep_df[
        (sweep_df["high_recall"] > baseline_recall) &
        (sweep_df["macro_f1"]   >= baseline_macro - MAX_F1_DROP)
    ]

    if candidates.empty:
        # Fall back: best high_recall
        best_row = sweep_df.loc[sweep_df["high_recall"].idxmax()]
        selection_note = (
            "No threshold satisfied both recall-improvement and macro-F1 constraints. "
            "Selected by highest high_recall."
        )
    else:
        # Among candidates, prefer highest macro_f1
        best_row = candidates.loc[candidates["macro_f1"].idxmax()]
        selection_note = (
            f"Selected threshold that improves high_recall over baseline "
            f"({baseline_recall:.4f}) while keeping macro-F1 drop ≤ {MAX_F1_DROP}; "
            "tie-broken by highest macro-F1."
        )

    best_t = float(best_row["threshold"])
    print(f"\nBaseline (t=0.50) — macro-F1: {baseline_macro:.4f}, high recall: {baseline_recall:.4f}")
    print(f"Selected threshold : {best_t}")
    print(f"Selection note     : {selection_note}\n")

    # ── Final evaluation on validation ───────────────────────────────────────
    val_pred_best = threshold_predict(val_proba, classes, best_t)
    val_metrics   = evaluate(y_val, val_pred_best, list(classes))

    print("── Validation metrics at selected threshold ──")
    for k, v in val_metrics.items():
        print(f"  {k:<22}: {v:.4f}")

    val_report = classification_report(y_val, val_pred_best, labels=list(classes), digits=4)
    val_cm     = confusion_matrix(y_val, val_pred_best, labels=list(classes))
    print(f"\n{val_report}")
    print("Confusion matrix (rows=true, cols=pred) — classes:", list(classes))
    print(pd.DataFrame(val_cm, index=classes, columns=classes).to_string())

    # ── Final evaluation on test ─────────────────────────────────────────────
    test_proba    = pipeline.predict_proba(X_test)
    test_pred     = threshold_predict(test_proba, classes, best_t)
    test_metrics  = evaluate(y_test, test_pred, list(classes))

    print("\n── Test metrics at selected threshold ──")
    for k, v in test_metrics.items():
        print(f"  {k:<22}: {v:.4f}")

    test_report = classification_report(y_test, test_pred, labels=list(classes), digits=4)
    test_cm     = confusion_matrix(y_test, test_pred, labels=list(classes))
    print(f"\n{test_report}")
    print("Confusion matrix (rows=true, cols=pred) — classes:", list(classes))
    print(pd.DataFrame(test_cm, index=classes, columns=classes).to_string())

    # ── Save sweep CSV ───────────────────────────────────────────────────────
    sweep_df.to_csv(CSV_OUT, index=False)
    print(f"\nSaved sweep table → {CSV_OUT}")

    # ── Save model bundle ────────────────────────────────────────────────────
    bundle = {
        "pipeline":           pipeline,
        "selected_threshold": best_t,
        "class_labels":       list(classes),
        "feature_config":     cfg,
    }
    joblib.dump(bundle, MODEL_OUT)
    print(f"Saved model bundle → {MODEL_OUT}")

    # ── Write markdown report ────────────────────────────────────────────────
    _write_report(
        sweep_df, best_t, selection_note,
        baseline_macro, baseline_recall,
        val_metrics, val_report, val_cm,
        test_metrics, test_report, test_cm,
        classes,
        X_train, X_val, X_test, y_train, y_val, y_test,
    )
    print(f"Saved report       → {MD_OUT}")


def _write_report(
    sweep_df, best_t, selection_note,
    baseline_macro, baseline_recall,
    val_metrics, val_report, val_cm,
    test_metrics, test_report, test_cm,
    classes,
    X_train, X_val, X_test, y_train, y_val, y_test,
):
    val_cm_df  = pd.DataFrame(val_cm,  index=classes, columns=classes)
    test_cm_df = pd.DataFrame(test_cm, index=classes, columns=classes)

    sweep_md = sweep_df.to_markdown(index=False, floatfmt=".4f")

    lines = [
        "# NariSafe — High-Risk Threshold Tuning Report",
        "",
        "## Model",
        "HistGradientBoostingClassifier (learning_rate=0.08, max_iter=200)  ",
        "Pipeline includes median imputation + standard scaling (numeric) and "
        "most-frequent imputation + one-hot encoding (categorical).  ",
        "No location features (city, latitude, longitude excluded).  ",
        "Trained on group-city split training set.",
        "",
        "## Dataset",
        f"| Split | Rows | High | Medium | Low |",
        f"|---|---|---|---|---|",
        f"| Train | {len(X_train):,} | {(y_train=='high').sum():,} | "
        f"{(y_train=='medium').sum():,} | {(y_train=='low').sum():,} |",
        f"| Val   | {len(X_val):,} | {(y_val=='high').sum():,} | "
        f"{(y_val=='medium').sum():,} | {(y_val=='low').sum():,} |",
        f"| Test  | {len(X_test):,} | {(y_test=='high').sum():,} | "
        f"{(y_test=='medium').sum():,} | {(y_test=='low').sum():,} |",
        "",
        "## Threshold Sweep — Validation Set",
        "",
        sweep_md,
        "",
        "## Threshold Selection",
        "",
        f"Baseline threshold : **0.50**  ",
        f"Baseline macro-F1  : {baseline_macro:.4f}  ",
        f"Baseline high recall: {baseline_recall:.4f}  ",
        f"Max tolerated macro-F1 drop: {MAX_F1_DROP}  ",
        "",
        f"**Selected threshold: {best_t}**  ",
        f"Rationale: {selection_note}",
        "",
        "## Validation Metrics at Selected Threshold",
        "",
        "| Metric | Value |",
        "|---|---|",
    ] + [
        f"| {k} | {v:.4f} |" for k, v in val_metrics.items()
    ] + [
        "",
        "### Validation Classification Report",
        "```",
        val_report,
        "```",
        "",
        "### Validation Confusion Matrix",
        val_cm_df.to_markdown(),
        "",
        "## Test Metrics at Selected Threshold",
        "",
        "| Metric | Value |",
        "|---|---|",
    ] + [
        f"| {k} | {v:.4f} |" for k, v in test_metrics.items()
    ] + [
        "",
        "### Test Classification Report",
        "```",
        test_report,
        "```",
        "",
        "### Test Confusion Matrix",
        test_cm_df.to_markdown(),
        "",
        "## Saved Artifacts",
        f"- `{CSV_OUT.relative_to(ROOT)}` — full threshold sweep table",
        f"- `{MD_OUT.relative_to(ROOT)}` — this report",
        f"- `{MODEL_OUT.relative_to(ROOT)}` — bundle: pipeline + selected_threshold + "
        "class_labels + feature_config",
        "",
        "## Usage Notes",
        "- Load the bundle with `joblib.load(...)` which returns a dict.",
        "- Call `bundle['pipeline'].predict_proba(X)` to get class probabilities.",
        "- Apply `bundle['selected_threshold']` on the 'high' class column to get "
        "threshold-adjusted predictions.",
        "- `bundle['class_labels']` gives the class order matching predict_proba columns.",
        "- Do **not** use `risk_score` as an input feature (target leakage).",
        "- Location features (city, latitude, longitude) are excluded from this model.",
    ]

    MD_OUT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
