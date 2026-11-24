"""Train and evaluate the first squash shot-outcome classifier.

Run from the project root:
    python train_model.py
"""

from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.squash_model import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    load_training_data, prepare_features
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "squash_synthetic_30_rallies.xlsx"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "squash_shot_model.joblib"


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    raw = load_training_data(DATA_PATH)

    # The target is direct winning shots, not simply the last shot in a rally.
    # Crucially, all outcome/future columns are excluded from the input features.
    labels = raw["Direct winning shot?"].astype(str).str.strip().str.lower()
    y = labels.map({"yes": 1, "no": 0})
    valid = y.notna()
    raw = raw.loc[valid].copy()
    y = y.loc[valid].astype(int)

    # Keep complete rallies together: no shots from one rally appear in both sets.
    groups = raw["Rally ID"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=17)
    train_idx, test_idx = next(splitter.split(raw, y, groups=groups))

    X = prepare_features(raw)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    preprocess = ColumnTransformer(
        transformers=[
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_FEATURES),
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=17,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    metrics = {
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_rallies": int(groups.iloc[train_idx].nunique()),
        "test_rallies": int(groups.iloc[test_idx].nunique()),
        "positive_rate_test": float(y_test.mean()),
        "confusion_matrix_labels_0_then_1": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": report,
    }
    if y_test.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
        metrics["average_precision"] = float(
            average_precision_score(y_test, probabilities)
        )

    joblib.dump(model, MODEL_PATH)
    (ARTIFACTS / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"Saved model to: {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved evaluation to: artifacts/evaluation_metrics.json")
    print("\nEvaluation (held-out rallies):")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (rows=true 0/1, columns=predicted 0/1):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("\nReminder: this workbook is synthetic. These scores do not measure")
    print("real-world squash performance.")


if __name__ == "__main__":
    main()
