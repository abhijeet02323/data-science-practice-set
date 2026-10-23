"""
Supervised Learning Model Implementation
Problem: Predict Titanic passenger survival (binary classification).
Dataset: Public Titanic CSV from the seaborn-data GitHub repository.
IMPORTANT: Run this script to compute the real metrics and plots. The report's
illustrative numbers/figure must not be represented as actual experiment results.
"""

from pathlib import Path
import urllib.request
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42
OUTPUT_DIR = Path("supervised_learning_outputs")
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"

def load_data():
    """Download the public dataset and load it into a DataFrame."""
    csv_path = OUTPUT_DIR / "titanic.csv"
    if not csv_path.exists():
        print("Downloading Titanic dataset...")
        urllib.request.urlretrieve(DATA_URL, csv_path)
    df = pd.read_csv(csv_path)
    return df

def main():
    df = load_data()
    print("\\n=== DATA OVERVIEW ===")
    print("Shape:", df.shape)
    print(df.head())
    print("\\nMissing values (top):")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    # Define prediction target and features. 'alive' is removed because it directly
    # encodes the target 'survived', avoiding target leakage.
    target = "survived"
    drop_cols = [c for c in ["alive", "class", "embark_town", "deck"] if c in df.columns]
    X = df.drop(columns=[target] + drop_cols)
    y = df[target].astype(int)

    # Keep only numeric and categorical predictors supported by this pipeline.
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced",
                                          random_state=RANDOM_STATE))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "precision": "precision",
               "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring,
                                return_train_score=False, n_jobs=None)
    print("\\n=== 5-FOLD CROSS-VALIDATION (TRAINING SET ONLY) ===")
    for metric in scoring:
        vals = cv_results[f"test_{metric}"]
        print(f"{metric}: {vals.mean():.4f} +/- {vals.std():.4f}")

    # Fit once on the complete training split, then evaluate only on held-out test set.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    print("\\n=== HELD-OUT TEST METRICS ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("\\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    with open(OUTPUT_DIR / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("5-fold CV metrics (mean +/- std)\\n")
        for metric in scoring:
            vals = cv_results[f"test_{metric}"]
            f.write(f"{metric}: {vals.mean():.4f} +/- {vals.std():.4f}\\n")
        f.write("\\nHeld-out test metrics\\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\\n")
        f.write("\\nClassification report\\n")
        f.write(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Did not survive", "Survived"],
        cmap="Blues", values_format="d", ax=ax
    )
    ax.set_title("Titanic Survival: Confusion Matrix")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title("Titanic Survival: ROC Curve")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curve.png", dpi=160)
    plt.close(fig)

    # Missingness overview
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        missing.plot(kind="bar", ax=ax)
        ax.set_title("Missing Values by Column")
        ax.set_ylabel("Missing row count")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "missing_values.png", dpi=160)
        plt.close(fig)

    # Save predictions for transparent inspection
    predictions = X_test.copy()
    predictions["actual_survived"] = y_test
    predictions["predicted_survived"] = y_pred
    predictions["predicted_survival_probability"] = y_prob
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    print(f"\\nOutputs saved in: {OUTPUT_DIR.resolve()}")
    print("Review metrics.txt and figures before writing final reported results.")

if __name__ == "__main__":
    main()
