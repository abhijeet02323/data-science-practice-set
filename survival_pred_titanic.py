"""
Integrative Data Science Capstone
Dataset: Public Titanic dataset
Tasks:
  1. Acquisition and data quality checks
  2. Cleaning and feature engineering
  3. Exploratory data analysis and plots
  4. Supervised learning: Logistic Regression classification
  5. Unsupervised learning: K-Means clustering on passenger attributes
  6. Evaluation and export of metrics/results

Run:
    pip install pandas numpy matplotlib seaborn scikit-learn
    python data_science_capstone.py

The script computes real results on execution. Do not report unrun or illustrative
values as experimental findings.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay,
    silhouette_score, adjusted_rand_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
OUT = Path("survival_pred_titanic_output")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"

def load_dataset():
    path = OUT / "titanic.csv"
    if not path.exists():
        print("Downloading public Titanic dataset...")
        urllib.request.urlretrieve(DATA_URL, path)
    return pd.read_csv(path)

def save_bar(series, title, xlabel, ylabel, filename):
    ax = series.plot(kind="bar", figsize=(7, 4))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG / filename, dpi=160)
    plt.close()

def main():
    df = load_dataset()
    print("\\n=== DATA OVERVIEW ===")
    print("Shape:", df.shape)
    print(df.head())
    print("\\nData types:\\n", df.dtypes)
    print("\\nMissing values:\\n", df.isna().sum().sort_values(ascending=False))

    # Save raw quality report before transformation
    quality = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
        "unique_values": df.nunique(dropna=True)
    })
    quality.to_csv(OUT / "data_quality_report.csv")

    # EDA figures
    save_bar(df["survived"].value_counts().sort_index(),
             "Target Distribution (0 = No, 1 = Yes)", "Survived", "Passenger count",
             "target_distribution.png")
    save_bar(df.groupby("sex")["survived"].mean(),
             "Survival Proportion by Sex", "Sex", "Survival proportion",
             "survival_by_sex.png")
    save_bar(df.groupby("pclass")["survived"].mean(),
             "Survival Proportion by Passenger Class", "Passenger class",
             "Survival proportion", "survival_by_class.png")
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="age", hue="survived", bins=25, kde=False, multiple="layer")
    plt.title("Age Distribution by Survival Outcome")
    plt.tight_layout()
    plt.savefig(FIG / "age_by_survival.png", dpi=160)
    plt.close()
    plt.figure(figsize=(7, 4))
    sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm",
                fmt=".2f", center=0)
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(FIG / "correlation_heatmap.png", dpi=160)
    plt.close()

    # Cleaning and feature engineering
    data = df.copy()
    data["family_size"] = data["sibsp"].fillna(0) + data["parch"].fillna(0) + 1
    data["is_alone"] = (data["family_size"] == 1).astype(int)
    # Target leakage / duplicates removed from predictors
    drop_cols = [c for c in ["survived", "alive", "class", "embark_town", "deck",
                              "who", "adult_male", "alone"] if c in data.columns]
    X = data.drop(columns=drop_cols)
    y = data["survived"].astype(int)

    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])
    classifier = Pipeline([
        ("preprocessor", preprocess),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=SEED))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
               "f1": "f1", "roc_auc": "roc_auc"}
    cv_scores = cross_validate(classifier, X_train, y_train, cv=cv, scoring=scoring)
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    prob = classifier.predict_proba(X_test)[:, 1]
    sup_metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob)
    }

    with open(OUT / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("SUPERVISED MODEL: Logistic Regression\\n")
        f.write("Five-fold cross-validation on training set (mean +/- std)\\n")
        for name in scoring:
            values = cv_scores[f"test_{name}"]
            f.write(f"{name}: {values.mean():.4f} +/- {values.std():.4f}\\n")
        f.write("\\nHeld-out test metrics\\n")
        for name, val in sup_metrics.items():
            f.write(f"{name}: {val:.4f}\\n")
        f.write("\\nClassification report\\n")
        f.write(classification_report(y_test, pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Did not survive", "Survived"],
        cmap="Blues", values_format="d", ax=ax
    )
    ax.set_title("Supervised Model Confusion Matrix")
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    RocCurveDisplay.from_predictions(y_test, prob, ax=ax)
    ax.set_title("Supervised Model ROC Curve")
    fig.tight_layout()
    fig.savefig(FIG / "roc_curve.png", dpi=160)
    plt.close(fig)

    # Unsupervised clustering: no target labels used for fitting.
    cluster_cols = ["age", "fare", "sibsp", "parch", "pclass", "family_size"]
    cluster_df = data[cluster_cols].copy()
    cluster_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    Z = cluster_pipe.fit_transform(cluster_df)
    kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=20)
    cluster_ids = kmeans.fit_predict(Z)
    silhouette = silhouette_score(Z, cluster_ids)
    cluster_output = data.copy()
    cluster_output["cluster"] = cluster_ids
    cluster_output.to_csv(OUT / "passengers_with_clusters.csv", index=False)
    profile = cluster_output.groupby("cluster")[cluster_cols + ["survived"]].mean()
    profile.to_csv(OUT / "cluster_profiles.csv")

    # PCA visualization
    pca = PCA(n_components=2, random_state=SEED)
    points = pca.fit_transform(Z)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=points[:, 0], y=points[:, 1], hue=cluster_ids,
                    palette="deep", s=35, alpha=0.7)
    plt.title("K-Means Passenger Segments (PCA Projection)")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.tight_layout()
    plt.savefig(FIG / "cluster_pca.png", dpi=160)
    plt.close()

    with open(OUT / "evaluation_report.txt", "a", encoding="utf-8") as f:
        f.write("\\nUNSUPERVISED MODEL: K-Means\\n")
        f.write(f"Silhouette score (k=3): {silhouette:.4f}\\n")
        f.write("Cluster profiles saved to cluster_profiles.csv\\n")
        f.write("Note: survival was not used to fit K-Means; cluster survival rates are descriptive only.\\n")

    print("\\n=== SUPERVISED TEST METRICS ===")
    for k, v in sup_metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"\\nK-Means silhouette score: {silhouette:.4f}")
    print("\\nCluster profiles:\\n", profile)
    print(f"\\nAll results saved to: {OUT.resolve()}")

if __name__ == "__main__":
    main()
