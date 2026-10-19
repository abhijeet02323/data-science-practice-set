"""
Exploratory Data Analysis (EDA) — Titanic Dataset
-------------------------------------------------
Downloads the public Titanic dataset, explores its structure, creates
annotated visualizations, and saves reusable outputs.

Install:
    pip install pandas matplotlib seaborn

Run:
    python titanic_eda.py

Outputs:
    data/titanic.csv
    outputs/figures/*.png
    outputs/eda_summary.txt
    outputs/cleaned_titanic.csv
"""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_URL = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "outputs" / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RAW_PATH = DATA_DIR / "titanic.csv"

sns.set_theme(style="whitegrid", context="notebook")


def load_data() -> pd.DataFrame:
    """Download dataset if absent, then load it."""
    if not RAW_PATH.exists():
        print("Downloading Titanic dataset...")
        urlretrieve(DATA_URL, RAW_PATH)
    df = pd.read_csv(RAW_PATH)
    if df.empty:
        raise ValueError("The downloaded dataset is empty.")
    return df


def print_overview(df: pd.DataFrame) -> None:
    """Print schema, dimensions, missingness, and descriptive statistics."""
    print("\\n--- DATASET SHAPE ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\\n--- DATA TYPES ---")
    print(df.dtypes)
    print("\\n--- FIRST FIVE ROWS ---")
    print(df.head())
    print("\\n--- DESCRIPTIVE STATISTICS ---")
    print(df.describe(include="all").T)
    print("\\n--- MISSING VALUES ---")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0] if missing.any() else "No missing values.")
    print(f"Duplicate rows: {df.duplicated().sum()}")


def create_visualizations(df: pd.DataFrame) -> None:
    """Generate multiple labeled charts and save them to outputs/figures."""
    # 1. Overall survival distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x="survived", hue="survived", legend=False, ax=ax)
    ax.set_title("Titanic Passengers: Survival Outcome")
    ax.set_xlabel("Survived (0 = No, 1 = Yes)")
    ax.set_ylabel("Number of passengers")
    for container in ax.containers:
        ax.bar_label(container)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "survival_outcome.png", dpi=160)
    plt.close(fig)

    # 2. Survival rate by sex
    survival_sex = df.groupby("sex", observed=True)["survived"].mean().reset_index()
    survival_sex["survival_rate_pct"] = survival_sex["survived"] * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=survival_sex, x="sex", y="survival_rate_pct", hue="sex",
                legend=False, ax=ax)
    ax.set_title("Survival Rate by Sex")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Survival rate (%)")
    ax.set_ylim(0, 100)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "survival_by_sex.png", dpi=160)
    plt.close(fig)

    # 3. Survival rate by passenger class
    survival_class = (
        df.groupby("pclass", observed=True)["survived"].mean()
        .mul(100).rename("survival_rate_pct").reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=survival_class, x="pclass", y="survival_rate_pct",
                hue="pclass", legend=False, ax=ax)
    ax.set_title("Survival Rate by Passenger Class")
    ax.set_xlabel("Passenger class (1 = first, 3 = third)")
    ax.set_ylabel("Survival rate (%)")
    ax.set_ylim(0, 100)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "survival_by_class.png", dpi=160)
    plt.close(fig)

    # 4. Age distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="age", bins=30, kde=True, ax=ax)
    ax.set_title("Passenger Age Distribution")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of passengers")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "age_distribution.png", dpi=160)
    plt.close(fig)

    # 5. Fare distribution with box plot to show spread/outliers
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="fare", ax=ax)
    ax.set_title("Distribution of Passenger Fares")
    ax.set_xlabel("Fare")
    ax.set_ylabel("Fare amount")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fare_boxplot.png", dpi=160)
    plt.close(fig)

    # 6. Correlation heatmap for numeric columns
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap of Numerical Features")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(fig)


def save_summary(df: pd.DataFrame) -> None:
    """Write key calculated statistics to a text report."""
    lines = [
        "TITANIC DATASET — EDA SUMMARY",
        "=" * 34,
        f"Rows: {len(df)}",
        f"Columns: {df.shape[1]}",
        f"Duplicate rows: {df.duplicated().sum()}",
        "",
        "Missing values by column:",
        df.isna().sum().to_string(),
        "",
        "Survival count:",
        df["survived"].value_counts(dropna=False).sort_index().to_string(),
        "",
        "Survival rate by sex (%):",
        (df.groupby("sex", observed=True)["survived"].mean().mul(100)
         .round(2).to_string()),
        "",
        "Survival rate by passenger class (%):",
        (df.groupby("pclass", observed=True)["survived"].mean().mul(100)
         .round(2).to_string()),
        "",
        "Numeric descriptive statistics:",
        df.describe().round(2).to_string(),
    ]
    (ROOT / "outputs" / "eda_summary.txt").write_text(
        "\\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    df = load_data()
    print_overview(df)
    create_visualizations(df)
    save_summary(df)

    # A separate copy with a simple analysis-friendly missing-value treatment.
    # Keep the original dataset unchanged; use median for age and mode for embark_town.
    analysis_df = df.copy()
    if "age" in analysis_df.columns:
        analysis_df["age"] = analysis_df["age"].fillna(analysis_df["age"].median())
    if "embark_town" in analysis_df.columns:
        mode = analysis_df["embark_town"].mode(dropna=True)
        if not mode.empty:
            analysis_df["embark_town"] = analysis_df["embark_town"].fillna(mode.iloc[0])
    analysis_df.to_csv(ROOT / "outputs" / "cleaned_titanic.csv", index=False)

    print(f"\\nCharts saved to: {FIGURE_DIR}")
    print(f"Summary saved to: {ROOT / 'outputs' / 'eda_summary.txt'}")
    print(f"Analysis copy saved to: {ROOT / 'outputs' / 'cleaned_titanic.csv'}")


if __name__ == "__main__":
    main()
