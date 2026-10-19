"""
Titanic Dataset Data Cleaning and Preprocessing
Run: python titanic_data_cleaning.py
Install: pip install pandas numpy matplotlib
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
OUTPUT_DIR = Path("titanic_cleaning_output")


def iqr_bounds(series, multiplier=1.5):
    """Return Tukey IQR lower and upper bounds, ignoring missing values."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - multiplier * iqr), float(q3 + multiplier * iqr)


def clean_dataset(raw):
    """Return cleaned DataFrame and an audit dictionary."""
    df = raw.copy()
    audit = {"rows_before": len(df), "columns_before": len(df.columns)}

    # Standardize strings and common blank placeholders.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA})

    # Remove only exact duplicate rows; do not guess a unique key.
    audit["exact_duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # Validate known categories. Unexpected values are made missing for review.
    expected = {
        "sex": {"male", "female"},
        "embarked": {"C", "Q", "S"},
        "class": {"First", "Second", "Third"},
        "who": {"man", "woman", "child"},
    }
    for col, allowed in expected.items():
        if col in df:
            invalid = df[col].notna() & ~df[col].isin(allowed)
            audit[f"invalid_{col}_count"] = int(invalid.sum())
            df.loc[invalid, col] = pd.NA

    # Impossible negative values are invalid for these variables.
    for col in ["age", "fare", "sibsp", "parch"]:
        if col in df:
            invalid = df[col].notna() & (df[col] < 0)
            audit[f"negative_{col}_count"] = int(invalid.sum())
            df.loc[invalid, col] = np.nan

    # Group-wise median age imputation; overall median is a fallback.
    if {"age", "pclass", "sex"}.issubset(df.columns):
        group_medians = df.groupby(["pclass", "sex"], observed=False)["age"].transform("median")
        df["age"] = df["age"].fillna(group_medians)
        df["age"] = df["age"].fillna(df["age"].median())

    # Mode is appropriate for a categorical port field.
    if "embarked" in df and df["embarked"].isna().any():
        mode = df["embarked"].mode(dropna=True)
        if not mode.empty:
            df["embarked"] = df["embarked"].fillna(mode.iloc[0])

    # Avoid fabricating cabin labels; retain only whether cabin info was recorded.
    cabin_col = "deck" if "deck" in df else ("cabin" if "cabin" in df else None)
    if cabin_col:
        df["cabin_known"] = df[cabin_col].notna().astype("int8")
        df = df.drop(columns=[cabin_col])

    # Flag potential outliers; IQR flags do not prove data errors.
    audit["iqr_outliers"] = {}
    for col in ["age", "fare", "sibsp", "parch"]:
        if col in df:
            low, high = iqr_bounds(df[col])
            mask = df[col].notna() & ((df[col] < low) | (df[col] > high))
            audit["iqr_outliers"][col] = {
                "lower_bound": low, "upper_bound": high, "count": int(mask.sum())
            }
            df[f"{col}_outlier_flag"] = mask.astype("int8")
            # Cap fare only in this model-ready copy; original remains in raw CSV.
            if col == "fare":
                df[col] = df[col].clip(lower=max(0, low), upper=high)

    if {"sibsp", "parch"}.issubset(df.columns):
        df["family_size"] = df["sibsp"] + df["parch"] + 1
        df["is_alone_recomputed"] = (df["family_size"] == 1).astype("int8")

    for col in ["adult_male", "alone"]:
        if col in df:
            df[col] = df[col].astype("int8")

    audit["rows_after"] = len(df)
    audit["columns_after"] = len(df.columns)
    audit["missing_after"] = df.isna().sum().to_dict()
    return df, audit


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_URL)
    raw.to_csv(OUTPUT_DIR / "titanic_raw.csv", index=False)
    missing_before = raw.isna().sum()
    duplicates_before = int(raw.duplicated().sum())

    cleaned, audit = clean_dataset(raw)
    cleaned.to_csv(OUTPUT_DIR / "titanic_cleaned.csv", index=False)

    # Missing-value comparison chart.
    missing_after = cleaned.isna().sum()
    chart = pd.DataFrame({
        "Before": missing_before,
        "After": missing_after.reindex(missing_before.index, fill_value=0),
    })
    chart = chart.loc[chart.max(axis=1) > 0]
    ax = chart.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Missing Values by Column: Before vs After")
    ax.set_ylabel("Missing cells")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missing_values_before_after.png", dpi=160)
    plt.close()

    if "fare" in raw:
        plt.figure(figsize=(7, 3.5))
        plt.boxplot(raw["fare"].dropna(), vert=False)
        plt.title("Fare Distribution Before Cleaning")
        plt.xlabel("Fare")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fare_outliers_boxplot.png", dpi=160)
        plt.close()

    if "age" in raw and "age" in cleaned:
        plt.figure(figsize=(8, 4))
        plt.hist(raw["age"].dropna(), bins=25, alpha=.65, label="Raw observed ages")
        plt.hist(cleaned["age"].dropna(), bins=25, alpha=.5, label="Cleaned ages")
        plt.title("Age Distribution Before and After Imputation")
        plt.xlabel("Age")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "age_distribution_before_after.png", dpi=160)
        plt.close()

    summary = {
        "source_url": DATA_URL,
        "shape_before": list(raw.shape),
        "duplicates_before": duplicates_before,
        "missing_before": missing_before.to_dict(),
        "audit": audit,
        "dtypes_after": cleaned.dtypes.astype(str).to_dict(),
    }
    (OUTPUT_DIR / "data_quality_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    print(f"Artifacts saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
