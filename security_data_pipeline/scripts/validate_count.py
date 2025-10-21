#!/usr/bin/env python3
"""
validate_counts.py — Quick sanity check for the unified security_master.parquet

Usage:
    python scripts/validate_counts.py --file data/security_master.parquet
"""

import argparse
from pathlib import Path
import pandas as pd

def summarize_numeric(df: pd.DataFrame):
    """
    Summarize columns by non-null count and NaN ratio.
    """
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        print("[WARN] No numeric columns detected.")
        return None

    summary = (
        df[num_cols]
        .notna()
        .sum()
        .sort_values(ascending=False)
        .rename("non_null_count")
        .to_frame()
    )
    summary["nan_ratio"] = 1 - (summary["non_null_count"] / len(df))
    return summary


def detect_category_presence(df: pd.DataFrame):
    """
    Detect whether each dataset family contributed actual data (non-NaN).
    """
    categories = {
        "Crime": ["violent_crime_count", "property_crime_count", "total_crime_count"],
        "Traffic": ["total_collisions", "fatal_collisions", "ksi_total"],
        "Operations": ["total_dispatch_calls", "arrests_count", "police_presence_index"],
        "Budget": ["gross_budget", "total_expenditure", "total_personnel"],
    }

    print("\n=== Dataset Family Validation ===")
    for fam, cols in categories.items():
        valid = False
        for c in cols:
            if c in df.columns and df[c].notna().any():
                valid = True
                break
        print(f"{fam:<12} → {'✅ Data Present' if valid else '❌ All NaN'}")
    print("=================================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the .parquet file to validate")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return

    print(f"[INFO] Loading {path} ...")
    df = pd.read_parquet(path)
    print(f"[INFO] Loaded dataframe with {len(df):,} rows and {len(df.columns):,} columns.\n")

    # Quick look at keys
    key_preview = df[["location_name", "year"]].drop_duplicates().head(10)
    print("Sample location/year pairs:")
    print(key_preview.to_string(index=False))
    print()

    # Category detection
    detect_category_presence(df)

    # Numeric summary
    summary = summarize_numeric(df)
    if summary is not None:
        print("\n=== Top 20 Non-Null Metrics ===")
        print(summary.head(20))
        print("=================================\n")


if __name__ == "__main__":
    main()
