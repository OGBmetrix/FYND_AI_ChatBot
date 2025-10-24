#!/usr/bin/env python3
"""
validate_geo.py
Quick checks for the geo-enriched parquet.
"""
import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to security_master_geo.parquet (or master)")
    args = ap.parse_args()

    p = Path(args.file)
    print(f"[INFO] Loading {p} ...")
    df = pd.read_parquet(p)
    print(f"[INFO] Loaded dataframe with {len(df):,} rows and {len(df.columns):,} columns.\n")

    cols = [c for c in ["location_name", "display_name", "location_pretty", "latitude", "longitude", "year", "total_crime_count", "total_collisions"] if c in df.columns]
    print("[INFO] Available key columns:", cols)

    print("\n=== Non-null counts for geo fields ===")
    for c in ["latitude", "longitude"]:
        if c in df.columns:
            non_null = df[c].notna().sum()
            print(f"{c:>12}: {non_null} non-null ({non_null/len(df):.1%})")
        else:
            print(f"{c:>12}: MISSING COLUMN")

    if {"latitude","longitude"}.issubset(df.columns):
        print("\n=== Sample rows with coordinates ===")
        print(df.loc[df["latitude"].notna() & df["longitude"].notna(), ["location_name","display_name","latitude","longitude","year"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
