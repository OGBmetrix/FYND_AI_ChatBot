#!/usr/bin/env python3
"""
enrich_geolocation.py
Attach latitude/longitude + friendly display_name to your master parquet using a reference CSV.

Usage:
  python scripts/enrich_geolocation.py \
      --in data/security_master.parquet \
      --ref data/location_reference.csv \
      --out data/security_master_geo.parquet \
      [--prefer-display]
"""

import argparse
import pandas as pd
from pathlib import Path

def normalize(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
         .str.title()
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input parquet path (security_master.parquet)")
    ap.add_argument("--ref", dest="ref_path", required=True, help="Location reference CSV")
    ap.add_argument("--out", dest="out_path", required=True, help="Output parquet path")
    ap.add_argument("--prefer-display", action="store_true", help="Prefer display_name for chatbot output")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    ref_path = Path(args.ref_path)
    out_path = Path(args.out_path)

    print(f"[INFO] Loading master: {in_path}")
    master = pd.read_parquet(in_path)

    print(f"[INFO] Loading reference: {ref_path}")
    ref = pd.read_csv(ref_path)

    # Normalize join key
    if "location_name" not in master.columns:
        raise SystemExit("[ERROR] 'location_name' column not found in master parquet.")

    master["location_name"] = normalize(master["location_name"])
    ref["location_name"] = normalize(ref["location_name"])

    # Merge
    print("[INFO] Merging reference...")
    merged = master.merge(ref, on="location_name", how="left", suffixes=("", "_ref"))

    # Fill coordinates
    for col in ("latitude", "longitude"):
        ref_col = f"{col}_ref"
        if col not in merged.columns:
            merged[col] = merged.get(ref_col)
        else:
            merged[col] = merged[col].fillna(merged.get(ref_col))
        if ref_col in merged.columns:
            merged.drop(columns=[ref_col], inplace=True, errors="ignore")

    # Display name
    if "display_name" not in merged.columns:
        merged["display_name"] = merged["location_name"]
    else:
        merged["display_name"] = merged["display_name"].fillna(merged["location_name"])

    if args.prefer_display:
        # Keep an alias column to use in UX / chatbot
        merged["location_pretty"] = merged["display_name"]
    else:
        merged["location_pretty"] = merged["location_name"]

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"[OK] Wrote {out_path} with {len(merged):,} rows and {len(merged.columns)} columns")
    # Quick sanity
    sample = merged.loc[merged["latitude"].notna() & merged["longitude"].notna(), ["location_name", "display_name", "latitude", "longitude"]].head(10)
    print("\n[INFO] Sample enriched rows:\n", sample.to_string(index=False))

if __name__ == "__main__":
    main()
