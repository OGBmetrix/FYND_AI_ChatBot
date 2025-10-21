#!/usr/bin/env python3
"""
Validate schema_map.yaml classification logic.
Scans all CSVs in data/raw and prints the detected batch family and canonical category.

Usage:
    python scripts/validate_schema.py --raw_dir data/raw --config config/schema_map.yaml
"""

import argparse
from pathlib import Path
import yaml
from scripts.data_pipeline import canonical_category_from_filename, classify_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Directory containing raw CSV files")
    parser.add_argument("--config", type=str, default="config/schema_map.yaml", help="Path to schema YAML file")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"[ERROR] Schema config not found: {config_path}")
        return

    if not raw_dir.exists():
        print(f"[ERROR] Raw directory not found: {raw_dir}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        print(f"[WARN] No CSV files found in {raw_dir}")
        return

    print("\n=== Schema Validation Report ===\n")
    results = []
    for fpath in files:
        family, cat = classify_file(fpath, schema)
        results.append((fpath.name, family, cat))
        print(f"{fpath.name:<45} →  {family.upper():<10}  |  {cat}")

    print("\n=== Summary ===")
    families = {}
    for _, fam, _ in results:
        families[fam] = families.get(fam, 0) + 1
    for fam, count in families.items():
        print(f"{fam.title():<10}: {count} file(s)")

    print("\n[OK] Schema classification complete.\n")

if __name__ == "__main__":
    main()
