
#!/usr/bin/env python3

"""
Security Intelligence Data Pipeline
Loads ~30 CSVs (traffic, crime, operations, budget), cleans & merges into one parquet:
data/security_master.parquet

Usage:
    python scripts/data_pipeline.py --raw_dir /mnt/data --out_dir /mnt/data/security_data_pipeline/data

Author: Anthony Ikenna Ogbonna
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# -------------------------
# Config & constants
# -------------------------

DEFAULT_SCHEMA_MAP = {
    "location_priority": [
        "neighbourhood",
        "neighborhood",
        "neighbourhood_name",
        "division",
        "zone_name",
        "patrol_zone",
        "location_name",
        "location",
        "area",
        "ward",
        "community",
        "district",
    ],
    "latitude_aliases": ["latitude", "lat", "y", "lat_wgs84"],
    "longitude_aliases": ["longitude", "lon", "long", "x", "lng", "lon_wgs84"],
    "date_priority": [
        "occurrence_date",
        "occ_date",
        "date",
        "reported_date",
        "occurrencedate",
        "event_date",
        "start_date",
        "issue_date",
    ],
    "year_aliases": ["year", "occ_year", "yr"],
    "count_hint_columns": [
        # if present, we treat each row as 1 count; we may also have explicit counts
        "accnum", "offence", "offence_type", "mci_category", "charge_type", "call_type"
    ],
    "drop_like": [
        # columns to drop if present
        "objectid", "fid", "globalid", "shape", "geom", "geometry", "wkt", "geojson",
        "index_", "_id", "unnamed"
    ],
    "category_keywords": {
        # filename (or mci_category/offence_type) keywords to canonical categories
        "assault": "assault",
        "auto_theft": "auto_theft",
        "break_and_enter": "break_and_enter",
        "robbery": "robbery",
        "theft_over": "theft_over",
        "theft_from_motor_vehicle": "theft_from_motor_vehicle",
        "mci": "mci",
        "shooting": "shootings_firearm",
        "firearm": "shootings_firearm",
        "homicide": "homicide",
        "victim": "victims",
        "reported_crime": "reported_crimes",
        "ksi": "ksi",
        "traffic_collision": "traffic_collisions",
        "pedestrian": "ksi_pedestrian",
        "cyclist": "ksi_cyclist",
        "motorcyclist": "ksi_motorcyclist",
        "fatal": "fatal_collisions",
        "patrol_zone": "patrol_zone",
        "police_boundaries": "police_boundaries",
        "dispatch": "dispatched_calls",
        "arrested_and_charged": "arrested_charged",
        "arrests_and_strip_searches": "strip_searches",
        "search_of_persons": "search_of_persons",
        "gross_operating_budget": "budget_operating",
        "gross_expenditures": "budget_expenditures",
        "personnel_by_rank": "personnel_by_rank",
        "neighbourhood_crime_rates": "neighbourhood_crime_rates",
        "bicycle_thefts": "bicycle_thefts",
        "traffic_collisions_open_data": "traffic_collisions"
    }
}

BATCH_BY_CATEGORY = {
    # broad families
    "traffic": {"ksi", "ksi_pedestrian", "ksi_cyclist", "ksi_motorcyclist", "fatal_collisions", "traffic_collisions"},
    "crime": {
        "assault", "auto_theft", "break_and_enter", "robbery", "theft_over",
        "theft_from_motor_vehicle", "mci", "shootings_firearm", "homicide",
        "victims", "reported_crimes", "neighbourhood_crime_rates", "bicycle_thefts"
    },
    "operations": {"patrol_zone", "police_boundaries", "dispatched_calls", "arrested_charged", "strip_searches", "search_of_persons"},
    "budget": {"budget_operating", "budget_expenditures", "personnel_by_rank"},
}

# Output columns target (not all will be present initially)
TARGET_COLUMNS = [
    "location_name", "location_type", "latitude", "longitude", "year",
    "violent_crime_count", "property_crime_count", "total_crime_count", "crime_rate_per_1000",
    "ksi_total", "ksi_pedestrian", "ksi_cyclist", "ksi_motorcyclist",
    "fatal_collisions", "total_collisions", "collision_density", "avg_injury_severity",
    "total_dispatch_calls", "priority_1_ratio", "arrests_count", "strip_search_count",
    "searches_count", "police_presence_index", "operational_density",
    "gross_budget", "total_expenditure", "salaries_and_benefits", "equipment_costs",
    "total_personnel", "supervisory_ratio", "budget_per_capita", "spending_efficiency",
    "security_score"
]


# -------------------------
# Helpers
# -------------------------

def std_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        pd.Index(df.columns)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df


def first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_location_type(used_col: Optional[str]) -> Optional[str]:
    if not used_col:
        return None
    if "neigh" in used_col:
        return "Neighbourhood"
    if "division" in used_col:
        return "Division"
    if "zone" in used_col or "patrol" in used_col:
        return "Patrol Zone"
    if "ward" in used_col:
        return "Ward"
    if "area" in used_col:
        return "Area"
    if "location" in used_col:
        return "Location"
    return used_col.title()


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def make_year(df: pd.DataFrame, schema: Dict) -> pd.DataFrame:
    # Try explicit year col first
    year_col = first_present(df, schema["year_aliases"])
    if year_col:
        df["year"] = coerce_numeric(df[year_col]).astype("Int64")
        return df

    # Try a date column and extract year
    date_col = first_present(df, schema["date_priority"])
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
        df["year"] = df[date_col].dt.year.astype("Int64")
        return df

    # Fallback: missing year
    df["year"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    return df


def normalize_location(df: pd.DataFrame, schema: Dict) -> pd.DataFrame:
    used = first_present(df, schema["location_priority"])
    if used is None:
        # try composite from division+zone or fallback
        if "division" in df.columns:
            df["location_name"] = df["division"].astype(str)
            used = "division"
        elif "zone" in df.columns:
            df["location_name"] = df["zone"].astype(str)
            used = "zone"
        else:
            df["location_name"] = "Unknown"
            used = None
    else:
        df["location_name"] = df[used].astype(str)

    # Clean up
    df["location_name"] = df["location_name"].str.strip().str.title()
    df["location_type"] = infer_location_type(used)

    # Coordinates if present
    lat_col = first_present(df, schema["latitude_aliases"])
    lon_col = first_present(df, schema["longitude_aliases"])
    if lat_col:
        df["latitude"] = coerce_numeric(df[lat_col])
    if lon_col:
        df["longitude"] = coerce_numeric(df[lon_col])

    return df


def drop_noise_columns(df: pd.DataFrame, schema: Dict) -> pd.DataFrame:
    drop_cols = []
    for col in df.columns:
        for bad in schema["drop_like"]:
            if bad in col:
                drop_cols.append(col)
                break
    if drop_cols:
        df = df.drop(columns=list(set(drop_cols)), errors="ignore")
    return df


def canonical_category_from_filename(fname: str, schema: Dict) -> Optional[str]:
    fn = fname.lower()
    for key, canon in schema["category_keywords"].items():
        if key in fn:
            return canon
    return None


def canonical_category_from_columns(df: pd.DataFrame) -> Optional[str]:
    # Try MCI or offence fields
    for col in ["mci_category", "offence_type", "offence", "category"]:
        if col in df.columns:
            # take most frequent as a hint
            top = df[col].astype(str).str.lower().value_counts(dropna=True).head(1)
            if len(top) > 0:
                val = top.index[0]
                # map common values
                if "assault" in val:
                    return "assault"
                if "robbery" in val:
                    return "robbery"
                if "auto" in val and "theft" in val:
                    return "auto_theft"
                if "break" in val:
                    return "break_and_enter"
                if "theft" in val and "motor" in val:
                    return "theft_from_motor_vehicle"
                if "theft" in val:
                    return "theft_over"
                if "homicide" in val:
                    return "homicide"
                if "shoot" in val or "firearm" in val:
                    return "shootings_firearm"
                if "bicycle" in val:
                    return "bicycle_thefts"
                if "victim" in val:
                    return "victims"
                if "mci" in val:
                    return "mci"
    return None


def classify_file(path: Path, schema: Dict) -> Tuple[str, str]:
    """
    Returns (batch_family, canonical_category)
    """
    cat = canonical_category_from_filename(path.name, schema) or "unknown"
    # Determine batch by category
    family = "unknown"
    for fam, cats in BATCH_BY_CATEGORY.items():
        if cat in cats:
            family = fam
            break
    return family, cat


def add_row_count(df: pd.DataFrame) -> pd.DataFrame:
    if "row_count" not in df.columns:
        df["row_count"] = 1
    return df


def soft_numeric_cols(df: pd.DataFrame) -> List[str]:
    # Return numeric-like columns including booleans (as ints)
    numcols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            numcols.append(c)
        elif df[c].dropna().isin([0,1,True,False]).all() and df[c].notna().any():
            # coerce boolean-like
            df[c] = df[c].astype(int)
            numcols.append(c)
    return numcols


def aggregate(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """
    Aggregates numeric columns by keys.
    If no numeric columns exist, counts rows instead.
    """
    numcols = soft_numeric_cols(df)
    agg_cols = [c for c in numcols if c not in keys]

    # Fallback: if no numeric cols, count rows
    if not agg_cols:
        df["row_count"] = 1
        agg_cols = ["row_count"]

    grouped = df.groupby(keys, dropna=False)[agg_cols].sum(min_count=1).reset_index()
    return grouped


# -------------------------
# Per-batch processors
# -------------------------

def process_generic_csv(path: Path, schema: Dict) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin1")
    df = std_colnames(df)
    df = drop_noise_columns(df, schema)
    df = normalize_location(df, schema)
    df = make_year(df, schema)
    return df


def process_traffic(paths: List[Path], schema: Dict) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = process_generic_csv(p, schema)
        cat = canonical_category_from_filename(p.name, schema) or canonical_category_from_columns(df) or "traffic_misc"
        df["traffic_category"] = cat

        # Normalize common flags
        for col in ["fatal", "fatality", "killed_or_seriously_injured", "ksi"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = add_row_count(df)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["location_name","year"])

    all_df = pd.concat(frames, ignore_index=True)
    # Specific rollups
    keys = [k for k in ["location_name", "year"] if k in all_df.columns]
    agg = aggregate(all_df, keys)

    # Rename common metrics if present
    colmap = {}
    if "fatal" in agg.columns:
        colmap["fatal"] = "fatal_collisions"
    if "row_count" in agg.columns:
        colmap["row_count"] = "total_collisions"
    agg = agg.rename(columns=colmap)

    return agg


def process_crime(paths: List[Path], schema: Dict) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = process_generic_csv(p, schema)
        cat = canonical_category_from_filename(p.name, schema) or canonical_category_from_columns(df) or "crime_misc"
        df["crime_category"] = cat
        df = add_row_count(df)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["location_name","year"])

    all_df = pd.concat(frames, ignore_index=True)
    keys = [k for k in ["location_name", "year", "crime_category"] if k in all_df.columns]
    agg = aggregate(all_df, keys)

    # Pivot categories to columns
    if "crime_category" in keys:
        pivot = agg.pivot_table(index=[c for c in keys if c != "crime_category"],
                                columns="crime_category",
                                values=[c for c in agg.columns if c not in keys],
                                aggfunc="sum").reset_index()
        # Flatten columns
        pivot.columns = ["_".join([str(c) for c in col if str(c) != ""]) for col in pivot.columns.to_flat_index()]
    else:
        pivot = agg

    # Compose higher-level counts
    # Violent = assault + robbery + shootings_firearm + homicide
    def sum_cols(df, cols):
        present = [c for c in cols if c in df.columns]
        if present:
            return df[present].sum(axis=1)
        return 0

    violent_candidates = [
        "row_count_assault", "row_count_robbery", "row_count_shootings_firearm", "row_count_homicide"
    ]
    property_candidates = [
        "row_count_theft_over", "row_count_auto_theft", "row_count_break_and_enter", "row_count_theft_from_motor_vehicle",
        "row_count_bicycle_thefts"
    ]

    pivot["violent_crime_count"] = sum_cols(pivot, violent_candidates)
    pivot["property_crime_count"] = sum_cols(pivot, property_candidates)
    pivot["total_crime_count"] = pivot[["violent_crime_count", "property_crime_count"]].fillna(0).sum(axis=1)

    # Crime rate per 1000 if neighbourhood crime rates present
    # Try to find a rate column
    rate_col = None
    for c in pivot.columns:
        if "crime_rate_per_1000" in c:
            rate_col = c
            break
    if rate_col:
        pivot["crime_rate_per_1000"] = pivot[rate_col]

    return pivot


def process_operations(paths: List[Path], schema: Dict) -> pd.DataFrame:
    """
    Safely process all operational datasets (dispatch, arrests, searches, etc.)
    and produce consistent output columns — even if some files or fields are missing.
    """

    frames = []
    for p in paths:
        df = process_generic_csv(p, schema)
        cat = canonical_category_from_filename(p.name, schema) or "ops_misc"
        df["ops_category"] = cat
        df = add_row_count(df)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["location_name", "year"])

    all_df = pd.concat(frames, ignore_index=True)
    keys = [k for k in ["location_name", "year", "ops_category"] if k in all_df.columns]
    agg = aggregate(all_df, keys)

    # Pivot the data
    pivot = agg.pivot_table(
        index=[c for c in keys if c != "ops_category"],
        columns="ops_category",
        values=[c for c in agg.columns if c not in keys],
        aggfunc="sum",
    ).reset_index()

    pivot.columns = [
        "_".join([str(c) for c in col if str(c) != ""]) for col in pivot.columns.to_flat_index()
    ]

    # Helper to safely sum multiple possible columns
    def pick(df, colname, cats):
        present_cols = [
            c for c in df.columns
            if any(c.endswith(f"_{cat}") for cat in cats) and c.startswith("row_count_")
        ]
        if present_cols:
            df[colname] = df[present_cols].sum(axis=1)
        else:
            # Always create column so later code doesn’t KeyError
            df[colname] = 0
        return df

    # Build safe operation metrics
    pivot = pick(pivot, "total_dispatch_calls", ["dispatched_calls"])
    pivot = pick(pivot, "arrests_count", ["arrested_charged", "arrested_and_charged"])
    pivot = pick(pivot, "strip_search_count", ["strip_searches"])
    pivot = pick(pivot, "searches_count", ["search_of_persons", "searches"])

    # --- Debugging block ---
    print("\n===== DEBUG: operations pivot columns =====")
    print(pivot.columns.tolist())
    print("===========================================\n")

    # Ensure safe placeholders
    for col in ["total_dispatch_calls", "arrests_count", "strip_search_count", "searches_count"]:
        if col not in pivot.columns:
            pivot[col] = 0

    # Compute police presence index safely
    pivot["police_presence_index"] = (
        pivot[["total_dispatch_calls", "arrests_count"]]
        .fillna(0)
        .sum(axis=1)
    )

    return pivot


def process_budget(paths: List[Path], schema: Dict) -> pd.DataFrame:
    """
    Handle budget and personnel datasets.
    Detects both 'SUBTYPE/COUNT_' long format and wide-format numeric tables (like Gross Operating Budget).
    """
    frames = []
    for p in paths:
        df = process_generic_csv(p, schema)
        cat = canonical_category_from_filename(p.name, schema) or "budget_misc"
        df["budget_category"] = cat
        df.columns = df.columns.str.strip().str.lower()

        # --- Case 1: SUBTYPE / COUNT_ long format ---
        if {"subtype", "count_"}.issubset(df.columns):
            df_pivot = (
                df.pivot_table(
                    index=["year", "section", "category"],
                    columns="subtype",
                    values="count_",
                    aggfunc="sum",
                )
                .reset_index()
            )
            df_pivot.columns = [
                str(c).strip().lower().replace(" ", "_").replace("($)", "").replace("%", "pct")
                for c in df_pivot.columns
            ]
            frames.append(df_pivot)
            continue

        # --- Case 2: Already wide-format numeric budget file ---
        num_cols = [
            c
            for c in df.columns
            if any(k in c for k in ["budget", "expenditure", "cost", "amount", "salaries"])
        ]
        if num_cols:
            # Keep key info only
            keep = [c for c in ["year", "section", "category"] if c in df.columns] + num_cols
            df_budget = df[keep].copy()
            for c in num_cols:
                df_budget[c] = pd.to_numeric(df_budget[c], errors="coerce")
            frames.append(df_budget)
            continue

    if not frames:
        return pd.DataFrame(columns=["year"])

    all_df = pd.concat(frames, ignore_index=True)

    # --- Aggregate by year and sum numeric values ---
    numcols = [c for c in all_df.columns if pd.api.types.is_numeric_dtype(all_df[c])]
    agg = all_df.groupby("year")[numcols].sum(min_count=1).reset_index()

    # --- Canonical column names ---
    colmap = {}
    for c in agg.columns:
        if "gross_operating_budget" in c:
            colmap[c] = "gross_budget"
        elif "expenditure" in c or "expenditures" in c:
            colmap[c] = "total_expenditure"
        elif "salaries" in c or "benefit" in c:
            colmap[c] = "salaries_and_benefits"
    agg = agg.rename(columns=colmap)

    return agg


# -------------------------
# Orchestration
# -------------------------

def collect_files(raw_dir: Path) -> List[Path]:
    files = sorted([p for p in raw_dir.glob("*.csv")])
    return files


def route_files(files: List[Path], schema: Dict):
    buckets = {"traffic": [], "crime": [], "operations": [], "budget": [], "unknown": []}
    for p in files:
        family, cat = classify_file(p, schema)
        buckets.get(family, buckets["unknown"]).append(p)
    return buckets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "raw"))
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parents[2] / "data"))
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[1] / "config" / "schema_map.yaml"))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir
    cleaned_dir = data_dir / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    # Load schema config if exists
    if Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as fh:
            schema = yaml.safe_load(fh)
    else:
        schema = DEFAULT_SCHEMA_MAP

    files = collect_files(raw_dir)
    if not files:
        print(f"[WARN] No CSV files found in {raw_dir}", file=sys.stderr)
        sys.exit(0)

    buckets = route_files(files, schema)
    print("[INFO] File buckets:", {k: len(v) for k, v in buckets.items()})

    # Process each family
    traffic_df = process_traffic(buckets["traffic"], schema)
    crime_df = process_crime(buckets["crime"], schema)
    ops_df = process_operations(buckets["operations"], schema)
    budget_df = process_budget(buckets["budget"], schema)

    # Merge
    master = None
    for df in [traffic_df, crime_df, ops_df, budget_df]:
        if df is None or df.empty:
            continue
        master = df if master is None else master.merge(df, on=[c for c in ["location_name", "year"] if c in df.columns and c in master.columns], how="outer")

    if master is None or master.empty:
        print("[WARN] Nothing to save; master dataframe is empty.", file=sys.stderr)
        sys.exit(0)

    # Derived metrics (safe guards)
    if "violent_crime_count" not in master.columns:
        master["violent_crime_count"] = np.nan
    if "property_crime_count" not in master.columns:
        master["property_crime_count"] = np.nan
    master["total_crime_count"] = master[["violent_crime_count", "property_crime_count"]].sum(axis=1, skipna=True)

    if "total_collisions" not in master.columns and "row_count" in master.columns:
        master["total_collisions"] = master["row_count"]

    # Simple normalized indices (rank pct across available rows)
    def rank_pct(col):
        if col not in master.columns:
            return np.nan
        s = master[col]
        if s.dropna().empty:
            return np.nan
        return s.rank(pct=True)

    # Placeholder presence and density if missing
    if "police_presence_index" not in master.columns:
        pres = master.get("total_dispatch_calls", pd.Series(np.nan, index=master.index)).fillna(0) + \
               master.get("arrests_count", pd.Series(np.nan, index=master.index)).fillna(0)
        master["police_presence_index"] = pres

    if "collision_density" not in master.columns and "total_collisions" in master.columns:
        denom = master.get("total_personnel", pd.Series(np.nan, index=master.index)).replace({0: np.nan})
        master["collision_density"] = master["total_collisions"] / denom

    # Security score (bounded 0..1)
    comp = (
        0.5 * rank_pct("total_crime_count") +
        0.3 * rank_pct("total_collisions") -
        0.2 * rank_pct("police_presence_index")
    )
    if isinstance(comp, pd.Series):
        comp = comp.clip(0, 1)
    master["security_score"] = comp

    # Save
    out_parquet = data_dir / "security_master.parquet"
    master.to_parquet(out_parquet, index=False)
    print(f"[OK] Wrote {out_parquet} with {len(master):,} rows and {len(master.columns)} columns")

    # Quick audit
    audit_lines = []
    audit_lines.append(f"# Security Data Pipeline — Audit Summary\n")
    audit_lines.append(f"- Rows: {len(master):,}\n- Columns: {len(master.columns)}\n")
    audit_lines.append("## Sample columns\n")
    cols = ", ".join(list(master.columns)[:25])
    audit_lines.append(f"`{cols} ...`\n")
    with open(Path(__file__).resolve().parents[2] / "reports" / "audit_summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(audit_lines))

if __name__ == "__main__":
    pd.set_option("display.max_columns", 200)
    main()
