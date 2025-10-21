# Security Intelligence Data Pipeline

This folder contains the generated pipeline to combine your 30 CSV datasets into a single
`security_master.parquet` ready for your AI chatbot.

## How to Run

1. Move/copy all your CSV files into: `security_data_pipeline/data/raw/`
   (This script was designed to also work if your CSVs are at `/mnt/data` — see `--raw_dir`.)

2. From a terminal on your machine (or this environment), run:
   ```bash
   python security_data_pipeline/scripts/data_pipeline.py      --raw_dir /mnt/data      --out_dir /mnt/data/security_data_pipeline/data      --config /mnt/data/security_data_pipeline/config/schema_map.yaml
   ```

3. Output:
   - Unified parquet: `/mnt/data/security_data_pipeline/data/security_master.parquet`
   - Audit summary: `/mnt/data/security_data_pipeline/reports/audit_summary.md`

## Notes

- The pipeline auto-detects location & date columns, normalizes into:
  - `location_name` (primary), `location_type` (context)
  - `latitude`, `longitude` (if present)
  - `year` (from `year` or parsed date)
- It aggregates by `location_name` + `year`, then merges four families:
  **traffic**, **crime**, **operations**, **budget**.

You can adjust aliases & keyword mappings in `config/schema_map.yaml`.