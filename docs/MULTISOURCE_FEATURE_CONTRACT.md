# Multi-source feature contract

`scripts/build_multisource_dataset.py` joins leakage-aware IBTrACS training samples with optional feature tables from ERA5 and GPM IMERG. The initial cohort comes from `data/processed/multisource_collection_plan.json`.

## Join keys

Every feature row must have these exact columns:

```text
storm_id,timestamp_utc
```

`timestamp_utc` must equal the supervised sample's `input_timestamp_utc` in UTC. It is the prediction issue time, never the target time. A feature that contains post-issue information must not be written to the table.

## ERA5 table

Write `data/processed/era5_features.csv`. Prefix every derived column with `era5_`. The initial recommended fields are:

```text
era5_u10_mps,era5_v10_mps,era5_wind10_mps,era5_mslp_pa,
era5_sst_k,era5_tcw_kg_m2,era5_dewpoint_k,era5_missing
```

## IMERG table

Write `data/processed/imerg_features.csv`. Prefix every derived column with `imerg_`. The initial recommended fields are:

```text
imerg_center_rate_mm_hr,imerg_mean_5deg_mm_hr,imerg_p95_5deg_mm_hr,
imerg_max_5deg_mm_hr,imerg_random_error_mean_mm_hr,
imerg_quality_index_mean,imerg_6h_accumulation_mm,
imerg_24h_accumulation_mm,imerg_missing
```

For every accumulation, only use intervals whose end time is at or before `timestamp_utc`.

## Missing data

The merge deliberately retains samples when a modality is absent. It writes:

```text
era5_available,imerg_available,multisource_complete
```

Training should use an explicit strategy: complete-case rows, missingness-aware features, or modality-specific models. Never treat a blank source measurement as zero.

## Run

```powershell
.\.venv\Scripts\python scripts/build_multisource_dataset.py
```

Add `--require-complete` only after both feature tables have been collected and validated.
