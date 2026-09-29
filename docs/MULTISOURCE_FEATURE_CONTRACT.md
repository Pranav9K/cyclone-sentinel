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
imerg_granule_start_utc,imerg_center_precipitation_cal_mm_hr,
imerg_window_mean_precipitation_cal_mm_hr,imerg_window_p95_precipitation_cal_mm_hr,
imerg_window_max_precipitation_cal_mm_hr,imerg_window_mean_random_error_mm_hr,
imerg_window_mean_quality_index,imerg_accumulation_6h_mm,
imerg_accumulation_24h_mm,imerg_missing
```

`imerg_granule_start_utc` records the source half-hour interval used for the
instantaneous features. For every accumulation, use only complete intervals
whose end time is at or before `timestamp_utc`; never replace missing rainfall
with zero. `imerg_accumulation_6h_mm` and `imerg_accumulation_24h_mm` are sums
of the ±2.5° window-mean rate across complete 30-minute intervals (`rate × 0.5`
hours), not future or partially filled intervals.

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

## Complete-case research benchmark

After rebuilding the joined table, run:

```powershell
.\.venv\Scripts\python scripts/train_multisource_baseline.py
```

The benchmark only accepts rows whose `multisource_complete` value is `true`.
It uses the ERA5 and IMERG feature columns together with the pre-issue
best-track state, preserves the existing train/test split, and writes ignored
model and metric artifacts under `data/processed/`. It intentionally exits
without creating a model when either source remains unavailable. This is a
research comparison, not an operational forecasting model.
