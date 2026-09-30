# Cyclone Sentinel

An AI-assisted tropical-cyclone monitoring and early-impact dashboard for the North Indian Ocean.

Deployed on - https://cyclone-sentinel.onrender.com/

## What works now

- The latest collected NOAA IBTrACS storm as a historical replay, selected by its most recent observation.
- Observed track plus a research-only 24/48/72-hour baseline forecast, with honest held-out error metrics.
- A reproducible 24-hour training set and ridge-regression benchmark for track and intensity.
- An event-aligned ten-storm collection plan (including the newest replayable storm), ERA5 feature pipeline, and GPM IMERG collector/extractor for the next multi-source layer.
- An operational-source watch for INSAT-3D metadata and near-real-time GPM IMERG Early/Late readiness, with explicit freshness safeguards.

The project is a research prototype. It is not an official warning or forecasting system.

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## API

- `GET /api/v1/storms` — available storms
- `GET /api/v1/storms/current` — most recent collected replayable storm
- `GET /api/v1/storms/{storm_id}` — collected storm state, observed track, forecast, and roadmap
- `GET /api/v1/storms/{storm_id}/prediction` — model-oriented prediction payload
- `GET /api/v1/data/status` — collection status and dataset manifest
- `GET /api/v1/data/catalog` — readiness of each planned data layer

## Begin data collection

The first reproducible collector downloads NOAA's official IBTrACS North Indian Ocean best-track subset and normalizes it into a model-ready CSV. It stores the source file locally under `data/raw/` and derived tracks under `data/processed/`; neither directory is committed to Git.

```powershell
.\.venv\Scripts\python scripts/collect_ibtracs.py --start-year 2000
```

This supplies labels for position, intensity, and timestamps. The project now also includes a checked ten-storm multi-source collection plan, an event-aligned ERA5 collector/extractor, and a GPM IMERG Final collector. See [the data-source guide](docs/DATA_SOURCES.md) before configuring external credentials. MOSDAC/INSAT imagery needs a registered research account, so it is intentionally not automated yet.

The source-join stage is available now and preserves missing-source flags rather than filling unavailable ERA5/IMERG data with zero:

```powershell
.\.venv\Scripts\python scripts/build_multisource_dataset.py
```

When an extracted feature row exactly matches a replay timestamp, the replay
automatically uses its ERA5 SST and IMERG rainfall values. Missing rows remain
blank and the UI retains its research-proxy labels; no seasonal proxy is
silently presented as collected data.

## Operational satellite products

The dashboard separates live-source status from historical training data. The
first integration polls MOSDAC's public INSAT-3D Imager RSS metadata; it rejects
stale feed items instead of presenting them as current imagery. GPM IMERG Early
and Late are the intended near-real-time rainfall products and require a local
Earthdata Login. See [the operational-product guide](docs/LIVE_SATELLITE_PRODUCTS.md).

## INSAT imagery pathway

After obtaining authorised MOSDAC imagery, place it under `data/raw/insat/`
and create a checksum-backed local catalog before any ML feature extraction:

```powershell
.\.venv\Scripts\python scripts/index_insat_imagery.py --dry-run
.\.venv\Scripts\python scripts/index_insat_imagery.py
```

See [the INSAT ingestion guide](docs/INSAT_IMAGE_INGESTION.md). The catalog
never creates a storm classification; it only makes real local files available
for the later image-decoding stage.

Once images are catalogued, create time-matched historical training pairs with:

```powershell
.\.venv\Scripts\python scripts/prepare_insat_training_index.py
```

## Next implementation steps

1. Run the local, credential-safe preflight. It checks only setup booleans and file counts; it never reads or prints secrets. Add `--write` to make the dashboard show its current result:

   ```powershell
   .\.venv\Scripts\python scripts/check_collection_readiness.py --write
   ```

2. Configure a CDS account locally, collect ERA5 for Biparjoy, then run the extractor.
3. Configure Earthdata locally, collect GPM IMERG for Biparjoy, then run its HDF5 feature extractor.
4. Train and evaluate the strict complete-case multi-source research benchmark once both feature layers have measured coverage:

   ```powershell
   .\.venv\Scripts\python scripts/train_multisource_baseline.py
   ```

   It refuses to produce a model until matching, non-missing ERA5 and IMERG rows exist.
5. Register with MOSDAC and add INSAT imagery/classification.
