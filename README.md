# Cyclone Sentinel

An AI-assisted tropical-cyclone monitoring and early-impact dashboard for the North Indian Ocean.

## What works now

- The latest collected NOAA IBTrACS storm as a historical replay, selected by its most recent observation.
- Observed track plus a research-only 24/48/72-hour baseline forecast, with honest held-out error metrics.
- A reproducible 24-hour training set and ridge-regression benchmark for track and intensity.
- An event-aligned nine-storm collection plan, ERA5 feature pipeline, and GPM IMERG collector/extractor for the next multi-source layer.
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

This supplies labels for position, intensity, and timestamps. The project now also includes a checked nine-storm multi-source collection plan, an event-aligned ERA5 collector/extractor, and a GPM IMERG Final collector. See [the data-source guide](docs/DATA_SOURCES.md) before configuring external credentials. MOSDAC/INSAT imagery needs a registered research account, so it is intentionally not automated yet.

The source-join stage is available now and preserves missing-source flags rather than filling unavailable ERA5/IMERG data with zero:

```powershell
.\.venv\Scripts\python scripts/build_multisource_dataset.py
```

## Operational satellite products

The dashboard separates live-source status from historical training data. The
first integration polls MOSDAC's public INSAT-3D Imager RSS metadata; it rejects
stale feed items instead of presenting them as current imagery. GPM IMERG Early
and Late are the intended near-real-time rainfall products and require a local
Earthdata Login. See [the operational-product guide](docs/LIVE_SATELLITE_PRODUCTS.md).

## Next implementation steps

1. Configure a CDS account locally, collect ERA5 for Biparjoy, then run the extractor.
2. Configure Earthdata locally, collect GPM IMERG for Biparjoy, then run its HDF5 feature extractor.
3. Train and evaluate a missingness-aware multi-source model once both feature layers have measured coverage.
4. Register with MOSDAC and add INSAT imagery/classification.
