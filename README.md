# Cyclone Sentinel

An AI-assisted tropical-cyclone monitoring and early-impact dashboard for the North Indian Ocean.

## What works now

- A historical Cyclone Biparjoy replay backed by collected NOAA IBTrACS observations.
- Observed track plus a research-only 24/48/72-hour baseline forecast, with honest held-out error metrics.
- A reproducible 24-hour training set and ridge-regression benchmark for track and intensity.
- An event-aligned plan and ERA5 collector for the next multi-source data layer.

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
- `GET /api/v1/storms/biparjoy-2023` — storm state, observed track, forecast, and risk
- `GET /api/v1/storms/biparjoy-2023/prediction` — model-oriented prediction payload
- `GET /api/v1/data/status` — collection status and dataset manifest
- `GET /api/v1/data/catalog` — readiness of each planned data layer

## Begin data collection

The first reproducible collector downloads NOAA's official IBTrACS North Indian Ocean best-track subset and normalizes it into a model-ready CSV. It stores the source file locally under `data/raw/` and derived tracks under `data/processed/`; neither directory is committed to Git.

```powershell
.\.venv\Scripts\python scripts/collect_ibtracs.py --start-year 2000
```

This supplies labels for position, intensity, and timestamps. The project now also includes a checked nine-storm multi-source collection plan and an event-aligned ERA5 collector. See [the data-source guide](docs/DATA_SOURCES.md) before configuring external credentials. MOSDAC/INSAT imagery needs a registered research account, so it is intentionally not automated yet.

The source-join stage is available now and preserves missing-source flags rather than filling unavailable ERA5/IMERG data with zero:

```powershell
.\.venv\Scripts\python scripts/build_multisource_dataset.py
```

## Next implementation steps

1. Run the ERA5 collector for Biparjoy after configuring a CDS account and API key.
2. Add GPM IMERG feature extraction around every storm/time window.
3. Join track, ERA5, and GPM features into a single time-aware training table.
4. Register with MOSDAC and add INSAT imagery/classification.
