# Multi-source data collection

Cyclone Sentinel collects event-aligned source windows rather than downloading a huge static basin archive. All timestamps are UTC and must only use information available at the prediction issue time.

## Ready now: IBTrACS labels and baseline

`scripts/collect_ibtracs.py` downloads NOAA IBTrACS North Indian Ocean best-track observations. These supply historical cyclone positions and intensity targets.

```powershell
.\.venv\Scripts\python scripts/collect_ibtracs.py --start-year 2000
.\.venv\Scripts\python scripts/prepare_training_data.py
.\.venv\Scripts\python scripts/train_ibtracs_baseline.py
```

## First multi-source cohort

Build and inspect the nine-storm cohort before downloading external grids:

```powershell
.\.venv\Scripts\python scripts/plan_multisource_collection.py
.\.venv\Scripts\python scripts/collect_era5.py --storm-id 2023156N10067 --dry-run
```

The plan contains a 5° envelope around each storm and exact historical timestamps. It is generated locally at `data/processed/multisource_collection_plan.json`.

## Join the source layers safely

After feature extraction, join tables using the checked source-time contract:

```powershell
.\.venv\Scripts\python scripts/build_multisource_dataset.py
```

It creates the initial nine-storm training cohort and preserves rows with missing environmental data using explicit availability flags. Read [the feature contract](MULTISOURCE_FEATURE_CONTRACT.md) before adding a new derived field.

## ERA5 atmospheric context

Use the official [ERA5 hourly single-levels dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) through the [Copernicus Climate Data Store API](https://cds.climate.copernicus.eu/how-to-api). ERA5 is reanalysis data, not satellite imagery.

Before a real download, each teammate needs to create a free CDS account, accept the dataset terms, configure their personal API key, and install the optional client:

```powershell
pip install cdsapi
.\.venv\Scripts\python scripts/collect_era5.py --storm-id 2023156N10067
```

The collector requests six variables: 10 m wind components, mean sea-level pressure, SST, total-column water vapour, and 2 m dewpoint temperature. It writes GRIB files under `data/raw/era5/` plus a provenance manifest. Do not commit source grids or API credentials.

## GPM IMERG rainfall

For historic training, use [GPM IMERG Final V07](https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_07/summary). It is research quality but has a finalization delay, so it must not be confused with a live warning source. NASA's [Earthdata Login guidance](https://urs.earthdata.nasa.gov/documentation/for_users/data_access) describes the free account and app authorization required for GES DISC download/OPeNDAP access.

Collect GPM after ERA5 is validated for Biparjoy. The desired compact outputs per label are centre rainfall rate, ±2.5° mean/p95/max rainfall, random-error mean, quality-index mean, 6-hour accumulation, and 24-hour accumulation. For operational display, later add IMERG Early or Late as a clearly separate feed.

## Satellite imagery

INSAT/MOSDAC imagery remains a separate workstream: register a research account first, retain each product's metadata/licensing, and store image tiles under `data/raw/insat/`. Do not train imagery classification until labels and time alignment are verified.
