# ERA5 event-feature extraction

`scripts/extract_era5_features.py` converts locally downloaded ERA5 grids into
one validated, event-aligned feature table. It does not call CDS, does not use
network access, and does not need a Copernicus credential.

The extractor joins each `storm_id` and planned UTC timestamp in
`data/processed/multisource_collection_plan.json` to the cyclone centre in
`data/processed/ibtracs_ni_tracks.csv`. It selects the nearest valid ERA5 grid
point, subject to the configured maximum distance (100 km by default).

## Safe readiness check

This checks the local plan, track coverage, and source-file headers without
creating artifacts or reading credentials:

```powershell
python scripts/extract_era5_features.py --dry-run
```

After `scripts/collect_era5.py` has downloaded the event windows, run:

```powershell
python scripts/extract_era5_features.py
```

The generated files are intentionally ignored by Git:

- `data/processed/era5_features.csv`
- `data/processed/era5_features_manifest.json`

## Output CSV contract

The downstream training-table merger keys this table on the first two columns.
The schema is fixed, including when no raw data are present.

| Column | Unit / meaning |
| --- | --- |
| `storm_id` | IBTrACS storm identifier |
| `timestamp_utc` | Exact ISO-8601 UTC event time, for example `2023-06-08T00:00:00Z` |
| `era5_u10_mps` | 10 m eastward wind component, m s-1 |
| `era5_v10_mps` | 10 m northward wind component, m s-1 |
| `era5_wind10_mps` | Derived speed: `sqrt(u10² + v10²)`, m s-1 |
| `era5_mslp_pa` | Mean sea-level pressure, Pa |
| `era5_sst_k` | Sea-surface temperature, K |
| `era5_tcw_kg_m2` | Total-column water vapour, kg m-2 |
| `era5_dewpoint_k` | 2 m dew-point temperature, K |
| `era5_missing` | `0` for every emitted, fully validated row |

The extractor does not invent placeholder measurements. If an expected event
has no readable, valid, close-enough grid point, it is omitted from the table.
The merger must treat an absent `(storm_id, timestamp_utc)` key as an
unavailable ERA5 layer. When no raw ERA5 source is available, the CSV contains
only the header and the manifest status is `waiting_for_raw_era5`.

## Raw source formats

The preferred source is a `.grib` file produced by `scripts/collect_era5.py`.
GRIB decoding is optional and needs a local decoder:

```powershell
pip install xarray cfgrib
```

NetCDF (`.nc` or `.netcdf`) needs `xarray`. These optional packages are not
needed for the dry run or CSV input.

A flat CSV may instead be placed anywhere under `data/raw/era5/`. It needs one
row per time/grid coordinate and these canonical concepts. The extractor
accepts the aliases shown in `--dry-run` and in the script, but the following
header is the portable form:

```text
timestamp_utc,latitude,longitude,u10_mps,v10_mps,mslp_pa,sst_k,tcw_kg_m2,dewpoint_k
```

Raw data must use ERA5's native units: m s-1 for wind components, Pa for mean
sea-level pressure, K for SST/dew point, and kg m-2 for total-column water
vapour. The extractor validates reasonable physical bounds and refuses to
silently convert hPa or Celsius values.

## Validation and provenance

`era5_features_manifest.json` records the source plan and track hashes,
discovered files, parser readiness, invalid source counts, expected/resolved
event keys, grid-point distance statistics, and explicit missing-data status.

Use exact timestamp matching by default. A narrow tolerance can be opted into
only when the downloaded source has a documented time offset:

```powershell
python scripts/extract_era5_features.py --time-tolerance-minutes 30
```

Do not expand the distance threshold casually: a far grid point weakens the
claim that the atmospheric context represents the cyclone centre.
