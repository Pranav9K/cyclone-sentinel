# IMERG event-feature extraction

`scripts/extract_imerg_features.py` turns locally collected GPM IMERG Final
V07 HDF5 granules into the time-aligned `imerg_features.csv` table. It never
contacts NASA and never reads Earthdata credentials.

## Readiness and run

Use the dry run to inspect the exact raw coverage needed by the full nine-storm
cohort:

```powershell
.\.venv\Scripts\python scripts/extract_imerg_features.py --dry-run
```

With no raw files available, a normal run creates one explicit missing row for
each planned issue time. This is safe and lets the merged training table retain
its missingness flags:

```powershell
.\.venv\Scripts\python scripts/extract_imerg_features.py
```

After real HDF5 downloads, install the optional local decoder and rerun:

```powershell
pip install h5py numpy
.\.venv\Scripts\python scripts/extract_imerg_features.py
```

## Causal feature policy

For each issue time, the extractor uses the latest **completed** half-hour
interval. For example, an issue at `2023-06-05T00:00:00Z` uses the granule from
`2023-06-04T23:30:00Z` to `2023-06-04T23:59:59Z`; it never uses the interval
beginning at midnight because that would include future data.

It calculates centre and ±2.5°-window statistics from that interval. The 6h
and 24h values are sums of the window-mean precipitation rate multiplied by
0.5 hours, only when all 12 or 48 causal intervals are valid. A missing
backing interval leaves the affected accumulation blank. A missing current
granule leaves every IMERG measurement blank and sets `imerg_missing=1`.

The HDF5 reader accepts both latitude-by-longitude and longitude-by-latitude
grid storage. It validates that the expected IMERG `Grid` fields and a valid
local window exist before emitting measurements.
