# INSAT imagery ingestion

Download an authorised MOSDAC INSAT product to a local folder outside Git,
normally `data/raw/insat/`. Do not place credentials in this repository.

Index the received files before any image-decoding or ML step:

```powershell
.\.venv\Scripts\python scripts/index_insat_imagery.py --dry-run
.\.venv\Scripts\python scripts/index_insat_imagery.py
```

The catalog at `data/processed/insat_image_catalog.json` records only local
file paths, checksums, formats, and any timestamp unambiguously present in the
filename. It does not infer a storm name, Dvorak category, cloud pattern, or
forecast. The dashboard changes INSAT from **Registration needed** to **Ready
to extract** only when this catalog contains accepted local image files.

## Build supervised training pairs

After cataloguing images, pair each file with the nearest historical IBTrACS
observation only when its parsed capture time is within 90 minutes:

```powershell
.\.venv\Scripts\python scripts/prepare_insat_training_index.py
```

This produces `data/processed/insat_training_index.csv` and a companion
manifest. The row's intensity label is a historical best-track label, not an
image-derived claim. Files without a clear capture time, or files too far from
any track observation, are retained as unpaired counts rather than guessed.
