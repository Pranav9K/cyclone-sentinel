# Prepare the IBTrACS baseline training data

Run this after `scripts/collect_ibtracs.py` has created
`data/processed/ibtracs_ni_tracks.csv`:

```powershell
.\.venv\Scripts\python scripts\prepare_training_data.py
```

The command writes these ignored, reproducible artifacts:

- `data/processed/training_samples.csv` — chronologically ordered supervised rows.
- `data/processed/training_samples_manifest.json` — source hash, configuration, schema, and split counts.
- `data/processed/training_samples_report.md` — readable validation and data-quality summary.

Each row uses an exact 24-hour later IBTrACS observation as the label. Its main
targets are `target_delta_latitude_24h`, `target_delta_longitude_24h`, and
`target_wind_knots`. The default input filter retains `TS` and `DS` NATURE
values; mixed (`MX`) and unreported (`NR`) records are excluded.

The current schema is intentionally fixed at a 24-hour horizon. To experiment
with a different input-quality policy, make the status filter explicit and
regenerate all three artifacts. For example:

```powershell
.\.venv\Scripts\python scripts\prepare_training_data.py --allowed-statuses TS,DS
```

The `split` field is assigned by whole storm in chronological order. Do not use
random row splitting: adjacent observations of one cyclone are highly similar
and would give misleading evaluation scores if they appeared in both train and
test data.
