# IBTrACS supervised training-data report

Generated at: 2026-09-26T11:59:18.559232+00:00

## Result

- Input: `data/processed/ibtracs_ni_tracks.csv`
- Output: `data/processed/training_samples.csv`
- Manifest: `data/processed/training_samples_manifest.json`
- Exact forecast horizon: 24 hours
- Allowed source statuses: DS, TS
- Source rows read: 9,633
- Unique valid observations: 9,358
- Supervised samples emitted: 6,575

## Validation and exclusions

| Check | Count |
| --- | ---: |
| missing pressure measurement | 910 |
| missing wind measurement | 883 |
| season timestamp mismatch | 58 |
| status not allowed | 275 |
| candidate input observations | 9,358 |
| emitted samples | 6,575 |
| excluded missing input wind | 465 |
| excluded missing target wind | 128 |
| excluded no exact horizon target | 2,190 |
| storms with supervised samples | 244 |
| storms without supervised samples | 20 |

## Chronological, storm-disjoint split

Each storm is assigned once, ordered by its first usable timestamp. This prevents the same cyclone appearing in both training and evaluation data.

| Split | Storms | Samples | First input | Last input |
| --- | ---: | ---: | --- | --- |
| train | 170 | 4,725 | 2000-03-27T12:00:00Z | 2019-09-25T06:00:00Z |
| validation | 36 | 928 | 2019-09-29T12:00:00Z | 2022-10-24T00:00:00Z |
| test | 38 | 922 | 2022-11-20T00:00:00Z | 2025-12-01T18:00:00Z |

## Model columns

- Features: `season, input_year, input_month, input_day_of_year, input_hour_utc, latitude, longitude, wind_knots, pressure_hpa, status, previous_6h_available, previous_6h_delta_latitude, previous_6h_delta_longitude, previous_6h_wind_change_knots`
- Targets: `target_delta_latitude_24h, target_delta_longitude_24h, target_wind_knots, target_wind_change_knots_24h`
- Identifiers and timestamps (`sample_id`, `storm_id`, `input_timestamp_utc`, `target_timestamp_utc`) are traceability fields, not model features.
- `pressure_hpa` and six-hour history fields may be blank; retain an imputation strategy within each training split only.
