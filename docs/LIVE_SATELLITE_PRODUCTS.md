# Operational satellite products

Historical best tracks and Final IMERG are for training and replay. They are
not a live-warning feed. The operational layer uses separate products with
their own latency and provenance.

| Product | Role | Availability in this project |
| --- | --- | --- |
| INSAT-3D Imager | Geostationary cloud-pattern imagery for the North Indian Ocean | Public MOSDAC RSS metadata poller; image download/decoding needs MOSDAC access and product selection |
| GPM IMERG Early Run V07 | Near-real-time precipitation, about four-hour latency | Earthdata Login required |
| GPM IMERG Late Run V07 | Higher-latency precipitation, about fourteen-hour latency | Earthdata Login required |
| GPM IMERG Final V07 | Delayed, gauge-adjusted historic training data | Offline training only; never presented as live |

## INSAT metadata poll

The first live integration verifies provider-published INSAT-3D Imager product
metadata without downloading imagery:

```powershell
.\.venv\Scripts\python scripts/poll_live_insat.py --dry-run
.\.venv\Scripts\python scripts/poll_live_insat.py
```

It writes `data/processed/live_insat_manifest.json`, which the dashboard reads.
The dashboard applies a six-hour freshness gate to the provider's publication
time: delayed feed items are labelled **Stale metadata**, not live. A current
metadata status still does not mean an image was locally decoded or that an
operational forecast exists.

## Before public deployment

1. Register for any MOSDAC product-download access required by the selected INSAT imagery product.
2. Create an Earthdata Login and authorize GES DISC for IMERG Early/Late.
3. Store credentials locally, never in Git or frontend code.
4. Add age checks and alert suppression: a delayed product must be marked stale, never shown as current.
5. Treat official IMD warnings as authoritative; this dashboard is a research aid.
