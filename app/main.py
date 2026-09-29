import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.historical_replay import (
    data_ready,
    latest_replay_id,
    list_all_storm_tracks,
    list_replays,
    prediction_payload as baseline_prediction,
    replay,
)

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "ibtracs_ni_manifest.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LIVE_SOURCES_PATH = PROJECT_ROOT / "config" / "live_satellite_sources.json"

app = FastAPI(
    title="Cyclone Sentinel API",
    version="0.1.0",
    description="Research API for historical cyclone replay and multi-source feature engineering.",
)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def read_json_object(path: Path) -> dict:
    """Read a generated manifest without making the public status API fragile."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def source_layer_status(source: str) -> str:
    """Report measured feature coverage separately from collection planning."""
    feature_manifest = read_json_object(PROCESSED_DIR / f"{source}_features_manifest.json")
    feature_output = feature_manifest.get("output", {})
    records_written = feature_output.get("records_written", 0) if isinstance(feature_output, dict) else 0
    if feature_manifest.get("status") == "ready" and isinstance(records_written, int) and records_written > 0:
        return "ready"
    if feature_manifest.get("status") == "partial" and isinstance(records_written, int) and records_written > 0:
        return "partial_coverage"
    if isinstance(feature_manifest.get("status"), str) and feature_manifest["status"].startswith("waiting_for_"):
        return "waiting_for_raw_data"

    collection_manifest = read_json_object(PROCESSED_DIR / f"{source}_manifest.json")
    if source == "era5":
        requests = collection_manifest.get("requests")
        if isinstance(requests, list) and requests:
            return "ready_for_feature_extraction"
        return "credentials_required"

    collection_status = collection_manifest.get("status")
    if collection_status == "complete":
        return "ready_for_feature_extraction"
    if collection_status == "planned_dry_run":
        return "planned"
    if collection_status == "blocked_local_earthdata_credentials_missing":
        return "credentials_required"
    return "credentials_required"


def insat_layer_status() -> str:
    """Report only locally indexed imagery readiness, never RSS metadata as imagery."""
    catalog = read_json_object(PROCESSED_DIR / "insat_image_catalog.json")
    summary = catalog.get("summary")
    accepted = summary.get("accepted_images", 0) if isinstance(summary, dict) else 0
    if catalog.get("status") == "ready_for_feature_extraction" and isinstance(accepted, int) and accepted > 0:
        return "ready_for_feature_extraction"
    if (PROCESSED_DIR / "insat_image_catalog.json").exists():
        return "waiting_for_raw_data"
    return "registration_required"


def insat_training_index_status() -> str:
    """Expose only the readiness of real image-to-track supervised pairing."""
    manifest = read_json_object(PROCESSED_DIR / "insat_training_index_manifest.json")
    counts = manifest.get("counts")
    paired = counts.get("paired", 0) if isinstance(counts, dict) else 0
    if manifest.get("status") == "ready_for_image_feature_extraction" and isinstance(paired, int) and paired > 0:
        return "ready_for_feature_extraction"
    return "waiting_for_source_features"


def live_satellite_products() -> dict:
    """Return operational-source readiness without exposing credentials or raw files."""
    source_config = read_json_object(LIVE_SOURCES_PATH)
    source_entries = source_config.get("sources", [])
    products: list[dict] = []
    for source in source_entries if isinstance(source_entries, list) else []:
        if not isinstance(source, dict):
            continue
        source_id = source.get("id")
        if not isinstance(source_id, str):
            continue
        product = {
            "id": source_id,
            "name": source.get("name", source_id),
            "provider": source.get("provider", "Unknown provider"),
            "mode": source.get("mode", "Operational source"),
            "coverage": source.get("coverage", "Not specified"),
            "nominal_latency": source.get("nominal_latency", "Not specified"),
            "status": "credentials_required" if source.get("authentication") == "Earthdata Login" else "not_polled",
            "latest": None,
        }
        manifest_reference = source.get("manifest")
        if isinstance(manifest_reference, str):
            manifest = read_json_object(PROJECT_ROOT / manifest_reference)
            if manifest.get("status") == "ready":
                latest = manifest.get("latest")
                product["latest"] = latest if isinstance(latest, dict) else None
                published_at = latest.get("published_at") if isinstance(latest, dict) else None
                max_age = source.get("max_metadata_age_hours")
                try:
                    published = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
                    if published.tzinfo is None:
                        raise ValueError
                    age_hours = (datetime.now(UTC) - published.astimezone(UTC)).total_seconds() / 3600
                    product["metadata_age_hours"] = round(age_hours, 1)
                    product["status"] = "metadata_current" if isinstance(max_age, (int, float)) and age_hours <= max_age else "stale"
                except (TypeError, ValueError):
                    product["status"] = "metadata_unverified"
            elif manifest.get("status") == "unavailable":
                product["status"] = "unavailable"
        products.append(product)
    return {
        "products": products,
        "next_step": "Use fresh provider metadata only; then configure local provider access for imagery and GPM IMERG Early/Late.",
    }


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "historical_baseline" if data_ready() else "data_unavailable"}


@app.get("/api/v1/storms")
def list_storms() -> list[dict[str, str]]:
    if not data_ready():
        return []
    return [
        {
            "id": str(item["id"]),
            "name": str(item["name"]),
            "season": str(item["season"]),
            "basin": "North Indian Ocean",
            "peak_category": str(item.get("peak_category", "Cyclonic Storm")),
            "last_observed_at": str(item["last_observed_at"]),
        }
        for item in list_replays()
    ]


@app.get("/api/v1/storms/basin/all")
def basin_storm_tracks() -> list[dict]:
    """Return simplified tracks of all collected storms for full-basin visualization."""
    if not data_ready():
        return []
    return list_all_storm_tracks()


@app.get("/api/v1/storms/current")
def current_storm() -> dict:
    """Return the most recent collected storm with a valid historical replay."""
    if not data_ready():
        raise HTTPException(status_code=503, detail="Collected replay data is unavailable")
    try:
        return replay(latest_replay_id())
    except ValueError:
        raise HTTPException(status_code=503, detail="No replayable collected storm is available") from None


@app.get("/api/v1/storms/{storm_id}")
def storm(storm_id: str) -> dict:
    if not data_ready():
        raise HTTPException(status_code=503, detail="Collected replay data is unavailable")
    try:
        return replay(storm_id)
    except (KeyError, ValueError):
        raise HTTPException(status_code=404, detail="Historical storm replay not found") from None


@app.get("/api/v1/storms/{storm_id}/prediction")
def prediction(storm_id: str) -> dict:
    if not data_ready():
        raise HTTPException(status_code=503, detail="Collected replay data is unavailable")
    try:
        return baseline_prediction(storm_id)
    except (KeyError, ValueError):
        raise HTTPException(status_code=404, detail="Historical storm replay not found") from None


@app.get("/api/v1/storms/{storm_id}/bulletin")
def storm_bulletin(storm_id: str) -> dict[str, str]:
    """Return a clearly labelled non-operational historical research brief."""
    if not data_ready():
        raise HTTPException(status_code=503, detail="Collected replay data is unavailable")
    try:
        item = replay(storm_id)
        return {
            "storm_id": item["id"],
            "name": item["name"],
            "issued_at": item["last_updated"],
            "bulletin": item.get("bulletin_text", ""),
        }
    except (KeyError, ValueError):
        raise HTTPException(status_code=404, detail="Historical storm replay not found") from None


@app.get("/api/v1/storms/{storm_id}/bulletin/text", response_class=PlainTextResponse)
def storm_bulletin_text(storm_id: str) -> str:
    """Return the raw text of the non-operational historical research brief."""
    if not data_ready():
        raise HTTPException(status_code=503, detail="Collected replay data is unavailable")
    try:
        item = replay(storm_id)
        return item.get("bulletin_text", "Bulletin unavailable")
    except (KeyError, ValueError):
        raise HTTPException(status_code=404, detail="Historical storm replay not found") from None


@app.get("/api/v1/storms/{storm_id}/dvorak")
def storm_dvorak(storm_id: str) -> dict:
    """Return an intensity-derived Dvorak-scale proxy, not image analysis."""
    if not data_ready():
        raise HTTPException(status_code=503, detail="Collected replay data is unavailable")
    try:
        item = replay(storm_id)
        return {"storm_id": item["id"], "name": item["name"], "dvorak": item.get("dvorak", {})}
    except (KeyError, ValueError):
        raise HTTPException(status_code=404, detail="Historical storm replay not found") from None


@app.get("/api/v1/storms/{storm_id}/ri")
def storm_ri(storm_id: str) -> dict:
    """Return a non-operational Rapid Intensification screening heuristic."""
    if not data_ready():
        raise HTTPException(status_code=503, detail="Collected replay data is unavailable")
    try:
        item = replay(storm_id)
        return {"storm_id": item["id"], "name": item["name"], "rapid_intensification": item.get("rapid_intensification", {})}
    except (KeyError, ValueError):
        raise HTTPException(status_code=404, detail="Historical storm replay not found") from None


@app.get("/api/v1/storms/{storm_id}/impact")
def storm_impact(storm_id: str) -> dict:
    """Return static coastal-proximity and wind-only impact screening context."""
    if not data_ready():
        raise HTTPException(status_code=503, detail="Collected replay data is unavailable")
    try:
        item = replay(storm_id)
        return {
            "storm_id": item["id"],
            "name": item["name"],
            "landfall": item.get("landfall", {}),
            "district_risk": item.get("district_risk", []),
        }
    except (KeyError, ValueError):
        raise HTTPException(status_code=404, detail="Historical storm replay not found") from None


@app.get("/api/v1/data/status")
def data_status() -> dict:
    """Report whether historical training data has been collected."""
    if MANIFEST_PATH.exists():
        manifest = read_json_object(MANIFEST_PATH)
        if manifest:
            return manifest
    return {
        "status": "not_collected",
        "source": "NOAA IBTrACS North Indian Ocean best-track archive",
        "next_step": "Run: python scripts/collect_ibtracs.py --start-year 2000",
    }


@app.get("/api/v1/data/catalog")
def data_catalog() -> dict:
    """Describe source-layer readiness without exposing raw data files."""
    readiness_manifest = read_json_object(PROCESSED_DIR / "collection_readiness.json")
    readiness_sources = readiness_manifest.get("sources") if isinstance(readiness_manifest.get("sources"), list) else []
    readiness_setup_needed = any(isinstance(item, dict) and item.get("status") in {"setup_required", "registration_required"} for item in readiness_sources)
    multisource_manifest_path = PROCESSED_DIR / "multisource_training_manifest.json"
    multisource_manifest = read_json_object(multisource_manifest_path)
    raw_counts = multisource_manifest.get("counts", {})
    multisource_counts = raw_counts if isinstance(raw_counts, dict) else {}
    multisource_status = (
        "ready"
        if multisource_counts.get("complete", 0) > 0
        else "waiting_for_source_features"
        if multisource_manifest_path.exists()
        else "pending"
    )
    layers = [
        {
            "id": "ibtracs",
            "name": "NOAA IBTrACS best tracks",
            "status": "ready" if MANIFEST_PATH.exists() else "pending",
            "purpose": "Historical track and intensity labels",
        },
        {
            "id": "supervised_samples",
            "name": "24-hour supervised samples",
            "status": "ready" if (PROCESSED_DIR / "training_samples_manifest.json").exists() else "pending",
            "purpose": "Leakage-aware baseline training examples",
        },
        {
            "id": "baseline",
            "name": "Ridge research baseline",
            "status": "ready" if (PROCESSED_DIR / "baseline_metrics.json").exists() else "pending",
            "purpose": "24-hour track and wind benchmark",
        },
        {
            "id": "collection_preflight",
            "name": "Local collection preflight",
            "status": "setup_required" if readiness_setup_needed else "ready" if readiness_manifest else "pending",
            "purpose": "Safe local check for provider setup, dependencies, and raw-file counts; it never exposes credentials.",
        },
        {
            "id": "era5",
            "name": "ERA5 atmospheric context",
            "status": source_layer_status("era5"),
            "purpose": "Wind, pressure, SST, water-vapour features",
        },
        {
            "id": "imerg",
            "name": "GPM IMERG rainfall",
            "status": source_layer_status("imerg"),
            "purpose": "Historic precipitation features",
        },
        {
            "id": "multisource_cohort",
            "name": "Event-aligned multi-source cohort",
            "status": multisource_status,
            "purpose": "Joined labels and source features for model experiments",
            "counts": multisource_counts,
        },
        {
            "id": "multisource_baseline",
            "name": "ERA5 + IMERG research benchmark",
            "status": "ready" if (PROCESSED_DIR / "multisource_baseline_metrics.json").exists() else "waiting_for_source_features",
            "purpose": "Strict complete-case research benchmark; never trained on missing source values",
        },
        {
            "id": "insat",
            "name": "MOSDAC INSAT imagery",
            "status": insat_layer_status(),
            "purpose": "Real image cataloguing and future satellite pattern classification",
        },
        {
            "id": "insat_supervised_index",
            "name": "INSAT + IBTrACS training pairs",
            "status": insat_training_index_status(),
            "purpose": "Timestamp-matched image files and historical intensity labels for future ML training",
        },
    ]
    return {
        "layers": layers,
        "next_milestone": readiness_manifest.get("next_action") if isinstance(readiness_manifest.get("next_action"), str) else "Run scripts/check_collection_readiness.py --write, then configure local CDS credentials and collect event-aligned ERA5.",
    }


@app.get("/api/v1/live/products")
def live_products() -> dict:
    """Describe real-time satellite product readiness and freshest provider metadata."""
    return live_satellite_products()
