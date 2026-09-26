import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.historical_replay import data_ready, list_replays, prediction_payload as baseline_prediction, replay
from app.sample_data import STORM, prediction_payload

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "ibtracs_ni_manifest.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

app = FastAPI(
    title="Cyclone Sentinel API",
    version="0.1.0",
    description="Research API for historical cyclone replay and multi-source feature engineering.",
)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "historical_baseline" if data_ready() else "demo"}


@app.get("/api/v1/storms")
def list_storms() -> list[dict[str, str]]:
    if data_ready():
        return [{"id": item["id"], "name": item["name"], "season": str(item["season"]), "basin": "North Indian Ocean"} for item in list_replays()]
    return [{"id": STORM["id"], "name": STORM["name"], "season": STORM["season"], "basin": STORM["basin"]}]


@app.get("/api/v1/storms/{storm_id}")
def storm(storm_id: str) -> dict:
    if data_ready():
        try:
            return replay(storm_id)
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail="Historical storm replay not found") from None
    if storm_id != STORM["id"]:
        raise HTTPException(status_code=404, detail="Storm not found")
    return STORM


@app.get("/api/v1/storms/{storm_id}/prediction")
def prediction(storm_id: str) -> dict:
    if data_ready():
        try:
            return baseline_prediction(storm_id)
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail="Historical storm replay not found") from None
    if storm_id != STORM["id"]:
        raise HTTPException(status_code=404, detail="Storm not found")
    return prediction_payload()


@app.get("/api/v1/data/status")
def data_status() -> dict:
    """Report whether official historical training data has been collected."""
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        "status": "not_collected",
        "source": "NOAA IBTrACS North Indian Ocean best-track archive",
        "next_step": "Run: python scripts/collect_ibtracs.py --start-year 2000",
    }


@app.get("/api/v1/data/catalog")
def data_catalog() -> dict:
    """Describe source-layer readiness without exposing raw data files."""
    multisource_manifest_path = PROCESSED_DIR / "multisource_training_manifest.json"
    multisource_counts: dict[str, int] = {}
    if multisource_manifest_path.exists():
        multisource_counts = json.loads(multisource_manifest_path.read_text(encoding="utf-8")).get("counts", {})
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
            "id": "era5",
            "name": "ERA5 atmospheric context",
            "status": "ready" if (PROCESSED_DIR / "era5_manifest.json").exists() else "credentials_required",
            "purpose": "Wind, pressure, SST, water-vapour features",
        },
        {
            "id": "imerg",
            "name": "GPM IMERG rainfall",
            "status": "ready" if (PROCESSED_DIR / "imerg_manifest.json").exists() else "credentials_required",
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
            "id": "insat",
            "name": "MOSDAC INSAT imagery",
            "status": "registration_required",
            "purpose": "Satellite pattern classification",
        },
    ]
    return {"layers": layers, "next_milestone": "Collect event-aligned ERA5 data for Cyclone Biparjoy."}
