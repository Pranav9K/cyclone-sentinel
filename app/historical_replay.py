"""Serve transparent historical replays backed by collected IBTrACS data.

This module is deliberately small and dependency-free.  It uses the trained
ridge-regression artifact only for *research replay* forecasts; it does not
turn historical best-track observations into a live safety forecast.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKS_PATH = PROJECT_ROOT / "data" / "processed" / "ibtracs_ni_tracks.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_model.json"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_metrics.json"
KNOTS_TO_KMPH = 1.852


@dataclass(frozen=True)
class Observation:
    storm_id: str
    season: int
    name: str
    time: datetime
    latitude: float
    longitude: float
    wind_knots: float | None
    pressure_hpa: float | None


def _optional_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _normalize_longitude(longitude: float) -> float:
    return (longitude + 180.0) % 360.0 - 180.0


def _longitude_delta(start: float, end: float) -> float:
    return _normalize_longitude(end - start)


def _season_features(timestamp: datetime) -> tuple[float, float]:
    angle = 2 * math.pi * timestamp.timetuple().tm_yday / 365.25
    return math.sin(angle), math.cos(angle)


@lru_cache(maxsize=1)
def load_tracks() -> dict[str, list[Observation]]:
    """Load valid collected observations, grouped and ordered by storm."""
    if not TRACKS_PATH.exists():
        return {}
    grouped: dict[str, list[Observation]] = {}
    with TRACKS_PATH.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            try:
                observation = Observation(
                    storm_id=row["storm_id"],
                    season=int(row["season"]),
                    name=row["name"].strip() or "UNNAMED",
                    time=datetime.fromisoformat(row["timestamp_utc"]).replace(tzinfo=UTC),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    wind_knots=_optional_float(row.get("wind_knots")),
                    pressure_hpa=_optional_float(row.get("pressure_hpa")),
                )
            except (KeyError, TypeError, ValueError):
                continue
            grouped.setdefault(observation.storm_id, []).append(observation)
    for storm in grouped.values():
        storm.sort(key=lambda item: item.time)
    return grouped


@lru_cache(maxsize=1)
def load_model() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        return {}
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def data_ready() -> bool:
    return bool(load_tracks() and load_model() and load_metrics())


def _predict_ridge(regressor: dict[str, Any], features: list[float]) -> float:
    means = regressor["feature_means"]
    scales = regressor["feature_scales"]
    coefficients = regressor["coefficients_standardized"]
    if not (len(features) == len(means) == len(scales) == len(coefficients) - 1):
        raise ValueError("Baseline inference contract does not match supplied features.")
    return coefficients[0] + sum(
        ((value - mean) / scale) * coefficient
        for value, mean, scale, coefficient in zip(features, means, scales, coefficients[1:])
    )


def _classification(wind_knots: float | None) -> str:
    if wind_knots is None:
        return "Intensity unavailable"
    # IMD wind-category thresholds, used only for a readable replay label.
    if wind_knots >= 120:
        return "Super Cyclonic Storm"
    if wind_knots >= 90:
        return "Extremely Severe Cyclonic Storm"
    if wind_knots >= 64:
        return "Very Severe Cyclonic Storm"
    if wind_knots >= 48:
        return "Severe Cyclonic Storm"
    if wind_knots >= 34:
        return "Cyclonic Storm"
    if wind_knots >= 28:
        return "Deep Depression"
    if wind_knots >= 17:
        return "Depression"
    return "Low Pressure Area"


def _issue_observation(track: list[Observation]) -> tuple[Observation, Observation, int]:
    """Pick a late replay point with a 6-hour history and 72-hour reference future."""
    by_time = {item.time: item for item in track}
    for index in range(len(track) - 1, -1, -1):
        current = track[index]
        previous = by_time.get(current.time - timedelta(hours=6))
        if previous and by_time.get(current.time + timedelta(hours=72)):
            return current, previous, index
    raise ValueError("Storm has insufficient 6-hour history and 72-hour replay horizon.")


def _forecast(current: Observation, previous: Observation, storm_start: datetime) -> list[dict[str, Any]]:
    model = load_model()
    track_models = model["track"]
    wind_model = model["intensity"]["wind_delta_knots"]
    metrics = load_metrics().get("test_metrics", {})
    base_error = float(metrics.get("track", {}).get("endpoint_mae_km", 150))

    state_lat, state_lon, state_wind, state_time = current.latitude, current.longitude, current.wind_knots, current.time
    prior_lat, prior_lon, prior_wind = previous.latitude, previous.longitude, previous.wind_knots
    forecast: list[dict[str, Any]] = []

    for hours in (24, 48, 72):
        latitude_motion = state_lat - prior_lat
        longitude_motion = _longitude_delta(prior_lon, state_lon)
        age_days = (state_time - storm_start).total_seconds() / 86_400
        season_sin, season_cos = _season_features(state_time)
        track_features = [state_lat, state_lon, latitude_motion, longitude_motion, age_days, season_sin, season_cos]
        latitude_delta = _predict_ridge(track_models["latitude_delta_degrees"], track_features)
        longitude_delta = _predict_ridge(track_models["longitude_delta_degrees"], track_features)

        wind_delta: float | None = None
        next_wind: float | None = None
        if state_wind is not None:
            wind_change = state_wind - prior_wind if prior_wind is not None else 0.0
            wind_features = [state_wind, wind_change, float(prior_wind is not None), state_lat, state_lon, age_days, season_sin, season_cos]
            wind_delta = _predict_ridge(wind_model, wind_features)
            next_wind = max(0.0, state_wind + wind_delta)

        next_latitude = state_lat + latitude_delta
        next_longitude = _normalize_longitude(state_lon + longitude_delta)
        # Recursive forecasts do not have 6-hour observations. Approximate the
        # latest six-hour motion from the just-predicted 24-hour displacement.
        # The model's motion features are 6-hour deltas, not 24-hour deltas.
        prior_lat = next_latitude - latitude_delta / 4
        prior_lon = _normalize_longitude(next_longitude - longitude_delta / 4)
        prior_wind = next_wind - wind_delta / 4 if next_wind is not None and wind_delta is not None else None
        state_lat, state_lon, state_wind = next_latitude, next_longitude, next_wind
        state_time += timedelta(hours=24)
        forecast.append({
            "hours": hours,
            "time": state_time.isoformat().replace("+00:00", "Z"),
            "lat": round(next_latitude, 3),
            "lon": round(next_longitude, 3),
            "wind_kmph": round(next_wind * KNOTS_TO_KMPH) if next_wind is not None else None,
            "radius_km": round(base_error * math.sqrt(hours / 24)),
        })
    return forecast


def _display_track(track: list[Observation], issue_index: int) -> list[dict[str, Any]]:
    visible = track[: issue_index + 1]
    # A 12-hour cadence keeps the replay legible without altering source data.
    result = [
        {"time": item.time.isoformat().replace("+00:00", "Z"), "lat": item.latitude, "lon": item.longitude}
        for index, item in enumerate(visible)
        if index % 4 == 0 or index == len(visible) - 1
    ]
    return result


def resolve_storm_id(selector: str) -> str:
    tracks = load_tracks()
    if selector in tracks:
        return selector
    normalized = selector.replace("-", " ").upper()
    for storm_id, observations in tracks.items():
        first = observations[0]
        candidates = {first.name.upper(), f"{first.name} {first.season}".upper(), f"{first.name}-{first.season}".upper()}
        if normalized in candidates:
            return storm_id
    raise KeyError(selector)


def list_replays(limit: int = 20) -> list[dict[str, str | int]]:
    named: list[dict[str, str | int]] = []
    for storm_id, track in load_tracks().items():
        if not track or track[0].name == "UNNAMED":
            continue
        first = track[0]
        named.append({"id": storm_id, "name": first.name.title(), "season": first.season, "points": len(track)})
    return sorted(named, key=lambda item: (int(item["season"]), str(item["name"])), reverse=True)[:limit]


def replay(selector: str) -> dict[str, Any]:
    storm_id = resolve_storm_id(selector)
    track = load_tracks()[storm_id]
    current, previous, issue_index = _issue_observation(track)
    forecast = _forecast(current, previous, track[0].time)
    metrics = load_metrics().get("test_metrics", {})
    track_metrics = metrics.get("track", {})
    intensity_metrics = metrics.get("intensity", {})
    wind_kmph = round(current.wind_knots * KNOTS_TO_KMPH) if current.wind_knots is not None else None
    return {
        "id": storm_id,
        "name": f"Cyclone {current.name.title()}",
        "season": str(current.season),
        "basin": "North Indian Ocean · Historical Replay",
        "data_mode": "Historical replay — NOAA IBTrACS observations + research baseline",
        "status": _classification(current.wind_knots),
        "last_updated": current.time.isoformat().replace("+00:00", "Z"),
        "current": {"lat": current.latitude, "lon": current.longitude, "wind_kmph": wind_kmph, "pressure_hpa": current.pressure_hpa, "rainfall_mm_hr": None},
        "classification": {
            "label": _classification(current.wind_knots),
            "confidence": None,
            "detail": "Category is derived from the best-track wind; this baseline has no calibrated confidence score.",
        },
        "model_metrics": {
            "track_error_km": track_metrics.get("endpoint_mae_km"),
            "intensity_mae_knots": intensity_metrics.get("wind_mae_knots"),
            "classification_f1": None,
            "model_name": "Ridge baseline v0.1 · 24-hour horizon",
        },
        "observed_track": _display_track(track, issue_index),
        "forecast_track": forecast,
        "district_risk": [
            {"district": "Impact layer pending", "level": "Prototype", "score": None, "drivers": ["Connect official IMD warning polygons", "Add coastal district boundaries", "Add GPM rainfall and exposure data"]},
        ],
        "provenance": {
            "observations": "NOAA IBTrACS v04r01, North Indian Ocean subset",
            "forecast": "Research-only 24-hour ridge baseline recursively rolled to 72 hours",
            "reference_future_available": True,
        },
    }


def prediction_payload(selector: str) -> dict[str, Any]:
    item = replay(selector)
    return {
        "storm_id": item["id"],
        "generated_at": item["last_updated"],
        "track": item["forecast_track"],
        "intensity": {"current_wind_kmph": item["current"]["wind_kmph"], "model": item["model_metrics"]["model_name"]},
        "uncertainty": {str(point["hours"]): point["radius_km"] for point in item["forecast_track"]},
        "disclaimer": "Historical research replay only. It is not an official or operational forecast; follow IMD warnings.",
    }
