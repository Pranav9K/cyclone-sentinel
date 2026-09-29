"""Serve transparent historical replays backed by collected IBTrACS data.

Supplementary visual modules are explicitly labelled research proxies until
real satellite and environmental sources are connected.
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

from app.bulletin_generator import generate_research_brief
from app.dvorak_engine import classify_pattern
from app.landfall_impact_engine import evaluate_landfall_and_impact
from app.ri_engine import evaluate_rapid_intensification

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKS_PATH = PROJECT_ROOT / "data" / "processed" / "ibtracs_ni_tracks.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_model.json"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_metrics.json"
ERA5_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "era5_features.csv"
IMERG_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "imerg_features.csv"
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
        parsed = float(value) if value not in (None, "") else None
        return parsed if parsed is None or math.isfinite(parsed) else None
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


def _load_feature_rows(path: Path, source_prefix: str) -> dict[tuple[str, str], dict[str, str]]:
    """Read an optional extracted feature table without treating absent data as zero."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            fields = set(reader.fieldnames or [])
            if {"storm_id", "timestamp_utc", f"{source_prefix}_missing"} - fields:
                return {}
            records: dict[tuple[str, str], dict[str, str]] = {}
            for row in reader:
                storm_id = (row.get("storm_id") or "").strip()
                timestamp = (row.get("timestamp_utc") or "").strip()
                if storm_id and timestamp:
                    records[(storm_id, timestamp)] = row
            return records
    except OSError:
        return {}


@lru_cache(maxsize=1)
def load_era5_features() -> dict[tuple[str, str], dict[str, str]]:
    return _load_feature_rows(ERA5_FEATURES_PATH, "era5")


@lru_cache(maxsize=1)
def load_imerg_features() -> dict[tuple[str, str], dict[str, str]]:
    return _load_feature_rows(IMERG_FEATURES_PATH, "imerg")


def _source_features(storm_id: str, timestamp: datetime) -> dict[str, Any]:
    """Return only source values that are present in a matching extracted row."""
    key = (storm_id, timestamp.isoformat().replace("+00:00", "Z"))
    era5 = load_era5_features().get(key)
    imerg = load_imerg_features().get(key)
    era5_available = bool(era5) and str(era5.get("era5_missing", "1")).strip().lower() not in {"1", "true", "yes"}
    imerg_available = bool(imerg) and str(imerg.get("imerg_missing", "1")).strip().lower() not in {"1", "true", "yes"}
    sst_k = _optional_float(era5.get("era5_sst_k") if era5_available and era5 else None)
    rainfall = _optional_float(imerg.get("imerg_center_precipitation_cal_mm_hr") if imerg_available and imerg else None)
    return {
        "era5_available": era5_available,
        "imerg_available": imerg_available,
        "era5_sst_celsius": round(sst_k - 273.15, 2) if sst_k is not None else None,
        "imerg_center_precipitation_mm_hr": rainfall,
    }


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


def _calculate_asymmetric_wind_radii(wind_knots: float | None) -> dict[str, dict[str, int] | None]:
    """Calculate quadrant-specific radii in km for 34kt, 50kt, and 64kt thresholds."""
    if wind_knots is None or wind_knots < 34.0:
        return {"r34_km": None, "r50_km": None, "r64_km": None}

    # 34 kt (Gale-force) base radius
    base_34 = min(320.0, max(120.0, 110.0 + (wind_knots - 34.0) * 1.5))
    r34 = {
        "ne": round(base_34 * 1.15),
        "se": round(base_34 * 0.95),
        "sw": round(base_34 * 0.80),
        "nw": round(base_34 * 1.05),
    }

    # 50 kt (Storm-force) radius
    r50 = None
    if wind_knots >= 50.0:
        base_50 = min(180.0, max(60.0, 55.0 + (wind_knots - 50.0) * 1.2))
        r50 = {
            "ne": round(base_50 * 1.15),
            "se": round(base_50 * 0.95),
            "sw": round(base_50 * 0.80),
            "nw": round(base_50 * 1.05),
        }

    # 64 kt (Hurricane-force) radius
    r64 = None
    if wind_knots >= 64.0:
        base_64 = min(110.0, max(35.0, 30.0 + (wind_knots - 64.0) * 0.9))
        r64 = {
            "ne": round(base_64 * 1.15),
            "se": round(base_64 * 0.95),
            "sw": round(base_64 * 0.80),
            "nw": round(base_64 * 1.05),
        }

    return {"r34_km": r34, "r50_km": r50, "r64_km": r64}


def _calculate_cone_polygon(
    origin_lat: float, origin_lon: float, forecast_track: list[dict[str, Any]]
) -> list[list[float]]:
    """Compute an error-scaled visual envelope from held-out baseline error."""
    if not forecast_track:
        return []

    # Assemble track points: [origin, p24, p48, p72] with respective uncertainty radii
    points = [{"lat": origin_lat, "lon": origin_lon, "radius_km": 25.0}] + [
        {"lat": pt["lat"], "lon": pt["lon"], "radius_km": pt["radius_km"]}
        for pt in forecast_track
    ]

    left_edge: list[list[float]] = []
    right_edge: list[list[float]] = []

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        dlat = p2["lat"] - p1["lat"]
        dlon = (p2["lon"] - p1["lon"]) * math.cos(math.radians(p1["lat"]))
        mag = math.hypot(dlat, dlon) or 1e-6
        # Unit normal vector (perpendicular to trajectory)
        norm_lat = -dlon / mag
        norm_lon = dlat / mag

        # Degrees per km roughly 1/111
        deg_lat1 = p1["radius_km"] / 111.0
        deg_lon1 = p1["radius_km"] / (111.0 * max(0.1, math.cos(math.radians(p1["lat"]))))

        left_edge.append([round(p1["lat"] + norm_lat * deg_lat1, 3), round(p1["lon"] + norm_lon * deg_lon1, 3)])
        right_edge.append([round(p1["lat"] - norm_lat * deg_lat1, 3), round(p1["lon"] - norm_lon * deg_lon1, 3)])

    # Add terminal semicircular arc around the last forecast point
    last_pt = points[-1]
    deg_lat_last = last_pt["radius_km"] / 111.0
    deg_lon_last = last_pt["radius_km"] / (111.0 * max(0.1, math.cos(math.radians(last_pt["lat"]))))

    arc_points: list[list[float]] = []
    for angle_deg in range(-90, 91, 15):
        rad = math.radians(angle_deg)
        # Approximate terminal cap
        arc_lat = last_pt["lat"] + deg_lat_last * math.cos(rad)
        arc_lon = last_pt["lon"] + deg_lon_last * math.sin(rad)
        arc_points.append([round(arc_lat, 3), round(arc_lon, 3)])

    # Construct closed polygon: left boundary + terminal arc + reversed right boundary
    polygon = left_edge + arc_points + list(reversed(right_edge))
    return polygon


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
    result = [
        {
            "time": item.time.isoformat().replace("+00:00", "Z"),
            "lat": item.latitude,
            "lon": item.longitude,
            "wind_kmph": round(item.wind_knots * KNOTS_TO_KMPH) if item.wind_knots is not None else None,
            "pressure_hpa": item.pressure_hpa,
            "stage": _classification(item.wind_knots),
        }
        for index, item in enumerate(visible)
        if index % 2 == 0 or index == len(visible) - 1
    ]
    return result


def resolve_storm_id(selector: str) -> str:
    if selector.strip().lower() == "current":
        return latest_replay_id()
    tracks = load_tracks()
    if selector in tracks:
        return selector
    normalized = selector.replace("-", " ").upper()
    for storm_id, observations in tracks.items():
        first = observations[0]
        candidates = {
            first.name.upper(),
            f"{first.name} {first.season}".upper(),
            f"{first.name}-{first.season}".upper(),
            f"CYCLONE {first.name}".upper(),
        }
        if normalized in candidates:
            return storm_id
    raise KeyError(selector)


def list_replays(limit: int = 100) -> list[dict[str, Any]]:
    """List collected storms that can support the 72-hour replay view."""
    named: list[tuple[datetime, dict[str, Any]]] = []
    for storm_id, track in load_tracks().items():
        if not track or track[0].name == "UNNAMED":
            continue
        try:
            current, _, _ = _issue_observation(track)
        except ValueError:
            continue
        first = track[0]
        last = track[-1]
        named.append((
            last.time,
            {
                "id": storm_id,
                "name": first.name.title(),
                "season": first.season,
                "points": len(track),
                "peak_category": _classification(max((obs.wind_knots or 0) for obs in track)),
                "last_observed_at": last.time.isoformat().replace("+00:00", "Z"),
            },
        ))
    return [item for _, item in sorted(named, key=lambda entry: entry[0], reverse=True)[:limit]]


def latest_replay_id() -> str:
    """Return the newest replayable storm based on its collected observation time."""
    available = list_replays(limit=1)
    if not available:
        raise ValueError("No collected storm can support a 72-hour historical replay.")
    return str(available[0]["id"])


def list_all_storm_tracks() -> list[dict[str, Any]]:
    """Return simplified tracks for all named storms for basin-wide exploration."""
    results: list[dict[str, Any]] = []
    for storm_id, track in load_tracks().items():
        if not track or track[0].name == "UNNAMED":
            continue
        max_wind = max((obs.wind_knots or 0) for obs in track)
        results.append({
            "id": storm_id,
            "name": f"Cyclone {track[0].name.title()}",
            "season": track[0].season,
            "peak_category": _classification(max_wind),
            "peak_wind_kmph": round(max_wind * KNOTS_TO_KMPH),
            "points": [[round(obs.latitude, 2), round(obs.longitude, 2)] for obs in track],
        })
    return results


def replay(selector: str) -> dict[str, Any]:
    storm_id = resolve_storm_id(selector)
    track = load_tracks()[storm_id]
    current, previous, issue_index = _issue_observation(track)
    forecast = _forecast(current, previous, track[0].time)
    metrics = load_metrics().get("test_metrics", {})
    track_metrics = metrics.get("track", {})
    intensity_metrics = metrics.get("intensity", {})

    wind_kmph = round(current.wind_knots * KNOTS_TO_KMPH) if current.wind_knots is not None else None
    source_features = _source_features(storm_id, current.time)

    # Research intensity-derived Dvorak-scale proxy; it contains no imagery analysis.
    dvorak = classify_pattern(
        wind_knots=current.wind_knots,
        pressure_hpa=current.pressure_hpa,
        latitude=current.latitude,
        longitude=current.longitude,
    )

    # ERA5 SST is injected only when the exact extracted row is present; shear
    # remains a declared seasonal proxy until an upper-air extractor is added.
    ri = evaluate_rapid_intensification(
        current_wind_knots=current.wind_knots,
        prior_wind_knots=previous.wind_knots,
        latitude=current.latitude,
        longitude=current.longitude,
        season_month=current.time.month,
        sea_surface_temperature_c=source_features["era5_sst_celsius"],
    )

    # Static coastal-proximity screen, not a coastline-intersection or warning engine.
    landfall, district_impacts = evaluate_landfall_and_impact(
        forecast_track=forecast,
        current_lat=current.latitude,
        current_lon=current.longitude,
        current_wind_knots=current.wind_knots,
        issue_time=current.time,
    )

    # Wind-extent display proxy derived only from wind speed.
    wind_radii = _calculate_asymmetric_wind_radii(current.wind_knots)

    # Held-out-error-scaled display envelope; no probabilistic calibration is implied.
    cone_polygon = _calculate_cone_polygon(current.latitude, current.longitude, forecast)

    base_payload = {
        "id": storm_id,
        "name": f"Cyclone {current.name.title()}",
        "season": str(current.season),
        "basin": "North Indian Ocean · Historical research replay",
        "data_mode": "IBTrACS historical replay + research screening modules",
        "status": _classification(current.wind_knots),
        "last_updated": current.time.isoformat().replace("+00:00", "Z"),
        "current": {
            "lat": current.latitude,
            "lon": current.longitude,
            "wind_kmph": wind_kmph,
            "pressure_hpa": current.pressure_hpa,
            "rainfall_mm_hr": source_features["imerg_center_precipitation_mm_hr"],
            "wind_radii": wind_radii,
        },
        "classification": {
            "label": _classification(current.wind_knots),
            "confidence": None,
            "detail": f"Intensity-derived Dvorak-scale proxy (CI{dvorak.ci_number:.1f}); no INSAT image model is connected.",
        },
        "dvorak": {
            "assessment_mode": dvorak.assessment_mode,
            "limitations": dvorak.limitations,
            "pattern_type": dvorak.pattern_type,
            "pattern_description": dvorak.pattern_description,
            "t_number": dvorak.t_number,
            "ci_number": dvorak.ci_number,
            "central_pressure_deficit_hpa": dvorak.central_pressure_deficit_hpa,
            "environmental_pressure_hpa": dvorak.environmental_pressure_hpa,
            "estimated_central_pressure_hpa": dvorak.estimated_central_pressure_hpa,
            "eye_characteristics": dvorak.eye_characteristics,
            "cloud_metrics": dvorak.cloud_metrics,
            "pattern_probabilities": dvorak.pattern_probabilities,
            "attribution": dvorak.attribution,
        },
        "rapid_intensification": {
            "ri_score": ri.ri_score,
            "screening_level": ri.screening_level,
            "status_label": ri.status_label,
            "input_mode": ri.input_mode,
            "limitations": ri.limitations,
            "summary": ri.summary,
            "favorable_factors": ri.favorable_factors,
            "inhibiting_factors": ri.inhibiting_factors,
            "factor_scores": ri.factor_scores,
            "projected_24h_wind_normal_knots": ri.projected_24h_wind_normal_knots,
            "projected_24h_wind_ri_knots": ri.projected_24h_wind_ri_knots,
        },
        "landfall": {
            "assessment_mode": landfall.assessment_mode,
            "limitations": landfall.limitations,
            "will_make_landfall": landfall.will_make_landfall,
            "landfall_point": landfall.landfall_point,
            "nearest_landmark": landfall.nearest_landmark,
            "nearest_port": landfall.nearest_port,
            "eta_hours": landfall.eta_hours,
            "eta_timestamp_utc": landfall.eta_timestamp_utc,
            "eta_timestamp_ist": landfall.eta_timestamp_ist,
            "wind_at_landfall_knots": landfall.wind_at_landfall_knots,
            "wind_at_landfall_kmph": landfall.wind_at_landfall_kmph,
            "gust_kmph": landfall.gust_kmph,
            "category_at_landfall": landfall.category_at_landfall,
            "storm_surge_meters": landfall.storm_surge_meters,
            "surge_warning_level": landfall.surge_warning_level,
            "coastline_sector": landfall.coastline_sector,
        },
        "model_metrics": {
            "track_error_km": track_metrics.get("endpoint_mae_km"),
            "intensity_mae_knots": intensity_metrics.get("wind_mae_knots"),
            "classification_f1": None,
            "model_name": "Ridge research baseline v0.1 · 24-hour horizon",
        },
        "observed_track": _display_track(track, issue_index),
        "forecast_track": forecast,
        "cone_polygon": cone_polygon,
        "district_risk": district_impacts,
        "provenance": {
            "observations": "NOAA IBTrACS v04r01, North Indian Ocean subset",
            "era5_sst": "Exact event-aligned ERA5 extraction" if source_features["era5_available"] else "Unavailable for this replay timestamp; RI uses a seasonal SST proxy.",
            "imerg_rainfall": "Exact event-aligned GPM IMERG extraction" if source_features["imerg_available"] else "Unavailable for this replay timestamp; rainfall is intentionally blank.",
            "forecast": "Ridge baseline with a held-out-error-scaled display envelope; not a calibrated probability cone.",
            "dvorak": "Intensity-to-Dvorak-scale proxy from best-track wind and pressure; no satellite imagery is used.",
            "ri_engine": "Best-track trend plus seasonal-proxy screening score; not a calibrated probability.",
            "reference_future_available": True,
        },
    }

    bulletin_text = generate_research_brief(base_payload)
    base_payload["bulletin_text"] = bulletin_text

    return base_payload


def prediction_payload(selector: str) -> dict[str, Any]:
    item = replay(selector)
    return {
        "storm_id": item["id"],
        "name": item["name"],
        "generated_at": item["last_updated"],
        "track": item["forecast_track"],
        "cone_polygon": item.get("cone_polygon", []),
        "intensity": {
            "current_wind_kmph": item["current"]["wind_kmph"],
            "category": item["status"],
            "dvorak_t_number": item["dvorak"]["t_number"],
            "rapid_intensification_risk": item["rapid_intensification"]["status_label"],
            "model": item["model_metrics"]["model_name"],
        },
        "landfall": item["landfall"],
        "wind_radii": item["current"]["wind_radii"],
        "uncertainty": {str(point["hours"]): point["radius_km"] for point in item["forecast_track"]},
        "disclaimer": "AI-assisted research model for SIH26070. Follow official IMD / RSMC bulletins for operational safety.",
    }
