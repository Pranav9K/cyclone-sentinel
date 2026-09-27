"""Coastal Landfall Detection, Storm Surge, and District Vulnerability Engine.

Implements the official Disaster Management decision-support layer for
Ministry of Earth Sciences (MoES) and NDMA/SDMA emergency operations centers.

Computes:
1. Exact Landfall point (lat, lon) and coastal landmark / port intersection
2. Landfall Time Window (ETA in hours and UTC/IST timestamps)
3. Projected Landfall Intensity, Category, and Peak Gust speed
4. Estimated Storm Surge Inundation Height (meters)
5. Coastal District Warning Tiers (Red, Orange, Yellow) with population exposure
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


@dataclass(frozen=True)
class LandfallForecast:
    will_make_landfall: bool
    landfall_point: dict[str, float] | None
    nearest_landmark: str
    nearest_port: str
    eta_hours: float | None
    eta_timestamp_utc: str | None
    eta_timestamp_ist: str | None
    wind_at_landfall_knots: float | None
    wind_at_landfall_kmph: float | None
    gust_kmph: float | None
    category_at_landfall: str
    storm_surge_meters: float | None
    surge_warning_level: str
    coastline_sector: str


# Prominent coastal districts along the North Indian Ocean coastline
COASTAL_DISTRICTS = [
    # Gujarat
    {"district": "Kutch", "state": "Gujarat", "lat": 23.25, "lon": 69.67, "population": 2092371, "port": "Kandla / Mundra Port", "surge_risk": "Very High"},
    {"district": "Devbhumi Dwarka", "state": "Gujarat", "lat": 22.24, "lon": 68.96, "population": 752484, "port": "Okha Port", "surge_risk": "High"},
    {"district": "Jamnagar", "state": "Gujarat", "lat": 22.47, "lon": 70.07, "population": 2160117, "port": "Bedi Port", "surge_risk": "High"},
    {"district": "Porbandar", "state": "Gujarat", "lat": 21.64, "lon": 69.60, "population": 585449, "port": "Porbandar Port", "surge_risk": "Moderate"},
    {"district": "Junagadh", "state": "Gujarat", "lat": 21.52, "lon": 70.45, "population": 2743082, "port": "Veraval Port", "surge_risk": "Moderate"},
    {"district": "Gir Somnath", "state": "Gujarat", "lat": 20.90, "lon": 70.37, "population": 1217477, "port": "Veraval / Jafrabad", "surge_risk": "Moderate"},
    {"district": "Bhavnagar", "state": "Gujarat", "lat": 21.76, "lon": 72.15, "population": 2880365, "port": "Alang / Bhavnagar Port", "surge_risk": "High"},
    # Maharashtra & Goa
    {"district": "Mumbai & Suburban", "state": "Maharashtra", "lat": 18.97, "lon": 72.82, "population": 12442373, "port": "JNP / Mumbai Port", "surge_risk": "High"},
    {"district": "Raigad", "state": "Maharashtra", "lat": 18.51, "lon": 73.18, "population": 2634200, "port": "Dighi Port", "surge_risk": "Moderate"},
    {"district": "Ratnagiri", "state": "Maharashtra", "lat": 16.99, "lon": 73.31, "population": 1615069, "port": "Jaigad Port", "surge_risk": "Moderate"},
    {"district": "Sindhudurg", "state": "Maharashtra", "lat": 16.11, "lon": 73.70, "population": 849651, "port": "Redi Port", "surge_risk": "Moderate"},
    {"district": "North Goa", "state": "Goa", "lat": 15.50, "lon": 73.83, "population": 818008, "port": "Mormugao Port", "surge_risk": "Low"},
    # Odisha
    {"district": "Kendrapara", "state": "Odisha", "lat": 20.50, "lon": 86.42, "population": 1440361, "port": "Paradip Anchorage", "surge_risk": "Extreme"},
    {"district": "Jagatsinghpur", "state": "Odisha", "lat": 20.27, "lon": 86.17, "population": 1136971, "port": "Paradip Port", "surge_risk": "Extreme"},
    {"district": "Puri", "state": "Odisha", "lat": 19.81, "lon": 85.83, "population": 1698730, "port": "Astaranga", "surge_risk": "Very High"},
    {"district": "Bhadrak", "state": "Odisha", "lat": 21.05, "lon": 86.51, "population": 1506522, "port": "Dhamra Port", "surge_risk": "Extreme"},
    {"district": "Balasore", "state": "Odisha", "lat": 21.49, "lon": 86.93, "population": 2317419, "port": "Chandipur Coast", "surge_risk": "Very High"},
    {"district": "Ganjam", "state": "Odisha", "lat": 19.38, "lon": 85.05, "population": 3529031, "port": "Gopalpur Port", "surge_risk": "High"},
    # West Bengal
    {"district": "South 24 Parganas", "state": "West Bengal", "lat": 22.16, "lon": 88.43, "population": 8161961, "port": "Sagar Island / Diamond Harbour", "surge_risk": "Extreme"},
    {"district": "North 24 Parganas", "state": "West Bengal", "lat": 22.72, "lon": 88.48, "population": 10009781, "port": "Sundarbans Delta", "surge_risk": "Very High"},
    {"district": "Purba Medinipur", "state": "West Bengal", "lat": 21.93, "lon": 87.77, "population": 5095875, "port": "Haldia / Digha Port", "surge_risk": "Extreme"},
    # Andhra Pradesh & Tamil Nadu
    {"district": "Visakhapatnam", "state": "Andhra Pradesh", "lat": 17.68, "lon": 83.21, "population": 4290589, "port": "Visakhapatnam Port", "surge_risk": "High"},
    {"district": "Krishna", "state": "Andhra Pradesh", "lat": 16.18, "lon": 81.13, "population": 4517398, "port": "Machilipatnam Port", "surge_risk": "Extreme"},
    {"district": "Nellore", "state": "Andhra Pradesh", "lat": 14.44, "lon": 79.98, "population": 2963557, "port": "Krishnapatnam Port", "surge_risk": "High"},
    {"district": "Chennai", "state": "Tamil Nadu", "lat": 13.08, "lon": 80.27, "population": 7088403, "port": "Chennai / Ennore Port", "surge_risk": "High"},
    {"district": "Nagapattinam", "state": "Tamil Nadu", "lat": 10.76, "lon": 79.84, "population": 1616450, "port": "Nagapattinam Port", "surge_risk": "Very High"},
]


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Great Circle Distance in km between two lat/lon coordinates."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def estimate_surge_height(wind_knots: float | None, basin_sector: str) -> tuple[float, str]:
    """Estimate peak coastal storm surge height in meters based on wind and bathymetry."""
    wind = wind_knots if wind_knots is not None else 45.0

    # Base surge proportional to kinetic wind energy
    if wind >= 120.0:
        base_surge = 5.2
    elif wind >= 90.0:
        base_surge = 3.8
    elif wind >= 64.0:
        base_surge = 2.6
    elif wind >= 48.0:
        base_surge = 1.6
    elif wind >= 34.0:
        base_surge = 0.9
    else:
        base_surge = 0.4

    # Shallow funneling amplifies surge: Bay of Bengal head (Odisha/Bengal) is world's highest surge zone
    if "Bengal" in basin_sector or "Odisha" in basin_sector:
        multiplier = 1.35
    elif "Gujarat" in basin_sector or "Kutch" in basin_sector:
        multiplier = 1.15
    else:
        multiplier = 0.95

    final_surge = round(base_surge * multiplier, 1)

    if final_surge >= 3.5:
        level = "CATASTROPHIC INUNDATION (>3.5m)"
    elif final_surge >= 2.0:
        level = "SEVERE INUNDATION (2.0 - 3.5m)"
    elif final_surge >= 1.0:
        level = "MODERATE INUNDATION (1.0 - 2.0m)"
    else:
        level = "LOW / TIDAL SURGE (<1.0m)"

    return final_surge, level


def evaluate_landfall_and_impact(
    forecast_track: list[dict[str, Any]],
    current_lat: float,
    current_lon: float,
    current_wind_knots: float | None,
    issue_time: datetime,
) -> tuple[LandfallForecast, list[dict[str, Any]]]:
    """Detect coastal landfall, compute ETA and surge, and tier affected districts."""
    if not forecast_track:
        default_forecast = LandfallForecast(
            will_make_landfall=False,
            landfall_point=None,
            nearest_landmark="Open sea navigation",
            nearest_port="None in vicinity",
            eta_hours=None,
            eta_timestamp_utc=None,
            eta_timestamp_ist=None,
            wind_at_landfall_knots=None,
            wind_at_landfall_kmph=None,
            gust_kmph=None,
            category_at_landfall="Depression over open sea",
            storm_surge_meters=None,
            surge_warning_level="Normal tide",
            coastline_sector="Deep Ocean",
        )
        return default_forecast, []

    # Check proximity to coastal landmarks along the projected trajectory
    min_dist = float("inf")
    closest_district = COASTAL_DISTRICTS[0]
    closest_forecast_pt: dict[str, Any] | None = None

    for pt in forecast_track:
        p_lat, p_lon = pt["lat"], pt["lon"]
        for dist in COASTAL_DISTRICTS:
            d = haversine_distance_km(p_lat, p_lon, dist["lat"], dist["lon"])
            if d < min_dist:
                min_dist = d
                closest_district = dist
                closest_forecast_pt = pt

    will_make_landfall = min_dist < 160.0  # Cyclone core within 160km of coastline
    wind_landfall_kt = closest_forecast_pt.get("wind_kmph", 80) / 1.852 if closest_forecast_pt and closest_forecast_pt.get("wind_kmph") else (current_wind_knots or 45.0)
    wind_landfall_kmph = round(wind_landfall_kt * 1.852, 0)
    gust_kmph = round(wind_landfall_kmph * 1.15, 0)

    # Classify landfall category
    if wind_landfall_kt >= 120:
        cat = "Super Cyclonic Storm"
    elif wind_landfall_kt >= 90:
        cat = "Extremely Severe Cyclonic Storm"
    elif wind_landfall_kt >= 64:
        cat = "Very Severe Cyclonic Storm"
    elif wind_landfall_kt >= 48:
        cat = "Severe Cyclonic Storm"
    elif wind_landfall_kt >= 34:
        cat = "Cyclonic Storm"
    else:
        cat = "Deep Depression"

    eta_hrs = closest_forecast_pt.get("hours", 48) if closest_forecast_pt else None
    if eta_hrs is not None:
        eta_utc = (issue_time + timedelta(hours=eta_hrs)).isoformat().replace("+00:00", "Z")
        # IST is UTC + 5:30
        eta_ist = (issue_time + timedelta(hours=eta_hrs, minutes=330)).strftime("%d-%b-%Y %H:%M IST")
    else:
        eta_utc = None
        eta_ist = None

    basin_sector = f"{closest_district['state']} Coastal Belt"
    surge_m, surge_lvl = estimate_surge_height(wind_landfall_kt, basin_sector)

    lf_point = (
        {"lat": closest_forecast_pt["lat"], "lon": closest_forecast_pt["lon"]}
        if closest_forecast_pt and will_make_landfall
        else None
    )

    landfall = LandfallForecast(
        will_make_landfall=will_make_landfall,
        landfall_point=lf_point,
        nearest_landmark=f"Near {closest_district['district']} Coastline ({closest_district['state']})",
        nearest_port=closest_district["port"],
        eta_hours=eta_hrs if will_make_landfall else None,
        eta_timestamp_utc=eta_utc if will_make_landfall else None,
        eta_timestamp_ist=eta_ist if will_make_landfall else None,
        wind_at_landfall_knots=round(wind_landfall_kt, 1) if will_make_landfall else None,
        wind_at_landfall_kmph=wind_landfall_kmph if will_make_landfall else None,
        gust_kmph=gust_kmph if will_make_landfall else None,
        category_at_landfall=cat if will_make_landfall else "Non-landfalling / Oceanic track",
        storm_surge_meters=surge_m if will_make_landfall else None,
        surge_warning_level=surge_lvl if will_make_landfall else "No surge threat",
        coastline_sector=basin_sector if will_make_landfall else "Open Basin",
    )

    # Calculate dynamic district vulnerability for all nearby districts
    district_impacts: list[dict[str, Any]] = []
    ref_lat = closest_forecast_pt["lat"] if closest_forecast_pt else current_lat
    ref_lon = closest_forecast_pt["lon"] if closest_forecast_pt else current_lon

    for dist in COASTAL_DISTRICTS:
        dist_km = haversine_distance_km(ref_lat, ref_lon, dist["lat"], dist["lon"])
        if dist_km > 500.0:
            continue

        if dist_km <= 90.0 and will_make_landfall:
            alert = "RED WARNING"
            level = "Severe"
            score = 92
            action = "Evacuate low-lying coastal belts; suspend port and marine operations completely"
        elif dist_km <= 180.0:
            alert = "ORANGE ALERT"
            level = "High"
            score = 76
            action = "Prepare cyclone shelters; restrict coastal movement and moor fishing vessels"
        elif dist_km <= 320.0:
            alert = "YELLOW WATCH"
            level = "Moderate"
            score = 54
            action = "Fishermen advised not to venture into deep sea; standby rescue personnel"
        else:
            alert = "GREEN ADVISORY"
            level = "Low"
            score = 28
            action = "Monitor official IMD bulletins regularly"

        rainfall_threat = "Extremely Heavy (>200 mm)" if dist_km <= 120 else "Heavy to Very Heavy (70-200 mm)" if dist_km <= 250 else "Moderate Scattered (15-60 mm)"
        peak_wind_expected = max(35, round(wind_landfall_kmph * max(0.35, 1.0 - (dist_km / 350.0))))

        district_impacts.append({
            "district": f"{dist['district']}, {dist['state']}",
            "level": level,
            "alert_tier": alert,
            "score": score,
            "distance_to_track_km": round(dist_km, 1),
            "population_exposed": dist["population"],
            "key_port": dist["port"],
            "expected_wind_kmph": peak_wind_expected,
            "rainfall_threat": rainfall_threat,
            "drivers": [
                f"Proximity: {dist_km:.0f} km from projected track corridor",
                f"Peak gust threat: up to {peak_wind_expected} km/h",
                f"Rainfall hazard: {rainfall_threat}",
                f"Directive: {action}",
            ],
        })

    # Sort by impact score descending
    district_impacts.sort(key=lambda item: item["score"], reverse=True)

    return landfall, district_impacts[:8]
