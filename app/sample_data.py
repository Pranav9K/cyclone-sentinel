"""Fixture data for the first dashboard vertical slice.

Replace this module with a repository/query layer once official data ingestion is enabled.
All values in this file are illustrative only.
"""

STORM = {
    "id": "biparjoy-2023",
    "name": "Cyclone Biparjoy",
    "season": "2023",
    "basin": "North Indian Ocean · Arabian Sea",
    "data_mode": "Demo replay — not an official forecast",
    "status": "Very Severe Cyclonic Storm",
    "last_updated": "2023-06-14T12:00:00Z",
    "current": {"lat": 20.4, "lon": 66.2, "wind_kmph": 135, "pressure_hpa": 958, "rainfall_mm_hr": 9.6},
    "classification": {"label": "Very Severe Cyclonic Storm", "confidence": 0.91},
    "model_metrics": {"track_error_km": 84, "intensity_mae_kmph": 12.4, "classification_f1": 0.88},
    "observed_track": [
        {"time": "2023-06-10T00:00:00Z", "lat": 12.8, "lon": 66.6},
        {"time": "2023-06-11T00:00:00Z", "lat": 14.1, "lon": 66.1},
        {"time": "2023-06-12T00:00:00Z", "lat": 15.5, "lon": 65.9},
        {"time": "2023-06-13T00:00:00Z", "lat": 17.8, "lon": 65.8},
        {"time": "2023-06-14T12:00:00Z", "lat": 20.4, "lon": 66.2},
    ],
    "forecast_track": [
        {"hours": 24, "time": "2023-06-15T12:00:00Z", "lat": 22.1, "lon": 67.3, "wind_kmph": 120, "radius_km": 95},
        {"hours": 48, "time": "2023-06-16T12:00:00Z", "lat": 23.5, "lon": 68.8, "wind_kmph": 92, "radius_km": 145},
        {"hours": 72, "time": "2023-06-17T12:00:00Z", "lat": 24.0, "lon": 70.2, "wind_kmph": 60, "radius_km": 210},
    ],
    "district_risk": [
        {"district": "Kachchh, Gujarat", "level": "Critical", "score": 91, "drivers": ["High wind exposure", "Likely landfall corridor", "Coastal inundation"]},
        {"district": "Devbhumi Dwarka, Gujarat", "level": "High", "score": 78, "drivers": ["Damaging winds", "Heavy rainfall", "Coastal inundation"]},
        {"district": "Jamnagar, Gujarat", "level": "High", "score": 71, "drivers": ["Gale-force winds", "Heavy rainfall"]},
        {"district": "Porbandar, Gujarat", "level": "Moderate", "score": 54, "drivers": ["Heavy rainfall", "Rough sea conditions"]},
    ],
}


def prediction_payload() -> dict:
    return {
        "storm_id": STORM["id"],
        "generated_at": STORM["last_updated"],
        "track": STORM["forecast_track"],
        "intensity": {"current_wind_kmph": 135, "peak_next_24h_kmph": 140, "trend": "weakening after 24 hours"},
        "uncertainty": {"24h_km": 95, "48h_km": 145, "72h_km": 210},
        "disclaimer": "Illustrative demo output only. Always follow official IMD warnings.",
    }

