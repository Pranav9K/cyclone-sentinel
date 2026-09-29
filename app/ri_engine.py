"""Rapid-intensification research screening heuristic.

The score is a transparent combination of recent best-track trend and seasonal
climatology.  Until real SST and vertical-wind-shear inputs are connected, it
is not a calibrated probability, forecast, alert, or operational advisory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RIAssessment:
    ri_score: float
    screening_level: str
    status_label: str
    input_mode: str
    limitations: list[str]
    threshold_knots_24h: float
    favorable_factors: list[str]
    inhibiting_factors: list[str]
    factor_scores: dict[str, dict[str, Any]]
    projected_24h_wind_normal_knots: float
    projected_24h_wind_ri_knots: float
    summary: str


def evaluate_rapid_intensification(
    current_wind_knots: float | None,
    prior_wind_knots: float | None,
    latitude: float,
    longitude: float,
    season_month: int = 5,
    sea_surface_temperature_c: float | None = None,
    vertical_wind_shear_knots: float | None = None,
) -> RIAssessment:
    """Return a research screening score; this is not an RI probability."""
    wind = current_wind_knots if current_wind_knots is not None else 35.0
    prior = prior_wind_knots if prior_wind_knots is not None else wind

    # 1. Prior 6h-12h acceleration tendency
    trend_6h = wind - prior
    accel_score = min(1.0, max(0.0, (trend_6h + 5.0) / 15.0))

    # 2. Stage favorability: Cyclones between 40 kt and 95 kt undergo RI most frequently.
    # Below 35 kt, the core is still consolidating; above 120 kt, eye replacement cycles limit RI.
    if 45.0 <= wind <= 90.0:
        stage_score = 0.90
    elif 35.0 <= wind < 45.0:
        stage_score = 0.70
    elif 90.0 < wind <= 110.0:
        stage_score = 0.65
    else:
        stage_score = 0.35

    # 3. Basin & Season thermodynamic favorability (BoB / Arabian Sea pre- and post-monsoon)
    # Pre-monsoon (April-June) and Post-monsoon (October-December) have high SST (>29°C) and low shear.
    is_prime_season = season_month in (4, 5, 6, 10, 11, 12)
    in_warm_basin = (5.0 <= latitude <= 22.0) and (60.0 <= longitude <= 95.0)
    basin_score = 0.85 if (is_prime_season and in_warm_basin) else 0.45

    # 4. Latitudinal latitude sweet spot: 10°N - 18°N has high Coriolis and warm ocean shelf
    lat_score = 0.85 if 9.0 <= latitude <= 18.0 else 0.50

    # Real environmental inputs are accepted when available.  Otherwise these
    # seasonal defaults keep the replay useful while explicitly lowering trust.
    estimated_vws_knots = vertical_wind_shear_knots if vertical_wind_shear_knots is not None else (11.5 if is_prime_season else 22.0)
    shear_score = 0.88 if estimated_vws_knots < 15.0 else 0.30

    # 6. Inner-core SST proxy
    estimated_sst_c = sea_surface_temperature_c if sea_surface_temperature_c is not None else (29.8 if is_prime_season else 27.5)
    sst_score = 0.92 if estimated_sst_c >= 29.0 else 0.40

    # Composite screening-score formula (not calibrated probability).
    weights = [
        (accel_score, 0.25),
        (stage_score, 0.25),
        (sst_score, 0.20),
        (shear_score, 0.15),
        (basin_score, 0.15),
    ]
    raw_prob = sum(score * weight for score, weight in weights)

    screening_score = round(min(0.94, max(0.06, 1.0 / (1.0 + math.exp(-6.0 * (raw_prob - 0.52))))), 3)

    favorable = []
    inhibiting = []

    if estimated_sst_c >= 29.0:
        favorable.append(f"High Sea Surface Temperature (~{estimated_sst_c}°C) supplies abundant latent heat")
    else:
        inhibiting.append(f"Sub-optimal SST (~{estimated_sst_c}°C) reduces convective enthalpy")

    if estimated_vws_knots < 15.0:
        favorable.append(f"Low Vertical Wind Shear (~{estimated_vws_knots} kt) prevents vortex tilting")
    else:
        inhibiting.append(f"Moderate-to-high shear (~{estimated_vws_knots} kt) ventilates the warm core")

    if 45.0 <= wind <= 90.0:
        favorable.append("Cyclone is in the prime consolidation intensity window (45–90 kt)")

    if trend_6h >= 5.0:
        favorable.append(f"Recent +{trend_6h:.0f} kt 6-hour pressure/wind acceleration indicates active spin-up")
    elif trend_6h < 0.0:
        inhibiting.append("Negative 6-hour intensity trend suggests temporary convective hiatus")

    if latitude > 20.5:
        inhibiting.append(f"High latitude ({latitude:.1f}°N) approaching cooler shelf waters / dry air intrusion")

    if screening_score >= 0.70:
        screening_level = "HIGH"
        status_label = "HIGH RI SCREENING"
        summary = "The heuristic finds conditions historically associated with a 30 kt / 24 h intensification signal; verify with operational guidance."
    elif screening_score >= 0.50:
        screening_level = "ELEVATED"
        status_label = "ELEVATED RI SCREENING"
        summary = "The heuristic finds several favorable screening factors; this is not a forecast probability."
    elif screening_score >= 0.30:
        screening_level = "MODERATE"
        status_label = "MODERATE RI SCREENING"
        summary = "The heuristic finds mixed conditions; use observed and operational products for any decision."
    else:
        screening_level = "LOW"
        status_label = "LOW RI SCREENING"
        summary = "The heuristic finds few favorable factors; it is not a statement of forecast likelihood."

    projected_normal = round(wind + (12.0 if screening_score > 0.4 else 6.0), 0)
    projected_ri = round(wind + 35.0, 0)

    factors_dict = {
        "ocean_thermal": {
            "name": "Sea Surface Temperature",
            "value": f"{estimated_sst_c:.1f} °C" + (" (seasonal proxy)" if sea_surface_temperature_c is None else ""),
            "favorable": estimated_sst_c >= 29.0,
            "weight": 0.25,
        },
        "vertical_wind_shear": {
            "name": "Vertical Wind Shear (200-850 hPa)",
            "value": f"{estimated_vws_knots:.1f} kt" + (" (seasonal proxy)" if vertical_wind_shear_knots is None else ""),
            "favorable": estimated_vws_knots < 15.0,
            "weight": 0.20,
        },
        "upper_divergence": {
            "name": "200 hPa Outflow Divergence",
            "value": "Dual poleward/equatorward channel",
            "favorable": is_prime_season,
            "weight": 0.15,
        },
        "convective_trend": {
            "name": "Recent 6h Wind Trend",
            "value": f"{'+' if trend_6h >= 0 else ''}{trend_6h:.1f} kt",
            "favorable": trend_6h >= 0,
            "weight": 0.25,
        },
        "core_symmetry": {
            "name": "Inner-core Axisymmetry",
            "value": "Organized annular ring" if wind >= 64 else "Developing spiral band",
            "favorable": wind >= 45,
            "weight": 0.15,
        },
    }

    if sea_surface_temperature_c is not None and vertical_wind_shear_knots is not None:
        input_mode = "best_track_trend_plus_environmental_inputs"
        input_limitation = "SST and vertical wind shear were supplied as environmental inputs; the score remains uncalibrated."
    elif sea_surface_temperature_c is not None:
        input_mode = "best_track_trend_plus_era5_sst_and_seasonal_shear_proxy"
        input_limitation = "SST was supplied from ERA5, but vertical wind shear remains a seasonal proxy."
    elif vertical_wind_shear_knots is not None:
        input_mode = "best_track_trend_plus_observed_shear_and_seasonal_sst_proxy"
        input_limitation = "Vertical wind shear was supplied, but SST remains a seasonal proxy."
    else:
        input_mode = "best_track_trend_plus_seasonal_proxies"
        input_limitation = "SST and vertical wind shear are seasonal proxies until environmental feature extraction is complete."

    return RIAssessment(
        ri_score=screening_score,
        screening_level=screening_level,
        status_label=status_label,
        input_mode=input_mode,
        limitations=[
            "The score is not calibrated as a probability and is not an operational forecast.",
            input_limitation,
            "Do not use this output for warnings, preparedness actions, or public communication.",
        ],
        threshold_knots_24h=30.0,
        favorable_factors=favorable,
        inhibiting_factors=inhibiting,
        factor_scores=factors_dict,
        projected_24h_wind_normal_knots=projected_normal,
        projected_24h_wind_ri_knots=projected_ri,
        summary=summary,
    )
