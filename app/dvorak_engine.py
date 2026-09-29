"""Dvorak-scale intensity proxy for historical cyclone replays.

This module derives a Dvorak-scale number from best-track wind and pressure.
It does **not** ingest or analyse INSAT imagery, cloud-top temperatures, or
eyewall structure.  Pattern, cloud and attribution values are explanatory
proxies only; they must never be presented as an official IMD/RSMC assessment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DvorakAssessment:
    assessment_mode: str
    limitations: list[str]
    pattern_type: str
    pattern_description: str
    t_number: float
    ci_number: float
    central_pressure_deficit_hpa: float
    environmental_pressure_hpa: float
    estimated_central_pressure_hpa: float
    eye_characteristics: dict[str, Any]
    cloud_metrics: dict[str, Any]
    pattern_probabilities: dict[str, float]
    attribution: list[dict[str, Any]]


# Dvorak-scale reference table used only to create the best-track proxy.
T_NUMBER_TABLE = [
    (1.0, 25.0, 23.0, 3.0),
    (1.5, 25.0, 25.0, 4.0),
    (2.0, 30.0, 28.0, 6.0),
    (2.5, 35.0, 33.0, 9.0),
    (3.0, 45.0, 42.0, 13.0),
    (3.5, 55.0, 50.0, 18.0),
    (4.0, 65.0, 60.0, 24.0),
    (4.5, 77.0, 70.0, 31.0),
    (5.0, 90.0, 82.0, 39.0),
    (5.5, 102.0, 93.0, 48.0),
    (6.0, 115.0, 105.0, 58.0),
    (6.5, 127.0, 116.0, 69.0),
    (7.0, 140.0, 128.0, 81.0),
    (7.5, 155.0, 142.0, 94.0),
    (8.0, 170.0, 156.0, 108.0),
]


def wind_to_t_number(wind_knots: float | None) -> float:
    """Map wind speed in knots to continuous Dvorak T-number."""
    if wind_knots is None or wind_knots <= 15.0:
        return 1.0
    if wind_knots >= 165.0:
        return 8.0

    # The reference table has two entries at 25 kt. Retain the higher T-number
    # at an identical wind threshold, then interpolate only between strictly
    # increasing winds. This prevents a valid 25-kt best-track observation
    # from dividing by zero and keeps sub-25-kt systems near T1.0.
    points: list[tuple[float, float]] = []
    for t_number, wind, _, _ in T_NUMBER_TABLE:
        if points and wind == points[-1][1]:
            points[-1] = (t_number, wind)
        else:
            points.append((t_number, wind))

    previous_t, previous_wind = 1.0, 15.0
    for t_number, wind in points:
        if wind_knots <= wind:
            fraction = (wind_knots - previous_wind) / (wind - previous_wind)
            return round(previous_t + fraction * (t_number - previous_t), 1)
        previous_t, previous_wind = t_number, wind

    return points[-1][0]


def calculate_pressure_deficit(wind_knots: float | None, latitude: float) -> float:
    """Calculate central pressure drop ΔP (hPa) using Mishra-Gupta NIO equation.

    ΔP = 0.0076 * (V_max)^1.89 for North Indian Ocean
    """
    if wind_knots is None or wind_knots <= 15.0:
        return 2.5
    v = max(wind_knots, 10.0)
    # Mishra-Gupta empirical formula tuned for Arabian Sea and Bay of Bengal
    delta_p = 0.0076 * (v ** 1.89)
    # Latitude adjustment for coriolis effect
    coriolis_factor = 1.0 + 0.008 * max(0.0, abs(latitude) - 12.0)
    return round(delta_p * coriolis_factor, 1)


def classify_pattern(
    wind_knots: float | None,
    pressure_hpa: float | None,
    latitude: float,
    longitude: float,
) -> DvorakAssessment:
    """Create a clearly labelled intensity-derived Dvorak-scale proxy."""
    wind = wind_knots if wind_knots is not None else 35.0
    t_num = wind_to_t_number(wind)
    ci_num = t_num  # CI is equal to or slightly higher than T-number during decay

    delta_p = calculate_pressure_deficit(wind, latitude)
    env_pressure = 1010.0  # Standard tropical mean environmental sea level pressure
    if pressure_hpa is not None and 900.0 <= pressure_hpa <= 1025.0:
        est_central_pressure = pressure_hpa
        delta_p = max(delta_p, round(env_pressure - pressure_hpa, 1))
    else:
        est_central_pressure = round(env_pressure - delta_p, 1)

    # Determine cloud pattern type and probability distribution
    if t_num >= 4.5:
        pattern = "Eye Pattern"
        description = "Well-defined circular or elliptical eye enclosed by dense convective eyewall ring with strong thermal contrast."
        eye_diam_km = round(20.0 + (8.0 - t_num) * 7.5, 1)
        eye_temp_c = round(-22.0 + (t_num - 4.5) * 5.5, 1)
        eyewall_temp_c = round(-72.0 - (t_num - 4.5) * 2.0, 1)
        probs = {
            "Eye Pattern": round(min(0.96, 0.55 + (t_num - 4.5) * 0.12), 2),
            "Central Dense Overcast (CDO)": round(max(0.02, 0.30 - (t_num - 4.5) * 0.08), 2),
            "Embedded Center Pattern": round(max(0.01, 0.12 - (t_num - 4.5) * 0.03), 2),
            "Curved Band Pattern": 0.02,
            "Shear Pattern": 0.01,
        }
    elif t_num >= 3.5:
        pattern = "Central Dense Overcast (CDO)"
        description = "Solid, uniform overcast cloud shield directly enveloping the circulation center without a clear eye."
        eye_diam_km = None
        eye_temp_c = None
        eyewall_temp_c = round(-68.0, 1)
        probs = {
            "Central Dense Overcast (CDO)": 0.62,
            "Embedded Center Pattern": 0.22,
            "Curved Band Pattern": 0.11,
            "Eye Pattern": 0.04,
            "Shear Pattern": 0.01,
        }
    elif t_num >= 2.5:
        pattern = "Curved Band Pattern"
        description = "Convective spiral rainbands wrapping cyclonically around the center, spanning 0.6 to 1.1 log-spiral turns."
        eye_diam_km = None
        eye_temp_c = None
        eyewall_temp_c = round(-58.0, 1)
        probs = {
            "Curved Band Pattern": 0.68,
            "Central Dense Overcast (CDO)": 0.18,
            "Shear Pattern": 0.09,
            "Embedded Center Pattern": 0.04,
            "Eye Pattern": 0.01,
        }
    else:
        pattern = "Shear Pattern"
        description = "Low-level circulation center exposed or partially displaced from deep convection by vertical wind shear."
        eye_diam_km = None
        eye_temp_c = None
        eyewall_temp_c = round(-48.0, 1)
        probs = {
            "Shear Pattern": 0.65,
            "Curved Band Pattern": 0.24,
            "Central Dense Overcast (CDO)": 0.08,
            "Embedded Center Pattern": 0.02,
            "Eye Pattern": 0.01,
        }

    # Normalize probabilities to sum to 1.0
    prob_total = sum(probs.values())
    normalized_probs = {k: round(v / prob_total, 3) for k, v in probs.items()}

    # Log-spiral wrap calculation
    spiral_wrap_turns = min(1.4, max(0.2, (t_num - 1.0) * 0.22))
    cloud_shield_diameter_km = round(220.0 + t_num * 55.0, 0)
    convective_symmetry = min(98.0, max(45.0, 35.0 + t_num * 8.5))

    eye_chars = {
        "present": pattern == "Eye Pattern",
        "diameter_km": eye_diam_km,
        "temperature_celsius": eye_temp_c,
        "eyewall_temperature_celsius": eyewall_temp_c,
        "eye_definition": "Distinct circular" if t_num >= 5.5 else "Ragged / cloud-filled" if pattern == "Eye Pattern" else "None",
    }

    cloud_metrics = {
        "cloud_shield_diameter_km": cloud_shield_diameter_km,
        "convective_symmetry_percent": round(convective_symmetry, 1),
        "log_spiral_wrap_turns": round(spiral_wrap_turns, 2),
        "min_cloud_top_temperature_celsius": eyewall_temp_c,
        "cold_cover_area_1000sqkm": round((cloud_shield_diameter_km / 2.0) ** 2 * math.pi / 1000.0, 1),
    }

    attribution = [
        {
            "feature": "Eyewall & Core Convection",
            "weight": round(0.35 + 0.05 * t_num, 2),
            "detail": f"Cold cloud top brightness temperature reached {eyewall_temp_c}°C in core ring",
        },
        {
            "feature": "Spiral Rainband Coherence",
            "weight": round(0.25, 2),
            "detail": f"Curved band structure completes {spiral_wrap_turns:.2f} log-spiral turns around vortex",
        },
        {
            "feature": "Core Convective Symmetry",
            "weight": round(0.20, 2),
            "detail": f"Circulation azimuthal uniformity measured at {convective_symmetry:.0f}%",
        },
        {
            "feature": "Thermal Eye/Surround Contrast",
            "weight": round(0.15 + (0.05 if pattern == 'Eye Pattern' else 0.0), 2),
            "detail": f"Gradient between center and convective ring is {abs(eyewall_temp_c - (eye_temp_c or -50.0)):.1f}°C",
        },
    ]

    return DvorakAssessment(
        assessment_mode="best_track_intensity_proxy",
        limitations=[
            "No satellite image pixels or cloud-top temperatures are used.",
            "Pattern labels and visual metrics are deterministic explanatory proxies, not observations.",
            "Not an official Dvorak analysis or operational intensity estimate.",
        ],
        pattern_type=pattern,
        pattern_description=description,
        t_number=t_num,
        ci_number=ci_num,
        central_pressure_deficit_hpa=delta_p,
        environmental_pressure_hpa=env_pressure,
        estimated_central_pressure_hpa=est_central_pressure,
        eye_characteristics=eye_chars,
        cloud_metrics=cloud_metrics,
        pattern_probabilities=normalized_probs,
        attribution=attribution,
    )
