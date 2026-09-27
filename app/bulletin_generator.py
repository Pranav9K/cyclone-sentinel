"""Official IMD / RSMC Tropical Cyclone Advisory Bulletin Generator.

Generates standard-compliant tropical cyclone bulletins modeled directly on
the India Meteorological Department (IMD) / Regional Specialised Meteorological
Centre (RSMC) New Delhi format.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


def generate_imd_bulletin(storm_data: dict[str, Any]) -> str:
    """Format an authentic IMD RSMC Tropical Cyclone Advisory Bulletin."""
    name = storm_data.get("name", "UNNAMED").upper().replace("CYCLONE ", "")
    season = storm_data.get("season", "2023")
    category = storm_data.get("status", "CYCLONIC STORM").upper()
    issue_utc_str = storm_data.get("last_updated", "")

    try:
        issue_time = datetime.fromisoformat(issue_utc_str.replace("Z", "+00:00"))
    except ValueError:
        issue_time = datetime.now()

    issue_ist = issue_time + timedelta(minutes=330)
    utc_formatted = issue_time.strftime("%d%H%M UTC")
    ist_formatted = issue_ist.strftime("%d-%b-%Y AT %H:%M IST")

    current = storm_data.get("current", {})
    lat = current.get("lat", 0.0)
    lon = current.get("lon", 0.0)
    wind_kmph = current.get("wind_kmph", 0)
    wind_kt = round(wind_kmph / 1.852) if wind_kmph else 35
    pressure_hpa = current.get("pressure_hpa", 992) or 992

    dvorak = storm_data.get("dvorak", {})
    t_num = dvorak.get("t_number", 3.5)
    ci_num = dvorak.get("ci_number", 3.5)
    pattern = dvorak.get("pattern_type", "Curved Band Pattern")
    delta_p = dvorak.get("central_pressure_deficit_hpa", 18.0)

    ri = storm_data.get("rapid_intensification", {})
    ri_alert = ri.get("status_label", "MODERATE RI RISK")
    ri_prob = ri.get("ri_probability", 0.35)

    landfall = storm_data.get("landfall", {})
    will_landfall = landfall.get("will_make_landfall", False)
    lf_landmark = landfall.get("nearest_landmark", "Open sea")
    lf_eta_ist = landfall.get("eta_timestamp_ist", "N/A")
    lf_surge = landfall.get("storm_surge_meters", 1.5)
    lf_surge_level = landfall.get("surge_warning_level", "Moderate")

    forecast = storm_data.get("forecast_track", [])

    lines = [
        "=" * 78,
        "INDIA METEOROLOGICAL DEPARTMENT",
        "REGIONAL SPECIALISED METEOROLOGICAL CENTRE - TROPICAL CYCLONES, NEW DELHI",
        "TROPICAL CYCLONE ADVISORY BULLETIN NO. 14 (RESEARCH REPLAY EDITION)",
        "=" * 78,
        f"FROM: RSMC - TROPICAL CYCLONES, NEW DELHI",
        f"TO: STORM WARNING CENTRES / DISASTER MANAGEMENT AUTHORITIES",
        f"TIME OF ISSUE: {utc_formatted} ({ist_formatted})",
        "",
        f"SUBJECT: {category} '{name}' OVER NORTH INDIAN OCEAN",
        "-" * 78,
        "",
        "1. CURRENT SYSTEM STATUS AND SYNOPTIC INTERPRETATION:",
        f"   THE {category} '{name}' OVER NORTH INDIAN OCEAN LAY CENTERED AT {issue_utc_str}",
        f"   NEAR LATITUDE {lat:.2f}°N AND LONGITUDE {lon:.2f}°E.",
        f"   ESTIMATED CENTRAL PRESSURE: {pressure_hpa:.1f} hPa.",
        f"   ESTIMATED CENTRAL PRESSURE DEFICIT (ΔP): {delta_p:.1f} hPa.",
        f"   MAXIMUM SUSTAINED SURFACE WIND SPEED: {wind_kt} KNOTS ({wind_kmph} KM/H) GUSTING TO {round(wind_kmph * 1.15)} KM/H.",
        f"   STATE OF SEA: PHENOMENAL TO VERY HIGH IN CORE CIRCULATION.",
        "",
        "2. SATELLITE DERIVED INTENSITY AND CLOUD PATTERN (INSAT-3D / DVORAK ANALYSIS):",
        f"   • DVORAK CLOUD PATTERN TYPE: {pattern.upper()}",
        f"   • ESTIMATED T-NUMBER: T{t_num:.1f} | CURRENT INTENSITY (CI) NO.: CI{ci_num:.1f}",
        f"   • CONVECTIVE SYMMETRY: {dvorak.get('cloud_metrics', {}).get('convective_symmetry_percent', 75)}%",
        f"   • LOG-SPIRAL RAINBAND WRAP: {dvorak.get('cloud_metrics', {}).get('log_spiral_wrap_turns', 0.8)} TURNS",
        f"   • MINIMUM CLOUD TOP BRIGHTNESS TEMP: {dvorak.get('cloud_metrics', {}).get('min_cloud_top_temperature_celsius', -65)}°C",
        "",
        "3. RAPID INTENSIFICATION (RI) DIAGNOSTIC ASSESSMENT:",
        f"   • STATUS: {ri_alert} (CALIBRATED PROBABILITY: {ri_prob * 100:.1f}%)",
        f"   • THRESHOLD: ≥ 30 KNOTS INTENSIFICATION WITHIN NEXT 24 HOURS",
        f"   • SYNOPSIS: {ri.get('summary', 'Favorable thermodynamic conditions persist.')}",
        "",
        "4. TRACK AND INTENSITY FORECAST TABLE:",
        f"   {'-' * 74}",
        f"   {'HORIZON':<10} {'VALID (UTC)':<22} {'LAT (°N)':<10} {'LON (°E)':<10} {'WIND (KT)':<11} {'UNCERTAINTY'}",
        f"   {'-' * 74}",
    ]

    for pt in forecast:
        hrs = f"+{pt.get('hours', 24)}H"
        valid = pt.get("time", "")
        f_lat = f"{pt.get('lat', 0.0):.2f}"
        f_lon = f"{pt.get('lon', 0.0):.2f}"
        f_wind_kt = round(pt.get("wind_kmph", 0) / 1.852) if pt.get("wind_kmph") else "—"
        uncert = f"±{pt.get('radius_km', 150)} KM"
        lines.append(f"   {hrs:<10} {valid:<22} {f_lat:<10} {f_lon:<10} {f_wind_kt:<11} {uncert}")

    lines.extend([
        f"   {'-' * 74}",
        "",
        "5. COASTAL LANDFALL AND STORM SURGE OUTLOOK:",
    ])

    if will_landfall:
        lines.extend([
            f"   • EXPECTED LANDFALL ZONE: {lf_landmark.upper()}",
            f"   • ESTIMATED TIME OF LANDFALL: {lf_eta_ist} (± 3 HOURS WINDOW)",
            f"   • ESTIMATED CATEGORY AT LANDFALL: {landfall.get('category_at_landfall', 'CYCLONIC STORM').upper()}",
            f"   • LANDFALL WIND: {landfall.get('wind_at_landfall_kmph', 120)} KM/H GUSTING TO {landfall.get('gust_kmph', 140)} KM/H",
            f"   • PROJECTED STORM SURGE: {lf_surge} METERS ({lf_surge_level.upper()}) ABOVE ASTRONOMICAL TIDE",
            f"   • SECTOR AT MAXIMUM SURGE RISK: {landfall.get('coastline_sector', 'COASTAL BELT').upper()}",
        ])
    else:
        lines.append("   • SYSTEM PROJECTED TO RECURVE / REMAIN OVER OPEN OCEAN WATERS OVER THE NEXT 72 HOURS.")

    lines.extend([
        "",
        "6. ACTION SUGGESTED FOR DISASTER MANAGEMENT AUTHORITIES (NDMA / SDMA / NDRF):",
        "   (A) TOTAL SUSPENSION OF FISHING OPERATIONS OVER RELEVANT SEA SECTORS.",
        "   (B) HOISTING OF LOCAL CAUTIONARY SIGNAL (LC-III / DANGER SIGNAL) AT ADJACENT PORTS.",
        "   (C) PRE-POSITIONING OF NDRF / SDRF TEAMS ALONG VULNERABLE LOW-LYING INUNDATION ZONES.",
        "   (D) REGULAR SCRUTINY OF RADAR (DWR), MOSDAC SATELLITE (INSAT-3D/3DR), AND IMD LIVE ADVISORIES.",
        "",
        "=" * 78,
        "END OF BULLETIN - REGIONAL SPECIALISED METEOROLOGICAL CENTRE, NEW DELHI",
        "=" * 78,
    ])

    return "\n".join(lines)
