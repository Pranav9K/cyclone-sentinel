"""Plain-text research-replay brief generator.

The output is intentionally distinct from IMD/RSMC products and must not be
used as an operational weather, marine, or disaster-management advisory.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


def generate_research_brief(storm_data: dict[str, Any]) -> str:
    """Format a transparent, non-operational summary of a historical replay."""
    name = storm_data.get("name", "UNNAMED").upper().replace("CYCLONE ", "")
    category = storm_data.get("status", "CYCLONIC STORM").upper()
    issue_utc = storm_data.get("last_updated", "")
    try:
        issue_time = datetime.fromisoformat(issue_utc.replace("Z", "+00:00"))
    except ValueError:
        issue_time = datetime.now()
    issue_ist = issue_time + timedelta(minutes=330)

    current = storm_data.get("current", {})
    wind_kmph = current.get("wind_kmph")
    wind_kt = round(wind_kmph / 1.852) if wind_kmph else "unavailable"
    dvorak = storm_data.get("dvorak", {})
    ri = storm_data.get("rapid_intensification", {})
    landfall = storm_data.get("landfall", {})

    lines = [
        "=" * 78,
        "CYCLONE SENTINEL — HISTORICAL RESEARCH REPLAY BRIEF",
        "NOT AN IMD/RSMC ADVISORY. NOT FOR OPERATIONAL OR PUBLIC-SAFETY DECISIONS.",
        "=" * 78,
        f"Replay issue point: {issue_utc} ({issue_ist.strftime('%d-%b-%Y %H:%M IST')})",
        f"System: {category} {name} | basin: North Indian Ocean",
        "",
        "1. OBSERVED BEST-TRACK STATE",
        f"   Position: {current.get('lat', 0.0):.2f}°N, {current.get('lon', 0.0):.2f}°E",
        f"   Wind: {wind_kt} kt ({wind_kmph if wind_kmph is not None else 'unavailable'} km/h)",
        f"   Pressure: {current.get('pressure_hpa') or 'unavailable'} hPa",
        "   Source: NOAA IBTrACS historical best-track subset.",
        "",
        "2. INTENSITY-DERIVED DVORAK-SCALE PROXY",
        f"   T-number / CI: T{dvorak.get('t_number', '—')} / CI{dvorak.get('ci_number', '—')}",
        f"   Proxy label: {dvorak.get('pattern_type', 'unavailable')}",
        "   Limitation: This is derived from wind/pressure; it does not analyse INSAT imagery.",
        "",
        "3. RAPID-INTENSIFICATION SCREENING HEURISTIC",
        f"   Screening level: {ri.get('status_label', 'unavailable')}",
        f"   Heuristic score: {ri.get('ri_score', 0) * 100:.1f}/100 (NOT a probability)",
        f"   Note: {ri.get('summary', 'No summary available')}",
        "",
        "4. BASELINE TRACK REPLAY",
        "   A ridge-regression research baseline projects 24/48/72-hour replay points.",
        "   Its envelope is scaled from held-out error; it is not a calibrated probability cone.",
    ]
    for point in storm_data.get("forecast_track", []):
        wind = point.get("wind_kmph")
        lines.append(
            f"   +{point.get('hours', '—')}h: {point.get('lat', 0.0):.2f}°N, "
            f"{point.get('lon', 0.0):.2f}°E | wind {wind if wind is not None else '—'} km/h | "
            f"error-scaled radius ±{point.get('radius_km', '—')} km"
        )

    lines.extend([
        "",
        "5. COASTAL-PROXIMITY SCREEN",
        f"   Nearest static district reference: {landfall.get('nearest_landmark', 'unavailable')}",
        "   This is not a landfall determination. Surge and district figures are research proxies only.",
        "",
        "For current weather and public-safety decisions, consult IMD/RSMC and local authorities.",
        "=" * 78,
    ])
    return "\n".join(lines)


# Compatibility alias for callers not yet migrated. It deliberately produces
# the same non-operational research brief.
generate_imd_bulletin = generate_research_brief
