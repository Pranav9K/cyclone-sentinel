"""Poll MOSDAC's public INSAT-3D Imager RSS feed for fresh product metadata.

The poller records provider-published metadata only. It does not impersonate a
MOSDAC user, fetch protected imagery, or claim an image was processed before a
local download and decoder are configured.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import xml.etree.ElementTree as element_tree
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://mosdac.gov.in/3dimager.xml"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "live_insat_manifest.json"
USER_AGENT = "CycloneSentinel-LiveINSAT/0.1 (+https://github.com/)"


def utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def local_path(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def https_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("The live feed URL must be an absolute HTTPS URL.")
    return value


def child_text(item: element_tree.Element, name: str) -> str | None:
    for child in item:
        if child.tag.rsplit("}", 1)[-1] == name and child.text:
            value = child.text.strip()
            if value:
                return value
    return None


def parse_published(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return utc_text(parsed)


def parse_feed(payload: bytes) -> list[dict[str, str | None]]:
    try:
        root = element_tree.fromstring(payload)
    except element_tree.ParseError as error:
        raise ValueError(f"MOSDAC feed XML could not be parsed: {error}") from error
    items: list[dict[str, str | None]] = []
    for item in root.iter():
        if item.tag.rsplit("}", 1)[-1] != "item":
            continue
        title = child_text(item, "title")
        link = child_text(item, "link")
        if not title or not link:
            continue
        items.append({
            "title": title,
            "link": link,
            "published_at": parse_published(child_text(item, "pubDate")),
            "guid": child_text(item, "guid"),
        })
    if not items:
        raise ValueError("MOSDAC feed did not contain any usable RSS items.")
    return items


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f"{path.name}.", dir=path.parent)
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll the public MOSDAC INSAT-3D Imager RSS feed.")
    parser.add_argument("--feed-url", default=DEFAULT_URL, help="HTTPS RSS feed URL.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Generated live-product manifest path.")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="HTTPS timeout (default: 30).")
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments without contacting MOSDAC or writing a manifest.")
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive.")
    feed_url = https_url(args.feed_url)
    manifest_path = local_path(args.manifest)
    if args.dry_run:
        print(json.dumps({"feed_url": feed_url, "manifest": str(manifest_path.relative_to(PROJECT_ROOT)).replace("\\", "/"), "network": False}, indent=2))
        return 0

    generated_at = utc_text(datetime.now(UTC))
    try:
        request = Request(feed_url, headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/xml, text/xml"})
        with urlopen(request, timeout=args.timeout_seconds) as response:
            payload = response.read(4 * 1024 * 1024 + 1)
        if len(payload) > 4 * 1024 * 1024:
            raise ValueError("MOSDAC feed exceeded the 4 MiB safety limit.")
        items = parse_feed(payload)
        manifest = {
            "schema_version": 1,
            "status": "ready",
            "generated_at_utc": generated_at,
            "source": {"provider": "MOSDAC / ISRO", "feed_url": feed_url, "product": "INSAT-3D Imager RSS metadata"},
            "items": items[:20],
            "latest": items[0],
            "note": "This manifest confirms published metadata, not a locally downloaded or decoded image.",
        }
    except (OSError, URLError, ValueError) as error:
        manifest = {
            "schema_version": 1,
            "status": "unavailable",
            "generated_at_utc": generated_at,
            "source": {"provider": "MOSDAC / ISRO", "feed_url": feed_url, "product": "INSAT-3D Imager RSS metadata"},
            "error": str(error),
            "note": "No satellite product metadata was accepted during this poll.",
        }
        write_manifest(manifest_path, manifest)
        print(f"ERROR: {error}")
        return 1
    write_manifest(manifest_path, manifest)
    print(f"Live INSAT metadata ready: {len(items)} RSS items; latest: {items[0]['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
