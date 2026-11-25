"""Find scenes for an area and period, from whichever configured provider is usable."""
from __future__ import annotations

import datetime as dt
import logging
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ceva.catalog.models import Scene
from ceva.catalog.providers import SPECS, TOA_SPECS, CollectionSpec, SceneRejected, baseline_tuple, normalise_item
from ceva.config import Config, SensorConfig
from ceva.credentials import Credentials, load_credentials

log = logging.getLogger(__name__)

PAGE_SIZE = 100
SEARCH_RETRIES = 8   # with back-off 2, 4, 8 ... s: about 8 minutes before giving up


@dataclass
class SearchResult:
    provider: str
    scenes: list[Scene]
    n_items: int                                   # items returned by the API before filtering
    rejected: Counter[str] = field(default_factory=Counter)        # reason -> count
    unusable_providers: dict[str, str] = field(default_factory=dict)   # provider -> why


def _specs(level: str) -> dict[tuple[str, str], CollectionSpec]:
    return TOA_SPECS if level == "toa" else SPECS


def provider_status(cfg: Config, family: str, creds: Credentials, level: str = "surface") -> dict[str, str | None]:
    """Configured providers for a sensor family and processing level, in priority order.

    The value is ``None`` when the provider can be used, otherwise the reason it cannot.
    A provider counts as usable only if its pixels can be read, not just searched.
    """
    sensor = cfg.sensors[family]
    status: dict[str, str | None] = {}
    for name in (sensor.toa_providers if level == "toa" else sensor.providers):
        if (name, family) not in _specs(level):
            status[name] = f"no {level} collection mapping for this sensor family"
        elif name == "cdse" and not creds.cdse_s3_ready:
            missing = [k for k in ("CDSE_S3_ACCESS_KEY", "CDSE_S3_SECRET_KEY") if not creds.get(k)]
            status[name] = f"missing {' and '.join(missing)}"
        elif name == "usgs":
            from ceva.catalog.usgs import download_status

            status[name] = download_status(cfg, creds)
        else:
            status[name] = None
    return status


def filter_and_normalise(
    items: Iterable[dict[str, Any]], spec: CollectionSpec, sensor: SensorConfig, level: str = "surface"
) -> tuple[list[Scene], Counter[str], int]:
    """Apply the sensor rules to raw STAC items. Returns scenes, rejection counts, item count.

    Filtering happens here rather than in the STAC query so that every provider
    is treated identically, whatever query extensions it supports.
    """
    scenes: list[Scene] = []
    rejected: Counter[str] = Counter()
    n_items = 0
    for item in items:
        n_items += 1
        props = item.get("properties") or {}
        if sensor.collection_categories is not None:
            category = props.get("landsat:collection_category")
            if category not in sensor.collection_categories:
                rejected[f"collection_category_{category}"] += 1
                continue
        try:
            scene = normalise_item(item, spec, level)
        except SceneRejected as exc:
            rejected[exc.reason] += 1
            continue
        if scene.platform not in sensor.platforms:
            rejected[f"platform_{scene.platform or 'unknown'}"] += 1
        elif (level == "surface" and scene.cloud_cover is not None
              and scene.cloud_cover > sensor.max_scene_cloud_cover):
            # Level-1 scenes are only looked up as twins of Level-2 ones, which were filtered already
            # (the two products estimate the cloud cover differently)
            rejected["cloud_cover_above_limit"] += 1
        else:
            scenes.append(scene)
    return scenes, rejected, n_items


def _acquisition_key(scene: Scene) -> tuple:
    """Same key for every processing of one Sentinel-2 acquisition over one MGRS tile."""
    stamp = re.search(r"\d{8}T\d{6}", scene.id)
    when = stamp.group(0) if stamp else scene.datetime.replace(microsecond=0).isoformat()
    return (scene.family, scene.platform, when, scene.grid_id)


def match_key(scene: Scene) -> tuple:
    """Same key for the Level-1 and Level-2 products of one acquisition over one tile or scene.

    Sentinel-2: platform, sensing time and MGRS tile. Landsat: sensor, path/row and date
    (the product ids differ only in the processing level and, sometimes, date).
    """
    if scene.family == "sentinel2":
        return _acquisition_key(scene)
    parts = scene.id.split("_")
    return (scene.family, parts[0], scene.grid_id, parts[3] if len(parts) > 3 else scene.datetime.date().isoformat())


def deduplicate(scenes: Iterable[Scene]) -> list[Scene]:
    """Keep one product per acquisition: the newest processing baseline, then the latest id.

    Some archives hold the same Sentinel-2 acquisition under several baselines.
    """
    best: dict[tuple, Scene] = {}
    for scene in scenes:
        key = _acquisition_key(scene) if scene.family == "sentinel2" else (scene.family, scene.id)
        current = best.get(key)
        rank = (baseline_tuple(scene.processing_baseline), scene.id)
        if current is None or rank > (baseline_tuple(current.processing_baseline), current.id):
            best[key] = scene
    return sorted(best.values(), key=lambda s: (s.datetime, s.id))


def _fetch_items(stac_url: str, collection: str, bbox: tuple[float, float, float, float],
                 start: dt.date, end: dt.date) -> Iterator[dict[str, Any]]:
    from pystac_client import Client
    from pystac_client.stac_api_io import StacApiIO
    from urllib3.util.retry import Retry

    # catalogues answer "429 too many requests" when many searches come at once: wait and retry
    retry = Retry(total=SEARCH_RETRIES, backoff_factor=2.0, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=None, respect_retry_after_header=True)
    search = Client.open(stac_url, stac_io=StacApiIO(max_retries=retry)).search(
        collections=[collection],
        bbox=list(bbox),
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        limit=PAGE_SIZE,
    )
    yield from search.items_as_dicts()


def search_scenes(
    cfg: Config,
    family: str,
    bbox: tuple[float, float, float, float],
    start: dt.date,
    end: dt.date,
    creds: Credentials | None = None,
    provider: str | None = None,
    level: str = "surface",
) -> SearchResult:
    """Scenes of one sensor family over ``bbox`` between two dates (inclusive).

    ``level`` is "surface" (Level-2 surface reflectance, the configured ``providers``) or
    "toa" (Level-1 top-of-atmosphere reflectance, the configured ``toa_providers``).
    Uses the first usable provider in the configured order unless ``provider`` is given.
    Asset links are stored unsigned; they are signed or authenticated when pixels are read.
    """
    if family not in cfg.sensors:
        raise ValueError(f"Unknown sensor family {family!r}. Configured: {sorted(cfg.sensors)}")
    creds = creds if creds is not None else load_credentials()
    status = provider_status(cfg, family, creds, level)

    if provider is not None:
        if provider not in status:
            raise ValueError(f"{provider!r} is not configured for {family}. Configured: {list(status)}")
        if status[provider] is not None:
            raise RuntimeError(f"Provider {provider} cannot be used for {family}: {status[provider]}")
        chosen = provider
    else:
        chosen = next((name for name, why in status.items() if why is None), None)
        if chosen is None:
            raise RuntimeError(f"No usable {level} provider for {family}: {status}")

    spec = _specs(level)[(chosen, family)]
    items = _fetch_items(cfg.providers[chosen].stac_url, spec.collection, bbox, start, end)
    scenes, rejected, n_items = filter_and_normalise(items, spec, cfg.sensors[family], level)
    unique = deduplicate(scenes)
    if len(unique) < len(scenes):
        rejected["duplicate_processing"] += len(scenes) - len(unique)
    log.info("%s/%s (%s): %d items, %d scenes kept", chosen, family, level, n_items, len(unique))
    return SearchResult(
        provider=chosen,
        scenes=unique,
        n_items=n_items,
        rejected=rejected,
        unusable_providers={name: why for name, why in status.items() if why is not None},
    )
