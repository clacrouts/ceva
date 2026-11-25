"""Provider-independent description of one satellite acquisition."""
from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

REFLECTANCE_BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")
MASK_BAND = "mask"
# Without these three a scene cannot produce a masked MNDWI.
REQUIRED_BANDS = ("green", "swir1", MASK_BAND)


@dataclass(frozen=True)
class BandAsset:
    """Where a band is stored and how to convert it: ``reflectance = stored * scale + offset``."""

    href: str
    scale: float = 1.0
    offset: float = 0.0
    nodata: float | None = None


@dataclass(frozen=True)
class Scene:
    """One acquisition over one grid cell, with the same fields whatever the provider."""

    id: str
    provider: str
    family: str                       # "landsat" or "sentinel2"
    platform: str                     # lower case, e.g. "landsat-5", "sentinel-2a"
    datetime: dt.datetime             # UTC, with the time of day (needed for tides)
    grid_id: str | None               # MGRS tile (Sentinel-2) or WRS-2 path+row (Landsat)
    cloud_cover: float | None         # percent, whole scene
    processing_baseline: str | None   # Sentinel-2 only
    mask_kind: str                    # "landsat_qa_pixel", "s2_scl", or "none" (Level-1 Sentinel-2)
    bands: dict[str, BandAsset]       # canonical band name -> asset
    meta: dict[str, Any] = field(default_factory=dict)   # level ("surface"/"toa"), Landsat entity id, sun elevation


_FRAME_COLUMNS = [
    "id", "provider", "family", "platform", "datetime", "grid_id",
    "cloud_cover", "processing_baseline", "mask_kind", "bands",
]


def scenes_to_frame(scenes: Iterable[Scene]) -> pd.DataFrame:
    """Flat table of scenes; ``bands`` is stored as a JSON string."""
    rows = [
        {
            "id": s.id, "provider": s.provider, "family": s.family, "platform": s.platform,
            "datetime": s.datetime, "grid_id": s.grid_id, "cloud_cover": s.cloud_cover,
            "processing_baseline": s.processing_baseline, "mask_kind": s.mask_kind,
            "bands": json.dumps({k: asdict(v) for k, v in s.bands.items()}, sort_keys=True),
        }
        for s in scenes
    ]
    frame = pd.DataFrame(rows, columns=_FRAME_COLUMNS)
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    return frame
