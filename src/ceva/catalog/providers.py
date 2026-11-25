"""What each STAC provider calls things, and how its items become a ``Scene``.

Adding a provider or a collection means adding one ``CollectionSpec``; nothing
else in the pipeline knows provider-specific asset names.

Reflectance scaling is taken from the item's own metadata when it declares it.
Otherwise it follows the published rules: Landsat Collection 2 Level-2 uses
``DN * 2.75e-5 - 0.2``; Sentinel-2 L2A uses ``DN / 10000``, minus 0.1 from
processing baseline 04.00 onwards (the BOA_ADD_OFFSET of -1000 DN).
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any

from ceva.catalog.models import MASK_BAND, REQUIRED_BANDS, BandAsset, Scene

LANDSAT_SCALING = (2.75e-5, -0.2)
S2_SCALE = 1e-4
S2_OFFSET = -0.1
S2_OFFSET_FROM_BASELINE = (4, 0)


class SceneRejected(ValueError):
    """An item that cannot be used. ``reason`` is a short label that can be counted."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class CollectionSpec:
    provider: str
    family: str
    collection: str
    asset_keys: dict[str, str]   # canonical band name -> this provider's asset key
    mask_kind: str


_PC_S2 = {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08", "swir1": "B11", "swir2": "B12", "mask": "SCL"}

SPECS: dict[tuple[str, str], CollectionSpec] = {
    ("planetary_computer", "landsat"): CollectionSpec(
        "planetary_computer", "landsat", "landsat-c2-l2",
        {"blue": "blue", "green": "green", "red": "red", "nir": "nir08",
         "swir1": "swir16", "swir2": "swir22", "mask": "qa_pixel"},
        "landsat_qa_pixel",
    ),
    ("planetary_computer", "sentinel2"): CollectionSpec(
        "planetary_computer", "sentinel2", "sentinel-2-l2a", _PC_S2, "s2_scl",
    ),
    ("earth_search", "sentinel2"): CollectionSpec(
        "earth_search", "sentinel2", "sentinel-2-c1-l2a",
        {"blue": "blue", "green": "green", "red": "red", "nir": "nir",
         "swir1": "swir16", "swir2": "swir22", "mask": "scl"},
        "s2_scl",
    ),
    ("cdse", "sentinel2"): CollectionSpec(
        "cdse", "sentinel2", "sentinel-2-l2a",
        {"blue": "B02_10m", "green": "B03_10m", "red": "B04_10m", "nir": "B08_10m",
         "swir1": "B11_20m", "swir2": "B12_20m", "mask": "SCL_20m"},
        "s2_scl",
    ),
}


# Top-of-atmosphere (Level-1) collections. They supply the reflectance bands only: the cloud
# mask still comes from the Level-2 product of the same acquisition (QA_PIXEL, SCL), so both
# kinds of chip are masked identically. Sentinel-2 L1C is scaled like L2A (DN / 10000, minus
# 0.1 from baseline 04.00, declared by the items). Landsat Level-1 reflectance needs each
# scene's MTL coefficients and sun elevation, so its scaling is filled in when it is read
# (see ``ceva.catalog.usgs``).
TOA_SPECS: dict[tuple[str, str], CollectionSpec] = {
    ("cdse", "sentinel2"): CollectionSpec(
        "cdse", "sentinel2", "sentinel-2-l1c",
        {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08", "swir1": "B11", "swir2": "B12"},
        "none",
    ),
    ("usgs", "landsat"): CollectionSpec(
        "usgs", "landsat", "landsat-c2l1",
        {"blue": "blue", "green": "green", "red": "red", "nir": "nir08", "swir1": "swir16", "swir2": "swir22",
         "mask": "qa_pixel"},
        "landsat_qa_pixel",
    ),
}


def baseline_tuple(baseline: str | None) -> tuple[int, int]:
    """'05.11', 'N0511' or '5.11' -> (5, 11). Unknown or malformed -> (0, 0)."""
    match = re.fullmatch(r"\s*N?(\d{1,2})\.?(\d{2})\s*", baseline or "")
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _declared_scaling(asset: dict[str, Any]) -> tuple[float, float | None, float | None] | None:
    """Scale, offset and nodata as declared by the asset, in any of the STAC raster layouts."""
    candidates: list[dict[str, Any]] = []
    if asset.get("raster:bands"):
        band = asset["raster:bands"][0]
        candidates.append({"scale": band.get("scale"), "offset": band.get("offset"), "nodata": band.get("nodata")})
    if asset.get("bands"):
        band = asset["bands"][0]
        candidates.append({"scale": band.get("raster:scale"), "offset": band.get("raster:offset"),
                           "nodata": band.get("nodata")})
    candidates.append({"scale": asset.get("raster:scale"), "offset": asset.get("raster:offset"),
                       "nodata": asset.get("nodata")})
    for c in candidates:
        scale = _as_float(c["scale"])
        if scale is not None:
            return scale, _as_float(c["offset"]), _as_float(c["nodata"])
    return None


def _reflectance_scaling(family: str, baseline: str | None, asset: dict[str, Any],
                         level: str = "surface") -> tuple[float, float, float | None]:
    declared = _declared_scaling(asset)
    if family == "landsat" and level == "toa":
        return float("nan"), float("nan"), 0.0      # per scene, from its MTL file, when read
    if family == "landsat":
        if declared is not None:
            return declared[0], declared[1] if declared[1] is not None else 0.0, declared[2]
        return LANDSAT_SCALING[0], LANDSAT_SCALING[1], 0.0

    if declared is not None and declared[1] is not None:
        if level == "toa" and declared[2] is None:
            return declared[0], declared[1], 0.0      # Level-1C: DN 0 is no data
        return declared
    # The offset is not declared, so it has to come from the processing baseline.
    if not baseline or baseline_tuple(baseline) == (0, 0):
        raise SceneRejected("sentinel2_unknown_processing_baseline")
    offset = S2_OFFSET if baseline_tuple(baseline) >= S2_OFFSET_FROM_BASELINE else 0.0
    scale = declared[0] if declared is not None else S2_SCALE
    nodata = declared[2] if declared is not None else 0.0
    return scale, offset, nodata


def _grid_id(family: str, item_id: str, props: dict[str, Any]) -> str | None:
    if family == "landsat":
        path, row = props.get("landsat:wrs_path"), props.get("landsat:wrs_row")
        if path is not None and row is not None:
            return f"{int(path):03d}{int(row):03d}"
        match = re.search(r"_(\d{6})_\d{8}_", item_id)
        return match.group(1) if match else None
    match = re.search(r"_T(\d{2}[A-Z]{3})(?:_|$)", item_id)
    if match:
        return match.group(1)
    code = props.get("s2:mgrs_tile") or str(props.get("grid:code") or "").removeprefix("MGRS-")
    return code or None


def normalise_item(item: dict[str, Any], spec: CollectionSpec, level: str = "surface") -> Scene:
    """Turn one STAC item (as a dict) into a ``Scene``, or raise ``SceneRejected``.

    ``level`` is "surface" (Level-2) or "toa" (Level-1: no mask band is required).
    """
    props, assets = item.get("properties") or {}, item.get("assets") or {}

    required = [n for n in REQUIRED_BANDS if n != MASK_BAND or level == "surface"]
    missing = [name for name in required if spec.asset_keys.get(name) not in assets]
    if missing:
        raise SceneRejected("missing_asset_" + "_".join(missing))

    raw_datetime = props.get("datetime") or props.get("start_datetime")
    if not raw_datetime:
        raise SceneRejected("no_datetime")
    when = dt.datetime.fromisoformat(raw_datetime)
    when = when.replace(tzinfo=dt.UTC) if when.tzinfo is None else when.astimezone(dt.UTC)

    baseline = None
    if spec.family == "sentinel2":
        baseline = props.get("s2:processing_baseline") or props.get("processing:version")

    bands: dict[str, BandAsset] = {}
    for name, key in spec.asset_keys.items():
        asset = assets.get(key)
        if asset is None:
            continue
        if name == MASK_BAND:
            bands[name] = BandAsset(href=asset["href"])
        else:
            scale, offset, nodata = _reflectance_scaling(spec.family, baseline, asset, level)
            bands[name] = BandAsset(href=asset["href"], scale=scale, offset=offset, nodata=nodata)
    meta: dict[str, Any] = {"level": level}
    if spec.family == "landsat":
        meta.update(entity_id=props.get("landsat:scene_id"), sun_elevation=_as_float(props.get("view:sun_elevation")))

    return Scene(
        id=item["id"],
        provider=spec.provider,
        family=spec.family,
        platform=str(props.get("platform") or "").lower(),
        datetime=when,
        grid_id=_grid_id(spec.family, item["id"], props),
        cloud_cover=_as_float(props.get("eo:cloud_cover")),
        processing_baseline=baseline,
        mask_kind=spec.mask_kind,
        bands=bands,
        meta=meta,
    )
