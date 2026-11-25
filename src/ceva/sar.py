"""Synthetic aperture radar (SAR): chips of radar backscatter around the transects.

Radar sees through clouds, by day and night, so it adds observations where and when the
imagery has none. Sentinel-1 (C band, 2014 onwards, every 6 to 12 days) is read from the
Planetary Computer's radiometrically terrain-corrected collection (``sentinel-1-rtc``:
gamma-nought backscatter on a 10 m UTM grid, VV and VH; CC BY 4.0, processed by Catalyst
from Copernicus Sentinel data). It needs no account: anonymous access tokens are issued (the
documented account procedure has been withdrawn). NISAR (L band) could be added as a second
mission.

Chips follow the optical ones (``ceva.chips``): one grid per coastal segment, read once per
acquisition on a grid covering every segment, stored in their own Icechunk repository
("<unit>-sar.icechunk", groups "<mission>/s<segment>"). Values are backscatter in decibels
as delivered (gamma nought); NaN where the image has no data. Each acquisition records its
orbit direction and relative orbit, because the radar's look direction changes how a beach
appears, so radar waterlines are aligned on the imagery per orbit direction (``ceva.sar_waterlines``).
"""
from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from ceva import __version__
from ceva.chips import ChipGrid, chip_grid, chips_path, grid_bbox_wgs84, read_band, segments, union_grid
from ceva.config import Config
from ceva.reader import gdal_environment, resolve_href

log = logging.getLogger(__name__)

LEVEL = "sar"                       # chips repository "<unit>-sar.icechunk"
POLARIZATIONS = ("vv", "vh")
DB_ENCODING = {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -32768}
MIN_LINEAR = 1e-5                   # -50 dB: floor before taking the logarithm
SCENE_IDS_DTYPE = "<U400"           # the products of one acquisition, space-separated
MAX_FAILED_FRACTION = 0.05
_UA = {"User-Agent": "ceva (+https://github.com/clacrouts/ceva)"}


@dataclass(frozen=True)
class SarScene:
    """One radar product (a slice of an acquisition)."""

    id: str
    mission: str
    platform: str
    datetime: dt.datetime
    orbit_state: str            # ascending / descending
    relative_orbit: int
    datatake: str
    hrefs: dict[str, str] = field(default_factory=dict)
    nodata: float | None = None


def search_sentinel1(cfg: Config, bbox: tuple[float, float, float, float], start: dt.date,
                     end: dt.date) -> list[SarScene]:
    """Sentinel-1 RTC products over ``bbox`` between two dates (inclusive), dual polarization only."""
    from pystac_client import Client

    c = cfg.sar.sentinel1
    catalog = Client.open(cfg.providers[c.provider].stac_url, headers=_UA)
    found = catalog.search(collections=[c.collection], bbox=bbox, datetime=f"{start}/{end}")
    scenes = []
    for item in found.items():
        p = item.properties
        if not all(pol in item.assets for pol in POLARIZATIONS):
            continue
        if p.get("sar:instrument_mode", "IW") != "IW":
            continue
        scenes.append(SarScene(
            id=item.id, mission="sentinel1", platform=str(p.get("platform", "sentinel-1")).lower(),
            datetime=item.datetime, orbit_state=str(p.get("sat:orbit_state", "")).lower(),
            relative_orbit=int(p.get("sat:relative_orbit", -1)),
            datatake=str(p.get("s1:datatake_id", item.id)),
            hrefs={pol: item.assets[pol].href for pol in POLARIZATIONS},
            nodata=c.nodata,
        ))
    return sorted(scenes, key=lambda s: s.datetime)


def group_acquisitions(scenes: list[SarScene]) -> list[list[SarScene]]:
    """Consecutive slices of one datatake are one acquisition."""
    groups: dict[tuple, list[SarScene]] = {}
    for s in scenes:
        groups.setdefault((s.platform, s.datatake, s.datetime.date()), []).append(s)
    return sorted(groups.values(), key=lambda g: g[0].datetime)


def to_db(linear: np.ndarray) -> np.ndarray:
    """Backscatter power to decibels; NaN stays NaN, zero and negatives become the -50 dB floor."""
    with np.errstate(invalid="ignore"):
        return (10.0 * np.log10(np.clip(linear, MIN_LINEAR, None))).astype("float32")


def read_acquisition(group: list[SarScene], grid: ChipGrid, provider: str) -> dict[str, np.ndarray] | None:
    """VV and VH backscatter (dB) of one acquisition on ``grid``: the first slice with data wins."""
    merged: dict[str, np.ndarray] | None = None
    for scene in group:
        data = {pol: read_band(resolve_href(provider, scene.hrefs[pol]), grid, True, scene.nodata)
                for pol in POLARIZATIONS}
        if merged is None:
            merged = data
        else:
            fill = ~np.isfinite(merged["vv"]) & np.isfinite(data["vv"])
            for pol in POLARIZATIONS:
                merged[pol] = np.where(fill, data[pol], merged[pol])
    if merged is None:
        return None
    valid = np.isfinite(merged["vv"]) & np.isfinite(merged["vh"]) & (merged["vv"] > 0) & (merged["vh"] > 0)
    if not valid.any():
        return None
    return {pol: np.where(valid, to_db(merged[pol]), np.nan).astype("float32") for pol in POLARIZATIONS}


def iter_sar_chips(cfg: Config, transects: gpd.GeoDataFrame, *, mission: str = "sentinel1",
                   start: dt.date | None = None, end: dt.date | None = None,
                   progress: Callable[[int, int], None] | None = None):
    """Radar chips of every acquisition, in blocks; yields ``(key, dataset)`` like ``ceva.chips.iter_chips``."""
    from ceva.credentials import load_credentials

    if mission != "sentinel1":
        raise NotImplementedError(f"radar mission {mission!r} is not implemented yet")
    c = cfg.sar.sentinel1
    start = max(start or cfg.time.start, c.start)
    end = end or cfg.time.end
    epsg = int(transects.estimate_utm_crs().to_epsg())
    groups_of_transects = segments(transects, cfg.chips.segment_km, epsg)
    ids = [",".join(str(t) for t in transects.iloc[rows]["transect_id"]) for rows in groups_of_transects]
    grids = [chip_grid(transects.iloc[rows], c.pixel_m, cfg.chips.margin_m, epsg) for rows in groups_of_transects]
    union = union_grid(grids)
    windows = [union.window_of(g) for g in grids]
    scenes = search_sentinel1(cfg, grid_bbox_wgs84(union), start, end)
    acquisitions = group_acquisitions(scenes)
    log.info("%s: %d products (%d acquisitions) from %s, read on a %dx%d grid for %d segments", mission,
             len(scenes), len(acquisitions), c.provider, union.width, union.height, len(grids))
    options = gdal_environment(c.provider, cfg, load_credentials())
    per_acquisition = sum(g.width * g.height for g in grids) * 4 * len(POLARIZATIONS)
    block = max(8, int(cfg.sar.block_bytes // max(per_acquisition, 1)))
    done, failed = 0, []

    def run(group: list[SarScene]):
        nonlocal done
        try:
            with rasterio.Env(**options):
                data = read_acquisition(group, union, c.provider)
        except Exception as exc:     # one unreadable product must not stop a build of thousands
            log.warning("could not read %s: %s", group[0].id, exc)
            done += 1
            return None, [s.id for s in group]
        cut = None
        if data is not None:
            cut = [{k: np.ascontiguousarray(v[w]) for k, v in data.items()} for w in windows]
            cut = [x if np.isfinite(x["vv"]).any() else None for x in cut]
        done += 1
        if progress is not None:
            progress(done, len(acquisitions))
        return cut, []

    with ThreadPoolExecutor(max_workers=cfg.sar.workers) as pool:
        for b0 in range(0, len(acquisitions), block):
            part = acquisitions[b0:b0 + block]
            results = list(pool.map(run, part))
            failed += [i for _, f in results for i in f]
            if acquisitions and len(failed) / len(acquisitions) > MAX_FAILED_FRACTION:
                raise RuntimeError(f"{len(failed)} of {len(acquisitions)} radar acquisitions could not be read, "
                                   f"e.g. {failed[:3]}")
            for k, grid in enumerate(grids):
                kept = [(g, cut[k]) for g, (cut, _) in zip(part, results, strict=True)
                        if cut is not None and cut[k] is not None]
                if not kept:
                    continue
                ds = _assemble(kept, grid, mission, c.provider, c.collection, start, end, len(failed))
                ds.attrs.update(transect_ids=ids[k], segment=k, family=mission)
                yield f"{mission}/s{k:03d}", ds


def _assemble(kept, grid: ChipGrid, mission, provider, collection, start, end, n_failed) -> xr.Dataset:
    dims = ("time", "y", "x")
    ds = xr.Dataset(
        {pol: (dims, np.stack([d[pol] for _, d in kept]).astype("float32")) for pol in POLARIZATIONS},
        coords={
            "time": pd.DatetimeIndex([pd.Timestamp(g[0].datetime).tz_convert(None) for g, _ in kept]),
            # fixed widths: blocks appended later must have the same string types
            "platform": ("time", np.array([g[0].platform for g, _ in kept], dtype="<U24")),
            "orbit_state": ("time", np.array([g[0].orbit_state for g, _ in kept], dtype="<U12")),
            "relative_orbit": ("time", np.array([g[0].relative_orbit for g, _ in kept], dtype="int16")),
            "scene_ids": ("time", np.array([" ".join(s.id for s in g) for g, _ in kept], dtype=SCENE_IDS_DTYPE)),
            "x": grid.x, "y": grid.y,
        },
        attrs={
            "title": f"CEVA radar chips ({mission})", "ceva_version": __version__, "provider": provider,
            "collection": collection, "period": f"{start} to {end}", "n_acquisitions_failed": n_failed,
            "backscatter": "gamma nought, radiometrically terrain corrected", "units": "dB",
            **grid.to_attrs(),
        },
    )
    for pol in POLARIZATIONS:
        ds[pol].attrs.update(long_name=f"{pol.upper()} backscatter (gamma nought)", units="dB")
    return ds.sortby("time")


def save_sar_chips(cfg: Config, name: str, chips, message: str) -> str:
    """Commit radar chips (a mapping or an iterator of (key, dataset)); returns the snapshot id."""
    import icechunk
    from icechunk.xarray import to_icechunk

    path = chips_path(cfg, name, LEVEL)
    path.parent.mkdir(parents=True, exist_ok=True)
    repo = icechunk.Repository.open_or_create(icechunk.local_filesystem_storage(str(path)))
    session = repo.writable_session("main")
    items = chips.items() if isinstance(chips, dict) else chips
    written: set[str] = set()
    for key, ds in items:
        if key in written:
            to_icechunk(ds, session, group=key, mode="a", append_dim="time")
            continue
        chunks = (min(ds.sizes["time"], 32), ds.sizes["y"], ds.sizes["x"])
        encoding = {pol: {**DB_ENCODING, "chunks": chunks} for pol in POLARIZATIONS}
        to_icechunk(ds, session, group=key, mode="w", encoding=encoding)
        written.add(key)
    if not written:
        raise RuntimeError("no radar acquisition holds any data over these transects")
    return session.commit(message)


def open_sar_chips(cfg: Config, name: str, ref: str = "main") -> dict[str, xr.Dataset]:
    """The stored radar chips, keyed "<mission>/s<segment>"; empty when none were built."""
    from ceva.chips import _open_repo

    if not chips_path(cfg, name, LEVEL).exists():
        return {}
    return _open_repo(cfg, name, LEVEL, ref)


def build_sar_chips(cfg: Config, name: str, transects: gpd.GeoDataFrame, **kwargs) -> str:
    """Read and commit the radar chips of a unit; returns the snapshot id."""
    missions = ",".join(cfg.sar.missions)
    return save_sar_chips(cfg, name, (kv for m in cfg.sar.missions for kv in iter_sar_chips(cfg, transects,
                                                                                           mission=m, **kwargs)),
                          f"radar chips ({missions}) for {name}, ceva {__version__}")
