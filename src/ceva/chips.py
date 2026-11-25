"""Small 2D images ("chips") around a set of transects, one per acquisition.

The transect cube holds reflectance along each transect only. Engines that look at a
pixel's neighbours, like CoastSat's classifier, need the image around the transects
instead. A chip grid is a north-up grid in the unit's UTM zone, one per sensor family,
at CoastSat's pixel sizes (Landsat 15 m, Sentinel-2 10 m). Each band is resampled once,
bilinearly, from its own grid onto the chip grid; the quality band is taken from the
nearest pixel. Values are stored unmasked, as in the cube.

Chip bands hold top-of-atmosphere reflectance (Level-1) or surface reflectance (Level-2),
per family as configured in ``chips.reflectance``. The quality band always comes from the
Level-2 product of the same acquisition, so both are masked the same way.

Chips are stored next to the cube in an Icechunk repository per reflectance level
("<unit>.icechunk" for surface, "<unit>-toa.icechunk" for top of atmosphere), one group
per family and coastal segment.
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from affine import Affine
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from ceva import __version__
from ceva.catalog import Scene, provider_status, search_scenes
from ceva.catalog.models import MASK_BAND
from ceva.catalog.search import SearchResult, match_key
from ceva.config import Config
from ceva.credentials import Credentials, load_credentials
from ceva.cube import MAX_FAILED_FRACTION, REFLECTANCE_ENCODING, _no_data, group_acquisitions
from ceva.reader import encode_qa, gdal_environment, resolve_href

log = logging.getLogger(__name__)

PAD_PIXELS = 2   # source pixels read beyond the chip, so bilinear weights exist at its edges
LEVELS = ("surface", "toa")
CHIP_BLOCK_BYTES = 2 * 2**30   # chips held in memory at once, before they are written
CLOUD_MARGIN = 0.02   # cloud fraction above CoastSat's limit before an acquisition is left unread

_SLOTS: dict[str, threading.BoundedSemaphore] = {}
_SLOTS_LOCK = threading.Lock()


def _slots(cfg: Config, provider: str) -> threading.BoundedSemaphore | None:
    """Shared limit on parallel reads from a provider that caps its connections (CDSE: 4)."""
    limit = cfg.providers[provider].max_connections if provider in cfg.providers else None
    if limit is None:
        return None
    with _SLOTS_LOCK:
        return _SLOTS.setdefault(provider, threading.BoundedSemaphore(limit))


@dataclass(frozen=True)
class ChipGrid:
    """A north-up grid: ``x0``/``y0`` are the left and top edges, in metres of ``epsg``."""

    epsg: int
    x0: float
    y0: float
    pixel_m: float
    width: int
    height: int

    @property
    def transform(self) -> Affine:
        return Affine(self.pixel_m, 0.0, self.x0, 0.0, -self.pixel_m, self.y0)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0 - self.height * self.pixel_m, self.x0 + self.width * self.pixel_m, self.y0)

    @property
    def x(self) -> np.ndarray:
        return self.x0 + (np.arange(self.width) + 0.5) * self.pixel_m

    @property
    def y(self) -> np.ndarray:
        return self.y0 - (np.arange(self.height) + 0.5) * self.pixel_m

    @property
    def georef(self) -> np.ndarray:
        """CoastSat's georef: maps pixel indices (row, column) to the centre of that pixel."""
        p = self.pixel_m
        return np.array([self.x0 + p / 2, p, 0.0, self.y0 - p / 2, 0.0, -p])

    def window_of(self, other: ChipGrid) -> tuple[slice, slice]:
        """Rows and columns of this grid that ``other`` covers (same projection and pixels)."""
        if other.epsg != self.epsg or other.pixel_m != self.pixel_m:
            raise ValueError("the grids do not share a projection and pixel size")
        c0 = int(round((other.x0 - self.x0) / self.pixel_m))
        r0 = int(round((self.y0 - other.y0) / self.pixel_m))
        if c0 < 0 or r0 < 0 or c0 + other.width > self.width or r0 + other.height > self.height:
            raise ValueError("the grid is not inside this one")
        return slice(r0, r0 + other.height), slice(c0, c0 + other.width)

    def to_attrs(self) -> dict:
        return {"epsg": self.epsg, "x0": self.x0, "y0": self.y0, "pixel_m": self.pixel_m,
                "width": self.width, "height": self.height}

    @classmethod
    def from_attrs(cls, attrs: dict) -> ChipGrid:
        return cls(int(attrs["epsg"]), float(attrs["x0"]), float(attrs["y0"]), float(attrs["pixel_m"]),
                   int(attrs["width"]), int(attrs["height"]))


def chip_grid(transects: gpd.GeoDataFrame, pixel_m: float, margin_m: float, epsg: int | None = None) -> ChipGrid:
    """The grid covering the transects plus ``margin_m``, edges on multiples of ``pixel_m``."""
    if epsg is None:
        epsg = int(transects.estimate_utm_crs().to_epsg())
    left, bottom, right, top = transects.to_crs(epsg).total_bounds
    x0 = np.floor((left - margin_m) / pixel_m) * pixel_m
    x1 = np.ceil((right + margin_m) / pixel_m) * pixel_m
    y0 = np.ceil((top + margin_m) / pixel_m) * pixel_m
    y1 = np.floor((bottom - margin_m) / pixel_m) * pixel_m
    return ChipGrid(epsg, float(x0), float(y0), float(pixel_m), int(round((x1 - x0) / pixel_m)),
                    int(round((y0 - y1) / pixel_m)))


def union_grid(grids: list[ChipGrid]) -> ChipGrid:
    """The smallest grid holding all of ``grids``, which share a projection and a pixel lattice."""
    p, epsg = grids[0].pixel_m, grids[0].epsg
    x0, y0 = min(g.x0 for g in grids), max(g.y0 for g in grids)
    x1 = max(g.x0 + g.width * p for g in grids)
    y1 = min(g.y0 - g.height * p for g in grids)
    return ChipGrid(epsg, x0, y0, p, int(round((x1 - x0) / p)), int(round((y0 - y1) / p)))


def read_band(path: str, grid: ChipGrid, reflectance: bool, nodata: float | None = None) -> np.ndarray:
    """Band 1 of ``path`` on ``grid``.

    Reflectance bands are resampled bilinearly into float32, NaN where the image has no
    data. Other bands (quality flags) take the nearest pixel and keep their type, 0 outside
    the image.
    """
    with rasterio.open(path) as src:
        dtype = "float32" if reflectance else src.dtypes[0]
        fill = np.nan if reflectance else 0
        out = np.full((grid.height, grid.width), fill, dtype=dtype)
        left, bottom, right, top = transform_bounds(f"EPSG:{grid.epsg}", src.crs, *grid.bounds, densify_pts=21)
        win = from_bounds(left, bottom, right, top, transform=src.transform)
        c0 = max(int(np.floor(win.col_off)) - PAD_PIXELS, 0)
        r0 = max(int(np.floor(win.row_off)) - PAD_PIXELS, 0)
        c1 = min(int(np.ceil(win.col_off + win.width)) + PAD_PIXELS, src.width)
        r1 = min(int(np.ceil(win.row_off + win.height)) + PAD_PIXELS, src.height)
        if c1 <= c0 or r1 <= r0:
            return out
        window = Window(c0, r0, c1 - c0, r1 - r0)
        data = src.read(1, window=window)
        if nodata is None:
            nodata = src.nodata
        if reflectance:
            source = data.astype("float32")
            if nodata is not None:
                source[data == nodata] = np.nan
            src_nodata, resampling = np.nan, Resampling.bilinear
        else:
            # 0 means "no data" in both quality bands (Landsat QA_PIXEL fill, Sentinel-2 SCL);
            # declaring it stops GDAL from rewriting real zeros as 1 to keep them apart from fill.
            source, src_nodata, resampling = data, 0, Resampling.nearest
        reproject(source, out, src_transform=src.window_transform(window), src_crs=src.crs,
                  src_nodata=src_nodata, dst_transform=grid.transform, dst_crs=f"EPSG:{grid.epsg}",
                  dst_nodata=fill, resampling=resampling)
    return out


@dataclass(frozen=True)
class ToaSource:
    """Where the top-of-atmosphere bands of each Level-2 scene are: its Level-1 twin."""

    scenes: dict[tuple, Scene]       # match key -> Level-1 scene
    gdal_options: dict
    slots: threading.BoundedSemaphore | None = None

    def twin(self, scene: Scene) -> Scene | None:
        return self.scenes.get(match_key(scene))


def read_quality(scene: Scene, grid: ChipGrid) -> np.ndarray:
    """The quality byte of ``scene`` on the grid (inside the provider's ``rasterio.Env``)."""
    raw = read_band(resolve_href(scene.provider, scene.bands[MASK_BAND].href), grid, False)
    return encode_qa(raw, scene.mask_kind)


def read_scene(scene: Scene, grid: ChipGrid, bands: tuple[str, ...],
               toa: ToaSource | None = None, qa: np.ndarray | None = None,
               local: dict[str, str] | None = None) -> dict[str, np.ndarray]:
    """Reflectance of ``bands`` and the quality byte (key ``qa``) on the grid.

    With ``toa``, the reflectance comes from the scene's Level-1 twin (which must exist) and
    only the quality band from ``scene``. ``qa``, when given, is the quality byte already read.
    ``local`` maps Level-1 asset links to copies already downloaded, read instead of the remote
    files. Must run inside ``rasterio.Env(**gdal_environment(...))`` for ``scene``'s provider.
    """
    out: dict[str, np.ndarray] = {}
    source = toa.twin(scene) if toa is not None else scene
    if source is None:
        raise LookupError(f"no top-of-atmosphere product for {scene.id}")
    local = local or {}

    def read_bands(names):
        for name in names:
            asset = source.bands[name]
            path = local.get(asset.href) or resolve_href(source.provider, asset.href)
            stored = read_band(path, grid, True, asset.nodata)
            out[name] = stored * np.float32(asset.scale) + np.float32(asset.offset)

    if toa is None:
        read_bands(bands)
    else:
        remote = [name for name in bands if source.bands[name].href not in local]
        with rasterio.Env(**toa.gdal_options):
            read_bands([name for name in bands if name not in remote])
            if toa.slots is None or not remote:
                read_bands(remote)
            else:
                with toa.slots:
                    read_bands(remote)
        out = {name: out[name] for name in bands}
    out["qa"] = read_quality(scene, grid) if qa is None else qa
    return out


def _merge_quality(qas: list[np.ndarray], family: str) -> np.ndarray:
    """The quality band of an acquisition, merged as ``_read_acquisition`` merges its scenes."""
    merged = qas[0]
    for qa in qas[1:]:
        merged = np.where(_no_data(merged, family) & ~_no_data(qa, family), qa, merged)
    return merged


def _usable_scenes(scenes: list[Scene], toa: ToaSource | None) -> list[Scene]:
    """With ``toa``, the Level-2 scenes that have a Level-1 twin (the others are left out, not failed)."""
    return [s for s in scenes if toa.twin(s) is not None] if toa is not None else list(scenes)


def _read_qualities(scenes: list[Scene], grid: ChipGrid) -> tuple[dict[str, np.ndarray], list[str]]:
    """The quality band of each scene, and the scenes that could not be read (inside their ``rasterio.Env``)."""
    quality: dict[str, np.ndarray] = {}
    failed: list[str] = []
    for scene in scenes:
        try:
            quality[scene.id] = read_quality(scene, grid)
        except Exception as exc:   # one unreadable scene must not stop a build of thousands
            log.warning("could not read %s: %s", scene.id, exc)
            failed.append(scene.id)
    return quality, failed


def _read_acquisition(scenes: list[Scene], grid: ChipGrid, bands: tuple[str, ...], gdal_options: dict,
                      toa: ToaSource | None = None, wanted: Callable[[np.ndarray], bool] | None = None,
                      quality: dict[str, np.ndarray] | None = None, local: dict[str, str] | None = None,
                      ) -> tuple[dict[str, np.ndarray] | None, list[str]]:
    """Merge the scenes of one acquisition: the first scene with data wins at each pixel.

    With ``toa``, Level-2 scenes without a Level-1 twin are left out (not counted as failed).
    With ``wanted``, the quality bands are read first and merged, and the reflectance bands only
    if ``wanted(quality)`` is true; otherwise the acquisition comes back empty (not as failed).
    ``quality`` holds quality bands already read (scene id -> band; scenes missing from it are
    left out); ``local`` maps Level-1 links to downloaded copies (see ``read_scene``).
    """
    merged: dict[str, np.ndarray] | None = None
    failed: list[str] = []
    if not scenes:
        return None, failed
    family = scenes[0].family
    scenes = _usable_scenes(scenes, toa)
    with rasterio.Env(**gdal_options):
        if quality is None and wanted is not None:
            quality, failed = _read_qualities(scenes, grid)
        if quality is not None:
            scenes = [s for s in scenes if s.id in quality]
            if not scenes or (wanted is not None
                              and not wanted(_merge_quality([quality[s.id] for s in scenes], family))):
                return None, failed
        for scene in scenes:
            try:
                data = read_scene(scene, grid, bands, toa, (quality or {}).get(scene.id), local)
            except Exception as exc:   # one unreadable scene must not stop a build of thousands
                log.warning("could not read %s: %s", scene.id, exc)
                failed.append(scene.id)
                continue
            if merged is None:
                merged = data
            else:
                fill = _no_data(merged["qa"], family) & ~_no_data(data["qa"], family)
                for key in merged:
                    merged[key] = np.where(fill, data[key], merged[key])
    if merged is None or not np.isfinite(merged[bands[0]]).any():
        return None, failed
    return merged, failed


def grid_bbox_wgs84(grid: ChipGrid) -> tuple[float, float, float, float]:
    return tuple(transform_bounds(f"EPSG:{grid.epsg}", "EPSG:4326", *grid.bounds, densify_pts=21))


def segments(transects: gpd.GeoDataFrame, max_km: float, epsg: int | None = None) -> list[np.ndarray]:
    """Groups of transects (row positions), consecutive along the coast and at most ``max_km`` long.

    A long coast becomes a series of short chips: the bounding box of the whole coast would be
    mostly land and sea far from any transect.
    """
    if epsg is None:
        epsg = int(transects.estimate_utm_crs().to_epsg())
    mid = np.array([[g.interpolate(0.5, normalized=True).x, g.interpolate(0.5, normalized=True).y]
                    for g in transects.to_crs(epsg).geometry])
    if len(mid) < 2:
        return [np.arange(len(mid))]
    centred = mid - mid.mean(axis=0)
    along = centred @ np.linalg.svd(centred, full_matrices=False)[2][0]
    order = np.argsort(along)
    out, current, first = [], [order[0]], along[order[0]]
    for i in order[1:]:
        if along[i] - first > max_km * 1000.0:
            out.append(np.array(current))
            current, first = [i], along[i]
        else:
            current.append(i)
    out.append(np.array(current))
    return out


def reflectance_levels(cfg: Config, families: tuple[str, ...], creds: Credentials) -> dict[str, str]:
    """The reflectance each family's chips can have now: the configured one, or surface
    reflectance (with a warning) when no top-of-atmosphere source is usable."""
    out = {}
    for family in families:
        level = cfg.chips.reflectance[family]
        if level == "toa":
            status = provider_status(cfg, family, creds, "toa")
            if not any(why is None for why in status.values()):
                log.warning("%s chips: no usable top-of-atmosphere source (%s); using surface reflectance",
                            family, "; ".join(f"{k}: {v}" for k, v in status.items()) or "none configured")
                level = "surface"
        out[family] = level
    return out


@dataclass
class _Plan:
    """What reading one family's chips of one unit needs: its grids, scenes and sources."""

    family: str
    grids: list[ChipGrid]
    union: ChipGrid
    windows: list
    ids: list[str]                   # transect ids of each segment, comma-separated
    result: SearchResult             # the Level-2 scenes
    options: dict                    # GDAL options of their provider
    toa: ToaSource | None
    toa_provider: str
    n_without: int                   # Level-2 scenes without a Level-1 twin
    groups: dict[tuple, list[Scene]]  # acquisition -> its scenes over this unit
    failed: list[str] = field(default_factory=list)


def _acquisition_id(group: list[Scene]) -> tuple:
    """The same for one acquisition whatever the unit (``group_acquisitions`` groups by satellite and day)."""
    return (group[0].family, group[0].platform, group[0].datetime.date())


def _plan(cfg: Config, transects: gpd.GeoDataFrame, family: str, level: str, start: dt.date, end: dt.date,
          creds: Credentials) -> _Plan:
    epsg = int(transects.estimate_utm_crs().to_epsg())
    groups_of_transects = segments(transects, cfg.chips.segment_km, epsg)
    ids = [",".join(str(t) for t in transects.iloc[rows]["transect_id"]) for rows in groups_of_transects]
    grids = [chip_grid(transects.iloc[rows], cfg.chips.pixel_m[family], cfg.chips.margin_m, epsg)
             for rows in groups_of_transects]
    union = union_grid(grids)
    bbox = grid_bbox_wgs84(union)
    result = search_scenes(cfg, family, bbox, start, end, creds=creds)
    toa, toa_provider, n_without = None, "", 0
    if level == "toa":
        found = search_scenes(cfg, family, bbox, start, end, creds=creds, level="toa")
        toa_provider = found.provider
        toa = ToaSource({match_key(s): s for s in found.scenes}, gdal_environment(found.provider, cfg, creds),
                        _slots(cfg, found.provider))
        n_without = sum(toa.twin(s) is None for s in result.scenes)
    groups = {_acquisition_id(g): g for g in group_acquisitions(result.scenes)}
    log.info("%s: %d scenes (%d acquisitions) from %s%s, read on a %dx%d grid for %d segments", family,
             len(result.scenes), len(groups), result.provider,
             f", {len(result.scenes) - n_without} with a Level-1 twin on {toa_provider}" if toa else "",
             union.width, union.height, len(grids))
    return _Plan(family, grids, union, [union.window_of(g) for g in grids], ids, result,
                 gdal_environment(result.provider, cfg, creds), toa, toa_provider, n_without, groups)


def _download(href: str, provider: str, dest: Path, toa: ToaSource) -> int:
    """Copy one Level-1 file to ``dest`` (whole, one connection of the provider's). Returns its size."""
    from osgeo import gdal

    partial = dest.with_name(dest.name + ".part")
    with rasterio.Env(**toa.gdal_options):
        if toa.slots is None:
            status = gdal.CopyFile(resolve_href(provider, href), str(partial))
        else:
            with toa.slots:
                status = gdal.CopyFile(resolve_href(provider, href), str(partial))
    if status != 0:
        partial.unlink(missing_ok=True)
        raise OSError(f"could not download {href}")
    partial.rename(dest)
    return dest.stat().st_size


def iter_chips_batch(
    cfg: Config,
    units: dict[str, gpd.GeoDataFrame],
    *,
    families: tuple[str, ...] = ("landsat", "sentinel2"),
    level: str = "surface",
    start: dt.date | None = None,
    end: dt.date | None = None,
    creds: Credentials | None = None,
    progress: Callable[[int, int], None] | None = None,
):
    """Read the chips of several units (each a set of transects) together, acquisition by acquisition.

    For each unit, each acquisition is read once on a grid covering all its coastal segments and
    cut into one chip per segment: the segment grids lie on that grid's pixels, so the chips are
    the same as if each had been read on its own. With ``chips.skip_cloudy``, each unit's quality
    band is read first, and its bands only if CoastSat would map at least one of its segments
    (``too_cloudy``); the segments it would skip as too cloudy are not stored.

    Top-of-atmosphere files that several units need (units on the same Sentinel-2 tile) are read
    once for all of them, as the provider's ``share_files`` says: "cache", each unit reads its window
    in turn and the blocks they share (the tile headers of a JPEG 2000 file, which has no tile index
    and must be walked through, and overlapping tiles) come from GDAL's block cache; "download", the
    file is copied once, whole, every unit's chips are cut from the copy, and the copy is deleted once
    the acquisition is done (fewer requests and faster; more bytes, unless many units share the file).
    Files one unit needs are read remotely, window only. Each way gives the same chips.

    Yields ``(unit, key, dataset)`` with ``key`` = "<family>/s<segment>", in time order; a key
    comes back once per block that has data on it (``save_chips`` appends them). Each dataset
    lists its transects in ``attrs["transect_ids"]`` (comma-separated).
    """
    import shutil
    import tempfile

    from ceva.engines_coastsat import too_cloudy

    creds = creds if creds is not None else load_credentials()
    start, end = start or cfg.time.start, end or cfg.time.end
    bands = tuple(cfg.chips.bands)
    for family in families:
        plans = {name: _plan(cfg, transects, family, level, start, end, creds) for name, transects in units.items()}
        acquisitions = sorted({a for plan in plans.values() for a in plan.groups},
                              key=lambda a: min(p.groups[a][0].datetime for p in plans.values() if a in p.groups))
        # bound the memory: the chips of one block of acquisitions at a time
        per_acquisition = sum(g.width * g.height for plan in plans.values() for g in plan.grids) * (4 * len(bands) + 1)
        block = max(8, int(CHIP_BLOCK_BYTES // max(per_acquisition, 1)))
        done, shared_files, shared_bytes = 0, 0, 0
        scratch = Path(tempfile.mkdtemp(prefix="l1c-", dir=_scratch_dir(cfg))) if level == "toa" else None

        def run(acquisition: tuple, family: str = family, plans: dict = plans, scratch: Path | None = scratch,
                n: int = len(acquisitions)):
            nonlocal done, shared_files, shared_bytes
            mine = {name: plan for name, plan in plans.items() if acquisition in plan.groups}
            clear: dict[str, list[bool]] = {}
            quality: dict[str, dict | None] = {}
            failed: dict[str, list[str]] = {name: [] for name in mine}
            for name, plan in mine.items():
                clear[name] = [True] * len(plan.windows)
                quality[name] = None
                if not cfg.chips.skip_cloudy:
                    continue
                scenes = _usable_scenes(plan.groups[acquisition], plan.toa)
                with rasterio.Env(**plan.options):
                    quality[name], failed[name] = _read_qualities(scenes, plan.union)
                kept = [quality[name][s.id] for s in scenes if s.id in quality[name]]
                clear[name] = ([not too_cloudy(qa[w], family, cfg, CLOUD_MARGIN)
                                for qa in [_merge_quality(kept, family)] for w in plan.windows] if kept
                               else [False] * len(plan.windows))
            wanting = [name for name in mine if any(clear[name])]
            # Level-1 files that more than one unit reads: downloaded once
            users: dict[str, tuple[str, Scene]] = {}
            count: dict[str, int] = {}
            for name in wanting:
                plan = mine[name]
                if plan.toa is None:
                    continue
                for scene in _usable_scenes(plan.groups[acquisition], plan.toa):
                    twin = plan.toa.twin(scene)
                    for band in bands:
                        href = twin.bands[band].href
                        users[href] = (twin.provider, twin)
                        count[href] = count.get(href, 0) + 1
            local: dict[str, str] = {}
            try:
                for href, (provider, twin) in users.items():
                    if count[href] < 2 or _share_policy(cfg, provider) != "download":
                        continue                       # read in turn by each unit, through the cache
                    dest = scratch / f"{twin.id}_{Path(href).name}"
                    try:
                        size = _download(href, provider, dest, mine[wanting[0]].toa)
                    except Exception as exc:          # read remotely instead
                        log.warning("could not download %s: %s", href, exc)
                        continue
                    local[href] = str(dest)
                    with _SLOTS_LOCK:
                        shared_files += 1
                        shared_bytes += size
                out = {}
                for name, plan in mine.items():
                    data = None
                    if name in wanting:
                        data, more = _read_acquisition(plan.groups[acquisition], plan.union, bands, plan.options,
                                                       plan.toa, quality=quality[name], local=local)
                        failed[name] += more
                    cut = None
                    if data is not None:   # copies, so that the large array is freed
                        cut = [{k: np.ascontiguousarray(v[w]) for k, v in data.items()} for w in plan.windows]
                        cut = [c if ok and np.isfinite(c[bands[0]]).any() else None
                               for c, ok in zip(cut, clear[name], strict=True)]
                    out[name] = (cut, failed[name])
            finally:
                for path in local.values():
                    Path(path).unlink(missing_ok=True)
            done += 1
            if progress is not None:
                progress(done, n)
            return out

        try:
            with ThreadPoolExecutor(max_workers=cfg.chips.workers) as pool:
                for b0 in range(0, len(acquisitions), block):
                    part = acquisitions[b0:b0 + block]
                    results = list(pool.map(run, part))
                    for name, plan in plans.items():
                        plan.failed += [i for r in results if name in r for i in r[name][1]]
                        if plan.result.scenes and len(plan.failed) / len(plan.result.scenes) > MAX_FAILED_FRACTION:
                            raise RuntimeError(f"{name}: {len(plan.failed)} of {len(plan.result.scenes)} scenes "
                                               f"could not be read, e.g. {plan.failed[:3]}")
                        for k, grid in enumerate(plan.grids):
                            kept = [(plan.groups[a], r[name][0][k]) for a, r in zip(part, results, strict=True)
                                    if name in r and r[name][0] is not None and r[name][0][k] is not None]
                            if not kept:
                                continue
                            ds = _assemble(kept, grid, bands, family, plan.result.provider, start, end,
                                           len(plan.failed), level)
                            ds.attrs.update(transect_ids=plan.ids[k], segment=k, family=family, reflectance=level,
                                            toa_provider=plan.toa_provider, n_scenes_without_toa=plan.n_without)
                            yield name, f"{family}/s{k:03d}", ds
        finally:
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)
        if shared_files:
            log.info("%s: %d top-of-atmosphere files shared by several units, downloaded once (%.1f GB)",
                     family, shared_files, shared_bytes / 1e9)


def _share_policy(cfg: Config, provider: str) -> str:
    """How units share a provider's files: ``ProviderConfig.share_files`` (download for unconfigured ones)."""
    return cfg.providers[provider].share_files if provider in cfg.providers else "download"


def _scratch_dir(cfg: Config) -> Path:
    """Where downloaded Level-1 files wait while they are cut (deleted afterwards)."""
    path = cfg.data_dir / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_chips(cfg: Config, transects: gpd.GeoDataFrame, **kwargs):
    """The chips of one unit (``iter_chips_batch`` with that unit alone); yields ``(key, dataset)``."""
    for _, key, ds in iter_chips_batch(cfg, {"unit": transects}, **kwargs):
        yield key, ds


def build_chips(cfg: Config, transects: gpd.GeoDataFrame, **kwargs) -> dict[str, xr.Dataset]:
    """All the chips in memory (small units and tests; ``save_chips`` streams them instead)."""
    parts: dict[str, list[xr.Dataset]] = {}
    for key, ds in iter_chips(cfg, transects, **kwargs):
        parts.setdefault(key, []).append(ds)
    out = {k: v[0] if len(v) == 1 else xr.concat(v, dim="time", combine_attrs="override").assign_attrs(v[-1].attrs)
           for k, v in parts.items()}
    if not out:
        raise RuntimeError("no acquisition holds any data over these transects")
    return out


def _assemble(kept, grid: ChipGrid, bands, family, provider, start, end, n_failed, level="surface") -> xr.Dataset:
    dims = ("time", "y", "x")
    data_vars = {name: (dims, np.stack([d[name] for _, d in kept]).astype("float32")) for name in bands}
    data_vars["qa"] = (dims, np.stack([d["qa"] for _, d in kept]))
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": pd.DatetimeIndex([pd.Timestamp(g[0].datetime).tz_convert(None) for g, _ in kept]),
            "platform": ("time", np.array([g[0].platform for g, _ in kept])),
            "family": ("time", np.array([g[0].family for g, _ in kept])),
            "scene_ids": ("time", np.array([" ".join(s.id for s in g) for g, _ in kept])),
            "x": grid.x, "y": grid.y,
        },
        attrs={
            "title": f"CEVA chips ({family})", "ceva_version": __version__, "provider": provider,
            "period": f"{start} to {end}", "n_scenes_failed": n_failed,
            "qa_meaning": "landsat: low byte of QA_PIXEL; sentinel2: scene classification (SCL)",
            **grid.to_attrs(),
        },
    )
    kind = "top-of-atmosphere" if level == "toa" else "surface"
    for name in bands:
        ds[name].attrs.update(long_name=f"{name} {kind} reflectance", units="1")
    return ds.sortby("time")


def chips_path(cfg: Config, name: str, level: str = "surface") -> Path:
    return cfg.data_dir / "chips" / (f"{name}.icechunk" if level == "surface" else f"{name}-{level}.icechunk")


class _ChipWriter:
    """Writes the chips of one unit into its Icechunk repository as they come, then commits."""

    def __init__(self, cfg: Config, name: str, level: str):
        import icechunk

        path = chips_path(cfg, name, level)
        path.parent.mkdir(parents=True, exist_ok=True)
        repo = icechunk.Repository.open_or_create(icechunk.local_filesystem_storage(str(path)))
        self.session = repo.writable_session("main")
        self.written: set[str] = set()

    def write(self, key: str, ds: xr.Dataset) -> None:
        from icechunk.xarray import to_icechunk

        if key in self.written:            # a later block of the same chips: add it along time
            to_icechunk(ds, self.session, group=key, mode="a", append_dim="time")
            return
        chunks = (min(ds.sizes["time"], 32), ds.sizes["y"], ds.sizes["x"])
        encoding = {v: dict(REFLECTANCE_ENCODING) for v in ds.data_vars if v != "qa"}
        for var in ds.data_vars:
            encoding.setdefault(var, {})["chunks"] = chunks
        to_icechunk(ds, self.session, group=key, mode="w", encoding=encoding)
        self.written.add(key)

    def commit(self, message: str) -> str:
        if not self.written:
            raise RuntimeError("no acquisition holds any data over these transects")
        return self.session.commit(message)


def save_chips(cfg: Config, name: str, chips, message: str, level: str = "surface") -> str:
    """Commit chips, given as a mapping or an iterator of (key, dataset), and return the snapshot id.

    Each dataset is written as soon as it comes, so a long coast never sits in memory at once.
    """
    writer = _ChipWriter(cfg, name, level)
    for key, ds in (chips.items() if isinstance(chips, dict) else chips):
        writer.write(key, ds)
    return writer.commit(message)


def save_chips_batch(cfg: Config, chips, messages: dict[str, str], level: str = "surface") -> dict[str, str]:
    """Commit the chips of several units, given as an iterator of (unit, key, dataset) (``iter_chips_batch``).

    Returns the snapshot id of each unit.
    """
    writers: dict[str, _ChipWriter] = {}
    for name, key, ds in chips:
        if name not in writers:
            writers[name] = _ChipWriter(cfg, name, level)
        writers[name].write(key, ds)
    missing = sorted(set(messages) - set(writers))
    if missing:
        raise RuntimeError(f"no acquisition holds any data over the transects of {', '.join(missing)}")
    return {name: writer.commit(messages[name]) for name, writer in writers.items()}


def open_chips(cfg: Config, name: str, ref: str = "main") -> dict[str, xr.Dataset]:
    """The stored chips, keyed by group ("<family>/s<segment>", or "<family>" for single-chip units).

    Each family's chips come at the configured reflectance (``chips.reflectance``) when they
    have been built so, else as surface reflectance, with a warning. Every dataset carries
    ``attrs["family"]`` and ``attrs["reflectance"]``; ``attrs["transect_ids"]`` lists its
    transects (absent for single-chip units, which cover them all). ``ref`` applies to the
    surface-reflectance repository.
    """
    out: dict[str, xr.Dataset] = {}
    wanted = dict(cfg.chips.reflectance)
    toa = _open_repo(cfg, name, "toa") if "toa" in wanted.values() and chips_path(cfg, name, "toa").exists() else {}
    toa_families = {k.split("/")[0] for k in toa}
    for key, ds in toa.items():
        if wanted.get(key.split("/")[0]) == "toa":
            out[key] = ds
    missing = [f for f, level in wanted.items() if level == "surface" or f not in toa_families]
    if missing:
        if not chips_path(cfg, name).exists():
            if not out:
                raise FileNotFoundError(f"No chips named {name!r} in {chips_path(cfg, name).parent}. Build them first.")
        else:
            for key, ds in _open_repo(cfg, name, "surface", ref).items():
                family = key.split("/")[0]
                if family in missing:
                    if wanted.get(family) == "toa":
                        log.warning("%s: no top-of-atmosphere chips for %s; using surface reflectance", name, family)
                        wanted[family] = "surface (fallback)"
                    out[key] = ds
    return out


def stored_families(cfg: Config, name: str, level: str) -> set[str]:
    """The families that have chips at this reflectance level."""
    if not chips_path(cfg, name, level).exists():
        return set()
    return {key.split("/")[0] for key in _repo_keys(_session(cfg, name, level))}


def _session(cfg: Config, name: str, level: str, ref: str = "main"):
    import icechunk

    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(chips_path(cfg, name, level))))
    try:
        return repo.readonly_session(ref)
    except Exception:
        return repo.readonly_session(snapshot_id=ref)


def _repo_keys(session) -> list[str]:
    """The chip groups in a repository: "<family>/s<segment>", or "<family>" (older single-chip units)."""
    import zarr

    try:
        root = zarr.open_group(session.store, mode="r")
    except zarr.errors.GroupNotFoundError:      # created, nothing committed yet
        return []
    keys = []
    for family in sorted(root.group_keys()):
        children = sorted(root[family].group_keys())
        keys += [f"{family}/{c}" for c in children] if children else [family]
    return keys


def _open_repo(cfg: Config, name: str, level: str, ref: str = "main") -> dict[str, xr.Dataset]:
    session = _session(cfg, name, level, ref)
    out = {}
    for key in _repo_keys(session):
        ds = xr.open_zarr(session.store, group=key, consolidated=False)
        for cname, coord in ds.coords.items():
            if coord.dtype.kind in "TUS":
                ds = ds.assign_coords({cname: (coord.dims, np.array([str(v) for v in coord.values], dtype=object))})
        ds.attrs["snapshot_id"] = session.snapshot_id
        ds.attrs.setdefault("family", key.split("/")[0])
        ds.attrs.setdefault("reflectance", level)
        out[key] = ds
    return out
