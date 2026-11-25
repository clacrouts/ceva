"""Per-image waterlines with CoastSat's detection (Vos et al. 2019), on CEVA's chips.

For every image, CoastSat's own code (coastsat-package, GPL-3.0-only) does the detection:
its neural network labels each pixel sand, white water, water or other; Otsu's method
separates the MNDWI of the sand pixels from that of the water pixels inside a buffer
around a reference shoreline, and that MNDWI contour is the waterline
(``find_wl_contours2``, or ``find_wl_contours1`` on MNDWI alone when the buffer holds
fewer than 50 sand pixels). Short contours and points near clouds or missing data are
dropped, and the waterline is intersected with each transect as CoastSat does
(``compute_intersection_QC``).

What CEVA provides instead of CoastSat's Google Earth Engine workflow:

* the imagery: read from STAC catalogues onto CoastSat's grids (Landsat 15 m, Sentinel-2
  10 m), as top-of-atmosphere reflectance like CoastSat's own (Level-1: Sentinel-2 L1C
  from the Copernicus Data Space; Landsat Level-1 from the USGS, given the download role) or as surface
  reflectance (Level-2; see ``chips.reflectance``); Landsat is resampled, not pan-sharpened;
* the cloud mask: CoastSat's rules applied to the Level-2 quality bands of the same
  acquisition (the Landsat QA_PIXEL flags CoastSat uses; for Sentinel-2 the scene
  classification classes closest to CoastSat's QA60 and s2cloudless mask), with
  CoastSat's removal of small cloud patches;
* reflectance floored at zero: surface reflectance can be slightly negative over dark
  water, which top-of-atmosphere reflectance, the classifier's training data, never is;
* the reference shoreline: drawn automatically through that year's annual (composite)
  shoreline on the transects, instead of digitised by hand once, and by default moved with
  the tide of each image (``perimage.reference: tide_stage``), so that on flat macrotidal
  beaches the search buffer stays on the waterline;
* the water-level correction, done afterwards by the pipeline as for the other engine.
"""
from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

from ceva.chips import ChipGrid
from ceva.config import Config
from ceva.cube import _no_data

log = logging.getLogger(__name__)

MIN_CLEAR_PIXELS = 100   # fewer usable pixels than this: nothing to classify


@dataclass
class Transects:
    """Transects in the chip projection, each from its landward end (CoastSat's convention)."""

    transect_id: np.ndarray
    origin: np.ndarray       # [n, 2] landward end
    end: np.ndarray          # [n, 2] seaward end

    @property
    def direction(self) -> np.ndarray:
        d = self.end - self.origin
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def as_coastsat(self) -> dict[str, np.ndarray]:
        return {str(t): np.array([o, e]) for t, o, e in zip(self.transect_id, self.origin, self.end, strict=True)}


def oriented_transects(transect_id, lon_start, lat_start, lon_end, lat_end, seaward_sign, epsg: int) -> Transects:
    """Transects with a known seaward direction, projected, landward end first."""
    to = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    start = np.column_stack(to.transform(np.asarray(lon_start), np.asarray(lat_start)))
    end = np.column_stack(to.transform(np.asarray(lon_end), np.asarray(lat_end)))
    sign = np.asarray(seaward_sign)
    keep = sign != 0
    origin = np.where((sign == 1)[:, None], start, end)[keep]
    seaward = np.where((sign == 1)[:, None], end, start)[keep]
    return Transects(np.asarray(transect_id, dtype=object)[keep], origin, seaward)


def _alongshore_order(t: Transects) -> np.ndarray:
    """Order of the transects along the coast: by position on the main axis of their origins."""
    if len(t.transect_id) < 2:
        return np.arange(len(t.transect_id))
    centred = t.origin - t.origin.mean(axis=0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    return np.argsort(centred @ axis)


def _by_year(t: Transects, annual: pd.DataFrame, years: np.ndarray, column: str = "x") -> pd.DataFrame:
    """``column`` of the annual table [transect, year] for these transects and years.

    Years without a value on a transect take the nearest years' (linear in between).
    """
    years = np.unique(np.asarray(years, dtype=int))
    all_years = np.arange(min(years.min(), annual["year"].min()), max(years.max(), annual["year"].max()) + 1)
    table = (annual.pivot_table(index="transect_id", columns="year", values=column, aggfunc="first")
             .reindex(index=t.transect_id, columns=all_years))
    return table.T.interpolate(limit_direction="both").T


def reference_line(t: Transects, x: np.ndarray, extend_m: float) -> np.ndarray | None:
    """The line through the point at distance ``x`` on each transect, in alongshore order.

    It is extended by ``extend_m`` beyond the end transects, so that contours long enough
    for CoastSat's length filter fit inside the buffer. None if no transect has a position.
    """
    order = _alongshore_order(t)
    x = np.asarray(x, dtype=float)[order]
    ok = np.isfinite(x)
    if not ok.any():
        return None
    pts = (t.origin[order] + x[:, None] * t.direction[order])[ok]
    if len(pts) == 1:
        normal = t.direction[order][ok][0]
        along = np.array([-normal[1], normal[0]])
        return np.array([pts[0] - extend_m * along, pts[0] + extend_m * along])
    first = pts[0] - pts[1]
    last = pts[-1] - pts[-2]
    return np.vstack([pts[0] + extend_m * first / np.linalg.norm(first), pts,
                      pts[-1] + extend_m * last / np.linalg.norm(last)])


def reference_lines(t: Transects, annual: pd.DataFrame, years: np.ndarray, extend_m: float) -> dict[int, np.ndarray]:
    """A reference shoreline per year: through that year's annual position on each transect.

    ``annual`` has transect_id, year and x (distance from the landward end).
    """
    filled = _by_year(t, annual, years)
    out: dict[int, np.ndarray] = {}
    for year in np.unique(np.asarray(years, dtype=int)):
        line = reference_line(t, filled[year].to_numpy(dtype=float), extend_m)
        if line is not None:
            out[int(year)] = line
    return out


def tide_following_lines(t: Transects, annual: pd.DataFrame, curves: pd.DataFrame, times: pd.DatetimeIndex,
                         tide: np.ndarray, extend_m: float) -> dict[pd.Timestamp, np.ndarray]:
    """A reference shoreline per image, moved with the tide (see ``ceva.engines.tide_stage_curves``).

    ``tide`` is [time, transect] in the order of ``t``. On each transect the line sits at that
    year's annual shoreline plus the move the tide-stage curve predicts between the composite's
    mean tide and the tide of the image.
    """
    from ceva.engines import stage_shift

    years = times.year.to_numpy()
    x_year, tide_year = _by_year(t, annual, years, "x"), _by_year(t, annual, years, "tide_mean")
    x = x_year[years].to_numpy(dtype=float).T                       # [time, transect]
    reference_tide = tide_year[years].to_numpy(dtype=float).T
    x = x + stage_shift(curves, t.transect_id, tide, reference_tide)
    out: dict[pd.Timestamp, np.ndarray] = {}
    for i, when in enumerate(times):
        line = reference_line(t, x[i], extend_m)
        if line is not None:
            out[pd.Timestamp(when)] = line
    return out


def _references_for(references: dict, times: np.ndarray) -> dict:
    """The part of ``references`` some images need: all of it when keyed by year, else theirs."""
    if not references or all(isinstance(k, int | np.integer) for k in references):
        return references
    wanted = (pd.Timestamp(w) for w in times)
    return {k: references[k] for k in wanted if k in references}


def satname(platform: str) -> str:
    """CoastSat's mission code for a CEVA platform name."""
    if platform.startswith("sentinel-2"):
        return "S2"
    number = platform.rsplit("-", 1)[-1]
    return {"4": "L5", "5": "L5", "7": "L7", "8": "L8", "9": "L9"}.get(number, "L8")


def coastsat_settings(cfg: Config, epsg: int) -> dict:
    c = cfg.coastsat
    return {
        "output_epsg": epsg, "max_dist_ref": c.max_dist_ref_m, "min_length_sl": c.min_length_sl_m,
        "dist_clouds": c.dist_clouds_m, "cloud_thresh": c.cloud_thresh, "min_beach_area": c.min_beach_area_m2,
        "along_dist": c.along_dist_m, "min_points": c.min_points, "max_std": c.max_std_m,
        "max_range": c.max_range_m, "min_chainage": c.min_chainage_m, "multiple_inter": c.multiple_inter,
        "prc_multiple": c.prc_multiple,
    }


def cloud_mask(qa: np.ndarray, family: str, cfg: Config) -> np.ndarray:
    """CoastSat's cloud mask on a Level-2 quality band (see the module notes)."""
    from scipy import ndimage
    from skimage import morphology

    c = cfg.coastsat
    if family == "landsat":
        cloud = (qa & sum(1 << bit for bit in c.landsat_cloud_bits)) != 0
    else:
        cloud = np.isin(qa, c.s2_cloud_classes)
    if c.min_cloud_pixels and cloud.any() and (~cloud).any():
        cloud = morphology.remove_small_objects(cloud, max_size=c.min_cloud_pixels - 1, connectivity=1)
    if c.cloud_mask_issue[family] and cloud.any():
        # CoastSat's cloud_mask_issue: keep only clouds wider than 6 pixels and larger than 100
        cloud = ndimage.binary_opening(cloud, structure=np.ones((6, 6), dtype=bool))
        cloud = morphology.remove_small_objects(cloud, max_size=99, connectivity=1)
    return cloud


def too_cloudy(qa: np.ndarray, family: str, cfg: Config, margin: float = 0.0) -> bool:
    """Whether CoastSat would skip this chip as too cloudy, judged from its quality band alone.

    The same test as the detection (``_detect_images``), except that the detection's no-data also
    counts pixels missing from the reflectance bands, which the quality band cannot see; ``margin``
    (on the cloud fraction) keeps the chips near the limit. Chip reading uses it to leave out
    acquisitions before their bands are downloaded.
    """
    from coastsat import SDS_shoreline

    nodata = _no_data(qa, family)
    cloud = cloud_mask(qa, family, cfg) | nodata
    if (~cloud).sum() < MIN_CLEAR_PIXELS:
        return True
    cover_all, cover, _ = SDS_shoreline.compute_cloud_metrics(cloud, nodata)
    return bool(cover_all > 0.99 or cover > cfg.coastsat.cloud_thresh + margin)


def shoreline_buffer(im_shape, georef, image_epsg, pixel_size, settings) -> np.ndarray:
    """CoastSat's ``create_shoreline_buffer``: the same pixels, about 20 times faster.

    CoastSat densifies the reference line to half a pixel, so once rounded consecutive points are
    at most one pixel apart and the line drawn between them is just those points (it draws each
    step with ``skimage.draw.line``, which is kept for longer steps); and dilating by a disk of
    radius r keeps the pixels within r of the line, which a distance transform finds directly
    instead of a (2r+1)-wide sliding maximum.
    """
    from coastsat import SDS_shoreline, SDS_tools
    from scipy import ndimage
    from skimage import draw

    if "reference_shoreline" not in settings:
        return np.ones(im_shape, dtype=bool)
    parts = settings.get("reference_shoreline_parts") or [settings["reference_shoreline"]]
    spacing = settings.get("ref_sl_densify_spacing", pixel_size / 2)
    line = np.zeros(im_shape, dtype=bool)
    for part in parts:
        ref = np.asarray(part)[:, :2]
        if len(ref) < 2:
            continue
        ref = SDS_tools.convert_epsg(ref, settings["output_epsg"], image_epsg)[:, :2]
        ref = SDS_shoreline.densify_line_coords(ref, spacing)
        pix = np.round(SDS_tools.convert_world2pix(ref, georef)).astype(int)
        a, b = pix[:-1], pix[1:]
        step = np.abs(b - a).max(axis=1)
        rows, cols = [a[step <= 1, 1], b[step <= 1, 1]], [a[step <= 1, 0], b[step <= 1, 0]]
        for j in np.flatnonzero(step > 1):
            r, c = draw.line(a[j, 1], a[j, 0], b[j, 1], b[j, 0])
            rows.append(r)
            cols.append(c)
        rr, cc = np.concatenate(rows), np.concatenate(cols)
        inside = (rr >= 0) & (rr < im_shape[0]) & (cc >= 0) & (cc < im_shape[1])
        line[rr[inside], cc[inside]] = True
    if not line.any():
        return line
    radius = int(np.ceil(settings["max_dist_ref"] / pixel_size))
    return ndimage.distance_transform_edt(~line) <= radius


def _far_from(points: np.ndarray, others: np.ndarray, distance: float) -> np.ndarray:
    """True for the points with no point of ``others`` closer than ``distance``.

    The nearest point comes from a k-d tree; within rounding of the limit, CoastSat's own test
    (``np.linalg.norm`` against all of them) decides.
    """
    from scipy.spatial import cKDTree

    if len(others) == 0:
        return np.ones(len(points), dtype=bool)
    nearest, _ = cKDTree(others).query(points, k=1)
    keep = nearest >= distance * (1 + 1e-9)
    unsure = ~keep & (nearest >= distance * (1 - 1e-9))
    for k in np.flatnonzero(unsure):
        keep[k] = not np.any(np.linalg.norm(points[k, :] - others, axis=1) < distance)
    return keep


def process_shoreline(contours, cloud_mask, im_nodata, georef, image_epsg, settings, **kwargs):
    """CoastSat's ``process_shoreline``: the same points, much faster.

    CoastSat measures the distance from every shoreline point to every cloud and no-data pixel,
    one array per point (most of the detection time when a chip has a no-data edge); here each
    point's nearest such pixel comes from a k-d tree (``_far_from``), and the arrays are joined
    once instead of grown point list by point list.
    """
    from coastsat import SDS_tools
    from shapely.geometry import LineString

    contours_world = SDS_tools.convert_pix2world(contours, georef)
    contours_epsg = SDS_tools.convert_epsg(contours_world, image_epsg, settings["output_epsg"])
    kept = [wl for wl in contours_epsg if LineString(wl[:, :2]).length >= settings["min_length_sl"]]
    x = np.concatenate([np.array([])] + [wl[:, 0] for wl in kept])
    y = np.concatenate([np.array([])] + [wl[:, 1] for wl in kept])
    shoreline = np.transpose(np.array([x, y]))
    for mask, distance in ((cloud_mask, settings["dist_clouds"]), (im_nodata, 30)):
        if len(shoreline) == 0:
            return shoreline
        if np.sum(mask) > 0:
            pixels = SDS_tools.convert_epsg(SDS_tools.convert_pix2world(np.argwhere(mask), georef),
                                            image_epsg, settings["output_epsg"])[:, :-1]
            shoreline = shoreline[_far_from(shoreline, pixels, distance)]
    return shoreline


def calculate_features(im_ms, cloud_mask, im_bool):
    """CoastSat's ``calculate_features``: the same matrix, assembled once instead of a column at a time."""
    from coastsat import SDS_tools

    indices = [SDS_tools.nd_index(im_ms[:, :, a], im_ms[:, :, b], cloud_mask)
               for a, b in ((3, 1), (4, 1), (3, 2), (4, 3), (0, 2))]     # NIR-G, SWIR-G, NIR-R, SWIR-NIR, B-R
    columns = [im_ms[im_bool, k] for k in range(im_ms.shape[2])] + [index[im_bool] for index in indices]
    columns += [SDS_tools.image_std(im_ms[:, :, k], 1)[im_bool] for k in range(im_ms.shape[2])]
    columns += [SDS_tools.image_std(index, 1)[im_bool] for index in indices]
    return np.stack(columns, axis=1)


@contextmanager
def faster_coastsat():
    """CoastSat's detection with the functions above in place of its own (same results)."""
    from coastsat import SDS_shoreline

    replaced = {"process_shoreline": process_shoreline, "calculate_features": calculate_features}
    original = {name: getattr(SDS_shoreline, name) for name in replaced}
    for name, function in replaced.items():
        setattr(SDS_shoreline, name, function)
    try:
        yield
    finally:
        for name, function in original.items():
            setattr(SDS_shoreline, name, function)


_CLASSIFIERS: dict = {}     # per process: loaded once, not once per block of images


def _classifier(sat: str, sand_color: str, cache: dict = _CLASSIFIERS):
    if (sat, sand_color) not in cache:
        import sklearn
        from coastsat import SDS_shoreline
        from coastsat.classification import models

        with warnings.catch_warnings():   # pickled with an older scikit-learn; predictions are unaffected
            warnings.simplefilter("ignore")
            cache[(sat, sand_color)] = SDS_shoreline.load_classifier(
                sat, sand_color, os.path.dirname(models.__file__), sklearn.__version__)
    return cache[(sat, sand_color)]


def _detect_images(cfg: Config, grid: ChipGrid, family: str, bands: np.ndarray, qa: np.ndarray,
                   times: np.ndarray, platforms: np.ndarray, references: dict[int, np.ndarray]) -> list[dict]:
    """CoastSat's waterline on a stack of images ([image, band, y, x]); one result row per image."""
    from coastsat import SDS_shoreline

    settings = coastsat_settings(cfg, grid.epsg)
    min_beach_pixels = np.ceil(cfg.coastsat.min_beach_area_m2 / grid.pixel_m**2)
    quiet = logging.getLogger("ceva.coastsat")
    quiet.setLevel(logging.WARNING)
    rows = []
    for i in range(len(times)):
        platform = str(platforms[i])
        sat = satname(platform)
        when = pd.Timestamp(times[i])
        row = {"time": when, "platform": platform, "family": family, "shoreline": np.empty((0, 2)),
               "threshold": np.nan, "method": "skipped", "cloud_cover": np.nan}
        rows.append(row)
        im_ms = np.moveaxis(bands[i], 0, -1).astype("float64")
        nodata = ~np.isfinite(im_ms).all(axis=-1) | _no_data(qa[i], family)
        cloud = cloud_mask(qa[i], family, cfg) | nodata
        im_ms = np.clip(np.nan_to_num(im_ms, nan=0.0), 0.0, None)
        if (~cloud).sum() < MIN_CLEAR_PIXELS:
            continue
        cover_all, cover, cloud_adv = SDS_shoreline.compute_cloud_metrics(cloud, nodata)
        row["cloud_cover"] = float(cover)
        if cover_all > 0.99 or cover > settings["cloud_thresh"]:
            continue
        reference = references.get(when)
        if reference is None:
            reference = references.get(when.year)
        if reference is None:
            continue
        settings["reference_shoreline"] = reference
        settings["min_length_sl"] = 200 if sat == "L7" else cfg.coastsat.min_length_sl_m
        buffer = shoreline_buffer(cloud.shape, grid.georef, grid.epsg, grid.pixel_m, settings)
        # CoastSat balances its sand and water samples with numpy's global random generator; seeding it
        # from the image time makes each detection reproducible, whatever the order or the process.
        np.random.seed(int(when.value // 1_000_000_000) % (2**32))
        try:
            with faster_coastsat():
                _, labels = SDS_shoreline.classify_image_NN(im_ms, cloud, min_beach_pixels,
                                                            _classifier(sat, cfg.coastsat.sand_color))
                row["method"] = "sand/water" if labels[buffer, 0].sum() >= 50 else "mndwi only"
                shoreline, threshold = SDS_shoreline.find_shoreline_optical(
                    im_ms, labels, buffer, cloud, cloud_adv, nodata, grid.georef, grid.epsg, sat,
                    f"{when:%Y-%m-%d}", settings, logger=quiet)
        except Exception as exc:   # CoastSat skips an image it cannot map; so does CEVA
            log.debug("no shoreline on %s %s: %s", platform, when, exc)
            row["method"] = "failed"
            continue
        row["shoreline"] = np.asarray(shoreline, dtype=float).reshape(-1, 2)
        row["threshold"] = float(threshold)
    return rows


_POOL = None
_POOL_WORKERS = 0


def detection_pool(workers: int):
    """The pool of detection processes, shared by every segment and sensor of a run.

    Starting a process costs about 10 s (importing CoastSat and its dependencies), and the
    radar model takes a few more seconds to load in each: a pool per segment and sensor spent
    minutes per region on that. Spawned, not forked: the parent has GDAL reader threads, and
    forking them can deadlock. A pool broken by a crashed process, or shut down, is replaced.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    global _POOL, _POOL_WORKERS
    stale = _POOL is not None and (getattr(_POOL, "_broken", False) or getattr(_POOL, "_shutdown_thread", False))
    if _POOL is None or _POOL_WORKERS != workers or stale:
        if _POOL is not None:
            _POOL.shutdown(wait=False, cancel_futures=True)
        _POOL = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        _POOL_WORKERS = workers
    return _POOL


def _detect_worker(config: dict, attrs: dict, family: str, bands, qa, times, platforms, references) -> list[dict]:
    """Process-pool entry point: the configuration travels as plain data."""
    warnings.simplefilter("ignore")
    return _detect_images(Config.model_validate(config), ChipGrid.from_attrs(attrs), family, bands, qa,
                          times, platforms, references)


def detect_shorelines(cfg: Config, ds: xr.Dataset, references: dict,
                      workers: int | None = None, chunk: int = 24, block: int = 240) -> pd.DataFrame:
    """CoastSat's waterline on every chip of one family and segment, in parallel processes.

    ``references`` holds the reference shorelines, keyed by year or by image time (an image's
    own line wins over its year's). The chips are read ``block`` images at a time, as float32,
    so memory stays bounded however long the record and large the chip. Returns one row per
    acquisition: time, platform, family, the waterline points (``shoreline``, [n, 2] in the chip
    projection, empty if none), the MNDWI threshold, the method CoastSat used, and the cloud
    cover over the chip.
    """
    grid = ChipGrid.from_attrs(ds.attrs)
    family = str(ds.attrs.get("family") or ds["family"].values[0])
    times, platforms = ds["time"].values, ds["platform"].values.astype(str)
    workers = cfg.chips.detect_workers if workers is None else workers

    def blocks():
        for start in range(0, len(times), block):
            part = ds.isel(time=slice(start, start + block))
            bands = np.stack([part[b].values.astype("float32", copy=False) for b in cfg.chips.bands], axis=1)
            yield start, bands, part["qa"].values                   # [time, band, y, x], [time, y, x]

    rows: list[dict] = []
    if workers <= 1 or len(times) <= chunk:
        for start, bands, qa in blocks():
            n = len(qa)
            rows += _detect_images(cfg, grid, family, bands, qa, times[start:start + n],
                                   platforms[start:start + n], _references_for(references, times[start:start + n]))
        return pd.DataFrame(rows)
    config, attrs = cfg.model_dump(mode="json"), grid.to_attrs()
    pool = detection_pool(workers)
    for start, bands, qa in blocks():
        n = len(qa)
        futures = [pool.submit(_detect_worker, config, attrs, family, bands[i:i + chunk], qa[i:i + chunk],
                               times[start + i:start + min(i + chunk, n)],
                               platforms[start + i:start + min(i + chunk, n)],
                               _references_for(references, times[start + i:start + min(i + chunk, n)]))
                   for i in range(0, n, chunk)]
        rows += [row for f in futures for row in f.result()]
        del bands, qa, futures
    return pd.DataFrame(rows)


def intersect(cfg: Config, detections: pd.DataFrame, t: Transects, epsg: int) -> pd.DataFrame:
    """CoastSat's quality-controlled intersection of each waterline with each transect.

    Returns transect_id, time and x_obs (distance from the landward end of the transect).
    """
    from coastsat import SDS_transects

    if detections.empty:       # typed, so that it merges with the (empty) detections on time
        return pd.DataFrame({"transect_id": pd.Series(dtype=object), "time": pd.Series(dtype="datetime64[ns]"),
                             "x_obs": pd.Series(dtype=float)})
    output = {"shorelines": list(detections["shoreline"]), "dates": list(detections["time"])}
    cross = SDS_transects.compute_intersection_QC(output, t.as_coastsat(), coastsat_settings(cfg, epsg),
                                                  use_progress_bar=False)
    frames = [pd.DataFrame({"transect_id": key, "time": detections["time"].to_numpy(),
                            "x_obs": np.asarray(values, dtype=float)}) for key, values in cross.items()]
    out = pd.concat(frames, ignore_index=True)
    return out[np.isfinite(out["x_obs"])].reset_index(drop=True)


def coastsat_waterlines(cfg: Config, chips: dict[str, xr.Dataset], transects: pd.DataFrame, annual: pd.DataFrame,
                        tide_fn, segment_cache: Path | None = None,
                        curves: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observed waterline of every acquisition on every transect, with its tide height.

    ``transects`` has transect_id, lon_start, lat_start, lon_end, lat_end and seaward_sign.
    ``annual`` is the annual (composite) shoreline table that places the reference lines.
    ``tide_fn(times)`` returns the tide [time, transect] in the order of ``transects``.
    With tide-stage ``curves``, each image gets its own reference line, moved with its tide.
    Returns the waterlines (the same columns as the index engine, plus ``method``) and
    one diagnostic row per acquisition. With ``segment_cache``, each chip's result is saved
    there as soon as it is done and reused on the next call, so a long coast can resume.
    """
    follow_tide = curves is not None and not curves.empty
    tables, diagnostics = [], []
    for key, ds in chips.items():
        done = segment_cache / f"{key.replace('/', '_')}" if segment_cache is not None else None
        if done is not None and (done / "hits.parquet").exists():
            tables.append(pd.read_parquet(done / "hits.parquet"))
            diagnostics.append(pd.read_parquet(done / "detections.parquet"))
            log.info("%s: reusing the detections of an earlier run", key)
            continue
        grid = ChipGrid.from_attrs(ds.attrs)
        mine = transects
        if ds.attrs.get("transect_ids"):
            mine = transects[transects["transect_id"].isin(str(ds.attrs["transect_ids"]).split(","))]
        t = oriented_transects(mine["transect_id"], mine["lon_start"], mine["lat_start"],
                               mine["lon_end"], mine["lat_end"], mine["seaward_sign"], grid.epsg)
        if len(t.transect_id) == 0:
            continue
        chip_times = pd.DatetimeIndex(ds["time"].values)
        if follow_tide:
            columns = pd.Index(transects["transect_id"]).get_indexer(t.transect_id)
            refs = tide_following_lines(t, annual, curves, chip_times, tide_fn(chip_times)[:, columns],
                                        cfg.coastsat.reference_extend_m)
        else:
            refs = reference_lines(t, annual, chip_times.year, cfg.coastsat.reference_extend_m)
        log.info("%s: detecting shorelines on %d chips (%d transects)", key, ds.sizes["time"], len(t.transect_id))
        detections = detect_shorelines(cfg, ds, refs)
        diagnostic = detections.drop(columns="shoreline").assign(n_points=detections["shoreline"].map(len), chip=key)
        mapped = detections[detections["shoreline"].map(len) > 0].reset_index(drop=True)
        hits = intersect(cfg, mapped, t, grid.epsg).merge(
            mapped[["time", "platform", "family", "threshold", "method"]], on="time")
        if done is not None:
            done.mkdir(parents=True, exist_ok=True)
            diagnostic.to_parquet(done / "detections.parquet", index=False)
            hits.to_parquet(done / "hits.parquet", index=False)       # written last: marks the chip as done
        tables.append(hits)
        diagnostics.append(diagnostic)
    w = pd.concat(tables, ignore_index=True)
    times = pd.DatetimeIndex(sorted(w["time"].unique()))
    tide = tide_fn(times)                                   # [time, transect]
    t_idx = times.get_indexer(w["time"])
    n_idx = pd.Index(transects["transect_id"]).get_indexer(w["transect_id"])
    w["tide"] = tide[t_idx, n_idx]
    w = w.sort_values(["time", "transect_id"]).reset_index(drop=True)
    columns = ["transect_id", "time", "platform", "family", "x_obs", "tide", "threshold", "method"]
    return w[columns], pd.concat(diagnostics, ignore_index=True)
