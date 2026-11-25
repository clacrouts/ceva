"""Radar waterlines, and their alignment on the imagery (inter-mission calibration).

Detection, for each of the two methods:

* CoastSat solution: CoastSat's SAR land/water model (``coastsat.SDS_sar_model``, a U-Net on
  [VV, VH, VV-VH] in dB, trained on Sentinel-1 sigma nought), or its Otsu fallback on
  mean(VV, VH); the waterline is the contour of the water mask within the reference buffer,
  intersected with the transects as for the imagery. The chips hold gamma nought, so they are
  brought to sigma nought at a nominal incidence angle first.
* CEVA solution: mean(VV, VH) in dB, median-filtered against speckle, sampled along each
  transect like the imagery; the waterline is where it crosses an Otsu threshold near the
  expected shoreline.

Both search near the same expected waterline as the imagery: that year's composite
shoreline from the imagery, moved with the tide.

Alignment: a radar waterline does not sit where an optical one does. Breaking waves are
bright like land and wet sand is dark like water, and how much depends on the waves, the
tide and the radar's look direction. The imagery is the reference: each radar waterline is
compared with the median of the imagery positions within ``sar.calibration.window_days``, and
the difference is modelled as an offset per transect and orbit direction plus terms in wave
height and tide. Only imagery is used to fit it, never surveys, so the validation stays
independent. The fitted model is subtracted from every radar waterline, including those far
from any image (cloudy seasons), which is where radar adds coverage.
"""
from __future__ import annotations

import logging
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod, Transformer

from ceva.chips import ChipGrid
from ceva.config import Config
from ceva.engines import Profiles, flag_outliers, tide_correct
from ceva.engines_coastsat import (
    coastsat_settings,
    intersect,
    oriented_transects,
    shoreline_buffer,
    tide_following_lines,
)

log = logging.getLogger(__name__)

MODEL_DIR = "models/coastsat-sar"         # under the data directory
MIN_VALID_PIXELS = 100
SAR_READ_BLOCK = 64                      # radar images decoded at once when sampling profiles
_GEOD = Geod(ellps="WGS84")


# ----------------------------------------------------------------------------- the model

def model_path(cfg: Config) -> Path:
    """CoastSat's SAR model, downloaded once into the data directory and checked against its hash."""
    from coastsat.SDS_sar_model import SAR_MODEL_CONFIG as spec

    folder = cfg.data_dir / MODEL_DIR
    path = folder / spec["model_name"]
    if not path.is_file():
        import pooch

        folder.mkdir(parents=True, exist_ok=True)
        pooch.retrieve(url=spec["url"], known_hash=spec["sha256"], fname=spec["model_name"], path=str(folder),
                       progressbar=False)
    return path


@lru_cache(maxsize=2)
def _session(path: str, threads: int = 1):
    """An onnxruntime session with a bounded thread pool (one per detection process)."""
    import onnxruntime as ort
    from coastsat.SDS_sar_model import read_model_spec

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
    return session, read_model_spec(session)


# ------------------------------------------------------------------- CoastSat solution

def _detect_sar_images(cfg: Config, grid: ChipGrid, vv: np.ndarray, vh: np.ndarray, times: np.ndarray,
                       references: dict, model: str | None) -> list[dict]:
    """CoastSat's radar waterline on a stack of chips ([image, y, x] in dB, gamma nought)."""
    from coastsat import SDS_shoreline
    from coastsat.SDS_sar_model import segment_water

    settings = coastsat_settings(cfg, grid.epsg)
    min_pixels = int(np.ceil(cfg.coastsat.min_beach_area_m2 / grid.pixel_m**2))
    shift = np.float32(10.0 * np.log10(np.cos(np.radians(cfg.sar.sentinel1.incidence_deg))))
    rows = []
    for i in range(len(times)):
        when = pd.Timestamp(times[i])
        row = {"time": when, "shoreline": np.empty((0, 2)), "method": "skipped"}
        rows.append(row)
        valid = np.isfinite(vv[i]) & np.isfinite(vh[i])
        reference = references.get(when)
        if reference is None:
            reference = references.get(when.year)
        if valid.sum() < MIN_VALID_PIXELS or reference is None:
            continue
        settings["reference_shoreline"] = reference
        buffer = shoreline_buffer(valid.shape, grid.georef, grid.epsg, grid.pixel_m, settings)
        try:
            if model is not None:
                session, spec = _session(model)
                composite = np.stack([vv[i] + shift, vh[i] + shift, vv[i] - vh[i]], axis=-1)
                water, _ = segment_water(np.nan_to_num(composite), valid, session, spec, cfg.sar.water_threshold)
                row["method"] = "model"
            else:
                image = (vv[i] + vh[i]) / 2.0
                image = np.where(valid, image, np.nanmedian(image[valid]))
                water, _ = SDS_shoreline.otsu_threshold(image, min_pixels)
                row["method"] = "otsu"
            water = water & valid
            line = SDS_shoreline.find_shoreline_SAR(water, buffer & valid, grid.georef, grid.epsg, settings)
        except Exception as exc:     # CoastSat skips an image it cannot map; so does CEVA
            log.debug("no radar shoreline on %s: %s", when, exc)
            row["method"] = "failed"
            continue
        row["shoreline"] = np.asarray(line, dtype=float).reshape(-1, 2)
    return rows


def _sar_worker(config: dict, attrs: dict, vv, vh, times, references, model) -> list[dict]:
    warnings.simplefilter("ignore")
    return _detect_sar_images(Config.model_validate(config), ChipGrid.from_attrs(attrs), vv, vh, times,
                              references, model)


def detect_sar_shorelines(cfg: Config, ds: xr.Dataset, references: dict, model: str | None,
                          workers: int | None = None, chunk: int = 32) -> pd.DataFrame:
    """CoastSat's radar waterline on every chip of one segment, in parallel processes."""
    from ceva.engines_coastsat import _references_for, detection_pool

    grid = ChipGrid.from_attrs(ds.attrs)
    times = ds["time"].values
    vv, vh = ds["vv"].values.astype("float32"), ds["vh"].values.astype("float32")
    workers = cfg.chips.detect_workers if workers is None else workers
    if workers <= 1 or len(times) <= chunk:
        return pd.DataFrame(_detect_sar_images(cfg, grid, vv, vh, times, _references_for(references, times), model))
    config, attrs = cfg.model_dump(mode="json"), grid.to_attrs()
    pool = detection_pool(workers)
    futures = [pool.submit(_sar_worker, config, attrs, vv[i:i + chunk], vh[i:i + chunk], times[i:i + chunk],
                           _references_for(references, times[i:i + chunk]), model)
               for i in range(0, len(times), chunk)]
    rows = [row for f in futures for row in f.result()]
    return pd.DataFrame(rows)


def coastsat_sar_waterlines(cfg: Config, chips: dict[str, xr.Dataset], transects: pd.DataFrame,
                            annual: pd.DataFrame, curves: pd.DataFrame, tide_fn) -> pd.DataFrame:
    """Radar waterlines of the CoastSat solution: transect_id, time, platform, family, orbit, x_obs, tide."""
    model = str(model_path(cfg)) if cfg.sar.detection == "model" else None
    tables = []
    for key, ds in chips.items():
        grid = ChipGrid.from_attrs(ds.attrs)
        mine = transects[transects["transect_id"].isin(str(ds.attrs.get("transect_ids", "")).split(","))] \
            if ds.attrs.get("transect_ids") else transects
        t = oriented_transects(mine["transect_id"], mine["lon_start"], mine["lat_start"], mine["lon_end"],
                               mine["lat_end"], mine["seaward_sign"], grid.epsg)
        if len(t.transect_id) == 0:
            continue
        times = pd.DatetimeIndex(ds["time"].values)
        columns = pd.Index(transects["transect_id"]).get_indexer(t.transect_id)
        refs = tide_following_lines(t, annual, curves, times, tide_fn(times)[:, columns],
                                    cfg.coastsat.reference_extend_m)
        log.info("%s: radar waterlines on %d chips (%d transects)", key, len(times), len(t.transect_id))
        detections = detect_sar_shorelines(cfg, ds, refs, model)
        mapped = detections[detections["shoreline"].map(len) > 0].reset_index(drop=True)
        if mapped.empty:
            continue
        meta = pd.DataFrame({"time": times, "platform": ds["platform"].values.astype(str),
                             "orbit_state": ds["orbit_state"].values.astype(str),
                             "relative_orbit": ds["relative_orbit"].values.astype(int)})
        hits = intersect(cfg, mapped, t, grid.epsg).merge(mapped[["time", "method"]], on="time").merge(meta, on="time")
        tables.append(hits)
    return _with_tide(tables, transects, tide_fn, cfg)


# ----------------------------------------------------------------------- CEVA solution

def _sampled_pixels(row: np.ndarray, col: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The pixels that bilinear sampling at (row, col) may read, inside the image.

    Those around each point, and one more on each side: at the last row or column, scipy's
    interpolation leans on the pixel before (with no weight, but a NaN there still counts).
    """
    r0 = np.floor(row.ravel()).astype(int)
    c0 = np.floor(col.ravel()).astype(int)
    step = np.array([-1, 0, 1, 2])
    rr = (r0[:, None, None] + step[None, :, None]).repeat(len(step), axis=2).ravel()
    cc = (c0[:, None, None] + step[None, None, :]).repeat(len(step), axis=1).ravel()
    inside = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    return np.divmod(np.unique(rr[inside] * shape[1] + cc[inside]), shape[1])


def _reflect(index: np.ndarray, n: int) -> np.ndarray:
    """Indices beyond the edges mirrored back inside (scipy.ndimage's "reflect" mode)."""
    index = np.where(index < 0, -index - 1, index)
    return np.where(index >= n, 2 * n - index - 1, index)


def _smoothed_at(vv: np.ndarray, vh: np.ndarray, pixels: tuple[np.ndarray, np.ndarray], size: int) -> np.ndarray:
    """Mean(VV, VH) in dB, median-filtered over ``size`` x ``size`` pixels, at ``pixels`` only (NaN elsewhere).

    The values are those of filtering the whole image (``scipy.ndimage.median_filter``, edges
    reflected, gaps first filled with the image's median, gaps NaN again afterwards), for a
    fraction of the work: sampling along the transects reads only these pixels.
    """
    from scipy.ndimage import median_filter

    mean_db = (vv + vh) / 2.0
    r, c = pixels
    out = np.full(mean_db.shape, np.nan, dtype=mean_db.dtype)
    centre_valid = np.isfinite(mean_db[r, c])
    if size > 1 and size % 2 == 0:        # even windows are not centred: filter the whole image
        valid = np.isfinite(mean_db)
        filled = np.where(valid, mean_db, np.nanmedian(mean_db) if valid.any() else 0.0)
        out[r, c] = np.where(centre_valid, median_filter(filled, size=size)[r, c], np.nan)
        return out
    if size <= 1:
        out[r, c] = np.where(centre_valid, mean_db[r, c], np.nan)
        return out
    offsets = np.arange(-(size // 2), size // 2 + 1)
    rows = _reflect(r[:, None, None] + offsets[None, :, None], mean_db.shape[0])
    cols = _reflect(c[:, None, None] + offsets[None, None, :], mean_db.shape[1])
    window = mean_db[rows, cols].reshape(len(r), -1)
    gaps = ~np.isfinite(window)
    if gaps.any():
        valid = np.isfinite(mean_db)
        window = np.where(gaps, np.nanmedian(mean_db) if valid.any() else 0.0, window)
    out[r, c] = np.where(centre_valid, np.median(window, axis=1), np.nan)
    return out


def sar_profiles(cfg: Config, chips: dict[str, xr.Dataset], cube: xr.Dataset, seaward_sign: np.ndarray,
                 tide_fn) -> tuple[Profiles, pd.DataFrame]:
    """Mean(VV, VH) backscatter (dB) along each transect at the imagery's samples, oriented land to sea.

    Each image is median-filtered (``sar.smoothing_px``) against speckle before it is sampled.
    The profiles travel in a ``Profiles`` (its ``index`` holds decibels here).
    """
    from scipy.ndimage import map_coordinates

    ids = cube["transect_id"].values.astype(str)
    distance = cube["distance"].values.astype("float64")
    length = cube["length"].values
    lon0, lat0 = cube["lon_start"].values, cube["lat_start"].values
    lon1, lat1 = cube["lon_end"].values, cube["lat_end"].values
    azimuth, _, _ = _GEOD.inv(lon0, lat0, lon1, lat1)
    n, k = len(ids), len(distance)
    lon, lat, _ = _GEOD.fwd(np.repeat(lon0, k), np.repeat(lat0, k), np.repeat(azimuth, k), np.tile(distance, n))
    lon, lat = lon.reshape(n, k), lat.reshape(n, k)
    beyond = distance[None, :] > length[:, None] + 1e-6

    times = pd.DatetimeIndex(sorted({t for ds in chips.values() for t in pd.DatetimeIndex(ds["time"].values)}))
    db = np.full((len(times), n, k), np.nan, dtype="float32")
    meta = {}
    size = cfg.sar.smoothing_px
    for ds in chips.values():
        grid = ChipGrid.from_attrs(ds.attrs)
        mine = np.isin(ids, str(ds.attrs.get("transect_ids", "")).split(",")) if ds.attrs.get("transect_ids") \
            else np.ones(n, bool)
        if not mine.any():
            continue
        x, y = Transformer.from_crs(4326, grid.epsg, always_xy=True).transform(lon[mine], lat[mine])
        col = (x - grid.x0) / grid.pixel_m - 0.5
        row = (grid.y0 - y) / grid.pixel_m - 0.5
        coords = [row.ravel(), col.ravel()]
        pixels = _sampled_pixels(row, col, (grid.height, grid.width))
        t_idx = times.get_indexer(pd.DatetimeIndex(ds["time"].values))
        for b0 in range(0, ds.sizes["time"], SAR_READ_BLOCK):       # bounded memory, however long the record
            part = ds.isel(time=slice(b0, b0 + SAR_READ_BLOCK))
            vv, vh = part["vv"].values, part["vh"].values
            for j in range(len(vv)):
                smooth = _smoothed_at(vv[j], vh[j], pixels, size)
                sampled = map_coordinates(smooth, coords, order=1, mode="constant", cval=np.nan)
                db[t_idx[b0 + j], mine] = sampled.reshape(row.shape)
        for t, p, o, r in zip(pd.DatetimeIndex(ds["time"].values), ds["platform"].values.astype(str),
                              ds["orbit_state"].values.astype(str), ds["relative_orbit"].values.astype(int),
                              strict=True):
            meta.setdefault(t, (p, o, r))
    db[:, beyond] = np.nan
    for i in np.flatnonzero(seaward_sign == -1):
        n_valid = int(np.searchsorted(distance, length[i] + 1e-6))
        db[:, i, :n_valid] = db[:, i, :n_valid][:, ::-1]
    db[:, seaward_sign == 0, :] = np.nan
    profiles = Profiles(index=db, distance=distance, time=times,
                        family=np.array(["sentinel1"] * len(times), dtype=object),
                        platform=np.array([meta[t][0] for t in times], dtype=object),
                        transect_id=ids.astype(object), seaward_sign=seaward_sign, tide=tide_fn(times))
    orbits = pd.DataFrame({"time": times, "orbit_state": [meta[t][1] for t in times],
                           "relative_orbit": [meta[t][2] for t in times]})
    return profiles, orbits


def _resample(profile: np.ndarray, distance: np.ndarray, fine: np.ndarray) -> np.ndarray:
    """Linear interpolation of [time, distance] rows onto ``fine``; NaN stays NaN."""
    pos = (fine - distance[0]) / (distance[1] - distance[0])
    k0 = np.clip(np.floor(pos).astype(int), 0, len(distance) - 2)
    w = (pos - k0).astype("float32")
    return profile[:, k0] * (1 - w) + profile[:, k0 + 1] * w


def _gather(fine_rows: np.ndarray, start: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """fine_rows[t, start[t] + offsets] (NaN outside), for start [T] and offsets [..., K]."""
    idx = start.reshape((-1,) + (1,) * offsets.ndim) + offsets[None]
    inside = (idx >= 0) & (idx < fine_rows.shape[1])
    rows = np.arange(len(start)).reshape((-1,) + (1,) * offsets.ndim)
    out = fine_rows[rows, np.clip(idx, 0, fine_rows.shape[1] - 1)]
    return np.where(inside, out, np.nan)


def match_templates(db: np.ndarray, distance: np.ndarray, reference: np.ndarray, anchor: np.ndarray,
                    groups: np.ndarray, c, block: int = 400) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Radar waterlines by matching each profile to a template learned at the imagery's waterlines.

    ``db`` [time, transect, distance]; ``reference`` [time, transect] the expected waterline
    that the search is centred on; ``anchor`` [time, transect] where the imagery puts the
    waterline at each radar time (NaN where no imagery is near); ``groups`` [time] the template
    each image uses (orbit direction and wave class). For every transect and group, the template
    is the median of the anchored profiles (each minus its mean); then every profile is slid
    within ``c.search_m`` of the reference and the shift with the highest correlation wins.
    Returns the waterline and the correlation [time, transect], and the templates.
    """
    step = c.step_m
    fine = np.arange(distance[0], distance[-1] + step / 2, step)
    k_off = np.round(np.arange(-c.half_width_m, c.half_width_m + step / 2, step) / step).astype(int)
    s_off = np.round(np.arange(-c.search_m, c.search_m + step / 2, step) / step).astype(int)
    n_time, n_transect = reference.shape
    x_obs = np.full((n_time, n_transect), np.nan)
    score = np.full((n_time, n_transect), np.nan)
    templates = []
    for i in range(n_transect):
        rows = _resample(db[:, i, :].astype("float32"), distance, fine)
        tpl = np.full((n_time, len(k_off)), np.nan, dtype="float32")
        for g in np.unique(groups):
            learn = np.flatnonzero((groups == g) & np.isfinite(anchor[:, i]))
            if len(learn) < c.min_learn:
                continue
            start = np.round((anchor[learn, i] - fine[0]) / step).astype(int)
            stack = _gather(rows[learn], start, k_off)
            stack -= np.nanmean(stack, axis=1, keepdims=True)
            enough = np.isfinite(stack).sum(axis=0) >= max(c.min_learn // 2, 1)
            template = np.where(enough, np.nanmedian(stack, axis=0), np.nan)
            tpl[groups == g] = template
            templates.append({"transect": i, "group": g, "n": len(learn), "template": template})
        todo = np.flatnonzero(np.isfinite(reference[:, i]) & np.isfinite(tpl).any(axis=1))
        for b0 in range(0, len(todo), block):
            t = todo[b0:b0 + block]
            start = np.round((reference[t, i] - fine[0]) / step).astype(int)
            seg = _gather(rows[t], start, s_off[:, None] + k_off[None, :])           # [T, S, K]
            tp = tpl[t][:, None, :]
            m = np.isfinite(seg) & np.isfinite(tp)
            n = m.sum(-1)
            a, b = np.where(m, seg, 0.0), np.where(m, tp, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                sa, sb = a.sum(-1), b.sum(-1)
                cov = (a * b).sum(-1) - sa * sb / n
                va = (a * a).sum(-1) - sa * sa / n
                vb = (b * b).sum(-1) - sb * sb / n
                r = cov / np.sqrt(va * vb)
            r[(n < c.min_valid * len(k_off)) | ~np.isfinite(r)] = -np.inf
            best = np.argmax(r, axis=1)
            r_best = r[np.arange(len(t)), best]
            # sub-step: the vertex of a parabola through the best shift and its neighbours
            lo, hi = np.clip(best - 1, 0, len(s_off) - 1), np.clip(best + 1, 0, len(s_off) - 1)
            r0, r1, r2 = r[np.arange(len(t)), lo], r_best, r[np.arange(len(t)), hi]
            with np.errstate(invalid="ignore", divide="ignore"):
                vertex = np.where((lo < best) & (hi > best) & np.isfinite(r0) & np.isfinite(r2),
                                  0.5 * (r0 - r2) / (r0 - 2 * r1 + r2), 0.0)
            vertex = np.clip(np.nan_to_num(vertex), -0.5, 0.5)
            ok = np.isfinite(r_best)
            x_obs[t[ok], i] = fine[0] + (start[ok] + s_off[best[ok]] + vertex[ok]) * step
            score[t[ok], i] = r_best[ok]
    return x_obs, score, templates


def ceva_sar_waterlines(cfg: Config, chips: dict[str, xr.Dataset], cube: xr.Dataset, seaward_sign: np.ndarray,
                        annual: pd.DataFrame, curves: pd.DataFrame, tide_fn, imagery: pd.DataFrame,
                        slopes: pd.Series, wave_fn) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Radar waterlines of the CEVA solution: templates learned at the imagery's waterlines, then matched.

    ``imagery`` is the corrected imagery waterlines (``x_corr``, ``use``); ``wave_fn(times)`` the wave
    level [time, transect] the imagery correction uses, so that the imagery positions can be moved
    back to where the waterline was at each radar time. Returns the waterlines (same columns as
    ``coastsat_sar_waterlines``, plus ``score``) and the templates (for the report).
    """
    from ceva.engines import reference_positions

    c = cfg.sar.template
    p, orbits = sar_profiles(cfg, chips, cube, seaward_sign, tide_fn)
    reference = reference_positions(p, annual, curves)
    n_time, n_transect = reference.shape
    t_idx, n_idx = np.meshgrid(np.arange(n_time), np.arange(n_transect), indexing="ij")
    pairs = pd.DataFrame({"transect_id": p.transect_id[n_idx.ravel()], "time": p.time[t_idx.ravel()]})
    x_img = imagery_reference(pairs, imagery, cfg.sar.calibration.window_days,
                              cfg.sar.calibration.min_imagery).reshape(n_time, n_transect)
    level = p.tide + wave_fn(p.time)
    tan_beta = slopes.reindex(p.transect_id.astype(str)).to_numpy()[None, :]
    anchor = x_img - (level - cfg.tides.reference_elevation_m) / tan_beta
    passes = orbits["orbit_state"].to_numpy().astype(str)
    wave = wave_fn(p.time).mean(axis=1)
    edges = np.nanquantile(wave, np.linspace(0, 1, c.wave_classes + 1)[1:-1]) if c.wave_classes > 1 else []
    wave_class = np.digitize(wave, edges)
    groups = np.array([f"{ps}|{wc}" for ps, wc in zip(passes, wave_class, strict=True)])
    x_obs, score, templates = match_templates(p.index, p.distance, reference, anchor, groups, c)
    ok = np.isfinite(x_obs) & (score >= c.min_score)
    t_ok, n_ok = np.nonzero(ok)
    w = pd.DataFrame({
        "transect_id": p.transect_id[n_ok].astype(str), "time": p.time[t_ok], "platform": p.platform[t_ok],
        "family": "sentinel1", "orbit_state": passes[t_ok],
        "relative_orbit": orbits["relative_orbit"].to_numpy()[t_ok], "x_obs": x_obs[t_ok, n_ok],
        "tide": p.tide[t_ok, n_ok], "threshold": np.nan, "method": "template", "score": score[t_ok, n_ok],
    })
    step = c.step_m
    offsets = np.arange(-c.half_width_m, c.half_width_m + step / 2, step)
    table = pd.DataFrame([{"transect_id": str(p.transect_id[t["transect"]]), "group": t["group"], "n_learned": t["n"],
                           "offset_m": float(o), "db": float(v)}
                          for t in templates for o, v in zip(offsets, t["template"], strict=True)])
    return w.sort_values(["time", "transect_id"]).reset_index(drop=True), table


def _with_tide(tables: list[pd.DataFrame], transects: pd.DataFrame, tide_fn, cfg: Config) -> pd.DataFrame:
    columns = ["transect_id", "time", "platform", "family", "orbit_state", "relative_orbit", "x_obs", "tide",
               "threshold", "method"]
    if not tables:
        return pd.DataFrame(columns=columns)
    w = pd.concat(tables, ignore_index=True)
    times = pd.DatetimeIndex(sorted(w["time"].unique()))
    tide = tide_fn(times)
    w["tide"] = tide[times.get_indexer(w["time"]), pd.Index(transects["transect_id"]).get_indexer(w["transect_id"])]
    w["family"] = "sentinel1"
    w["threshold"] = np.nan
    return w.sort_values(["time", "transect_id"]).reset_index(drop=True)[columns]


# ------------------------------------------------------------------ alignment on imagery

def imagery_reference(sar: pd.DataFrame, imagery: pd.DataFrame, window_days: float, min_n: int) -> np.ndarray:
    """Median of the imagery positions (``x_corr``, used ones) within ``window_days`` of each radar waterline."""
    out = np.full(len(sar), np.nan)
    window = np.timedelta64(int(window_days * 86400), "s")
    used = imagery[imagery["use"]]
    by_transect = {tid: (g["time"].to_numpy("datetime64[s]"), g["x_corr"].to_numpy())
                   for tid, g in used.sort_values("time").groupby("transect_id")}
    positions = sar.reset_index(drop=True)
    for tid, g in positions.groupby("transect_id"):
        if tid not in by_transect:
            continue
        t_img, x_img = by_transect[tid]
        t = g["time"].to_numpy("datetime64[s]")
        lo, hi = np.searchsorted(t_img, t - window), np.searchsorted(t_img, t + window, side="right")
        for row, a, b in zip(g.index, lo, hi, strict=True):
            if b - a >= min_n:
                out[row] = np.median(x_img[a:b])
    return out


def _mad(r) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    return 1.4826 * float(np.median(np.abs(r - np.median(r)))) if len(r) else np.nan


def align_on_imagery(cfg: Config, sar: pd.DataFrame, imagery: pd.DataFrame, slopes: pd.Series,
                     hs: np.ndarray | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Radar waterlines moved onto the imagery: ``x_level`` (corrected for the water level like the
    imagery: tide, plus ``wave`` when given), ``x_ref`` (imagery reference), ``x_corr`` (aligned),
    ``offset`` (what was removed), ``outlier`` and ``use``.

    Returns the waterlines and the calibration: one row per transect and orbit direction (offset,
    pairs used, spread of the residuals) with the fitted terms in its attrs.
    """
    c = cfg.sar.calibration
    w = sar.reset_index(drop=True).copy()
    if "wave" not in w:
        w["wave"] = np.float32(0.0)
    w["level"] = w["tide"] + w["wave"]
    w["x_level"] = tide_correct(w, slopes, cfg.tides.reference_elevation_m, level="level")
    w["x_ref"] = imagery_reference(w, imagery, c.window_days, c.min_imagery)
    w["hs_anomaly"] = 0.0
    if hs is not None:
        w["hs_anomaly"] = hs - pd.Series(hs).groupby(w["transect_id"]).transform("mean").to_numpy()
    w["tide_c"] = w["tide"] - w.groupby("transect_id")["tide"].transform("mean")
    terms = [{"hs": "hs_anomaly", "tide": "tide_c"}[t] for t in c.terms]
    group = w["transect_id"].astype(str) + "|" + w["orbit_state"].astype(str)
    diff = (w["x_level"] - w["x_ref"]).to_numpy()
    paired = np.isfinite(diff)
    keep = paired.copy()
    beta = np.zeros(len(terms))
    X = w[terms].to_numpy(dtype=float) if terms else np.zeros((len(w), 0))
    for _ in range(4):
        rest = diff - X @ beta
        by_group = pd.Series(rest[keep]).groupby(group[keep].to_numpy())
        offset_g, n_g = by_group.median(), by_group.size()
        by_pass = pd.Series(rest[keep]).groupby(w["orbit_state"].to_numpy()[keep]).median()
        enough = n_g[n_g >= c.min_pairs].index
        offset = group.map(offset_g.reindex(enough)).fillna(w["orbit_state"].map(by_pass)).to_numpy()
        if terms:
            target = diff - offset
            beta, *_ = np.linalg.lstsq(X[keep], target[keep], rcond=None)
        resid = diff - offset - X @ beta
        mad = 1.4826 * np.nanmedian(np.abs(resid[keep] - np.nanmedian(resid[keep])))
        keep = paired & (np.abs(resid) <= c.outlier_mad_k * max(mad, 1.0))
    w["offset"] = offset + X @ beta
    yearly = {}
    if c.per_year:
        # the radar's drift from one year to the next (satellites joining or leaving, processing):
        # a unit-wide offset per year, from that year's pairs, removed too
        year = w["time"].dt.year.to_numpy()
        rest = diff - w["offset"].to_numpy()
        by_year = pd.Series(rest[keep]).groupby(year[keep])
        drift = by_year.median()[by_year.size() >= c.min_pairs]
        yearly = {int(k): float(v) for k, v in drift.items()}
        w["offset"] = w["offset"] + pd.Series(year).map(drift).fillna(0.0).to_numpy()
    w["x_corr"] = w["x_level"] - w["offset"]
    resid = (w["x_corr"] - w["x_ref"]).to_numpy()
    # outliers: against the imagery where it exists, else against the radar's own running median
    far = np.abs(resid) > c.outlier_mad_k * max(mad, 1.0)
    running = flag_outliers(w, "x_corr", c.outlier_mad_k).to_numpy()
    w["outlier"] = np.where(np.isfinite(resid), far, running)
    w["low_water"] = False
    w["use"] = ~w["outlier"]
    table = pd.DataFrame({"group": group, "transect_id": w["transect_id"], "orbit_state": w["orbit_state"],
                          "resid": resid, "paired": paired, "kept": keep, "offset": offset})
    calibration = table.groupby(["transect_id", "orbit_state"]).agg(
        offset=("offset", "first"), n_radar=("offset", "size"), n_pairs=("paired", "sum"), n_used=("kept", "sum"),
        resid_median=("resid", "median"), resid_mad=("resid", _mad),
    ).reset_index()
    calibration.attrs = {"terms": dict(zip(c.terms, map(float, beta), strict=True)), "yearly_offsets": yearly,
                         "resid_mad": float(mad),
                         "n_pairs": int(paired.sum()), "n_used": int(keep.sum())}
    return w.drop(columns=["tide_c"]), calibration


def source_yearly(imagery: pd.DataFrame, radar: pd.DataFrame | None, cfg: Config) -> dict[str, pd.DataFrame]:
    """Yearly median of each source (``imagery``, ``radar``) with the variance of that median.

    The variance is sigma^2 / min(n, cap): sigma is the source's spread of single positions
    around its running median (at this unit), n its observations that year, and ``cap``
    (``sar.effective_per_year``) bounds how many count as independent, because errors of images
    close in time are alike. Radar also drifts from the imagery from one year to the next
    (satellites joining or leaving, processing): its variance gets a year-level term tau^2, the
    spread of the yearly radar minus imagery differences beyond what sampling explains. The
    estimates are in each table's ``attrs`` (sigma, tau).
    """
    from ceva.engines import annual_from_waterlines

    cap = cfg.sar.effective_per_year
    out = {}
    for source, w in (("imagery", imagery), ("radar", radar)):
        if w is None or w.empty:
            continue
        used = w[w["use"]].sort_values(["transect_id", "time"])
        running = used.groupby("transect_id")["x_corr"].transform(
            lambda s: s.rolling(11, center=True, min_periods=3).median())
        resid = (used["x_corr"] - running).dropna()
        sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) if len(resid) else np.nan
        y = annual_from_waterlines(w, "x_corr", label=source)
        y["var"] = sigma**2 / np.minimum(y["n_obs"], cap)
        y.attrs = {"sigma": sigma, "tau": 0.0}
        out[source] = y
    if "radar" in out:
        both = out["imagery"].merge(out["radar"], on=["transect_id", "year"], suffixes=("_i", "_r"))
        tau2 = 0.0
        if len(both) >= 10:
            d = both["x_r"] - both["x_i"]
            d = d - d.groupby(both["transect_id"]).transform("median")     # per-transect offsets are aligned already
            spread = (1.4826 * float(np.median(np.abs(d - np.median(d))))) ** 2
            tau2 = max(spread - float((both["var_i"] + both["var_r"]).median()), 0.0)
        out["radar"]["var"] = out["radar"]["var"] + tau2
        out["radar"].attrs["tau"] = float(np.sqrt(tau2))
    return out


def combined_yearly(imagery: pd.DataFrame, radar: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Yearly positions from imagery and aligned radar together: the two yearly medians of
    ``source_yearly`` averaged with weights 1 / variance. Returns transect_id, year, x, n_obs,
    n_imagery, n_radar, se (standard error of x), sensors="combined"; the estimates in ``attrs``.
    """
    sources = source_yearly(imagery, radar, cfg)
    parts = pd.concat([y.assign(source=k) for k, y in sources.items()], ignore_index=True)
    parts["weight"] = 1.0 / parts["var"]
    parts["wx"] = parts["weight"] * parts["x"]
    for k in ("imagery", "radar"):
        parts[f"n_{k}"] = np.where(parts["source"] == k, parts["n_obs"], 0)
    g = parts.groupby(["transect_id", "year"])
    out = pd.DataFrame({"x": g["wx"].sum() / g["weight"].sum(), "n_obs": g["n_obs"].sum(),
                        "n_imagery": g["n_imagery"].sum(), "n_radar": g["n_radar"].sum(),
                        "se": np.sqrt(1.0 / g["weight"].sum())}).reset_index()
    out["sensors"] = "combined"
    out.attrs = {"sigma_imagery": sources["imagery"].attrs["sigma"],
                 "sigma_radar": sources["radar"].attrs["sigma"] if "radar" in sources else None,
                 "tau_radar": sources["radar"].attrs["tau"] if "radar" in sources else None}
    return out
