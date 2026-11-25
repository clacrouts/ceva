"""Image evidence: the satellite images behind each shoreline, for the dashboard's image viewer.

For every 2 km segment of coast, each image CEVA used is cut along the coast: the sea at the top (so
the image's vertical axis is the card chart's, metres seaward up), 10 m pixels, and across the shore
±300 m around the long-term shoreline (CEVA's chip margin and matching corridor), wider where the
waterlines range further, up to half a transect. Images are shown in natural colour with one fixed
stretch, so a change of brightness between images is real. They come from the scenes the cube was
built from, merged as the cube merges them.

Each segment gives:

* ``<segment>.mp4``: the images in time order, a short H.264 video, keyframes every ``KEYFRAME_EVERY``
  images, image i at i / ``FPS`` seconds. The frames are small and alike: on the six survey beaches a
  video took 0.75 KB per image against 1.1 KB for WebP images of the same quality, and the viewer lands
  on the exact image when it seeks.
* ``<segment>.json``: where the frame sits on the map (its four corners), and for every image and
  transect the waterline CEVA detected (``x_obs``) and the shoreline after its tide and wave correction
  (``x_corr``), which the viewer draws over the image, with the tide, the waves and the year's positions.

``pack`` puts the segments of an area into one uncompressed ZIP (what is uploaded to Zenodo: anyone can
download and open it) and writes an index of where each file's bytes start, so the viewer reads one
segment with HTTP range requests.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from pyproj import Transformer
from scipy.ndimage import map_coordinates

from ceva.config import Config

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
SEGMENT_KM = 2.0
PIXEL_M = 10.0
HALF_WIDTH_M = 300.0       # across the shore, each side of the long-term shoreline
MAX_HALF_WIDTH_M = 1000.0  # half a transect
BEYOND_M = 100.0           # beyond the waterlines' 2-98 % range, and beyond the end transects
STRETCH = (0.0, 0.30)      # surface reflectance shown from black to white, the same for every image
GAMMA = 0.75               # display = scaled ** GAMMA: dark water and wet sand a little brighter
NO_DATA = 96               # grey where the image has no data
FPS = 10
KEYFRAME_EVERY = 10
VIDEO_CRF = 24             # H.264 quality (lower is better and larger)
BLOCK_MAX_PIXELS = 250_000     # one read per image covers neighbouring segments up to this size (25 km2)
GAP_KM = 1.0                   # transects further apart than this along the coast start a new segment
MIN_ALONG_M = 600.0            # a frame is at least this long along the coast (an isolated transect)
READ_BANDS = ("red", "green", "blue")


def _apply(t: Affine, x, y) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return t.a * x + t.b * y + t.c, t.d * x + t.e * y + t.f


@dataclass(frozen=True)
class FrameGrid:
    """A segment's frame: a rectangle along the coast, the sea at the top, ``PIXEL_M`` pixels.

    ``transform`` maps a pixel (column, row) to metres of ``epsg``; it turns with the coast.
    """

    epsg: int
    transform: Affine
    width: int
    height: int

    @property
    def corners(self) -> np.ndarray:
        """Top-left, top-right, bottom-right, bottom-left, in metres of ``epsg``."""
        x, y = _apply(self.transform, [0, self.width, self.width, 0], [0, 0, self.height, self.height])
        return np.column_stack([x, y])

    def pixel_of(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Column and row (fractional) of points in metres of ``epsg``."""
        return _apply(~self.transform, x, y)

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        cols, rows = np.meshgrid(np.arange(self.width) + 0.5, np.arange(self.height) + 0.5)
        return _apply(self.transform, cols, rows)


def _ends(lines: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Landward and seaward ends of the transects (``seaward_sign`` +1: stored land to sea)."""
    first = np.array([g.coords[0][:2] for g in lines.geometry])
    last = np.array([g.coords[-1][:2] for g in lines.geometry])
    forward = (lines["seaward_sign"].to_numpy() >= 0)[:, None]
    return np.where(forward, first, last), np.where(forward, last, first)


def frame_grid(lines: gpd.GeoDataFrame, waterlines: pd.DataFrame) -> FrameGrid | None:
    """The frame of a segment from its transects (in metres) and the waterlines CEVA used there.

    Along the coast it spans the transects' long-term shorelines plus ``BEYOND_M``; across, ±300 m
    around them, wider to hold 2-98 % of the detected and corrected waterlines plus ``BEYOND_M``.
    None when no waterline was used.
    """
    land, sea = _ends(lines)
    length = np.linalg.norm(sea - land, axis=1)
    d = (sea - land) / np.where(length > 0, length, 1)[:, None]
    used = waterlines[waterlines["use"]]
    ids = lines["transect_id"].astype(str).to_numpy()
    median = used.groupby("transect_id")["x_corr"].median().reindex(ids).to_numpy()
    ok = np.isfinite(median)
    if not ok.any():
        return None
    v = d[ok].mean(axis=0)
    v /= np.linalg.norm(v)
    u = np.array([v[1], -v[0]])          # the seaward direction turned clockwise: with the sea up, u runs right
    shore = land[ok] + median[ok, None] * d[ok]
    origin = shore.mean(axis=0)
    along = (shore - origin) @ u
    a0, a1 = along.min() - BEYOND_M, along.max() + BEYOND_M
    if a1 - a0 < MIN_ALONG_M:
        a0, a1 = (a0 + a1 - MIN_ALONG_M) / 2, (a0 + a1 + MIN_ALONG_M) / 2
    row = pd.Series(np.arange(len(ids)), index=ids)
    i = row.reindex(used["transect_id"].astype(str)).to_numpy()
    keep = np.isfinite(i)
    i = i[keep].astype(int)
    across = np.concatenate([((land[i] + used[col].to_numpy()[keep, None] * d[i]) - origin) @ v
                             for col in ("x_obs", "x_corr")])
    lo, hi = np.nanpercentile(across, [2, 98]) if np.isfinite(across).any() else (0.0, 0.0)
    c0 = max(min(-HALF_WIDTH_M, lo - BEYOND_M), -MAX_HALF_WIDTH_M)
    c1 = min(max(HALF_WIDTH_M, hi + BEYOND_M), MAX_HALF_WIDTH_M)
    width = 2 * math.ceil((a1 - a0) / (2 * PIXEL_M))        # even: H.264 needs it
    height = 2 * math.ceil((c1 - c0) / (2 * PIXEL_M))
    top_left = origin + a0 * u + c1 * v
    transform = Affine(PIXEL_M * u[0], -PIXEL_M * v[0], top_left[0], PIXEL_M * u[1], -PIXEL_M * v[1], top_left[1])
    epsg = int(lines.crs.to_epsg())
    return FrameGrid(epsg, transform, width, height)


def block_grid(frames: list[FrameGrid], pixel_m: float = PIXEL_M):
    """The north-up grid (``ceva.chips.ChipGrid``) read once per image for these frames."""
    from ceva.chips import ChipGrid

    corners = np.concatenate([f.corners for f in frames])
    x0 = math.floor(corners[:, 0].min() / pixel_m) * pixel_m - 2 * pixel_m
    x1 = math.ceil(corners[:, 0].max() / pixel_m) * pixel_m + 2 * pixel_m
    y0 = math.ceil(corners[:, 1].max() / pixel_m) * pixel_m + 2 * pixel_m
    y1 = math.floor(corners[:, 1].min() / pixel_m) * pixel_m - 2 * pixel_m
    return ChipGrid(frames[0].epsg, float(x0), float(y0), float(pixel_m), int(round((x1 - x0) / pixel_m)),
                    int(round((y0 - y1) / pixel_m)))


def blocks(frames: list[FrameGrid], max_pixels: int = BLOCK_MAX_PIXELS) -> list[list[int]]:
    """Neighbouring frames (in order along the coast) grouped so that each group's grid stays small."""
    out: list[list[int]] = []
    for i, frame in enumerate(frames):
        if out:
            grid = block_grid([frames[j] for j in out[-1]] + [frame])
            if grid.width * grid.height <= max_pixels:
                out[-1].append(i)
                continue
        out.append([i])
    return out


def cut(block: dict[str, np.ndarray], grid, frame: FrameGrid) -> dict[str, np.ndarray]:
    """The frame's pixels from a block read on the north-up ``grid``: bilinear for reflectance."""
    x, y = frame.centres()
    col = (x - grid.x0) / grid.pixel_m - 0.5
    row = (grid.y0 - y) / grid.pixel_m - 0.5
    out = {}
    for name, band in block.items():
        order, fill = (0, 0) if name == "qa" else (1, np.nan)
        out[name] = map_coordinates(band, [row, col], order=order, mode="constant", cval=fill)
    return out


def render(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Natural colour, uint8 [height, width, 3], with the same stretch for every image."""
    rgb = np.stack([red, green, blue], axis=-1)
    lo, hi = STRETCH
    scaled = np.clip((np.nan_to_num(rgb, nan=lo) - lo) / (hi - lo), 0.0, 1.0) ** GAMMA
    out = np.round(scaled * 255).astype(np.uint8)
    out[~np.isfinite(rgb).all(axis=-1)] = NO_DATA
    return out


def cloud_share(qa: np.ndarray, family: str, cfg: Config) -> float:
    """Share of the frame's pixels with data that the quality flags reject (clouds, shadows, snow)."""
    from ceva.cube import _no_data

    data = ~_no_data(qa, family)
    if not data.any():
        return 1.0
    if family == "landsat":
        bad = (qa & sum(1 << b for b in cfg.masking.landsat_qa_drop_bits)) != 0
    else:
        bad = np.isin(qa, cfg.masking.s2_scl_drop_classes)
    return float((bad & data).sum() / data.sum())


def ffmpeg_path() -> str | None:
    import sys

    beside = Path(sys.executable).with_name("ffmpeg")
    return str(beside) if beside.exists() else shutil.which("ffmpeg")


class VideoWriter:
    """Frames of one size, piped to ffmpeg (H.264, yuv420p, constant frame rate, keyframes every
    ``KEYFRAME_EVERY``), so that frame i starts at i / FPS seconds."""

    def __init__(self, path: Path, width: int, height: int, ffmpeg: str | None = None):
        ffmpeg = ffmpeg or ffmpeg_path()
        if ffmpeg is None:
            raise FileNotFoundError("ffmpeg is not installed (conda-forge: ffmpeg)")
        self.path, self.count = path, 0
        self.proc = subprocess.Popen(
            [ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(VIDEO_CRF),
             "-g", str(KEYFRAME_EVERY), "-keyint_min", str(KEYFRAME_EVERY), "-sc_threshold", "0",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.count += 1

    def close(self) -> None:
        if self.count == 0:                       # nothing to encode: no file
            self.proc.kill()
            self.proc.wait()
            self.path.unlink(missing_ok=True)
            return
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed on {self.path.name}")


def _dm(values: pd.Series | np.ndarray) -> list:
    """Metres as whole decimetres, None where missing (compact JSON)."""
    return [None if not np.isfinite(v) else int(round(v * 10)) for v in np.asarray(values, dtype=float)]


def segment_lines(lines: gpd.GeoDataFrame, frame: FrameGrid) -> list[dict]:
    """Each transect in the frame's pixels: its landward end and its step per metre seaward."""
    land, sea = _ends(lines)
    length = np.linalg.norm(sea - land, axis=1)
    step = (sea - land) / np.where(length > 0, length, 1)[:, None]
    c0, r0 = frame.pixel_of(land[:, 0], land[:, 1])
    c1, r1 = frame.pixel_of(land[:, 0] + step[:, 0], land[:, 1] + step[:, 1])
    return [{"id": str(t), "x0": round(float(a), 2), "y0": round(float(b), 2), "dx": round(float(c - a), 5),
             "dy": round(float(e - b), 5)} for t, a, b, c, e in zip(lines["transect_id"], c0, r0, c1, r1, strict=True)]


def lonlat(frame: FrameGrid, points: np.ndarray) -> list[list[float]]:
    to = Transformer.from_crs(frame.epsg, 4326, always_xy=True)
    lon, lat = to.transform(points[:, 0], points[:, 1])
    return [[round(float(a), 6), round(float(b), 6)] for a, b in zip(lon, lat, strict=True)]


@dataclass
class SegmentPlan:
    name: str
    lines: gpd.GeoDataFrame          # the segment's transects, in metres
    frame: FrameGrid
    waterlines: pd.DataFrame         # the segment's waterlines (all, flags kept)
    times: pd.DatetimeIndex          # images with at least one waterline used in the segment


def coast_segments(lines: gpd.GeoDataFrame, segment_km: float = SEGMENT_KM, gap_km: float = GAP_KM) -> list[np.ndarray]:
    """Groups of transects (row positions), consecutive along the coast, each spanning at most
    ``segment_km`` of coast; a gap of more than ``gap_km`` between neighbours also starts a group.

    The transects are chained from one end of the coast to the other, each to its nearest
    remaining neighbour, so a curved coast is measured along its length, not along one axis.
    """
    mid = np.array([[g.interpolate(0.5, normalized=True).x, g.interpolate(0.5, normalized=True).y]
                    for g in lines.geometry])
    n = len(mid)
    if n < 2:
        return [np.arange(n)]
    far = int(np.argmax(np.linalg.norm(mid - mid[0], axis=1)))
    start = int(np.argmax(np.linalg.norm(mid - mid[far], axis=1)))      # one end of the coast's longest extent
    order, left = [start], np.ones(n, dtype=bool)
    left[start] = False
    for _ in range(n - 1):
        d = np.linalg.norm(mid - mid[order[-1]], axis=1)
        d[~left] = np.inf
        order.append(int(np.argmin(d)))
        left[order[-1]] = False
    step = np.linalg.norm(np.diff(mid[order], axis=0), axis=1)
    groups, after_gap, current, run = [], [True], [order[0]], 0.0
    for i, gap in zip(order[1:], step, strict=True):
        if gap > gap_km * 1000.0 or run + gap > segment_km * 1000.0:
            groups.append(current)
            after_gap.append(gap > gap_km * 1000.0)
            current, run = [i], 0.0
        else:
            current.append(i)
            run += gap
    groups.append(current)
    merged: list[list[int]] = []
    for k, group in enumerate(groups):       # a lone transect cut off by length joins the segment before it
        if merged and len(group) == 1 and not after_gap[k]:
            merged[-1] += group
        else:
            merged.append(group)
    return [np.array(g) for g in merged]


def plan_segments(name: str, transects: gpd.GeoDataFrame, waterlines: pd.DataFrame,
                  segment_km: float = SEGMENT_KM) -> list[SegmentPlan]:
    """The segments of an area (a unit or a survey site), in order along the coast."""
    oriented = transects[transects["seaward_sign"] != 0].reset_index(drop=True)
    if oriented.empty:
        return []
    epsg = int(oriented.estimate_utm_crs().to_epsg())
    metric = oriented.to_crs(epsg)
    plans = []
    for k, rows in enumerate(coast_segments(metric, segment_km)):
        lines = metric.iloc[rows].reset_index(drop=True)
        mine = waterlines[waterlines["transect_id"].astype(str).isin(lines["transect_id"].astype(str))]
        frame = frame_grid(lines, mine)
        if frame is None:
            continue
        times = pd.DatetimeIndex(sorted(mine.loc[mine["use"], "time"].unique()))
        plans.append(SegmentPlan(f"{name}-{k + 1}", lines, frame, mine, times))
    return plans


def _scenes_by_id(cfg: Config, grids: list, families: set[str], start, end, creds) -> dict:
    """The catalogue's scenes over the grids, by id (the cube keeps only the ids)."""
    from ceva.catalog import search_scenes
    from ceva.chips import grid_bbox_wgs84

    boxes = np.array([grid_bbox_wgs84(g) for g in grids])
    bbox = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())
    out = {}
    for family in sorted(families):
        result = search_scenes(cfg, family, bbox, start, end, creds=creds)
        out.update({s.id: s for s in result.scenes})
    return out


def _in_order(pool: ThreadPoolExecutor, fn: Callable, items, ahead: int):
    """``pool.map`` that reads at most ``ahead`` items ahead of the consumer (memory stays bounded)."""
    from collections import deque

    pending, items = deque(), iter(items)
    for item in items:
        pending.append(pool.submit(fn, item))
        if len(pending) >= ahead:
            break
    while pending:
        result = pending.popleft().result()
        nxt = next(items, None)
        if nxt is not None:
            pending.append(pool.submit(fn, nxt))
        yield result


def _acquisitions(cfg: Config, name: str, catalogue: dict, waterlines: pd.DataFrame,
                  times: pd.DatetimeIndex) -> dict[pd.Timestamp, list[str]]:
    """The scenes of each image used, by its time: from the cube when it is kept, otherwise from the
    catalogue, grouped as the cube builder groups them (one satellite, one day: one acquisition)."""
    from ceva.cube import cube_path, group_acquisitions, open_cube

    if cube_path(cfg, name).exists():
        cube = open_cube(cfg, name)
        ids = pd.Series(cube["scene_ids"].values, index=pd.DatetimeIndex(cube["time"].values))
        return {t: str(ids[t]).split() for t in times if t in ids.index}
    by_day = {(g[0].platform, g[0].datetime.date()): [s.id for s in g]
              for g in group_acquisitions(list(catalogue.values()))}
    platform = waterlines.drop_duplicates("time").set_index("time")["platform"].astype(str)
    return {t: by_day[key] for t in times if (key := (platform.get(t, ""), t.date())) in by_day}


def build(cfg: Config, name: str, out: Path, *, segment_km: float = SEGMENT_KM, workers: int = 8,
          progress: Callable[[int, int], None] | None = None) -> list[dict]:
    """The frames of an area with stored results (``name``: a unit or a survey site).

    Returns, for each segment, what was written (files, frames, sizes).
    """
    from ceva.chips import _read_acquisition
    from ceva.credentials import load_credentials
    from ceva.reader import gdal_environment

    results = cfg.data_dir / "results" / name / "ceva"
    transects = gpd.read_parquet(results / "transects.parquet")
    waterlines = pd.read_parquet(results / "waterlines.parquet")
    yearly = pd.read_parquet(results / "shorelines_imagery.parquet", columns=["transect_id", "year", "x", "se_m"])
    plans = plan_segments(name, transects, waterlines, segment_km)
    if not plans:
        return []
    times = pd.DatetimeIndex(sorted(set().union(*[set(p.times) for p in plans])))
    groups = blocks([p.frame for p in plans])
    grids = [block_grid([plans[i].frame for i in g]) for g in groups]
    creds = load_credentials()
    families = set(waterlines.loc[waterlines["time"].isin(times), "family"].astype(str))
    catalogue = _scenes_by_id(cfg, grids, families, times.min().date(), times.max().date(), creds)
    scene_ids = _acquisitions(cfg, name, catalogue, waterlines, times)
    options: dict[str, dict] = {}

    def read(time: pd.Timestamp) -> tuple[pd.Timestamp, dict[int, dict] | None]:
        ids = scene_ids.get(time, [])
        scenes = [catalogue[i] for i in ids if i in catalogue]
        if not scenes:
            return time, None
        provider = scenes[0].provider
        if provider not in options:
            options[provider] = gdal_environment(provider, cfg, creds)
        frames = {}
        for g, grid in zip(groups, grids, strict=True):
            wanted = [i for i in g if time in plans[i].times]
            if not wanted:
                continue
            data, _ = _read_acquisition(scenes, grid, (*READ_BANDS, ), options[provider])
            if data is None:
                continue
            for i in wanted:
                part = cut(data, grid, plans[i].frame)
                frames[i] = {"rgb": render(part["red"], part["green"], part["blue"]),
                             "cloud": cloud_share(part["qa"].astype(np.int64), scenes[0].family, cfg)}
        return time, frames

    out.mkdir(parents=True, exist_ok=True)
    writers = {i: VideoWriter(out / f"{p.name}.mp4", p.frame.width, p.frame.height) for i, p in enumerate(plans)}
    kept: dict[int, list[pd.Timestamp]] = {i: [] for i in range(len(plans))}
    cloud: dict[int, list[float]] = {i: [] for i in range(len(plans))}
    missing = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (time, frames) in enumerate(_in_order(pool, read, times, 2 * workers), start=1):
            if frames is None:
                missing += 1
            for i, item in (frames or {}).items():
                writers[i].write(item["rgb"])
                kept[i].append(time)
                cloud[i].append(item["cloud"])
            if progress is not None:
                progress(done, len(times))
    for writer in writers.values():
        writer.close()
    if missing:
        log.warning("%s: %d of %d images not found in the catalogue or without data", name, missing, len(times))

    return [_write_segment(out, plan, kept[i], cloud[i], yearly, name) for i, plan in enumerate(plans) if kept[i]]


def _write_segment(out: Path, plan: SegmentPlan, times: list[pd.Timestamp], cloud: list[float],
                   yearly: pd.DataFrame, area: str) -> dict:
    """The segment's JSON: frame on the map, transects, and per image the waterlines and their correction."""
    ids = plan.lines["transect_id"].astype(str).tolist()
    w = plan.waterlines.assign(transect_id=plan.waterlines["transect_id"].astype(str))
    w = w[w["time"].isin(times)].set_index(["time", "transect_id"])
    index = pd.MultiIndex.from_product([pd.DatetimeIndex(times), ids], names=["time", "transect_id"])
    w = w.reindex(index)
    per_image = w.groupby(level="time", sort=False).first()     # tide, waves, platform: one per image
    flag = np.where(w["use"].fillna(False).astype(bool), 0,
                    np.where(w["outlier"].fillna(False).astype(bool), 1, np.where(w["x_obs"].notna(), 2, -1)))
    years = yearly[yearly["transect_id"].astype(str).isin(ids)]
    payload = {
        "v": FORMAT_VERSION, "name": plan.name, "area": area, "pixel_m": PIXEL_M,
        "width": plan.frame.width, "height": plan.frame.height,
        "corners": lonlat(plan.frame, plan.frame.corners), "fps": FPS, "video": f"{plan.name}.mp4",
        "transects": segment_lines(plan.lines, plan.frame),
        "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
        "platform": per_image["platform"].astype(str).tolist(),
        "tide": _dm(per_image["tide"]), "wave": _dm(per_image["wave"]), "level": _dm(per_image["level"]),
        "cloud": [int(round(100 * c)) for c in cloud],
        "x_obs": _dm(w["x_obs"]), "x_corr": _dm(w["x_corr"]), "flag": flag.astype(int).tolist(),
        "yearly": {t: {"year": g["year"].astype(int).tolist(), "x": _dm(g["x"]), "se": _dm(g["se_m"])}
                   for t, g in years.sort_values("year").groupby(years["transect_id"].astype(str))},
    }
    path = out / f"{plan.name}.json"
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return {"name": plan.name, "area": area, "frames": len(times), "width": plan.frame.width,
            "height": plan.frame.height, "transects": len(ids), "files": [path.name, payload["video"]],
            "bytes": {"json": path.stat().st_size, "video": (out / payload["video"]).stat().st_size}}


def pack(folder: Path, archive: str, segments: list[dict], index_path: Path) -> dict:
    """The segments' files in one uncompressed ZIP, and an index of each file's bytes in it.

    The index (``{"archive", "segments": [{name, area, corners, frames, transects, files: {file: [start, size]}}]}``)
    lets the viewer fetch one file with a single HTTP range request, from the local server or Zenodo.
    """
    zpath = folder / archive
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
        for seg in segments:
            for file in seg["files"]:
                z.write(folder / file, arcname=file)
    where: dict[str, list[int]] = {}
    with zipfile.ZipFile(zpath) as z, open(zpath, "rb") as raw:
        for info in z.infolist():
            raw.seek(info.header_offset)
            header = raw.read(30)
            name_len, extra_len = int.from_bytes(header[26:28], "little"), int.from_bytes(header[28:30], "little")
            where[info.filename] = [info.header_offset + 30 + name_len + extra_len, info.file_size]
    entries = []
    for seg in segments:
        meta = json.loads((folder / f"{seg['name']}.json").read_text())
        entries.append({"name": seg["name"], "area": seg["area"], "label": seg.get("label", seg["area"]),
                        "corners": meta["corners"], "frames": seg["frames"],
                        "transects": [t["id"] for t in meta["transects"]],
                        "first": meta["time"][0] if meta["time"] else None,
                        "last": meta["time"][-1] if meta["time"] else None,
                        "files": {f: where[f] for f in seg["files"]}})
    index = {"v": FORMAT_VERSION, "archive": archive, "bytes": zpath.stat().st_size, "segments": entries}
    index_path.write_text(json.dumps(index, separators=(",", ":")))
    return index


# ---------------------------------------------------------------------- many units: build, then pack
def work_dir(cfg: Config) -> Path:
    """Where segments are built before they are packed (``<data dir>/frames/work``)."""
    return cfg.data_dir / "frames" / "work"


def is_built(work: Path, name: str) -> bool:
    return (work / f"{name}.done.json").exists()


def mark_built(work: Path, name: str, label: str, segments: list[dict]) -> None:
    """Record a unit as built (its segments' files are in ``work``): a later build skips it."""
    (work / f"{name}.done.json").write_text(json.dumps({"name": name, "label": label, "segments": segments}))


def built_units(work: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(work.glob("*.done.json"))]


def _countries(cfg: Config, ids: dict[str, list[str]]) -> dict[str, str | None]:
    """The most common country (GCTS) of each unit's transects; None where none is known."""
    from ceva.transects.gcts import available_indexes

    wanted = sorted({t for v in ids.values() for t in v})
    if not wanted:
        return {}
    known = pd.concat([pd.read_parquet(p, columns=["transect_id", "country"], filters=[("transect_id", "in", wanted)])
                       for p in available_indexes(cfg)]).drop_duplicates("transect_id")
    known = known.set_index("transect_id")["country"]
    out = {}
    for unit, tids in ids.items():
        found = known.reindex(tids).dropna()
        out[unit] = str(found.mode().iloc[0]) if len(found) else None
    return out


def pack_all(cfg: Config, work: Path, dest: Path, by_country: bool = True) -> dict:
    """The built units in uncompressed ZIPs (one per country, or one), with the indexes the viewer reads:

    * ``index.json``: ``{"v": 2, "archives": {name: bytes}, "units": {unit: {label, country, segments, frames}}}``
    * ``units/<unit>.json``: the unit's segments, each with its archive and where its files' bytes are.

    The map loads a unit's index only when one of its transects is opened.
    """
    units = [u for u in built_units(work) if u["segments"]]

    def transect_ids(unit: dict) -> list[str]:
        metas = [json.loads((work / f"{s['name']}.json").read_text()) for s in unit["segments"]]
        return [t["id"] for m in metas for t in m["transects"]]

    ids = {u["name"]: transect_ids(u) for u in units}
    country = _countries(cfg, ids) if by_country else {}
    groups: dict[str, list[dict]] = {}
    for u in units:
        archive = f"frames-{country.get(u['name']) or 'other'}.zip" if by_country else "frames.zip"
        groups.setdefault(archive, []).extend({**s, "area": u["name"], "label": u["label"]} for s in u["segments"])
    (dest / "units").mkdir(parents=True, exist_ok=True)
    archives: dict[str, int] = {}
    entries: dict[str, list[dict]] = {}
    for archive, segments in sorted(groups.items()):
        part = pack(work, archive, segments, work / f"{archive}.index.json")
        if (work / archive).resolve() != (dest / archive).resolve():
            shutil.move(str(work / archive), str(dest / archive))
        archives[archive] = part["bytes"]
        for entry in part["segments"]:
            entries.setdefault(entry["area"], []).append({**entry, "archive": archive})
    summary = {}
    for u in units:
        mine = entries.get(u["name"], [])
        (dest / "units" / f"{u['name']}.json").write_text(json.dumps(
            {"v": 2, "unit": u["name"], "label": u["label"], "segments": mine}, separators=(",", ":")))
        summary[u["name"]] = {"label": u["label"], "country": country.get(u["name"]), "segments": len(mine),
                              "frames": sum(e["frames"] for e in mine)}
    index = {"v": 2, "archives": archives, "units": summary}
    (dest / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    return index
