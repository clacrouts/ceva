"""The transect cube: reflectance along every transect for every acquisition.

Dimensions are (time, transect, distance). ``distance`` is metres from the
first vertex of the transect. The cube holds what the sensors measured
(green and SWIR1 reflectance) and the provider's quality flags, unmasked, so
masking rules, water indices and thresholds can change without reading any
imagery again. It is stored in an Icechunk repository: every build is a
commit, and results can always name the exact data they came from.
"""
from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from ceva import __version__
from ceva.catalog import Scene, search_scenes
from ceva.config import Config
from ceva.credentials import Credentials, load_credentials
from ceva.reader import LANDSAT_FILL, S2_NODATA, gdal_environment, sample_scene
from ceva.transects.sampling import TransectSamples, sample_transects

log = logging.getLogger(__name__)

REFLECTANCE_ENCODING = {"dtype": "int16", "scale_factor": 1e-4, "_FillValue": -32768}
MAX_FAILED_FRACTION = 0.05


def group_acquisitions(scenes: list[Scene]) -> list[list[Scene]]:
    """Scenes of one satellite on one day are one acquisition (adjacent tiles of the same pass)."""
    groups: dict[tuple[str, dt.date], list[Scene]] = {}
    for scene in sorted(scenes, key=lambda s: s.datetime):
        groups.setdefault((scene.platform, scene.datetime.date()), []).append(scene)
    return sorted(groups.values(), key=lambda g: g[0].datetime)


def _no_data(qa: np.ndarray, family: str) -> np.ndarray:
    return (qa & LANDSAT_FILL).astype(bool) if family == "landsat" else qa == S2_NODATA


def _sample_acquisition(scenes: list[Scene], samples: TransectSamples, bands: tuple[str, ...],
                        gdal_options: dict) -> tuple[dict[str, np.ndarray] | None, list[str]]:
    """Merge the scenes of one acquisition: first scene with data wins at each sample."""
    merged: dict[str, np.ndarray] | None = None
    failed: list[str] = []
    family = scenes[0].family
    with rasterio.Env(**gdal_options):
        for scene in scenes:
            try:
                data = sample_scene(scene, samples.lon, samples.lat, bands)
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


def build_cube(
    cfg: Config,
    transects: gpd.GeoDataFrame,
    *,
    families: tuple[str, ...] = ("landsat", "sentinel2"),
    start: dt.date | None = None,
    end: dt.date | None = None,
    creds: Credentials | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> xr.Dataset:
    """Search, read and assemble the cube for a set of transects."""
    creds = creds if creds is not None else load_credentials()
    start, end = start or cfg.time.start, end or cfg.time.end
    samples = sample_transects(transects, cfg.cube.spacing_m)
    bands = tuple(cfg.cube.bands)

    jobs: list[tuple[list[Scene], dict]] = []
    providers: dict[str, str] = {}
    for family in families:
        result = search_scenes(cfg, family, samples.bbox, start, end, creds=creds)
        providers[family] = result.provider
        options = gdal_environment(result.provider, cfg, creds)
        jobs += [(group, options) for group in group_acquisitions(result.scenes)]
        log.info("%s: %d scenes from %s", family, len(result.scenes), result.provider)
    jobs.sort(key=lambda job: job[0][0].datetime)

    done = 0

    def run(job: tuple[list[Scene], dict]):
        nonlocal done
        out = _sample_acquisition(job[0], samples, bands, job[1])
        done += 1
        if progress is not None:
            progress(done, len(jobs))
        return out

    with ThreadPoolExecutor(max_workers=cfg.cube.workers) as pool:
        results = list(pool.map(run, jobs))

    failed = [scene_id for _, ids in results for scene_id in ids]
    n_scenes = sum(len(job[0]) for job in jobs)
    if n_scenes and len(failed) / n_scenes > MAX_FAILED_FRACTION:
        raise RuntimeError(f"{len(failed)} of {n_scenes} scenes could not be read, e.g. {failed[:3]}")

    kept = [(job[0], data) for job, (data, _) in zip(jobs, results, strict=True) if data is not None]
    if not kept:
        raise RuntimeError("no acquisition holds any data over these transects")

    dims = ("time", "transect", "distance")
    data_vars = {name: (dims, np.stack([data[name] for _, data in kept]).astype("float32")) for name in bands}
    data_vars["qa"] = (dims, np.stack([data["qa"] for _, data in kept]))
    times = pd.DatetimeIndex([pd.Timestamp(group[0].datetime).tz_convert(None) for group, _ in kept])
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": times,
            "platform": ("time", np.array([group[0].platform for group, _ in kept])),
            "family": ("time", np.array([group[0].family for group, _ in kept])),
            "scene_ids": ("time", np.array([" ".join(s.id for s in group) for group, _ in kept])),
            "transect_id": ("transect", samples.transect_id),
            "distance": samples.distance,
            "length": ("transect", samples.length),
            "lon_start": ("transect", samples.start[:, 0]),
            "lat_start": ("transect", samples.start[:, 1]),
            "lon_end": ("transect", samples.end[:, 0]),
            "lat_end": ("transect", samples.end[:, 1]),
        },
        attrs={
            "title": "CEVA transect cube",
            "ceva_version": __version__,
            "spacing_m": cfg.cube.spacing_m,
            "providers": ", ".join(f"{k}: {v}" for k, v in providers.items()),
            "period": f"{start} to {end}",
            "n_scenes_failed": len(failed),
            "qa_meaning": "landsat: low byte of QA_PIXEL; sentinel2: scene classification (SCL)",
        },
    )
    for name in bands:
        ds[name].attrs.update(long_name=f"{name} surface reflectance", units="1")
    ds["distance"].attrs.update(long_name="distance from the first vertex of the transect", units="m")
    return ds.sortby("time")


def cube_path(cfg: Config, name: str) -> Path:
    return cfg.data_dir / "cubes" / f"{name}.icechunk"


def save_cube(cfg: Config, name: str, ds: xr.Dataset, message: str) -> str:
    """Commit the cube to its Icechunk repository and return the snapshot id."""
    import icechunk
    from icechunk.xarray import to_icechunk

    path = cube_path(cfg, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    repo = icechunk.Repository.open_or_create(icechunk.local_filesystem_storage(str(path)))
    session = repo.writable_session("main")
    encoding = {v: dict(REFLECTANCE_ENCODING) for v in ds.data_vars if v != "qa"}
    n_time, n_transect, n_distance = (ds.sizes[d] for d in ("time", "transect", "distance"))
    chunks = (min(n_time, 256), min(n_transect, 256), n_distance)
    for var in ds.data_vars:
        encoding.setdefault(var, {})["chunks"] = chunks
    to_icechunk(ds, session, mode="w", encoding=encoding)
    return session.commit(message)


def open_cube(cfg: Config, name: str, ref: str = "main") -> xr.Dataset:
    """Open a stored cube. ``ref`` is a branch name or a snapshot id."""
    import icechunk

    path = cube_path(cfg, name)
    if not path.exists():
        raise FileNotFoundError(f"No cube named {name!r} in {path.parent}. Build it first.")
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(path)))
    try:
        session = repo.readonly_session(ref)
    except Exception:
        session = repo.readonly_session(snapshot_id=ref)
    ds = xr.open_zarr(session.store, consolidated=False)
    for name, coord in ds.coords.items():       # plain Python strings, whatever the stored string type
        if coord.dtype.kind in "TUS":
            ds = ds.assign_coords({name: (coord.dims, np.array([str(v) for v in coord.values], dtype=object))})
    ds.attrs["snapshot_id"] = session.snapshot_id
    return ds
