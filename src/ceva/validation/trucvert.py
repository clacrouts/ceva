"""Truc Vert, France: 16 years of beach surveys on a high-energy meso-macrotidal coast.

Castelle, Bujan, Marieu, Ferreira (2020), Scientific Data 7:410, data at
https://osf.io/jftw8/. Each survey is a digital elevation model on a fixed local
grid (2 m cross-shore, 20 m alongshore), with elevations relative to mean sea
level. It is also one of the sites of the satellite-derived-shoreline benchmark
of Vos et al. (2023).

The satellite transects used here are the survey grid lines themselves, extended
landward and seaward, so satellite and survey positions share one axis.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import LineString

from ceva.config import Config
from ceva.shoreline import seaward_edge

log = logging.getLogger(__name__)

NAME = "trucvert"
OSF_FILES = "https://api.osf.io/v2/nodes/jftw8/files/osfstorage/?page[size]=100"
ATTRIBUTION = ("Truc Vert beach surveys: Castelle, Bujan, Marieu and Ferreira (2020), "
               "Scientific Data 7:410, https://osf.io/jftw8/")
MHWS_M = 1.5          # mean high water springs above mean sea level (Vos et al. 2023)
BENCHMARK_SLOPE = 0.05
X_START, X_END = -180.0, 820.0   # extent of the satellite transects in the survey's cross-shore frame (m)
_GEOD = Geod(ellps="WGS84")
_UA = {"User-Agent": "ceva (+https://github.com/clacrouts/ceva)"}


def raw_dir(cfg: Config) -> Path:
    return cfg.data_dir / "raw" / "insitu" / "trucvert"


def download(cfg: Config) -> Path:
    """Fetch every survey file from OSF (about 250 MB); files already present are skipped."""
    dest = raw_dir(cfg)
    dest.mkdir(parents=True, exist_ok=True)
    files, url = [], OSF_FILES
    while url:
        with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=60) as response:
            page = json.load(response)
        files += [(f["attributes"]["name"], f["links"]["download"], f["attributes"].get("size") or 0)
                  for f in page["data"] if f["attributes"]["kind"] == "file"]
        url = page["links"].get("next")

    def fetch(item: tuple[str, str, int]) -> None:
        name, link, size = item
        target = dest / name
        if target.exists() and target.stat().st_size == size:
            return
        with urllib.request.urlopen(urllib.request.Request(link, headers=_UA), timeout=120) as response:
            target.write_bytes(response.read())

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(fetch, files))
    return dest


def _grid(cfg: Config) -> dict[str, np.ndarray]:
    import netCDF4

    path = raw_dir(cfg) / "Grids.nc"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run `ceva validate trucvert --download`.")
    with netCDF4.Dataset(path) as nc:
        return {
            "x": np.array(nc["xlg"][:, 0], dtype="float64"),     # cross-shore, increasing seaward
            "y": np.array(nc["ylg"][0, :], dtype="float64"),     # alongshore
            "lon": np.array(nc["long"][:], dtype="float64"),     # [x, y]
            "lat": np.array(nc["latg"][:], dtype="float64"),
        }


def _columns(grid: dict[str, np.ndarray], every: int) -> np.ndarray:
    return np.arange(0, grid["y"].size, every)


def transect_name(alongshore_m: float) -> str:
    return f"tv{int(round(alongshore_m)):+05d}"


def transects(cfg: Config, every: int = 5) -> gpd.GeoDataFrame:
    """One transect per ``every``-th survey grid line (5 gives one every 100 m), land to sea."""
    grid = _grid(cfg)
    rows = []
    for j in _columns(grid, every):
        lon0, lat0 = grid["lon"][0, j], grid["lat"][0, j]
        azimuth, _, _ = _GEOD.inv(lon0, lat0, grid["lon"][-1, j], grid["lat"][-1, j])
        lon_a, lat_a, _ = _GEOD.fwd(lon0, lat0, azimuth, X_START - grid["x"][0])
        lon_b, lat_b, _ = _GEOD.fwd(lon0, lat0, azimuth, X_END - grid["x"][0])
        rows.append({"transect_id": transect_name(grid["y"][j]), "alongshore_m": float(grid["y"][j]),
                     "geometry": LineString([(lon_a, lat_a), (lon_b, lat_b)])})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def surveys(cfg: Config, level_m: float = 0.0, every: int = 5, min_land_run_m: float = 50.0) -> pd.DataFrame:
    """Surveyed position of an elevation contour on each transect, for every survey.

    Uses the same rule as the satellite shorelines (seaward edge of the beach), with
    the contour ``level_m`` in metres above mean sea level. Positions are metres from
    the landward end of the transects returned by ``transects``. Also returns the
    beach-face slope between mean sea level and mean high water springs.
    """
    import netCDF4

    grid = _grid(cfg)
    columns = _columns(grid, every)
    names = np.array([transect_name(y) for y in grid["y"][columns]])
    distance = grid["x"] - X_START
    rows = []
    for path in sorted(raw_dir(cfg).glob("*_TrucVert_*.nc")):
        with netCDF4.Dataset(path) as nc:
            if "zg" not in nc.variables:
                continue
            z = np.ma.filled(nc["zg"][:].astype("float64"), np.nan)
        z[z < -100] = np.nan
        z = z[:, columns].T                                                # [transect, x]
        if not np.isfinite(z).any():
            continue
        x = seaward_edge(level_m - z, distance, 0.0, min_land_run_m, leading_gap_is_land=True)
        slope = np.full(len(columns), np.nan)
        for i in range(len(columns)):
            face = np.isfinite(z[i]) & (z[i] >= 0.0) & (z[i] <= MHWS_M) & (np.abs(distance - x[i]) < 150)
            if face.sum() >= 5:
                slope[i] = -np.polyfit(distance[face], z[i][face], 1)[0]
        when = pd.Timestamp(path.stem[-10:]) + pd.Timedelta(hours=12)
        rows.append(pd.DataFrame({"transect_id": names, "time": when, "x": x, "slope": slope,
                                  "campaign": path.name.split("_")[0]}))
    out = pd.concat(rows, ignore_index=True)
    return out[np.isfinite(out["x"])].reset_index(drop=True)
