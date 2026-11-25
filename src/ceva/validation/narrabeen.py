"""Narrabeen–Collaroy, Australia: monthly beach profiles on five lines since 1976.

Turner et al. (2016), Scientific Data 3:160024, WRL/UNSW, http://narrabeen.wrl.unsw.edu.au/.
Microtidal (spring range about 1.3 m), moderate to high-energy, embayed sandy beach.
Elevations are in metres above AHD, which the SDS benchmark (Vos et al. 2023) takes as
mean sea level. The profile file and the transect lines are those distributed with that
benchmark (https://github.com/SatelliteShorelines/SDS_Benchmark).

The satellite transects are the benchmark lines extended ``LANDWARD_M`` behind the
survey origin, so the survey chainage c is c + LANDWARD_M on the satellite transect.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import LineString

from ceva.config import Config
from ceva.shoreline import seaward_edge

NAME = "narrabeen"
BENCHMARK_SITE = "NARRABEEN"
ATTRIBUTION = ("Narrabeen–Collaroy profiles: Turner et al. (2016), Scientific Data 3:160024, WRL UNSW; "
               "as distributed with the SDS benchmark (Vos et al. 2023)")
PROFILES = ("PF1", "PF2", "PF4", "PF6", "PF8")
BENCHMARK_SLOPE = 0.1
LANDWARD_M = 200.0
SEAWARD_M = 600.0
SPACING_M = 10.0
_REPO = "https://raw.githubusercontent.com/SatelliteShorelines/SDS_Benchmark/main/datasets/NARRABEEN"
_GEOD = Geod(ellps="WGS84")
_UA = {"User-Agent": "ceva (+https://github.com/clacrouts/ceva)"}


def raw_dir(cfg: Config) -> Path:
    return cfg.data_dir / "raw" / "insitu" / "narrabeen"


def _files(cfg: Config) -> dict[str, tuple[Path, str]]:
    return {"profiles": (raw_dir(cfg) / "Narrabeen_Profiles.csv", f"{_REPO}/raw/Narrabeen_Profiles.csv"),
            "transects": (raw_dir(cfg) / "NARRABEEN_transects.geojson", f"{_REPO}/NARRABEEN_transects.geojson")}


def download(cfg: Config) -> Path:
    for path, url in _files(cfg).values():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=300) as response:
                path.write_bytes(response.read())
    return raw_dir(cfg)


def transect_name(profile: str) -> str:
    return f"nb-{profile}"


def transects(cfg: Config) -> gpd.GeoDataFrame:
    """The five survey lines, from LANDWARD_M behind the survey origin to SEAWARD_M in front of it."""
    path = _files(cfg)["transects"][0]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run `ceva validate narrabeen --download`.")
    rows = []
    for feature in json.loads(path.read_text())["features"]:
        name = feature["properties"]["name"]
        (lon0, lat0), (lon1, lat1) = feature["geometry"]["coordinates"][0], feature["geometry"]["coordinates"][-1]
        azimuth, _, _ = _GEOD.inv(lon0, lat0, lon1, lat1)
        lon_a, lat_a, _ = _GEOD.fwd(lon0, lat0, azimuth + 180.0, LANDWARD_M)
        lon_b, lat_b, _ = _GEOD.fwd(lon0, lat0, azimuth, SEAWARD_M)
        rows.append({"transect_id": transect_name(name), "profile": name,
                     "geometry": LineString([(lon_a, lat_a), (lon_b, lat_b)])})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def surveys(cfg: Config, level_m: float = 0.0, min_land_run_m: float = 50.0) -> pd.DataFrame:
    """Surveyed position of the ``level_m`` contour (m AHD) for every survey, in transect metres."""
    path = _files(cfg)["profiles"][0]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run `ceva validate narrabeen --download`.")
    data = pd.read_csv(path)
    grid = np.arange(0.0, LANDWARD_M + SEAWARD_M + SPACING_M / 2, SPACING_M)
    rows = []
    for (profile, date), g in data.groupby(["Profile ID", "Date"], sort=False):
        if len(g) < 3:
            continue
        g = g.sort_values("Chainage")
        chainage = g["Chainage"].to_numpy(dtype="float64") + LANDWARD_M
        elevation = g["Elevation"].to_numpy(dtype="float64")
        z = np.interp(grid, chainage, elevation, left=np.nan, right=np.nan)
        x = float(seaward_edge(level_m - z, grid, 0.0, min_land_run_m, leading_gap_is_land=True))
        # Beach-face slope on the surveyed points around mean sea level (10 m apart, so a wide band).
        face = (elevation >= -0.5) & (elevation <= 1.5) & (np.abs(chainage - x) < 60)
        slope = -np.polyfit(chainage[face], elevation[face], 1)[0] if face.sum() >= 2 else np.nan
        rows.append({"transect_id": transect_name(profile), "time": pd.Timestamp(date) + pd.Timedelta(hours=12),
                     "x": x, "slope": slope})
    out = pd.DataFrame(rows)
    return out[np.isfinite(out["x"])].reset_index(drop=True)
