"""Perranporth and Slapton Sands, south-west England: beach profiles since 2007.

Konstantinou et al. (2023), "Satellite-based shoreline detection along high-energy
macrotidal coasts and influence of beach state", Marine Geology 462:107082; data on
Zenodo, doi:10.5281/zenodo.7383895 (CC-BY-4.0), from the South West Regional Coastal
Monitoring Programme and the University of Plymouth.

* Perranporth: macrotidal (spring range about 6.3 m), high-energy, dissipative to
  intermediate sandy beach facing west-north-west.
* Slapton Sands: macrotidal (spring range about 4.3 m), steep fine-gravel barrier in
  Start Bay, facing east-south-east, sheltered from the dominant westerly waves.

Points are in UTM zone 30N; elevations in metres above Ordnance Datum Newlyn, within about
0.1-0.2 m of present mean sea level there. The satellite transects follow the surveyed
profile lines: the main axis of every point ever surveyed on a line, from ``LANDWARD_M``
behind its landward end to ``SEAWARD_M`` beyond its seaward end.
"""
from __future__ import annotations

import datetime as dt
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString

from ceva.config import Config
from ceva.shoreline import seaward_edge

ATTRIBUTION = ("Perranporth and Slapton Sands profiles: Konstantinou et al. (2023), Zenodo "
               "doi:10.5281/zenodo.7383895 (CC-BY-4.0); South West Regional Coastal Monitoring Programme")
ZENODO = "https://zenodo.org/records/7383895/files"
EPSG_UTM = 32630
LANDWARD_M = 100.0
SEAWARD_M = 300.0
SPACING_M = 2.0
MAX_OFF_LINE_M = 25.0      # surveyed points farther than this from the line are not on it
MIN_SURVEYS = 30
_UA = {"User-Agent": "ceva (+https://github.com/clacrouts/ceva)"}


@dataclass(frozen=True)
class Site:
    name: str          # CEVA site key
    file: str          # file on Zenodo
    group: str         # top-level group in the MATLAB file
    prefix: str        # transect id prefix
    min_land_run_m: float

    # -- the interface the validation code expects from a site module
    @property
    def NAME(self) -> str:   # noqa: N802 - mirrors the module constant of the other sites
        return self.name

    @property
    def ATTRIBUTION(self) -> str:   # noqa: N802
        return ATTRIBUTION

    def raw_path(self, cfg: Config) -> Path:
        return cfg.data_dir / "raw" / "insitu" / "ukmacro" / self.file

    def download(self, cfg: Config) -> Path:
        path = self.raw_path(cfg)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(f"{ZENODO}/{self.file}?download=1", headers=_UA)
            with urllib.request.urlopen(request, timeout=1800) as response:
                path.write_bytes(response.read())
        return path

    def transects(self, cfg: Config) -> gpd.GeoDataFrame:
        lines = _lines(str(self.raw_path(cfg)), self.group)
        to_wgs84 = Transformer.from_crs(EPSG_UTM, 4326, always_xy=True)
        rows = []
        for profile, line in lines.items():
            (x0, y0), (x1, y1) = line["start"], line["end"]
            lon, lat = to_wgs84.transform([x0, x1], [y0, y1])
            rows.append({"transect_id": f"{self.prefix}-{profile}", "profile": profile,
                         "geometry": LineString(zip(lon, lat, strict=True))})
        return gpd.GeoDataFrame(rows, crs="EPSG:4326")

    def surveys(self, cfg: Config, level_m: float = 0.0) -> pd.DataFrame:
        """Surveyed position of the ``level_m`` contour (m ODN) for every survey, in transect metres."""
        path = str(self.raw_path(cfg))
        lines = _lines(path, self.group)
        rows = []
        for profile, line in lines.items():
            origin, axis = np.asarray(line["start"]), np.asarray(line["axis"])
            grid = np.arange(0.0, line["length"] + SPACING_M / 2, SPACING_M)
            for when, east, north, elev in _surveys(path, self.group)[profile]:
                d = np.column_stack([east, north]) - origin
                along = d @ axis
                off = np.abs(d @ np.array([-axis[1], axis[0]]))
                ok = np.isfinite(elev) & (off <= MAX_OFF_LINE_M)
                if ok.sum() < 5:
                    continue
                order = np.argsort(along[ok])
                s, z = along[ok][order], elev[ok][order]
                profile_z = np.interp(grid, s, z, left=np.nan, right=np.nan)
                profile_z[(grid < s[0]) | (grid > s[-1])] = np.nan
                x = float(seaward_edge(level_m - profile_z, grid, 0.0, self.min_land_run_m, leading_gap_is_land=True))
                if not np.isfinite(x) or np.nanmin(profile_z) > level_m - 0.2:
                    continue                                 # the survey stops before the contour
                face = (z >= level_m - 0.5) & (z <= level_m + 1.5) & (np.abs(s - x) < 60)
                slope = -np.polyfit(s[face], z[face], 1)[0] if face.sum() >= 3 else np.nan
                rows.append({"transect_id": f"{self.prefix}-{profile}", "time": when, "x": x, "slope": slope})
        return pd.DataFrame(rows, columns=["transect_id", "time", "x", "slope"])


def _datenum(value: float) -> pd.Timestamp:
    """MATLAB datenum to a timestamp; whole days (no time recorded) are taken at noon."""
    day = dt.datetime.fromordinal(int(value)) - dt.timedelta(days=366)
    frac = value - int(value)
    return pd.Timestamp(day) + (pd.Timedelta(days=frac) if frac > 0 else pd.Timedelta(hours=12))


@lru_cache(maxsize=4)
def _surveys(path: str, group: str) -> dict[str, list[tuple]]:
    """Every survey of every profile line: (time, easting, northing, elevation)."""
    import h5py

    out: dict[str, list[tuple]] = {}
    with h5py.File(path, "r") as h:
        profiles = h[group]["profData"]
        for name in profiles:
            g = profiles[name]
            refs = {k: g[k][()].ravel() for k in ("dnum", "lon", "lat", "elev")}
            surveys = []
            for i in range(len(refs["dnum"])):
                dnum = np.ravel(h[refs["dnum"][i]][()])
                east = np.ravel(h[refs["lon"][i]][()]).astype("float64")     # "lon"/"lat" hold UTM easting/northing
                north = np.ravel(h[refs["lat"][i]][()]).astype("float64")
                elev = np.ravel(h[refs["elev"][i]][()]).astype("float64")
                if dnum.size and east.size == elev.size == north.size and east.size >= 5:
                    surveys.append((_datenum(float(dnum[0])), east, north, elev))
            if len(surveys) >= MIN_SURVEYS:
                out[name] = surveys
    return out


@lru_cache(maxsize=4)
def _lines(path: str, group: str) -> dict[str, dict]:
    """A straight line per profile: the main axis of all its points, oriented land to sea."""
    lines = {}
    for name, surveys in _surveys(path, group).items():
        pts = np.vstack([np.column_stack([e, n]) for _, e, n, _ in surveys])
        z = np.concatenate([el for _, _, _, el in surveys])
        ok = np.isfinite(pts).all(axis=1) & np.isfinite(z)
        pts, z = pts[ok], z[ok]
        centre = pts.mean(axis=0)
        axis = np.linalg.svd(pts - centre, full_matrices=False)[2][0]
        s = (pts - centre) @ axis
        if np.corrcoef(s, z)[0, 1] > 0:              # elevation must fall seaward
            axis, s = -axis, -s
        lo, hi = np.percentile(s, [1, 99])
        start = centre + (lo - LANDWARD_M) * axis
        end = centre + (hi + SEAWARD_M) * axis
        lines[name] = {"start": tuple(start), "end": tuple(end), "axis": tuple(axis),
                       "length": float(hi - lo + LANDWARD_M + SEAWARD_M)}
    return lines


PERRANPORTH = Site("perranporth", "PPTdata.mat", "PPT", "pp", 50.0)
SLAPTON = Site("slapton", "SLSdata.mat", "SLS", "ss", 20.0)
