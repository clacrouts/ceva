"""JARKUS, the Netherlands: yearly beach profiles along the whole Dutch coast since 1965.

Rijkswaterstaat, served by Deltares over OPeNDAP. Elevations are relative to NAP,
the Dutch ordnance datum, which is within about 0.1 m of present mean sea level.
The data may be used freely for research provided RIJKSWATERSTAAT is acknowledged.

Profiles are yearly, so this site validates the annual shorelines and the long-term
rates, on a coast with groynes, a sea wall and very large sand nourishments.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString

from ceva.config import Config
from ceva.shoreline import seaward_edge

log = logging.getLogger(__name__)

NAME = "jarkus"
OPENDAP = "https://opendap.deltares.nl/thredds/dodsC/opendap/rijkswaterstaat/jarkus/profiles/transect.nc"
ATTRIBUTION = "JARKUS coastal profiles: RIJKSWATERSTAAT, served by Deltares (opendap.deltares.nl)"
HALF_LENGTH_M = 500.0
FIRST_YEAR = 1984
MIN_YEARS = 25
MIN_SPACING_M = 500.0
MAX_GAP_M = 60.0
_WINDOW = (-1500.0, 2500.0)   # cross-shore range downloaded, metres from the reference pole line


def _fill_small_gaps(z: np.ndarray, spacing: float) -> np.ndarray:
    """Linear interpolation across gaps of at most MAX_GAP_M along the last axis."""
    out = z.copy()
    x = np.arange(z.shape[-1])
    flat = out.reshape(-1, z.shape[-1])
    for row in flat:
        ok = np.isfinite(row)
        if ok.sum() < 2 or ok.all():
            continue
        idx = np.flatnonzero(ok)
        filled = np.interp(x, idx, row[idx], left=np.nan, right=np.nan)
        gap = np.diff(idx)
        for a, g in zip(idx[:-1], gap, strict=True):
            if g * spacing > MAX_GAP_M:
                filled[a + 1:a + g] = np.nan
        row[:] = filled
    return out


@dataclass(frozen=True)
class Area:
    """One JARKUS coastal area ("kustvak") as a validation site."""

    area: str          # JARKUS area name, e.g. "Noord-Holland"
    name: str          # CEVA site key
    prefix: str        # transect id prefix

    @property
    def NAME(self) -> str:   # noqa: N802 - mirrors the module constant of the other sites
        return self.name

    @property
    def ATTRIBUTION(self) -> str:   # noqa: N802
        return ATTRIBUTION

    def raw_path(self, cfg: Config) -> Path:
        return cfg.data_dir / "raw" / "insitu" / "jarkus" / f"{self.area}.nc"

    def download(self, cfg: Config, force: bool = False) -> Path:
        """Save the profiles of one coastal area locally (about 30 MB)."""
        import xarray as xr

        dest = self.raw_path(cfg)
        if dest.exists() and not force:
            with xr.open_dataset(dest) as existing:
                if "time_topo" in existing:
                    return dest
        ds = xr.open_dataset(OPENDAP)
        names = np.char.strip(ds["areaname"].values.astype(str))
        rows = np.flatnonzero(names == self.area)
        cross = ds["cross_shore"].values
        cols = np.flatnonzero((cross >= _WINDOW[0]) & (cross <= _WINDOW[1]))
        subset = ds[["altitude", "x", "y", "id", "time_topo"]].isel(
            alongshore=slice(rows[0], rows[-1] + 1), cross_shore=slice(cols[0], cols[-1] + 1),
            time=ds["time"].dt.year.values >= FIRST_YEAR - 1,
        ).load()
        subset.attrs = {"source": OPENDAP, "attribution": ATTRIBUTION, "area": self.area}
        dest.parent.mkdir(parents=True, exist_ok=True)
        subset.to_netcdf(dest)
        return dest

    def _contours(self, cfg: Config, level_m: float) -> tuple[pd.DataFrame, dict]:
        """Position of the contour for every profile and year, in the JARKUS cross-shore frame."""
        import xarray as xr

        path = self.raw_path(cfg)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run `ceva validate {self.name} --download`.")
        ds = xr.open_dataset(path).load()
        cross = ds["cross_shore"].values
        z = _fill_small_gaps(ds["altitude"].values, float(cross[1] - cross[0]))      # [time, profile, cross]
        position = seaward_edge(level_m - z, cross, 0.0, cfg.annual.min_land_run_m, leading_gap_is_land=True)
        # A profile that stops before reaching the sea gives no shoreline.
        lowest = np.nanmin(np.where(np.isfinite(z), z, np.inf), axis=-1)
        reaches_sea = lowest < level_m - 0.3
        position = np.where(reaches_sea, position, np.nan)
        years = ds["time"].dt.year.values
        t_idx, p_idx = np.nonzero(np.isfinite(position))
        # Survey date of each profile; mid-year when the file does not record it.
        if "time_topo" in ds:
            surveyed = pd.to_datetime(ds["time_topo"].values[t_idx, p_idx])
        else:
            surveyed = pd.DatetimeIndex([pd.NaT] * len(t_idx))
        midyear = pd.to_datetime([f"{y}-07-01" for y in years[t_idx]])
        frame = pd.DataFrame({"profile": ds["id"].values[p_idx].astype(int), "year": years[t_idx],
                              "cross_shore": position[t_idx, p_idx],
                              "time": np.where(pd.isna(surveyed), midyear, surveyed),
                              "dated": ~pd.isna(surveyed)})
        return frame[frame["year"] >= FIRST_YEAR], {"ds": ds, "cross": cross}

    def _selected(self, cfg: Config) -> pd.DataFrame:
        """Profiles with a long record, at least MIN_SPACING_M apart, and where their transect starts."""
        frame, ctx = self._contours(cfg, 0.0)
        stats = frame.groupby("profile")["cross_shore"].agg(n="size", centre="median").reset_index()
        stats = stats[stats["n"] >= MIN_YEARS].sort_values("profile")
        keep, last = [], -np.inf
        for row in stats.itertuples():
            alongshore_m = (row.profile % 1_000_000) * 10.0
            if alongshore_m - last >= MIN_SPACING_M:
                keep.append(row.profile)
                last = alongshore_m
        stats = stats[stats["profile"].isin(keep)].copy()
        stats["start"] = (stats["centre"] / 5.0).round() * 5.0 - HALF_LENGTH_M
        return stats

    def transect_name(self, profile: int) -> str:
        return f"{self.prefix}{profile}"

    def transects(self, cfg: Config) -> gpd.GeoDataFrame:
        """Satellite transects along the selected JARKUS profile lines, 1 km long, land to sea."""
        import xarray as xr

        selected = self._selected(cfg)
        ds = xr.open_dataset(self.raw_path(cfg)).load()
        cross = ds["cross_shore"].values
        ids = ds["id"].values.astype(int)
        to_wgs84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
        rows = []
        for row in selected.itertuples():
            p = int(np.flatnonzero(ids == row.profile)[0])
            a = int(np.argmin(np.abs(cross - row.start)))
            b = int(np.argmin(np.abs(cross - (row.start + 2 * HALF_LENGTH_M))))
            lon, lat = to_wgs84.transform(ds["x"].values[p, [a, b]], ds["y"].values[p, [a, b]])
            rows.append({"transect_id": self.transect_name(row.profile), "profile": row.profile,
                         "alongshore_m": (row.profile % 1_000_000) * 10.0,
                         "geometry": LineString(zip(lon, lat, strict=True))})
        return gpd.GeoDataFrame(rows, crs="EPSG:4326")

    def surveys(self, cfg: Config, level_m: float = 0.0) -> pd.DataFrame:
        """Surveyed position of the contour per transect and year, in metres from the transect start.

        ``time`` is the survey date of that profile (``dated`` False when the file has no date
        and mid-year is assumed).
        """
        selected = self._selected(cfg).set_index("profile")
        frame, _ = self._contours(cfg, level_m)
        frame = frame[frame["profile"].isin(selected.index)].copy()
        frame["x"] = frame["cross_shore"] - frame["profile"].map(selected["start"])
        frame["transect_id"] = frame["profile"].map(self.transect_name)
        frame = frame[(frame["x"] > 0) & (frame["x"] < 2 * HALF_LENGTH_M)]
        return frame[["transect_id", "year", "time", "dated", "x"]].reset_index(drop=True)


NOORD_HOLLAND = Area("Noord-Holland", "jarkus", "jk")
# Walcheren (Zeeland): spring range about 4 m against about 2 m in Noord-Holland.
WALCHEREN = Area("Walcheren", "walcheren", "jw")

# Noord-Holland, as module-level functions (the interface of the other site modules)
raw_path = NOORD_HOLLAND.raw_path
download = NOORD_HOLLAND.download
transects = NOORD_HOLLAND.transects
surveys = NOORD_HOLLAND.surveys
transect_name = NOORD_HOLLAND.transect_name
