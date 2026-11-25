"""Wave-driven water level at the shore, from offshore waves.

Breaking waves raise the water level at the shore and push the swash up the beach,
so a satellite waterline sits higher than the tide alone would put it, most of all
in storms. CEVA corrects this with a relation fitted to surveys (``empirical_level``,
see ceva.validation.waterline_level): the waterline rises by ``hs_coef`` metres per
metre of significant wave height above the transect's usual wave height.

Waves come from the ERA5 ocean-wave reanalysis (0.5°, hourly, 1940 onwards): either
downloaded once from the Copernicus Climate Data Store and read locally (``ceva.era5``,
free with an account, any use), or read through the Open-Meteo marine API, which needs
no account but is for non-commercial use and refuses continental-scale runs.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod

from ceva.config import Config

log = logging.getLogger(__name__)

OPEN_METEO = "https://marine-api.open-meteo.com/v1/marine"
ERA5_MODEL = "era5_ocean"
ATTRIBUTION = "Waves: ERA5 (Hersbach et al. 2020, Copernicus Climate Change Service) via Open-Meteo (CC-BY 4.0)"
_GEOD = Geod(ellps="WGS84")
_UA = {"User-Agent": "ceva (+https://github.com/clacrouts/ceva)"}


def offshore_point(lon_start, lat_start, lon_end, lat_end, seaward_sign, distance_m: float) -> tuple[float, float]:
    """A point ``distance_m`` beyond the seaward ends of the transects, along their mean direction."""
    from ceva.shoreline import NoShoreline

    sign = np.asarray(seaward_sign)
    use = sign != 0
    if not use.any():
        raise NoShoreline("no transect has a known seaward direction")
    lon_land = np.where(sign > 0, lon_start, lon_end)[use]
    lat_land = np.where(sign > 0, lat_start, lat_end)[use]
    lon_sea = np.where(sign > 0, lon_end, lon_start)[use]
    lat_sea = np.where(sign > 0, lat_end, lat_start)[use]
    azimuth, _, _ = _GEOD.inv(lon_land, lat_land, lon_sea, lat_sea)
    mean_azimuth = np.degrees(np.arctan2(np.sin(np.radians(azimuth)).mean(), np.cos(np.radians(azimuth)).mean()))
    lon, lat, _ = _GEOD.fwd(lon_sea.mean(), lat_sea.mean(), mean_azimuth, distance_m)
    return float(lon), float(lat)


def _fetch_open_meteo(lon: float, lat: float, start: dt.date, end: dt.date) -> pd.DataFrame:
    params = {"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "models": ERA5_MODEL,
              "hourly": "wave_height,wave_peak_period,wave_direction", "timezone": "GMT",
              "start_date": start.isoformat(), "end_date": end.isoformat()}
    url = OPEN_METEO + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=120) as response:
        hourly = json.load(response)["hourly"]
    return pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "hs": pd.to_numeric(pd.Series(hourly["wave_height"]), errors="coerce"),
        "tp": pd.to_numeric(pd.Series(hourly["wave_peak_period"]), errors="coerce"),
        "direction": pd.to_numeric(pd.Series(hourly["wave_direction"]), errors="coerce"),
    })


def cache_path(cfg: Config, lon: float, lat: float) -> Path:
    return cfg.data_dir / "raw" / "waves" / f"{ERA5_MODEL}_{lat:+.2f}_{lon:+.2f}.parquet"


def era5_waves(cfg: Config, lon: float, lat: float, start: dt.date, end: dt.date, chunk_years: int = 8) -> pd.DataFrame:
    """Hourly ERA5 waves (hs [m], tp [s], direction [deg]) at a point, cached on disk."""
    path = cache_path(cfg, lon, lat)
    if path.exists():
        cached = pd.read_parquet(path)
        if cached["time"].min().date() <= start and cached["time"].max().date() >= end - dt.timedelta(days=2):
            return cached
    frames = []
    year = start.year
    while year <= end.year:
        last = min(dt.date(year + chunk_years - 1, 12, 31), end)
        frames.append(_fetch_open_meteo(lon, lat, max(start, dt.date(year, 1, 1)), last))
        year += chunk_years
    out = pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def waves_for_unit(cfg: Config, lon_start, lat_start, lon_end, lat_end, seaward_sign,
                   start: dt.date, end: dt.date, attempts: int = 4) -> tuple[pd.DataFrame, tuple[float, float]]:
    """ERA5 waves offshore of a set of transects, moving further out if the grid cell is land.

    With ``waves.source: cds_era5``, the nearest sea cell of the local ERA5 store (``ceva.era5``).
    """
    distance = cfg.waves.offshore_distance_m
    if cfg.waves.source == "cds_era5":
        from ceva.era5 import cds_waves

        lon, lat = offshore_point(lon_start, lat_start, lon_end, lat_end, seaward_sign, distance)
        return cds_waves(cfg, lon, lat, start, end)
    for _ in range(attempts):
        lon, lat = offshore_point(lon_start, lat_start, lon_end, lat_end, seaward_sign, distance)
        waves = era5_waves(cfg, lon, lat, start, end)
        if waves["hs"].notna().mean() > 0.9:
            return waves, (lon, lat)
        log.info("no ERA5 waves at %.3f, %.3f (land cell?); moving offshore", lon, lat)
        distance += 20_000
    raise RuntimeError(f"no ERA5 wave data within {distance / 1000:.0f} km offshore of these transects")


def hs_at(waves: pd.DataFrame, times) -> np.ndarray:
    """Significant wave height interpolated at ``times``."""
    t_wave = waves["time"].to_numpy("datetime64[s]").astype("float64")
    t_img = pd.DatetimeIndex(times).to_numpy("datetime64[s]").astype("float64")
    return np.interp(t_img, t_wave, waves["hs"].interpolate(limit_direction="both").to_numpy())


def tide_range(tide: pd.Series, transect_id: pd.Series) -> pd.Series:
    """Tide range of each transect, as four standard deviations of the tide at image times."""
    return 4.0 * tide.groupby(transect_id).transform("std")


def empirical_level(hs, hs_mean, hs_coef: float) -> np.ndarray:
    """Extra height of the satellite waterline (m) when the waves are above their usual height.

    Fitted to surveys: within every validation site the waterline sits higher by roughly the
    same amount per metre of wave height (set-up, swash, wet sand). Only the departure from
    the transect's mean wave height is used: the mean offset differs between beaches in a way
    the validation sites cannot model, so it is left to the tide correction.
    """
    return (hs_coef * (np.asarray(hs, "float64") - np.asarray(hs_mean, "float64"))).astype("float32")
