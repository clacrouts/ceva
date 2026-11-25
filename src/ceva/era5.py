"""ERA5 ocean waves from the Copernicus Climate Data Store, downloaded once for a region and read locally.

The Open-Meteo API serves the same ERA5 waves, but its free use is non-commercial and limited: a
continent's units need decades of hourly data at hundreds of points, which it refuses (HTTP 429).
Here the hourly waves of a box (``waves.cds.area``) on the 0.5° ERA5 wave grid are requested year by
year (``ceva waves download``), then the sea cells near the coast are written to one local Zarr store,
cell by cell, so that a cell's whole record is a few reads (``ceva waves build``). A unit then takes
the series of the sea cell nearest its offshore point.

Account: the CDS key in ``~/.config/ceva/tokens/cds.env`` (``CDSAPI_URL``, ``CDSAPI_KEY``); the ERA5
licence must be accepted once on the dataset page (the API answers "required licences not accepted"
until then). Data: ERA5, Hersbach et al. (2020), Copernicus Climate Change Service, Copernicus licence.
"""
from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ceva.config import Config
from ceva.credentials import Credentials, load_credentials

log = logging.getLogger(__name__)

DATASET = "reanalysis-era5-single-levels"
VARIABLES = {     # CEVA column -> CDS variable, and the short name in the files
    "hs": ("significant_height_of_combined_wind_waves_and_swell", "swh"),
    "tp": ("peak_wave_period", "pp1d"),
    "direction": ("mean_wave_direction", "mwd"),
}
LICENCE_PAGE = f"https://cds.climate.copernicus.eu/datasets/{DATASET}?tab=download#manage-licences"
ATTRIBUTION = "Waves: ERA5 (Hersbach et al. 2020), Copernicus Climate Change Service (C3S) Climate Data Store"


def _tag(cfg: Config) -> str:
    n, w, s, e = cfg.waves.cds.area
    return f"N{n:g}_W{w:g}_S{s:g}_E{e:g}"


def year_path(cfg: Config, year: int) -> Path:
    return cfg.data_dir / "raw" / "era5_waves" / _tag(cfg) / f"{year}.nc"


def store_path(cfg: Config) -> Path:
    return cfg.data_dir / "waves" / f"era5_{_tag(cfg)}.zarr"


def client(creds: Credentials | None = None):
    """A CDS client with the key of ``cds.env`` (no ~/.cdsapirc needed)."""
    import cdsapi

    creds = creds if creds is not None else load_credentials()
    if not creds.has("CDSAPI_URL", "CDSAPI_KEY"):
        raise RuntimeError("no CDS key: put CDSAPI_URL and CDSAPI_KEY in ~/.config/ceva/tokens/cds.env")
    return cdsapi.Client(url=creds.get("CDSAPI_URL"), key=creds.get("CDSAPI_KEY"), quiet=True, progress=False)


def request(cfg: Config, year: int) -> dict:
    """The CDS request of one year of hourly waves over the configured box."""
    return {
        "product_type": ["reanalysis"],
        "variable": [VARIABLES[v][0] for v in cfg.waves.cds.variables],
        "year": [str(year)], "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)], "time": [f"{h:02d}:00" for h in range(24)],
        "area": list(cfg.waves.cds.area), "data_format": "netcdf", "download_format": "unarchived",
    }


def download(cfg: Config, years, cds=None, progress: Callable[[int, int], None] | None = None) -> list[Path]:
    """Download the years not yet on disk (one request each; the CDS queue sets the pace). Returns the files."""
    years = list(years)
    paths = []
    for k, year in enumerate(years, 1):
        path = year_path(cfg, year)
        if not path.exists():
            cds = cds if cds is not None else client()
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_name(path.name + ".part")
            log.info("ERA5 waves %d: requesting", year)
            cds.retrieve(DATASET, request(cfg, year), str(part))
            part.replace(path)
        paths.append(path)
        if progress is not None:
            progress(k, len(years))
    return paths


def _open_year(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path)
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if "expver" in ds.coords and "time" not in ds["expver"].dims:   # newest months: a scalar label only
        ds = ds.drop_vars("expver")
    return ds


def coastal_cells(valid: np.ndarray, reach: int) -> np.ndarray:
    """Sea cells within ``reach`` cells of a land (never valid) cell: where waves meet a coast."""
    from scipy.ndimage import binary_dilation

    land = ~valid
    near_land = binary_dilation(land, structure=np.ones((3, 3), bool), iterations=reach) if reach > 0 else land
    return valid & near_land


def build_store(cfg: Config, years, chunk_cells: int = 64) -> Path:
    """Write the coastal sea cells of the downloaded years into one Zarr store: (time, cell), per variable."""
    years = sorted(years)
    first = _open_year(year_path(cfg, years[0]))
    short = [VARIABLES[v][1] for v in cfg.waves.cds.variables]
    valid = first[short[0]].notnull().any("time").values                # sea, wherever waves exist
    keep = coastal_cells(valid, cfg.waves.cds.coastal_reach_cells)
    rows, cols = np.nonzero(keep)
    lat, lon = first["latitude"].values[rows], first["longitude"].values[cols]
    path = store_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    for k, year in enumerate(years):
        ds = _open_year(year_path(cfg, year))
        data = {col: (("time", "cell"), ds[name].values[:, rows, cols].astype("float32"))
                for col, name in zip(cfg.waves.cds.variables, short, strict=True)}
        out = xr.Dataset(data, coords={"time": ds["time"].values, "lat": ("cell", lat), "lon": ("cell", lon)})
        out.attrs.update(source=ATTRIBUTION, area=list(cfg.waves.cds.area))
        if k == 0:
            encoding = {v: {"chunks": (len(out["time"]), chunk_cells)} for v in data}
            out.to_zarr(path, mode="w", encoding=encoding, consolidated=True)
        else:
            out.to_zarr(path, append_dim="time", consolidated=True)
        log.info("ERA5 waves %d: %d coastal cells stored", year, len(rows))
    _open_store.cache_clear()
    return path


@lru_cache(maxsize=2)
def _open_store(path: str) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=True)


def cds_waves(cfg: Config, lon: float, lat: float, start: dt.date, end: dt.date) -> tuple[pd.DataFrame, tuple]:
    """Hourly waves (hs, tp, direction) of the stored sea cell nearest (lon, lat), and that cell."""
    path = store_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"no ERA5 wave store {path}: run `ceva waves download` then `ceva waves build`")
    ds = _open_store(str(path))
    cell_lat, cell_lon = ds["lat"].values, ds["lon"].values
    distance = np.hypot((cell_lon - lon) * np.cos(np.radians(lat)), cell_lat - lat)
    i = int(np.argmin(distance))
    series = ds.isel(cell=i).sel(time=slice(str(start), f"{end} 23:59")).load()
    frame = pd.DataFrame({"time": pd.DatetimeIndex(series["time"].values)})
    for col in ("hs", "tp", "direction"):
        frame[col] = series[col].values.astype("float64") if col in series else np.nan
    return frame, (float(cell_lon[i]), float(cell_lat[i]))
