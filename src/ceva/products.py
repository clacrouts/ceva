"""Yearly shoreline products: imagery only, and imagery + radar.

Both give, for every transect and year, where the shoreline was (at the reference elevation,
mean sea level by default), as a distance along the transect and as a point:

* ``imagery``: the median of the year's corrected waterlines from Landsat and Sentinel-2.
* ``imagery_radar``: that median and the yearly median of the Sentinel-1 radar waterlines
  (aligned on the imagery), averaged with weights from their precision
  (``ceva.sar_waterlines.combined_yearly``). Years with too few clear images but enough radar
  have a position here only.

Columns: transect_id, year, x (m from the landward end of the transect), n_imagery, n_radar,
se_m (standard error of x, from each source's scatter, the observations that year, and the
radar's year-to-year drift; an estimate, not a validated accuracy), geometry (the shoreline
point, WGS84). Rates: ordinary least squares on the yearly positions of each transect.

Files: ``shorelines_<product>.parquet`` (GeoParquet) and ``rates_<product>.parquet`` next to
the other results.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import Point

from ceva.config import Config
from ceva.engines import change_rates

PRODUCTS = {
    "imagery": "Landsat and Sentinel-2",
    "imagery_radar": "Landsat and Sentinel-2, with Sentinel-1 radar aligned on them",
}
MIN_RATE_YEARS = 5
_GEOD = Geod(ellps="WGS84")


def shoreline_points(transects: gpd.GeoDataFrame, transect_id: pd.Series, x: pd.Series) -> list[Point | None]:
    """The point ``x`` metres from the landward end of each transect (None where it has no seaward side)."""
    t = transects.set_index("transect_id")
    first = t.geometry.map(lambda g: g.coords[0])
    last = t.geometry.map(lambda g: g.coords[-1])
    sign = t["seaward_sign"]
    land = np.where(sign > 0, first, last)
    sea = np.where(sign > 0, last, first)
    land = pd.Series(list(land), index=t.index)
    sea = pd.Series(list(sea), index=t.index)
    ids = transect_id.astype(str).to_numpy()
    if len(ids) == 0:
        return []
    lon0, lat0 = np.array([land[i][:2] for i in ids]).T
    lon1, lat1 = np.array([sea[i][:2] for i in ids]).T
    azimuth, _, _ = _GEOD.inv(lon0, lat0, lon1, lat1)
    lon, lat, _ = _GEOD.fwd(lon0, lat0, azimuth, x.to_numpy(dtype=float))
    oriented = sign.reindex(ids).to_numpy() != 0
    return [Point(a, b) if ok else None for a, b, ok in zip(lon, lat, oriented, strict=True)]


def yearly_products(cfg: Config, transects: gpd.GeoDataFrame, imagery: pd.DataFrame,
                    radar: pd.DataFrame | None) -> dict[str, gpd.GeoDataFrame]:
    """The yearly products of one unit (``imagery_radar`` only when there is radar)."""
    from ceva.sar_waterlines import combined_yearly, source_yearly

    transects = transects.assign(transect_id=transects["transect_id"].astype(str))
    sources = source_yearly(imagery, radar, cfg)
    image = sources["imagery"]
    tables = {"imagery": pd.DataFrame({
        "transect_id": image["transect_id"].astype(str), "year": image["year"], "x": image["x"],
        "n_imagery": image["n_obs"], "n_radar": 0, "se_m": np.sqrt(image["var"])})}
    if "radar" in sources:
        both = combined_yearly(imagery, radar, cfg)
        tables["imagery_radar"] = pd.DataFrame({
            "transect_id": both["transect_id"].astype(str), "year": both["year"], "x": both["x"],
            "n_imagery": both["n_imagery"], "n_radar": both["n_radar"], "se_m": both["se"]})
    out = {}
    for name, table in tables.items():
        table = table[table["transect_id"].isin(transects.loc[transects["seaward_sign"] != 0, "transect_id"])]
        table = table.sort_values(["transect_id", "year"]).reset_index(drop=True)
        table["product"] = name
        out[name] = gpd.GeoDataFrame(table, geometry=shoreline_points(transects, table["transect_id"], table["x"]),
                                     crs="EPSG:4326")
    return out


def product_rates(product: pd.DataFrame, min_years: int = MIN_RATE_YEARS) -> pd.DataFrame:
    """Rate of change (m/yr, positive seaward) of each transect, with its 95 % interval."""
    rates = change_rates(product[["transect_id", "year", "x"]], min_years)
    return rates.assign(product=product["product"].iloc[0] if len(product) else "")


def write_products(out: Path, products: dict[str, gpd.GeoDataFrame]) -> dict[str, Path]:
    """Write each product and its rates; remove the files of a product no longer made."""
    paths = {}
    for name in PRODUCTS:
        shorelines, rates = out / f"shorelines_{name}.parquet", out / f"rates_{name}.parquet"
        if name not in products:
            shorelines.unlink(missing_ok=True)
            rates.unlink(missing_ok=True)
            continue
        products[name].to_parquet(shorelines, index=False)
        product_rates(products[name]).to_parquet(rates, index=False)
        paths[name] = shorelines
    return paths


def load_products(out: Path) -> dict[str, tuple[gpd.GeoDataFrame, pd.DataFrame]]:
    """The products stored in a results folder: name -> (yearly shorelines, rates)."""
    return {name: (gpd.read_parquet(out / f"shorelines_{name}.parquet"), pd.read_parquet(out / f"rates_{name}.parquet"))
            for name in PRODUCTS if (out / f"shorelines_{name}.parquet").exists()}


def coverage(products: dict[str, gpd.GeoDataFrame]) -> dict:
    """How many transect-years each product has, and what radar adds."""
    out = {name: {"transect_years": int(len(p)), "median_se_m": float(p["se_m"].median()) if len(p) else None}
           for name, p in products.items()}
    if "imagery_radar" in products:
        keys = lambda p: set(zip(p["transect_id"], p["year"], strict=True))  # noqa: E731
        out["radar_only_transect_years"] = len(keys(products["imagery_radar"]) - keys(products["imagery"]))
    return out
