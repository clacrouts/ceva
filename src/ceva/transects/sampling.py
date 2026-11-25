"""Sample points along transects, on the ellipsoid.

Working with geodesic distances along each transect avoids any map projection,
so the same code is exact in every UTM zone, across the antimeridian and near
the poles.
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from pyproj import Geod

_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class TransectSamples:
    transect_id: np.ndarray   # [N]
    distance: np.ndarray      # [K] metres from the first vertex of each transect
    lon: np.ndarray           # [N, K]; NaN beyond the end of a shorter transect
    lat: np.ndarray           # [N, K]
    length: np.ndarray        # [N] metres
    start: np.ndarray         # [N, 2] lon, lat of the first vertex
    end: np.ndarray           # [N, 2] lon, lat of the last vertex

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (float(np.nanmin(self.lon)), float(np.nanmin(self.lat)),
                float(np.nanmax(self.lon)), float(np.nanmax(self.lat)))


def sample_transects(transects: gpd.GeoDataFrame, spacing_m: float, id_column: str = "transect_id") -> TransectSamples:
    """Points every ``spacing_m`` metres along the straight line joining each
    transect's first and last vertex."""
    if transects.empty:
        raise ValueError("no transects to sample")
    if transects.crs is None:
        raise ValueError("transects need a CRS")
    if transects.crs.to_epsg() != 4326:
        transects = transects.to_crs(4326)

    coords = [np.asarray(g.coords) for g in transects.geometry]
    start = np.array([c[0][:2] for c in coords], dtype="float64")
    end = np.array([c[-1][:2] for c in coords], dtype="float64")
    azimuth, _, length = _GEOD.inv(start[:, 0], start[:, 1], end[:, 0], end[:, 1])

    n = len(coords)
    k = int(np.floor(length.max() / spacing_m + 1e-9)) + 1
    distance = np.arange(k, dtype="float64") * spacing_m
    lon, lat, _ = _GEOD.fwd(
        np.repeat(start[:, 0], k), np.repeat(start[:, 1], k), np.repeat(azimuth, k), np.tile(distance, n)
    )
    lon, lat = lon.reshape(n, k), lat.reshape(n, k)
    beyond = distance[None, :] > length[:, None] + 1e-6
    lon[beyond] = np.nan
    lat[beyond] = np.nan
    return TransectSamples(
        transect_id=transects[id_column].astype(str).to_numpy(),
        distance=distance, lon=lon, lat=lat, length=np.asarray(length), start=start, end=end,
    )
