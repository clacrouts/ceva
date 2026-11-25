"""Worldwide processing grid.

Tiles are Bing-scheme quadkeys (the Web Mercator tile pyramid). A tile at zoom
``z`` is the ``z``-character prefix of every finer quadkey inside it, so "all
transects of a tile" is a plain string-range query on an index sorted by the
zoom-12 quadkey of each transect's origin.

Nothing here assumes a hemisphere, a UTM zone or a bounding box: tiles never
straddle the antimeridian, and each tile gets its own metric CRS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mercantile
import numpy as np

INDEX_ZOOM = 12
_MERCATOR_LAT_LIMIT = 85.05112878
_METRES_PER_DEGREE = 111_320.0


def quadkeys(lon, lat, zoom: int = INDEX_ZOOM) -> np.ndarray:
    """Quadkey of the tile containing each (lon, lat), vectorised.

    Latitudes beyond the Web Mercator limit (about 85.05°) fall in the last tile row.
    """
    lon = np.atleast_1d(np.asarray(lon, dtype="float64"))
    lat = np.atleast_1d(np.asarray(lat, dtype="float64"))
    n = 1 << zoom
    x = np.floor((lon + 180.0) / 360.0 * n).astype("int64")
    sin_lat = np.sin(np.radians(np.clip(lat, -_MERCATOR_LAT_LIMIT, _MERCATOR_LAT_LIMIT)))
    y = np.floor((0.5 - np.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * np.pi)) * n).astype("int64")
    x = np.clip(x, 0, n - 1)
    y = np.clip(y, 0, n - 1)
    shifts = np.arange(zoom - 1, -1, -1)
    digits = ((x[:, None] >> shifts) & 1) + 2 * ((y[:, None] >> shifts) & 1)
    codes = np.ascontiguousarray((digits + ord("0")).astype("uint8"))
    return codes.view(f"S{zoom}").ravel().astype(str)


def working_epsg(lon: float, lat: float) -> int:
    """Metric CRS for a location: UTM between 80°S and 84°N, UPS near the poles."""
    if lat >= 84.0:
        return 32661
    if lat <= -80.0:
        return 32761
    zone = min(int((lon + 180.0) // 6) + 1, 60)
    return (32600 if lat >= 0 else 32700) + zone


@dataclass(frozen=True, order=True)
class Tile:
    """One processing tile, identified by its quadkey."""

    quadkey: str

    def __post_init__(self) -> None:
        if not self.quadkey or set(self.quadkey) - set("0123"):
            raise ValueError(f"{self.quadkey!r} is not a quadkey (digits 0-3 only)")

    @property
    def zoom(self) -> int:
        return len(self.quadkey)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """west, south, east, north in degrees."""
        b = mercantile.bounds(mercantile.quadkey_to_tile(self.quadkey))
        return (b.west, b.south, b.east, b.north)

    @property
    def centre(self) -> tuple[float, float]:
        w, s, e, n = self.bounds
        return ((w + e) / 2.0, (s + n) / 2.0)

    @property
    def epsg(self) -> int:
        """The single metric CRS used for every raster and transect of this tile."""
        return working_epsg(*self.centre)

    def buffered_bounds(self, buffer_m: float) -> tuple[float, float, float, float]:
        """Bounds grown by ``buffer_m`` metres, clamped to the valid lon/lat range.

        At the antimeridian the buffer is cut at ±180° rather than wrapped.
        """
        w, s, e, n = self.bounds
        dlat = buffer_m / _METRES_PER_DEGREE
        widest_lat = min(max(abs(s), abs(n)), 89.0)
        dlon = buffer_m / (_METRES_PER_DEGREE * math.cos(math.radians(widest_lat)))
        return (max(w - dlon, -180.0), max(s - dlat, -90.0), min(e + dlon, 180.0), min(n + dlat, 90.0))

    def index_range(self, index_zoom: int = INDEX_ZOOM) -> tuple[str, str]:
        """Smallest and largest ``index_zoom`` quadkey inside this tile (inclusive)."""
        pad = index_zoom - self.zoom
        if pad < 0:
            raise ValueError(f"tile zoom {self.zoom} is finer than the index zoom {index_zoom}")
        return self.quadkey + "0" * pad, self.quadkey + "3" * pad


def tiles_for_bbox(bbox: tuple[float, float, float, float], zoom: int) -> list[Tile]:
    """Tiles at ``zoom`` intersecting a lon/lat box. West > east means the box
    crosses the antimeridian."""
    w, s, e, n = bbox
    return sorted(Tile(mercantile.quadkey(t)) for t in mercantile.tiles(w, s, e, n, zooms=zoom))
