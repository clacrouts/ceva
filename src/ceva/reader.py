"""Read satellite bands at arbitrary points, on each image's own grid.

Images are never resampled onto a common raster. Each band is read once, in
its native projection and resolution, and interpolated at the requested
points. That keeps one interpolation step between the sensor and the
shoreline, whatever mix of sensors, UTM zones and pixel sizes is involved.
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Any

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from scipy.ndimage import map_coordinates

from ceva.catalog.models import MASK_BAND, Scene
from ceva.config import Config
from ceva.credentials import Credentials

# GDAL fixes the size of its HTTP requests and of its block cache when the first remote file of the
# process is opened, whatever options later reads pass: set them for the whole process, here, before
# any read. Small requests (GDAL's default 16 kB) read a window of a JPEG 2000 file (which has no tile
# index) with about 9 MB instead of 55 MB with 4 MB requests; the cache keeps the blocks that units
# on one tile share while they read the same file in turn.
PROCESS_GDAL = {
    "CPL_VSIL_CURL_CHUNK_SIZE": "16384",
    "CPL_VSIL_CURL_CACHE_SIZE": str(1024 * 2**20),
}
for _key, _value in PROCESS_GDAL.items():
    os.environ.setdefault(_key, _value)

GDAL_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff,.jp2",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_TIMEOUT": "60",
    "VSI_CACHE": "TRUE",
}

# Landsat QA_PIXEL: only the low byte is kept (bit 0 fill, 1 dilated cloud, 2 cirrus,
# 3 cloud, 4 cloud shadow, 5 snow, 6 clear, 7 water).
LANDSAT_FILL = 1
S2_NODATA = 0
READ_BLOCK = 512   # pixels; matches the internal tiling of the cloud-optimised files

_SIGN_LOCK = threading.Lock()


def gdal_environment(provider: str, cfg: Config, creds: Credentials) -> dict[str, Any]:
    """GDAL options to open this provider's assets (pass to ``rasterio.Env``).

    S3 keys travel in a rasterio ``AWSSession`` (under ``session``), the only way rasterio
    accepts them; they are never set as process-wide environment variables.
    """
    options: dict[str, Any] = dict(GDAL_OPTIONS)
    if provider in cfg.providers:
        options.update(cfg.providers[provider].gdal)
    if provider == "cdse":
        from rasterio.session import AWSSession

        endpoint = cfg.providers["cdse"].s3_endpoint
        options.update(
            session=AWSSession(aws_access_key_id=creds.get("CDSE_S3_ACCESS_KEY"),
                               aws_secret_access_key=creds.get("CDSE_S3_SECRET_KEY"),
                               region_name="default", endpoint_url=endpoint),
            AWS_S3_ENDPOINT=endpoint,
            AWS_VIRTUAL_HOSTING="FALSE",
            AWS_HTTPS="YES",
        )
    return options


def resolve_href(provider: str, href: str) -> str:
    """Turn a stored asset link into something GDAL can open right now."""
    if provider == "planetary_computer":
        import planetary_computer

        # Tokens are cached per storage container. Without the lock, every worker
        # thread asks for the same token at once and the service throttles them all.
        with _SIGN_LOCK:
            return planetary_computer.sign(href)
    if href.startswith("s3://"):
        return "/vsis3/" + href[len("s3://"):]
    return href


@lru_cache(maxsize=64)
def _to_crs(wkt: str) -> Transformer:
    return Transformer.from_crs("EPSG:4326", wkt, always_xy=True)


def read_at_points(path: str, lon: np.ndarray, lat: np.ndarray, method: str = "bilinear",
                   nodata: float | None = 0) -> np.ndarray:
    """Band 1 of ``path`` at lon/lat points.

    ``bilinear`` returns float32 with NaN outside the image, on nodata, and
    wherever a nodata pixel takes part in the interpolation. ``nearest``
    returns the raw values, with 0 outside the image.
    """
    shape = lon.shape
    lon, lat = np.ravel(lon), np.ravel(lat)
    finite = np.isfinite(lon) & np.isfinite(lat)
    with rasterio.open(path) as src:
        out = (np.zeros(lon.shape, dtype=src.dtypes[0]) if method == "nearest"
               else np.full(lon.shape, np.nan, dtype="float32"))
        xs, ys = _to_crs(src.crs.to_wkt()).transform(lon[finite], lat[finite])
        inverse = ~src.transform
        cols = np.full(lon.shape, np.nan)
        rows = np.full(lon.shape, np.nan)
        cols[finite] = inverse.a * xs + inverse.b * ys + inverse.c - 0.5   # pixel-centre coordinates
        rows[finite] = inverse.d * xs + inverse.e * ys + inverse.f - 0.5
        inside = finite & (rows > -0.5) & (rows < src.height - 0.5) & (cols > -0.5) & (cols < src.width - 0.5)
        if not inside.any():
            return out.reshape(shape)
        if nodata is None:
            nodata = src.nodata
        # Read only the image blocks that hold points: a coastline is a thin line
        # across the image, so its bounding box is mostly pixels nobody needs.
        idx = np.flatnonzero(inside)
        cell = (np.floor(rows[idx] / READ_BLOCK).astype(np.int64) * 1_000_000
                + np.floor(cols[idx] / READ_BLOCK).astype(np.int64))
        for key in np.unique(cell):
            sel = idx[cell == key]
            r0 = max(int(np.floor(rows[sel].min())) - 1, 0)
            r1 = min(int(np.ceil(rows[sel].max())) + 2, src.height)
            c0 = max(int(np.floor(cols[sel].min())) - 1, 0)
            c1 = min(int(np.ceil(cols[sel].max())) + 2, src.width)
            data = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
            rr, cc = rows[sel] - r0, cols[sel] - c0
            if method == "nearest":
                ri = np.clip(np.rint(rr).astype(int), 0, data.shape[0] - 1)
                ci = np.clip(np.rint(cc).astype(int), 0, data.shape[1] - 1)
                out[sel] = data[ri, ci]
            else:
                values = data.astype("float32")
                if nodata is not None:
                    values[data == nodata] = np.nan
                out[sel] = map_coordinates(values, [rr, cc], order=1, mode="nearest", prefilter=False)
    return out.reshape(shape)


def encode_qa(raw: np.ndarray, mask_kind: str) -> np.ndarray:
    """Quality flags as one byte per sample (see module constants)."""
    if mask_kind == "landsat_qa_pixel":
        low = (raw.astype("uint16") & 0xFF).astype("uint8")
        return np.where(raw == 0, LANDSAT_FILL, low).astype("uint8")
    return raw.astype("uint8")   # Sentinel-2 scene classification, 0 = no data


def sample_scene(scene: Scene, lon: np.ndarray, lat: np.ndarray, bands: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Reflectance of ``bands`` and the quality byte (key ``qa``) at the points.

    Must run inside ``rasterio.Env(**gdal_environment(...))``.
    """
    out: dict[str, np.ndarray] = {}
    for name in bands:
        asset = scene.bands[name]
        stored = read_at_points(resolve_href(scene.provider, asset.href), lon, lat, "bilinear", asset.nodata)
        out[name] = stored * asset.scale + asset.offset
    raw = read_at_points(resolve_href(scene.provider, scene.bands[MASK_BAND].href), lon, lat, "nearest")
    out["qa"] = encode_qa(raw, scene.mask_kind)
    return out
