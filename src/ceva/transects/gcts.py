"""Global Coastal Transect System (GCTS): download, tile index, per-tile loading.

GCTS (Calkoen et al. 2025, CC-BY-4.0) provides about 11 million cross-shore
transects worldwide, 100 m apart and 2 km long, with stable ``transect_id``s.
Using it instead of transects cast from our own shoreline gives every region
the same reference geometry and removes artefacts at the edge of an area of
interest.

The published files are one GeoParquet per continent with three large row
groups in no spatial order, and their ``quadkey`` column is not the tile of
the transect. ``build_index`` therefore rewrites each file sorted by the
zoom-12 quadkey of the transect origin, in small row groups, so that loading
one tile reads a few kilobytes instead of the whole continent.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from ceva.config import Config
from ceva.grid import INDEX_ZOOM, Tile, quadkeys, tiles_for_bbox

log = logging.getLogger(__name__)

CONTINENTS = ("Africa", "Antarctica", "Asia", "Europe", "North_America", "Oceania", "South_America")
ZENODO_API = "https://zenodo.org/api/records/{record}"
USER_AGENT = "ceva (+https://github.com/clacrouts/ceva)"
INDEX_KEY = f"qk{INDEX_ZOOM}"
ROW_GROUP_SIZE = 16_384
SOURCE_COLUMNS = [
    "transect_id", "lon", "lat", "bearing", "utm_epsg", "continent", "country",
    "common_country_name", "common_region_name", "osm_coastline_is_closed",
    "osm_coastline_length", "geometry",
]


def _check_continent(continent: str) -> None:
    if continent not in CONTINENTS:
        raise ValueError(f"Unknown GCTS continent {continent!r}. Choose from {list(CONTINENTS)}.")


def raw_path(cfg: Config, continent: str) -> Path:
    return cfg.data_dir / "raw" / "gcts" / cfg.transects.gcts.release / f"{continent}.parquet"


def index_dir(cfg: Config) -> Path:
    return cfg.data_dir / "transects" / "gcts" / cfg.transects.gcts.release


def index_path(cfg: Config, continent: str) -> Path:
    return index_dir(cfg) / f"{continent}.parquet"


def zenodo_file(record: int, key: str, dest: Path) -> Path:
    """Download one file of a Zenodo record to ``dest`` and verify its checksum."""
    headers = {"User-Agent": USER_AGENT}
    meta = requests.get(ZENODO_API.format(record=record), headers=headers, timeout=60)
    meta.raise_for_status()
    files = {f["key"]: f for f in meta.json()["files"]}
    if key not in files:
        raise FileNotFoundError(f"Zenodo record {record} has no file {key!r}: {sorted(files)}")
    algorithm, _, expected = files[key]["checksum"].partition(":")
    digest = hashlib.new(algorithm)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with requests.get(files[key]["links"]["self"], headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(part, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                digest.update(chunk)
    if digest.hexdigest() != expected:
        part.unlink()
        raise OSError(f"Checksum mismatch for {key} (Zenodo {record}): got {digest.hexdigest()}, expected {expected}")
    part.replace(dest)
    log.info("Downloaded %s (%d bytes)", dest, dest.stat().st_size)
    return dest


def download(cfg: Config, continent: str, force: bool = False) -> Path:
    """Download one continent file from Zenodo and verify its checksum."""
    _check_continent(continent)
    dest = raw_path(cfg, continent)
    if dest.exists() and not force:
        return dest
    return zenodo_file(cfg.transects.gcts.zenodo_record, f"{continent}.parquet", dest)


def build_index(cfg: Config, continent: str, force: bool = False) -> Path:
    """Rewrite a raw continent file sorted by the quadkey of each transect origin."""
    _check_continent(continent)
    src, dest = raw_path(cfg, continent), index_path(cfg, continent)
    if dest.exists() and not force:
        return dest
    if not src.exists():
        raise FileNotFoundError(f"{src} is missing. Run `ceva transects prepare --continent {continent}`.")
    gdf = gpd.read_parquet(src, columns=SOURCE_COLUMNS)
    gdf[INDEX_KEY] = quadkeys(gdf["lon"].to_numpy(), gdf["lat"].to_numpy(), INDEX_ZOOM)
    gdf = gdf.sort_values([INDEX_KEY, "transect_id"], kind="stable").reset_index(drop=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    gdf.to_parquet(part, index=False, row_group_size=ROW_GROUP_SIZE, compression="zstd")
    part.replace(dest)
    log.info("Indexed %d transects into %s", len(gdf), dest)
    return dest


def prepare(cfg: Config, continent: str, force: bool = False) -> Path:
    """Download (if needed) and index one continent."""
    download(cfg, continent, force=force)
    return build_index(cfg, continent, force=force)


def available_indexes(cfg: Config) -> list[Path]:
    return sorted(index_dir(cfg).glob("*.parquet"))


def load_tile(cfg: Config, tile: Tile) -> gpd.GeoDataFrame:
    """Transects whose origin lies in ``tile`` (WGS84), from every indexed continent."""
    indexes = available_indexes(cfg)
    if not indexes:
        raise FileNotFoundError(
            f"No GCTS index in {index_dir(cfg)}. Run `ceva transects prepare --continent <name>` first."
        )
    lo, hi = tile.index_range()
    frames = [gpd.read_parquet(p, filters=[(INDEX_KEY, ">=", lo), (INDEX_KEY, "<=", hi)]) for p in indexes]
    crs = frames[0].crs
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs)
    gdf["tile"] = tile.quadkey
    return gdf


def load_bbox(cfg: Config, bbox: tuple[float, float, float, float], zoom: int | None = None) -> gpd.GeoDataFrame:
    """Transects whose origin lies in a lon/lat box, tagged with their processing tile."""
    zoom = cfg.grid.processing_zoom if zoom is None else zoom
    frames = [load_tile(cfg, tile) for tile in tiles_for_bbox(bbox, zoom)]
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)
    w, s, e, n = bbox
    in_lon = gdf["lon"].between(w, e) if w <= e else (gdf["lon"] >= w) | (gdf["lon"] <= e)
    return gdf[in_lon & gdf["lat"].between(s, n)].reset_index(drop=True)
