"""The published dataset, loaded once: the rates and yearly shorelines of the Zenodo record (GeoParquet), the
dashboard's JSON files (manifest, summary, validation, image index) and, when published, the coast it does
not measure. A unit's comparisons and image segments are fetched when a question needs them.

Sources are local paths or web addresses (downloaded once into a cache folder).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SITE_URL = "https://clacrouts.github.io/ceva-dashboard/"
DATA_DOI = "10.5281/zenodo.24493619"        # all versions of the dataset
IMAGES_DOI = "10.5281/zenodo.24500960"      # all versions of the image archive
CODE_URL = "https://github.com/clacrouts/ceva"
USER_AGENT = "ceva-assistant (+https://github.com/clacrouts/ceva)"
EARTH_KM = 6371.0088
GCTS_STEP_M = 100

RATE_COLUMNS = [
    "transect_id", "country", "unit", "rate", "rate_lo", "rate_hi", "r2", "n_years", "first_year", "last_year",
    "change_class", "significant", "se_median_m", "shore_type", "coastal_type", "is_built_environment",
    "has_defence", "sds_rate", "sds_qa", "cvi", "cvi_class", "rank_geomorphology", "rank_shoreline_change",
    "rank_coastal_slope", "rank_sea_level", "rank_waves", "rank_tide", "coastal_slope_pct",
    "sea_level_trend_mm_yr", "uplift_mm_yr", "relative_sea_level_mm_yr", "mean_tidal_range_m",
    "mean_wave_height_m", "geometry",
]


def is_url(source: str) -> bool:
    return str(source).startswith(("http://", "https://"))


def fetch(source: str, cache: Path) -> Path:
    """A local copy of a file given by its path or web address (downloaded once into ``cache``)."""
    if not is_url(source):
        return Path(source)
    import httpx

    parts = source.rstrip("/").split("/")
    dest = Path(cache) / (parts[-2] if parts[-1] == "content" else parts[-1])     # Zenodo: .../files/<name>/content
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with httpx.stream("GET", source, follow_redirects=True, timeout=300, headers={"User-Agent": USER_AGENT}) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    part.replace(dest)
    log.info("downloaded %s", dest.name)
    return dest


def read_json(base: str, path: str) -> dict | None:
    """A JSON file of the dashboard (``data/manifest.json``, ...), from its folder or its web address; None when
    it is not there."""
    if is_url(base):
        import httpx

        r = httpx.get(base.rstrip("/") + "/" + path, timeout=30, follow_redirects=True,
                      headers={"User-Agent": USER_AGENT})
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    p = Path(base) / path
    return json.loads(p.read_text()) if p.exists() else None


def _xyz(lon, lat) -> np.ndarray:
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _km(chord) -> np.ndarray:
    return 2 * EARTH_KM * np.arcsin(np.clip(np.asarray(chord) / 2, 0, 1))


def _chord(km: float) -> float:
    return float(2 * np.sin(km / EARTH_KM / 2))


def haversine_m(lon0, lat0, lon1, lat1) -> np.ndarray:
    lon0, lat0, lon1, lat1 = (np.radians(np.asarray(v, float)) for v in (lon0, lat0, lon1, lat1))
    a = np.sin((lat1 - lat0) / 2) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin((lon1 - lon0) / 2) ** 2
    return 2 * EARTH_KM * 1000 * np.arcsin(np.sqrt(a))


@dataclass
class Dataset:
    """The dataset in memory. ``rates`` has one row per transect (in the order of the yearly series), with the
    latest shoreline point (lon, lat) and the transect's ends (land_lon/lat, sea_lon/lat)."""

    rates: pd.DataFrame
    years: np.ndarray                 # the yearly positions, sorted by transect then year
    x: np.ndarray
    se: np.ndarray
    images: np.ndarray
    starts: np.ndarray                # yearly positions of transect i: starts[i]:starts[i + 1]
    site: str = SITE_URL              # where the dashboard's files are read (folder or address)
    public: str = SITE_URL            # the dashboard's address, for links
    manifest: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    validation: dict | None = None
    frames: dict | None = None
    coverage: pd.DataFrame | None = None
    _unit_files: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        from scipy.spatial import cKDTree

        self.position = pd.Series(np.arange(len(self.rates)), index=self.rates["transect_id"].to_numpy())
        self._tree = cKDTree(_xyz(self.rates["lon"], self.rates["lat"]))
        self._coverage_tree = (cKDTree(_xyz(self.coverage["lon"], self.coverage["lat"]))
                               if self.coverage is not None and len(self.coverage) else None)
        names = {code: c.get("name", code) for code, c in self.summary.get("countries", {}).items()}
        self.country_names = names
        self._by_name = {n.lower(): code for code, n in names.items()} | {c.lower(): c for c in names}

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, rates: str, shorelines: str, site: str = SITE_URL, cache: str | Path = "ceva-cache",
             coverage: str | None = None) -> Dataset:
        """Read the two GeoParquet files of the dataset (paths or addresses) and the dashboard's files at ``site``
        (its folder or its address). ``coverage`` is the dashboard's ``data/coverage.parquet`` when it is
        published elsewhere than at ``site``."""
        import geopandas as gpd
        import pyarrow.parquet as pq
        import shapely

        cache = Path(cache)
        path = fetch(rates, cache)
        columns = [c for c in RATE_COLUMNS if c in pq.read_schema(path).names]
        r = gpd.read_parquet(path, columns=columns)
        r["transect_id"] = r["transect_id"].astype(str)
        r = r.drop_duplicates("transect_id").sort_values("transect_id").reset_index(drop=True)
        land, sea = shapely.get_point(r.geometry.values, 0), shapely.get_point(r.geometry.values, -1)
        r["land_lon"], r["land_lat"] = shapely.get_x(land), shapely.get_y(land)
        r["sea_lon"], r["sea_lat"] = shapely.get_x(sea), shapely.get_y(sea)
        r = pd.DataFrame(r.drop(columns="geometry"))

        s = pq.read_table(fetch(shorelines, cache), columns=["transect_id", "year", "x", "se_m", "n_imagery"])
        s = s.to_pandas()
        s["transect_id"] = s["transect_id"].astype(str)
        s = s[s["transect_id"].isin(set(r["transect_id"]))].sort_values(["transect_id", "year"])
        ids, first = np.unique(s["transect_id"].to_numpy(), return_index=True)
        r = r[r["transect_id"].isin(set(ids))].reset_index(drop=True)          # transects with positions
        starts = np.append(first, len(s))
        x = s["x"].to_numpy(float)
        last = x[starts[1:] - 1]                                                # latest position of each transect
        length = haversine_m(r["land_lon"], r["land_lat"], r["sea_lon"], r["sea_lat"])
        f = np.clip(last / np.where(length > 0, length, 1), 0, 1)
        r["lon"] = r["land_lon"] + (r["sea_lon"] - r["land_lon"]) * f
        r["lat"] = r["land_lat"] + (r["sea_lat"] - r["land_lat"]) * f

        cover = None
        source = coverage or (site.rstrip("/") + "/data/coverage.parquet" if is_url(site)
                              else str(Path(site) / "data" / "coverage.parquet"))
        try:
            cover = pd.read_parquet(fetch(source, cache))
        except Exception as error:                                              # not published (yet)
            log.info("no coverage of the coast not measured: %s", error)
        return cls(rates=r, years=s["year"].to_numpy(int), x=x, se=s["se_m"].to_numpy(float),
                   images=s["n_imagery"].to_numpy(int), starts=starts, site=site,
                   public=site if is_url(site) else SITE_URL,
                   manifest=read_json(site, "data/manifest.json") or {},
                   summary=read_json(site, "data/summary.json") or {},
                   validation=read_json(site, "data/validation.json"),
                   frames=read_json(site, "frames/index.json"), coverage=cover)

    # ------------------------------------------------------------------ lookups
    def index_of(self, transect_id: str) -> int | None:
        i = self.position.get(str(transect_id).strip())
        return None if i is None else int(i)

    def series(self, i: int) -> pd.DataFrame:
        sl = slice(self.starts[i], self.starts[i + 1])
        return pd.DataFrame({"year": self.years[sl], "x": self.x[sl], "se": self.se[sl], "images": self.images[sl]})

    def near(self, lon: float, lat: float, radius_km: float, limit: int) -> tuple[np.ndarray, np.ndarray]:
        """Transects within ``radius_km`` of a point, nearest first: their rows and distances (km)."""
        dist, idx = self._tree.query(_xyz([lon], [lat])[0], k=limit, distance_upper_bound=_chord(radius_km))
        dist, idx = np.atleast_1d(dist), np.atleast_1d(idx)
        ok = np.isfinite(dist)
        return idx[ok], _km(dist[ok])

    def nearest(self, lon: float, lat: float) -> tuple[int, float]:
        dist, idx = self._tree.query(_xyz([lon], [lat])[0])
        return int(idx), float(_km(dist))

    def coverage_near(self, lon: float, lat: float, radius_km: float) -> pd.DataFrame | None:
        """The coast not measured within ``radius_km``: one row per transect of it (``why``), with distances."""
        if self._coverage_tree is None:
            return None
        idx = self._coverage_tree.query_ball_point(_xyz([lon], [lat])[0], _chord(radius_km))
        rows = self.coverage.iloc[sorted(idx)].copy()
        rows["km"] = haversine_m(lon, lat, rows["lon"], rows["lat"]) / 1000
        return rows

    def country(self, text: str | None) -> str | None:
        """An ISO code from a code or an English country name, as the dataset names it."""
        if not text:
            return None
        return self._by_name.get(str(text).strip().lower())

    def link(self, i: int, zoom: int = 15) -> str:
        r = self.rates.iloc[i]
        unit = f"&u={r['unit']}" if "unit" in self.rates and pd.notna(r["unit"]) else ""
        return f"{self.public.rstrip('/')}/?t={r['transect_id']}{unit}#map={zoom}/{r['lat']:.5f}/{r['lon']:.5f}"

    def map_link(self, lon: float, lat: float, zoom: int = 11) -> str:
        return f"{self.public.rstrip('/')}/#map={zoom}/{lat:.5f}/{lon:.5f}"

    def unit_file(self, kind: str, unit: str) -> dict | None:
        """A unit's file of the dashboard: ``refs`` (the other products at its transects) or ``frames`` (its
        image segments); kept once read."""
        key = (kind, unit)
        if key not in self._unit_files:
            path = f"data/refs/{unit}.json" if kind == "refs" else f"frames/units/{unit}.json"
            try:
                self._unit_files[key] = read_json(self.site, path)
            except Exception as error:
                log.warning("%s: %s", path, error)
                return None
        return self._unit_files[key]

    def source(self) -> dict:
        """What every answer cites."""
        m = self.manifest
        return {"dataset": f"CEVA shorelines (code {m.get('ceva_version', '?')}), doi:{DATA_DOI}",
                "licence": m.get("licence", "CC-BY-4.0"), "map": self.public}
