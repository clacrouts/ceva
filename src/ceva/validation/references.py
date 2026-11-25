"""Published shoreline-change products to compare CEVA with, matched to CEVA's transects.

Three products give the change of European coasts:

* ``sm``: the Deltares Shoreline Monitor, on the same GCTS transects, 1984 to 2023. Its rates come
  with the Global Coastal Typology (Calkoen et al. 2025, Zenodo 15607678, CC-BY-4.0).
* ``emodnet_sat``: EMODnet Geology's satellite coastline migration (made by Deltares), 2007-2017,
  transects every 500 m (CC BY-SA 4.0).
* ``emodnet_field``: EMODnet Geology's coastline migration from field data: national surveys
  compiled by EMODnet partners, as erosion, stable or accretion over each source's period
  (CC BY-SA 4.0).

``prepare`` downloads each product once into ``<data dir>/raw/references`` and keeps what the
comparison needs in ``<data dir>/references/<key>.parquet`` (WGS84).

``match`` takes the transects of one unit. For each product it finds the feature nearest to the
transect, on the stretch of it within ``CORRIDOR_M`` of CEVA's latest shoreline (the Shoreline
Monitor: the same transect id). The stretch, not the shoreline point: a product's coastline can sit
tens of metres from CEVA's waterline (a mapped coastline, a land marker), most of all on wide
beaches. It then fits CEVA's rate on the years the product covers, so the two rates describe the
same period. Where the product starts before the satellite record (1984), CEVA's rate covers only
the years after 1984.
"""
from __future__ import annotations

import json
import logging
import math
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from ceva.change_metrics import linear_regression_rate
from ceva.config import Config

log = logging.getLogger(__name__)

METRIC_CRS = 3035                      # ETRS89 / LAEA Europe: distances in metres across Europe
FIRST_YEAR = 1984                      # first year of the satellite record (Landsat 5)
EUROPE = (-32.0, 27.0, 45.0, 72.0)     # lon/lat box kept from the pan-European products
USER_AGENT = "ceva (+https://github.com/clacrouts/ceva)"
CLASSES = ("erosion", "stable", "accretion")

EMODNET_WFS = "https://drive.emodnet-geology.eu/geoserver/tno/wfs"
EMODNET_LAYERS = {"emodnet_sat": "coastal_migration_satellite", "emodnet_field": "coastal_migration_fd_000_0_80k"}
EMODNET_PAGE = 50_000
CORRIDOR_M = 300.0                     # stretch of the transect, either side of CEVA's shoreline, that is matched


@dataclass(frozen=True)
class Product:
    key: str
    name: str
    provider: str
    period: str
    coverage: str
    method: str
    kind: str                     # "satellite" or "field" (surveys and maps)
    licence: str
    url: str
    citation: str
    stable_m_yr: float | None     # |rate| up to which a coast is stable; None: by uncertainty (the product's own rule)
    match_m: float                # largest distance from the transect (near CEVA's shoreline) to the product's feature


PRODUCTS = {
    "sm": Product(
        "sm", "Shoreline Monitor", "Deltares", "1984-2023", "Every GCTS transect (100 m)",
        "Shorelines from yearly composite Landsat and Sentinel-2 images (method of Luijendijk et al. 2018); "
        "rate fitted over the years",
        "satellite", "CC-BY-4.0", "https://zenodo.org/records/15607678",
        "Calkoen et al. (2025), Global Coastal Typology, Zenodo 15607678; Luijendijk et al. (2018), Sci. Rep. 8:6641",
        0.5, 50.0),
    "emodnet_sat": Product(
        "emodnet_sat", "EMODnet coastline migration (satellite)", "EMODnet Geology, Deltares", "2007-2017",
        "Europe, transects every 500 m",
        "Shorelines from yearly composite Sentinel-2 and Landsat 5, 7 and 8 images; rate over the decade",
        "satellite", "CC-BY-SA-4.0", "https://emodnet.ec.europa.eu/geonetwork/srv/api/records/8fbb18deb12b8ea73a91e8a1141e4ea395142ac5",
        "EMODnet Geology, coastline migration from satellite data", 0.5, 150.0),
    "emodnet_field": Product(
        "emodnet_field", "EMODnet coastline migration (field data)", "EMODnet Geology", "varies by source",
        "Europe, coast segments",
        "National surveys and maps compiled by EMODnet partners, as landward migration, stable (within "
        "0.5 m/yr) or seaward migration over each source's period",
        "field", "CC-BY-SA-4.0", "https://emodnet.ec.europa.eu/geonetwork/srv/api/records/a8a18774430c368ecb18350f9e3b1636ee71ce47",
        "EMODnet Geology, coastline migration from field data", 0.5, 100.0),
}

COLUMNS = ["transect_id", "product", "ref_id", "distance_m", "rate", "unc", "cls", "y0", "y1", "qa", "offset_m",
           "source", "ceva_rate", "ceva_lo", "ceva_hi", "ceva_n", "ceva_y0", "ceva_y1", "ceva_cls"]


def product_info(keys) -> list[dict]:
    return [asdict(PRODUCTS[k]) for k in keys]


def raw_dir(cfg: Config) -> Path:
    return cfg.data_dir / "raw" / "references"


def path(cfg: Config, key: str) -> Path:
    return cfg.data_dir / "references" / f"{key}.parquet"


def available(cfg: Config) -> list[str]:
    return [k for k in PRODUCTS if path(cfg, k).exists()]


# ------------------------------------------------------------------ downloads

def _get(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=600) as response, open(part, "wb") as f:
        while chunk := response.read(1 << 20):
            f.write(chunk)
    part.rename(dest)
    return dest


def _emodnet_pages(cfg: Config, layer: str) -> list[Path]:
    """The layer as GeoJSON pages (WGS84), fetched once."""
    folder = raw_dir(cfg) / "emodnet" / layer
    pages = []
    for start in range(0, 10_000_000, EMODNET_PAGE):
        page = folder / f"page_{start:06d}.json"
        if not page.exists():
            _get(f"{EMODNET_WFS}?service=WFS&version=2.0.0&request=GetFeature&outputFormat=application/json"
                 f"&srsName=EPSG:4326&sortBy=id&typeNames=tno:{layer}&count={EMODNET_PAGE}&startIndex={start}", page)
        returned = json.loads(page.read_text()).get("numberReturned", 0)
        if returned == 0:
            page.unlink()
            break
        pages.append(page)
        if returned < EMODNET_PAGE:
            break
    return pages


def _read_pages(pages: list[Path]) -> gpd.GeoDataFrame:
    frames = [gpd.read_file(p) for p in pages]
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)


def _in_europe(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return g.iloc[g.sindex.query(shapely.box(*EUROPE), predicate="intersects")].sort_index()


# ------------------------------------------------------------------ building each product

def _build_sm(cfg: Config) -> gpd.GeoDataFrame:
    """The Shoreline Monitor's rate of each GCTS transect and its latest shoreline point."""
    folder = cfg.data_dir / "raw" / "typology" / str(cfg.transects.typology.zenodo_record)
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Global Coastal Typology file in {folder}: prepare the typology first")
    columns = {"transect_id": "transect_id", "sds:change_rate": "rate", "sds:obs_primary_count": "n_obs",
               "sds:reference_datetime": "time", "sds:reference_lon": "lon", "sds:reference_lat": "lat",
               "sds:tr_stdev": "stdev", "sds:tr_is_qa": "qa"}
    d = pd.concat([pd.read_parquet(f, columns=list(columns)).rename(columns=columns) for f in files])
    d = d[d["rate"].notna() & d["lon"].notna()]
    return gpd.GeoDataFrame({
        "transect_id": d["transect_id"].astype(str).to_numpy(), "rate": d["rate"].astype("float32").to_numpy(),
        "n_obs": d["n_obs"].astype("float32").to_numpy(), "y0": np.int16(FIRST_YEAR),
        "y1": d["time"].dt.year.astype("int16").to_numpy(), "stdev": d["stdev"].astype("float32").to_numpy(),
        "qa": d["qa"].fillna(False).astype(bool).to_numpy(),
    }, geometry=gpd.points_from_xy(d["lon"], d["lat"]), crs=4326)


def _build_emodnet_sat(cfg: Config) -> gpd.GeoDataFrame:
    g = _in_europe(_read_pages(_emodnet_pages(cfg, EMODNET_LAYERS["emodnet_sat"])))
    return gpd.GeoDataFrame({
        "ref_id": g["transect_id"].astype(str).to_numpy(), "rate": g["changerate"].astype("float32").to_numpy(),
        "unc": g["changerate_unc"].astype("float32").to_numpy(),
        "cls": g["cr_class3"].astype("string").str.lower().to_numpy(), "y0": np.int16(2007), "y1": np.int16(2017),
        "source": g["sediment_label"].astype("string").to_numpy(),
    }, geometry=g.geometry.values, crs=4326)


def _build_emodnet_field(cfg: Config) -> gpd.GeoDataFrame:
    g = _in_europe(_read_pages(_emodnet_pages(cfg, EMODNET_LAYERS["emodnet_field"])))
    g = g[g["migration"].isin(CLASSES)]
    period = g["period"].astype("string").str.extract(r"(\d{4})\s*-\s*(\d{4})")
    return gpd.GeoDataFrame({
        "ref_id": g["id"].astype(str).to_numpy(), "cls": g["migration"].astype(str).to_numpy(),
        "y0": pd.to_numeric(period[0], errors="coerce").astype("Int16").to_numpy(),
        "y1": pd.to_numeric(period[1], errors="coerce").astype("Int16").to_numpy(),
    }, geometry=g.geometry.values, crs=4326)


_BUILDERS = {"sm": _build_sm, "emodnet_sat": _build_emodnet_sat, "emodnet_field": _build_emodnet_field}


def prepare(cfg: Config, keys: list[str] | None = None, force: bool = False) -> dict[str, int]:
    """Download and reduce each product (once). Returns the number of features of each."""
    counts = {}
    for key in keys or list(PRODUCTS):
        if key not in PRODUCTS:
            raise ValueError(f"unknown reference product {key!r}; choose from {list(PRODUCTS)}")
        dest = path(cfg, key)
        if dest.exists() and not force:
            counts[key] = int(len(gpd.read_parquet(dest, columns=["geometry"])))
            continue
        g = _BUILDERS[key](cfg)
        dest.parent.mkdir(parents=True, exist_ok=True)
        g.to_parquet(dest)
        counts[key] = int(len(g))
        log.info("%s: %d features", key, len(g))
    return counts


_LOADED: dict[tuple[str, float], gpd.GeoDataFrame] = {}


def load(cfg: Config, key: str) -> gpd.GeoDataFrame:
    """One product in metres (ETRS89-LAEA); the Shoreline Monitor indexed by transect id."""
    p = path(cfg, key)
    cache_key = (str(p), p.stat().st_mtime)
    if cache_key not in _LOADED:
        g = gpd.read_parquet(p).to_crs(METRIC_CRS)
        if key == "sm":
            g = g.drop_duplicates("transect_id").set_index("transect_id", drop=False)
        _LOADED[cache_key] = g
    return _LOADED[cache_key]


def load_all(cfg: Config, keys: list[str] | None = None) -> dict[str, gpd.GeoDataFrame]:
    return {k: load(cfg, k) for k in (keys or available(cfg))}


# ------------------------------------------------------------------ matching one unit

def period_rate(years: np.ndarray, x: np.ndarray, y0, y1) -> tuple[float, float, float, int, int, int]:
    """CEVA's rate (and 95 % range) over the years ``y0``-``y1``, from 1984 at the earliest.

    It needs at least five yearly positions and half the years of the period, the first within
    the first fifth of the period and the last within the last fifth. Otherwise NaN.
    """
    none = (math.nan, math.nan, math.nan, 0, 0, 0)
    if y0 is None or y1 is None or pd.isna(y0) or pd.isna(y1):
        return none
    lo, hi = max(int(y0), FIRST_YEAR), int(y1)
    span = hi - lo + 1
    if span < 5 or len(years) == 0:
        return none
    keep = (years >= lo) & (years <= hi)
    n = int(keep.sum())
    if n < max(5, math.ceil(0.5 * span)):
        return none
    used = years[keep]
    if used.min() > lo + 0.2 * span or used.max() < hi - 0.2 * span:
        return none
    rate, rate_lo, rate_hi, _ = linear_regression_rate(used.astype(float), x[keep])
    return float(rate), float(rate_lo), float(rate_hi), n, lo, hi


def threshold_class(rate: float, stable: float) -> str | None:
    if rate is None or not np.isfinite(rate):
        return None
    return "erosion" if rate < -stable else "accretion" if rate > stable else "stable"


def significance_class(lo: float, hi: float) -> str | None:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return "erosion" if hi < 0 else "accretion" if lo > 0 else "stable"


def ceva_class(rate: float, lo: float, hi: float, stable: float | None) -> str | None:
    """CEVA's class over a product's years, as the map shows it: eroding or accreting only where the
    trend is significant (its 95 % range excludes zero), and beyond the product's stable band if it has one."""
    by_sign = significance_class(lo, hi)
    if stable is None or by_sign is None:
        return by_sign
    return threshold_class(rate, stable) if by_sign != "stable" else "stable"


def _nearest(ref: gpd.GeoDataFrame, points: gpd.GeoDataFrame, max_m: float) -> pd.DataFrame:
    x0, y0, x1, y1 = points.total_bounds
    near = ref.iloc[ref.sindex.query(shapely.box(x0 - max_m, y0 - max_m, x1 + max_m, y1 + max_m))]
    if near.empty:
        return pd.DataFrame()
    joined = gpd.sjoin_nearest(points, near, how="inner", max_distance=max_m, distance_col="distance_m")
    joined = joined.sort_values("distance_m").drop_duplicates("transect_id")
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))


def _same_transect(sm: gpd.GeoDataFrame, points: gpd.GeoDataFrame, shores: gpd.GeoDataFrame,
                   transects: gpd.GeoDataFrame) -> pd.DataFrame:
    """The Shoreline Monitor on the same transect, and how far its latest shoreline is from CEVA's
    shoreline of the same year (positive: the Monitor's is further seaward)."""
    ids = [t for t in points["transect_id"] if t in sm.index]
    if not ids:
        return pd.DataFrame()
    found = pd.DataFrame(sm.loc[ids])
    lines = transects.set_index("transect_id").reindex(ids)
    line_m = gpd.GeoSeries(lines.geometry.values, crs=transects.crs).to_crs(METRIC_CRS).values
    sign = np.where(lines["seaward_sign"].fillna(1).to_numpy() >= 0, 1.0, -1.0)
    ref_point = found.geometry.values
    year_points = shores.assign(_key=list(zip(shores["transect_id"], shores["year"], strict=True)))
    year_points = gpd.GeoSeries(year_points.geometry.values, index=year_points["_key"], crs=shores.crs)
    year_points = year_points[~year_points.index.duplicated()].to_crs(METRIC_CRS)
    ours = year_points.reindex(list(zip(ids, found["y1"].astype(int), strict=True))).values
    distance = shapely.distance(ref_point, line_m)
    offset = (shapely.line_locate_point(line_m, ref_point) - shapely.line_locate_point(line_m, ours)) * sign
    valid = (distance <= PRODUCTS["sm"].match_m) & ~shapely.is_missing(ours)
    found["offset_m"] = np.where(valid, offset, np.nan)
    found["distance_m"] = np.where(np.isfinite(distance), distance, np.nan)
    found["ref_id"] = found["transect_id"]
    return found.drop(columns=["geometry"]).reset_index(drop=True)


def match(refs: dict[str, gpd.GeoDataFrame], points: gpd.GeoDataFrame, shores: gpd.GeoDataFrame,
          transects: gpd.GeoDataFrame) -> pd.DataFrame:
    """Each product's value at the transects of one unit, with CEVA's rate over the product's years.

    ``points``: one row per transect at its latest shoreline (``transect_id``, geometry);
    the products are matched to the transect line within ``CORRIDOR_M`` of that point;
    ``shores``: the yearly positions (``transect_id``, ``year``, ``x``, geometry);
    ``transects``: the transect lines (``transect_id``, ``seaward_sign``). One row per transect and
    product found, with the columns of ``COLUMNS``.
    """
    if not refs or points.empty:
        return pd.DataFrame(columns=COLUMNS)
    transects = transects.astype({"transect_id": str})
    at = gpd.GeoDataFrame(points[["transect_id"]].astype({"transect_id": str}), geometry=points.geometry.values,
                          crs=points.crs).to_crs(METRIC_CRS)
    lines = gpd.GeoSeries(transects.set_index("transect_id").reindex(at["transect_id"]).geometry.values,
                          crs=transects.crs).to_crs(METRIC_CRS).values
    stretch = shapely.intersection(lines, shapely.buffer(at.geometry.values, CORRIDOR_M))
    stretch = np.where(shapely.is_missing(stretch) | shapely.is_empty(stretch), at.geometry.values, stretch)
    near = gpd.GeoDataFrame({"transect_id": at["transect_id"].to_numpy()}, geometry=stretch, crs=METRIC_CRS)
    found = []
    for key, ref in refs.items():
        if key == "sm":
            hit = _same_transect(ref, at, shores.astype({"transect_id": str}), transects)
        else:
            hit = _nearest(ref, near, PRODUCTS[key].match_m)
        if not hit.empty:
            found.append(hit.assign(product=key))
    if not found:
        return pd.DataFrame(columns=COLUMNS)
    table = pd.concat(found, ignore_index=True).reindex(columns=COLUMNS)
    for column in ("cls", "source", "ref_id"):
        table[column] = table[column].astype(object)
    series = {str(tid): (g["year"].to_numpy(dtype=int), g["x"].to_numpy(dtype=float))
              for tid, g in shores.sort_values("year").groupby("transect_id")}
    empty = (np.array([], dtype=int), np.array([], dtype=float))
    fits = [period_rate(*series.get(tid, empty), y0, y1)
            for tid, y0, y1 in zip(table["transect_id"], table["y0"], table["y1"], strict=True)]
    fits = pd.DataFrame(fits, columns=["ceva_rate", "ceva_lo", "ceva_hi", "ceva_n", "ceva_y0", "ceva_y1"])
    for column in fits:
        table[column] = fits[column].to_numpy()
    table["ceva_cls"] = [ceva_class(r, lo, hi, PRODUCTS[p].stable_m_yr) for p, r, lo, hi in
                         zip(table["product"], table["ceva_rate"], table["ceva_lo"], table["ceva_hi"], strict=True)]
    ref_threshold = table["product"].map(lambda p: PRODUCTS[p].stable_m_yr)
    by_rate = table["cls"].isna() & ref_threshold.notna() & table["rate"].notna()
    table.loc[by_rate, "cls"] = [threshold_class(r, s) for r, s in zip(table.loc[by_rate, "rate"],
                                                                        ref_threshold[by_rate], strict=True)]
    return table


def card_entries(table: pd.DataFrame) -> dict[str, dict]:
    """What a transect's card shows of each product: its rate (``r``), years (``y``), class (``k``),
    CEVA's rate over the same years (``c``) and the distance to the product's feature (``d``); for the
    Shoreline Monitor also its quality flag (``q``) and how far its latest shoreline is from CEVA's (``o``)."""
    def num(v, digits=2):
        return None if v is None or pd.isna(v) else round(float(v), digits)

    out: dict[str, dict] = {}
    for row in table.itertuples(index=False):
        entry = {"r": num(row.rate), "y": [None if pd.isna(row.y0) else int(row.y0),
                                           None if pd.isna(row.y1) else int(row.y1)],
                 "c": num(row.ceva_rate), "k": row.cls if isinstance(row.cls, str) else None,
                 "d": num(row.distance_m, 0)}
        if row.product == "sm":
            entry["q"] = int(bool(row.qa)) if not pd.isna(row.qa) else None
            entry["o"] = num(row.offset_m, 0)
        out.setdefault(str(row.transect_id), {})[row.product] = {k: v for k, v in entry.items() if v is not None}
    return out
