"""The full dataset for download and citation (Zenodo): every product of the runs, with its documentation.

``ceva export OUT --run <runs>`` writes:

* ``<name>_rates.parquet`` (GeoParquet, one row per transect, drawn from its landward end): rate of shoreline
  change and its 95 % range, change class, years, precision, country, coastal type, the Shoreline Monitor's
  rate, and the coastal vulnerability index with its ranks and inputs (when ``ceva vulnerability compute``
  has run);
* ``<name>_shorelines.parquet`` (GeoParquet, points): the yearly shoreline position of each transect, its
  standard error and the number of images behind it;
* ``<name>_<CC>_waterlines.parquet`` (Parquet), one per country (ISO code ``CC``): every waterline of every
  image, with its tide, wave level, corrected position and whether it was used. They make most of the size;

and ``README.md`` (files, columns, method, validation, limits, licence, how to cite), ``metadata.json`` (a
Zenodo deposit's metadata) and ``SHA256SUMS``. A Zenodo record holds up to 100 files: 95 countries at most.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import LineString

from ceva import __version__
from ceva.change_metrics import classify_lrr
from ceva.config import Config

log = logging.getLogger(__name__)

PRODUCTS = ("rates", "shorelines", "waterlines")
WHOLE = ("rates", "shorelines")            # one file each for the whole area; the waterlines one per country
ZENODO_FILES = 100
WATERLINE_COLUMNS = ["transect_id", "time", "platform", "family", "x_obs", "tide", "wave", "level", "x_corr",
                     "threshold", "outlier", "low_water", "use"]
SITE_URL = "https://clacrouts.github.io/ceva-dashboard/"     # the map and figures, the project's public page
SITE_CODE_URL = "https://github.com/clacrouts/ceva-dashboard"
CODE_URL = "https://github.com/clacrouts/ceva"                     # the software that made the dataset
IMAGES_DOI = "10.5281/zenodo.24500960"                             # the satellite images behind it (all versions)
RELATED = [
    {"identifier": "10.5281/zenodo.14056925", "relation": "isDerivedFrom", "scheme": "doi"},    # GCTS
    {"identifier": "10.5281/zenodo.15607678", "relation": "isDerivedFrom", "scheme": "doi"},    # typology
    {"identifier": "10.17882/79489", "relation": "references", "scheme": "doi"},                # EOT20
    {"identifier": "10.24381/cds.adbb2d47", "relation": "references", "scheme": "doi"},         # ERA5 single levels
    {"identifier": SITE_URL, "relation": "isSupplementedBy", "scheme": "url"},             # the dashboard
    {"identifier": SITE_CODE_URL, "relation": "isSupplementedBy", "scheme": "url"},
    {"identifier": IMAGES_DOI, "relation": "isSupplementedBy", "scheme": "doi"},
    {"identifier": CODE_URL, "relation": "isCompiledBy", "scheme": "url"},
    {"identifier": "10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f", "relation": "references", "scheme": "doi"},  # GEBCO
    {"identifier": "10.5281/zenodo.5560862", "relation": "references", "scheme": "doi"},     # GIA grids, Steffen 2021
    {"identifier": "10.1002/2014JB011176", "relation": "references", "scheme": "doi"},       # ICE-6G_C, Peltier 2015
]

COLUMN_DOCS = {
    "rates": [
        ("transect_id", "GCTS transect id (Calkoen et al. 2025): stable worldwide"),
        ("country", "ISO 3166-1 alpha-2 code of the transect (GCTS)"),
        ("unit", "processing unit (a tile of about 30 km cut along the coast): the dashboard's files and links use it"),
        ("rate", "linear rate of shoreline change, m/yr, positive seaward (accretion), negative landward (erosion)"),
        ("rate_lo, rate_hi", "95 % range of the rate (t distribution of the least-squares slope)"),
        ("r2", "coefficient of determination of the linear fit"),
        ("n_years, first_year, last_year", "years with a position, first and last of them"),
        ("change_class", "Luijendijk et al. (2018) class of the rate: stable within ±0.5 m/yr, then erosion "
                         "or accretion, intense (1–3 m/yr), severe (3–5), extreme (over 5)"),
        ("significant", "the 95 % range of the rate excludes zero"),
        ("se_median_m", "median standard error of the yearly positions, m: precision, not validated accuracy"),
        ("shore_type, coastal_type, is_built_environment, has_defence",
         "Global Coastal Typology (Calkoen et al. 2025): sediment type, geomorphology, built-up shore, "
         "coastal defences"),
        ("sds_rate, sds_qa", "the Deltares Shoreline Monitor's change rate (m/yr) and its quality flag, from the "
                             "same typology files, for comparison"),
        ("geometry", "the transect (EPSG:4326), from its landward end, along which x is measured"),
    ],
    "shorelines": [
        ("transect_id, country", "as in rates"),
        ("year", "calendar year"),
        ("x", "shoreline position that year: metres from the landward end of the transect, at mean sea level"),
        ("se_m", "standard error of x, m (from the scatter of the year's waterlines and their number)"),
        ("n_imagery", "Landsat and Sentinel-2 waterlines behind the position"),
        ("geometry", "the shoreline point (EPSG:4326)"),
    ],
    "waterlines": [
        ("transect_id, country", "as in rates"),
        ("time", "acquisition time, UTC"),
        ("platform, family", "satellite (landsat-5 ... landsat-9, sentinel-2a/b/c), and its family"),
        ("x_obs", "waterline seen in the image: metres from the landward end of the transect"),
        ("tide", "tide height at that time, m above mean sea level (EOT20)"),
        ("wave", "wave-driven rise of the waterline, m (empirical: 0.48 x (Hs - the transect's mean Hs), ERA5)"),
        ("level", "tide + wave, m"),
        ("x_corr", "the waterline moved to mean sea level: x_obs + level / beach slope"),
        ("threshold", "water-index threshold of the image (Otsu)"),
        ("outlier, low_water, use", "flagged as an outlier; water level too low to correct; used in the "
                                    "yearly position"),
    ],
}

VULNERABILITY_DOCS = [
    ("cvi, cvi_class", "coastal vulnerability index, sqrt(product of the six ranks / 6) (Thieler and "
                       "Hammar-Klose 1999), and its class: low, moderate, high or very high, the quartiles "
                       "of the index over the dataset"),
    ("rank_geomorphology, rank_shoreline_change, rank_coastal_slope, rank_sea_level, rank_waves, rank_tide",
     "the six ranks, 1 (low vulnerability) to 5 (very high)"),
    ("coastal_slope_pct", "regional slope across the shore, %: GEBCO 2024 from 5 km offshore to the highest "
                          "ground within 5 km inland"),
    ("sea_level_trend_mm_yr, uplift_mm_yr, relative_sea_level_mm_yr",
     "trend of the sea surface at the nearest sea cell (Copernicus Marine altimetry, its correction for "
     "glacial isostatic adjustment undone with the ICE-6G_C VM5a geoid rate), vertical land motion of "
     "glacial isostatic adjustment (ICE-6G_C VM5a), and their difference, mm/yr"),
    ("mean_tidal_range_m, mean_wave_height_m", "mean daily tidal range (EOT20, one year) and mean significant "
                                               "wave height (ERA5), m, of the transect's unit"),
]


def _orient(line, sign: int):
    """The transect from its landward end (``seaward_sign`` +1: stored land to sea)."""
    if line is None:
        return None
    coords = list(line.coords)
    return LineString(coords if sign >= 0 else coords[::-1])


def _countries(cfg: Config, ids: list[str]) -> pd.Series:
    from ceva.transects.gcts import available_indexes

    found = pd.concat([pd.read_parquet(p, columns=["transect_id", "country"], filters=[("transect_id", "in", ids)])
                       for p in available_indexes(cfg)]).drop_duplicates("transect_id")
    return found.set_index("transect_id")["country"]


VULNERABILITY_COLUMNS = ["cvi", "cvi_class", "rank_geomorphology", "rank_shoreline_change", "rank_coastal_slope",
                         "rank_sea_level", "rank_waves", "rank_tide", "coastal_slope_pct", "sea_level_trend_mm_yr",
                         "uplift_mm_yr", "relative_sea_level_mm_yr", "mean_tidal_range_m", "mean_wave_height_m"]


def unit_tables(cfg: Config, name: str, method: str = "ceva", index: pd.DataFrame | None = None,
                products: tuple[str, ...] = PRODUCTS) -> dict[str, pd.DataFrame] | None:
    """The products of one unit, each with the country of every transect; the rates with their unit and the
    vulnerability ``index`` (``ceva.vulnerability.read``) when given."""
    from ceva.transects import typology

    out = cfg.data_dir / "results" / name / method
    if not (out / "rates_imagery.parquet").exists():
        return None
    rates = pd.read_parquet(out / "rates_imagery.parquet")
    shores = gpd.read_parquet(out / "shorelines_imagery.parquet")
    lines = gpd.read_parquet(out / "transects.parquet").set_index("transect_id")
    water = pd.read_parquet(out / "waterlines.parquet", columns=WATERLINE_COLUMNS) if "waterlines" in products else None
    ids = sorted(set(rates["transect_id"].astype(str)) | set(shores["transect_id"].astype(str)))
    country = _countries(cfg, ids)
    se = shores.groupby("transect_id")["se_m"].median().rename("se_median_m")
    rates = rates.drop(columns=["product"], errors="ignore").join(se, on="transect_id")
    rates["change_class"] = rates["rate"].map(lambda r: classify_lrr(r) if pd.notna(r) else "insufficient_data")
    rates["significant"] = (rates["rate_lo"] > 0) | (rates["rate_hi"] < 0)
    try:
        kinds = typology.load(cfg, ids)[["transect_id", "shore_type", "coastal_type", "is_built_environment",
                                          "has_defence", "sds_rate", "sds_qa"]]
        rates = rates.merge(kinds, on="transect_id", how="left")
    except FileNotFoundError:
        pass
    if index is not None:
        from ceva.vulnerability import CLASSES

        known = index.astype({"transect_id": str}).drop_duplicates("transect_id")
        known = known.assign(cvi_class=known["cvi_class"].map(dict(enumerate(CLASSES))))
        rates = rates.astype({"transect_id": str}).merge(known[["transect_id", *VULNERABILITY_COLUMNS]],
                                                         on="transect_id", how="left")
    geometry = [_orient(lines["geometry"].get(t), int(lines["seaward_sign"].get(t, 1))) for t in rates["transect_id"]]
    rates = gpd.GeoDataFrame(rates, geometry=geometry, crs=lines.crs)
    shores = shores[["transect_id", "year", "x", "se_m", "n_imagery", "geometry"]]
    tables = {k: v for k, v in {"rates": rates, "shorelines": shores, "waterlines": water}.items() if k in products}
    for table in tables.values():
        table.insert(1, "country", table["transect_id"].astype(str).map(country))
    if "rates" in tables:
        tables["rates"].insert(2, "unit", name)
    return tables


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


PLACE = "Europe and the Mediterranean"
COVERAGE = ("The sand and gravel coasts (Global Coastal Typology) of Europe south of the Arctic circle, with the "
            "Black Sea and the Sea of Azov but not the Caspian, and of every country around the Mediterranean.")


def _geo_metadata(schemas: list[pa.Schema]) -> dict:
    """The GeoParquet metadata of parts written by geopandas, for all of them: bounding box and geometry types."""
    metas = [json.loads(s.metadata[b"geo"]) for s in schemas]
    geo = metas[0]
    column = geo["primary_column"]
    boxes = [m["columns"][column]["bbox"] for m in metas if m["columns"][column].get("bbox")]
    if boxes:
        geo["columns"][column]["bbox"] = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                                          max(b[2] for b in boxes), max(b[3] for b in boxes)]
    types = {g for m in metas for g in m["columns"][column]["geometry_types"]}
    geo["columns"][column]["geometry_types"] = sorted(types)
    return geo


def _concat(files: list[Path], dest: Path) -> int:
    """One Parquet file from parts, streamed part by part: their columns unified (a column missing from a part,
    or empty there, takes the type it has elsewhere), and GeoParquet metadata covering them all. Returns rows."""
    schemas = [pq.read_schema(f) for f in files]
    schema = pa.unify_schemas([s.remove_metadata() for s in schemas], promote_options="permissive")
    if all(s.metadata and b"geo" in s.metadata for s in schemas):
        schema = schema.with_metadata({b"geo": json.dumps(_geo_metadata(schemas)).encode()})
    rows = 0
    with pq.ParquetWriter(dest, schema, compression="zstd") as writer:
        for f in files:
            part = pq.read_table(f)
            columns = [part[n].cast(schema.field(n).type) if n in part.column_names
                       else pa.nulls(part.num_rows, schema.field(n).type) for n in schema.names]
            writer.write_table(pa.Table.from_arrays(columns, schema=schema))
            rows += part.num_rows
    return rows


def export(cfg: Config, out: Path, units: list[str], name: str = "ceva-shorelines", version: str = "1.1",
           method: str = "ceva", place: str = PLACE, coverage: str = COVERAGE,
           products: tuple[str, ...] = PRODUCTS) -> dict:
    """Write the dataset of these units into ``out``: rates and yearly shorelines in one file each, the
    waterlines in one file per country. Returns its summary.

    ``place`` names the coasts in the titles, ``coverage`` says exactly which transects are in; ``products``
    leaves some out (a light copy: rates and shorelines)."""
    from ceva import vulnerability

    out = Path(out)
    parts = out / ".parts"
    shutil.rmtree(parts, ignore_errors=True)
    done, index = [], vulnerability.read(cfg)
    counts: dict[str, dict[str, int]] = {}
    for unit in units:
        tables = unit_tables(cfg, unit, method, index, products)
        if tables is None:
            log.warning("%s: no %s results, left out", unit, method)
            continue
        for product, table in tables.items():
            country = table["country"].fillna("XX").astype(str)
            for cc, n in country.value_counts().items():
                counts.setdefault(cc, dict.fromkeys(PRODUCTS, 0))[product] += int(n)
            groups = [("", table)] if product in WHOLE else table.groupby(country)
            for cc, g in groups:
                folder = parts / product / cc
                folder.mkdir(parents=True, exist_ok=True)
                g.reset_index(drop=True).to_parquet(folder / f"{unit}.parquet", index=False)
        done.append(unit)
    if not done:
        raise RuntimeError("no unit has results to export")
    counts = dict(sorted(counts.items()))
    countries = list(counts)
    if len(WHOLE) + len(countries) + 3 > ZENODO_FILES:
        raise RuntimeError(f"{len(countries)} countries: more files than a Zenodo record holds ({ZENODO_FILES})")
    for product in (p for p in WHOLE if p in products):
        _concat(sorted((parts / product).glob("*.parquet")), out / f"{name}_{product}.parquet")
    for cc in countries:
        files = sorted((parts / "waterlines" / cc).glob("*.parquet"))
        if files:
            _concat(files, out / f"{name}_{cc}_waterlines.parquet")
    shutil.rmtree(parts)
    summary = {"name": name, "version": version, "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
               "ceva_version": __version__, "method": method, "units": len(done), "countries": counts,
               "place": place, "coverage": coverage, "vulnerability": index is not None, "products": list(products)}
    (out / "README.md").write_text(readme(summary))
    (out / "metadata.json").write_text(json.dumps(zenodo_metadata(summary), indent=1))
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{_sha256(p)}  {p.name}\n" for p in files))
    log.info("exported %d units, %d countries, to %s", len(done), len(countries), out)
    return summary


def readme(summary: dict) -> str:
    """The dataset's README: what it is, its files and columns, method, validation, limits, licence, citation."""
    name, index = summary["name"], summary.get("vulnerability", False)
    rows = "\n".join(f"| {cc} | {c.get('rates', 0):,} | {c.get('shorelines', 0):,} | {c.get('waterlines', 0):,} |"
                     for cc, c in summary["countries"].items())
    products = summary.get("products", PRODUCTS)
    files = "\n".join(line for product, line in (
        ("rates", f"* `{name}_rates.parquet` (GeoParquet): one row per transect, for the whole area."),
        ("shorelines", f"* `{name}_shorelines.parquet` (GeoParquet): one row per transect and year, for the "
                       "whole area."),
        ("waterlines", f"* `{name}_CC_waterlines.parquet` (Parquet), one per country `CC` (ISO 3166-1 alpha-2, "
                       "`XX` where GCTS gives none): one row per image and transect."),
    ) if product in products)
    docs = {k: v for k, v in COLUMN_DOCS.items() if k in products}
    if index and "rates" in docs:
        docs["rates"] = docs["rates"][:-1] + VULNERABILITY_DOCS + docs["rates"][-1:]
    columns = "\n\n".join(f"**{product}**\n\n| Column | Meaning |\n|---|---|\n" + "\n".join(
        f"| `{c}` | {m} |" for c, m in docs[product]) for product in docs)
    title = f"satellite-derived shorelines of the sand and gravel coasts of {summary['place']}"
    index_line = (" With a coastal vulnerability index of every transect (USGS method)." if index else "")
    index_step = """
6. The coastal vulnerability index (Thieler and Hammar-Klose 1999, USGS) ranks six variables from 1 (low) to
   5 (very high): the landform (Global Coastal Typology), the rate of change, the regional slope (GEBCO 2024,
   from 5 km offshore to the highest ground within 5 km inland), the relative sea-level rise (the sea-surface
   trend of Copernicus Marine satellite altimetry minus the land uplift of glacial isostatic adjustment,
   ICE-6G_C VM5a), the mean ERA5 wave height and the mean EOT20 tidal range. CVI = sqrt(product of the
   ranks / 6); its classes are the quartiles of the index over the dataset: they compare coasts with each
   other, they are not a hazard or risk assessment.""" if index else ""
    index_sources = """ Terrain and bathymetry: GEBCO Compilation Group (2024), GEBCO 2024 Grid,
doi:10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f. Sea-level trends: generated using E.U. Copernicus Marine
Service Information (OMI_CLIMATE_SL_GLOBAL_regional_trends). Glacial isostatic adjustment: ICE-6G_C (VM5a),
Peltier et al. (2015), doi:10.1002/2014JB011176, grids of Steffen (2021), doi:10.5281/zenodo.5560862,
CC-BY-4.0.""" if index else ""
    return f"""# {name} {summary['version']}: {title}

Yearly shoreline positions and rates of change, 1984 to 2025, on the transects of the Global Coastal
Transect System (one every 100 m along the coast), from every clear Landsat 4–9 and Sentinel-2 image,
moved to mean sea level with the tide and the waves of each image.{index_line} Made with CEVA
{summary['ceva_version']} ({CODE_URL}, method: the CEVA solution), {summary['created'][:10]}. Map and figures:
{SITE_URL}; the satellite images behind the shorelines: doi:{IMAGES_DOI}.

{summary['coverage']}

## Files

{files}

`SHA256SUMS` lists the checksums. All coordinates are WGS84 (EPSG:4326); every file has the `country`
column, so a country is one filter away.

| Country | Transects | Yearly positions | Waterlines |
|---|---|---|---|
{rows}

## Columns

{columns}

## Method in short

1. Each Landsat (Collection 2 Level-2, 1984 onwards) and Sentinel-2 (Level-2A, 2015 onwards) image is read
   along every transect; clouds, shadows and no-data are masked from the image's quality band.
2. The waterline is where the modified normalised difference water index (green, short-wave infrared)
   crosses the image's Otsu threshold, searched near where the tide of the image puts the shoreline.
3. Each waterline is moved to mean sea level: the EOT20 tide at the image time, plus a wave term fitted on
   surveyed beaches (ERA5 significant wave height), divided by the beach slope estimated from the images
   (Vos et al. 2020). Outliers and images at too low a water level are flagged, not removed.
4. The yearly position is the median of the year's corrected waterlines; the rate is the least-squares
   slope through the yearly positions (at least 5 years), with its 95 % range.
5. Only transects classed as sand or gravel coasts by the Global Coastal Typology are processed.{index_step}

The code, and a notebook that runs every step on one beach: {CODE_URL}.

## Validation and limits

The method was checked against surveyed beach profiles at six beaches (Truc Vert, France; Narrabeen,
Australia; Noord-Holland and Walcheren, the Netherlands; Slapton Sands and Perranporth, United Kingdom):
yearly positions were 8 to 39 m from the surveyed ones (root mean square), the largest at the
macrotidal Perranporth. `se_m` is a precision from the data themselves, not a validated accuracy. Rates
over a few metres a year, or over many years, are more reliable than any single year. Global tide models
are weakest in estuaries and lagoons; muddy, rocky and engineered coasts are not covered. The validation
beaches are all in temperate climates. This is not an official erosion map: national and regional
observatories are the reference for planning.

## Licence and attribution

CC-BY-4.0. Contains modified Copernicus Sentinel data (ESA / EU). Landsat imagery courtesy of the U.S.
Geological Survey. Derived from the Global Coastal Transect System (doi:10.5281/zenodo.14056925) and the
Global Coastal Typology (doi:10.5281/zenodo.15607678), Calkoen et al. (2025), CC-BY-4.0. Tides: EOT20,
Hart-Davis et al. (2021), doi:10.17882/79489, CC-BY-4.0. Waves: ERA5, Hersbach et al. (2020), Copernicus
Climate Change Service, doi:10.24381/cds.adbb2d47.{index_sources}

## How to cite

Cite this record by its DOI, and the sources above.
"""


def zenodo_metadata(summary: dict) -> dict:
    """The metadata of a Zenodo deposit (to review before uploading: creators, affiliations)."""
    index = summary.get("vulnerability", False)
    return {"metadata": {
        "title": f"Satellite-derived shorelines of the sand and gravel coasts of {summary['place']}, 1984-2025"
                 + (", with a coastal vulnerability index" if index else "")
                 + f" ({summary['name']} {summary['version']})",
        "upload_type": "dataset",
        "description": ("<p>Yearly shoreline positions and rates of change from Landsat 4-9 and Sentinel-2, corrected "
                        "for tide and waves, on the transects of the Global Coastal Transect System"
                        + (", and the coastal vulnerability index of each transect (USGS method)" if index else "")
                        + f". {summary['coverage']} Rates and yearly shorelines in one GeoParquet file each, "
                        "waterlines in one Parquet file per country; see README.md for the columns, the method, its "
                        f"validation and its limits. Made with CEVA: <a href=\"{CODE_URL}\">{CODE_URL}</a>.</p>"),
        "creators": [{"name": "Lacrouts, Clément"}],
        "keywords": ["shoreline", "coastal erosion", "satellite-derived shoreline", "Landsat", "Sentinel-2",
                     "Europe", "Mediterranean", "GCTS"] + (["coastal vulnerability index"] if index else []),
        "license": "cc-by-4.0", "version": summary["version"], "access_right": "open",
        "related_identifiers": RELATED,
        "notes": f"Made with CEVA {summary['ceva_version']} ({summary['method']}) on {summary['created'][:10]}.",
    }}
