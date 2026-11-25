"""The full dataset for download and citation (Zenodo): every product of a run, by country, with its documentation.

``ceva export OUT --run <run>`` writes, for each country (ISO code ``CC``):

* ``<name>_<CC>_rates.parquet`` (GeoParquet, the transects from their landward end): rate of shoreline change
  and its 95 % range, change class, years, precision, coastal type, the Shoreline Monitor's rate, and the
  coastal vulnerability index with its ranks and inputs (when ``ceva vulnerability compute`` has run);
* ``<name>_<CC>_shorelines.parquet`` (GeoParquet, points): the yearly shoreline position of each transect,
  its standard error and the number of images behind it;
* ``<name>_<CC>_waterlines.parquet`` (Parquet): every waterline of every image, with its tide, wave level,
  corrected position and whether it was used;

and ``README.md`` (columns, method, validation, limits, licence, how to cite), ``metadata.json`` (a Zenodo
deposit's metadata) and ``SHA256SUMS``. A Zenodo record holds up to 100 files, so 31 countries at most fit
in one record with the documentation.
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
import pyarrow.parquet as pq
from shapely.geometry import LineString

from ceva import __version__
from ceva.change_metrics import classify_lrr
from ceva.config import Config

log = logging.getLogger(__name__)

PRODUCTS = ("rates", "shorelines", "waterlines")
WATERLINE_COLUMNS = ["transect_id", "time", "platform", "family", "x_obs", "tide", "wave", "level", "x_corr",
                     "threshold", "outlier", "low_water", "use"]
SITE_URL = "https://clacrouts.github.io/ceva-dashboard/"     # the map and figures, the project's public page
SITE_CODE_URL = "https://github.com/clacrouts/ceva-dashboard"
RELATED = [
    {"identifier": "10.5281/zenodo.14056925", "relation": "isDerivedFrom", "scheme": "doi"},    # GCTS
    {"identifier": "10.5281/zenodo.15607678", "relation": "isDerivedFrom", "scheme": "doi"},    # typology
    {"identifier": "10.17882/79489", "relation": "references", "scheme": "doi"},                # EOT20
    {"identifier": "10.24381/cds.adbb2d47", "relation": "references", "scheme": "doi"},         # ERA5 single levels
    {"identifier": SITE_URL, "relation": "isSupplementedBy", "scheme": "url"},             # the dashboard
    {"identifier": SITE_CODE_URL, "relation": "isSupplementedBy", "scheme": "url"},
]

COLUMN_DOCS = {
    "rates": [
        ("transect_id", "GCTS transect id (Calkoen et al. 2025): stable worldwide"),
        ("country", "ISO 3166-1 alpha-2 code of the transect (GCTS)"),
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


def unit_tables(cfg: Config, name: str, method: str = "ceva",
                index: pd.DataFrame | None = None) -> dict[str, pd.DataFrame] | None:
    """The three products of one unit, each with the country of every transect; the rates with the
    vulnerability ``index`` (``ceva.vulnerability.read``) when given."""
    from ceva.transects import typology

    out = cfg.data_dir / "results" / name / method
    if not (out / "rates_imagery.parquet").exists():
        return None
    rates = pd.read_parquet(out / "rates_imagery.parquet")
    shores = gpd.read_parquet(out / "shorelines_imagery.parquet")
    lines = gpd.read_parquet(out / "transects.parquet").set_index("transect_id")
    water = pd.read_parquet(out / "waterlines.parquet", columns=WATERLINE_COLUMNS)
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
    tables = {"rates": rates, "shorelines": shores, "waterlines": water}
    for table in tables.values():
        table.insert(1, "country", table["transect_id"].astype(str).map(country))
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


def export(cfg: Config, out: Path, units: list[str], name: str = "ceva-shorelines", version: str = "1.0",
           method: str = "ceva", place: str = PLACE, coverage: str = COVERAGE) -> dict:
    """Write the dataset of these units into ``out`` (one file per country and product). Returns its summary.

    ``place`` names the coasts in the titles, ``coverage`` says exactly which transects are in."""
    out = Path(out)
    parts = out / ".parts"
    shutil.rmtree(parts, ignore_errors=True)
    from ceva import vulnerability

    done, index = [], vulnerability.read(cfg)
    for unit in units:
        tables = unit_tables(cfg, unit, method, index)
        if tables is None:
            log.warning("%s: no %s results, left out", unit, method)
            continue
        for product, table in tables.items():
            for cc, g in table.groupby("country", dropna=False):
                folder = parts / product / (str(cc) if pd.notna(cc) else "XX")
                folder.mkdir(parents=True, exist_ok=True)
                g.reset_index(drop=True).to_parquet(folder / f"{unit}.parquet", index=False)
        done.append(unit)
    if not done:
        raise RuntimeError("no unit has results to export")
    countries = sorted({p.name for p in (parts / "rates").iterdir()})
    if 3 * len(countries) + 3 > 100:
        raise RuntimeError(f"{len(countries)} countries: more files than a Zenodo record holds (100)")
    counts: dict[str, dict] = {}
    for cc in countries:
        counts[cc] = {}
        for product in PRODUCTS:
            files = sorted((parts / product / cc).glob("*.parquet")) if (parts / product / cc).exists() else []
            if not files:
                continue
            dest = out / f"{name}_{cc}_{product}.parquet"
            if product == "waterlines":            # large: streamed, file by file, to one schema
                schema = pq.read_schema(files[0])
                with pq.ParquetWriter(dest, schema, compression="zstd") as writer:
                    rows = 0
                    for f in files:
                        table = pq.read_table(f).cast(schema)
                        writer.write_table(table)
                        rows += table.num_rows
            else:
                table = pd.concat([gpd.read_parquet(f) for f in files], ignore_index=True)
                table = gpd.GeoDataFrame(table, geometry="geometry", crs=4326)
                table.to_parquet(dest, index=False, compression="zstd")
                rows = len(table)
            counts[cc][product] = rows
    shutil.rmtree(parts)
    summary = {"name": name, "version": version, "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
               "ceva_version": __version__, "method": method, "units": len(done), "countries": counts,
               "place": place, "coverage": coverage}
    (out / "README.md").write_text(readme(summary))
    (out / "metadata.json").write_text(json.dumps(zenodo_metadata(summary), indent=1))
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{_sha256(p)}  {p.name}\n" for p in files))
    log.info("exported %d units, %d countries, to %s", len(done), len(countries), out)
    return summary


def readme(summary: dict) -> str:
    """The dataset's README: what it is, its files and columns, method, validation, limits, licence, citation."""
    rows = "\n".join(f"| {cc} | {c.get('rates', 0):,} | {c.get('shorelines', 0):,} | {c.get('waterlines', 0):,} |"
                     for cc, c in summary["countries"].items())
    columns = "\n\n".join(f"**{product}**\n\n| Column | Meaning |\n|---|---|\n" + "\n".join(
        f"| `{c}` | {m} |" for c, m in COLUMN_DOCS[product]) for product in PRODUCTS)
    title = f"satellite-derived shorelines of the sand and gravel coasts of {summary['place']}"
    return f"""# {summary['name']} {summary['version']}: {title}

Yearly shoreline positions and rates of change, 1984 to 2025, on the transects of the Global Coastal
Transect System (one every 100 m along the coast), from every clear Landsat 4–9 and Sentinel-2 image,
moved to mean sea level with the tide and the waves of each image. Made with CEVA {summary['ceva_version']}
(method: the CEVA solution), {summary['created'][:10]}. Map and figures: {SITE_URL}

{summary['coverage']}

## Files

For each country `CC` (ISO 3166-1 alpha-2): `{summary['name']}_CC_rates.parquet` (GeoParquet, one row per
transect), `{summary['name']}_CC_shorelines.parquet` (GeoParquet, one row per transect and year) and
`{summary['name']}_CC_waterlines.parquet` (Parquet, one row per image and transect). `SHA256SUMS` lists
the checksums. All coordinates are WGS84 (EPSG:4326).

| Country | Transects with a rate | Yearly positions | Waterlines |
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
5. Only transects classed as sand or gravel coasts by the Global Coastal Typology are processed.

## Validation and limits

The method was checked against surveyed beach profiles at six beaches (Truc Vert, France; Narrabeen,
Australia; Noord-Holland and Walcheren, the Netherlands; Slapton Sands and Perranporth, United Kingdom):
yearly positions were 8 to 39 m from the surveyed ones (root mean square), the largest at the
macrotidal Perranporth. `se_m` is a precision from the data themselves, not a validated accuracy. Rates
over a few metres a year, or over many years, are more reliable than any single year. Global tide models
are weakest in estuaries and lagoons; muddy, rocky and engineered coasts are not covered. This is not an
official erosion map: national and regional observatories are the reference for planning.

## Licence and attribution

CC-BY-4.0. Contains modified Copernicus Sentinel data (ESA / EU). Landsat imagery courtesy of the U.S.
Geological Survey. Derived from the Global Coastal Transect System (doi:10.5281/zenodo.14056925) and the
Global Coastal Typology (doi:10.5281/zenodo.15607678), Calkoen et al. (2025), CC-BY-4.0. Tides: EOT20,
Hart-Davis et al. (2021), doi:10.17882/79489, CC-BY-4.0. Waves: ERA5, Hersbach et al. (2020), Copernicus
Climate Change Service, doi:10.24381/cds.adbb2d47.

## How to cite

Cite this record by its DOI, and the sources above.
"""


def zenodo_metadata(summary: dict) -> dict:
    """The metadata of a Zenodo deposit (to review before uploading: creators, affiliations)."""
    return {"metadata": {
        "title": f"Satellite-derived shorelines of the sand and gravel coasts of {summary['place']}, 1984-2025 "
                 f"({summary['name']} {summary['version']})",
        "upload_type": "dataset",
        "description": ("<p>Yearly shoreline positions and rates of change from Landsat 4-9 and Sentinel-2, corrected "
                        "for tide and waves, on the transects of the Global Coastal Transect System. "
                        f"{summary['coverage']} GeoParquet and Parquet files by country; see README.md "
                        "for the columns, the method, its validation and its limits.</p>"),
        "creators": [{"name": "Lacrouts, Clément"}],
        "keywords": ["shoreline", "coastal erosion", "satellite-derived shoreline", "Landsat", "Sentinel-2",
                     "Europe", "Mediterranean", "GCTS"],
        "license": "cc-by-4.0", "version": summary["version"], "access_right": "open",
        "related_identifiers": RELATED,
        "notes": f"Made with CEVA {summary['ceva_version']} ({summary['method']}) on {summary['created'][:10]}.",
    }}
