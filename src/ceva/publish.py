"""The dashboard's data: what the static site (``ceva-dashboard``) reads, written from the results of units.

``ceva publish`` gathers the yearly shoreline product and the rates of the CEVA solution for the units
of a batch run (and any named regions), adds each transect's country and coastal type, and writes into
the dashboard's ``data/`` folder:

* ``shoreline.pmtiles``: one point per transect, at its latest shoreline position, with its rate of
  change, 95 % interval, change class, precision, coastal type and the Shoreline Monitor's rate (vector
  tiles made by tippecanoe; the map reads them straight from the file with HTTP range requests).
* ``series/<unit>.json``: the yearly positions of the transects of one unit, fetched when one is clicked,
  with the ranks and inputs of their coastal vulnerability index (``ceva vulnerability compute``).
* ``summary.json``: totals by change class and by vulnerability class, for the whole coast and by country
  (the narrative sections).
* ``stats.json``: the transects grouped by what the site filters on (country, class, coastal type, shore,
  significance, built-up shore, defences, precision), with their rates: the filtered figures and the
  summary by country.
* ``refs/<unit>.json``: where other products cover a transect (the Shoreline Monitor, EMODnet's
  satellite and field data), their rate and years and CEVA's rate over the same years, shown on its card. These values
  stay under each product's own licence (``manifest.json``).
* ``validation.json``: how CEVA compares with those products and with surveys (the validation page).
* ``manifest.json``: what was published, when, from which method and code version, with attributions.

Everything is precomputed: the site needs no server, so any static host serves it.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ceva import __version__, vulnerability
from ceva.change_metrics import classify_lrr
from ceva.config import Config

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
CLASSES = ["extreme_erosion", "severe_erosion", "intense_erosion", "erosion", "stable", "accretion",
           "intense_accretion", "severe_accretion", "extreme_accretion"]
ATTRIBUTIONS = [
    "Contains modified Copernicus Sentinel data (ESA / EU)",
    "Landsat imagery courtesy of the U.S. Geological Survey (public domain)",
    "Transects: Global Coastal Transect System, Calkoen et al. (2025), CC-BY-4.0",
    "Coastal type and Shoreline Monitor rates: Global Coastal Typology, Calkoen et al. (2025), CC-BY-4.0",
    "Tides: EOT20, Hart-Davis et al. (2021), CC-BY-4.0",
    "Waves: ERA5, Hersbach et al. (2020), Copernicus Climate Change Service",
]
VULNERABILITY_ATTRIBUTIONS = [
    "Terrain and bathymetry: GEBCO Compilation Group (2024), GEBCO 2024 Grid, "
    "doi:10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f",
    "Sea-level trends: E.U. Copernicus Marine Service Information (OMI_CLIMATE_SL_GLOBAL_regional_trends)",
    "Glacial isostatic adjustment: ICE-6G_C (VM5a), Peltier et al. (2015), grids of Steffen (2021), CC-BY-4.0",
]
REFERENCE_ATTRIBUTIONS = {
    "emodnet_sat": "Reference rates: EMODnet Geology, coastline migration from satellite data (Deltares), CC BY-SA 4.0",
    "emodnet_field": "Reference classes: EMODnet Geology, coastline migration from field data, CC BY-SA 4.0",
}


def _round(values, digits: int):
    return [None if not np.isfinite(v) else round(float(v), digits) for v in np.asarray(values, dtype=float)]


def read_unit(cfg: Config, name: str, method: str = "ceva") -> tuple[pd.DataFrame, gpd.GeoDataFrame,
                                                                       gpd.GeoDataFrame] | None:
    """The rates, yearly shorelines and transects of one unit, or None if it has no results."""
    out = cfg.data_dir / "results" / name / method
    if not (out / "rates_imagery.parquet").exists():
        return None
    return (pd.read_parquet(out / "rates_imagery.parquet"), gpd.read_parquet(out / "shorelines_imagery.parquet"),
            gpd.read_parquet(out / "transects.parquet"))


def gather_unit(cfg: Config, name: str, method: str = "ceva",
                frames: tuple | None = None) -> tuple[gpd.GeoDataFrame, dict] | None:
    """One unit: a point per transect (latest shoreline, with its rate) and the yearly series of each."""
    frames = frames or read_unit(cfg, name, method)
    if frames is None:
        return None
    rates, shores, transects = frames
    transects = transects.set_index("transect_id")
    if shores.empty:
        return None
    latest = shores.sort_values("year").groupby("transect_id").tail(1).set_index("transect_id")
    se = shores.groupby("transect_id")["se_m"].median()
    points = latest[["geometry"]].join(rates.set_index("transect_id"), how="left").join(se.rename("se"))
    points = gpd.GeoDataFrame(points.reset_index(), geometry="geometry", crs=shores.crs)
    points["unit"] = name
    series = {}
    for tid, g in shores.sort_values("year").groupby("transect_id"):
        line = transects.loc[tid, "geometry"] if tid in transects.index else None
        sign = int(transects.loc[tid, "seaward_sign"]) if tid in transects.index else 1
        coords = list(line.coords) if line is not None else []
        if not coords:
            land = sea = None
        else:                                  # stored land to sea when the seaward sign is +1
            land, sea = (coords[0], coords[-1]) if sign >= 0 else (coords[-1], coords[0])
        series[str(tid)] = {
            "year": g["year"].astype(int).tolist(), "x": _round(g["x"], 1), "se": _round(g["se_m"], 1),
            "n": g["n_imagery"].astype(int).tolist(),
            "line": [[round(land[0], 6), round(land[1], 6)], [round(sea[0], 6), round(sea[1], 6)]] if land else None,
        }
    return points, series


def enrich(cfg: Config, points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Country (GCTS) and coastal type (Global Coastal Typology) of each transect, where known."""
    from ceva.transects import typology
    from ceva.transects.gcts import available_indexes

    ids = points["transect_id"].astype(str).tolist()
    countries = pd.concat([pd.read_parquet(p, columns=["transect_id", "country", "common_country_name"],
                                           filters=[("transect_id", "in", ids)]) for p in available_indexes(cfg)])
    points = points.merge(countries.drop_duplicates("transect_id"), on="transect_id", how="left")
    try:
        kinds = typology.load(cfg, ids)
    except FileNotFoundError:
        kinds = pd.DataFrame(columns=["transect_id"])
    keep = ["transect_id", "shore_type", "coastal_type", "is_built_environment", "has_defence", "sds_rate", "sds_qa"]
    return points.merge(kinds[[c for c in keep if c in kinds]], on="transect_id", how="left")


def feature_table(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The properties the map shows, short and rounded (they are in every vector tile)."""
    classes = points["rate"].map(lambda r: classify_lrr(r) if pd.notna(r) else "insufficient_data")
    significant = (points["rate_lo"] > 0) | (points["rate_hi"] < 0)
    table = pd.DataFrame({
        "id": points["transect_id"].astype(str), "unit": points["unit"],
        "rate": points["rate"].round(2), "lo": points["rate_lo"].round(2), "hi": points["rate_hi"].round(2),
        "cls": classes, "sig": significant.fillna(False).astype(int),
        "se": points["se"].round(1), "ny": points["n_years"].fillna(0).astype(int),
        "y0": points["first_year"], "y1": points["last_year"],
        "cc": points.get("country"), "shore": points.get("shore_type"), "coast": points.get("coastal_type"),
        "built": points.get("is_built_environment"), "def": points.get("has_defence"),
        "sds": points["sds_rate"].round(2) if "sds_rate" in points else None,
        "smd": points["smd"].round(2) if "smd" in points else None,
        "cvi": points["cvi"].round(2) if "cvi" in points else None,
        "cvic": points["cvi_class"].where(points["cvi_class"] >= 0).astype("Int64") if "cvi_class" in points else None,
    })
    return gpd.GeoDataFrame(table, geometry=points.geometry.values, crs=points.crs).to_crs(4326)


def summary(table: pd.DataFrame) -> dict:
    """Totals by change class for the whole coast and each country."""
    def block(g: pd.DataFrame) -> dict:
        rated = g[g["cls"] != "insufficient_data"]
        bounds = g.total_bounds if hasattr(g, "total_bounds") else [np.nan] * 4
        return {
            "transects": int(len(g)), "rated": int(len(rated)),
            "bounds": [round(float(v), 4) for v in bounds],
            "classes": {c: int((rated["cls"] == c).sum()) for c in CLASSES},
            "significant_erosion": int(((rated["hi"] < 0)).sum()),
            "significant_accretion": int(((rated["lo"] > 0)).sum()),
            "median_rate": None if rated.empty else round(float(rated["rate"].median()), 2),
            **({"vulnerability": vulnerability_block(g)} if "cvic" in g and g["cvic"].notna().any() else {}),
        }

    names = table.get("cn", table.get("cc"))
    return {"all": block(table),
            "countries": {str(cc): {"name": str(names[g.index[0]]) if names is not None else str(cc), **block(g)}
                          for cc, g in table.groupby("cc") if pd.notna(cc)}}


def vulnerability_block(g: pd.DataFrame) -> dict:
    """Transects by vulnerability class (low to very high) and the median index, where it is known."""
    known = g[g["cvic"].notna()]
    return {"known": int(len(known)), "classes": [int((known["cvic"] == k).sum()) for k in range(4)],
            "median": round(float(known["cvi"].median()), 2)}


def vulnerability_entries(index: pd.DataFrame, ids) -> dict[str, dict]:
    """What a transect's card shows of its index: the six ranks (``ceva.vulnerability.VARIABLES``) and the
    inputs behind them."""
    from ceva.vulnerability import VARIABLES

    rows = index.reindex([str(i) for i in ids]).dropna(subset=["cvi"])
    ranks = rows[[f"rank_{v}" for v in VARIABLES]].to_numpy()
    inputs = {"slope": "coastal_slope_pct", "sl": "sea_level_trend_mm_yr", "vlm": "uplift_mm_yr",
              "rsl": "relative_sea_level_mm_yr", "tide": "mean_tidal_range_m", "hs": "mean_wave_height_m"}
    values = {k: _round(rows[c], 2) for k, c in inputs.items()}
    return {str(tid): {"r": [int(r) for r in ranks[i]], **{k: values[k][i] for k in inputs}}
            for i, tid in enumerate(rows.index)}


PRECISION_BINS = [5, 10, 20]     # m, standard error of a yearly position: under 5, 5-10, 10-20, over 20 (or unknown)
STAT_KEYS = ["cc", "coast", "shore", "cls", "sig", "built", "def", "prec"]


def _flag(values: pd.Series) -> np.ndarray:
    """0 no, 1 yes, 2 unknown."""
    known = values.notna()
    return np.where(known, values.where(known, False).astype(bool).astype(int), 2)


def filter_stats(table: pd.DataFrame) -> dict:
    """The transects grouped by everything the site filters on, each group with its rates (cm/yr, sorted).

    The site adds groups up for any choice of country, change class, coastal type, shore, significance,
    built-up shore, defences and precision, and takes medians from their rates: its figures match the
    filtered map exactly, with no per-transect file to download.
    """
    def column(name: str, missing: str) -> pd.Series:
        values = table[name] if name in table else None
        values = values if values is not None else pd.Series(index=table.index, dtype=object)
        return values.astype(object).where(values.notna(), missing).astype(str)

    t = pd.DataFrame({"cc": column("cc", ""), "coast": column("coast", "unknown"), "shore": column("shore", "unknown"),
                      "cls": table["cls"].astype(str), "sig": table["sig"].astype(int).to_numpy(),
                      "built": _flag(table["built"]) if "built" in table else 2,
                      "def": _flag(table["def"]) if "def" in table else 2,
                      "prec": np.digitize(table["se"].fillna(np.inf), PRECISION_BINS),
                      "rate": (table["rate"] * 100).round()})
    keys = {k: sorted(t[k].unique()) for k in ("cc", "coast", "shore")}
    keys["cls"] = CLASSES + ["insufficient_data"]
    for k, values in keys.items():
        t[k] = t[k].map({v: i for i, v in enumerate(values)})
    groups = [[*(int(v) for v in key), int(len(g)), sorted(int(r) for r in g["rate"].dropna())]
              for key, g in t.groupby(STAT_KEYS, sort=True)]
    return {"v": FORMAT_VERSION, "keys": keys, "precision_bins": PRECISION_BINS,
            "columns": STAT_KEYS + ["n", "rates"], "groups": groups}


def _tippecanoe() -> str | None:
    """tippecanoe on the PATH, or installed next to this Python (conda-forge: tippecanoe)."""
    import sys

    beside = Path(sys.executable).with_name("tippecanoe")
    return shutil.which("tippecanoe") or (str(beside) if beside.exists() else None)


GITHUB_FILE_LIMIT = 100 * 2**20      # GitHub refuses larger files in a push (GitHub Pages hosting)


POINT_TILES = ["-l", "shoreline", "-Z2", "-z11", "-B6", "-r1.8", "--drop-densest-as-needed",
               "--extend-zooms-if-still-dropping"]
LINE_TILES = ["-l", "coverage", "-Z3", "-z11", "--drop-smallest-as-needed", "--extend-zooms-if-still-dropping"]


def write_tiles(table: gpd.GeoDataFrame, path: Path, tippecanoe: str | None = None, options=POINT_TILES) -> Path:
    """Vector tiles (tippecanoe, PMTiles). The points: every one from zoom 6, thinned below (and where dense).

    Tiles stop at zoom 11 (about 5 m precision); the map enlarges them beyond. Every zoom level
    repeats every point, so each level more adds about a sixth to the file."""
    tippecanoe = tippecanoe or _tippecanoe()
    if tippecanoe is None:
        raise FileNotFoundError("tippecanoe is not installed (conda-forge: tippecanoe)")
    source = path.with_suffix(".geojsonl")
    with open(source, "w") as f:
        for row in json.loads(table.to_json(drop_id=True))["features"]:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    subprocess.run([tippecanoe, "-o", str(path), "--force", "--quiet", *options, str(source)], check=True)
    source.unlink()
    size = path.stat().st_size
    if size > 0.9 * GITHUB_FILE_LIMIT:
        log.warning("%s is %.0f MB, close to GitHub's 100 MB limit per file: split the tiles by region or host "
                    "them elsewhere (see the dashboard's README)", path.name, size / 2**20)
    return path


GCTS_STEP_M = 100                    # GCTS transect ids: <coastline>tr<metres along it>, every 100 m


def coverage_lines(transects: pd.DataFrame, measured: set[str], shore_types: list[str]) -> gpd.GeoDataFrame:
    """The coast the dataset does not measure, as lines along it: runs of consecutive transects (100 m apart
    on one GCTS coastline) of one kind, through their origins; a lone transect is drawn 100 m long, across
    its bearing.

    ``transects`` has transect_id, lon, lat, bearing and shore_type; ``measured`` holds the published ids.
    The kind (``why``) is the coast type when the dataset does not measure that type (``shore_types``),
    else ``no_result``: a measured type where no rate could be made (a spit, a lagoon, a narrow island)."""
    import shapely

    t = transects[~transects["transect_id"].astype(str).isin(measured)].copy()
    kind = t["shore_type"].fillna("unknown").astype(str)
    t["why"] = np.where(kind.isin(shore_types), "no_result", kind)
    parts = t["transect_id"].astype(str).str.extract(r"^(cl\d+s\d+)tr(\d+)$")
    t["line"], t["pos"] = parts[0], parts[1].astype("int64")
    t = t.sort_values(["line", "pos"]).reset_index(drop=True)
    new = ((t["line"] != t["line"].shift()) | (t["why"] != t["why"].shift())
           | (t["pos"] - t["pos"].shift() != GCTS_STEP_M))
    t["run"] = new.cumsum() - 1
    size = t.groupby("run")["run"].transform("size")
    lone, many = t[size == 1], t[size > 1]
    along = np.radians(lone["bearing"].to_numpy() + 90.0)
    dlat = GCTS_STEP_M / 2 / 111_320 * np.cos(along)
    dlon = GCTS_STEP_M / 2 / (111_320 * np.cos(np.radians(lone["lat"].to_numpy()))) * np.sin(along)
    x, y = lone["lon"].to_numpy(), lone["lat"].to_numpy()
    short = shapely.linestrings(np.stack([np.column_stack([x - dlon, y - dlat]),
                                          np.column_stack([x + dlon, y + dlat])], axis=1))
    long_ids = many["run"].to_numpy()
    long = shapely.linestrings(many[["lon", "lat"]].to_numpy(), indices=np.unique(long_ids, return_inverse=True)[1])
    why = pd.concat([lone["why"], many.drop_duplicates("run")["why"]], ignore_index=True)
    return gpd.GeoDataFrame({"why": why}, geometry=np.concatenate([short, long]), crs=4326)


def write_coverage(cfg: Config, out: Path, measured: set[str], area) -> dict[str, int]:
    """``coverage.pmtiles``: the coast of ``area`` (``ceva.batch.load_area``) the dataset does not measure.
    Returns the number of transects of each kind, measured ones included."""
    from ceva.batch import in_area
    from ceva.transects import typology
    from ceva.transects.gcts import available_indexes

    frames = []
    for index in available_indexes(cfg):
        t = pd.read_parquet(index, columns=["transect_id", "lon", "lat", "bearing", "continent"])
        frames.append(t[in_area(area, t["lon"], t["lat"], t["continent"])])
    t = pd.concat(frames, ignore_index=True)
    t = t.merge(typology.load(cfg)[["transect_id", "shore_type"]], on="transect_id", how="left")
    lines = coverage_lines(t, measured, cfg.batch.shore_types)
    write_tiles(lines, out / "coverage.pmtiles", options=LINE_TILES)
    counts = t.assign(why=np.where(t["transect_id"].astype(str).isin(measured), "measured",
                                   t["shore_type"].fillna("unknown").astype(str)))
    counts.loc[(counts["why"].isin(cfg.batch.shore_types)), "why"] = "no_result"
    return {k: int(v) for k, v in counts["why"].value_counts().items()}


def _write_json(path: Path, data: dict, indent: int | None = None) -> None:
    path.write_text(json.dumps(data, indent=indent, separators=None if indent else (",", ":")))


def _remove_stale(folder: Path, keep: set[str]) -> None:
    for stale in folder.glob("*.json"):                   # units of an earlier publication
        if stale.stem not in keep:
            stale.unlink()


def publish(cfg: Config, out: Path, units: list[str], method: str = "ceva", references: bool = True,
            run: dict | None = None, area=None) -> dict:
    """Write the dashboard's data for these units into ``out``. Returns the manifest.

    With ``references`` (and products prepared by ``ceva references prepare``), each transect is
    also compared with the other products that cover it. ``run`` describes the batch run (planned
    and finished units), so the validation page says whether it is provisional. With ``area``, the
    coast of that area the dataset does not measure is drawn too (``write_coverage``).
    """
    from ceva.validation import agreement
    from ceva.validation import references as refs_mod

    out = Path(out)
    for folder in ("series", "refs"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    refs = refs_mod.load_all(cfg) if references else {}
    index = vulnerability.read(cfg)
    if index is not None:
        edges, made_from = index.attrs.get("class_edges", []), index.attrs.get("sources", {})
        index = index.astype({"transect_id": str}).drop_duplicates("transect_id").set_index("transect_id")
    frames, written, with_refs, pairs = [], [], [], []
    for name in units:
        unit = read_unit(cfg, name, method)
        got = gather_unit(cfg, name, method, unit)
        if got is None:
            log.warning("%s: no %s results, left out", name, method)
            continue
        points, series = got
        frames.append(points)
        if index is not None:
            for tid, entry in vulnerability_entries(index, series).items():
                series[tid]["vi"] = entry
        _write_json(out / "series" / f"{name}.json", {"v": FORMAT_VERSION, "unit": name, "transects": series})
        written.append(name)
        if refs:
            _, shores, transects = unit
            matched = refs_mod.match(refs, points, shores, transects)
            if not matched.empty:
                pairs.append(matched.assign(unit=name))
                _write_json(out / "refs" / f"{name}.json", {"v": FORMAT_VERSION, "unit": name,
                                                            "transects": refs_mod.card_entries(matched)})
                with_refs.append(name)
    if not frames:
        raise RuntimeError("no unit has results to publish")
    _remove_stale(out / "series", set(written))
    _remove_stale(out / "refs", set(with_refs))
    pairs = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(columns=refs_mod.COLUMNS + ["unit"])
    points = enrich(cfg, pd.concat(frames, ignore_index=True))
    if index is not None:
        points = points.join(index[["cvi", "cvi_class"]], on=points["transect_id"].astype(str))
    sm = pairs[(pairs["product"] == "sm") & pairs["qa"].astype("boolean").fillna(False)]
    points["smd"] = points["transect_id"].astype(str).map(
        (sm["ceva_rate"] - sm["rate"]).set_axis(sm["transect_id"].astype(str)))
    table = feature_table(points)
    table["cn"] = points.get("common_country_name")
    write_tiles(table, out / "shoreline.pmtiles")
    stats = summary(table)
    (out / "summary.json").write_text(json.dumps(stats, indent=1))
    _write_json(out / "stats.json", filter_stats(table))
    products = [k for k in refs_mod.PRODUCTS if k in set(pairs["product"])]
    validation = None
    if products:
        validation = agreement.payload(pairs, table, agreement.load_sites(cfg), run)
        _write_json(out / "validation.json", validation)
    w, s, e, n = table.total_bounds
    years = [int(table["y0"].min()), int(table["y1"].max())]
    manifest = {
        "format": FORMAT_VERSION, "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "ceva_version": __version__, "method": method, "units": len(written), "transects": int(len(table)),
        "years": years, "bounds": [round(float(v), 4) for v in (w, s, e, n)],
        "classes": CLASSES, "files": {"tiles": "shoreline.pmtiles", "series": "series/{unit}.json",
                                      "summary": "summary.json", "stats": "stats.json"},
        "licence": "CC-BY-4.0", "attribution": ATTRIBUTIONS + [REFERENCE_ATTRIBUTIONS[k] for k in products
                                                             if k in REFERENCE_ATTRIBUTIONS],
    }
    if run is not None:                                   # the site says it is a preview until the run ends
        manifest["provisional"] = bool(run.get("planned") and run.get("finished") is not None
                                       and run["finished"] < run["planned"])
    if area is not None:
        manifest["files"]["coverage"] = "coverage.pmtiles"
        manifest["coverage"] = write_coverage(cfg, out, set(table["id"]), area)
    else:
        (out / "coverage.pmtiles").unlink(missing_ok=True)
    if index is not None and table["cvic"].notna().any():
        manifest["vulnerability"] = {
            "transects": int(table["cvic"].notna().sum()), "class_edges": edges,
            "classes": list(vulnerability.CLASSES), "variables": list(vulnerability.VARIABLES),
            "sources": made_from}
        manifest["attribution"] += VULNERABILITY_ATTRIBUTIONS
    if products:
        manifest["files"] |= {"refs": "refs/{unit}.json", "validation": "validation.json"}
        manifest["references"] = refs_mod.product_info(products)
        quality = agreement.survey_ranges(validation.get("sites", []))
        if quality:
            manifest["validation"] = quality
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("published %d transects of %d units to %s", len(table), len(written), out)
    return manifest
