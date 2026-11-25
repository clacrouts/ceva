"""Run the pipeline for one unit (a set of transects): cube, tides, waves, engines, tables.

Results are Parquet tables in ``<data_dir>/results/<name>/<variant>/``, with a
``run.json`` recording the cube snapshot, the configuration and the software
version they came from. There are two variants, one per shoreline detection method,
run on the same cube so that they can be compared: ``coastsat`` (CoastSat's pixel
classifier and 2D contour on image chips) and ``ceva`` (CEVA's MNDWI profiles along each
transect with an Otsu threshold). Everything else (tide, search window that follows the
tide, wave correction, slope) is shared.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from ceva import __version__
from ceva.config import Config
from ceva.cube import build_cube, open_cube, save_cube
from ceva.engines import (
    Profiles,
    annual_from_waterlines,
    annual_shorelines,
    change_rates,
    estimate_slopes,
    flag_outliers,
    load_profiles,
    perimage_waterlines,
    tide_correct,
    tide_stage_curves,
)
from ceva.tides import TideTable

log = logging.getLogger(__name__)

SENSOR_SUBSETS = ("all", "landsat", "sentinel2")     # "all" is the imagery (Landsat and Sentinel-2)
RADAR_SUBSETS = ("sentinel1", "combined")              # radar alone; imagery and aligned radar together
IMAGERY = ("landsat", "sentinel2")
DEFAULT_VARIANT = "ceva"   # the production method; CoastSat is kept for comparison at validation sites

# Named method variants, as configuration overrides on top of the defaults.
VARIANTS: dict[str, dict] = {
    # CoastSat's detection (pixel classifier, sand/water threshold, 2D contour) on chips, with
    # Sentinel-2 as top-of-atmosphere reflectance
    "coastsat": {"perimage": {"engine": "coastsat"}},
    # CEVA's detection: MNDWI along each transect, Otsu threshold near the expected shoreline
    "ceva": {"perimage": {"engine": "ceva"}},
}


@dataclass
class Results:
    name: str
    transects: gpd.GeoDataFrame     # one row per transect: geometry, slope, orientation
    waterlines: pd.DataFrame        # per image: x_obs, tide, wave, level, x_corr, outlier, use
    annual: pd.DataFrame            # per year: engine, sensors, x, n_obs
    rates: pd.DataFrame             # per transect: engine, sensors, rate and interval
    run: dict
    radar: pd.DataFrame | None = None              # radar waterlines, aligned on the imagery
    radar_calibration: pd.DataFrame | None = None  # per transect and orbit direction


def results_dir(cfg: Config, name: str, variant: str = DEFAULT_VARIANT) -> Path:
    return cfg.data_dir / "results" / name / variant


def build(cfg: Config, name: str, transects: gpd.GeoDataFrame, **kwargs) -> str:
    """Build the cube for these transects and commit it. Returns the snapshot id."""
    ds = build_cube(cfg, transects, **kwargs)
    message = f"{name}: {ds.sizes['time']} acquisitions, {ds.sizes['transect']} transects, {ds.attrs['period']}"
    snapshot = save_cube(cfg, name, ds, message)
    log.info("cube %s committed as %s", name, snapshot)
    return snapshot


def build_chips(cfg: Config, name: str, transects: gpd.GeoDataFrame,
                families: tuple[str, ...] = ("landsat", "sentinel2"), only_missing: bool = False,
                **kwargs) -> dict[str, str]:
    """Read the 2D chips around these transects (for the CoastSat engine) and commit them.

    Each family is read at the configured reflectance when its source is usable, else as
    surface reflectance. With ``only_missing``, families already stored at that level are
    skipped. Returns the snapshot id per reflectance level written.
    """
    return build_chips_batch(cfg, {name: transects}, families, only_missing, **kwargs).get(name, {})


def build_chips_batch(cfg: Config, units: dict[str, gpd.GeoDataFrame],
                      families: tuple[str, ...] = ("landsat", "sentinel2"), only_missing: bool = False,
                      **kwargs) -> dict[str, dict[str, str]]:
    """``build_chips`` for several units at once: top-of-atmosphere files they share are read once.

    Units on the same Sentinel-2 tile need the same Level-1C files; read together, each file is
    downloaded once for all of them (``ceva.chips.iter_chips_batch``). The units must share the
    chip settings (``cfg``). Returns, per unit, the snapshot id per reflectance level written.
    """
    from ceva.chips import iter_chips_batch, reflectance_levels, save_chips_batch, segments, stored_families
    from ceva.credentials import load_credentials

    creds = kwargs.pop("creds", None) or load_credentials()
    levels = reflectance_levels(cfg, families, creds)
    snapshots: dict[str, dict[str, str]] = {name: {} for name in units}
    for level in sorted(set(levels.values())):
        for family in (f for f in families if levels[f] == level):
            todo = {name: t for name, t in units.items()
                    if not (only_missing and family in stored_families(cfg, name, level))}
            if not todo:
                continue
            messages = {name: f"{name}: {family} chips ({level} reflectance) for {len(t)} transects in "
                              f"{len(segments(t, cfg.chips.segment_km))} coastal segments" for name, t in todo.items()}
            written = save_chips_batch(cfg, iter_chips_batch(cfg, todo, families=(family,), level=level,
                                                             creds=creds, **kwargs), messages, level)
            for name, snapshot in written.items():
                snapshots[name][level] = snapshot
                log.info("%s chips %s committed as %s", level, name, snapshot)
    return snapshots


def _coastsat_cached(cfg: Config, name: str, chip_snapshots: dict, annual: pd.DataFrame, compute,
                     curves: pd.DataFrame | None = None):
    """CoastSat detection, reused by the variants that differ only in the water-level correction.

    Keyed by everything the detection depends on: the chips, the CoastSat and chip settings,
    the reference lines (through the annual shorelines and the tide-stage curves) and the tide model.
    """
    import importlib.metadata

    try:
        package = importlib.metadata.version("coastsat-package")
    except importlib.metadata.PackageNotFoundError:
        package = None
    inputs = {
        "chips": chip_snapshots, "coastsat": cfg.coastsat.model_dump(mode="json"),
        "chip_cfg": cfg.chips.model_dump(mode="json"), "tides": cfg.tides.model_dump(mode="json"),
        "annual": annual[["transect_id", "year", "x"]].round(3).astype(str).values.tolist(),
        "coastsat_package": package, "ceva": __version__,
    }
    if curves is not None:
        inputs["annual_tide"] = annual["tide_mean"].round(3).astype(str).tolist()
        inputs["curves"] = curves.round(3).astype(str).values.tolist()
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]
    folder = cfg.data_dir / "results" / name / "_cache" / f"coastsat-{key}"
    if (folder / "waterlines.parquet").exists():
        log.info("reusing CoastSat detections %s", key)
        return pd.read_parquet(folder / "waterlines.parquet"), pd.read_parquet(folder / "detections.parquet")
    folder.mkdir(parents=True, exist_ok=True)
    raw, detections = compute(folder / "segments")      # each finished segment is kept there, for a restart
    raw.to_parquet(folder / "waterlines.parquet", index=False)
    detections.to_parquet(folder / "detections.parquet", index=False)
    return raw, detections


def _radar_waterlines(cfg: Config, name: str, ds, profiles: Profiles, annual: pd.DataFrame, curves: pd.DataFrame,
                      imagery: pd.DataFrame, slopes: pd.DataFrame, waves, chips: dict, tide_fn: TideTable):
    """Radar waterlines of the configured detection method, aligned on the imagery's waterlines.

    Returns the radar waterlines, the alignment table (per transect and orbit direction) and,
    for the CEVA solution, the radar templates learned at the imagery's waterlines. Detections
    are cached, keyed by everything they depend on; the alignment is redone every run.
    """
    from ceva.sar_waterlines import align_on_imagery, ceva_sar_waterlines, coastsat_sar_waterlines

    unit = pd.DataFrame({
        "transect_id": profiles.transect_id, "lon_start": ds["lon_start"].values, "lat_start": ds["lat_start"].values,
        "lon_end": ds["lon_end"].values, "lat_end": ds["lat_end"].values, "seaward_sign": profiles.seaward_sign,
    })
    ids = pd.Index(profiles.transect_id.astype(str))
    tan_beta = slopes.set_index("transect_id")["slope_used"]
    tan_beta.index = tan_beta.index.astype(str)

    # the imagery's wave correction, at any time: coefficient x (Hs - the transect's mean Hs at its images)
    if waves is not None and cfg.waves.component == "empirical":
        from ceva.waves import empirical_level, hs_at

        coef = cfg.waves.empirical.hs_coef[cfg.perimage.engine]
        hs_imagery = pd.Series(hs_at(waves, imagery["time"]), index=imagery.index)
        hs_mean = hs_imagery.groupby(imagery["transect_id"].astype(str)).mean().reindex(ids).fillna(hs_imagery.mean())

        def wave_fn(when):
            hs = hs_at(waves, pd.DatetimeIndex(when))
            return empirical_level(hs[:, None], hs_mean.to_numpy()[None, :], coef)
    else:
        hs = None

        def wave_fn(when):
            return np.zeros((len(when), len(ids)), dtype="float32")

    inputs = {
        "radar_chips": {k: c.attrs.get("snapshot_id") for k, c in chips.items()}, "engine": cfg.perimage.engine,
        "sar": cfg.sar.model_dump(mode="json", exclude={"calibration", "effective_per_year"}),
        "coastsat": cfg.coastsat.model_dump(mode="json"),
        "perimage": cfg.perimage.model_dump(mode="json", include={"max_dist_ref_m", "min_valid_fraction",
                                                                  "tide_stage_bins", "tide_stage_min_obs"}),
        "tides": cfg.tides.model_dump(mode="json"), "ceva": __version__,
        "annual": annual[["transect_id", "year", "x", "tide_mean"]].round(3).astype(str).values.tolist(),
        "curves": curves.round(3).astype(str).values.tolist(),
    }
    if cfg.perimage.engine == "ceva":      # the templates are learned at the imagery's waterlines
        inputs["imagery"] = str(pd.util.hash_pandas_object(
            imagery[["transect_id", "time", "x_corr", "use"]], index=False).sum())
        inputs["waves"] = cfg.waves.model_dump(mode="json")
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]
    folder = cfg.data_dir / "results" / name / "_cache" / f"radar-{key}"
    templates = None
    if (folder / "waterlines.parquet").exists():
        log.info("reusing radar detections %s", key)
        raw = pd.read_parquet(folder / "waterlines.parquet")
        if (folder / "templates.parquet").exists():
            templates = pd.read_parquet(folder / "templates.parquet")
    else:
        if cfg.perimage.engine == "coastsat":
            raw = coastsat_sar_waterlines(cfg, chips, unit, annual, curves, tide_fn)
        else:
            raw, templates = ceva_sar_waterlines(cfg, chips, ds, profiles.seaward_sign, annual, curves, tide_fn,
                                                 imagery, tan_beta, wave_fn)
        folder.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(folder / "waterlines.parquet", index=False)
        if templates is not None:
            templates.to_parquet(folder / "templates.parquet", index=False)
    if raw.empty:
        return None, None, None
    times = pd.DatetimeIndex(sorted(raw["time"].unique()))
    level = wave_fn(times)
    raw = raw.copy()
    raw["wave"] = level[times.get_indexer(raw["time"]), ids.get_indexer(raw["transect_id"].astype(str))]
    hs = None
    if waves is not None:
        from ceva.waves import hs_at

        hs = hs_at(waves, raw["time"])
    radar, calibration = align_on_imagery(cfg, raw, imagery, tan_beta, hs)
    return radar, calibration, templates


def uses_chips(cfg: Config) -> bool:
    return cfg.perimage.engine == "coastsat"


def _unit_waves(cfg: Config, ds, sign: np.ndarray, times: pd.DatetimeIndex):
    if cfg.waves.source == "none" or cfg.waves.component == "none":
        return None, None
    from ceva.waves import waves_for_unit

    start = (times.min() - pd.Timedelta(days=2)).date()
    end = min((times.max() + pd.Timedelta(days=2)).date(), dt.date.today() - dt.timedelta(days=7))
    return waves_for_unit(cfg, ds["lon_start"].values, ds["lat_start"].values, ds["lon_end"].values,
                          ds["lat_end"].values, sign, start, end)


def _slopes(cfg: Config, waterlines: pd.DataFrame, level: str) -> tuple[pd.DataFrame, float]:
    c = cfg.perimage
    est = estimate_slopes(waterlines, c.slope_bounds, level=level, method=c.slope_method)
    unit = float(np.nanmedian(est["slope"])) if est["slope"].notna().any() else c.default_slope
    est["slope_used"] = est["slope"].fillna(unit)
    est["slope_source"] = np.where(est["slope"].notna(), "estimated", "unit median")
    return est, unit


def correct_waterlines(cfg: Config, waterlines: pd.DataFrame, waves: pd.DataFrame | None,
                       fixed_slope: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Add wave level, total water level, slope and corrected position to per-image waterlines.

    The slope is estimated from the tide, then again from tide + the wave correction.
    """
    w = waterlines.copy()
    w["wave"] = np.float32(0.0)
    w["level"] = w["tide"]
    if fixed_slope is not None:
        slopes = pd.DataFrame({"transect_id": w["transect_id"].unique()})
        slopes["slope"], slopes["n_slope_obs"] = np.nan, 0
        slopes["slope_used"], slopes["slope_source"], unit = fixed_slope, "given", fixed_slope
    else:
        slopes, unit = _slopes(cfg, w, "tide")
    if waves is not None and cfg.waves.component == "empirical":
        from ceva.waves import empirical_level, hs_at

        hs = pd.Series(hs_at(waves, w["time"]), index=w.index)
        coef = cfg.waves.empirical.hs_coef[cfg.perimage.engine]
        w["wave"] = empirical_level(hs, hs.groupby(w["transect_id"]).transform("mean"), coef)
        w["level"] = w["tide"] + w["wave"]
        if fixed_slope is None:
            slopes, unit = _slopes(cfg, w, "level")
    by_transect = slopes.set_index("transect_id")["slope_used"]
    w["x_corr"] = tide_correct(w, by_transect, cfg.tides.reference_elevation_m, level="level")
    w["outlier"] = flag_outliers(w, "x_corr", cfg.perimage.outlier_mad_k)
    threshold = cfg.perimage.min_total_water_level_m
    w["low_water"] = (w["level"] < threshold) if threshold is not None else False
    w["use"] = ~w["outlier"] & ~w["low_water"]
    return w, slopes, unit


def run(cfg: Config, name: str, variant: str = DEFAULT_VARIANT, fixed_slope: float | None = None) -> Results:
    """Run both engines on a stored cube and write the result tables for one variant."""
    ds = open_cube(cfg, name).load()
    times = pd.DatetimeIndex(ds["time"].values)
    lon = (ds["lon_start"].values + ds["lon_end"].values) / 2.0
    lat = (ds["lat_start"].values + ds["lat_end"].values) / 2.0
    chips, radar_chips = None, None
    if cfg.perimage.engine == "coastsat":
        from ceva.chips import open_chips

        chips = open_chips(cfg, name)
    if cfg.sar.enabled:
        from ceva.sar import open_sar_chips

        radar_chips = open_sar_chips(cfg, name) or None
    # the tide model reads its files on every call: one call for every time this run needs
    tides = TideTable(cfg, lon, lat)
    tides.prefetch(times.append([pd.DatetimeIndex(c["time"].values)
                                 for c in [*(chips or {}).values(), *(radar_chips or {}).values()]]))
    tide = tides(times)
    profiles: Profiles = load_profiles(ds, cfg, tide)
    subsets = {"all": None, "landsat": profiles.family == "landsat", "sentinel2": profiles.family == "sentinel2"}

    composites = [annual_shorelines(profiles, cfg, subset, label) for label, subset in subsets.items()
                  if subset is None or subset.any()]
    curves = tide_stage_curves(profiles, cfg)
    detections, chip_snapshots, chip_levels = None, None, None
    if cfg.perimage.engine == "coastsat":
        from ceva.engines_coastsat import coastsat_waterlines

        chip_snapshots = {key: c.attrs.get("snapshot_id") for key, c in chips.items()}
        chip_levels = {key.split("/")[0]: c.attrs.get("reflectance", "surface") for key, c in chips.items()}
        unit = pd.DataFrame({
            "transect_id": profiles.transect_id,
            "lon_start": ds["lon_start"].values, "lat_start": ds["lat_start"].values,
            "lon_end": ds["lon_end"].values, "lat_end": ds["lat_end"].values, "seaward_sign": profiles.seaward_sign,
        })
        def detect(segment_cache):
            return coastsat_waterlines(cfg, chips, unit, composites[0], tides, segment_cache, curves)

        raw, detections = _coastsat_cached(cfg, name, chip_snapshots, composites[0], detect, curves)
        times = times.union(pd.DatetimeIndex(raw["time"].unique()))
    else:
        raw = perimage_waterlines(profiles, composites[0], cfg, curves)
    radar_times = (pd.DatetimeIndex(sorted({t for c in radar_chips.values() for t in c["time"].values}))
                   if radar_chips else pd.DatetimeIndex([]))
    waves, wave_point = _unit_waves(cfg, ds, profiles.seaward_sign, times.union(radar_times))
    waterlines, slopes, unit_slope = correct_waterlines(cfg, raw, waves, fixed_slope)
    radar, radar_calibration, radar_templates, radar_snapshot = None, None, None, None
    if radar_chips:
        radar, radar_calibration, radar_templates = _radar_waterlines(
            cfg, name, ds, profiles, composites[0], curves, waterlines, slopes, waves, radar_chips, tides)
        radar_snapshot = next(iter(radar_chips.values())).attrs.get("snapshot_id")

    per_image = [annual_from_waterlines(waterlines if label == "all" else waterlines[waterlines["family"] == label],
                                        "x_corr", label=label) for label in subsets]
    if radar is not None and not radar.empty:
        from ceva.sar_waterlines import combined_yearly

        combined = combined_yearly(waterlines, radar, cfg)
        radar_calibration.attrs["combination"] = dict(combined.attrs)
        per_image += [annual_from_waterlines(radar, "x_corr", label="sentinel1"), combined]
    annual = pd.concat([f.assign(engine="annual") for f in composites]
                       + [f.assign(engine="perimage") for f in per_image], ignore_index=True)
    if annual.empty:
        from ceva.shoreline import NoShoreline

        raise NoShoreline("no yearly position on any transect")
    rates = pd.concat([change_rates(g).assign(engine=engine, sensors=sensors)
                       for (engine, sensors), g in annual.groupby(["engine", "sensors"])], ignore_index=True)

    table = pd.DataFrame({
        "transect_id": profiles.transect_id, "seaward_sign": profiles.seaward_sign,
        "length": ds["length"].values, "lon": lon, "lat": lat,
    }).merge(slopes, on="transect_id", how="left")
    geometry = [LineString([(a, b), (c, d)]) for a, b, c, d in zip(
        ds["lon_start"].values, ds["lat_start"].values, ds["lon_end"].values, ds["lat_end"].values, strict=True)]
    transects = gpd.GeoDataFrame(table, geometry=geometry, crs="EPSG:4326")
    from ceva.products import coverage, write_products, yearly_products

    products = yearly_products(cfg, transects, waterlines, radar)

    composite_thresholds = composites[0].groupby("year")["threshold"].first()
    run_info = {
        "name": name, "variant": variant, "ceva_version": __version__,
        "cube_snapshot": ds.attrs.get("snapshot_id"), "cube_providers": ds.attrs.get("providers"),
        "tide_model": cfg.tides.model,
        "config_sha256": hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()[:16],
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "n_acquisitions": int(ds.sizes["time"]), "n_transects": int(ds.sizes["transect"]),
        "perimage_engine": cfg.perimage.engine, "chips_snapshot": chip_snapshots,
        "threshold_annual": cfg.annual.threshold, "threshold_perimage": cfg.perimage.threshold,
        "median_threshold_annual": float(composite_thresholds.median()),
        "median_threshold_perimage": float(waterlines.drop_duplicates("time")["threshold"].median()),
        "slope_method": cfg.perimage.slope_method, "unit_slope": unit_slope, "fixed_slope": fixed_slope,
        "search_reference": "tide_stage", "chips_reflectance": chip_levels,
        "wave_component": cfg.waves.component if waves is not None else "none",
        "wave_point": list(wave_point) if wave_point else None,
        "min_total_water_level_m": cfg.perimage.min_total_water_level_m,
        "radar_chips_snapshot": radar_snapshot,
        "radar_detection": (cfg.sar.detection if cfg.perimage.engine == "coastsat" else "otsu along transects")
        if radar is not None else None,
        "radar_calibration": radar_calibration.attrs if radar_calibration is not None else None,
        "products": coverage(products),
    }
    out = results_dir(cfg, name, variant)
    out.mkdir(parents=True, exist_ok=True)
    if curves is not None:
        curves.to_parquet(out / "tide_stage.parquet", index=False)
    transects.to_parquet(out / "transects.parquet", index=False)
    waterlines.to_parquet(out / "waterlines.parquet", index=False)
    annual.to_parquet(out / "annual.parquet", index=False)
    rates.to_parquet(out / "rates.parquet", index=False)
    write_products(out, products)
    if detections is not None:
        detections.to_parquet(out / "detections.parquet", index=False)
    for stale in ("radar_waterlines.parquet", "radar_calibration.parquet", "radar_templates.parquet"):
        (out / stale).unlink(missing_ok=True)
    if radar is not None:
        radar.to_parquet(out / "radar_waterlines.parquet", index=False)
        radar_calibration.to_parquet(out / "radar_calibration.parquet", index=False)
        if radar_templates is not None:
            radar_templates.to_parquet(out / "radar_templates.parquet", index=False)
    (out / "run.json").write_text(json.dumps(run_info, indent=2))
    return Results(name, transects, waterlines, annual, rates, run_info, radar, radar_calibration)


def load(cfg: Config, name: str, variant: str = DEFAULT_VARIANT) -> Results:
    out = results_dir(cfg, name, variant)
    if not (out / "run.json").exists():
        raise FileNotFoundError(f"No results for {name!r} ({variant}) in {out}. Run the pipeline first.")
    radar = out / "radar_waterlines.parquet"
    return Results(
        name, gpd.read_parquet(out / "transects.parquet"), pd.read_parquet(out / "waterlines.parquet"),
        pd.read_parquet(out / "annual.parquet"), pd.read_parquet(out / "rates.parquet"),
        json.loads((out / "run.json").read_text()),
        pd.read_parquet(radar) if radar.exists() else None,
        pd.read_parquet(out / "radar_calibration.parquet") if radar.exists() else None,
    )


def variant_config(cfg: Config, variant: str) -> Config:
    """The configuration for a named variant."""
    from ceva.config import deep_merge

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
    merged = deep_merge(cfg.model_dump(mode="json"), VARIANTS[variant])
    return Config.model_validate(merged)
