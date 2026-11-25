"""Validation report: compare pipeline results with surveys and write one interactive page.

Each site payload gathers everything the page shows as plain lists and numbers; the
page draws its charts in the browser, so every point can be hovered and every chart
zoomed. Several method variants (see ``ceva.pipeline.VARIANTS``) are compared on the
same data cube; the detailed charts show one of them (the current default unless chosen).
"""
from __future__ import annotations

import datetime as dt
import json
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from ceva import __version__, pipeline
from ceva.config import Config
from ceva.engines import change_rates, estimate_slopes, flag_outliers, tide_correct
from ceva.validation import benchmark, jarkus, narrabeen, trucvert
from ceva.validation.compare import error_metrics, match_in_time, metrics_table
from ceva.validation.ukmacro import PERRANPORTH, SLAPTON

SLOPES_TESTED = (0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3)
LEVEL_THRESHOLDS = tuple(float(v) for v in np.round(np.arange(-2.0, 1.01, 0.25), 2))
MIN_SURVEYS_PER_YEAR = 4
PLOTLY_CDN = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/{version}/plotly.min.js"
VARIANT_LABELS = {
    "coastsat": "CoastSat solution: CoastSat's pixel classifier, sand/water threshold and 2D contour on image "
                "chips (Sentinel-2 as top-of-atmosphere reflectance)",
    "ceva": "CEVA solution: MNDWI along each transect, Otsu threshold near the expected shoreline",
}
SHARED_METHOD = ("Both: search window that follows the tide, tide + wave correction fitted to surveys, "
                 "spectral beach slope")
BENCHMARK_LABELS = {"coastsat": "CEVA: CoastSat solution", "ceva": "CEVA: CEVA solution"}
SITE_INFO = {
    "trucvert": {
        "module": trucvert, "title": "Truc Vert, France", "dense": True, "min_rate_years": 8,
        "kind": "Open sandy beach facing the Bay of Biscay. Meso-macrotidal (spring range up to 5 m), "
                "high-energy (winter waves up to 10 m). Topographic surveys every two to four weeks, 2003–2019.",
    },
    "narrabeen": {
        "module": narrabeen, "title": "Narrabeen–Collaroy, Australia", "dense": True, "min_rate_years": 10,
        "kind": "Embayed sandy beach near Sydney. Microtidal (spring range about 1.3 m), moderate to high "
                "waves. Monthly profiles on five lines since 1976; satellite comparison 1984–2019.",
    },
    "jarkus": {
        "module": jarkus, "title": "Noord-Holland, the Netherlands", "dense": False, "min_rate_years": 15,
        "kind": "55 km of sandy coast with groynes, a sea wall and very large sand nourishments. "
                "Microtidal to mesotidal. One surveyed profile per year on each line (JARKUS), 1984–2025.",
    },
    "walcheren": {
        "module": jarkus.WALCHEREN, "title": "Walcheren, the Netherlands", "dense": False, "min_rate_years": 15,
        "kind": "Sandy coast of the former island of Walcheren (Zeeland), facing the Western Scheldt mouth and the "
                "North Sea, with dunes, groynes and nourishments. Mesotidal to macrotidal (spring range about 4 m). "
                "One surveyed profile per year on each line (JARKUS), 1984–2025.",
    },
    "perranporth": {
        "module": PERRANPORTH, "title": "Perranporth, England", "dense": True, "min_rate_years": 8,
        # Used to fit the waterline model only with the search window that follows the tide
        # (perimage.reference: tide_stage): the tidal excursion, about 500 m, exceeds a fixed window, whose
        # heights would describe a detection failure, not where the waterline sits.
        "kind": "Wide sandy beach on the north coast of Cornwall facing the Atlantic. Macrotidal (spring range about "
                "6.3 m), high-energy, with a flat low-tide terrace. Profiles about monthly, 2006–2019.",
    },
    "slapton": {
        "module": SLAPTON, "title": "Slapton Sands, England", "dense": True, "min_rate_years": 8,
        "kind": "Steep fine-gravel barrier in Start Bay facing east-south-east, sheltered from the dominant westerly "
                "waves. Macrotidal (spring range about 4.3 m). Profiles about monthly, 2007–2020.",
    },
}


# ------------------------------------------------------------------------- helpers

def _r(values, digits: int = 1) -> list:
    """Rounded list with NaN as null, ready for JSON."""
    array = np.asarray(values, dtype="float64")
    return [None if not np.isfinite(v) else round(float(v), digits) for v in array]


def _records(frame: pd.DataFrame, digits: int = 2) -> list[dict]:
    out = frame.copy()
    for column in out.select_dtypes("number"):
        out[column] = out[column].astype("float64").round(digits)
    return json.loads(out.to_json(orient="records"))


def _corr(a, b) -> float:
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else float("nan")


def _pair_metrics(pairs: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Metrics of x against x_survey, plus spread after removing each transect's own bias."""
    rows = []
    for key, g in pairs.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key, strict=True)) | error_metrics(g["error"])
        sat = g["x"] - g.groupby("transect_id")["x"].transform("mean")
        ref = g["x_survey"] - g.groupby("transect_id")["x_survey"].transform("mean")
        row["r"] = _corr(sat, ref)
        resid = g["error"] - g.groupby("transect_id")["error"].transform("mean")
        row["std_within"] = float(resid.std(ddof=1)) if len(g) > 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _rate_pairs(annual: pd.DataFrame, survey_annual: pd.DataFrame, min_years: int) -> pd.DataFrame:
    """Satellite and surveyed rates per transect, each fitted on the years both have."""
    rows = []
    for (engine, sensors), g in annual.groupby(["engine", "sensors"], sort=True):
        both = g.merge(survey_annual, on=["transect_id", "year"])
        sat = change_rates(both[["transect_id", "year", "x"]], min_years)
        ref = change_rates(both[["transect_id", "year"]].assign(x=both["x_survey"]), min_years)
        merged = sat.merge(ref, on="transect_id", suffixes=("", "_survey"))
        rows.append(merged.assign(engine=engine, sensors=sensors))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _rate_metrics(rates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (engine, sensors), g in rates.groupby(["engine", "sensors"], sort=True):
        d = (g["rate"] - g["rate_survey"]).astype("float64")
        rows.append({"engine": engine, "sensors": sensors, "n": len(g), "bias": d.mean(),
                     "rmse": float(np.sqrt((d ** 2).mean())), "r": _corr(g["rate"], g["rate_survey"]),
                     "first_year": int(g["first_year"].min()), "last_year": int(g["last_year"].max())})
    return pd.DataFrame(rows)


def _survey_annual(surveys: pd.DataFrame) -> pd.DataFrame:
    if "year" not in surveys:
        surveys = surveys.assign(year=surveys["time"].dt.year)
    return surveys.groupby(["transect_id", "year"])["x"].agg(x_survey="median", n_surveys="size").reset_index()


def _season(month: pd.Series) -> pd.Series:
    return month.map({12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                      6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"})


def _annual_series(annual: pd.DataFrame) -> dict:
    out: dict[str, dict] = {}
    for (tid, engine, sensors), g in annual.sort_values("year").groupby(["transect_id", "engine", "sensors"]):
        out.setdefault(tid, {})[f"{engine}|{sensors}"] = {"year": g["year"].tolist(), "x": _r(g["x"]),
                                                          "n": g["n_obs"].tolist()}
    return out


def _rate_records(rates: pd.DataFrame) -> list[dict]:
    if rates.empty:
        return []
    return _records(rates[["transect_id", "engine", "sensors", "rate", "rate_lo", "rate_hi", "rate_survey",
                           "rate_lo_survey", "rate_hi_survey", "n_years", "first_year", "last_year"]], 3)


# ------------------------------------------------------------------ variant summary

def _variant_summary(cfg: Config, site: str, variant: str, surveys: pd.DataFrame,
                     survey_annual: pd.DataFrame, dense: bool, min_rate_years: int) -> tuple[dict, dict]:
    """Headline metrics of one variant, and the intermediate tables the detailed charts need."""
    res = pipeline.load(cfg, site, variant)
    out = {"variant": variant, "label": VARIANT_LABELS.get(variant, variant), "run": res.run}
    pairs = res.annual.merge(survey_annual, on=["transect_id", "year"])
    pairs["error"] = pairs["x"] - pairs["x_survey"]
    out["yearly"] = _records(_pair_metrics(pairs, ["engine", "sensors"]))
    rates = _rate_pairs(res.annual, survey_annual, min_rate_years)
    out["rates"] = _records(_rate_metrics(rates), 3) if not rates.empty else []
    extra = {"results": res, "pairs": pairs, "rates": rates}
    if dense:
        w = res.waterlines
        used = w[w["use"]]
        observed = match_in_time(used, surveys, "x_obs")
        corrected = match_in_time(used, surveys, "x_corr")
        tables = [
            metrics_table(observed.assign(group="all sensors"), ["group"]).assign(position="as observed"),
            metrics_table(observed.rename(columns={"family": "group"}), ["group"]).assign(position="as observed"),
            metrics_table(corrected.assign(group="all sensors"), ["group"]).assign(position="corrected"),
            metrics_table(corrected.rename(columns={"family": "group"}), ["group"]).assign(position="corrected"),
            metrics_table(corrected.rename(columns={"platform": "group"}), ["group"]).assign(position="corrected"),
        ]
        out["perimage"] = _records(pd.concat(tables, ignore_index=True))
        seasons = corrected.assign(season=_season(corrected["time"].dt.month))
        out["seasons"] = _records(pd.DataFrame(
            [{"season": s} | error_metrics(g["error"]) for s, g in seasons.groupby("season")]))
        out["n_waterlines"], out["n_used"] = int(len(w)), int(w["use"].sum())
        extra |= {"observed": observed, "corrected": corrected}
    radar = _radar_summary(cfg, site, variant, res, surveys, survey_annual, dense)
    if radar is not None:
        out["radar"] = radar
    return out, extra


def _radar_summary(cfg: Config, site: str, variant: str, res: pipeline.Results, surveys: pd.DataFrame,
                   survey_annual: pd.DataFrame, dense: bool) -> dict | None:
    """Radar against the surveys over the radar years: single images (as detected and aligned on the
    imagery), yearly positions and rates (imagery, radar, both), and the learned radar signature."""
    r = res.radar
    if r is None or r.empty:
        return None
    start = r["time"].min()
    first = int(start.year) + (1 if start.month > 1 else 0)      # the first full year of radar
    era = res.annual[(res.annual["engine"] == "perimage") & (res.annual["year"] >= first)]
    out = {"first_year": first, "n_images": int(r["time"].nunique()), "n_waterlines": int(len(r)),
           "used": float(r["use"].mean()), "alignment": res.run.get("radar_calibration") or {},
           "obs_per_year": {s: float(g["n_obs"].median()) for s, g in era.groupby("sensors")}}
    pairs = era.merge(survey_annual, on=["transect_id", "year"])
    pairs["error"] = pairs["x"] - pairs["x_survey"]
    out["yearly"] = _records(_pair_metrics(pairs, ["sensors"]))
    rates = _rate_pairs(era, survey_annual[survey_annual["year"] >= first], 5)
    out["rates"] = _records(_rate_metrics(rates), 3) if not rates.empty else []
    if dense:
        imagery = res.waterlines[res.waterlines["use"] & (res.waterlines["time"].dt.year >= first)]
        rows = [{"source": label} | error_metrics(match_in_time(frame, surveys, column)["error"])
                for label, frame, column in (("imagery", imagery, "x_corr"), ("radar, as detected", r, "x_level"),
                                             ("radar, aligned", r[r["use"]], "x_corr"))]
        aligned = match_in_time(r[r["use"]], surveys, "x_corr")
        rows += [{"source": f"radar, aligned, {p}"} | error_metrics(g["error"])
                 for p, g in aligned.groupby("orbit_state")]
        out["perimage"] = _records(pd.DataFrame(rows))
    path = pipeline.results_dir(cfg, site, variant) / "radar_templates.parquet"
    if path.exists():
        t = pd.read_parquet(path).groupby(["group", "offset_m"])["db"].median().reset_index()
        out["signature"] = {g: {"offset": d["offset_m"].tolist(), "db": _r(d["db"], 2)} for g, d in t.groupby("group")}
    return out


# ----------------------------------------------------------------------- dense sites

def _water_level_curve(w: pd.DataFrame, surveys: pd.DataFrame) -> list[dict]:
    """Per-image error when images below a total water level are dropped (Castelle et al. 2021)."""
    base = w[~w["outlier"]]
    rows = []
    for threshold in (None, *LEVEL_THRESHOLDS):
        kept = base if threshold is None else base[base["level"] >= threshold]
        m = error_metrics(match_in_time(kept, surveys, "x_corr")["error"]) if len(kept) else {"n": 0}
        rows.append({"threshold": threshold, "n_images": int(kept["time"].nunique())} | m)
    return _records(pd.DataFrame(rows), 2)


def _slope_used(w: pd.DataFrame) -> pd.Series:
    """Per-transect slope implied by the stored correction (x_corr - x_obs = level / slope)."""
    shift = w["x_corr"] - w["x_obs"]
    ratio = (w["level"] / shift).where(shift.abs() > 1e-6)
    return ratio.groupby(w["transect_id"]).median()


def _slope_sensitivity(w: pd.DataFrame, surveys: pd.DataFrame, cfg: Config) -> list[dict]:
    """Error when one slope is used everywhere. Wave set-up scales with the slope, so it is rescaled."""
    used = w["transect_id"].map(_slope_used(w))
    rows = []
    for slope in SLOPES_TESTED:
        trial = w.assign(level=w["tide"] + w["wave"] * slope / used)
        trial = trial.assign(x_trial=tide_correct(trial, slope, cfg.tides.reference_elevation_m, level="level"))
        trial["outlier"] = flag_outliers(trial, "x_trial", cfg.perimage.outlier_mad_k)
        m = error_metrics(match_in_time(trial[~trial["outlier"]], surveys, "x_trial")["error"])
        rows.append({"slope": slope, "std": m["std"], "bias": m["bias"], "rmse": m["rmse"], "n": m["n"]})
    return _records(pd.DataFrame(rows), 3)


def dense_payload(cfg: Config, site: str, variants: list[str], detail: str = "default") -> dict:
    info = SITE_INFO[site]
    module = info["module"]
    surveys = module.surveys(cfg, cfg.tides.reference_elevation_m)
    survey_annual = _survey_annual(surveys)
    survey_annual = survey_annual[survey_annual["n_surveys"] >= MIN_SURVEYS_PER_YEAR]

    summaries, extras = {}, {}
    for v in variants:
        summaries[v], extras[v] = _variant_summary(cfg, site, v, surveys, survey_annual, True, info["min_rate_years"])
    d = extras[detail]
    res, corrected, observed = d["results"], d["corrected"], d["observed"]
    w = res.waterlines

    # Benchmark: same surveys, transects and matching for the published algorithms.
    bench, offsets = None, {}
    if site in benchmark.SITES:
        benchmark.download(cfg, site)
        ceva = {BENCHMARK_LABELS.get(v, f"CEVA: {v}"): extras[v]["results"].waterlines for v in variants}
        seaward = benchmark.seaward_directions(res.transects)
        bench = _records(benchmark.score(cfg, site, surveys, ceva, seaward))
        # raw waterlines of each solution against the published CoastSat waterlines of the same images
        for v in ("ceva", "coastsat"):
            if v in extras:
                offsets[v] = benchmark.same_image_offset(cfg, site, extras[v]["results"].waterlines)
        if "coastsat" in extras:
            offsets["coastsat_geolocated"] = benchmark.same_image_offset(
                cfg, site, extras["coastsat"]["results"].waterlines, seaward)

    gradient_tide, intercept_tide = np.polyfit(observed["tide"], observed["error"], 1)
    gradient_level, intercept_level = np.polyfit(observed["level"], observed["error"], 1)
    survey_slope = surveys.groupby("transect_id")["slope"].median() if "slope" in surveys else pd.Series(dtype=float)
    used = w[w["use"]]
    regression = estimate_slopes(used, cfg.perimage.slope_bounds, level="tide",
                                 method="regression").set_index("transect_id")["slope"]
    spectral = res.transects.set_index("transect_id")["slope"]

    series = {}
    for tid, g in w.sort_values("time").groupby("transect_id"):
        s = surveys[surveys["transect_id"] == tid].sort_values("time")
        series[tid] = {
            "sat": {"t": g["time"].dt.strftime("%Y-%m-%dT%H:%M").tolist(), "x": _r(g["x_corr"]),
                    "xo": _r(g["x_obs"]), "tide": _r(g["tide"], 2), "wave": _r(g["wave"], 2),
                    "p": g["platform"].tolist(), "out": (~g["use"]).astype(int).tolist()},
            "survey": {"t": s["time"].dt.strftime("%Y-%m-%d").tolist(), "x": _r(s["x"])},
        }
    edges = np.arange(-2.5, 2.51, 0.25)
    pairs = d["pairs"]
    t = res.transects
    median = lambda s: float(np.nanmedian(s)) if s.notna().any() else None  # noqa: E731
    return {
        "key": site, "title": info["title"], "kind": info["kind"], "dense": True, "detail": detail,
        "attribution": module.ATTRIBUTION, "run": res.run, "variants": [summaries[v] for v in variants],
        "benchmark": bench, "same_image_offset": offsets,
        "transects": _records(pd.DataFrame({
            "id": t["transect_id"], "lon": t["lon"], "lat": t["lat"], "slope": t["slope"],
            "slope_used": t["slope_used"], "slope_source": t["slope_source"], "sign": t["seaward_sign"],
            "slope_regression": t["transect_id"].map(regression), "slope_survey": t["transect_id"].map(survey_slope),
        }), 4),
        "slopes": {"spectral_median": median(spectral), "regression_median": median(regression),
                   "survey_median": median(survey_slope), "benchmark": getattr(module, "BENCHMARK_SLOPE", None)},
        "perimage_metrics": summaries[detail]["perimage"],
        "annual_metrics": _records(_pair_metrics(pairs, ["engine", "sensors"])),
        "annual_pairs": _records(pairs[["transect_id", "year", "engine", "sensors", "x", "x_survey", "error",
                                        "n_obs"]]),
        "rates": _rate_records(d["rates"]),
        "rate_metrics": summaries[detail]["rates"],
        "annual_series": _annual_series(res.annual),
        "survey_annual": {tid: {"year": g["year"].tolist(), "x": _r(g["x_survey"])}
                          for tid, g in survey_annual.sort_values("year").groupby("transect_id")},
        "matched": {"tid": corrected["transect_id"].tolist(), "t": corrected["time"].dt.strftime("%Y-%m-%d").tolist(),
                    "p": corrected["platform"].tolist(), "f": corrected["family"].tolist(),
                    "x": _r(corrected["x_corr"]), "xo": _r(corrected["x_obs"]), "xs": _r(corrected["x_survey"]),
                    "tide": _r(corrected["tide"], 2), "level": _r(corrected["level"], 2),
                    "gap": _r(corrected["gap_days"], 1)},
        "tide_response": {"gradient": round(float(gradient_tide), 2), "intercept": round(float(intercept_tide), 2),
                          "gradient_level": round(float(gradient_level), 2),
                          "intercept_level": round(float(intercept_level), 2),
                          "apparent_slope": round(float(-1.0 / gradient_level), 4) if gradient_level < 0 else None},
        "slope_sensitivity": _slope_sensitivity(w, surveys, cfg),
        "water_level_curve": _water_level_curve(w, surveys),
        "tide_sampling": {"edges": _r(edges, 2), **{
            family: np.histogram(w.loc[w["family"] == family].drop_duplicates("time")["tide"], edges)[0].tolist()
            for family in ("landsat", "sentinel2")}},
        "series": series,
        "n_outliers": int((~w["use"]).sum()), "n_waterlines": int(len(w)),
        "survey_range": [str(surveys["time"].min().date()), str(surveys["time"].max().date())],
    }


# ---------------------------------------------------------------------------- JARKUS

def _survey_centred(cfg: Config, variant_cfg: Config, res: pipeline.Results, surveys: pd.DataFrame, profiles,
                    half_window_days: int = 90, perimage_days: int = 45) -> pd.DataFrame:
    """Satellite positions around each dated survey, for both engines.

    Per image: median of the corrected positions within ``perimage_days`` (at least 3).
    Composite: tide-filtered composite of the images within ``half_window_days``, with
    that year's composite threshold.
    """
    from ceva.engines import window_composites

    dated = surveys[surveys["dated"]].rename(columns={"x": "x_survey"})
    w = res.waterlines[res.waterlines["use"]]
    by_transect = {tid: g for tid, g in w.groupby("transect_id")}
    rows = []
    for s in dated.itertuples():
        g = by_transect.get(s.transect_id)
        if g is None:
            continue
        near = g[(g["time"] - s.time).abs() <= pd.Timedelta(days=perimage_days)]
        if len(near) >= 3:
            rows.append({"engine": "perimage", "sensors": "all", "transect_id": s.transect_id, "year": s.year,
                         "x": float(near["x_corr"].median()), "x_survey": s.x_survey, "n_obs": len(near)})
    composites = res.annual[(res.annual["engine"] == "annual") & (res.annual["sensors"] == "all")]
    thresholds = composites.groupby("year")["threshold"].first()
    windows = dated.assign(start=dated["time"] - pd.Timedelta(days=half_window_days),
                           end=dated["time"] + pd.Timedelta(days=half_window_days),
                           threshold=dated["year"].map(thresholds).fillna(cfg.annual.index_threshold))
    comp = window_composites(profiles, variant_cfg, windows)
    comp = comp[np.isfinite(comp["x"])]
    rows += [{"engine": "annual", "sensors": "all", "transect_id": r.transect_id, "year": r.year, "x": r.x,
              "x_survey": r.x_survey, "n_obs": r.n_obs} for r in comp.itertuples()]
    out = pd.DataFrame(rows)
    out["error"] = out["x"] - out["x_survey"]
    return out


def _variant_config(cfg: Config, site: str, variant: str) -> Config:
    """The configuration of a variant; an archived run (a copied results folder) uses the variant it ran as."""
    if variant not in pipeline.VARIANTS:
        run = json.loads((pipeline.results_dir(cfg, site, variant) / "run.json").read_text())
        variant = run.get("variant", pipeline.DEFAULT_VARIANT)
    return pipeline.variant_config(cfg, variant)


def annual_payload(cfg: Config, site: str, variants: list[str], detail: str = "default") -> dict:
    from ceva.cube import open_cube
    from ceva.engines import load_profiles
    from ceva.tides import tide_heights

    info = SITE_INFO[site]
    module = info["module"]
    surveys = module.surveys(cfg, cfg.tides.reference_elevation_m)
    survey_annual = surveys.rename(columns={"x": "x_survey"})[["transect_id", "year", "x_survey"]]

    ds = open_cube(cfg, site).load()
    tide = tide_heights(cfg, (ds["lon_start"].values + ds["lon_end"].values) / 2.0,
                        (ds["lat_start"].values + ds["lat_end"].values) / 2.0, pd.DatetimeIndex(ds["time"].values))

    summaries, extras = {}, {}
    for v in variants:
        summaries[v], extras[v] = _variant_summary(cfg, site, v, surveys, survey_annual, False,
                                                   info["min_rate_years"])
        variant_cfg = _variant_config(cfg, site, v)
        profiles = load_profiles(ds, variant_cfg, tide)
        centred = _survey_centred(cfg, variant_cfg, extras[v]["results"], surveys, profiles)
        summaries[v]["survey_centred"] = _records(_pair_metrics(centred, ["engine", "sensors"]))
    d = extras[detail]
    res, pairs = d["results"], d["pairs"]
    t = res.transects
    return {
        "key": site, "title": info["title"], "kind": info["kind"], "dense": False, "detail": detail,
        "attribution": module.ATTRIBUTION, "run": res.run, "variants": [summaries[v] for v in variants],
        "dated_fraction": round(float(surveys["dated"].mean()), 3),
        "transects": _records(pd.DataFrame({
            "id": t["transect_id"], "lon": t["lon"], "lat": t["lat"], "slope": t["slope"],
            "slope_used": t["slope_used"], "slope_source": t["slope_source"], "sign": t["seaward_sign"]}), 5),
        "slopes": {"spectral_median": float(np.nanmedian(t["slope"])) if t["slope"].notna().any() else None},
        "annual_metrics": _records(_pair_metrics(pairs, ["engine", "sensors"])),
        "annual_pairs": _records(pairs[["transect_id", "year", "engine", "sensors", "x", "x_survey", "error",
                                        "n_obs"]]),
        "rates": _rate_records(d["rates"]),
        "rate_metrics": summaries[detail]["rates"],
        "annual_series": _annual_series(res.annual),
        "survey_annual": {tid: {"year": g["year"].tolist(), "x": _r(g["x_survey"])}
                          for tid, g in survey_annual.sort_values("year").groupby("transect_id")},
        "n_outliers": int((~res.waterlines["use"]).sum()), "n_waterlines": int(len(res.waterlines)),
    }


# ----------------------------------------------------------------------------- page

def waterline_payload(cfg: Config, sites: list[str], variant: str = pipeline.DEFAULT_VARIANT) -> dict:
    """Where the satellite waterline sits above the tide, per site, and the fitted model tested site by site."""
    from ceva.validation import waterline_level as wl

    tables = {s: wl.elevation_table(cfg, s, variant) for s in sites}
    loso = wl.leave_one_site_out(cfg, tables)
    coef = wl.fit(pd.concat(tables.values(), ignore_index=True))
    offsets = wl.site_offsets(tables).set_index("site")

    def binned(t: pd.DataFrame, column: str, edges: np.ndarray, min_n: int = 10) -> dict:
        cut = pd.cut(t[column], edges)
        grouped = t.groupby(cut, observed=True)["dz"]
        g = pd.DataFrame({"median": grouped.median(), "n": grouped.size(),
                          "q25": grouped.quantile(0.25), "q75": grouped.quantile(0.75)})
        g = g[g["n"] >= min_n]
        mids = [float(i.mid) for i in g.index]
        return {"x": mids, "dz": g["median"].round(3).tolist(), "q25": g["q25"].round(3).tolist(),
                "q75": g["q75"].round(3).tolist(), "n": g["n"].astype(int).tolist()}

    out_sites = []
    for s, t in tables.items():
        z = t["z_wl"] - t["tide"]
        out_sites.append({
            "key": s, "title": SITE_INFO[s]["title"], "n": int(len(t)), "dz_mean": round(float(t["dz"].mean()), 3),
            "surveyed_fraction": round(float(t["z_wl"].notna().mean()), 3),
            "surveyed_height_mean": round(float(z.mean()), 3) if z.notna().any() else None,
            "tide_range": round(float(t["tide_range"].median()), 2), "hs_median": round(float(t["hs"].median()), 2),
            "hs_mean": round(float(t["hs"].mean()), 3), "slope": round(float(offsets.at[s, "slope"]), 3),
            "offset_m": round(float(offsets.at[s, "offset_m"]), 1),
            "by_hs": binned(t, "hs", np.arange(0.0, 5.01, 0.25)),
            "by_tide": binned(t, "tide", np.arange(-3.0, 3.01, 0.25)),
        })
    return {"variant": variant, "coef": [round(float(c), 3) for c in coef], "sites": out_sites,
            "loso": _records(loso, 2)}


def site_payload(cfg: Config, site: str, variants: list[str], detail: str = "default") -> dict:
    """Everything the page shows for one site; the detailed charts use the ``detail`` variant."""
    if detail not in variants:
        detail = "default"
    return (dense_payload if SITE_INFO[site]["dense"] else annual_payload)(cfg, site, variants, detail)


def write_report(payloads: list[dict], path: Path, standalone: bool = True, inline_plotly: bool = True,
                 notes_html: str | None = None, waterline: dict | None = None) -> Path:
    """Write the page.

    ``standalone`` adds the document wrapper for opening the file directly.
    ``inline_plotly`` embeds the charting library (about 5 MB) so the page works
    offline and behind networks that block CDNs; otherwise it is loaded from cdnjs.
    ``notes_html`` is an optional reviewed commentary shown at the top of the page.
    """
    template = (resources.files("ceva.validation") / "report_template.html").read_text(encoding="utf-8")
    shown = list(dict.fromkeys(v["variant"] for p in payloads for v in p["variants"]))     # in page order
    data = {"generated": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"), "version": __version__,
            "sites": payloads, "variant_labels": {v: VARIANT_LABELS.get(v, v) for v in shown},
            "shared_method": SHARED_METHOD,
            "notes": notes_html, "waterline": waterline}
    if inline_plotly:
        from plotly.offline import get_plotlyjs

        # The bundle holds one raw U+FFFD inside a regex; the escape is equivalent and keeps the file clean.
        bundle = get_plotlyjs().replace("</script", "<\\/script").replace("�", "\\uFFFD")
        library = "<script>" + bundle + "</script>"
    else:
        from plotly.offline import get_plotlyjs_version

        # the same version as the bundled copy, so the page behaves the same either way
        library = f'<script src="{PLOTLY_CDN.format(version=get_plotlyjs_version())}"></script>'
    body = template.replace('<script src="__PLOTLY__"></script>', library).replace(
        "__DATA__", json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/"))
    if standalone:
        body = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1"></head><body>\n'
                + body + "\n</body></html>\n")
    else:
        # Embedded pages (e.g. a hosted viewer) may not be allowed to save files: no "download plot" button.
        body = body.replace('modeBarButtonsToRemove: ["lasso2d", "select2d"]',
                            'modeBarButtonsToRemove: ["lasso2d", "select2d", "toImage"]')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path
