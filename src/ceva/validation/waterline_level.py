"""Where the satellite waterline sits above the tide, measured against surveys.

For every satellite waterline matched in time to a survey, ``dz`` is the height above the
tide that the water-level correction must use to put the waterline exactly on the
surveyed shoreline, with the beach slope the pipeline uses:

    dz = reference elevation + slope * (surveyed position - observed position) - tide

Every matched waterline counts, wherever it falls on the profile. Within a site, ``dz``
grows with the wave height by a similar amount everywhere; that part is what
``waves.component = "empirical"`` corrects. Its mean differs between beaches (``site_offsets``
lists it with tide range, waves and slope), which a beach-type model will need more sites to fit.

The surveyed beach elevation at the waterline (``z_wl``, from contours extracted at a
ladder of levels) is kept as a physical check. It is missing where the waterline lies
beyond the surveyed part of the profile, which is common on surveys that stop near the
low-tide line, so it is not used for fitting.

``fit`` estimates the model with every site weighted equally, and ``leave_one_site_out``
measures how well it predicts a site it was not fitted on, in metres of shoreline position.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ceva import pipeline
from ceva.config import Config
from ceva.validation.compare import match_in_time

log = logging.getLogger(__name__)

LEVELS = tuple(float(v) for v in np.round(np.arange(-2.5, 3.51, 0.25), 2))
# The model used by waves.component = "empirical": dz = hs_coef * (Hs - mean Hs of the transect)
FEATURES = ("hs_anomaly",)


def survey_ladder(cfg: Config, site: str, levels: tuple[float, ...] = LEVELS) -> pd.DataFrame:
    """Surveyed position of each contour in ``levels``: transect_id, time, level, x. Cached per site."""
    from ceva.validation.report import SITE_INFO

    path = cfg.data_dir / "raw" / "insitu" / "ladders" / f"{site}.parquet"
    if path.exists():
        cached = pd.read_parquet(path)
        if set(levels) <= set(cached["level"].round(2)):
            return cached[cached["level"].round(2).isin(levels)].reset_index(drop=True)
    module = SITE_INFO[site]["module"]
    frames = []
    for z in levels:
        s = module.surveys(cfg, z)
        frames.append(s[["transect_id", "time", "x"]].assign(level=z))
    ladder = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    ladder.to_parquet(path, index=False)
    return ladder


def waterline_elevation(waterlines: pd.DataFrame, ladder: pd.DataFrame) -> np.ndarray:
    """Surveyed elevation at each waterline's ``x_obs``, from surveys matched in time (NaN otherwise)."""
    w = waterlines[["transect_id", "time", "x_obs"]].reset_index(drop=True).assign(key=lambda d: np.arange(len(d)))
    levels = np.array(sorted(ladder["level"].unique()))
    xs = np.full((len(w), len(levels)), np.nan)
    for j, z in enumerate(levels):
        m = match_in_time(w, ladder[ladder["level"] == z], "x_obs")
        xs[m["key"].to_numpy(), j] = m["x_survey"].to_numpy()
    out = np.full(len(w), np.nan)
    x_obs = w["x_obs"].to_numpy()
    for i in range(len(w)):
        ok = np.isfinite(xs[i])
        if ok.sum() < 4:
            continue
        order = np.argsort(xs[i][ok])
        x, z = xs[i][ok][order], levels[ok][order]
        keep = np.r_[True, np.diff(z) < 0]          # the part of the profile that falls seaward
        x, z = x[keep], z[keep]
        if len(x) >= 3 and x[0] <= x_obs[i] <= x[-1]:
            out[i] = np.interp(x_obs[i], x, z)
    return out


def _unit_waves(cfg: Config, res) -> pd.DataFrame:
    """ERA5 waves of the unit: the run's wave point, or found from the transects for a run without waves."""
    from ceva.waves import cache_path, waves_for_unit

    if res.run.get("wave_point"):
        lon, lat = res.run["wave_point"]
        return pd.read_parquet(cache_path(cfg, lon, lat))
    t = res.transects
    ends = np.array([[g.coords[0][0], g.coords[0][1], g.coords[-1][0], g.coords[-1][1]] for g in t.geometry])
    times = pd.DatetimeIndex(res.waterlines["time"])
    start = (times.min() - pd.Timedelta(days=2)).date()
    end = min((times.max() + pd.Timedelta(days=2)).date(), (pd.Timestamp.today() - pd.Timedelta(days=7)).date())
    sign = t["seaward_sign"].to_numpy()
    waves, _ = waves_for_unit(cfg, ends[:, 0], ends[:, 1], ends[:, 2], ends[:, 3], sign, start, end)
    return waves


def elevation_table(cfg: Config, site: str, variant: str = pipeline.DEFAULT_VARIANT,
                    surveyed_elevation: bool = True) -> pd.DataFrame:
    """Waterlines of one site matched to a survey, with ``dz``, the model features and ``z_wl``.

    The detections are those of ``variant``; slopes and outliers are worked out again from the
    tide alone, so that what is measured does not depend on the wave coefficient being fitted.
    """
    from ceva.validation.report import SITE_INFO
    from ceva.waves import hs_at, tide_range

    res = pipeline.load(cfg, site, variant)
    tide_only = cfg.model_copy(update={"waves": cfg.waves.model_copy(update={"component": "none"})})
    raw = res.waterlines.drop(columns=["wave", "level", "x_corr", "outlier", "low_water", "use"], errors="ignore")
    w, slopes, _ = pipeline.correct_waterlines(tide_only, raw, None)
    w = w[w["use"]].reset_index(drop=True)
    waves = _unit_waves(cfg, res)
    w["hs"] = hs_at(waves, w["time"])
    w["tp"] = hs_at(waves.assign(hs=waves["tp"]), w["time"])     # same interpolation, peak period
    # over every image of the transect, not only those matched to a survey (as the pipeline does)
    w["hs_anomaly"] = w["hs"] - w.groupby("transect_id")["hs"].transform("mean")
    w["tide_range"] = tide_range(w["tide"], w["transect_id"])
    surveys = SITE_INFO[site]["module"].surveys(cfg, cfg.tides.reference_elevation_m)
    w = match_in_time(w, surveys, "x_obs").reset_index(drop=True)    # adds x_survey
    w["slope_used"] = w["transect_id"].map(slopes.set_index("transect_id")["slope_used"])
    w["dz"] = cfg.tides.reference_elevation_m + w["slope_used"] * (w["x_survey"] - w["x_obs"]) - w["tide"]
    if surveyed_elevation:
        w["z_wl"] = waterline_elevation(w, survey_ladder(cfg, site))
    w["intercept"] = 1.0
    w["site"] = site
    return w[w["dz"].notna() & w["hs"].notna()].reset_index(drop=True)


def fit(table: pd.DataFrame, features: tuple[str, ...] = FEATURES, within_site: bool = True) -> np.ndarray:
    """Least-squares coefficients of ``dz`` on ``features``, every site weighing the same.

    ``within_site`` removes each site's means first, so that the coefficients describe how
    ``dz`` varies inside a site and the sites' different mean offsets do not leak into them.
    """
    y = table["dz"]
    x = table[list(features)]
    if within_site:
        y = y - y.groupby(table["site"]).transform("mean")
        x = x - x.groupby(table["site"]).transform("mean")
    weight = np.sqrt(1.0 / table.groupby("site")["dz"].transform("size").to_numpy())
    coef, *_ = np.linalg.lstsq(x.to_numpy() * weight[:, None], y.to_numpy() * weight, rcond=None)
    return coef


def site_offsets(tables: dict[str, pd.DataFrame], features: tuple[str, ...] = FEATURES) -> pd.DataFrame:
    """Per site: the mean height still needed after the fitted model, with the beach's tide range,
    waves and slope. This is what a beach-type model would have to explain."""
    every = pd.concat(tables.values(), ignore_index=True)
    coef = fit(every, features)
    rows = []
    for site, t in tables.items():
        remaining = t["dz"] - t[list(features)].to_numpy() @ coef
        rows.append({"site": site, "n": int(len(t)), "dz_mean": float(t["dz"].mean()),
                     "dz_remaining": float(remaining.mean()),
                     # as a shoreline error: satellite minus survey, negative = satellite landward
                     "offset_m": float((-remaining / t["slope_used"]).mean()),
                     "tide_range_m": float(t["tide_range"].median()), "hs_mean_m": float(t["hs"].mean()),
                     "slope": float(t["slope_used"].median())})
    return pd.DataFrame(rows)


def predicted_position(table: pd.DataFrame, dz: np.ndarray, reference_elevation_m: float) -> np.ndarray:
    """Waterline moved to the reference elevation with tide + ``dz`` and the slope the pipeline used."""
    return (table["x_obs"] + (table["tide"] + dz - reference_elevation_m) / table["slope_used"]).to_numpy()


def leave_one_site_out(cfg: Config, tables: dict[str, pd.DataFrame],
                       features: tuple[str, ...] = FEATURES) -> pd.DataFrame:
    """For each site: fitted on the others, scored on it; plus the tide-only correction for reference.

    Errors are shoreline positions (satellite minus survey, metres) of the images matched to a survey.
    """
    rows = []
    every = pd.concat(tables.values(), ignore_index=True)
    for held, test in tables.items():
        coef = fit(every[every["site"] != held], features)
        for label, dz in (("tide only", np.zeros(len(test))), ("model", test[list(features)].to_numpy() @ coef)):
            e = predicted_position(test, dz, cfg.tides.reference_elevation_m) - test["x_survey"].to_numpy()
            rows.append({"site": held, "correction": label, "n": int(e.size), "bias": float(e.mean()),
                         "std": float(e.std(ddof=1)), "rmse": float(np.sqrt((e ** 2).mean())),
                         "dz_observed": float(test["dz"].mean()), "dz_predicted": float(dz.mean()),
                         **{f"coef_{f}": float(c) for f, c in zip(features, coef, strict=True)}})
    return pd.DataFrame(rows)


DESCRIPTORS = {
    "tide_range_m": "tide range (m)",
    "hs_mean_m": "mean wave height (m)",
    "slope": "beach slope used",
    "iribarren": "Iribarren number (slope / sqrt(Hs / L0))",
}


def offset_models(tables: dict[str, pd.DataFrame], features: tuple[str, ...] = FEATURES) -> pd.DataFrame:
    """Can a beach's mean offset be predicted from what is known everywhere? Tested site by site.

    For each descriptor, the mean height still needed after the wave correction is regressed on
    it across the other sites (intercept + one coefficient), then predicted for the held-out
    site. ``error_m`` is the shoreline error left at that site, against ``error_none_m`` with no
    offset model; a useful descriptor makes it smaller at most sites.
    """
    offsets = site_offsets(tables, features).set_index("site")
    l0 = {}
    for site, t in tables.items():
        tp = t["tp"].mean() if "tp" in t else 10.0
        l0[site] = 9.81 * tp ** 2 / (2 * np.pi)
    offsets["iribarren"] = offsets["slope"] / np.sqrt(offsets["hs_mean_m"] / pd.Series(l0))
    rows = []
    for name in DESCRIPTORS:
        for held in offsets.index:
            train = offsets.drop(index=held)
            if len(train) < 3:
                continue
            b, a = np.polyfit(train[name], train["dz_remaining"], 1)
            predicted = a + b * offsets.at[held, name]
            slope = offsets.at[held, "slope"]
            rows.append({"descriptor": name, "site": held, "value": float(offsets.at[held, name]),
                         "dz_remaining": float(offsets.at[held, "dz_remaining"]), "dz_predicted": float(predicted),
                         "error_none_m": float(-offsets.at[held, "dz_remaining"] / slope),
                         "error_m": float(-(offsets.at[held, "dz_remaining"] - predicted) / slope),
                         "intercept": float(a), "coef": float(b)})
    return pd.DataFrame(rows)
