"""Shoreline engines. Both read the same transect cube and write the same kind of table.

``annual``    after DEA Coastlines (Bishop-Taylor et al. 2021): for each year, the
              median water index of the observations taken near mean sea level, then
              the sub-pixel crossing of a land/water threshold. One robust position per
              transect and year.
``perimage``  after CoastSat (Vos et al. 2019): one waterline per image, moved to the
              reference elevation using the total water level (tide, optionally plus
              wave set-up or run-up) and the beach slope. Keeps the seasonal and storm
              signal, at the price of more noise.

The land/water threshold is either fixed or chosen per composite or per image with
Otsu's method on the index values near the shoreline, so that it sits between the
sand and water populations actually present (CoastSat does this on classified sand
and water pixels). Positions are metres along the transect from its landward end,
positive seaward.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from skimage.filters import threshold_otsu

from ceva.config import Config
from ceva.shoreline import clear_sky, mndwi, nearest_crossing, orient_seaward, seaward_edge

CHUNK = 64              # acquisitions processed at once, to bound memory
OTSU_MIN_SAMPLES = 100  # fewer samples than this near the shoreline: keep the fixed threshold
OTSU_MIN_CLASS = 0.1    # each side of the first guess must hold at least this share of the samples


@dataclass
class Profiles:
    index: np.ndarray           # [time, transect, distance] masked MNDWI, land to sea
    distance: np.ndarray        # [distance] metres from the landward end
    time: pd.DatetimeIndex
    family: np.ndarray          # [time]
    platform: np.ndarray        # [time]
    transect_id: np.ndarray     # [transect]
    seaward_sign: np.ndarray    # [transect] +1 as stored, -1 flipped, 0 undetermined (excluded)
    tide: np.ndarray            # [time, transect] metres above mean sea level

    @property
    def year(self) -> np.ndarray:
        return self.time.year.to_numpy()


def load_profiles(ds: xr.Dataset, cfg: Config, tide: np.ndarray) -> Profiles:
    """Masked water index for every acquisition, with each transect oriented land to sea."""
    family = ds["family"].values.astype(object)
    index = mndwi(ds["green"].values, ds["swir1"].values)
    clear = clear_sky(ds["qa"].values, family == "landsat",
                      cfg.masking.landsat_qa_drop_bits, cfg.masking.s2_scl_drop_classes)
    index[~clear] = np.nan

    distance = ds["distance"].values.astype("float64")
    length = ds["length"].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        typical = np.nanmedian(index, axis=0)
    sign = orient_seaward(typical, cfg.annual.index_threshold)
    for i in np.flatnonzero(sign == -1):
        n_valid = int(np.searchsorted(distance, length[i] + 1e-6))
        index[:, i, :n_valid] = index[:, i, :n_valid][:, ::-1]
    index[:, sign == 0, :] = np.nan
    return Profiles(
        index=index, distance=distance, time=pd.DatetimeIndex(ds["time"].values),
        family=family, platform=ds["platform"].values.astype(object),
        transect_id=ds["transect_id"].values.astype(object), seaward_sign=sign, tide=tide,
    )


# -------------------------------------------------------------------------- threshold

def otsu_near(index: np.ndarray, distance: np.ndarray, centre: np.ndarray, window_m: float,
              fallback: float) -> float:
    """Otsu threshold of the index samples within ``window_m`` of ``centre`` (one value per profile).

    ``index`` is [profile, distance]. Falls back to ``fallback`` when there are too few
    samples or they are not split on both sides of it.
    """
    near = np.abs(distance[None, :] - np.asarray(centre)[:, None]) <= window_m
    values = index[near & np.isfinite(index)]
    if values.size < OTSU_MIN_SAMPLES:
        return fallback
    below = (values <= fallback).mean()
    if not OTSU_MIN_CLASS <= below <= 1 - OTSU_MIN_CLASS:
        return fallback
    return float(threshold_otsu(values))


def detect_edge(composite: np.ndarray, distance: np.ndarray, cfg: Config) -> tuple[np.ndarray, float]:
    """Shoreline of composite profiles [transect, distance] and the threshold used."""
    a = cfg.annual
    first = seaward_edge(composite, distance, a.index_threshold, a.min_land_run_m)
    if a.threshold == "fixed" or not np.isfinite(first).any():
        return first, a.index_threshold
    threshold = otsu_near(composite, distance, first, a.otsu_window_m, a.index_threshold)
    return seaward_edge(composite, distance, threshold, a.min_land_run_m), threshold


# ----------------------------------------------------------------------------- annual

def tide_half_window(tide: np.ndarray, observed: np.ndarray, central_fraction: float) -> np.ndarray:
    """Half-width [transect] of the accepted tide band: a fraction of the observed tide range."""
    masked = np.where(observed, tide, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tide_range = np.nanmax(masked, axis=0) - np.nanmin(masked, axis=0)
    return central_fraction * tide_range / 2.0


def eligible_for_composite(p: Profiles, cfg: Config, subset: np.ndarray | None = None) -> np.ndarray:
    """[time, transect] observations usable in a composite: mostly cloud-free and near mean sea level."""
    use = np.ones(len(p.time), dtype=bool) if subset is None else subset
    observed = np.isfinite(p.index).mean(axis=-1) >= 0.5
    half = tide_half_window(p.tide, observed, cfg.annual.tide_central_fraction)       # from all sensors
    near_msl = np.abs(p.tide - cfg.tides.reference_elevation_m) <= half[None, :]
    return observed & near_msl & use[:, None]


def annual_shorelines(p: Profiles, cfg: Config, subset: np.ndarray | None = None, label: str = "all") -> pd.DataFrame:
    """One shoreline per transect and year from tide-filtered median composites.

    ``subset`` selects acquisitions (for example one sensor family). Returns
    transect_id, year, x, n_obs, tide_mean, gapfilled, threshold, sensors.
    """
    a = cfg.annual
    use = np.ones(len(p.time), dtype=bool) if subset is None else subset
    eligible = eligible_for_composite(p, cfg, subset)

    rows = []
    for year in np.unique(p.year[use]):
        in_year = p.year == year
        in_window = np.abs(p.year - year) <= a.gapfill_years
        n_year = (eligible & in_year[:, None]).sum(axis=0)
        composite, n_obs, tide_mean = _composite(p, eligible & in_year[:, None])
        sparse = n_year < a.min_observations
        if sparse.any() and a.gapfill_years > 0:
            wide, n_wide, tide_wide = _composite(p, eligible & in_window[:, None], only=sparse)
            composite[sparse], n_obs[sparse], tide_mean[sparse] = wide[sparse], n_wide[sparse], tide_wide[sparse]
        enough = n_obs >= a.min_observations
        x, threshold = detect_edge(np.where(enough[:, None], composite, np.nan), p.distance, cfg)
        rows.append(pd.DataFrame({
            "transect_id": p.transect_id, "year": int(year), "x": x, "n_obs": n_obs,
            "tide_mean": tide_mean, "gapfilled": sparse, "threshold": threshold, "sensors": label,
        }))
    out = pd.concat(rows, ignore_index=True)
    return out[np.isfinite(out["x"])].reset_index(drop=True)


def window_composites(p: Profiles, cfg: Config, windows: pd.DataFrame) -> pd.DataFrame:
    """Composite shoreline of one transect over an arbitrary time window, for each row of ``windows``.

    ``windows`` has transect_id, start, end (timestamps) and threshold (the index threshold
    to use, for example that year's composite threshold). Returns the windows with x and n_obs.
    """
    eligible = eligible_for_composite(p, cfg)
    position = {tid: i for i, tid in enumerate(p.transect_id)}
    xs, ns = [], []
    for row in windows.itertuples():
        i = position.get(row.transect_id)
        chosen = (eligible[:, i] & (p.time >= row.start) & (p.time <= row.end)) if i is not None else None
        n = int(chosen.sum()) if chosen is not None else 0
        if n < cfg.annual.min_observations:
            xs.append(np.nan)
            ns.append(n)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            profile = np.nanmedian(p.index[chosen, i, :], axis=0)
        xs.append(float(seaward_edge(profile, p.distance, row.threshold, cfg.annual.min_land_run_m)))
        ns.append(n)
    return windows.assign(x=xs, n_obs=ns)


def _composite(p: Profiles, selected: np.ndarray, only: np.ndarray | None = None):
    """Median index profile per transect over the selected [time, transect] observations."""
    n_transects = selected.shape[1]
    composite = np.full((n_transects, len(p.distance)), np.nan, dtype="float32")
    tide_mean = np.full(n_transects, np.nan, dtype="float32")
    n_obs = selected.sum(axis=0)
    times = np.flatnonzero(selected.any(axis=1))
    if times.size:
        todo = np.flatnonzero(n_obs > 0 if only is None else (n_obs > 0) & only)
        stack = np.where(selected[times][:, todo, None], p.index[times][:, todo, :], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN columns are expected
            composite[todo] = np.nanmedian(stack, axis=0)
            tide_mean[todo] = np.nanmean(np.where(selected[times][:, todo], p.tide[times][:, todo], np.nan), axis=0)
    return composite, n_obs, tide_mean


# ------------------------------------------------------------ tide-following search window

def _non_increasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least-squares fit of ``y`` that never increases (pool adjacent violators)."""
    blocks: list[list[float]] = []            # [weighted mean, weight, count]
    for value, weight in zip(y, w, strict=True):
        blocks.append([float(value), float(weight), 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([np.full(n, v) for v, _, n in blocks])


def tide_stage_curves(p: Profiles, cfg: Config) -> pd.DataFrame:
    """Where each transect's waterline sits at each tide level, from composites binned by tide.

    The observations of all years on a transect are split into ``tide_stage_bins`` groups of
    equal size by their tide height, and each group's median index profile gives one
    shoreline (as for the annual composites). A rising tide cannot move the waterline
    seaward, so the positions are made non-increasing with the tide. The curve describes
    the shape of the beach, which changes far less than its position, so one curve over the
    whole record is enough. Returns transect_id, tide (median of the group), x, n_obs.
    """
    c = cfg.perimage
    n_bins = c.tide_stage_bins
    observed = np.isfinite(p.index).mean(axis=-1) >= 0.5
    tide = np.where(observed, p.tide, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        edges = np.nanquantile(tide, np.linspace(0.0, 1.0, n_bins + 1), axis=0)     # [bin edge, transect]
    xs, tides, counts = [], [], []
    for k in range(n_bins):
        upper = (tide <= edges[k + 1]) if k == n_bins - 1 else (tide < edges[k + 1])
        selected = observed & (tide >= edges[k]) & upper
        composite, n_obs, tide_mean = _composite(p, selected)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tide_median = np.nanmedian(np.where(selected, p.tide, np.nan), axis=0)
        enough = n_obs >= c.tide_stage_min_obs
        x, _ = detect_edge(np.where(enough[:, None], composite, np.nan), p.distance, cfg)
        xs.append(np.where(enough, x, np.nan))
        tides.append(tide_median)
        counts.append(n_obs)
    xs, tides, counts = np.array(xs), np.array(tides), np.array(counts)          # [bin, transect]
    rows = []
    for i, transect_id in enumerate(p.transect_id):
        ok = np.isfinite(xs[:, i]) & np.isfinite(tides[:, i])
        if ok.sum() < 2:
            continue
        order = np.argsort(tides[ok, i])
        z, n = tides[ok, i][order], counts[ok, i][order]
        x = _non_increasing(xs[ok, i][order], n.astype(float))
        rows.append(pd.DataFrame({"transect_id": transect_id, "tide": z, "x": x, "n_obs": n}))
    columns = ["transect_id", "tide", "x", "n_obs"]
    return pd.concat(rows, ignore_index=True)[columns] if rows else pd.DataFrame(columns=columns)


def stage_shift(curves: pd.DataFrame, transect_id: np.ndarray, tide: np.ndarray, tide_ref: np.ndarray) -> np.ndarray:
    """Expected move of the waterline [time, transect] between ``tide_ref`` and ``tide``.

    Read off each transect's tide-stage curve, linear in between its points and, beyond
    them, along the straight line through its two ends (a planar beach). Zero for
    transects without a curve and where either tide is unknown.
    """
    tide = np.atleast_2d(np.asarray(tide, dtype=float))
    tide_ref = np.broadcast_to(np.asarray(tide_ref, dtype=float), tide.shape)
    out = np.zeros(tide.shape)
    if curves.empty:
        return out
    groups = dict(tuple(curves.groupby("transect_id", sort=False)))
    for i, tid in enumerate(transect_id):
        g = groups.get(tid)
        if g is None or len(g) < 2:
            continue
        z, x = g["tide"].to_numpy(float), g["x"].to_numpy(float)
        gradient = (x[-1] - x[0]) / (z[-1] - z[0]) if z[-1] > z[0] else 0.0

        def at(level, z=z, x=x, gradient=gradient):
            inside = np.interp(level, z, x)
            return np.where(level < z[0], x[0] + (level - z[0]) * gradient,
                            np.where(level > z[-1], x[-1] + (level - z[-1]) * gradient, inside))

        shift = at(tide[:, i]) - at(tide_ref[:, i])
        out[:, i] = np.where(np.isfinite(shift), shift, 0.0)
    return out


# --------------------------------------------------------------------------- per image

def reference_positions(p: Profiles, annual: pd.DataFrame, curves: pd.DataFrame | None = None) -> np.ndarray:
    """Search centre [time, transect]: that year's annual shoreline, else the nearest year's.

    With tide-stage ``curves``, the centre also follows the tide: it moves from the annual
    shoreline, which sits at the composite's mean tide, by as much as the curve says the
    waterline moves between that tide and the image's.
    """
    years = np.arange(p.year.min(), p.year.max() + 1)

    def per_year(column: str) -> np.ndarray:
        table = (annual.pivot(index="transect_id", columns="year", values=column)
                 .reindex(index=p.transect_id, columns=years))
        return table.T.interpolate(limit_direction="both").T.to_numpy()[:, p.year - years[0]].T

    reference = per_year("x")
    if curves is not None and not curves.empty:
        reference = reference + stage_shift(curves, p.transect_id, p.tide, per_year("tide_mean"))
    return reference


def perimage_waterlines(p: Profiles, annual: pd.DataFrame, cfg: Config,
                        curves: pd.DataFrame | None = None) -> pd.DataFrame:
    """Observed waterline of every acquisition on every transect, with its tide height."""
    c = cfg.perimage
    reference = reference_positions(p, annual, curves)
    n_time = len(p.time)
    x_obs = np.full((n_time, len(p.transect_id)), np.nan)
    threshold = np.full(n_time, c.index_threshold, dtype="float32")
    if c.threshold == "otsu":
        for t in range(n_time):
            threshold[t] = otsu_near(p.index[t], p.distance, reference[t], c.max_dist_ref_m, c.index_threshold)
    for lo in range(0, n_time, CHUNK):
        hi = min(lo + CHUNK, n_time)
        x_obs[lo:hi] = nearest_crossing(p.index[lo:hi], p.distance, threshold[lo:hi, None],
                                        reference[lo:hi], c.max_dist_ref_m, c.min_valid_fraction)
    t_idx, n_idx = np.nonzero(np.isfinite(x_obs))
    return pd.DataFrame({
        "transect_id": p.transect_id[n_idx], "time": p.time[t_idx], "platform": p.platform[t_idx],
        "family": p.family[t_idx], "x_obs": x_obs[t_idx, n_idx], "tide": p.tide[t_idx, n_idx],
        "threshold": threshold[t_idx],
    })


# ----------------------------------------------------------------------- beach slope

def estimate_slopes(waterlines: pd.DataFrame, bounds: tuple[float, float], min_obs: int = 30,
                    level: str = "tide", method: str = "regression") -> pd.DataFrame:
    """Beach-face slope tan(beta) per transect, from how the waterline follows the water level.

    ``level`` is the water-level column (tide, or tide plus waves). ``method`` is
    ``regression`` or ``spectral`` (see the two functions below). Returns transect_id,
    slope (NaN when not determined) and n_slope_obs.
    """
    estimate = _slope_regression if method == "regression" else _slope_spectral
    rows = []
    for transect_id, g in waterlines.groupby("transect_id", sort=False):
        slope, n = estimate(g, level, bounds, min_obs)
        rows.append({"transect_id": transect_id, "slope": slope, "n_slope_obs": n})
    return pd.DataFrame(rows, columns=["transect_id", "slope", "n_slope_obs"])


def _slope_regression(g: pd.DataFrame, level: str, bounds, min_obs: int) -> tuple[float, int]:
    """A higher water level puts the waterline further landward by level / tan(beta).

    Positions and levels are first centred within each year, so that long-term change does
    not leak into the estimate; outliers beyond 3 scaled MADs are dropped before the final fit.
    """
    year = g["time"].dt.year
    dx = (g["x_obs"] - g.groupby(year)["x_obs"].transform("mean")).to_numpy()
    dz = (g[level] - g.groupby(year)[level].transform("mean")).to_numpy()
    n = len(g)
    if n < min_obs or np.ptp(dz) == 0:
        return np.nan, n
    gradient = (dx * dz).sum() / (dz ** 2).sum()
    residual = dx - gradient * dz
    mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    keep = np.abs(residual - np.median(residual)) <= 3 * mad if mad > 0 else np.ones(n, dtype=bool)
    if keep.sum() < min_obs:
        return np.nan, n
    gradient = (dx[keep] * dz[keep]).sum() / (dz[keep] ** 2).sum()
    if gradient < 0 and bounds[0] <= -1.0 / gradient <= bounds[1]:
        return -1.0 / gradient, int(keep.sum())
    return np.nan, int(keep.sum())


# Cycles per day: sub-seasonal frequencies, up to the Nyquist frequency of 8-day sampling.
SPECTRAL_FREQ = np.linspace(1 / 120.0, 1 / 16.0, 240)
SPECTRAL_SLOPES = 80


class _Periodogram:
    """``scipy.signal.lombscargle`` (default options) of many series sampled at the same times.

    Its sines and cosines depend only on the times and the frequencies: they are computed once,
    and each series then costs two dot products instead of four trigonometric arrays. Every
    step is scipy's own, in the same order, so each periodogram is the same to the last bit.
    """

    def __init__(self, t: np.ndarray, omega: np.ndarray):
        x = np.asarray(t, dtype=np.float64)
        weights = np.ones_like(x, dtype=np.float64)
        self.weights = (weights * (1.0 / weights.sum())).reshape(-1, 1)
        freqst = np.asarray(omega, dtype=np.float64).reshape(1, -1) * x.reshape(-1, 1)
        coswt, sinwt = np.cos(freqst), np.sin(freqst)
        cc = np.dot(self.weights.T, coswt * coswt)
        ss = 1.0 - cc
        cs = np.dot(self.weights.T, coswt * sinwt)
        tau = 0.5 * np.arctan2(2.0 * cs, cc - ss)
        freqst_tau = freqst - tau
        self.cos, self.sin = np.cos(freqst_tau), np.sin(freqst_tau)
        cc = np.dot(self.weights.T, self.cos * self.cos)
        ss = 1.0 - cc
        epsneg = np.finfo(np.float64).epsneg
        cc[cc < epsneg] = epsneg
        ss[ss < epsneg] = epsneg
        self.cc, self.ss, self.n = cc, ss, x.shape[0]

    def __call__(self, y: np.ndarray) -> np.ndarray:
        weights_y = self.weights * np.asarray(y, dtype=np.float64).reshape(-1, 1)
        yc = np.dot(weights_y.T, self.cos)
        ys = np.dot(weights_y.T, self.sin)
        a = yc / self.cc
        b = ys / self.ss
        pgram = np.squeeze(2.0 * (a * yc + b * ys))
        pgram *= float(self.n) / 4.0
        return pgram


def _slope_spectral(g: pd.DataFrame, level: str, bounds, min_obs: int) -> tuple[float, int]:
    """After Vos et al. (2020): the slope that removes the tidal signal from the shoreline series.

    Satellites sample the tide at a fixed time of day, so the tide seen in the images is
    aliased into a few sub-seasonal frequencies. A wrong slope leaves energy at exactly
    those frequencies in the corrected series. For each candidate slope the corrected
    series' Lomb-Scargle spectrum is weighted by the spectrum of the water level itself,
    and the slope with the least weighted energy is kept. Only images from 1999 onwards
    are used when there are enough, for denser sampling (Landsat 7 and later).
    """
    recent = g[g["time"] >= "1999-01-01"]
    use = recent if len(recent) >= min_obs else g
    n = len(use)
    if n < min_obs:
        return np.nan, n
    t = use["time"].to_numpy("datetime64[s]").astype("float64") / 86400.0
    x = np.array(use["x_obs"], dtype="float64")       # copies: modified in place below
    z = np.array(use[level], dtype="float64")
    year = use["time"].dt.year.to_numpy()
    # Remove each year's mean so trends and inter-annual change do not dominate.
    for y in np.unique(year):
        sel = year == y
        x[sel] -= x[sel].mean()
        z[sel] -= z[sel].mean()
    periodogram = _Periodogram(t, 2 * np.pi * SPECTRAL_FREQ)
    weight = periodogram(z)
    if not np.isfinite(weight).all() or weight.sum() <= 0:
        return np.nan, n
    weight = weight / weight.sum()
    slopes = np.geomspace(bounds[0], bounds[1], SPECTRAL_SLOPES)
    energy = np.array([(weight * periodogram(x + z / s)).sum() for s in slopes])
    best = int(np.argmin(energy))
    if best in (0, len(slopes) - 1):       # minimum on the edge of the search range: not determined
        return np.nan, n
    return float(slopes[best]), n


# ------------------------------------------------------------------- correction etc.

def tide_correct(waterlines: pd.DataFrame, slopes: pd.Series | float, reference_elevation_m: float,
                 level: str = "tide") -> pd.Series:
    """Waterline moved to the reference elevation: x + (level - reference) / tan(beta)."""
    tan_beta = waterlines["transect_id"].map(slopes) if isinstance(slopes, pd.Series) else slopes
    return waterlines["x_obs"] + (waterlines[level] - reference_elevation_m) / tan_beta


def flag_outliers(waterlines: pd.DataFrame, column: str, k: float, window: int = 11,
                  floor_m: float = 10.0) -> pd.Series:
    """True where a position departs from its transect's running median by more than
    ``k`` scaled MADs (and at least ``floor_m`` metres)."""
    ordered = waterlines.sort_values(["transect_id", "time"])
    running = ordered.groupby("transect_id")[column].transform(
        lambda s: s.rolling(window, center=True, min_periods=3).median())
    residual = (ordered[column] - running).abs()
    mad = 1.4826 * residual.groupby(ordered["transect_id"]).transform("median")
    return (residual > np.maximum(k * mad, floor_m)).reindex(waterlines.index)


def annual_from_waterlines(waterlines: pd.DataFrame, column: str, min_obs: int = 3,
                           label: str = "all") -> pd.DataFrame:
    """Yearly median of per-image positions (rows marked ``use``, else non-outliers)."""
    if "use" in waterlines:
        good = waterlines[waterlines["use"]]
    else:
        good = waterlines[~waterlines["outlier"]] if "outlier" in waterlines else waterlines
    grouped = good.groupby(["transect_id", good["time"].dt.year.rename("year")])[column]
    out = grouped.agg(x="median", n_obs="size").reset_index()
    out["sensors"] = label
    return out[out["n_obs"] >= min_obs].reset_index(drop=True)


# ------------------------------------------------------------------------------ rates

def change_rates(annual: pd.DataFrame, min_years: int = 5) -> pd.DataFrame:
    """Linear rate of change (m/yr, positive seaward) per transect, with its 95 % interval."""
    from ceva.change_metrics import linear_regression_rate

    rows = []
    for transect_id, g in annual.groupby("transect_id", sort=False):
        if len(g) < min_years:
            continue
        rate, lo, hi, r2 = linear_regression_rate(g["year"].to_numpy(dtype=float), g["x"].to_numpy(dtype=float))
        rows.append({"transect_id": transect_id, "rate": rate, "rate_lo": lo, "rate_hi": hi, "r2": r2,
                     "n_years": len(g), "first_year": int(g["year"].min()), "last_year": int(g["year"].max())})
    return pd.DataFrame(rows, columns=["transect_id", "rate", "rate_lo", "rate_hi", "r2", "n_years",
                                       "first_year", "last_year"])
