"""Compare satellite positions with surveyed ones."""
from __future__ import annotations

import numpy as np
import pandas as pd


def match_in_time(satellite: pd.DataFrame, survey: pd.DataFrame, column: str,
                  max_days: float = 10.0, min_days: float = 3.0) -> pd.DataFrame:
    """Surveyed position at each satellite date, as in the benchmark of Vos et al. (2023).

    If surveys exist within ``max_days`` both before and after the image, the surveyed
    position is interpolated linearly between them. Otherwise the nearest survey is used
    if it is within ``min_days``. Other satellite dates are left out.

    Returns the satellite rows that could be matched, with ``x_survey``, ``error``
    (satellite minus survey, positive = satellite further seaward) and ``gap_days``.
    """
    out = []
    for transect_id, sat in satellite.groupby("transect_id", sort=False):
        ref = survey[survey["transect_id"] == transect_id].sort_values("time")
        if len(ref) < 2:
            continue
        t_ref = ref["time"].to_numpy().astype("datetime64[s]").astype("float64") / 86400.0
        x_ref = ref["x"].to_numpy(dtype="float64")
        t_sat = sat["time"].to_numpy().astype("datetime64[s]").astype("float64") / 86400.0

        after = np.searchsorted(t_ref, t_sat)
        before = after - 1
        has_before, has_after = before >= 0, after < len(t_ref)
        gap_before = np.where(has_before, t_sat - t_ref[np.clip(before, 0, None)], np.inf)
        gap_after = np.where(has_after, t_ref[np.clip(after, None, len(t_ref) - 1)] - t_sat, np.inf)

        interpolated = np.interp(t_sat, t_ref, x_ref)
        nearest = np.where(gap_before <= gap_after, x_ref[np.clip(before, 0, None)],
                           x_ref[np.clip(after, None, len(t_ref) - 1)])
        both = (gap_before <= max_days) & (gap_after <= max_days)
        close = np.minimum(gap_before, gap_after) <= min_days
        x_survey = np.where(both, interpolated, np.where(close, nearest, np.nan))

        matched = sat.assign(x_survey=x_survey, gap_days=np.minimum(gap_before, gap_after))
        matched["error"] = matched[column] - matched["x_survey"]
        out.append(matched[np.isfinite(matched["error"])])
    if not out:
        return satellite.iloc[0:0].assign(x_survey=np.nan, gap_days=np.nan, error=np.nan)
    return pd.concat(out, ignore_index=True)


def error_metrics(errors: pd.Series) -> dict[str, float]:
    """Bias (mean), spread (standard deviation), RMSE and median absolute error, in metres."""
    e = errors.to_numpy(dtype="float64")
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"n": 0, "bias": np.nan, "std": np.nan, "rmse": np.nan, "mae": np.nan}
    return {"n": int(e.size), "bias": float(e.mean()), "std": float(e.std(ddof=1)) if e.size > 1 else np.nan,
            "rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.median(np.abs(e)))}


def metrics_table(matched: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Error metrics for each group, plus the correlation between satellite and survey
    after removing each transect's own mean (so it measures change, not location)."""
    rows = []
    for key, g in matched.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key, strict=True)) | error_metrics(g["error"])
        sat = g["x_survey"] + g["error"]
        sat_anomaly = sat - sat.groupby(g["transect_id"]).transform("mean")
        survey_anomaly = g["x_survey"] - g["x_survey"].groupby(g["transect_id"]).transform("mean")
        row["r"] = float(np.corrcoef(sat_anomaly, survey_anomaly)[0, 1]) if len(g) > 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
