"""Shoreline change metrics (DSAS-style; Himmelstoss et al. 2018).

Pure functions on per-transect time series of cross-shore distance x(t)
(seaward positive, metres). No I/O, no plotting.
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ChangeResult:
    ref_year: int
    cmp_year: int
    nsm_m: float          # Net Shoreline Movement: x(cmp_year) - x(ref_year)
    epr_m_per_yr: float   # End Point Rate: NSM / (cmp_year - ref_year)
    lrr_m_per_yr: float   # Linear Regression Rate: OLS slope over all years
    lrr_ci95_lo: float
    lrr_ci95_hi: float
    lrr_r2: float
    n_years: int


def nsm_epr(years: np.ndarray, x: np.ndarray, ref_year: int, cmp_year: int) -> tuple[float, float]:
    """NSM and EPR between two specific years (Himmelstoss et al. 2018)."""
    x_ref = x[years == ref_year]
    x_cmp = x[years == cmp_year]
    if len(x_ref) == 0 or len(x_cmp) == 0:
        return float("nan"), float("nan")
    nsm = float(x_cmp[0] - x_ref[0])
    epr = nsm / (cmp_year - ref_year)
    return nsm, epr


def linear_regression_rate(years: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float]:
    """OLS slope (m/yr) with 95% CI (t-distribution) and R-squared.

    Needs at least 3 annual points to fit a meaningful trend + CI.
    """
    if len(years) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    res = stats.linregress(years, x)
    n = len(years)
    t_crit = stats.t.ppf(0.975, df=n - 2)
    ci_halfwidth = t_crit * res.stderr
    return res.slope, res.slope - ci_halfwidth, res.slope + ci_halfwidth, res.rvalue ** 2


def compute_change(years: np.ndarray, x: np.ndarray, ref_year: int, cmp_year: int) -> ChangeResult:
    """Full DSAS-style change summary for one transect's annual series."""
    nsm, epr = nsm_epr(years, x, ref_year, cmp_year)
    slope, ci_lo, ci_hi, r2 = linear_regression_rate(years, x)
    return ChangeResult(
        ref_year=ref_year, cmp_year=cmp_year,
        nsm_m=nsm, epr_m_per_yr=epr,
        lrr_m_per_yr=slope, lrr_ci95_lo=ci_lo, lrr_ci95_hi=ci_hi, lrr_r2=r2,
        n_years=len(years),
    )


def classify_lrr(lrr_m_per_yr: float) -> str:
    """Luijendijk et al. 2018 (Sci. Rep. 8:6641) shoreline-change classes."""
    if np.isnan(lrr_m_per_yr):
        return "insufficient_data"
    av = abs(lrr_m_per_yr)
    if av <= 0.5:
        return "stable"
    label = "accretion" if lrr_m_per_yr > 0 else "erosion"
    if av <= 1.0:
        return label
    tier = "intense" if av <= 3.0 else "severe" if av <= 5.0 else "extreme"
    return f"{tier}_{label}"


def is_significant(nsm_m: float, ci_hi: float, sigma_s_m: float = 10.0, k: float = 2 * 2**0.5) -> bool:
    """Vos et al. 2019 significance test: |NSM| exceeds k*sigma_s AND the
    95% CI upper bound on the trend stays below zero (erosion case)."""
    if np.isnan(nsm_m) or np.isnan(ci_hi):
        return False
    return abs(nsm_m) > k * sigma_s_m and ci_hi < 0
