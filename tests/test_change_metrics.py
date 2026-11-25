"""Change metrics: net movement, end-point and regression rates, classes, significance."""
import numpy as np
import pytest

from ceva.change_metrics import classify_lrr, compute_change, is_significant, linear_regression_rate


def test_linear_regression_recovers_a_known_rate():
    years = np.arange(2000, 2011)
    x = 100.0 - 2.5 * (years - 2000)
    slope, lo, hi, r2 = linear_regression_rate(years, x)
    assert slope == pytest.approx(-2.5)
    assert lo == pytest.approx(-2.5) and hi == pytest.approx(-2.5)
    assert r2 == pytest.approx(1.0)


def test_fewer_than_three_years_gives_no_trend():
    assert all(np.isnan(v) for v in linear_regression_rate(np.array([2000, 2001]), np.array([1.0, 2.0])))


def test_compute_change_end_points():
    years = np.array([2000, 2005, 2010])
    result = compute_change(years, np.array([50.0, 40.0, 20.0]), 2000, 2010)
    assert result.nsm_m == pytest.approx(-30.0)
    assert result.epr_m_per_yr == pytest.approx(-3.0)


@pytest.mark.parametrize(
    ("rate", "label"),
    [(0.2, "stable"), (-0.7, "erosion"), (0.7, "accretion"), (-2.0, "intense_erosion"),
     (-4.0, "severe_erosion"), (6.0, "extreme_accretion"), (float("nan"), "insufficient_data")],
)
def test_classification(rate, label):
    assert classify_lrr(rate) == label


def test_significance_needs_large_movement_and_a_negative_upper_bound():
    assert is_significant(-40.0, -0.5)
    assert not is_significant(-20.0, -0.5)    # below 2*sqrt(2)*10 m
    assert not is_significant(-40.0, 0.1)     # trend interval reaches zero
