"""The search window that follows the tide, on a flat macrotidal beach whose waterline is known."""
import numpy as np
import pandas as pd
import pytest

from ceva.config import load_config
from ceva.engines import (
    Profiles,
    _non_increasing,
    annual_shorelines,
    perimage_waterlines,
    reference_positions,
    stage_shift,
    tide_stage_curves,
)
from ceva.engines_coastsat import Transects, _references_for, tide_following_lines

DISTANCE = np.arange(0.0, 1600.0, 10.0)
SLOPE = 0.012          # Perranporth-like: 3 m of tide moves the waterline 250 m
X0 = 800.0


def macrotidal(n_transects: int = 3, n_time: int = 400, seed: int = 1) -> Profiles:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2016-01-03", periods=n_time, freq="5D")
    tide = rng.uniform(-3.0, 3.0, size=(n_time, n_transects)).astype("float32")
    waterline = X0 - tide / SLOPE
    index = np.tanh((DISTANCE[None, None, :] - waterline[:, :, None]) / 30.0).astype("float32") * 0.5
    family = np.array(["sentinel2"] * n_time, dtype=object)
    return Profiles(index=index, distance=DISTANCE, time=times, family=family,
                    platform=np.array(["sentinel-2a"] * n_time, dtype=object),
                    transect_id=np.array([f"t{i}" for i in range(n_transects)], dtype=object),
                    seaward_sign=np.ones(n_transects, dtype="int8"), tide=tide)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_non_increasing_pools_violators():
    out = _non_increasing(np.array([5.0, 6.0, 3.0, 4.0, 1.0]), np.ones(5))
    assert out == pytest.approx([5.5, 5.5, 3.5, 3.5, 1.0])
    assert _non_increasing(np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 1.0])) == pytest.approx([3.0, 2.0, 1.0])


def test_curves_measure_how_far_the_tide_moves_the_waterline(cfg):
    curves = tide_stage_curves(macrotidal(), cfg)
    assert set(curves["transect_id"]) == {"t0", "t1", "t2"}
    for _, g in curves.groupby("transect_id"):
        assert len(g) == cfg.perimage.tide_stage_bins
        assert np.all(np.diff(g["x"]) <= 0)                              # landward as the tide rises
        gradient = np.polyfit(g["tide"], g["x"], 1)[0]
        assert gradient == pytest.approx(-1 / SLOPE, rel=0.1)
        assert np.interp(0.0, g["tide"], g["x"]) == pytest.approx(X0, abs=15)


def test_stage_shift_is_linear_beyond_the_curve_and_zero_without_one():
    curves = pd.DataFrame({"transect_id": ["a"] * 3, "tide": [-1.0, 0.0, 1.0], "x": [100.0, 50.0, 0.0]})
    shift = stage_shift(curves, np.array(["a", "b"]), np.array([[2.0, 2.0], [-2.0, 1.0], [0.5, np.nan]]), 0.0)
    assert shift[:, 0] == pytest.approx([-100.0, 100.0, -25.0])
    assert (shift[:, 1] == 0).all()


def test_the_window_follows_the_tide_where_a_fixed_one_misses(cfg):
    p = macrotidal()
    annual = annual_shorelines(p, cfg)
    truth = X0 - p.tide / SLOPE
    curves = tide_stage_curves(p, cfg)
    fixed = reference_positions(p, annual)
    following = reference_positions(p, annual, curves)
    assert np.abs(following - truth).max() < 40.0
    assert np.abs(fixed - truth).max() > 200.0

    def share_found(curves_or_none) -> float:
        """Share of all image-transect pairs whose waterline is found within 15 m of the truth."""
        w = perimage_waterlines(p, annual, cfg, curves_or_none)
        expected = X0 - w["tide"].to_numpy() / SLOPE
        return float((np.abs(w["x_obs"].to_numpy() - expected) < 15.0).sum() / truth.size)

    assert share_found(curves) > 0.95
    assert share_found(None) < 0.8


def test_coastsat_reference_line_per_image_moves_with_the_tide():
    t = Transects(np.array(["a", "b", "c"], dtype=object),
                  origin=np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]]),
                  end=np.array([[1000.0, 0.0], [1000.0, 100.0], [1000.0, 200.0]]))
    annual = pd.DataFrame({"transect_id": ["a", "b", "c"], "year": 2020, "x": 500.0, "tide_mean": 0.0})
    curves = pd.DataFrame({"transect_id": np.repeat(["a", "b", "c"], 2), "tide": [-1.0, 1.0] * 3,
                           "x": [550.0, 450.0] * 3})
    times = pd.DatetimeIndex(["2020-03-01 11:00", "2020-03-06 11:00"])
    lines = tide_following_lines(t, annual, curves, times, np.array([[1.0, 1.0, 1.0], [-2.0, -2.0, -2.0]]), 300.0)
    first, second = lines[times[0]], lines[times[1]]
    assert first[1:-1, 0] == pytest.approx([450.0] * 3)                # 1 m above the composite: 50 m landward
    assert second[1:-1, 0] == pytest.approx([600.0] * 3)               # beyond the curve, on its straight line
    assert len(first) == 5                                             # extended at both ends
    assert set(_references_for(lines, times[:1].to_numpy())) == {times[0]}
    by_year = {2020: first}
    assert _references_for(by_year, times.to_numpy()) is by_year
