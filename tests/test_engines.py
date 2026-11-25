"""Engines on a synthetic beach whose shoreline, slope and tide are known exactly."""
import numpy as np
import pandas as pd
import pytest

from ceva.config import load_config
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
)

DISTANCE = np.arange(0.0, 1000.0, 10.0)
SLOPE = 0.04
RATE = -1.5            # m/yr, erosion
X0 = 520.0             # mean-sea-level shoreline in 2000


def synthetic(n_transects: int = 3, years=range(2000, 2012), per_year: int = 24, seed: int = 0) -> Profiles:
    rng = np.random.default_rng(seed)
    times = pd.DatetimeIndex([pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=15 * i + 3)
                              for y in years for i in range(per_year)])
    tide = rng.uniform(-0.5, 0.5, size=(len(times), n_transects)).astype("float32")
    msl = X0 + RATE * (times.year.to_numpy() - 2000)
    waterline = msl[:, None] - tide / SLOPE                         # high tide = further landward
    index = ((DISTANCE[None, None, :] - waterline[:, :, None]) * 0.01).astype("float32")
    family = np.where(np.arange(len(times)) % 3 == 0, "sentinel2", "landsat").astype(object)
    return Profiles(index=index, distance=DISTANCE, time=times, family=family, platform=family,
                    transect_id=np.array([f"t{i}" for i in range(n_transects)], dtype=object),
                    seaward_sign=np.ones(n_transects, dtype="int8"), tide=tide)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_annual_composites_recover_position_and_rate(cfg):
    annual = annual_shorelines(synthetic(), cfg)
    truth = X0 + RATE * (annual["year"] - 2000)
    # Only tides in the central half of the range are used: |tide| <= 0.25 m, at most 6 m of excursion,
    # and the median of about 12 such images per year is closer still.
    assert (annual["x"] - truth).abs().max() < 5.0
    assert (annual["x"] - truth).abs().mean() < 2.0
    assert annual["tide_mean"].abs().max() < 0.2
    rates = change_rates(annual)
    assert rates["rate"].to_numpy() == pytest.approx(RATE, abs=0.4)
    # A 95 % interval misses now and then; it must at least come close and not be degenerate.
    assert ((rates["rate_lo"] - 0.2 < RATE) & (RATE < rates["rate_hi"] + 0.2)).all()
    assert ((rates["rate_hi"] - rates["rate_lo"]) > 0.1).all()


def test_sparse_years_borrow_neighbours_and_say_so(cfg):
    p = synthetic()
    drop = (p.time.year == 2005) & (np.arange(len(p.time)) % 24 >= 2)      # leave 2 images in 2005
    p.index[drop] = np.nan
    annual = annual_shorelines(p, cfg)
    row = annual[(annual["year"] == 2005) & (annual["transect_id"] == "t0")].iloc[0]
    assert bool(row["gapfilled"]) and row["n_obs"] >= cfg.annual.min_observations
    assert not annual[annual["year"] == 2008]["gapfilled"].any()


def test_sensor_subset_only_reports_its_own_years(cfg):
    p = synthetic()
    subset = (p.family == "sentinel2") & (p.time.year >= 2006)
    annual = annual_shorelines(p, cfg, subset, "sentinel2")
    assert annual["year"].min() == 2006 and set(annual["sensors"]) == {"sentinel2"}


def test_perimage_waterlines_slope_and_tide_correction(cfg):
    p = synthetic()
    annual = annual_shorelines(p, cfg)
    w = perimage_waterlines(p, annual, cfg)
    truth_msl = X0 + RATE * (w["time"].dt.year - 2000)
    assert (w["x_obs"] - (truth_msl - w["tide"] / SLOPE)).abs().max() < 0.1

    slopes = estimate_slopes(w, cfg.perimage.slope_bounds)
    assert slopes["slope"].to_numpy() == pytest.approx(SLOPE, rel=0.02)

    corrected = tide_correct(w, slopes.set_index("transect_id")["slope"], 0.0)
    assert (corrected - truth_msl).abs().max() < 1.5
    assert (tide_correct(w, SLOPE, 0.0) - truth_msl).abs().max() < 0.1

    yearly = annual_from_waterlines(w.assign(x_corr=corrected, outlier=False), "x_corr")
    assert change_rates(yearly)["rate"].to_numpy() == pytest.approx(RATE, abs=0.05)


def test_slope_is_not_reported_when_the_waterline_ignores_the_tide(cfg):
    p = synthetic()
    w = perimage_waterlines(p, annual_shorelines(p, cfg), cfg)
    w["x_obs"] = 500.0 + np.random.default_rng(1).normal(0, 5, len(w))
    assert estimate_slopes(w, cfg.perimage.slope_bounds)["slope"].isna().all()


def test_outliers_are_flagged_not_removed():
    time = pd.date_range("2020-01-01", periods=60, freq="5D")
    x = 400.0 + np.random.default_rng(2).normal(0, 3, 60)
    x[[10, 40]] += 80.0
    w = pd.DataFrame({"transect_id": "t0", "time": time, "x": x})
    flags = flag_outliers(w, "x", k=3.0)
    assert flags.sum() == 2 and flags.iloc[[10, 40]].all() and len(flags) == len(w)


def test_load_profiles_flips_reversed_transects_and_drops_undetermined(cfg):
    import xarray as xr

    n_time, k = 30, len(DISTANCE)
    land_to_sea = (DISTANCE - 400.0) * 0.01
    green = np.where(land_to_sea > 0, 0.08, 0.10)
    swir = np.where(land_to_sea > 0, 0.01, 0.30)
    stack = lambda a: np.broadcast_to(np.stack([a, a[::-1], np.full(k, a[0])]), (n_time, 3, k)).copy()  # noqa: E731
    dims = ("time", "transect", "distance")
    ds = xr.Dataset(
        {"green": (dims, stack(green)), "swir1": (dims, stack(swir)),
         "qa": (dims, np.full((n_time, 3, k), 64, dtype="uint8"))},
        coords={"time": pd.date_range("2020-01-01", periods=n_time, freq="10D"), "distance": DISTANCE,
                "family": ("time", np.array(["landsat"] * n_time, dtype=object)),
                "platform": ("time", np.array(["landsat-8"] * n_time, dtype=object)),
                "transect_id": ("transect", np.array(["a", "b", "c"], dtype=object)),
                "length": ("transect", np.full(3, DISTANCE[-1]))},
    )
    p = load_profiles(ds, cfg, np.zeros((n_time, 3), dtype="float32"))
    assert p.seaward_sign.tolist() == [1, -1, 0]
    assert np.nanmax(np.abs(p.index[:, 0] - p.index[:, 1])) < 1e-6      # same profile once flipped
    assert np.isnan(p.index[:, 2]).all()
