"""Otsu thresholds, spectral slope, wave-corrected waterlines, survey-centred composites, the two variants."""
import numpy as np
import pandas as pd
import pytest

from ceva.config import load_config
from ceva.engines import (
    Profiles,
    annual_shorelines,
    detect_edge,
    estimate_slopes,
    otsu_near,
    perimage_waterlines,
    window_composites,
)
from ceva.pipeline import VARIANTS, correct_waterlines, variant_config

DISTANCE = np.arange(0.0, 1000.0, 10.0)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def two_population_profile(shoreline_m: float, land=-0.1, water=0.6, ramp_m=60.0):
    """Beach at ``land`` (wet sand reads just below 0), water at ``water``, and a linear
    transition of ``ramp_m`` centred on ``shoreline_m``."""
    fraction = np.clip((DISTANCE - (shoreline_m - ramp_m / 2)) / ramp_m, 0.0, 1.0)
    return (land + (water - land) * fraction).astype("float32")


def test_otsu_sits_between_the_two_populations():
    profiles = np.stack([two_population_profile(500.0 + 10 * i) for i in range(6)])
    threshold = otsu_near(profiles, DISTANCE, np.full(6, 480.0), 150.0, 0.0)
    assert 0.1 < threshold < 0.4           # between the beach (-0.1) and water (0.6) levels, well above 0


def test_otsu_falls_back_when_one_class_is_missing():
    flat = np.full((3, len(DISTANCE)), 0.5, dtype="float32")
    assert otsu_near(flat, DISTANCE, np.full(3, 500.0), 150.0, 0.0) == 0.0
    sparse = two_population_profile(500.0)[None, :]
    assert otsu_near(sparse, DISTANCE, np.array([500.0]), 150.0, 0.0) == 0.0     # too few samples


def test_composite_edge_moves_seaward_of_wet_sand_with_otsu(cfg):
    composite = np.stack([two_population_profile(500.0) for _ in range(6)])
    fixed_cfg = cfg.model_copy(update={"annual": cfg.annual.model_copy(update={"threshold": "fixed"})})
    x_fixed, t_fixed = detect_edge(composite, DISTANCE, fixed_cfg)
    x_otsu, t_otsu = detect_edge(composite, DISTANCE, cfg)
    assert t_fixed == 0.0 and 0.1 < t_otsu < 0.4
    assert np.all(np.abs(x_fixed - 478.6) < 1.0)         # fixed 0: 21 m landward of the transition centre
    assert np.all((x_otsu > x_fixed + 8.0) & (x_otsu < 515.0))   # Otsu: well inside the transition


def synthetic_profiles(slope=0.04, n_years=10, per_year=40, seed=3, wave=None):
    """Waterline = shoreline - level / slope, with level = tide (+ wave)."""
    rng = np.random.default_rng(seed)
    days = np.sort(rng.choice(np.arange(n_years * 365), n_years * per_year, replace=False))
    times = pd.Timestamp("2000-01-01") + pd.to_timedelta(days, unit="D") + pd.Timedelta(hours=10.5)
    # Tide sampled at a fixed time of day: aliased semi-diurnal signal plus a spring-neap cycle.
    t = days.astype(float)
    tide = (1.2 * np.cos(2 * np.pi * t / 14.77) * np.cos(2 * np.pi * t / 0.5175) + 0.3 * rng.normal(size=t.size))
    level = tide + (0 if wave is None else wave)
    x = 500.0 - level / slope + rng.normal(0, 2.0, t.size)
    return pd.DataFrame({"transect_id": "t0", "time": times, "x_obs": x, "tide": tide, "level": level,
                         "family": "landsat", "platform": "landsat-8"})


@pytest.mark.parametrize("method", ["spectral", "regression"])
@pytest.mark.parametrize("slope", [0.03, 0.08])
def test_slope_methods_recover_a_known_slope(method, slope):
    w = synthetic_profiles(slope=slope)
    est = estimate_slopes(w, (0.01, 0.3), level="tide", method=method)
    assert est["slope"].iloc[0] == pytest.approx(slope, rel=0.15)


def test_spectral_slope_is_undetermined_without_a_tide_signal():
    w = synthetic_profiles()
    w["x_obs"] = 500.0 + np.random.default_rng(0).normal(0, 5, len(w))
    est = estimate_slopes(w, (0.01, 0.3), level="tide", method="spectral")
    assert np.isnan(est["slope"].iloc[0])


def test_wave_correction_removes_the_wave_driven_spread_and_recovers_slope(cfg):
    from ceva.waves import empirical_level, hs_at

    rng = np.random.default_rng(1)
    w = synthetic_profiles(slope=0.05)
    hours = pd.date_range(w["time"].min().floor("D") - pd.Timedelta(days=1), w["time"].max().ceil("D"), freq="h")
    hs = 1.0 + 1.5 * (1 + np.sin(2 * np.pi * np.arange(len(hours)) / (24 * 9))) / 2 + rng.uniform(0, 0.2, len(hours))
    table = pd.DataFrame({"time": hours, "hs": hs, "tp": 11.0, "direction": 270.0})
    hs_img = hs_at(table, w["time"])
    wave = empirical_level(hs_img, hs_img.mean(), cfg.waves.empirical.hs_coef[cfg.perimage.engine])
    w = w.drop(columns="level")
    w["x_obs"] = w["x_obs"] - wave / 0.05           # the observed waterline rises with the waves
    out, slopes, unit = correct_waterlines(cfg, w, table)
    assert slopes["slope"].iloc[0] == pytest.approx(0.05, rel=0.15)
    assert np.allclose(out["wave"], wave, atol=1e-5) and (out["level"] - out["tide"] - out["wave"]).abs().max() < 1e-5
    assert (out.loc[out["use"], "x_corr"] - 500.0).abs().median() < 3.0
    tide_only, _, _ = correct_waterlines(cfg, w, None)
    assert out.loc[out["use"], "x_corr"].std() < tide_only.loc[tide_only["use"], "x_corr"].std() - 1.0


def test_low_water_images_are_excluded_when_asked(cfg):
    w = synthetic_profiles()
    strict = variant_config(cfg, "coastsat").model_copy(
        update={"perimage": cfg.perimage.model_copy(update={"min_total_water_level_m": 0.0})})
    out, _, _ = correct_waterlines(strict, w.drop(columns="level"), None)
    assert (out["low_water"] == (out["level"] < 0.0)).all()
    assert not out.loc[out["low_water"], "use"].any()


def make_profiles(n_time=120, shoreline=500.0, wet=True):
    times = pd.date_range("2010-01-01", periods=n_time, freq="9D")
    rng = np.random.default_rng(0)
    tide = rng.uniform(-1, 1, size=(n_time, 4)).astype("float32")
    base = two_population_profile(shoreline) if wet else ((DISTANCE - shoreline) * 0.01).astype("float32")
    index = np.broadcast_to(base, (n_time, 4, len(DISTANCE))).copy()
    return Profiles(index=index, distance=DISTANCE, time=times, family=np.array(["landsat"] * n_time, dtype=object),
                    platform=np.array(["landsat-8"] * n_time, dtype=object),
                    transect_id=np.array([f"t{i}" for i in range(4)], dtype=object),
                    seaward_sign=np.ones(4, dtype="int8"), tide=tide)


def test_annual_and_perimage_thresholds_are_reported(cfg):
    p = make_profiles()
    annual = annual_shorelines(p, cfg)
    assert (annual["threshold"] > 0.1).all() and annual["x"].between(487.0, 515.0).all()
    w = perimage_waterlines(p, annual, cfg)
    assert (w["threshold"] > 0.1).all() and w["x_obs"].between(487.0, 515.0).all()


def test_window_composites(cfg):
    p = make_profiles(wet=False)
    windows = pd.DataFrame({"transect_id": ["t0", "t1", "nope"],
                            "start": pd.to_datetime(["2010-01-01", "2012-06-01", "2010-01-01"]),
                            "end": pd.to_datetime(["2010-06-30", "2012-06-05", "2010-06-30"]),
                            "threshold": [0.0, 0.0, 0.0]})
    out = window_composites(p, cfg, windows)
    assert out["x"].iloc[0] == pytest.approx(500.0, abs=0.5) and out["n_obs"].iloc[0] >= 5
    assert np.isnan(out["x"].iloc[1]) and out["n_obs"].iloc[1] < 5          # too few images in 5 days
    assert np.isnan(out["x"].iloc[2])


def test_the_two_variants_differ_only_by_their_detection(cfg):
    assert set(VARIANTS) == {"coastsat", "ceva"}
    assert variant_config(cfg, "ceva") == cfg               # the production method is the default
    assert cfg.perimage.engine == "ceva" and cfg.waves.component == "empirical"
    coastsat = variant_config(cfg, "coastsat")
    assert coastsat.perimage.engine == "coastsat"
    assert coastsat.model_copy(update={"perimage": coastsat.perimage.model_copy(update={"engine": "ceva"})}) == cfg
    with pytest.raises(ValueError):
        variant_config(cfg, "nope")
