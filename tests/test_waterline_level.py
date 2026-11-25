"""The surveyed elevation of satellite waterlines and the empirical waterline model."""
import numpy as np
import pandas as pd
import pytest

from ceva.config import load_config
from ceva.validation.waterline_level import fit, waterline_elevation
from ceva.waves import empirical_level, tide_range


def planar_ladder(slope=0.05, x0=500.0, times=("2020-01-01", "2020-01-15")):
    """Surveys of a planar beach: the contour at level z sits at x0 - z / slope."""
    rows = []
    for t in pd.to_datetime(list(times)):
        for z in np.round(np.arange(-2.0, 3.01, 0.25), 2):
            rows.append({"transect_id": "a", "time": t, "level": float(z), "x": x0 - z / slope})
    return pd.DataFrame(rows)


def test_waterline_elevation_reads_the_surveyed_profile():
    ladder = planar_ladder()
    w = pd.DataFrame({"transect_id": ["a", "a", "a", "b"],
                      "time": pd.to_datetime(["2020-01-10", "2020-01-11", "2020-06-01", "2020-01-10"]),
                      "x_obs": [480.0, 530.0, 500.0, 500.0]})
    z = waterline_elevation(w, ladder)
    assert np.allclose(z[:2], [1.0, -1.5])          # (500 - 480) * 0.05, (500 - 530) * 0.05
    assert np.isnan(z[2]) and np.isnan(z[3])        # no survey within days; no survey of that transect


def test_waterline_elevation_ignores_positions_beyond_the_surveyed_profile():
    w = pd.DataFrame({"transect_id": ["a"], "time": pd.to_datetime(["2020-01-10"]), "x_obs": [700.0]})
    assert np.isnan(waterline_elevation(w, planar_ladder())[0])


def test_height_needed_puts_the_waterline_on_the_surveyed_shoreline(monkeypatch, tmp_path):
    from ceva import pipeline
    from ceva.validation import waterline_level as wl

    cfg = load_config(overrides={"data_dir": str(tmp_path)})
    times = pd.to_datetime(["2020-01-05", "2020-01-08"])
    waterlines = pd.DataFrame({"transect_id": "a", "time": times, "x_obs": [480.0, 520.0], "tide": [0.5, -0.5],
                               "use": True, "platform": "landsat-8", "family": "landsat"})
    transects = pd.DataFrame({"transect_id": ["a"], "slope_used": [0.05]})
    run = {"wave_point": [0.0, 0.0]}
    monkeypatch.setattr(pipeline, "load", lambda *a, **k: pipeline.Results("x", transects, waterlines, None, None, run))
    surveys = planar_ladder().query("level == 0").drop(columns="level")             # shoreline at x = 500
    monkeypatch.setattr(wl, "survey_ladder", lambda *a, **k: planar_ladder())
    import ceva.validation.report as report
    from ceva.waves import cache_path

    module = type("FakeSite", (), {"surveys": staticmethod(lambda c, z: surveys)})
    monkeypatch.setitem(report.SITE_INFO, "fake", {"module": module})
    waves = pd.DataFrame({"time": pd.date_range("2019-12-01", "2020-02-01", freq="h"), "hs": 1.0, "tp": 8.0})
    path = cache_path(cfg, 0.0, 0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    waves.to_parquet(path)
    t = wl.elevation_table(cfg, "fake")
    # 0.05 * (500 - 480) - 0.5 = 0.5 ; 0.05 * (500 - 520) + 0.5 = -0.5
    assert np.allclose(t["dz"], [0.5, -0.5])
    assert np.allclose(t["z_wl"], [1.0, -1.0])         # the surveyed elevation of each waterline
    fixed = wl.predicted_position(t, t["dz"].to_numpy(), 0.0)
    assert np.allclose(fixed, 500.0)


def test_fit_weighs_sites_equally():
    rng = np.random.default_rng(1)
    frames = []
    for site, n, tr in (("big", 5000, 1.5), ("small", 50, 4.0), ("mid", 400, 2.0)):
        hs = rng.uniform(0.5, 3.0, n)
        frames.append(pd.DataFrame({"site": site, "intercept": 1.0, "hs": hs, "tide_range": tr,
                                    "dz": -0.9 + 0.5 * hs + 0.2 * tr + rng.normal(0, 0.05, n)}))
    coef = fit(pd.concat(frames, ignore_index=True), ("intercept", "hs", "tide_range"), within_site=False)
    assert np.allclose(coef, [-0.9, 0.5, 0.2], atol=0.05)


def test_within_site_fit_ignores_the_sites_mean_offsets():
    rng = np.random.default_rng(2)
    frames = []
    for site, offset in (("a", 1.0), ("b", -0.5), ("c", 0.2)):
        anomaly = rng.normal(0, 0.5, 300)
        frames.append(pd.DataFrame({"site": site, "hs_anomaly": anomaly + 0.3,      # matched images: not centred
                                    "dz": offset + 0.8 * anomaly + rng.normal(0, 0.05, 300)}))
    assert np.allclose(fit(pd.concat(frames, ignore_index=True)), [0.8], atol=0.03)


def test_empirical_level_and_tide_range():
    tide = pd.Series([-1.0, 1.0, -1.0, 1.0, -0.5, 0.5])
    tid = pd.Series(["a", "a", "a", "a", "b", "b"])
    tr = tide_range(tide, tid)
    assert np.allclose(tr[tid == "a"], 4 * np.std([-1, 1, -1, 1], ddof=1))
    assert np.allclose(empirical_level([1.0, 3.0], [2.0, 2.0], 0.75), [-0.75, 0.75])


def test_empirical_model_needs_a_coefficient_for_the_engine():
    from ceva.config import Config

    data = load_config().model_dump(mode="json")
    data["waves"]["component"] = "empirical"
    data["perimage"]["engine"] = "coastsat"
    del data["waves"]["empirical"]["hs_coef"]["coastsat"]
    with pytest.raises(ValueError, match="no coefficient for the 'coastsat' engine"):
        Config.model_validate(data)


def test_correction_with_the_empirical_model():
    from ceva.pipeline import correct_waterlines

    cfg = load_config(overrides={"waves": {"component": "empirical"}})
    times = pd.date_range("2020-01-01", periods=60, freq="7D")
    tide = np.tile([-1.0, 1.0], 30)
    w = pd.DataFrame({"transect_id": "a", "time": times, "platform": "landsat-8", "family": "landsat",
                      "tide": tide, "x_obs": 500.0 - tide / 0.05, "threshold": 0.0})
    hourly = pd.date_range("2019-12-01", "2021-03-01", freq="h")
    hs = np.where(hourly.isin(times[::2]), 3.0, 1.0)                 # stormy at every other image
    waves = pd.DataFrame({"time": hourly, "hs": hs, "tp": 10.0})
    out, slopes, _ = correct_waterlines(cfg, w, waves, fixed_slope=0.05)
    coef = cfg.waves.empirical.hs_coef[cfg.perimage.engine]
    expected = coef * np.tile([1.0, -1.0], 30)                         # Hs 3 or 1 around a mean of 2
    assert np.allclose(out["wave"], expected, atol=1e-3)
    # moved to mean sea level: x_obs + (tide + wave) / slope
    assert np.allclose(out["x_corr"], 500.0 + expected / 0.05, atol=0.05)
