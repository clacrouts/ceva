"""Radar: chips, the CEVA solution's template matching, alignment on the imagery, combined yearly positions."""
import datetime as dt

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import LineString

import ceva.sar as sar
from ceva.config import load_config
from ceva.sar_waterlines import align_on_imagery, combined_yearly, match_templates

EPSG = 32630
X0, Y0 = 600_000.0, 4_960_000.0


@pytest.fixture(scope="module")
def config():
    return load_config()


def _raster(path, values):
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1], count=1,
                       dtype="float32", crs=f"EPSG:{EPSG}", transform=from_origin(X0, Y0, 10.0, 10.0),
                       nodata=-32768) as dst:
        dst.write(values.astype("float32"), 1)
    return str(path)


def test_backscatter_to_decibels():
    db = sar.to_db(np.array([1.0, 0.01, 0.0, np.nan]))
    assert db[0] == pytest.approx(0.0) and db[1] == pytest.approx(-20.0)
    assert db[2] == pytest.approx(-50.0) and np.isnan(db[3])


def test_radar_chips_round_trip_in_blocks(cfg, tmp_path, monkeypatch):
    pytest.importorskip("icechunk")
    vv = _raster(tmp_path / "vv.tif", np.full((200, 300), 0.05))       # -13 dB
    vh = _raster(tmp_path / "vh.tif", np.full((200, 300), 0.01))       # -20 dB
    start = dt.datetime(2020, 1, 1, 6, tzinfo=dt.UTC)
    scenes = [sar.SarScene(f"S1A_{'X' * (5 + 40 * (i % 2))}_{i:03d}_rtc", "sentinel1", "sentinel-1a",
                           start + dt.timedelta(days=6 * i), "descending" if i % 2 else "ascending", 8 + i % 2,
                           f"dt{i}", {"vv": vv, "vh": vh}, -32768.0) for i in range(20)]
    monkeypatch.setattr(sar, "search_sentinel1", lambda *a, **k: scenes)
    monkeypatch.setattr(sar, "resolve_href", lambda provider, href: href)
    to_ll = Transformer.from_crs(EPSG, 4326, always_xy=True)
    (lon0, lon1), (lat0, lat1) = to_ll.transform([X0 + 1000, X0 + 1000], [Y0 - 500, Y0 - 1500])
    transects = gpd.GeoDataFrame({"transect_id": ["a"]}, geometry=[LineString([(lon0, lat0), (lon1, lat1)])],
                                 crs="EPSG:4326")
    cfg = cfg.model_copy(update={"sar": cfg.sar.model_copy(update={"block_bytes": 1, "workers": 2})})
    sar.save_sar_chips(cfg, "unit", sar.iter_sar_chips(cfg, transects), "radar")      # blocks of 8, appended
    chips = sar.open_sar_chips(cfg, "unit")
    ds = chips["sentinel1/s000"]
    assert ds.sizes["time"] == 20 and set(ds["orbit_state"].values) == {"ascending", "descending"}
    assert np.allclose(ds["vv"].values, -13.01, atol=0.02) and np.allclose(ds["vh"].values, -20.0, atol=0.02)


def _beach(distance, x, rng, rough=1.0):
    """Radar profile: dunes, a dark wet band landward of the waterline, a bright surf zone, then the sea."""
    p = np.full(distance.shape, -15.0)
    p[(distance > x - 60) & (distance <= x)] = -21.0
    p[(distance > x) & (distance <= x + 120)] = -21.0 + 4.0 * rough
    p[distance > x + 120] = -21.5
    return p + rng.normal(0, 1.2, distance.shape)


def test_templates_learned_from_the_imagery_find_the_waterline(config):
    rng = np.random.default_rng(2)
    distance = np.arange(0.0, 1200.0, 10.0)
    n_time = 300
    truth = 600.0 + 40.0 * np.sin(np.arange(n_time) / 15.0)
    db = np.stack([_beach(distance, x, rng) for x in truth])[:, None, :]
    reference = (truth + rng.uniform(-60, 60, n_time))[:, None]     # where the search is centred
    anchor = np.where(np.arange(n_time) % 2 == 0, truth, np.nan)[:, None]   # imagery on half the dates
    groups = np.array(["descending|0"] * n_time)
    x_obs, score, templates = match_templates(db, distance, reference, anchor, groups, config.sar.template)
    err = x_obs[:, 0] - truth
    assert len(templates) == 1 and np.nanmedian(score) > 0.6
    assert np.nanmedian(np.abs(err)) < 6.0
    assert np.nanmedian(np.abs(err[1::2])) < 6.0          # also where no imagery was near


def test_alignment_removes_offsets_per_orbit_direction_and_wave_dependence(config):
    rng = np.random.default_rng(0)
    times = pd.date_range("2018-01-01", periods=400, freq="2D")
    imagery = pd.DataFrame({"transect_id": "t0", "time": times[::3], "x_corr": 500 + rng.normal(0, 3, len(times[::3])),
                            "use": True})
    passes = np.where(np.arange(len(times)) % 2 == 0, "ascending", "descending")
    hs = 1.5 + rng.uniform(-1, 1, len(times))
    offset = np.where(passes == "ascending", 30.0, -20.0)
    tide = rng.uniform(-1, 1, len(times))
    slope = 0.05
    x_level = 500 + offset + 8.0 * (hs - hs.mean()) + rng.normal(0, 4, len(times))
    radar = pd.DataFrame({"transect_id": "t0", "time": times, "orbit_state": passes, "tide": tide,
                          "x_obs": x_level - tide / slope})
    out, calibration = align_on_imagery(config, radar, imagery, pd.Series({"t0": slope}), hs)
    assert (out.loc[out["use"], "x_corr"] - 500).abs().median() < 4.0
    by_pass = calibration.set_index("orbit_state")["offset"]
    assert by_pass["ascending"] == pytest.approx(30.0, abs=3) and by_pass["descending"] == pytest.approx(-20.0, abs=3)
    assert calibration.attrs["terms"]["hs"] == pytest.approx(8.0, abs=1.5)


def test_combined_yearly_weights_each_family_by_its_precision(config):
    rng = np.random.default_rng(1)

    def family(n, x, sigma):
        t = pd.Timestamp("2019-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 364, n)), unit="D")
        return pd.DataFrame({"transect_id": "t0", "time": t, "x_corr": x + rng.normal(0, sigma, n), "use": True})

    out = combined_yearly(family(40, 500.0, 5.0), family(150, 520.0, 40.0), config)
    assert len(out) == 1 and out["n_obs"].iloc[0] == 190
    assert 500.0 < out["x"].iloc[0] < 504.0            # mostly the imagery: about 16 times its weight


def test_per_year_offset_removes_a_radar_drift_between_years(config):
    rng = np.random.default_rng(3)
    times = pd.date_range("2019-01-01", "2022-12-31", freq="3D")
    imagery = pd.DataFrame({"transect_id": "t0", "time": times[::4], "x_corr": 500.0, "use": True})
    drift = np.where(times.year == 2022, -12.0, 0.0)          # one satellite fewer that year
    radar = pd.DataFrame({"transect_id": "t0", "time": times, "orbit_state": "descending", "tide": 0.0,
                          "x_obs": 500.0 + drift + rng.normal(0, 3, len(times))})
    plain = config.sar.calibration
    for per_year, expected in ((False, -12.0), (True, 0.0)):
        cfg = config.model_copy(update={"sar": config.sar.model_copy(update={
            "calibration": plain.model_copy(update={"per_year": per_year, "terms": []})})})
        out, calibration = align_on_imagery(cfg, radar, imagery, pd.Series({"t0": 0.05}), None)
        in_2022 = out[out["time"].dt.year == 2022]
        assert (in_2022["x_corr"] - 500.0).median() == pytest.approx(expected, abs=4.0)
    assert calibration.attrs["yearly_offsets"][2022] == pytest.approx(-9.0, abs=4.0)
