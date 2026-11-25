import datetime as dt

import numpy as np
import pandas as pd
import pytest
from pyproj import Geod

from ceva import waves


def test_offshore_point_lies_beyond_the_seaward_ends():
    geod = Geod(ellps="WGS84")
    lon_s, lat_s = np.array([-1.25, -1.25]), np.array([44.70, 44.72])
    lon_e, lat_e, _ = geod.fwd(lon_s, lat_s, np.array([270.0, 270.0]), np.array([1000.0, 1000.0]))
    lon, lat = waves.offshore_point(lon_s, lat_s, lon_e, lat_e, np.array([1, 1]), 20_000)
    assert lon < lon_e.min() and abs(lat - 44.71) < 0.01
    _, _, dist = geod.inv(lon_e.mean(), lat_e.mean(), lon, lat)
    assert dist == pytest.approx(20_000, rel=0.01)
    # Flipped transects: the seaward end is the start.
    lon2, _ = waves.offshore_point(lon_e, lat_e, lon_s, lat_s, np.array([-1, -1]), 20_000)
    assert lon2 == pytest.approx(lon, abs=1e-6)
    with pytest.raises(ValueError):
        waves.offshore_point(lon_s, lat_s, lon_e, lat_e, np.array([0, 0]), 20_000)


def test_era5_waves_are_cached(cfg, monkeypatch):
    calls = []

    def fake(lon, lat, start, end):
        calls.append((start, end))
        time = pd.date_range(start, end + dt.timedelta(days=1), freq="h", inclusive="left")
        return pd.DataFrame({"time": time, "hs": 1.0, "tp": 9.0, "direction": 260.0})

    monkeypatch.setattr(waves, "_fetch_open_meteo", fake)
    first = waves.era5_waves(cfg, -1.5, 44.75, dt.date(2000, 1, 1), dt.date(2010, 12, 31), chunk_years=4)
    assert len(calls) == 3 and first["time"].is_monotonic_increasing and first["time"].is_unique
    again = waves.era5_waves(cfg, -1.5, 44.75, dt.date(2001, 1, 1), dt.date(2009, 1, 1))
    assert len(calls) == 3 and len(again) == len(first)


def test_waves_for_unit_moves_offshore_from_land_cells(cfg, monkeypatch):
    seen = []

    def fake(cfg_, lon, lat, start, end):
        seen.append(lon)
        hs = np.nan if len(seen) == 1 else 1.0
        return pd.DataFrame({"time": pd.date_range(start, periods=10, freq="h"), "hs": hs, "tp": 9.0,
                             "direction": 0.0})

    monkeypatch.setattr(waves, "era5_waves", fake)
    cfg = cfg.model_copy(update={"waves": cfg.waves.model_copy(update={"source": "open_meteo_era5"})})
    lon_s, lat_s = np.array([-1.25]), np.array([44.7])
    _, point = waves.waves_for_unit(cfg, lon_s, lat_s, np.array([-1.26]), lat_s, np.array([1]),
                                    dt.date(2020, 1, 1), dt.date(2020, 1, 2))
    assert len(seen) == 2 and point[0] < seen[0]
