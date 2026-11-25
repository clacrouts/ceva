"""The Perranporth / Slapton Sands loader, on a small file with the layout of the published MATLAB files."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ceva.validation.ukmacro import Site, _datenum

h5py = pytest.importorskip("h5py")


def datenum(day: dt.date) -> float:
    return float(day.toordinal() + 366)


def write_site(path, group="TST", n_surveys=35, slope=0.05, shore_east=500_100.0):
    """One profile line running east (seaward), a planar beach whose 0 m contour is at ``shore_east``."""
    with h5py.File(path, "w") as h:
        refs = h.create_group("#refs#")
        prof = h.create_group(f"{group}/profData/P01")
        rng = np.random.default_rng(0)
        cols = {k: [] for k in ("dnum", "lon", "lat", "elev")}
        for i in range(n_surveys):
            east = np.arange(500_000.0, 500_250.0, 5.0)
            north = 5_600_000.0 + rng.normal(0, 0.5, east.size)
            elev = (shore_east - east) * slope + (0.2 if i % 2 else 0.0)        # every other survey 4 m seaward
            values = {"dnum": np.array([[datenum(dt.date(2010, 1, 1) + dt.timedelta(days=30 * i))]]),
                      "lon": east[None, :], "lat": north[None, :], "elev": elev[None, :]}
            for k, v in values.items():
                d = refs.create_dataset(f"{k}{i}", data=v)
                cols[k].append(d.ref)
        for k, r in cols.items():
            prof.create_dataset(k, data=np.array(r, dtype=h5py.ref_dtype)[None, :])


def test_datenum():
    assert _datenum(datenum(dt.date(2015, 3, 2))) == pd.Timestamp("2015-03-02 12:00")
    assert _datenum(datenum(dt.date(2015, 3, 2)) + 0.25) == pd.Timestamp("2015-03-02 06:00")


def test_transect_and_contours_from_a_synthetic_file(tmp_path, monkeypatch):
    from ceva.config import load_config

    cfg = load_config(overrides={"data_dir": str(tmp_path)})
    site = Site("test", "TSTdata.mat", "TST", "ts", 20.0)
    path = site.raw_path(cfg)
    path.parent.mkdir(parents=True)
    write_site(path)
    tr = site.transects(cfg)
    assert list(tr["transect_id"]) == ["ts-P01"]
    line = tr.to_crs(32630).geometry.iloc[0]
    assert line.coords[0][0] < line.coords[-1][0]                   # from land (west) to sea (east)
    s = site.surveys(cfg, 0.0)
    assert len(s) == 35 and s["time"].min() == pd.Timestamp("2010-01-01 12:00")
    origin_east = line.coords[0][0]
    x_even = s.iloc[::2]["x"].to_numpy() + origin_east
    x_odd = s.iloc[1::2]["x"].to_numpy() + origin_east
    assert np.allclose(x_even, 500_100.0, atol=2.5)                 # within the 2 m grid
    assert np.allclose(x_odd - x_even[: len(x_odd)], 4.0, atol=2.5)
    assert np.allclose(s["slope"], 0.05, atol=0.005)
