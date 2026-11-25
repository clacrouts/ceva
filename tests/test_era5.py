"""ERA5 waves from the Climate Data Store: requests, the local store of coastal cells, the nearest sea cell."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ceva import era5


@pytest.fixture
def wcfg(cfg, tmp_path):
    waves = cfg.waves.model_copy(update={"source": "cds_era5", "cds": cfg.waves.cds.model_copy(
        update={"area": (46.0, -3.0, 44.0, -1.0), "coastal_reach_cells": 1})})
    return cfg.model_copy(update={"data_dir": tmp_path, "waves": waves})


def year_file(year: int) -> xr.Dataset:
    """A 5 x 5 grid (0.5 degrees): land on the east column, sea elsewhere; swh = row + col / 10 + hour / 1000."""
    times = pd.date_range(f"{year}-01-01", periods=48, freq="h")
    lat, lon = np.arange(46.0, 43.9, -0.5), np.arange(-3.0, -0.9, 0.5)
    swh = (np.arange(5)[None, :, None] + np.arange(5)[None, None, :] / 10 + np.arange(48)[:, None, None] / 1000
           + (year - 2020)).astype("float32")
    swh[:, :, 4] = np.nan                                   # land
    return xr.Dataset({"swh": (("valid_time", "latitude", "longitude"), swh)},
                      coords={"valid_time": times, "latitude": lat, "longitude": lon, "expver": "0001"})


class FakeCDS:
    def __init__(self):
        self.requests = []

    def retrieve(self, dataset, request, target):
        self.requests.append((dataset, request))
        year_file(int(request["year"][0])).to_netcdf(target)


def test_years_are_requested_once_over_the_box(wcfg):
    cds = FakeCDS()
    era5.download(wcfg, [2020, 2021], cds=cds)
    era5.download(wcfg, [2020, 2021, 2022], cds=cds)              # resumes: only 2022 is new
    assert [r["year"] for _, r in cds.requests] == [["2020"], ["2021"], ["2022"]]
    dataset, request = cds.requests[0]
    assert dataset == "reanalysis-era5-single-levels" and request["area"] == [46.0, -3.0, 44.0, -1.0]
    assert request["variable"] == ["significant_height_of_combined_wind_waves_and_swell"]
    assert len(request["time"]) == 24 and len(request["day"]) == 31


def test_the_store_keeps_coastal_sea_cells_and_gives_the_nearest(wcfg):
    era5.download(wcfg, [2020, 2021], cds=FakeCDS())
    path = era5.build_store(wcfg, [2020, 2021])
    store = xr.open_zarr(path)
    assert store.sizes["time"] == 96 and store.sizes["cell"] == 5     # the sea column next to the land only
    assert np.allclose(store["lon"].values, -1.5)
    waves, cell = era5.cds_waves(wcfg, -1.2, 45.1, dt.date(2021, 1, 1), dt.date(2021, 1, 1))
    assert cell == (-1.5, 45.0) and len(waves) == 24
    assert waves["hs"].iloc[0] == pytest.approx(2 + 0.3 + 1)          # row 2, column 3, year 2021, hour 0
    assert waves["tp"].isna().all()


def test_unit_waves_use_the_local_store_when_configured(wcfg):
    from ceva.waves import waves_for_unit

    era5.download(wcfg, [2020], cds=FakeCDS())
    era5.build_store(wcfg, [2020])
    waves, point = waves_for_unit(wcfg, np.array([-1.0]), np.array([44.5]), np.array([-1.3]), np.array([44.5]),
                                  np.array([1]), dt.date(2020, 1, 1), dt.date(2020, 1, 2))
    assert point[0] == -1.5 and len(waves) == 48
