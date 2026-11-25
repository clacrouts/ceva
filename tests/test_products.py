"""The two yearly products: imagery only, and imagery + radar."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pyproj import Geod
from shapely.geometry import LineString

from ceva.config import load_config
from ceva.products import coverage, load_products, write_products, yearly_products

GEOD = Geod(ellps="WGS84")


@pytest.fixture(scope="module")
def config():
    return load_config()


def transects():
    lines = [LineString([(-1.25, 44.70), (-1.26, 44.70)]),      # stored land to sea
             LineString([(-1.26, 44.71), (-1.25, 44.71)]),      # stored sea to land
             LineString([(-1.26, 44.72), (-1.25, 44.72)])]      # no shoreline found
    return gpd.GeoDataFrame({"transect_id": ["a", "b", "c"], "seaward_sign": [1, -1, 0]}, geometry=lines,
                            crs="EPSG:4326")


def waterlines(years, per_year, x, sigma, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for tid in ("a", "b", "c"):
        for y in years:
            t = pd.Timestamp(f"{y}-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 364, per_year)), unit="D")
            rows.append(pd.DataFrame({"transect_id": tid, "time": t, "x_corr": x + rng.normal(0, sigma, per_year),
                                      "use": True}))
    return pd.concat(rows, ignore_index=True)


def test_two_products_and_what_radar_adds(config, tmp_path):
    imagery = waterlines([2018, 2019, 2020], 20, 100.0, 8.0, 0)
    radar = waterlines([2019, 2020, 2021], 60, 100.0, 15.0, 1)            # 2021: radar only (a cloudy year)
    products = yearly_products(config, transects(), imagery, radar)
    assert set(products) == {"imagery", "imagery_radar"}
    img, both = products["imagery"], products["imagery_radar"]
    assert sorted(img["year"].unique()) == [2018, 2019, 2020] and set(img["transect_id"]) == {"a", "b"}
    assert sorted(both["year"].unique()) == [2018, 2019, 2020, 2021]
    assert (both.loc[both["year"] == 2021, "n_imagery"] == 0).all()
    paired = both[both["year"] == 2019].set_index("transect_id")["se_m"]
    alone = img[img["year"] == 2019].set_index("transect_id")["se_m"]
    assert (paired < alone).all()                          # two sources: a smaller standard error
    # the point lies x metres from the landward end, whichever way the transect is stored
    for tid, start in (("a", (-1.25, 44.70)), ("b", (-1.25, 44.71))):
        p = img[img["transect_id"] == tid].iloc[0]
        _, _, dist = GEOD.inv(start[0], start[1], p.geometry.x, p.geometry.y)
        assert dist == pytest.approx(p["x"], abs=0.01)
    assert coverage(products)["radar_only_transect_years"] == 2
    write_products(tmp_path, products)
    stored = load_products(tmp_path)
    assert set(stored) == {"imagery", "imagery_radar"}
    shorelines, rates = stored["imagery_radar"]
    assert len(shorelines) == len(both) and set(rates.columns) >= {"transect_id", "rate", "rate_lo", "rate_hi"}


def test_without_radar_only_the_imagery_product(config, tmp_path):
    products = yearly_products(config, transects(), waterlines([2018, 2019], 10, 50.0, 5.0, 2), None)
    assert set(products) == {"imagery"}
    (tmp_path / "shorelines_imagery_radar.parquet").write_text("stale")
    write_products(tmp_path, products)
    assert not (tmp_path / "shorelines_imagery_radar.parquet").exists()
