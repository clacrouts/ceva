"""Comparison with other products: matching, same-period rates, classes, the validation page's numbers."""
import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from ceva.validation import agreement
from ceva.validation import references as refs

M_LON = 79_000.0          # metres per degree of longitude near 44.7 N (approximately)
M_LAT = 111_000.0


def test_rate_over_a_products_years_needs_enough_of_them():
    years = np.arange(1984, 2026)
    x = 100.0 - 1.0 * (years - 1984)
    rate, lo, hi, n, y0, y1 = refs.period_rate(years, x, 2000, 2021)
    assert rate == pytest.approx(-1.0) and n == 22 and (y0, y1) == (2000, 2021)
    assert refs.period_rate(years, x, 1950, 2011)[4] == 1984              # starts with the satellite record
    sparse = years[(years < 2003) | (years > 2018)]
    assert math.isnan(refs.period_rate(sparse, x[np.isin(years, sparse)], 2000, 2021)[0])   # half the years missing
    late = years[years >= 2012]
    assert math.isnan(refs.period_rate(late, x[years >= 2012], 2000, 2021)[0])  # nothing near the start
    assert math.isnan(refs.period_rate(years, x, 2019, 2021)[0])                # too short
    assert math.isnan(refs.period_rate(years, x, None, 2021)[0])


def test_ceva_calls_change_only_where_it_is_significant():
    assert refs.ceva_class(-0.8, -1.2, -0.4, 0.5) == "erosion"
    assert refs.ceva_class(-0.8, -1.9, 0.3, 0.5) == "stable"              # not significant
    assert refs.ceva_class(-0.3, -0.4, -0.2, 0.5) == "stable"             # significant but within the band
    assert refs.ceva_class(-0.3, -0.4, -0.2, None) == "erosion"           # a product without a stable band
    assert refs.ceva_class(0.9, 0.6, 1.2, None) == "accretion"
    assert refs.ceva_class(math.nan, math.nan, math.nan, 0.5) is None


def _unit():
    """One transect (land to the east, sea to the west), shoreline at -1.26 E, retreating 1 m/yr;
    a second one 2 km north that no product covers."""
    lines = [LineString([(-1.25, 44.70), (-1.27, 44.70)]), LineString([(-1.25, 44.718), (-1.27, 44.718)])]
    transects = gpd.GeoDataFrame({"transect_id": ["a", "b"], "seaward_sign": [1, 1]}, geometry=lines, crs=4326)
    years = list(range(1984, 2026))
    rows = [{"transect_id": tid, "year": y, "x": 100.0 - (y - 1984), "se_m": 4.0, "n_imagery": 10}
            for tid in ("a", "b") for y in years]
    geometry = [Point(-1.26, 44.70 if r["transect_id"] == "a" else 44.718) for r in rows]
    shores = gpd.GeoDataFrame(rows, geometry=geometry, crs=4326)
    points = gpd.GeoDataFrame({"transect_id": ["a", "b"]}, geometry=[Point(-1.26, 44.70), Point(-1.26, 44.718)],
                              crs=4326)
    return points, shores, transects


def _references():
    sm = gpd.GeoDataFrame({"transect_id": ["a"], "rate": [-0.8], "n_obs": [40.0], "y0": [1984], "y1": [2023],
                           "stdev": [10.0], "qa": [True]},
                          geometry=[Point(-1.26 - 20 / M_LON, 44.70)], crs=4326).to_crs(refs.METRIC_CRS)
    sm = sm.set_index("transect_id", drop=False)

    def near(east_m, north_m):       # a point east (landward) and north of CEVA's shoreline on transect "a"
        return Point(-1.26 + east_m / M_LON, 44.70 + north_m / M_LAT)

    sat = gpd.GeoDataFrame({"ref_id": ["behind_the_beach", "far_inland"], "rate": [-0.5, 3.0], "unc": [0.3, 0.3],
                            "cls": ["stable", "accretion"], "y0": [2007, 2007], "y1": [2017, 2017],
                            "source": ["Sandy", "Sandy"]},
                           geometry=[near(60, 10), near(600, 5)], crs=4326).to_crs(refs.METRIC_CRS)
    field = gpd.GeoDataFrame({"ref_id": ["7"], "cls": ["erosion"], "y0": [1950], "y1": [2011]},
                             geometry=[LineString([(-1.2605, 44.69), (-1.2605, 44.705)])], crs=4326
                             ).to_crs(refs.METRIC_CRS)
    return {"sm": sm, "emodnet_sat": sat, "emodnet_field": field}


def test_each_product_is_matched_along_the_transect_with_ceva_over_its_years():
    points, shores, transects = _unit()
    table = refs.match(_references(), points, shores, transects).set_index("product")
    assert set(table["transect_id"]) == {"a"}                              # "b" is 2 km away from everything
    assert set(table.index) == {"sm", "emodnet_sat", "emodnet_field"}
    sm, sat, field = (table.loc[k] for k in ("sm", "emodnet_sat", "emodnet_field"))
    assert sm["ceva_rate"] == pytest.approx(-1.0) and sm["cls"] == "erosion" and sm["ceva_cls"] == "erosion"
    assert sm["offset_m"] == pytest.approx(20.0, abs=1.0)                  # the Monitor's shoreline is 20 m seaward
    # the product's transect 60 m behind CEVA's waterline, 10 m from its transect, not the closer one 600 m inland
    assert sat["ref_id"] == "behind_the_beach" and sat["distance_m"] == pytest.approx(10.0, abs=1.0)
    assert (sat["ceva_y0"], sat["ceva_y1"]) == (2007, 2017) and sat["ceva_cls"] == "erosion"
    assert field["ceva_y0"] == 1984 and field["ceva_rate"] == pytest.approx(-1.0)
    cards = refs.card_entries(table.reset_index())
    card = dict(cards["a"]["emodnet_sat"])
    assert card.pop("d") == pytest.approx(10.0, abs=1.0)
    assert card == {"r": -0.5, "y": [2007, 2017], "c": -1.0, "k": "stable"}
    assert cards["a"]["sm"]["q"] == 1 and cards["a"]["sm"]["o"] == pytest.approx(20.0, abs=1.0)


def test_nothing_to_compare_gives_an_empty_table():
    points, shores, transects = _unit()
    assert refs.match({}, points, shores, transects).empty
    assert list(refs.match({}, points, shores, transects).columns) == refs.COLUMNS


def _pairs(n=200, seed=1):
    rng = np.random.default_rng(seed)
    rate = rng.normal(0, 1, n)
    ceva = rate + rng.normal(-0.1, 0.3, n)
    cls = np.where(rate < -0.5, "erosion", np.where(rate > 0.5, "accretion", "stable"))
    ceva_cls = np.where(ceva < -0.5, "erosion", np.where(ceva > 0.5, "accretion", "stable"))
    ids = [f"t{i}" for i in range(n)]
    pairs = pd.DataFrame({"transect_id": ids, "product": "sm", "ref_id": ids, "rate": rate, "cls": cls,
                          "ceva_rate": ceva, "ceva_lo": ceva - 0.4, "ceva_hi": ceva + 0.4, "ceva_cls": ceva_cls,
                          "ceva_y0": 1984, "ceva_y1": 2023, "y0": 1984, "y1": 2023, "qa": True,
                          "offset_m": rng.normal(-10, 15, n), "distance_m": 0.0, "unit": "u1"})
    table = gpd.GeoDataFrame({"id": ids, "unit": "u1", "cc": "FR", "cn": "France", "coast": "dune", "built": False,
                              "def": np.arange(n) % 2 == 0, "se": rng.uniform(2, 14, n), "rate": ceva, "ny": 40,
                              "y0": 1984, "y1": 2025},
                             geometry=[Point(-1.2, 44.7)] * n, crs=4326)
    return pairs, table


def test_agreement_numbers_and_chart_data():
    pairs, table = _pairs()
    m = agreement.metrics(pairs)
    diff = pairs["ceva_rate"] - pairs["rate"]
    assert m["n"] == 200 and m["bias"] == pytest.approx(diff.median(), abs=1e-3)
    assert m["mad"] == pytest.approx(diff.abs().median(), abs=1e-3) and 0.8 < m["r"] <= 1
    assert m["agree"] == pytest.approx((pairs["cls"] == pairs["ceva_cls"]).mean(), abs=1e-3) and m["kappa"] > 0.5
    assert agreement.kappa(pd.Series(["erosion", "stable"]), pd.Series(["erosion", "stable"])) == 1.0
    hist = agreement.histogram(pd.Series([-9.0, -0.05, 0.05, 0.05, 9.0]), 1.0, 0.1)
    assert hist["below"] == 1 and hist["above"] == 1 and sum(hist["counts"]) == 3 and len(hist["counts"]) == 20
    grid = agreement.density(pd.Series([0.05, 5.0]), pd.Series([0.05, 0.0]), limit=1.0, step=0.5)
    assert grid["outside"] == 1 and grid["counts"][2][2] == 1
    conf = agreement.confusion(pairs)["counts"]
    assert sum(map(sum, conf)) == 200 and conf[1][1] > conf[1][0]


def test_validation_payload_says_whether_it_is_provisional():
    pairs, table = _pairs()
    sites = {"generated": "2026-11-19", "sites": [{"key": "trucvert", "methods": {
        "ceva": {"yearly": {"rmse": 13.5}, "rates": {"rmse": 0.95}}}}]}
    out = agreement.payload(pairs, table, sites, {"planned": 10, "finished": 4})
    assert out["provisional"] and out["transects"] == 200 and out["years"] == [1984, 2025]
    block = out["products"][0]
    assert block["key"] == "sm" and block["overall"]["n"] == 200 and block["share"] == 1.0
    assert [r["key"] for r in block["by"]["defence"]] == ["no", "yes"]
    assert [r["key"] for r in block["by"]["precision"]][0] == "<5"
    assert block["offset"]["median"] == pytest.approx(pairs["offset_m"].median(), abs=0.1)
    assert len(block["examples"]) == agreement.N_EXAMPLES and block["examples"][0]["unit"] == "u1"
    assert not agreement.payload(pairs, table, None, {"planned": 10, "finished": 10})["provisional"]
    assert agreement.survey_ranges(sites["sites"]) == {"yearly_rmse": [13.5, 13.5], "rate_rmse": [0.95, 0.95],
                                                       "sites": 1}
    json.dumps(out)                                                       # ready for the page


def test_publish_compares_with_prepared_products(cfg, tmp_path, monkeypatch):
    from ceva import publish

    if publish._tippecanoe() is None:
        pytest.skip("tippecanoe is not installed")
    cfg = cfg.model_copy(update={"data_dir": tmp_path / "data"})
    out_dir = cfg.data_dir / "results" / "u1" / "ceva"
    out_dir.mkdir(parents=True)
    points, shores, transects = _unit()
    transects.to_parquet(out_dir / "transects.parquet")
    shores.assign(product="imagery", n_radar=0).to_parquet(out_dir / "shorelines_imagery.parquet")
    pd.DataFrame({"transect_id": ["a", "b"], "rate": [-1.0, -1.0], "rate_lo": [-1.01, -1.01], "rate_hi": [-0.99, -0.99],
                  "r2": 1.0, "n_years": 42, "first_year": 1984, "last_year": 2025, "product": "imagery"}
                 ).to_parquet(out_dir / "rates_imagery.parquet")
    index = tmp_path / "gcts.parquet"
    pd.DataFrame({"transect_id": ["a", "b"], "country": "FR", "common_country_name": "France"}).to_parquet(index)
    monkeypatch.setattr("ceva.transects.gcts.available_indexes", lambda cfg: [index])
    monkeypatch.setattr("ceva.transects.typology.load", lambda cfg, ids=None: pd.DataFrame(
        {"transect_id": ["a", "b"], "shore_type": "sandy_gravel_or_small_boulder_sediments", "coastal_type": "dune",
         "is_built_environment": False, "has_defence": False, "sds_rate": [-0.8, np.nan], "sds_qa": True}))
    (cfg.data_dir / "references").mkdir(parents=True)
    for key, g in _references().items():
        g.reset_index(drop=True).to_crs(4326).to_parquet(refs.path(cfg, key))
    site = tmp_path / "site" / "data"
    (site / "refs").mkdir(parents=True)
    (site / "refs" / "gone.json").write_text("{}")
    manifest = publish.publish(cfg, site, ["u1"], run={"name": "r", "planned": 2, "finished": 1})
    assert manifest["provisional"] and manifest["files"]["validation"] == "validation.json"
    assert [r["key"] for r in manifest["references"]] == ["sm", "emodnet_sat", "emodnet_field"]
    assert any("BY-SA" in a for a in manifest["attribution"])
    cards = json.loads((site / "refs" / "u1.json").read_text())["transects"]
    assert set(cards) == {"a"} and set(cards["a"]) == {"sm", "emodnet_sat", "emodnet_field"}
    assert not (site / "refs" / "gone.json").exists()
    validation = json.loads((site / "validation.json").read_text())
    assert validation["provisional"] and [p["key"] for p in validation["products"]][0] == "sm"
