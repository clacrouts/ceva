"""The dashboard's data: tiles, series, summary and manifest from the results of units."""
import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from ceva import publish


@pytest.fixture
def results(cfg, tmp_path, monkeypatch):
    cfg = cfg.model_copy(update={"data_dir": tmp_path / "data"})
    out = cfg.data_dir / "results" / "u1" / "ceva"
    out.mkdir(parents=True)
    ids = ["a", "b"]
    lines = [LineString([(-1.25, 44.7), (-1.27, 44.7)]), LineString([(-1.27, 44.71), (-1.25, 44.71)])]
    gpd.GeoDataFrame({"transect_id": ids, "seaward_sign": [1, -1]}, geometry=lines, crs=4326).to_parquet(
        out / "transects.parquet")
    years = list(range(2000, 2010))
    rows = [{"transect_id": tid, "year": y, "x": 100 + k * (y - 2000) * 2.0, "n_imagery": 12, "n_radar": 0,
             "se_m": 3.0, "product": "imagery"} for k, tid in ((-1, "a"), (1, "b")) for y in years]
    gpd.GeoDataFrame(rows, geometry=[Point(-1.26, 44.7 + 0.01 * (r["transect_id"] == "b")) for r in rows],
                     crs=4326).to_parquet(out / "shorelines_imagery.parquet")
    pd.DataFrame({"transect_id": ids, "rate": [-2.0, 2.0], "rate_lo": [-2.2, 1.8], "rate_hi": [-1.8, 2.2], "r2": 1.0,
                  "n_years": 10, "first_year": 2000, "last_year": 2009, "product": "imagery"}
                 ).to_parquet(out / "rates_imagery.parquet")
    index = tmp_path / "gcts.parquet"
    pd.DataFrame({"transect_id": ids, "country": "FR", "common_country_name": "France"}).to_parquet(index)
    monkeypatch.setattr("ceva.transects.gcts.available_indexes", lambda cfg: [index])
    monkeypatch.setattr("ceva.transects.typology.load", lambda cfg, ids=None: pd.DataFrame(
        {"transect_id": ["a", "b"], "shore_type": "sandy_gravel_or_small_boulder_sediments", "coastal_type": "dune",
         "is_built_environment": False, "has_defence": False, "sds_rate": [-1.5, 1.0], "sds_qa": True}))
    return cfg


def test_units_become_tiles_series_summary_and_manifest(results, tmp_path):
    if publish._tippecanoe() is None:
        pytest.skip("tippecanoe is not installed")
    out = tmp_path / "site" / "data"
    (out / "series").mkdir(parents=True)
    (out / "series" / "old_unit.json").write_text("{}")             # from an earlier publication
    manifest = publish.publish(results, out, ["u1", "missing"])
    assert not (out / "series" / "old_unit.json").exists()
    assert manifest["transects"] == 2 and manifest["units"] == 1 and manifest["years"] == [2000, 2009]
    assert (out / "shoreline.pmtiles").stat().st_size > 0
    series = json.loads((out / "series" / "u1.json").read_text())["transects"]
    assert series["a"]["year"][:2] == [2000, 2001] and series["a"]["x"][1] == 98.0
    assert series["b"]["line"] == [[-1.25, 44.71], [-1.27, 44.71]]       # oriented land to sea (stored sea to land)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["all"]["classes"]["intense_erosion"] == 1 and summary["all"]["classes"]["intense_accretion"] == 1
    assert summary["countries"]["FR"]["name"] == "France" and summary["all"]["significant_erosion"] == 1
    stats = json.loads((out / "stats.json").read_text())
    assert stats["keys"]["cc"] == ["FR"] and stats["keys"]["coast"] == ["dune"]
    assert manifest["files"]["stats"] == "stats.json"
    by_class = {stats["keys"]["cls"][g[3]]: g for g in stats["groups"]}
    assert by_class["intense_erosion"][-2:] == [1, [-200]] and by_class["intense_accretion"][-2:] == [1, [200]]
    assert {g[4:8] for g in map(tuple, stats["groups"])} == {(1, 0, 0, 0)}     # significant, natural, se 3 m


def _waterlines(cfg, unit, ids):
    pd.DataFrame({"transect_id": ids, "time": pd.to_datetime(["2001-06-01"] * len(ids)),
                  "platform": "landsat-7", "family": "landsat", "x_obs": 100.0, "tide": 0.5, "wave": 0.1,
                  "level": 0.6, "x_corr": 104.0, "threshold": 0.1, "outlier": False, "low_water": False,
                  "use": True}).to_parquet(cfg.data_dir / "results" / unit / "ceva" / "waterlines.parquet")


def test_the_dataset_has_rates_and_shorelines_whole_and_waterlines_by_country(results, tmp_path):
    import shutil

    import pyarrow.parquet as pq

    from ceva import export

    _waterlines(results, "u1", ["a", "b"])
    # a second unit in another country, without a coastal type: columns empty there, a wider bounding box
    u1, u2 = (results.data_dir / "results" / u / "ceva" for u in ("u1", "u2"))
    shutil.copytree(u1, u2)
    for name in ("transects", "shorelines_imagery"):
        g = gpd.read_parquet(u2 / f"{name}.parquet")
        g["transect_id"] = g["transect_id"].map({"a": "c", "b": "d"})
        g.geometry = g.geometry.translate(xoff=3.0)
        g.to_parquet(u2 / f"{name}.parquet")
    rates = pd.read_parquet(u2 / "rates_imagery.parquet").assign(transect_id=["c", "d"])
    rates.to_parquet(u2 / "rates_imagery.parquet")
    _waterlines(results, "u2", ["c", "d"])
    pd.DataFrame({"transect_id": ["a", "b", "c", "d"], "country": ["FR", "FR", "ES", "ES"],
                  "common_country_name": ["France", "France", "Spain", "Spain"]}).to_parquet(tmp_path / "gcts.parquet")

    out = tmp_path / "zenodo"
    summary = export.export(results, out, ["u1", "u2"], version="0.1")
    names = sorted(p.name for p in out.iterdir())
    assert names == ["README.md", "SHA256SUMS", "ceva-shorelines_ES_waterlines.parquet",
                     "ceva-shorelines_FR_waterlines.parquet", "ceva-shorelines_rates.parquet",
                     "ceva-shorelines_shorelines.parquet", "metadata.json"]
    assert summary["countries"] == {"ES": {"rates": 2, "shorelines": 20, "waterlines": 2},
                                    "FR": {"rates": 2, "shorelines": 20, "waterlines": 2}}
    rates = gpd.read_parquet(out / "ceva-shorelines_rates.parquet").set_index("transect_id")
    assert rates.loc["a", "change_class"] == "intense_erosion" and bool(rates.loc["a", "significant"])
    assert list(rates.loc["b", "geometry"].coords)[0] == (-1.25, 44.71)       # from the landward end
    assert rates.loc["a", "sds_rate"] == -1.5 and pd.isna(rates.loc["c", "sds_rate"])
    assert rates.loc["c", "country"] == "ES"
    geo = json.loads(pq.read_schema(out / "ceva-shorelines_shorelines.parquet").metadata[b"geo"])
    west, _, east, _ = geo["columns"]["geometry"]["bbox"]
    assert west == pytest.approx(-1.26) and east == pytest.approx(1.74)       # both units
    assert len(gpd.read_parquet(out / "ceva-shorelines_shorelines.parquet")) == 40
    meta = json.loads((out / "metadata.json").read_text())["metadata"]
    assert meta["license"] == "cc-by-4.0" and meta["version"] == "0.1"
    assert {"identifier": export.CODE_URL, "relation": "isCompiledBy", "scheme": "url"} in meta["related_identifiers"]
    assert "ceva-shorelines_FR_waterlines.parquet" in (out / "SHA256SUMS").read_text()
    assert "ceva-shorelines_rates.parquet" in (out / "README.md").read_text()
    assert not (out / ".parts").exists()


def test_the_coast_not_measured_is_drawn_as_runs_of_one_kind():
    from ceva.publish import coverage_lines

    ids = [f"cl1s00tr{100 * k:08d}" for k in range(7)] + ["cl2s00tr00000500"]
    transects = pd.DataFrame({
        "transect_id": ids, "lon": [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 5.0], "lat": 45.0, "bearing": 0.0,
        "shore_type": ["rock", "rock", "sand", "sand", "mud", "rock", "rock", "sand"],
    })
    lines = coverage_lines(transects, measured={ids[2]}, shore_types=["sand"])
    # rock x2 | (measured) | sand without a result | mud | rock x2 | a lone sand transect on another coastline
    assert sorted(lines["why"]) == ["mud", "no_result", "no_result", "rock", "rock"]
    lengths = sorted(round(g.length, 4) for g in lines.geometry)
    assert lengths.count(0.001) == 2                     # the two runs of two transects
    lone = lines.geometry[[g.coords[0][0] > 4 for g in lines.geometry]].iloc[0]
    assert abs(lone.coords[0][1] - lone.coords[1][1]) < 1e-9 and lone.length > 0    # 100 m along the coast (east-west)


def _index(cfg):
    from ceva import vulnerability

    table = pd.DataFrame({"transect_id": ["a", "b"], "unit": "u1", "coastal_slope_pct": [0.4, 1.5],
                          "sea_level_trend_mm_yr": 3.1, "uplift_mm_yr": [-0.4, 0.2],
                          "relative_sea_level_mm_yr": [3.5, 2.9], "mean_tidal_range_m": 2.9, "mean_wave_height_m": 1.8,
                          "cvi": [27.4, 5.5], "cvi_class": [3, 0]})
    for k, name in enumerate(vulnerability.VARIABLES):
        table[f"rank_{name}"] = [[5, 4, 4, 5, 5, 3][k], [5, 1, 1, 3, 5, 3][k]]
    table.attrs["class_edges"] = [8.0, 15.0, 22.0]
    vulnerability.write(cfg, table)


def test_the_vulnerability_index_goes_to_the_cards_the_summary_and_the_dataset(results, tmp_path):
    from ceva import export

    _index(results)
    if publish._tippecanoe() is not None:
        out = tmp_path / "site" / "data"
        manifest = publish.publish(results, out, ["u1"])
        assert manifest["vulnerability"]["class_edges"] == [8.0, 15.0, 22.0]
        assert manifest["vulnerability"]["transects"] == 2 and any("GEBCO" in a for a in manifest["attribution"])
        card = json.loads((out / "series" / "u1.json").read_text())["transects"]["a"]["vi"]
        assert card["r"] == [5, 4, 4, 5, 5, 3] and card["slope"] == 0.4 and card["rsl"] == 3.5
        block = json.loads((out / "summary.json").read_text())["countries"]["FR"]["vulnerability"]
        assert block == {"known": 2, "classes": [1, 0, 0, 1], "median": 16.45}
    _waterlines(results, "u1", ["a"])
    export.export(results, tmp_path / "zenodo", ["u1"], version="0.1")
    rates = gpd.read_parquet(tmp_path / "zenodo" / "ceva-shorelines_rates.parquet").set_index("transect_id")
    assert rates.loc["a", "cvi_class"] == "very_high" and rates.loc["b", "rank_coastal_slope"] == 1
