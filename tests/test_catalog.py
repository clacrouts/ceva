import datetime as dt

import pytest

from ceva.catalog import deduplicate, provider_status, scenes_to_frame
from ceva.catalog.providers import SPECS, SceneRejected, baseline_tuple, normalise_item
from ceva.catalog.search import filter_and_normalise
from ceva.config import load_config
from ceva.credentials import Credentials


def _assets(keys, **extra):
    return {key: {"href": f"https://example.org/{key}.tif", **extra} for key in keys}


def landsat_item(item_id="LT05_L2SP_200029_19900828_02_T1", platform="landsat-5", category="T1", cloud=12.0):
    assets = _assets(["blue", "green", "red", "nir08", "swir16", "swir22"],
                     **{"raster:bands": [{"scale": 2.75e-05, "offset": -0.2, "nodata": 0}]})
    assets["qa_pixel"] = {"href": "https://example.org/qa_pixel.tif"}
    return {
        "id": item_id,
        "properties": {
            "platform": platform, "datetime": "1990-08-28T10:08:05.002025Z", "eo:cloud_cover": cloud,
            "landsat:wrs_path": "200", "landsat:wrs_row": "029", "landsat:collection_category": category,
        },
        "assets": assets,
    }


def pc_s2_item(baseline, item_id="S2A_MSIL2A_20220723T105631_R094_T30TXR_20220724T010203", cloud=5.0):
    return {
        "id": item_id,
        "properties": {
            "platform": "Sentinel-2A", "datetime": "2022-07-23T10:56:31.024000Z",
            "eo:cloud_cover": cloud, "s2:processing_baseline": baseline, "s2:mgrs_tile": "30TXR",
        },
        "assets": _assets(["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]),
    }


def test_landsat_item_uses_declared_scaling_and_wrs_grid():
    scene = normalise_item(landsat_item(), SPECS[("planetary_computer", "landsat")])
    assert scene.platform == "landsat-5"
    assert scene.grid_id == "200029"
    assert scene.datetime == dt.datetime(1990, 8, 28, 10, 8, 5, 2025, tzinfo=dt.UTC)
    assert (scene.bands["green"].scale, scene.bands["green"].offset) == (2.75e-05, -0.2)
    assert scene.bands["swir1"].href.endswith("swir16.tif")
    assert (scene.bands["mask"].scale, scene.bands["mask"].offset) == (1.0, 0.0)
    assert scene.mask_kind == "landsat_qa_pixel"
    assert scene.processing_baseline is None


@pytest.mark.parametrize(("baseline", "offset"), [("02.12", 0.0), ("03.01", 0.0), ("04.00", -0.1), ("05.10", -0.1)])
def test_sentinel2_offset_follows_the_processing_baseline(baseline, offset):
    scene = normalise_item(pc_s2_item(baseline), SPECS[("planetary_computer", "sentinel2")])
    assert scene.platform == "sentinel-2a"
    assert scene.grid_id == "30TXR"
    assert scene.bands["green"].scale == pytest.approx(1e-4)
    assert scene.bands["green"].offset == pytest.approx(offset)


def test_sentinel2_without_baseline_is_rejected_not_guessed():
    item = pc_s2_item(None)
    with pytest.raises(SceneRejected, match="unknown_processing_baseline"):
        normalise_item(item, SPECS[("planetary_computer", "sentinel2")])


def test_earth_search_declared_scaling_is_used():
    item = {
        "id": "S2B_T30TXR_20250731T110824_L2A",
        "properties": {"platform": "sentinel-2b", "datetime": "2025-07-31T11:08:24Z",
                       "s2:processing_baseline": "05.11", "eo:cloud_cover": 1.0},
        "assets": {
            **_assets(["blue", "green", "red", "nir", "swir16", "swir22"],
                      **{"raster:bands": [{"scale": 0.0001, "offset": -0.1, "nodata": 0}]}),
            "scl": {"href": "https://example.org/scl.tif"},
        },
    }
    scene = normalise_item(item, SPECS[("earth_search", "sentinel2")])
    assert scene.grid_id == "30TXR"
    assert (scene.bands["swir1"].scale, scene.bands["swir1"].offset) == (0.0001, -0.1)


def test_declared_scale_without_offset_still_gets_the_baseline_offset():
    item = pc_s2_item("05.10")
    for asset in item["assets"].values():
        asset["raster:bands"] = [{"scale": 0.0001}]
    scene = normalise_item(item, SPECS[("planetary_computer", "sentinel2")])
    assert scene.bands["green"].offset == pytest.approx(-0.1)


def test_cdse_item():
    item = {
        "id": "S2A_MSIL2A_20160723T105622_N0500_R094_T30TXR_20231028T031858",
        "properties": {"platform": "sentinel-2a", "datetime": "2016-07-23T10:56:22.026000Z",
                       "processing:version": "05.00", "eo:cloud_cover": 19.65, "grid:code": "MGRS-30TXR"},
        "assets": {k: {"href": f"s3://eodata/Sentinel-2/x/{k}.jp2"}
                   for k in ["B02_10m", "B03_10m", "B04_10m", "B08_10m", "B11_20m", "B12_20m", "SCL_20m", "B03_20m"]},
    }
    scene = normalise_item(item, SPECS[("cdse", "sentinel2")])
    assert scene.processing_baseline == "05.00"
    assert scene.bands["green"].href == "s3://eodata/Sentinel-2/x/B03_10m.jp2"
    assert scene.bands["green"].offset == pytest.approx(-0.1)
    assert scene.grid_id == "30TXR"


def test_missing_required_asset_is_rejected():
    item = landsat_item()
    del item["assets"]["swir16"]
    with pytest.raises(SceneRejected, match="missing_asset_swir1"):
        normalise_item(item, SPECS[("planetary_computer", "landsat")])


def test_filtering_counts_every_rejection():
    cfg = load_config()
    items = [
        landsat_item(),
        landsat_item(item_id="LT05_L2SP_200029_19900913_02_T2", category="T2"),
        landsat_item(item_id="LM03_L2SP_200029_19800101_02_T1", platform="landsat-3"),
        landsat_item(item_id="LT05_L2SP_200029_19900929_02_T1", cloud=99.0),
    ]
    scenes, rejected, n_items = filter_and_normalise(items, SPECS[("planetary_computer", "landsat")],
                                                    cfg.sensors["landsat"])
    assert n_items == 4
    assert [s.id for s in scenes] == ["LT05_L2SP_200029_19900828_02_T1"]
    assert rejected == {"collection_category_T2": 1, "platform_landsat-3": 1, "cloud_cover_above_limit": 1}


def test_duplicate_processings_keep_the_newest_baseline():
    spec = SPECS[("planetary_computer", "sentinel2")]
    old = normalise_item(pc_s2_item("04.00", "S2A_MSIL2A_20220723T105631_R094_T30TXR_20220724T010203"), spec)
    new = normalise_item(pc_s2_item("05.10", "S2A_MSIL2A_20220723T105631_R094_T30TXR_20240101T000000"), spec)
    other_tile = normalise_item(pc_s2_item("04.00", "S2A_MSIL2A_20220723T105631_R094_T30TXQ_20220724T010203"), spec)
    kept = deduplicate([old, new, other_tile])
    assert {s.id for s in kept} == {new.id, other_tile.id}


def test_baseline_tuple():
    assert baseline_tuple("05.11") == (5, 11)
    assert baseline_tuple("N0500") == (5, 0)
    assert baseline_tuple(None) == (0, 0)
    assert baseline_tuple("latest") == (0, 0)


def test_provider_status_needs_both_cdse_keys():
    cfg = load_config()
    half = provider_status(cfg, "sentinel2", Credentials({"CDSE_S3_ACCESS_KEY": "a"}))
    assert half["cdse"] == "missing CDSE_S3_SECRET_KEY"
    assert half["planetary_computer"] is None
    full = provider_status(cfg, "sentinel2", Credentials({"CDSE_S3_ACCESS_KEY": "a", "CDSE_S3_SECRET_KEY": "b"}))
    assert full["cdse"] is None
    assert list(provider_status(cfg, "landsat", Credentials())) == ["planetary_computer"]


def test_scene_table_round_trip_columns():
    scene = normalise_item(landsat_item(), SPECS[("planetary_computer", "landsat")])
    frame = scenes_to_frame([scene])
    assert frame.loc[0, "id"] == scene.id
    assert str(frame["datetime"].dt.tz) == "UTC"
    assert '"swir1"' in frame.loc[0, "bands"]
    assert scenes_to_frame([]).empty
