import pytest
from pydantic import ValidationError

from ceva.config import DATA_DIR_ENV, deep_merge, load_config, packaged_regions


def test_defaults_load_without_region():
    cfg = load_config()
    assert cfg.region is None
    assert cfg.grid.processing_zoom == 10
    assert set(cfg.sensors) == {"landsat", "sentinel2"}


def test_region_is_merged_on_defaults():
    cfg = load_config("gironde")
    assert cfg.region.name == "gironde"
    assert cfg.region.bbox_wgs84[0] == pytest.approx(-1.279822)
    assert cfg.sensors["sentinel2"].providers[0] == "planetary_computer"
    assert cfg.sensors["sentinel2"].toa_providers == ["cdse"]


def test_packaged_regions_listed():
    assert {"gironde", "truc_vert"} <= set(packaged_regions())


def test_unknown_region_names_the_choices():
    with pytest.raises(FileNotFoundError, match="gironde"):
        load_config("atlantis")


def test_region_from_path(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("region:\n  name: custom\n  bbox_wgs84: [150.0, -34.0, 151.0, -33.0]\n  continent: Oceania\n")
    assert load_config(path).region.name == "custom"


def test_data_dir_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "from_env"))
    assert load_config().data_dir == (tmp_path / "from_env").resolve()
    explicit = load_config(overrides={"data_dir": str(tmp_path / "explicit")})
    assert explicit.data_dir == (tmp_path / "explicit").resolve()


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError):
        load_config(overrides={"grid": {"procesing_zoom": 9}})


def test_unknown_provider_is_rejected():
    with pytest.raises(ValidationError, match="unknown providers"):
        load_config(overrides={"sensors": {"landsat": {"providers": ["nowhere"]}}})


def test_bad_bbox_is_rejected():
    with pytest.raises(ValidationError):
        load_config("gironde", overrides={"region": {"bbox_wgs84": [-1.0, 46.0, 0.0, 45.0]}})


def test_antimeridian_bbox_is_accepted():
    cfg = load_config("gironde", overrides={"region": {"bbox_wgs84": [179.0, -18.0, -179.0, -16.0]}})
    assert cfg.region.bbox_wgs84[0] > cfg.region.bbox_wgs84[2]


def test_deep_merge_replaces_lists_and_keeps_siblings():
    merged = deep_merge({"a": {"x": [1, 2], "y": 1}}, {"a": {"x": [3]}})
    assert merged == {"a": {"x": [3], "y": 1}}
