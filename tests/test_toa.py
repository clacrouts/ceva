"""Top-of-atmosphere chips: Level-1 twins of Level-2 scenes, their scaling, and how chips are read."""
import datetime as dt
import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from ceva.catalog.models import BandAsset, Scene
from ceva.catalog.providers import TOA_SPECS, normalise_item
from ceva.catalog.search import filter_and_normalise, match_key
from ceva.catalog.usgs import band_number, toa_scaling
from ceva.chips import ChipGrid, ToaSource, _read_acquisition, chips_path
from ceva.config import load_config

EPSG = 32630
BANDS = ("blue", "green", "red", "nir", "swir1")
X0, Y0 = 600_000.0, 4_960_000.0


def write_raster(path, values, res, dtype="uint16"):
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1], count=1,
                       dtype=dtype, crs=f"EPSG:{EPSG}", transform=from_origin(X0, Y0, res, res), nodata=0) as dst:
        dst.write(values.astype(dtype), 1)
    return str(path)


def l1c_item(item_id="S2A_MSIL1C_20200602T105631_N0500_R094_T30TXQ_20230617T073825", cloud=95.0):
    band = {"raster:scale": 0.0001, "raster:offset": -0.1}
    return {
        "id": item_id,
        "properties": {"platform": "sentinel-2a", "datetime": "2020-06-02T10:56:31.024Z", "eo:cloud_cover": cloud,
                       "processing:version": "05.00", "grid:code": "MGRS-30TXQ"},
        "assets": {k: {"href": f"s3://eodata/Sentinel-2/x/{k}.jp2", **band}
                   for k in ["B02", "B03", "B04", "B08", "B11", "B12"]},
    }


def scene(scene_id, family="sentinel2", platform="sentinel-2a", grid="30TXQ", **bands) -> Scene:
    return Scene(scene_id, "local", family, platform, dt.datetime(2020, 6, 2, 10, 56, tzinfo=dt.UTC), grid, 10.0,
                 "05.00", "s2_scl", {k: v if isinstance(v, BandAsset) else BandAsset(v) for k, v in bands.items()})


def test_level1c_item_needs_no_mask_and_keeps_its_declared_scaling():
    s = normalise_item(l1c_item(), TOA_SPECS[("cdse", "sentinel2")], "toa")
    assert "mask" not in s.bands and s.meta["level"] == "toa"
    assert (s.bands["green"].scale, s.bands["green"].offset, s.bands["green"].nodata) == (0.0001, -0.1, 0.0)


def test_level1_twins_are_not_cloud_filtered():
    """The Level-2 scene was filtered already; the two products estimate cloud cover differently."""
    cfg = load_config()
    kept, rejected, _ = filter_and_normalise([l1c_item(cloud=95.0)], TOA_SPECS[("cdse", "sentinel2")],
                                             cfg.sensors["sentinel2"], "toa")
    assert len(kept) == 1 and not rejected


def test_level1_and_level2_products_of_one_acquisition_share_a_key():
    l2a = scene("S2A_MSIL2A_20200602T105631_R094_T30TXQ_20201014T120000")
    l1c = scene("S2A_MSIL1C_20200602T105631_N0500_R094_T30TXQ_20230617T073825")
    other_tile = scene("S2A_MSIL1C_20200602T105631_N0500_R094_T30TXR_20230617T073825", grid="30TXR")
    assert match_key(l2a) == match_key(l1c) != match_key(other_tile)
    l2 = scene("LC08_L2SP_200029_20200627_20200823_02_T1", "landsat", "landsat-8", "200029")
    l1 = scene("LC08_L1TP_200029_20200627_20200823_02_T1", "landsat", "landsat-8", "200029")
    l1_reprocessed = scene("LC08_L1TP_200029_20200627_20210101_02_T1", "landsat", "landsat-8", "200029")
    assert match_key(l2) == match_key(l1) == match_key(l1_reprocessed)


def test_landsat_toa_scaling_folds_in_the_sun_elevation():
    mtl = {"LANDSAT_METADATA_FILE": {
        "LEVEL1_RADIOMETRIC_RESCALING": {"REFLECTANCE_MULT_BAND_3": "2.0000E-05",
                                         "REFLECTANCE_ADD_BAND_3": "-0.100000"},
        "IMAGE_ATTRIBUTES": {"SUN_ELEVATION": "30.0"}}}
    scale, offset = toa_scaling(mtl, 3)
    assert scale == pytest.approx(4e-5) and offset == pytest.approx(-0.2)     # sin(30 degrees) = 0.5
    assert toa_scaling(mtl, 3, sun_elevation=90.0) == pytest.approx((2e-5, -0.1))
    assert 10_000 * scale + offset == pytest.approx((2e-5 * 10_000 - 0.1) / math.sin(math.radians(30)))
    assert band_number("s3://x/LC08_L1TP_200029_20200627_20200823_02_T1_B3.TIF") == 3
    with pytest.raises(ValueError):
        toa_scaling(mtl, 3, sun_elevation=-5.0)


def test_toa_chips_take_bands_from_the_twin_and_the_mask_from_level2(tmp_path):
    bands = ("blue", "green", "red", "nir", "swir1")
    surface = {b: write_raster(tmp_path / f"sr_{b}.tif", np.full((40, 40), 1500), 10.0) for b in bands}
    toa = {b: write_raster(tmp_path / f"toa_{b}.tif", np.full((40, 40), 2500), 10.0) for b in bands}
    scl = write_raster(tmp_path / "scl.tif", np.full((20, 20), 6), 20.0, dtype="uint8")
    scaled = lambda path: BandAsset(path, scale=1e-4, offset=-0.1, nodata=0)   # noqa: E731
    l2a = scene("S2A_MSIL2A_20200602T105631_R094_T30TXQ_20201014T120000",
                **{b: scaled(p) for b, p in surface.items()}, mask=scl)
    l1c = scene("S2A_MSIL1C_20200602T105631_N0500_R094_T30TXQ_20230617T073825",
                **{b: scaled(p) for b, p in toa.items()})
    lonely = scene("S2B_MSIL2A_20200607T105619_R094_T30TXQ_20201014T120000",
                   **{b: scaled(p) for b, p in surface.items()}, mask=scl)
    grid = ChipGrid(EPSG, X0 + 50.0, Y0 - 50.0, 10.0, 20, 20)
    source = ToaSource({match_key(l1c): l1c}, {})
    data, failed = _read_acquisition([l2a], grid, bands, {}, source)
    assert not failed and np.allclose(data["green"], 0.15) and (data["qa"] == 6).all()
    sr, _ = _read_acquisition([l2a], grid, bands, {})
    assert np.allclose(sr["green"], 0.05)
    assert _read_acquisition([lonely], grid, bands, {}, source) == (None, [])   # no twin: left out, not failed


def test_each_reflectance_level_has_its_own_repository():
    cfg = load_config()
    assert chips_path(cfg, "x").name == "x.icechunk"
    assert chips_path(cfg, "x", "toa").name == "x-toa.icechunk"


def test_open_chips_prefers_toa_and_falls_back_to_surface(cfg):
    pytest.importorskip("icechunk")
    import xarray as xr

    from ceva.chips import open_chips, save_chips

    def chip(value, level):
        ds = xr.Dataset({b: (("time", "y", "x"), np.full((1, 2, 2), value, dtype="float32")) for b in BANDS},
                        coords={"time": [np.datetime64("2020-01-01")], "x": [0.5, 1.5], "y": [1.5, 0.5]})
        ds["qa"] = (("time", "y", "x"), np.ones((1, 2, 2), dtype="uint8"))
        return ds.assign_attrs(reflectance=level, family="x")

    save_chips(cfg, "u", {"landsat/s000": chip(0.1, "surface"), "sentinel2/s000": chip(0.2, "surface")}, "sr")
    save_chips(cfg, "u", {"sentinel2/s000": chip(0.3, "toa")}, "toa", level="toa")
    toa_cfg = cfg.model_copy(update={"chips": cfg.chips.model_copy(
        update={"reflectance": {"landsat": "toa", "sentinel2": "toa"}})})
    got = open_chips(toa_cfg, "u")
    assert sorted(got) == ["landsat/s000", "sentinel2/s000"]
    assert got["sentinel2/s000"].attrs["reflectance"] == "toa" and float(got["sentinel2/s000"]["green"].max()) == 0.3
    assert got["landsat/s000"].attrs["reflectance"] == "surface"           # no TOA chips for Landsat
    surface_cfg = cfg.model_copy(update={"chips": cfg.chips.model_copy(
        update={"reflectance": {"landsat": "surface", "sentinel2": "surface"}})})
    assert float(open_chips(surface_cfg, "u")["sentinel2/s000"]["green"].max()) == pytest.approx(0.2)
