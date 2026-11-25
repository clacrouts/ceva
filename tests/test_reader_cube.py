"""Reading at points, transect sampling and cube storage, on small local rasters."""
import datetime as dt

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj import Geod, Transformer
from rasterio.transform import from_origin
from shapely.geometry import LineString

from ceva.catalog.models import BandAsset, Scene
from ceva.cube import group_acquisitions, open_cube, save_cube
from ceva.reader import encode_qa, read_at_points, sample_scene
from ceva.transects.sampling import sample_transects

EPSG = 32630
X0, Y0, RES, SIZE = 600_000.0, 4_960_000.0, 30.0, 1200       # larger than one read block


def write_raster(path, values, res=RES, dtype="uint16", nodata=0):
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1], count=1,
                       dtype=dtype, crs=f"EPSG:{EPSG}", transform=from_origin(X0, Y0, res, res), nodata=nodata,
                       tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(values.astype(dtype), 1)
    return str(path)


def plane(size=SIZE, res=RES):
    """Raster whose value at each pixel centre is 1000 + easting offset in metres / 10."""
    cols = (np.arange(size) + 0.5) * res
    return np.broadcast_to(1000 + cols / 10.0, (size, size)).round().astype("uint16")


def lonlat(x, y):
    return Transformer.from_crs(EPSG, 4326, always_xy=True).transform(x, y)


def test_bilinear_reads_a_plane_exactly_across_block_edges(tmp_path):
    path = write_raster(tmp_path / "plane.tif", plane())
    x = X0 + np.array([45.0, 600.0, 15_345.0, 15_375.0, 30_000.0])      # 15 360 m is a 512-pixel block edge
    y = Y0 - np.array([45.0, 900.0, 15_345.0, 15_375.0, 30_000.0])
    lon, lat = lonlat(x, y)
    values = read_at_points(path, np.array(lon), np.array(lat))
    assert values == pytest.approx(1000 + (x - X0) / 10.0, abs=0.6)       # raster is rounded to integers


def test_points_outside_and_nodata_give_nan(tmp_path):
    data = plane()
    data[100:110, 100:110] = 0
    path = write_raster(tmp_path / "holes.tif", data)
    lon, lat = lonlat(X0 + np.array([-500.0, 105 * RES, 300 * RES]), Y0 - np.array([300.0, 105 * RES, 300 * RES]))
    values = read_at_points(path, np.array(lon), np.array(lat))
    assert np.isnan(values[0]) and np.isnan(values[1]) and np.isfinite(values[2])
    assert np.isnan(read_at_points(path, np.array([np.nan]), np.array([np.nan])))[0]


def test_nearest_returns_raw_values_and_zero_outside(tmp_path):
    data = np.full((SIZE, SIZE), 21824, dtype="uint16")
    data[:, 600:] = 22280
    path = write_raster(tmp_path / "qa.tif", data)
    lon, lat = lonlat(X0 + np.array([599.4 * RES, 600.6 * RES, -90.0]), Y0 - np.array([3000.0, 3000.0, 3000.0]))
    assert read_at_points(path, np.array(lon), np.array(lat), "nearest").tolist() == [21824, 22280, 0]


def test_encode_qa():
    landsat = encode_qa(np.array([0, 21824, 22280, 21952], dtype="uint16"), "landsat_qa_pixel")
    assert landsat.tolist() == [1, 21824 & 0xFF, 22280 & 0xFF, 21952 & 0xFF]
    assert encode_qa(np.array([0, 6, 9], dtype="uint8"), "s2_scl").tolist() == [0, 6, 9]


def test_sample_scene_applies_scale_and_offset_per_band(tmp_path):
    green = write_raster(tmp_path / "green.tif", np.full((SIZE, SIZE), 10_000, dtype="uint16"))
    swir = write_raster(tmp_path / "swir.tif", np.full((SIZE // 2, SIZE // 2), 2_000, dtype="uint16"), res=2 * RES)
    mask = write_raster(tmp_path / "scl.tif", np.full((SIZE // 2, SIZE // 2), 6, dtype="uint8"), res=2 * RES,
                        dtype="uint8")
    scene = Scene(id="S2A_TEST", provider="local", family="sentinel2", platform="sentinel-2a",
                  datetime=dt.datetime(2022, 7, 1, 11, tzinfo=dt.UTC), grid_id="30TXQ", cloud_cover=0.0,
                  processing_baseline="05.10", mask_kind="s2_scl",
                  bands={"green": BandAsset(green, 1e-4, -0.1, 0), "swir1": BandAsset(swir, 1e-4, -0.1, 0),
                         "mask": BandAsset(mask)})
    lon, lat = lonlat(X0 + np.array([[900.0, 5000.0]]), Y0 - np.array([[900.0, 5000.0]]))
    out = sample_scene(scene, np.array(lon), np.array(lat), ("green", "swir1"))
    assert out["green"] == pytest.approx(0.9) and out["swir1"] == pytest.approx(0.1)
    assert out["qa"].tolist() == [[6, 6]] and out["green"].shape == (1, 2)


def test_sample_transects_spacing_is_metric_and_ends_are_padded():
    geod = Geod(ellps="WGS84")
    lon1, lat1, _ = geod.fwd(-1.25, 44.74, 275.0, 2000.0)
    lon2, lat2, _ = geod.fwd(-1.25, 44.75, 275.0, 1000.0)
    gdf = gpd.GeoDataFrame({"transect_id": ["long", "short"]}, crs="EPSG:4326",
                           geometry=[LineString([(-1.25, 44.74), (lon1, lat1)]),
                                     LineString([(-1.25, 44.75), (lon2, lat2)])])
    s = sample_transects(gdf, 10.0)
    assert s.distance[[0, -1]].tolist() == [0.0, 2000.0] and s.lon.shape == (2, 201)
    _, _, step = geod.inv(s.lon[0, :-1], s.lat[0, :-1], s.lon[0, 1:], s.lat[0, 1:])
    assert step == pytest.approx(10.0, abs=1e-6)
    assert np.isfinite(s.lon[1, :101]).all() and np.isnan(s.lon[1, 101:]).all()
    assert s.length == pytest.approx([2000.0, 1000.0], abs=1e-6)


def _scene(scene_id, platform, when):
    return Scene(scene_id, "p", "landsat", platform, when, None, None, None, "landsat_qa_pixel", {})


def test_same_satellite_same_day_is_one_acquisition():
    day = dt.datetime(2020, 5, 1, 10, 50, tzinfo=dt.UTC)
    groups = group_acquisitions([
        _scene("a", "landsat-8", day), _scene("b", "landsat-8", day + dt.timedelta(seconds=24)),
        _scene("c", "landsat-7", day), _scene("d", "landsat-8", day + dt.timedelta(days=16)),
    ])
    assert sorted(len(g) for g in groups) == [1, 1, 2]


def test_cube_round_trip_and_history(cfg):
    import pandas as pd
    import xarray as xr

    rng = np.random.default_rng(0)
    dims = ("time", "transect", "distance")
    ds = xr.Dataset(
        {"green": (dims, rng.uniform(0, 0.5, (4, 3, 11)).astype("float32")),
         "swir1": (dims, rng.uniform(0, 0.5, (4, 3, 11)).astype("float32")),
         "qa": (dims, rng.integers(0, 255, (4, 3, 11)).astype("uint8"))},
        coords={"time": pd.date_range("2020-01-01", periods=4, freq="16D"), "distance": np.arange(11) * 10.0,
                "transect_id": ("transect", np.array(["a", "b", "c"])),
                "platform": ("time", np.array(["landsat-8"] * 4)), "family": ("time", np.array(["landsat"] * 4))},
    )
    ds["green"][0, 0, 0] = np.nan
    first = save_cube(cfg, "unit", ds, "first build")
    back = open_cube(cfg, "unit").load()
    assert back.attrs["snapshot_id"] == first
    assert np.isnan(back["green"].values[0, 0, 0])
    assert np.nanmax(np.abs(back["green"].values - ds["green"].values)) <= 5.1e-5      # stored as 1e-4 integers
    assert (back["qa"].values == ds["qa"].values).all()
    assert back["transect_id"].values.tolist() == ["a", "b", "c"] and back["family"].values[0] == "landsat"

    second = save_cube(cfg, "unit", ds.isel(time=slice(0, 2)), "rebuilt, shorter")
    assert second != first
    assert open_cube(cfg, "unit").sizes["time"] == 2
    assert open_cube(cfg, "unit", first).sizes["time"] == 4          # the earlier version is still readable
    with pytest.raises(FileNotFoundError):
        open_cube(cfg, "missing")
