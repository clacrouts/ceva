"""Chips read once per acquisition for all segments, in blocks: the same chips as segment by segment."""
import datetime as dt

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import LineString

import ceva.chips as chips
from ceva.catalog import SearchResult
from ceva.catalog.models import BandAsset, Scene

EPSG = 32630
X0, Y0 = 600_000.0, 4_960_000.0
BANDS = ("blue", "green", "red", "nir", "swir1")


def write_raster(path, values, res, dtype="uint16"):
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1], count=1,
                       dtype=dtype, crs=f"EPSG:{EPSG}", transform=from_origin(X0, Y0, res, res), nodata=0) as dst:
        dst.write(values.astype(dtype), 1)
    return str(path)


@pytest.fixture
def scene_files(tmp_path):
    """Twelve Sentinel-2 acquisitions over a 6 x 3 km area; each band a different gentle ramp."""
    rows, cols = np.mgrid[0:300, 0:600]
    scl = write_raster(tmp_path / "scl.tif", np.full((150, 300), 6), 20.0, dtype="uint8")
    scenes = []
    for i in range(12):
        bands = {}
        for j, band in enumerate(BANDS):
            values = 1000 + 100 * i + 10 * j + cols + rows / 2.0
            bands[band] = BandAsset(write_raster(tmp_path / f"a{i}_{band}.tif", values, 10.0), 1e-4, 0.0, 0.0)
        bands["mask"] = BandAsset(scl)
        when = dt.datetime(2020, 1, 1, 11, tzinfo=dt.UTC) + dt.timedelta(days=5 * i)
        scenes.append(Scene(f"S2A_MSIL2A_{when:%Y%m%dT%H%M%S}_R094_T30TXQ_X", "local", "sentinel2", "sentinel-2a",
                            when, "30TXQ", 1.0, "05.00", "s2_scl", bands))
    return scenes


def transects():
    to_ll = Transformer.from_crs(EPSG, 4326, always_xy=True)
    lines = []
    for x in (X0 + 800.0, X0 + 1300.0, X0 + 4500.0, X0 + 5000.0):     # two groups 3 km apart
        (lon0, lon1), (lat0, lat1) = to_ll.transform([x, x], [Y0 - 900.0, Y0 - 2100.0])
        lines.append(LineString([(lon0, lat0), (lon1, lat1)]))
    return gpd.GeoDataFrame({"transect_id": ["a", "b", "c", "d"]}, geometry=lines, crs="EPSG:4326")


def test_union_grid_and_windows():
    a = chips.ChipGrid(EPSG, 1000.0, 5000.0, 10.0, 30, 20)
    b = chips.ChipGrid(EPSG, 1200.0, 4900.0, 10.0, 40, 10)
    u = chips.union_grid([a, b])
    assert (u.x0, u.y0, u.width, u.height) == (1000.0, 5000.0, 60, 20)
    assert u.window_of(b) == (slice(10, 20), slice(20, 60))
    with pytest.raises(ValueError):
        a.window_of(b)


def test_one_read_per_acquisition_gives_the_chips_of_each_segment(cfg, scene_files, monkeypatch):
    pytest.importorskip("icechunk")
    monkeypatch.setattr(chips, "search_scenes", lambda *a, **k: SearchResult("local", scene_files, len(scene_files)))
    cfg = cfg.model_copy(update={"chips": cfg.chips.model_copy(update={"segment_km": 1.0, "workers": 4})})
    tr = transects()
    one_block = chips.build_chips(cfg, tr, families=("sentinel2",))
    monkeypatch.setattr(chips, "CHIP_BLOCK_BYTES", 1)                  # blocks of 8 acquisitions
    blocks = list(chips.iter_chips(cfg, tr, families=("sentinel2",)))
    assert [k for k, _ in blocks] == ["sentinel2/s000", "sentinel2/s001"] * 2
    assert sorted(one_block) == ["sentinel2/s000", "sentinel2/s001"]

    # the same pixels as a chip read on its own grid
    groups = chips.segments(tr, 1.0, EPSG)
    for k, rows in enumerate(groups):
        grid = chips.chip_grid(tr.iloc[rows], 10.0, cfg.chips.margin_m, EPSG)
        alone, _ = chips._read_acquisition([scene_files[3]], grid, BANDS, {})
        ds = one_block[f"sentinel2/s{k:03d}"]
        assert ds.sizes["time"] == 12 and ds.attrs["transect_ids"] == ",".join(tr.iloc[rows]["transect_id"])
        for band in BANDS:
            assert np.allclose(ds[band].isel(time=3).values, alone[band], equal_nan=True, atol=1e-6)

    chips.save_chips(cfg, "unit", iter(blocks), "blocks")
    stored = chips.open_chips(cfg.model_copy(update={"chips": cfg.chips.model_copy(
        update={"reflectance": {"landsat": "surface", "sentinel2": "surface"}})}), "unit")
    for key, ds in one_block.items():
        assert stored[key].sizes["time"] == 12
        assert (stored[key]["time"].values == ds["time"].values).all()
        assert np.allclose(stored[key]["green"].values, ds["green"].values, atol=1e-4, equal_nan=True)


def test_cloudy_acquisitions_are_neither_read_nor_stored(cfg, scene_files, tmp_path, monkeypatch):
    monkeypatch.setattr(chips, "search_scenes", lambda *a, **k: SearchResult("local", scene_files, len(scene_files)))
    cfg = cfg.model_copy(update={"chips": cfg.chips.model_copy(update={"segment_km": 1.0, "workers": 2})})
    cloudy_west = np.full((150, 300), 6)
    cloudy_west[:, :150] = 9                                  # cloud over the first segment only
    overcast = write_raster(tmp_path / "overcast.tif", np.full((150, 300), 9), 20.0, dtype="uint8")
    half = write_raster(tmp_path / "half.tif", cloudy_west, 20.0, dtype="uint8")
    scene_files[2].bands["mask"] = BandAsset(overcast)
    scene_files[5].bands["mask"] = BandAsset(half)
    reads = []
    original = chips.read_band
    monkeypatch.setattr(chips, "read_band", lambda path, *a, **k: reads.append(path) or original(path, *a, **k))
    got = chips.build_chips(cfg, transects(), families=("sentinel2",))
    times = {k: list(ds["time"].values) for k, ds in got.items()}
    overcast_time, half_time = (np.datetime64(scene_files[i].datetime.replace(tzinfo=None)) for i in (2, 5))
    assert overcast_time not in times["sentinel2/s000"] + times["sentinel2/s001"]     # never stored
    assert half_time not in times["sentinel2/s000"] and half_time in times["sentinel2/s001"]
    assert len(times["sentinel2/s000"]) == 10 and len(times["sentinel2/s001"]) == 11
    assert not any("a2_" in path for path in reads)           # its bands were never downloaded
    assert any("a5_" in path for path in reads)

    everything = chips.build_chips(cfg.model_copy(update={"chips": cfg.chips.model_copy(
        update={"skip_cloudy": False})}), transects(), families=("sentinel2",))
    assert all(ds.sizes["time"] == 12 for ds in everything.values())


def test_units_on_one_tile_download_each_level1_file_once(cfg, scene_files, tmp_path, monkeypatch):
    """Two units over the same Sentinel-2 files: the same chips as read apart, each Level-1 file fetched once."""
    import shutil
    from dataclasses import replace

    twins = []
    for s in scene_files:                       # Level-1 twins: same acquisition and tile, their own band files
        bands = {}
        for band, asset in s.bands.items():
            if band == "mask":
                continue
            copy = tmp_path / ("l1c_" + asset.href.split("/")[-1])
            shutil.copy(asset.href, copy)
            bands[band] = BandAsset(str(copy), asset.scale, asset.offset, asset.nodata)
        twins.append(replace(s, id=s.id.replace("MSIL2A", "MSIL1C"), mask_kind="none", bands=bands))

    def search(cfg, family, bbox, start, end, creds=None, level="surface", provider=None):
        found = twins if level == "toa" else scene_files
        return SearchResult("local", found, len(found))

    monkeypatch.setattr(chips, "search_scenes", search)
    cfg = cfg.model_copy(update={"chips": cfg.chips.model_copy(update={"segment_km": 1.0, "workers": 3}),
                                 "data_dir": tmp_path / "data"})
    tr = transects()
    units = {"west": tr.iloc[:2], "east": tr.iloc[2:]}
    downloads, remote = [], []
    original_download, original_read = chips._download, chips.read_band
    monkeypatch.setattr(chips, "_download", lambda href, *a: downloads.append(href) or original_download(href, *a))
    monkeypatch.setattr(chips, "read_band", lambda path, *a, **k: remote.append(path) or original_read(path, *a, **k))

    apart = {name: dict(chips.iter_chips(cfg, t, families=("sentinel2",), level="toa")) for name, t in units.items()}
    assert not downloads                        # one unit alone reads windows of the remote files
    l1c_reads_apart = sum(path.split("/")[-1].startswith("l1c_") for path in remote)
    remote.clear()
    together: dict = {}
    for name, key, ds in chips.iter_chips_batch(cfg, units, families=("sentinel2",), level="toa"):
        together.setdefault(name, {})[key] = ds
    assert sorted(downloads) == sorted(set(downloads)) and len(downloads) == 12 * len(BANDS)   # each file once
    level1 = {asset.href for twin in twins for asset in twin.bands.values()}
    assert not level1 & set(remote)             # no Level-1 file read remotely: all from the copies
    assert l1c_reads_apart == 2 * 12 * len(BANDS)
    for name in units:
        assert sorted(together[name]) == sorted(apart[name])
        for key, ds in apart[name].items():
            assert ds.identical(together[name][key])
    assert not any((tmp_path / "data" / "tmp").iterdir())      # the copies are gone

    # sharing through the cache instead: nothing downloaded, the same chips
    downloads.clear()
    monkeypatch.setattr(chips, "_share_policy", lambda cfg, provider: "cache")
    cached: dict = {}
    for name, key, ds in chips.iter_chips_batch(cfg, units, families=("sentinel2",), level="toa"):
        cached.setdefault(name, {})[key] = ds
    assert not downloads
    assert all(ds.identical(cached[name][key]) for name in units for key, ds in apart[name].items())
