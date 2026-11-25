"""Chips and the CoastSat engine, on small local rasters and synthetic beaches."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import LineString

from ceva.chips import ChipGrid, chip_grid, read_band
from ceva.engines_coastsat import (
    cloud_mask,
    oriented_transects,
    reference_lines,
    satname,
)

EPSG = 32630
X0, Y0 = 600_000.0, 4_960_000.0


def lonlat(x, y):
    return Transformer.from_crs(EPSG, 4326, always_xy=True).transform(x, y)


def write_raster(path, values, res, dtype="uint16", nodata=0):
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1], count=1,
                       dtype=dtype, crs=f"EPSG:{EPSG}", transform=from_origin(X0, Y0, res, res),
                       nodata=nodata) as dst:
        dst.write(values.astype(dtype), 1)
    return str(path)


# ------------------------------------------------------------------------------- chips

def test_chip_grid_covers_the_transects_on_pixel_multiples():
    tr = gpd.GeoDataFrame(geometry=[LineString([lonlat(X0 + 1003, Y0 - 2007), lonlat(X0 + 1817, Y0 - 1211)])],
                          crs="EPSG:4326")
    grid = chip_grid(tr, 15.0, 100.0, epsg=EPSG)
    left, bottom, right, top = grid.bounds
    assert left <= X0 + 1003 - 100 and right >= X0 + 1817 + 100
    assert bottom <= Y0 - 2007 - 100 and top >= Y0 - 1211 + 100
    assert left % 15 == 0 and top % 15 == 0
    assert np.isclose(grid.x[0], left + 7.5) and np.isclose(grid.y[0], top - 7.5)
    # CoastSat's georef maps pixel index (row, col) = (0, 0) to the centre of the first pixel
    assert np.allclose(grid.georef[[0, 3]], [grid.x[0], grid.y[0]])


def test_read_band_resamples_a_plane_exactly_and_marks_missing_data(tmp_path):
    res = 30.0
    cols = (np.arange(200) + 0.5) * res
    values = np.broadcast_to(1000 + cols / 10.0, (200, 200)).copy()
    values[150:, :] = 0                                     # no data in the south
    path = write_raster(tmp_path / "plane.tif", values, res, dtype="float32")
    grid = ChipGrid(EPSG, X0 + 1500.0, Y0 - 1500.0, 15.0, 80, 120)   # straddles the no-data edge
    out = read_band(path, grid, reflectance=True)
    expected = 1000 + (grid.x - X0) / 10.0
    inside = grid.y > Y0 - 150 * res + res                  # rows whose bilinear support is all valid
    assert np.allclose(out[inside], np.broadcast_to(expected, out.shape)[inside], atol=1e-3)
    assert np.isnan(out[grid.y < Y0 - 150 * res - res]).all()


def test_read_band_takes_the_nearest_flag_and_zero_outside(tmp_path):
    flags = np.zeros((40, 40), dtype="uint8")
    flags[:, 20:] = 9
    path = write_raster(tmp_path / "scl.tif", flags, 20.0, dtype="uint8", nodata=None)
    grid = ChipGrid(EPSG, X0 + 300.0, Y0 - 100.0, 10.0, 80, 10)       # runs past the east edge
    out = read_band(path, grid, reflectance=False)
    assert out.dtype == np.uint8
    assert set(np.unique(out[:, grid.x < X0 + 400 - 10])) == {0}
    assert set(np.unique(out[:, (grid.x > X0 + 400 + 10) & (grid.x < X0 + 800 - 10)])) == {9}
    assert set(np.unique(out[:, grid.x > X0 + 800 + 10])) == {0}


# ---------------------------------------------------------------------- transects, refs

def test_transects_start_at_their_landward_end():
    t = oriented_transects(["a", "b", "c"], [0.0, 0.0, 0.0], [45.0, 45.01, 45.02], [-0.01, -0.01, -0.01],
                           [45.0, 45.01, 45.02], [1, -1, 0], EPSG)
    assert list(t.transect_id) == ["a", "b"]                # no seaward direction: dropped
    assert t.origin[0, 0] > t.end[0, 0]                     # a: start (east) is landward
    assert t.origin[1, 0] < t.end[1, 0]                     # b: end (west) is landward


def test_reference_lines_follow_the_annual_positions_and_extend_past_the_ends():
    xs = X0 + np.array([1000.0, 1000.0, 1000.0])
    ys = Y0 - np.array([1000.0, 1100.0, 1200.0])
    to_ll = Transformer.from_crs(EPSG, 4326, always_xy=True)
    lon_s, lat_s = to_ll.transform(xs + 400, ys)
    lon_e, lat_e = to_ll.transform(xs - 400, ys)
    t = oriented_transects(["n", "m", "s"], lon_s, lat_s, lon_e, lat_e, [1, 1, 1], EPSG)
    annual = pd.DataFrame({"transect_id": ["n", "m", "s", "n", "m"], "year": [2000, 2000, 2000, 2002, 2002],
                           "x": [400.0, 410.0, 420.0, 300.0, 310.0]})
    refs = reference_lines(t, annual, np.array([2000, 2001, 2002, 2010]), extend_m=50.0)
    line = refs[2000]
    assert line.shape == (5, 2)                             # three points plus one extension at each end
    inner = line[1:-1]
    order = np.argsort(inner[:, 1])[::-1]
    assert np.allclose(inner[order, 0], xs + 400 - np.array([400.0, 410.0, 420.0]), atol=0.5)
    assert np.isclose(np.linalg.norm(line[0] - line[1]), 50.0)
    # 2001: halfway between 2000 and 2002; transect s has no 2002 value and keeps 420
    mid = refs[2001][1:-1]
    assert np.allclose(sorted(xs + 400 - mid[:, 0]), [350.0, 360.0, 420.0], atol=0.5)
    assert np.allclose(refs[2010], refs[2002])             # after the last year: the last year's line


def test_satname():
    assert [satname(p) for p in ("landsat-4", "landsat-5", "landsat-7", "landsat-8", "landsat-9",
                                 "sentinel-2a", "sentinel-2c")] == ["L5", "L5", "L7", "L8", "L9", "S2", "S2"]


# ------------------------------------------------------------------------- cloud mask

def test_cloud_mask_drops_specks_and_thin_lines_but_keeps_clouds(cfg):
    qa = np.full((100, 100), 6, dtype="uint8")              # water
    qa[10:12, 10:12] = 9                                    # a speck: not a cloud
    qa[40:42, :] = 9                                        # a thin line along the beach: not a cloud
    qa[60:90, 60:90] = 9                                    # a cloud
    mask = cloud_mask(qa, "sentinel2", cfg)
    assert not mask[10:12, 10:12].any() and not mask[40:42, :].any()
    assert mask[65:85, 65:85].all()
    landsat = np.zeros((50, 50), dtype="uint8")
    landsat[5:25, 5:25] = 1 << 3                            # cloud bit
    landsat[30:40, 30:40] = 1 << 4                          # shadow bit: CoastSat keeps shadows
    m = cloud_mask(landsat, "landsat", cfg)
    assert m[5:25, 5:25].all() and not m[30:40, 30:40].any()


# ------------------------------------------------------------------ synthetic detection

WATER = (0.05, 0.04, 0.02, 0.01, 0.005)                    # blue, green, red, nir, swir1
SAND = (0.20, 0.25, 0.30, 0.35, 0.40)
VEGETATION = (0.03, 0.06, 0.04, 0.30, 0.15)


def synthetic_chips(shore_x: float, n_images: int = 3, pixel: float = 15.0):
    """Water to the west, sand to the east of ``shore_x``, vegetation behind; north-south coast."""
    grid = ChipGrid(EPSG, X0, Y0, pixel, 60, 120)
    rng = np.random.default_rng(0)
    frac = np.clip((grid.x - (shore_x - pixel / 2)) / pixel, 0, 1)          # sub-pixel mixing at the edge
    veg = grid.x > shore_x + 300
    bands = {}
    for k, name in enumerate(("blue", "green", "red", "nir", "swir1")):
        row = WATER[k] * (1 - frac) + SAND[k] * frac
        row = np.where(veg, VEGETATION[k], row)
        img = np.broadcast_to(row, (grid.height, grid.width))
        bands[name] = (("time", "y", "x"), np.stack([img + rng.normal(0, 0.002, img.shape) for _ in range(n_images)]))
    bands["qa"] = (("time", "y", "x"), np.zeros((n_images, grid.height, grid.width), dtype="uint8"))
    ds = xr.Dataset(bands, coords={
        "time": pd.date_range("2020-06-01", periods=n_images, freq="16D"),
        "platform": ("time", ["landsat-8"] * n_images), "family": ("time", ["landsat"] * n_images),
        "x": grid.x, "y": grid.y,
    }, attrs=grid.to_attrs())
    for name in ("blue", "green", "red", "nir", "swir1"):
        ds[name] = ds[name].astype("float32")
    return ds, grid


def test_coastsat_engine_finds_a_synthetic_shoreline(cfg):
    pytest.importorskip("coastsat")
    from ceva.engines_coastsat import coastsat_waterlines

    shore_x = X0 + 400.0
    ds, grid = synthetic_chips(shore_x)
    ys = Y0 - np.array([600.0, 900.0, 1200.0])
    to_ll = Transformer.from_crs(EPSG, 4326, always_xy=True)
    lon_s, lat_s = to_ll.transform(np.full(3, X0 + 700.0), ys)     # landward start, 300 m east of the shore
    lon_e, lat_e = to_ll.transform(np.full(3, X0 + 100.0), ys)
    unit = pd.DataFrame({"transect_id": ["a", "b", "c"], "lon_start": lon_s, "lat_start": lat_s,
                         "lon_end": lon_e, "lat_end": lat_e, "seaward_sign": [1, 1, 1]})
    annual = pd.DataFrame({"transect_id": ["a", "b", "c"], "year": 2020, "x": [280.0, 300.0, 320.0]})
    w, diagnostics = coastsat_waterlines(cfg, {"landsat": ds}, unit, annual,
                                         lambda times: np.full((len(times), 3), 0.5))
    assert (diagnostics["n_points"] > 0).all()
    assert set(w["transect_id"]) == {"a", "b", "c"} and len(w) == 9
    assert np.allclose(w["x_obs"], 300.0, atol=grid.pixel_m)   # the edge is 300 m from the landward start
    assert np.allclose(w["tide"], 0.5)
    assert w["threshold"].notna().all()


def test_coastsat_detections_are_reused_across_water_level_variants(cfg):
    from ceva.pipeline import _coastsat_cached

    annual = pd.DataFrame({"transect_id": ["a"], "year": [2020], "x": [300.0]})
    calls = []

    def compute(segment_cache):
        calls.append(1)
        return (pd.DataFrame({"transect_id": ["a"], "time": [pd.Timestamp("2020-06-01")], "x_obs": [301.0]}),
                pd.DataFrame({"time": [pd.Timestamp("2020-06-01")], "method": ["sand/water"]}))

    first, _ = _coastsat_cached(cfg, "unit", {"landsat": "S1"}, annual, compute)
    again, diag = _coastsat_cached(cfg, "unit", {"landsat": "S1"}, annual, compute)
    assert len(calls) == 1 and again.equals(first) and list(diag["method"]) == ["sand/water"]
    _coastsat_cached(cfg, "unit", {"landsat": "S2"}, annual, compute)     # new chips: computed again
    moved = annual.assign(x=[320.0])                                       # new reference lines: again
    _coastsat_cached(cfg, "unit", {"landsat": "S1"}, moved, compute)
    assert len(calls) == 3


def test_half_pixel_error_of_published_coastsat_depends_on_beach_orientation():
    from ceva.validation.benchmark import half_pixel_error, seaward_directions

    to_ll = Transformer.from_crs(EPSG, 4326, always_xy=True)
    west = LineString([to_ll.transform(X0 + 400, Y0), to_ll.transform(X0 - 400, Y0)])    # starts east, runs west
    east = LineString([to_ll.transform(X0 - 400, Y0), to_ll.transform(X0 + 400, Y0)])
    tr = gpd.GeoDataFrame({"transect_id": ["w", "e", "flat"], "seaward_sign": [1, 1, 0]},
                          geometry=[west, east, west], crs="EPSG:4326")
    seaward = seaward_directions(tr)
    assert set(seaward) == {"w", "e"}
    assert np.allclose(seaward["w"], (-1, 0), atol=1e-3) and np.allclose(seaward["e"], (1, 0), atol=1e-3)
    err = half_pixel_error(seaward, "landsat")      # published points sit 7.5 m west and 7.5 m north
    assert np.isclose(err["w"], 7.5, atol=0.01)     # west-facing beach: published line too far seaward
    assert np.isclose(err["e"], -7.5, atol=0.01)    # east-facing beach: too far landward
    assert np.isclose(half_pixel_error({"n": (0.0, 1.0)}, "sentinel2")["n"], 5.0)


def test_segments_cut_a_long_coast_into_short_consecutive_groups():
    from ceva.chips import segments

    to_ll = Transformer.from_crs(EPSG, 4326, always_xy=True)
    ys = Y0 - np.arange(0, 10_000, 500.0)                        # 20 transects along 9.5 km of coast
    rng = np.random.default_rng(3)
    order = rng.permutation(len(ys))                              # stored in no particular order
    lines = [LineString([to_ll.transform(X0 + 400, y), to_ll.transform(X0 - 400, y)]) for y in ys[order]]
    tr = gpd.GeoDataFrame({"transect_id": [f"t{i}" for i in order]}, geometry=lines, crs="EPSG:4326")
    groups = segments(tr, 3.0, EPSG)
    assert sorted(np.concatenate(groups).tolist()) == list(range(len(ys)))
    for g in groups:
        span = np.ptp(ys[order][g])
        assert span <= 3000.0 + 1e-6
    assert len(groups) == 4                                        # 0-3 km, 3.5-6.5, 7-9.5 ... as consecutive runs


def test_parallel_detection_gives_the_serial_result(cfg):
    pytest.importorskip("coastsat")
    from ceva.engines_coastsat import detect_shorelines, oriented_transects, reference_lines

    ds, grid = synthetic_chips(X0 + 400.0, n_images=6)
    ys = Y0 - np.array([600.0, 1200.0])
    to_ll = Transformer.from_crs(EPSG, 4326, always_xy=True)
    lon_s, lat_s = to_ll.transform(np.full(2, X0 + 700.0), ys)
    lon_e, lat_e = to_ll.transform(np.full(2, X0 + 100.0), ys)
    t = oriented_transects(["a", "b"], lon_s, lat_s, lon_e, lat_e, [1, 1], EPSG)
    refs = reference_lines(t, pd.DataFrame({"transect_id": ["a", "b"], "year": 2020, "x": [300.0, 300.0]}),
                           np.array([2020]), 300.0)
    serial = detect_shorelines(cfg, ds, refs, workers=1)
    parallel = detect_shorelines(cfg, ds, refs, workers=2, chunk=2)
    assert list(serial["time"]) == list(parallel["time"])
    assert np.allclose(serial["threshold"], parallel["threshold"], equal_nan=True)
    for a, b in zip(serial["shoreline"], parallel["shoreline"], strict=True):
        assert np.allclose(a, b)


def test_segmented_chips_round_trip(cfg):
    pytest.importorskip("icechunk")
    from ceva.chips import open_chips, save_chips

    ds, _ = synthetic_chips(X0 + 400.0, n_images=2)
    a = ds.assign_attrs(transect_ids="a,b", family="landsat", segment=0)
    b = ds.assign_attrs(transect_ids="c", family="landsat", segment=1)
    save_chips(cfg, "unit", iter([("landsat/s000", a), ("landsat/s001", b)]), "test")
    chips = open_chips(cfg, "unit")
    assert list(chips) == ["landsat/s000", "landsat/s001"]
    assert chips["landsat/s001"].attrs["transect_ids"] == "c" and chips["landsat/s000"].attrs["family"] == "landsat"
    assert np.allclose(chips["landsat/s000"]["blue"].values, ds["blue"].values, atol=1e-4)


def test_coastsat_waterlines_resume_from_finished_segments(cfg, tmp_path, monkeypatch):
    pytest.importorskip("coastsat")
    import ceva.engines_coastsat as engine

    ds, _ = synthetic_chips(X0 + 400.0, n_images=3)
    ys = Y0 - np.array([600.0, 900.0])
    to_ll = Transformer.from_crs(EPSG, 4326, always_xy=True)
    lon_s, lat_s = to_ll.transform(np.full(2, X0 + 700.0), ys)
    lon_e, lat_e = to_ll.transform(np.full(2, X0 + 100.0), ys)
    unit = pd.DataFrame({"transect_id": ["a", "b"], "lon_start": lon_s, "lat_start": lat_s,
                         "lon_end": lon_e, "lat_end": lat_e, "seaward_sign": [1, 1]})
    annual = pd.DataFrame({"transect_id": ["a", "b"], "year": 2020, "x": [300.0, 300.0]})
    tide = lambda times: np.zeros((len(times), 2))                             # noqa: E731
    chips = {"landsat/s000": ds.assign_attrs(transect_ids="a", family="landsat"),
             "landsat/s001": ds.assign_attrs(transect_ids="b", family="landsat")}
    first, _ = engine.coastsat_waterlines(cfg, chips, unit, annual, tide, tmp_path / "segments")
    assert sorted(p.name for p in (tmp_path / "segments").iterdir()) == ["landsat_s000", "landsat_s001"]

    def fail(*args, **kwargs):
        raise AssertionError("a finished segment was detected again")

    monkeypatch.setattr(engine, "detect_shorelines", fail)
    again, _ = engine.coastsat_waterlines(cfg, chips, unit, annual, tide, tmp_path / "segments")
    assert again.equals(first)
