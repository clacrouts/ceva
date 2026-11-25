"""The faster replacements give exactly what they replace."""
import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import map_coordinates, median_filter

from ceva.chips import ChipGrid
from ceva.engines_coastsat import calculate_features, process_shoreline, shoreline_buffer, too_cloudy
from ceva.sar_waterlines import _sampled_pixels, _smoothed_at

SDS_shoreline = pytest.importorskip("coastsat.SDS_shoreline")
EPSG = 32630
GRID = ChipGrid(EPSG, 600_000.0, 4_960_000.0, 10.0, 180, 140)


def settings(**extra):
    return {"output_epsg": EPSG, "max_dist_ref": 150, "min_length_sl": 200, "dist_clouds": 300, **extra}


def wiggly_line(rng, n=12, margin=-200.0):
    """A reference line across the grid, partly beyond its edges."""
    x = np.linspace(GRID.x0 + margin, GRID.x0 + GRID.width * 10 - margin, n)
    y = GRID.y0 - 700 + np.cumsum(rng.normal(0, 60, n))
    return np.column_stack([x, y])


def test_shoreline_buffer_is_coastsats(cfg):
    rng = np.random.default_rng(0)
    lines = [wiggly_line(rng) for _ in range(6)]
    lines += [np.array([[GRID.x0 + 300, GRID.y0 - 300]] * 2),                        # a single point
              np.array([[GRID.x0 - 5000, GRID.y0 + 5000], [GRID.x0 - 4000, GRID.y0 + 4000]])]   # outside
    for line in lines:
        for dist in (100, 150, 155):
            s = settings(reference_shoreline=line, max_dist_ref=dist)
            expected = SDS_shoreline.create_shoreline_buffer((GRID.height, GRID.width), GRID.georef, EPSG, 10, s)
            got = shoreline_buffer((GRID.height, GRID.width), GRID.georef, EPSG, 10, s)
            assert np.array_equal(got, expected)
    assert shoreline_buffer((5, 4), GRID.georef, EPSG, 10, settings()).all()          # no reference: everywhere


def test_process_shoreline_is_coastsats():
    rng = np.random.default_rng(1)
    for case in range(5):
        contours = []
        for _ in range(4):                       # contours in pixel coordinates (row, column), some short
            n = int(rng.integers(3, 120))
            rows = 20 + np.cumsum(rng.uniform(-0.5, 1.0, n)) + case
            cols = rng.uniform(0, 1) * 100 + np.cumsum(rng.uniform(0.2, 1.0, n))
            contours.append(np.column_stack([rows, cols]))
        cloud = np.zeros((GRID.height, GRID.width), bool)
        for _ in range(case):                    # case 0: no cloud at all
            r, c = rng.integers(0, GRID.height), rng.integers(0, GRID.width)
            cloud[max(r - 8, 0):r + 8, max(c - 8, 0):c + 8] = True
        nodata = np.zeros_like(cloud)
        nodata[:, :case * 3] = True
        s = settings()
        expected = SDS_shoreline.process_shoreline(contours, cloud, nodata, GRID.georef, EPSG, s)
        got = process_shoreline(contours, cloud, nodata, GRID.georef, EPSG, s)
        assert got.shape == expected.shape and np.array_equal(got, expected)


def test_far_from_decides_at_the_limit_like_coastsat():
    from ceva.engines_coastsat import _far_from

    others = np.array([[0.0, 0.0], [10.0, 0.0]])
    points = np.array([[0.0, 30.0], [0.0, 29.999999999], [5.0, 3.0], [100.0, 100.0]])
    expected = [not np.any(np.linalg.norm(p - others, axis=1) < 30) for p in points]
    assert _far_from(points, others, 30).tolist() == expected == [True, False, False, True]
    assert _far_from(points, np.empty((0, 2)), 30).all()


def test_calculate_features_is_coastsats():
    rng = np.random.default_rng(2)
    im = rng.uniform(0, 0.4, (40, 50, 5))
    im[3:6, 7:9, :] = 0.0                       # flat patch: zero standard deviation
    cloud = np.zeros((40, 50), bool)
    cloud[10:14, 20:30] = True
    where = np.ones((40, 50), bool)
    expected = SDS_shoreline.calculate_features(im, cloud, where)
    got = calculate_features(im, cloud, where)
    assert got.dtype == expected.dtype and np.array_equal(got, expected, equal_nan=True)


@pytest.mark.parametrize("size", [1, 3, 5, 4])
def test_radar_smoothing_at_the_sampled_pixels_is_the_whole_image_filter(size):
    rng = np.random.default_rng(3)
    vv = rng.normal(-12, 3, (60, 70)).astype("float64")
    vh = rng.normal(-20, 3, (60, 70)).astype("float64")
    vv[20:25, 30:40] = np.nan                     # a gap, filled with the image median by the filter
    vv[0, :5] = np.nan
    row = np.concatenate([rng.uniform(-2, 61, 200), [0.0, 59.0, 3.0]])     # some beyond the edges
    col = np.concatenate([rng.uniform(-2, 71, 200), [0.0, 69.0, 7.0]])
    mean_db = (vv + vh) / 2.0
    valid = np.isfinite(mean_db)
    filled = np.where(valid, mean_db, np.nanmedian(mean_db))
    smooth = median_filter(filled, size=size) if size > 1 else filled
    smooth[~valid] = np.nan
    expected = map_coordinates(smooth, [row, col], order=1, mode="constant", cval=np.nan)
    pixels = _sampled_pixels(row, col, vv.shape)
    got = map_coordinates(_smoothed_at(vv, vh, pixels, size), [row, col], order=1, mode="constant", cval=np.nan)
    assert np.array_equal(got, expected, equal_nan=True)


def test_tide_table_models_each_time_once(cfg, monkeypatch):
    import ceva.tides as tides

    calls = []

    def fake(cfg, lon, lat, times):
        calls.append(len(times))
        hours = (pd.DatetimeIndex(times).asi8 // 3_600_000_000_000).astype(float)
        return (np.sin(hours)[:, None] + lon[None, :]).astype("float32")

    monkeypatch.setattr(tides, "tide_heights", fake)
    lon, lat = np.array([1.0, 2.0, 3.0]), np.zeros(3)
    table = tides.TideTable(cfg, lon, lat)
    first = pd.date_range("2020-01-01", periods=10, freq="7h")
    table.prefetch(first.append(first))
    shuffled = first[[3, 1, 1, 9]]
    assert np.array_equal(table(shuffled), fake(cfg, lon, lat, shuffled)) and calls == [10, 4]
    later = pd.DatetimeIndex(["2019-06-01 12:00", "2020-01-01 07:00"])        # one new time, one known
    assert np.array_equal(table(later), fake(cfg, lon, lat, later))
    assert calls == [10, 4, 1, 2]


def test_too_cloudy_reads_the_quality_band_like_the_detection(cfg):
    clear = np.full((60, 60), 6, dtype="uint8")             # Sentinel-2 scene class 6: water
    assert not too_cloudy(clear, "sentinel2", cfg)
    cloudy = clear.copy()
    cloudy[:40] = 9                                         # cloud, high probability: two thirds of the chip
    assert too_cloudy(cloudy, "sentinel2", cfg)
    empty = np.zeros_like(clear)                            # no data at all
    assert too_cloudy(empty, "sentinel2", cfg)
    border = clear.copy()
    border[:, :30] = 0                                      # half no data, the rest clear
    assert not too_cloudy(border, "sentinel2", cfg)


def test_one_detection_pool_serves_every_segment():
    from ceva.engines_coastsat import detection_pool

    pool = detection_pool(2)
    assert detection_pool(2) is pool                        # reused: processes start once per run
    other = detection_pool(3)
    assert other is not pool and detection_pool(3) is other
    other.shutdown()
    assert detection_pool(3) is not other                   # a pool shut down is replaced


def test_periodogram_is_scipys_lombscargle():
    from scipy.signal import lombscargle

    from ceva.engines import SPECTRAL_FREQ, _Periodogram

    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(0, 9000, 500))
    omega = 2 * np.pi * SPECTRAL_FREQ
    periodogram = _Periodogram(t, omega)
    for _ in range(3):
        y = rng.normal(0, 5, 500)
        assert np.array_equal(periodogram(y), lombscargle(t, y, omega))


def test_a_segment_without_any_waterline_merges_cleanly(cfg):
    from ceva.engines_coastsat import intersect, oriented_transects

    t = oriented_transects(pd.Series(["a"]), pd.Series([-1.25]), pd.Series([44.70]), pd.Series([-1.26]),
                           pd.Series([44.70]), pd.Series([1]), EPSG)
    detections = pd.DataFrame({"time": pd.DatetimeIndex([]), "shoreline": [], "platform": [], "family": [],
                               "threshold": [], "method": []})
    hits = intersect(cfg, detections, t, EPSG).merge(detections[["time", "platform", "family", "threshold", "method"]],
                                                     on="time")
    assert hits.empty and str(hits["time"].dtype).startswith("datetime64")
