"""Image evidence: frames along the coast, the lines in their pixels, segments, sheets and the packed archive."""
import json
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from ceva import frames as fr

UTM = 32630


def _coast(xs, sign=1):
    """Transects across a straight east-west coast, land to the south, 2 km long (x from the land end)."""
    lines = [LineString([(x, 0), (x, 2000)]) if sign > 0 else LineString([(x, 2000), (x, 0)]) for x in xs]
    return gpd.GeoDataFrame({"transect_id": [f"t{i}" for i in range(len(xs))], "seaward_sign": sign},
                            geometry=lines, crs=UTM)


def _waterlines(ids, x_obs=1050.0, x_corr=1000.0):
    rows = [{"transect_id": t, "time": pd.Timestamp("2020-01-01") + pd.Timedelta(days=k), "x_obs": x_obs + k,
             "x_corr": x_corr + k % 3, "use": True, "outlier": False} for t in ids for k in range(10)]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("sign", [1, -1])
def test_a_frame_has_the_sea_at_the_top_and_the_coast_running_right(sign):
    lines = _coast(range(0, 1001, 100), sign)
    frame = fr.frame_grid(lines, _waterlines(lines["transect_id"]))
    assert (frame.width, frame.height) == (120, 60)         # 1 km of transects + 2 x 100 m; +-300 m across
    assert fr.frame_grid(_coast([0, 100]), _waterlines(["t0", "t1"])).width == fr.MIN_ALONG_M / fr.PIXEL_M
    col, row = frame.pixel_of(np.array([0.0, 1000.0, 0.0]), np.array([1001.0, 1001.0, 1200.0]))
    assert col[1] > col[0]                                    # east is to the right: not mirrored
    assert row[2] < row[0] == pytest.approx(30, abs=0.5)      # seaward is up; the shoreline mid-height
    tl, tr, br, bl = frame.corners
    assert tl[1] > bl[1] and tr[0] > tl[0]


def test_the_frame_widens_where_the_waterlines_range_further_but_not_beyond_half_a_transect():
    lines = _coast([0, 100])
    wide = fr.frame_grid(lines, _waterlines(lines["transect_id"], x_obs=1600.0))
    assert wide.height * fr.PIXEL_M > 600 and wide.height * fr.PIXEL_M <= 2 * fr.MAX_HALF_WIDTH_M
    assert fr.frame_grid(lines, _waterlines(lines["transect_id"]).assign(use=False)) is None


def test_lines_are_given_in_the_frame_pixels_from_the_landward_end():
    lines = _coast([0, 100, 200], sign=-1)                    # stored sea to land: x still from the land end
    frame = fr.frame_grid(lines, _waterlines(lines["transect_id"]))
    t = fr.segment_lines(lines, frame)[1]
    x = 1001.0                                                # the long-term shoreline
    assert t["x0"] + x * t["dx"] == pytest.approx(30, abs=0.5)    # the middle of a 600 m frame
    assert t["y0"] + x * t["dy"] == pytest.approx(30, abs=0.5)
    assert t["dy"] == pytest.approx(-1 / fr.PIXEL_M) and t["dx"] == pytest.approx(0, abs=1e-6)


def test_segments_follow_a_curved_coast_and_break_at_gaps():
    # an L-shaped coast: 3 km east, then 3 km north, then a 4 km gap and 1 km more
    pts = [(x, 0) for x in range(0, 3001, 100)] + [(3000, y) for y in range(100, 3001, 100)]
    pts += [(3000, y) for y in range(7000, 8001, 100)]
    lines = gpd.GeoDataFrame({"transect_id": range(len(pts))},
                             geometry=[LineString([(x, y), (x + 1, y + 1)]) for x, y in pts], crs=UTM)
    groups = fr.coast_segments(lines.sample(frac=1, random_state=1).reset_index(drop=True))
    assert sum(len(g) for g in groups) == len(pts)
    assert max(len(g) for g in groups) <= 21 and len(groups) == 4   # 6 km of coast in 2 km pieces, then the island
    shuffled = lines.sample(frac=1, random_state=1).reset_index(drop=True)
    for g in fr.coast_segments(shuffled):                       # each piece is a stretch of coast, not scattered
        xy = np.array([shuffled.geometry.iloc[i].coords[0] for i in g])
        assert np.linalg.norm(np.diff(xy, axis=0), axis=1).max() <= 150


def test_render_uses_one_stretch_and_marks_missing_data():
    band = np.array([[0.0, 0.30, np.nan]], dtype="float32")
    out = fr.render(band, band, band)
    assert out[0, 0].tolist() == [0, 0, 0] and out[0, 1].tolist() == [255, 255, 255]
    assert out[0, 2].tolist() == [fr.NO_DATA] * 3


def test_the_archive_index_points_at_the_right_bytes(tmp_path):
    (tmp_path / "s.json").write_text(json.dumps({"corners": [[0, 0]] * 4, "time": ["2020-01-01T10:00"],
                                                 "transects": [{"id": "t0"}, {"id": "t1"}]}))
    (tmp_path / "s.mp4").write_bytes(b"video bytes")
    seg = {"name": "s", "area": "a", "label": "A beach", "frames": 10, "files": ["s.json", "s.mp4"]}
    index = fr.pack(tmp_path, "pack.zip", [seg], tmp_path / "index.json")
    raw = (tmp_path / "pack.zip").read_bytes()
    for file, (start, size) in index["segments"][0]["files"].items():
        assert raw[start:start + size] == (tmp_path / file).read_bytes()
    assert zipfile.ZipFile(tmp_path / "pack.zip").namelist() == seg["files"]   # an ordinary ZIP too
    assert index["segments"][0]["transects"] == ["t0", "t1"]                    # the map finds a transect's images
    assert index["segments"][0]["label"] == "A beach"


def test_units_are_packed_by_country_with_one_small_index_each(cfg, tmp_path, monkeypatch):
    work, dest = tmp_path / "work", tmp_path / "site"
    work.mkdir()
    for unit, seg in (("u1", "u1-1"), ("u2", "u2-1")):
        (work / f"{seg}.json").write_text(json.dumps({"corners": [[0, 0]] * 4, "time": ["2020-01-01T10:00"],
                                                     "transects": [{"id": f"{unit}t0"}]}))
        (work / f"{seg}.mp4").write_bytes(f"video of {seg}".encode())
        files = [f"{seg}.json", f"{seg}.mp4"]
        fr.mark_built(work, unit, unit, [{"name": seg, "area": unit, "frames": 1, "files": files}])
    assert fr.is_built(work, "u1") and not fr.is_built(work, "u3")
    monkeypatch.setattr(fr, "_countries", lambda cfg, ids: {"u1": "FR", "u2": "PT"})
    index = fr.pack_all(cfg, work, dest)
    assert sorted(index["archives"]) == ["frames-FR.zip", "frames-PT.zip"]
    assert index["units"]["u2"] == {"label": "u2", "country": "PT", "segments": 1, "frames": 1}
    seg = json.loads((dest / "units" / "u2.json").read_text())["segments"][0]
    assert seg["archive"] == "frames-PT.zip" and seg["transects"] == ["u2t0"]
    start, size = seg["files"]["u2-1.mp4"]
    assert (dest / "frames-PT.zip").read_bytes()[start:start + size] == b"video of u2-1"


def test_units_are_built_side_by_side_and_never_twice(tmp_path):
    import subprocess
    import sys

    from ceva.cli import _build_side_by_side, _building_elsewhere

    work, log = tmp_path / "work", tmp_path / "log"
    work.mkdir()
    finish = ("import json, pathlib, sys, time; time.sleep({wait}); "
              "pathlib.Path(sys.argv[1], sys.argv[-1].partition('=')[0] + '.done.json')"
              ".write_text(json.dumps({{'segments': [{{'frames': 3}}]}}))")
    noted = ("import sys, time; name, start = sys.argv[-1].partition('=')[0], time.time(); "
             + finish.format(wait=0.5) + "; open(sys.argv[2], 'a').write(f'{name} {start} {time.time()}\\n')")

    def command(name):                                 # a stand-in for one unit's process
        code = "raise SystemExit(2)" if name == "bad" else noted
        return [sys.executable, "-c", code, str(work), str(log), "frames", "build", f"{name}={name}"]

    (work / "u5.done.json").write_text("{}")                                   # built before
    other = subprocess.Popen([sys.executable, "-c", finish.format(wait=1), str(work), "frames", "build", "u3=u3"])
    try:
        assert "u3" in _building_elsewhere(set()) and "u3" not in _building_elsewhere({other.pid})
        done = _build_side_by_side(work, ["bad", "u1", "u2", "u3", "u4", "u5"], command, jobs=2, min_free_gb=0,
                                   poll=0.05)
    finally:
        other.wait()
    runs = {name: (float(start), float(end)) for name, start, end in map(str.split, log.read_text().splitlines())}
    assert done == 5 and sorted(runs) == ["u1", "u2", "u4"]          # u3 was another process's, u5 was built
    overlap = max(sum(s <= t < e for s, e in runs.values()) for t, _ in runs.values())
    assert overlap == 2                                               # two at a time, never more


def test_without_its_cube_a_unit_finds_its_scenes_grouped_by_satellite_and_day(cfg, tmp_path):
    import datetime as dt

    from ceva.catalog import Scene

    cfg = cfg.model_copy(update={"data_dir": tmp_path / "data"})
    utc = dt.UTC

    def scene(i, platform, when):
        return Scene(i, "pc", "sentinel2", platform, when, None, None, None, "s2_scl", {})

    morning = dt.datetime(2020, 5, 1, 10, 30, tzinfo=utc)
    scenes = [scene("a", "sentinel-2a", morning), scene("b", "sentinel-2a", morning + dt.timedelta(seconds=4)),
              scene("c", "sentinel-2b", morning + dt.timedelta(days=3))]
    catalogue = {s.id: s for s in scenes}
    times = pd.DatetimeIndex([pd.Timestamp("2020-05-01 10:30:00"), pd.Timestamp("2020-05-04 10:30:00")])
    waterlines = pd.DataFrame({"time": times, "platform": ["sentinel-2a", "sentinel-2b"]})
    found = fr._acquisitions(cfg, "no-cube-here", catalogue, waterlines, times)
    assert found == {times[0]: ["a", "b"], times[1]: ["c"]}         # two tiles of one pass are one acquisition
