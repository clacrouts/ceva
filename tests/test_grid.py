import mercantile
import numpy as np
import pytest

from ceva.grid import Tile, quadkeys, tiles_for_bbox, working_epsg

GIRONDE_BBOX = (-1.279822, 45.373822, -0.723581, 45.693203)


@pytest.mark.parametrize("zoom", [1, 6, 10, 12])
def test_quadkeys_match_mercantile(zoom):
    rng = np.random.default_rng(42)
    lon = rng.uniform(-179.9, 179.9, 400)
    lat = rng.uniform(-84.0, 84.0, 400)
    expected = [mercantile.quadkey(mercantile.tile(x, y, zoom)) for x, y in zip(lon, lat, strict=True)]
    assert quadkeys(lon, lat, zoom).tolist() == expected


def test_coarser_tile_is_a_prefix_of_the_finer_one():
    fine = quadkeys([-1.2, 151.2, -70.6], [45.6, -33.9, -33.4], 12)
    coarse = quadkeys([-1.2, 151.2, -70.6], [45.6, -33.9, -33.4], 10)
    assert all(f.startswith(c) for f, c in zip(fine, coarse, strict=True))


def test_extreme_coordinates_stay_in_range():
    keys = quadkeys([180.0, -180.0, 0.0, 0.0], [0.0, 0.0, 89.9, -89.9], 12)
    assert all(len(k) == 12 and set(k) <= set("0123") for k in keys)


def test_gironde_tiles_include_those_holding_its_transects():
    found = {t.quadkey for t in tiles_for_bbox(GIRONDE_BBOX, 10)}
    assert {"0313313302", "0313313303", "0313313320", "0313313321"} <= found


def test_tile_geometry():
    tile = Tile("0313313302")
    w, s, e, n = tile.bounds
    assert w < e and s < n
    assert tile.zoom == 10
    assert tile.epsg == 32630
    assert tile.index_range() == ("031331330200", "031331330233")
    bw, bs, be, bn = tile.buffered_bounds(2000)
    assert bw < w and bs < s and be > e and bn > n
    # 2 km is about 0.018 degrees of latitude
    assert s - bs == pytest.approx(2000 / 111_320, rel=1e-6)


def test_tile_rejects_bad_quadkeys():
    with pytest.raises(ValueError):
        Tile("0412")
    with pytest.raises(ValueError):
        Tile("")
    with pytest.raises(ValueError):
        Tile("0" * 13).index_range()


@pytest.mark.parametrize(
    ("lon", "lat", "epsg"),
    [
        (-1.0, 45.5, 32630),     # Gironde
        (151.2, -33.9, 32756),   # Sydney
        (-180.0, 10.0, 32601),
        (180.0, 10.0, 32660),
        (0.0, 88.0, 32661),      # UPS north
        (0.0, -85.0, 32761),     # UPS south
    ],
)
def test_working_epsg(lon, lat, epsg):
    assert working_epsg(lon, lat) == epsg


def test_bbox_across_the_antimeridian_gives_tiles_on_both_sides():
    tiles = tiles_for_bbox((179.5, -17.0, -179.5, -16.0), 8)
    wests = [t.bounds[0] for t in tiles]
    assert any(w > 170 for w in wests) and any(w < -170 for w in wests)
