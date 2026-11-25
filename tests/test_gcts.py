import geopandas as gpd
import pyarrow.parquet as pq
import pytest
from shapely.geometry import LineString

from ceva.grid import Tile, quadkeys
from ceva.transects import gcts

# (transect_id, lon, lat): three in the Gironde box (two tiles), one far away.
POINTS = [
    ("cl1s00tr00000300", -1.20, 45.60),
    ("cl1s00tr00000100", -1.19, 45.61),
    ("cl1s00tr00000200", -0.80, 45.40),
    ("cl2s00tr00000100", 2.00, 51.00),
]


def _write_raw(cfg, continent="Europe"):
    ids, lons, lats = zip(*POINTS, strict=True)
    gdf = gpd.GeoDataFrame(
        {
            "transect_id": ids, "lon": lons, "lat": lats, "bearing": 270.0, "utm_epsg": 32630,
            "continent": "EU", "country": "FR", "common_country_name": "France",
            "common_region_name": "Nouvelle-Aquitaine", "osm_coastline_is_closed": False,
            "osm_coastline_length": 1000,
            # GCTS ships a quadkey column that is not the tile of the transect.
            "quadkey": "020000111111",
        },
        geometry=[LineString([(x - 0.01, y), (x + 0.01, y)]) for x, y in zip(lons, lats, strict=True)],
        crs="EPSG:4326",
    )
    path = gcts.raw_path(cfg, continent)
    path.parent.mkdir(parents=True)
    gdf.to_parquet(path)
    return path


def test_index_is_sorted_by_origin_quadkey_and_drops_the_source_quadkey(cfg):
    _write_raw(cfg)
    index = gcts.build_index(cfg, "Europe")
    table = pq.read_table(index)
    keys = table.column(gcts.INDEX_KEY).to_pylist()
    assert keys == sorted(keys)
    assert "quadkey" not in table.column_names
    assert sorted(keys) == sorted(quadkeys([p[1] for p in POINTS], [p[2] for p in POINTS]).tolist())


def test_load_tile_returns_only_that_tile(cfg):
    _write_raw(cfg)
    gcts.build_index(cfg, "Europe")
    tile = Tile(quadkeys([-1.20], [45.60], cfg.grid.processing_zoom)[0])
    gdf = gcts.load_tile(cfg, tile)
    assert sorted(gdf["transect_id"]) == ["cl1s00tr00000100", "cl1s00tr00000300"]
    assert set(gdf["tile"]) == {tile.quadkey}
    assert gdf.crs.to_epsg() == 4326
    assert gdf.geometry.geom_type.eq("LineString").all()


def test_load_bbox_keeps_region_transects_and_tags_tiles(cfg):
    _write_raw(cfg)
    gcts.build_index(cfg, "Europe")
    gdf = gcts.load_bbox(cfg, cfg.region.bbox_wgs84)
    assert len(gdf) == 3
    assert gdf["tile"].nunique() == 2
    assert "cl2s00tr00000100" not in set(gdf["transect_id"])


def test_empty_tile_gives_an_empty_frame(cfg):
    _write_raw(cfg)
    gcts.build_index(cfg, "Europe")
    assert gcts.load_tile(cfg, Tile("2222222222")).empty


def test_missing_index_explains_what_to_run(cfg):
    with pytest.raises(FileNotFoundError, match="ceva transects prepare"):
        gcts.load_tile(cfg, Tile("0313313302"))


def test_unknown_continent(cfg):
    with pytest.raises(ValueError, match="Europe"):
        gcts.build_index(cfg, "Atlantis")
