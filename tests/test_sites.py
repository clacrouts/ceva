"""Survey loaders and benchmark scoring on small synthetic files."""
import json

import numpy as np
import pandas as pd
import pytest
from pyproj import Geod

from ceva.validation import benchmark, narrabeen


def write_narrabeen(cfg, shoreline_chainage=(80.0, 90.0)):
    folder = narrabeen.raw_dir(cfg)
    folder.mkdir(parents=True)
    rows = []
    for date, c0 in zip(("2010-01-05", "2010-02-03"), shoreline_chainage, strict=True):
        for chainage in range(0, 200, 10):
            rows.append({"Site": "NARRA", "Profile ID": "PF1", "Date": date, "Chainage": chainage,
                         "Elevation": 0.1 * (c0 - chainage), "Flag": "GPS"})
    pd.DataFrame(rows).to_csv(folder / "Narrabeen_Profiles.csv", index=False)
    lon1, lat1, _ = Geod(ellps="WGS84").fwd(151.30, -33.70, 110.0, 300.0)
    geo = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"name": "PF1"},
           "geometry": {"type": "LineString", "coordinates": [[151.30, -33.70], [lon1, lat1]]}}]}
    (folder / "NARRABEEN_transects.geojson").write_text(json.dumps(geo))


def test_narrabeen_transects_extend_behind_the_survey_origin(cfg):
    write_narrabeen(cfg)
    t = narrabeen.transects(cfg)
    assert t["transect_id"].tolist() == ["nb-PF1"]
    lon0, lat0 = t.geometry.iloc[0].coords[0]
    geod = Geod(ellps="WGS84")
    _, _, behind = geod.inv(lon0, lat0, 151.30, -33.70)
    _, _, total = geod.inv(*t.geometry.iloc[0].coords[0], *t.geometry.iloc[0].coords[-1])
    assert behind == pytest.approx(narrabeen.LANDWARD_M, abs=0.5)
    assert total == pytest.approx(narrabeen.LANDWARD_M + narrabeen.SEAWARD_M, abs=0.5)


def test_narrabeen_survey_contour_in_transect_metres(cfg):
    write_narrabeen(cfg)
    s = narrabeen.surveys(cfg, 0.0)
    assert s["x"].tolist() == pytest.approx([80.0 + narrabeen.LANDWARD_M, 90.0 + narrabeen.LANDWARD_M], abs=0.01)
    assert s["slope"].tolist() == pytest.approx([0.1, 0.1], abs=1e-6)
    assert (s["time"].dt.hour == 12).all()


def test_benchmark_scoring_uses_the_offset_and_same_matching(cfg):
    site = benchmark.SITES["narrabeen"]
    survey = pd.DataFrame({"transect_id": "nb-PF1", "time": pd.date_range("2010-01-01", periods=420, freq="7D"),
                           "x": 300.0 + 10 * np.sin(np.arange(420) / 5)})
    image_times = survey["time"] + pd.Timedelta(days=3)
    truth = np.interp(image_times.astype("int64"), survey["time"].astype("int64"), survey["x"])
    folder = benchmark.raw_dir(cfg) / site.code / "CoastSat" / "tidally_corrected_timeseries_MSL"
    folder.mkdir(parents=True)
    pd.DataFrame({"dates": image_times.dt.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                  "PF1": truth - site.offset - 5.0, "satname": "L8"}).to_csv(
        folder / "PF1_timeseries_tidally_corrected.csv", index=False)
    ceva = pd.DataFrame({"transect_id": "nb-PF1", "time": image_times, "family": "landsat",
                         "x_corr": truth + 2.0, "use": True})
    table = benchmark.score(cfg, "narrabeen", survey, {"CEVA test": ceva})
    per_image = table[table["scale"] == "per image"].set_index("algorithm")
    assert per_image.loc["CoastSat", "bias"] == pytest.approx(-5.0, abs=1e-6)
    assert per_image.loc["CEVA test", "bias"] == pytest.approx(2.0, abs=1e-6)
    assert per_image.loc["CoastSat", "n"] == per_image.loc["CEVA test", "n"]
    assert set(table["scale"]) == {"per image", "yearly"}
