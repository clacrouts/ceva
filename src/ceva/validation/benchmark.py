"""Score the published algorithms of the SDS benchmark (Vos et al. 2023) against our surveys.

The benchmark distributes each team's shoreline series per transect
(https://github.com/SatelliteShorelines/SDS_Benchmark/tree/main/submissions). Scoring
them with the same survey contours, transects and time matching as CEVA gives a
like-for-like comparison with the state of the art, independent of how the
benchmark paper computed its own tables.

Team positions are distances from the benchmark transect origin; our transects start
``offset`` metres behind that origin on the same line.

CoastSat's published shorelines carry a half-pixel geolocation shift: CoastSat converts
contour coordinates, which count from pixel centres, with the image's GeoTIFF transform,
which counts from the pixel corner. Every point therefore sits half a pixel west and north
of where its pixels put it (7.5 m each way on CoastSat's 15 m Landsat grid, 5 m on the
10 m Sentinel-2 grid). Across a transect that is a few metres seaward or landward depending
on which way the beach faces. ``score`` can add CoastSat's series with that shift removed.
"""
from __future__ import annotations

import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ceva.config import Config
from ceva.validation.compare import match_in_time, metrics_table

log = logging.getLogger(__name__)

REPO = "https://raw.githubusercontent.com/SatelliteShorelines/SDS_Benchmark/main/submissions"
_UA = {"User-Agent": "ceva (+https://github.com/clacrouts/ceva)"}


@dataclass(frozen=True)
class Site:
    code: str                   # benchmark folder name
    profiles: tuple[str, ...]   # benchmark transects scored
    offset: float               # benchmark chainage + offset = CEVA distance
    transect_id: dict           # benchmark profile -> CEVA transect id


SITES = {
    "trucvert": Site("TRUCVERT", ("-400", "-300", "-200", "-100"), 200.0,
                     {p: f"tv{int(p):+05d}" for p in ("-400", "-300", "-200", "-100")}),
    "narrabeen": Site("NARRABEEN", ("PF1", "PF2", "PF4", "PF6", "PF8"), 200.0,
                      {p: f"nb-{p}" for p in ("PF1", "PF2", "PF4", "PF6", "PF8")}),
}

# (team, product label, folder, file pattern, sensor)
_CORRECTED = "{p}_timeseries_tidally_corrected.csv"
PRODUCTS = [
    ("CoastSat", "per image", "tidally_corrected_timeseries_MSL", _CORRECTED, "landsat"),
    ("CoastSat", "per image", "tidally_corrected_timeseries_MSL_S2", _CORRECTED, "sentinel2"),
    ("SHOREX", "per image", "tidally_corrected_timeseries_MSL", _CORRECTED, "landsat"),
    ("CASSIE", "per image", "tidally_corrected_timeseries_MSL", _CORRECTED, "landsat"),
    ("CASSIE", "per image", "tidally_corrected_timeseries_MSL_S2", _CORRECTED, "sentinel2"),
    ("Deltares", "yearly composite", "Transect_Mapping/CSV/Annual/cleaned_timeseries", "{p}_timerseries_cleaned.csv",
     "landsat"),
]


COASTSAT_PIXEL_M = {"landsat": 15.0, "sentinel2": 10.0}
GEOLOCATED = "CoastSat, half-pixel shift removed"


def seaward_directions(transects: gpd.GeoDataFrame) -> dict[str, tuple[float, float]]:
    """Unit vector (east, north) pointing seaward along each transect, in its UTM zone."""
    projected = transects.to_crs(transects.estimate_utm_crs())
    out = {}
    for tid, line, sign in zip(projected["transect_id"], projected.geometry, projected["seaward_sign"], strict=True):
        if sign == 0:
            continue
        a, b = np.asarray(line.coords[0]), np.asarray(line.coords[-1])
        d = (b - a) if sign == 1 else (a - b)
        out[str(tid)] = tuple(d / np.linalg.norm(d))
    return out


def half_pixel_error(seaward: dict[str, tuple[float, float]], sensors: str) -> dict[str, float]:
    """Cross-shore error of CoastSat's published positions from the half-pixel shift, per transect.

    Positive: the published position is seaward of where the image's pixels put it.
    """
    half = COASTSAT_PIXEL_M[sensors] / 2.0
    return {tid: -half * ux + half * uy for tid, (ux, uy) in seaward.items()}


def raw_dir(cfg: Config) -> Path:
    return cfg.data_dir / "raw" / "insitu" / "sds_benchmark" / "submissions"


def download(cfg: Config, site: str) -> Path:
    s = SITES[site]
    for team, _, folder, pattern, _ in PRODUCTS:
        for p in s.profiles:
            name = pattern.format(p=p)
            dest = raw_dir(cfg) / s.code / team / folder / name
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = f"{REPO}/team_{team}/{s.code}/{folder}/{name}"
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=120) as response:
                    dest.write_bytes(response.read())
            except urllib.error.HTTPError as exc:
                log.info("no %s %s for %s %s (%s)", team, folder, s.code, p, exc.code)
    return raw_dir(cfg) / s.code


def team_series(cfg: Config, site: str) -> pd.DataFrame:
    """All downloaded team series: team, product, sensors, transect_id, time, x (CEVA distance)."""
    s = SITES[site]
    frames = []
    for team, product, folder, pattern, sensor in PRODUCTS:
        for p in s.profiles:
            path = raw_dir(cfg) / s.code / team / folder / pattern.format(p=p)
            if not path.exists():
                continue
            data = pd.read_csv(path)
            time_col = "dates" if "dates" in data else "Date"
            value_col = p if p in data else "Distance"
            x = pd.to_numeric(data[value_col], errors="coerce")
            frames.append(pd.DataFrame({
                "team": team, "product": product, "sensors": sensor, "transect_id": s.transect_id[p],
                "time": pd.to_datetime(data[time_col], utc=True).dt.tz_convert(None), "x": x + s.offset,
            }).dropna(subset=["x"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def yearly_survey(surveys: pd.DataFrame, min_surveys: int = 4) -> pd.DataFrame:
    by = surveys.assign(year=surveys["time"].dt.year).groupby(["transect_id", "year"])["x"]
    out = by.agg(x_survey="median", n_surveys="size").reset_index()
    return out[out["n_surveys"] >= min_surveys]


def score(cfg: Config, site: str, surveys: pd.DataFrame, ceva: dict[str, pd.DataFrame],
          seaward: dict[str, tuple[float, float]] | None = None) -> pd.DataFrame:
    """Per-image and yearly metrics for each team and each CEVA variant on the benchmark transects.

    ``ceva`` maps a label (e.g. "CEVA, default") to its waterlines table (time, family,
    x_corr, use). Per-image rows use the ±10-day matching; yearly rows compare the median
    of the year's positions with the median of that year's surveys. With ``seaward``
    (see ``seaward_directions``), CoastSat's series are also scored with the half-pixel
    shift removed.
    """
    s = SITES[site]
    keep = set(s.transect_id.values())
    surveys = surveys[surveys["transect_id"].isin(keep)]
    annual_survey = yearly_survey(surveys)
    series = team_series(cfg, site)
    candidates = []
    for (team, product, sensors), g in series.groupby(["team", "product", "sensors"]):
        candidates.append((team, product, sensors, g[["transect_id", "time", "x"]]))
        if team == "CoastSat" and seaward is not None:
            error = g["transect_id"].map(half_pixel_error(seaward, sensors))
            candidates.append((GEOLOCATED, product, sensors, g[["transect_id", "time"]].assign(x=g["x"] - error)))
    for label, w in ceva.items():
        w = w[w["transect_id"].isin(keep) & w["use"]]
        for sensors in ("landsat", "sentinel2"):
            g = w[w["family"] == sensors]
            candidates.append((label, "per image", sensors, g[["transect_id", "time"]].assign(x=g["x_corr"])))

    rows = []
    for name, product, sensors, g in candidates:
        if g.empty:
            continue
        if product == "per image":
            matched = match_in_time(g, surveys, "x")
            if len(matched) >= 5:
                m = metrics_table(matched.assign(group="x"), ["group"]).iloc[0].to_dict()
                rows.append({"algorithm": name, "product": product, "sensors": sensors, "scale": "per image"} | m)
            yearly = g.groupby(["transect_id", g["time"].dt.year.rename("year")])["x"].agg(["median", "size"])
            yearly = yearly[yearly["size"] >= 3].reset_index().rename(columns={"median": "x"})
        else:
            yearly = g.assign(year=g["time"].dt.year)[["transect_id", "year", "x"]]
        pairs = yearly.merge(annual_survey, on=["transect_id", "year"])
        if len(pairs) >= 5:
            pairs["error"] = pairs["x"] - pairs["x_survey"]
            m = metrics_table(pairs.assign(group="x"), ["group"]).iloc[0].to_dict()
            rows.append({"algorithm": name, "product": product, "sensors": sensors, "scale": "yearly"} | m)
    out = pd.DataFrame(rows).drop(columns=["group"], errors="ignore")
    return out.sort_values(["scale", "sensors", "rmse"]).reset_index(drop=True)


def same_image_offset(cfg: Config, site: str, ceva_raw: pd.DataFrame,
                      seaward: dict[str, tuple[float, float]] | None = None) -> dict:
    """CEVA raw waterline minus CoastSat raw waterline on images both processed (positive = CEVA seaward).

    With ``seaward``, CoastSat's half-pixel shift is removed from its positions first.
    """
    s = SITES[site]
    diffs = []
    for p in s.profiles:
        url = f"{REPO}/team_CoastSat/{s.code}/raw_timeseries/{p}_timeseries_raw.csv"
        dest = raw_dir(cfg) / s.code / "CoastSat" / "raw_timeseries" / f"{p}_timeseries_raw.csv"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=120) as response:
                    dest.write_bytes(response.read())
            except urllib.error.HTTPError:
                continue
        data = pd.read_csv(dest)
        theirs = pd.DataFrame({"day": pd.to_datetime(data["dates"], utc=True).dt.tz_convert(None).dt.floor("D"),
                               "x_cs": pd.to_numeric(data[p], errors="coerce") + s.offset}).dropna()
        if seaward is not None and "satname" in data:
            family = data.loc[theirs.index, "satname"].map(lambda n: "sentinel2" if n == "S2" else "landsat")
            error = {f: half_pixel_error(seaward, f)[s.transect_id[p]] for f in COASTSAT_PIXEL_M}
            theirs["x_cs"] = theirs["x_cs"] - family.map(error)
        ours = ceva_raw[ceva_raw["transect_id"] == s.transect_id[p]]
        ours = ours.assign(day=ours["time"].dt.floor("D"))
        both = ours.merge(theirs, on="day")
        diffs.append(both["x_obs"] - both["x_cs"])
    d = pd.concat(diffs) if diffs else pd.Series(dtype=float)
    if d.empty:
        return {"n": 0}
    return {"n": int(d.size), "median": float(d.median()), "mad": float((d - d.median()).abs().median())}
