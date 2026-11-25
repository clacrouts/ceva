"""The assistant's tools: read-only questions to the dataset, each answer small, rounded, with links to the map
and the flags a reader needs (trend not significant, low precision, coast not measured).

Every method of ``Tools`` whose name does not start with ``_`` is one tool; its docstring is what the model
reads to choose it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ceva.assistant.data import CODE_URL, DATA_DOI, GCTS_STEP_M, IMAGES_DOI, Dataset

MAX_RADIUS_KM = 20.0
SHORT_RECORD = 20                 # years with a position: most transects have 30 to 42
MAX_ITEMS = 25
CLASSES = ["extreme_erosion", "severe_erosion", "intense_erosion", "erosion", "stable", "accretion",
           "intense_accretion", "severe_accretion", "extreme_accretion"]
CLASS_RANGES = {
    "extreme_erosion": "faster than -5 m/yr", "severe_erosion": "-5 to -3 m/yr", "intense_erosion": "-3 to -1 m/yr",
    "erosion": "-1 to -0.5 m/yr", "stable": "within +-0.5 m/yr", "accretion": "0.5 to 1 m/yr",
    "intense_accretion": "1 to 3 m/yr", "severe_accretion": "3 to 5 m/yr", "extreme_accretion": "faster than 5 m/yr",
}
VARIABLES = {
    "geomorphology": ("landform (Global Coastal Typology)", "coastal_type", ""),
    "shoreline_change": ("rate of shoreline change", "rate", "m/yr"),
    "coastal_slope": ("regional slope across the shore (GEBCO 2024)", "coastal_slope_pct", "%"),
    "sea_level": ("relative sea-level rise (altimetry minus glacial uplift)", "relative_sea_level_mm_yr", "mm/yr"),
    "waves": ("mean significant wave height (ERA5)", "mean_wave_height_m", "m"),
    "tide": ("mean tidal range (EOT20)", "mean_tidal_range_m", "m"),
}
NOT_MEASURED = {
    "rocky_shore_platform_or_large_boulders": "rocky shore: CEVA measures sand and gravel beaches only",
    "no_sediment_or_shore_platform": "cliff, rock platform or built structure: not measured (a cliff retreats at "
                                     "its top, which the waterline does not show)",
    "muddy_sediments": "marsh or tidal flat: its shoreline is the vegetation edge, which needs another method",
    "unknown": "coast of unknown type in the Global Coastal Typology: not measured",
    "no_result": "sand or gravel, but no shoreline could be measured (water at both ends of the transects: a "
                 "spit, a lagoon, a narrow island; or too few usable images)",
}


def _r(value, digits: int = 2):
    return None if value is None or pd.isna(value) else round(float(value), digits)


def _flag(value) -> bool | None:
    return None if value is None or pd.isna(value) else bool(value)


class Tools:
    def __init__(self, ds: Dataset):
        self.ds = ds

    # ------------------------------------------------------------------ shared pieces
    def _brief(self, i: int, distance_km: float | None = None) -> dict:
        r = self.ds.rates.iloc[i]
        out = {"id": r["transect_id"], "country": r.get("country"),
               "rate_m_per_yr": _r(r["rate"]), "rate_95_range": [_r(r["rate_lo"]), _r(r["rate_hi"])],
               "change_class": r.get("change_class"), "significant": _flag(r.get("significant")),
               "years": [int(r["first_year"]), int(r["last_year"])] if pd.notna(r.get("first_year")) else None,
               "years_with_position": int(r["n_years"]) if pd.notna(r.get("n_years")) else None,
               "precision_m": _r(r.get("se_median_m"), 1), "coastal_type": r.get("coastal_type"),
               "map": self.ds.link(i)}
        if distance_km is not None:
            out["distance_km"] = _r(distance_km, 2)
        if pd.notna(r.get("cvi", np.nan)):
            out["vulnerability"] = {"index": _r(r["cvi"], 1), "class": r.get("cvi_class")}
        flags = []
        if out["significant"] is False:
            flags.append("trend not significant: its 95 % range includes zero")
        if (out["precision_m"] or 0) > 20:
            flags.append("low precision: yearly positions scatter by more than 20 m")
        if (out["years_with_position"] or 99) < SHORT_RECORD:
            flags.append(f"short record: {out['years_with_position']} years with a position (most transects have "
                         "30 to 42)")
        if flags:
            out["flags"] = flags
        return out

    def _with_source(self, answer: dict) -> dict:
        return answer | {"source": self.ds.source()}

    def _mask(self, country: str | None, bbox: list[float] | None, coastal_type: str | None,
              shore: str | None) -> tuple[np.ndarray | None, str]:
        """Rows of the area asked about, and how to name it; None and a message when it is not understood."""
        r, ds = self.ds.rates, self.ds
        mask = np.ones(len(r), bool)
        names = []
        if country:
            code = ds.country(country)
            if code is None:
                return None, f"unknown country {country!r}: give an ISO code or the English name of one of " \
                             f"{sorted(ds.country_names.values())}"
            mask &= (r["country"] == code).to_numpy()
            names.append(ds.country_names.get(code, code))
        if bbox:
            if len(bbox) != 4:
                return None, "bbox is [west, south, east, north] in degrees"
            w, s, e, n = (float(v) for v in bbox)
            mask &= (r["lon"].between(w, e) & r["lat"].between(s, n)).to_numpy()
            names.append(f"box {w:.2f},{s:.2f},{e:.2f},{n:.2f}")
        if coastal_type:
            mask &= (r["coastal_type"] == coastal_type).to_numpy()
            names.append(coastal_type.replace("_", " "))
        if shore:
            built = r["is_built_environment"].astype("boolean").fillna(False).to_numpy(bool)
            defended = r["has_defence"].astype("boolean").fillna(False).to_numpy(bool)
            test = {"natural": ~built & ~defended, "built": built, "defended": defended}.get(shore)
            if test is None:
                return None, "shore is natural, built or defended"
            mask &= test
            names.append(f"{shore} shores")
        return mask, ", ".join(names) or "all coasts published"

    def _figures(self, rows: pd.DataFrame) -> dict:
        rated = rows[rows["rate"].notna()]
        n = max(len(rated), 1)
        sig = rated["significant"].astype("boolean").fillna(False).to_numpy(bool)
        out = {"transects": int(len(rows)), "coast_km": round(len(rows) * GCTS_STEP_M / 1000),
               "with_rate": int(len(rated)),
               "retreating_share": round(float((sig & (rated["rate"] < 0)).sum()) / n, 3),
               "advancing_share": round(float((sig & (rated["rate"] > 0)).sum()) / n, 3),
               "median_rate_m_per_yr": _r(rated["rate"].median()),
               "faster_than_1_m_per_yr_share": round(float((rated["rate"].abs() > 1).sum()) / n, 3)}
        out["class_shares"] = {c: round(float((rated["change_class"] == c).sum()) / n, 3) for c in CLASSES}
        if "se_median_m" in rows and len(rows):
            out["precision_under_10_m_share"] = round(float((rows["se_median_m"] < 10).mean()), 3)
        return out

    def _vulnerability_figures(self, rows: pd.DataFrame) -> dict | None:
        if "cvi" not in rows or rows["cvi"].notna().sum() == 0:
            return None
        known = rows[rows["cvi"].notna()]
        n = len(known)
        out = {"with_index": int(n), "median_index": _r(known["cvi"].median(), 1),
               "class_shares": {c: round(float((known["cvi_class"] == c).sum()) / n, 3)
                                for c in ("low", "moderate", "high", "very_high")}}
        top = {v: round(float((known[f"rank_{v}"] >= 4).mean()), 3) for v in VARIABLES if f"rank_{v}" in known}
        out["share_ranked_4_or_5"] = dict(sorted(top.items(), key=lambda kv: -kv[1]))
        return out

    # ------------------------------------------------------------------ the tools
    def about_dataset(self) -> dict[str, Any]:
        """What the dataset is: version, coverage, years, method in short, accuracy against surveys, the
        vulnerability index, licence, how to cite, links. Use it for questions about the data themselves."""
        m, ds = self.ds.manifest, self.ds
        quality = m.get("validation") or {}
        vul = m.get("vulnerability") or {}
        return self._with_source({
            "name": "CEVA: satellite-derived shorelines of sand and gravel coasts",
            "code_version": m.get("ceva_version"), "generated": m.get("generated"),
            "years": m.get("years"), "transects": m.get("transects"),
            "countries": sorted(ds.country_names.values()),
            "transect_spacing_m": GCTS_STEP_M,
            "method": "every clear Landsat (1984-) and Sentinel-2 (2015-) image is read along transects of the Global "
                      "Coastal Transect System; the waterline is moved to mean sea level with the tide (EOT20) and "
                      "waves (ERA5); a yearly position is the median of the year's waterlines; the rate is a "
                      "straight-line fit through the yearly positions, with its 95 % range. Sand and gravel coasts "
                      "only.",
            "accuracy": {"survey_beaches": quality.get("sites"),
                         "yearly_position_rmse_m": quality.get("yearly_rmse"),
                         "rate_rmse_m_per_yr": quality.get("rate_rmse"),
                         "note": "against field surveys; precision_m of a transect is its repeatability, not its "
                                 "accuracy"},
            "change_classes": CLASS_RANGES,
            "vulnerability_index": {
                "method": "USGS coastal vulnerability index (Thieler and Hammar-Klose 1999): six variables ranked 1 "
                          "(low) to 5 (very high), CVI = sqrt(product of ranks / 6)",
                "variables": {k: v[0] for k, v in VARIABLES.items()},
                "classes": "quartiles of the index over the coast published: low, moderate, high, very_high",
                "class_edges": [_r(e, 1) for e in vul.get("class_edges", [])],
            } if vul else "not published",
            "caveat": "not an official erosion map: national and regional observatories are the reference for "
                      "planning",
            "licence": m.get("licence", "CC-BY-4.0"),
            "cite": f"the dataset doi:{DATA_DOI} and its sources (see the record's README)",
            "links": {"map": ds.public, "data": f"https://doi.org/{DATA_DOI}", "images": f"https://doi.org/{IMAGES_DOI}",
                      "code": CODE_URL},
        })

    def transects_near(self, lon: float, lat: float, radius_km: float = 3.0, limit: int = 10) -> dict[str, Any]:
        """The measured transects nearest a point (longitude, latitude in degrees), within radius_km (at most 20):
        rate of shoreline change with its 95 % range, class, significance, precision, coastal type, vulnerability
        and a map link for each, plus their median. For "how is the beach at X changing?": give the coordinates
        of the place."""
        radius = min(max(float(radius_km), 0.1), MAX_RADIUS_KM)
        idx, dist = self.ds.near(lon, lat, radius, min(max(int(limit), 1), MAX_ITEMS))
        if len(idx) == 0:
            j, km = self.ds.nearest(lon, lat)
            return self._with_source({
                "found": 0, "note": f"no measured transect within {radius:g} km; the nearest is {km:.1f} km away. "
                                    "The coast here may not be sand or gravel (not_measured tells why).",
                "nearest": self._brief(j, km)})
        rows = self.ds.rates.iloc[idx]
        return self._with_source({
            "found": int(len(idx)), "radius_km": radius,
            "median_rate_m_per_yr": _r(rows["rate"].median()),
            "transects": [self._brief(i, d) for i, d in zip(idx, dist, strict=True)],
            "map": self.ds.map_link(lon, lat, 13)})

    def transect_detail(self, transect_id: str) -> dict[str, Any]:
        """Everything about one transect (id like cl30793s01tr02679265): its yearly shoreline positions with their
        standard errors and number of images, the rate, the Shoreline Monitor's rate, the built-up shore and
        defences, the vulnerability index and whether satellite images can be viewed."""
        i = self.ds.index_of(transect_id)
        if i is None:
            return {"error": f"no transect {transect_id!r} in the dataset"}
        r = self.ds.rates.iloc[i]
        s = self.ds.series(i)
        mean = s["x"].mean()
        out = self._brief(i) | {
            "r2": _r(r.get("r2")), "images_total": int(s["images"].sum()),
            "built_up_shore": _flag(r.get("is_built_environment")), "coastal_defences": _flag(r.get("has_defence")),
            "shoreline_monitor_rate_m_per_yr": _r(r.get("sds_rate")),
            "yearly_positions": {"note": "metres seaward of the transect's mean position; se_m is one standard error",
                                 "rows": [[int(y), _r(x - mean, 1), _r(e, 1), int(n)] for y, x, e, n in
                                          s[["year", "x", "se", "images"]].itertuples(index=False)],
                                 "columns": ["year", "position_m", "se_m", "images"]},
            "images": bool(self.ds.frames and r.get("unit") in self.ds.frames.get("units", {})),
        }
        if pd.notna(r.get("cvi", np.nan)):
            out["vulnerability"] = self._transect_vulnerability(i)
        return self._with_source(out)

    def change_between(self, start_year: int, end_year: int, transect_ids: list[str] | None = None,
                       lon: float | None = None, lat: float | None = None, radius_km: float = 1.0) -> dict[str, Any]:
        """Shoreline change over a chosen period, for given transect ids or the transects within radius_km of a
        point: the rate fitted over the years of that period (with its 95 % range), and the net movement between
        the first and last years with a position in it. For "since 2000?" or "between 1990 and 2010?"."""
        from ceva.change_metrics import linear_regression_rate

        y0, y1 = int(start_year), int(end_year)
        if y1 - y0 < 4:
            return {"error": "the period needs at least 5 years"}
        if transect_ids:
            idx = [i for i in (self.ds.index_of(t) for t in transect_ids[:MAX_ITEMS]) if i is not None]
        elif lon is not None and lat is not None:
            idx = list(self.ds.near(lon, lat, min(float(radius_km), MAX_RADIUS_KM), MAX_ITEMS)[0])
        else:
            return {"error": "give transect_ids, or lon and lat"}
        results = []
        for i in idx:
            s = self.ds.series(i)
            s = s[(s["year"] >= y0) & (s["year"] <= y1)]
            item = {"id": self.ds.rates.iloc[i]["transect_id"], "years_with_position": int(len(s)),
                    "map": self.ds.link(i)}
            if len(s) >= 5:
                slope, lo, hi, _ = linear_regression_rate(s["year"].to_numpy(float), s["x"].to_numpy(float))
                net = float(s["x"].iloc[-1] - s["x"].iloc[0])
                item |= {"rate_m_per_yr": _r(slope), "rate_95_range": [_r(lo), _r(hi)],
                         "significant": bool(lo > 0 or hi < 0),
                         "net_movement_m": _r(net, 1), "between": [int(s["year"].iloc[0]), int(s["year"].iloc[-1])]}
            else:
                item["note"] = "fewer than 5 years with a position in the period: no rate"
            results.append(item)
        rates = [t["rate_m_per_yr"] for t in results if t.get("rate_m_per_yr") is not None]
        return self._with_source({
            "period": [y0, y1], "transects": results,
            "median_rate_m_per_yr": _r(np.median(rates)) if rates else None,
            "note": "positive is seaward (accretion), negative landward (erosion)"})

    def area_summary(self, country: str | None = None, bbox: list[float] | None = None,
                     coastal_type: str | None = None, shore: str | None = None) -> dict[str, Any]:
        """Figures for a country (ISO code or English name), a box [west, south, east, north], a coastal type
        (dune, sediment_plain, wetland, inlet, moderately_sloped, cliffed_or_steep, bedrock_plain,
        engineered_structures) and/or a shore (natural, built, defended): transects, km of coast, shares
        retreating and advancing (significant trends), median rate, classes, precision, vulnerability; and the
        same for the whole dataset to compare."""
        mask, name = self._mask(country, bbox, coastal_type, shore)
        if mask is None:
            return {"error": name}
        rows = self.ds.rates[mask]
        if rows.empty:
            return {"error": f"no measured transect in {name}"}
        out = {"area": name} | self._figures(rows)
        vul = self._vulnerability_figures(rows)
        if vul:
            out["vulnerability"] = vul
        whole = self._figures(self.ds.rates)
        out["whole_dataset"] = {k: whole[k] for k in ("transects", "retreating_share", "advancing_share",
                                                       "median_rate_m_per_yr")}
        w, s, e, n = rows["lon"].min(), rows["lat"].min(), rows["lon"].max(), rows["lat"].max()
        out["map"] = self.ds.map_link((w + e) / 2, (s + n) / 2, 7 if country else 9)
        return self._with_source(out)

    def rank_hotspots(self, direction: str = "erosion", country: str | None = None, bbox: list[float] | None = None,
                      by: str = "stretch", n: int = 10) -> dict[str, Any]:
        """Where the shoreline moves fastest: erosion (retreat) or accretion (advance), in a country or box, as
        stretches of coast (about 1 km: the median rate of its transects, at least 5 of them) or as single
        transects with a significant trend over at least 20 years. n at most 25."""
        if direction not in ("erosion", "accretion"):
            return {"error": "direction is erosion or accretion"}
        mask, name = self._mask(country, bbox, None, None)
        if mask is None:
            return {"error": name}
        rows = self.ds.rates[mask & self.ds.rates["rate"].notna().to_numpy()]
        n = min(max(int(n), 1), MAX_ITEMS)
        sign = -1 if direction == "erosion" else 1
        if by == "transect":
            sig = rows["significant"].astype("boolean").fillna(False).to_numpy(bool)
            long = (rows["n_years"] >= SHORT_RECORD).to_numpy()
            best = rows[sig & long & (np.sign(rows["rate"]) == sign)].sort_values("rate", ascending=sign < 0).head(n)
            items = [self._brief(int(self.ds.index_of(t))) for t in best["transect_id"]]
        else:
            parts = rows["transect_id"].str.extract(r"^(cl\d+s\d+)tr(\d+)$")
            key = parts[0] + ":" + (parts[1].astype("int64") // 1000).astype(str)
            moving = rows["significant"].astype("boolean").fillna(False) & (np.sign(rows["rate"]) == sign)
            g = rows.assign(key=key.to_numpy(), moving=moving.to_numpy()).groupby("key")
            stretch = pd.DataFrame({"n": g.size(), "median_rate": g["rate"].median(),
                                    "share_significant": g["moving"].mean(), "lon": g["lon"].mean(),
                                    "lat": g["lat"].mean(), "country": g["country"].first(),
                                    "middle": g["transect_id"].agg(lambda s: s.iloc[len(s) // 2])})
            stretch = stretch[(stretch["n"] >= 5) & (np.sign(stretch["median_rate"]) == sign)]
            stretch = stretch.sort_values("median_rate", ascending=sign < 0).head(n)
            items = [{"median_rate_m_per_yr": _r(s.median_rate), "transects": int(s.n),
                      "share_significant": _r(s.share_significant), "country": s.country,
                      "centre": [_r(s.lon, 5), _r(s.lat, 5)],
                      "map": self.ds.link(int(self.ds.index_of(s.middle)), 14)} for s in stretch.itertuples()]
        return self._with_source({"area": name, "direction": direction, "by": by, "items": items,
                                  "note": "stretch: about 1 km of one GCTS coastline" if by != "transect"
                                  else f"transects with at least {SHORT_RECORD} years with a position"})

    def compare_products(self, country: str | None = None, transect_id: str | None = None) -> dict[str, Any]:
        """How CEVA agrees with other shoreline products (the Deltares Shoreline Monitor, EMODnet satellite and
        field data): for one transect (their rates and years, and CEVA's rate over the same years), or for a
        country or the whole dataset (shares in the same class of change, within 0.5 m/yr, in opposite
        directions)."""
        if transect_id:
            i = self.ds.index_of(transect_id)
            if i is None:
                return {"error": f"no transect {transect_id!r} in the dataset"}
            unit = self.ds.rates.iloc[i].get("unit")
            unit_refs = (self.ds.unit_file("refs", unit) or {}) if unit else {}
            refs = unit_refs.get("transects", {}).get(str(transect_id), {})
            names = {p["key"]: p["product"]["name"] for p in (self.ds.validation or {}).get("products", [])}
            items = {names.get(k, k): {"rate_m_per_yr": v.get("r"), "years": v.get("y"), "class": v.get("k"),
                                       "ceva_rate_same_years": v.get("c"), "distance_m": v.get("d")}
                     for k, v in refs.items()}
            return self._with_source({"transect": self._brief(i), "products": items or "no other product here"})
        v = self.ds.validation
        if not v:
            return {"error": "the comparison is not published"}
        code = self.ds.country(country) if country else None
        if country and code is None:
            return {"error": f"unknown country {country!r}"}
        english = self.ds.country_names.get(code) if code else None
        keep = ("n_cls", "agree", "within_05", "opposite")
        out = []
        for p in v["products"]:
            entry = {"product": p["product"]["name"], "years": p.get("years"),
                     "whole_dataset": {k: p["overall"].get(k) for k in keep}}
            if english:
                here = next((c for c in p["by"].get("country", []) if c["key"] == english), None)
                entry[english] = {k: here.get(k) for k in keep} if here else "does not cover this country"
            out.append(entry)
        return self._with_source({"products": out, "keys": {
            "n_cls": "transects compared", "agree": "share in the same class of change (erosion, stable, accretion)",
            "within_05": "share of rates within 0.5 m/yr of each other", "opposite": "share in opposite directions"},
            "note": "the other products have their own errors: agreement is not proof of accuracy"})

    def not_measured(self, lon: float, lat: float, radius_km: float = 2.0) -> dict[str, Any]:
        """Why there is no data near a point: the kinds of coast not measured within radius_km (rock, cliffs,
        structures, mud, or sand without a result), and the nearest measured transect."""
        radius = min(max(float(radius_km), 0.1), MAX_RADIUS_KM)
        j, km = self.ds.nearest(lon, lat)
        out = {"nearest_measured": self._brief(j, km)}
        rows = self.ds.coverage_near(lon, lat, radius)
        if rows is None:
            out["note"] = ("the map of the coast not measured is not published yet. CEVA measures sand and gravel "
                           "coasts only; " + "; ".join(NOT_MEASURED.values()))
        elif rows.empty:
            out["note"] = f"no transect of the coast within {radius:g} km: this point may be inland or at sea"
        else:
            kinds = rows.groupby("why")["km"].agg(["size", "min"]).sort_values("min")
            out["not_measured"] = [{"kind": k, "transects": int(r["size"]), "nearest_km": _r(r["min"], 2),
                                    "why": NOT_MEASURED.get(k, NOT_MEASURED["unknown"])} for k, r in kinds.iterrows()]
        return self._with_source(out)

    def images_for(self, transect_id: str) -> dict[str, Any]:
        """Whether the satellite images behind a transect's shorelines can be viewed, and the link to the image
        viewer (every image of its 2 km segment, with the waterline detected and the corrected shoreline)."""
        i = self.ds.index_of(transect_id)
        if i is None:
            return {"error": f"no transect {transect_id!r} in the dataset"}
        unit = self.ds.rates.iloc[i].get("unit")
        frames = self.ds.frames or {}
        if not unit or unit not in frames.get("units", {}):
            return self._with_source({"images": False, "note": "no images published for this transect"})
        index = self.ds.unit_file("frames", unit) or {}
        seg = next((s for s in index.get("segments", []) if str(transect_id) in s.get("transects", [])), None)
        if seg is None:
            return self._with_source({"images": False, "note": "no image segment covers this transect"})
        base = self.ds.public.rstrip("/")
        return self._with_source({
            "images": True, "segment": seg["name"], "images_in_segment": seg.get("frames"),
            "viewer": f"{base}/frames.html?segment={seg['name']}&transect={transect_id}&unit={unit}",
            "card": self.ds.link(i), "archive": f"https://doi.org/{IMAGES_DOI}"})

    def vulnerability(self, transect_id: str | None = None, country: str | None = None) -> dict[str, Any]:
        """The coastal vulnerability index (USGS method): for one transect, its index, class and the six ranks
        (1 low to 5 very high) with the values behind them; for a country (or the whole dataset when neither is
        given), its classes, median index and which variables most often rank high."""
        vul = self.ds.manifest.get("vulnerability")
        if not vul or "cvi" not in self.ds.rates:
            return {"error": "the vulnerability index is not published"}
        edges = [_r(e, 1) for e in vul.get("class_edges", [])]
        if transect_id:
            i = self.ds.index_of(transect_id)
            if i is None:
                return {"error": f"no transect {transect_id!r} in the dataset"}
            return self._with_source({"transect": transect_id, "map": self.ds.link(i), "class_edges": edges}
                                     | self._transect_vulnerability(i))
        mask, name = self._mask(country, None, None, None)
        if mask is None:
            return {"error": name}
        out = {"area": name, "class_edges": edges} | (self._vulnerability_figures(self.ds.rates[mask]) or {})
        if country:
            out["whole_dataset"] = self._vulnerability_figures(self.ds.rates)
        return self._with_source(out | {"note": "classes compare coasts with each other (quartiles); not a hazard "
                                                "or risk assessment"})

    def _transect_vulnerability(self, i: int) -> dict:
        r = self.ds.rates.iloc[i]
        ranks = {}
        for key, (label, column, unit) in VARIABLES.items():
            value = r.get(column)
            ranks[key] = {"rank": None if pd.isna(r.get(f"rank_{key}")) else int(r[f"rank_{key}"]), "what": label,
                          "value": value if isinstance(value, str) else _r(value, 2), "unit": unit}
        if pd.notna(r.get("sea_level_trend_mm_yr")):
            ranks["sea_level"]["detail"] = {"sea_surface_mm_yr": _r(r["sea_level_trend_mm_yr"]),
                                            "land_uplift_mm_yr": _r(r["uplift_mm_yr"])}
        return {"index": _r(r.get("cvi"), 1), "class": r.get("cvi_class"), "ranks": ranks}


def tool_names() -> list[str]:
    return [n for n in dir(Tools) if not n.startswith("_") and callable(getattr(Tools, n))]
