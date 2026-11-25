"""How CEVA's rates compare with other products and with surveys: the dashboard's validation page.

``payload`` turns the matched rates of every published transect (``references.match``) into the
numbers and the chart data the page draws: for each product, the agreement overall, by country,
coastal type, built-up or defended coast and CEVA's own precision; the joint distribution of the
two rates; the distribution of their difference; and how often both put a coast in the same class
(erosion, stable, accretion). ``sites_summary`` adds the accuracy against surveys at the
validation beaches, CEVA's two detection methods and the published algorithms of the SDS
benchmark side by side.

Differences are CEVA minus the product, in m/yr (positive: CEVA sees the coast further seaward
with time than the product does). Medians are used where outliers would dominate a mean.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ceva.config import Config
from ceva.validation.references import CLASSES, PRODUCTS, product_info

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
RATE_LIMIT = 3.0             # m/yr: extent of the joint-distribution chart
RATE_STEP = 0.1
DIFF_LIMIT = 4.0             # m/yr: extent of the difference histogram
DIFF_STEP = 0.1
OFFSET_LIMIT = 150.0         # m: extent of the position-offset histogram
OFFSET_STEP = 5.0
MIN_GROUP = 30               # smallest group reported
N_EXAMPLES = 12
PRECISION_BINS = [0.0, 5.0, 10.0, 20.0, math.inf]
PRECISION_LABELS = ["<5", "5-10", "10-20", ">20"]
PUBLISHED = ("perimage", "all")    # engine and sensors of the published yearly positions


def _num(v, digits: int = 3):
    return None if v is None or not np.isfinite(v) else round(float(v), digits)


def kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa of two classifications (agreement beyond chance)."""
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    if len(a) == 0:
        return math.nan
    observed = float((a == b).mean())
    expected = float(sum(a.eq(c).mean() * b.eq(c).mean() for c in CLASSES))
    return math.nan if expected >= 1 else (observed - expected) / (1 - expected)


def metrics(d: pd.DataFrame) -> dict:
    """Agreement of CEVA's rate (``ceva_rate``) with the product's (``rate``) and of their classes."""
    rated = d[np.isfinite(d["rate"].astype(float)) & np.isfinite(d["ceva_rate"].astype(float))]
    diff = (rated["ceva_rate"] - rated["rate"]).astype(float)
    out = {"n": int(len(rated))}
    if len(rated):
        out |= {
            "bias": _num(diff.median()), "mean": _num(diff.mean()), "mad": _num(diff.abs().median()),
            "rmse": _num(np.sqrt((diff ** 2).mean())),
            "r": _num(rated["ceva_rate"].astype(float).corr(rated["rate"].astype(float))) if len(rated) > 2 else None,
            "rho": _num(rated["ceva_rate"].astype(float).corr(rated["rate"].astype(float), method="spearman"))
            if len(rated) > 2 else None,
            "within_05": _num((diff.abs() <= 0.5).mean()), "within_1": _num((diff.abs() <= 1.0).mean()),
            "in_ci": _num(((rated["rate"] >= rated["ceva_lo"]) & (rated["rate"] <= rated["ceva_hi"])).mean()),
        }
    classed = d[d["cls"].isin(CLASSES) & d["ceva_cls"].isin(CLASSES)]
    out["n_cls"] = int(len(classed))
    if len(classed):
        out["agree"] = _num((classed["cls"] == classed["ceva_cls"]).mean())
        out["kappa"] = _num(kappa(classed["cls"], classed["ceva_cls"]))
        opposite = ((classed["cls"] == "erosion") & (classed["ceva_cls"] == "accretion")) | \
                   ((classed["cls"] == "accretion") & (classed["ceva_cls"] == "erosion"))
        out["opposite"] = _num(opposite.mean())
    return out


def histogram(values: pd.Series, limit: float, step: float) -> dict:
    v = pd.Series(values, dtype="float64").dropna().to_numpy()
    edges = np.arange(-limit, limit + step / 2, step)
    counts, _ = np.histogram(v[(v >= -limit) & (v < limit)], bins=edges)     # beyond the frame: counted apart
    return {"start": -limit, "step": step, "counts": counts.astype(int).tolist(),
            "below": int((v < -limit).sum()), "above": int((v >= limit).sum()), "n": int(v.size)}


def density(x: pd.Series, y: pd.Series, limit: float = RATE_LIMIT, step: float = RATE_STEP) -> dict:
    """Counts on a square grid: rows are CEVA's rate (y), columns the product's (x)."""
    x, y = np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    inside = (np.abs(x) < limit) & (np.abs(y) < limit)
    edges = np.arange(-limit, limit + step / 2, step)
    counts, _, _ = np.histogram2d(y[inside], x[inside], bins=[edges, edges])
    return {"start": -limit, "step": step, "counts": counts.astype(int).tolist(), "outside": int((~inside).sum()),
            "n": int(x.size)}


def confusion(d: pd.DataFrame) -> dict:
    """Rows: the product's class; columns: CEVA's."""
    classed = d[d["cls"].isin(CLASSES) & d["ceva_cls"].isin(CLASSES)]
    table = pd.crosstab(classed["cls"], classed["ceva_cls"]).reindex(index=list(CLASSES), columns=list(CLASSES),
                                                                     fill_value=0)
    return {"classes": list(CLASSES), "counts": table.astype(int).values.tolist()}


def groups(d: pd.DataFrame, column: str, order: list | None = None) -> list[dict]:
    rows = []
    for key, g in d.groupby(column, observed=True, dropna=True):
        m = metrics(g)
        if m["n"] >= MIN_GROUP or m["n_cls"] >= MIN_GROUP:
            rows.append({"key": str(key)} | m)
    if order is not None:
        rank = {str(k): i for i, k in enumerate(order)}
        rows.sort(key=lambda r: rank.get(r["key"], len(rank)))
    else:
        rows.sort(key=lambda r: -r["n"])
    return rows


def _flag(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return "yes" if value in (True, 1, "true", "True") else "no"


def product_block(d: pd.DataFrame, key: str, table: pd.DataFrame) -> dict:
    """Everything the page shows about one product."""
    product = PRODUCTS[key]
    if key == "sm":
        d = d[d["qa"].astype("boolean").fillna(False)]                  # the Monitor's own quality flag
    elif key == "emodnet_sat":                                          # one CEVA transect per product transect
        d = d.sort_values("distance_m").drop_duplicates("ref_id")
    rated = d[np.isfinite(d["rate"].astype(float)) & np.isfinite(d["ceva_rate"].astype(float))]
    block = {
        "key": key, "matched": int(d["transect_id"].nunique()), "share": _num(d["transect_id"].nunique() / len(table)),
        "years": [int(d["y0"].min()), int(d["y1"].max())] if d["y0"].notna().any() else None,
        "ceva_years": [int(rated["ceva_y0"].min()), int(rated["ceva_y1"].max())] if len(rated) else None,
        "overall": metrics(d), "confusion": confusion(d),
        "by": {
            "country": groups(d, "cn"),
            "coastal_type": groups(d, "coast"),
            "built": groups(d.assign(flag=d["built"].map(_flag)), "flag", ["no", "yes"]),
            "defence": groups(d.assign(flag=d["def"].map(_flag)), "flag", ["no", "yes"]),
            "precision": groups(d.assign(bin=pd.cut(d["se"], PRECISION_BINS, labels=PRECISION_LABELS, right=False)),
                                "bin", PRECISION_LABELS),
        },
    }
    if len(rated):
        block["density"] = density(rated["rate"], rated["ceva_rate"])
        block["difference"] = histogram(rated["ceva_rate"] - rated["rate"], DIFF_LIMIT, DIFF_STEP)
        worst = rated.assign(diff=rated["ceva_rate"] - rated["rate"])
        worst = worst.reindex(worst["diff"].abs().sort_values(ascending=False).index).head(N_EXAMPLES)
        block["examples"] = [{"id": r.transect_id, "unit": r.unit, "cn": r.cn if isinstance(r.cn, str) else None,
                              "rate": _num(r.rate, 2), "ceva": _num(r.ceva_rate, 2), "diff": _num(r.diff, 2)}
                             for r in worst.itertuples(index=False)]
    if key == "sm":
        offsets = d["offset_m"].astype(float).dropna()
        block["offset"] = histogram(offsets, OFFSET_LIMIT, OFFSET_STEP) | {
            "median": _num(offsets.median(), 1), "mad": _num((offsets - offsets.median()).abs().median(), 1)}
    if key == "emodnet_field":
        overlap = (d["ceva_y1"] - d["ceva_y0"] + 1) / (d["y1"].astype(float) - d["y0"].astype(float) + 1)
        block["overlap_median"] = _num(overlap[d["ceva_rate"].notna()].median(), 2)
        block["no_period"] = int(d["y0"].isna().sum())
    block["product"] = product_info([key])[0] | {"kind": product.kind}
    return block


def payload(pairs: pd.DataFrame, table: pd.DataFrame, sites: dict | None = None, run: dict | None = None) -> dict:
    """The validation page's data. ``pairs``: ``references.match`` of every published unit (with a
    ``unit`` column); ``table``: the published transects (``feature_table`` with ``cn``)."""
    context = pd.DataFrame(table.drop(columns="geometry", errors="ignore")).rename(columns={"id": "transect_id"})
    context = context[["transect_id", "unit", "cc", "cn", "coast", "built", "def", "se", "rate", "ny"]].rename(
        columns={"rate": "rate_full", "unit": "unit_published"})
    pairs = pairs.merge(context, on="transect_id", how="left")
    if "unit" not in pairs:
        pairs["unit"] = pairs["unit_published"]
    run = run or {}
    planned, finished = run.get("planned"), run.get("finished")
    out = {
        "v": FORMAT_VERSION, "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "provisional": bool(planned and finished is not None and finished < planned), "run": run,
        "transects": int(len(table)), "units": int(table["unit"].nunique()),
        "years": [int(table["y0"].min()), int(table["y1"].max())] if len(table) else None,
        "precision_median": _num(table["se"].median(), 1),
        "products": [],
    }
    for key in PRODUCTS:
        d = pairs[pairs["product"] == key]
        if d.empty:
            continue
        out["products"].append(product_block(d, key, table))
    if sites:
        out["sites"] = sites.get("sites", [])
        out["sites_generated"] = sites.get("generated")
    return out


# ------------------------------------------------------------------ the validation beaches

def _pick(records: list[dict] | None, **match) -> dict | None:
    for r in records or []:
        if all(r.get(k) == v for k, v in match.items()):
            return r
    return None


def _keep(record: dict | None, keys=("n", "bias", "std", "rmse", "mae", "r", "std_within", "first_year",
                                     "last_year")) -> dict | None:
    return None if record is None else {k: _num(record[k], 2) if isinstance(record.get(k), float) else record.get(k)
                                        for k in keys if k in record}


def sites_summary(cfg: Config, sites: list[str] | None = None,
                  variants: tuple[str, ...] = ("ceva", "coastsat")) -> dict:
    """Accuracy against surveys at each validation beach, from the stored results of each method."""
    from ceva import pipeline
    from ceva.validation import benchmark, report

    out = []
    for site in sites or list(report.SITE_INFO):
        info = report.SITE_INFO[site]
        done = [v for v in variants if (pipeline.results_dir(cfg, site, v) / "run.json").exists()]
        if "ceva" not in done:
            log.warning("%s: no CEVA results, left out", site)
            continue
        module = info["module"]
        surveys = module.surveys(cfg, cfg.tides.reference_elevation_m)
        if info["dense"]:
            survey_annual = report._survey_annual(surveys)
            survey_annual = survey_annual[survey_annual["n_surveys"] >= report.MIN_SURVEYS_PER_YEAR]
        else:
            survey_annual = surveys.rename(columns={"x": "x_survey"})[["transect_id", "year", "x_survey"]]
        entry = {"key": site, "title": info["title"], "dense": info["dense"], "kind": info["kind"],
                 "attribution": module.ATTRIBUTION, "survey_years": [int(survey_annual["year"].min()),
                                                                   int(survey_annual["year"].max())],
                 "methods": {}}
        extras = {}
        for v in done:
            summary, extras[v] = report._variant_summary(cfg, site, v, surveys, survey_annual, info["dense"],
                                                         info["min_rate_years"])
            # the published product: per-image waterlines of all optical sensors, then yearly medians
            yearly = _pick(summary["yearly"], engine=PUBLISHED[0], sensors=PUBLISHED[1])
            rates = _pick(summary["rates"], engine=PUBLISHED[0], sensors=PUBLISHED[1])
            perimage = _pick(summary.get("perimage"), group="all sensors", position="corrected")
            entry["methods"][v] = {"yearly": _keep(yearly), "rates": _keep(rates), "perimage": _keep(perimage),
                                   "transects": int(extras[v]["results"].transects["transect_id"].nunique())}
        if info["dense"] and site in benchmark.SITES:
            try:
                benchmark.download(cfg, site)
                ceva = {report.BENCHMARK_LABELS.get(v, v): extras[v]["results"].waterlines for v in done}
                seaward = benchmark.seaward_directions(extras["ceva"]["results"].transects)
                scores = benchmark.score(cfg, site, surveys, ceva, seaward)
                scores = scores[scores["scale"] == "per image"]
                entry["benchmark"] = [{"algorithm": r.algorithm, "sensors": r.sensors, "n": int(r.n),
                                       "rmse": _num(r.rmse, 2), "bias": _num(r.bias, 2), "std": _num(r.std, 2)}
                                      for r in scores.itertuples(index=False)]
            except Exception as error:                                   # the benchmark is optional context
                log.warning("%s: benchmark not scored (%s)", site, error)
        out.append(entry)
    return {"generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "sites": out}


def survey_ranges(sites: list[dict]) -> dict | None:
    """Smallest and largest error of CEVA's yearly positions and rates across the survey beaches
    (root mean square), for the caveat on each card."""
    yearly = [s["methods"]["ceva"]["yearly"]["rmse"] for s in sites if (s["methods"].get("ceva") or {}).get("yearly")]
    rates = [s["methods"]["ceva"]["rates"]["rmse"] for s in sites if (s["methods"].get("ceva") or {}).get("rates")]
    if not yearly or not rates:
        return None
    return {"yearly_rmse": [min(yearly), max(yearly)], "rate_rmse": [min(rates), max(rates)], "sites": len(sites)}


def sites_path(cfg: Config) -> Path:
    return cfg.data_dir / "reports" / "sites_summary.json"


def load_sites(cfg: Config) -> dict | None:
    p = sites_path(cfg)
    return json.loads(p.read_text()) if p.exists() else None
