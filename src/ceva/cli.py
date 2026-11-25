"""Command line entry point: ``ceva --help``."""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from ceva import __version__
from ceva.config import Config, load_config, packaged_regions
from ceva.credentials import TOKEN_FILES, load_credentials, load_local_settings, token_file, tokens_dir

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="CEVA: worldwide satellite-derived shoreline pipeline.")
transects_app = typer.Typer(no_args_is_help=True, help="Reference transects (GCTS).")
app.add_typer(transects_app, name="transects")
batch_app = typer.Typer(no_args_is_help=True, help="Many units in turn (a country, a continent): plan, run, status.")
app.add_typer(batch_app, name="batch")
waves_app = typer.Typer(no_args_is_help=True, help="ERA5 waves from the Copernicus Climate Data Store.")
app.add_typer(waves_app, name="waves")
references_app = typer.Typer(no_args_is_help=True,
                             help="Other shoreline-change products and the survey beaches, to compare CEVA with.")
app.add_typer(references_app, name="references")
vulnerability_app = typer.Typer(no_args_is_help=True,
                                help="The coastal vulnerability index of the published transects.")
app.add_typer(vulnerability_app, name="vulnerability")


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show progress messages.")] = False) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    load_local_settings()


def _region_config(region: str) -> Config:
    cfg = load_config(region)
    if cfg.region is None:
        raise typer.BadParameter(f"{region} does not define a region")
    return cfg


@app.command()
def info() -> None:
    """Show the version, data directory, credentials and provider status."""
    from ceva.catalog import provider_status
    from ceva.transects import gcts

    cfg = load_config()
    creds = load_credentials()
    typer.echo(f"ceva {__version__}")
    typer.echo(f"data dir     : {cfg.data_dir}{'' if cfg.data_dir.exists() else '  (does not exist yet)'}")
    typer.echo(f"token files  : {tokens_dir()}")
    status = creds.status()
    for service, keys in TOKEN_FILES.items():
        present = "present" if token_file(service).is_file() else "absent"
        keys_set = ", ".join(f"{k} {'set' if status[k] else 'missing'}" for k in keys)
        typer.echo(f"  {service + '.env':<24} {present}: {keys_set}")
    typer.echo(f"regions      : {', '.join(packaged_regions())}")
    for family in cfg.sensors:
        for level, label in (("surface", "surface reflectance"), ("toa", "top of atmosphere, for the chips")):
            status = provider_status(cfg, family, creds, level)
            if not status:
                continue
            typer.echo(f"{family} providers ({label}), in order of preference:")
            for name, why in status.items():
                typer.echo(f"  {name:<20} {'usable' if why is None else 'not usable: ' + why}")
    typer.echo(f"chip reflectance (configured): {cfg.chips.reflectance}")
    indexes = [p.stem for p in gcts.available_indexes(cfg)]
    typer.echo(f"GCTS {cfg.transects.gcts.release} indexed continents: {', '.join(indexes) or 'none'}")


@app.command()
def tiles(region: str) -> None:
    """List the processing tiles of a region and their transect counts."""
    from ceva.grid import tiles_for_bbox
    from ceva.transects import gcts

    cfg = _region_config(region)
    bbox = cfg.region.bbox_wgs84
    counts: dict[str, int] = {}
    try:
        counts = gcts.load_bbox(cfg, bbox)["tile"].value_counts().to_dict()
    except FileNotFoundError as exc:
        typer.echo(f"note: {exc}", err=True)
    typer.echo(f"{'tile':<14}{'EPSG':<8}{'west':>9}{'south':>9}{'east':>9}{'north':>9}{'transects':>11}")
    for tile in tiles_for_bbox(bbox, cfg.grid.processing_zoom):
        w, s, e, n = tile.bounds
        typer.echo(f"{tile.quadkey:<14}{tile.epsg:<8}{w:>9.3f}{s:>9.3f}{e:>9.3f}{n:>9.3f}"
                   f"{counts.get(tile.quadkey, 0):>11}")
    typer.echo(f"{sum(counts.values())} transects inside the {cfg.region.name} box")


@transects_app.command("prepare")
def transects_prepare(
    continent: Annotated[Optional[str], typer.Option(help="GCTS continent file, e.g. Europe.")] = None,
    region: Annotated[Optional[str], typer.Option(help="Use the continent of this region.")] = None,
    force: Annotated[bool, typer.Option(help="Download and index again.")] = False,
) -> None:
    """Download one GCTS continent file and build its tile index, and the coastal typology of its transects."""
    from ceva.transects import gcts, typology

    if (continent is None) == (region is None):
        raise typer.BadParameter("give exactly one of --continent or --region")
    cfg = _region_config(region) if region else load_config()
    name = continent or cfg.region.continent
    path = gcts.prepare(cfg, name, force=force)
    typer.echo(f"GCTS {name} ready: {path}")
    typer.echo(cfg.transects.gcts.attribution.strip())
    if cfg.transects.typology is not None:
        path = typology.prepare(cfg, name, force=force)
        typer.echo(f"coastal typology {name} ready: {path}")
        typer.echo(cfg.transects.typology.attribution.strip())


@app.command()
def scenes(
    region: str,
    family: Annotated[str, typer.Argument(help="Sensor family: landsat or sentinel2.")],
    start: Annotated[Optional[str], typer.Option(help="First day, YYYY-MM-DD.")] = None,
    end: Annotated[Optional[str], typer.Option(help="Last day, YYYY-MM-DD.")] = None,
    provider: Annotated[Optional[str], typer.Option(help="Force one provider.")] = None,
    out: Annotated[Optional[Path], typer.Option(help="Write the scene list to this Parquet file.")] = None,
) -> None:
    """Search the scenes available over a region and summarise them."""
    from ceva.catalog import scenes_to_frame, search_scenes

    cfg = _region_config(region)
    first = dt.date.fromisoformat(start) if start else cfg.time.start
    last = dt.date.fromisoformat(end) if end else cfg.time.end
    result = search_scenes(cfg, family, cfg.region.bbox_wgs84, first, last, provider=provider)

    for name, why in result.unusable_providers.items():
        typer.echo(f"provider {name} not usable: {why}")
    typer.echo(f"provider {result.provider}: {result.n_items} items, {len(result.scenes)} scenes kept "
               f"({first} to {last})")
    for reason, count in sorted(result.rejected.items()):
        typer.echo(f"  rejected {count:>5}  {reason}")

    frame = scenes_to_frame(result.scenes)
    if not frame.empty:
        table = (frame.assign(year=frame["datetime"].dt.year)
                 .pivot_table(index="year", columns="platform", values="id", aggfunc="count", fill_value=0))
        typer.echo(table.to_string())
        baselines = frame["processing_baseline"].dropna().value_counts().sort_index()
        if not baselines.empty:
            typer.echo("processing baselines: " + ", ".join(f"{b} ({c})" for b, c in baselines.items()))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out, index=False)
        typer.echo(f"wrote {out}")


def _progress(done: int, total: int) -> None:
    if done == total or done % 200 == 0:
        typer.echo(f"  {done}/{total} acquisitions read")


@app.command()
def process(
    regions: Annotated[list[str], typer.Argument(help="One or more regions; regions on the same Sentinel-2 tile "
                                                      "share their top-of-atmosphere downloads.")],
    start: Annotated[Optional[str], typer.Option(help="First day, YYYY-MM-DD.")] = None,
    end: Annotated[Optional[str], typer.Option(help="Last day, YYYY-MM-DD.")] = None,
    method: Annotated[str, typer.Option(help="Shoreline detection: ceva (production) or coastsat.")] = "ceva",
    rebuild: Annotated[bool, typer.Option(help="Read the imagery and radar again even if stored.")] = False,
) -> None:
    """Read the imagery (and radar) of regions on their GCTS transects; write the yearly shoreline products.

    The image chips of all the regions are read together, so that each Level-1C file is downloaded
    once for every region it covers.
    """
    from ceva import pipeline
    from ceva.chips import chips_path
    from ceva.cube import cube_path
    from ceva.sar import LEVEL, build_sar_chips
    from ceva.transects import gcts

    if method not in pipeline.VARIANTS:
        raise typer.BadParameter(f"--method must be one of {list(pipeline.VARIANTS)}")
    configs = {}
    for region in regions:
        cfg = pipeline.variant_config(_region_config(region), method)
        configs[cfg.region.name] = cfg
    first_cfg = next(iter(configs.values()))
    shared = ("chips", "sensors", "providers", "coastsat", "time", "data_dir")
    for name, cfg in configs.items():
        different = [k for k in shared if getattr(cfg, k) != getattr(first_cfg, k)]
        if different:
            raise typer.BadParameter(f"{name} has its own {', '.join(different)} settings: process it on its own")
    first = dt.date.fromisoformat(start) if start else None
    last = dt.date.fromisoformat(end) if end else None
    transects: dict[str, object] = {}

    def transects_of(name):
        if name not in transects:
            transects[name] = gcts.load_bbox(configs[name], configs[name].region.bbox_wgs84)
        return transects[name]

    for name, cfg in configs.items():
        if rebuild or not cube_path(cfg, name).exists():
            typer.echo(f"{name}: {len(transects_of(name))} transects; reading imagery along them")
            snapshot = pipeline.build(cfg, name, transects_of(name), progress=_progress, start=first, end=last)
            typer.echo(f"{name}: cube committed: {snapshot}")
    if pipeline.uses_chips(first_cfg):
        typer.echo(f"image chips for CoastSat's detection ({', '.join(configs)})")
        pipeline.build_chips_batch(first_cfg, {name: transects_of(name) for name in configs},
                                   only_missing=not rebuild, progress=_progress, start=first, end=last)
    for name, cfg in configs.items():
        if cfg.sar.enabled and (rebuild or not chips_path(cfg, name, LEVEL).exists()):
            typer.echo(f"{name}: radar chips ({', '.join(cfg.sar.missions)})")
            build_sar_chips(cfg, name, transects_of(name), progress=_progress, start=first, end=last)
    for name, cfg in configs.items():
        _report_run(pipeline, cfg, name, method)


def _report_run(pipeline, cfg, name: str, method: str) -> None:
    from ceva.products import PRODUCTS

    results = pipeline.run(cfg, name, method)
    out = pipeline.results_dir(cfg, name, method)
    oriented = int((results.transects["seaward_sign"] != 0).sum())
    typer.echo(f"{name}: {oriented} of {len(results.transects)} transects cross a shoreline; "
               f"{len(results.waterlines)} imagery waterlines"
               + (f", {int(results.radar['use'].sum())} radar waterlines" if results.radar is not None else ""))
    typer.echo(f"waves: {results.run['wave_component']} (ERA5 at {results.run['wave_point']})")
    summary = results.run.get("products", {})
    for product, label in PRODUCTS.items():
        if product in summary:
            p = summary[product]
            typer.echo(f"{product:14s} {p['transect_years']} transect-years, median standard error "
                       f"{p['median_se_m']:.1f} m ({label}): {out / f'shorelines_{product}.parquet'}")
    if "radar_only_transect_years" in summary:
        typer.echo(f"radar adds {summary['radar_only_transect_years']} transect-years the imagery alone does not have")


@app.command()
def validate(
    sites: Annotated[Optional[list[str]], typer.Argument(
        help="trucvert, narrabeen, jarkus, walcheren, perranporth, slapton (default: all).")] = None,
    variants: Annotated[Optional[list[str]], typer.Option(
        "--variant", help="Methods to compare: coastsat, ceva (default: both).")] = None,
    download: Annotated[bool, typer.Option(help="Fetch the survey data first.")] = False,
    rebuild: Annotated[bool, typer.Option(help="Read the imagery again even if a cube exists.")] = False,
    skip_run: Annotated[bool, typer.Option(help="Reuse stored results; only rewrite the report.")] = False,
    notes: Annotated[Optional[Path], typer.Option(help="HTML file with reviewed commentary for the page.")] = None,
    out: Annotated[Optional[Path], typer.Option(help="Where to write the report.")] = None,
    detail: Annotated[str, typer.Option(
        help="Method shown in the detailed charts.")] = "ceva",
    waterline: Annotated[bool, typer.Option(
        help="Add where the satellite waterline sits above the tide (needs at least three sites).")] = True,
) -> None:
    """Run the pipeline at the validation sites and write the interactive report."""
    import json

    from ceva import pipeline
    from ceva.cube import cube_path
    from ceva.validation import report

    cfg = load_config()
    chosen_variants = variants or list(pipeline.VARIANTS)
    if detail not in chosen_variants:
        raise typer.BadParameter(f"--detail {detail} must be one of the methods run")
    payloads = []
    for site in sites or list(report.SITE_INFO):
        if site not in report.SITE_INFO:
            raise typer.BadParameter(f"unknown site {site!r}; choose from {list(report.SITE_INFO)}")
        module = report.SITE_INFO[site]["module"]
        if download:
            typer.echo(f"{site}: downloading surveys")
            module.download(cfg)
        if rebuild or not cube_path(cfg, site).exists():
            typer.echo(f"{site}: reading imagery")
            pipeline.build(cfg, site, module.transects(cfg), progress=_progress)
        site_variants = [v for v in chosen_variants if report.SITE_INFO[site].get("chips", True)
                         or not pipeline.uses_chips(pipeline.variant_config(cfg, v))]
        chip_configs = {json.dumps(c.chips.model_dump(mode="json"), sort_keys=True): c
                        for c in (pipeline.variant_config(cfg, v) for v in site_variants) if pipeline.uses_chips(c)}
        if not skip_run:
            for chip_cfg in chip_configs.values():
                typer.echo(f"{site}: image chips for the CoastSat engine ({chip_cfg.chips.reflectance})")
                pipeline.build_chips(chip_cfg, site, module.transects(chip_cfg), only_missing=not rebuild,
                                     progress=_progress)
            if cfg.sar.enabled:
                from ceva.chips import chips_path
                from ceva.sar import LEVEL, build_sar_chips

                if rebuild or not chips_path(cfg, site, LEVEL).exists():
                    typer.echo(f"{site}: radar chips ({', '.join(cfg.sar.missions)})")
                    build_sar_chips(cfg, site, module.transects(cfg), progress=_progress)
        if not skip_run:
            for variant in site_variants:
                typer.echo(f"{site}: computing shorelines ({variant})")
                pipeline.run(pipeline.variant_config(cfg, variant), site, variant)
        available = [v for v in chosen_variants if (pipeline.results_dir(cfg, site, v) / "run.json").exists()]
        if len(available) < len(chosen_variants):
            typer.echo(f"{site}: no results for {sorted(set(chosen_variants) - set(available))}; left out")
        typer.echo(f"{site}: comparing with surveys")
        payloads.append(report.site_payload(cfg, site, available, detail))
    notes_html = notes.read_text(encoding="utf-8") if notes else None
    level_sites = [p["key"] for p in payloads if report.SITE_INFO[p["key"]].get("calibrate", True)]
    waterline_data = None
    if waterline and len(level_sites) >= 3:
        typer.echo("waterline height above the tide, all sites")
        # measured on the detections of the configured engine (the variant of the same name)
        source = cfg.perimage.engine if cfg.perimage.engine in chosen_variants else chosen_variants[0]
        waterline_data = report.waterline_payload(cfg, level_sites, source)
    path = report.write_report(payloads, out or cfg.data_dir / "reports" / "validation.html", notes_html=notes_html,
                               waterline=waterline_data)
    typer.echo(f"report: {path}")


@app.command("calibrate-waterline")
def calibrate_waterline(
    sites: Annotated[Optional[list[str]], typer.Argument(
        help="Sites with surveys (default: every validation site).")] = None,
    variant: Annotated[str, typer.Option(help="Method whose waterlines are measured: ceva or coastsat.")] = "ceva",
) -> None:
    """Fit the height of the satellite waterline above the tide to the surveys; test it site by site."""
    import json

    import pandas as pd

    from ceva.validation import waterline_level

    cfg = load_config()
    from ceva.validation.report import SITE_INFO

    chosen = sites or [k for k, info in SITE_INFO.items() if info.get("calibrate", True)]
    tables = {}
    for site in chosen:
        typer.echo(f"{site}: waterlines matched to surveys (survey contours are cached after the first run)")
        tables[site] = waterline_level.elevation_table(cfg, site, variant)
        dz = tables[site]["dz"].mean()
        typer.echo(f"  {len(tables[site])} waterlines, mean height needed above the tide {dz:+.2f} m")
    loso = waterline_level.leave_one_site_out(cfg, tables)
    typer.echo("\nEach site predicted by a model fitted on the others (shoreline error, m):")
    columns = ["site", "correction", "n", "bias", "std", "rmse", "dz_observed", "dz_predicted"]
    typer.echo(loso[columns].round(2).to_string(index=False))
    coef = waterline_level.fit(pd.concat(tables.values(), ignore_index=True))
    fitted = {"hs_coef": round(float(coef[0]), 3)}
    typer.echo(f"\nFitted on all sites: {fitted} (put this under waves.empirical in the configuration)")
    offsets = waterline_level.site_offsets(tables)
    typer.echo("\nMean offset each beach keeps (what a beach-type model would have to explain):")
    typer.echo(offsets.round(3).to_string(index=False))
    models = waterline_level.offset_models(tables) if len(tables) >= 4 else pd.DataFrame()
    if not models.empty:
        typer.echo("\nPredicting each beach's offset from one descriptor, fitted on the other beaches "
                   "(shoreline error left, m):")
        gained = models.assign(gain=models["error_none_m"].abs() - models["error_m"].abs())
        summary = gained.groupby("descriptor").agg(
            mean_abs_error_none=("error_none_m", lambda v: v.abs().mean()),
            mean_abs_error=("error_m", lambda v: v.abs().mean()),
            sites_improved=("gain", lambda v: int((v > 0).sum())), sites=("gain", "size"))
        typer.echo(summary.round(2).to_string())
    out = cfg.data_dir / "reports" / "waterline_model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"variant": variant, "sites": chosen, "fitted": fitted,
                               "leave_one_site_out": loso.round(3).to_dict(orient="records"),
                               "site_offsets": offsets.round(3).to_dict(orient="records"),
                               "offset_models": models.round(3).to_dict(orient="records")}, indent=2))
    typer.echo(f"written: {out}")


def _countries(value: str) -> list[str] | None:
    from ceva.batch import EU27

    if value.lower() == "all":
        return None
    return list(EU27) if value.upper() == "EU27" else [c.strip().upper() for c in value.split(",") if c.strip()]


@batch_app.command("plan")
def batch_plan(
    run: Annotated[str, typer.Argument(help="Name of the run, e.g. eu27.")],
    countries: Annotated[str, typer.Option(help="EU27, ISO codes separated by commas (FR,PT), or all.")] = "EU27",
    max_transects: Annotated[Optional[int], typer.Option(help="Most transects in a unit.")] = None,
    shore_types: Annotated[Optional[str], typer.Option(
        help="Coastal Typology sediment classes, separated by commas, or 'all' (default: batch.shore_types).")] = None,
    area: Annotated[Optional[str], typer.Option(
        help="Only the transects inside this area: a GeoJSON file, or the name of one in ceva/config/areas "
             "(europe-mediterranean).")] = None,
    after: Annotated[Optional[str], typer.Option(
        help="Earlier runs this one extends (separated by commas): their transects are left out, their unit "
             "names never reused.")] = None,
) -> None:
    """Cut the transects of some countries into units and write the plan; print what the run will store."""
    import pandas as pd

    from ceva.batch import load_area, load_plan, plan_units, storage_estimate, write_plan

    cfg = load_config()
    kinds = None if shore_types is None else ([] if shore_types == "all" else shore_types.split(","))
    plan = plan_units(cfg, _countries(countries), max_transects, kinds, area=load_area(area) if area else None,
                      after=pd.concat([load_plan(cfg, r.strip()) for r in after.split(",")]) if after else None)
    path = write_plan(cfg, run, plan)
    n = int(plan["n_transects"].sum())
    est = storage_estimate(n)
    typer.echo(f"{len(plan)} units on {plan['tile'].nunique()} tiles, {n} transects: {path}")
    typer.echo(f"to store: cubes {est['cubes'][0] / 1e12:.2f}-{est['cubes'][1] / 1e12:.2f} TB, "
               f"results {est['results'][0] / 1e9:.0f}-{est['results'][1] / 1e9:.0f} GB")


def _years(first: Optional[int], last: Optional[int], cfg) -> range:
    return range(first or cfg.time.start.year, (last or cfg.time.end.year) + 1)


@waves_app.command("download")
def waves_download(
    first_year: Annotated[Optional[int], typer.Option(help="First year (default: time.start).")] = None,
    last_year: Annotated[Optional[int], typer.Option(help="Last year (default: time.end).")] = None,
) -> None:
    """Download the hourly ERA5 waves of ``waves.cds.area``, one request a year (resumes where it stopped)."""
    from ceva import era5

    cfg = load_config()
    years = _years(first_year, last_year, cfg)
    try:
        paths = era5.download(cfg, years, progress=lambda k, n: typer.echo(f"  {k}/{n} years"))
    except Exception as exc:
        if "licence" in str(exc).lower():
            raise typer.BadParameter(f"accept the ERA5 licence once: {era5.LICENCE_PAGE}") from exc
        raise
    typer.echo(f"{len(paths)} years in {paths[0].parent}")


@waves_app.command("build")
def waves_build(
    first_year: Annotated[Optional[int], typer.Option(help="First year (default: time.start).")] = None,
    last_year: Annotated[Optional[int], typer.Option(help="Last year (default: time.end).")] = None,
) -> None:
    """Write the coastal sea cells of the downloaded years into the local wave store."""
    from ceva import era5

    cfg = load_config()
    path = era5.build_store(cfg, _years(first_year, last_year, cfg))
    typer.echo(f"wave store: {path}")
    typer.echo(era5.ATTRIBUTION)


def _run_units(cfg, run: str | None, units: str | None, only: str | None = None) -> tuple[list[str], dict | None]:
    """The units named, plus the finished units of one or more batch runs (names separated by commas, e.g.
    a run and the run that extends it), and how far those runs are.

    ``only`` keeps the units it lists: names separated by commas, or ``@file`` with one per line.
    """
    from ceva.batch import load_plan, read_status

    names = [u.strip() for u in (units or "").split(",") if u.strip()]
    info = None
    runs = [r.strip() for r in (run or "").split(",") if r.strip()]
    if runs:
        info = {"name": "+".join(runs), "planned": 0, "processed": 0, "failed": 0, "empty": 0}
        for one in runs:
            status = read_status(cfg, one)
            names += sorted(u for u, e in status.items() if e["state"] in ("processed", "uploaded"))
            states = [e["state"] for u, e in status.items() if not u.startswith("_")]
            info["planned"] += int(load_plan(cfg, one)["name"].nunique())
            info["processed"] += sum(s in ("processed", "uploaded") for s in states)
            info["failed"] += sum(s == "failed" for s in states)
            info["empty"] += sum(s == "empty" for s in states)
        info["finished"] = info["processed"] + info["failed"] + info["empty"]
        if len(set(names)) != len(names):
            raise typer.BadParameter(f"runs {runs} share unit names: plan the later one with --after")
    if only:
        text = Path(only[1:]).read_text() if only.startswith("@") else only
        keep = {u.strip() for u in text.replace(",", "\n").splitlines() if u.strip()}
        names = [n for n in names if n in keep]
        if not names:
            raise typer.BadParameter("none of the units of --only is finished or named")
    if not names:
        raise typer.BadParameter("give --run and/or --units")
    return names, info


@app.command()
def publish(
    out: Annotated[Path, typer.Argument(help="The dashboard's data folder, e.g. ../ceva-dashboard/data.")],
    run: Annotated[Optional[str], typer.Option(help="Publish the finished units of these batch runs "
                                                    "(names separated by commas).")] = None,
    units: Annotated[Optional[str], typer.Option(help="And/or these units or regions, separated by commas.")] = None,
    method: Annotated[str, typer.Option(help="Shoreline detection whose results are published.")] = "ceva",
    area: Annotated[Optional[str], typer.Option(
        help="Also draw the coast of this area that is not measured (other coast types, no result): a GeoJSON "
             "file or the name of one in ceva/config/areas (europe-mediterranean).")] = None,
    references: Annotated[bool, typer.Option(help="Compare with the prepared reference products "
                                                  "(cards and validation page).")] = True,
    only: Annotated[Optional[str], typer.Option(
        help="Only these units, separated by commas, or @file with one per line: a preview of some "
             "areas before a run is finished (the site says it is a preview).")] = None,
) -> None:
    """Write the dashboard's data (map tiles, series, summary, manifest, comparisons) from the results of units."""
    from ceva import publish as pub
    from ceva.batch import load_area

    cfg = load_config()
    names, info = _run_units(cfg, run, units, only)
    manifest = pub.publish(cfg, out, names, method, references=references, run=info,
                           area=load_area(area) if area else None)
    typer.echo(f"{manifest['transects']} transects of {manifest['units']} units, {manifest['years'][0]}-"
               f"{manifest['years'][1]}: {out}")
    if manifest.get("references"):
        typer.echo(f"compared with: {', '.join(r['key'] for r in manifest['references'])}"
                   f"{' (provisional: the run is not finished)' if manifest.get('provisional') else ''}")


frames_app = typer.Typer(no_args_is_help=True, help="The satellite images behind the shorelines, for the "
                                                    "dashboard's image viewer: build, then pack.")
app.add_typer(frames_app, name="frames")


@frames_app.command("build")
def frames_build(
    names: Annotated[Optional[list[str]], typer.Argument(help="Units or survey sites with stored results, each "
                                                              "optionally named for the viewer: NAME=Label.")] = None,
    run: Annotated[Optional[str], typer.Option(help="And all the processed units of this batch run.")] = None,
    shard: Annotated[str, typer.Option(help="K/N: only every N-th unit, from the K-th (N processes side "
                                            "by side).")] = "1/1",
    workers: Annotated[int, typer.Option(min=1, help="Parallel image reads.")] = 8,
    jobs: Annotated[int, typer.Option(min=1, help="Units built side by side, each in its own process (about "
                                                  "one core and up to 2 GB each).")] = 1,
) -> None:
    """Cut every image used, 2 km of coast at a time: a video (H.264) and a JSON file of the detected and
    corrected waterlines per segment, in <data dir>/frames/work. Units already built are skipped, and so
    are units another process is building, so a stopped build resumes where it was."""
    from ceva import frames as fr

    cfg = load_config()
    work = fr.work_dir(cfg)
    items = [n.partition("=") for n in names or []]
    labels = {name: label or name for name, _, label in items}
    if run:
        names_run, _ = _run_units(cfg, run, None)
        labels.update({u: labels.get(u, u) for u in names_run})
    k, _, n = shard.partition("/")
    k, n = int(k), int(n or 1)
    if not 1 <= k <= n:
        raise typer.BadParameter("--shard K/N with 1 <= K <= N")
    todo = sorted(labels)[k - 1::n]
    if len(todo) == 1:                            # one unit: built here
        name = todo[0]
        if not fr.is_built(work, name):
            fr.mark_built(work, name, labels[name], fr.build(cfg, name, work, workers=workers))
        return
    import sys

    def command(name: str) -> list[str]:
        return [sys.executable, "-I", "-c", "from ceva.cli import app; app()", "frames", "build",
                f"{name}={labels[name]}", "--workers", str(workers)]

    done = _build_side_by_side(work, todo, command, jobs, cfg.batch.min_free_memory_gb)
    typer.echo(f"{done} of {len(todo)} units built" + (f" (shard {k}/{n})" if n > 1 else ""))


def _building_elsewhere(skip: set[int]) -> set[str]:
    """The units other processes are building now: those of a ``frames build NAME=Label ...`` command line,
    such as the unit processes of a stopped build, which go on to the end of their unit."""
    import psutil

    units: set[str] = set()
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = p.info["cmdline"] or []
        if p.info["pid"] in skip or "frames" not in cmd:
            continue
        i = cmd.index("frames")
        if cmd[i + 1:i + 2] == ["build"]:
            units.update(a.partition("=")[0] for a in cmd[i + 2:] if "=" in a and not a.startswith("-"))
    return units


def _build_side_by_side(work: Path, todo: list[str], command, jobs: int, min_free_gb: float,
                        poll: float = 5.0) -> int:
    """Build ``todo`` with up to ``jobs`` units at a time, each in a fresh process (``command(name)``), whose
    memory goes back to the machine when it ends. A unit starts only while the machine keeps ``min_free_gb``
    for its other users, and not while another process builds it; it is then waited for, and built here if
    that process ends without finishing it. Returns how many of ``todo`` are built."""
    import os
    import subprocess
    import time

    import psutil

    from ceva import frames as fr

    env = {**os.environ, "MALLOC_ARENA_MAX": "2", "GDAL_CACHEMAX": "256"}
    queue = [name for name in todo if not fr.is_built(work, name)]
    done = len(todo) - len(queue)
    running: dict[str, tuple[subprocess.Popen, float]] = {}
    while queue or running:
        for name, (proc, started) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[name]
            if proc.returncode != 0 or not fr.is_built(work, name):
                typer.echo(f"{name}: failed (exit {proc.returncode})")
                continue
            done += 1
            record = json.loads((work / f"{name}.done.json").read_text())["segments"]
            images = sum(s["frames"] for s in record)
            size = sum(s["bytes"]["video"] + s["bytes"]["json"] for s in record if "bytes" in s)
            typer.echo(f"{name}: {len(record)} segments, {images} images, {size / 1e6:.1f} MB, "
                       f"{(time.time() - started) / 60:.1f} min ({done}/{len(todo)})")
        if queue and len(running) < jobs and psutil.virtual_memory().available >= min_free_gb * 1e9:
            busy = _building_elsewhere({os.getpid(), *(proc.pid for proc, _ in running.values())})
            name = next((name for name in queue if name not in busy), None)
            if name is not None:
                queue.remove(name)
                if fr.is_built(work, name):                       # finished meanwhile by another process
                    done += 1
                else:
                    running[name] = (subprocess.Popen(command(name), env=env), time.time())
                continue                                          # fill the other free places now
        time.sleep(poll)
    return done


@frames_app.command("pack")
def frames_pack(
    dest: Annotated[Path, typer.Argument(help="Where the archives and their indexes go (the dashboard's frames/).")],
    by_country: Annotated[bool, typer.Option(help="One archive per country (Zenodo: at most 100 files).")] = True,
) -> None:
    """Put the built segments into uncompressed ZIPs with their indexes: index.json (the units with images)
    and units/<unit>.json (each segment's bytes in its archive), read by the viewer with range requests."""
    from ceva import frames as fr

    cfg = load_config()
    index = fr.pack_all(cfg, fr.work_dir(cfg), dest, by_country=by_country)
    total = sum(index["archives"].values())
    typer.echo(f"{len(index['units'])} units, {sum(u['segments'] for u in index['units'].values())} segments in "
               f"{len(index['archives'])} archives, {total / 1e9:.2f} GB: {dest}")


@vulnerability_app.command("prepare")
def vulnerability_prepare() -> None:
    """Get the index's sources: GEBCO 2024 over the area, the glacial-rebound grid, the sea-level trends."""
    from ceva import vulnerability

    for name, where in vulnerability.prepare(load_config()).items():
        typer.echo(f"{name}: {where}")


@vulnerability_app.command("compute")
def vulnerability_compute(
    run: Annotated[Optional[str],
                   typer.Option(help="The finished units of these batch runs (separated by commas).")] = None,
    units: Annotated[Optional[str], typer.Option(help="And/or these units or regions, separated by commas.")] = None,
    method: Annotated[str, typer.Option(help="Shoreline detection whose rates are used.")] = "ceva",
) -> None:
    """Compute the index of every published transect of these units; publish and export read it."""
    from ceva import vulnerability

    cfg = load_config()
    names, _ = _run_units(cfg, run, units)
    vulnerability.prepare(cfg)
    table = vulnerability.compute(cfg, names, method)
    where = vulnerability.write(cfg, table)
    known = table["cvi"].notna()
    typer.echo(f"{int(known.sum())} of {len(table)} transects with an index; class edges "
               f"{[round(e, 2) for e in table.attrs['class_edges']]}: {where}")


@references_app.command("prepare")
def references_prepare(
    only: Annotated[Optional[str], typer.Option(help="These products only, separated by commas "
                                                     "(sm, emodnet_sat, emodnet_field).")] = None,
    force: Annotated[bool, typer.Option(help="Download and rebuild even if prepared.")] = False,
) -> None:
    """Download the products CEVA is compared with and keep what the comparison needs."""
    from ceva.validation import references as refs

    cfg = load_config()
    keys = [k.strip() for k in only.split(",")] if only else None
    for key, n in refs.prepare(cfg, keys, force).items():
        p = refs.PRODUCTS[key]
        typer.echo(f"{key}: {n} features ({p.name}, {p.provider}, {p.period}; {p.licence})")


@references_app.command("sites")
def references_sites(
    sites: Annotated[Optional[list[str]], typer.Option(
        "--site", help="Survey beaches (default: all with results).")] = None,
) -> None:
    """Score the stored results of the survey beaches for the validation page (no processing)."""
    from ceva.validation import agreement

    cfg = load_config()
    data = agreement.sites_summary(cfg, sites)
    path = agreement.sites_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
    for site in data["sites"]:
        m = site["methods"]["ceva"]
        typer.echo(f"{site['key']}: yearly RMSE {m['yearly']['rmse'] if m['yearly'] else '-'} m, "
                   f"rate RMSE {m['rates']['rmse'] if m['rates'] else '-'} m/yr")
    typer.echo(f"written: {path}")


@app.command()
def export(
    out: Annotated[Path, typer.Argument(help="Folder of the dataset (one Zenodo record).")],
    run: Annotated[Optional[str], typer.Option(help="Export the finished units of these batch runs "
                                                    "(names separated by commas).")] = None,
    units: Annotated[Optional[str], typer.Option(help="And/or these units or regions, separated by commas.")] = None,
    version: Annotated[str, typer.Option(help="Version of the dataset.")] = "1.0",
    name: Annotated[str, typer.Option(help="File name prefix.")] = "ceva-shorelines",
    place: Annotated[Optional[str], typer.Option(help="The coasts, as named in the titles.")] = None,
    coverage: Annotated[Optional[str], typer.Option(help="Which transects are in, in one sentence.")] = None,
) -> None:
    """Write the full dataset for download and citation: rates, yearly shorelines and waterlines by country."""
    from ceva import export as exp

    cfg = load_config()
    names, _ = _run_units(cfg, run, units)
    summary = exp.export(cfg, out, names, name, version, place=place or exp.PLACE, coverage=coverage or exp.COVERAGE)
    size = sum(p.stat().st_size for p in Path(out).iterdir() if p.is_file())
    typer.echo(f"{summary['units']} units, {len(summary['countries'])} countries, {size / 1e9:.2f} GB: {out}")


@app.command()
def keepalive() -> None:
    """Refresh the dates of the environment and of the data directory (for disks that delete old files).

    Runs during batch runs by themselves; when nothing runs, run this at least once a week.
    """
    from ceva.batch import keepalive as touch

    typer.echo(f"{touch(load_config())} files refreshed")


@batch_app.command("run")
def batch_run(
    run: Annotated[str, typer.Argument(help="Name of a planned run.")],
    method: Annotated[str, typer.Option(help="Shoreline detection: ceva or coastsat.")] = "ceva",
    limit: Annotated[Optional[int], typer.Option(help="Stop after this many units.")] = None,
    workers: Annotated[int, typer.Option(min=1, help="Worker processes side by side, each on its own tiles, "
                                                     "within batch.memory_budget_gb (no online store yet).")] = 1,
    worker: Annotated[Optional[int], typer.Option(hidden=True, help="Run as worker K of a coordinator.")] = None,
) -> None:
    """Process the units of a plan in turn; upload each to the online store; resume where it stopped.

    Create a file named STOP in <data dir>/batch/<run>/ to stop after the current unit(s).
    """
    from ceva.batch import Coordinator, Runner
    from ceva.storage import remote_store

    cfg = load_config()
    if worker is not None:
        raise typer.Exit(Runner(cfg, run, method, worker=worker).work())
    store = remote_store(cfg)
    if workers > 1:
        if store is not None or limit is not None:
            raise typer.BadParameter("--workers runs without an online store and without --limit")
        typer.echo(f"{workers} workers, memory budget {cfg.batch.memory_budget_gb:.0f} GB")
        summary = Coordinator(cfg, run, method, workers).run()
    else:
        typer.echo(f"online store: {store if store is not None else 'none yet (units wait locally)'}")
        summary = Runner(cfg, run, method).run(limit)
    typer.echo(f"units: {summary}")


@batch_app.command("status")
def batch_status(run: Annotated[str, typer.Argument(help="Name of a planned run.")]) -> None:
    """Units done, waiting, failed; transects, time per unit and the bytes stored."""
    from ceva.batch import RECENT_UNITS, load_plan, read_status, recent_pace

    cfg = load_config()
    plan, state = load_plan(cfg, run), read_status(cfg, run)
    counts: dict[str, int] = {}
    for event in state.values():
        counts[event["state"]] = counts.get(event["state"], 0) + 1
    done = [e for e in state.values() if e["state"] in ("processed", "uploaded")]
    n_done = sum(int(plan.set_index("name").loc[e["unit"], "n_transects"]) for e in done)
    typer.echo(f"{len(plan)} units, {int(plan['n_transects'].sum())} transects; states: {counts}")
    pace = recent_pace(plan, state)
    if pace is not None:
        units_per_hour, left = pace
        typer.echo(f"{n_done} transects done; last {RECENT_UNITS} units: {units_per_hour:.0f} units/h; "
                   f"about {left / 86400:.1f} days left")
    uploaded = sum(e.get("bytes", 0) for e in state.values() if e["state"] == "uploaded")
    waiting = sum(e.get("local_bytes", 0) for e in state.values() if e["state"] == "processed")
    typer.echo(f"online: {uploaded / 1e9:.1f} GB; waiting locally: {waiting / 1e9:.1f} GB")
    failed = [e for e in state.values() if e["state"] == "failed"]
    for e in failed[:10]:
        typer.echo(f"  failed {e['unit']}: {e.get('error', '')[:150]}")


if __name__ == "__main__":
    app()
