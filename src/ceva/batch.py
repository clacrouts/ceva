"""Many units in turn (a country, a continent): plan, run, upload, resume.

A unit is one processing tile (``grid.processing_zoom``, about 28 km wide at 45°N), cut along the
coast into groups of at most ``batch.max_transects`` consecutive GCTS transects, so that a unit's
cube fits in memory. Each unit is processed on its own (its cube, then the chosen method). Once its
results are written, they stay in the data directory, or, with an online store (``ceva.storage``),
are uploaded, checked there and deleted locally. Locally, the run pauses while the (shared) disk has
less than ``storage.min_free_gb`` free, and keeps a unit's cube only while more than
``storage.keep_cubes_above_gb`` is free (results are always kept; a cube can be rebuilt). After
``batch.max_consecutive_failures`` failures in a row (a service refusing requests) it pauses longer
each time, then stops.

Everything about a run sits in ``<data dir>/batch/<run>/``: ``plan.parquet`` (the units), and
``status.jsonl`` (one line per event; the last line of a unit is its state), so a run can be
stopped (create a file named ``STOP`` there) and resumed with the same command. Long runs also
refresh the dates of the files they need, which a disk that deletes old files would otherwise remove.

Several workers (``Coordinator``, ``ceva batch run --workers N``) process tiles side by side, each
tile in a fresh process that exits once the tile is done, so its memory goes back to the machine;
the units of a tile stay together so that they share the image blocks they read. A worker claims a
tile with a file in ``claims/`` (created atomically, so no two workers take the same tile) and
writes its events to ``status-w<k>.jsonl``; ``read_status`` merges them. Memory is shared with other
users of the machine: a worker starts a unit only while the machine keeps ``batch.min_free_memory_gb``
available and the other busy workers use less than ``START_SHARE`` of ``batch.memory_budget_gb``;
beyond the budget (or below ``MEMORY_FLOOR_GB`` available) the coordinator stops the newest unit,
which is retried at the end, once every tile has been taken.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from ceva.config import Config
from ceva.shoreline import NoShoreline

log = logging.getLogger(__name__)

EU27 = ("AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE AX").split()
AREAS_DIR = Path(__file__).parent / "config" / "areas"
BYTES_PER_TRANSECT_ACQUISITION = 634     # the cube at Truc Vert: 302 MB for 237 transects x 2009 acquisitions
RESULT_BYTES_PER_TRANSECT = 37_000       # results of the CEVA solution at Truc Vert: 8.8 MB for 237 transects
POLL_S = 600
FAILURE_PAUSES_S = (600, 1200, 2400, 4800)
TOUCH_EVERY_S = 86_400
# several workers
TILE_DONE = 10               # a worker's exit code: one tile done, there may be more
WORKER_STAGGER_S = 60        # workers start one after another
WATCH_S = 5                  # the coordinator's polling interval
MEMORY_POLL_S = 30           # a worker waiting for memory looks again after this long
RESTART_PAUSE_S = 60         # before a stopped or crashed worker is started again
MAX_CRASHES = 20             # worker crashes (not memory stops) after which the run stops
START_SHARE = 0.6            # a unit starts only while the other busy workers use less than this share of the budget
MEMORY_FLOOR_GB = 4.0        # below this much available on the machine, the newest unit is stopped
MEMORY_STOP = "worker stopped: over the memory budget"
RECENT_UNITS = 40            # the time left is estimated at the pace of the last units done


def batch_dir(cfg: Config, run: str) -> Path:
    return cfg.data_dir / "batch" / run


def load_area(value: str | Path) -> list[tuple[frozenset[str] | None, object]]:
    """An area to plan: a GeoJSON file of lon/lat polygons, or the name of one in ``ceva/config/areas``.

    A feature can be limited to some GCTS continents (property ``continents``, e.g. ``["AF"]``), so that
    a box drawn around a sea keeps one side of it only.
    """
    import shapely.geometry

    path = Path(value)
    if not path.suffix:
        path = AREAS_DIR / f"{value}.geojson"
    features = json.loads(path.read_text())["features"]
    return [(frozenset(f.get("properties", {}).get("continents") or ()) or None, shapely.geometry.shape(f["geometry"]))
            for f in features]


def in_area(area, lon, lat, continent) -> np.ndarray:
    """Which transect origins lie in the area (and on one of the continents its polygon allows)."""
    import shapely

    lon, lat, continent = np.asarray(lon, float), np.asarray(lat, float), np.asarray(continent, object)
    keep = np.zeros(lon.size, bool)
    for continents, polygon in area:
        on = np.ones(lon.size, bool) if continents is None else np.isin(continent, list(continents))
        keep |= on & shapely.contains_xy(polygon, lon, lat)
    return keep


def plan_units(cfg: Config, countries: list[str] | None, max_transects: int | None = None,
               shore_types: list[str] | None = None, area=None, after: pd.DataFrame | None = None) -> pd.DataFrame:
    """The units of the transects of ``countries`` (ISO alpha-2 codes, None for all): name, tile, n_transects,
    transect_ids.

    Only the transects whose Global Coastal Typology sediment class is in ``shore_types`` (default
    ``batch.shore_types``; all transects when empty) are kept, and only those inside ``area``
    (``load_area``) when one is given. ``after`` is the plan of an earlier run that this one extends: its
    transects are left out, and in a tile they share, this run's units are numbered after its units, so
    that a unit name always means the same transects.
    """
    from ceva.transects.gcts import INDEX_KEY, available_indexes

    max_transects = max_transects or cfg.batch.max_transects
    shore_types = cfg.batch.shore_types if shore_types is None else shore_types
    columns = ["transect_id", "country", INDEX_KEY] + (["lon", "lat", "continent"] if area is not None else [])
    frames = [pd.read_parquet(p, columns=columns) for p in available_indexes(cfg)]
    if not frames:
        raise FileNotFoundError("no GCTS index: run `ceva transects prepare --continent <name>` first")
    t = pd.concat(frames, ignore_index=True)
    if countries is not None:
        t = t[t["country"].isin(countries)]
    if area is not None:
        t = t[in_area(area, t["lon"], t["lat"], t["continent"])]
    first = {}
    if after is not None:
        t = t[~t["transect_id"].isin({i for ids in after["transect_ids"] for i in ids})]
        numbers = after["name"].str.rsplit("-", n=1).str[1].astype(int)
        first = (numbers.groupby(after["tile"]).max() + 1).to_dict()
    if shore_types:
        from ceva.transects import typology

        kinds = typology.load(cfg)[["transect_id", "shore_type"]]
        t = t.merge(kinds, on="transect_id", how="inner")
        t = t[t["shore_type"].isin(shore_types)]
    t["tile"] = t[INDEX_KEY].str[:cfg.grid.processing_zoom]
    rows = []
    for tile, g in t.sort_values(["tile", "transect_id"]).groupby("tile", sort=True):
        ids = g["transect_id"].astype(str).tolist()       # GCTS ids run along each coastline
        parts = math.ceil(len(ids) / max_transects)
        size = math.ceil(len(ids) / parts)
        for k in range(parts):
            chunk = ids[k * size:(k + 1) * size]
            rows.append({"name": f"{tile}-{first.get(tile, 0) + k}", "tile": tile, "n_transects": len(chunk),
                         "transect_ids": chunk})
    return pd.DataFrame(rows, columns=["name", "tile", "n_transects", "transect_ids"])


def storage_estimate(n_transects: int, acquisitions: tuple[int, int] = (2000, 2600)) -> dict[str, tuple[float, float]]:
    """Bytes the run will store: cubes and results (low and high, by the acquisitions per transect)."""
    cubes = tuple(n_transects * a * BYTES_PER_TRANSECT_ACQUISITION for a in acquisitions)
    results = tuple(n_transects * RESULT_BYTES_PER_TRANSECT * a / acquisitions[0] for a in acquisitions)
    return {"cubes": cubes, "results": results}


def write_plan(cfg: Config, run: str, plan: pd.DataFrame) -> Path:
    path = batch_dir(cfg, run) / "plan.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_parquet(path, index=False)
    return path


def load_plan(cfg: Config, run: str) -> pd.DataFrame:
    path = batch_dir(cfg, run) / "plan.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no plan for {run!r}: run `ceva batch plan {run}` first")
    return pd.read_parquet(path)


def read_status(cfg: Config, run: str) -> dict[str, dict]:
    """The last event of each unit, from ``status.jsonl`` and the workers' ``status-w<k>.jsonl``."""
    folder = batch_dir(cfg, run)
    events = []
    for path in [folder / "status.jsonl", *sorted(folder.glob("status-w*.jsonl"))]:
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:           # empty, or cut short by a stopped worker
                    continue
    events.sort(key=lambda e: e.get("time", ""))       # stable: file order within a second
    return {event["unit"]: event for event in events if "unit" in event}


def recent_pace(plan: pd.DataFrame, state: dict[str, dict]) -> tuple[float, float] | None:
    """Units per hour and seconds left, at the pace of the last ``RECENT_UNITS`` units done.

    A unit costs a fixed time (the image searches and reads) plus a time per transect, fitted on those
    units; the units that ran side by side (``--workers``) share the wall-clock time. Units still to do
    exclude the failed ones, retried at the end.
    """
    sizes = plan.set_index("name")["n_transects"].astype(float)
    finished = {u for u, e in state.items() if e["state"] in ("processed", "uploaded", "failed", "empty")}
    timed = sorted((e for e in state.values() if e["state"] in ("processed", "uploaded") and "cube_s" in e
                    and e["unit"] in sizes.index), key=lambda e: e["time"])[-RECENT_UNITS:]
    if len(timed) < 2:
        return None
    n = sizes[[e["unit"] for e in timed]].to_numpy()
    seconds = np.array([e["cube_s"] + e.get("run_s", 0) for e in timed])
    end = np.array([dt.datetime.fromisoformat(e["time"]).timestamp() for e in timed])
    span = end.max() - (end - seconds).min()
    if span <= 0:
        return None
    side_by_side = seconds.sum() / span                 # below 1 when units waited between them
    per_transect, fixed = np.polyfit(n, seconds, 1) if np.ptp(n) > 0 else (0.0, seconds.mean())
    per_transect, fixed = max(per_transect, 0.0), max(fixed, 0.0)
    left = sizes[~sizes.index.isin(finished)]
    return len(timed) / span * 3600, float((fixed * len(left) + per_transect * left.sum()) / side_by_side)


class Runner:
    """Processes the units of a plan in turn; the pipeline and the store can be replaced in tests."""

    def __init__(self, cfg: Config, run: str, method: str = "ceva", *, store_factory: Callable | None = None,
                 pipeline=None, sleep: Callable[[float], None] = time.sleep, worker: int | None = None):
        from ceva import pipeline as default_pipeline
        from ceva.storage import remote_store

        self.pipeline = pipeline or default_pipeline
        self.cfg = self.pipeline.variant_config(cfg, method)
        self.run_name, self.method = run, method
        self.dir = batch_dir(cfg, run)
        self.store_factory = store_factory or (lambda: remote_store(self.cfg))
        self.sleep = sleep
        self.worker = worker
        if worker is not None:                        # several workers share the image servers
            reads = min(self.cfg.cube.workers, self.cfg.batch.worker_reads)
            self.cfg = self.cfg.model_copy(update={"cube": self.cfg.cube.model_copy(update={"workers": reads})})
        self.status_file = self.dir / ("status.jsonl" if worker is None else f"status-w{worker}.jsonl")
        self.state = read_status(cfg, run)
        self._tile_cache: tuple[str, object] | None = None
        self._touched = 0.0

    # ------------------------------------------------------------------ bookkeeping
    def _record(self, unit: str, state: str, **info) -> None:
        event = {"unit": unit, "state": state, "time": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), **info}
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
        self.state[unit] = event

    def _local_paths(self, name: str) -> list[Path]:
        from ceva.cube import cube_path

        return [cube_path(self.cfg, name), self.cfg.data_dir / "results" / name]

    def _waiting_bytes(self) -> int:
        return sum(e.get("local_bytes", 0) for e in self.state.values() if e["state"] in ("processed", "failed"))

    def _free_bytes(self) -> int:
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(self.cfg.data_dir).free

    def _wait_for_disk(self) -> bool:
        """Pause while the data disk is nearly full; False if a stop came first."""
        while (free := self._free_bytes()) < self.cfg.storage.min_free_gb * 1e9:
            log.warning("%.0f GB free on the data disk (less than %.0f GB): pausing", free / 1e9,
                        self.cfg.storage.min_free_gb)
            if self.stop_requested():
                return False
            self.touch_inputs()
            self.sleep(POLL_S)
        return True

    def stop_requested(self) -> bool:
        return (self.dir / "STOP").exists()

    def touch_inputs(self, force: bool = False) -> None:
        """Refresh the dates of the environment, the inputs and the outputs (some shared disks delete files by age).
        With several workers, the coordinator does it."""
        if self.worker is not None or (not force and time.time() - self._touched < TOUCH_EVERY_S):
            return
        keepalive(self.cfg)
        self._touched = time.time()

    # ------------------------------------------------------------------ one unit
    def _transects(self, unit: pd.Series):
        from ceva.grid import Tile
        from ceva.transects import gcts

        if self._tile_cache is None or self._tile_cache[0] != unit["tile"]:
            self._tile_cache = (unit["tile"], gcts.load_tile(self.cfg, Tile(unit["tile"])))
        tile = self._tile_cache[1]
        return tile[tile["transect_id"].astype(str).isin(set(unit["transect_ids"]))].reset_index(drop=True)

    def process(self, unit: pd.Series) -> None:
        from ceva.cube import cube_path

        name = unit["name"]
        t0 = time.time()
        if not cube_path(self.cfg, name).exists():
            self.pipeline.build(self.cfg, name, self._transects(unit))
        t1 = time.time()
        results = self.pipeline.run(self.cfg, name, self.method)
        products = results.run.get("products", {}) if hasattr(results, "run") else {}
        kept = True
        if self.store_factory() is None and self._free_bytes() < self.cfg.storage.keep_cubes_above_gb * 1e9:
            shutil.rmtree(cube_path(self.cfg, name), ignore_errors=True)        # rebuildable; results stay
            kept = False
        size = sum(_size(p) for p in self._local_paths(name))
        self._record(name, "processed", n_transects=int(unit["n_transects"]), cube_s=round(t1 - t0, 1),
                     run_s=round(time.time() - t1, 1), local_bytes=size, cube_kept=kept,
                     transect_years=products.get("imagery", {}).get("transect_years"))

    def upload(self, name: str, store) -> None:
        from ceva.cube import cube_path

        cube = cube_path(self.cfg, name)
        n_cube, b_cube = 0, 0
        if cube.exists():
            n_cube, b_cube = store.upload_tree(cube, "cubes", cube.name)
            if not store.check_icechunk(local_snapshot(cube), "cubes", cube.name):
                raise RuntimeError(f"{name}: the uploaded cube does not match the local one")
        results = self.cfg.data_dir / "results" / name / self.method
        n_res, b_res = store.upload_tree(results, "results", name, self.method)
        info = {"files": n_cube + n_res, "bytes": b_cube + b_res, "cube_bytes": b_cube}
        store.put_json({**self.state.get(name, {}), **info, "state": "uploaded"}, "status", f"{name}.json")
        if not self.cfg.storage.keep_local:
            for path in self._local_paths(name):
                shutil.rmtree(path, ignore_errors=True)
        self._record(name, "uploaded", **info)

    def upload_waiting(self, store) -> None:
        for name, event in list(self.state.items()):
            if event["state"] == "processed":
                try:
                    self.upload(name, store)
                except Exception as exc:          # kept locally; tried again later
                    log.warning("%s: upload failed: %s", name, exc)
                    info = {k: v for k, v in event.items() if k not in ("unit", "state", "time")}
                    self._record(name, "processed", **{**info, "upload_error": str(exc)[:300]})

    def wait_for_store(self):
        """Pause until the online store is configured (or a stop is asked); returns it, or None."""
        log.warning("%.1f GB wait locally (budget %.0f GB): pausing until ~/.config/ceva/tokens/storage.env "
                    "is filled in", self._waiting_bytes() / 1e9, self.cfg.storage.local_budget_gb)
        while not self.stop_requested():
            store = self.store_factory()
            if store is not None:
                return store
            self.touch_inputs()
            self.sleep(POLL_S)
        return None

    def _within_budget(self, store):
        """Wait, uploading, until the units waiting locally fit the budget; False if a stop came first.

        Without a store, wait for it; with one whose uploads fail (a wrong key, an outage), retry them
        every few minutes rather than fill the disk.
        """
        budget = self.cfg.storage.local_budget_gb * 1e9
        while self._waiting_bytes() > budget:
            if store is None:
                store = self.wait_for_store()
                if store is None:
                    return False
            self.upload_waiting(store)
            if self._waiting_bytes() > budget:
                log.warning("uploads failing and %.1f GB waiting locally: retrying in %d s",
                            self._waiting_bytes() / 1e9, POLL_S)
                if self.stop_requested():
                    return False
                self.sleep(POLL_S)
                store = self.store_factory()
        return store

    # ------------------------------------------------------------------ the run
    def run(self, limit: int | None = None) -> dict[str, int]:
        plan = load_plan(self.cfg, self.run_name)
        self.touch_inputs(force=True)
        done, in_a_row, pauses = 0, 0, 0
        for attempt in range(2):                      # a second pass for the units that failed
            for _, unit in plan.iterrows():
                if self.stop_requested() or (limit is not None and done >= limit):
                    return self.summary()
                name = unit["name"]
                state = self.state.get(name, {}).get("state")
                if state in ("uploaded", "empty") or (state == "failed" and attempt == 0):
                    continue
                store = self.store_factory()
                if state != "processed":
                    if store is not None:
                        store = self._within_budget(store)
                        if store is False:
                            return self.summary()
                    if not self._wait_for_disk():
                        return self.summary()
                    try:
                        self.process(unit)
                        done, in_a_row, pauses = done + 1, 0, 0
                    except NoShoreline as exc:        # nothing to measure there: finished, not failed
                        self._record_empty(name, exc)
                        done += 1
                    except Exception as exc:          # one bad unit must not stop a continent
                        log.exception("%s failed", name)
                        self._record(name, "failed", error=f"{type(exc).__name__}: {exc}"[:500],
                                     local_bytes=sum(_size(p) for p in self._local_paths(name)))
                        in_a_row += 1
                        if in_a_row >= self.cfg.batch.max_consecutive_failures:
                            if pauses == len(FAILURE_PAUSES_S) or self.stop_requested():
                                log.error("%d units failed in a row (last: %s): stopping the run", in_a_row, exc)
                                self._record("_run", "stopped", error=f"{in_a_row} failures in a row: {exc}"[:500])
                                return self.summary()
                            log.warning("%d units failed in a row: pausing %d min", in_a_row,
                                        FAILURE_PAUSES_S[pauses] // 60)
                            self.sleep(FAILURE_PAUSES_S[pauses])
                            pauses += 1
                            in_a_row = 0
                        continue
                if store is not None:
                    self.upload_waiting(store)
                self.touch_inputs()
        return self.summary()

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for unit, event in self.state.items():
            if unit != "_run":
                counts[event["state"]] = counts.get(event["state"], 0) + 1
        return counts

    # ------------------------------------------------------------------ one worker among several
    def work(self) -> int:
        """Claim the next tile no worker has taken and process its units; once every tile is taken,
        one failed unit no worker has retried. Returns ``TILE_DONE``, or 0 when nothing is left (or
        a stop was asked)."""
        if self.store_factory() is not None:
            raise RuntimeError("several workers with an online store are not supported yet: use one worker")
        plan = load_plan(self.cfg, self.run_name)
        for attempt in (0, 1):
            for claim, units in self._candidates(plan, attempt):
                if self.stop_requested() or not self._wait_for_memory():
                    return 0
                if not self._claim(claim):
                    continue                          # another worker took it meanwhile
                self.state = read_status(self.cfg, self.run_name)
                for _, unit in units.iterrows():
                    state = self.state.get(unit["name"], {}).get("state")
                    if state in ("processed", "uploaded", "empty") or (attempt == 0 and state == "failed"):
                        continue
                    if self.stop_requested() or not self._wait_for_memory() or not self._wait_for_disk():
                        return 0
                    self._process_one(unit, claim, attempt)
                return TILE_DONE
        return 0

    def _candidates(self, plan: pd.DataFrame, attempt: int):
        """(claim, units) in plan order: the tiles with a unit not yet tried, then the failed units."""
        self.state = read_status(self.cfg, self.run_name)
        claims = self.dir / "claims"
        states = plan["name"].map(lambda n: self.state.get(n, {}).get("state"))
        if attempt == 0:
            todo = plan[states.isna()]
            for tile in todo["tile"].drop_duplicates():
                if not (claims / tile).exists():
                    yield tile, plan[plan["tile"] == tile]
        else:
            for _, unit in plan[states == "failed"].iterrows():
                if not (claims / f"{unit['name']}.retry").exists():
                    yield f"{unit['name']}.retry", plan[plan["name"] == unit["name"]]

    def _claim(self, claim: str) -> bool:
        folder = self.dir / "claims"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(folder / claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            json.dump({"worker": self.worker, "pid": os.getpid(), "time": time.time()}, f)
        return True

    def _record_empty(self, name: str, exc: Exception) -> None:
        log.info("%s: no shoreline to measure (%s)", name, exc)
        self._record(name, "empty", reason=str(exc)[:300],
                     local_bytes=sum(_size(p) for p in self._local_paths(name)))

    def _process_one(self, unit: pd.Series, claim: str, attempt: int) -> None:
        name = unit["name"]
        current = self.dir / f"current-w{self.worker}.json"
        current.write_text(json.dumps({"unit": name, "claim": claim, "attempt": attempt, "time": time.time()}))
        try:
            self.process(unit)
        except NoShoreline as exc:                    # nothing to measure there: finished, not failed
            self._record_empty(name, exc)
        except Exception as exc:                      # one bad unit must not stop a continent
            log.exception("%s failed", name)
            self._record(name, "failed", error=f"{type(exc).__name__}: {exc}"[:500],
                         local_bytes=sum(_size(p) for p in self._local_paths(name)))
        finally:
            current.unlink(missing_ok=True)

    def _memory_ok(self) -> bool:
        """Room on the machine for one more unit, leaving the other users theirs."""
        import psutil

        if psutil.virtual_memory().available < self.cfg.batch.min_free_memory_gb * 1e9:
            return False
        try:
            watch = json.loads((self.dir / "workers.json").read_text())
        except (OSError, json.JSONDecodeError):
            return True
        if time.time() - watch.get("time", 0) > 60:   # no coordinator watching
            return True
        others = sum(v for k, v in watch.get("busy", {}).items() if k != str(self.worker))
        return others < START_SHARE * self.cfg.batch.memory_budget_gb * 1e9

    def _wait_for_memory(self) -> bool:
        """Pause while the machine is short of memory; False if a stop came first."""
        warned = False
        while not self._memory_ok():
            if self.stop_requested():
                return False
            if not warned:
                log.warning("not enough memory for another unit now: waiting")
                warned = True
            self.sleep(MEMORY_POLL_S)
        return True


class Coordinator:
    """Runs ``workers`` worker processes over a plan and keeps them within the machine's memory.

    Each worker processes one tile and exits (``TILE_DONE``), and is started again until it finds
    nothing left (exit 0). A worker that crashes, or that the coordinator stops because the workers
    together use more than ``batch.memory_budget_gb`` (or the machine has less than
    ``MEMORY_FLOOR_GB`` available), has its unit recorded as failed, to be retried at the end, and is
    started again after ``RESTART_PAUSE_S``. Only a unit among several running is stopped: one unit
    alone may use what it needs, as with a single worker. ``spawn``, ``memory``, ``sleep`` and
    ``clock`` can be replaced in tests.
    """

    def __init__(self, cfg: Config, run: str, method: str = "ceva", workers: int = 2, *,
                 spawn: Callable | None = None, memory: Callable | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.cfg, self.run_name, self.method, self.n = cfg, run, method, workers
        self.dir = batch_dir(cfg, run)
        self.spawn = spawn or self._spawn
        self.memory = memory or _memory_snapshot
        self.sleep, self.clock = sleep, clock
        self._stopped_by_us: set[int] = set()

    def stop_requested(self) -> bool:
        return (self.dir / "STOP").exists()

    # ------------------------------------------------------------------ the workers
    def _spawn(self, k: int):
        cmd = [sys.executable, "-I", "-c", "from ceva.cli import app; app()"]
        if logging.getLogger().isEnabledFor(logging.INFO):
            cmd.append("-v")
        cmd += ["batch", "run", self.run_name, "--method", self.method, "--worker", str(k)]
        env = {**os.environ, "MALLOC_ARENA_MAX": "2"}     # fewer malloc arenas: less memory kept by threads
        with open(self.dir / f"run-w{k}.log", "a") as out:
            return subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)

    def _current(self, k: int) -> dict | None:
        try:
            return json.loads((self.dir / f"current-w{k}.json").read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _release(self, k: int, reason: str) -> None:
        """A worker ended in the middle of a unit: record it as failed and free its tile."""
        current = self._current(k)
        (self.dir / f"current-w{k}.json").unlink(missing_ok=True)
        if current is None:
            return
        event = {"unit": current["unit"], "state": "failed",
                 "time": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "error": f"worker stopped: {reason}"}
        with open(self.dir / "status.jsonl", "a") as f:
            f.write(json.dumps(event) + "\n")
        if current.get("attempt") == 0:               # the rest of the tile goes back to the other workers
            (self.dir / "claims" / current["claim"]).unlink(missing_ok=True)

    def _watch(self, procs: dict) -> None:
        """Share the workers' memory with them (``workers.json``) and stop the newest unit if they use too much."""
        rss, available = self.memory({k: p.pid for k, p in procs.items()})
        busy = {k: rss.get(k, 0) for k in procs if self._current(k) is not None}
        tmp = self.dir / "workers.json.part"
        tmp.write_text(json.dumps({"time": time.time(), "busy": {str(k): v for k, v in busy.items()},
                                   "budget": self.cfg.batch.memory_budget_gb * 1e9}))
        tmp.replace(self.dir / "workers.json")
        over = sum(rss.values()) > self.cfg.batch.memory_budget_gb * 1e9 or available < MEMORY_FLOOR_GB * 1e9
        running = [k for k in busy if k not in self._stopped_by_us]
        if over and len(running) >= 2:
            newest = max(running, key=lambda k: (self._current(k) or {}).get("time", 0))
            log.warning("workers use %.1f GB (%.1f GB available on the machine): stopping worker %d (%s)",
                        sum(rss.values()) / 1e9, available / 1e9, newest, (self._current(newest) or {}).get("unit"))
            procs[newest].kill()
            self._stopped_by_us.add(newest)

    def _succeeded_since(self, since: str) -> bool:
        return any(e["state"] in ("processed", "uploaded") and e.get("time", "") > since
                   for u, e in read_status(self.cfg, self.run_name).items() if u != "_run")

    def _failures_in_a_row(self, since: str) -> int:
        events = sorted((e for u, e in read_status(self.cfg, self.run_name).items()
                         if u != "_run" and e.get("time", "") > since), key=lambda e: e["time"])
        count = 0
        for event in reversed(events):
            if str(event.get("error", "")).startswith(MEMORY_STOP):
                continue                              # stopped for memory: not a refusing service
            if event["state"] != "failed":
                break
            count += 1
        return count

    # ------------------------------------------------------------------ the run
    def run(self) -> dict[str, int]:
        lock = self.dir / "workers.lock"
        self.dir.mkdir(parents=True, exist_ok=True)
        if lock.exists():
            import psutil

            pid = int(lock.read_text().strip() or 0)
            if pid and pid != os.getpid() and psutil.pid_exists(pid):
                raise RuntimeError(f"workers already running for {self.run_name} (coordinator pid {pid})")
        lock.write_text(str(os.getpid()))
        shutil.rmtree(self.dir / "claims", ignore_errors=True)          # claims of an earlier run
        for stale in self.dir.glob("current-w*.json"):
            stale.unlink()
        threading.Thread(target=self._keepalive, daemon=True).start()
        try:
            self._loop()
        finally:
            lock.unlink(missing_ok=True)
            (self.dir / "workers.json").unlink(missing_ok=True)
        counts: dict[str, int] = {}
        for unit, event in read_status(self.cfg, self.run_name).items():
            if unit != "_run":
                counts[event["state"]] = counts.get(event["state"], 0) + 1
        return counts

    def _keepalive(self) -> None:
        while True:
            try:
                keepalive(self.cfg)
            except Exception as exc:                   # never stop the run for it
                log.warning("refreshing file dates failed: %s", exc)
            time.sleep(TOUCH_EVERY_S)

    def _loop(self) -> None:
        start = self.clock()
        due = {k: start + k * WORKER_STAGGER_S for k in range(self.n)}
        procs: dict[int, object] = {}
        crashes, pauses, resume_at = 0, 0, 0.0
        mark = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        stopping = False
        while due or procs:
            now = self.clock()
            stopping = stopping or self.stop_requested()
            for k, at in list(due.items()):
                if stopping:
                    del due[k]
                elif now >= at and now >= resume_at:
                    procs[k] = self.spawn(k)
                    del due[k]
            if procs:
                self._watch(procs)
            for k, proc in list(procs.items()):
                code = proc.poll()
                if code is None:
                    continue
                del procs[k]
                if code == TILE_DONE:
                    due[k] = now
                elif code == 0:
                    continue                          # nothing left for this worker
                else:
                    memory = k in self._stopped_by_us
                    self._stopped_by_us.discard(k)
                    self._release(k, MEMORY_STOP.removeprefix("worker stopped: ") if memory else f"exit code {code}")
                    crashes += 0 if memory else 1
                    if crashes > MAX_CRASHES:
                        log.error("%d worker crashes: stopping the run", crashes)
                        stopping = True
                    else:
                        due[k] = now + RESTART_PAUSE_S
            failures = self._failures_in_a_row(mark)
            if failures == 0 and self._succeeded_since(mark):
                pauses = 0
            if failures >= self.cfg.batch.max_consecutive_failures and not stopping:
                mark = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
                if pauses == len(FAILURE_PAUSES_S):
                    log.error("%d units failed in a row: stopping the run", failures)
                    with open(self.dir / "status.jsonl", "a") as f:
                        f.write(json.dumps({"unit": "_run", "state": "stopped", "time": mark,
                                            "error": f"{failures} failures in a row"}) + "\n")
                    stopping = True
                else:
                    log.warning("%d units failed in a row: no new tile for %d min", failures,
                                FAILURE_PAUSES_S[pauses] // 60)
                    resume_at = now + FAILURE_PAUSES_S[pauses]
                    pauses += 1
            if due or procs:
                self.sleep(WATCH_S)


def _memory_snapshot(pids: dict[int, int]) -> tuple[dict[int, int], int]:
    """Resident memory of each worker (with its child processes) and the memory available on the machine."""
    import psutil

    rss = {}
    for k, pid in pids.items():
        try:
            proc = psutil.Process(pid)
            total = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.Error:
                    pass
            rss[k] = total
        except psutil.Error:
            rss[k] = 0
    return rss, psutil.virtual_memory().available


def keepalive(cfg: Config, roots: list[Path] | None = None) -> int:
    """Refresh the dates of the environment and of everything in the data directory. Returns the files touched.

    Some shared disks delete files by age; this keeps the environment (whose files carry their
    build dates), the inputs and the stored results. Run it at least weekly when nothing runs.
    """
    count = 0
    for root in roots if roots is not None else (Path(sys.prefix), cfg.data_dir):
        for folder, _, files in os.walk(root):
            for name in files:
                try:
                    os.utime(os.path.join(folder, name), follow_symlinks=False)
                    count += 1
                except OSError:
                    pass
    return count


def local_snapshot(path: Path) -> str:
    """The snapshot at the tip of ``main`` of a local Icechunk repository."""
    import icechunk

    return icechunk.Repository.open(icechunk.local_filesystem_storage(str(path))).readonly_session("main").snapshot_id


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0
