"""Batch runs: units cut from tiles, processed in turn, uploaded, resumed; the S3 store's layout."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ceva import batch
from ceva.credentials import Credentials
from ceva.storage import RemoteStore


@pytest.fixture
def data(cfg, tmp_path, monkeypatch):
    """A GCTS-like index: tile A (zoom 10 prefix 0000000000) with 7 transects, tile B with 2, one other country."""
    rows = [(f"cl1s01tr{i:05d}", "FR", "000000000000") for i in range(7)]
    rows += [(f"cl2s01tr{i:05d}", "PT", "000000000100") for i in range(2)] + [("cl3s01tr00001", "GB", "000000000200")]
    index = tmp_path / "index.parquet"
    pd.DataFrame(rows, columns=["transect_id", "country", "qk12"]).to_parquet(index)
    monkeypatch.setattr("ceva.transects.gcts.available_indexes", lambda cfg: [index])
    return cfg.model_copy(update={"data_dir": tmp_path / "data"})


def test_tiles_are_cut_along_the_coast_into_units(data):
    plan = batch.plan_units(data, ["FR", "PT"], max_transects=3, shore_types=[])
    assert plan["name"].tolist() == ["0000000000-0", "0000000000-1", "0000000000-2", "0000000001-0"]
    assert plan["n_transects"].tolist() == [3, 3, 1, 2]
    assert list(plan.loc[0, "transect_ids"]) == ["cl1s01tr00000", "cl1s01tr00001", "cl1s01tr00002"]
    est = batch.storage_estimate(1000)
    assert est["cubes"][0] == 1000 * 2000 * batch.BYTES_PER_TRANSECT_ACQUISITION


class FakePipeline:
    """Writes a tiny cube folder and results folder per unit, as the real pipeline lays them out."""

    def __init__(self, fail=()):
        self.built, self.ran, self.fail = [], [], set(fail)

    def variant_config(self, cfg, method):
        return cfg

    def build(self, cfg, name, transects):
        if name in self.fail:
            raise RuntimeError("no imagery")
        cube = cfg.data_dir / "cubes" / f"{name}.icechunk"
        (cube / "chunks").mkdir(parents=True)
        (cube / "chunks" / "c0").write_bytes(b"x" * 1000)
        (cube / "repo").write_bytes(b"r")
        self.built.append(name)

    def run(self, cfg, name, method):
        out = cfg.data_dir / "results" / name / method
        out.mkdir(parents=True)
        (out / "shorelines_imagery.parquet").write_bytes(b"p" * 100)
        self.ran.append(name)
        return SimpleNamespace(run={"products": {"imagery": {"transect_years": 5}}})


class FakeStore:
    def __init__(self):
        self.objects: dict[str, int] = {}
        self.order: list[str] = []

    def upload_tree(self, local, *parts):
        files = sorted(p for p in Path(local).rglob("*") if p.is_file())
        files.sort(key=lambda p: p.relative_to(local).parts[:1] == ("repo",))
        for p in files:
            key = "/".join((*parts, p.relative_to(local).as_posix()))
            self.objects[key] = p.stat().st_size
            self.order.append(key)
        return len(files), sum(p.stat().st_size for p in files)

    def check_icechunk(self, snapshot, *parts):
        return True

    def put_json(self, obj, *parts):
        self.objects["/".join(parts)] = 1


@pytest.fixture(autouse=True)
def no_walks(monkeypatch):
    """Batch runs refresh file dates; in tests, never over the whole environment."""
    monkeypatch.setattr(batch.Runner, "touch_inputs", lambda self, force=False: None)


@pytest.fixture
def no_tiles(monkeypatch):
    # a roomy disk unless a test says otherwise: the real one is shared and may be nearly full
    monkeypatch.setattr(batch.Runner, "_free_bytes", lambda self: 10**15)
    monkeypatch.setattr(batch.Runner, "_transects", lambda self, unit: None)
    monkeypatch.setattr(batch, "local_snapshot", lambda path: "S1")
    monkeypatch.setattr(batch.Runner, "touch_inputs", lambda self, force=False: None)


def test_units_are_processed_uploaded_and_freed_then_resumed(data, no_tiles):
    batch.write_plan(data, "r", batch.plan_units(data, ["FR", "PT"], max_transects=3, shore_types=[]))
    store, pipe = FakeStore(), FakePipeline(fail={"0000000000-1"})
    summary = batch.Runner(data, "r", store_factory=lambda: store, pipeline=pipe).run()
    assert summary == {"uploaded": 3, "failed": 1}
    assert pipe.built.count("0000000000-1") == 0 and pipe.ran == ["0000000000-0", "0000000000-2", "0000000001-0"]
    assert "cubes/0000000000-0.icechunk/chunks/c0" in store.objects
    assert "results/0000000000-0/ceva/shorelines_imagery.parquet" in store.objects
    cube = "cubes/0000000000-0.icechunk/"
    assert store.order.index(cube + "repo") > store.order.index(cube + "chunks/c0")      # entry point last
    assert not (data.data_dir / "cubes" / "0000000000-0.icechunk").exists()        # freed once uploaded
    assert not (data.data_dir / "results" / "0000000000-0").exists()
    # resumed: only the failed unit is tried again
    pipe2 = FakePipeline()
    summary = batch.Runner(data, "r", store_factory=lambda: store, pipeline=pipe2).run()
    assert pipe2.ran == ["0000000000-1"] and summary == {"uploaded": 4}


def test_without_online_storage_units_stay_local_and_cubes_go_when_space_runs_low(data, no_tiles, monkeypatch):
    batch.write_plan(data, "w", batch.plan_units(data, ["FR", "PT"], max_transects=3, shore_types=[]))
    pipe = FakePipeline()
    # plenty of room, until the last unit: above min_free_gb (no pause) but below keep_cubes_above_gb
    monkeypatch.setattr(batch.Runner, "_free_bytes", lambda self: 200e9 if len(pipe.ran) == 4 else 10**15)
    summary = batch.Runner(data, "w", store_factory=lambda: None, pipeline=pipe, sleep=pytest.fail).run()
    assert summary == {"processed": 4} and pipe.ran == ["0000000000-0", "0000000000-1", "0000000000-2",
                                                        "0000000001-0"]
    assert (data.data_dir / "results" / "0000000001-0" / "ceva").exists()                 # results stay
    state = batch.read_status(data, "w")
    assert [state[u]["cube_kept"] for u in pipe.ran] == [True, True, True, False]
    assert not (data.data_dir / "cubes" / "0000000001-0.icechunk").exists()


def test_a_nearly_full_disk_pauses_the_run(data, no_tiles, monkeypatch):
    batch.write_plan(data, "d", batch.plan_units(data, ["PT"], max_transects=3, shore_types=[]))
    free = iter([0, 0, 10**15])
    monkeypatch.setattr(batch.Runner, "_free_bytes", lambda self: next(free, 10**15))
    sleeps = []
    pipe = FakePipeline()
    batch.Runner(data, "d", store_factory=lambda: None, pipeline=pipe, sleep=sleeps.append).run()
    assert sleeps == [batch.POLL_S, batch.POLL_S] and pipe.ran == ["0000000001-0"]


def test_failures_in_a_row_pause_then_stop_the_run(data, no_tiles):
    batch.write_plan(data, "x", batch.plan_units(data, ["FR", "PT"], max_transects=1, shore_types=[]))
    cfg = data.model_copy(update={"batch": data.batch.model_copy(update={"max_consecutive_failures": 1})})
    every = {f"0000000000-{k}" for k in range(7)} | {"0000000001-0", "0000000001-1"}
    sleeps = []
    summary = batch.Runner(cfg, "x", store_factory=lambda: None, pipeline=FakePipeline(fail=every),
                           sleep=sleeps.append).run()
    assert sleeps == list(batch.FAILURE_PAUSES_S) and summary == {"failed": len(batch.FAILURE_PAUSES_S) + 1}
    assert batch.read_status(cfg, "x")["_run"]["state"] == "stopped"


def test_only_the_chosen_coast_types_are_planned(data, monkeypatch):
    kinds = pd.DataFrame({"transect_id": [f"cl1s01tr{i:05d}" for i in range(7)] + ["cl2s01tr00000", "cl2s01tr00001"],
                          "shore_type": ["sandy_gravel_or_small_boulder_sediments"] * 4 + ["muddy_sediments"] * 3
                          + ["rocky_shore_platform_or_large_boulders", "sandy_gravel_or_small_boulder_sediments"]})
    monkeypatch.setattr("ceva.transects.typology.load", lambda cfg: kinds)
    sandy = ["sandy_gravel_or_small_boulder_sediments"]
    plan = batch.plan_units(data, ["FR", "PT"], max_transects=3, shore_types=sandy)
    assert plan["n_transects"].sum() == 5 and plan["name"].tolist() == ["0000000000-0", "0000000000-1", "0000000001-0"]
    assert batch.plan_units(data, ["FR", "PT"], max_transects=3, shore_types=[])["n_transects"].sum() == 9


def test_typology_is_reduced_and_loaded(data, monkeypatch):
    from ceva.transects import typology

    raw = pd.DataFrame({source: ["b", "a"] if name == "transect_id" else [0.5, 0.25] if name.startswith("p_")
                        or name == "sds_rate" else [True, False] if name in ("is_built_environment", "has_defence",
                                                                             "sds_qa") else ["x", "y"]
                        for source, name in typology.COLUMNS.items()})
    raw["unused"] = 1
    monkeypatch.setattr(typology, "zenodo_file", lambda record, key, dest: dest.parent.mkdir(parents=True,
                        exist_ok=True) or raw.to_parquet(dest) or dest)
    path = typology.prepare(data, "Europe")
    assert list(pd.read_parquet(path).columns) == list(typology.COLUMNS.values())
    loaded = typology.load(data, ["a"])
    assert loaded["transect_id"].tolist() == ["a"] and loaded["sds_rate"].tolist() == [0.25]


def test_keepalive_refreshes_the_data_directory(data):
    import os

    old = data.data_dir / "results" / "u" / "x.parquet"
    old.parent.mkdir(parents=True)
    old.write_text("x")
    os.utime(old, (0, 0))
    assert batch.keepalive(data, roots=[data.data_dir]) >= 1
    assert old.stat().st_mtime > 1e9


def test_stop_file_ends_the_run_after_the_current_unit(data, no_tiles):
    batch.write_plan(data, "s", batch.plan_units(data, ["FR"], max_transects=3, shore_types=[]))
    pipe = FakePipeline()
    runner = batch.Runner(data, "s", store_factory=lambda: None, pipeline=pipe)
    original = pipe.run

    def run_and_stop(cfg, name, method):
        (batch.batch_dir(data, "s") / "STOP").write_text("")
        return original(cfg, name, method)

    pipe.run = run_and_stop
    assert runner.run() == {"processed": 1}


class FakeS3:
    def __init__(self):
        self.keys = []

    def upload_file(self, filename, bucket, key):
        self.keys.append((bucket, key))


def test_remote_store_mirrors_the_data_directory_and_hides_its_keys(cfg, tmp_path):
    creds = Credentials({"STORAGE_ENDPOINT_URL": "https://acct.eu.r2.cloudflarestorage.com", "STORAGE_BUCKET": "b",
                         "STORAGE_ACCESS_KEY_ID": "id", "STORAGE_SECRET_ACCESS_KEY": "very-secret"})
    store = RemoteStore(cfg, creds, client=FakeS3())
    repo = tmp_path / "u.icechunk"
    (repo / "snapshots").mkdir(parents=True)
    (repo / "snapshots" / "S1").write_bytes(b"s")
    (repo / "repo").write_bytes(b"r")
    assert store.upload_tree(repo, "cubes", "u.icechunk") == (2, 2)
    assert store.client.keys == [("b", "ceva/cubes/u.icechunk/snapshots/S1"), ("b", "ceva/cubes/u.icechunk/repo")]
    assert "very-secret" not in repr(store) and "id" not in repr(store).replace("cloudflarestorage", "")


def test_failing_uploads_pause_the_run_instead_of_filling_the_disk(data, no_tiles):
    batch.write_plan(data, "f", batch.plan_units(data, ["FR", "PT"], max_transects=3, shore_types=[]))
    cfg = data.model_copy(update={"storage": data.storage.model_copy(update={"local_budget_gb": 1e-6})})

    class Broken(FakeStore):
        def upload_tree(self, local, *parts):
            raise OSError("access denied")

    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        (batch.batch_dir(cfg, "f") / "STOP").write_text("")       # the user stops it to fix the key

    pipe = FakePipeline()
    summary = batch.Runner(cfg, "f", store_factory=lambda: Broken(), pipeline=pipe, sleep=sleep).run()
    assert pipe.ran == ["0000000000-0"] and summary == {"processed": 1} and sleeps


def test_tide_runs_do_not_wrap_the_output_again():
    import sys

    import eo_tides.utils

    from ceva import tides

    tides._no_colorama()
    before = sys.stdout
    for _ in range(3):
        eo_tides.utils.list_models(directory=".", show_available=False, show_supported=False, raise_error=False)
    assert sys.stdout is before


# ---------------------------------------------------------------------- several workers

def test_workers_status_files_are_merged_by_time(data):
    folder = batch.batch_dir(data, "m")
    folder.mkdir(parents=True)
    (folder / "status.jsonl").write_text('{"unit": "a", "state": "processed", "time": "2026-11-19T10:00:00+00:00"}\n'
                                         '{"unit": "b", "state": "failed", "time": "2026-11-19T10:05:00+00:00"}\n')
    (folder / "status-w1.jsonl").write_text('{"unit": "b", "state": "processed", "time": "2026-11-20T08:00:00+00:00"}\n'
                                            '{"unit": "c", "sta')           # cut short by a stopped worker
    state = batch.read_status(data, "m")
    assert {u: e["state"] for u, e in state.items()} == {"a": "processed", "b": "processed"}


def test_workers_share_out_tiles_and_retry_a_failed_unit_once_at_the_end(data, no_tiles, monkeypatch):
    monkeypatch.setattr(batch.Runner, "_memory_ok", lambda self: True)
    batch.write_plan(data, "w", batch.plan_units(data, ["FR", "PT"], max_transects=3, shore_types=[]))
    pipe = FakePipeline(fail={"0000000000-1"})
    w0, w1 = (batch.Runner(data, "w", store_factory=lambda: None, pipeline=pipe, worker=k) for k in (0, 1))
    assert w0.cfg.cube.workers == min(data.cube.workers, data.batch.worker_reads)
    assert w0.work() == batch.TILE_DONE                   # tile 0000000000: its three units together
    assert w1.work() == batch.TILE_DONE                   # tile 0000000001, not the one w0 claimed
    assert pipe.ran == ["0000000000-0", "0000000000-2", "0000000001-0"]
    assert w0.work() == batch.TILE_DONE                   # every tile taken: the failed unit, once
    assert w1.work() == 0 and w0.work() == 0              # not again by the other worker
    folder = batch.batch_dir(data, "w")
    assert sorted(p.name for p in (folder / "claims").iterdir()) == ["0000000000", "0000000000-1.retry", "0000000001"]
    failed = [line for line in (folder / "status-w0.jsonl").read_text().splitlines() if '"failed"' in line]
    assert len(failed) == 2 and not (folder / "status.jsonl").exists()
    assert {u: e["state"] for u, e in batch.read_status(data, "w").items()} == {
        "0000000000-0": "processed", "0000000000-1": "failed", "0000000000-2": "processed", "0000000001-0": "processed"}


def test_a_worker_leaves_memory_to_the_other_users(data, monkeypatch):
    runner = batch.Runner(data, "mem", store_factory=lambda: None, pipeline=FakePipeline(), worker=0)
    available = {"bytes": 10e9}
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=available["bytes"]))
    assert not runner._memory_ok()                        # under batch.min_free_memory_gb on the machine
    available["bytes"] = 40e9
    assert runner._memory_ok()
    folder = batch.batch_dir(data, "mem")
    folder.mkdir(parents=True, exist_ok=True)
    busy = 0.7 * data.batch.memory_budget_gb * 1e9
    (folder / "workers.json").write_text(json.dumps({"time": time.time(), "busy": {"1": busy, "0": busy}}))
    assert not runner._memory_ok()                        # the others use most of the budget
    (folder / "workers.json").write_text(json.dumps({"time": time.time() - 600, "busy": {"1": busy}}))
    assert runner._memory_ok()                            # nobody watching any more
    available["bytes"] = 1e9
    (folder / "STOP").write_text("")
    assert runner._wait_for_memory() is False


class FakeProc:
    """A worker process that exits at a given time with a given code, or when killed."""

    def __init__(self, k, clock, exit_at, code):
        self.k, self.pid, self.clock, self.exit_at, self.code, self.killed = k, 1000 + k, clock, exit_at, code, False

    def poll(self):
        if self.killed:
            return -9
        return self.code if self.clock() >= self.exit_at else None

    def kill(self):
        self.killed = True


def test_over_the_memory_budget_the_newest_unit_stops_and_is_retried_later(data, monkeypatch):
    monkeypatch.setattr(batch, "keepalive", lambda cfg: 0)
    folder = batch.batch_dir(data, "c")
    now = {"t": 0.0}
    clock = lambda: now["t"]                                          # noqa: E731
    spawned = []

    def spawn(k):
        first = spawned.count(k) == 0
        spawned.append(k)
        if not first:
            return FakeProc(k, clock, 0, 0)                           # nothing left
        (folder / "claims").mkdir(exist_ok=True)
        (folder / "claims" / f"T{k}").write_text("")
        (folder / f"current-w{k}.json").write_text(json.dumps({"unit": f"u{k}", "claim": f"T{k}", "attempt": 0,
                                                                "time": float(k)}))
        if k == 0:
            def done():
                if clock() >= 200:
                    (folder / "current-w0.json").unlink(missing_ok=True)
                return clock()
            return FakeProc(k, done, 200, batch.TILE_DONE)
        return FakeProc(k, clock, float("inf"), 0)

    def memory(pids):                                                 # 20 + 15 GB: over a 30 GB budget
        return {k: (20e9 if k == 0 else 15e9) for k in pids}, 30e9

    def sleep(seconds):
        now["t"] += seconds

    cfg = data.model_copy(update={"batch": data.batch.model_copy(update={"memory_budget_gb": 30.0})})
    folder.mkdir(parents=True)
    summary = batch.Coordinator(cfg, "c", workers=2, spawn=spawn, memory=memory, sleep=sleep, clock=clock).run()
    assert spawned == [0, 1, 1, 0]                        # 1 stopped, started again; 0 after its tile
    state = batch.read_status(data, "c")
    assert state["u1"]["state"] == "failed" and state["u1"]["error"] == batch.MEMORY_STOP
    assert not (folder / "claims" / "T1").exists() and (folder / "claims" / "T0").exists()
    assert summary == {"failed": 1} and not (folder / "workers.lock").exists()


def test_the_time_left_counts_the_units_running_side_by_side():
    """Two workers, each unit 100 s + 1 s per transect: half the remaining units' time, failed units aside."""
    plan = pd.DataFrame({"name": [f"u{i}" for i in range(1, 8)], "n_transects": [10, 50, 90, 50, 100, 20, 1000]})

    def done(unit, n, end_s, state="processed"):
        end = pd.Timestamp("2026-11-20T08:00:00+00:00") + pd.Timedelta(seconds=end_s)
        return {"unit": unit, "state": state, "time": end.isoformat(), "cube_s": 100 + n - 10, "run_s": 10}

    state = {"u1": done("u1", 10, 110), "u3": done("u3", 90, 300),       # worker 0
             "u2": done("u2", 50, 150), "u4": done("u4", 50, 300),       # worker 1
             "u7": {"unit": "u7", "state": "failed", "time": "2026-11-20T08:05:00+00:00"}}
    units_per_hour, left = batch.recent_pace(plan, state)
    assert units_per_hour == pytest.approx(48)                         # 4 units in 300 s
    assert left == pytest.approx(((100 + 100) + (100 + 20)) / 2)       # u5 and u6, two at a time
    assert batch.recent_pace(plan, {"u1": state["u1"]}) is None


def test_a_preview_publishes_only_the_listed_units_that_are_finished(data, tmp_path):
    from ceva.cli import _run_units

    batch.write_plan(data, "p", batch.plan_units(data, ["FR", "PT"], max_transects=3, shore_types=[]))
    folder = batch.batch_dir(data, "p")
    (folder / "status.jsonl").write_text(
        '{"unit": "0000000000-0", "state": "processed", "time": "2026-11-20T08:00:00+00:00"}\n'
        '{"unit": "0000000001-0", "state": "processed", "time": "2026-11-20T08:05:00+00:00"}\n')
    listed = tmp_path / "area.txt"
    listed.write_text("0000000001-0\n0000000000-2\n")                    # the second one is not finished
    names, info = _run_units(data, "p", None, f"@{listed}")
    assert names == ["0000000001-0"] and info["planned"] == 4 and info["finished"] == 2
    assert _run_units(data, "p", None, "0000000000-0")[0] == ["0000000000-0"]


def test_a_later_run_extends_an_earlier_one_without_reusing_unit_names(data):
    first = batch.plan_units(data, ["FR"], max_transects=3, shore_types=[]).iloc[:2]     # 6 of tile A's 7
    later = batch.plan_units(data, None, max_transects=3, shore_types=[], after=first)
    assert later["name"].tolist() == ["0000000000-2", "0000000001-0", "0000000002-0"]  # numbered after A-1
    assert later.loc[0, "transect_ids"] == ["cl1s01tr00006"]                            # only what is left
    assert not set(first["name"]) & set(later["name"])


def test_the_area_keeps_one_side_of_each_sea():
    area = batch.load_area("europe-mediterranean")
    points = [  # lon, lat, GCTS continent, kept
        (10.0, 60.0, "EU", True), (15.0, 68.0, "EU", False),        # Norway south and north of the Arctic circle
        (44.3, 66.0, "EU", True), (47.5, 45.0, "EU", False),        # the White Sea, the Caspian
        (37.5, 46.5, "EU", True), (-16.5, 28.3, "EU", True),        # the Sea of Azov, the Canaries
        (10.2, 36.8, "AF", True), (-8.0, 33.0, "AF", False),        # Tunis, Morocco's Atlantic coast
        (32.3, 30.6, "AF", False), (33.5, 27.5, "AF", False),       # the Suez Canal, the Red Sea
        (35.5, 33.9, "AS", True), (33.4, 35.1, "AS", True),         # Beirut, Cyprus
        (41.6, 41.6, "AS", True), (35.0, 29.5, "AS", False),        # Batumi, Eilat
        (50.0, 40.4, "AS", False), (10.2, 36.8, "AS", False),       # Baku; a point of another continent
    ]
    lon, lat, continent, kept = zip(*points, strict=True)
    assert batch.in_area(area, lon, lat, continent).tolist() == list(kept)


def test_units_with_nothing_to_measure_are_empty_not_failed(data, no_tiles):
    from ceva.shoreline import NoShoreline

    class Spit(FakePipeline):
        def run(self, cfg, name, method):
            if name == "0000000000-1":
                raise NoShoreline("no transect has a known seaward direction")
            return super().run(cfg, name, method)

    batch.write_plan(data, "e", batch.plan_units(data, ["FR", "PT"], max_transects=3, shore_types=[]))
    summary = batch.Runner(data, "e", store_factory=lambda: None, pipeline=Spit()).run()
    assert summary == {"processed": 3, "empty": 1}
    state = batch.read_status(data, "e")
    assert state["0000000000-1"]["reason"] == "no transect has a known seaward direction"
    again = FakePipeline()
    assert batch.Runner(data, "e", store_factory=lambda: None, pipeline=again).run() == {"processed": 3, "empty": 1}
    assert again.ran == []                                                      # not tried again
