"""P0-2: the planning collector, its counters and its guarantees.

Measurement must never change a result, so besides counting, these tests
check scoping (per run, per thread), the no-op fast path, and that plans and
output frames are identical with the collector on and off.
"""
import json
import shutil
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from videowipe import perfstats
from videowipe.detect import (
    DBNetDetector,
    TextBox,
    refine_temporal_presence,
)
from videowipe.engine import WipeEngine
from videowipe.api import WipeRequest
from videowipe.plan import Segment

ZERO = dict.fromkeys(perfstats.COUNTERS, 0)


def counts(row):
    return {key: row[key] for key in perfstats.COUNTERS}


# ── collector ───────────────────────────────────────────────────────────────

def test_phase_scoping_nesting_and_wall_time():
    handle = perfstats.start()
    perfstats.add("detect_calls")  # outside any phase
    with perfstats.phase("refine"):
        perfstats.add("frames_converted", 3)
        time.sleep(0.02)
        with perfstats.phase("boundary_probe"):
            perfstats.add("frames_converted")
        perfstats.add("frames_converted")
    perfstats.begin("coarse_sampling")
    perfstats.add("frames_grabbed", 2)
    perfstats.end()
    snapshot = perfstats.finish(handle)

    assert list(snapshot) == ["coarse_sampling", "refine", "boundary_probe", "other"]
    assert snapshot["refine"]["frames_converted"] == 4
    assert snapshot["boundary_probe"]["frames_converted"] == 1
    assert snapshot["coarse_sampling"]["frames_grabbed"] == 2
    assert snapshot["other"]["detect_calls"] == 1
    assert snapshot["refine"]["wall_s"] >= 0.02  # inclusive of the nested phase
    assert snapshot["refine"]["entries"] == 1
    for row in snapshot.values():
        assert set(perfstats.COUNTERS) <= row.keys()
    json.dumps(snapshot)  # plain numbers only


def test_phase_can_be_entered_more_than_once_and_accumulates():
    handle = perfstats.start()
    for _ in range(3):
        with perfstats.phase("refine"):
            perfstats.add("detect_calls", 2)
    snapshot = perfstats.finish(handle)
    assert snapshot["refine"]["entries"] == 3 and snapshot["refine"]["detect_calls"] == 6


def test_distinct_marks_count_each_identity_once():
    handle = perfstats.start()
    with perfstats.phase("refine"):
        for frame in (3, 3, 5, 3, 5):
            perfstats.mark("recovery_frames", frame)
    assert perfstats.finish(handle)["refine"]["recovery_frames"] == 2


def test_open_phase_is_closed_when_the_run_ends_early():
    handle = perfstats.start()
    perfstats.begin("coarse_sampling")
    perfstats.add("frames_converted")
    snapshot = perfstats.finish(handle)  # e.g. an exception skipped end()
    assert snapshot["coarse_sampling"]["frames_converted"] == 1
    assert snapshot["coarse_sampling"]["wall_s"] >= 0


def test_helpers_are_noops_without_a_collector():
    assert perfstats.current() is None
    perfstats.add("detect_calls")
    perfstats.mark("recovery_frames", 1)
    perfstats.forward("hl")
    perfstats.begin("refine")
    perfstats.end()
    with perfstats.phase("refine"):
        pass
    items = iter([1, 2])
    assert perfstats.counted(items) is items  # no wrapper at all
    detector = SimpleNamespace(detect=lambda frame: ["boxes", frame])
    assert perfstats.detect(detector, "f") == ["boxes", "f"]
    assert perfstats.finish(None) == {}


def test_detect_wrapper_counts_and_passes_exceptions_through():
    class Failing:
        def detect(self, frame):
            raise RuntimeError("boom")

    handle = perfstats.start()
    with perfstats.phase("refine"):
        assert perfstats.detect(SimpleNamespace(detect=lambda f: [f]), 7) == [7]
        with pytest.raises(RuntimeError, match="boom"):
            perfstats.detect(Failing(), 0, recovery=True)
    row = perfstats.finish(handle)["refine"]
    assert (row["detect_calls"], row["recovery_detect_calls"]) == (2, 1)


def test_collector_is_thread_safe():
    stats = perfstats.PerfStats()
    stats.begin("refine")

    def worker():
        for _ in range(2000):
            stats.add("detect_calls")
            stats.add_many(forward_passes=1, forward_hl=1)
            stats.mark("recovery_frames", threading.get_ident())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stats.end()
    row = stats.snapshot()["refine"]
    assert row["detect_calls"] == 16000 and row["forward_passes"] == 16000
    assert row["forward_hl"] == 16000 and row["recovery_frames"] == 8


def test_consecutive_runs_do_not_leak_into_each_other():
    first = perfstats.start()
    with perfstats.phase("refine"):
        perfstats.add("detect_calls", 5)
        perfstats.mark("recovery_frames", 1)
    first_snapshot = perfstats.finish(first)
    assert perfstats.current() is None
    perfstats.add("detect_calls", 100)  # between runs: dropped, not carried over

    second = perfstats.start()
    with perfstats.phase("refine"):
        perfstats.add("detect_calls", 2)
    second_snapshot = perfstats.finish(second)
    assert first_snapshot["refine"]["detect_calls"] == 5
    assert second_snapshot["refine"]["detect_calls"] == 2
    assert second_snapshot["refine"]["recovery_frames"] == 0


def test_a_collector_is_bound_to_the_thread_that_started_it():
    seen = {}

    def other_thread():
        perfstats.add("detect_calls", 50)  # no collector in this context
        seen["current"] = perfstats.current()

    handle = perfstats.start()
    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join()
    with perfstats.phase("refine"):
        perfstats.add("detect_calls")
    snapshot = perfstats.finish(handle)
    assert seen["current"] is None
    assert snapshot["refine"]["detect_calls"] == 1 and "other" not in snapshot


def test_collection_works_when_the_whole_run_is_on_a_worker_thread():
    result = {}

    def run():
        handle = perfstats.start()
        with perfstats.phase("refine"):
            perfstats.add("frames_converted", 4)
        result["snapshot"] = perfstats.finish(handle)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert result["snapshot"]["refine"]["frames_converted"] == 4
    assert perfstats.current() is None


def test_helper_threads_can_adopt_the_collector():
    handle = perfstats.start()
    stats = perfstats.current()

    def helper():
        with perfstats.activate(stats), perfstats.phase("refine"):
            perfstats.add("frames_grabbed", 7)

    thread = threading.Thread(target=helper)
    thread.start()
    thread.join()
    assert perfstats.finish(handle)["refine"]["frames_grabbed"] == 7


@pytest.mark.parametrize("value", ["0", "off", "False", "no"])
def test_environment_switch_disables_collection(monkeypatch, value):
    monkeypatch.setenv("VIDEOWIPE_PERF_STATS", value)
    assert perfstats.start() is None
    assert perfstats.finish(None) == {}


# ── scripted counts ─────────────────────────────────────────────────────────

W, H, N = 64, 48, 12


def _write_indexed_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 8, (W, H))
    assert writer.isOpened()
    for index in range(N):
        writer.write(np.full((H, W, 3), index * 10, np.uint8))
    writer.release()
    return str(path)


def _box(x1, y1, x2, y2):
    return TextBox(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64), 0.9)


class ScriptedDetector:
    """Full frames: even index -> box over the candidate, odd -> nothing.

    Recovery crops (smaller than the frame) see a box in every variant only
    while processing frame 3, so exactly one recovery attempt succeeds.
    """

    def __init__(self):
        self.index = None

    def detect(self, frame):
        if frame.shape[:2] == (H, W):
            self.index = int(frame[0, 0, 0]) // 10
            return [_box(12, 30, 52, 42)] if self.index % 2 == 0 else []
        return [_box(10, 8, 30, 16)] if self.index == 3 else []


def test_refine_counters_on_a_scripted_scenario(tmp_path):
    video = _write_indexed_video(tmp_path / "indexed.avi")
    candidate = SimpleNamespace(
        id="A", bbox=(10, 28, 54, 44), detector_backed=True,
        presence_frames=[], temporal_sample_indices=[],
    )
    result = SimpleNamespace(
        detector=ScriptedDetector(), candidates=[candidate], sample_indices=[0],
        frame_shape=(H, W), sampled_frame_boxes={},
    )
    handle = perfstats.start()
    with perfstats.phase("refine"):
        refine_temporal_presence(video, result, {"A": [Segment(2, 8)]}, N)
    row = perfstats.finish(handle)["refine"]

    # Frames 0..7 are advanced; 2..7 are converted, 0 and 1 are only grabbed.
    assert (row["frames_converted"], row["frames_grabbed"]) == (6, 2)
    # Frames 2, 4, 6 find a box (no recovery); 3, 5, 7 are empty and recover.
    assert row["recovery_calls"] == 3 and row["recovery_frames"] == 3
    assert row["recovery_detect_calls"] == 3 * 6  # six contrast variants each
    assert row["recovery_hits"] == 1              # only frame 3 agrees
    assert row["detect_calls"] == 6 + 18
    # Present: 2, 4, 6 by detection plus 3 by recovery.
    assert row["frames_present"] == 4
    assert row["forward_passes"] == 0             # the stub is not a DBNet
    assert candidate.presence_frames == [2, 3, 4, 6]


def _dbnet_like(hl_boxes):
    """A DBNetDetector whose network outputs nothing, with counted forwards."""
    detector = object.__new__(DBNetDetector)
    detector._input_w = detector._input_h = 64
    detector._scale = 1 / 255
    detector._mean = (0, 0, 0)
    detector._bin_thresh, detector._box_thresh, detector._unclip_ratio = .3, .5, 1.5
    detector._adaptive = False
    detector._manual_only = False
    detector._net = SimpleNamespace(
        setInput=lambda blob: None,
        forward=lambda: np.zeros((1, 1, 64, 64), np.float32),
    )
    detector._hl_model = SimpleNamespace(
        setInputParams=lambda **kwargs: None,
        detect=lambda frame: (
            [np.array([[[1, 1]], [[9, 1]], [[9, 9]], [[1, 9]]])] * hl_boxes,
            [0.9] * hl_boxes,
        ),
    )
    return detector


def test_empty_frame_costs_two_forward_passes_and_a_hit_costs_one():
    frame = np.zeros((48, 64, 3), np.uint8)
    handle = perfstats.start()
    with perfstats.phase("refine"):
        _dbnet_like(hl_boxes=0).detect(frame)   # hl finds nothing -> manual too
        _dbnet_like(hl_boxes=1).detect(frame)   # hl finds a box -> one pass
        manual = _dbnet_like(hl_boxes=1)
        manual._manual_only = True
        manual.detect(frame)                    # manual only -> one pass
    row = perfstats.finish(handle)["refine"]
    assert (row["forward_hl"], row["forward_manual"]) == (2, 2)
    assert row["forward_passes"] == 4
    assert row["detect_calls"] == 0  # forwards are counted inside the detector


def test_worst_case_empty_frame_is_fourteen_forward_passes(tmp_path):
    """2 for the frame itself plus 6 recovery variants x 2 (the PRD's P2-3 case)."""
    video = _write_indexed_video(tmp_path / "indexed.avi")
    candidate = SimpleNamespace(
        id="A", bbox=(10, 28, 54, 44), detector_backed=True,
        presence_frames=[], temporal_sample_indices=[],
    )
    result = SimpleNamespace(
        detector=_dbnet_like(hl_boxes=0), candidates=[candidate], sample_indices=[0],
        frame_shape=(H, W), sampled_frame_boxes={},
    )
    handle = perfstats.start()
    with perfstats.phase("refine"):
        refine_temporal_presence(video, result, {"A": [Segment(4, 5)]}, N)
    row = perfstats.finish(handle)["refine"]
    assert row["frames_converted"] == 1 and row["frames_grabbed"] == 4
    assert row["detect_calls"] == 1 + 6
    assert row["recovery_calls"] == 1 and row["recovery_hits"] == 0
    assert row["forward_passes"] == 14
    assert (row["forward_hl"], row["forward_manual"]) == (7, 7)
    assert row["frames_present"] == 0


class ProbeDetector:
    """No text on frames 3 and 4, a subtitle box everywhere else."""

    def detect(self, frame):
        if frame.shape[:2] != (H, W):
            return []
        return [] if int(frame[0, 0, 0]) // 10 in (3, 4) else [_box(12, 30, 52, 42)]


def test_boundary_probe_counters_on_a_scripted_gap(tmp_path):
    from videowipe.plan import MaskAsset, Source, TemporalResolution, Track, WipePlan
    from videowipe.planning import _add_spatial_segments

    video = _write_indexed_video(tmp_path / "indexed.avi")
    track = Track("c1", "subtitle", "subtitle", "remove", (10, 28, 54, 44), 1, 1,
                  "default", [Segment(0, 3), Segment(5, 8)], "c1")
    plan = WipePlan("wipe_plan", 1, Source("x.avi", "a" * 64, W, H, 4, N), {},
                    TemporalResolution(0, 0, 0), MaskAsset("m.npz", ""), [track])
    observed = {index: [_box(12, 30, 52, 42)] for index in (0, 1, 2, 5, 6, 7)}
    result = SimpleNamespace(
        frame_shape=(H, W), detector=ProbeDetector(), sampled_frame_boxes=observed,
        candidates=[SimpleNamespace(id="c1", type="subtitle", detector_backed=True)],
    )
    handle = perfstats.start()
    _add_spatial_segments(plan, result, video_path=video)
    stats = perfstats.finish(handle)

    # The two-frame gap (3, 4) and the zone after the last segment (8) are
    # probed: three frames, reached by grabbing 0..2, then 5..7.
    probe = stats["boundary_probe"]
    assert probe["frames_converted"] == 3 and probe["frames_grabbed"] == 3 + 3
    # One detect per probed frame; the two empty gap frames each get a recovery
    # attempt of six variants that finds nothing.
    assert probe["detect_calls"] == 3 + 2 * 6
    assert probe["recovery_calls"] == 2 and probe["recovery_frames"] == 2
    assert probe["recovery_detect_calls"] == 12 and probe["recovery_hits"] == 0
    assert probe["frames_present"] == 1  # only frame 8 has a box
    assert "descender_recovery" in stats


# ── engine integration ──────────────────────────────────────────────────────

PLAN_W, PLAN_H, PLAN_FRAMES = 96, 64, 30


def _write_plan_video(path, frames=PLAN_FRAMES):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (PLAN_W, PLAN_H))
    for _ in range(frames):
        frame = np.zeros((PLAN_H, PLAN_W, 3), dtype=np.uint8)
        frame[50:60, 8:88] = 200
        frame[5:15, 38:58] = 200
        writer.write(frame)
    writer.release()


class PlanDetector:
    def detect(self, frame):
        if frame.shape[:2] != (PLAN_H, PLAN_W):
            return []
        return [
            TextBox(np.array([[8, 50], [88, 50], [88, 60], [8, 60]]), 0.9, "subtitle"),
            TextBox(np.array([[38, 5], [58, 5], [58, 15], [38, 15]]), 0.95, "MangoTV"),
        ]


def test_engine_plan_reports_each_phase_and_isolates_runs(tmp_path):
    video = tmp_path / "input.mp4"
    _write_plan_video(video)
    engine = WipeEngine(task="clean")
    assert engine.last_detection_stats == {}

    def plan(name):
        engine.plan(WipeRequest(video=video, output_dir=tmp_path / name, detector=PlanDetector()))
        return engine.last_detection_stats

    first = plan("one")
    assert list(first) == ["coarse_sampling", "band_fallback", "refine", "descender_recovery"]
    # Every frame is a sample, read once, converted once, detected once.
    coarse = first["coarse_sampling"]
    assert coarse["frames_converted"] == PLAN_FRAMES and coarse["frames_grabbed"] == 0
    assert coarse["detect_calls"] == PLAN_FRAMES and coarse["forward_passes"] == 0
    # The subtitle is active on every frame and all of them were sampled, so
    # refine converts them but reuses the sampled evidence instead of detecting.
    refine = first["refine"]
    assert refine["frames_converted"] == PLAN_FRAMES and refine["detect_calls"] == 0
    assert refine["frames_present"] == PLAN_FRAMES
    assert refine["wall_s"] > 0
    # The light band fallback runs two variants of the bottom band per sample.
    fallback = first["band_fallback"]
    assert fallback["frames_converted"] == PLAN_FRAMES and fallback["detect_calls"] == 2 * PLAN_FRAMES
    assert first["descender_recovery"]["frames_converted"] == PLAN_FRAMES

    second = plan("two")
    scrub = lambda stats: {p: {**row, "wall_s": 0} for p, row in stats.items()}
    assert scrub(second) == scrub(first)  # same counts, not doubled
    returned = engine.last_detection_stats
    returned["refine"]["frames_converted"] = -1
    assert engine.last_detection_stats["refine"]["frames_converted"] == PLAN_FRAMES


def test_engine_plan_on_a_worker_thread_is_counted(tmp_path):
    video = tmp_path / "input.mp4"
    _write_plan_video(video, frames=8)
    engine = WipeEngine(task="clean")
    errors = []

    def run():
        try:
            engine.plan(WipeRequest(video=video, output_dir=tmp_path / "out", detector=PlanDetector()))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert not errors
    assert engine.last_detection_stats["coarse_sampling"]["frames_converted"] == 8


def test_fast_mode_has_no_refine_phase(tmp_path):
    video = tmp_path / "input.mp4"
    _write_plan_video(video, frames=8)
    engine = WipeEngine(task="clean")
    engine.plan(WipeRequest(
        video=video, output_dir=tmp_path / "out", detector=PlanDetector(), detect_mode="fast",
    ))
    stats = engine.last_detection_stats
    assert "coarse_sampling" in stats and "refine" not in stats


# ── results are unchanged ───────────────────────────────────────────────────

def _decode(path):
    reader = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        frames.append(frame)
    reader.release()
    return frames


def _process(tmp_path, name, monkeypatch, enabled):
    from videowipe.inpainters import sttn

    if enabled:
        monkeypatch.delenv("VIDEOWIPE_PERF_STATS", raising=False)
    else:
        monkeypatch.setenv("VIDEOWIPE_PERF_STATS", "0")
    video = tmp_path / "input.mp4"
    if not video.exists():
        _write_plan_video(video)
    engine = WipeEngine(task="clean", gap=10)
    painter = sttn.STTNInpainter()
    painter.backend = SimpleNamespace(
        benchmark_metadata=lambda: {"device": "cpu", "weight_sha256": "test"},
    )
    engine._task_impl.inpainter = painter
    engine._task_impl.backend = painter.backend
    engine._model_loaded = True
    output = tmp_path / name
    out_path = engine.process(video=str(video), output=str(output), detector=PlanDetector())
    plan = json.loads((output / "wipe_plan.json").read_text())
    masks = dict(np.load(output / "wipe_plan_masks.npz"))
    benchmark = json.loads((output / "benchmark.json").read_text())
    return out_path, plan, masks, benchmark, engine


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires ffmpeg")
def test_plan_and_output_frames_are_identical_with_the_collector_on_and_off(tmp_path, monkeypatch):
    from videowipe.inpainters import sttn

    monkeypatch.setattr(sttn, "_process_segment", lambda frames, *a: [
        np.full((120, 640, 3), 90, np.uint8) for _ in frames])
    out_on, plan_on, masks_on, bm_on, engine_on = _process(tmp_path, "on", monkeypatch, True)
    out_off, plan_off, masks_off, bm_off, engine_off = _process(tmp_path, "off", monkeypatch, False)

    assert plan_on == plan_off
    assert masks_on.keys() == masks_off.keys()
    assert all(np.array_equal(masks_on[key], masks_off[key]) for key in masks_on)
    frames_on, frames_off = _decode(out_on), _decode(out_off)
    assert len(frames_on) == len(frames_off) == PLAN_FRAMES
    assert all(np.array_equal(a, b) for a, b in zip(frames_on, frames_off))

    stats = bm_on["timing"].pop("detection_stats")
    assert stats["coarse_sampling"]["frames_converted"] == PLAN_FRAMES
    assert "detection_stats" not in bm_off["timing"]
    assert engine_off.last_detection_stats == {}
    # The existing timing keys keep their meaning in both runs.
    assert set(bm_off["timing"]) <= set(bm_on["timing"])
    for key in ("detection_s", "inpainting_s", "prediction_s", "composition_s"):
        assert key in bm_on["timing"] and key in bm_off["timing"]
    # The inpainting split lands in benchmark.json next to them.
    assert bm_on["timing"]["frames_written"] == PLAN_FRAMES
    assert bm_on["timing"]["model_s"] >= 0
    assert (bm_on["timing"]["segments"], bm_off["timing"]["segments"]) == (3, 3)


# ── server surface ──────────────────────────────────────────────────────────

def test_timing_validation_accepts_numbers_and_stat_objects_only():
    from videowipe.server.jobs import _valid_timing

    assert _valid_timing("plan_s", 1.5) and _valid_timing("n", 3)
    assert not _valid_timing("plan_s", -1) and not _valid_timing("plan_s", True)
    assert not _valid_timing("plan_s", float("inf")) and not _valid_timing("plan_s", "1")
    assert not _valid_timing("plan_s", {"refine": {"detect_calls": 1}})  # only *_stats
    assert _valid_timing("detection_stats", {"refine": {"wall_s": 0.5, "detect_calls": 3}})
    assert _valid_timing("review_stats", {})
    assert not _valid_timing("detection_stats", {"refine": {"wall_s": "x"}})
    assert not _valid_timing("detection_stats", {"refine": 1})
    assert not _valid_timing("detection_stats", {"refine": {"detect_calls": -1}})


STATS = {"coarse_sampling": {"wall_s": 0.25, "entries": 1, "frames_converted": 8, "detect_calls": 8}}


def test_job_timings_carry_the_stats_and_survive_a_restart(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from test_server import FakeEngine, _post_video, _wait_for_state, _write_test_video
    from videowipe.server import app as server_app
    from videowipe.server import jobs

    class StatsEngine(FakeEngine):
        last_detection_stats = STATS

    jobs.reset_jobs()
    monkeypatch.setenv("VIDEOWIPE_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(server_app, "_engine", StatsEngine())
    try:
        with TestClient(server_app.app) as client:
            video = tmp_path / "input.mp4"
            _write_test_video(video)
            job_id = _post_video(client, video).json()["id"]
            snapshot = _wait_for_state(client, job_id, "preview_ready")
            assert snapshot["timings"]["detection_stats"] == STATS
            assert snapshot["timings"]["plan_s"] > 0
            jobs.reset_jobs()  # simulate a restart: the manifest must still load
            restored = client.get("/jobs/current").json()
            assert restored["id"] == job_id
            assert restored["timings"]["detection_stats"] == STATS
    finally:
        jobs.reset_jobs()
        monkeypatch.setattr(server_app, "_engine", None)


def test_engines_without_stats_leave_job_timings_unchanged(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from test_server import FakeEngine, _post_video, _wait_for_state, _write_test_video
    from videowipe.server import app as server_app
    from videowipe.server import jobs

    jobs.reset_jobs()
    monkeypatch.setenv("VIDEOWIPE_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(server_app, "_engine", FakeEngine())
    try:
        with TestClient(server_app.app) as client:
            video = tmp_path / "input.mp4"
            _write_test_video(video)
            job_id = _post_video(client, video).json()["id"]
            snapshot = _wait_for_state(client, job_id, "preview_ready")
            assert "detection_stats" not in snapshot["timings"]
    finally:
        jobs.reset_jobs()
        monkeypatch.setattr(server_app, "_engine", None)
