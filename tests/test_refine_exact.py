"""Exactness tests for the refine / skip / trial-positioning speed-ups.

``original_refine_temporal_presence`` below is a verbatim copy of the function
as it was before the bookkeeping and decode changes; the current implementation
must reproduce its results exactly. The only intended differences are listed in
``test_early_exit_*`` and ``test_short_file_in_the_unread_tail_*``.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Iterable

import cv2
import numpy as np
import pytest

from videowipe.detect import (
    CleanDetectionResult,
    TextBox,
    _box_overlaps_bbox,
    _contrast_recovered_boxes,
    _needs_contrast_recovery,
    refine_temporal_presence,
)
from videowipe.errors import ProcessingCancelledError
from videowipe.plan import Segment
from videowipe.video import skip_frames, skip_frames_concurrently


# ── Oracle: the original implementation, unchanged apart from its name ──────

def original_refine_temporal_presence(
    video_path: str,
    result: CleanDetectionResult,
    candidate_segments: dict[str, Iterable[object]],
    expected_frame_count: int,
    progress=None,
    check_cancelled=None,
) -> list[str]:
    """Densely recheck detector-backed remove candidates inside coarse segments.

    Detection remains sequential and each active video frame is passed to the
    detector once.  A detector error is negative evidence: keeping that frame
    is safer than carrying forward a coarse remove decision.  A frame whose
    normal detection succeeds but finds nothing may still recover evidence
    through :func:`_contrast_recovered_boxes` — per-frame positive evidence
    only; empty, failed, and protected content never gains presence.
    """
    detector = result.detector
    candidates = {
        candidate.id: candidate
        for candidate in result.candidates
        if candidate.id in candidate_segments and getattr(
            candidate, "detector_backed", bool(candidate.presence_frames)
        )
    }
    if detector is None or not candidates:
        return []

    segments = {
        candidate_id: [
            (int(segment.start), int(segment.end))
            for segment in candidate_segments[candidate_id]
        ]
        for candidate_id in candidates
    }
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for temporal refinement: {video_path}")

    warnings: list[str] = []
    failed = getattr(result, "failed_frame_indices", set())
    result.failed_frame_indices = failed
    try:
        frame_index = 0
        while frame_index < expected_frame_count:
            if check_cancelled is not None:
                check_cancelled()
            ok, frame = cap.read()
            if not ok:
                raise ValueError(
                    f"Temporal refinement decoded {frame_index} frames; "
                    f"expected {expected_frame_count}"
                )
            active = [
                candidate
                for candidate_id, candidate in candidates.items()
                if any(start <= frame_index < end for start, end in segments[candidate_id])
            ]
            if active:
                for candidate in active:
                    candidate.temporal_sample_indices = sorted(set(
                        candidate.temporal_sample_indices or result.sample_indices
                    ) | {frame_index})
                    candidate.presence_frames = [
                        index for index in candidate.presence_frames if index != frame_index
                    ]
                if frame_index in failed:
                    boxes = []
                    detect_failed = True
                    result.sampled_frame_boxes[frame_index] = []
                elif frame_index in result.sampled_frame_boxes:
                    boxes = result.sampled_frame_boxes[frame_index]
                    detect_failed = False
                else:
                    try:
                        boxes = detector.detect(frame)
                        detect_failed = False
                    except ProcessingCancelledError:
                        raise
                    except Exception as exc:  # safety boundary: failed frame remains keep.
                        warnings.append(
                            f"temporal refinement failed at frame {frame_index}; "
                            f"keeping frame: {exc}"
                        )
                        boxes = []
                        detect_failed = True
                        failed.add(frame_index)
                    result.sampled_frame_boxes[frame_index] = boxes
                for candidate in active:
                    if any(
                        _box_overlaps_bbox(
                            box, candidate.bbox,
                            frame.shape[1], frame.shape[0],
                        )
                        for box in boxes
                    ):
                        candidate.presence_frames.append(frame_index)
                    if not detect_failed and _needs_contrast_recovery(
                        boxes, candidate.bbox, frame.shape[1], frame.shape[0],
                    ):
                        # White flash / cross-dissolve recovery: per-frame
                        # positive evidence from contrast-restored variants of
                        # this candidate's own band. Failed detection above
                        # never gets a second chance.
                        try:
                            recovered = _contrast_recovered_boxes(
                                detector, frame, candidate.bbox,
                            )
                        except ProcessingCancelledError:
                            raise
                        except Exception as exc:
                            warnings.append(
                                f"temporal recovery failed at frame {frame_index}; "
                                f"keeping frame: {exc}"
                            )
                            failed.add(frame_index)
                            result.sampled_frame_boxes[frame_index] = []
                            for affected in active:
                                affected.presence_frames = [
                                    index for index in affected.presence_frames
                                    if index != frame_index
                                ]
                            break
                        if recovered:
                            candidate.presence_frames.append(frame_index)
                            result.sampled_frame_boxes[frame_index] = (
                                list(result.sampled_frame_boxes[frame_index])
                                + recovered
                            )
            frame_index += 1
            if progress is not None:
                progress(frame_index, expected_frame_count)
    finally:
        cap.release()

    for candidate in candidates.values():
        candidate.presence_frames = sorted(set(candidate.presence_frames))
    return warnings


# ── Fixtures ────────────────────────────────────────────────────────────────

WIDTH, HEIGHT, FRAMES = 64, 48, 40
STEP = 5  # frame i is filled with the value i * STEP (lossless FFV1).


def make_video(path, count=FRAMES):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 8, (WIDTH, HEIGHT))
    assert writer.isOpened()
    for index in range(count):
        writer.write(np.full((HEIGHT, WIDTH, 3), index * STEP, np.uint8))
    writer.release()
    return str(path)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("refine") / "indexed.avi")


def _box(x1, y1, x2, y2):
    return TextBox(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64), 0.9)


class StubDetector:
    """Deterministic boxes derived from pixel content, with a call log."""

    def __init__(self, raises=()):
        self.raises = set(raises)
        self.calls = []
        self.current = None

    def detect(self, frame):
        full = frame.shape[:2] == (HEIGHT, WIDTH)
        if full:
            self.current = int(frame[0, 0, 0]) // STEP
        self.calls.append((full, tuple(frame.shape), int(frame.sum())))
        index = self.current
        if full:
            if index in self.raises:
                raise RuntimeError(f"stub failure at {index}")
            if index % 3 == 0:
                return [_box(12, 32, 48, 40)]   # wide overlap with candidate A
            if index % 3 == 1:
                return [_box(2, 0, 8, 5)]       # elsewhere: triggers recovery
            return []                           # empty: triggers recovery
        # Contrast-recovery crops: agree across variants for some frames only.
        if index % 4 == 1:
            return [_box(10, 8, 30, 16)]
        if index % 7 == 3:
            raise RuntimeError(f"stub recovery failure at {index}")
        return []


def make_candidate(cid, bbox, presence=(), temporal=()):
    return SimpleNamespace(
        id=cid, bbox=bbox, detector_backed=True,
        presence_frames=list(presence), temporal_sample_indices=list(temporal),
    )


def make_result(detector, with_cache=True):
    candidates = [
        make_candidate("A", (10, 30, 50, 44), presence=[30, 5, 5, 17, 1, 12]),
        make_candidate("B", (0, 0, 30, 12), presence=[2, 3], temporal=[3, 4]),
        make_candidate("C", (0, 0, 63, 47), presence=[0]),
        make_candidate("D", (0, 20, 20, 30)),  # never active in any scenario
    ]
    result = SimpleNamespace(
        detector=detector, candidates=candidates, sample_indices=[0, 10, 20, 30],
        frame_shape=(HEIGHT, WIDTH), sampled_frame_boxes={},
    )
    if with_cache:
        result.sampled_frame_boxes = {4: [_box(12, 32, 48, 40)], 6: [], 20: [_box(2, 0, 8, 5)]}
        result.failed_frame_indices = {8, 9}
    return result


def snapshot(result, warnings, progress, detector):
    def boxes(items):
        return [(b.points.tolist(), b.confidence, b.text) for b in items]
    return {
        "candidates": {
            c.id: (list(c.presence_frames), list(c.temporal_sample_indices))
            for c in result.candidates
        },
        "boxes": {i: boxes(b) for i, b in sorted(result.sampled_frame_boxes.items())},
        "failed": sorted(getattr(result, "failed_frame_indices", set())),
        "warnings": list(warnings),
        "progress": list(progress),
        "calls": list(detector.calls),
    }


def run(function, path, segments, expected, raises=(), with_cache=True, **kwargs):
    detector = StubDetector(raises)
    result = make_result(detector, with_cache)
    progress = []
    warnings = function(
        path, result, segments, expected,
        progress=lambda done, total: progress.append((done, total)), **kwargs,
    )
    return snapshot(result, warnings, progress, detector)


def segs(**rows):
    return {key: [Segment(a, b) for a, b in value] for key, value in rows.items()}


FULL = segs(A=[(2, 10), (15, 20)], B=[(0, 25)], C=[(0, FRAMES)])
EARLY = segs(A=[(2, 10), (15, 20)], B=[(0, 25)])
SPARSE = segs(A=[(5, 9), (12, 14)])
EVERY = segs(C=[(0, FRAMES)])


def compare(path, segments, expected=FRAMES, **kwargs):
    old = run(original_refine_temporal_presence, path, segments, expected, **kwargs)
    new = run(refine_temporal_presence, path, segments, expected, **kwargs)
    return old, new


# ── P1-1 / P1-2: refine equality against the original ──────────────────────

@pytest.mark.parametrize("raises", [(), (3, 12, 17), (0, 5, 24)])
@pytest.mark.parametrize("with_cache", [True, False])
def test_refine_matches_original_when_the_whole_file_is_read(video, raises, with_cache):
    old, new = compare(video, FULL, raises=raises, with_cache=with_cache)
    assert new == old
    # Make sure the scenario exercises what it claims to.
    assert any(not full for full, _, _ in old["calls"]), "recovery never ran"
    assert old["candidates"]["D"] == ([], [])
    if raises:
        assert old["warnings"]


def test_refine_recovery_failure_path_matches_original(video):
    old, new = compare(video, FULL)
    assert new == old
    assert any("temporal recovery failed" in w for w in old["warnings"])


def test_candidate_active_on_every_frame_matches_original(video):
    old, new = compare(video, EVERY)
    assert new == old
    assert old["candidates"]["C"][1] == list(range(FRAMES))


def test_frames_without_an_active_candidate_match_original(video):
    old, new = compare(video, SPARSE)
    assert new["candidates"] == old["candidates"]
    assert new["boxes"] == old["boxes"]
    assert new["failed"] == old["failed"]
    assert new["warnings"] == old["warnings"]
    assert new["calls"] == old["calls"]
    assert set(new["boxes"]) >= {5, 6, 7, 8, 12, 13}
    assert 10 not in new["boxes"]


@pytest.mark.parametrize("segments", [EARLY, SPARSE])
def test_early_exit_results_match_and_final_progress_is_kept(video, segments):
    old, new = compare(video, segments)
    for key in ("candidates", "boxes", "failed", "warnings", "calls"):
        assert new[key] == old[key], key
    stop = max(s.end for rows in segments.values() for s in rows)
    assert stop < FRAMES
    # Same events for every frame that is still decoded, then the same final
    # event as before; the silent tail's per-frame events are no longer sent.
    assert new["progress"][:stop] == old["progress"][:stop]
    assert new["progress"][-1] == old["progress"][-1] == (FRAMES, FRAMES)
    assert len(new["progress"]) == stop + 1


def test_progress_is_identical_when_nothing_is_skipped(video):
    old, new = compare(video, FULL)
    assert new["progress"] == old["progress"] == [(i, FRAMES) for i in range(1, FRAMES + 1)]


def test_short_file_inside_the_needed_range_still_raises(tmp_path):
    short = make_video(tmp_path / "short.avi", count=30)
    for function in (original_refine_temporal_presence, refine_temporal_presence):
        with pytest.raises(ValueError, match="decoded 30 frames; expected 40"):
            run(function, short, FULL, FRAMES)


def test_short_file_inside_a_gap_of_the_needed_range_still_raises(tmp_path):
    short = make_video(tmp_path / "short.avi", count=12)
    segments = segs(A=[(2, 5), (20, 25)])
    for function in (original_refine_temporal_presence, refine_temporal_presence):
        with pytest.raises(ValueError, match="decoded 12 frames; expected 40"):
            run(function, short, segments, FRAMES)


def test_short_file_in_the_unread_tail_is_the_one_documented_difference(tmp_path):
    short = make_video(tmp_path / "short.avi", count=30)
    with pytest.raises(ValueError, match="decoded 30 frames; expected 40"):
        run(original_refine_temporal_presence, short, EARLY, FRAMES)
    new = run(refine_temporal_presence, short, EARLY, FRAMES)
    full = make_video(tmp_path / "full.avi")
    assert new == run(refine_temporal_presence, full, EARLY, FRAMES)


@pytest.mark.parametrize("cancel_at", [0, 5, 13])
def test_interrupted_run_leaves_the_same_candidate_state(video, cancel_at):
    def run_cancelled(function):
        detector = StubDetector((3,))
        result = make_result(detector)
        seen = []

        def check():
            seen.append(1)
            if len(seen) > cancel_at:
                raise ProcessingCancelledError("cancelled")

        with pytest.raises(ProcessingCancelledError):
            function(video, result, FULL, FRAMES, check_cancelled=check)
        return snapshot(result, [], [], detector)

    assert run_cancelled(refine_temporal_presence) == run_cancelled(original_refine_temporal_presence)


def test_progress_callback_failure_leaves_the_same_candidate_state(video):
    def run_failing(function):
        detector = StubDetector()
        result = make_result(detector)

        def progress(done, total):
            if done == 9:
                raise RuntimeError("stop")

        with pytest.raises(RuntimeError, match="stop"):
            function(video, result, FULL, FRAMES, progress=progress)
        return snapshot(result, [], [], detector)

    assert run_failing(refine_temporal_presence) == run_failing(original_refine_temporal_presence)


# ── P1-1: scaling ───────────────────────────────────────────────────────────

class FakeCapture:
    """Minimal capture: frames are tiny and cost nothing to produce."""

    def __init__(self, total):
        self.total = total
        self.position = 0
        self.frame = np.zeros((20, 20, 3), np.uint8)

    def isOpened(self):
        return True

    def grab(self):
        if self.position >= self.total:
            return False
        self.position += 1
        return True

    def retrieve(self):
        return True, self.frame

    def read(self):
        return (self.grab(), self.frame)

    def release(self):
        pass


def _refine_stub_time(monkeypatch, frames):
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: FakeCapture(frames))
    wide = _box(0, 0, 19, 19)  # overlaps and spans the band: no recovery runs
    detector = SimpleNamespace(detect=lambda frame: [wide])
    candidate = make_candidate("A", (0, 0, 19, 19))
    result = SimpleNamespace(
        detector=detector, candidates=[candidate], sample_indices=[0],
        frame_shape=(20, 20), sampled_frame_boxes={},
    )
    started = time.perf_counter()
    refine_temporal_presence("ignored.avi", result, {"A": [Segment(0, frames)]}, frames)
    elapsed = time.perf_counter() - started
    assert candidate.presence_frames == list(range(frames))
    assert candidate.temporal_sample_indices == sorted({0} | set(range(frames)))
    return elapsed


def test_refine_bookkeeping_scales_linearly(monkeypatch):
    # The per-frame list rebuilds were quadratic: 4x the frames cost about 16x.
    small = min(_refine_stub_time(monkeypatch, 4000) for _ in range(3))
    large = min(_refine_stub_time(monkeypatch, 16000) for _ in range(3))
    assert large < small * 9, (small, large)


# ── P1-2(c): skip_frames ────────────────────────────────────────────────────

class CountingCapture:
    def __init__(self, path):
        self.reader = cv2.VideoCapture(path)
        self.reads = 0
        self.grabs = 0

    def read(self):
        self.reads += 1
        return self.reader.read()

    def grab(self):
        self.grabs += 1
        return self.reader.grab()

    def retrieve(self):
        return self.reader.retrieve()

    def release(self):
        self.reader.release()


def _old_skip(reader, count):
    for _ in range(count):
        assert reader.read()[0]


@pytest.mark.parametrize("count", [0, 1, 7, 39])
def test_skip_frames_lands_on_the_same_frame_as_read_based_skipping(video, count):
    expected = cv2.VideoCapture(video)
    _old_skip(expected, count)
    _, wanted = expected.read()
    expected.release()
    reader = CountingCapture(video)
    skip_frames(reader, count)
    ok, frame = reader.read()
    reader.release()
    assert ok and np.array_equal(frame, wanted)
    assert int(frame[0, 0, 0]) == count * STEP
    assert (reader.grabs, reader.reads) == (count, 1)


def test_skip_frames_uses_read_for_readers_without_grab(video):
    real = cv2.VideoCapture(video)
    calls = []

    def read():
        calls.append(1)
        return real.read()

    fake = SimpleNamespace(read=read)
    skip_frames(fake, 6)
    assert len(calls) == 6
    ok, frame = real.read()
    real.release()
    assert ok and int(frame[0, 0, 0]) == 6 * STEP


def test_skip_frames_reports_short_reads_for_grab_readers(video):
    reader = CountingCapture(video)
    with pytest.raises(ValueError, match="decoded 40"):
        skip_frames(reader, 50)
    reader.release()


def test_skip_frames_cancellation_stops_before_decoding():
    calls = []
    reader = SimpleNamespace(grab=lambda: calls.append(1) or True)

    def cancelled():
        raise ProcessingCancelledError("cancelled")

    with pytest.raises(ProcessingCancelledError):
        skip_frames(reader, 10, cancelled)
    assert calls == []


# ── P1-2(d): descender recovery ─────────────────────────────────────────────

def test_recover_descenders_matches_read_based_decoding(tmp_path, monkeypatch):
    from videowipe import planning
    from videowipe.plan import MaskAsset, Source, TemporalResolution, Track, WipePlan, _spatial_mask

    path = tmp_path / "outline.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 8, (160, 100))
    for i in range(16):
        frame = np.full((100, 160, 3), 50, np.uint8)
        if i in (5, 9):
            frame[85:95, 60:65] = 0
            frame[85:90, 61:64] = 255
        writer.write(frame)
    writer.release()
    original_rows = [
        dict(start=5, end=6, boxes=[[50, 80, 75, 92]]),
        dict(start=9, end=10, boxes=[[50, 80, 75, 92]]),
    ]

    def build():
        track = Track("c1", "subtitle", "subtitle", "remove", (30, 75, 130, 99), 1, 1,
                      "default", [Segment(5, 10)], "c1")
        # Frames 6-8 have no boxes for any track: the gap is only advanced.
        track.spatial_segments = [dict(row) for row in original_rows]
        track.mask = _spatial_mask(track, (100, 160))
        plan = WipePlan("wipe_plan", 3, Source("outline.avi", "a" * 64, 160, 100, 8, 16), {},
                        TemporalResolution(4, 4, 4), MaskAsset("wipe_plan_masks.npz", ""), [track])
        return plan, track

    plan, track = build()
    planning._recover_descenders(plan, [track], str(path), None)
    grabbed = track.spatial_segments
    assert grabbed != original_rows  # the outline evidence did extend the boxes

    capture = cv2.VideoCapture

    class ReadOnly:
        def __init__(self, source):
            self.reader = capture(source)

        def isOpened(self):
            return self.reader.isOpened()

        def read(self):
            return self.reader.read()

        def release(self):
            self.reader.release()

    monkeypatch.setattr(cv2, "VideoCapture", ReadOnly)
    plan2, track2 = build()
    planning._recover_descenders(plan2, [track2], str(path), None)
    assert grabbed == track2.spatial_segments


# ── P1-8: concurrent skip ───────────────────────────────────────────────────

def test_concurrent_skip_leaves_both_readers_at_the_same_position(video):
    first, second = CountingCapture(video), CountingCapture(video)
    skip_frames_concurrently((first, second), 17)
    frames = [first.read()[1], second.read()[1]]
    first.release()
    second.release()
    assert all(int(f[0, 0, 0]) == 17 * STEP for f in frames)
    assert (first.grabs, second.grabs) == (17, 17)


def test_concurrent_skip_works_with_read_only_readers(video):
    readers = [cv2.VideoCapture(video), cv2.VideoCapture(video)]
    fakes = [SimpleNamespace(read=r.read) for r in readers]
    skip_frames_concurrently(fakes, 11)
    assert [int(r.read()[1][0, 0, 0]) for r in readers] == [11 * STEP] * 2
    for r in readers:
        r.release()


def test_concurrent_skip_calls_the_callback_once_per_frame_and_never_in_parallel(video):
    first, second = CountingCapture(video), CountingCapture(video)
    inside = []
    overlaps = []
    count = []

    def callback():
        inside.append(1)
        if len(inside) > 1:
            overlaps.append(1)
        time.sleep(0.0005)
        count.append(1)
        inside.pop()

    skip_frames_concurrently((first, second), 12, callback)
    first.release()
    second.release()
    assert len(count) == 24 and not overlaps


class SlowFake:
    def __init__(self, delay=0.0):
        self.grabs = 0
        self.delay = delay

    def grab(self):
        self.grabs += 1
        time.sleep(self.delay)
        return True


def test_callback_exception_propagates_and_stops_both_skips():
    first, second = SlowFake(0.001), SlowFake(0.001)
    calls = []

    def callback():
        calls.append(threading.current_thread())
        if len(calls) == 9:
            raise ProcessingCancelledError("cancelled")

    with pytest.raises(ProcessingCancelledError):
        skip_frames_concurrently((first, second), 100000, callback)
    assert first.grabs < 100 and second.grabs < 100
    assert len(calls) == 9


def test_callback_exception_in_the_worker_thread_propagates_and_stops_both_skips():
    first, second = SlowFake(0.001), SlowFake(0.001)
    main = threading.current_thread()
    calls = []

    def callback():
        calls.append(1)
        if threading.current_thread() is not main:
            raise ProcessingCancelledError("cancelled in worker")

    with pytest.raises(ProcessingCancelledError, match="in worker"):
        skip_frames_concurrently((first, second), 100000, callback)
    assert first.grabs < 100 and second.grabs < 100


def test_reader_failure_in_either_thread_propagates_and_stops_the_other():
    class Dies(SlowFake):
        def grab(self):
            self.grabs += 1
            return self.grabs <= 5

    for order in ("first", "second"):
        dying, other = Dies(), SlowFake(0.001)
        readers = (dying, other) if order == "first" else (other, dying)
        started = time.perf_counter()
        with pytest.raises(ValueError, match="decoded 5"):
            skip_frames_concurrently(readers, 100000)
        assert other.grabs < 1000 and time.perf_counter() - started < 5


def test_single_reader_is_skipped_directly(video):
    reader = CountingCapture(video)
    skip_frames_concurrently([reader], 4)
    assert reader.grabs == 4
    reader.release()


# ── P1-8: trial positioning and grab in the inference loop ──────────────────

def test_trial_output_is_identical_when_inference_frames_are_grabbed(tmp_path, monkeypatch):
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("requires ffmpeg")
    from videowipe.inpainters import sttn
    from videowipe.inpainters.base import InpaintJob

    path = tmp_path / "gap.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 8, (160, 100))
    for index in range(16):
        writer.write(np.full((100, 160, 3), index, np.uint8))
    writer.release()
    mask = np.zeros((100, 160, 1), np.uint8)
    mask[80:90, 35:125] = 1
    capture_type = cv2.VideoCapture
    pixels = []
    blend = sttn._blend_frame_regions

    def capture(*args):
        out = blend(*args)
        pixels.append(out.copy())
        return out

    def frame_mask(index):
        # Segments of four frames: only the last two have anything to remove,
        # so the first two are inactive and need no pixels for inference.
        return mask[:, :, 0] if index >= 8 else np.zeros((100, 160), np.uint8)

    class Spy:
        def __init__(self, source, hide_grab):
            self.reader = capture_type(source)
            self.counts = {"read": 0, "grab": 0}
            if not hide_grab:
                self.grab = self._grab

        def _grab(self):
            self.counts["grab"] += 1
            return self.reader.grab()

        def isOpened(self):
            return self.reader.isOpened()

        def read(self):
            self.counts["read"] += 1
            return self.reader.read()

        def release(self):
            self.reader.release()

    painter = sttn.STTNInpainter()
    painter.backend = object()

    def run_once(hide_grab, interval):
        with monkeypatch.context() as patch:
            patch.setattr(sttn, "_process_segment",
                          lambda frames, *args: [np.zeros_like(f) for f in frames])
            patch.setattr(sttn, "_blend_frame_regions", capture)
            patch.setattr(sttn, "_VIDEO_CAPTURE_TYPE", Spy)
            patch.setattr(cv2, "VideoCapture", lambda p: Spy(p, hide_grab))
            pixels.clear()
            reader = Spy(str(path), hide_grab)
            try:
                painter.inpaint(InpaintJob(str(path), mask, str(tmp_path), 8, 16, 160, 100,
                                           reader=reader, gap=4, trial_range=interval,
                                           frame_mask=frame_mask))
            finally:
                reader.release()
            return [p.copy() for p in pixels], reader.counts

    for interval in (None, (5, 15)):
        with_grab, grab_counts = run_once(False, interval)
        read_only, read_counts = run_once(True, interval)
        assert len(with_grab) == len(read_only) > 0
        assert all(np.array_equal(a, b) for a, b in zip(with_grab, read_only))
        assert grab_counts["grab"] > 0 and read_counts["grab"] == 0
