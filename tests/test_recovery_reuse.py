"""Exactness tests for reusing identical contrast-recovery calls within a run.

``original_contrast_recovered_boxes`` below is a verbatim copy (renamed only)
of ``_contrast_recovered_boxes`` as it is on ``perf/detector-calls``, before
recovery learnt to answer an identical question from an earlier answer. For
random frames and bboxes, repeated frames, different bboxes on identical
pixels, different detector states on entry and a detector that fails, the
function running under ``recovery_reuse.scope()`` must return what the
original returns, leave the detector's ``_manual_only`` as it found it, make
only calls the original would have made, and keep the counters truthful.
"""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import videowipe.detect as detect
from videowipe import perfstats, planning, recovery_reuse
from videowipe.detect import (
    TextBox,
    TextDetector,
    _bbox,
    _contrast_recovered_boxes,
    _contrast_restored_variants,
    _later_variants_can_confirm,
    _rects_agree,
    refine_temporal_presence,
)
from videowipe.plan import Segment


# ── Oracle: the function as it was, unchanged apart from its name ───────────

def original_contrast_recovered_boxes(
    detector: TextDetector,
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> list[TextBox]:
    """Second-chance detection for one candidate band on the current frame.

    Runs detection on contrast-restored variants of the band around *bbox*.
    A recovered box must (a) have its center inside the candidate bbox and
    (b) be agreed on by at least two independent restorations; single-variant
    hits stay negative. Returns boxes mapped back to frame coordinates.
    """
    perfstats.add("recovery_calls")
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]
    margin_y = max(8, (y2 - y1 + 1) // 4)
    margin_x = max(8, (x2 - x1 + 1) // 8)
    cx1, cy1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    cx2, cy2 = min(width - 1, x2 + margin_x), min(height - 1, y2 + margin_y)
    crop = frame[cy1:cy2 + 1, cx1:cx2 + 1]
    if crop.size == 0:
        return []

    def band_boxes(variant_crop: np.ndarray) -> list[tuple[TextBox, tuple[int, int, int, int]]]:
        found: list[tuple[TextBox, tuple[int, int, int, int]]] = []
        for box in perfstats.detect(detector, variant_crop, recovery=True):
            bx1, by1, bx2, by2 = _bbox(box.points, crop.shape[1], crop.shape[0])
            fx1, fy1, fx2, fy2 = bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1
            center_x, center_y = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                # TextBox carries crop-local polygon points; remap them so the
                # returned evidence lives in frame coordinates like every other
                # detector box.
                remapped = TextBox(
                    points=box.points.astype(np.float64) + np.array([cx1, cy1]),
                    confidence=box.confidence,
                    text=box.text,
                )
                found.append((remapped, (fx1, fy1, fx2, fy2)))
        return found

    # Flip state of the shared detector must not leak: the manual fallback is
    # a per-call choice here, not a mode switch for the rest of the video.
    manual_only = getattr(detector, "_manual_only", None)
    try:
        per_variant: list[tuple[str, list[tuple[TextBox, tuple[int, int, int, int]]]]] = []
        confirmed: dict[tuple[int, int, int, int], TextBox] = {}
        variants = list(_contrast_restored_variants(crop))
        families = [family for family, _ in variants]
        for position, (family, variant_crop) in enumerate(variants):
            # Stop only when the remaining variants cannot change the returned
            # boxes: no remaining one can pair with an earlier or a later
            # variant of another family that may hold entries. Only trailing
            # variants are ever dropped, so no later call sees a different
            # detector state, and the ``finally`` below restores the flag as
            # it always did. (A skipped call is not made, so a detector error
            # it would have raised is not seen; none of the six variants
            # differs from the others in shape or type.)
            if not _later_variants_can_confirm(per_variant, families[position:]):
                perfstats.add("recovery_variants_skipped", len(variants) - position)
                break
            entries = band_boxes(variant_crop)
            for earlier_family, earlier in per_variant:
                if family == earlier_family:
                    continue
                for entry in entries:
                    for other in earlier:
                        if _rects_agree(entry[1], other[1]):
                            confirmed.setdefault(entry[1], entry[0])
                            confirmed.setdefault(other[1], other[0])
            per_variant.append((family, entries))
            # Other glyphs may only agree with a later variant; one match
            # must not stop evidence collection for the rest of the band.
        if confirmed:
            perfstats.add("recovery_hits")
        return [confirmed[coords] for coords in sorted(confirmed)]
    finally:
        if manual_only is not None:
            detector._manual_only = manual_only  # noqa: SLF001


# ── Deterministic stand-in for DBNetDetector ────────────────────────────────

FRAME_H, FRAME_W = 120, 300
BBOX = (20, 40, 200, 90)
EDGE_A = (0, 0, FRAME_W - 1, FRAME_H - 1)       # crop clamps to the whole frame
EDGE_B = (1, 1, FRAME_W - 2, FRAME_H - 2)       # ... and so does this one


class ContentDetector:
    """Boxes from bright blobs; answers depend only on pixels and entry state.

    Like ``DBNetDetector`` it tries a strict "high-level" pass first and, when
    that finds nothing, a looser "manual" pass that latches ``_manual_only`` if
    it finds boxes, so the same pixels give different answers in the two
    states. ``fail_at`` raises on that (zero-based) call.
    """

    def __init__(self, fail_at=None):
        self._manual_only = False
        self.calls = 0
        self.fail_at = fail_at

    def reuse_token(self):
        return ("fake", self._manual_only)

    @staticmethod
    def _blobs(frame, threshold, pad):
        gray = frame.astype(np.float32).mean(axis=2)
        mask = (gray > threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 15), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        boxes = []
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            if area < 30 or h > 20 or w > 150:       # glyph lines, not washed-out regions
                continue
            boxes.append(TextBox(np.array(
                [[x - pad, y - pad], [x + w + pad, y - pad],
                 [x + w + pad, y + h + pad], [x - pad, y + h + pad]], np.float32),
                0.5 + area / (10.0 * frame.shape[0] * frame.shape[1])))
        return boxes

    def detect(self, frame):
        n, self.calls = self.calls, self.calls + 1
        if self.fail_at == n:
            raise RuntimeError(f"stub failure at call {n}")
        if not self._manual_only:
            boxes = self._blobs(frame, 200, 0)
            if boxes:
                return boxes
        boxes = self._blobs(frame, 140, 1)
        if boxes and not self._manual_only:
            self._manual_only = True
        return boxes


class UntokenedDetector(ContentDetector):
    reuse_token = None            # says nothing about purity: never reused


def make_frame(seed, strength=None, text=True):
    """Dark noisy frame with two bright text-like lines inside ``BBOX``."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(20, 70, (FRAME_H, FRAME_W, 3), np.uint8)
    if text:
        level = int(strength if strength is not None else rng.integers(130, 256))
        for row, (x1, x2) in zip((52, 68), ((40, 120), (60, 170))):
            for x in range(x1, x2, 9):
                frame[row:row + 9, x:x + 6] = level
    return frame


def same_boxes(a, b):
    assert len(a) == len(b)
    for left, right in zip(a, b):
        assert left.points.dtype == right.points.dtype
        assert np.array_equal(left.points, right.points)
        assert left.confidence == right.confidence and left.text == right.text


def call(function, detector, frame, bbox, state=False):
    """One recovery call from the given entry state: boxes or the raised error."""
    detector._manual_only = state
    try:
        boxes, error = function(detector, frame, bbox), None
    except Exception as exc:  # noqa: BLE001 - compared between old and new
        boxes, error = None, f"{type(exc).__name__}: {exc}"
    return boxes, error, detector._manual_only


def counted(function, *args, **kwargs):
    handle = perfstats.start()
    with perfstats.phase("refine"):
        out = function(*args, **kwargs)
    return out, perfstats.finish(handle)["refine"]


def positive_frame():
    for seed in range(60):
        frame = make_frame(seed, strength=235)
        boxes = original_contrast_recovered_boxes(ContentDetector(), frame, BBOX)
        if boxes:
            return frame
    raise AssertionError("no seed produced a recovery hit")


# ── Random sequences: same answers, same state, no extra calls ──────────────

def _sequence(seed):
    rng = random.Random(seed)
    base = [make_frame(s, None if s % 3 else 235, text=s % 4 != 0) for s in range(5)]
    frames = list(base)
    frames += [f.copy() for f in base[:3]]                        # equal pixels, new objects
    outside = base[1].copy()
    outside[:20, :] ^= 0x55                                       # differs outside the crop
    outside[:, 260:] ^= 0x55
    frames.append(outside)
    boxes = [BBOX, (30, 45, 150, 85), (0, 5, 80, 40), (220, 90, 299, 119),
             EDGE_A, EDGE_B, (10, 10, 12, 12)]
    return [(rng.choice(frames), rng.choice(boxes), rng.random() < 0.3) for _ in range(70)]


@pytest.mark.parametrize("seed", range(6))
def test_random_sequences_match_the_original(seed):
    old_detector, new_detector = ContentDetector(), ContentDetector()
    reused = positives = negatives = reused_positive = reused_negative = 0
    with recovery_reuse.scope() as memo:
        assert memo is not None
        for frame, bbox, state in _sequence(seed):
            (old, old_row) = counted(call, original_contrast_recovered_boxes,
                                     old_detector, frame, bbox, state)
            before = new_detector.calls
            (new, new_row) = counted(call, _contrast_recovered_boxes,
                                     new_detector, frame, bbox, state)
            assert new[1] is old[1] is None
            same_boxes(old[0], new[0])
            assert new[2] is old[2] is state                      # state left as found
            assert new_row["recovery_calls"] == old_row["recovery_calls"] == 1
            assert new_row["recovery_hits"] == old_row["recovery_hits"]
            if new_row["recovery_reused"]:
                reused += 1
                assert new_detector.calls == before               # no detector call at all
                assert new_row["recovery_detect_calls"] == 0
                assert new_row["detect_calls"] == 0
                reused_positive += bool(old[0])
                reused_negative += not old[0]
            else:
                assert new_row["recovery_detect_calls"] == old_row["recovery_detect_calls"]
            positives += bool(old[0])
            negatives += not old[0]
    # The reused calls are exactly the calls the original made on top of ours.
    assert old_detector.calls - new_detector.calls > 0
    assert positives and negatives and reused and reused_positive and reused_negative


def test_entry_state_is_part_of_the_key():
    frame = make_frame(1, strength=145)      # the strict pass misses it, the loose one finds it
    detector, oracle = ContentDetector(), ContentDetector()
    answers = {}
    with recovery_reuse.scope() as memo:
        for state in (False, True, False, True, True, False):
            old = call(original_contrast_recovered_boxes, oracle, frame, BBOX, state)
            new = call(_contrast_recovered_boxes, detector, frame, BBOX, state)
            same_boxes(old[0], new[0])
            assert new[2] is state
            answers[state] = old[0]
        assert len(memo) == 2                                      # one entry per state
    assert len(answers[False]) != len(answers[True])              # the state matters here
    assert detector.calls == oracle.calls // 3                    # 2 computed, 4 reused


def test_identical_pixels_under_a_different_bbox_are_not_shared():
    frame = make_frame(2, strength=235)
    # Both bboxes clamp to the whole frame, so the crops are the same pixels.
    assert crop_of(frame, EDGE_A).shape == crop_of(frame, EDGE_B).shape
    assert np.array_equal(crop_of(frame, EDGE_A), crop_of(frame, EDGE_B))
    detector, oracle = ContentDetector(), ContentDetector()
    with recovery_reuse.scope() as memo:
        for bbox in (EDGE_A, EDGE_B, EDGE_A, EDGE_B):
            old = call(original_contrast_recovered_boxes, oracle, frame, bbox)
            new = call(_contrast_recovered_boxes, detector, frame, bbox)
            same_boxes(old[0], new[0])
        assert len(memo) == 2
    per_call = oracle.calls // 4
    assert detector.calls == 2 * per_call      # the second pair came from the memo


def crop_of(frame, bbox):
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]
    margin_y, margin_x = max(8, (y2 - y1 + 1) // 4), max(8, (x2 - x1 + 1) // 8)
    return frame[max(0, y1 - margin_y):min(height - 1, y2 + margin_y) + 1,
                 max(0, x1 - margin_x):min(width - 1, x2 + margin_x) + 1]


def test_a_different_frame_size_still_matches_the_original():
    frame = make_frame(3, strength=235)
    bigger = np.pad(frame, ((0, 0), (0, 40), (0, 0)), constant_values=0)
    detector, oracle = ContentDetector(), ContentDetector()
    with recovery_reuse.scope():
        for image in (frame, bigger, frame, bigger):
            old = call(original_contrast_recovered_boxes, oracle, image, BBOX)
            new = call(_contrast_recovered_boxes, detector, image, BBOX)
            same_boxes(old[0], new[0])


# ── Failures, aliasing, counters ────────────────────────────────────────────

def test_an_exception_matches_the_original_and_is_never_remembered():
    frame = positive_frame()
    for fail_at in (0, 2, 4):
        old_detector, new_detector = ContentDetector(fail_at), ContentDetector(fail_at)
        with recovery_reuse.scope() as memo:
            old = call(original_contrast_recovered_boxes, old_detector, frame, BBOX, True)
            new = call(_contrast_recovered_boxes, new_detector, frame, BBOX, True)
            assert new[1] == old[1] and old[1] is not None
            assert new[2] is old[2] is True                          # restored on the way out
            assert len(memo) == 0
            # The failing call is spent; the same question now succeeds, is
            # computed (not replayed from the failure) and then remembered.
            old = call(original_contrast_recovered_boxes, old_detector, frame, BBOX, True)
            before = new_detector.calls
            new = call(_contrast_recovered_boxes, new_detector, frame, BBOX, True)
            assert new[1] is old[1] is None
            same_boxes(old[0], new[0])
            assert new_detector.calls > before and len(memo) == 1
            computed = new_detector.calls
            again = call(_contrast_recovered_boxes, new_detector, frame, BBOX, True)
            same_boxes(old[0], again[0])
            assert new_detector.calls == computed


def test_returned_boxes_are_private_copies():
    frame = positive_frame()
    oracle = original_contrast_recovered_boxes(ContentDetector(), frame, BBOX)
    assert oracle
    detector = ContentDetector()
    with recovery_reuse.scope():
        first = _contrast_recovered_boxes(detector, frame, BBOX)
        for box in first:
            box.points += 1000.0
            box.text = "changed"
        first.append("junk")
        second = _contrast_recovered_boxes(detector, frame, BBOX)
        same_boxes(oracle, second)
        second[0].points[:] = -1
        third = _contrast_recovered_boxes(detector, frame, BBOX)
        same_boxes(oracle, third)
        assert not {id(a) for a in second} & {id(b) for b in third}
        assert all(a.points is not b.points for a, b in zip(second, third))


def test_counters_stay_truthful():
    frame, negative = positive_frame(), make_frame(9, text=False)
    detector, oracle = ContentDetector(), ContentDetector()
    _, once = counted(original_contrast_recovered_boxes, oracle, frame, BBOX)

    def twice():
        for image in (frame, frame, negative, negative, negative):
            _contrast_recovered_boxes(detector, image, BBOX)

    with recovery_reuse.scope():
        _, row = counted(twice)
    _, per_negative = counted(original_contrast_recovered_boxes, ContentDetector(), negative, BBOX)
    assert row["recovery_calls"] == 5
    assert row["recovery_reused"] == 3
    assert row["recovery_hits"] == 2 * once["recovery_hits"] == 2        # reused hits still count
    assert row["recovery_detect_calls"] == row["detect_calls"] == (
        once["recovery_detect_calls"] + per_negative["recovery_detect_calls"])


# ── Scope: nothing outlives a run, nothing leaks between detectors ──────────

def test_no_reuse_outside_a_scope():
    assert recovery_reuse.current() is None
    frame, detector = positive_frame(), ContentDetector()
    _contrast_recovered_boxes(detector, frame, BBOX)
    after_one = detector.calls
    _contrast_recovered_boxes(detector, frame, BBOX)
    assert detector.calls == 2 * after_one


def test_each_scope_has_its_own_memo():
    frame, detector = positive_frame(), ContentDetector()
    with recovery_reuse.scope() as first:
        _contrast_recovered_boxes(detector, frame, BBOX)
        assert recovery_reuse.current() is first and len(first) == 1
    assert recovery_reuse.current() is None
    calls = detector.calls
    with recovery_reuse.scope() as second:
        assert second is not first and len(second) == 0
        _contrast_recovered_boxes(detector, frame, BBOX)
    assert detector.calls == 2 * calls


def test_a_scope_is_dropped_when_the_run_raises():
    with pytest.raises(KeyError):
        with recovery_reuse.scope():
            raise KeyError("run failed")
    assert recovery_reuse.current() is None


def test_a_detector_that_does_not_vouch_for_itself_is_never_reused():
    frame, detector = positive_frame(), UntokenedDetector()
    with recovery_reuse.scope() as memo:
        _contrast_recovered_boxes(detector, frame, BBOX)
        calls = detector.calls
        _contrast_recovered_boxes(detector, frame, BBOX)
        assert detector.calls == 2 * calls and len(memo) == 0


def test_a_memo_belongs_to_one_detector():
    frame = positive_frame()
    first, second = ContentDetector(), ContentDetector()
    with recovery_reuse.scope() as memo:
        _contrast_recovered_boxes(first, frame, BBOX)
        _contrast_recovered_boxes(second, frame, BBOX)
        assert second.calls == first.calls          # computed, not borrowed
        assert len(memo) == 1
        before = second.calls
        _contrast_recovered_boxes(second, frame, BBOX)
        assert second.calls == before


def test_the_memo_is_bounded_and_drops_the_oldest():
    frames = [make_frame(seed, strength=235) for seed in (4, 5, 6)]
    detector = ContentDetector()
    with recovery_reuse.scope(max_entries=2) as memo:
        for frame in frames:
            _contrast_recovered_boxes(detector, frame, BBOX)
        assert len(memo) == 2
        before = detector.calls
        _contrast_recovered_boxes(detector, frames[2], BBOX)        # still there
        assert detector.calls == before
        _contrast_recovered_boxes(detector, frames[0], BBOX)        # evicted: computed
        assert detector.calls > before and len(memo) == 2


@pytest.mark.parametrize("value, entries", [
    (None, recovery_reuse.DEFAULT_ENTRIES), ("", recovery_reuse.DEFAULT_ENTRIES),
    ("on", recovery_reuse.DEFAULT_ENTRIES), ("junk", recovery_reuse.DEFAULT_ENTRIES),
    ("12", 12), ("off", 0), ("OFF", 0), ("0", 0), ("false", 0), ("no", 0), ("-3", 0),
])
def test_environment_switch(monkeypatch, value, entries):
    if value is None:
        monkeypatch.delenv(recovery_reuse.ENV, raising=False)
    else:
        monkeypatch.setenv(recovery_reuse.ENV, value)
    assert recovery_reuse.configured_entries() == entries
    frame, detector = positive_frame(), ContentDetector()
    with recovery_reuse.scope() as memo:
        assert (memo is None) == (entries == 0)
        _contrast_recovered_boxes(detector, frame, BBOX)
        calls = detector.calls
        _contrast_recovered_boxes(detector, frame, BBOX)
        assert detector.calls == (2 * calls if entries == 0 else calls)


def test_the_real_detector_vouches_for_exactly_its_state():
    token = detect.DBNetDetector.reuse_token
    stub = SimpleNamespace(_manual_only=False)
    assert token(stub) == ("dbnet", False)
    stub._manual_only = True
    assert token(stub) == ("dbnet", True)


# ── Through refinement and the planning scopes ──────────────────────────────

N_FRAMES = 12


def _write_video(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 8, (FRAME_W, FRAME_H))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()
    return str(path)


def _refine(video, function, monkeypatch, scoped):
    monkeypatch.setattr(detect, "_contrast_recovered_boxes", function)
    detector = ContentDetector()
    candidate = SimpleNamespace(id="A", bbox=BBOX, detector_backed=True,
                                presence_frames=[0], temporal_sample_indices=[])
    result = SimpleNamespace(detector=detector, candidates=[candidate], sample_indices=[0],
                             frame_shape=(FRAME_H, FRAME_W), sampled_frame_boxes={})
    handle = perfstats.start()
    with perfstats.phase("refine"):
        if scoped:
            with recovery_reuse.scope():
                warnings = refine_temporal_presence(video, result, {"A": [Segment(0, N_FRAMES)]}, N_FRAMES)
        else:
            warnings = refine_temporal_presence(video, result, {"A": [Segment(0, N_FRAMES)]}, N_FRAMES)
    return candidate, result, warnings, detector, perfstats.finish(handle)["refine"]


def test_refinement_over_repeating_frames_is_identical_with_fewer_calls(tmp_path, monkeypatch):
    # Static stretches (as skip-coded frames decode): the same band, frame after
    # frame, around a few different pictures, one of them a recovery hit.
    pictures = [make_frame(7, text=False), make_frame(8, strength=235), make_frame(9, strength=150)]
    order = [0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 1]
    video = _write_video(tmp_path / "static.avi", [pictures[i] for i in order])
    old = _refine(video, original_contrast_recovered_boxes, monkeypatch, scoped=False)
    new = _refine(video, _contrast_recovered_boxes, monkeypatch, scoped=True)
    (old_candidate, old_result, old_warnings, old_detector, old_row) = old
    (new_candidate, new_result, new_warnings, new_detector, new_row) = new
    assert new_candidate.presence_frames == old_candidate.presence_frames
    assert new_candidate.temporal_sample_indices == old_candidate.temporal_sample_indices
    assert new_warnings == old_warnings
    assert new_result.failed_frame_indices == old_result.failed_frame_indices
    assert list(new_result.sampled_frame_boxes) == list(old_result.sampled_frame_boxes)
    for index in old_result.sampled_frame_boxes:
        same_boxes(old_result.sampled_frame_boxes[index], new_result.sampled_frame_boxes[index])
    assert new_detector._manual_only is old_detector._manual_only
    assert new_row["recovery_reused"] > 0 and old_row["recovery_reused"] == 0
    assert new_row["recovery_calls"] == old_row["recovery_calls"]
    assert new_row["recovery_hits"] == old_row["recovery_hits"]
    # Every detector call that is gone is a recovery call that was reused.
    saved = old_row["recovery_detect_calls"] - new_row["recovery_detect_calls"]
    assert saved > 0
    assert new_row["detect_calls"] == old_row["detect_calls"] - saved
    assert new_detector.calls == old_detector.calls - saved


def test_planning_runs_refinement_and_probes_inside_one_scope(monkeypatch):
    seen = []
    plan = SimpleNamespace(warnings=[], remove_tracks=[])
    monkeypatch.setattr(planning, "build_wipe_plan", lambda *a, **k: plan)
    monkeypatch.setattr(planning, "validate_plan", lambda *a, **k: None)

    def refine(*args, **kwargs):
        seen.append(("refine", recovery_reuse.current()))
        return []

    def probes(*args, **kwargs):
        seen.append(("probes", recovery_reuse.current()))

    monkeypatch.setattr(planning, "refine_temporal_presence", refine)
    monkeypatch.setattr(planning, "_add_spatial_segments", probes)
    result = SimpleNamespace(candidates=[], sample_indices=[], frame_shape=(FRAME_H, FRAME_W))
    source = SimpleNamespace(frame_count=N_FRAMES)
    planning._finalize_result("video.mp4", result, source, refine=True)
    assert [name for name, _ in seen] == ["refine", "probes"]
    assert seen[0][1] is not None and seen[0][1] is seen[1][1]
    assert recovery_reuse.current() is None
    seen.clear()
    planning._finalize_result("video.mp4", result, source, refine=True)
    assert seen[0][1] is not None and recovery_reuse.current() is None


def test_review_refinement_runs_inside_a_scope_too(tmp_path, monkeypatch):
    seen = []
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"sample_indices": [], "candidates": [], "boxes": {}}))
    source = SimpleNamespace(height=FRAME_H, width=FRAME_W, frame_count=N_FRAMES)
    machine = SimpleNamespace(remove_tracks=[], tracks=[], source=source)
    reviewed = SimpleNamespace(remove_tracks=[SimpleNamespace(id="x", segments=[1])], warnings=[])
    refined = SimpleNamespace(tracks=[SimpleNamespace(id="x", segments=[1])])
    monkeypatch.setattr(planning, "build_wipe_plan", lambda *a, **k: refined)
    monkeypatch.setattr(planning, "validate_plan", lambda *a, **k: None)
    monkeypatch.setattr(planning, "refine_temporal_presence",
                        lambda *a, **k: seen.append(recovery_reuse.current()) or [])
    monkeypatch.setattr(planning, "_add_spatial_segments",
                        lambda *a, **k: seen.append(recovery_reuse.current()))
    planning.refine_review("video.mp4", machine, reviewed, str(evidence), ContentDetector())
    assert len(seen) == 2 and seen[0] is not None and seen[0] is seen[1]
    assert recovery_reuse.current() is None
