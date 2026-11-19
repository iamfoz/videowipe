"""Exactness tests for pruning contrast recovery (P2-3a).

``original_contrast_recovered_boxes`` below is a verbatim copy (renamed only)
of the function as it was before recovery learnt to stop once its outcome is
fixed. Recovery returns boxes only where two variants of *different* families
agree, so for every scripted pattern of "which variant sees which boxes" the
pruned function must return exactly what the original returns, leave the
detector's ``_manual_only`` flag as it found it, and only ever skip the calls
whose result could not have mattered.
"""
from __future__ import annotations

import itertools
import random
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import videowipe.detect as detect
from videowipe import perfstats
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


# ── Oracle: the pre-change function, unchanged apart from its name ──────────

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
        for family, variant_crop in _contrast_restored_variants(crop):
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


# ── Scripted detector ───────────────────────────────────────────────────────

FRAME_H, FRAME_W = 120, 300
BBOX = (20, 40, 200, 90)  # crop = rows 28..102, cols 0..222: local y + 28 = frame y


def _poly(x1, y1, x2, y2, order=(0, 1, 2, 3)):
    corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return np.array([corners[i] for i in order], np.float64)


# Crop-local polygons. A and A2 overlap by more than half (they agree) but have
# different coordinates; AR has A's extent with its corners in another order
# (same key, different object); B is a second glyph; C's centre is outside the
# candidate, so recovery ignores it.
BOXES = {
    "A": _poly(40, 20, 80, 45),
    "A2": _poly(42, 21, 82, 46),
    "AR": _poly(40, 20, 80, 45, order=(2, 3, 0, 1)),
    "B": _poly(120, 20, 160, 45),
    "C": _poly(100, 0, 140, 5),
}
IGNORED = {"C"}


class ScriptedDetector:
    """Answers the n-th call from ``pattern[n]`` and logs each call.

    ``flip_after`` lists call numbers after which the detector latches
    ``_manual_only`` the way ``DBNetDetector`` does when its manual path
    rescues a call. ``fail_at`` raises on that call.
    """

    def __init__(self, pattern, flip_after=(), fail_at=None):
        self.pattern = pattern
        self.flip_after = set(flip_after)
        self.fail_at = fail_at
        self._manual_only = False
        self.log: list[tuple[int, bool]] = []

    def detect(self, frame):
        n = len(self.log)
        self.log.append((n, self._manual_only))
        if self.fail_at == n:
            raise RuntimeError(f"stub failure at call {n}")
        assert n < len(self.pattern), f"call {n} beyond a {len(self.pattern)}-variant script"
        boxes = [TextBox(BOXES[name].copy(), 0.9, name) for name in self.pattern[n]]
        if n in self.flip_after:
            self._manual_only = True
        return boxes


def _frame():
    return np.random.default_rng(3).integers(0, 255, (FRAME_H, FRAME_W, 3), np.uint8)


def _same_boxes(a, b):
    assert len(a) == len(b)
    for left, right in zip(a, b):
        assert np.array_equal(left.points, right.points)
        assert left.confidence == right.confidence and left.text == right.text


def _visible(pattern_entry):
    return [name for name in pattern_entry if name not in IGNORED]


def _run(function, pattern, **detector_kwargs):
    detector = ScriptedDetector(pattern, **detector_kwargs)
    handle = perfstats.start()
    with perfstats.phase("refine"):
        try:
            boxes = function(detector, _frame(), BBOX)
            error = None
        except Exception as exc:  # noqa: BLE001 - compared between both runs
            boxes, error = None, f"{type(exc).__name__}: {exc}"
    return detector, boxes, error, perfstats.finish(handle)["refine"]


def _expected_skipped(pattern):
    """Independent statement of the rule for the standard six variants:
    the sixth (darkened, strong) is dropped when the four other-family
    variants (contrast, highpass, contrast, highpass) saw nothing."""
    return int(all(not _visible(pattern[i]) for i in (0, 1, 3, 4)))


OPTIONS = [(), ("A",), ("A2",), ("AR",), ("B",), ("A", "B"), ("C",)]


def _patterns():
    # Every 3-way choice per variant over three different shapes of evidence,
    # then a seeded random sample of the full 7-option space.
    for combo in itertools.product([(), ("A",), ("A2", "B")], repeat=6):
        yield combo
    # The only patterns where the rule can fire: contrast/highpass variants
    # empty, any evidence in the two darkened ones.
    for second, sixth in itertools.product(OPTIONS, repeat=2):
        yield ((), (), second, (), (), sixth)
    rng = random.Random(20260101)
    for _ in range(200):
        yield tuple(rng.choice(OPTIONS) for _ in range(6))


# ── Same boxes, same state; fewer calls only when the outcome is fixed ──────

def test_pruned_recovery_matches_the_original_on_every_scripted_pattern():
    skipped_somewhere = kept_somewhere = confirmed_somewhere = 0
    for pattern in _patterns():
        old_detector, old_boxes, old_error, old_row = _run(original_contrast_recovered_boxes, pattern)
        new_detector, new_boxes, new_error, new_row = _run(_contrast_recovered_boxes, pattern)
        assert old_error is None and new_error is None
        _same_boxes(old_boxes, new_boxes)
        assert old_row["recovery_hits"] == new_row["recovery_hits"]
        assert old_row["recovery_calls"] == new_row["recovery_calls"] == 1
        # Calls the pruned version made are a prefix of the original's, with
        # the same detector flag at each.
        assert new_detector.log == old_detector.log[:len(new_detector.log)]
        assert len(old_detector.log) == 6
        expected = _expected_skipped(pattern)
        assert len(new_detector.log) == 6 - expected, pattern
        assert new_row["recovery_variants_skipped"] == expected
        assert new_row["recovery_detect_calls"] == 6 - expected
        assert old_detector._manual_only is new_detector._manual_only is False
        skipped_somewhere += expected
        kept_somewhere += 1 - expected
        confirmed_somewhere += bool(new_boxes)
    assert skipped_somewhere > 40 and kept_somewhere > 400 and confirmed_somewhere > 100


@pytest.mark.parametrize("flip_after", [(0,), (2,), (4,), (5,), (1, 3)])
def test_detector_state_is_restored_and_identical_when_the_flag_latches(flip_after):
    for pattern in [((),) * 6, (("A",), (), (), (), (), ("A2",)), ((), (), ("B",), (), (), ("B",)),
                    ((), (), (), (), ("A",), ("A",)), ((), (), ("A",), (), (), ("A",))]:
        old = _run(original_contrast_recovered_boxes, pattern, flip_after=flip_after)
        new = _run(_contrast_recovered_boxes, pattern, flip_after=flip_after)
        _same_boxes(old[1], new[1])
        assert new[0].log == old[0].log[:len(new[0].log)]
        # Always restored to the value before recovery, not left latched.
        assert old[0]._manual_only is new[0]._manual_only is False


# ── Cases that decide the rule ──────────────────────────────────────────────

def test_sixth_variant_still_runs_when_it_is_the_only_confirmation():
    # Only the strong "darkened" variant agrees with the strong "highpass"
    # one (variants 4 and 5): stopping after five would lose the box.
    pattern = ((), (), (), (), ("A",), ("A2",))
    detector, boxes, error, row = _run(_contrast_recovered_boxes, pattern)
    assert error is None and len(detector.log) == 6 and len(boxes) == 2
    assert row["recovery_variants_skipped"] == 0 and row["recovery_hits"] == 1
    _same_boxes(boxes, _run(original_contrast_recovered_boxes, pattern)[1])


def test_a_match_does_not_stop_collection_for_other_glyphs():
    # Variants 0 and 1 confirm A; B only agrees between variants 4 and 5.
    pattern = (("A",), ("A",), (), (), ("B",), ("B",))
    detector, boxes, _, _ = _run(_contrast_recovered_boxes, pattern)
    assert len(detector.log) == 6
    assert sorted(b.text for b in boxes) == ["A", "B"]


def test_same_family_cannot_confirm_so_the_sixth_is_skipped():
    # Variant 2 (darkened) saw something, but only variant 5 is also darkened.
    pattern = ((), (), ("A",), (), (), ("A",))
    detector, boxes, _, row = _run(_contrast_recovered_boxes, pattern)
    assert boxes == [] and len(detector.log) == 5
    assert row["recovery_variants_skipped"] == 1 and row["recovery_hits"] == 0


def test_nothing_found_anywhere_skips_exactly_the_last_call():
    detector, boxes, _, row = _run(_contrast_recovered_boxes, ((),) * 6)
    assert boxes == [] and len(detector.log) == 5
    assert (row["recovery_detect_calls"], row["recovery_variants_skipped"]) == (5, 1)


def test_ignored_boxes_outside_the_candidate_do_not_count_as_evidence():
    pattern = (("C",), ("C",), ("C",), ("C",), ("C",), ())
    assert len(_run(_contrast_recovered_boxes, pattern)[0].log) == 5


def test_the_rule_in_isolation():
    can = _later_variants_can_confirm
    names = ["contrast", "highpass", "darkened", "contrast", "highpass", "darkened"]
    nothing = [(f, []) for f in names]
    assert can([], names)                                   # nothing run yet
    assert can(nothing[:4], names[4:])                       # 4 and 5 may agree
    assert not can(nothing[:5], names[5:])                   # lone last variant
    assert can([*nothing[:4], ("highpass", [1])], names[5:])  # other family has entries
    assert not can([("contrast", []), ("highpass", []), ("darkened", [1]),
                    ("contrast", []), ("highpass", [])], names[5:])
    assert not can([], ["highpass", "highpass"])             # one family only
    assert not can([("highpass", [1])], ["highpass"])
    assert can([("highpass", [1])], ["darkened"])
    assert not can([], [])


@pytest.fixture
def variants_of(monkeypatch):
    def install(families):
        def fake(crop):
            return [(family, crop) for family in families]
        monkeypatch.setattr(detect, "_contrast_restored_variants", fake)
        monkeypatch.setitem(globals(), "_contrast_restored_variants", fake)
    return install


@pytest.mark.parametrize("families", [
    ["contrast", "highpass", "darkened"],
    ["highpass", "highpass"],
    ["contrast", "contrast", "highpass"],
    ["darkened"],
])
def test_other_variant_lists_follow_the_same_rule(variants_of, families):
    variants_of(families)
    options = [(), ("A",), ("A2", "B")]
    for pattern in itertools.product(options, repeat=len(families)):
        old = _run(original_contrast_recovered_boxes, pattern)
        new = _run(_contrast_recovered_boxes, pattern)
        assert old[2] is None and new[2] is None
        _same_boxes(old[1], new[1])
        assert new[0].log == old[0].log[:len(new[0].log)]
        assert len(old[0].log) == len(families)


# ── Failure handling ────────────────────────────────────────────────────────

@pytest.mark.parametrize("fail_at", [0, 1, 2, 3, 4])
def test_a_failing_call_that_is_made_fails_identically(fail_at):
    pattern = ((),) * 6
    old = _run(original_contrast_recovered_boxes, pattern, fail_at=fail_at)
    new = _run(_contrast_recovered_boxes, pattern, fail_at=fail_at)
    assert old[2] == new[2] == f"RuntimeError: stub failure at call {fail_at}"
    assert old[0].log == new[0].log and new[0]._manual_only is False


def test_the_only_unobservable_difference_is_a_failure_on_a_call_that_is_never_made():
    """A skipped call is not made, so an error it would have raised is not
    seen. The skipped sixth variant has the same shape and type as the five
    before it, so a detector that fails only on it is not expected."""
    pattern = ((),) * 6
    old = _run(original_contrast_recovered_boxes, pattern, fail_at=5)
    new = _run(_contrast_recovered_boxes, pattern, fail_at=5)
    assert old[2] == "RuntimeError: stub failure at call 5" and new[2] is None
    assert new[1] == []


# ── Through the refine loop ─────────────────────────────────────────────────

N_FRAMES = 12


def _write_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 8, (FRAME_W, FRAME_H))
    assert writer.isOpened()
    for index in range(N_FRAMES):
        frame = np.full((FRAME_H, FRAME_W, 3), 40, np.uint8)
        frame[0, 0] = index
        writer.write(frame)
    writer.release()
    return str(path)


class FrameAwareDetector:
    """Full frames: a box on every third frame. Recovery calls on a frame
    follow ``plans[index]`` (default: nothing)."""

    def __init__(self, plans):
        self.plans = plans
        self._manual_only = False
        self.index = None
        self.n = 0
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if frame.shape[:2] == (FRAME_H, FRAME_W):
            self.index, self.n = int(frame[0, 0, 0]), 0
            if self.index % 3 == 0:
                return [TextBox(_poly(30, 50, 190, 70), 0.9)]
            return []
        pattern = self.plans.get(self.index, ((),) * 6)
        n, self.n = self.n, self.n + 1
        boxes = [TextBox(BOXES[name].copy(), 0.9) for name in pattern[n]]
        if self.index == 7 and n == 1:
            self._manual_only = True   # latches inside recovery on one frame
        return boxes


def _refine(video, detector, function, monkeypatch):
    monkeypatch.setattr(detect, "_contrast_recovered_boxes", function)
    candidate = SimpleNamespace(
        id="A", bbox=BBOX, detector_backed=True, presence_frames=[0], temporal_sample_indices=[],
    )
    result = SimpleNamespace(
        detector=detector, candidates=[candidate], sample_indices=[0],
        frame_shape=(FRAME_H, FRAME_W), sampled_frame_boxes={},
    )
    handle = perfstats.start()
    with perfstats.phase("refine"):
        warnings = refine_temporal_presence(video, result, {"A": [Segment(0, N_FRAMES)]}, N_FRAMES)
    return candidate, result, warnings, perfstats.finish(handle)["refine"]


def test_refine_results_are_identical_with_pruned_recovery(tmp_path, monkeypatch):
    video = _write_video(tmp_path / "frames.avi")
    plans = {
        1: (("A",), (), (), (), (), ()),                 # one variant only: negative
        2: (("A",), ("A2",), (), (), (), ()),            # two families agree: hit
        4: ((), (), (), (), ("B",), ("B",)),             # needs the sixth call: hit
        5: ((), (), ("A",), (), (), ("A",)),             # same family twice: negative
        7: (("B",), (), (), (), ("B",), ()),             # flag latches mid-recovery
        10: (("A", "B"), ("B",), (), ("A",), (), ()),    # B confirmed twice, A twice
    }
    old_detector, new_detector = FrameAwareDetector(plans), FrameAwareDetector(plans)
    old = _refine(video, old_detector, original_contrast_recovered_boxes, monkeypatch)
    new = _refine(video, new_detector, _contrast_recovered_boxes, monkeypatch)

    old_candidate, old_result, old_warnings, old_row = old
    new_candidate, new_result, new_warnings, new_row = new
    assert new_candidate.presence_frames == old_candidate.presence_frames
    assert new_candidate.temporal_sample_indices == old_candidate.temporal_sample_indices
    assert new_warnings == old_warnings == []
    assert new_result.failed_frame_indices == old_result.failed_frame_indices
    assert list(new_result.sampled_frame_boxes) == list(old_result.sampled_frame_boxes)
    for index in old_result.sampled_frame_boxes:
        _same_boxes(old_result.sampled_frame_boxes[index], new_result.sampled_frame_boxes[index])
    assert old_detector._manual_only is new_detector._manual_only is False
    # Hits at 2, 4 and 10 (plus the detected 3, 6, 9, 0 frames).
    assert {2, 4, 10} <= set(new_candidate.presence_frames)
    assert not {1, 5} & set(new_candidate.presence_frames)

    # Only the sixth call of recovery attempts whose first five saw nothing
    # in the other families was dropped; every other counter is untouched.
    skipped = new_row["recovery_variants_skipped"]
    assert skipped > 0
    assert old_row["detect_calls"] - new_row["detect_calls"] == skipped
    for key in ("recovery_calls", "recovery_frames", "recovery_hits", "frames_present",
                "frames_converted", "frames_grabbed"):
        assert old_row[key] == new_row[key], key
    assert old_detector.calls - new_detector.calls == skipped


def test_a_failing_recovery_is_reported_the_same_way(tmp_path, monkeypatch):
    video = _write_video(tmp_path / "frames.avi")

    class Failing(FrameAwareDetector):
        def detect(self, frame):
            if frame.shape[:2] != (FRAME_H, FRAME_W) and self.index == 4 and self.n == 2:
                raise RuntimeError("variant failed")
            return super().detect(frame)

    old = _refine(video, Failing({}), original_contrast_recovered_boxes, monkeypatch)
    new = _refine(video, Failing({}), _contrast_recovered_boxes, monkeypatch)
    assert new[2] == old[2] and len(new[2]) == 1 and "frame 4" in new[2][0]
    assert new[1].failed_frame_indices == old[1].failed_frame_indices == {4}
    assert new[0].presence_frames == old[0].presence_frames
    assert new[1].sampled_frame_boxes[4] == old[1].sampled_frame_boxes[4] == []


# ── The real detector's flag ────────────────────────────────────────────────

def _dbnet_with_a_rescuing_manual_path():
    from videowipe.detect import DBNetDetector

    detector = object.__new__(DBNetDetector)
    detector._input_w = detector._input_h = 64
    detector._scale, detector._mean = 1 / 255, (0, 0, 0)
    detector._bin_thresh, detector._box_thresh, detector._unclip_ratio = .3, .5, 1.5
    detector._adaptive = False
    detector._manual_only = False
    prob = np.zeros((1, 1, 64, 64), np.float32)
    prob[0, 0, 2:15, 5:55] = 0.9
    detector._net = SimpleNamespace(setInput=lambda blob: None, forward=lambda: prob.copy())
    detector._hl_model = SimpleNamespace(
        setInputParams=lambda **kwargs: None, detect=lambda frame: ([], []),
    )
    return detector


def test_real_detector_flag_is_restored_after_recovery():
    frame = _frame()
    results = []
    for function in (original_contrast_recovered_boxes, _contrast_recovered_boxes):
        detector = _dbnet_with_a_rescuing_manual_path()
        boxes = function(detector, frame, BBOX)
        results.append((boxes, detector._manual_only))
    (old_boxes, old_flag), (new_boxes, new_flag) = results
    _same_boxes(old_boxes, new_boxes)
    assert old_flag is new_flag is False
    assert len(new_boxes) >= 1  # every variant sees the same block: all agree
