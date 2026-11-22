"""Sparse tracks: recurring boxes become a candidate, one-off boxes never do."""
import math
import types

import cv2
import numpy as np
import pytest

from videowipe import sparse_tracks
from videowipe.detect import TextBox, detect_clean_candidates, refine_temporal_presence
from videowipe.plan import Segment, predicted_mask_at
from videowipe.planning import finalize, prepare
from videowipe.sparse_tracks import (
    MIN_EPISODES,
    SUPPORT_FLOOR,
    enabled,
    find_tracks,
    min_support,
    sparse_candidates,
)

SHAPE = (360, 640)  # (height, width)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(sparse_tracks.ENV_SWITCH, raising=False)


def line_box(width, y=40, height=17, centre=320):
    x1 = centre - width // 2
    return (x1, y, x1 + width - 1, y + height - 1)


def samples_with(rows, n):
    """``n`` samples; ``rows`` maps a sample ordinal to the boxes on it."""
    return [(ordinal * 5, list(rows.get(ordinal, []))) for ordinal in range(n)]


def appearances(starts, length, widths=(200, 120, 260, 160)):
    """Lines of ``length`` consecutive samples starting at each of ``starts``."""
    rows = {}
    for k, start in enumerate(starts):
        for ordinal in range(start, start + length):
            rows[ordinal] = [line_box(widths[k % len(widths)])]
    return rows


# ── switch and threshold ─────────────────────────────────────────────────────

@pytest.mark.parametrize("value, expected", [
    (None, True), ("", True),
    ("nonsense", True),
    ("off", False), ("OFF", False), ("0", False), (" false ", False), ("no", False),
    ("on", True), ("ON", True), (" 1 ", True), ("true", True), ("yes", True),
])
def test_switch(value, expected):
    environ = {} if value is None else {sparse_tracks.ENV_SWITCH: value}
    assert enabled(environ) is expected


def test_the_switch_beats_the_default(monkeypatch):
    for default in (True, False):
        monkeypatch.setattr(sparse_tracks, "DEFAULT_ON", default)
        assert enabled({}) is default
        assert enabled({sparse_tracks.ENV_SWITCH: "on"}) is True
        assert enabled({sparse_tracks.ENV_SWITCH: "off"}) is False


def test_the_default_is_on():
    # The measurements in docs/sparse-tracks.md are what justify this.
    assert sparse_tracks.DEFAULT_ON is True
    assert enabled({}) is True


def test_support_grows_with_the_number_of_samples():
    assert min_support(0) == SUPPORT_FLOOR
    assert min_support(50) == SUPPORT_FLOOR
    previous = 0
    for n in (50, 120, 300, 600, 1200, 4300):
        assert min_support(n) >= previous
        previous = min_support(n)
    assert min_support(600) == math.ceil(math.sqrt(600))
    assert min_support(1200) > min_support(600) > min_support(120)


# ── the rule on boxes ────────────────────────────────────────────────────────

def test_a_recurring_row_is_one_track():
    n = 300
    need = min_support(n)
    starts = [20 + 30 * k for k in range(8)]  # eight separate appearances
    length = math.ceil(need / len(starts)) + 1
    tracks = find_tracks(samples_with(appearances(starts, length), n), SHAPE)
    assert len(tracks) == 1
    track = tracks[0]
    assert track.episodes == len(starts)
    assert len(track.samples) == len(starts) * length >= need
    x1, y1, x2, y2 = track.bbox
    # Lines of different lengths in one row: the region spans the longest.
    assert x1 <= 320 - 130 and x2 >= 320 + 130 and y1 <= 40 and y2 >= 56
    assert track.supports(line_box(150)) and track.supports(line_box(60))
    assert not track.supports(line_box(150, y=200))
    assert not track.supports(line_box(150, height=40))
    # Inside the region but off the row: the centre sits 8 px low.
    assert not track.supports(line_box(150, y=48))


def test_support_just_below_the_threshold_is_not_enough():
    n = 600
    need = min_support(n)
    starts = [10 + 70 * k for k in range(8)]
    just_short = [(need - 1) // len(starts) + (1 if i < (need - 1) % len(starts) else 0)
                  for i in range(len(starts))]
    rows = {}
    for start, length in zip(starts, just_short):
        for ordinal in range(start, start + length):
            rows[ordinal] = [line_box(200)]
    assert sum(just_short) == need - 1
    assert find_tracks(samples_with(rows, n), SHAPE) == []
    rows[580] = [line_box(200)]
    rows[581] = [line_box(200)]
    assert len(find_tracks(samples_with(rows, n), SHAPE)) == 1


def test_few_separate_stretches_are_not_a_track():
    # Plenty of samples, but one long scene: the rule wants separate appearances.
    n = 300
    rows = appearances([100], 60)
    assert MIN_EPISODES > 1
    assert find_tracks(samples_with(rows, n), SHAPE) == []
    two = appearances([50, 200], 30)
    assert find_tracks(samples_with(two, n), SHAPE) == []
    three = appearances([30, 130, 230], 12)
    assert len(find_tracks(samples_with(three, n), SHAPE)) == 1


def test_one_off_boxes_never_qualify():
    n = 600
    rng = np.random.default_rng(3)
    rows = {}
    for ordinal in range(n):
        boxes = []
        for _ in range(4):  # four wide boxes a sample, anywhere
            width = int(rng.integers(60, 400))
            boxes.append(line_box(
                width, y=int(rng.integers(0, 330)),
                height=int(rng.integers(10, 30)), centre=int(rng.integers(80, 560)),
            ))
        rows[ordinal] = boxes
    assert find_tracks(samples_with(rows, n), SHAPE) == []


def test_noise_that_grows_with_the_samples_does_not_get_in():
    # The same noise process over a short and a long video: a recurring spot
    # collects boxes in proportion to the number of samples, the threshold
    # grows with it.
    rng = np.random.default_rng(11)
    for n in (120, 1200, 4000):
        rows = {}
        for ordinal in range(n):
            rows[ordinal] = [
                line_box(int(rng.integers(40, 120)), y=int(rng.integers(0, 330)),
                         height=int(rng.integers(12, 24)), centre=int(rng.integers(60, 580)))
                for _ in range(3)
            ]
        assert find_tracks(samples_with(rows, n), SHAPE) == [], n


def test_narrow_boxes_are_not_lines_of_text():
    n = 300
    rows = {ordinal: [line_box(20)] for ordinal in range(0, n, 2)}
    assert find_tracks(samples_with(rows, n), SHAPE) == []


def test_boxes_of_existing_candidates_are_skipped():
    n = 300
    starts = [20 + 30 * k for k in range(8)]
    rows = appearances(starts, 4)
    data = samples_with(rows, n)
    assert len(find_tracks(data, SHAPE)) == 1
    assert find_tracks(data, SHAPE, skip=lambda box: box[1] <= 45 <= box[3]) == []


def test_boxes_that_do_not_overlap_horizontally_are_not_one_track():
    # The same row, but text on the left in some appearances and on the right
    # in the others: neither place has enough of its own.
    n = 300
    need = min_support(n)
    left = appearances([20, 50, 80, 110], 3, widths=(100,))
    right = appearances([140, 170, 200, 230], 3, widths=(100,))
    rows = {}
    for ordinal in left:
        rows[ordinal] = [line_box(100, centre=90)]
    for ordinal in right:
        rows[ordinal] = [line_box(100, centre=550)]
    assert len(left) + len(right) >= need > len(left)
    assert find_tracks(samples_with(rows, n), SHAPE) == []


def test_two_rows_are_two_tracks():
    n = 300
    starts = [20 + 30 * k for k in range(8)]
    top = appearances(starts, 4)
    low = {ordinal + 17: [line_box(180, y=300)] for ordinal in top}
    rows = {ordinal: top.get(ordinal, []) + low.get(ordinal, []) for ordinal in set(top) | set(low)}
    tracks = find_tracks(samples_with(rows, n), SHAPE)
    assert [round(t.y_centre) for t in tracks] == [48, 308]


# ── from boxes to candidates ─────────────────────────────────────────────────

def text_box(box):
    x1, y1, x2, y2 = box
    return TextBox(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32), 0.9)


def as_text_boxes(samples):
    return [(sample, [text_box(box) for box in boxes]) for sample, boxes in samples]


def existing(bbox, kind):
    return types.SimpleNamespace(bbox=bbox, type=kind)


def test_candidate_carries_only_its_own_evidence():
    n = 300
    starts = [20 + 30 * k for k in range(8)]
    data = samples_with(appearances(starts, 4), n)
    found = sparse_candidates(as_text_boxes(data), SHAPE, [], start_index=3)
    assert [c.id for c in found] == ["c3"]
    candidate = found[0]
    assert candidate.type == "subtitle" and candidate.default_remove
    assert candidate.detector_backed
    present = {sample for sample, boxes in data if boxes}
    assert set(candidate.presence_frames) == present and len(present) == 32
    assert candidate.mask.shape == (360, 640, 1)
    assert candidate.mask[candidate.bbox[1], candidate.bbox[0], 0] == 1
    assert "8 separate stretches" in candidate.reason
    # Its own evidence rule accepts a box in the row and nothing else.
    assert candidate.evidence_filter(line_box(100))
    assert not candidate.evidence_filter(line_box(100, y=200))


def test_a_track_the_classifier_would_keep_is_not_admitted():
    # Narrow text in the middle of the frame is scene text, kept by default.
    n = 300
    rows = {}
    for start in [20 + 30 * k for k in range(8)]:
        for ordinal in range(start, start + 4):
            rows[ordinal] = [line_box(50, y=170, centre=320)]
    assert len(find_tracks(samples_with(rows, n), SHAPE)) == 1
    assert sparse_candidates(as_text_boxes(samples_with(rows, n)), SHAPE, [], 1) == []


def test_the_row_of_a_subtitle_candidate_belongs_to_it():
    n = 300
    starts = [20 + 30 * k for k in range(8)]
    data = as_text_boxes(samples_with(appearances(starts, 4), n))
    row = existing((100, 38, 540, 60), "subtitle")
    assert sparse_candidates(data, SHAPE, [row], 4) == []
    # A logo at the same height elsewhere does not claim the row.
    logo = existing((560, 30, 639, 70), "logo")
    assert len(sparse_candidates(data, SHAPE, [logo], 4)) == 1


def _thumbnails(data, same):
    rng = np.random.default_rng(5)
    fixed = rng.integers(0, 255, (180, 320), dtype=np.uint8)
    return {
        sample: (fixed if same else rng.integers(0, 255, (180, 320), dtype=np.uint8))
        for sample, _boxes in data
    }


def test_a_static_graphic_that_returns_is_not_a_line_of_text():
    # The same pixels every time it is on screen: a logo or a watermark that is
    # shown only part of the time. Text that changes from line to line is the
    # other case, and only that is admitted.
    n = 300
    starts = [20 + 30 * k for k in range(8)]
    data = samples_with(appearances(starts, 4), n)
    boxes = as_text_boxes(data)
    assert len(sparse_candidates(boxes, SHAPE, [], 3, _thumbnails(data, same=False))) == 1
    assert sparse_candidates(boxes, SHAPE, [], 3, _thumbnails(data, same=True)) == []
    # Without thumbnails there is nothing to compare, and the track stands.
    assert len(sparse_candidates(boxes, SHAPE, [], 3, None)) == 1


# ── in the pipeline ──────────────────────────────────────────────────────────

FPS = 5
FRAMES = 600  # 120 s, so 120 samples at one a second
W, H = 160, 100
LINES = [range(40, 55), range(160, 175), range(280, 295), range(400, 415),
         range(520, 535), range(580, 595)]
ONE_OFF = range(100, 115)  # a box in another row, once


def _frame(index):
    frame = np.zeros((H, W, 3), np.uint8)
    frame[0, 0] = (index % 256, index // 256, 7)
    return frame


def _index(frame):
    return int(frame[0, 0, 0]) + 256 * int(frame[0, 0, 1])


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("sparse") / "sparse.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (W, H))
    assert writer.isOpened()
    for index in range(FRAMES):
        writer.write(_frame(index))
    writer.release()
    return str(path)


class Stub:
    """Boxes of a caption row (top of the frame) on the scheduled frames only."""

    def __init__(self, lines=LINES, one_off=ONE_OFF, strays=()):
        self.strays = set(strays)
        self.on = {}
        for k, frames in enumerate(lines):
            for index in frames:
                self.on[index] = (30 + 8 * (k % 3), 70 + 6 * (k % 3))   # varying widths
        self.one_off = set(one_off)
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        index = _index(frame)
        boxes = []
        if index in self.on:
            x1, x2 = self.on[index]
            boxes.append(TextBox(np.array([[x1, 10], [x2 + 60, 10], [x2 + 60, 22], [x1, 22]]), 0.95))
        if index in self.strays:
            # Inside the caption region, but not a box of its row.
            boxes.append(TextBox(np.array([[50, 8], [130, 8], [130, 40], [50, 40]]), 0.9))
        if index in self.one_off:
            boxes.append(TextBox(np.array([[20, 60], [120, 60], [120, 72], [20, 72]]), 0.9))
        return boxes


def run(video, switch, detector=None, mode="balanced"):
    environ = {sparse_tracks.ENV_SWITCH: switch} if switch else {}
    import os
    old = os.environ.copy()
    os.environ.update(environ)
    try:
        draft = prepare(video, detector=detector or Stub(), detect_mode=mode, ocr="off",
                        warn=lambda _message: None)
        return draft, finalize(draft)
    finally:
        os.environ.clear()
        os.environ.update(old)


def removed_frames(plan):
    return {index for index in range(FRAMES) if predicted_mask_at(plan, index).any()}


def test_off_is_exactly_today(video):
    draft, plan = run(video, "off")
    assert draft._result.candidates == []
    assert plan.remove_tracks == []
    assert draft._result.sample_indices == sorted(draft._result.sample_indices)
    assert len(draft._result.sample_indices) == 120
    # Nothing extra was asked of the detector either: the three calls a sample
    # of balanced mode makes (the frame and its two band variants), and no
    # refinement of a track that does not exist.
    detector = Stub()
    run(video, "off", detector=detector)
    assert detector.calls == 3 * 120


def test_unset_means_the_default(video):
    draft, _plan = run(video, None)
    assert bool(draft._result.candidates) is sparse_tracks.DEFAULT_ON


def test_on_removes_the_recurring_row_on_its_own_evidence(video):
    draft, plan = run(video, "on")
    candidates = draft._result.candidates
    assert len(candidates) == 1 and "sparse track" in candidates[0].reason
    assert len(draft._result.sample_indices) == 120
    expected = set().union(*LINES)
    removed = removed_frames(plan)
    # Every frame with a caption box and nothing else: the one-off box in
    # another row was never a track and is not inferred into one, and the
    # frames between the appearances are untouched.
    assert removed == expected
    assert not removed & set(ONE_OFF)


def test_fast_mode_never_uses_the_rule(video):
    # Captions on 45 % of the time: below fast's 0.50 consistency, above what
    # the rule needs, so only the exclusion keeps it out of fast.
    lines = [range(start, start + 18) for start in range(10, 600, 40)]
    draft, plan = run(video, "on", detector=Stub(lines=lines, one_off=[]), mode="fast")
    assert draft._result.candidates == []
    assert plan.remove_tracks == []
    balanced, _ = run(video, "on", detector=Stub(lines=lines, one_off=[]), mode="balanced")
    assert balanced._result.candidates, "the rule's own tracks exist for the other modes"


def test_boundary_frames_with_stray_boxes_are_not_added(video):
    # Boxes of another shape just after each line, inside the region: they are
    # not evidence for the track, so the segments end where the line does.
    strays = {index for frames in LINES for index in range(frames[-1] + 1, frames[-1] + 3)}
    _draft, plan = run(video, "on", detector=Stub(one_off=[], strays=strays))
    expected = set().union(*LINES)
    assert removed_frames(plan) == expected
    # Segments follow the evidence too: they may overhang a line by the one
    # frame the midpoint rule gives a neighbour, never reach the strays.
    covered = {index for track in plan.remove_tracks for segment in track.segments
               for index in range(segment.start, segment.end)}
    assert expected <= covered
    assert all(any(abs(index - frame) <= 1 for frame in expected) for index in covered)


def test_two_appearances_are_not_enough_however_long(video):
    detector = Stub(lines=[range(40, 100), range(300, 360)], one_off=[])
    draft, _plan = run(video, "on", detector=detector)
    assert draft._result.candidates == []


def test_refinement_counts_only_boxes_the_candidate_accepts(video):
    # A box that overlaps the candidate's region but is not its row does not
    # make a frame present.
    detector = Stub(one_off=[])
    result = detect_clean_candidates(
        video, detector, sample_count=120, sparse_tracks_on=True,
    )
    candidate = result.candidates[0]
    assert candidate.evidence_filter is not None
    stray = 300  # a frame in no line
    cx1, cy1, cx2, _cy2 = candidate.bbox

    class Stray:
        def detect(self, frame):
            index = _index(frame)
            if index == stray:
                # Inside the region horizontally and vertically, wrong row height.
                return [TextBox(np.array([[cx1 + 20, cy1], [cx2 - 20, cy1],
                                          [cx2 - 20, cy1 + 40], [cx1 + 20, cy1 + 40]]), 0.9)]
            return detector.detect(frame)

    result.detector = Stray()
    result.sampled_frame_boxes = {}
    refine_temporal_presence(video, result, {candidate.id: [Segment(290, 310)]}, FRAMES)
    assert stray not in candidate.presence_frames
    plain = detect_clean_candidates(video, detector, sample_count=120, sparse_tracks_on=True)
    plain_candidate = plain.candidates[0]
    plain_candidate.evidence_filter = None
    plain.detector = Stray()
    plain.sampled_frame_boxes = {}
    refine_temporal_presence(video, plain, {plain_candidate.id: [Segment(290, 310)]}, FRAMES)
    assert stray in plain_candidate.presence_frames, "the stray box overlaps, so the filter is what excludes it"
