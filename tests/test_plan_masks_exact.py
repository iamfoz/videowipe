"""Exactness of the plan mask projection against the pre-optimisation code.

The ``_oracle_*`` functions below are verbatim copies of the originals from
upstream commit 20872d4 (only renamed). Every mask the optimised code produces
must match them bit for bit: same dtype, shape and values, compared with
``np.array_equal`` and never ``allclose``.
"""
from __future__ import annotations

import time
from bisect import bisect_right
from collections.abc import Callable
from copy import deepcopy
from functools import lru_cache

import cv2
import numpy as np
import pytest

from videowipe.errors import InvalidInputError
from videowipe.plan import (
    MaskAsset, Segment, Source, TemporalResolution, Track, WipePlan,
    _execution_tracks, _project_mask, _spatial_mask, _union_projection,
    execution_masks, is_temporal, predicted_mask_at, remove_union_mask,
)

# ── oracles: verbatim from 20872d4:src/videowipe/plan.py ─────────────────────

def _oracle_spatial_mask(track, shape, frame_index=None, *, active_only=False):
    mask = np.zeros(shape, dtype=np.uint8)
    for row in track.spatial_segments:
        if frame_index is not None and not row["start"] <= frame_index < row["end"]:
            continue
        if active_only and not any(s.start < row["end"] and s.end > row["start"] for s in track.segments):
            continue
        for x1, y1, x2, y2 in row["boxes"]:
            mask[y1:y2+1, x1:x2+1] = 1
    return mask


def _oracle_project_mask(
    prepared: list[tuple[Track, np.ndarray]],
    active_tracks: tuple[int, ...],
    frame_shape: tuple[int, int],
    feather_radius: int = 0,
    frame_index: int | None = None,
) -> np.ndarray:
    mask = np.zeros(frame_shape, dtype=np.uint8)
    for index in active_tracks:
        if prepared[index][0].action == "remove":
            track, spatial = prepared[index]
            if frame_index is not None and track.spatial_segments is not None:
                spatial = _oracle_spatial_mask(track, frame_shape, frame_index)
            np.maximum(mask, spatial, out=mask)
    if feather_radius > 0 and mask.any():
        flat = mask.astype(np.float32)
        kernel = max(3, feather_radius * 2 + 1)
        blurred = cv2.GaussianBlur(flat, (kernel, kernel), feather_radius / 2.0)
        mask = np.clip(np.where(flat >= 1.0, 1.0, blurred), 0.0, 1.0)
    for index in active_tracks:
        if prepared[index][0].action == "protect":
            mask[prepared[index][1] != 0] = 0
    return mask


def _oracle_active_tracks(
    prepared: list[tuple[Track, np.ndarray]], frame_index: int
) -> tuple[int, ...]:
    return tuple(
        index
        for index, (track, _mask) in enumerate(prepared)
        if any(segment.contains(frame_index) for segment in track.segments)
    )


def _oracle_execution_masks(
    plan: WipePlan, feather_radius: int = 0
) -> tuple[np.ndarray, Callable[[int], np.ndarray] | None]:
    """Project a plan into its static mask and optional temporal frame mask.

    The static mask is ``(H, W, 1)``. Temporal plans also receive a bounded,
    one-entry cached ``frame_mask(index) -> (H, W)`` projection. Cached masks
    are immutable so one consumer cannot corrupt a later frame.
    """
    prepared = _execution_tracks(plan)
    for track, mask in prepared:
        if not mask.any():
            raise InvalidInputError(
                f"track {track.id}: remove track mask is empty; cannot execute"
            )
    frame_shape = (plan.source.height, plan.source.width)
    static_mask = _oracle_union_projection(prepared, frame_shape, feather_radius)[:, :, None]
    if not prepared or not is_temporal(plan):
        return static_mask, None

    @lru_cache(maxsize=1)
    def mask_for(active_tracks: tuple[int, ...], spatial_frame: int | None) -> np.ndarray:
        mask = _oracle_project_mask(
            prepared, active_tracks, frame_shape, feather_radius, spatial_frame
        )
        mask.setflags(write=False)
        return mask

    spatial_starts = {i: [row["start"] for row in track.spatial_segments]
                      for i, (track, _) in enumerate(prepared) if track.spatial_segments is not None}

    def frame_mask(frame_index: int) -> np.ndarray:
        active = _oracle_active_tracks(prepared, frame_index)
        boundary = None
        for i in active:
            if i not in spatial_starts:
                continue
            rows = prepared[i][0].spatial_segments
            index = bisect_right(spatial_starts[i], frame_index) - 1
            changed = (rows[index]["start"] if frame_index < rows[index]["end"]
                       else rows[index]["end"]) if index >= 0 else 0
            boundary = max(boundary or 0, changed)
        # Every frame between joint spatial boundaries has exactly the same
        # alpha. Reuse it instead of blurring a full frame on every video frame.
        return mask_for(active, boundary)

    # Private producer guarantee: these cached results never change after return.
    frame_mask._videowipe_stable_masks = True
    return static_mask, frame_mask


def _oracle_union_projection(prepared, frame_shape, feather_radius=0):
    # Without protection the legacy union is exact and avoids repeated blurs.
    if not any(track.action == "protect" for track, _ in prepared):
        return _oracle_project_mask(prepared, tuple(range(len(prepared))), frame_shape, feather_radius)
    result = np.zeros(frame_shape, dtype=np.float32 if feather_radius else np.uint8)
    boundaries = sorted({s.start for track, _ in prepared for s in track.segments}
                        | {s.end for track, _ in prepared for s in track.segments})
    boundaries = sorted(set(boundaries) | {row[key] for track, _ in prepared
                         for row in (track.spatial_segments or []) for key in ("start", "end")})
    for frame in boundaries[:-1]:
        np.maximum(result, _oracle_project_mask(prepared, _oracle_active_tracks(prepared, frame),
                                        frame_shape, feather_radius, frame), out=result)
    return result


def _oracle_remove_union_mask(plan: WipePlan) -> np.ndarray:
    """Pixels possibly removed at any time, respecting time-limited protection."""
    return _oracle_union_projection(_execution_tracks(plan),
                             (plan.source.height, plan.source.width)).astype(bool)


# ── helpers ──────────────────────────────────────────────────────────────────

ENGINE_FEATHER = 4  # engine._DEFAULT_FEATHER_RADIUS
RADII = (0, ENGINE_FEATHER, 11)


def assert_same(new, old, what=""):
    assert new.dtype == old.dtype, f"{what}: dtype {new.dtype} != {old.dtype}"
    assert new.shape == old.shape, f"{what}: shape {new.shape} != {old.shape}"
    assert np.array_equal(new, old), f"{what}: values differ"


def make_plan(width, height, frames, tracks, schema=3):
    return WipePlan(
        "wipe_plan", schema, Source("x.mp4", "a" * 64, width, height, 25.0, frames), {},
        TemporalResolution(0, 0, 0), MaskAsset("wipe_plan_masks.npz", ""), tracks)


def spatial_track(tid, rows, segments, shape, action="remove"):
    t = Track(tid, "subtitle", tid, action, (0, 0, shape[1] - 1, shape[0] - 1), 1, 1,
              "default", segments, tid, spatial_segments=rows)
    t.mask = _spatial_mask(t, shape)
    return t


def static_track(tid, box, segments, shape, action="remove", value=1):
    mask = np.zeros(shape, np.uint8)
    x1, y1, x2, y2 = box
    mask[y1:y2 + 1, x1:x2 + 1] = value
    return Track(tid, "region", tid, action, tuple(box), 1, 1, "user", segments, tid, mask)


def random_box(rng, width, height):
    """A box of a random kind, including edge, corner, 1-pixel and full-frame ones."""
    kind = rng.integers(0, 9)
    if kind == 0:
        return [0, 0, width - 1, height - 1]
    x1 = int(rng.integers(0, width)); y1 = int(rng.integers(0, height))
    x2 = int(rng.integers(x1, width)); y2 = int(rng.integers(y1, height))
    if kind == 1:
        x1 = y1 = 0                                   # top-left corner
    elif kind == 2:
        x2, y2 = width - 1, height - 1                # bottom-right corner
    elif kind == 3:
        x1, y2 = 0, height - 1                        # bottom-left corner
    elif kind == 4:
        x2, y1 = width - 1, 0                         # top-right corner
    elif kind == 5:
        x2, y2 = x1, y1                               # single pixel
    elif kind == 6:
        x1, x2 = 0, width - 1                         # full-width band
    elif kind == 7:
        y1, y2 = 0, height - 1                        # full-height band
    return [x1, y1, x2, y2]


def random_rows(rng, width, height, frames, count, adjacent=0.4, max_boxes=3):
    """Ordered disjoint rows with a mix of adjacent rows and gaps between rows."""
    rows, cursor = [], int(rng.integers(0, 3))
    for _ in range(count):
        length = int(rng.integers(1, 4))
        if cursor + length > frames:
            break
        rows.append({"start": cursor, "end": cursor + length, "boxes": [
            random_box(rng, width, height) for _ in range(int(rng.integers(1, max_boxes + 1)))]})
        cursor += length + (0 if rng.random() < adjacent else int(rng.integers(1, 4)))
    return rows


def compare_plan(plan, radii=RADII, *, every=1, reverse_too=True):
    """Compare every frame, the union and predicted masks for each radius."""
    frames = plan.source.frame_count
    for radius in radii:
        new_static, new_frame = execution_masks(plan, radius)
        old_static, old_frame = _oracle_execution_masks(plan, radius)
        assert_same(new_static, old_static, f"static r={radius}")
        assert (new_frame is None) == (old_frame is None)
        if new_frame is None:
            continue
        assert new_frame._videowipe_stable_masks is True
        order = list(range(0, frames, every))
        for i in order + (order[::-1] if reverse_too else []):
            new = new_frame(i)
            assert not new.flags.writeable
            assert_same(new, old_frame(i), f"frame {i} r={radius}")
    prepared = _execution_tracks(plan)
    shape = (plan.source.height, plan.source.width)
    for radius in radii:
        assert_same(_union_projection(prepared, shape, radius),
                    _oracle_union_projection(prepared, shape, radius), f"union r={radius}")
    assert_same(remove_union_mask(plan), _oracle_remove_union_mask(plan), "remove_union_mask")
    for i in range(0, frames, every):
        active = _oracle_active_tracks(prepared, i)
        expected = _oracle_project_mask(prepared, active, shape, frame_index=i)
        assert_same(_project_mask(prepared, active, shape, frame_index=i), expected, f"predicted {i}")
        assert_same(predicted_mask_at(plan, i), expected.astype(bool), f"predicted_mask_at {i}")


# ── plans built by the repository's existing tests ───────────────────────────

def test_existing_fixture_plans_are_bit_identical():
    from test_plan_protection import protected_plan
    from test_spatial_subtitles import local_plan
    from test_wipe_plan import _temporal_execution_plan
    from videowipe.server.review import compile_review

    compare_plan(local_plan())
    compare_plan(protected_plan())
    compare_plan(_temporal_execution_plan())
    compare_plan(_temporal_execution_plan(mask_value=255))
    protected, _ = compile_review(deepcopy(local_plan()), protections=[
        {"id": "p_test", "bbox": [0, 1, 4, 7], "segments": [[0, 1]]}])
    compare_plan(protected)
    shortened, _ = compile_review(deepcopy(local_plan()), segment_overrides={"c1": [[2, 6]]})
    compare_plan(shortened)


# ── synthetic plans ──────────────────────────────────────────────────────────

SIZES = [(64, 48), (65, 47), (203, 121), (1279, 719)]


@pytest.mark.parametrize("width,height", SIZES)
def test_static_tracks_only(width, height):
    shape = (height, width)
    plan = make_plan(width, height, 12, [
        static_track("a", [0, 0, width // 2, height // 3], [Segment(0, 5)], shape),
        static_track("b", [width // 3, height // 2, width - 1, height - 1], [Segment(3, 12)], shape),
    ], schema=2)
    compare_plan(plan)
    always = make_plan(width, height, 6, [static_track("a", [1, 1, 20, 9], [Segment(0, 6)], shape)])
    compare_plan(always)


@pytest.mark.parametrize("width,height", SIZES)
@pytest.mark.parametrize("seed", [0, 1])
def test_many_rows_with_varied_boxes(width, height, seed):
    rng = np.random.default_rng(seed * 1000 + width)
    frames = 90
    rows = random_rows(rng, width, height, frames, 40)
    shape = (height, width)
    plan = make_plan(width, height, frames, [spatial_track("s", rows, [Segment(0, frames)], shape)])
    compare_plan(plan, every=1 if width < 300 else 2)


@pytest.mark.parametrize("width,height", [(64, 48), (203, 121), (1279, 719)])
def test_overlapping_remove_tracks_and_protection(width, height):
    rng = np.random.default_rng(width)
    frames = 60
    shape = (height, width)
    tracks = [
        spatial_track("s1", random_rows(rng, width, height, frames, 14), [Segment(0, 40)], shape),
        spatial_track("s2", random_rows(rng, width, height, frames, 14), [Segment(10, 25), Segment(30, 60)], shape),
        static_track("r", random_box(rng, width, height), [Segment(5, 45)], shape),
        static_track("p1", [width // 4, height // 4, 3 * width // 4, 3 * height // 4],
                     [Segment(8, 20), Segment(33, 38)], shape, action="protect"),
        static_track("p2", random_box(rng, width, height), [Segment(0, 60)], shape, action="protect"),
    ]
    compare_plan(make_plan(width, height, frames, tracks), every=1 if width < 300 else 3)


def test_adjacent_rows_gaps_and_edge_boxes_with_protect():
    width, height, frames = 130, 70, 24
    shape = (height, width)
    rows = [
        {"start": 0, "end": 3, "boxes": [[0, 0, 0, 0]]},                         # one pixel, corner
        {"start": 3, "end": 6, "boxes": [[width - 1, height - 1, width - 1, height - 1]]},  # adjacent
        {"start": 8, "end": 9, "boxes": [[0, 0, width - 1, height - 1]]},        # full frame, gap before
        {"start": 9, "end": 12, "boxes": [[0, 30, width - 1, 33]]},              # full-width band
        {"start": 12, "end": 14, "boxes": [[60, 0, 63, height - 1], [0, 0, 5, 5]]},
        {"start": 16, "end": 24, "boxes": [[width - 6, 2, width - 1, 9], [64, 64, 70, 69]]},
    ]
    tracks = [spatial_track("s", rows, [Segment(0, 24)], shape),
              static_track("p", [10, 10, 100, 40], [Segment(2, 11), Segment(15, 20)], shape, "protect")]
    compare_plan(make_plan(width, height, frames, tracks))


def test_unordered_rows_fall_back_to_the_linear_scan():
    """Execution validates, so unordered rows are rejected before any bisect."""
    width, height = 40, 30
    shape = (height, width)
    rows = [{"start": 4, "end": 8, "boxes": [[1, 1, 5, 5]]},
            {"start": 0, "end": 5, "boxes": [[20, 10, 30, 20]]}]
    track = spatial_track("s", [rows[0]], [Segment(0, 10)], shape)
    track.spatial_segments = rows
    plan = make_plan(width, height, 10, [track])
    with pytest.raises(InvalidInputError):
        execution_masks(plan, 4)
    with pytest.raises(InvalidInputError):
        _oracle_execution_masks(plan, 4)
    # The private projection stays correct for overlapping, unsorted rows.
    prepared = [(track, np.ones(shape, np.uint8))]
    for frame in range(10):
        for radius in (0, 4):
            assert_same(_project_mask(prepared, (0,), shape, radius, frame),
                        _oracle_project_mask(prepared, (0,), shape, radius, frame))


# ── the blur window itself ───────────────────────────────────────────────────

@pytest.mark.parametrize("radius", [1, 2, 4, 7, 12])
def test_cropped_blur_equals_full_frame_blur_for_random_boxes(radius):
    rng = np.random.default_rng(radius)
    for _ in range(250):
        width = int(rng.integers(20, 500)); height = int(rng.integers(20, 300))
        box = random_box(rng, width, height)
        track = spatial_track("s", [{"start": 0, "end": 1, "boxes": [box]}], [Segment(0, 1)], (height, width))
        prepared = [(track, track.mask)]
        assert_same(_project_mask(prepared, (0,), (height, width), radius, 0),
                    _oracle_project_mask(prepared, (0,), (height, width), radius, 0),
                    f"{width}x{height} box={box} r={radius}")


def test_returned_masks_are_stable_and_independent():
    from test_spatial_subtitles import local_plan
    _, frame_mask = execution_masks(local_plan(), 1)
    first = frame_mask(0)
    snapshot = first.copy()
    for i in range(6):
        frame_mask(i)
    assert np.array_equal(first, snapshot) and not first.flags.writeable
    assert frame_mask(0) is frame_mask(1)  # one-entry cache still shares across a row


# ── performance ──────────────────────────────────────────────────────────────

def test_thirty_thousand_rows_speed_and_exactness():
    width, height = 1920, 1080
    rows_count, row_len = 30_000, 2
    frames = rows_count * row_len
    rng = np.random.default_rng(7)
    rows = []
    for i in range(rows_count):
        x1 = int(rng.integers(0, width - 400)); y1 = int(rng.integers(0, height - 80))
        rows.append({"start": i * row_len, "end": i * row_len + row_len,
                     "boxes": [[x1, y1, x1 + int(rng.integers(1, 400)), y1 + int(rng.integers(1, 80))]]})
    shape = (height, width)
    plan = make_plan(width, height, frames, [spatial_track("s", rows, [Segment(0, frames)], shape)])
    _, new_frame = execution_masks(plan, ENGINE_FEATHER)
    _, old_frame = _oracle_execution_masks(plan, ENGINE_FEATHER)
    picks = [int(i) for i in np.linspace(0, frames - 1, 1000)]  # each lands on a different row

    start = time.perf_counter()
    for i in picks:
        new_frame(i)
    new_per_call = (time.perf_counter() - start) / len(picks)

    subset = picks[::25]
    start = time.perf_counter()
    for i in subset:
        old_frame(i)
    old_per_call = (time.perf_counter() - start) / len(subset)
    ratio = old_per_call / new_per_call
    print(f"\nframe_mask over {rows_count} rows at {width}x{height}: "
          f"old {old_per_call * 1e3:.2f} ms/call, new {new_per_call * 1e3:.2f} ms/call, "
          f"speed-up {ratio:.1f}x")
    for i in subset:
        assert_same(new_frame(i), old_frame(i), f"frame {i}")
    assert ratio >= 3.0


def test_protected_union_is_faster_and_exact():
    width, height, frames = 1280, 720, 400
    rng = np.random.default_rng(3)
    shape = (height, width)
    rows = random_rows(rng, width, height, frames, 120, max_boxes=1)
    tracks = [spatial_track("s", rows, [Segment(0, frames)], shape),
              static_track("p", [300, 200, 900, 500], [Segment(50, 150), Segment(220, 260)], shape, "protect")]
    plan = make_plan(width, height, frames, tracks)
    prepared = _execution_tracks(plan)
    start = time.perf_counter()
    new = _union_projection(prepared, shape, ENGINE_FEATHER)
    new_time = time.perf_counter() - start
    start = time.perf_counter()
    old = _oracle_union_projection(prepared, shape, ENGINE_FEATHER)
    old_time = time.perf_counter() - start
    print(f"\nprotected union, {len(rows)} rows at {width}x{height}: "
          f"old {old_time:.2f} s, new {new_time:.3f} s, speed-up {old_time / new_time:.1f}x")
    assert_same(new, old, "union")
    assert new_time < old_time
