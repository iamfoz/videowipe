"""P2-6: compositing only inside the mask's box must not change a single byte.

The oracles below are verbatim copies of the pre-change code
(``perf/instrumentation``): ``_blend_frame_regions`` and the per-frame
composition steps of ``STTNInpainter.inpaint`` (resize each band's prediction to
full width, convert colour, blend, optionally stack the original on top, take
the bytes). Every test compares the current code with them using
``np.array_equal`` / byte equality.
"""
import gc
import itertools
import time
import weakref
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from videowipe.inpainters import sttn
from videowipe.inpainters.base import InpaintJob
from videowipe.plan import (
    MaskAsset, Segment, Source, TemporalResolution, Track, WipePlan,
    execution_masks,
)

DEFAULT_FEATHER = 4  # videowipe.engine._DEFAULT_FEATHER_RADIUS


# ---------------------------------------------------------------------------
# Oracles: the code as it was before this change.
# ---------------------------------------------------------------------------

def old_blend_frame_regions(frame_ori, comp_frames, modes, mask):
    """Blend prepared model crops without converting the whole frame to float."""
    active = [
        (mode, comp)
        for mode, comp in zip(modes, comp_frames)
        if comp is not None
    ]
    frame = frame_ori.copy()
    if not active:
        return frame

    work_from = min(mode[0] for mode, _comp in active)
    work_to = max(mode[1] for mode, _comp in active)
    work = frame_ori[work_from:work_to].astype(np.float32)
    for (from_h, to_h), comp in active:
        mask_area = mask[from_h:to_h].astype(np.float32)
        relative = slice(from_h - work_from, to_h - work_from)
        work[relative] = (
            mask_area * comp
            + (1.0 - mask_area) * work[relative]
        )
    frame[work_from:work_to] = np.clip(work, 0, 255).astype(np.uint8)
    return frame


def old_compose(frame_ori, comps, modes, mask, ori_w, dual):
    """The composition steps of the old per-frame loop. ``comps[k]`` is band k's
    120x640 prediction for this frame, or None."""
    frame_comps = []
    for k in range(len(modes)):
        if comps.get(k) is not None:
            mode_height = modes[k][1] - modes[k][0]
            comp = cv2.resize(
                comps[k], (ori_w, mode_height),
            )
            comp = cv2.cvtColor(
                np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB
            ).astype(np.float32)
            frame_comps.append(comp)
        else:
            frame_comps.append(None)
    frame = old_blend_frame_regions(frame_ori, frame_comps, modes, mask)
    if dual:
        frame = np.vstack([frame_ori, frame])
    return frame.tobytes()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rand_frame(rng, w, h):
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def rand_preds(rng, modes, present=None):
    return {
        k: rng.randint(0, 256, (120, 640, 3), dtype=np.uint8)
        for k in range(len(modes)) if present is None or k in present
    }


def feather(box_mask, radius):
    """The mask the plan producer builds (plan._project_mask, without protection)."""
    mask = box_mask.astype(np.uint8)
    if radius > 0 and mask.any():
        flat = mask.astype(np.float32)
        kernel = max(3, radius * 2 + 1)
        blurred = cv2.GaussianBlur(flat, (kernel, kernel), radius / 2.0)
        mask = np.clip(np.where(flat >= 1.0, 1.0, blurred), 0.0, 1.0)
    return mask


def boxes_mask(w, h, boxes, radius=0, dtype=None):
    mask = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in boxes:
        mask[y0:y1, x0:x1] = 1
    mask = feather(mask, radius)
    if dtype is not None:
        mask = mask.astype(dtype)
    return mask[:, :, None]


def check(frame, mask, modes, preds, dual, *, own_frame=False):
    """Compare the new composition with the oracle; the source frame is intact."""
    w = frame.shape[1]
    before = frame.copy()
    expected = old_compose(frame, preds, modes, mask, w, dual)
    facts = sttn._MaskFacts(mask, modes)
    got = sttn._compose_frame(
        frame.copy() if own_frame else frame, preds, modes, facts,
        sttn._BandUpscaler(w), w, dual=dual, own_frame=own_frame,
        legacy_mask=facts.alpha,
    )
    assert got.tobytes() == expected
    if not own_frame:
        assert np.array_equal(frame, before), "the source frame must not be modified"


# ---------------------------------------------------------------------------
# 1. Compositing equals the whole-band code
# ---------------------------------------------------------------------------

SIZES = [(1920, 1080), (1280, 720), (1001, 563), (641, 300), (333, 211), (1280, 480)]


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("w,h", SIZES)
@pytest.mark.parametrize("radius", [0, DEFAULT_FEATHER, 9])
def test_single_small_box_matches_oracle(w, h, radius, dual):
    rng = np.random.RandomState(w * 7 + h + radius)
    frame = rand_frame(rng, w, h)
    y0 = h - max(8, h // 12) - 3
    mask = boxes_mask(w, h, [(w // 7, y0, w // 7 + w // 2, y0 + max(8, h // 12))], radius)
    modes = sttn.get_inpaint_mode(h, int(w * 3 / 16), mask)
    assert modes
    check(frame, mask, modes, rand_preds(rng, modes), dual)


@pytest.mark.parametrize("dual", [False, True])
def test_4k_width_matches_oracle(dual):
    rng = np.random.RandomState(4)
    w, h = 3840, 2160
    frame = rand_frame(rng, w, h)
    mask = boxes_mask(w, h, [(300, 1900, 2500, 2010), (100, 80, 700, 160)], DEFAULT_FEATHER)
    modes = sttn.get_inpaint_mode(h, int(w * 3 / 16), mask)
    assert len(modes) == 2
    check(frame, mask, modes, rand_preds(rng, modes), dual)


@pytest.mark.parametrize("dual", [False, True])
def test_empty_mask_leaves_frame_untouched(dual):
    rng = np.random.RandomState(1)
    w, h = 640, 360
    frame = rand_frame(rng, w, h)
    mask = np.zeros((h, w, 1), np.float32)
    modes = [(120, 360)]
    check(frame, mask, modes, rand_preds(rng, modes), dual)
    facts = sttn._MaskFacts(mask, modes)
    assert facts.active == () and facts.boxes == (None,)


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
def test_full_mask_matches_oracle(dtype, dual):
    rng = np.random.RandomState(2)
    w, h = 1280, 480
    frame = rand_frame(rng, w, h)
    mask = np.ones((h, w, 1), dtype)
    modes = [(0, 240), (240, 480)]
    check(frame, mask, modes, rand_preds(rng, modes), dual)


@pytest.mark.parametrize("dual", [False, True])
def test_several_boxes_and_boxes_on_every_edge(dual):
    rng = np.random.RandomState(3)
    w, h = 1280, 480
    frame = rand_frame(rng, w, h)
    boxes = [
        (0, 0, 40, 30),            # top-left corner of the frame
        (w - 50, 0, w, 20),        # top-right corner
        (0, h - 25, 90, h),        # bottom-left corner
        (w - 70, h - 25, w, h),    # bottom-right corner
        (500, 230, 700, 250),      # straddles the 240 band boundary
        (600, 100, 601, 101),      # a single pixel
    ]
    for radius in (0, DEFAULT_FEATHER):
        mask = boxes_mask(w, h, boxes, radius)
        for modes in ([(0, 240), (240, 480)], [(0, 480)], [(0, 100), (100, 480)]):
            check(frame, mask, modes, rand_preds(rng, modes), dual)
        modes = sttn.get_inpaint_mode(h, int(w * 3 / 16), mask)
        check(frame, mask, modes, rand_preds(rng, modes), dual)


@pytest.mark.parametrize("dual", [False, True])
def test_overlapping_and_abutting_bands(dual):
    """Later bands read the unrounded float result of earlier ones."""
    rng = np.random.RandomState(5)
    w, h = 1280, 600
    frame = rand_frame(rng, w, h)
    mask = boxes_mask(w, h, [(100, 190, 900, 330), (200, 330, 1100, 420)], DEFAULT_FEATHER)
    layouts = [
        [(200, 440), (150, 350)],          # overlapping, later band on top
        [(150, 350), (200, 440)],          # overlapping, other order
        [(100, 300), (300, 500)],          # abutting
        [(100, 300), (250, 500), (0, 600)],  # three, nested overlaps
        [(0, 200), (400, 600)],            # disjoint: neither contains the mask boxes fully
    ]
    for modes in layouts:
        for present in (None, {0}, {1}):
            preds = rand_preds(rng, modes, None if present is None else present & set(range(len(modes))))
            check(frame, mask, modes, preds, dual)


def test_overlapping_boxes_with_a_bridging_band():
    """A group is built from overlap, not from adjacency in band order."""
    rng = np.random.RandomState(6)
    w, h = 1280, 600
    frame = rand_frame(rng, w, h)
    mask = boxes_mask(w, h, [(0, 10, 300, 150), (900, 300, 1200, 460), (100, 100, 1000, 400)], 0)
    modes = [(0, 160), (290, 470), (80, 420)]
    for dual in (False, True):
        check(frame, mask, modes, rand_preds(rng, modes), dual)


@pytest.mark.parametrize("dual", [False, True])
def test_random_masks_and_bands(dual):
    rng = np.random.RandomState(8)
    for trial in range(25):
        w = int(rng.choice([640, 800, 1280, 1366, 1920]))
        h = int(rng.randint(240, 700))
        frame = rand_frame(rng, w, h)
        boxes = []
        for _ in range(rng.randint(1, 4)):
            x0 = rng.randint(0, w - 1); y0 = rng.randint(0, h - 1)
            boxes.append((x0, y0, rng.randint(x0 + 1, w + 1), rng.randint(y0 + 1, min(h, y0 + 120) + 1)))
        radius = int(rng.choice([0, 3, 4, 9, 25]))
        mask = boxes_mask(w, h, boxes, radius, dtype=rng.choice([None, np.float64]))
        band_count = rng.randint(1, 4)
        modes = []
        for _ in range(band_count):
            top = rng.randint(0, h - 20)
            modes.append((top, int(rng.randint(top + 10, min(h, top + int(w * 3 / 16)) + 1))))
        present = {k for k in range(band_count) if rng.rand() < 0.8}
        check(frame, mask, modes, rand_preds(rng, modes, present), dual)


def test_noise_alpha_with_extreme_values():
    """Dense random alpha including exact 0, exact 1, tiny and denormal values."""
    rng = np.random.RandomState(9)
    w, h = 1280, 480
    frame = rand_frame(rng, w, h)
    alpha = rng.rand(h, w, 1).astype(np.float32)
    alpha[rng.rand(h, w, 1) < 0.5] = 0.0
    alpha[rng.rand(h, w, 1) < 0.1] = 1.0
    alpha[rng.rand(h, w, 1) < 0.05] = np.float32(1e-45)
    alpha[rng.rand(h, w, 1) < 0.05] = np.float32(1.0 - 2 ** -24)
    alpha[:, :300] = 0.0
    modes = [(0, 240), (200, 480)]
    for dual in (False, True):
        check(frame, alpha, modes, rand_preds(rng, modes), dual)


def test_three_channel_alpha_matches_oracle():
    rng = np.random.RandomState(10)
    w, h = 1280, 480
    frame = rand_frame(rng, w, h)
    alpha = np.zeros((h, w, 3), np.float32)
    alpha[300:360, 100:700, 0] = 0.5
    alpha[320:380, 200:300, 2] = 1.0
    modes = [(240, 480)]
    check(frame, alpha, modes, rand_preds(rng, modes), False)


def test_mask_kinds_from_the_real_plan_producer():
    """Static (H, W, 1) and temporal (H, W) masks from ``execution_masks``."""
    rng = np.random.RandomState(11)
    w, h = 1280, 480

    def plan(tracks):
        return WipePlan('wipe_plan', 3, Source('x.mp4', 'a' * 64, w, h, 25, 100), {},
                        TemporalResolution(0, 0, 0), MaskAsset('wipe_plan_masks.npz', ''), tracks)

    def track(name, box, segments, action='remove'):
        mask = np.zeros((h, w), np.uint8)
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = 1
        return Track(name, 'region', name, action, (x0, y0, x1 - 1, y1 - 1), 1, 1,
                     'user', segments, name, mask)

    plans = [
        plan([track('a', (100, 400, 900, 440), [Segment(0, 100)])]),
        plan([track('a', (100, 400, 900, 440), [Segment(0, 50)]),
              track('b', (50, 20, 400, 80), [Segment(30, 100)])]),
        plan([track('a', (100, 380, 900, 440), [Segment(0, 100)]),
              track('p', (300, 390, 500, 430), [Segment(10, 20)], action='protect')]),
    ]
    frame = rand_frame(rng, w, h)
    for radius in (0, DEFAULT_FEATHER, 13):
        for p in plans:
            static, temporal = execution_masks(p, radius)
            candidates = [static]
            if temporal is not None:
                candidates += [temporal(i) for i in (0, 15, 35, 60, 99)]
            for mask in candidates:
                if not mask.any():
                    continue
                modes = sttn.get_inpaint_mode(h, int(w * 3 / 16), mask)
                for dual in (False, True):
                    check(frame, mask[:, :, None] if mask.ndim == 2 else mask, modes,
                          rand_preds(rng, modes), dual)


def test_in_place_frame_gets_the_same_bytes_and_no_copy():
    rng = np.random.RandomState(12)
    w, h = 1280, 480
    frame = rand_frame(rng, w, h)
    mask = boxes_mask(w, h, [(100, 400, 900, 440)], DEFAULT_FEATHER)
    modes = [(240, 480)]
    preds = rand_preds(rng, modes)
    expected = old_compose(frame, preds, modes, mask, w, False)
    owned = frame.copy()
    facts = sttn._MaskFacts(mask, modes)
    got = sttn._compose_frame(owned, preds, modes, facts, sttn._BandUpscaler(w), w,
                              dual=False, own_frame=True, legacy_mask=mask)
    assert got is owned and got.tobytes() == expected
    # Not owned: the input is left alone, and an untouched frame is not copied.
    got = sttn._compose_frame(frame, preds, modes, facts, sttn._BandUpscaler(w), w,
                              dual=False, own_frame=False, legacy_mask=mask)
    assert got is not frame and got.tobytes() == expected
    untouched = sttn._compose_frame(frame, {}, modes, facts, sttn._BandUpscaler(w), w,
                                    dual=False, own_frame=False, legacy_mask=mask)
    assert untouched is frame
    # Dual never edits the source frame in place, even when told it owns it.
    before = frame.copy()
    dual = sttn._compose_frame(frame, preds, modes, facts, sttn._BandUpscaler(w), w,
                               dual=True, own_frame=True, legacy_mask=mask)
    assert np.array_equal(frame, before)
    assert dual.tobytes() == old_compose(frame, preds, modes, mask, w, True)


def test_non_contiguous_frames_match_and_are_written_correctly():
    rng = np.random.RandomState(13)
    w, h = 1280, 480
    mask = boxes_mask(w, h, [(100, 400, 900, 440)], DEFAULT_FEATHER)
    modes = [(240, 480)]
    preds = rand_preds(rng, modes)
    base = rng.randint(0, 256, (h, w, 4), dtype=np.uint8)
    frame = base[:, :, :3]                       # non-contiguous view
    expected = old_compose(frame, preds, modes, mask, w, False)
    facts = sttn._MaskFacts(mask, modes)
    got = sttn._compose_frame(frame, preds, modes, facts, sttn._BandUpscaler(w), w,
                              dual=False, own_frame=False, legacy_mask=mask)
    assert got.tobytes() == expected
    pipe = []
    sink = SimpleNamespace(write=lambda data: pipe.append(bytes(data)))
    sttn._write_frame(sink, frame)
    sttn._write_frame(sink, np.ascontiguousarray(frame))
    assert pipe[0] == pipe[1] == frame.tobytes()


def test_replaced_blend_seam_is_still_honoured(monkeypatch):
    """Code that substitutes ``_blend_frame_regions`` still sees every frame."""
    rng = np.random.RandomState(14)
    w, h = 1280, 480
    frame = rand_frame(rng, w, h)
    mask = boxes_mask(w, h, [(100, 400, 900, 440)], DEFAULT_FEATHER)
    modes = [(240, 480)]
    calls = []
    original = sttn._blend_frame_regions

    def spy(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(sttn, "_blend_frame_regions", spy)
    facts = sttn._MaskFacts(mask, modes)
    got = sttn._compose_frame(frame, rand_preds(rng, modes), modes, facts,
                              sttn._BandUpscaler(w), w, dual=False, own_frame=False,
                              legacy_mask=mask)
    assert calls == [1] and got.shape == frame.shape


# ---------------------------------------------------------------------------
# 2. Cropped resize reproduces the full resize
# ---------------------------------------------------------------------------

def full_crop(pred, w, band_h, box):
    y0, y1, x0, x1 = box
    resized = cv2.resize(pred, (w, band_h))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1].astype(np.float32)


@pytest.mark.parametrize("w", [1920, 1280, 2560, 3840, 960, 1000, 1366, 854, 205, 640, 641])
def test_cropped_resize_is_identical_for_random_boxes(w):
    rng = np.random.RandomState(w)
    for band_h in sorted({int(w * 3 / 16), max(1, int(w * 3 / 16) - 37), 120, 7}):
        upscaler = sttn._BandUpscaler(w)
        for _ in range(25):
            x0 = int(rng.randint(0, w)); x1 = int(rng.randint(x0 + 1, w + 1))
            y0 = int(rng.randint(0, band_h)); y1 = int(rng.randint(y0 + 1, band_h + 1))
            pred = rng.randint(0, 256, (120, 640, 3), dtype=np.uint8)
            got = upscaler.crop(pred, band_h, (y0, y1, x0, x1))
            assert np.array_equal(got, full_crop(pred, w, band_h, (y0, y1, x0, x1)))


def test_common_resolutions_really_use_the_cropped_resize():
    """The shortcut is not dead code at the sizes that matter."""
    for w in (1280, 1920, 2560, 3840):
        band_h = int(w * 3 / 16)
        upscaler = sttn._BandUpscaler(w)
        window = upscaler._window((120, 640, 3), band_h, (band_h - 40, band_h - 8, w // 10, w // 2))
        assert window is not None, w
        dy0, dy1, _sy0, _sy1, dx0, dx1 = window[:6]
        assert (dy1 - dy0) * (dx1 - dx0) < band_h * w // 2


def test_non_uint8_predictions_use_the_full_resize_and_still_match():
    rng = np.random.RandomState(15)
    w, h = 1280, 480
    frame = rand_frame(rng, w, h)
    mask = boxes_mask(w, h, [(100, 400, 900, 440)], DEFAULT_FEATHER)
    modes = [(240, 480)]
    preds = {0: rng.randint(0, 256, (120, 640, 3)).astype(np.float32)}
    for dual in (False, True):
        check(frame, mask, modes, preds, dual)


def test_rejected_windows_fall_back_to_full_resize(monkeypatch):
    """If the on-machine comparison fails, the full resize is used."""
    rng = np.random.RandomState(16)
    w, band_h = 1920, 360
    upscaler = sttn._BandUpscaler(w)
    monkeypatch.setattr(upscaler, "_verified", lambda *args: False)
    pred = rng.randint(0, 256, (120, 640, 3), dtype=np.uint8)
    box = (300, 340, 400, 900)
    assert upscaler._window(pred.shape, band_h, box) is None
    assert np.array_equal(upscaler.crop(pred, band_h, box), full_crop(pred, w, band_h, box))


def test_unstable_weight_ratios_are_never_windowed():
    # 640 -> 1000 reduces to 16/25: some 11-bit weights sit within float32
    # error of a rounding boundary, so position could matter. It must refuse.
    assert not sttn._axis_weights_are_stable(640, 1000)
    assert sttn._axis_weights_are_stable(640, 1920)
    assert sttn._axis_weights_are_stable(120, 360)
    # Horizontally the ratio is refused, so that axis is always the full width.
    assert sttn._BandUpscaler._axis(400, 900, 640, 1000) == (0, 1000, 0, 640, 0, 1000)


# ---------------------------------------------------------------------------
# 3. Mask-derived data: cached by identity, bounded, never stale
# ---------------------------------------------------------------------------

def test_facts_describe_the_box_per_band():
    mask = np.zeros((300, 400), np.float32)
    mask[120:130, 50:90] = 0.5
    mask[250:251, 399:400] = 1.0
    facts = sttn._MaskFacts(mask, [(100, 200), (200, 300), (0, 100)])
    assert facts.alpha.shape == (300, 400, 1)
    assert facts.boxes == ((120, 130, 50, 90), (250, 251, 399, 400), None)
    assert facts.active == (0, 1)
    alpha, remainder = facts.float_alpha(0)
    assert alpha.dtype == remainder.dtype == np.float32
    assert np.array_equal(alpha, mask[120:130, 50:90, None])
    assert np.array_equal(remainder, np.float32(1.0) - alpha)
    assert not alpha.flags.writeable and not remainder.flags.writeable


def test_cache_returns_the_same_facts_for_the_same_array_and_is_bounded():
    modes = [(0, 50)]
    cache = sttn._MaskCache(modes, max_entries=3)
    arrays = [np.zeros((60, 80), np.uint8) for _ in range(6)]
    for a in arrays:
        a[5:9, 5:9] = 1
    first = cache.get(arrays[0])
    assert cache.get(arrays[0]) is first
    for a in arrays:
        cache.get(a)
        assert len(cache) <= 3
    assert len(cache) == 3
    # The most recently used survive; the oldest was evicted.
    assert cache.get(arrays[5]) is cache.get(arrays[5])
    assert cache.get(arrays[0]) is not first


def test_cache_never_returns_stale_data_when_another_array_reuses_an_id():
    modes = [(0, 50)]
    cache = sttn._MaskCache(modes, max_entries=4)
    a = np.zeros((60, 80), np.uint8)
    a[5:9, 5:9] = 1
    facts_a = cache.get(a)
    ref = weakref.ref(a)
    # While cached, the array is kept alive, so its id cannot be reused.
    key = id(a)
    del a
    gc.collect()
    assert ref() is not None
    # Force the pathological case anyway: a different array under a's key.
    b = np.zeros((60, 80), np.uint8)
    b[20:30, 40:70] = 1
    cache._entries[id(b)] = cache._entries.pop(key)
    facts_b = cache.get(b)
    assert facts_b is not facts_a
    assert facts_b.boxes == ((20, 30, 40, 70),)
    assert facts_b.source is b


def test_cache_derived_data_is_bounded_in_bytes_and_still_correct():
    modes = [(0, 200)]
    cache = sttn._MaskCache(modes, max_entries=8, budget=40_000)
    masks = []
    for i in range(6):
        m = np.zeros((200, 300), np.float32)
        m[10 + i:50 + i, 20:120] = 0.25 * (i + 1)
        masks.append(m)
    for _round in range(3):
        for m in masks:
            facts = cache.get(m)
            alpha, remainder = facts.float_alpha(0)
            y0, y1, x0, x1 = facts.boxes[0]
            assert np.array_equal(alpha, m[y0:y1, x0:x1, None])
            assert np.array_equal(remainder, 1.0 - alpha)
            assert cache.derived_bytes <= 40_000
            assert cache.derived_bytes == sum(e.derived_bytes for e in cache._entries.values())
    # A budget smaller than one pair keeps nothing but still computes correctly.
    tiny = sttn._MaskCache(modes, budget=10)
    facts = tiny.get(masks[0])
    alpha, _ = facts.float_alpha(0)
    assert tiny.derived_bytes == 0 and facts._derived == {}
    assert np.array_equal(alpha, masks[0][10:50, 20:120, None])


def test_cache_budget_scales_with_available_memory_and_can_be_overridden(monkeypatch):
    monkeypatch.delenv("VIDEOWIPE_BLEND_CACHE_MB", raising=False)
    monkeypatch.setattr(sttn, "_available_memory_bytes", lambda: 1 << 30)
    small = sttn._blend_cache_budget()
    monkeypatch.setattr(sttn, "_available_memory_bytes", lambda: 16 << 30)
    large = sttn._blend_cache_budget()
    assert 0 < small < large <= 512 << 20
    monkeypatch.setattr(sttn, "_available_memory_bytes", lambda: 0)
    assert sttn._blend_cache_budget() > 0
    monkeypatch.setenv("VIDEOWIPE_BLEND_CACHE_MB", "0")
    assert sttn._blend_cache_budget() == 0
    monkeypatch.setenv("VIDEOWIPE_BLEND_CACHE_MB", "3")
    assert sttn._blend_cache_budget() == 3 << 20


# ---------------------------------------------------------------------------
# 4. End to end: the exact bytes written to the ffmpeg pipe
# ---------------------------------------------------------------------------

E2E_W, E2E_H, E2E_FRAMES, E2E_GAP = 1280, 480, 10, 4


def stub_predict(frames, drop=None):
    """A deterministic stand-in for the model: depends on every frame of the
    segment, on position and on channel, so any misplacement shows."""
    stack = np.asarray(frames, dtype=np.float32)
    mean = stack.mean(axis=0)
    out = []
    for index, frame in enumerate(stack):
        if drop is not None and index == drop:
            out.append(None)
            continue
        pattern = (np.arange(640, dtype=np.float32)[None, :, None] * 0.7
                   + np.arange(120, dtype=np.float32)[:, None, None] * 1.3
                   + np.array([0, 40, 90], np.float32)[None, None, :] + index * 11)
        out.append(((0.5 * frame + 0.3 * mean + pattern) % 256).astype(np.uint8))
    return out


class RecordingSink:
    def __init__(self):
        self.writes = []

    def write(self, data):
        self.writes.append(bytes(data))
        return len(data)

    def close(self):
        pass


class FakePipe:
    def __init__(self):
        self.stdin = RecordingSink()
        self.returncode = 0

    def wait(self):
        return 0

    def poll(self):
        return 0

    def terminate(self):
        pass


class ListReader:
    def __init__(self, frames):
        self.frames = frames
        self.index = 0

    def read(self):
        if self.index == len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def isOpened(self):
        return True

    def release(self):
        pass


def make_clip(path):
    rng = np.random.RandomState(99)
    frames = []
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 8, (E2E_W, E2E_H))
    assert writer.isOpened()
    base = rng.randint(0, 256, (E2E_H, E2E_W, 3), dtype=np.uint8)
    for index in range(E2E_FRAMES):
        frame = np.roll(base, index * 9, axis=1)
        frame = (frame.astype(np.int16) + rng.randint(-20, 21, frame.shape)).clip(0, 255).astype(np.uint8)
        frames.append(frame)
        writer.write(frame)
    writer.release()
    return frames


def oracle_stream(frames, static_mask, frame_mask, stable, dual, trial, drop):
    """What the pre-change ``inpaint`` wrote, reproduced step by step."""
    w, h = E2E_W, E2E_H
    mode = sttn.get_inpaint_mode(h, int(w * 3 / 16), static_mask)
    length = len(frames)
    first, last = trial or (0, length)
    chunks = []
    for i in range(first // E2E_GAP, (last + E2E_GAP - 1) // E2E_GAP):
        start_f, end_f = i * E2E_GAP, min((i + 1) * E2E_GAP, length)
        segment_masks = {}
        active = set(range(len(mode)))
        if frame_mask is not None:
            active = set()
            for index in range(max(start_f, first), min(end_f, last)):
                alpha = np.asarray(frame_mask(index))
                if not stable:
                    alpha = alpha.copy()
                if alpha.ndim == 2:
                    alpha = alpha[:, :, None]
                segment_masks[index] = alpha
                for k, (top, bottom) in enumerate(mode):
                    if k not in active and np.any(alpha[top:bottom] != 0):
                        active.add(k)
        comps = {}
        for k in sorted(active):
            from_h, to_h = mode[k]
            resized = [
                cv2.resize(frames[f][from_h:to_h], (640, 120), interpolation=cv2.INTER_LINEAR)
                .astype(np.float32)
                for f in range(start_f, end_f)
            ]
            comps[k] = stub_predict(resized, drop)
        for j in range(end_f - start_f):
            if not first <= start_f + j < last:
                continue
            per_frame = {
                k: comps[k][j] for k in range(len(mode))
                if comps.get(k) and j < len(comps[k]) and comps[k][j] is not None
            }
            blend_mask = segment_masks[start_f + j] if frame_mask is not None else static_mask
            chunks.append(old_compose(frames[start_f + j], per_frame, mode, blend_mask, w, dual))
    return b"".join(chunks)


def run_inpaint(tmp_path, monkeypatch, frames, video, static_mask, frame_mask, dual, trial, drop):
    pipe = FakePipe()
    monkeypatch.setattr(sttn.subprocess, "Popen", lambda *a, **k: pipe)
    monkeypatch.setattr(sttn, "_process_segment",
                        lambda fr, *args: stub_predict(fr, drop))
    painter = sttn.STTNInpainter()
    painter.backend = SimpleNamespace(benchmark_metadata=lambda: {"device": "cpu", "weight_sha256": "t"})
    reader = cv2.VideoCapture(str(video)) if video else ListReader(frames)
    metrics = {}
    try:
        painter.inpaint(InpaintJob(
            video_path=str(video) if video else "", mask=static_mask, output_dir=str(tmp_path),
            fps=8, frame_count=len(frames), width=E2E_W, height=E2E_H, reader=reader,
            gap=E2E_GAP, frame_mask=frame_mask, dual=dual, trial_range=trial, metrics=metrics,
        ))
    finally:
        if hasattr(reader, "release"):
            reader.release()
    return b"".join(pipe.stdin.writes), pipe.stdin.writes, metrics


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("blendclip") / "clip.avi"
    return str(path), make_clip(path)


def _static_masks(radius):
    return boxes_mask(E2E_W, E2E_H, [(120, 410, 1000, 450), (30, 20, 300, 60)], radius)


def _temporal(radius, stable, kind):
    """A frame_mask over the 10-frame clip. Frames 0-3 have nothing to remove
    (an inactive segment), 4-5 only the top-left box, 6-9 both."""
    top = boxes_mask(E2E_W, E2E_H, [(30, 20, 300, 60)], radius)[:, :, 0]
    both = boxes_mask(E2E_W, E2E_H, [(30, 20, 300, 60), (120, 410, 1000, 450)], radius)[:, :, 0]
    empty = np.zeros_like(top)
    for a in (top, both, empty):
        a.setflags(write=False)
    buffer = np.zeros_like(top)

    def pick(index):
        return empty if index < 4 else top if index < 6 else both

    if kind == "stable":
        def frame_mask(index):
            return pick(index)
        frame_mask._videowipe_stable_masks = True
    elif kind == "stable3d":
        def frame_mask(index):
            return pick(index)[:, :, None]  # a fresh view each call: identity is the base array's
        frame_mask._videowipe_stable_masks = True
    else:  # an unmarked producer that reuses one buffer
        def frame_mask(index):
            buffer[...] = pick(index)
            return buffer
    return frame_mask


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("source", ["video", "fake_reader"])
@pytest.mark.parametrize("radius", [0, DEFAULT_FEATHER])
def test_end_to_end_static_mask_bytes_match_the_old_implementation(
        tmp_path, monkeypatch, clip, radius, source, dual):
    video, frames = clip
    mask = _static_masks(radius)
    expected = oracle_stream(frames, mask, None, False, dual, None, None)
    got, writes, metrics = run_inpaint(
        tmp_path, monkeypatch, frames, video if source == "video" else None,
        mask, None, dual, None, None)
    assert len(writes) == E2E_FRAMES
    assert got == expected
    assert metrics["frames_written"] == E2E_FRAMES
    for key in ("blend_s", "pipe_write_s", "decode_output_s", "mask_eval_s"):
        assert key in metrics


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("kind", ["stable", "stable3d", "unmarked"])
@pytest.mark.parametrize("radius", [0, DEFAULT_FEATHER])
def test_end_to_end_temporal_mask_bytes_match_the_old_implementation(
        tmp_path, monkeypatch, clip, radius, kind, dual):
    video, frames = clip
    static = _static_masks(radius)
    frame_mask = _temporal(radius, kind != "unmarked", kind)
    expected = oracle_stream(frames, static, frame_mask, kind != "unmarked", dual, None, None)
    got, writes, _ = run_inpaint(tmp_path, monkeypatch, frames, video, static, frame_mask, dual, None, None)
    assert len(writes) == E2E_FRAMES
    assert got == expected
    # Frames 0-3 have no active band at all: they are the source, byte for byte.
    frame_bytes = E2E_W * E2E_H * 3 * (2 if dual else 1)
    for index in range(4):
        data = writes[index]
        assert len(data) == frame_bytes
        assert data[-E2E_W * E2E_H * 3:] == frames[index].tobytes()


@pytest.mark.parametrize("dual", [False, True])
def test_end_to_end_trial_range_and_missing_predictions(tmp_path, monkeypatch, clip, dual):
    video, frames = clip
    static = _static_masks(DEFAULT_FEATHER)
    frame_mask = _temporal(DEFAULT_FEATHER, True, "stable")
    expected = oracle_stream(frames, static, frame_mask, True, dual, (3, 9), 2)
    got, writes, _ = run_inpaint(tmp_path, monkeypatch, frames, video, static, frame_mask, dual, (3, 9), 2)
    assert len(writes) == 6
    assert got == expected


def test_end_to_end_timers_and_counters_still_account_for_composition(tmp_path, monkeypatch, clip):
    video, frames = clip
    got, writes, metrics = run_inpaint(
        tmp_path, monkeypatch, frames, video, _static_masks(DEFAULT_FEATHER), None, False, None, None)
    parts = metrics["decode_output_s"] + metrics["blend_s"] + metrics["pipe_write_s"]
    assert abs(parts - metrics["composition_s"]) <= 0.02 * metrics["composition_s"] + 0.004
    assert metrics["output_frames_read"] == metrics["frames_written"] == E2E_FRAMES


# ---------------------------------------------------------------------------
# 5. Speed: a small box must cost clearly less than the whole band
# ---------------------------------------------------------------------------

def test_small_box_composition_is_much_faster_than_the_old_code(capsys):
    rng = np.random.RandomState(21)
    w, h = 1920, 1080
    frame = rand_frame(rng, w, h)
    mask = boxes_mask(w, h, [(300, 930, 1100, 990)], DEFAULT_FEATHER)
    modes = sttn.get_inpaint_mode(h, int(w * 3 / 16), mask)
    preds = rand_preds(rng, modes)
    facts = sttn._MaskFacts(mask, modes)
    upscaler = sttn._BandUpscaler(w)

    def old():
        return old_compose(frame, preds, modes, mask, w, False)

    def new():
        out = sttn._compose_frame(frame, preds, modes, facts, upscaler, w, dual=False,
                                  own_frame=False, legacy_mask=mask)
        return out.tobytes()

    assert old() == new()          # also warms caches and verifies the window

    def best(fn, repeats=7, inner=4):
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(inner):
                fn()
            times.append((time.perf_counter() - start) / inner)
        return min(times)

    old_s, new_s = best(old), best(new)
    with capsys.disabled():
        print(f"\ncomposition per 1080p frame, small box: old {old_s * 1e3:.2f} ms, "
              f"new {new_s * 1e3:.2f} ms, {old_s / new_s:.1f}x")
    assert old_s / new_s >= 2.0
