"""Cropping the model's input around the mask (``VIDEOWIPE_MODEL_CROP``).

Unit tests cover the box geometry, the ``auto`` rule, the per-segment choice and
the prediction-cache key. The end-to-end tests drive ``STTNInpainter.inpaint``
with a small deterministic model and compare the bytes written to the encoder
pipe with oracles that do not use ``videowipe.inpainters.mask_crop``:

* ``off`` (and an unset variable) must be byte-identical to the whole-band
  pipeline, rebuilt here from ``_process_segment``, ``cv2.resize`` and the
  blend;
* ``auto`` must equal the same pipeline fed a crop box that the test works out
  from the mask by hand.
"""
import hashlib
import json
import warnings

import cv2
import numpy as np
import pytest

import videowipe.fidelity  # noqa: F401  (imports videowipe.timing before any test replaces subprocess.Popen)
from videowipe.inpainters import mask_crop, sttn
from videowipe.inpainters.base import InpaintJob
from videowipe.inpainters.mask_crop import ModelCrop, fit_box
from videowipe.inpainters.prediction_cache import PredictionCache


# ---------------------------------------------------------------------------
# Geometry and the auto rule
# ---------------------------------------------------------------------------

def check_box(crop, union, band, frame_width):
    y0, y1, x0, x1 = union
    assert crop.width * 3 == crop.height * 16, "16:3 exactly"
    assert crop.width % 16 == 0
    assert crop.width >= 640, "never enlarged on its way into a 640 wide model input"
    assert 0 <= crop.left and crop.right <= frame_width
    assert band[0] <= crop.top and crop.bottom <= band[1]
    assert crop.left <= x0 and x1 <= crop.right and crop.top <= y0 and y1 <= crop.bottom


BAND = (720, 1080)       # the bottom band of a 1920 x 1080 frame: 360 rows


def test_a_wide_enough_mask_gets_a_tight_16_by_3_box():
    union = (800, 860, 500, 1400)                    # 900 x 60 of 1920 x 1080
    crop = fit_box(union, BAND, 1920, 1080)
    check_box(crop, union, BAND, 1920)
    margin = round(0.04 * 1080)
    assert crop.width == -(-(900 + 2 * margin) // 16) * 16      # width plus the margin, up to a multiple of 16
    assert crop.height == crop.width * 3 // 16


def test_a_tall_mask_widens_the_box_to_keep_the_aspect():
    union = (760, 1000, 900, 1100)                   # only 200 wide but 240 tall
    crop = fit_box(union, BAND, 1920, 1080)
    check_box(crop, union, BAND, 1920)
    assert crop.width >= (240 + 2 * 43) * 16 / 3


@pytest.mark.parametrize("width", [600, 400, 100])
def test_a_box_narrower_than_the_model_input_is_not_used(width):
    # Cropping it would enlarge it; the band stays (measured: -8.5 dB at 480p).
    union = (800, 820, 600, 600 + width - 2 * 43)
    assert fit_box(union, BAND, 1920, 1080) is None


def test_the_threshold_is_640_after_the_margin_and_rounding():
    margin = round(0.04 * 1080)
    inner = 640 - 2 * margin                          # exactly 640 wide with margins
    rows = (800, 820)
    assert fit_box(rows + (100, 100 + inner), BAND, 1920, 1080).width == 640
    assert fit_box(rows + (100, 100 + inner - 1), BAND, 1920, 1080).width == 640   # rounds up to 640
    assert fit_box(rows + (100, 100 + inner - 17), BAND, 1920, 1080) is None       # 624 after rounding


def test_a_frame_narrower_than_the_model_input_never_crops():
    assert fit_box((100, 120, 10, 300), (0, 112), 600, 360) is None
    assert fit_box((100, 120, 10, 590), (0, 112), 600, 360) is None


def test_full_width_text_keeps_the_band():
    assert fit_box((800, 820, 20, 1900), BAND, 1920, 1080) is None
    assert fit_box((800, 820, 20, 1880), BAND, 1920, 1080) is None   # wider than the frame after the margin


def test_a_box_that_is_the_whole_band_is_not_a_crop():
    # 1920 x 360 at rows 720-1080 is the band itself: nothing to gain, keep the band path.
    assert fit_box((730, 1000, 45, 1877), BAND, 1920, 1080) is None
    assert fit_box((730, 1000, 45, 1850), BAND, 1920, 1080) is not None      # a little narrower: 1888 x 354


def test_a_band_too_short_for_the_box_keeps_the_band():
    # The topmost band can be short; a 640 wide box needs 120 rows.
    assert fit_box((10, 30, 100, 650), (0, 100), 1920, 1080) is None
    crop = fit_box((10, 30, 100, 650), (0, 120), 1920, 1080)
    assert crop.width == 640 and crop.height == 120


def test_the_box_is_clamped_into_the_frame_and_the_band():
    for union in [(720, 740, 0, 700), (1050, 1080, 1220, 1920), (720, 740, 600, 1300)]:
        crop = fit_box(union, BAND, 1920, 1080)
        check_box(crop, union, BAND, 1920)


def test_random_masks_always_give_a_valid_box_or_none():
    rng = np.random.RandomState(3)
    seen = 0
    for _ in range(3000):
        frame_w = int(rng.choice([480, 640, 854, 1280, 1920, 3840]))
        frame_h = frame_w * 9 // 16
        band_h = frame_w * 3 // 16
        top = int(rng.randint(0, frame_h - band_h + 1))
        band = (top, top + band_h)
        x0 = int(rng.randint(0, frame_w - 10)); x1 = int(rng.randint(x0 + 1, frame_w + 1))
        y0 = int(rng.randint(band[0], band[1] - 1)); y1 = int(rng.randint(y0 + 1, band[1] + 1))
        crop = fit_box((y0, y1, x0, x1), band, frame_w, frame_h)
        if crop is not None:
            check_box(crop, (y0, y1, x0, x1), band, frame_w)
            seen += 1
    assert seen > 300


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------

def test_the_default_is_off(monkeypatch):
    monkeypatch.delenv("VIDEOWIPE_MODEL_CROP", raising=False)
    assert mask_crop.crop_mode() == "off"


@pytest.mark.parametrize("value,expected", [("off", "off"), ("auto", "auto"), (" AUTO ", "auto"), ("Off", "off"), ("", "off")])
def test_the_switch_values(monkeypatch, value, expected):
    monkeypatch.setenv("VIDEOWIPE_MODEL_CROP", value)
    assert mask_crop.crop_mode() == expected


def test_an_unknown_value_is_off_with_a_warning(monkeypatch):
    monkeypatch.setenv("VIDEOWIPE_MODEL_CROP", "on")
    with pytest.warns(RuntimeWarning, match="off or auto"):
        assert mask_crop.crop_mode() == "off"


# ---------------------------------------------------------------------------
# Prediction-cache key
# ---------------------------------------------------------------------------

def test_the_key_without_a_crop_is_the_old_key(tmp_path):
    identity = {"backend": "x", "weights": "y"}
    cache = PredictionCache(tmp_path, identity)
    old = hashlib.sha256(json.dumps(dict(identity, frames=[0, 25], crop=[743, 1080]),
                                    sort_keys=True).encode()).hexdigest()
    assert cache.key(0, 25, 743, 1080) == old
    assert cache.key(0, 25, 743, 1080, crop=None) == old


def test_the_box_is_part_of_the_key(tmp_path):
    cache = PredictionCache(tmp_path, {"backend": "x"})
    band = cache.key(0, 25, 743, 1080)
    keys = {band}
    for crop in [ModelCrop(100, 800, 788, 929), ModelCrop(101, 800, 789, 929), ModelCrop(100, 801, 788, 930),
                 ModelCrop(100, 800, 868, 942)]:
        key = cache.key(0, 25, 743, 1080, crop=crop)
        assert key not in keys, "an old or different entry must never answer for this box"
        keys.add(key)
        assert key == cache.key(0, 25, 743, 1080, crop=tuple(crop)), "stable"
    assert len(keys) == 5


# ---------------------------------------------------------------------------
# Choosing boxes per segment
# ---------------------------------------------------------------------------

W, H = 1280, 480
GAP, FRAMES, STRIDE = 12, 36, 2
BANDS = [(240, 480), (0, 240)]           # what get_inpaint_mode gives for the masks below
LOW_ROWS, UP_ROWS = (300, 340), (60, 100)


def alpha(*regions):
    """Soft-edged alpha for ``(rows, cols)`` regions, shape (H, W, 1)."""
    mask = np.zeros((H, W, 1), np.float32)
    for (r0, r1), (c0, c1) in regions:
        mask[r0:r1, c0:c1] = 1.0
        mask[r0:r1, c0:c0 + 5] = 0.5
        mask[r0:r1, c1 - 5:c1] = 0.5
    mask.setflags(write=False)
    return mask


WIDE_LOW = alpha((LOW_ROWS, (300, 940)))             # 640 wide: boxed, 688 x 129
NARROW_LOW = alpha((LOW_ROWS, (500, 900)))           # 400 wide: below the threshold, the band stays
SHIFTED_LOW = alpha((LOW_ROWS, (420, 1060)))         # as wide, elsewhere
TOP_WIDE = alpha((UP_ROWS, (100, 1100)))             # the other band
NONE = alpha()


def union(*masks):
    out = np.maximum.reduce(masks)
    out.setflags(write=False)
    return out


BOTH = union(WIDE_LOW, TOP_WIDE)
EVERYTHING = union(WIDE_LOW, NARROW_LOW, SHIFTED_LOW, TOP_WIDE)


def stable(function):
    function._videowipe_stable_masks = True
    return function


def make_cropper(frame_mask=None, static=None, simple=True):
    job = InpaintJob(video_path="x", mask=static if static is not None else EVERYTHING, frame_mask=frame_mask,
                     width=W, height=H, frame_count=FRAMES, fps=8, gap=GAP, output_dir=".")
    return mask_crop.SegmentCropper(BANDS, W, H, job, sttn._MaskCache(BANDS), sttn._MaskFacts)


def test_a_static_mask_gets_the_same_box_in_every_segment():
    cropper = make_cropper(static=BOTH)
    plans = [cropper.plan(s, s + GAP, {0, 1}, {}) for s in (0, 12, 24)]
    assert plans[0] == plans[1] == plans[2]
    assert set(plans[0]) == {0, 1}, "each band has a box of its own"
    for k, crop in plans[0].items():
        top, bottom = BANDS[k]
        assert top <= crop.top and crop.bottom <= bottom
    assert plans[0][0].width == 688 and plans[0][1].width == 1040


def test_the_box_follows_the_mask_from_segment_to_segment():
    layout = {0: WIDE_LOW, 1: NARROW_LOW, 2: SHIFTED_LOW}
    cropper = make_cropper(frame_mask=stable(lambda index: layout[index // GAP]))
    first, second, third = (cropper.plan(s, s + GAP, {0}, {}) for s in (0, 12, 24))
    assert first[0].left == 300 - 24 and first[0].width == 688
    assert second == {}, "400 wide would be enlarged: the band stays for this segment"
    assert third[0].left != first[0].left and third[0].width == 688
    assert cropper.bands_cropped == 2 and cropper.bands_kept == 1


def test_the_box_covers_the_whole_segment_not_one_frame():
    def frame_mask(index):                           # the text moves right across the segment
        return alpha((LOW_ROWS, (300 + 20 * (index % GAP), 940 + 20 * (index % GAP))))
    cropper = make_cropper(frame_mask=lambda i: frame_mask(i))
    crop = cropper.plan(0, GAP, {0}, {})[0]
    assert crop.left <= 300 and crop.right >= 940 + 20 * (GAP - 1)


def test_frames_outside_a_trial_range_still_count():
    layout = lambda index: WIDE_LOW if index % GAP < 6 else SHIFTED_LOW
    mask = stable(layout)
    full = make_cropper(frame_mask=mask)
    known = {i: sttn._MaskFacts(layout(i), BANDS) for i in range(0, 4)}    # only the emitted frames are known
    trial = make_cropper(frame_mask=mask)
    assert trial.plan(0, GAP, {0}, known) == full.plan(0, GAP, {0}, {}), "a trial and a full run pick one box"


def test_only_active_bands_get_a_box():
    cropper = make_cropper(static=BOTH)
    assert set(cropper.plan(0, GAP, {1}, {})) == {1}
    assert cropper.plan(0, GAP, set(), {}) == {}


def test_a_mask_that_the_box_path_cannot_read_never_crops():
    class Odd:
        simple = False
        boxes = ((0, 1, 0, 0), None)

    cropper = make_cropper(frame_mask=lambda i: WIDE_LOW)
    cropper._facts = lambda index: Odd()
    assert cropper.plan(0, GAP, {0}, {}) == {}


# ---------------------------------------------------------------------------
# Composition of a cropped prediction
# ---------------------------------------------------------------------------

def oracle_blend_region(frame, comp_bgr, crop, alpha_full):
    """Blend a prediction already scaled to the crop (BGR uint8) into the frame."""
    out = frame.copy()
    comp = cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    region = frame[crop.top:crop.bottom, crop.left:crop.right].astype(np.float32)
    a = alpha_full[crop.top:crop.bottom, crop.left:crop.right, None].astype(np.float32)
    out[crop.top:crop.bottom, crop.left:crop.right] = np.clip(a * comp + (1.0 - a) * region, 0, 255).astype(np.uint8)
    return out


@pytest.mark.parametrize("windowed", ["auto", "off"])
@pytest.mark.parametrize("dual", [False, True])
def test_a_cropped_prediction_composes_to_the_oracle_on_every_path(monkeypatch, windowed, dual):
    monkeypatch.setenv("VIDEOWIPE_BLEND_WINDOW_RESIZE", windowed)
    rng = np.random.RandomState(11)
    frame = rng.randint(0, 256, (H, W, 3), dtype=np.uint8)
    prediction = rng.randint(0, 256, (120, 640, 3), dtype=np.uint8)
    facts = sttn._MaskFacts(WIDE_LOW, BANDS)
    crop = fit_box(facts.boxes[0], BANDS[0], W, H)
    assert crop is not None
    tagged = {0: mask_crop.tag(prediction, crop, BANDS[0][0])}
    scaled = cv2.resize(prediction, (crop.width, crop.height))
    expected = oracle_blend_region(frame, scaled, crop, WIDE_LOW[:, :, 0])
    if dual:
        expected = np.vstack([frame, expected])
    upscaler = sttn._BandUpscaler(W)
    fast = sttn._compose_frame(frame, tagged, BANDS, facts, upscaler, W, dual=dual, own_frame=False,
                               legacy_mask=facts.alpha)
    legacy = sttn._compose_legacy(frame, tagged, BANDS, facts.alpha, W, dual)
    assert fast.tobytes() == expected.tobytes()
    assert legacy.tobytes() == expected.tobytes()


def test_a_cropped_prediction_refuses_a_mask_outside_its_box():
    rng = np.random.RandomState(2)
    prediction = rng.randint(0, 256, (120, 640, 3), dtype=np.uint8)
    crop = ModelCrop(100, 300, 788, 429)
    tagged = mask_crop.tag(prediction, crop, 240)
    with pytest.raises(RuntimeError, match="outside the model's crop"):
        tagged.region(sttn._BandUpscaler(W), (60, 100, 90, 400))


def test_tag_and_model_input_do_nothing_without_a_crop():
    frame = np.arange(24, dtype=np.uint8).reshape(4, 2, 3)
    prediction = np.zeros((120, 640, 3), np.uint8)
    assert mask_crop.tag(prediction, None, 10) is prediction
    assert mask_crop.model_input(frame, 1, 3, None).base is frame or np.shares_memory(mask_crop.model_input(frame, 1, 3, None), frame)
    assert np.array_equal(mask_crop.model_input(frame, 1, 3, None), frame[1:3])
    assert np.array_equal(mask_crop.model_input(frame, 1, 3, ModelCrop(1, 2, 2, 4)), frame[2:4, 1:2])


# ---------------------------------------------------------------------------
# End to end through STTNInpainter.inpaint
# ---------------------------------------------------------------------------

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


class FakeModel:
    """A deterministic stand-in for the network; outputs depend on the pixels
    it is given and on which frames share a window."""

    def __init__(self):
        self.windows = 0
        self.inputs = []

    def benchmark_metadata(self):
        return {"device": "cpu", "weight_sha256": "fake-weights"}

    def preprocess(self, frames):
        self.inputs.append(np.asarray(frames, dtype=np.float32))
        return self.inputs[-1]

    def encode(self, frames):
        means = frames.mean(axis=(1, 2))
        sample = frames[:, ::30, ::100, :].reshape(len(frames), -1)
        return np.concatenate([means, sample], axis=1)

    def transform(self, selected):
        self.windows += 1
        n = len(selected)
        weights = np.arange(1, n + 1, dtype=np.float32)[:, None]
        return selected * 0.5 + (selected * weights).sum(axis=0) / n * 0.5 + np.arange(n)[:, None] * 0.37

    def decode(self, transformed):
        pattern = (np.arange(640, dtype=np.float32)[None, :, None] * 0.7
                   + np.arange(120, dtype=np.float32)[:, None, None] * 1.3
                   + np.array([0, 40, 90], np.float32)[None, None, :])
        means = transformed[:, None, None, :3] * 41.0
        return ((means + pattern[None]) % 256).astype(np.uint8)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("cropclip") / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 8, (W, H))
    assert writer.isOpened()
    yy, xx = np.mgrid[0:H, 0:W]
    for index in range(FRAMES):
        base = np.stack([(xx // 5 + index * 3) % 256, (yy // 3 + index * 5) % 256,
                         ((xx + yy) // 7 + index * 11) % 256], axis=-1).astype(np.uint8)
        writer.write(base)
    writer.release()
    return str(path)


def source_frames(video):
    reader = cv2.VideoCapture(video)
    frames = []
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        frames.append(frame)
    reader.release()
    assert len(frames) == FRAMES
    return frames


def per_frame(layout):
    @stable
    def frame_mask(index):
        return layout(index)
    return frame_mask


# Segment 0 boxed in both bands, segment 1 narrow in the lower band, segment 2 shifted.
LAYOUT_MASKS = (union(WIDE_LOW, TOP_WIDE), union(NARROW_LOW, TOP_WIDE), union(SHIFTED_LOW, TOP_WIDE))
LAYOUT = lambda index: LAYOUT_MASKS[index // GAP]


def run(tmp_path, monkeypatch, video, *, mode, frame_mask=None, mask=EVERYTHING, dual=False, trial=None,
        cache=None, skip="auto", legacy=False, windowed="auto", model=None):
    """One ``inpaint`` call; returns ``(bytes written to the pipe, metrics, model)``."""
    model = model or FakeModel()
    pipe = FakePipe()
    painter = sttn.STTNInpainter(neighbor_stride=STRIDE)
    painter.backend = model
    metrics = {}
    reader = cv2.VideoCapture(video)
    with monkeypatch.context() as patch:
        if mode is None:
            patch.delenv("VIDEOWIPE_MODEL_CROP", raising=False)
        else:
            patch.setenv("VIDEOWIPE_MODEL_CROP", mode)
        patch.setenv("VIDEOWIPE_SKIP_WINDOWS", skip)
        patch.setenv("VIDEOWIPE_BLEND_WINDOW_RESIZE", windowed)
        patch.setattr(sttn.subprocess, "Popen", lambda *a, **k: pipe)
        if legacy:
            original = sttn._blend_frame_regions
            patch.setattr(sttn, "_blend_frame_regions", lambda *a: original(*a))
        try:
            painter.inpaint(InpaintJob(
                video_path=video, mask=mask, output_dir=str(tmp_path), fps=8, frame_count=FRAMES,
                width=W, height=H, reader=reader, gap=GAP, frame_mask=frame_mask, dual=dual,
                trial_range=trial, metrics=metrics,
                prediction_cache_dir=str(cache) if cache else None,
            ))
        finally:
            reader.release()
    return b"".join(pipe.stdin.writes), metrics, model


def old_blend(frame_ori, comp_frames, modes, mask):
    """``_blend_frame_regions`` as it stood before the compositing rewrite."""
    active = [(mode, comp) for mode, comp in zip(modes, comp_frames) if comp is not None]
    frame = frame_ori.copy()
    if not active:
        return frame
    work_from = min(mode[0] for mode, _comp in active)
    work_to = max(mode[1] for mode, _comp in active)
    work = frame_ori[work_from:work_to].astype(np.float32)
    for (from_h, to_h), comp in active:
        mask_area = mask[from_h:to_h].astype(np.float32)
        relative = slice(from_h - work_from, to_h - work_from)
        work[relative] = mask_area * comp + (1.0 - mask_area) * work[relative]
    frame[work_from:work_to] = np.clip(work, 0, 255).astype(np.uint8)
    return frame


def oracle_run(video, masks, boxes, *, dual=False, trial=None):
    """The expected bytes, built without ``mask_crop``.

    ``masks(index)`` is the alpha of a frame; ``boxes`` maps ``(segment, band)``
    to the ``ModelCrop`` the model should see (absent: the whole band).
    """
    frames = source_frames(video)
    model = FakeModel()
    first, last = trial or (0, FRAMES)
    out = []
    for segment in range(0, FRAMES // GAP):
        start, end = segment * GAP, (segment + 1) * GAP
        predictions = {}
        for k, (top, bottom) in enumerate(BANDS):
            if not any(masks(i)[top:bottom].any() for i in range(start, end)):
                continue
            box = boxes.get((segment, k))
            small = []
            for i in range(start, end):
                region = frames[i][top:bottom] if box is None else frames[i][box.top:box.bottom, box.left:box.right]
                small.append(cv2.resize(region, (640, 120), interpolation=cv2.INTER_LINEAR).astype(np.float32))
            predictions[k] = sttn._process_segment(small, model, 640, 120, 5, STRIDE)
        for i in range(start, end):
            if not first <= i < last:
                continue
            alpha_i = masks(i)
            frame = frames[i]
            comps = []
            for k, (top, bottom) in enumerate(BANDS):
                box = boxes.get((segment, k))
                if k not in predictions or box is not None:
                    comps.append(None)
                    continue
                comp = cv2.resize(predictions[k][i - start], (W, bottom - top))
                comps.append(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB).astype(np.float32))
            result = old_blend(frame, comps, BANDS, alpha_i)
            for k in range(len(BANDS)):
                box = boxes.get((segment, k))
                if k in predictions and box is not None:
                    scaled = cv2.resize(predictions[k][i - start], (box.width, box.height))
                    result = oracle_blend_region(result, scaled, box, alpha_i[:, :, 0])
            out.append((np.vstack([frame, result]) if dual else result).tobytes())
    return b"".join(out)


# The boxes worked out by hand for LAYOUT (margin 19 px = 4% of 480):
#   WIDE_LOW   cols 300-940, rows 300-340: 640 + 38 = 678 -> 688 wide, 129 high, centred on (620, 320)
#   TOP_WIDE   cols 100-1100, rows 60-100: 1000 + 38 = 1038 -> 1040 wide, 195 high, centred on (600, 80)
#   SHIFTED    cols 420-1060: 688 wide again, centred on x = 740
#   NARROW_LOW cols 500-900: 438 -> 448 < 640, so the band is kept
EXPECTED_BOXES = {
    (0, 0): ModelCrop(276, 256, 964, 385),
    (1, 1): ModelCrop(80, 0, 1120, 195),
    (0, 1): ModelCrop(80, 0, 1120, 195),
    (2, 1): ModelCrop(80, 0, 1120, 195),
    (2, 0): ModelCrop(396, 256, 1084, 385),
}


def test_the_hand_worked_boxes_are_what_fit_box_gives():
    for index, expected in [(0, EXPECTED_BOXES[(0, 0)]), (24, EXPECTED_BOXES[(2, 0)])]:
        facts = sttn._MaskFacts(LAYOUT(index), BANDS)
        assert fit_box(facts.boxes[0], BANDS[0], W, H) == expected
    facts = sttn._MaskFacts(LAYOUT(0), BANDS)
    assert fit_box(facts.boxes[1], BANDS[1], W, H) == EXPECTED_BOXES[(0, 1)]


@pytest.mark.parametrize("mode", [None, "off", "OFF", "bogus"])
@pytest.mark.parametrize("dual", [False, True])
def test_off_is_byte_identical_to_the_whole_band_pipeline(tmp_path, monkeypatch, video, mode, dual):
    mask = per_frame(LAYOUT)
    reference = oracle_run(video, LAYOUT, {}, dual=dual)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        got, metrics, _model = run(tmp_path, monkeypatch, video, mode=mode, frame_mask=mask, dual=dual)
    assert len(reference) == FRAMES * W * H * 3 * (2 if dual else 1)
    assert got == reference
    assert "mask_crop" not in json.dumps(sorted(metrics))


def test_off_never_builds_a_cropper_and_never_tags_a_prediction(tmp_path, monkeypatch, video):
    def boom(*a, **k):
        raise AssertionError("the crop code ran with the switch off")
    monkeypatch.setattr(mask_crop.SegmentCropper, "plan", boom)
    monkeypatch.setattr(mask_crop, "CroppedPrediction", type("Never", (), {"__init__": boom}))
    got, _m, _model = run(tmp_path, monkeypatch, video, mode="off", frame_mask=per_frame(LAYOUT))
    assert got == oracle_run(video, LAYOUT, {})


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("windowed", ["auto", "off"])
def test_auto_equals_the_pipeline_fed_the_hand_worked_boxes(tmp_path, monkeypatch, video, dual, legacy, windowed):
    boxes = {key: value for key, value in EXPECTED_BOXES.items()}
    mask = per_frame(LAYOUT)
    reference = oracle_run(video, LAYOUT, boxes, dual=dual)
    got, metrics, model = run(tmp_path, monkeypatch, video, mode="auto", frame_mask=mask, dual=dual,
                              legacy=legacy, windowed=windowed)
    assert got == reference
    off = oracle_run(video, LAYOUT, {}, dual=dual)
    assert reference != off, "the crop must change something here"
    # Same model cost: one 640x120 inference per band and segment either way.
    _, off_metrics, off_model = run(tmp_path, monkeypatch, video, mode="off", frame_mask=mask, dual=dual)
    assert metrics["bands_inferred"] == off_metrics["bands_inferred"]
    assert metrics["windows"] == off_metrics["windows"] and model.windows == off_model.windows
    assert all(x.shape[1:] == (120, 640, 3) for x in model.inputs)


def test_the_model_is_given_the_box_resized_not_the_band(tmp_path, monkeypatch, video):
    _got, _m, model = run(tmp_path, monkeypatch, video, mode="auto", frame_mask=per_frame(LAYOUT))
    frames = source_frames(video)
    box = EXPECTED_BOXES[(0, 0)]
    expected = [cv2.resize(frames[i][box.top:box.bottom, box.left:box.right], (640, 120),
                           interpolation=cv2.INTER_LINEAR).astype(np.float32) for i in range(GAP)]
    seen = [x for x in model.inputs if np.array_equal(x, np.asarray(expected))]
    assert len(seen) == 1


def test_pixels_outside_the_mask_are_untouched(tmp_path, monkeypatch, video):
    got, _m, _model = run(tmp_path, monkeypatch, video, mode="auto", frame_mask=per_frame(LAYOUT))
    frames = source_frames(video)
    written = np.frombuffer(got, np.uint8).reshape(FRAMES, H, W, 3)
    for i in (0, 13, 30):
        outside = LAYOUT(i)[:, :, 0] == 0
        assert np.array_equal(written[i][outside], frames[i][outside])
        assert not np.array_equal(written[i][~outside], frames[i][~outside])


def test_a_narrow_mask_keeps_the_band_and_the_exact_off_bytes(tmp_path, monkeypatch, video):
    mask = per_frame(lambda index: NARROW_LOW)
    auto, _m, _model = run(tmp_path, monkeypatch, video, mode="auto", frame_mask=mask, mask=NARROW_LOW)
    off, _m2, _model2 = run(tmp_path, monkeypatch, video, mode="off", frame_mask=mask, mask=NARROW_LOW)
    assert auto == off


def test_a_static_mask_without_a_per_frame_callback(tmp_path, monkeypatch, video):
    mask = BOTH
    boxes = {(s, 0): EXPECTED_BOXES[(0, 0)] for s in range(3)}
    boxes.update({(s, 1): EXPECTED_BOXES[(0, 1)] for s in range(3)})
    got, _m, _model = run(tmp_path, monkeypatch, video, mode="auto", mask=mask)
    assert got == oracle_run(video, lambda i: mask, boxes)


@pytest.mark.parametrize("trial", [(5, 20), (13, 14), (24, 36), (0, 12)])
def test_a_trial_range_gives_the_frames_of_the_full_run(tmp_path, monkeypatch, video, trial):
    mask = per_frame(LAYOUT)
    full, _m, _model = run(tmp_path, monkeypatch, video, mode="auto", frame_mask=mask)
    got, _m2, _model2 = run(tmp_path, monkeypatch, video, mode="auto", frame_mask=mask, trial=trial)
    frame_size = W * H * 3
    assert got == full[trial[0] * frame_size:trial[1] * frame_size]


def test_sparse_masks_skip_windows_exactly_with_a_crop(tmp_path, monkeypatch, video):
    layout = lambda index: (WIDE_LOW if index in (3, 4, 30) else TOP_WIDE if index in (9, 25) else NONE)
    mask = per_frame(layout)
    off_skip, off_metrics, _ = run(tmp_path, monkeypatch, video, mode="auto", frame_mask=mask, skip="off")
    on_skip, on_metrics, _ = run(tmp_path, monkeypatch, video, mode="auto", frame_mask=mask, skip="on")
    assert on_skip == off_skip
    assert on_metrics["windows"] < off_metrics["windows"]


def test_the_cache_serves_a_crop_and_never_confuses_it_with_a_band(tmp_path, monkeypatch, video):
    mask = per_frame(LAYOUT)
    cache = tmp_path / "predictions"
    first, m1, _ = run(tmp_path, monkeypatch, video, mode="auto", frame_mask=mask, cache=cache)
    assert m1["prediction_hits"] == 0 and m1["prediction_writes"] > 0
    entries = {p.name for p in cache.glob("*.npz")}
    second, m2, model2 = run(tmp_path, monkeypatch, video, mode="auto", frame_mask=mask, cache=cache)
    assert second == first
    assert m2["prediction_misses"] == 0 and m2["bands_inferred"] == 0 and model2.windows == 0
    # The band run that follows reads only the entry whose input was the band all along
    # (segment 1, lower band: the mask was too narrow to crop) and infers the other five.
    off, m3, _ = run(tmp_path, monkeypatch, video, mode="off", frame_mask=mask, cache=cache)
    assert m3["prediction_hits"] == 1 and m3["bands_inferred"] == 5
    assert off == oracle_run(video, LAYOUT, {})
    assert entries < {p.name for p in cache.glob("*.npz")}, "the band entries sit beside the crop entries"
    # And the other way round: a band entry is not used for a crop.
    third, m4, _ = run(tmp_path, monkeypatch, video, mode="auto", frame_mask=mask, cache=cache)
    assert third == first and m4["bands_inferred"] == 0
