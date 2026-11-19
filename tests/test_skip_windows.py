"""P2-2: skipping model windows that cannot affect an emitted pixel is exact.

``oracle_process_segment`` is a verbatim copy of ``_process_segment`` as it was
before this change (``perf/blend-box``). The property tests compare the current
function with it bit for bit over many segment lengths, strides and ``needed``
sets, using a backend whose every output depends on exactly which frames are in
the window. The end-to-end tests drive ``STTNInpainter.inpaint`` with a small
deterministic model and compare the bytes written to the encoder pipe with
``VIDEOWIPE_SKIP_WINDOWS=off`` and with the old function.
"""
import importlib.util
import itertools
import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from videowipe.inpainters import sttn
from videowipe.inpainters.base import InpaintJob
from videowipe.inpainters.prediction_cache import PredictionCache
from videowipe.inpainters.sttn import blend_frames, get_ref_index


# ---------------------------------------------------------------------------
# Oracle: the function as it was before this change.
# ---------------------------------------------------------------------------

def oracle_process_segment(frames, backend, w, h, ref_length, neighbor_stride):
    """Inpaint a batch of frames through the model (backend-agnostic)."""
    video_length = len(frames)
    feats = backend.encode(backend.preprocess(frames))

    comp_frames_np = np.zeros((video_length, h, w, 3), dtype=np.float32)
    counts = np.zeros((video_length,), dtype=np.int32)

    for f in range(0, video_length, neighbor_stride):
        neighbor_ids = [
            i for i in range(
                max(0, f - neighbor_stride),
                min(video_length, f + neighbor_stride + 1),
            )
        ]
        ref_ids = get_ref_index(neighbor_ids, video_length, ref_length)

        ids = neighbor_ids + ref_ids
        selected = feats[ids]
        transformed = backend.transform(selected)
        decoded = backend.decode(transformed[: len(neighbor_ids)])

        blend_frames(comp_frames_np, decoded, np.array(neighbor_ids), counts)

    comp_frames = []
    for idx in range(video_length):
        if counts[idx]:
            comp_frames.append(np.clip(comp_frames_np[idx] / counts[idx], 0, 255).astype(np.uint8))
        else:
            comp_frames.append(None)
    return comp_frames


# ---------------------------------------------------------------------------
# A backend whose outputs depend on exactly which ids are in the window
# ---------------------------------------------------------------------------

class WindowBackend:
    """Every transformed row depends on the whole window (its members, their
    order and their number), and every decoded value on its transformed row."""

    def __init__(self):
        self.encoded = 0
        self.windows = []          # the ``selected`` array of every transform call

    def preprocess(self, frames):
        return np.asarray(frames, dtype=np.float32)

    def encode(self, frames):
        self.encoded += 1
        return frames.reshape(len(frames), -1) + np.arange(len(frames))[:, None] * 0.5

    def transform(self, selected):
        self.windows.append(selected.copy())
        n = len(selected)
        weights = np.arange(1, n + 1, dtype=np.float32)[:, None]
        mixed = (selected * weights).sum(axis=0) / n
        return selected * 3.1 + mixed * (n + 0.37)

    def decode(self, transformed):
        return ((transformed * 17.0) % 256.0).astype(np.float32).reshape(len(transformed), 1, 1, 3)


def tiny_frames(length, seed):
    rng = np.random.RandomState(seed)
    return [rng.uniform(0, 255, (1, 1, 3)).astype(np.float32) for _ in range(length)]


def covering_windows(length, stride):
    """Independent description of the windows: ``f -> ids`` and ``idx -> [f]``."""
    windows = {f: list(range(max(0, f - stride), min(length, f + stride + 1)))
               for f in range(0, length, stride)}
    covers = {idx: [f for f, ids in windows.items() if idx in ids] for idx in range(length)}
    return windows, covers


def check_against_oracle(length, stride, ref_length, needed, seed):
    frames = tiny_frames(length, seed)
    old_backend, new_backend = WindowBackend(), WindowBackend()
    expected = oracle_process_segment(frames, old_backend, 1, 1, ref_length, stride)
    got = sttn._process_segment(frames, new_backend, 1, 1, ref_length, stride, needed)

    assert len(got) == length and new_backend.encoded == 1  # every frame is encoded
    assert all(frame is not None for frame in expected)

    windows, covers = covering_windows(length, stride)
    wanted = set(range(length)) if needed is None else set(needed)
    ran = [f for f, ids in windows.items() if wanted.intersection(ids)]
    # Exactly the windows that touch a needed frame run, with the same inputs.
    assert len(new_backend.windows) == len(ran)
    old_by_f = dict(zip(windows, old_backend.windows))
    for call, f in zip(new_backend.windows, ran):
        assert call.shape == old_by_f[f].shape and np.array_equal(call, old_by_f[f])
    assert sttn._count_windows(length, stride, needed) == (len(ran), len(windows))

    for idx in range(length):
        complete = all(f in ran for f in covers[idx])
        if idx in wanted:
            assert got[idx] is not None, idx
        assert (got[idx] is not None) == complete, (idx, covers[idx], ran)
        if got[idx] is not None:
            assert got[idx].dtype == expected[idx].dtype
            assert np.array_equal(got[idx], expected[idx]), idx
    return got, ran, windows


@pytest.mark.parametrize("stride,ref_length", [(5, 5), (2, 5), (1, 3), (3, 4), (7, 5), (5, 2)])
def test_every_returned_frame_is_bit_identical_to_the_full_computation(stride, ref_length):
    rng = random.Random(stride * 100 + ref_length)
    checked = 0
    for length in list(range(1, 28)) + [40]:
        for trial in range(6):
            density = rng.choice([0.0, 0.05, 0.15, 0.4, 0.8, 1.0])
            needed = {i for i in range(length) if rng.random() < density}
            check_against_oracle(length, stride, ref_length, needed, seed=length * 31 + trial)
            checked += 1
    assert checked > 150


@pytest.mark.parametrize("stride", [1, 2, 5, 6])
def test_every_single_needed_frame_and_every_pair(stride):
    for length in (1, 2, 5, 6, 11, 16):
        for single in range(length):
            check_against_oracle(length, stride, 5, [single], seed=length + single)
        for pair in itertools.combinations(range(length), 2):
            check_against_oracle(length, stride, 5, pair, seed=7)


@pytest.mark.parametrize("stride,ref_length", [(5, 5), (2, 5), (1, 3), (7, 5)])
def test_default_needed_is_the_old_function_everywhere(stride, ref_length):
    for length in range(1, 30):
        frames = tiny_frames(length, length)
        old_backend, new_backend = WindowBackend(), WindowBackend()
        expected = oracle_process_segment(frames, old_backend, 1, 1, ref_length, stride)
        for kwargs in ({}, {"needed": None}):
            new_backend = WindowBackend()
            got = sttn._process_segment(frames, new_backend, 1, 1, ref_length, stride, **kwargs)
            assert len(got) == len(expected)
            assert all(a is not None and np.array_equal(a, b) for a, b in zip(got, expected))
            assert len(new_backend.windows) == len(old_backend.windows)
            assert all(np.array_equal(a, b) for a, b in zip(new_backend.windows, old_backend.windows))
        assert sttn._count_windows(length, stride) == (len(old_backend.windows),) * 2


def test_needed_covering_everything_runs_every_window():
    check_against_oracle(23, 5, 5, range(23), seed=3)
    got, ran, windows = check_against_oracle(23, 5, 5, np.arange(23), seed=3)
    assert ran == list(windows)


def test_nothing_needed_runs_no_window_and_returns_nothing():
    backend = WindowBackend()
    got = sttn._process_segment(tiny_frames(12, 1), backend, 1, 1, 5, 5, [])
    assert got == [None] * 12 and backend.windows == [] and backend.encoded == 1


def test_a_partly_covered_frame_is_never_returned():
    # Stride 2, length 12: frame 2 is covered by windows 0, 2 and 4. Needing
    # frame 1 runs windows 0 and 2 only, so frame 2 would be a two-window mean.
    got, ran, _ = check_against_oracle(12, 2, 5, [1], seed=5)
    assert ran == [0, 2]
    assert [i for i, frame in enumerate(got) if frame is not None] == [0, 1]
    # The same frame is returned once its remaining window runs as well.
    got, ran, _ = check_against_oracle(12, 2, 5, [1, 3], seed=5)
    assert 2 in [i for i, frame in enumerate(got) if frame is not None]


def test_the_window_count_matches_the_calls_made():
    for length, stride, needed in ((25, 5, [3]), (25, 5, [3, 4, 21]), (7, 2, range(7)), (9, 3, [])):
        backend = WindowBackend()
        sttn._process_segment(tiny_frames(length, 2), backend, 1, 1, 5, stride, needed)
        assert sttn._count_windows(length, stride, needed)[0] == len(backend.windows)


@pytest.mark.parametrize("value,expected", [
    (None, True), ("", True), ("auto", True), ("on", True), ("ON", True), ("off", False),
    (" Off ", False),
])
def test_override_switch(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("VIDEOWIPE_SKIP_WINDOWS", raising=False)
    else:
        monkeypatch.setenv("VIDEOWIPE_SKIP_WINDOWS", value)
    assert sttn._skip_windows_enabled() is expected


# ---------------------------------------------------------------------------
# The compositor ignores a prediction in a band with no alpha (legacy path)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("kind", ["uint8", "float", "soft", "negative_zero"])
def test_legacy_composition_ignores_predictions_for_bands_without_alpha(dual, kind):
    rng = np.random.RandomState(11)
    h, w = 160, 320
    modes = [(100, 160), (40, 100), (0, 40)]       # the middle band overlaps nothing
    modes_overlapping = [(100, 160), (70, 130), (0, 40)]
    for bands in (modes, modes_overlapping):
        frame = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w, 1), np.float32 if kind != "uint8" else np.uint8)
        mask[5:30, 20:200] = 1                       # only the top band has alpha
        if kind == "soft":
            mask[5:30, 20:200] = rng.uniform(0.1, 1.0, (25, 180, 1))
        if kind == "negative_zero":
            mask[100:160, :] = -0.0
        predictions = {k: rng.randint(0, 256, (120, 640, 3), dtype=np.uint8) for k in range(3)}
        facts = sttn._MaskFacts(mask, bands)
        assert facts.active == (2,)
        only_active = {k: p for k, p in predictions.items() if k in facts.active}
        everything = sttn._compose_legacy(frame, predictions, bands, facts.alpha, w, dual)
        reduced = sttn._compose_legacy(frame, only_active, bands, facts.alpha, w, dual)
        assert everything.tobytes() == reduced.tobytes()
        # And the full compositor agrees on either input.
        upscaler = sttn._BandUpscaler(w)
        for given in (predictions, only_active):
            out = sttn._compose_frame(frame, given, bands, facts, upscaler, w, dual=dual,
                                      own_frame=False, legacy_mask=facts.alpha)
            assert out.tobytes() == everything.tobytes()


# ---------------------------------------------------------------------------
# End to end through STTNInpainter.inpaint with a small deterministic model
# ---------------------------------------------------------------------------

W, H, FRAMES, GAP, STRIDE = 320, 160, 36, 12, 2
ROWS_LOW, ROWS_UP = (110, 150), (10, 50)       # bands 0 / (1 and 2) of the full mask


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
    """A deterministic stand-in for the network: window outputs depend on the
    frames, on which frames share the window and on their order."""

    def __init__(self):
        self.encoded = 0
        self.windows = 0

    def benchmark_metadata(self):
        return {"device": "cpu", "weight_sha256": "fake-weights"}

    def preprocess(self, frames):
        return np.asarray(frames, dtype=np.float32)

    def encode(self, frames):
        self.encoded += 1
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
    path = tmp_path_factory.mktemp("skipclip") / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 8, (W, H))
    assert writer.isOpened()
    rng = np.random.RandomState(5)
    base = rng.randint(0, 256, (H, W, 3), dtype=np.uint8)
    for index in range(FRAMES):
        frame = np.roll(base, index * 7, axis=1).astype(np.int16) + index * 3
        writer.write(frame.clip(0, 255).astype(np.uint8))
    writer.release()
    return str(path)


def _alpha(rows):
    mask = np.zeros((H, W, 1), np.float32)
    if rows is not None:
        mask[rows[0]:rows[1], 30:290] = 1.0
        mask[rows[0]:rows[1], 30:36] = 0.5          # soft edge
    mask.setflags(write=False)
    return mask


ALPHAS = {"none": _alpha(None), "low": _alpha(ROWS_LOW), "up": _alpha(ROWS_UP)}
BOTH = np.maximum(ALPHAS["low"], ALPHAS["up"])


def per_frame_mask(layout):
    """``layout(index) -> 'none' | 'low' | 'up'`` as a stable plan-style callback."""
    def frame_mask(index):
        return ALPHAS[layout(index)]
    frame_mask._videowipe_stable_masks = True
    return frame_mask


def run(tmp_path, monkeypatch, video, *, frame_mask=None, dual=False, trial=None, cache=None,
        skip="auto", oracle=False, legacy=False, model=None):
    """One ``inpaint`` call; returns ``(bytes written to the pipe, metrics, model)``."""
    model = model or FakeModel()
    pipe = FakePipe()
    painter = sttn.STTNInpainter(neighbor_stride=STRIDE)
    painter.backend = model
    metrics = {}
    reader = cv2.VideoCapture(video)
    with monkeypatch.context() as patch:
        patch.setenv("VIDEOWIPE_SKIP_WINDOWS", skip)
        patch.setattr(sttn.subprocess, "Popen", lambda *a, **k: pipe)
        if oracle:
            patch.setattr(sttn, "_process_segment",
                          lambda fr, backend, w, h, r, s, needed=None:
                          oracle_process_segment(fr, backend, w, h, r, s))
        if legacy:
            original = sttn._blend_frame_regions
            patch.setattr(sttn, "_blend_frame_regions", lambda *a: original(*a))
        try:
            painter.inpaint(InpaintJob(
                video_path=video, mask=BOTH, output_dir=str(tmp_path), fps=8, frame_count=FRAMES,
                width=W, height=H, reader=reader, gap=GAP, frame_mask=frame_mask, dual=dual,
                trial_range=trial, metrics=metrics,
                prediction_cache_dir=str(cache) if cache else None,
            ))
        finally:
            reader.release()
    return b"".join(pipe.stdin.writes), metrics, model


# Sparse: the lower band is wanted at a few frames, the upper band at a couple.
SPARSE = lambda i: "low" if i in (3, 4, 17, 30) else "up" if i in (9, 25) else "none"
SPARSE_OTHER = lambda i: "low" if i in (1, 4, 20, 21) else "up" if i in (13, 27, 35) else "none"
DENSE = lambda i: "low" if i % 2 else "up"


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("dual", [False, True])
def test_sparse_mask_bytes_match_the_old_function_and_fewer_windows_run(
        tmp_path, monkeypatch, video, dual, legacy):
    mask = per_frame_mask(SPARSE)
    reference, ref_metrics, _ = run(tmp_path, monkeypatch, video, frame_mask=mask, dual=dual,
                                    legacy=legacy, skip="off", oracle=True)
    off, off_metrics, off_model = run(tmp_path, monkeypatch, video, frame_mask=mask, dual=dual,
                                      legacy=legacy, skip="off")
    on, on_metrics, on_model = run(tmp_path, monkeypatch, video, frame_mask=mask, dual=dual,
                                   legacy=legacy, skip="on")
    assert len(reference) == FRAMES * W * H * 3 * (2 if dual else 1)
    assert off == reference
    assert on == reference
    # Every window the model really ran is counted, and none of the others.
    assert off_metrics["windows"] == off_model.windows == ref_metrics["windows"]
    assert off_metrics["windows_skipped"] == 0
    assert on_metrics["windows"] == on_model.windows
    assert on_metrics["windows"] < off_metrics["windows"]
    assert on_metrics["windows"] + on_metrics["windows_skipped"] == off_metrics["windows"]
    assert on_metrics["bands_inferred"] == off_metrics["bands_inferred"]


@pytest.mark.parametrize("skip", ["auto", "on", "off", "garbage"])
def test_dense_mask_is_unchanged_and_reports_consistent_counts(tmp_path, monkeypatch, video, skip):
    mask = per_frame_mask(DENSE)
    reference, ref_metrics, _ = run(tmp_path, monkeypatch, video, frame_mask=mask, skip="off", oracle=True)
    got, metrics, model = run(tmp_path, monkeypatch, video, frame_mask=mask, skip=skip)
    assert got == reference
    assert metrics["windows"] == model.windows
    assert metrics["windows"] + metrics["windows_skipped"] == ref_metrics["windows"]
    if skip == "off":
        assert metrics["windows_skipped"] == 0


@pytest.mark.parametrize("trial", [(5, 20), (13, 14), (0, 12), (11, 13), (24, 36), (30, 35)])
@pytest.mark.parametrize("temporal", [False, True])
def test_trial_ranges_edge_segments(tmp_path, monkeypatch, video, trial, temporal):
    mask = per_frame_mask(DENSE) if temporal else None
    reference, ref_metrics, _ = run(tmp_path, monkeypatch, video, frame_mask=mask, trial=trial,
                                    skip="off", oracle=True)
    got, metrics, model = run(tmp_path, monkeypatch, video, frame_mask=mask, trial=trial)
    assert got == reference
    assert metrics["windows"] == model.windows
    assert metrics["windows"] + metrics["windows_skipped"] == ref_metrics["windows"]
    first, last = trial
    if (first, last) != (0, 12) and not (first % GAP == 0 and last % GAP == 0):
        # An edge segment is only partly emitted: some of its windows are not needed.
        assert metrics["windows_skipped"] > 0


def test_static_mask_with_the_whole_video_skips_nothing(tmp_path, monkeypatch, video):
    reference, ref_metrics, _ = run(tmp_path, monkeypatch, video, skip="off", oracle=True)
    got, metrics, _ = run(tmp_path, monkeypatch, video)
    assert got == reference
    assert metrics["windows_skipped"] == 0 and metrics["windows"] == ref_metrics["windows"]


# ---------------------------------------------------------------------------
# The prediction cache with partly valid entries
# ---------------------------------------------------------------------------

def entries(directory):
    """``{file name: (pixels, valid)}`` of every cache entry in ``directory``."""
    found = {}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as stored:
            found[path.name] = (stored["pixels"].copy(), stored["valid"].copy())
    return found


def test_cache_read_counts_a_partial_entry_as_a_miss(tmp_path):
    cache = PredictionCache(tmp_path, {"a": 1})
    key = cache.key(0, 3, 0, 120)
    pixels = np.full((120, 640, 3), 7, np.uint8)
    cache.write(key, [pixels, None, pixels])
    got = cache.read(key, 3, need=[0, 2])
    assert got[1] is None and got[0] is not None and cache.hits == 1 and cache.misses == 0
    got = cache.read(key, 3, need=[0, 1])
    assert got[1] is None and cache.hits == 1 and cache.misses == 1
    cache.read(key, 3)                         # no ``need``: every frame
    assert cache.hits == 1 and cache.misses == 2


def test_sparse_run_then_other_frames_then_subset(tmp_path, monkeypatch, video):
    cache = tmp_path / "predictions"
    mask_a, mask_b = per_frame_mask(SPARSE), per_frame_mask(SPARSE_OTHER)
    ref_a, _, _ = run(tmp_path, monkeypatch, video, frame_mask=mask_a, skip="off", oracle=True)
    ref_b, _, _ = run(tmp_path, monkeypatch, video, frame_mask=mask_b, skip="off", oracle=True)

    # A complete reference entry set, computed without skipping, to compare frames against.
    complete = tmp_path / "complete"
    run(tmp_path, monkeypatch, video, frame_mask=per_frame_mask(DENSE), cache=complete, skip="off")
    full = entries(complete)

    # (a) The first run stores a partial entry.
    got_a, metrics_a, model_a = run(tmp_path, monkeypatch, video, frame_mask=mask_a, cache=cache)
    assert got_a == ref_a
    after_a = entries(cache)
    assert after_a and metrics_a["prediction_writes"] == len(after_a)
    assert any(not valid.all() for _p, valid in after_a.values())
    assert metrics_a["windows_skipped"] > 0 and metrics_a["prediction_hits"] == 0

    # A second run needing frames that are not valid in those entries computes them
    # and writes exactly what a full computation would have produced.
    got_b, metrics_b, model_b = run(tmp_path, monkeypatch, video, frame_mask=mask_b, cache=cache)
    assert got_b == ref_b
    assert model_b.windows == metrics_b["windows"] > 0
    assert metrics_b["frames_read"] > 0                    # the inference reader was not grabbed past
    assert metrics_b["windows"] + metrics_b["windows_skipped"] > metrics_b["windows"]

    # (b) The entries now hold the union, every valid frame equals the full computation,
    # and what was valid before is untouched.
    after_b = entries(cache)
    assert after_b.keys() >= after_a.keys()
    grew = False
    for name, (pixels, valid) in after_b.items():
        if name in after_a:
            before_pixels, before_valid = after_a[name]
            assert (valid >= before_valid).all()
            assert np.array_equal(pixels[before_valid], before_pixels[before_valid])
            grew = grew or bool((valid & ~before_valid).any())
        if name in full:
            full_pixels, full_valid = full[name]
            assert full_valid.all()
            assert np.array_equal(pixels[valid], full_pixels[valid])
    assert grew

    # (c) A third run that needs a subset of what is stored is a pure hit.
    def subset(index):
        return SPARSE(index) if index in (3, 4, 17, 30, 9, 25) else "none"
    ref_c, _, _ = run(tmp_path, monkeypatch, video, frame_mask=per_frame_mask(subset), skip="off", oracle=True)
    got_c, metrics_c, model_c = run(tmp_path, monkeypatch, video, frame_mask=per_frame_mask(subset), cache=cache)
    assert got_c == ref_c
    assert model_c.encoded == 0 and model_c.windows == 0
    assert metrics_c["windows"] == 0 and metrics_c["segments_inferred"] == 0
    assert metrics_c["frames_read"] == 0 and metrics_c["frames_grabbed"] == FRAMES
    assert metrics_c["prediction_misses"] == 0 and metrics_c["prediction_writes"] == 0
    assert entries(cache).keys() == after_b.keys()

    # Replaying mask A or B is also a pure hit now.
    for mask, ref in ((mask_a, ref_a), (mask_b, ref_b)):
        got, metrics, model = run(tmp_path, monkeypatch, video, frame_mask=mask, cache=cache)
        assert got == ref and model.windows == 0 and metrics["prediction_writes"] == 0


def test_a_partial_entry_is_never_taken_for_a_complete_one(tmp_path, monkeypatch, video):
    cache = tmp_path / "predictions"
    run(tmp_path, monkeypatch, video, frame_mask=per_frame_mask(SPARSE), cache=cache)
    # A static mask needs every frame of every band.
    reference, _, _ = run(tmp_path, monkeypatch, video, skip="off", oracle=True)
    got, metrics, model = run(tmp_path, monkeypatch, video, cache=cache)
    assert got == reference
    assert model.windows > 0 and metrics["prediction_writes"] > 0
    # Everything is now valid, so a repeat is a pure hit.
    again, metrics, model = run(tmp_path, monkeypatch, video, cache=cache)
    assert again == reference and model.windows == 0
    assert all(valid.all() for _p, valid in entries(cache).values())


def test_skipping_off_ignores_partial_entries_and_completes_them(tmp_path, monkeypatch, video):
    cache = tmp_path / "predictions"
    run(tmp_path, monkeypatch, video, frame_mask=per_frame_mask(SPARSE), cache=cache)
    reference, ref_metrics, _ = run(tmp_path, monkeypatch, video, skip="off", oracle=True)
    got, metrics, model = run(tmp_path, monkeypatch, video, cache=cache, skip="off")
    assert got == reference
    assert metrics["windows"] == ref_metrics["windows"] and metrics["windows_skipped"] == 0
    assert all(valid.all() for _p, valid in entries(cache).values())


@pytest.mark.parametrize("temporal", [False, True])
def test_trial_ranges_populate_and_extend_the_cache(tmp_path, monkeypatch, video, temporal):
    cache = tmp_path / "predictions"
    mask = per_frame_mask(DENSE) if temporal else None
    ref_full, _, _ = run(tmp_path, monkeypatch, video, frame_mask=mask, skip="off", oracle=True)
    ref_trial, _, _ = run(tmp_path, monkeypatch, video, frame_mask=mask, trial=(5, 20),
                          skip="off", oracle=True)

    got, metrics, _ = run(tmp_path, monkeypatch, video, frame_mask=mask, trial=(5, 20), cache=cache)
    assert got == ref_trial
    assert metrics["windows_skipped"] > 0
    partial = entries(cache)
    assert any(not valid.all() for _p, valid in partial.values())

    # The same trial again is a pure hit.
    got, metrics, model = run(tmp_path, monkeypatch, video, frame_mask=mask, trial=(5, 20), cache=cache)
    assert got == ref_trial and model.windows == 0

    # The whole video needs the frames outside the trial; the entries only grow.
    got, metrics, model = run(tmp_path, monkeypatch, video, frame_mask=mask, cache=cache)
    assert got == ref_full
    assert metrics["windows"] > 0
    grown = entries(cache)
    for name, (pixels, valid) in grown.items():
        if name in partial:
            before_pixels, before_valid = partial[name]
            assert (valid >= before_valid).all()
            assert np.array_equal(pixels[before_valid], before_pixels[before_valid])
    # And a trial elsewhere now needs nothing from the model.
    ref_other, _, _ = run(tmp_path, monkeypatch, video, frame_mask=mask, trial=(22, 31),
                          skip="off", oracle=True)
    got, metrics, model = run(tmp_path, monkeypatch, video, frame_mask=mask, trial=(22, 31), cache=cache)
    assert got == ref_other and model.windows == 0


def test_a_corrupt_entry_is_recomputed_and_replaced(tmp_path, monkeypatch, video):
    cache = tmp_path / "predictions"
    mask = per_frame_mask(SPARSE)
    reference, _, _ = run(tmp_path, monkeypatch, video, frame_mask=mask, skip="off", oracle=True)
    run(tmp_path, monkeypatch, video, frame_mask=mask, cache=cache)
    victim = sorted(cache.glob("*.npz"))[0]
    victim.write_bytes(b"not a zip")
    with pytest.warns(RuntimeWarning, match="Corrupt"):
        got, metrics, model = run(tmp_path, monkeypatch, video, frame_mask=mask, cache=cache)
    assert got == reference and metrics["prediction_corrupt"] == 1 and model.windows > 0
    again, _, model = run(tmp_path, monkeypatch, video, frame_mask=mask, cache=cache)
    assert again == reference and model.windows == 0


# ---------------------------------------------------------------------------
# scripts/skip_windows_estimate.py replays the segment loop without the model
# ---------------------------------------------------------------------------

def _estimate_script():
    path = Path(__file__).parents[1] / "scripts" / "skip_windows_estimate.py"
    spec = importlib.util.spec_from_file_location("skip_windows_estimate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("layout,trial", [
    (SPARSE, None), (DENSE, None), (SPARSE_OTHER, None), (SPARSE, (5, 20)), (DENSE, (13, 14)),
    (None, None), (None, (5, 20)),
])
def test_estimate_script_counts_what_inpaint_runs(tmp_path, monkeypatch, video, layout, trial):
    script = _estimate_script()
    mask = per_frame_mask(layout) if layout else None
    _, metrics, _ = run(tmp_path, monkeypatch, video, frame_mask=mask, trial=trial)
    counted = script.count_windows(FRAMES, W, H, BOTH, mask, gap=GAP, neighbor_stride=STRIDE,
                                   trial_range=trial)
    assert counted["windows_run"] == metrics["windows"]
    assert counted["windows_full"] == metrics["windows"] + metrics["windows_skipped"]
    assert counted["band_segments"] == metrics["bands_inferred"]
    assert counted["segments"] == metrics["segments"]
