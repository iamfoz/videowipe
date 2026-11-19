"""P0-1: the split inpainting timers must account for prediction and composition.

The model is replaced by a stub that sleeps, so ``model_s`` is a known
quantity, while decode, resize, blend and the ffmpeg pipe run for real.
"""
import shutil
import subprocess
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from videowipe.inpainters import sttn
from videowipe.inpainters.base import InpaintJob

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires ffmpeg")

W, H, FPS, FRAMES, GAP = 640, 360, 4, 20, 4
MODEL_SLEEP_S = 0.04
EXISTING_KEYS = {"prediction_s", "composition_s", "encoding_finalize_s", "inpainting_s"}
PREDICTION_PARTS = ("cache_read_s", "decode_infer_s", "resize_s", "model_s", "cache_write_s")
COMPOSITION_PARTS = ("decode_output_s", "blend_s", "pipe_write_s")


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("clip") / "source.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        f"testsrc2=size={W}x{H}:rate={FPS}:duration={FRAMES // FPS}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ], check=True)
    return path


@pytest.fixture
def stub_model(monkeypatch):
    calls = []

    def predict(frames, *args):
        calls.append(len(frames))
        time.sleep(MODEL_SLEEP_S)
        value = np.mean(frames, axis=0).astype(np.uint8)
        return [value for _ in frames]

    monkeypatch.setattr(sttn, "_process_segment", predict)
    return calls


def _inpaint(video, tmp_path, *, interval=None, cache=None, temporal=False, dual=False,
             neighbor_stride=5):
    mask = np.zeros((H, W, 1), dtype=np.uint8)
    mask[290:340, 40:600] = 1
    painter = sttn.STTNInpainter(neighbor_stride=neighbor_stride)
    painter.backend = SimpleNamespace(
        benchmark_metadata=lambda: {"device": "cpu", "weight_sha256": "test"},
    )
    metrics = {}
    reader = cv2.VideoCapture(str(video))
    try:
        painter.inpaint(InpaintJob(
            video_path=str(video), mask=mask, output_dir=str(tmp_path),
            fps=FPS, frame_count=FRAMES, width=W, height=H, reader=reader, gap=GAP,
            frame_mask=(lambda index: mask) if temporal else None,
            trial_range=interval, dual=dual, metrics=metrics,
            prediction_cache_dir=str(cache) if cache else None,
        ))
    finally:
        reader.release()
    return metrics


def _within(parts, total, slack=0.004):
    # 2% of the lumped timer, plus a few milliseconds for timer rounding
    # (every value is rounded to 0.1 ms) and scheduler noise on tiny runs.
    return abs(parts - total) <= 0.02 * total + slack


@pytest.mark.parametrize("temporal", [False, True])
def test_parts_account_for_prediction_and_composition(video, tmp_path, stub_model, temporal):
    metrics = _inpaint(video, tmp_path, temporal=temporal)
    assert EXISTING_KEYS <= metrics.keys()
    prediction = sum(metrics[key] for key in PREDICTION_PARTS)
    composition = sum(metrics[key] for key in COMPOSITION_PARTS)
    assert metrics["prediction_s"] >= MODEL_SLEEP_S * (FRAMES // GAP)
    assert _within(prediction, metrics["prediction_s"]), (prediction, metrics)
    assert _within(composition, metrics["composition_s"]), (composition, metrics)
    # The stub sleeps, so nearly all of the prediction is model time.
    assert metrics["model_s"] >= MODEL_SLEEP_S * (FRAMES // GAP)
    assert metrics["decode_infer_s"] > 0 and metrics["resize_s"] > 0
    assert metrics["blend_s"] > 0 and metrics["pipe_write_s"] > 0
    assert (metrics["mask_eval_s"] > 0) is temporal
    # Parts never exceed the lumped timers or the whole loop.
    assert prediction <= metrics["prediction_s"] + 0.004
    assert (metrics["prediction_s"] + metrics["composition_s"] + metrics["mask_eval_s"]
            <= metrics["inpainting_s"] + 0.004)


def test_counts_describe_the_work_done(video, tmp_path, stub_model):
    metrics = _inpaint(video, tmp_path)
    segments = FRAMES // GAP
    assert metrics["segments"] == segments
    assert metrics["frames_read"] == FRAMES and metrics["frames_grabbed"] == 0
    assert metrics["output_frames_read"] == FRAMES and metrics["frames_written"] == FRAMES
    assert metrics["segments_inferred"] == segments and metrics["segments_cached"] == 0
    # One model window per band per segment (gap 4 < neighbour stride 5).
    assert metrics["bands_inferred"] == len(stub_model)
    assert metrics["windows"] == metrics["bands_inferred"]
    assert metrics["prefix_skip_s"] == 0 and metrics["prefix_frames"] == 0
    assert metrics["cache_read_s"] == metrics["cache_write_s"] == 0


def test_windows_follow_the_neighbour_stride(video, tmp_path, stub_model):
    metrics = _inpaint(video, tmp_path, neighbor_stride=2)
    # ceil(4 / 2) windows for every band in every segment
    assert metrics["windows"] == 2 * metrics["bands_inferred"]


def test_trial_reports_prefix_skip_and_positions_by_grab(video, tmp_path, stub_model):
    metrics = _inpaint(video, tmp_path, interval=(16, 20), dual=True)
    # Only the last segment is inferred; frames 0..15 are skipped by both readers.
    assert metrics["prefix_frames"] == 16
    assert metrics["prefix_skip_s"] > 0
    assert metrics["segments"] == 1 and metrics["frames_read"] == GAP
    assert metrics["frames_written"] == 4 and metrics["output_frames_read"] == GAP


def test_cache_hits_are_counted_and_skip_the_decode_conversion(video, tmp_path, stub_model):
    cache = tmp_path / "predictions"
    first = _inpaint(video, tmp_path, cache=cache)
    assert first["cache_write_s"] > 0 and first["segments_cached"] == 0
    assert _within(sum(first[key] for key in PREDICTION_PARTS), first["prediction_s"])
    calls_before = len(stub_model)
    second = _inpaint(video, tmp_path, cache=cache)
    assert len(stub_model) == calls_before  # nothing recomputed
    segments = FRAMES // GAP
    assert second["segments_cached"] == segments and second["segments_inferred"] == 0
    # No band needs pixels, so the inference reader only advances.
    assert second["frames_grabbed"] == FRAMES and second["frames_read"] == 0
    assert second["cache_read_s"] > 0 and second["model_s"] < MODEL_SLEEP_S
    assert _within(sum(second[key] for key in PREDICTION_PARTS), second["prediction_s"])
    assert _within(sum(second[key] for key in COMPOSITION_PARTS), second["composition_s"])
