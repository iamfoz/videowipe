"""Exact frame counts for containers that do not store one (WebM/Matroska)."""
from __future__ import annotations

import shutil
import subprocess

import cv2
import pytest

from videowipe import video
from videowipe.detect import _sample_frames_with_indices

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="requires ffmpeg to build fixtures"
)


def _decoded(path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


@pytest.fixture(scope="module")
def vfr_webm(tmp_path_factory):
    """A VP9 WebM with irregular timestamps, like many YouTube downloads:
    OpenCV estimates duration x fps and overshoots the real frame count."""
    path = tmp_path_factory.mktemp("webm") / "vfr.webm"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30",
        "-t", "4", "-vf", "setpts='if(lt(N,60),N/30/TB,(N/30+0.02*mod(N,3))/TB)'",
        "-fps_mode", "vfr", "-c:v", "libvpx-vp9", "-b:v", "100k", "-deadline", "realtime",
        str(path),
    ], check=True)
    return path


def test_webm_estimate_overshoots_and_helper_is_exact(vfr_webm):
    cap = cv2.VideoCapture(str(vfr_webm))
    estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
    cap.release()
    decoded = _decoded(vfr_webm)
    assert estimate > decoded, "fixture no longer reproduces the overshoot"
    assert video.frame_count(str(vfr_webm)) == decoded


def test_sampling_reaches_the_real_last_frame(vfr_webm):
    indices = [i for i, _ in _sample_frames_with_indices(str(vfr_webm), 30)]
    assert indices[-1] == _decoded(vfr_webm) - 1


def test_fallback_without_ffprobe_decodes_to_count(vfr_webm, monkeypatch):
    monkeypatch.setattr(video, "_probe", lambda *a, **k: None)
    video._counted_frames.cache_clear()
    try:
        assert video.frame_count(str(vfr_webm)) == _decoded(vfr_webm)
    finally:
        video._counted_frames.cache_clear()


def test_indexed_container_keeps_opencv_count(tmp_path):
    path = tmp_path / "cfr.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25",
        "-t", "2", "-c:v", "mpeg4", str(path),
    ], check=True)
    cap = cv2.VideoCapture(str(path))
    estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
    cap.release()
    assert video.frame_count(str(path)) == estimate == 50
