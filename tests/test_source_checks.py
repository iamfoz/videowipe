"""Warnings for sources with more than 8 bits per sample or an HDR transfer."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from videowipe import source_checks
from videowipe.source_checks import bit_depth, describe, warn_about_source

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg and ffprobe are required")


@pytest.fixture(autouse=True)
def _forget_sources():
    source_checks._warned.clear()
    yield
    source_checks._warned.clear()


@pytest.mark.parametrize("pix_fmt, raw, expected", [
    ("yuv420p", None, 8), ("yuvj420p", "8", 8), ("nv12", None, 8), ("nv16", None, 8),
    ("nv21", "N/A", 8), ("bgr24", None, 8), ("yuv410p", None, 8), ("gray", None, 8),
    ("yuv420p10le", None, 10), ("yuv422p10be", "N/A", 10), ("p010le", None, 10),
    ("p210le", None, 10), ("yuv444p12le", None, 12), ("gbrp12be", None, 12),
    ("yuv420p9le", None, 9), ("gray16le", None, 16), ("yuva444p16le", None, 16),
    ("yuv420p", "10", 10), (None, None, 8), ("", "0", 8),
])
def test_bit_depth(pix_fmt, raw, expected):
    assert bit_depth(pix_fmt, raw) == expected


def test_describe():
    assert describe({"pix_fmt": "yuv420p", "color_transfer": "bt709"}) == []
    assert describe({}) == []
    ten = describe({"pix_fmt": "yuv420p10le"})
    assert len(ten) == 1 and "10-bit" in ten[0] and ten[0].startswith("Warning:")
    for transfer, name in (("smpte2084", "PQ"), ("arib-std-b67", "HLG")):
        hdr = describe({"pix_fmt": "yuv420p10le", "color_transfer": transfer})
        assert len(hdr) == 1 and f"HDR ({name})" in hdr[0]


def test_a_source_that_cannot_be_probed_is_not_reported(tmp_path, capsys):
    missing = tmp_path / "nothing.mp4"
    assert warn_about_source(str(missing)) == []
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    assert warn_about_source(str(broken)) == []
    assert capsys.readouterr().out == ""


def _clip(path, *options):
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         "testsrc=s=64x48:r=25:d=0.4", *options, str(path)],
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip("this ffmpeg cannot encode the test clip")
    return str(path)


@needs_ffmpeg
def test_real_sources_are_classified_and_reported_once(tmp_path, capsys):
    plain = _clip(tmp_path / "plain.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p")
    assert warn_about_source(plain) == []
    assert capsys.readouterr().out == ""

    ten = _clip(tmp_path / "ten.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p10le")
    lines = warn_about_source(ten)
    assert len(lines) == 1 and "10-bit" in lines[0]
    assert capsys.readouterr().out.strip() == lines[0]
    assert warn_about_source(ten) == []  # once per source
    assert capsys.readouterr().out == ""

    # Tags set on the frames as well, after the conversion: ffmpeg 9 writes only the
    # matrix from the output options alone, and converts to them if the frames lack them.
    hdr = _clip(tmp_path / "hdr.mp4", "-vf",
                "format=yuv420p10le,"
                "setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc",
                "-c:v", "libx264", "-pix_fmt", "yuv420p10le",
                "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                "-colorspace", "bt2020nc")
    lines = warn_about_source(hdr)
    assert len(lines) == 1 and "HDR (PQ)" in lines[0]


@needs_ffmpeg
def test_reading_a_source_for_a_plan_gives_the_warning(tmp_path, capsys):
    from videowipe.plan import compute_source

    ten = _clip(tmp_path / "ten.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p10le")
    first = compute_source(ten)
    assert "10-bit" in capsys.readouterr().out
    assert compute_source(ten) == first  # the plan's source is unchanged, and said once
    assert capsys.readouterr().out == ""
