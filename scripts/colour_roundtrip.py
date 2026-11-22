#!/usr/bin/env python3
"""Measure what the cleaning pipeline does to colour.

Sources of known colour (SMPTE bars, saturated flat patches and real content),
correctly converted and tagged for each class below, go through the real
``STTNInpainter.inpaint`` (OpenCV decode, ``bgr24`` pipe, ffmpeg ``libx264`` encode)
with a stub model whose only effect is inside a small mask. Source and output are
then decoded to RGB in a common space, each honouring its own tags the way a
correct player does, and compared on the flat patches and the whole frame.

Three probes say why:

* ``decode``: which matrix and range OpenCV's decoder used (from the BGR values of
  known YCbCr patches, for each tag combination);
* ``encode``: which matrix and range ffmpeg's ``bgr24`` -> ``yuv420p`` conversion
  used (from the YCbCr values of known BGR patches);
* ``passthrough``: how far YCbCr patches of known value move through the whole
  pipeline, in 8-bit code values.

Usage::

    python scripts/colour_roundtrip.py                 # everything
    python scripts/colour_roundtrip.py --classes hd709_tv,sd601_625_tv --frames 25
    python scripts/colour_roundtrip.py --colour off    # VIDEOWIPE_COLOUR for the run
    python scripts/colour_roundtrip.py --probe decode  # one section only

Needs ``ffmpeg`` and ``ffprobe`` on the PATH, OpenCV and numpy. Takes a few minutes
(the HD classes dominate); ``--frames`` and ``--classes`` shorten it. The numbers
depend on the OpenCV build (which ffmpeg libraries it bundles) and on the system
ffmpeg, so run it on the machine that matters; the first lines say which.

Interpretation of a stream when it is decoded for comparison (see ``interpret``):
the matrix and range tags if present; otherwise what players conventionally do
(BT.709 for anything wider than 1024 or taller than 576, BT.601 below; limited
range). Note that ffmpeg's own swscale reads an untagged stream as BT.601 whatever
its size; the table follows the players because they are what people watch with.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    import videowipe  # noqa: F401
except ImportError:  # run from a checkout that is not installed
    sys.path.insert(0, str(ROOT / "src"))
from videowipe.colour import (
    COEFFICIENTS,
    PROBE_YUV,
    TAG_MATRIX,
    probe_patch_rects,
    probe_yuv_frame,
    rgb_to_yuv,
    yuv_to_rgb,
)

# ---------------------------------------------------------------------------------
# Colour maths (pure functions; tests/test_colour_roundtrip.py checks them)
# ---------------------------------------------------------------------------------

#: ffmpeg's scale-filter name of each matrix
SCALE_NAME = {"bt601": "bt601", "bt709": "bt709", "bt2020": "bt2020"}


def conventional_matrix(width: int, height: int) -> str:
    """What players assume for a stream with no matrix tag."""
    return "bt709" if width > 1024 or height > 576 else "bt601"


def interpret(stream: dict) -> tuple[str, str]:
    """``(matrix, range)`` a correct player uses for an ffprobe stream dict:
    the tags when present, else the conventions; ``range`` is ``tv`` or ``pc``."""
    matrix = TAG_MATRIX.get(stream.get("color_space"))
    if matrix is None:
        matrix = conventional_matrix(int(stream["width"]), int(stream["height"]))
    rng = stream.get("color_range")
    if rng not in ("tv", "pc"):
        rng = "pc" if str(stream.get("pix_fmt", "")).startswith("yuvj") else "tv"
    return matrix, rng


def srgb_to_lab(rgb):
    """CIE L*a*b* (D65) of sRGB values in 0..255, last axis 3."""
    c = np.asarray(rgb, np.float64) / 255
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def delta_e76(a, b):
    """CIE76 colour difference between two sRGB colours (or arrays of them)."""
    return np.linalg.norm(srgb_to_lab(a) - srgb_to_lab(b), axis=-1)


def chroma_shift(a, b):
    """Length of the difference of the (Cb, Cr) of two RGB colours (BT.709 full
    range, 8-bit code values): the part of a colour error that is not brightness."""
    ca = rgb_to_yuv(a, "bt709", "pc")[..., 1:]
    cb = rgb_to_yuv(b, "bt709", "pc")[..., 1:]
    return np.linalg.norm(ca - cb, axis=-1)


def psnr(a, b, mask=None) -> float:
    """PSNR in dB of two uint8 arrays; *mask* (bool, same leading shape) selects pixels."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if mask is not None:
        a, b = a[..., mask, :], b[..., mask, :]
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)


# ---------------------------------------------------------------------------------
# Source classes
# ---------------------------------------------------------------------------------

HD = (1280, 720)
SD625 = (720, 576)
SD525 = (720, 480)


@dataclass(frozen=True)
class SourceClass:
    name: str
    size: tuple
    matrix: str            # what the source is converted with (and, if tagged, tagged as)
    range: str = "tv"      # "tv" or "pc"
    pix_fmt: str = "yuv420p"
    tags: dict | None = field(default_factory=dict)  # None: write no tags at all
    note: str = ""


def _tags(space, primaries, trc, rng="tv"):
    return {"colorspace": space, "color_primaries": primaries, "color_trc": trc,
            "color_range": rng}


CLASSES = [
    SourceClass("hd709_tv", HD, "bt709", "tv", tags=_tags("bt709", "bt709", "bt709"),
                note="BT.709 limited, tagged (the common case)"),
    SourceClass("sd601_625_tv", SD625, "bt601", "tv",
                tags=_tags("bt470bg", "bt470bg", "smpte170m"), note="BT.601 625-line, tagged"),
    SourceClass("sd601_525_tv", SD525, "bt601", "tv",
                tags=_tags("smpte170m", "smpte170m", "smpte170m"), note="BT.601 525-line, tagged"),
    SourceClass("hd709_pc", HD, "bt709", "pc", tags=_tags("bt709", "bt709", "bt709", "pc"),
                note="BT.709 full range (yuvj420p)"),
    SourceClass("sd601_pc", SD625, "bt601", "pc", tags=_tags("bt470bg", "bt470bg", "smpte170m", "pc"),
                note="BT.601 full range"),
    SourceClass("hd_untagged", HD, "bt709", "tv", tags=None, note="HD, no tags (converted with 709)"),
    SourceClass("sd_untagged", SD625, "bt601", "tv", tags=None, note="SD, no tags (converted with 601)"),
    SourceClass("hd601_tv", HD, "bt601", "tv", tags=_tags("bt470bg", "bt470bg", "smpte170m"),
                note="HD frame size, BT.601 tagged"),
    SourceClass("sd709_tv", SD625, "bt709", "tv", tags=_tags("bt709", "bt709", "bt709"),
                note="SD frame size, BT.709 tagged"),
    SourceClass("hd2020_tv", HD, "bt2020", "tv", tags=_tags("bt2020nc", "bt2020", "bt2020-10"),
                note="BT.2020 SDR (10-bit transfer tag, 8-bit samples)"),
    SourceClass("hd2020_mtx", HD, "bt2020", "tv", tags={"colorspace": "bt2020nc", "color_range": "tv"},
                note="BT.2020 matrix tag only (no primaries or transfer tags)"),
    SourceClass("hd709_422", HD, "bt709", "tv", pix_fmt="yuv422p",
                tags=_tags("bt709", "bt709", "bt709"), note="BT.709 4:2:2 8-bit"),
    SourceClass("hd709_444", HD, "bt709", "tv", pix_fmt="yuv444p",
                tags=_tags("bt709", "bt709", "bt709"), note="BT.709 4:4:4 8-bit"),
    SourceClass("hd709_10bit", HD, "bt709", "tv", pix_fmt="yuv420p10le",
                tags=_tags("bt709", "bt709", "bt709"), note="BT.709 10-bit (decoded to 8-bit)"),
]

BARS = {  # SMPTE 75 % bars, left to right
    "bar_grey": (191, 191, 191), "bar_yellow": (191, 191, 0), "bar_cyan": (0, 191, 191),
    "bar_green": (0, 191, 0), "bar_magenta": (191, 0, 191), "bar_red": (191, 0, 0),
    "bar_blue": (0, 0, 191),
}
SWATCHES = {  # flat, saturated or typical colours
    "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "cyan": (0, 255, 255),
    "magenta": (255, 0, 255), "yellow": (255, 255, 0), "white": (255, 255, 255),
    "grey": (128, 128, 128), "black": (0, 0, 0), "skin": (194, 150, 130),
    "orange": (255, 128, 0),
}


def layout(width: int, height: int):
    """Rectangles ``(x0, y0, x1, y1)`` of the flat patches' interiors, the content
    region's top, and the rectangle the stub model blanks (the mask)."""
    patches = {}
    bar_h, swatch_h = int(height * 0.22), int(height * 0.18)
    margin = 12
    for i, name in enumerate(BARS):
        x0, x1 = width * i // len(BARS), width * (i + 1) // len(BARS)
        patches[name] = (x0 + margin, margin, x1 - margin, bar_h - margin)
    for i, name in enumerate(SWATCHES):
        x0, x1 = width * i // len(SWATCHES), width * (i + 1) // len(SWATCHES)
        patches[name] = (x0 + margin, bar_h + margin, x1 - margin, bar_h + swatch_h - margin)
    content_top = bar_h + swatch_h
    mask = (int(width * 0.05), height - int(height * 0.12), int(width * 0.30), height - int(height * 0.04))
    return patches, content_top, mask


def reference_frames(width, height, frames, content=None):
    """``frames`` RGB frames (uint8, N x H x W x 3) of the test picture."""
    _, top, _ = layout(width, height)
    out = np.zeros((frames, height, width, 3), np.uint8)
    bar_h, swatch_h = int(height * 0.22), int(height * 0.18)
    for i, rgb in enumerate(BARS.values()):
        out[:, :bar_h, width * i // len(BARS):width * (i + 1) // len(BARS)] = rgb
    for i, rgb in enumerate(SWATCHES.values()):
        out[:, bar_h:bar_h + swatch_h, width * i // len(SWATCHES):width * (i + 1) // len(SWATCHES)] = rgb
    rest = height - top
    real = None
    if content and os.path.isfile(content):
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", "2", "-i", content, "-frames:v", str(frames),
             "-vf", f"scale={width}:{rest}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
            capture_output=True, check=False).stdout
        if len(raw) == frames * rest * width * 3:
            real = np.frombuffer(raw, np.uint8).reshape(frames, rest, width, 3)
    if real is None:  # a moving gradient when no clip is available
        xs = np.linspace(0, 255, width)[None, :]
        ys = np.linspace(0, 255, rest)[:, None]
        for n in range(frames):
            out[n, top:, :, 0] = (xs + n * 3) % 256
            out[n, top:, :, 1] = ys
            out[n, top:, :, 2] = (xs + ys) / 2
    else:
        out[:, top:] = real
    return out


# ---------------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------------


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, capture_output=True, **kwargs)


def probe_stream(path) -> dict:
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=pix_fmt,width,height,color_range,color_space,color_primaries,"
               "color_transfer,nb_frames", "-of", "json", str(path)], text=True).stdout
    return json.loads(out)["streams"][0]


def tag_args(tags):
    args = []
    for key, value in (tags or {}).items():
        args += [f"-{key}", value]
    return args


_SETPARAMS_NAME = {"colorspace": "colorspace", "color_primaries": "color_primaries",
                   "color_trc": "color_trc", "color_range": "range"}


def frame_tags(tags):
    """A ``setparams`` filter labelling frames with *tags* without converting them;
    ``None`` labels them unknown. Newer ffmpeg (9.0.2) writes only what the frames
    carry, converts pixels to match colour options on the output, and lets the
    scale filter label what it converts, so labels must be set here."""
    if tags is None:
        return "setparams=colorspace=unknown:color_primaries=unknown:color_trc=unknown:range=unknown"
    if not tags:
        return "null"
    return "setparams=" + ":".join(f"{_SETPARAMS_NAME[k]}={v}" for k, v in tags.items())


def make_source(path, spec: SourceClass, frames_rgb, fps=25):
    """Encode *frames_rgb* losslessly (``-crf 0``) as *spec*: converted with its
    matrix and range, tagged as its tags say (none at all when ``tags`` is None)."""
    h, w = frames_rgb.shape[1:3]
    scale = (f"scale=out_color_matrix={SCALE_NAME[spec.matrix]}:out_range="
             f"{'tv' if spec.range == 'tv' else 'pc'}:flags=accurate_rnd+full_chroma_int,"
             f"format={spec.pix_fmt},{frame_tags(spec.tags)}")
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-vf", scale, "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "0", *tag_args(spec.tags), str(path)]
    run(cmd, input=frames_rgb.tobytes())


def decode_rgb(path, stream: dict, frames=None):
    """RGB frames (uint8, N x H x W x 3) of *path*, converted the way a correct
    player would read its tags (see :func:`interpret`)."""
    matrix, rng = interpret(stream)
    w, h = int(stream["width"]), int(stream["height"])
    vf = (f"scale=in_color_matrix={SCALE_NAME[matrix]}:in_range={rng}:out_range=pc:"
          "flags=accurate_rnd+full_chroma_int+full_chroma_inp,format=rgb24")
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if frames:
        cmd += ["-frames:v", str(frames)]
    cmd += ["-vf", vf, "-f", "rawvideo", "-"]
    raw = run(cmd).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def decode_yuv(path, w, h, frames=1):
    """Raw 4:2:0 planes of *path* exactly as stored (no pixel-format or range
    conversion: a full-range stream is read as its own yuvj420p)."""
    raw = run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", str(frames),
               "-f", "rawvideo", "-"]).stdout
    n = w * h
    frame = n + 2 * (w // 2) * (h // 2)
    data = np.frombuffer(raw, np.uint8)[: frame * frames].reshape(frames, frame)
    y = data[:, :n].reshape(frames, h, w)
    cb = data[:, n:n + (w // 2) * (h // 2)].reshape(frames, h // 2, w // 2)
    cr = data[:, n + (w // 2) * (h // 2):].reshape(frames, h // 2, w // 2)
    return y, cb, cr


# ---------------------------------------------------------------------------------
# The real pipeline with a stub model
# ---------------------------------------------------------------------------------


def import_pipeline():
    from videowipe.inpainters import sttn
    from videowipe.inpainters.base import InpaintJob
    from videowipe.tasks.base import read_frame_info
    return sttn, InpaintJob, read_frame_info


PIPELINE_LOG: list = []  # the lines the last run_pipeline() printed


def run_pipeline(source, out_dir, mask_box=None) -> str:
    """Clean *source* with a stub model: the mask (``mask_box``, or a small corner
    box) is blanked, everything else is decoded, piped and encoded as in a real
    run. Returns the output path; what the run printed is in ``PIPELINE_LOG``."""
    import contextlib
    import io
    sttn, InpaintJob, read_frame_info = import_pipeline()
    reader, info = read_frame_info(str(source))
    w, h = info["W_ori"], info["H_ori"]
    x0, y0, x1, y1 = mask_box or layout(w, h)[2]
    mask = np.zeros((h, w, 1), np.uint8)
    mask[y0:y1, x0:x1] = 1
    painter = sttn.STTNInpainter()
    painter.backend = object()
    original = sttn._process_segment
    sttn._process_segment = lambda frames, *args: [np.zeros_like(f) for f in frames]
    captured = io.StringIO()
    try:
        job = InpaintJob(str(source), mask, str(out_dir), info["fps"], info["len"], w, h,
                         reader=reader, gap=10)
        with contextlib.redirect_stdout(captured):
            return painter.inpaint(job).output_path
    finally:
        PIPELINE_LOG[:] = [line for line in captured.getvalue().splitlines()
                           if not line.startswith(("Processing frames", "Saved to"))]
        sttn._process_segment = original
        reader.release()


# ---------------------------------------------------------------------------------
# Section 1: the table
# ---------------------------------------------------------------------------------


def compare(src_rgb, out_rgb, size):
    """Metrics of *out_rgb* against *src_rgb* (both N x H x W x 3 uint8)."""
    w, h = size
    patches, content_top, (mx0, my0, mx1, my1) = layout(w, h)
    n = min(len(src_rgb), len(out_rgb))
    src, out = src_rgb[:n], out_rgb[:n]
    mean_abs = []
    maxima = []
    shifts = []
    des = []
    signed = []
    for name, (x0, y0, x1, y1) in patches.items():
        a = src[:, y0:y1, x0:x1].astype(np.float64)
        b = out[:, y0:y1, x0:x1].astype(np.float64)
        d = b - a
        mean_abs.append(np.abs(d).mean(axis=(0, 1, 2)))
        maxima.append(np.abs(d).max(axis=(0, 1, 2)))
        ma, mb = a.mean(axis=(0, 1, 2)), b.mean(axis=(0, 1, 2))
        shifts.append(float(chroma_shift(ma, mb)))
        des.append(float(delta_e76(ma, mb)))
        signed.append(mb - ma)
    keep = np.ones((h, w), bool)
    keep[max(0, my0 - 8):my1 + 8, max(0, mx0 - 8):mx1 + 8] = False
    real = keep.copy()
    real[:content_top] = False  # the real-content part of the picture only
    return {
        "frames": n,
        "patch_mean_abs": np.mean(mean_abs, axis=0).round(2).tolist(),
        "patch_max_abs": np.max(maxima, axis=0).astype(int).tolist(),
        "patch_signed_mean": np.mean(signed, axis=0).round(2).tolist(),
        "chroma_shift": round(float(np.mean(shifts)), 2),
        "chroma_shift_max": round(float(np.max(shifts)), 2),
        "delta_e_mean": round(float(np.mean(des)), 2),
        "delta_e_max": round(float(np.max(des)), 2),
        "worst_patch": max(zip(des, patches), key=lambda t: t[0])[1],
        "psnr": round(psnr(src, out, keep), 1),
        "psnr_content": round(psnr(src, out, real), 1),
        "patch_detail": {k: [round(float(v), 1) for v in s] for k, s in zip(patches, signed)},
    }


def class_row(spec: SourceClass, frames, content, workdir):
    w, h = spec.size
    ref = reference_frames(w, h, frames, content)
    source = Path(workdir) / f"{spec.name}.mp4"
    make_source(source, spec, ref)
    out_dir = Path(workdir) / f"out_{spec.name}"
    out_dir.mkdir()
    output = run_pipeline(source, out_dir)
    src_stream, out_stream = probe_stream(source), probe_stream(output)
    src_rgb = decode_rgb(source, src_stream)
    out_rgb = decode_rgb(output, out_stream)
    result = compare(src_rgb, out_rgb, spec.size)
    # The source's own conversion error against the reference picture (sanity).
    x0, y0, x1, y1 = layout(w, h)[0]["red"]
    result["source_vs_reference_max"] = int(np.abs(
        src_rgb[:, y0:y1, x0:x1].astype(int) - ref[:len(src_rgb), y0:y1, x0:x1].astype(int)).max())
    result["messages"] = list(PIPELINE_LOG)
    result["class"] = spec.name
    result["note"] = spec.note
    result["source_tags"] = tags_of(src_stream)
    result["output_tags"] = tags_of(out_stream)
    result["source_read_as"] = "/".join(interpret(src_stream))
    result["output_read_as"] = "/".join(interpret(out_stream))
    return result


def tags_of(stream):
    return {"pix_fmt": stream.get("pix_fmt"), "range": stream.get("color_range", "-"),
            "space": stream.get("color_space", "-"), "primaries": stream.get("color_primaries", "-"),
            "trc": stream.get("color_transfer", "-")}


def print_table(rows):
    def fmt_tags(t):
        return f"{t['pix_fmt']} {t['range']}/{t['space']}/{t['primaries']}/{t['trc']}"
    print("\n== Round trip: output against source, both read as a correct player would ==")
    print("Errors are output minus source in 8-bit RGB code values on the flat patches "
          "(18 bars and swatches); PSNR (dB) is over the whole frame outside the mask / over its real-content part.\n")
    head = ["class", "mean|e| R,G,B", "max|e|", "chroma", "dE76 mean/max", "PSNR all/content", "read as src -> out"]
    print(" | ".join(head))
    for r in rows:
        print(" | ".join([
            r["class"],
            ",".join(f"{v:.2f}" for v in r["patch_mean_abs"]),
            ",".join(str(v) for v in r["patch_max_abs"]),
            f"{r['chroma_shift']:.2f}",
            f"{r['delta_e_mean']:.2f}/{r['delta_e_max']:.2f} ({r['worst_patch']})",
            f"{r['psnr']:.1f}/{r['psnr_content']:.1f}",
            f"{r['source_read_as']} -> {r['output_read_as']}",
        ]))
    print("\nTags (ffprobe: pix_fmt range/space/primaries/transfer):")
    for r in rows:
        print(f"  {r['class']:14s} source {fmt_tags(r['source_tags'])}")
        print(f"  {'':14s} output {fmt_tags(r['output_tags'])}")


# ---------------------------------------------------------------------------------
# Section 2: probes
# ---------------------------------------------------------------------------------

def make_probe_clip(path, w, h, space, rng, frames=12, primaries=None, trc=None):
    raw = probe_yuv_frame(w, h) * frames
    tags = {"colorspace": space} if space else {}
    if primaries:
        tags["color_primaries"] = primaries
    if trc:
        tags["color_trc"] = trc
    tags["color_range"] = rng
    run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
         "-r", "25", "-i", "-", "-vf", frame_tags(tags), "-c:v", "libx264", "-crf", "0", *tag_args(tags),
         str(path)], input=raw)


def patch_means(planes, w, h):
    y, cb, cr = planes
    out = []
    for x0, y0, x1, y1 in probe_patch_rects(w, h):
        out.append((y[0, y0:y1, x0:x1].mean(), cb[0, y0 // 2:y1 // 2, x0 // 2:x1 // 2].mean(),
                    cr[0, y0 // 2:y1 // 2, x0 // 2:x1 // 2].mean()))
    return np.array(out)


#: tag sets for the decode probe: (matrix, primaries, transfer)
DECODE_TAGS = [(None, None, None), ("bt709", None, None), ("smpte170m", None, None),
               ("bt2020nc", None, None), ("bt709", "bt709", "bt709"),
               ("bt470bg", "bt470bg", "smpte170m"), ("bt2020nc", "bt2020", "bt2020-10")]


def probe_decode(workdir):
    """Which matrix and range does OpenCV's decoder use, for each tag combination?"""
    import cv2
    print("\n== Probe: OpenCV decode (BGR from known YCbCr patches) ==")
    print("tags (matrix/primaries/transfer, range) at size -> best matrix/range, mean|error| "
          "in code values (next best)")
    rows = []
    for (w, h), (space, primaries, trc), rng in itertools.product([HD, SD625], DECODE_TAGS, ["tv", "pc"]):
        path = Path(workdir) / f"probe_{w}_{space}_{primaries}_{rng}.mp4"
        make_probe_clip(path, w, h, space, rng, primaries=primaries, trc=trc)
        cap = cv2.VideoCapture(str(path))
        ok, bgr = cap.read()
        cap.release()
        label = f"{space}/{primaries}/{trc} {rng} at {w}x{h}"
        if not ok:
            print(f"  {label}: could not decode")
            continue
        rgb = bgr[..., ::-1].astype(np.float64)
        measured = np.array([rgb[y0:y1, x0:x1].mean(axis=(0, 1)) for x0, y0, x1, y1 in
                             probe_patch_rects(w, h)])
        errors = {}
        for matrix, r in itertools.product(COEFFICIENTS, ["tv", "pc"]):
            errors[(matrix, r)] = float(np.abs(yuv_to_rgb(np.array(PROBE_YUV), matrix, r) - measured).mean())
        (matrix, r) = min(errors, key=errors.get)
        err = errors[(matrix, r)]
        second = sorted(errors.values())[1]
        verdict = f"{matrix}/{r}" if err < 1.6 else "none of them"
        print(f"  {label:48s}: {verdict:12s} err {err:5.2f} (next best {second:.1f})")
        rows.append({"tags": [space, primaries, trc, rng], "size": [w, h], "matrix": matrix, "range": r,
                     "error": err})
    return rows


def probe_encode(workdir, extra=()):
    """Which matrix and range does ffmpeg's bgr24 -> yuv420p conversion use (with the
    pipeline's own encoder options)?"""
    print("\n== Probe: ffmpeg encode (YCbCr from known BGR patches, bgr24 -> yuv420p) ==")
    results = []
    for (w, h) in (HD, SD625):
        patches = np.array(list(SWATCHES.values()) + list(BARS.values()), np.uint8)
        frame = np.zeros((h, w, 3), np.uint8)
        pw = w // len(patches)
        rects = []
        for i, rgb in enumerate(patches):
            frame[:, i * pw:(i + 1) * pw] = rgb
            rects.append((i * pw + 8, 8, (i + 1) * pw - 8, h - 8))
        path = Path(workdir) / f"encode_{w}.mp4"
        run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{w}x{h}",
             "-pix_fmt", "bgr24", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "0",
             "-pix_fmt", "yuv420p", *extra, str(path)],
            input=frame[..., ::-1].tobytes() * 3)
        y, cb, cr = decode_yuv(path, w, h)
        measured = np.array([(y[0, y0:y1, x0:x1].mean(), cb[0, y0 // 2:y1 // 2, x0 // 2:x1 // 2].mean(),
                              cr[0, y0 // 2:y1 // 2, x0 // 2:x1 // 2].mean()) for x0, y0, x1, y1 in rects])
        errors = {(m, r): float(np.abs(rgb_to_yuv(patches, m, r) - measured).mean())
                  for m, r in itertools.product(COEFFICIENTS, ["tv", "pc"])}
        (matrix, r) = min(errors, key=errors.get)
        err = errors[(matrix, r)]
        stream = probe_stream(path)
        print(f"  {w}x{h}: {matrix}/{r}, err {err:.2f} code values; output tags "
              f"{stream.get('color_range', '-')}/{stream.get('color_space', '-')}/"
              f"{stream.get('color_primaries', '-')}/{stream.get('color_transfer', '-')}")
        results.append({"size": [w, h], "matrix": matrix, "range": r, "error": err})
    return results


def probe_passthrough(workdir):
    """Known YCbCr patches through the whole pipeline: the largest change in each
    plane, in 8-bit code values."""
    print("\n== Probe: YCbCr code values through the pipeline (output - source) ==")
    print("source tags -> max|dY|, max|dCb|, max|dCr| over the patches; output tags")
    rows = []
    for (w, h), space, rng in itertools.product([HD, SD625], ["bt709", "smpte170m", None], ["tv", "pc"]):
        source = Path(workdir) / f"pass_{w}_{space}_{rng}.mp4"
        make_probe_clip(source, w, h, space, rng, frames=12)
        out_dir = Path(workdir) / f"pass_out_{w}_{space}_{rng}"
        out_dir.mkdir()
        output = run_pipeline(source, out_dir, mask_box=(w - 14, h - 14, w - 2, h - 2))
        a = patch_means(decode_yuv(source, w, h), w, h)
        # The output is decoded as it is (yuv420p); a full-range stream is read
        # as its raw planes, so ranges compare like for like.
        b = patch_means(decode_yuv(output, w, h), w, h)
        d = np.abs(b - a).max(axis=0)
        stream = probe_stream(output)
        print(f"  {w}x{h} {space!s:10s} {rng}: dY {d[0]:5.2f}  dCb {d[1]:5.2f}  dCr {d[2]:5.2f}   "
              f"out {stream.get('color_range', '-')}/{stream.get('color_space', '-')}")
        rows.append({"size": [w, h], "space": space, "range": rng, "delta": d.round(2).tolist(),
                     "messages": list(PIPELINE_LOG)})
    return rows


# ---------------------------------------------------------------------------------


def install_sim601():
    """Make the pipeline read every source as a BT.601-only decoder would (the
    reference Mac's OpenCV 4.14.0: it ignores BT.709 and BT.2020 matrix tags and
    honours the range tag), with the ffmpeg reader standing in for it."""
    from videowipe import colour, decode
    from videowipe.inpainters import sttn
    from videowipe.tasks import base

    def open_capture(path, like=None):
        source = colour.probe_source(path)
        return decode.FfmpegFrameReader(path, "bt601", (source and source.range) or "tv")

    base.open_capture = sttn.open_capture = open_capture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--classes", help="comma-separated class names (default: all)")
    parser.add_argument("--frames", type=int, default=50, help="frames per class (default 50)")
    parser.add_argument("--content", default=str(ROOT / "input" / "detext_examples" / "english1.mp4"),
                        help="clip used for the real-content part of the picture")
    parser.add_argument("--probe", choices=["decode", "encode", "passthrough", "table", "all"],
                        default="all")
    parser.add_argument("--colour", help="set VIDEOWIPE_COLOUR for the run (e.g. off, auto)")
    parser.add_argument("--decode", help="set VIDEOWIPE_DECODE for the run (opencv, auto, ffmpeg), "
                        "or 'sim601': a decoder that reads every source as BT.601 with the "
                        "source's range, like the reference Mac's OpenCV")
    parser.add_argument("--json", help="write the results to this file")
    parser.add_argument("--keep", help="keep the generated files in this directory")
    args = parser.parse_args(argv)
    if args.colour:
        os.environ["VIDEOWIPE_COLOUR"] = args.colour
    if args.decode == "sim601":
        install_sim601()
    elif args.decode:
        os.environ["VIDEOWIPE_DECODE"] = args.decode
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        sys.exit("ffmpeg and ffprobe are required")
    import cv2
    first = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False).stdout.splitlines()[0]
    builds = [line.strip() for line in cv2.getBuildInformation().splitlines()
              if line.strip().startswith(("avcodec:", "swscale:", "FFMPEG:"))]
    print(f"OpenCV {cv2.__version__} ({'; '.join(builds)})")
    print(f"System {first}")
    print(f"VIDEOWIPE_COLOUR={os.environ.get('VIDEOWIPE_COLOUR', '(unset)')} "
          f"VIDEOWIPE_DECODE={args.decode or os.environ.get('VIDEOWIPE_DECODE', '(unset)')}")
    wanted = set(args.classes.split(",")) if args.classes else None
    results = {}
    holder = tempfile.TemporaryDirectory() if not args.keep else None
    workdir = args.keep or holder.name
    os.makedirs(workdir, exist_ok=True)
    try:
        if args.probe in ("decode", "all"):
            results["decode"] = probe_decode(workdir)
        if args.probe in ("encode", "all"):
            results["encode"] = probe_encode(workdir)
        if args.probe in ("passthrough", "all"):
            results["passthrough"] = probe_passthrough(workdir)
        if args.probe in ("table", "all"):
            rows = [class_row(spec, args.frames, args.content, workdir)
                    for spec in CLASSES if wanted is None or spec.name in wanted]
            print_table(rows)
            results["table"] = rows
    finally:
        if holder:
            holder.cleanup()
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
