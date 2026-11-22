#!/usr/bin/env python3
"""Measure what decoding a BT.709 picture with the wrong matrix costs, and what the
ffmpeg reader (``videowipe.decode``) gets back.

A picture of known colour (every colour of an 8 x 8 x 8 grid of the RGB cube, as flat
32-pixel patches, so 4:2:0 chroma does not blur their interiors) is converted to YCbCr
the way a BT.709 HD source is made, stored losslessly (untagged, and tagged BT.709),
and decoded three ways:

``opencv``
    ``cv2.VideoCapture`` on *this* machine.
``ffmpeg``
    ``videowipe.decode.FfmpegFrameReader`` with the matrix and range a player uses
    (the tags, else BT.709 for HD).
``bt601-only``
    A decoder that ignores matrix tags and always uses BT.601 limited range (what
    the reference Mac's OpenCV 4.14.0 does, measured in ``docs/colour.md``),
    simulated by running the same reader with BT.601.

Per decoder: the mean and largest absolute error per channel in 8-bit code values over
the grid, the number of patches that are wrong by more than 8 code values and, for the
BT.601-only reading, how many colours lie outside the RGB cube before the decoder clips
them and by how much (the part no later step can restore). With ``--content FILE...`` the same is reported for real
frames (the correct decode is the reference; the share of pixels that differ by more
than 4 and 8 code values, and the largest difference).

Usage::

    python scripts/decode_clipping.py
    python scripts/decode_clipping.py --content input/detext_examples/others.mp4 --frames 3
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    import videowipe  # noqa: F401
except ImportError:  # run from a checkout that is not installed
    sys.path.insert(0, str(ROOT / "src"))
from videowipe import colour, decode
from videowipe.timing import _run

WIDTH, HEIGHT, PATCH = 1280, 720, 32
LEVELS = np.round(np.linspace(0, 255, 8)).astype(int)


def grid_colours() -> np.ndarray:
    """All 512 colours of the 8 x 8 x 8 grid, RGB, shape (512, 3)."""
    return np.array([(r, g, b) for r in LEVELS for g in LEVELS for b in LEVELS], np.float64)


def patch_rects(count: int):
    columns = WIDTH // PATCH
    for i in range(count):
        x, y = (i % columns) * PATCH, (i // columns) * PATCH
        yield x + 8, y + 8, x + PATCH - 8, y + PATCH - 8


def render_yuv420(colours, matrix: str, rng: str = "tv"):
    """Raw yuv420p bytes of one frame of flat patches of *colours* (RGB), converted
    with *matrix* (rounded to 8 bits, as an encoder's input is)."""
    yuv = np.rint(colour.rgb_to_yuv(colours, matrix, rng)).clip(0, 255).astype(np.uint8)
    y = np.full((HEIGHT, WIDTH), 16, np.uint8)
    cb = np.full((HEIGHT // 2, WIDTH // 2), 128, np.uint8)
    cr = np.full((HEIGHT // 2, WIDTH // 2), 128, np.uint8)
    for (v_y, v_cb, v_cr), (x0, y0, x1, y1) in zip(yuv, patch_rects(len(yuv))):
        y[y0 - 8:y1 + 8, x0 - 8:x1 + 8] = v_y
        cb[(y0 - 8) // 2:(y1 + 8) // 2, (x0 - 8) // 2:(x1 + 8) // 2] = v_cb
        cr[(y0 - 8) // 2:(y1 + 8) // 2, (x0 - 8) // 2:(x1 + 8) // 2] = v_cr
    return y.tobytes() + cb.tobytes() + cr.tobytes(), yuv


def write_source(path: Path, raw: bytes, tags: bool) -> None:
    """A lossless H.264 file of *raw*, tagged BT.709 limited range when *tags*."""
    options = colour._labelled((), "bt709", "bt709", "bt709", "tv") if tags else []
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s",
           f"{WIDTH}x{HEIGHT}", "-r", "25", "-i", "-", "-frames:v", "2", "-c:v", "libx264",
           "-qp", "0", "-pix_fmt", "yuv420p", *options, str(path)]
    result = _run(cmd, input=raw * 2, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-500:])


def read_first(reader) -> np.ndarray:
    try:
        ok, frame = reader.read()
    finally:
        reader.release()
    if not ok:
        raise RuntimeError("could not decode the first frame")
    return frame


def patch_means(frame, count: int) -> np.ndarray:
    """Mean RGB of each patch interior of a BGR frame."""
    rgb = frame[..., ::-1].astype(np.float64)
    return np.array([rgb[y0:y1, x0:x1].mean(axis=(0, 1)) for x0, y0, x1, y1 in patch_rects(count)])


def score(measured, truth, unclipped=None) -> dict:
    error = np.abs(measured - truth)
    wrong = error.max(axis=1) > 8
    out = {"mean_err_rgb": [round(float(v), 2) for v in error.mean(axis=0)],
           "max_err_rgb": [round(float(v), 1) for v in error.max(axis=0)],
           "patches_wrong_over_8": int(wrong.sum())}
    if unclipped is not None:
        outside = np.maximum(-unclipped, unclipped - 255).clip(0)
        out["patches_clipped"] = int((outside.max(axis=1) > 0.5).sum())
        out["max_overshoot"] = round(float(outside.max()), 1)
    return out


def measure_grid(workdir: Path) -> dict:
    colours = grid_colours()
    raw, yuv = render_yuv420(colours, "bt709")
    results = {}
    for name, tags in (("untagged HD", False), ("BT.709 tagged HD", True)):
        path = workdir / (name.replace(" ", "_") + ".mp4")
        write_source(path, raw, tags)
        wanted = decode.source_conversion(colour.probe_source(str(path)))
        # What the source's YCbCr is, decoded without clipping, by each matrix
        row = {"correct matrix": list(wanted)}
        for label, matrix in (("float bt709", "bt709"), ("float bt601", "bt601")):
            lo = colour.COEFFICIENTS[matrix]
            kr, kb = lo
            kg = 1 - kr - kb
            wide = yuv.astype(np.float64)
            y, cb, cr = ((wide[:, 0] - 16) / 219, (wide[:, 1] - 128) / 224, (wide[:, 2] - 128) / 224)
            r = y + 2 * (1 - kr) * cr
            b = y + 2 * (1 - kb) * cb
            g = (y - kr * r - kb * b) / kg
            row[label] = np.stack([r, g, b], axis=-1) * 255
        row["opencv"] = score(patch_means(read_first(cv2.VideoCapture(str(path))), len(colours)), colours)
        row["ffmpeg"] = score(patch_means(read_first(decode.FfmpegFrameReader(str(path), *wanted)),
                                          len(colours)), colours)
        row["bt601-only"] = score(
            patch_means(read_first(decode.FfmpegFrameReader(str(path), "bt601", "tv")), len(colours)),
            colours, unclipped=row["float bt601"])
        row["float bt601"] = score(np.clip(row["float bt601"], 0, 255), colours,
                                   unclipped=row["float bt601"])
        del row["float bt709"]
        results[name] = row
    return results


def measure_content(path: str, frames: int) -> dict:
    """Real frames: the correct decode against OpenCV's and a BT.601-only decoder's."""
    source = colour.probe_source(path)
    wanted = decode.source_conversion(source)
    if wanted is None:
        return {"error": "matrix tag not handled"}
    out = {"correct": list(wanted), "frames": frames}
    for name, make in (("opencv", lambda: cv2.VideoCapture(path)),
                       ("bt601-only", lambda: decode.FfmpegFrameReader(path, "bt601", wanted[1]))):
        correct = decode.FfmpegFrameReader(path, *wanted)
        other = make()
        diffs = []
        try:
            for _ in range(frames):
                ok_a, a = correct.read()
                ok_b, b = other.read()
                if not (ok_a and ok_b):
                    break
                diffs.append(np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2))
        finally:
            correct.release()
            other.release()
        d = np.concatenate([x.ravel() for x in diffs]) if diffs else np.zeros(1)
        out[name] = {"mean_abs": round(float(d.mean()), 3), "share_over_4": round(float((d > 4).mean()), 5),
                     "share_over_8": round(float((d > 8).mean()), 5), "max": int(d.max())}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--content", nargs="*", default=[], metavar="FILE", help="also measure real video")
    parser.add_argument("--frames", type=int, default=3, help="frames of each content file")
    parser.add_argument("--json", metavar="FILE")
    args = parser.parse_args(argv)
    import subprocess
    first = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    print(f"OpenCV {cv2.__version__}; {first}")
    with tempfile.TemporaryDirectory(prefix="videowipe-clip-") as folder:
        grid = measure_grid(Path(folder))
    for name, row in grid.items():
        print(f"\n{name} (decoded as {row['correct matrix'][0]} {row['correct matrix'][1]} by a player)")
        for label in ("opencv", "ffmpeg", "bt601-only"):
            print(f"  {label:12s} {row[label]}")
    content = {}
    for path in args.content:
        content[path] = measure_content(path, args.frames)
        print(f"\n{path}: {content[path]}")
    if args.json:
        Path(args.json).write_text(json.dumps({"grid": grid, "content": content}, indent=2, default=list))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
