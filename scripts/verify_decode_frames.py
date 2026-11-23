#!/usr/bin/env python3
"""Check, frame for frame, that the ffmpeg reader yields OpenCV's frames in OpenCV's order.

``videowipe.decode.FfmpegFrameReader`` replaces ``cv2.VideoCapture`` in the inpainting
path, which identifies frames by their sequential-read ordinal. This script reads a
source with both, in lockstep, and compares frame *i* of one with frame *i* of the
other. The readers' colour conversions differ by design, so the ffmpeg reader is
first run with the conversion this OpenCV measurably uses for the source (the
decode probe of ``videowipe.colour``); then the frames should agree to within the
rounding of two swscale versions, whatever the colours would be under the correct
matrix. Per file it reports

* the frame counts and sizes of both readers (they must be equal);
* the largest per-frame mean absolute difference and the largest single-pixel
  difference between OpenCV's frame *i* and the ffmpeg frame *i* (same conversion);
* ``misordered``: frames *i* for which a neighbouring frame of the other reader
  (*i-1*, *i+1*) is a better match than frame *i* itself (a frame dropped,
  duplicated or shifted shows up here, unless the picture really is static);
* with ``--correct`` also the difference to the ffmpeg reader's *correct* frame
  (the matrix the source calls for), which is the colour correction itself.

``--generate DIR`` first writes the variable-rate, WebM, edit-list, B-frame, rotated
and odd-sized sources of the test suite into DIR and checks those as well.

Usage::

    python scripts/verify_decode_frames.py input/detext_examples/*.mp4
    python scripts/verify_decode_frames.py --generate /tmp/decode-cases --correct
    python scripts/verify_decode_frames.py clip.webm --json result.json

Exit status 1 when any source's counts, sizes or order disagree.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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

#: mean absolute difference (8-bit code values) that counts as "the same frame":
#: swscale versions differ by a rounding step in some pixels
SAME = 0.75


def open_pair(path: str, conversion=None):
    """``(OpenCV capture, ffmpeg reader)`` of *path* converting alike (what this
    OpenCV measurably does for the source, unless *conversion* is given)."""
    source = colour.probe_source(path)
    if conversion is None:
        measured = None
        if source is not None:
            measured = colour.decoder_conversion(source.width, source.height, source.space,
                                                 source.primaries, source.transfer, source.range)
        conversion = measured or ("bt601", "tv")
    return cv2.VideoCapture(path), decode.FfmpegFrameReader(path, *conversion), conversion


def _mad(a, b) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def compare(path: str, max_frames: int | None = None, correct: bool = False) -> dict:
    """Read *path* with both readers in lockstep; the numbers described above."""
    theirs, ours, conversion = open_pair(path)
    right = None
    wanted = None
    if correct:
        source = colour.probe_source(path)
        wanted = decode.source_conversion(source) if source else None
        if wanted is not None:
            right = decode.FfmpegFrameReader(path, *wanted)
    out = {"path": str(path), "opencv_conversion": list(conversion), "frames": 0,
           "opencv_frames": 0, "ffmpeg_frames": 0, "shape_mismatch": 0, "max_mean_diff": 0.0,
           "max_pixel_diff": 0, "misordered": 0, "first_misordered": None,
           "ffmpeg_ends_early": False, "ffmpeg_has_extra": False}
    if correct:
        out["correct_conversion"] = list(wanted) if wanted else None
        out["max_mean_diff_correct"] = 0.0
        out["max_pixel_diff_correct"] = 0
    prev_t = prev_o = None
    try:
        if not theirs.isOpened():
            out["error"] = "OpenCV cannot open it"
            return out
        i = 0
        while max_frames is None or i < max_frames:
            ok_t, a = theirs.read()
            ok_o, b = ours.read()
            out["opencv_frames"] += bool(ok_t)
            out["ffmpeg_frames"] += bool(ok_o)
            if ok_t != ok_o:
                out["ffmpeg_ends_early" if ok_t else "ffmpeg_has_extra"] = True
                i += 1
                break
            if not ok_t:
                break
            if a.shape != b.shape:
                out["shape_mismatch"] += 1
            else:
                d = _mad(a, b)
                out["max_mean_diff"] = max(out["max_mean_diff"], d)
                out["max_pixel_diff"] = max(out["max_pixel_diff"],
                                            int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()))
                if prev_t is not None and prev_t.shape == a.shape:
                    # a neighbour is a clearly better match than the frame itself
                    # (they differ from each other, so this is not a static picture)
                    near = min(_mad(a, prev_o), _mad(prev_t, b))
                    if d > SAME and near < d - 0.25:
                        out["misordered"] += 1
                        out["first_misordered"] = out["first_misordered"] or i
                if right is not None:
                    ok_r, c = right.read()
                    if ok_r and c.shape == a.shape:
                        out["max_mean_diff_correct"] = max(out["max_mean_diff_correct"], _mad(a, c))
                        out["max_pixel_diff_correct"] = max(
                            out["max_pixel_diff_correct"],
                            int(np.abs(a.astype(np.int16) - c.astype(np.int16)).max()))
            prev_t, prev_o = a, b
            i += 1
        out["frames"] = i
        if max_frames is None and not out["ffmpeg_ends_early"] and not out["ffmpeg_has_extra"]:
            # both reached the end together; neither has a frame more
            ok_t, _ = theirs.read()
            ok_o, _ = ours.read()
            out["ffmpeg_has_extra"] = bool(ok_o and not ok_t)
            out["ffmpeg_ends_early"] = bool(ok_t and not ok_o)
    finally:
        theirs.release()
        ours.release()
        if right is not None:
            right.release()
    out["ok"] = (not out["shape_mismatch"] and not out["misordered"] and not out["ffmpeg_ends_early"]
                 and not out["ffmpeg_has_extra"] and out["opencv_frames"] == out["ffmpeg_frames"]
                 and out["max_mean_diff"] <= SAME * 4)
    return out


# -- generated sources ---------------------------------------------------------------------


def _ffmpeg(*args, timeout=600):
    result = _run(["ffmpeg", "-y", "-v", "error", *args], timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-800:])


def generate(folder: Path, frames: int = 90) -> dict[str, Path]:
    """The awkward sources, written to *folder*: ``{name: path}``."""
    folder.mkdir(parents=True, exist_ok=True)
    cases = {}
    content = f"testsrc2=size=320x240:rate=25,format=yuv420p"
    base = folder / "base.mp4"
    _ffmpeg("-f", "lavfi", "-i", content, "-frames:v", str(frames), "-c:v", "libx264", "-bf", "3",
            "-g", "30", "-crf", "12", "-pix_fmt", "yuv420p", str(base))
    cases["bframes"] = base
    # variable frame rate: irregular gaps and bursts in the timestamps
    vfr = folder / "vfr.mp4"
    _ffmpeg("-f", "lavfi", "-i", content, "-frames:v", str(frames), "-vf",
            "setpts='PTS+0.45/TB*gte(N,30)+0.2/TB*gte(N,31)-0.18/TB*gte(N,60)+sin(N)*0.1/TB'",
            "-fps_mode", "passthrough", "-c:v", "libx264", "-bf", "2", "-crf", "12",
            "-pix_fmt", "yuv420p", str(vfr))
    cases["vfr_mp4"] = vfr
    for name, codec in (("vfr_webm", ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "20"]),
                        ("vfr_vp8_webm", ["-c:v", "libvpx", "-b:v", "2M"])):
        path = folder / f"{name}.webm"
        try:
            _ffmpeg("-f", "lavfi", "-i", content, "-frames:v", str(frames), "-vf",
                    "setpts='PTS+0.3/TB*gte(N,40)+sin(N)*0.2/TB'", "-fps_mode", "passthrough",
                    *codec, "-pix_fmt", "yuv420p", str(path))
            cases[name] = path
        except RuntimeError:
            pass  # this ffmpeg has no such encoder
    # edit lists: a cut inside a GOP (pre-roll frames hidden by the edit list) and a
    # delayed start (an empty edit)
    cut = folder / "editlist_cut.mp4"
    _ffmpeg("-ss", "1.0", "-i", str(base), "-c", "copy", str(cut))
    cases["editlist_cut"] = cut
    delayed = folder / "editlist_delay.mp4"
    _ffmpeg("-itsoffset", "1.5", "-i", str(base), "-c", "copy", str(delayed))
    cases["editlist_delay"] = delayed
    cut_vfr = folder / "editlist_cut_vfr.mp4"
    _ffmpeg("-ss", "1.2", "-i", str(vfr), "-c", "copy", str(cut_vfr))
    cases["editlist_cut_vfr"] = cut_vfr
    # open-GOP HEVC in Matroska, and a rotated stream, and an odd size
    mkv = folder / "hevc_bframes.mkv"
    try:
        _ffmpeg("-f", "lavfi", "-i", content, "-frames:v", str(frames), "-c:v", "libx265",
                "-x265-params", "log-level=error:bframes=4", "-pix_fmt", "yuv420p", str(mkv))
        cases["hevc_mkv"] = mkv
    except RuntimeError:
        pass
    rotated = folder / "rotated.mp4"
    _ffmpeg("-display_rotation", "90", "-i", str(base), "-c", "copy", str(rotated))
    cases["rotated"] = rotated
    odd = folder / "odd_size.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=321x241:rate=25,format=yuv420p", "-frames:v", "40",
            "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-vf", "pad=322:242", str(odd))
    cases["odd_size"] = odd
    tagged = folder / "bt709_tagged.mp4"
    _ffmpeg("-i", str(base), "-c", "copy", "-bsf:v",
            "h264_metadata=matrix_coefficients=1:colour_primaries=1:transfer_characteristics=1",
            str(tagged))
    cases["bt709_tagged"] = tagged
    # The colour tags go on the frames (``setparams``, after any conversion) as well as
    # on the output: newer ffmpeg converts the pixels to colour options given on the
    # output alone, and writes only the tags the frames carry.
    for fmt in ("yuv422p", "yuv444p"):
        path = folder / f"{fmt}.mp4"
        _ffmpeg("-f", "lavfi", "-i", content, "-frames:v", "40", "-c:v", "libx264", "-crf", "12",
                "-pix_fmt", fmt, *colour._labelled((), "bt709", None, None, None, pix_fmt=fmt), str(path))
        cases[fmt] = path
    full = folder / "full_range.mp4"
    _ffmpeg("-f", "lavfi", "-i", content, "-frames:v", "40", "-c:v", "libx264", "-crf", "12",
            "-pix_fmt", "yuv420p",
            *colour._labelled(("scale=out_range=pc",), "bt709", None, None, "pc"), str(full))
    cases["full_range"] = full
    ten = folder / "yuv420p10.mp4"
    _ffmpeg("-f", "lavfi", "-i", content, "-frames:v", "40", "-c:v", "libx264", "-crf", "12",
            "-pix_fmt", "yuv420p10le",
            *colour._labelled((), "bt709", None, None, None, pix_fmt="yuv420p10le"), str(ten))
    cases["yuv420p10"] = ten
    return cases


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sources", nargs="*", help="video files to check")
    parser.add_argument("--generate", metavar="DIR", help="also write and check the awkward sources")
    parser.add_argument("--max-frames", type=int, help="stop after this many frames of each")
    parser.add_argument("--correct", action="store_true",
                        help="also report the difference to the correctly converted frames")
    parser.add_argument("--json", metavar="FILE", help="write the numbers here")
    args = parser.parse_args(argv)
    paths = [Path(p) for p in args.sources]
    if args.generate:
        paths += list(generate(Path(args.generate)).values())
    if not paths:
        parser.error("name a source or use --generate")
    results, failed = [], False
    for path in paths:
        result = compare(str(path), args.max_frames, args.correct)
        results.append(result)
        failed |= not result.get("ok", False)
        line = (f"{path.name:24s} frames opencv={result['opencv_frames']} ffmpeg={result['ffmpeg_frames']} "
                f"max mean|diff|={result['max_mean_diff']:.3f} max px={result['max_pixel_diff']} "
                f"misordered={result['misordered']} {'OK' if result.get('ok') else 'FAIL'}")
        if args.correct and "max_mean_diff_correct" in result:
            line += (f"  vs correct: mean {result['max_mean_diff_correct']:.2f} "
                     f"px {result['max_pixel_diff_correct']}")
        print(line)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
