"""Give the cleaned video the source's colour tags, with a conversion that matches.

The pipeline decodes with OpenCV to 8-bit BGR, pipes the frames to ffmpeg and
encodes ``libx264 -pix_fmt yuv420p``. Upstream sets no colour primaries, transfer,
matrix or range on the way in or out, so two things are left to chance:

* OpenCV converts YCbCr to BGR with a matrix and range of its own choosing (which
  depends on the OpenCV build: some honour the source's tags, others always use
  BT.601 limited range), and
* ffmpeg converts BGR back to YCbCr with swscale's default, BT.601 limited range,
  and the file says nothing about it, so players guess (BT.709 for HD, BT.601
  for SD).

Where those agree with each other and with the player's guess the picture is
right; where they do not (a tagged BT.709 HD source on an OpenCV that honours the
tags, a BT.601 or BT.2020 source of HD size, a full-range source, ...) every
colour moves, by up to dozens of 8-bit code values on saturated colours.
``docs/colour.md`` has the measurements.

This module makes the encode the inverse of what the decoder did, and says so:

1. The source's tags are read with ffprobe.
2. The matrix and range OpenCV *really* uses for a stream tagged like this one
   are measured on this machine (a small tagged clip of known colours decoded
   with ``cv2.VideoCapture``), not assumed.
3. ffmpeg is told to convert BGR to YCbCr with that matrix and range
   (``-vf scale=out_color_matrix=..:out_range=..``) and the output is tagged with
   the source's own tags. The YCbCr values of untouched pixels then go through
   unchanged, and the tags describe them truthfully. The tags are set on the
   frames (``setparams``) as well as on the output: newer ffmpeg (measured on
   9.0.2) treats colour options on the output as a request to convert the pixels
   to them, and writes only the tags the frames carry.
4. The result is checked by running ffmpeg on known colours first (the tags come
   out as asked and the YCbCr values are those of the chosen matrix and range).

Nothing changes for a source that has no colour tags (its output is exactly what
it was), nor for HDR sources (``videowipe.source_checks`` already warns about
those; they are not handled). Where any step fails the job runs as before and one
line says so. ``VIDEOWIPE_COLOUR`` selects the behaviour: ``auto`` (default) or
``off`` (upstream's command exactly).
"""

from __future__ import annotations

import functools
import itertools
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np

from videowipe.source_checks import HDR_TRANSFERS
from videowipe.timing import _run

ENV_VAR = "VIDEOWIPE_COLOUR"
MODES = ("auto", "off")

#: (Kr, Kb) of each matrix ffmpeg's swscale can convert with here. BT.470 B/G and
#: SMPTE 170M share BT.601's coefficients.
COEFFICIENTS = {
    "bt601": (0.299, 0.114),
    "bt709": (0.2126, 0.0722),
    "bt2020": (0.2627, 0.0593),
}
#: the matrix a ``color_space`` tag (as ffprobe prints it) means
TAG_MATRIX = {
    "bt709": "bt709",
    "smpte170m": "bt601",
    "bt470bg": "bt601",
    "bt2020nc": "bt2020",
}
_UNSET = (None, "", "unknown", "unspecified")
#: what swscale does when asked for nothing (and what upstream's command gets)
DEFAULT_CONVERSION = ("bt601", "tv")


def colour_mode() -> str:
    """``auto`` (default) or ``off``, from ``VIDEOWIPE_COLOUR``."""
    value = os.environ.get(ENV_VAR, "auto").strip().lower()
    return value if value in MODES else "auto"


# -- colour maths ------------------------------------------------------------------


def yuv_to_rgb(yuv, matrix: str, rng: str):
    """Float RGB in 0..255 (clipped) from 8-bit Y, Cb, Cr (last axis of *yuv*)."""
    kr, kb = COEFFICIENTS[matrix]
    kg = 1 - kr - kb
    y, cb, cr = (np.asarray(yuv, np.float64)[..., i] for i in range(3))
    if rng == "tv":
        y, cb, cr = (y - 16) / 219, (cb - 128) / 224, (cr - 128) / 224
    else:
        y, cb, cr = y / 255, (cb - 128) / 255, (cr - 128) / 255
    r = y + 2 * (1 - kr) * cr
    b = y + 2 * (1 - kb) * cb
    g = (y - kr * r - kb * b) / kg
    return np.clip(np.stack([r, g, b], axis=-1) * 255, 0, 255)


def rgb_to_yuv(rgb, matrix: str, rng: str):
    """Float 8-bit Y, Cb, Cr (unrounded) from RGB in 0..255."""
    kr, kb = COEFFICIENTS[matrix]
    kg = 1 - kr - kb
    r, g, b = (np.asarray(rgb, np.float64)[..., i] / 255 for i in range(3))
    y = kr * r + kg * g + kb * b
    cb = (b - y) / (2 * (1 - kb))
    cr = (r - y) / (2 * (1 - kr))
    if rng == "tv":
        out = (16 + 219 * y, 128 + 224 * cb, 128 + 224 * cr)
    else:
        out = (255 * y, 128 + 255 * cb, 128 + 255 * cr)
    return np.stack(out, axis=-1)


def best_hypothesis(errors: dict, limit: float = 1.6, margin: float = 0.5):
    """The ``(matrix, range)`` with the smallest error in *errors*, or ``None``
    when even that is above *limit* or the next best is not clearly worse."""
    ranked = sorted(errors, key=errors.get)
    best, second = errors[ranked[0]], errors[ranked[1]]
    return ranked[0] if best <= limit and second - best >= margin else None


#: YCbCr triplets inside the RGB cube under any of the matrices, so that a
#: conversion with the wrong one still gives distinct, unclipped colours
PROBE_YUV = [(60, 100, 160), (90, 170, 110), (120, 128, 128), (150, 90, 170), (180, 150, 100),
             (110, 160, 160), (140, 110, 100), (80, 128, 190), (200, 128, 128), (50, 128, 128),
             (130, 190, 128), (100, 128, 80), (160, 128, 170), (170, 100, 128), (70, 150, 140),
             (190, 140, 120)]
_PROBE_MARGIN = 16


def probe_patch_rects(width: int, height: int, columns: int = 4):
    """The interior ``(x0, y0, x1, y1)`` of each :data:`PROBE_YUV` patch."""
    rows = len(PROBE_YUV) // columns
    pw, ph = width // columns, height // rows
    return [((i % columns) * pw + _PROBE_MARGIN, (i // columns) * ph + _PROBE_MARGIN,
             (i % columns + 1) * pw - _PROBE_MARGIN, (i // columns + 1) * ph - _PROBE_MARGIN)
            for i in range(len(PROBE_YUV))]


def probe_yuv_frame(width: int, height: int):
    """One yuv420p frame (width and height even) of the :data:`PROBE_YUV` patches,
    as raw bytes: Y, then Cb, then Cr."""
    y = np.zeros((height, width), np.uint8)
    cb = np.zeros((height // 2, width // 2), np.uint8)
    cr = np.zeros((height // 2, width // 2), np.uint8)
    m = _PROBE_MARGIN
    for (yy, bb, rr), (x0, y0, x1, y1) in zip(PROBE_YUV, probe_patch_rects(width, height)):
        y[y0 - m:y1 + m, x0 - m:x1 + m] = yy
        cb[(y0 - m) // 2:(y1 + m) // 2, (x0 - m) // 2:(x1 + m) // 2] = bb
        cr[(y0 - m) // 2:(y1 + m) // 2, (x0 - m) // 2:(x1 + m) // 2] = rr
    return y.tobytes() + cb.tobytes() + cr.tobytes()


# -- the source ---------------------------------------------------------------------


@dataclass(frozen=True)
class SourceColour:
    """The colour description of a source's first video stream, as ffprobe says it
    (``None`` for what the file does not say)."""

    width: int
    height: int
    pix_fmt: str = ""
    space: str | None = None
    primaries: str | None = None
    transfer: str | None = None
    range: str | None = None  # "tv", "pc" or None

    @property
    def tagged(self) -> bool:
        return any(v is not None for v in (self.space, self.primaries, self.transfer, self.range))

    @property
    def hdr(self) -> bool:
        return self.transfer in HDR_TRANSFERS


def _tag(value):
    return None if value in _UNSET else str(value)


def parse_probe(text: str) -> SourceColour | None:
    """Build a :class:`SourceColour` from ``ffprobe -of json`` output, or ``None``."""
    try:
        stream = json.loads(text)["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    pix_fmt = str(stream.get("pix_fmt") or "")
    rng = stream.get("color_range")
    rng = rng if rng in ("tv", "pc") else ("pc" if pix_fmt.startswith("yuvj") else None)
    return SourceColour(width, height, pix_fmt, _tag(stream.get("color_space")),
                        _tag(stream.get("color_primaries")), _tag(stream.get("color_transfer")), rng)


def probe_source(path: str, timeout: float = 60) -> SourceColour | None:
    """Probe *path*; ``None`` on any failure (no ffprobe, error, no video stream)."""
    try:
        out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                    "stream=pix_fmt,width,height,color_range,color_space,color_primaries,"
                    "color_transfer", "-of", "json", path], timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_probe(out.stdout.decode("utf-8", "replace")) if out.returncode == 0 else None


# -- what the decoder does ---------------------------------------------------------


def _tag_options(space, primaries, transfer, rng):
    options = []
    for flag, value in (("-colorspace", space), ("-color_primaries", primaries),
                        ("-color_trc", transfer), ("-color_range", rng)):
        if value is not None:
            options += [flag, value]
    return options


def _frame_tags(space, primaries, transfer, rng) -> str:
    """A ``setparams`` filter that labels the frames with these tags without
    converting them ("" when there is nothing to label). Newer ffmpeg converts
    the pixels to match colour options given on the output unless the frames
    already carry them, and writes only what the frames carry."""
    fields = [f"{name}={value}" for name, value in (
        ("colorspace", space), ("color_primaries", primaries),
        ("color_trc", transfer), ("range", rng)) if value is not None]
    return "setparams=" + ":".join(fields) if fields else ""


def _labelled(filters, space, primaries, transfer, rng, pix_fmt="yuv420p"):
    """``-vf`` with *filters*, a conversion to *pix_fmt*, then the frame labels,
    followed by the same tags as output options (which older ffmpeg needs to
    write them). The labels must come after the conversion: on BGR frames they
    tell the conversion which matrix to use (BT.709 for a BT.709 label, on
    ffmpeg 6.1 as on 9), which is not the point; after it they only label."""
    labels = _frame_tags(space, primaries, transfer, rng)
    chain = [*filters, f"format={pix_fmt}", labels] if labels else list(filters)
    return (["-vf", ",".join(chain)] if chain else []) + _tag_options(space, primaries, transfer, rng)


@functools.lru_cache(maxsize=32)
def decoder_conversion(width: int, height: int, space, primaries, transfer, rng) -> tuple | None:
    """``(matrix, range)`` that ``cv2.VideoCapture`` converts to BGR with, for an
    H.264 stream of this size carrying these tags, measured by decoding a clip of
    known YCbCr values; ``None`` when it matches none of BT.601/BT.709/BT.2020 in
    either range (for instance when the decoder also converts between primaries),
    or anything fails."""
    import cv2

    # The source's size (even), but never smaller than the patches need.
    width, height = max(width, 256) // 2 * 2, max(height, 256) // 2 * 2
    # Plain H.264 High profile, nearly lossless: the same decoder path as an
    # ordinary source (lossless x264 would be High 4:4:4, which hardware
    # decoders skip).
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p",
           "-s", f"{width}x{height}", "-r", "25", "-i", "-", "-frames:v", "2",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "1", "-profile:v", "high",
           "-pix_fmt", "yuv420p", *_labelled((), space, primaries, transfer, rng)]
    raw = probe_yuv_frame(width, height)
    with tempfile.TemporaryDirectory(prefix="videowipe-colour-") as folder:
        path = os.path.join(folder, "probe.mp4")
        try:
            if _run(cmd + [path], input=raw * 2, timeout=120).returncode != 0:
                return None
            capture = cv2.VideoCapture(path)
            try:
                ok, bgr = capture.read()
            finally:
                capture.release()
        except (OSError, subprocess.SubprocessError, cv2.error):
            return None
    if not ok or bgr is None or bgr.shape[:2] != (height, width):
        return None
    rgb = bgr[..., ::-1].astype(np.float64)
    measured = np.array([rgb[y0:y1, x0:x1].mean(axis=(0, 1))
                         for x0, y0, x1, y1 in probe_patch_rects(width, height)])
    errors = {(m, r): float(np.abs(yuv_to_rgb(np.array(PROBE_YUV), m, r) - measured).mean())
              for m, r in itertools.product(COEFFICIENTS, ("tv", "pc"))}
    return best_hypothesis(errors)


# -- the plan ------------------------------------------------------------------------


def _scale_filters(matrix: str, rng: str):
    """The filter that makes ffmpeg convert BGR with *matrix* and *rng* (none for
    the conversion it makes anyway)."""
    if (matrix, rng) == DEFAULT_CONVERSION:
        return []
    return [f"scale=out_color_matrix={matrix}:out_range={rng}"]


#: the largest YCbCr error (8-bit code values) :func:`check_encode` accepts
ENCODE_TOLERANCE = 4.0


def check_encode(options, matrix: str, rng: str, expected_tags: dict) -> bool:
    """Run ffmpeg with *options* on known colours: true when it accepts them, the
    output carries *expected_tags* (ffprobe names) and its YCbCr values are those
    of *matrix* and *rng*."""
    colours = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128), (255, 255, 255),
                        (194, 150, 130)], np.uint8)
    width, height, stripe = 192, 64, 32
    frame = np.zeros((height, width, 3), np.uint8)
    for i, rgb in enumerate(colours):
        frame[:, i * stripe:(i + 1) * stripe] = rgb
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-vcodec", "rawvideo", "-s",
           f"{width}x{height}", "-pix_fmt", "bgr24", "-r", "25", "-i", "-",
           "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
           *options]
    with tempfile.TemporaryDirectory(prefix="videowipe-colour-") as folder:
        path = os.path.join(folder, "check.mp4")
        try:
            if _run(cmd + [path], input=frame[..., ::-1].tobytes() * 2, timeout=120).returncode != 0:
                return False
            info = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=color_range,color_space,color_primaries,color_transfer",
                         "-of", "json", path], timeout=60)
            planes = _run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1", "-f", "rawvideo", "-"],
                          timeout=60)
            if info.returncode != 0 or planes.returncode != 0:
                return False
            stream = json.loads(info.stdout)["streams"][0]
        except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError):
            return False
    seen = {"-colorspace": stream.get("color_space"), "-color_primaries": stream.get("color_primaries"),
            "-color_trc": stream.get("color_transfer"), "-color_range": stream.get("color_range")}
    if any(seen.get(flag) != value for flag, value in expected_tags.items()):
        return False
    data = np.frombuffer(planes.stdout, np.uint8)
    n = width * height
    if len(data) != n + n // 2:
        return False
    y = data[:n].reshape(height, width)
    cb = data[n:n + n // 4].reshape(height // 2, width // 2)
    cr = data[n + n // 4:].reshape(height // 2, width // 2)
    got = np.array([(y[16:48, i * stripe + 8:i * stripe + 24].mean(),
                     cb[8:24, i * stripe // 2 + 4:i * stripe // 2 + 12].mean(),
                     cr[8:24, i * stripe // 2 + 4:i * stripe // 2 + 12].mean())
                    for i in range(len(colours))])
    # Another swscale rounds chroma differently by a code value or two; the nearest wrong
    # matrix is 8 or more codes away on these colours (BT.709 against BT.2020 in Y).
    return float(np.abs(rgb_to_yuv(colours, matrix, rng) - got).max()) <= ENCODE_TOLERANCE


@dataclass
class OutputColour:
    """What one STTN job does about colour tags: ``options`` go to the first output
    (the H.264 MP4), ``second`` to a further output when the command has one (the
    optional HEVC copy). Empty ``options`` means the job is exactly upstream's."""

    options: tuple = ()
    second: tuple = ()
    description: str = ""
    #: false when the source has no tags to keep and only the conversion changes
    tagged: bool = True

    @property
    def active(self) -> bool:
        return bool(self.options)

    def rewrite(self, cmd):
        """*cmd* with the options added to its outputs, or ``None`` when *cmd* is
        not the shape this code knows how to rewrite."""
        try:
            codec = cmd.index("-c:v")
            if cmd[codec + 1] != "libx264":
                return None
            pix = cmd.index("-pix_fmt", codec)
            if cmd[pix + 1] != "yuv420p":
                return None
            movflags = cmd.index("-movflags", pix)
            if cmd[movflags + 1] != "+faststart":
                return None
        except (ValueError, IndexError):
            return None
        # Filters or colour options already present (anywhere before the first
        # output) mean someone else has taken charge of this part.
        before = cmd[:movflags]
        if any(flag in before for flag in ("-vf", "-filter:v", "-filter_complex", "-colorspace",
                                           "-color_primaries", "-color_trc", "-color_range", "-sws_flags")):
            return None
        end = movflags + 3  # "-movflags", "+faststart", <file>
        out = list(cmd[:movflags]) + list(self.options) + list(cmd[movflags:end])
        rest = list(cmd[end:])
        if rest and self.second:
            if "-vf" in rest or "-filter:v" in rest:
                return None
            rest[-1:-1] = list(self.second)  # before the further output's file name
        return out + rest

    def command(self, cmd):
        """*cmd* for this job: with the colour options when they apply (one line says
        so), else (with a warning when they should have) *cmd* itself."""
        if not self.active:
            return cmd
        rewritten = self.rewrite(cmd)
        if rewritten is None:
            print("Warning: could not keep the source's colour tags with this ffmpeg "
                  "command; the output carries none.")
            self.options = self.second = ()
            return cmd
        if self.tagged:
            print(f"Keeping the source's colour tags ({self.description}).")
        else:
            print(f"Encoding with the matrix the frames were decoded with ({self.description}).")
        return rewritten


def _describe(matrix: str, rng: str, source: SourceColour) -> str:
    names = {"bt601": "BT.601", "bt709": "BT.709", "bt2020": "BT.2020"}
    return f"{names[matrix]}, {'full' if rng == 'pc' else 'limited'} range; the source is " \
           f"{source.space or 'untagged'}"


#: after a conversion: frames in yuv420p carrying no colour labels at all
_UNLABELLED = ("format=yuv420p,setparams=colorspace=unknown:color_primaries=unknown:"
               "color_trc=unknown:range=unknown")


def _output_plan(source: SourceColour, matrix: str, rng: str, tagged: bool, write_tags: bool = True):
    """The ``(options, second, description)`` for frames whose BGR values are the
    source's true colours under *matrix* and *rng*, or ``None`` when this ffmpeg does
    not apply them as asked. With *tagged* false (a source with no tags at all) or
    *write_tags* false (an 8-bit copy of an HDR source: its tags would claim an HDR
    8-bit stream) nothing is written to the file and only the conversion is chosen."""
    tagged = tagged and write_tags
    tags = {"-colorspace": source.space, "-color_primaries": source.primaries,
            "-color_trc": source.transfer, "-color_range": rng if tagged else None}
    if not write_tags:
        tags = {}
    tags = {flag: value for flag, value in tags.items() if value is not None}
    options = _labelled(_scale_filters(matrix, rng), tags.get("-colorspace"),
                        tags.get("-color_primaries"), tags.get("-color_trc"),
                        tags.get("-color_range"))
    if options and not tags:
        # A source with no tags stays untagged. Newer ffmpeg's scale filter labels
        # what it converts (measured on 9.1: the output came out tagged bt709), so
        # the labels are reset after the conversion; older ffmpeg writes none anyway.
        options = ["-vf", options[1] + "," + _UNLABELLED, *options[2:]]
    if options and not check_encode(options, matrix, rng, tags):
        return None
    # The HEVC copy is limited range by design. Its conversion is changed only
    # where BGR is already the picture's true colours (the decoder read the tags
    # as a player would), so that converting once, to limited range, is right.
    second = ()
    if tagged and source.space is not None:
        second = tuple(["-vf", f"scale=out_color_matrix={matrix}:out_range=tv,"
                        + _frame_tags(source.space, source.primaries, source.transfer, "tv")]
                       + _tag_options(source.space, source.primaries, source.transfer, None))
    elif not tagged and (matrix, "tv") != DEFAULT_CONVERSION:
        second = ("-vf", f"scale=out_color_matrix={matrix}:out_range=tv")
    return tuple(options), second, _describe(matrix, rng, source)


@functools.lru_cache(maxsize=32)
def _reader_plan(source: SourceColour, matrix: str, rng: str):
    return _output_plan(source, matrix, rng, source.tagged, not source.hdr)


def reader_encode_plan(source: SourceColour, matrix: str, rng: str) -> OutputColour | None:
    """What the encode does for frames that :mod:`videowipe.decode`'s ffmpeg reader
    converted to BGR with *matrix* and *rng* (so BGR is the true picture), or
    ``None`` when this ffmpeg cannot apply it. Inactive (upstream's command) when
    the source has no tags and the matrix is ffmpeg's own default."""
    plan = _reader_plan(source, matrix, rng)
    if plan is None:
        return None
    options, second, description = plan
    return OutputColour(options, second, description, tagged=source.tagged and not source.hdr)


def plan_output_colour(job) -> OutputColour:
    """Decide how this job tags (and converts) its output's colours, and say so
    when it cannot be done."""
    path = getattr(job, "video_path", None)
    if colour_mode() == "off" or not path or not os.path.isfile(path):
        return OutputColour()
    source = probe_source(path)
    if source is None:
        print("Warning: could not read the source's colour tags; the output carries none.")
        return OutputColour()
    from videowipe import decode  # lazy: decode imports this module

    by_reader = decode.reader_conversion(getattr(job, "reader", None))
    if by_reader is not None:
        # The frames were decoded by ffmpeg with a matrix and range we know, so the
        # encode needs no measuring: it is the inverse of that.
        plan = reader_encode_plan(source, *by_reader)
        if plan is not None:
            from videowipe import highbit  # lazy: highbit imports this module

            kept = highbit.current(path)
            if kept is not None and kept.active:
                plan.second = ()  # the 10-bit output has its own options (highbit.py)
        if plan is None:
            print("Warning: this ffmpeg did not apply the matrix the frames were decoded "
                  "with; the output's colours are converted with ffmpeg's default.")
            return OutputColour()
        return plan
    if source.hdr or not source.tagged:
        return OutputColour()  # HDR is not handled; no tags means nothing to keep
    if source.space is not None and source.space not in TAG_MATRIX:
        print(f"Warning: the source's colour matrix ({source.space}) is not one this code "
              "converts with; the output carries no colour tags.")
        return OutputColour()
    decoded = decoder_conversion(source.width, source.height, source.space, source.primaries,
                                 source.transfer, source.range)
    if decoded is None:
        print("Warning: could not tell how this OpenCV converts the source's colours; the "
              "output carries no colour tags.")
        return OutputColour()
    matrix, rng = decoded
    if rng != (source.range or "tv"):
        # BGR then holds the wrong levels, and no tag on the output would describe them.
        print("Warning: this OpenCV read the source's colour range differently from its tag; "
              "the output carries no colour tags.")
        return OutputColour()
    plan = _output_plan(source, matrix, rng, True)
    if plan is None:
        print("Warning: this ffmpeg did not apply the source's colour tags with a matching "
              "conversion; the output carries none.")
        return OutputColour()
    options, second, description = plan
    # The HEVC copy's conversion changes only where the decoder read the tags as a
    # player would (see _output_plan).
    if (matrix, rng) != (TAG_MATRIX.get(source.space), source.range or "tv"):
        second = ()
    return OutputColour(options, second, description)
