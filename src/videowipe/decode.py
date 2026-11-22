"""Decode the source's frames with the colour matrix and range the source was made with.

OpenCV converts the source's YCbCr to BGR with a matrix of its own choosing. The
OpenCV measured on the reference Mac (4.14.0, avcodec 61.19) uses BT.601 for every
source: it ignores BT.709 and BT.2020 matrix tags (it honours the full-range tag),
and it treats an untagged HD source as BT.601 although players read it as BT.709.
A BT.709 picture decoded as BT.601 has saturated colours pushed outside the RGB
cube and clipped (up to 39 code values of green lost), and clipped values cannot
be restored by any later step. ``docs/decode.md`` has the measurements.

This module provides :class:`FfmpegFrameReader`, a drop-in for the
``cv2.VideoCapture`` the inpainting path reads from (``read``, ``grab``,
``retrieve``, ``get``, ``release``): ffmpeg decodes the source and converts it to
``bgr24`` with the source's own matrix and range (its tags, or the BT.709 / BT.601
size convention that players use when there are none), and the frames come down a
pipe. Frames are identified by their sequential-read ordinal, exactly as with
OpenCV: the reader yields one frame per decoded frame, in decode order, with no
timestamp-driven duplication or dropping (``-fps_mode passthrough``), and never
seeks. ``scripts/verify_decode_frames.py`` checks that, frame for frame, against
OpenCV on any source.

``VIDEOWIPE_DECODE`` selects the behaviour:

``auto`` (default)
    Measure what this OpenCV does for a stream tagged like the source (the same
    probe :mod:`videowipe.colour` already uses). When it converts with the correct
    matrix and range, OpenCV keeps decoding, so nothing changes. When it does not,
    use the ffmpeg reader.
``ffmpeg``
    Always use the ffmpeg reader where it can be used.
``opencv``
    Upstream's decoder.

Wherever the ffmpeg reader cannot be used (HDR, a matrix this code does not
convert, a pixel format that is not YCbCr, an ffmpeg that does not decode known
colours as asked, frames that do not match OpenCV's, ``VIDEOWIPE_COLOUR=off``)
the job decodes with OpenCV and one line says so. :mod:`videowipe.colour` reads
the matrix and range from the reader itself, so the encode is the inverse of the
decode that really happened.

Detection and planning keep reading with OpenCV (``docs/decode.md`` says why).
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
import tempfile
import threading
import weakref

import cv2
import numpy as np

from videowipe import colour
from videowipe.colour import TAG_MATRIX
from videowipe.source_checks import bit_depth
from videowipe.timing import _run

ENV_VAR = "VIDEOWIPE_DECODE"
MODES = ("auto", "opencv", "ffmpeg")

_REAL_CAPTURE = cv2.VideoCapture
#: pixel-format name prefixes of YCbCr formats swscale converts with a matrix
_YCBCR_PREFIXES = ("yuv", "nv", "p0", "p2")
#: how many leading frames :func:`_matches_opencv` compares
_PREFIX_FRAMES = 6


def decode_mode() -> str:
    """``auto`` (default), ``ffmpeg`` or ``opencv``, from ``VIDEOWIPE_DECODE``."""
    value = os.environ.get(ENV_VAR, "auto").strip().lower()
    return value if value in MODES else "auto"


def conventional_matrix(width: int, height: int) -> str:
    """The matrix players assume for a stream with no matrix tag."""
    return "bt709" if width > 1024 or height > 576 else "bt601"


def source_conversion(source) -> tuple[str, str] | None:
    """``(matrix, range)`` a correct player converts *source* with: its tags, else
    the size convention (BT.709 above 1024 wide or 576 high, else BT.601) and
    limited range. ``None`` for a matrix tag this code does not convert with."""
    if source.space is None:
        matrix = conventional_matrix(source.width, source.height)
    else:
        matrix = TAG_MATRIX.get(source.space)
        if matrix is None:
            return None
    return matrix, source.range or "tv"


# -- the reader --------------------------------------------------------------------


def reader_command(path: str, matrix: str, rng: str, sync=("-fps_mode", "passthrough")) -> list[str]:
    """The ffmpeg command that writes the first video stream of *path* to stdout as
    ``bgr24``, converted with *matrix* and *rng* (which override the stream's own
    tags), one frame per decoded frame."""
    return ["ffmpeg", "-nostdin", "-v", "error", "-nostats", "-i", path,
            "-map", "0:v:0", "-an", "-sn", "-dn", *sync,
            "-vf", f"scale=in_color_matrix={matrix}:in_range={rng},format=bgr24",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]


_announced: set[str] = set()


def _stop(proc) -> None:
    """Close a reader's ffmpeg process (safe to call twice)."""
    try:
        proc.stdout.close()
    except OSError:
        pass
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class FfmpegFrameReader:
    """Stands in for the ``cv2.VideoCapture`` of the inpainting path, with frames
    from ffmpeg (see the module doc).

    ``read``, ``grab``, ``retrieve`` and ``release`` use the ffmpeg process, which
    starts on the first frame requested; ``isOpened`` and ``get(CAP_PROP_...)`` are
    a real capture's, opened on the same file for its properties. ``set`` is not
    supported (callers never seek). (It cannot be a subclass of
    ``cv2.VideoCapture``: Python subclasses of OpenCV's bound classes corrupt the
    heap when freed.)
    """

    #: ``isinstance(reader, CAPTURE_TYPES)`` is how the inpainter tells a real
    #: reader from a test double

    def __init__(self, path: str, matrix: str, rng: str, sync=("-fps_mode", "passthrough"),
                 note: str = ""):
        self._capture = cv2.VideoCapture(path)
        self.path, self.matrix, self.range, self.sync, self.note = path, matrix, rng, sync, note
        self._proc = None
        self._lock = threading.Lock()
        self._shape = None
        self._buffer = None
        self.frames_delivered = 0

    # -- process ----------------------------------------------------------------

    def _start(self) -> bool:
        width = int(self.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(self.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        if width <= 0 or height <= 0:
            return False
        self._shape = (height, width, 3)
        self._stderr = tempfile.TemporaryFile()
        try:
            self._proc = subprocess.Popen(
                reader_command(self.path, self.matrix, self.range, self.sync),
                stdout=subprocess.PIPE, stderr=self._stderr, bufsize=0)
        except OSError:
            self._proc = None
            return False
        # Whoever forgets release() must not leave ffmpeg running (not __del__: the
        # OpenCV base class does not survive being touched at interpreter exit).
        weakref.finalize(self, _stop, self._proc)
        try:  # a big pipe lets ffmpeg decode ahead of the consumer (Linux only)
            import fcntl
            fcntl.fcntl(self._proc.stdout.fileno(), getattr(fcntl, "F_SETPIPE_SZ", 1031), 1 << 21)
        except (ImportError, OSError, ValueError):
            pass
        if self.note and self.path not in _announced:
            _announced.add(self.path)
            print(self.note)
        return True

    def _fill(self, buffer: np.ndarray) -> bool:
        if self._proc is None and (self._shape is not None or not self._start()):
            return False
        view = memoryview(buffer).cast("B")
        done = 0
        while done < len(view):
            got = self._proc.stdout.readinto(view[done:])
            if not got:
                return False  # end of stream (or ffmpeg stopped); a partial frame is dropped
            done += got
        return True

    # -- the VideoCapture interface ---------------------------------------------------

    def read(self, image=None):
        with self._lock:
            if self._shape is None and not self._start():
                return False, None
            frame = np.empty(self._shape, np.uint8)
            if not self._fill(frame):
                return False, None
            self.frames_delivered += 1
            return True, frame

    def grab(self) -> bool:
        with self._lock:
            if self._shape is None and not self._start():
                return False
            if self._buffer is None:
                self._buffer = np.empty(self._shape, np.uint8)
            if not self._fill(self._buffer):
                self._grabbed = False
                return False
            self.frames_delivered += 1
            self._grabbed = True
            return True

    def retrieve(self, image=None, flag=0):
        if not getattr(self, "_grabbed", False):
            return False, None
        return True, self._buffer.copy()

    def isOpened(self) -> bool:
        return self._capture.isOpened()

    def get(self, prop):
        return self._capture.get(prop)

    def set(self, prop, value):
        return False

    def release(self):
        with self._lock:
            proc, self._proc = self._proc, None
            self._shape = self._shape or ()
            if proc is not None:
                _stop(proc)
            stderr = getattr(self, "_stderr", None)
            if stderr is not None:
                stderr.close()
                self._stderr = None
        self._capture.release()



#: what counts as a real frame reader (``cv2.VideoCapture`` and its stand-in)
CAPTURE_TYPES = (cv2.VideoCapture, FfmpegFrameReader)


def reader_conversion(reader) -> tuple[str, str] | None:
    """``(matrix, range)`` the frames of *reader* were converted to BGR with, when
    it is an :class:`FfmpegFrameReader`; ``None`` for anything else (OpenCV's own
    conversion, which :mod:`videowipe.colour` measures)."""
    if isinstance(reader, FfmpegFrameReader):
        return reader.matrix, reader.range
    return None


# -- the decision ------------------------------------------------------------------


def _bgr_of(patches_yuv, matrix, rng):
    return colour.yuv_to_rgb(np.array(patches_yuv), matrix, rng)


def _reader_decodes_known_colours(matrix: str, rng: str, sync) -> bool:
    """Run the reader on a clip of known YCbCr patches, tagged with a *different*
    matrix and range so the override is tested too: its BGR must be what *matrix*
    and *rng* give."""
    width = height = 256
    wrong_space = "bt709" if matrix == "bt601" else "smpte170m"
    wrong_range = "tv" if rng == "pc" else "pc"
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s",
           f"{width}x{height}", "-r", "25", "-i", "-", "-frames:v", "2", "-c:v", "libx264",
           "-preset", "ultrafast", "-crf", "1", "-profile:v", "high", "-pix_fmt", "yuv420p",
           *colour._labelled((), wrong_space, None, None, wrong_range)]
    raw = colour.probe_yuv_frame(width, height)
    with tempfile.TemporaryDirectory(prefix="videowipe-decode-") as folder:
        path = os.path.join(folder, "probe.mp4")
        try:
            if _run(cmd + [path], input=raw * 2, timeout=120).returncode != 0:
                return False
            reader = FfmpegFrameReader(path, matrix, rng, sync)
            try:
                ok, bgr = reader.read()
            finally:
                reader.release()
        except (OSError, subprocess.SubprocessError, cv2.error):
            return False
    if not ok or bgr is None or bgr.shape[:2] != (height, width):
        return False
    rgb = bgr[..., ::-1].astype(np.float64)
    measured = np.array([rgb[y0:y1, x0:x1].mean(axis=(0, 1))
                         for x0, y0, x1, y1 in colour.probe_patch_rects(width, height)])
    return float(np.abs(_bgr_of(colour.PROBE_YUV, matrix, rng) - measured).mean()) <= 1.6


def _small_grey(frame) -> np.ndarray:
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float64)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation of two pictures (Pearson on the dense ranks of their pixels), so
    that any monotone change of tone, which is what two decoders' different colour
    conversions amount to, leaves it high; 1.0 when either is flat (it cannot tell)."""
    ra = np.unique(a, return_inverse=True)[1].ravel().astype(np.float64)
    rb = np.unique(b, return_inverse=True)[1].ravel().astype(np.float64)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denominator = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return 1.0 if denominator < 1e-6 else float((ra * rb).sum() / denominator)


def _matches_opencv(path: str, matrix: str, rng: str, sync) -> str | None:
    """``None`` when the ffmpeg reader's first frames are OpenCV's first frames, in
    the same order and size; else a short reason. A smoke test (the first frames
    are where timestamp offsets and edit lists show); the full per-frame
    comparison is ``scripts/verify_decode_frames.py``."""
    theirs = _REAL_CAPTURE(path)
    ours = FfmpegFrameReader(path, matrix, rng, sync)
    try:
        if not theirs.isOpened():
            return "OpenCV cannot open it"
        a, b = [], []
        for _ in range(_PREFIX_FRAMES):
            ok1, x = theirs.read()
            ok2, y = ours.read()
            if ok1 != ok2:
                return "the frame counts differ"
            if not ok1:
                break
            if x.shape != y.shape:
                return f"the frame sizes differ ({x.shape[1]}x{x.shape[0]} against {y.shape[1]}x{y.shape[0]})"
            a.append(_small_grey(x))
            b.append(_small_grey(y))
    finally:
        theirs.release()
        ours.release()
    if not a:
        return "no frames"
    for i, frame in enumerate(b):
        # the same picture (their colours may differ), and no other frame of the prefix
        # is a clearly better match
        scores = [_correlation(frame, other) for other in a]
        if scores[i] < 0.8 or scores[i] < max(scores) - 0.01:
            return f"frame {i} does not correspond"
    return None


@functools.lru_cache(maxsize=32)
def _decide(path: str, size: int, mtime_ns: int, mode: str, colour_mode: str):
    """The ``(matrix, range, sync, note)`` to decode *path* with ffmpeg, or ``None``.
    One line is printed when ffmpeg was wanted but cannot be used."""
    if mode == "opencv":
        return None
    forced = mode == "ffmpeg"

    def refuse(why: str, quiet: bool = False):
        if not quiet:
            print(f"Warning: {why}; decoding with OpenCV.")
        return None

    if colour_mode == "off":
        # The encode is then ffmpeg's default (BT.601) whatever the decode did.
        return refuse("VIDEOWIPE_DECODE needs VIDEOWIPE_COLOUR=auto, which is off", quiet=not forced)
    source = colour.probe_source(path)
    if source is None:
        return refuse("could not read the source's colour description")
    if source.hdr:
        return refuse("the source is HDR, which the ffmpeg reader does not handle", quiet=not forced)
    if not source.pix_fmt.startswith(_YCBCR_PREFIXES):
        return refuse(f"the source's pixel format ({source.pix_fmt or 'unknown'}) is not YCbCr",
                      quiet=not forced)
    wanted = source_conversion(source)
    if wanted is None:
        return refuse(f"the source's colour matrix ({source.space}) is not one the ffmpeg reader "
                      "converts with")
    matrix, rng = wanted
    seen = ""
    if not forced and bit_depth(source.pix_fmt) > 8:
        # The 8-bit probe says nothing about OpenCV's high-bit-depth path, which
        # measurably differs from the correct conversion by more than the matrix
        # (docs/decode.md): decode such sources with ffmpeg.
        seen = "this OpenCV's conversion of high-bit-depth sources is not exact"
    elif not forced:
        measured = colour.decoder_conversion(source.width, source.height, source.space,
                                             source.primaries, source.transfer, source.range)
        if measured == wanted:
            return None  # OpenCV already reads this source correctly: nothing to change
        seen = "this OpenCV reads the source as " + (
            "an unrecognised conversion" if measured is None else
            colour._describe(*measured, source).split(";")[0])
    for sync in (("-fps_mode", "passthrough"), ("-vsync", "passthrough")):
        if _reader_decodes_known_colours(matrix, rng, sync):
            break
    else:
        return refuse("this ffmpeg did not decode known colours as asked")
    if colour.reader_encode_plan(source, matrix, rng) is None:
        return refuse("this ffmpeg did not encode with the matrix the frames were decoded with")
    why = _matches_opencv(path, matrix, rng, sync)
    if why is not None:
        return refuse(f"the ffmpeg reader's frames do not match OpenCV's ({why})")
    note = (f"Decoding with ffmpeg ({colour._describe(matrix, rng, source).split(';')[0]}); "
            + ("VIDEOWIPE_DECODE=ffmpeg." if forced else f"{seen}."))
    return matrix, rng, sync, note


def decide(video_path: str):
    """``(matrix, range, sync, note)`` for decoding *video_path* with ffmpeg, or
    ``None`` to decode with OpenCV (cached per file; fallbacks print one line)."""
    mode = decode_mode()
    if mode == "opencv" or not video_path or not os.path.isfile(video_path):
        return None
    if cv2.VideoCapture is not _REAL_CAPTURE:  # a test double stands in for decoding
        return None
    try:
        stat = os.stat(video_path)
    except OSError:
        return None
    return _decide(os.path.realpath(video_path), stat.st_size, stat.st_mtime_ns, mode,
                   colour.colour_mode())


def open_capture(video_path: str, like=None):
    """A frame reader for *video_path*.

    With no *like*: an :class:`FfmpegFrameReader` when :func:`decide` says so, else a
    plain ``cv2.VideoCapture`` (upstream's). With *like* (the reader a job already
    uses for the same file): a reader that converts exactly as *like* does, so two
    readers of one job never disagree about the colours.
    """
    if like is not None:
        if isinstance(like, FfmpegFrameReader):
            reader = FfmpegFrameReader(video_path, like.matrix, like.range, like.sync)
            if reader.isOpened():
                return reader
            reader.release()
        return cv2.VideoCapture(video_path)
    plan = decide(video_path)
    if plan is None:
        return cv2.VideoCapture(video_path)
    matrix, rng, sync, note = plan
    reader = FfmpegFrameReader(video_path, matrix, rng, sync, note)
    if not reader.isOpened():
        reader.release()
        return cv2.VideoCapture(video_path)
    return reader


def _main(argv=None) -> int:  # pragma: no cover - a convenience for people
    """``python -m videowipe.decode FILE``: say how FILE would be decoded."""
    argv = sys.argv[1:] if argv is None else argv
    for path in argv:
        plan = decide(path)
        print(f"{path}: " + ("OpenCV" if plan is None else plan[3]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
