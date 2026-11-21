"""Keep the source's frame timestamps in the cleaned video (variable-rate sources).

Upstream pipes raw ``bgr24`` frames to ffmpeg with ``-r <fps>``, so every output
frame is placed on a constant-rate grid while the audio, taken from the source
file, keeps its real timing. For a constant-rate source that is exact. For a
variable-rate source (screen recordings, phone video, many web downloads) the
picture drifts away from the sound by up to a few frames, and by more whenever
the source has a long gap.

This module decides, per job, whether the source's own timestamps are needed
and usable, and if so lets the existing ffmpeg pipeline carry them:

* The source is probed with ffprobe (packet timestamps only; nothing is decoded).
* A source whose frames sit on a regular grid is *constant*. Nothing changes for
  it: the ffmpeg arguments and the bytes written to stdin are exactly upstream's.
* A variable source whose packet count equals the job's frame count gets its
  timestamps back. Instead of ``rawvideo`` the frames are wrapped in a minimal
  Matroska stream (one cluster per frame, each stamped with the source frame's
  time) and read with ``-f matroska``; the MP4 is written without retiming.
* Anything else (ffprobe failed, the counts differ, this ffmpeg cannot do it)
  falls back to upstream's behaviour and says so in one line.

``VIDEOWIPE_SOURCE_TIMING`` selects the behaviour: ``auto`` (default), ``off``
(always upstream's constant-rate behaviour) or ``on`` (use the source's
timestamps whenever they are usable, even for a constant source).

Frames are matched to timestamps only by their sequential ordinal, as everywhere
else in the pipeline (see :mod:`videowipe.video`).
"""

from __future__ import annotations

import functools
import itertools
import json
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from subprocess import (
    Popen as _Popen,  # bound at import: unaffected by patches of subprocess.Popen
)

ENV_VAR = "VIDEOWIPE_SOURCE_TIMING"
MODES = ("auto", "off", "on")

# Matroska timestamps are written in nanoseconds, which is finer than any
# container's time base, so a source tick survives the round trip exactly.
_SCALE_NS = 1


def _run(cmd, input=None, timeout=None, text=False) -> subprocess.CompletedProcess:
    """``subprocess.run`` with captured output, through the ``Popen`` that was
    current at import, so that code which replaces ``subprocess.Popen`` to watch
    the cleaning pipeline's ffmpeg does not also catch these probes."""
    with _Popen(cmd, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text) as process:
        try:
            out, err = process.communicate(input, timeout=timeout)
        except BaseException:
            process.kill()
            raise
    return subprocess.CompletedProcess(cmd, process.returncode, out, err)


def timing_mode() -> str:
    """``auto`` (default), ``off`` or ``on``, from ``VIDEOWIPE_SOURCE_TIMING``."""
    value = os.environ.get(ENV_VAR, "auto").strip().lower()
    return value if value in MODES else "auto"


# --- Probing -------------------------------------------------------------------


@dataclass(frozen=True)
class SourceClock:
    """Frame timing of a source's first video stream."""

    pts: tuple  # packet pts in presentation order, in ``time_base`` ticks
    time_base: Fraction
    format_start: Fraction  # seconds; what ffmpeg subtracts from every stream
    stream_start: Fraction | None
    r_frame_rate: Fraction | None
    avg_frame_rate: Fraction | None
    constant_rate: Fraction | None  # the rate every frame sits on, else None

    @property
    def constant(self) -> bool:
        return self.constant_rate is not None

    def seconds(self, ordinal: int) -> Fraction:
        """The frame's presentation time relative to the format's start."""
        return self.pts[ordinal] * self.time_base - self.format_start


def _fraction(value) -> Fraction | None:
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    return result


def _positive(value) -> Fraction | None:
    result = _fraction(value)
    return result if result is not None and result > 0 else None


def constant_rate_of(pts, time_base: Fraction, rates) -> Fraction | None:
    """The first of *rates* for which every frame is within one time-base tick
    of ``pts[0] + i / rate``; ``None`` when there is none.

    One tick is the resolution the container can express, so a file whose
    timestamps were rounded to milliseconds (Matroska, WebM) is still constant.
    """
    count = len(pts)
    if count == 0:
        return None
    for rate in rates:
        if rate is None or rate <= 0:
            continue
        # |(pts_i - pts_0) - i * tb_den / (rate * tb_num)| <= 1, scaled to integers.
        scale = rate.numerator * time_base.numerator
        grid = rate.denominator * time_base.denominator
        first = pts[0]
        if all(abs((p - first) * scale - i * grid) <= scale for i, p in enumerate(pts)):
            return rate
    return None


def parse_probe(text: str) -> SourceClock | None:
    """Build a :class:`SourceClock` from ``ffprobe -of json`` output, or ``None``
    when the timestamps are missing, ``N/A`` or duplicated."""
    try:
        data = json.loads(text)
        stream = data["streams"][0]
        time_base = _positive(stream["time_base"])
        format_start = _fraction(data["format"]["start_time"])
        packets = data["packets"]
        pts = [int(packet["pts"]) for packet in packets]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    if time_base is None or format_start is None or not pts:
        return None
    pts.sort()
    if any(b == a for a, b in zip(pts, pts[1:])):
        return None
    r_rate = _positive(stream.get("r_frame_rate"))
    avg_rate = _positive(stream.get("avg_frame_rate"))
    return SourceClock(
        tuple(pts), time_base, format_start, _fraction(stream.get("start_time")),
        r_rate, avg_rate, constant_rate_of(pts, time_base, (r_rate, avg_rate)),
    )


def probe_source(path: str, timeout: float = 600) -> SourceClock | None:
    """Probe *path*; ``None`` on any failure (no ffprobe, error, unusable pts)."""
    try:
        result = _run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries",
             "stream=r_frame_rate,avg_frame_rate,time_base,start_time"
             ":format=start_time:packet=pts",
             "-of", "json", path],
            timeout=timeout, text=True,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not isinstance(result.stdout, str):
        return None
    return parse_probe(result.stdout)


# --- Matroska frame writer ---------------------------------------------------------


def _vint(n: int) -> bytes:
    for length in range(1, 9):
        if n < (1 << (7 * length)) - 1:
            return ((1 << (7 * length)) | n).to_bytes(length, "big")
    raise ValueError("EBML size too large")


def _element(ident: str, data: bytes) -> bytes:
    return bytes.fromhex(ident) + _vint(len(data)) + data


def _uint(value: int) -> bytes:
    return value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")


_UNKNOWN_SIZE = b"\x01\xff\xff\xff\xff\xff\xff\xff"


def matroska_header(width: int, height: int, scale_ns: int = _SCALE_NS) -> bytes:
    """EBML header and the start of a Segment (of unknown size) that holds one
    uncompressed ``bgr24`` video track. Frames follow as clusters."""
    ebml = _element("1A45DFA3", (
        _element("4286", _uint(1)) + _element("42F7", _uint(1))
        + _element("42F2", _uint(4)) + _element("42F3", _uint(8))
        + _element("4282", b"matroska") + _element("4287", _uint(4))
        + _element("4285", _uint(2))
    ))
    info = _element("1549A966", (
        _element("2AD7B1", _uint(scale_ns)) + _element("4D80", b"videowipe")
        + _element("5741", b"videowipe")
    ))
    video = _element("E0", (
        _element("B0", _uint(width)) + _element("BA", _uint(height))
        + _element("2EB524", b"BGR\x18")
    ))
    track = _element("AE", (
        _element("D7", _uint(1)) + _element("73C5", _uint(1)) + _element("83", _uint(1))
        + _element("9C", _uint(0)) + _element("86", b"V_UNCOMPRESSED") + video
    ))
    return (ebml + bytes.fromhex("18538067") + _UNKNOWN_SIZE + info
            + _element("1654AE6B", track))


def matroska_frame_parts(timestamp: int, duration: int, frame_bytes: int):
    """The bytes of one cluster before and after the frame's pixels: a cluster
    holding the timestamp and a block group whose block (track 1, offset 0)
    carries the frame, with ``duration`` as its BlockDuration (so the last frame
    of a file, or a one-frame file, still has a length)."""
    stamp = _element("E7", _uint(timestamp))
    block_header = b"\x81" + struct.pack(">h", 0) + b"\x00"
    block_prefix = bytes.fromhex("A1") + _vint(len(block_header) + frame_bytes) + block_header
    length = _element("9B", _uint(duration))
    group_size = len(block_prefix) + frame_bytes + len(length)
    cluster_size = len(stamp) + 1 + len(_vint(group_size)) + group_size
    prefix = (bytes.fromhex("1F43B675") + _vint(cluster_size) + stamp
              + bytes.fromhex("A0") + _vint(group_size) + block_prefix)
    return prefix, length


class MatroskaFrameWriter:
    """A stand-in for ``Popen.stdin`` that wraps whole raw frames in Matroska.

    Each frame of ``frame_bytes`` bytes is stamped with the next ``(timestamp,
    duration)`` pair of *timestamps* (integer ticks of ``scale_ns`` nanoseconds). Buffers of any
    size (a whole frame, a ``memoryview`` of one, or pieces of one) may be
    written; a frame is emitted once all its bytes have arrived. Everything else
    (``close``, ``flush``, ``fileno`` ...) passes through to the real stream.
    """

    def __init__(self, stream, width: int, height: int, frame_bytes: int,
                 timestamps, scale_ns: int = _SCALE_NS):
        self._stream = stream
        self._header = matroska_header(width, height, scale_ns)
        self._frame_bytes = frame_bytes
        self._timestamps = timestamps
        self._ordinal = 0
        self._pending = bytearray()

    @property
    def frames_written(self) -> int:
        return self._ordinal

    def _emit(self, pixels) -> None:
        if self._ordinal >= len(self._timestamps):
            raise RuntimeError(
                f"more frames than source timestamps ({len(self._timestamps)})"
            )
        if self._header is not None:
            self._stream.write(self._header)
            self._header = None
        prefix, suffix = matroska_frame_parts(*self._timestamps[self._ordinal], self._frame_bytes)
        self._stream.write(prefix)
        self._stream.write(pixels)
        self._stream.write(suffix)
        self._ordinal += 1

    def write(self, data) -> int:
        view = memoryview(data)
        if view.format != "B" or view.ndim != 1:
            view = view.cast("B")
        total = len(view)
        offset = 0
        if self._pending:
            take = min(self._frame_bytes - len(self._pending), total)
            self._pending += view[:take]
            offset = take
            if len(self._pending) == self._frame_bytes:
                self._emit(bytes(self._pending))
                self._pending.clear()
        while total - offset >= self._frame_bytes:
            self._emit(view[offset:offset + self._frame_bytes])
            offset += self._frame_bytes
        if offset < total:
            self._pending += view[offset:]
        return total

    def __getattr__(self, name):
        return getattr(self._stream, name)


# --- ffmpeg capability -----------------------------------------------------------------


def _self_test_stream(offset: bool):
    """(stream bytes, expected output pts in 1/1000 s) for the capability probe."""
    width, height = 16, 16
    frame_bytes = width * height * 3
    # Irregular gaps that are not multiples of any frame rate: neither a constant
    # grid nor an encoder time base of 1 / frame rate can reproduce them.
    times_ms = [83, 116, 250, 283] if offset else [0, 33, 167, 200]
    shifted = [t - times_ms[0] for t in times_ms]
    out = bytearray(matroska_header(width, height))
    for i, t in enumerate(shifted):
        gap = (shifted[i + 1] - t) if i + 1 < len(shifted) else 40
        prefix, suffix = matroska_frame_parts(t * 1_000_000, gap * 1_000_000, frame_bytes)
        out += prefix + bytes(frame_bytes) + suffix
    return bytes(out), times_ms


@functools.cache
def ffmpeg_timing_options(offset: bool = False) -> tuple | None:
    """The ffmpeg output options that keep each frame's timestamp: no retiming
    (``-fps_mode passthrough``, or ``-vsync passthrough`` on builds that lack
    it) and an encoder time base taken from the input (``-enc_time_base demux``,
    or ``-1`` on builds older than that spelling; ffmpeg 9 accepts only
    ``demux``; without it ffmpeg rounds every frame to a grid of 1 / frame
    rate). ``None``
    unless they have been shown to keep the timestamps of a small off-grid
    Matroska stream on this machine (``offset``: one that starts late).

    This is a capability check, not a version check: it runs ffmpeg."""
    data, expected = _self_test_stream(offset)
    for option, time_base in itertools.product(("-fps_mode", "-vsync"), ("demux", "-1")):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "probe.mp4")
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-nostats"]
            if offset:
                cmd += ["-itsoffset", "0.083"]
            cmd += ["-f", "matroska", "-i", "-", option, "passthrough",
                    "-enc_time_base", time_base,
                    "-video_track_timescale", "1000", "-c:v", "libx264",
                    "-preset", "ultrafast", "-pix_fmt", "yuv420p", target]
            try:
                done = _run(cmd, input=data, timeout=60)
                if done.returncode != 0:
                    continue
                probe = _run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "packet=pts", "-of", "csv=p=0", target],
                    timeout=60, text=True,
                )
            except (OSError, ValueError, subprocess.SubprocessError):
                return None
            try:
                got = sorted(int(line) for line in probe.stdout.split())
            except ValueError:
                continue
            if got == expected:
                return (option, "passthrough", "-enc_time_base", time_base)
    return None


# --- The per-job plan -------------------------------------------------------------------


def _decimal(value: Fraction) -> str:
    """Seconds as ffmpeg parses them (microsecond resolution)."""
    return f"{float(value):.6f}"


def _output_timescale(time_base: Fraction) -> int:
    """MP4 track timescale in which every source tick is a whole number."""
    scale = time_base.denominator
    while scale < 1000:
        scale *= 2
    return scale


@dataclass
class SourceTiming:
    """What one STTN job does about timestamps. ``clock`` is set when the source's
    own timestamps are used; otherwise the job is exactly upstream's."""

    first: int = 0
    last: int = 0
    trial: bool = False
    fps: float = 0.0
    clock: SourceClock | None = None

    @property
    def active(self) -> bool:
        return self.clock is not None

    # -- ffmpeg command ------------------------------------------------------------

    def _rewrite(self, cmd):
        """The command for the timestamp mode, or ``None`` when *cmd* is not the
        shape this code knows how to rewrite."""
        clock = self.clock
        try:
            at = cmd.index("-f")
            if (cmd[at:at + 4] != ["-f", "rawvideo", "-vcodec", "rawvideo"]
                    or cmd[at + 4] != "-s"
                    or cmd[at + 6:at + 12] != ["-pix_fmt", "bgr24", "-r", str(self.fps), "-i", "-"]):
                return None
            size = cmd[at + 5]
            width, height = (int(v) for v in size.split("x"))
            end = at + 12
            head, rest = list(cmd[:at]), list(cmd[end:])
            options = ffmpeg_timing_options(self.video_offset() != 0)
            if options is None:
                return None
            if self.trial:
                seek = rest.index("-ss")
                if rest[seek + 1] != str(self.first / self.fps):
                    return None
                rest[seek + 1] = _decimal(self.start_offset())
                length = rest.index("-t")
                if rest[length + 1] != str((self.last - self.first) / self.fps):
                    return None
                rest[length + 1] = _decimal(self.duration())
            movflags = rest.index("-movflags")
            if rest[movflags + 1] != "+faststart":
                return None
            first_output_ends = movflags + 3  # "-movflags", "+faststart", <file>
        except (ValueError, IndexError):
            return None
        source = []
        if self.video_offset() != 0:
            source += ["-itsoffset", _decimal(self.video_offset())]
        source += ["-f", "matroska", "-i", "-"]
        # A further output (the optional HEVC copy) has its own encoder: it needs
        # the same options, but not the MP4-only track timescale.
        if first_output_ends < len(rest):
            rest[first_output_ends:first_output_ends] = options
        rest[movflags:movflags] = [
            *options, "-video_track_timescale", str(_output_timescale(clock.time_base)),
        ]
        self._size = (width, height)
        return head + source + rest

    def start_offset(self) -> Fraction:
        """Source time of frame ``first`` (what ``-ss`` seeks the audio to)."""
        return self.clock.seconds(self.first)

    def video_offset(self) -> Fraction:
        """Where the first video frame sits on the output timeline: the real
        offset between the video and the file start for a whole run, zero for a
        trial (the audio is cut at the first frame's time)."""
        return Fraction(0) if self.trial else self.clock.seconds(0)

    def duration(self) -> Fraction:
        """Length of the trial: from frame ``first`` to the next frame after the
        range (or one more frame interval when the range ends the video)."""
        clock = self.clock
        if self.last < len(clock.pts):
            end = clock.seconds(self.last)
        else:
            gap = (clock.seconds(self.last - 1) - clock.seconds(self.last - 2)
                   if self.last >= 2 else Fraction(1, 25))
            end = clock.seconds(self.last - 1) + gap
        return end - self.start_offset()

    def timestamps(self):
        """``(timestamp, duration)`` in ns for frames ``first`` .. ``last - 1``;
        a frame lasts until the next one (the last repeats the previous gap)."""
        clock = self.clock
        count = len(clock.pts)
        base = clock.pts[self.first]  # a whole run starts at frame 0
        scale = Fraction(10**9, _SCALE_NS) * clock.time_base
        rows = []
        for i in range(self.first, self.last):
            if i + 1 < count:
                ticks = clock.pts[i + 1] - clock.pts[i]
            elif i > 0:
                ticks = clock.pts[i] - clock.pts[i - 1]
            else:
                ticks = round(1 / (clock.time_base * (self.fps or 25)))
            rows.append((round((clock.pts[i] - base) * scale), round(ticks * scale)))
        return rows

    # -- process -----------------------------------------------------------------------

    def command(self, cmd):
        """The ffmpeg command for this job: rewritten for the source's timestamps
        when they are kept (with a warning and upstream's command when it cannot
        be), else *cmd* itself."""
        rewritten = self._rewrite(cmd) if self.active else None
        if self.active and rewritten is None:
            print("Warning: could not keep the source's frame timestamps with this "
                  "ffmpeg command; the output is retimed to a constant frame rate "
                  "and audio may drift.")
            self.clock = None
        return cmd if rewritten is None else rewritten

    def attach(self, pipe):
        """Replace ``pipe.stdin`` with the writer that stamps each frame, when
        :meth:`command` rewrote the command. Returns *pipe*."""
        if self.active:
            width, height = self._size
            pipe.stdin = MatroskaFrameWriter(
                pipe.stdin, width, height, width * height * 3, self.timestamps(),
            )
            kind = "constant-rate" if self.clock.constant else "variable-rate"
            print(f"Keeping the source's frame timestamps ({kind} source, "
                  f"{len(self.clock.pts)} frames).")
        return pipe

    def popen(self, cmd, stderr_file):
        """``subprocess.Popen`` for the job's ffmpeg command: :meth:`command`,
        then :meth:`attach`."""
        pipe = subprocess.Popen(self.command(cmd), stdin=subprocess.PIPE, stderr=stderr_file)
        return self.attach(pipe)


def _warn_retimed(reason: str) -> None:
    print(f"Warning: {reason}; the output is retimed to a constant frame rate "
          "and audio may drift.")


def plan_source_timing(job) -> SourceTiming:
    """Decide how this job treats the source's timestamps (and say so)."""
    first, last = job.trial_range or (0, job.frame_count)
    plan = SourceTiming(first, last, job.trial_range is not None, job.fps)
    mode = timing_mode()
    if mode == "off" or not job.video_path or not os.path.isfile(job.video_path):
        return plan
    clock = probe_source(job.video_path)
    if clock is None:
        print("Warning: could not read the source's frame timestamps; assuming a "
              "constant frame rate. If the source is variable-rate, audio may drift.")
        return plan
    if clock.constant and mode != "on":
        return plan
    if len(clock.pts) != job.frame_count:
        if not clock.constant or mode == "on":
            _warn_retimed(
                f"the source's timestamps cannot be matched to the decoded frames "
                f"({len(clock.pts)} packets, {job.frame_count} frames)"
            )
        return plan
    offset = clock.seconds(0)
    if offset < 0:
        _warn_retimed("the source's first frame precedes its start time")
        return plan
    if ffmpeg_timing_options(offset != 0 and not plan.trial) is None:
        _warn_retimed("this ffmpeg cannot keep the source's frame timestamps")
        return plan
    plan.clock = clock
    return plan
