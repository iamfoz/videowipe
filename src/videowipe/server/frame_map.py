"""Seconds <-> frame ordinals from a source's real frame timestamps.

The web page and the server turn the seconds a person types or sees into frame
ordinals with ``frame = seconds * fps``. For a variable-frame-rate source that
is wrong by up to a couple of frames, because ``fps`` is only an average. This
module holds the source's real per-frame times (read once with
:func:`videowipe.timing.probe_source`) and converts through them.

Frame identity is the sequential-read ordinal everywhere (frame 0, 1, 2 ... in
decode order), so ordinal ``i`` is the ``i``-th presentation timestamp. Times
are on the cleaned output's own timeline (``SourceClock.seconds``): seconds
from the start of the container.

Only a source that is variable-rate gets a map. For a constant-rate source,
:func:`frame_map` returns ``None`` and every caller keeps upstream's
``seconds * fps`` arithmetic, so its numbers do not change. ``None`` is also
what a failed probe, a packet count that differs from the decoded frame count,
or ``VIDEOWIPE_SOURCE_TIMING=off`` gives. ``VIDEOWIPE_SOURCE_TIMING=on`` builds
the map for a constant-rate source as well.
"""

from __future__ import annotations

import os
import threading
from bisect import bisect_right
from collections import OrderedDict

from videowipe.timing import probe_source, timing_mode

_DIGITS = 6  # microseconds: finer than any container's time base is useful
_FALLBACK_GAP = 1 / 25


class FrameMap:
    """Start time of every frame, and the end of the last: ``frame_count + 1``
    strictly increasing values in seconds. Between two frame times the ordinal
    is interpolated, so a fractional frame is a fraction of that frame's
    duration."""

    __slots__ = ("times",)

    def __init__(self, times):
        self.times = tuple(times)
        if len(self.times) < 2 or any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("frame times must be strictly increasing")

    @property
    def frame_count(self) -> int:
        return len(self.times) - 1

    @property
    def duration(self) -> float:
        """The end of the last frame, in seconds."""
        return self.times[-1]

    def seconds(self, frame: float) -> float:
        """The time at which *frame* starts (clamped to the video)."""
        count = self.frame_count
        frame = min(max(frame, 0), count)
        whole = int(frame)
        if whole >= count or frame == whole:
            return self.times[whole]
        return self.times[whole] + (frame - whole) * (self.times[whole + 1] - self.times[whole])

    def frames(self, seconds: float) -> float:
        """The (fractional) ordinal at *seconds*, clamped to ``0 .. frame_count``."""
        times = self.times
        at = bisect_right(times, seconds) - 1
        if at < 0:
            return 0.0
        if at >= self.frame_count:
            return float(self.frame_count)
        return at + (seconds - times[at]) / (times[at + 1] - times[at])

    def to_list(self) -> list:
        return list(self.times)

    @classmethod
    def from_clock(cls, clock, frame_count: int, fps: float = 0.0) -> "FrameMap | None":
        """The map for *clock*, or ``None`` if its packets cannot stand for the
        *frame_count* decoded frames."""
        if frame_count <= 0 or len(clock.pts) != frame_count:
            return None
        times = [round(float(clock.seconds(i)), _DIGITS) for i in range(frame_count)]
        if times[0] < 0:
            return None  # the first frame precedes the container's start
        if frame_count > 1:
            gap = times[-1] - times[-2]
        else:
            gap = 1 / fps if fps > 0 else _FALLBACK_GAP
        times.append(round(times[-1] + gap, _DIGITS))
        try:
            return cls(times)
        except ValueError:
            return None


# --- One probe per source -----------------------------------------------------------

_CACHE: "OrderedDict[tuple, FrameMap | None]" = OrderedDict()
_CACHE_LIMIT = 16
_LOCK = threading.Lock()


def _identity(path: str, frame_count: int, fps: float, mode: str) -> tuple:
    stat = os.stat(path)
    return (os.path.realpath(path), stat.st_size, stat.st_mtime_ns, frame_count, fps, mode)


def frame_map(video_path, frame_count: int, fps: float = 0.0) -> FrameMap | None:
    """The :class:`FrameMap` of *video_path*, or ``None`` when upstream's constant
    rate arithmetic is the right (or only) answer. Probed once per file and
    cached, failures included."""
    mode = timing_mode()
    if mode == "off" or not video_path:
        return None
    try:
        key = _identity(str(video_path), frame_count, fps, mode)
    except OSError:
        return None
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    result = _build(str(video_path), frame_count, fps, mode)
    with _LOCK:
        _CACHE[key] = result
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
    return result


def _build(path: str, frame_count: int, fps: float, mode: str) -> FrameMap | None:
    clock = probe_source(path)
    if clock is None:
        print("Warning: could not read the source's frame timestamps; the page and "
              "server convert seconds to frames at a constant rate.")
        return None
    if clock.constant and mode != "on":
        return None
    result = FrameMap.from_clock(clock, frame_count, fps)
    if result is None:
        print(f"Warning: the source's timestamps cannot be matched to the decoded frames "
              f"({len(clock.pts)} packets, {frame_count} frames); the page and server "
              "convert seconds to frames at a constant rate.")
    return result


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def trial_seconds(frame_range, fps: float, frames: FrameMap | None) -> tuple[float, float]:
    """``(start, duration)`` in seconds of frames ``first .. last - 1``."""
    first, last = frame_range
    if frames is None:
        return first / fps, (last - first) / fps
    start = frames.seconds(first)
    return start, frames.seconds(last) - start
