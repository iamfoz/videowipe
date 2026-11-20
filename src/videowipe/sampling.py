"""How many frames detection samples from a video.

Each detect mode fixes a sample count (24, 50 or 80).  On a short clip that is
plenty, but on a long one a single sample speaks for a long stretch of video: at
50 samples a 70,000-frame (47 minute) video has a sample about every 57 seconds,
and a sample that lands in a pause between lines marks its whole window absent.
Subtitle lines last one to five seconds, so a sample is only sure to land on
every line when the samples are about a second apart.

The rule keeps the mode's fixed count and adds samples only once the video is
long enough that the count would spread them thinner than the mode's interval::

    count = max(mode_count, ceil(duration / interval))

===========================  =============  ==============================
mode                         interval       fixed count kept for clips up to
===========================  =============  ==============================
fast                         (never grows)  any length: 24 samples
balanced                     1 s            50 s
balanced, subtitle target    1.25 s         62 s
sensitive                    1.25 s         100 s
===========================  =============  ==============================

``fast`` does not refine, so extra samples buy it little, and it exists to be
quick.  A mode that runs the forced band fallback (``sensitive``, or any mode
asked for subtitles) spends seven detector calls per sample (two bands, three
variants, plus the frame itself) against light fallback's three, so its
interval is a quarter longer.  Short clips, including the three ~20 s
calibration clips, are sampled exactly as before.

Switches, both environment variables read on every call:

``VIDEOWIPE_SAMPLE_INTERVAL``
    Seconds between samples for balanced (sensitive scales with it).  ``off``,
    ``0`` or any non-positive number restores the fixed counts exactly.  Unset
    or unreadable means the default.
``VIDEOWIPE_SAMPLE_MEMORY_MB``
    Upper bound on the memory the extra samples may hold.  Each successfully
    detected sample keeps a greyscale appearance thumbnail and its boxes, so
    the cap is ``budget / bytes_per_sample`` and never goes below the mode's
    own count.  The default is the smaller of 256 MB and 1/16 of the memory the
    system reports as available (the container's limit if it has one; 256 MB
    when nothing can be read).  ``0`` or ``off`` removes the cap.

Duration is ``frame_count / fps`` with OpenCV's frame count and average rate,
the same quantities ``plan.compute_source`` records.  On a variable-rate
source the average rate stands in for real time, so the spacing in seconds is
nominal: frames are still chosen by sequential ordinal, never by time seek.
Only :func:`videowipe.planning.prepare` applies the rule; a caller that passes
``sample_count`` to ``detect_clean_candidates`` directly gets exactly that many.
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping

ENV_INTERVAL = "VIDEOWIPE_SAMPLE_INTERVAL"
ENV_MEMORY = "VIDEOWIPE_SAMPLE_MEMORY_MB"

# Seconds between samples with the light band fallback.  Modes that do not
# refine never grow; the forced fallback costs more per sample, so it scales
# the interval.
DEFAULT_INTERVAL_SECONDS = 1.0
GROWING_MODES = frozenset({"balanced", "sensitive"})
FORCED_FALLBACK_SCALE = 1.25
DEFAULT_MEMORY_MB = 256.0
MEMORY_FRACTION = 1.0 / 16.0

# Mirrors detect._appearance_thumbnail (max width 320, greyscale) plus a
# generous allowance for the boxes and dictionary entries kept per sample.
_THUMBNAIL_WIDTH = 320
_PER_SAMPLE_OVERHEAD_BYTES = 4096

_OFF_WORDS = {"off", "false", "no", "none", "disable", "disabled"}


def sample_interval_seconds(environ: Mapping[str, str] | None = None) -> float | None:
    """Balanced sample interval in seconds, or ``None`` when switched off."""
    env = os.environ if environ is None else environ
    raw = env.get(ENV_INTERVAL)
    if raw is None or not raw.strip():
        return DEFAULT_INTERVAL_SECONDS
    text = raw.strip().lower()
    if text in _OFF_WORDS:
        return None
    try:
        value = float(text)
    except ValueError:
        return DEFAULT_INTERVAL_SECONDS
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _read_int(path: str) -> int | None:
    try:
        with open(path, encoding="ascii") as handle:
            text = handle.read().strip()
    except (OSError, UnicodeDecodeError):
        return None
    return int(text) if text.isdigit() else None


def _available_memory_bytes() -> int | None:
    """Memory a new allocation can expect, or ``None`` when the system will not say.

    Linux reports ``MemAvailable`` and a container may add a cgroup limit; the
    smaller wins.  Elsewhere only the physical total is known.
    """
    figures: list[int] = []
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    figures.append(int(line.split()[1]) * 1024)
                    break
    except (OSError, ValueError, IndexError, UnicodeDecodeError):
        pass
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        limit = _read_int(path)
        # An unlimited cgroup reports "max" (v2) or a huge sentinel (v1).
        if limit is not None and limit < (1 << 60):
            figures.append(limit)
    if figures:
        return min(figures)
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None
    return int(total) if total > 0 else None


def _default_memory_bytes() -> int:
    budget = int(DEFAULT_MEMORY_MB * 1024 * 1024)
    available = _available_memory_bytes()
    if available:
        budget = min(budget, int(available * MEMORY_FRACTION))
    return budget


def sample_memory_bytes(environ: Mapping[str, str] | None = None) -> int | None:
    """Memory budget for sampled evidence in bytes, or ``None`` for no cap."""
    env = os.environ if environ is None else environ
    raw = env.get(ENV_MEMORY)
    if raw is None or not raw.strip():
        return _default_memory_bytes()
    text = raw.strip().lower()
    if text in _OFF_WORDS:
        return None
    try:
        value = float(text)
    except ValueError:
        return _default_memory_bytes()
    if not math.isfinite(value) or value <= 0:
        return None
    return int(value * 1024 * 1024)


def bytes_per_sample(width: int, height: int) -> int:
    """Estimated memory held for one sampled frame (thumbnail plus overhead)."""
    if width <= 0 or height <= 0:
        return _THUMBNAIL_WIDTH * _THUMBNAIL_WIDTH + _PER_SAMPLE_OVERHEAD_BYTES
    thumb_w = min(width, _THUMBNAIL_WIDTH)
    thumb_h = max(1, round(height * thumb_w / width))
    return thumb_w * thumb_h + _PER_SAMPLE_OVERHEAD_BYTES


def interval_scale(mode: str, subtitle_fallback: str) -> float | None:
    """Multiplier on the interval for a mode, or ``None`` if it never grows."""
    if mode not in GROWING_MODES:
        return None
    return FORCED_FALLBACK_SCALE if subtitle_fallback == "force" else 1.0


def effective_sample_count(
    mode: str,
    subtitle_fallback: str,
    count: int,
    total_frames: int,
    fps: float,
    *,
    width: int = 0,
    height: int = 0,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Sample count for a video: the mode's ``count``, grown for long videos.

    ``subtitle_fallback`` is the mode's resolved band-fallback setting (an
    explicit subtitle request upgrades it to ``"force"``).

    Never below ``count``, never above ``total_frames`` once raised, and
    unchanged whenever the rule is off, the mode does not grow, the rate or
    length is unknown, or the video is short.
    """
    scale = interval_scale(mode, subtitle_fallback)
    interval = sample_interval_seconds(environ)
    if count <= 0 or scale is None or interval is None:
        return count
    if total_frames <= 0 or not fps or not math.isfinite(fps) or fps <= 0:
        return count
    # The tolerance keeps an exact multiple (50 s at 1 s) from rounding up.
    wanted = math.ceil((total_frames / fps) / (interval * scale) - 1e-9)
    if wanted <= count:
        return count
    budget = sample_memory_bytes(environ)
    if budget is not None:
        wanted = min(wanted, int(budget // bytes_per_sample(width, height)))
    return max(count, min(wanted, total_frames))


def planned_sample_count(
    video_path: str, mode: str, subtitle_fallback: str, count: int,
) -> int:
    """:func:`effective_sample_count` for a file; *count* if it cannot be probed.

    Opens the file only to read its frame count, rate and frame size; no frame
    is decoded and nothing is seeked.
    """
    if interval_scale(mode, subtitle_fallback) is None or sample_interval_seconds() is None:
        return count
    import cv2

    try:
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                return count
            total = int((capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) + 0.5)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            capture.release()
    except (cv2.error, OSError, ValueError, TypeError):
        return count
    return effective_sample_count(
        mode, subtitle_fallback, count, total, fps, width=width, height=height,
    )
