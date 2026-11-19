"""Sequential source-frame positioning shared by detection and execution."""

import threading
from concurrent.futures import ThreadPoolExecutor


class _SkipAborted(Exception):
    """Raised inside a skip whose sibling has already failed."""


def skip_frames_concurrently(readers, count: int, check_cancelled=None) -> None:
    """Run :func:`skip_frames` on every reader at once (decode releases the GIL).

    ``check_cancelled`` is called once per skipped frame per reader, exactly as
    in sequential skips, but never from two threads at the same time. The first
    reader is advanced on the calling thread. An exception from either skip
    (including one raised by ``check_cancelled``) stops the other promptly and
    is re-raised to the caller.
    """
    readers = list(readers)
    if len(readers) < 2:
        for reader in readers:
            skip_frames(reader, count, check_cancelled)
        return
    if len(readers) > 2:
        raise ValueError("skip_frames_concurrently supports at most two readers")
    stop = threading.Event()
    lock = threading.Lock()

    def guarded():
        if stop.is_set():
            raise _SkipAborted()
        if check_cancelled is not None:
            with lock:
                check_cancelled()

    def worker():
        try:
            skip_frames(readers[1], count, guarded)
        except BaseException:
            stop.set()
            raise

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker)
        try:
            skip_frames(readers[0], count, guarded)
        except BaseException as exc:
            stop.set()
            if isinstance(exc, _SkipAborted):
                # The other reader failed first; report its real error.
                future.result()
            raise
        future.result()


def skip_frames(reader, count: int, check_cancelled=None) -> None:
    """Consume exactly ``count`` frames from the reader's current position.

    Callers own the reader and track its next ordinal. OpenCV's successful
    frame seek can still land on another image for timestamp-irregular video;
    reading forward preserves the same ordinals as a full run from frame zero.
    """
    # grab() advances the decoder by one frame exactly as read() does but skips
    # the colour conversion and copy; readers without it (fakes, wrappers)
    # fall back to read().
    advance = getattr(reader, "grab", None)
    if advance is None:
        def advance():
            return reader.read()[0]
    for decoded in range(count):
        if check_cancelled is not None:
            check_cancelled()
        if not advance():
            raise ValueError(
                f"Cannot advance source reader by {count} frames; decoded {decoded}"
            )


# --- Frame count ---------------------------------------------------------------
#
# OpenCV's CAP_PROP_FRAME_COUNT is exact for containers that index every frame
# (MP4/MOV), but Matroska/WebM stores no frame count, so OpenCV estimates
# duration x fps. For variable-frame-rate sources such as YouTube WebM that
# estimate can overshoot, and the pipeline then asks for a frame that does not
# exist ("Cannot decode sampled source frame N" at the last index). When the
# container has no frame count we count the video packets with ffprobe, which
# only demuxes and is fast even for long videos.

import functools as _functools
import os as _os
import subprocess as _subprocess
from subprocess import run as _run  # bound at import: unaffected by patches of subprocess.run


def _probe(args: list[str], video_path: str, timeout: int) -> str | None:
    try:
        out = _run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", *args,
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=timeout, check=True,
        ).stdout
    except (OSError, ValueError, _subprocess.SubprocessError):
        return None
    return out.strip() if isinstance(out, str) else None


def _as_count(text: str | None) -> int | None:
    try:
        value = int((text or "").splitlines()[0])
    except (ValueError, IndexError):
        return None
    return value if value > 0 else None


@_functools.lru_cache(maxsize=64)
def _counted_frames(video_path: str, _size: int, _mtime_ns: int) -> int | None:
    """Exact frame count for containers without one, else None (keyed by file identity)."""
    if _as_count(_probe(["-show_entries", "stream=nb_frames"], video_path, 30)) is not None:
        return None  # the container indexes its frames: OpenCV's count is exact
    counted = _as_count(_probe(
        ["-count_packets", "-show_entries", "stream=nb_read_packets"], video_path, 600))
    if counted is not None:
        return counted
    # No usable ffprobe: decode-count with OpenCV (slow, but correct).
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return None
        n = 0
        while cap.grab():
            n += 1
        return n or None
    finally:
        cap.release()


def frame_count(video_path: str, capture=None) -> int:
    """Number of decodable frames in *video_path*.

    Uses OpenCV's count when the container indexes its frames and an exact
    count otherwise (see above). *capture* is an already-open
    ``cv2.VideoCapture`` for the same file, used for the estimate if given.
    """
    import cv2

    own = capture is None
    cap = cv2.VideoCapture(video_path) if own else capture
    try:
        estimate = int((cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) + 0.5)
    finally:
        if own:
            cap.release()
    try:
        stat = _os.stat(video_path)
    except OSError:
        return estimate
    exact = _counted_frames(_os.path.realpath(video_path), stat.st_size, stat.st_mtime_ns)
    return exact if exact is not None else estimate
