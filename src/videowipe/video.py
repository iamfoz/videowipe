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
