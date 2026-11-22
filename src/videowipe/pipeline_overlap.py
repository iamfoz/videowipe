"""Overlap decoding, composing and writing with the model (``VIDEOWIPE_PIPELINE_OVERLAP``).

The inpainting loop in :mod:`videowipe.inpainters.sttn` is strictly serial: for
each segment it decodes and resizes the frames, runs the model, decodes the
frames again, composes them and writes them to ffmpeg. None of the stages
depends on the *result* of a later one, so on a machine with spare cores (or
whose model runs on an accelerator) the decode of segment N+1 and the compose
and write of segment N can happen while the model is busy.

This module overlaps them without changing a single byte:

* every frame is still decoded, composed and written by the same functions, in
  the same order, and each stage is one thread, so nothing is reordered;
* the readers only read ahead (they never skip or seek), and only the real
  ``cv2.VideoCapture`` or a reader that declares itself safe is read ahead;
* ``VIDEOWIPE_PIPELINE_OVERLAP=off`` hands the original functions and readers
  straight back, so the serial code path is exactly what it was.

Three bounded queues carry frames between the stages. Each holds at most
:attr:`Plan.depth` frames (default: one segment, at most ``DEFAULT_DEPTH_CAP``;
less when the machine reports little free memory), so the extra memory is at
most ``3 * depth`` full-resolution frames plus the three in flight.

Errors raised by a stage are re-raised in the main thread at its next frame
(or when it finishes); an exception in the main thread (including a
cancellation raised from a progress callback) aborts every stage and discards
queued frames without waiting for them to be written.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque

ENV_MODE = "VIDEOWIPE_PIPELINE_OVERLAP"
ENV_DEPTH = "VIDEOWIPE_PIPELINE_DEPTH"

#: Cores needed (usable by this process) before ``auto`` turns the overlap on.
MIN_CPUS_FOR_AUTO = 3
#: Most frames one queue holds by default, whatever the segment length.
DEFAULT_DEPTH_CAP = 32
#: Fraction of the memory the machine reports as available that the queues may use.
MEMORY_FRACTION = 8
#: Absolute ceiling for the queues, in bytes.
MEMORY_CEILING = 1024 * 1024 * 1024
#: Join timeout when stopping a reader (seconds); stages are daemon threads.
JOIN_TIMEOUT = 10.0
#: How long an abort waits for the writer to leave a write that is already
#: under way. The caller ends ffmpeg next, which frees a write stuck on a pipe.
WRITER_ABORT_TIMEOUT = 2.0


class Aborted(Exception):
    """Raised inside a stage when the pipeline is being torn down."""


class _Closed(Exception):
    """Raised by :meth:`Channel.get` once the producer has finished and the queue is empty."""


class Channel:
    """A bounded FIFO between two threads that can be closed, failed or aborted.

    ``put`` blocks while ``capacity`` items are queued; ``get`` blocks while it
    is empty. ``close`` (producer finished) lets the consumer drain what is
    queued; ``abort`` wakes both sides at once and drops what is queued.
    """

    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self._items = deque()
        self._cond = threading.Condition()
        self._closed = False
        self._aborted = False
        self.high_water = 0
        self.put_wait_s = 0.0
        self.get_wait_s = 0.0

    def __len__(self):
        with self._cond:
            return len(self._items)

    def put(self, item) -> None:
        with self._cond:
            if len(self._items) >= self.capacity and not self._aborted:
                started = time.perf_counter()
                while len(self._items) >= self.capacity and not self._aborted:
                    self._cond.wait()
                self.put_wait_s += time.perf_counter() - started
            if self._aborted:
                raise Aborted()
            self._items.append(item)
            self.high_water = max(self.high_water, len(self._items))
            self._cond.notify_all()

    def get(self):
        with self._cond:
            if not self._items and not self._closed and not self._aborted:
                started = time.perf_counter()
                while not self._items and not self._closed and not self._aborted:
                    self._cond.wait()
                self.get_wait_s += time.perf_counter() - started
            if self._aborted:
                raise Aborted()
            if not self._items:
                raise _Closed()
            item = self._items.popleft()
            self._cond.notify_all()
            return item

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def abort(self) -> None:
        with self._cond:
            self._aborted = True
            self._items.clear()
            self._cond.notify_all()


class _Stage:
    """One daemon thread that runs ``target`` and remembers how it ended."""

    def __init__(self, name, target):
        self.name = name
        self.error = None
        self._target = target
        self._thread = threading.Thread(target=self._run, name=f"videowipe-{name}", daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        try:
            self._target()
        except (Aborted, _Closed):
            pass
        except BaseException as exc:  # reported to the main thread, never lost
            self.error = exc

    def join(self, timeout=None) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()


class PrefetchReader:
    """Reads a capture-like object ahead on a thread; same ``read``/``grab`` results.

    Frames come back in exactly the order the capture produces them, so a
    consumer sees what it would have seen reading directly. The first failed
    read ends the stream (every later ``read`` returns ``(False, None)``), as
    with a capture at its end. An exception raised by the capture is raised by
    the ``read`` that reaches it, after the frames before it. Read-ahead stops
    when the queue is full, so at most ``depth`` decoded frames are held.
    """

    def __init__(self, capture, depth, *, owns=False, check=None, name="reader"):
        self._capture = capture
        self._owns = owns
        self._check = check
        self.channel = Channel(depth)
        self._ended = False
        if hasattr(capture, "grab"):
            # Only offered when the capture has it, so callers that choose
            # between ``grab`` and ``read`` with ``hasattr`` choose as before.
            self.grab = self._grab
        self._stage = _Stage(name, self._pump)
        self._stage.start()

    def _pump(self):
        channel = self.channel
        try:
            while True:
                try:
                    success, image = self._capture.read()
                except BaseException as exc:
                    channel.put(("error", exc))
                    return
                channel.put(("frame", (success, image)))
                if not success:
                    return
        finally:
            channel.close()

    def read(self):
        if self._check is not None:
            self._check()
        if self._ended:
            return False, None
        try:
            kind, payload = self.channel.get()
        except _Closed:
            self._ended = True
            return False, None
        if kind == "error":
            self._ended = True
            raise payload
        if not payload[0]:
            self._ended = True
        return payload

    def _grab(self):
        return self.read()[0]

    def stop(self) -> None:
        self.channel.abort()
        self._stage.join(JOIN_TIMEOUT)

    def release(self):
        self.stop()
        if self._owns:
            self._capture.release()

    def __getattr__(self, name):
        # Anything else (isOpened, get ...) is the capture's own. Positions
        # reported by it run ahead of what has been consumed.
        return getattr(self._capture, name)


class _Pending:
    """A composition that has been requested but not yet run."""

    __slots__ = ("args", "kwargs")

    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


class BackgroundWriter:
    """Composes and writes frames, in order, on one thread.

    ``compose`` returns a token and ``write`` queues it; the thread then runs
    the real compose and write functions for each token in turn. ``write``
    blocks while ``depth`` frames are already queued.
    """

    def __init__(self, compose, write, depth, owner):
        self._compose = compose
        self._write = write
        self._owner = owner
        self.channel = Channel(depth)
        self.compose_s = 0.0
        self.write_s = 0.0
        self.frames = 0
        self.error = None
        self._stage = _Stage("writer", self._pump)
        self._stage.start()

    def _pump(self):
        clock = time.perf_counter
        channel = self.channel
        try:
            while True:
                stream, pending = channel.get()
                started = clock()
                frame = self._compose(*pending.args, **pending.kwargs)
                composed = clock()
                self._write(stream, frame)
                self.write_s += clock() - composed
                self.compose_s += composed - started
                self.frames += 1
        except (_Closed, Aborted):
            return
        except BaseException as exc:
            # Record the error first, then wake a producer that is blocked on a
            # full queue, so it finds the error rather than a bare abort.
            self.error = exc
            channel.abort()

    # Drop-in replacements for the two call sites in the inpaint loop.
    def compose(self, *args, **kwargs):
        return _Pending(args, kwargs)

    def write(self, stream, pending):
        self._owner.check()
        try:
            self.channel.put((stream, pending))
        except Aborted:
            self._owner.check()
            raise

    def finish(self):
        """Wait until everything queued has been written."""
        self.channel.close()
        self._stage.join()

    def stop(self):
        self.channel.abort()
        return self._stage.join(WRITER_ABORT_TIMEOUT)


class Plan:
    """The decision for one job: whether to overlap, and how deep the queues are."""

    def __init__(self, enabled, depth, reason):
        self.enabled = enabled
        self.depth = depth
        self.reason = reason

    def __repr__(self):
        return f"Plan(enabled={self.enabled!r}, depth={self.depth!r}, reason={self.reason!r})"


def usable_cpus() -> int:
    """CPUs this process may run on (respects affinity and container limits)."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def requested_mode() -> str:
    """``off``, ``on`` or ``auto`` from ``VIDEOWIPE_PIPELINE_OVERLAP`` (unknown values mean auto)."""
    value = os.environ.get(ENV_MODE, "auto").strip().lower()
    if value in ("off", "0", "false", "no", "none"):
        return "off"
    if value in ("on", "1", "true", "yes", "force"):
        return "on"
    return "auto"


def queue_depth(frame_bytes, gap, available_bytes) -> int:
    """Frames each queue may hold: a segment's worth, within the memory budget.

    ``VIDEOWIPE_PIPELINE_DEPTH`` overrides it. Without a usable memory figure
    the default is used; otherwise three queues must fit in
    ``min(available / MEMORY_FRACTION, MEMORY_CEILING)``.
    """
    override = os.environ.get(ENV_DEPTH)
    if override is not None:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    wanted = max(1, min(int(gap), DEFAULT_DEPTH_CAP))
    if not available_bytes or frame_bytes <= 0:
        return wanted
    budget = min(available_bytes // MEMORY_FRACTION, MEMORY_CEILING)
    affordable = int(budget // (3 * frame_bytes))
    return max(0, min(wanted, affordable))


def decide(frame_bytes, gap, available_bytes, cpus=None, mode=None) -> Plan:
    """Choose from what the machine shows: usable cores and available memory."""
    mode = requested_mode() if mode is None else mode
    if mode == "off":
        return Plan(False, 0, "VIDEOWIPE_PIPELINE_OVERLAP=off")
    depth = queue_depth(frame_bytes, gap, available_bytes)
    if mode == "on":
        return Plan(True, max(1, depth), "VIDEOWIPE_PIPELINE_OVERLAP=on")
    cpus = usable_cpus() if cpus is None else cpus
    if cpus < MIN_CPUS_FOR_AUTO:
        return Plan(False, 0, f"{cpus} usable CPU(s); the overlap needs {MIN_CPUS_FOR_AUTO}")
    if depth < 2:
        return Plan(False, 0, "not enough free memory for two frames per queue")
    return Plan(True, depth, f"{cpus} usable CPUs")


class _Off:
    """The serial code path: the original functions and readers, untouched."""

    enabled = False
    reason = "off"

    def __init__(self, reader=None, output_reader=None, compose=None, write=None):
        self.reader = reader
        self.output_reader = output_reader
        self.compose = compose
        self.write = write

    def finish(self):
        pass

    def abort(self):
        pass

    def metrics(self):
        return {}


#: Placeholder bound before the pipeline exists, so ``finally`` blocks can call it.
OFF = _Off()


class Overlap:
    """The running pipeline: wrapped readers, a background writer, one error slot."""

    enabled = True

    def __init__(self, plan, reader, output_reader, compose, write, capture_type=None):
        self.plan = plan
        self.reason = plan.reason
        self._stages = []
        self.writer = BackgroundWriter(compose, write, plan.depth, self)
        self.compose = self.writer.compose
        self.write = self.writer.write
        self._readers = []
        self._capture_type = capture_type
        self.reader = self._wrap(reader, False, "reader")
        self.output_reader = self._wrap(output_reader, True, "output-reader")

    def _wrap(self, capture, owns, name):
        """Read ahead only on a capture that is known to tolerate it.

        That is an exact ``capture_type`` (the real ``cv2.VideoCapture``) or an
        object that says so with a true ``videowipe_prefetch_safe`` attribute.
        Anything else (test doubles, wrappers, a reader that is not safe to use
        from another thread or that counts its reads) is used as it is.
        """
        safe = (type(capture) is self._capture_type
                or getattr(capture, "videowipe_prefetch_safe", False) is True)
        if capture is None or not safe or not callable(getattr(capture, "read", None)):
            return capture
        wrapped = PrefetchReader(capture, self.plan.depth, owns=owns,
                                 check=self.check, name=name)
        self._readers.append(wrapped)
        return wrapped

    def check(self):
        """Raise a stage's error in the calling (main) thread, if there is one."""
        error = self.writer.error
        if error is not None:
            raise error

    def finish(self):
        """Normal end: wait for the writer, stop the readers, raise any stage error."""
        self.writer.finish()
        for reader in self._readers:
            reader.stop()
        self.check()

    def abort(self):
        """Tear everything down without waiting for queued frames to be written."""
        for reader in self._readers:
            reader.stop()
        self.writer.stop()

    def metrics(self):
        readers = {reader._stage.name: reader.channel for reader in self._readers}
        out = {
            "pipeline_overlap": 1,
            "pipeline_depth": self.plan.depth,
            "overlap_compose_s": round(self.writer.compose_s, 4),
            "overlap_write_s": round(self.writer.write_s, 4),
            "overlap_submit_wait_s": round(self.writer.channel.put_wait_s, 4),
            "overlap_writer_idle_s": round(self.writer.channel.get_wait_s, 4),
            "overlap_writer_high_water": self.writer.channel.high_water,
        }
        for name, channel in readers.items():
            key = name.replace("-", "_")
            out[f"overlap_{key}_wait_s"] = round(channel.get_wait_s, 4)
            out[f"overlap_{key}_high_water"] = channel.high_water
        return out


def attach(job, reader, output_reader, compose, write, *, frame_bytes,
           available_bytes=0, capture_type=None, cpus=None, mode=None):
    """Return the overlap for this job, or the serial path when it is off.

    The result always has ``reader``, ``output_reader``, ``compose``, ``write``,
    ``finish()``, ``abort()`` and ``metrics()``.
    """
    plan = decide(frame_bytes, job.gap, available_bytes, cpus=cpus, mode=mode)
    if not plan.enabled:
        return _Off(reader, output_reader, compose, write)
    return Overlap(plan, reader, output_reader, compose, write, capture_type)
