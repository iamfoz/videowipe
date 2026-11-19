"""Cheap per-run counters and phase timers for planning work.

Measurement only: nothing here may change a plan, a mask or a pixel. A
:class:`PerfStats` collector is created for one planning run (see
:func:`start`), bound to the running context with a ``contextvars`` variable,
and read back with :meth:`PerfStats.snapshot`. Instrumentation points call the
module-level helpers (:func:`add`, :func:`detect`, :func:`begin` ...), which
are a single context-variable lookup and return when no collector is active,
so uninstrumented runs pay almost nothing.

Scoping rules:

* one collector per plan/run, never shared between runs, so the long-lived web
  server cannot leak counts from one job into the next;
* the collector is found through a ``ContextVar``: code that runs on the thread
  that started it (the normal case, including a worker thread that calls the
  engine) is counted; a helper thread started by that code must adopt the
  collector with :func:`activate` (or run under ``contextvars.copy_context``);
* counters and timers are guarded by a lock, so several threads may feed one
  collector.

``VIDEOWIPE_PERF_STATS=0`` (also ``off``/``false``/``no``) disables collection.
"""
from __future__ import annotations

import contextvars
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

#: Planning phases, in the order the pipeline runs them.
PHASES = (
    "coarse_sampling",
    "band_fallback",
    "refine",
    "boundary_probe",
    "descender_recovery",
    "review_refine",
)

#: Counters every phase reports (zero when unused) so the JSON shape is stable.
COUNTERS = (
    "frames_grabbed",        # frames advanced with grab()/skip, never converted
    "frames_converted",      # frames converted to pixels (retrieve()/read())
    "detect_calls",          # TextDetector.detect() invocations, any detector
    "forward_passes",        # DBNetDetector network forwards (hl + manual)
    "forward_hl",            #   of which through the high-level OpenCV API
    "forward_manual",        #   of which through the manual path
    "recovery_frames",       # distinct frames that reached contrast recovery
    "recovery_calls",        # _contrast_recovered_boxes invocations
    "recovery_detect_calls", # detect() calls made inside recovery
    "recovery_hits",         # recovery invocations that returned >= 1 box
    "recovery_reused",       # recovery invocations answered from an identical earlier one
    "recovery_variants_skipped",  # recovery variants not run: outcome already fixed
    "frames_present",        # frames found present for a candidate
    "band_crops_stored",     # sampled frames whose band crops the fallback reused
    "band_second_pass",      # band fallbacks that had to decode the file again
)

_FALSE = {"0", "off", "false", "no"}
_ACTIVE: contextvars.ContextVar["PerfStats | None"] = contextvars.ContextVar(
    "videowipe_perfstats", default=None,
)
_OTHER = "other"


def enabled() -> bool:
    """Whether collection is allowed (``VIDEOWIPE_PERF_STATS``, default on)."""
    return os.environ.get("VIDEOWIPE_PERF_STATS", "1").strip().lower() not in _FALSE


class PerfStats:
    """Thread-safe counters and wall timers grouped by planning phase."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wall: dict[str, float] = {}
        self._entries: dict[str, int] = {}
        self._counters: dict[str, dict[str, int]] = {}
        self._marks: dict[tuple[str, str], set] = {}
        self._stack: list[tuple[str, float]] = []

    # -- phases ---------------------------------------------------------

    def begin(self, name: str) -> None:
        with self._lock:
            self._stack.append((name, time.perf_counter()))
            self._entries[name] = self._entries.get(name, 0) + 1
            self._counters.setdefault(name, {})
            self._wall.setdefault(name, 0.0)

    def end(self) -> None:
        with self._lock:
            self._pop_locked()

    def _pop_locked(self) -> None:
        if self._stack:
            name, started = self._stack.pop()
            self._wall[name] = self._wall.get(name, 0.0) + (time.perf_counter() - started)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        self.begin(name)
        try:
            yield
        finally:
            self.end()

    def close(self) -> None:
        """Close any phase left open by an interrupted run."""
        with self._lock:
            while self._stack:
                self._pop_locked()

    @property
    def current_phase(self) -> str:
        with self._lock:
            return self._stack[-1][0] if self._stack else _OTHER

    # -- counters -------------------------------------------------------

    def add(self, key: str, n: int = 1) -> None:
        with self._lock:
            phase = self._stack[-1][0] if self._stack else _OTHER
            row = self._counters.setdefault(phase, {})
            row[key] = row.get(key, 0) + n

    def add_many(self, **counts: int) -> None:
        with self._lock:
            phase = self._stack[-1][0] if self._stack else _OTHER
            row = self._counters.setdefault(phase, {})
            for key, n in counts.items():
                row[key] = row.get(key, 0) + n

    def mark(self, key: str, ident: Any) -> None:
        """Count ``ident`` once per phase for ``key`` (distinct-frame counters)."""
        with self._lock:
            phase = self._stack[-1][0] if self._stack else _OTHER
            self._marks.setdefault((phase, key), set()).add(ident)

    # -- output ---------------------------------------------------------

    def snapshot(self) -> dict[str, dict[str, float]]:
        """``{phase: {"wall_s", "entries", <counters>...}}`` for phases that ran."""
        with self._lock:
            names = set(self._counters) | set(self._wall)
            names |= {phase for phase, _ in self._marks}
            ordered = [p for p in PHASES if p in names]
            ordered += sorted(names - set(PHASES))
            out: dict[str, dict[str, float]] = {}
            for name in ordered:
                row: dict[str, float] = {
                    "wall_s": round(self._wall.get(name, 0.0), 4),
                    "entries": self._entries.get(name, 0),
                }
                row.update(dict.fromkeys(COUNTERS, 0))
                row.update(self._counters.get(name, {}))
                for (phase, key), seen in self._marks.items():
                    if phase == name:
                        row[key] = len(seen)
                out[name] = row
            return out


# -- module-level helpers (no-ops without an active collector) -----------------

def current() -> "PerfStats | None":
    return _ACTIVE.get()


def start() -> "tuple[PerfStats, contextvars.Token] | None":
    """Bind a fresh collector to this context; ``None`` when disabled."""
    if not enabled():
        return None
    stats = PerfStats()
    return stats, _ACTIVE.set(stats)


def finish(handle: "tuple[PerfStats, contextvars.Token] | None") -> dict:
    """Unbind the collector from :func:`start` and return its snapshot."""
    if handle is None:
        return {}
    stats, token = handle
    stats.close()
    try:
        _ACTIVE.reset(token)
    except ValueError:  # finished on a different context than it started in
        _ACTIVE.set(None)
    return stats.snapshot()


@contextmanager
def activate(stats: "PerfStats | None") -> Iterator[None]:
    """Make an existing collector current (for helper threads)."""
    token = _ACTIVE.set(stats)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def add(key: str, n: int = 1) -> None:
    stats = _ACTIVE.get()
    if stats is not None:
        stats.add(key, n)


def mark(key: str, ident: Any) -> None:
    stats = _ACTIVE.get()
    if stats is not None:
        stats.mark(key, ident)


def forward(kind: str) -> None:
    """Count one network forward pass (``kind`` is ``"hl"`` or ``"manual"``)."""
    stats = _ACTIVE.get()
    if stats is not None:
        stats.add_many(forward_passes=1, **{f"forward_{kind}": 1})


def begin(name: str) -> None:
    stats = _ACTIVE.get()
    if stats is not None:
        stats.begin(name)


def end() -> None:
    stats = _ACTIVE.get()
    if stats is not None:
        stats.end()


@contextmanager
def phase(name: str) -> Iterator[None]:
    stats = _ACTIVE.get()
    if stats is None:
        yield
        return
    with stats.phase(name):
        yield


def detect(detector: Any, frame: Any, *, recovery: bool = False) -> Any:
    """``detector.detect(frame)``, counting the call.

    Counted at the call site so any pluggable ``TextDetector`` is covered; the
    result and any exception pass through untouched.
    """
    stats = _ACTIVE.get()
    if stats is None:
        return detector.detect(frame)
    if recovery:
        stats.add_many(detect_calls=1, recovery_detect_calls=1)
    else:
        stats.add("detect_calls")
    return detector.detect(frame)


def counted(items: Iterable, key: str = "frames_converted") -> Iterable:
    """Iterate ``items`` counting each one under ``key`` (identity if inactive)."""
    stats = _ACTIVE.get()
    if stats is None:
        return items

    def generate() -> Iterator:
        for item in items:
            stats.add(key)
            yield item

    return generate()
