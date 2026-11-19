"""Exact reuse of contrast-recovery answers within one planning run.

Contrast recovery (``detect._contrast_recovered_boxes``) is a pure function of
the band's pixels, the candidate bbox, the frame size and the detector's state
on entry. Boundary probes ask for bands that refinement has already examined
(same frame, same bbox), so the same question is put to the detector twice.
Within one run the second answer is the first answer: this module remembers it.

Exactness rules, all enforced here or at the one call site:

* the key is the SHA-256 of the crop pixels plus the crop and frame shapes, the
  pixel type, the bbox and the detector's :meth:`reuse_token`, which carries
  every piece of detector state that can change what ``detect`` returns (for
  ``DBNetDetector`` that is ``_manual_only``). A detector without a
  ``reuse_token`` is never reused, because nothing says its answers depend on
  pixels alone;
* only completed calls are stored, so an exception is never remembered;
* stored and returned boxes are private copies, so a caller that changes what
  it was given cannot change a later answer;
* a run owns its memo (:func:`scope`, bound to a ``ContextVar`` like
  :mod:`videowipe.perfstats`), so nothing survives into the next job in a
  long-lived server; the memo is also bound to one detector object;
* it is bounded: least recently used entries are dropped past the limit.

``VIDEOWIPE_RECOVERY_REUSE`` sets the limit as a number of entries (an entry is
a short key and the few boxes of one recovery, well under a kilobyte);
``off``/``0``/``false``/``no`` switches reuse off and ``on`` keeps the default.
"""
from __future__ import annotations

import contextvars
import functools
import hashlib
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

ENV = "VIDEOWIPE_RECOVERY_REUSE"
DEFAULT_ENTRIES = 4096
_OFF = {"0", "off", "false", "no"}
_ON = {"on", "true", "yes"}

_ACTIVE: contextvars.ContextVar["RecoveryMemo | None"] = contextvars.ContextVar(
    "videowipe_recovery_reuse", default=None,
)


def configured_entries() -> int:
    """Entries a run may keep (``0`` means reuse is off)."""
    raw = os.environ.get(ENV, "").strip().lower()
    if not raw or raw in _ON:
        return DEFAULT_ENTRIES
    if raw in _OFF:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_ENTRIES


def _copy_boxes(boxes: list) -> list:
    return [type(box)(points=np.array(box.points, copy=True),
                      confidence=box.confidence, text=box.text) for box in boxes]


class RecoveryMemo:
    """Least-recently-used map from a recovery key to its returned boxes."""

    def __init__(self, max_entries: int = DEFAULT_ENTRIES) -> None:
        self._max = max(1, int(max_entries))
        self._entries: "OrderedDict[tuple, list]" = OrderedDict()
        self._detector: Any = None
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @staticmethod
    def key(detector: Any, bbox: Any, frame: np.ndarray, crop: np.ndarray) -> tuple | None:
        """The full input of one recovery call, or ``None`` when not reusable."""
        token_of = getattr(detector, "reuse_token", None)
        if not callable(token_of):
            return None
        try:
            key = (
                token_of(), tuple(bbox), tuple(frame.shape), tuple(crop.shape),
                crop.dtype.str,
                hashlib.sha256(np.ascontiguousarray(crop).data).digest(),
            )
            hash(key)
        except (TypeError, ValueError):
            return None
        return None if key[0] is None else key

    def get(self, detector: Any, key: tuple) -> list | None:
        """A private copy of the stored boxes, or ``None`` (an empty list is a hit)."""
        with self._lock:
            if detector is not self._detector:
                return None
            boxes = self._entries.get(key)
            if boxes is None:
                return None
            self._entries.move_to_end(key)
            return _copy_boxes(boxes)

    def put(self, detector: Any, key: tuple, boxes: list) -> None:
        with self._lock:
            if detector is not self._detector:
                self._entries.clear()   # answers belong to one detector only
                self._detector = detector
            self._entries[key] = _copy_boxes(boxes)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)


def current() -> "RecoveryMemo | None":
    """The memo of the running planning scope, if there is one."""
    return _ACTIVE.get()


@contextmanager
def scope(max_entries: int | None = None) -> Iterator["RecoveryMemo | None"]:
    """Give the code run inside its own memo (``None`` when reuse is off)."""
    entries = configured_entries() if max_entries is None else max_entries
    if entries <= 0:
        token = _ACTIVE.set(None)
        memo = None
    else:
        memo = RecoveryMemo(entries)
        token = _ACTIVE.set(memo)
    try:
        yield memo
    finally:
        _ACTIVE.reset(token)


def scoped(function):
    """Run *function* inside its own :func:`scope`."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with scope():
            return function(*args, **kwargs)
    return wrapper
