"""Candidates for subtitle or caption tracks that are on screen only rarely.

Detection keeps a pixel when the detector's boxes cover it on at least
``consistency`` of the samples (0.40 balanced, 0.30 sensitive).  A track on
screen a smaller share of the time never reaches that, however many samples
are taken, so it never becomes a candidate and is never removed.

This module adds a second, independent route to a candidate that rests on
*recurrence* alone.  Text boxes that keep returning to the same place, with
the same height, on enough separate samples are one track, whatever share of
the time it occupies.  Nothing here looks at a frame other than through the
boxes the detector reported on it, and a candidate's presence is only ever the
samples (and later frames) on which one of its own boxes was found.

The rule
--------

1. Take every box the detector reported on a sample, except those that belong
   to a candidate that already exists (inside its box, or on the row of a
   subtitle candidate) and those too narrow to be a line of text
   (:data:`MIN_WIDTH_FRACTION` of the frame width).
2. Boxes *agree* when their vertical centres are within 3 px or a quarter of
   their height (whichever is more), their heights are within a factor of
   1.43, and the shorter one lies at least half inside the other's horizontal
   extent (lines of different lengths in one row overlap in the middle, split
   halves of one long line do too).
3. From the box that agrees with the most *distinct samples*, collect its
   agreeing boxes and call them a track; repeat on what is left.
4. A track qualifies only if it has boxes on at least ``min_support(n)``
   samples, in at least :data:`MIN_EPISODES` separate stretches (runs of
   consecutive samples), where ``n`` is the number of samples that were
   examined.  ``min_support`` grows with ``sqrt(n)``: detector noise recurs
   at a place a number of times that grows with the number of samples, so a
   fixed threshold would admit more of it on a long video.  The support must
   also exceed what boxes thrown down at random rows would put in the same
   row (:func:`chance_support`): the more boxes the detector reports in all,
   the more a row has to hold.

A qualifying track becomes a candidate only if the usual region
classification calls it a subtitle (:func:`videowipe.detect._classify_region`),
the same test a dense track passes, so a caption that would be kept as scene
text or a watermark stays kept; if its appearance is the same on every sample
(correlation at least ``detect._STABLE_APPEARANCE_CORRELATION``) it is a
static graphic, a logo or a watermark that happens to be on screen only part of
the time, and is not taken for text that changes line to line.

Switch: ``VIDEOWIPE_SPARSE_TRACKS``: ``off`` (or ``0``, ``false``, ``no``)
restores exactly the behaviour without this module, ``on`` (or ``1``, ``true``,
``yes``) enables it; unset means :data:`DEFAULT_ON`.  The ``fast`` mode never
uses it (it does not refine, so a sparse track's coarse windows would be
removed whole).  See ``docs/sparse-tracks.md`` for the measurements.
"""
from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

ENV_SWITCH = "VIDEOWIPE_SPARSE_TRACKS"

# A box narrower than this share of the frame is not a line of text.
MIN_WIDTH_FRACTION = 0.06
MIN_BOX_HEIGHT = 6
# Separate stretches of consecutive samples a track must appear in.
MIN_EPISODES = 3
# Support needed: max(SUPPORT_FLOOR, ceil(SUPPORT_SLOPE * sqrt(n))).
SUPPORT_FLOOR = 8
SUPPORT_SLOPE = 1.0

# Work bound on a video with a great many boxes: at most this many heads are
# tried (those with the most boxes in their row), for at most this many tracks.
MAX_HEADS = 1500
MAX_TRACK_SEARCHES = 40

# A row must beat the chance count by this many standard deviations.
CHANCE_SIGMAS = 4.0

# Agreement between two boxes.
Y_TOLERANCE_PX = 3.0
Y_TOLERANCE_HEIGHT = 0.25
HEIGHT_RATIO = 1.43
X_CONTAINMENT = 0.5
# Slack a later frame's box gets against a track's recorded geometry.
SUPPORT_Y_TOLERANCE_HEIGHT = 0.35
SUPPORT_HEIGHT_RATIO = 1.6
# Margin added to the box extent to make the candidate's region, the same
# 14 px the frequency-map candidates get from their morphological closing.
REGION_MARGIN_X = 14
REGION_MARGIN_Y = 2

# What an unset switch means.
DEFAULT_ON = True
_ON = {"on", "1", "true", "yes", "enable", "enabled"}
_OFF = {"off", "0", "false", "no", "disable", "disabled"}

BoxTuple = tuple[int, int, int, int]


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether sparse tracks are switched on (unset or unreadable: :data:`DEFAULT_ON`)."""
    env = os.environ if environ is None else environ
    value = (env.get(ENV_SWITCH) or "").strip().lower()
    if value in _ON:
        return True
    if value in _OFF:
        return False
    return DEFAULT_ON


def min_support(n_samples: int) -> int:
    """Samples a track must appear on, for ``n_samples`` examined."""
    if n_samples <= 0:
        return SUPPORT_FLOOR
    return max(SUPPORT_FLOOR, math.ceil(SUPPORT_SLOPE * math.sqrt(n_samples)))


def chance_support(boxes: int, row_height: float, frame_height: int) -> int:
    """Boxes a row of ``row_height`` px would collect if the ``boxes`` examined
    were spread evenly over the frame, plus :data:`CHANCE_SIGMAS` deviations."""
    if boxes <= 0 or frame_height <= 0:
        return 0
    mean = boxes * min(1.0, row_height / frame_height)
    return math.ceil(mean + CHANCE_SIGMAS * math.sqrt(mean))


@dataclass
class SparseTrack:
    """One recurring row of text boxes."""

    bbox: BoxTuple                 # where box centres fall (margins included)
    y_centre: float
    height: float
    samples: list[int]             # ordinals of the samples that hold a box of it
    episodes: int

    def supports(self, box: BoxTuple) -> bool:
        """Whether *box* (x1, y1, x2, y2) is a box of this track's row."""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bx1, by1, bx2, by2 = self.bbox
        if not (bx1 <= cx <= bx2 and by1 <= cy <= by2):
            return False
        height = y2 - y1 + 1
        tolerance = max(Y_TOLERANCE_PX, SUPPORT_Y_TOLERANCE_HEIGHT * self.height)
        return (
            abs(cy - self.y_centre) <= tolerance
            and self.height / SUPPORT_HEIGHT_RATIO <= height <= self.height * SUPPORT_HEIGHT_RATIO
        )


def _episodes(ordinals: np.ndarray) -> int:
    ordinals = np.unique(ordinals)
    if len(ordinals) == 0:
        return 0
    return 1 + int(np.sum(np.diff(ordinals) > 1))


def find_tracks(
    frame_boxes: Sequence[tuple[int, Sequence[BoxTuple]]],
    frame_shape: tuple[int, int],
    skip: Callable[[BoxTuple], bool] | None = None,
) -> list[SparseTrack]:
    """Tracks that recur on enough separate samples.

    *frame_boxes* is one ``(sample_index, boxes)`` pair per examined sample, in
    sample order, boxes as ``(x1, y1, x2, y2)`` pixels.  *skip* names boxes that
    already belong to a candidate.  Returns the tracks in top-to-bottom order;
    each carries the ordinals (positions in *frame_boxes*) of its samples.
    """
    height, width = frame_shape
    n = len(frame_boxes)
    need = min_support(n)
    rows: list[tuple[int, int, int, int, int]] = []
    for ordinal, (_sample, boxes) in enumerate(frame_boxes):
        for box in boxes:
            x1, y1, x2, y2 = box
            if x2 - x1 + 1 < MIN_WIDTH_FRACTION * width or y2 - y1 + 1 < MIN_BOX_HEIGHT:
                continue
            if skip is not None and skip(box):
                continue
            rows.append((ordinal, x1, y1, x2, y2))
    if len(rows) < need:
        return []
    total_boxes = len(rows)
    arr = np.asarray(rows, dtype=np.float64)
    ordinal, x1, y1, x2, y2 = arr.T
    yc = (y1 + y2) / 2.0
    box_h = y2 - y1 + 1
    box_w = x2 - x1 + 1

    # A box cannot head a qualifying track unless enough boxes share its row.
    order = np.argsort(yc, kind="stable")
    sorted_yc = yc[order]
    tol = np.maximum(Y_TOLERANCE_PX, Y_TOLERANCE_HEIGHT * box_h)
    in_row = (
        np.searchsorted(sorted_yc, yc + tol, side="right")
        - np.searchsorted(sorted_yc, yc - tol, side="left")
    )
    alive = np.ones(len(arr), dtype=bool)
    # What a head's row must hold: the support floor, and more than chance
    # would put there given how many boxes were reported in all.
    required = np.array([
        max(need, chance_support(total_boxes, 2.0 * t + 1.0, height)) for t in tol
    ])
    heads = [int(i) for i in np.flatnonzero(in_row >= required)]
    if len(heads) > MAX_HEADS:
        heads = sorted(heads, key=lambda i: (-in_row[i], i))[:MAX_HEADS]

    def agreeing(anchor: int) -> np.ndarray:
        overlap = np.minimum(x2, x2[anchor]) - np.maximum(x1, x1[anchor]) + 1
        return (
            alive
            & (np.abs(yc - yc[anchor]) <= tol[anchor])
            & (box_h * HEIGHT_RATIO >= box_h[anchor])
            & (box_h <= box_h[anchor] * HEIGHT_RATIO)
            & (overlap >= X_CONTAINMENT * np.minimum(box_w, box_w[anchor]))
        )

    tracks: list[SparseTrack] = []
    for _search in range(MAX_TRACK_SEARCHES):
        if not heads:
            break
        # The head whose agreeing boxes clear its requirement by the most.
        best_margin, best_mask = None, None
        for anchor in heads:
            if not alive[anchor]:
                continue
            mask = agreeing(anchor)
            margin = len(np.unique(ordinal[mask])) - int(required[anchor])
            if best_margin is None or margin > best_margin:
                best_margin, best_mask = margin, mask
        if best_mask is None or best_margin < 0:
            break
        episodes = _episodes(ordinal[best_mask])
        alive &= ~best_mask
        heads = [i for i in heads if alive[i]]
        if episodes < MIN_EPISODES:
            continue
        member = np.flatnonzero(best_mask)
        region = (
            max(0, int(x1[member].min()) - REGION_MARGIN_X),
            max(0, int(y1[member].min()) - REGION_MARGIN_Y),
            min(width - 1, int(x2[member].max()) + REGION_MARGIN_X),
            min(height - 1, int(y2[member].max()) + REGION_MARGIN_Y),
        )
        tracks.append(SparseTrack(
            bbox=region,
            y_centre=float(np.median(yc[member])),
            height=float(np.median(box_h[member])),
            samples=sorted({int(v) for v in ordinal[member]}),
            episodes=episodes,
        ))
    tracks.sort(key=lambda t: (t.bbox[1], t.bbox[0]))
    return tracks


def sparse_candidates(
    sample_boxes: Sequence[tuple[int, Sequence]],
    frame_shape: tuple[int, int],
    existing: Sequence,
    start_index: int,
    thumbnails: Mapping[int, np.ndarray] | None = None,
):
    """Candidates for the sparse tracks the examined samples reveal.

    *sample_boxes* is the detector output for every examined sample as
    ``(sample_index, [TextBox, ...])``; *existing* the candidates already
    found, whose boxes and rows are left alone; *start_index* the number the
    first new id continues from (``c<start_index>``).  *thumbnails* are the
    appearance thumbnails of the samples (as kept by detection): a track whose
    appearance is the same on every sample is a static graphic that keeps
    returning, a logo or a watermark, not text that changes from line to line,
    and is left to the existing rules for overlays.
    """
    from videowipe import detect

    height, width = frame_shape
    converted = [
        (sample, [detect._bbox(box.points, width, height) for box in boxes])
        for sample, boxes in sample_boxes
    ]
    n = len(converted)
    if n == 0:
        return []
    owners = [(c.bbox, c.type == "subtitle") for c in existing]

    def skip(box: BoxTuple) -> bool:
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        for (ox1, oy1, ox2, oy2), is_subtitle in owners:
            if is_subtitle:
                # The rest of a subtitle candidate's row (the halves of a long
                # line fall outside its centre-weighted box) is its own text.
                if oy1 <= cy <= oy2:
                    return True
            elif ox1 <= cx <= ox2 and oy1 <= cy <= oy2:
                return True
        return False

    candidates = []
    for track in find_tracks(converted, frame_shape, skip=skip):
        x1, y1, x2, y2 = track.bbox
        zone = detect._zone(track.bbox, width, height)
        fraction = len(track.samples) / n
        target_type, reason, default_remove = detect._classify_region(
            track.bbox, zone, [], width, height, presence_fraction=fraction,
        )
        if target_type != "subtitle" or not default_remove:
            continue
        if thumbnails:
            stability = detect._appearance_stability(
                thumbnails, [converted[ordinal][0] for ordinal in track.samples],
                track.bbox, frame_shape,
            )
            if stability is not None and stability >= detect._STABLE_APPEARANCE_CORRELATION:
                continue
        mask = np.zeros((height, width, 1), dtype=np.uint8)
        mask[y1:y2 + 1, x1:x2 + 1, 0] = 1
        candidates.append(detect.CleanCandidate(
            id=f"c{start_index + len(candidates)}",
            type=target_type,
            label=detect._candidate_label(target_type, zone),
            bbox=track.bbox,
            confidence=fraction,
            frame_fraction=fraction,
            reason=(
                f"sparse track: {reason}; boxes on {len(track.samples)} of {n} "
                f"samples in {track.episodes} separate stretches"
            ),
            default_remove=default_remove,
            presence_frames=[converted[ordinal][0] for ordinal in track.samples],
            mask=mask,
            detector_backed=True,
            evidence_filter=track.supports,
        ))
    return candidates
