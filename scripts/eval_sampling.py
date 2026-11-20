#!/usr/bin/env python3
"""Score the planning pipeline against a synthetic clip with known subtitle times.

Pair this with ``scripts/gen_long_clip.py``.  It runs the same two calls the
engine makes (``planning.prepare`` then ``planning.finalize``) and compares the
final plan, frame by frame, with ``<clip>.truth.json``:

* subtitle-line recall: the share of lines the plan removes (a line is *found*
  when at least half, or at least 90 % for the strict figure, of its interior
  frames have the removal mask over at least half of the line's glyph
  rectangle) and the share of subtitle frames covered;
* false-removal frames: frames with no text on screen (one frame of margin
  either side of every line) where the removal mask is non-empty away from the
  watermark, and the mean removed area outside on-screen text;
* per-track candidate formation, and for every line that is lost, where it was
  lost: ``no_candidate`` (no candidate region covers it), ``not_removed``
  (candidate formed but its action is keep), ``window_absent`` (a remove track
  covers it but the coarse segments built from the samples do not) or
  ``refine_dropped`` (coarse segments cover it, dense refinement then dropped
  it);
* cost: detector calls (logical, by phase), wall time and peak RSS.

``--detector-cache FILE`` memoises detector results by image content, so
repeated evaluations of one clip only pay for frames not seen before.  Cached
runs report the same logical call counts; their wall time is then dominated by
everything except the detector, and ``est_detector_seconds`` (calls times the
mean latency of an uncached call, stored in the cache) says what the detector
would have cost.  The detector keeps a little state between calls, so a cache
filled by one run can differ in rare corner cases from a cold run.

``--sample-interval`` sets ``VIDEOWIPE_SAMPLE_INTERVAL`` for the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import resource
import sys
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from videowipe.detect import TextBox, _default_detector
from videowipe.plan import (
    _active_tracks,
    _execution_tracks,
    _project_mask,
    build_wipe_plan,
    save_wipe_plan,
)
from videowipe.planning import finalize, prepare

COVER = 0.5
STRICT = 0.9
PAD = 8


class CountingDetector:
    """Wrap a detector: count calls by phase and optionally memoise by content."""

    def __init__(self, inner, cache_path: str | None = None) -> None:
        self._inner = inner
        self._cache_path = cache_path
        self._cache: dict[bytes, list[tuple[np.ndarray, float, str]]] = {}
        self._latency = 0.0
        self._latency_n = 0
        self.phase = "prepare"
        self.calls: dict[str, int] = {}
        self.misses: dict[str, int] = {}
        self._prior_latency = None
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as handle:
                blob = pickle.load(handle)
            self._cache = blob["entries"]
            self._prior_latency = blob.get("mean_latency")
            self._latency, self._latency_n = blob.get("latency_sum", 0.0), blob.get("latency_n", 0)

    @staticmethod
    def _key(frame: np.ndarray) -> bytes:
        digest = hashlib.blake2b(np.ascontiguousarray(frame).tobytes(), digest_size=16)
        digest.update(repr(frame.shape).encode())
        return digest.digest()

    def detect(self, frame: np.ndarray) -> list[TextBox]:
        self.calls[self.phase] = self.calls.get(self.phase, 0) + 1
        key = self._key(frame) if self._cache_path else None
        if key is not None and key in self._cache:
            return [TextBox(points=p.copy(), confidence=c, text=t) for p, c, t in self._cache[key]]
        self.misses[self.phase] = self.misses.get(self.phase, 0) + 1
        started = time.perf_counter()
        boxes = self._inner.detect(frame)
        self._latency += time.perf_counter() - started
        self._latency_n += 1
        if key is not None:
            self._cache[key] = [(b.points.copy(), float(b.confidence), b.text) for b in boxes]
        return boxes

    @property
    def mean_latency(self) -> float:
        if self._latency_n:
            return self._latency / self._latency_n
        return float(self._prior_latency or 0.0)

    def save(self) -> None:
        if not self._cache_path:
            return
        tmp = self._cache_path + ".tmp"
        with open(tmp, "wb") as handle:
            pickle.dump({
                "entries": self._cache, "mean_latency": self.mean_latency,
                "latency_sum": self._latency, "latency_n": self._latency_n,
            }, handle)
        os.replace(tmp, self._cache_path)


def _rect_cover(mask: np.ndarray, rect: list[int]) -> float:
    x1, y1, x2, y2 = rect
    region = mask[y1:y2 + 1, x1:x2 + 1]
    return float(region.mean()) if region.size else 0.0


def _segment_frames(segments, frame_count: int) -> np.ndarray:
    out = np.zeros(frame_count, dtype=bool)
    for segment in segments:
        out[int(segment.start):int(segment.end)] = True
    return out


def _interior(line: dict) -> tuple[int, int]:
    first, end = int(line["first_frame"]), int(line["end_frame"])
    if end - first >= 4:
        return first + 1, end - 1
    return first, end


def evaluate_plan(plan, provisional, truth: dict) -> dict:
    """Frame-by-frame comparison of a plan with the truth file."""
    frame_count = int(plan.source.frame_count)
    height, width = int(plan.source.height), int(plan.source.width)
    lines = truth["lines"]
    prepared = _execution_tracks(plan)

    # Candidate coverage (any action) and remove-track segment coverage.
    cand_masks = [
        (t, (np.asarray(t.mask) > 0) if t.mask is not None else None) for t in plan.tracks
    ]
    prov_by_id = {t.id: t for t in provisional.tracks}

    def covering_tracks(rect):
        found = []
        for track, mask in cand_masks:
            if mask is not None and _rect_cover(mask, rect) >= COVER:
                found.append(track)
        return found

    on_screen: list[list[int]] = [[] for _ in range(frame_count)]
    near_text = np.zeros(frame_count + 2, dtype=bool)
    for index, line in enumerate(lines):
        first, end = _interior(line)
        for frame in range(first, min(end, frame_count)):
            on_screen[frame].append(index)
        lo = max(0, int(line["first_frame"]) - 1)
        hi = min(frame_count, int(line["end_frame"]) + 1)
        near_text[lo:hi] = True

    wm = truth["watermark"]["rect"]
    wm_pad = [max(0, wm[0] - PAD), max(0, wm[1] - PAD), min(width - 1, wm[2] + PAD), min(height - 1, wm[3] + PAD)]
    covered = [0] * len(lines)
    interior_len = [_interior(l)[1] - _interior(l)[0] for l in lines]
    wm_cov = 0
    wm_total = 0
    false_frames = 0
    no_text_frames = 0
    false_area = 0.0
    allowed = np.zeros((height, width), dtype=bool)
    for frame in range(frame_count):
        active = _active_tracks(prepared, frame)
        if active:
            mask = _project_mask(prepared, active, (height, width), frame_index=frame) > 0
        else:
            mask = None
        wm_total += 1
        if mask is not None and _rect_cover(mask, wm) >= COVER:
            wm_cov += 1
        for index in on_screen[frame]:
            if mask is not None and _rect_cover(mask, lines[index]["rect"]) >= COVER:
                covered[index] += 1
        if mask is None:
            if not near_text[frame]:
                no_text_frames += 1
            continue
        allowed[:] = False
        x1, y1, x2, y2 = wm_pad
        allowed[y1:y2 + 1, x1:x2 + 1] = True
        for index in on_screen[frame]:
            rx1, ry1, rx2, ry2 = lines[index]["rect"]
            allowed[max(0, ry1 - PAD):ry2 + PAD + 1, max(0, rx1 - PAD):rx2 + PAD + 1] = True
        outside = mask & ~allowed
        false_area += float(outside.sum()) / (height * width)
        if not near_text[frame]:
            no_text_frames += 1
            if outside.any():
                false_frames += 1

    # Loss attribution per line.
    prov_cov = {}
    for track in plan.tracks:
        ptrack = prov_by_id.get(track.id)
        prov_cov[track.id] = _segment_frames(ptrack.segments, frame_count) if ptrack else np.zeros(frame_count, bool)

    result: dict = {"watermark_frame_recall": wm_cov / max(1, wm_total)}
    for name in ("dialogue", "caption"):
        idx = [i for i, l in enumerate(lines) if l["track"] == name]
        total_frames = sum(interior_len[i] for i in idx)
        cov_frames = sum(covered[i] for i in idx)
        found = [i for i in idx if interior_len[i] and covered[i] / interior_len[i] >= COVER]
        strict = [i for i in idx if interior_len[i] and covered[i] / interior_len[i] >= STRICT]
        lost: dict[str, int] = {}
        coarse_cov_frames = 0
        with_candidate = 0
        for i in idx:
            line = lines[i]
            cands = covering_tracks(line["rect"])
            first, end = _interior(line)
            removers = [t for t in cands if t.action == "remove"]
            if cands:
                with_candidate += 1
            coarse = np.zeros(frame_count, bool)
            for t in removers:
                coarse |= prov_cov[t.id]
            coarse_frames = int(coarse[first:end].sum())
            coarse_cov_frames += coarse_frames
            if i in found:
                continue
            if not cands:
                lost["no_candidate"] = lost.get("no_candidate", 0) + 1
            elif not removers:
                lost["not_removed"] = lost.get("not_removed", 0) + 1
            elif coarse_frames / max(1, end - first) < COVER:
                lost["window_absent"] = lost.get("window_absent", 0) + 1
            else:
                lost["refine_dropped"] = lost.get("refine_dropped", 0) + 1
        result[name] = {
            "lines": len(idx),
            "lines_found": len(found),
            "line_recall": len(found) / max(1, len(idx)),
            "lines_found_strict": len(strict),
            "line_recall_strict": len(strict) / max(1, len(idx)),
            "frame_recall": cov_frames / max(1, total_frames),
            "coarse_frame_recall": coarse_cov_frames / max(1, total_frames),
            "lines_with_candidate": with_candidate,
            "candidate_formed": with_candidate >= max(1, len(idx) // 2),
            "lost_by": lost,
        }
    result["false_removal_frames"] = false_frames
    result["no_text_frames"] = no_text_frames
    result["false_removal_share_of_no_text"] = false_frames / max(1, no_text_frames)
    result["false_removed_area_mean"] = false_area / max(1, frame_count)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("clip")
    parser.add_argument("--truth", help="Defaults to <clip without .mp4>.truth.json")
    parser.add_argument("--mode", default="balanced", choices=["fast", "balanced", "sensitive"])
    parser.add_argument("--targets", nargs="*", default=[], help="e.g. subtitle")
    parser.add_argument("--sample-interval", help="Sets VIDEOWIPE_SAMPLE_INTERVAL (seconds, or off)")
    parser.add_argument("--detector-cache", help="Pickle file memoising detector output")
    parser.add_argument("--save-plan", help="Directory to write wipe_plan.json and masks into")
    parser.add_argument("--output", help="Write the result JSON here")
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    if args.sample_interval is not None:
        os.environ["VIDEOWIPE_SAMPLE_INTERVAL"] = args.sample_interval
    clip = Path(args.clip)
    truth_path = Path(args.truth) if args.truth else clip.with_suffix(".truth.json")
    truth = json.loads(truth_path.read_text())

    detector = CountingDetector(_default_detector(), args.detector_cache)
    warnings: list[str] = []
    started = time.perf_counter()
    draft = prepare(
        str(clip), detector=detector, targets=args.targets, detect_mode=args.mode,
        ocr="off", warn=warnings.append,
    )
    t_prepare = time.perf_counter() - started
    result = draft._result
    provisional = build_wipe_plan(
        result.candidates, result.sample_indices, len(result.sample_indices),
        draft._source, result.frame_shape,
        request=draft._resolved_request,
    )
    samples = sorted(result.sample_indices)
    gaps = [b - a for a, b in zip(samples, samples[1:])] or [0]
    detector.phase = "finalize"
    started = time.perf_counter()
    plan = finalize(draft)
    t_finalize = time.perf_counter() - started
    if args.save_plan:
        save_wipe_plan(plan, args.save_plan)
    detector.save()

    fps = float(plan.source.fps) or 1.0
    metrics = evaluate_plan(plan, provisional, truth)
    calls = dict(detector.calls)
    total_calls = sum(calls.values())
    latency = detector.mean_latency
    out = {
        "label": args.label,
        "clip": clip.name,
        "mode": args.mode,
        "targets": args.targets,
        "sample_interval_env": os.environ.get("VIDEOWIPE_SAMPLE_INTERVAL"),
        "frames": int(plan.source.frame_count),
        "fps": fps,
        "samples": len(samples),
        "max_gap_frames": int(max(gaps)),
        "max_gap_seconds": round(max(gaps) / fps, 2),
        "detector_calls": calls,
        "detector_calls_total": total_calls,
        "detector_uncached": dict(detector.misses),
        "est_detector_seconds": round(total_calls * latency, 1),
        "plan_seconds_wall": round(t_prepare + t_finalize, 1),
        "prepare_seconds_wall": round(t_prepare, 1),
        "finalize_seconds_wall": round(t_finalize, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "remove_tracks": len(plan.remove_tracks),
        "tracks": len(plan.tracks),
        "warnings": list(plan.warnings)[:6] + warnings[:3],
        "metrics": metrics,
    }
    text = json.dumps(out, indent=1)
    if args.output:
        Path(args.output).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
