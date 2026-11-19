#!/usr/bin/env python
"""Measure how often contrast recovery repeats an earlier call's exact inputs.

Usage:
    python scripts/recovery_reuse_probe.py VIDEO [VIDEO ...] -o OUT.json
        [--device auto|cpu|mps|cuda] [--work-dir DIR] [--no-calls]

For every video the script runs the planning flow ``videowipe clean`` uses up
to the plan (``WipeEngine.plan``, defaults of the CLI), with two wrappers
installed from this script only (the package is not edited):

* ``_contrast_recovered_boxes`` in ``videowipe.detect`` (refine) and in
  ``videowipe.planning`` (boundary probes), recording per call the planning
  phase, the frame index (taken from the ``recovery_frames`` mark that both
  callers emit just before the call), the bbox, the frame shape, the detector's
  ``_manual_only`` on entry and exit, a SHA-256 of the crop pixels, the
  detector calls made inside, whether boxes came back, and the wall time;
* ``perfstats.detect`` (every detector call, recovery or not), recording the
  phase, a SHA-256 of the input pixels, the detector's ``_manual_only`` on
  entry, the seconds the call took and a hash of the boxes it returned.

The result of a recovery call is a pure function of the crop pixels, the bbox,
the frame size and the detector state on entry, so a call whose full key equals
an earlier call's key would have returned an equal answer (the script also
checks that, by comparing the returned boxes of every repeat with the first
occurrence). The report counts those repeats per phase, how many follow an
empty result, what a one-entry "previous call for this bbox" memo would catch,
the detector calls and seconds they represent against planning's totals, and
the coarser keys (crop pixels only, and any detector input at all) that a
per-variant memo would use. It also times hashing against one detector call.

Hashing and bookkeeping time is measured and subtracted from the reported
planning wall time (``wall_adjusted_s``); wall-clock numbers on a shared
machine are rough, counts are exact. Nothing here changes a result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

# Measure the code beside this script, not whichever copy happens to be installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from videowipe import detect as detect_module
from videowipe import perfstats, planning
from videowipe.api import WipeRequest
from videowipe.engine import WipeEngine
from videowipe.plan import JSON_FILENAME

PHASE_ROWS = ("refine", "boundary_probe", "review_refine", "other")


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _boxes_digest(boxes) -> str:
    h = hashlib.sha256()
    for box in boxes:
        h.update(np.asarray(box.points, dtype=np.float64).tobytes())
        h.update(repr(float(box.confidence)).encode())
        h.update(repr(box.text).encode())
        h.update(b"|")
    return h.hexdigest()


def _phase() -> str:
    stats = perfstats.current()
    return stats.current_phase if stats is not None else "other"


def crop_rect(bbox, frame_shape):
    """The rectangle ``_contrast_recovered_boxes`` crops (same arithmetic)."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    height, width = frame_shape[:2]
    margin_y = max(8, (y2 - y1 + 1) // 4)
    margin_x = max(8, (x2 - x1 + 1) // 8)
    return (max(0, x1 - margin_x), max(0, y1 - margin_y),
            min(width - 1, x2 + margin_x), min(height - 1, y2 + margin_y))


class Recorder:
    """Installs the wrappers and keeps the raw per-call records."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.detects: list[dict] = []
        self.hash_s = 0.0          # time spent hashing: probe overhead, not planning
        self.detector = None
        self.sample_crop = None    # first recovery crop, for the overhead benchmark
        self._open: dict | None = None
        self._frame_mark = None
        self._orig = {}

    # -- installation -------------------------------------------------------

    def install(self) -> None:
        self._orig = {
            "mark": perfstats.mark,
            "detect": perfstats.detect,
            "recover_detect": detect_module._contrast_recovered_boxes,
            "recover_planning": planning._contrast_recovered_boxes,
        }
        perfstats.mark = self._mark
        perfstats.detect = self._detect
        detect_module._contrast_recovered_boxes = self._recover
        planning._contrast_recovered_boxes = self._recover

    def uninstall(self) -> None:
        perfstats.mark = self._orig["mark"]
        perfstats.detect = self._orig["detect"]
        detect_module._contrast_recovered_boxes = self._orig["recover_detect"]
        planning._contrast_recovered_boxes = self._orig["recover_planning"]

    # -- wrappers -----------------------------------------------------------

    def _mark(self, key, ident):
        if key == "recovery_frames":
            self._frame_mark = ident
        return self._orig["mark"](key, ident)

    def _detect(self, detector, frame, *, recovery=False):
        self.detector = self.detector or detector
        t0 = time.perf_counter()
        digest = _digest(frame)
        state = getattr(detector, "_manual_only", None)
        self.hash_s += time.perf_counter() - t0
        row = {"phase": _phase(), "recovery": bool(recovery), "hash": digest,
               "shape": list(frame.shape), "state": state, "error": None}
        started = time.perf_counter()
        try:
            out = self._orig["detect"](detector, frame, recovery=recovery)
        except BaseException as exc:
            row["seconds"] = time.perf_counter() - started
            row["error"] = type(exc).__name__
            self.detects.append(row)
            raise
        row["seconds"] = time.perf_counter() - started
        t0 = time.perf_counter()
        row["boxes"] = len(out)
        row["result"] = _boxes_digest(out)
        self.hash_s += time.perf_counter() - t0
        self.detects.append(row)
        if self._open is not None:
            self._open["detects"] += 1
            self._open["detect_s"] += row["seconds"]
        return out

    def _recover(self, detector, frame, bbox):
        self.detector = self.detector or detector
        t0 = time.perf_counter()
        x1, y1, x2, y2 = crop_rect(bbox, frame.shape)
        crop = frame[y1:y2 + 1, x1:x2 + 1]
        digest = _digest(crop) if crop.size else None
        if self.sample_crop is None and crop.size:
            self.sample_crop = crop.copy()
        self.hash_s += time.perf_counter() - t0
        row = {
            "phase": _phase(), "frame": self._frame_mark,
            "bbox": [int(v) for v in bbox], "frame_shape": list(frame.shape),
            "crop_rect": [x1, y1, x2, y2], "crop_shape": list(crop.shape),
            "crop_hash": digest, "state_in": getattr(detector, "_manual_only", None),
            "detects": 0, "detect_s": 0.0, "error": None,
        }
        first_detect = len(self.detects)
        self._open = row
        started = time.perf_counter()
        try:
            out = self._orig["recover_detect"](detector, frame, bbox)
        except BaseException as exc:
            row["error"] = type(exc).__name__
            raise
        finally:
            row["seconds"] = time.perf_counter() - started
            row["state_out"] = getattr(detector, "_manual_only", None)
            row["detect_first"] = first_detect
            self._open = None
            self.calls.append(row)
        t0 = time.perf_counter()
        row["returned"] = bool(out)
        row["result"] = _boxes_digest(out)
        self.hash_s += time.perf_counter() - t0
        return out


# ── analysis ─────────────────────────────────────────────────────────────────

def _full_key(call):
    return (call["crop_hash"], tuple(call["crop_shape"]), tuple(call["bbox"]),
            tuple(call["frame_shape"]), call["state_in"])


def _crop_key(call):
    return (call["crop_hash"], tuple(call["crop_shape"]), call["state_in"])


def _detect_key(row):
    return (row["hash"], tuple(row["shape"]), row["state"])


def _blank():
    return {"calls": 0, "detect_calls": 0, "detect_s": 0.0, "wall_s": 0.0}


def _add(into, call):
    into["calls"] += 1
    into["detect_calls"] += call["detects"]
    into["detect_s"] = round(into["detect_s"] + call["detect_s"], 4)
    into["wall_s"] = round(into["wall_s"] + call["seconds"], 4)


def analyse(calls, detects, stats, wall_s, hash_s) -> dict:
    """Per-phase tables of repeated recovery calls and detector inputs."""
    ok = [c for c in calls if not c["error"] and c["crop_hash"] is not None]
    total_detects = len(detects)
    total_detect_s = sum(d["seconds"] for d in detects)
    adjusted = wall_s - hash_s

    out: dict = {"planning": {
        "wall_s": round(wall_s, 3), "probe_overhead_s": round(hash_s, 3),
        "wall_adjusted_s": round(adjusted, 3),
        "detect_calls": total_detects,
        "detect_calls_perfstats": sum(int(v.get("detect_calls", 0)) for v in stats.values()),
        "detect_s": round(total_detect_s, 3),
        "recovery_calls": len(calls),
        "recovery_calls_perfstats": sum(int(v.get("recovery_calls", 0)) for v in stats.values()),
        "recovery_detect_calls": sum(c["detects"] for c in calls),
        "recovery_detect_s": round(sum(c["detect_s"] for c in calls), 3),
        "recovery_wall_s": round(sum(c["seconds"] for c in calls), 3),
        "recovery_hits": sum(1 for c in calls if c.get("returned")),
        "recovery_errors": sum(1 for c in calls if c["error"]),
        "entry_states": sorted({repr(c["state_in"]) for c in calls}),
        "state_changed_by_call": sum(1 for c in calls if c["state_in"] != c["state_out"]),
        "phase_detect_calls": {p: sum(1 for d in detects if d["phase"] == p)
                               for p in sorted({d["phase"] for d in detects})},
    }}

    first_seen: dict = {}      # full key -> index of earliest call
    last_seen: dict = {}       # full key -> index of the most recent call
    last_for_bbox: dict = {}   # bbox -> key of the previous call for that bbox
    last_overall = None
    crop_first: dict = {}
    per_phase = {p: {"all": _blank(), "repeat": _blank(), "repeat_same_phase": _blank(),
                     "repeat_cross_phase": _blank(), "repeat_after_negative": _blank(),
                     "repeat_after_positive": _blank(), "repeat_prev_for_bbox": _blank(),
                     "repeat_prev_overall": _blank(), "crop_only_repeat": _blank(),
                     "crop_only_extra_vs_full": _blank()} for p in PHASE_ROWS}
    mismatches = 0
    cross_pairs: dict = {}
    for index, call in enumerate(ok):
        phase = call["phase"] if call["phase"] in per_phase else "other"
        row = per_phase[phase]
        _add(row["all"], call)
        key, ckey = _full_key(call), _crop_key(call)
        if key in first_seen:
            earlier = ok[last_seen[key]]
            first = ok[first_seen[key]]
            _add(row["repeat"], call)
            same = earlier["phase"] == call["phase"]
            _add(row["repeat_same_phase" if same else "repeat_cross_phase"], call)
            if not same:
                pair = f"{earlier['phase']}->{call['phase']}"
                cross_pairs[pair] = cross_pairs.get(pair, 0) + 1
            _add(row["repeat_after_positive" if first["returned"] else "repeat_after_negative"], call)
            if first["result"] != call["result"]:
                mismatches += 1
            if last_for_bbox.get(tuple(call["bbox"])) == key:
                _add(row["repeat_prev_for_bbox"], call)
            if last_overall == key:
                _add(row["repeat_prev_overall"], call)
        if ckey in crop_first:
            _add(row["crop_only_repeat"], call)
            if key not in first_seen:
                _add(row["crop_only_extra_vs_full"], call)
        first_seen.setdefault(key, index)
        crop_first.setdefault(ckey, index)
        last_seen[key] = index
        last_for_bbox[tuple(call["bbox"])] = key
        last_overall = key

    totals = {name: _blank() for name in next(iter(per_phase.values()))}
    for row in per_phase.values():
        for name, value in row.items():
            for field in totals[name]:
                totals[name][field] = round(totals[name][field] + value[field], 4)
    out["recovery_by_phase"] = per_phase
    out["recovery_total"] = totals
    out["recovery_cross_phase_pairs"] = cross_pairs
    out["repeat_result_mismatches"] = mismatches

    rep = totals["repeat"]
    out["repeat_share"] = {
        "of_detect_calls": round(rep["detect_calls"] / total_detects, 4) if total_detects else None,
        "of_detect_seconds": round(rep["detect_s"] / total_detect_s, 4) if total_detect_s else None,
        "recovery_wall_of_planning_wall": round(rep["wall_s"] / adjusted, 4) if adjusted > 0 else None,
    }
    prev = totals["repeat_prev_for_bbox"]
    out["prev_for_bbox_share"] = {
        "of_detect_calls": round(prev["detect_calls"] / total_detects, 4) if total_detects else None,
        "recovery_wall_of_planning_wall": round(prev["wall_s"] / adjusted, 4) if adjusted > 0 else None,
    }

    # Detector inputs at any level: every call, then recovery-only calls, then
    # whole-frame (non-recovery) calls, with the phase pair of each repeat.
    def repeats(rows):
        seen: dict = {}
        count, seconds, pairs = 0, 0.0, {}
        for row in rows:
            key = _detect_key(row)
            if key in seen:
                count += 1
                seconds += row["seconds"]
                pair = f"{seen[key]}->{row['phase']}"
                pairs[pair] = pairs.get(pair, 0) + 1
            seen.setdefault(key, row["phase"])
        return {"calls": len(rows), "repeats": count, "repeat_s": round(seconds, 4),
                "phase_pairs": pairs}

    out["detector_input_repeats"] = {
        "all_calls": repeats(detects),
        "recovery_calls_only": repeats([d for d in detects if d["recovery"]]),
        "whole_frame_calls_only": repeats([d for d in detects if not d["recovery"]]),
    }
    mism = {}
    seen_result: dict = {}
    for row in detects:
        if row["error"]:
            continue
        key = _detect_key(row)
        if key in seen_result and seen_result[key] != row["result"]:
            mism[row["phase"]] = mism.get(row["phase"], 0) + 1
        seen_result.setdefault(key, row["result"])
    out["detector_input_repeats"]["result_mismatches"] = mism
    return out


def overhead_benchmark(recorder: Recorder) -> dict:
    """Hashing and comparison cost against one detector call on a real crop."""
    crop, detector = recorder.sample_crop, recorder.detector
    if crop is None or detector is None:
        return {"available": False}

    def best(fn, runs):
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    other = crop.copy()
    state = getattr(detector, "_manual_only", None)
    detect_s = best(lambda: detector.detect(crop), 5)
    if state is not None:
        detector._manual_only = state  # a probe call may flip it; keep the entry state
    return {
        "available": True, "crop_shape": list(crop.shape), "crop_bytes": int(crop.nbytes),
        "sha256_s": best(lambda: _digest(crop), 50),
        "array_equal_s": best(lambda: np.array_equal(crop, other), 50),
        "detect_call_s": detect_s,
    }


# ── driver ───────────────────────────────────────────────────────────────────

def probe(video: str, work_dir: str, device: str, keep_calls: bool) -> dict:
    out_dir = os.path.join(work_dir, Path(video).stem)
    engine = WipeEngine(task="clean", device=device)
    recorder = Recorder()
    record: dict = {"video": os.path.abspath(video)}
    recorder.install()
    started = time.perf_counter()
    try:
        plan = engine.plan(WipeRequest(video=video, output_dir=out_dir))
        record["plan_path"] = os.path.join(out_dir, JSON_FILENAME)
        record["remove_tracks"] = len(plan.remove_tracks)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        wall = time.perf_counter() - started
        recorder.uninstall()
    stats = engine.last_detection_stats
    record["perfstats"] = stats
    record["analysis"] = analyse(recorder.calls, recorder.detects, stats, wall, recorder.hash_s)
    try:
        record["overhead"] = overhead_benchmark(recorder)
    except Exception as exc:
        record["overhead"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if keep_calls:
        record["calls"] = recorder.calls
    try:
        engine.cleanup()
    except Exception as exc:
        print(f"warning: engine cleanup failed: {exc}", file=sys.stderr)
    return record


def summary(record: dict) -> str:
    a = record["analysis"]
    p, tot = a["planning"], a["recovery_total"]
    lines = [f"{Path(record['video']).name}: wall {p['wall_adjusted_s']}s "
             f"(probe overhead {p['probe_overhead_s']}s), {p['detect_calls']} detect calls, "
             f"{p['recovery_calls']} recovery calls "
             f"({p['recovery_detect_calls']} detect calls inside), {p['recovery_hits']} hits"]
    for phase, row in a["recovery_by_phase"].items():
        if row["all"]["calls"]:
            lines.append(
                f"  {phase:15s} calls {row['all']['calls']:5d}  repeats {row['repeat']['calls']:5d}"
                f"  (same-phase {row['repeat_same_phase']['calls']}, cross {row['repeat_cross_phase']['calls']},"
                f" prev-for-bbox {row['repeat_prev_for_bbox']['calls']},"
                f" after-negative {row['repeat_after_negative']['calls']})")
    lines.append(f"  repeats total {tot['repeat']['calls']} = {tot['repeat']['detect_calls']} detect calls"
                 f" ({a['repeat_share']['of_detect_calls']} of planning's), "
                 f"{tot['repeat']['wall_s']}s ({a['repeat_share']['recovery_wall_of_planning_wall']} of wall);"
                 f" mismatches {a['repeat_result_mismatches']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("videos", nargs="+")
    parser.add_argument("-o", "--output", required=True, help="JSON report to write")
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--work-dir", help="where plans are written (default: a temporary directory)")
    parser.add_argument("--no-calls", action="store_true", help="omit the raw per-call records")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as scratch:
        work_dir = args.work_dir or scratch
        records = []
        for video in args.videos:
            record = probe(video, work_dir, args.device, not args.no_calls)
            records.append(record)
            print(summary(record), flush=True)
            if "error" in record:
                print(f"  planning failed: {record['error']}", file=sys.stderr)
        Path(args.output).write_text(
            json.dumps({"schema_version": 1, "videos": records}, indent=1, default=str),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
