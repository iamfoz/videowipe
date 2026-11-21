#!/usr/bin/env python3
"""Where does ``frame = seconds * fps`` disagree with a source's real timestamps?

For each video, reads the per-frame presentation timestamps with ffprobe (the
same probe the cleaning pipeline uses) and compares them with the constant-rate
arithmetic that the web UI and the server use: frame ordinal = seconds * fps,
with fps taken from OpenCV (``CAP_PROP_FPS``).

Frame identity is the sequential-read ordinal (frame 0, 1, 2 ... in decode
order), so ordinal ``i`` is the ``i``-th presentation timestamp.

Usage: python scripts/vfr_timing_report.py [VIDEO ...]
       (default: the bundled samples under input/detext_examples)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from videowipe.timing import probe_source  # noqa: E402


def upstream_seconds_to_frames(start, end, fps, frame_count):
    """Upstream's ``seconds_to_frames`` (server/review.py), verbatim in behaviour."""
    frames = [value * fps for value in (start, end)]
    frames = [round(v) if abs(v - round(v)) < 1e-9 else v for v in frames]
    return math.floor(frames[0]), min(frame_count, math.ceil(frames[1]))


def report(path: Path) -> dict | None:
    clock = probe_source(str(path))
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int((cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) + 0.5)
    cap.release()
    if clock is None or fps <= 0:
        print(f"{path.name}: could not probe")
        return None
    base = clock.pts[0]
    t = [float((p - base) * clock.time_base) for p in clock.pts]
    n = len(t)
    gaps: dict[float, int] = {}
    for a, b in zip(t, t[1:]):
        key = round(b - a, 6)
        gaps[key] = gaps.get(key, 0) + 1
    nominal = float(max(gaps, key=lambda g: gaps[g])) if gaps else 1 / fps
    end = t[-1] + (t[-1] - t[-2] if n > 1 else nominal)

    # Seconds -> ordinal: the frame the constant arithmetic names at each frame's
    # true start time, against that frame's real ordinal.
    forward = [upstream_seconds_to_frames(t[i], t[i] + 1e-6, fps, frame_count)[0] - i for i in range(n)]
    # Ordinal -> seconds: the time the arithmetic gives frame i, against its real time.
    backward = [i / fps - t[i] for i in range(n)]
    # A 3-second trial typed in at each frame start: the range the server picks,
    # against the range that really covers [t_i, t_i + 3).
    trial_errors = []
    for i in range(n):
        want_last = next((j for j in range(i, n) if t[j] >= t[i] + 3.0 - 1e-9), n)
        got = upstream_seconds_to_frames(t[i], min(t[i] + 3.0, frame_count / fps), fps, frame_count)
        trial_errors.append((got[0] - i, got[1] - want_last))

    print(f"== {path.name}")
    print(f"   frames: {n} packets, {frame_count} decoded (OpenCV); "
          f"fps OpenCV {fps:.4f}, r_frame_rate {clock.r_frame_rate}, avg {clock.avg_frame_rate}")
    print(f"   classified {'constant' if clock.constant else 'VARIABLE'}; gaps (s: count) {dict(sorted(gaps.items()))}")
    print(f"   duration: frame_count/fps = {frame_count / fps:.4f} s; real end = {end:.4f} s "
          f"({(frame_count / fps - end) / nominal:+.2f} frames)")
    off = sum(1 for v in forward if v)
    print(f"   seconds->frame at each frame's own start time: {off}/{n} frames mis-numbered, "
          f"error {min(forward):+d}..{max(forward):+d} frames")
    print(f"   frame->seconds (i/fps) vs real time: {min(backward) / nominal:+.2f}..{max(backward) / nominal:+.2f} "
          f"frames ({min(backward) * 1000:+.0f}..{max(backward) * 1000:+.0f} ms)")
    firsts = [a for a, _ in trial_errors]
    lasts = [b for _, b in trial_errors]
    print(f"   3 s trial starting at each frame: first-frame error {min(firsts):+d}..{max(firsts):+d}, "
          f"last-frame error {min(lasts):+d}..{max(lasts):+d}; "
          f"{sum(1 for a, b in trial_errors if a or b)}/{n} trials get a different range")
    return {"name": path.name, "constant": clock.constant, "off": off, "n": n}


def main(argv):
    paths = [Path(a) for a in argv] or sorted((ROOT / "input" / "detext_examples").glob("*.mp4"))
    for path in paths:
        report(path)


if __name__ == "__main__":
    main(sys.argv[1:])
