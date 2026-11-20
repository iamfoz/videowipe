#!/usr/bin/env python3
"""How the sample count changes the chance that a track clears ``consistency``.

Detection builds a per-pixel frequency map over the samples and keeps pixels
seen on at least ``consistency`` of them (balanced: 0.40).  For the pixels every
line of a subtitle track shares, that frequency is the share of samples on
which the track is on screen.  This script draws many timelines with
``scripts/gen_long_clip.py``'s generator, samples each uniformly with N
samples, and reports how often the track clears the threshold, grouped by its
true on-screen share.

True presence comes from the generator's ground truth, so the result isolates
sampling noise from detector behaviour.  It is a simulation, not a pipeline
run.

    python scripts/sim_sampling_consistency.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SAMPLE_COUNTS = (50, 150, 300, 600, 1200)
BINS = [(0.30, 0.34), (0.34, 0.38), (0.38, 0.40), (0.40, 0.42), (0.42, 0.46), (0.46, 0.52)]


def load_generator():
    path = Path(__file__).with_name("gen_long_clip.py")
    spec = importlib.util.spec_from_file_location("gen_long_clip", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(threshold: float = 0.40, fps: int = 5, seeds: int = 300) -> None:
    gen = load_generator()
    rows = []
    for duration in (600, 1200):
        total = duration * fps
        for seed in range(1, seeds + 1):
            # Vary the pause share so the true share of the track spans the threshold.
            pause_share = 0.05 + 0.5 * ((seed * 0.6180339887) % 1.0)
            timeline = gen.build_timeline(duration, seed, pause_share=pause_share, fps=fps)
            on = np.zeros(total, dtype=bool)
            for line in timeline["lines"]:
                if line["track"] == "dialogue":
                    on[int(round(line["start"] * fps)):int(round(line["end"] * fps))] = True
            row = {"duration": duration, "share": float(on.mean())}
            for n in SAMPLE_COUNTS:
                indices = np.linspace(0, total - 1, n, dtype=int)
                row[n] = bool(on[indices].mean() >= threshold)
            rows.append(row)

    print(f"threshold {threshold:.2f}; P(track forms a candidate) by number of samples")
    print("clip     true share  runs  " + "  ".join(f"N={n:<5d}" for n in SAMPLE_COUNTS))
    for duration in (600, 1200):
        for low, high in BINS:
            chosen = [r for r in rows if r["duration"] == duration and low <= r["share"] < high]
            if not chosen:
                continue
            probs = [np.mean([r[n] for r in chosen]) for n in SAMPLE_COUNTS]
            print(f"{duration:5d} s  {low:.2f}-{high:.2f}  {len(chosen):4d}  "
                  + "  ".join(f"{p:7.2f}" for p in probs))


if __name__ == "__main__":
    main()
