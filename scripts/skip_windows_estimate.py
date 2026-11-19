#!/usr/bin/env python
"""Estimate how many STTN model windows window skipping saves on real videos.

Usage:
    python scripts/skip_windows_estimate.py VIDEO [VIDEO ...]
        [-o OUTDIR] [--device auto|cpu|mps|cuda] [--gap N] [--replan]
        [--json FILE]

For every video the script plans the clean exactly as ``videowipe clean``
does (``WipeEngine.plan``: detection and temporal refinement, no inpainting
model is loaded), then replays the inpainting segment loop WITHOUT the model:
the same band layout (``get_inpaint_mode``), the same per-frame masks
(``execution_masks`` with the engine's feather radius), the same segments of
``--gap`` frames, and, per band of each segment, the windows the model would
run today and the windows that run when only the frames that reach an emitted
pixel are computed (``VIDEOWIPE_SKIP_WINDOWS``, the default).

Planning is the slow part (minutes per clip on a CPU). The plan is written to
``OUTDIR/<video name>/`` and reused on later runs; ``--replan`` forgets it.
Model time is roughly proportional to the window count (a little less, since
every frame is still encoded), so the percentage is an upper bound on the
model-time saving. Nothing here changes a result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Measure the code beside this script, not whichever copy happens to be installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from videowipe.inpainters import sttn

NEIGHBOR_STRIDE = 5   # STTNInpainter's default


def count_windows(frame_count, width, height, static_mask, frame_mask=None, *,
                  gap=25, neighbor_stride=NEIGHBOR_STRIDE, trial_range=None):
    """Replay ``STTNInpainter.inpaint``'s segment loop and count model windows.

    ``static_mask`` is the plan's ``(H, W, 1)`` mask and ``frame_mask`` its
    optional per-frame callback (``execution_masks``). Returns a dict with the
    frames, segments, band-segments inferred, and the windows run today
    (``windows_full``) and with skipping (``windows_skipped_run``).
    """
    first, last = trial_range or (0, frame_count)
    mode = sttn.get_inpaint_mode(height, int(width * 3 / 16), static_mask)
    mask_cache = sttn._MaskCache(mode)
    totals = dict(frames=last - first, segments=0, bands=len(mode), band_segments=0,
                  windows_full=0, windows_run=0)
    for segment in range(first // gap, (last + gap - 1) // gap):
        start_f, end_f = segment * gap, min((segment + 1) * gap, frame_count)
        emitted = range(max(start_f, first), min(end_f, last))
        totals["segments"] += 1
        if frame_mask is None:
            needed = {k: [i - start_f for i in emitted] for k in range(len(mode))}
        else:
            needed = {}
            for index in emitted:
                alpha = np.asarray(frame_mask(index))
                if getattr(frame_mask, "_videowipe_stable_masks", False):
                    facts = mask_cache.get(alpha)
                else:
                    facts = sttn._MaskFacts(alpha.copy(), mode)
                for k in facts.active:
                    needed.setdefault(k, []).append(index - start_f)
        for local in needed.values():
            run, total = sttn._count_windows(end_f - start_f, neighbor_stride, local)
            totals["band_segments"] += 1
            totals["windows_full"] += total
            totals["windows_run"] += run
    return totals


def plan_video(video, out_dir, device, gap, replan):
    """The video's clean plan, reused from ``out_dir`` when it already exists."""
    from videowipe.api import WipeRequest
    from videowipe.engine import WipeEngine
    from videowipe.plan import JSON_FILENAME, load_wipe_plan

    plan_path = out_dir / JSON_FILENAME
    if replan or not plan_path.is_file():
        engine = WipeEngine(task="clean", device=device, gap=gap)
        try:
            engine.plan(WipeRequest(video=str(video), output_dir=str(out_dir)))
        finally:
            engine.cleanup()
    return load_wipe_plan(str(plan_path), video_path=str(video))


def estimate(video, out_dir, device="auto", gap=25, replan=False):
    from videowipe.engine import _DEFAULT_FEATHER_RADIUS
    from videowipe.plan import execution_masks

    plan = plan_video(video, out_dir, device, gap, replan)
    static_mask, frame_mask = execution_masks(plan, _DEFAULT_FEATHER_RADIUS)
    source = plan.source
    result = count_windows(source.frame_count, source.width, source.height,
                           static_mask, frame_mask, gap=gap)
    result["video"] = str(video)
    result["temporal"] = frame_mask is not None
    full, run = result["windows_full"], result["windows_run"]
    result["saved_pct"] = round(100.0 * (full - run) / full, 1) if full else 0.0
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("videos", nargs="+", help="input videos")
    parser.add_argument("-o", "--output", default="skip_windows_estimate",
                        help="where plans are written and reused")
    parser.add_argument("--device", default="auto", help="detector device")
    parser.add_argument("--gap", type=int, default=25, help="frames per segment (engine default 25)")
    parser.add_argument("--replan", action="store_true", help="ignore plans found in OUTPUT")
    parser.add_argument("--json", default=None, help="also write the results here")
    args = parser.parse_args(argv)

    rows = []
    for video in args.videos:
        video = Path(video)
        out_dir = Path(args.output) / video.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        rows.append(estimate(video, out_dir, args.device, args.gap, args.replan))

    header = ("video", "frames", "segments", "band-segments", "windows today",
              "windows skipping", "saved")
    table = [header] + [
        (Path(r["video"]).name, r["frames"], r["segments"], r["band_segments"],
         r["windows_full"], r["windows_run"], f"{r['saved_pct']}%")
        for r in rows
    ]
    widths = [max(len(str(line[i])) for line in table) for i in range(len(header))]
    for line in table:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(line)))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
