"""Scratch prototype for Q-9(b): feed the model a crop around the mask.

Not part of the pipeline.  The real pipeline always hands the model a
full-width band (``width * 3 / 16`` rows, scaled to 640x120).  Here the same
backend and ``_process_segment`` are driven from a box fitted around the
mask's bounding box instead, so a 1080p subtitle is shrunk ~1.5x rather than
3x.  ``--mode band`` reproduces the pipeline's own geometry through the same
code path (a validity check against the harness' ``current`` scores);
``--mode crop`` uses the tight box; ``cropmin`` is the same but never narrower than 640 px, so the crop is not enlarged.  Scores come from the same metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import inpaint_quality_harness as H  # noqa: E402
import inpaint_quality_metrics as M  # noqa: E402

MODEL_W, MODEL_H = 640, 120
ASPECT = MODEL_W / MODEL_H


def fit_box(bbox, frame_w, frame_h, margin, min_width=0):
    """Smallest 16:3 box containing ``bbox`` grown by ``margin``, kept inside the frame.

    ``min_width`` stops the box from being narrower than the model input, i.e.
    the crop is never enlarged on its way into the model.
    """
    x0, y0, x1, y1 = bbox
    x0, y0, x1, y1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin
    width = max(x1 - x0, (y1 - y0) * ASPECT, min_width)
    width = min(width, frame_w)
    height = width / ASPECT
    if height > frame_h:
        height = frame_h
        width = height * ASPECT
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(round(min(max(cx - width / 2, 0), frame_w - width)))
    top = int(round(min(max(cy - height / 2, 0), frame_h - height)))
    return left, top, left + int(round(width)), top + int(round(height))


def band_box(frame_w, frame_h, mode_rows):
    return 0, mode_rows[0], frame_w, mode_rows[1]


def run(clip, mode, down, work, ref_length=5, neighbor_stride=5, save=(4, 12, 20, 28, 36, 45)):
    from videowipe.inpainters import sttn
    from videowipe.backends import load_backend

    alphas = H._AlphaFrames(clip)
    width, height = clip["width"], clip["height"]
    union = alphas.union()
    ys, xs = np.nonzero(union > 0)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    if mode == "band":
        mode_rows = sttn.get_inpaint_mode(height, int(width * 3 / 16), union[:, :, None])
        assert len(mode_rows) == 1
        box = band_box(width, height, mode_rows[0])
    else:
        box = fit_box(bbox, width, height, margin=round(0.04 * height),
                      min_width=MODEL_W if mode == "cropmin" else 0)
    left, top, right, bottom = box
    crop_w, crop_h = right - left, bottom - top
    interp = H.INTERP[down]
    backend = load_backend(H.WEIGHT)
    cap = cv2.VideoCapture(str(clip["dir"] / "burned.mp4"))
    rows = clip["rows"]
    outs = []
    started = time.monotonic()
    for start in range(0, H.FRAMES, 25):
        end = min(start + 25, H.FRAMES)
        originals, small = [], []
        for _ in range(start, end):
            ok, frame = cap.read()
            assert ok
            originals.append(frame[top:bottom, left:right].copy())
            small.append(cv2.resize(originals[-1], (MODEL_W, MODEL_H), interpolation=interp).astype(np.float32))
        comps = sttn._process_segment(small, backend, MODEL_W, MODEL_H, ref_length, neighbor_stride)
        for j, comp in enumerate(comps):
            index = start + j
            alpha = alphas(index)[top:bottom, left:right, None]
            big = cv2.resize(comp, (crop_w, crop_h))
            big = cv2.cvtColor(np.array(big).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32)
            blended = np.clip(alpha * big + (1.0 - alpha) * originals[j].astype(np.float32), 0, 255).astype(np.uint8)
            outs.append((index, blended))
    cap.release()
    backend.cleanup()
    # Place the crop back into the scoring rows (full width) to reuse the metrics.
    burned = H.read_rows(clip["dir"] / "burned.mp4", rows)
    captured = []
    for index, blended in outs:
        row = burned[index].copy()
        y0 = max(rows[0], top)
        y1 = min(rows[1], bottom)
        row[y0 - rows[0]:y1 - rows[0], left:right] = blended[y0 - top:y1 - top]
        captured.append(row)
    truth = H.read_rows(clip["dir"] / "clean.mkv", rows)
    crop_alpha = [clip[f"alpha{alphas.line(i)}"] for i in range(H.FRAMES)]
    crop_glyph = [clip[f"glyph{alphas.line(i)}"] for i in range(H.FRAMES)]
    scores = M.score_clip(captured, truth, crop_alpha, crop_glyph)
    scores["seconds"] = round(time.monotonic() - started, 1)
    scores["box"] = list(box)
    scores["shrink"] = round(crop_w / MODEL_W, 2)
    name = f"proto_{mode}_{down}"
    out_dir = work / "runs" / clip["name"] / name
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    for i in save:
        cv2.imwrite(str(out_dir / "frames" / f"{i:03d}.png"), captured[i])
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=1))
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--modes", default="band,crop")
    ap.add_argument("--down", default="linear")
    args = ap.parse_args()
    for name in args.clips.split(","):
        clip = H.build_clip(name, args.work)
        for mode in args.modes.split(","):
            scores = run(clip, mode, args.down, args.work)
            print(name, mode, args.down, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                      for k, v in scores.items()}), flush=True)


if __name__ == "__main__":
    main()
