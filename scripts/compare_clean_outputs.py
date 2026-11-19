#!/usr/bin/env python3
"""Compare two cleaned videos against the inpainting quality gate.

Use it to check that a change which alters the numbers (for example half
precision, ``VIDEOWIPE_PRECISION=float16``) has not made the output worse:

    python scripts/compare_clean_outputs.py REFERENCE.mp4 CANDIDATE.mp4 \\
        [--plan path/to/wipe_plan.json] [--json out.json]

Both videos are decoded with OpenCV and must have the same frame count and
size. The tool reports the mean and maximum absolute pixel difference over the
whole frame and, when a plan is given, inside the plan's removal mask, plus a
flicker indicator: how much the mask content changes from one frame to the
next, in each video and between them.

Gate (``plans/STTN_INFERENCE_SPEED.md``, PRD constraint C4), on the 0-255 scale:

* mean absolute difference inside the removal mask <= 0.5
* temporal-difference error inside the removal mask <= 0.5, that is the mean of
  |(candidate[i] - candidate[i-1]) - (reference[i] - reference[i-1])|

Both files have been through H.264 (CRF 18), so encoder noise is part of every
difference reported here and the gate is therefore conservative: a candidate
that is bit-identical before encoding can still show a small non-zero figure.

Exit status: 0 PASS, 1 FAIL, 2 the inputs cannot be compared, 3 no verdict
(no plan given, so there is no removal mask to apply the gate to).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

GATE_MASKED_MAE = 0.5
GATE_TEMPORAL_MAE = 0.5
ENCODER_NOTE = (
    "Both files went through H.264 CRF 18 encoding, so encoder noise is included in "
    "every figure and the gate is therefore conservative."
)


class CompareError(Exception):
    """The two videos cannot be compared (missing file, size or count mismatch)."""


def _open(path: str) -> "cv2.VideoCapture":
    if not os.path.isfile(path):
        raise CompareError(f"video not found: {path}")
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise CompareError(f"cannot open video: {path}")
    return capture


def probe(path: str) -> dict:
    """Frame count (by decoding, as container counts are not trusted) and size."""
    capture = _open(path)
    try:
        count = 0
        size = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if size is None:
                size = (int(frame.shape[1]), int(frame.shape[0]))
            count += 1
        return {"frames": count, "size": size}
    finally:
        capture.release()


def load_mask_provider(plan_path: str, feather_radius: int = 0):
    """Return ``mask_at(frame_index) -> bool (H, W)`` from a saved wipe plan.

    Uses the public ``load_wipe_plan`` / ``execution_masks`` helpers, so the
    mask is the plan's removal region as the engine would execute it. With
    ``feather_radius`` 0 it is the hard removal region, without the blend edge.
    """
    from videowipe.plan import execution_masks, load_wipe_plan

    plan = load_wipe_plan(plan_path)
    static, temporal = execution_masks(plan, feather_radius)
    static_bool = np.asarray(static)[:, :, 0] > 0

    def mask_at(index: int) -> np.ndarray:
        if temporal is None:
            return static_bool
        return np.asarray(temporal(index)) > 0

    mask_at.frame_count = plan.source.frame_count
    mask_at.size = (plan.source.width, plan.source.height)
    return mask_at


def compare_videos(reference: str, candidate: str, mask_at=None, progress=None) -> dict:
    """Compare two videos frame by frame and return the figures (no verdict).

    ``mask_at(index)`` returns a boolean (H, W) removal mask for a frame, or
    ``None`` for no mask. Differences are on the 0-255 scale over all channels.
    """
    ref_info, cand_info = probe(reference), probe(candidate)
    if ref_info["frames"] != cand_info["frames"]:
        raise CompareError(
            f"frame counts differ: reference {ref_info['frames']}, candidate {cand_info['frames']}")
    if ref_info["size"] != cand_info["size"]:
        raise CompareError(
            f"frame sizes differ: reference {ref_info['size']}, candidate {cand_info['size']}")
    if ref_info["frames"] == 0:
        raise CompareError("the videos contain no frames")

    ref_cap, cand_cap = _open(reference), _open(candidate)
    frame_sum = frame_max = 0.0
    frame_pixels = 0
    mask_sum = mask_max = 0.0
    mask_pixels = 0
    masked_frames = 0
    temporal_err_sum = ref_flicker_sum = cand_flicker_sum = 0.0
    temporal_pixels = 0
    previous = None
    try:
        for index in range(ref_info["frames"]):
            ok_a, a = ref_cap.read()
            ok_b, b = cand_cap.read()
            if not (ok_a and ok_b):
                raise CompareError(f"video ended early at frame {index}")
            a = a.astype(np.int16)
            b = b.astype(np.int16)
            diff = np.abs(a - b)
            frame_sum += float(diff.sum())
            frame_max = max(frame_max, float(diff.max()))
            frame_pixels += diff.size
            mask = mask_at(index) if mask_at is not None else None
            if mask is not None and mask.shape != diff.shape[:2]:
                raise CompareError(
                    f"plan mask is {mask.shape[1]}x{mask.shape[0]} but the videos are "
                    f"{diff.shape[1]}x{diff.shape[0]}: wrong plan for these videos?")
            if mask is not None and mask.any():
                inside = diff[mask]
                mask_sum += float(inside.sum())
                mask_max = max(mask_max, float(inside.max()))
                mask_pixels += inside.size
                masked_frames += 1
            if previous is not None and mask is not None:
                prev_a, prev_b, prev_mask = previous
                both = mask & prev_mask
                if both.any():
                    da = (a - prev_a)[both]
                    db = (b - prev_b)[both]
                    temporal_err_sum += float(np.abs(da - db).sum())
                    ref_flicker_sum += float(np.abs(da).sum())
                    cand_flicker_sum += float(np.abs(db).sum())
                    temporal_pixels += da.size
            previous = (a, b, mask) if mask is not None else None
            if progress is not None and (index + 1) % 100 == 0:
                progress(index + 1, ref_info["frames"])
    finally:
        ref_cap.release()
        cand_cap.release()

    result = {
        "frames": ref_info["frames"],
        "size": list(ref_info["size"]),
        "whole_frame": {"mean_abs_diff": frame_sum / frame_pixels, "max_abs_diff": frame_max},
        "mask": None,
        "flicker": None,
    }
    if mask_at is not None:
        result["mask"] = {
            "frames_with_mask": masked_frames,
            "pixels": mask_pixels // 3 if mask_pixels else 0,
            "mean_abs_diff": (mask_sum / mask_pixels) if mask_pixels else None,
            "max_abs_diff": mask_max if mask_pixels else None,
        }
        result["flicker"] = {
            "pixels": temporal_pixels // 3 if temporal_pixels else 0,
            "reference_mean_abs_frame_change": (ref_flicker_sum / temporal_pixels) if temporal_pixels else None,
            "candidate_mean_abs_frame_change": (cand_flicker_sum / temporal_pixels) if temporal_pixels else None,
            "temporal_diff_error": (temporal_err_sum / temporal_pixels) if temporal_pixels else None,
        }
    return result


def judge(result: dict) -> dict:
    """Apply the gate to ``compare_videos`` figures: verdict PASS, FAIL or NO VERDICT."""
    mask = result.get("mask")
    flicker = result.get("flicker")
    checks = {}
    if mask is None or mask.get("mean_abs_diff") is None:
        return {"verdict": "NO VERDICT", "checks": checks,
                "reason": "no removal mask (give --plan, or the plan has no masked frames)"}
    checks["masked_mean_abs_diff"] = {
        "value": mask["mean_abs_diff"], "limit": GATE_MASKED_MAE,
        "passed": bool(mask["mean_abs_diff"] <= GATE_MASKED_MAE)}
    if flicker and flicker.get("temporal_diff_error") is not None:
        checks["temporal_diff_error"] = {
            "value": flicker["temporal_diff_error"], "limit": GATE_TEMPORAL_MAE,
            "passed": bool(flicker["temporal_diff_error"] <= GATE_TEMPORAL_MAE)}
    passed = all(check["passed"] for check in checks.values())
    return {"verdict": "PASS" if passed else "FAIL", "checks": checks, "reason": None}


def format_report(result: dict, verdict: dict, reference: str, candidate: str) -> str:
    def num(value, digits=4):
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = [
        f"Reference: {reference}",
        f"Candidate: {candidate}",
        f"Frames: {result['frames']}   size: {result['size'][0]}x{result['size'][1]}",
        "",
        "Whole frame (0-255 scale)",
        f"  mean abs diff {num(result['whole_frame']['mean_abs_diff'])}   "
        f"max abs diff {num(result['whole_frame']['max_abs_diff'], 0)}",
    ]
    mask = result.get("mask")
    if mask is not None:
        lines += [
            "",
            f"Inside the removal mask ({mask['frames_with_mask']} frames with a mask)",
            f"  mean abs diff {num(mask['mean_abs_diff'])}   max abs diff {num(mask['max_abs_diff'], 0)}",
        ]
        flicker = result["flicker"]
        lines += [
            "",
            "Flicker inside the mask (mean abs change from one frame to the next)",
            f"  reference {num(flicker['reference_mean_abs_frame_change'])}   "
            f"candidate {num(flicker['candidate_mean_abs_frame_change'])}   "
            f"temporal-difference error {num(flicker['temporal_diff_error'])}",
        ]
    lines += ["", f"Gate: masked mean abs diff <= {GATE_MASKED_MAE}, "
                  f"temporal-difference error <= {GATE_TEMPORAL_MAE} (0-255 scale)"]
    for name, check in verdict["checks"].items():
        lines.append(f"  {name}: {check['value']:.4f} (limit {check['limit']}) "
                     f"{'ok' if check['passed'] else 'EXCEEDED'}")
    if verdict["reason"]:
        lines.append(f"  {verdict['reason']}")
    lines += ["", ENCODER_NOTE, f"Verdict: {verdict['verdict']}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two cleaned videos against the inpainting quality gate.")
    parser.add_argument("reference", help="reference clean (for example float32)")
    parser.add_argument("candidate", help="candidate clean (for example float16)")
    parser.add_argument("--plan", help="wipe_plan.json whose removal mask defines the region to judge")
    parser.add_argument("--json", dest="json_path", help="also write the figures to this JSON file")
    parser.add_argument("--feather", type=int, default=0,
                        help="feather radius applied to the plan mask (default 0: hard removal region)")
    args = parser.parse_args(argv)

    try:
        mask_at = None
        if args.plan:
            try:
                mask_at = load_mask_provider(args.plan, args.feather)
            except CompareError:
                raise
            except Exception as exc:  # plan problems are reported, not tracebacks
                raise CompareError(f"cannot use plan {args.plan}: {exc}") from exc
        result = compare_videos(
            args.reference, args.candidate, mask_at,
            progress=lambda done, total: print(f"  compared {done}/{total} frames", file=sys.stderr))
    except CompareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    verdict = judge(result)
    print(format_report(result, verdict, args.reference, args.candidate))
    if args.json_path:
        payload = dict(result, gate=verdict, reference=args.reference, candidate=args.candidate,
                       note=ENCODER_NOTE)
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return {"PASS": 0, "FAIL": 1}.get(verdict["verdict"], 3)


if __name__ == "__main__":
    sys.exit(main())
