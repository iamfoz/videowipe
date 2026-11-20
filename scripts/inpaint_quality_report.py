"""Tables from the harness' ``runs/<clip>/<variant>/scores.json`` files.

Prints (Markdown) per-clip metrics and the paired difference of every variant
against ``current`` per resolution group, with the spread across clips so the
reader can tell a real effect from noise.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

GROUPS = {"480p": ("e1", "e2", "e3"), "1080p": ("c1", "c2", "o1", "o2"), "4K": ("k1", "k2")}
# metric -> (label, higher is better?)
METRICS = {
    "core_psnr": ("core PSNR dB", True),
    "core_ssim": ("core SSIM", True),
    "core_mae": ("core MAE", False),
    "inner_mae": ("inner-edge MAE", False),
    "seam_mae": ("seam MAE", False),
    "seam_psnr": ("seam PSNR dB", True),
    "seam_ssim": ("seam SSIM", True),
    "inner_bias": ("inner-edge luma bias", None),
    "sharp_ratio": ("sharpness (Lap.var / truth)", True),
    "ghost_energy": ("ghost edge excess", None),
    "ghost_corr": ("ghost corr.", False),
    "flicker_extra": ("extra flicker", False),
    "flicker_out": ("flicker (out)", None),
}


def load(runs: Path) -> dict[str, dict[str, dict[str, float]]]:
    data: dict[str, dict[str, dict[str, float]]] = {}
    for path in sorted(runs.glob("*/*/scores.json")):
        clip, variant = path.parent.parent.name, path.parent.name
        data.setdefault(clip, {})[variant] = json.loads(path.read_text())
    return data


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def per_clip(data, variants, keys=("core_psnr", "core_ssim", "seam_mae", "sharp_ratio", "ghost_energy", "flicker_extra")):
    lines = []
    for clip, runs in data.items():
        lines.append(f"\n**{clip}**\n")
        lines.append("| variant | " + " | ".join(METRICS[k][0] for k in keys) + " |")
        lines.append("|---|" + "---:|" * len(keys))
        for variant in variants:
            if variant in runs:
                lines.append(f"| {variant} | " + " | ".join(fmt(runs[variant][k]) for k in keys) + " |")
    return "\n".join(lines)


def paired(data, variants, baseline="current", keys=None):
    """Mean +/- SD of (variant - baseline) per resolution group, and sign counts."""
    keys = keys or list(METRICS)
    lines = []
    for group, clips in GROUPS.items():
        present = [c for c in clips if c in data and baseline in data[c]]
        if not present:
            continue
        lines.append(f"\n**{group}** ({len(present)} clip(s): {', '.join(present)})\n")
        lines.append("| variant | " + " | ".join(METRICS[k][0] for k in keys) + " |")
        lines.append("|---|" + "---:|" * len(keys))
        for variant in variants:
            if variant == baseline:
                continue
            row = []
            usable = [c for c in present if variant in data[c]]
            if not usable:
                continue
            for k in keys:
                diffs = [data[c][variant][k] - data[c][baseline][k] for c in usable]
                mean = statistics.mean(diffs)
                sd = statistics.stdev(diffs) if len(diffs) > 1 else float("nan")
                better = METRICS[k][1]
                wins = sum((d > 0) == bool(better) and d != 0 for d in diffs) if better is not None else None
                cell = f"{mean:+.3f}" + ("" if sd != sd else f" (sd {sd:.3f})")
                if wins is not None:
                    cell += f" [{wins}/{len(diffs)}]"
                row.append(cell)
            lines.append(f"| {variant} | " + " | ".join(row) + " |")
    return "\n".join(lines)


COMPACT = ("core_psnr", "core_ssim", "seam_mae", "inner_bias", "sharp_ratio", "ghost_corr", "flicker_extra")


def compact(data, variants, baseline="current"):
    """``current -> variant`` means per resolution group, over the clips that have both."""
    lines = []
    for group, clips in GROUPS.items():
        lines.append(f"\n**{group}**\n")
        lines.append("| variant | clips | " + " | ".join(METRICS[k][0] for k in COMPACT) + " |")
        lines.append("|---|---|" + "---:|" * len(COMPACT))
        for variant in variants:
            if variant == baseline:
                continue
            usable = [c for c in clips if c in data and baseline in data[c] and variant in data[c]]
            if not usable:
                continue
            cells = []
            for k in COMPACT:
                before = statistics.mean(data[c][baseline][k] for c in usable)
                after = statistics.mean(data[c][variant][k] for c in usable)
                cells.append(f"{before:.3f} -> {after:.3f}")
            lines.append(f"| {variant} | {','.join(usable)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--variants", default="current,area,area_cubic,area_lanczos,linear_cubic,masked_zero,masked_mean,seam")
    ap.add_argument("--baseline", default="current")
    ap.add_argument("--keys", default=",".join(("core_psnr", "core_ssim", "seam_mae", "inner_bias", "sharp_ratio", "flicker_extra")),
                    help="metrics for the paired table")
    ap.add_argument("--per-clip", action="store_true")
    ap.add_argument("--compact", action="store_true", help="current -> variant means per resolution group")
    args = ap.parse_args()
    data = load(args.runs)
    variants = args.variants.split(",")
    if args.compact:
        print(compact(data, variants, args.baseline))
        return
    if args.per_clip:
        print(per_clip(data, variants))
    print("\n### Paired difference against", args.baseline, "(variant - baseline; [n better / n clips])")
    print(paired(data, variants, args.baseline, args.keys.split(",")))


if __name__ == "__main__":
    main()
