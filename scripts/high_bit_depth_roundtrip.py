#!/usr/bin/env python3
"""Measure what the cleaning pipeline does to 10-bit and HDR sources, with and without
``VIDEOWIPE_HIGH_BIT_DEPTH``.

Sources: lossless HEVC (x265 ``lossless=1``, Main 10) clips of a smooth test picture,
written at 10 bits and tagged as 10-bit SDR (BT.709), HDR10 (BT.2020, PQ, with a
mastering display and content light level) and HLG (BT.2020, ARIB STD-B67). The
picture has, from the top:

``ramp``
    a horizontal luma ramp over the whole range (64 to 940, 1.4 codes per pixel);
``shallow``
    a ramp of only 64 ten-bit codes (16 eight-bit levels) where banding is plainest;
``chroma``
    a Cb ramp and a Cr ramp at constant luma.

Each goes through the real ``STTNInpainter.inpaint`` with a stub model (the mask box,
in a corner clear of the rows measured, is blanked; the rest is decoded, piped and
encoded as in a real run). Every file written is decoded to ``yuv420p10le`` and its
first frame compared with the source's, outside the mask box: the root-mean-square
and largest error in 10-bit code values, how many distinct levels the ramps keep (the
source has 876 and 64; 8 bits keep at most 220 and 16) and the longest run of equal
values (banding), and whether the tags and the HDR metadata survive.

Usage::

    python scripts/high_bit_depth_roundtrip.py                  # off and auto, all three
    python scripts/high_bit_depth_roundtrip.py --kinds pq --modes auto --frames 8 --keep DIR
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    import videowipe  # noqa: F401
except ImportError:  # run from a checkout that is not installed
    sys.path.insert(0, str(ROOT / "src"))

WIDTH, HEIGHT = 640, 360
MASK_BOX = (520, 300, 620, 350)  # x0, y0, x1, y1: bottom right, clear of the rows measured
ROWS = {"ramp": (20, 160), "shallow": (170, 230), "chroma": (240, 290)}

KINDS = {
    "sdr10": dict(tags=dict(colorprim="bt709", transfer="bt709", colormatrix="bt709")),
    "pq": dict(tags=dict(colorprim="bt2020", transfer="smpte2084", colormatrix="bt2020nc"),
               hdr=("master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)"
                    "L(10000000,1):max-cll=1000,400:hdr10=1:hdr10-opt=1:repeat-headers=1")),
    "hlg": dict(tags=dict(colorprim="bt2020", transfer="arib-std-b67", colormatrix="bt2020nc")),
}


def load_roundtrip():
    spec = importlib.util.spec_from_file_location("colour_roundtrip", ROOT / "scripts" / "colour_roundtrip.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def picture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 10-bit Y, Cb, Cr planes (uint16) of the test picture."""
    y = np.full((HEIGHT, WIDTH), 512, np.uint16)
    cb = np.full((HEIGHT // 2, WIDTH // 2), 512, np.uint16)
    cr = np.full((HEIGHT // 2, WIDTH // 2), 512, np.uint16)
    x = np.arange(WIDTH)
    y[ROWS["ramp"][0]:ROWS["ramp"][1]] = np.rint(64 + 876 * x / (WIDTH - 1)).astype(np.uint16)
    y[ROWS["shallow"][0]:ROWS["shallow"][1]] = np.rint(400 + 64 * x / (WIDTH - 1)).astype(np.uint16)
    top, bottom = ROWS["chroma"]
    y[top:bottom] = 500
    half = np.arange(WIDTH // 2)
    cb[top // 2:bottom // 2] = np.rint(400 + 224 * half / (WIDTH // 2 - 1)).astype(np.uint16)
    cr[top // 2:bottom // 2] = np.rint(624 - 224 * half / (WIDTH // 2 - 1)).astype(np.uint16)
    return y, cb, cr


def make_source(path: Path, kind: str, frames: int) -> None:
    spec = KINDS[kind]
    y, cb, cr = picture()
    raw = (y.astype("<u2").tobytes() + cb.astype("<u2").tobytes() + cr.astype("<u2").tobytes()) * frames
    params = ["log-level=error", "lossless=1", "range=limited",
              *[f"{k}={v}" for k, v in spec["tags"].items()]]
    if "hdr" in spec:
        params.append(spec["hdr"])
    # The tags are also set on the frames (setparams, after the conversion to the output's
    # pixel format) and on the output: ffmpeg 9 converts the pixels to colour options given
    # on the output alone and writes only the tags the frames carry, which would make the
    # source something other than a limited-range PQ, HLG or BT.709 clip.
    from videowipe import colour

    tags = spec["tags"]
    labels = colour._labelled((), tags["colormatrix"], tags["colorprim"], tags["transfer"], "tv",
                              pix_fmt="yuv420p10le")
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s",
           f"{WIDTH}x{HEIGHT}", "-r", "25", "-i", "-", "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
           *labels, "-x265-params", ":".join(params), "-tag:v", "hvc1", str(path)]
    result = subprocess.run(cmd, input=raw, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-500:])


def probe(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
                          "-show_frames", "-read_intervals", "%+#1", "-of", "json", path],
                         capture_output=True, text=True)
    data = json.loads(out.stdout)
    stream = data["streams"][0]
    tags = {k: stream.get(k) for k in ("pix_fmt", "color_range", "color_space", "color_primaries",
                                       "color_transfer")}
    kinds = {s.get("side_data_type") for f in data.get("frames", [])[:1] for s in f.get("side_data_list", [])}
    tags["mastering_display"] = "Mastering display metadata" in kinds
    tags["content_light_level"] = "Content light level metadata" in kinds
    return tags


def decode_planes(path: str):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1", "-pix_fmt", "yuv420p10le",
                          "-f", "rawvideo", "-"], capture_output=True).stdout
    data = np.frombuffer(raw, "<u2")
    n = WIDTH * HEIGHT
    y = data[:n].reshape(HEIGHT, WIDTH).astype(np.int32)
    cb = data[n:n + n // 4].reshape(HEIGHT // 2, WIDTH // 2).astype(np.int32)
    cr = data[n + n // 4:].reshape(HEIGHT // 2, WIDTH // 2).astype(np.int32)
    return y, cb, cr


def longest_run(row: np.ndarray) -> int:
    changes = np.flatnonzero(np.diff(row) != 0)
    edges = np.concatenate([[-1], changes, [len(row) - 1]])
    return int(np.diff(edges).max())


def measure(source: str, output: str) -> dict:
    sy, scb, scr = (p.astype(np.int32) for p in picture())
    oy, ocb, ocr = decode_planes(output)
    keep = np.ones((HEIGHT, WIDTH), bool)
    x0, y0, x1, y1 = MASK_BOX
    keep[y0 - 8:y1 + 8, x0 - 8:x1 + 8] = False
    result = {}
    for name, (top, bottom) in ROWS.items():
        rows = np.zeros((HEIGHT, WIDTH), bool)
        rows[top:bottom] = True
        sel = rows & keep
        err = (oy - sy)[sel].astype(np.float64)
        entry = {"rms": round(float(np.sqrt((err ** 2).mean())), 2), "max": int(np.abs(err).max())}
        if name in ("ramp", "shallow"):
            middle = (top + bottom) // 2
            entry["levels"] = int(len(np.unique(oy[middle])))
            entry["source_levels"] = int(len(np.unique(sy[middle])))
            entry["longest_run"] = longest_run(oy[middle])
            entry["source_longest_run"] = longest_run(sy[middle])
        else:
            half = np.zeros((HEIGHT // 2, WIDTH // 2), bool)
            half[top // 2:bottom // 2] = True
            for label, a, b in (("cb", ocb, scb), ("cr", ocr, scr)):
                e = (a - b)[half & keep[::2, ::2]].astype(np.float64)
                entry[label + "_rms"] = round(float(np.sqrt((e ** 2).mean())), 2)
                entry[label + "_levels"] = int(len(np.unique(a[(top + bottom) // 4])))
        result[name] = entry
    result["tags"] = probe(output)
    return result


def run(kind: str, mode: str, frames: int, workdir: Path, rr) -> dict:
    os.environ["VIDEOWIPE_HIGH_BIT_DEPTH"] = mode
    source = workdir / f"{kind}_src.mp4"
    make_source(source, kind, frames)
    out_dir = workdir / f"out_{kind}_{mode}"
    out_dir.mkdir()
    output = Path(rr.run_pipeline(source, out_dir, MASK_BOX))
    files = {p.suffix: p for p in out_dir.iterdir() if p.suffix in (".mp4", ".mkv")}
    entry = {"kind": kind, "mode": mode, "source_tags": probe(str(source)),
             "messages": list(rr.PIPELINE_LOG), "outputs": {}}
    for suffix, path in sorted(files.items()):
        entry["outputs"][suffix] = measure(str(source), str(path))
        entry["outputs"][suffix]["bytes"] = path.stat().st_size
    return entry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kinds", default=",".join(KINDS), help="comma-separated: sdr10, pq, hlg")
    parser.add_argument("--modes", default="off,auto", help="VIDEOWIPE_HIGH_BIT_DEPTH values to run")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--json")
    parser.add_argument("--keep", help="keep the generated files in this directory")
    args = parser.parse_args(argv)
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        sys.exit("ffmpeg and ffprobe are required")
    rr = load_roundtrip()
    first = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    print(first)
    holder = tempfile.TemporaryDirectory() if not args.keep else None
    base = Path(args.keep or holder.name)
    base.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for kind in args.kinds.split(","):
            for mode in args.modes.split(","):
                workdir = base / f"{kind}_{mode}"
                workdir.mkdir(exist_ok=True)
                entry = run(kind, mode, args.frames, workdir, rr)
                results.append(entry)
                print(f"\n== {kind}, VIDEOWIPE_HIGH_BIT_DEPTH={mode}")
                print("  source tags:", entry["source_tags"])
                for line in entry["messages"]:
                    print("  >", line)
                for suffix, m in entry["outputs"].items():
                    print(f"  {suffix} ({m['bytes']} bytes) tags: {m['tags']}")
                    for part in ("ramp", "shallow", "chroma"):
                        print(f"    {part:8s} {m[part]}")
    finally:
        if holder is not None:
            holder.cleanup()
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
