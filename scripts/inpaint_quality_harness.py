"""Ground-truth harness for STTN inpainting-quality experiments.

Real footage that is free of text in a chosen band is cut into a short clip.
Subtitle-like text is burned into it with ffmpeg ``drawtext`` (white fill,
black outline; a CJK-capable font for CJK lines) and the unburned frames are
kept as ground truth.  The *real* ``STTNInpainter.inpaint`` is then run with
the exact glyph mask (dilated 7x7 as the detector does, feathered with the
pipeline's own 4 px recipe) and the output frames, captured just before
encoding, are scored against the truth with :mod:`inpaint_quality_metrics`.

Experiments are monkeypatched in (nothing upstream is edited): the harness
swaps the module's ``cv2`` for a shim that picks the resize filters, masks the
model input, and wraps the compositor for the seam colour match.  Predictions
are cached per (downscale, input-mask) pair so that variants that only differ
in the upscale or in compositing reuse one model run.

Sub-commands: ``build`` (make clips), ``run`` (score variants), ``sheet``
(contact sheets), ``report`` (tables).  Run with ``--help`` for details.
Work and results live outside the repository (``--work``).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import inpaint_quality_metrics as M  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "input" / "detext_examples"
CJK_FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"  # WenQuanYi Zen Hei
LATIN_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
WEIGHT = str(Path.home() / ".videowipe" / "weights" / "sttn.onnx")
FRAMES = 50
SWITCH = 25  # the burned line changes here, as consecutive subtitles do
FPS = 25
LINES = {
    "cjk": ("他们终于抓到了那个鬱鬱寡歡的組長", "龍騰虎躍驚濤駭浪的歲月已經過去"),
    "latin": ("I mean, are you going to use me like a napkin?", "Why napkin? Why not socks, Berk?"),
}

# name: (source video, output height, first frame, y fraction, language, style)
CONFIGS = {
    "e1": ("english1", 480, 60, 0.50, "cjk", "regular"),
    "e2": ("english1", 480, 250, 0.50, "latin", "regular"),
    "e3": ("english1", 480, 400, 0.46, "cjk", "thin"),
    "c1": ("chinese1", 1080, 150, 0.58, "cjk", "regular"),
    "c2": ("chinese1", 1080, 350, 0.60, "cjk", "thin"),
    "o1": ("others", 1080, 100, 0.50, "cjk", "regular"),
    "o2": ("others", 1080, 300, 0.52, "latin", "thin"),
    "k1": ("chinese1", 2160, 150, 0.58, "cjk", "regular"),
    "k2": ("chinese1", 2160, 350, 0.60, "cjk", "thin"),
}

# Variant = (downscale, upscale, model-input masking, seam match)
VARIANTS = {
    "current": ("linear", "linear", "off", "off"),
    "area": ("area", "linear", "off", "off"),
    "area_cubic": ("area", "cubic", "off", "off"),
    "area_lanczos": ("area", "lanczos", "off", "off"),
    "linear_cubic": ("linear", "cubic", "off", "off"),
    "masked_zero": ("linear", "linear", "zero", "off"),
    "masked_mean": ("linear", "linear", "mean", "off"),
    "masked_zero_area": ("area", "linear", "zero", "off"),
    "masked_telea": ("linear", "linear", "telea", "off"),
    "seam": ("linear", "linear", "off", "on"),
    # Same settings as "current" with its own prediction cache: a determinism check.
    "current_b": ("linear", "linear", "off", "off"),
}
INTERP = {
    "linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4,
}


# --------------------------------------------------------------------------
# clip construction
# --------------------------------------------------------------------------
def _run(cmd, **kw):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **kw)


def text_params(height: int, style: str) -> tuple[int, int]:
    """Font size and outline width for a frame height and style."""
    if style == "regular":
        size = round(height / 20)
        return size, max(1, round(size / 16))
    size = round(height / 28)
    return size, max(1, round(size / 40))


def _drawtext(workdir: Path, cfg, width: int, height: int, fill: str, border: str) -> str:
    _src, _res, _start, yfrac, lang, style = cfg
    size, borderw = text_params(height, style)
    font = CJK_FONT if lang == "cjk" else LATIN_FONT
    filters = []
    for i, line in enumerate(LINES[lang]):
        path = workdir / f"line{i}.txt"
        path.write_text(line, encoding="utf-8")
        enable = f"lt(n\\,{SWITCH})" if i == 0 else f"gte(n\\,{SWITCH})"
        filters.append(
            f"drawtext=fontfile={font}:textfile={path}:expansion=none:fontsize={size}"
            f":fontcolor={fill}:borderw={borderw}:bordercolor={border}"
            f":x=(w-text_w)/2:y=h*{yfrac}:enable='{enable}'"
        )
    return ",".join(filters)


def build_clip(name: str, work: Path) -> dict:
    """Create ``clean.mkv`` (truth), ``burned.mp4`` and ``meta.npz`` for a config."""
    cfg = CONFIGS[name]
    src, height, start, _y, _lang, _style = cfg
    cdir = work / "clips" / name
    meta_path = cdir / "meta.npz"
    if meta_path.exists():
        return load_meta(name, work)
    cdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(SAMPLES / f"{src}.mp4"))
    for _ in range(start):
        cap.grab()
    ok, first = cap.read()
    assert ok
    src_h, src_w = first.shape[:2]
    width = round(height * src_w / src_h)
    frames = [first]
    for _ in range(FRAMES - 1):
        ok, f = cap.read()
        assert ok
        frames.append(f)
    cap.release()
    if (width, height) != (src_w, src_h):
        frames = [cv2.resize(f, (width, height), interpolation=cv2.INTER_LANCZOS4) for f in frames]
    # Lossless truth.
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-", "-c:v", "ffv1",
         str(cdir / "clean.mkv")], stdin=subprocess.PIPE)
    for f in frames:
        enc.stdin.write(f.tobytes())
    enc.stdin.close()
    assert enc.wait() == 0
    del frames
    # Burned input, encoded as a real source would be.
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(cdir / "clean.mkv"),
          "-vf", _drawtext(cdir, cfg, width, height, "white", "black"),
          "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
          str(cdir / "burned.mp4")])
    # Exact glyph masks: the same text on black, white fill and white outline.
    _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
          "-i", f"color=c=black:s={width}x{height}:r={FPS},format=gray",
          "-vf", _drawtext(cdir, cfg, width, height, "white", "white"),
          "-frames:v", str(FRAMES), "-f", "rawvideo", "-pix_fmt", "gray", str(cdir / "glyph.raw")])
    raw = np.memmap(cdir / "glyph.raw", dtype=np.uint8, mode="r", shape=(FRAMES, height, width))
    glyphs = [np.array(raw[index] > 16) for index in (0, SWITCH)]  # limited-range black is 16
    del raw
    (cdir / "glyph.raw").unlink()
    union = glyphs[0] | glyphs[1]
    ys, xs = np.nonzero(union)
    y0, y1 = max(0, ys.min() - 24), min(height, ys.max() + 25)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    alphas = [pipeline_alpha(g) for g in glyphs]
    np.savez_compressed(
        meta_path, width=width, height=height, rows=[y0, y1], cols=[x0, x1],
        glyph0=glyphs[0][y0:y1], glyph1=glyphs[1][y0:y1],
        alpha0=alphas[0][y0:y1], alpha1=alphas[1][y0:y1])
    return load_meta(name, work)


def pipeline_alpha(glyph: np.ndarray) -> np.ndarray:
    """Detector-style 7x7 dilation, then plan._project_mask's 4 px feather."""
    dil = cv2.dilate(glyph.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    flat = dil.astype(np.float32)
    radius = 4
    kernel = max(3, radius * 2 + 1)
    blurred = cv2.GaussianBlur(flat, (kernel, kernel), radius / 2.0)
    return np.clip(np.where(flat >= 1.0, 1.0, blurred), 0.0, 1.0).astype(np.float32)


def load_meta(name: str, work: Path) -> dict:
    cdir = work / "clips" / name
    with np.load(cdir / "meta.npz") as z:
        meta = {k: z[k] for k in z.files}
    meta.update(name=name, dir=cdir, width=int(meta["width"]), height=int(meta["height"]),
                rows=tuple(int(v) for v in meta["rows"]), cols=tuple(int(v) for v in meta["cols"]))
    return meta


def read_rows(path: Path, rows: tuple[int, int], count: int = FRAMES) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out = []
    for _ in range(count):
        ok, f = cap.read()
        assert ok
        out.append(f[rows[0]:rows[1]].copy())
    cap.release()
    return out


# --------------------------------------------------------------------------
# seam colour match (experiment Q-9a)
# --------------------------------------------------------------------------
def seam_match(comp: np.ndarray, orig: np.ndarray, alpha: np.ndarray,
               ring: int = 14, sigma: float = 6.0) -> np.ndarray:
    """Add a smooth colour offset to the prediction so it meets the original.

    ``comp``/``orig`` are float32 (H, W, 3) of the same band, ``alpha`` the
    float32 (H, W) blend mask.  The low-pass of ``orig - comp`` measured in a
    ring of untouched pixels just outside the mask is spread over the mask by
    normalised convolution.  Only the low frequencies are touched.
    """
    core = alpha >= 0.999
    if not core.any():
        return comp
    ys, xs = np.nonzero(alpha > 0)
    pad = ring + int(4 * sigma)
    y0, y1 = max(0, ys.min() - pad), min(alpha.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(alpha.shape[1], xs.max() + pad + 1)
    a = alpha[y0:y1, x0:x1]
    c = comp[y0:y1, x0:x1]
    o = orig[y0:y1, x0:x1]
    dist = cv2.distanceTransform((a < 0.999).astype(np.uint8), cv2.DIST_L2, 3)
    ring_mask = ((a == 0) & (dist <= ring)).astype(np.float32)
    k = int(sigma * 6) | 1
    weight = cv2.GaussianBlur(ring_mask, (k, k), sigma)
    diff = (o - c) * ring_mask[..., None]
    offset = cv2.GaussianBlur(diff, (k, k), sigma) / np.maximum(weight, 1e-4)[..., None]
    offset *= (weight > 1e-3)[..., None]
    result = comp.copy()
    result[y0:y1, x0:x1] = c + offset * (a > 0)[..., None]
    return result


# --------------------------------------------------------------------------
# variant runner
# --------------------------------------------------------------------------
class _Cv2Shim:
    """Stand-in for ``cv2`` inside inpainters.sttn that applies the variant."""

    def __init__(self, variant, alpha_for_frame, model_size=(640, 120)):
        self.down, self.up, self.masking, _seam = VARIANTS[variant]
        self.alpha_for_frame = alpha_for_frame
        self.size = model_size
        self._hard: dict = {}

    def __getattr__(self, name):
        return getattr(cv2, name)

    def _model_mask(self, frame_index, from_h, to_h):
        key = (self.alpha_for_frame.line(frame_index), from_h, to_h)
        if key not in self._hard:
            band = (self.alpha_for_frame(frame_index)[from_h:to_h] >= 0.999).astype(np.float32)
            small = cv2.resize(band, self.size, interpolation=cv2.INTER_AREA)
            self._hard[key] = (small > 0)[..., None]
        return self._hard[key]

    def resize(self, src, dsize, *args, **kwargs):
        if "interpolation" in kwargs:  # the model-input downscale site
            out = cv2.resize(src, dsize, interpolation=INTERP[self.down])
            if self.masking == "off":
                return out
            caller = sys._getframe(1).f_locals
            hard = self._model_mask(caller["start_f"] + caller["local_index"],
                                    caller["from_h"], caller["to_h"])
            if self.masking == "telea":
                filled = cv2.inpaint(out, hard[..., 0].astype(np.uint8), 3, cv2.INPAINT_TELEA)
                return filled.astype(np.float32)
            out = out.astype(np.float32)
            if self.masking == "zero":
                fill = np.float32(127.5)  # 0.0 in the model's [-1, 1] space
            else:
                keep = ~hard[..., 0]
                fill = out[keep].mean(axis=0) if keep.any() else np.float32(127.5)
            return np.where(hard, fill, out).astype(np.float32)
        return cv2.resize(src, dsize, interpolation=INTERP[self.up])  # the prediction upscale


class _AlphaFrames:
    """Full-frame per-frame alpha, built on demand from the two line masks."""

    _videowipe_stable_masks = True

    def __init__(self, meta):
        self.meta = meta
        self.shape = (meta["height"], meta["width"])
        y0, y1 = meta["rows"]
        self.rows = slice(y0, y1)
        self.cache = {}

    def line(self, index):
        return 0 if index < SWITCH else 1

    def __call__(self, index):
        line = self.line(index)
        if line not in self.cache:
            full = np.zeros(self.shape, np.float32)
            full[self.rows] = self.meta[f"alpha{line}"]
            self.cache[line] = full
        return self.cache[line]

    def union(self):
        return np.maximum(self(0), self(SWITCH))


def run_variant(clip: dict, variant: str, work: Path, save_frames=(4, 12, 20, 28, 36, 45)) -> dict:
    """Run the real STTNInpainter for a variant and score the captured frames."""
    from videowipe.inpainters import sttn
    from videowipe.inpainters.base import InpaintJob

    down, up, masking, seam = VARIANTS[variant]
    cache_dir = work / "predcache" / clip["name"] / f"{down}-{masking}{'-b' if variant.endswith('_b') else ''}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir = work / "runs" / clip["name"] / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = _AlphaFrames(clip)
    rows = slice(*clip["rows"])
    captured: list[np.ndarray] = []
    original_blend = sttn._blend_frame_regions
    original_cv2 = sttn.cv2

    def wrapper(frame_ori, comps, modes, mask):
        if seam == "on":
            comps = list(comps)
            for k, ((top, bottom), comp) in enumerate(zip(modes, comps)):
                if comp is not None:
                    comps[k] = seam_match(comp, frame_ori[top:bottom].astype(np.float32),
                                          mask[top:bottom, :, 0])
        frame = original_blend(frame_ori, comps, modes, mask)
        captured.append(frame[rows].copy())
        return frame

    sttn.cv2 = _Cv2Shim(variant, alphas)
    sttn._blend_frame_regions = wrapper
    painter = sttn.STTNInpainter()
    painter.load(WEIGHT)
    video = str(clip["dir"] / "burned.mp4")
    reader = cv2.VideoCapture(video)
    started = time.monotonic()
    try:
        outcome = painter.inpaint(InpaintJob(
            video_path=video, reader=reader, mask=alphas.union()[:, :, None],
            frame_mask=alphas, output_dir=str(out_dir), width=clip["width"],
            height=clip["height"], fps=FPS, frame_count=FRAMES, gap=25,
            prediction_cache_dir=str(cache_dir)))
    finally:
        reader.release()
        painter.cleanup()
        sttn.cv2, sttn._blend_frame_regions = original_cv2, original_blend
    elapsed = time.monotonic() - started
    assert len(captured) == FRAMES, len(captured)
    truth = read_rows(clip["dir"] / "clean.mkv", clip["rows"])
    crop_alpha = [clip[f"alpha{alphas.line(i)}"] for i in range(FRAMES)]
    crop_glyph = [clip[f"glyph{alphas.line(i)}"] for i in range(FRAMES)]
    scores = M.score_clip(captured, truth, crop_alpha, crop_glyph)
    scores["seconds"] = round(elapsed, 1)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    for i in save_frames:
        cv2.imwrite(str(frames_dir / f"{i:03d}.png"), captured[i])
    np.save(out_dir / "output_crop.npy", np.stack(captured))
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=1))
    return scores


# --------------------------------------------------------------------------
# contact sheets
# --------------------------------------------------------------------------
def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (len(text) * 9 + 8, 18), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return img


def contact_sheet(clip: dict, variants: list[str], work: Path, dest: Path,
                  frames=(4, 28, 45), zoom: int = 3, width_px: int | None = None) -> Path:
    """Stack truth / input / each variant for a few frames, tight crop at zoom x."""
    x0, x1 = clip["cols"]
    span = width_px or (420 if clip["height"] >= 1080 else 280)
    if clip["height"] >= 2160:
        span = 560
    x1 = min(x1, x0 + span)
    rows = clip["rows"]
    truth = read_rows(clip["dir"] / "clean.mkv", rows)
    burned = read_rows(clip["dir"] / "burned.mp4", rows)
    sheet_rows = []
    for i in frames:
        panels = [("truth", truth[i]), ("input", burned[i])]
        for v in variants:
            image = cv2.imread(str(work / "runs" / clip["name"] / v / "frames" / f"{i:03d}.png"))
            panels.append((v, image))
        stack = [label(cv2.resize(p[:, x0:x1], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST),
                       f"{n} f{i}") for n, p in panels]
        sheet_rows.append(np.vstack(stack))
    sheet = np.hstack(sheet_rows)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), sheet)
    return dest


SHEET_SETS = {
    "q7_downscale": ["current", "area", "area_cubic"],
    "q8_masked_input": ["current", "masked_zero", "masked_mean", "masked_telea"],
    "q9a_seam_match": ["current", "seam"],
    "q9b_crop": ["current", "proto_band_linear", "proto_crop_linear", "proto_cropmin_linear"],
}


def write_sheets(work: Path, out: Path, clips: list[str]) -> Path:
    """One contact sheet per (question, clip) and a Markdown index with the metrics."""
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Inpainting-quality contact sheets", "",
             "Each sheet stacks, per column (frame), truth / burned input / the listed variants, "
             "cropped tightly to the text and enlarged 3x (nearest neighbour).", ""]
    for name in clips:
        clip = build_clip(name, work)
        for key, wanted in SHEET_SETS.items():
            have = [v for v in wanted if (work / "runs" / name / v / "frames").is_dir()]
            if len(have) < 2:
                continue
            dest = contact_sheet(clip, have, work, out / f"{key}_{name}.png")
            lines.append(f"- `{dest}`  ({name}: {', '.join(have)})")
    index = out / "INDEX.md"
    index.write_text("\n".join(lines) + "\n")
    return index


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "run", "sheet", "sheets"])
    ap.add_argument("--out", type=Path, help="sheets: output folder")
    ap.add_argument("--work", type=Path, required=True, help="scratch directory outside the repository")
    ap.add_argument("--clips", default=",".join(CONFIGS))
    ap.add_argument("--variants", default="current")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.command == "sheets":
        print(write_sheets(args.work, args.out, args.clips.split(",")))
        return
    for name in args.clips.split(","):
        if args.command == "build":
            build_clip(name, args.work)
            print("built", name)
            continue
        clip = build_clip(name, args.work)
        variants = args.variants.split(",")
        if args.command == "run":
            for v in variants:
                scores = run_variant(clip, v, args.work)
                print(name, v, json.dumps({k: round(x, 4) for k, x in scores.items()}), flush=True)
        else:
            dest = args.work / "results" / f"{name}_{'_'.join(variants)}.png"
            print(contact_sheet(clip, variants, args.work, dest))


if __name__ == "__main__":
    main()
