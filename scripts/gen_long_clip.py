#!/usr/bin/env python3
"""Generate a long synthetic clip with subtitle lines at KNOWN times.

The clip is a moving, textured, text-free background (ffmpeg ``gradients`` plus
a ``geq`` plasma) with three burned-in text tracks rendered by libass from a
generated ASS file:

* ``dialogue``  bottom-centre lines of 1-5 s with realistic gaps, separated by
  several long pauses (20-120 s) and one stretch with no subtitle text at all;
* ``caption``   a top-centre track that is visible only a small share of the
  time (about 10-20 %);
* ``watermark`` a persistent semi-transparent top-right mark, on for the whole
  clip.

Everything is derived from ``--seed`` so the same arguments always give the
same timeline.  Alongside ``<name>.mp4`` the script writes ``<name>.ass`` and
``<name>.truth.json``: every line's frame range and glyph rectangle, measured
from a text-only render of the same ASS file (so rectangles are real libass
output, not estimates).  ``scripts/eval_sampling.py`` scores a plan against
that truth file.

Line times are whole frames and ``--fps`` must divide 100 (5, 10, 20, 25, ...),
so ASS centisecond rounding never moves a line by a frame.  Lines are one row
by default (``--two-line-share`` adds two-row lines, whose upper row the
frequency-map candidate does not cover; that is a spatial matter, separate from
the temporal sampling this clip is meant to measure).

The clips are synthetic.  They exercise the sampling mechanism with exact
ground truth; they are not a substitute for real footage.  Do not commit the
videos.

Example::

    python scripts/gen_long_clip.py --out /tmp/clips --name long5 --duration 300
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

WORDS = (
    "about after again always another around because before between brother "
    "careful change country daughter distance early evening every family father "
    "friend garden happy heavy himself important island justice kitchen letter "
    "listen morning mother mountain never nothing office people picture promise "
    "question quiet remember river secret silver simple station summer teacher "
    "tonight together understand village weather window winter without wonder "
    "yesterday young answer believe bridge candle coffee dinner engine forest "
    "golden hidden journey killer lonely market narrow orange pocket reason "
    "shadow temple travel useful visit wander yellow zero"
).split()

BG_W, BG_H = 640, 360
FONTS_DIR = "/usr/share/fonts/truetype/dejavu"

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Dialogue,DejaVu Sans,22,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,1.6,0,2,20,20,24,1
Style: Caption,DejaVu Sans,20,&H0000E6FF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,1.6,0,8,20,20,30,1
Style: Mark,DejaVu Sans,16,&H66FFFFFF,&H000000FF,&H66000000,&H00000000,-1,0,0,0,100,100,0,0,1,1.0,0,9,16,16,14,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_time(seconds: float) -> str:
    cs = int(round(seconds * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


class Grid:
    """Rounds times to whole frames."""

    def __init__(self, fps: int) -> None:
        self.fps = fps

    def __call__(self, value: float) -> float:
        return round(value * self.fps) / self.fps


def _text(rng: random.Random, lo: int, hi: int) -> str:
    words = [rng.choice(WORDS) for _ in range(rng.randint(lo, hi))]
    words[0] = words[0].capitalize()
    text = " ".join(words)
    return text + rng.choice([".", ".", "?", "!", "..."])


def build_timeline(
    duration: float, seed: int, pause_share: float = 0.30,
    caption_every: tuple[float, float] = (8.0, 30.0),
    fps: int = 10, two_line_share: float = 0.0,
) -> dict:
    """Return the line timeline for ``duration`` seconds (deterministic)."""
    _r1 = Grid(fps)
    rng = random.Random(seed)
    k = max(1, round(duration / 150.0))
    total_pause = duration * pause_share
    weights = [rng.uniform(0.5, 1.5) for _ in range(k)]
    scale = total_pause / sum(weights)
    pauses_len = [min(120.0, max(20.0, w * scale)) for w in weights]
    if duration < 90:
        pauses_len = [min(length, duration * 0.25) for length in pauses_len]
    speech_total = duration - sum(pauses_len)
    speech_w = [rng.uniform(0.6, 1.4) for _ in range(k + 1)]
    speech_len = [speech_total * w / sum(speech_w) for w in speech_w]

    t = 0.0
    speech: list[tuple[float, float]] = []
    pauses: list[tuple[float, float]] = []
    for index in range(k + 1):
        speech.append((_r1(t), _r1(t + speech_len[index])))
        t += speech_len[index]
        if index < k:
            pauses.append((_r1(t), _r1(t + pauses_len[index])))
            t += pauses_len[index]
    # The longest pause is the stretch with no text at all (no captions either).
    no_text = max(pauses, key=lambda p: p[1] - p[0])

    lines: list[dict] = []
    for start, end in speech:
        cursor = start + rng.uniform(0.5, 2.0)
        while cursor < end - 1.0:
            length = _r1(rng.uniform(1.0, 5.0))
            first, last = _r1(cursor), _r1(cursor + length)
            if last > end:
                break
            text = _text(rng, 2, 8)
            if two_line_share and len(text) > 34 and rng.random() < two_line_share:
                cut = text.rfind(" ", 0, len(text) // 2 + 6)
                if cut > 0:
                    text = text[:cut] + "\\N" + text[cut + 1:]
            lines.append({"track": "dialogue", "start": first, "end": last, "text": text})
            if rng.random() < 0.7:
                gap = rng.uniform(0.2, 1.2)
            else:
                gap = rng.uniform(1.2, 3.5)
            cursor = last + gap

    cursor = rng.uniform(*caption_every) * 0.5
    while cursor < duration - 4.0:
        length = _r1(rng.uniform(2.0, 4.0))
        first, last = _r1(cursor), _r1(cursor + length)
        if not (last > no_text[0] and first < no_text[1]):
            lines.append({
                "track": "caption", "start": first, "end": last,
                "text": _text(rng, 2, 4),
            })
        cursor = last + rng.uniform(*caption_every)

    for index, line in enumerate(sorted(lines, key=lambda x: (x["start"], x["track"]))):
        line["id"] = f"L{index:04d}"
    lines.sort(key=lambda x: x["id"])
    return {
        "duration": duration,
        "seed": seed,
        "speech": [list(s) for s in speech],
        "pauses": [list(p) for p in pauses],
        "no_text_stretch": list(no_text),
        "lines": lines,
    }


def write_ass(timeline: dict, path: Path) -> None:
    rows = [ASS_HEADER.format(w=BG_W, h=BG_H)]
    rows.append(
        f"Dialogue: 0,{_ass_time(0)},{_ass_time(timeline['duration'] + 1.0)},"
        "Mark,,0,0,0,,@videowipe.test\n"
    )
    for line in timeline["lines"]:
        style = "Dialogue" if line["track"] == "dialogue" else "Caption"
        rows.append(
            f"Dialogue: 0,{_ass_time(line['start'])},{_ass_time(line['end'])},"
            f"{style},,0,0,0,,{line['text']}\n"
        )
    path.write_text("".join(rows), encoding="utf-8")


def _ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-nostdin", "-nostats", "-loglevel", "error", "-y", *args]
    subprocess.run(cmd, check=True)


def _subtitle_filter(ass: Path) -> str:
    # The path is a plain file name under a directory we created, so a single
    # level of quoting is enough for the filter graph.
    return f"subtitles=filename='{ass}':fontsdir='{FONTS_DIR}'"


def render_video(
    ass: Path, out: Path, duration: float, fps: int, width: int, height: int,
    seed: int, crf: int,
) -> None:
    scale = "" if (width, height) == (BG_W, BG_H) else f",scale={width}:{height}:flags=bicubic"
    graph = (
        "geq=lum='lum(X,Y)+34*sin(X/19+T*1.1)*sin(Y/13-T*0.7)"
        "+22*sin((X*0.6+Y)/9+T*2.1)':cb='cb(X,Y)':cr='cr(X,Y)',"
        "noise=alls=5:allf=t" + scale + "," + _subtitle_filter(ass)
    )
    _ffmpeg([
        "-f", "lavfi", "-i",
        f"gradients=s={BG_W}x{BG_H}:r={fps}:d={duration}:n=5:speed=0.04:seed={seed}",
        "-vf", graph, "-pix_fmt", "yuv420p", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", str(crf), "-g", "50",
        "-movflags", "+faststart", str(out),
    ])


def render_text_only(
    ass: Path, out: Path, duration: float, fps: int, width: int, height: int,
) -> None:
    _ffmpeg([
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}:d={duration}",
        "-vf", _subtitle_filter(ass), "-pix_fmt", "yuv420p", "-c:v", "libx264",
        "-qp", "8", "-preset", "veryfast", str(out),
    ])


def _glyph_rect(gray: np.ndarray, region: np.ndarray, floor: int, pad: int):
    ys, xs = np.where((gray > floor) & region)
    if len(xs) == 0:
        return None
    h, w = gray.shape
    return [
        max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
        min(w - 1, int(xs.max()) + pad), min(h - 1, int(ys.max()) + pad),
    ]


def measure_rects(text_video: Path, timeline: dict, fps: int) -> dict:
    """Glyph rectangle of every line, from a middle frame of the text-only render."""
    pad = 3
    cap = cv2.VideoCapture(str(text_video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {text_video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("cannot read first text-only frame")
    gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    corner = np.zeros_like(gray, dtype=bool)
    corner[: int(h * 0.15), int(w * 0.6):] = True
    mark = _glyph_rect(gray, corner, 40, pad)
    if mark is None:
        raise RuntimeError("watermark not visible in the first text-only frame")
    timeline["watermark"] = {"rect": mark, "first_frame": 0, "end_frame": total}

    wanted: dict[int, list[dict]] = {}
    for line in timeline["lines"]:
        first_frame = int(round(line["start"] * fps))
        last_frame = int(round(line["end"] * fps))
        line["first_frame"], line["end_frame"] = first_frame, min(last_frame, total)
        mid = (first_frame + max(first_frame + 1, last_frame)) // 2
        wanted.setdefault(min(mid, total - 1), []).append(line)
    position = 1  # frame 0 was consumed above
    for target in sorted(wanted):
        if target < position:
            cap.release()
            cap = cv2.VideoCapture(str(text_video))
            position = 0
        while position < target:
            if not cap.grab():
                raise RuntimeError("text-only render ended early")
            position += 1
        ok, frame = cap.read()
        position += 1
        if not ok:
            raise RuntimeError(f"cannot decode text-only frame {target}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for line in wanted[target]:
            region = np.zeros_like(gray, dtype=bool)
            if line["track"] == "dialogue":
                region[int(h * 0.55):, :] = True
            else:
                region[: int(h * 0.45), : max(1, mark[0] - 2 * pad)] = True
            rect = _glyph_rect(gray, region, 90, pad)
            if rect is None:
                raise RuntimeError(f"line {line['id']} has no visible glyphs at frame {target}")
            line["rect"] = rect
    cap.release()
    return timeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--name", default="clip", help="Base file name")
    parser.add_argument("--duration", type=float, required=True, help="Seconds")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--size", default="640x360", help="Output WxH (text is rendered at this size)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pause-share", type=float, default=0.30)
    parser.add_argument("--two-line-share", type=float, default=0.0,
                        help="Share of long dialogue lines set on two rows")
    parser.add_argument("--crf", type=int, default=24)
    args = parser.parse_args(argv)

    if shutil.which("ffmpeg") is None:
        print("ffmpeg is required", file=sys.stderr)
        return 2
    if args.fps <= 0 or 100 % args.fps:
        print("--fps must divide 100 so line times are exact centiseconds", file=sys.stderr)
        return 2
    width, height = (int(v) for v in args.size.lower().split("x"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = out / args.name
    duration = float(args.duration)

    timeline = build_timeline(
        duration, args.seed, pause_share=args.pause_share, fps=args.fps,
        two_line_share=args.two_line_share,
    )
    ass = base.with_suffix(".ass")
    write_ass(timeline, ass)
    text_only = base.with_suffix(".text.mp4")
    render_text_only(ass, text_only, duration, args.fps, width, height)
    timeline = measure_rects(text_only, timeline, args.fps)
    text_only.unlink()
    video = base.with_suffix(".mp4")
    render_video(ass, video, duration, args.fps, width, height, args.seed, args.crf)

    cap = cv2.VideoCapture(str(video))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    timeline.update({
        "fps": args.fps, "width": width, "height": height,
        "frame_count": frame_count, "video": video.name,
        "generator": "scripts/gen_long_clip.py",
    })
    with open(base.with_suffix(".truth.json"), "w", encoding="utf-8") as handle:
        json.dump(timeline, handle, indent=1)
    dialogue = [l for l in timeline["lines"] if l["track"] == "dialogue"]
    caption = [l for l in timeline["lines"] if l["track"] == "caption"]
    share = lambda ls: sum(l["end"] - l["start"] for l in ls) / duration
    print(
        f"{video}: {frame_count} frames, {len(dialogue)} dialogue lines "
        f"({share(dialogue):.0%} of time), {len(caption)} caption lines "
        f"({share(caption):.0%}), no-text stretch {timeline['no_text_stretch']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
