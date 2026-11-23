#!/usr/bin/env python3
"""Check that the ffmpeg reader converts with the matrix and range it is given, whatever
the clip's own colour labels say, and which of the two this ffmpeg follows.

``videowipe.decode.reader_command`` relabels the frames with the matrix and range it
converts with (``setparams``) and gives the same to the scaler (``in_color_matrix``,
``in_range``). ffmpeg 6.1.1 takes the scaler's options over the frames' own colour
properties; newer ffmpeg may do the opposite. This script builds clips of known YCbCr
patches labelled with the *wrong* matrix and range and decodes each four ways:

``full``
    the reader's own command (label and options). Must be right.
``label only``
    the scaler's options dropped, so the frames' labels are all there is: the case
    where the frames win. Must be right too, which is the point of the relabel.
``options only``
    the relabel dropped: right where the options win, wrong where the frames do.
``neither``
    both dropped: the clip as labelled, which is wrong by construction (the control
    that shows the clip's labels would change the picture).

Each is the mean absolute error, in 8-bit RGB code values, of the decoded patches
against what the matrix and range ask for (the reader's start-up check accepts up to
1.6). Usage::

    python scripts/verify_reader_override.py

Exit status 1 when ``full`` or ``label only`` is wrong for any case.
"""

from __future__ import annotations

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
from videowipe import colour, decode

SIZE = 256
#: (matrix, range the reader is asked for, the labels the clip carries instead)
MISLABELLED = [
    ("bt709", "tv", "smpte170m", "tv"),
    ("bt601", "tv", "bt709", "tv"),
    ("bt601", "pc", "bt2020nc", "tv"),
    ("bt2020", "pc", "bt709", "tv"),
]
#: shown but not required: reading a clip labelled full range as limited range. No
#: job asks for that (the reader follows a source's own range), and on ffmpeg 9.1
#: no label or option achieves it (measured on the reference Mac: about 7.5 out).
INFORMATIONAL = [
    ("bt709", "tv", "smpte170m", "pc"),
    ("bt601", "tv", "bt709", "pc"),
]
#: the start-up check's limit (``decode._reader_decodes_known_colours``)
LIMIT = 1.6


def probe_clip(folder, space, rng) -> str:
    """A clip of known YCbCr patches (``colour.PROBE_YUV``) labelled *space* and *rng*."""
    path = Path(folder) / f"probe_{space}_{rng}.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s",
           f"{SIZE}x{SIZE}", "-r", "25", "-i", "-", "-frames:v", "2", "-c:v", "libx264",
           "-preset", "ultrafast", "-crf", "1", "-pix_fmt", "yuv420p",
           *colour._labelled((), space, None, None, rng)]
    raw = colour.probe_yuv_frame(SIZE, SIZE)
    subprocess.run(cmd + [str(path)], input=raw * 2, check=True)
    return str(path)


def patch_error(frame_bytes, matrix, rng) -> float:
    """Mean absolute error (code values) of the patches of a decoded ``bgr24`` frame
    against what *matrix* and *rng* give for the probe's YCbCr."""
    frame = np.frombuffer(frame_bytes, np.uint8)[:SIZE * SIZE * 3]
    rgb = frame.reshape(SIZE, SIZE, 3)[..., ::-1].astype(np.float64)
    measured = np.array([rgb[y0:y1, x0:x1].mean(axis=(0, 1))
                         for x0, y0, x1, y1 in colour.probe_patch_rects(SIZE, SIZE)])
    return float(np.abs(decode._bgr_of(colour.PROBE_YUV, matrix, rng) - measured).mean())


def decode_with(cmd) -> bytes:
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def with_filters(cmd, keep):
    """*cmd* with only the ``-vf`` filters for which ``keep(filter)`` is true."""
    at = cmd.index("-vf") + 1
    return cmd[:at] + [",".join(f for f in cmd[at].split(",") if keep(f))] + cmd[at + 1:]


def variants(cmd) -> dict:
    """The reader's command and the three ways of taking part of it away."""
    no_options = with_filters(cmd, lambda f: not f.startswith("scale="))
    return {"full": cmd,
            "label only": no_options,
            "options only": with_filters(cmd, lambda f: not f.startswith("setparams=")),
            "neither": with_filters(no_options, lambda f: not f.startswith("setparams="))}


def measure(folder, matrix, rng, wrong_space, wrong_range) -> dict:
    """``{variant: error}`` for one case."""
    clip = probe_clip(folder, wrong_space, wrong_range)
    cmd = decode.reader_command(clip, matrix, rng)
    return {name: patch_error(decode_with(command), matrix, rng) for name, command in variants(cmd).items()}


def main() -> int:
    ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    print(ffmpeg)
    print(f"{'asked':14s} {'clip labelled':18s} " + " ".join(f"{n:>12s}" for n in ("full", "label only",
                                                                                   "options only", "neither")))
    failed = False
    with tempfile.TemporaryDirectory(prefix="videowipe-override-") as folder:
        for required, cases in ((True, MISLABELLED), (False, INFORMATIONAL)):
            for matrix, rng, space, wrong_range in cases:
                errors = measure(folder, matrix, rng, space, wrong_range)
                if required:
                    failed |= errors["full"] > LIMIT or errors["label only"] > LIMIT
                print(f"{matrix + ' ' + rng:14s} {space + ' ' + wrong_range:18s} "
                      + " ".join(f"{errors[n]:12.2f}" for n in ("full", "label only", "options only",
                                                                 "neither"))
                      + ("" if required else "   (not required)"))
    print("FAIL: the reader does not convert as asked" if failed else
          f"OK: full and label only are within {LIMIT}; 'options only' and 'neither' show what "
          "this ffmpeg follows")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
