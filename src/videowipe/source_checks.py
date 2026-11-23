"""Warnings about sources the cleaned video cannot represent faithfully.

Frames are decoded to 8-bit BGR and encoded as 8-bit ``yuv420p`` with no tone
mapping, so a source with more than 8 bits per sample loses precision and an HDR
source (PQ or HLG transfer) comes out with wrong colours. Nothing here changes
the output: it only says so, once per source, as early as the source is read.
A source that cannot be probed is not reported.
"""

from __future__ import annotations

import json
import re
import subprocess

from videowipe.timing import _run

HDR_TRANSFERS = {"smpte2084": "PQ", "arib-std-b67": "HLG"}

_warned: set[str] = set()


def bit_depth(pix_fmt: str | None, bits_per_raw_sample=None) -> int:
    """Bits per sample from ffprobe's ``bits_per_raw_sample`` when it gives one,
    otherwise from the pixel format's name (``yuv420p10le``, ``p010le``,
    ``gbrp12be``, ``gray16le``); 8 when neither says more."""
    try:
        bits = int(bits_per_raw_sample)
    except (TypeError, ValueError):
        bits = 0
    if bits > 0:
        return bits
    name = pix_fmt or ""
    # Every format above 8 bits ends in its depth and byte order; requiring the
    # byte order keeps 8-bit names such as nv12 and nv16 out.
    match = re.search(r"(?:^p[024]|[a-z])(9|1[0246])(?:le|be)$", name)
    return int(match.group(1)) if match else 8


def describe(stream: dict) -> list[str]:
    """The warning lines for one ffprobe video stream (none for 8-bit SDR)."""
    lines = []
    transfer = HDR_TRANSFERS.get(stream.get("color_transfer"))
    depth = bit_depth(stream.get("pix_fmt"), stream.get("bits_per_raw_sample"))
    # With the bit-depth path (videowipe.highbit) the MP4 review copy is still 8-bit but
    # an HEVC copy keeps the source's bit depth where this ffmpeg can: for HDR sources
    # unless VIDEOWIPE_HIGH_BIT_DEPTH=off, for SDR ones when it is on.
    from videowipe.highbit import mode as high_bit_mode
    chosen = high_bit_mode()
    kept = (" The HEVC copy keeps the source's bit depth where this ffmpeg can "
            "(docs/high-bit-depth.md).")
    opt_in = (" Set VIDEOWIPE_HIGH_BIT_DEPTH=on to also write a 10-bit HEVC copy that keeps "
              "it (docs/high-bit-depth.md).")
    if depth <= 8 or chosen == "off":
        kept = opt_in = ""
    if transfer:
        lines.append(
            f"Warning: the source is HDR ({transfer}); the cleaned video is 8-bit SDR "
            "with no tone mapping, so its colours will be wrong." + kept
        )
    elif depth > 8:
        lines.append(
            f"Warning: the source is {depth}-bit; the cleaned video is made from 8-bit "
            "frames, so gradients may show banding." + (kept if chosen == "on" else opt_in)
        )
    return lines


def probe(path: str) -> dict | None:
    """The first video stream's pixel format, depth and transfer, or ``None``."""
    try:
        # Through timing._run, so code that replaces subprocess.run or Popen to
        # watch the pipeline's own commands does not see this probe.
        out = _run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=pix_fmt,bits_per_raw_sample,color_transfer", "-of", "json", path],
            timeout=30,
        )
        if out.returncode != 0:
            return None
        return json.loads(out.stdout)["streams"][0]
    except (OSError, ValueError, KeyError, IndexError, TypeError, subprocess.SubprocessError):
        return None


def warn_about_source(path: str) -> list[str]:
    """Print the warnings for *path* the first time it is seen in this process;
    returns the lines printed."""
    if path in _warned:
        return []
    _warned.add(path)
    stream = probe(path)
    lines = describe(stream) if stream else []
    for line in lines:
        print(line)
    return lines
