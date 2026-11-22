"""Optional H.265 (HEVC) export alongside the H.264 review copy.

The STTN pipeline pipes cleaned frames into one ffmpeg process that writes an
H.264/AAC MP4. The web UI plays that MP4 in the browser for review, so it is
always kept. When the H.265 output codec is selected, the same ffmpeg process
gets a second output: an HEVC/Opus MKV encoded directly from the cleaned
frames (not re-encoded from the H.264 copy).

Encoder settings follow the user's HandBrake presets marked [VTB]:

=====================  =========  =============  ===============
Preset                 Max height  VT quality     Audio (Opus)
=====================  =========  =============  ===============
480p Stereo DVD        576        50             80 kbps stereo
720p Stereo            720        45             80 kbps stereo
1080p Stereo           larger     45             80 kbps stereo
1080p Surround         larger     45             256 kbps 5.1
=====================  =========  =============  ===============

All use 10-bit HEVC (Main 10), limited colour range, MKV. VideoToolbox
(``hevc_videotoolbox``) is used when ffmpeg has it and it works on this
machine; otherwise ``libx265`` with an approximately equivalent CRF.
Resolution is never changed (the presets' autocrop and size limits are not
applied: VideoWipe keeps the source frame geometry). Audio tracks that are
already Opus are copied; others are encoded per track, choosing the stereo or
surround settings from the track's channel count.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from subprocess import run as _run  # bound at import: unaffected by test patches

CODECS = ("h264", "h265")
HEVC_SUFFIX = ".mkv"


def _default_codec() -> str:
    value = os.environ.get("VIDEOWIPE_OUTPUT_CODEC", "h264").strip().lower()
    return value if value in CODECS else "h264"


_lock = threading.Lock()
_selected: list[str] = []  # stack of explicitly selected codecs (one job runs at a time)


def current_codec() -> str:
    """The output codec for runs started now (innermost ``output_codec``)."""
    with _lock:
        return _selected[-1] if _selected else _default_codec()


@contextmanager
def output_codec(codec: str | None):
    """Select the output codec for runs inside the block. ``None`` keeps the
    current setting."""
    if codec is None:
        yield
        return
    codec = codec.lower()
    if codec not in CODECS:
        raise ValueError(f"unknown output codec {codec!r}; choose from {', '.join(CODECS)}")
    with _lock:
        _selected.append(codec)
    try:
        yield
    finally:
        with _lock:
            _selected.remove(codec)


def hevc_path(mp4_path: str) -> str:
    """Path of the HEVC export that accompanies an H.264 review MP4."""
    return os.path.splitext(mp4_path)[0] + HEVC_SUFFIX


# --- Presets -----------------------------------------------------------------


@dataclass(frozen=True)
class Preset:
    name: str
    vt_quality: int  # HandBrake VideoQualitySlider for VideoToolbox (0-100)
    x265_crf: int  # libx265 fallback: approximate equivalent
    stereo_kbps: int = 80
    surround_kbps: int = 256


PRESETS = {
    "480p": Preset("x265 [480p] [Stereo] [DVD] [VTB]", vt_quality=50, x265_crf=20),
    "720p": Preset("x265 [720p] [Stereo] [VTB]", vt_quality=45, x265_crf=22),
    "1080p": Preset("x265 [1080p] [Stereo|Surround] [VTB]", vt_quality=45, x265_crf=22),
}


def choose_preset(height: int) -> Preset:
    if height <= 576:
        return PRESETS["480p"]
    if height <= 720:
        return PRESETS["720p"]
    return PRESETS["1080p"]


# --- Probing -------------------------------------------------------------------


def _capture(cmd: list[str], timeout: int = 60) -> str | None:
    try:
        result = _run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    out = result.stdout if isinstance(result.stdout, str) else ""
    err = result.stderr if isinstance(result.stderr, str) else ""
    return out + err if result.returncode == 0 else None


def probe_audio(path: str) -> list[dict]:
    """Audio streams of *path*: ``[{"codec": "aac", "channels": 2}, ...]``."""
    out = _capture([
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=codec_name,channels", "-of", "json", path,
    ])
    if not out:
        return []
    try:
        streams = json.loads(out).get("streams", [])
    except ValueError:
        return []
    return [{"codec": str(s.get("codec_name", "")), "channels": int(s.get("channels") or 2)}
            for s in streams]


@functools.lru_cache(maxsize=None)
def _encoder_help(name: str) -> str:
    return _capture(["ffmpeg", "-hide_banner", "-h", f"encoder={name}"]) or ""


@functools.lru_cache(maxsize=None)
def _videotoolbox_works() -> bool:
    """True when hevc_videotoolbox exists *and* can encode 10-bit constant-
    quality HEVC here (missing on non-Apple builds; constant quality needs
    Apple Silicon)."""
    if "hevc_videotoolbox" not in _encoder_help("hevc_videotoolbox"):
        return False
    return _capture([
        "ffmpeg", "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
        "testsrc=size=256x144:rate=25", "-frames:v", "2", "-c:v", "hevc_videotoolbox",
        "-q:v", "50", "-profile:v", "main10", "-pix_fmt", "p010le", "-f", "null", "-",
    ], timeout=60) is not None


def video_encoder() -> str:
    """``hevc_videotoolbox`` when usable, else ``libx265``."""
    if _videotoolbox_works():
        return "hevc_videotoolbox"
    if "libx265" in _encoder_help("libx265"):
        return "libx265"
    raise RuntimeError(
        "H.265 output needs an ffmpeg with hevc_videotoolbox (macOS) or libx265"
    )


# --- ffmpeg arguments ------------------------------------------------------------


def video_args(encoder: str, preset: Preset) -> list[str]:
    if encoder == "hevc_videotoolbox":
        args = ["-c:v", "hevc_videotoolbox", "-q:v", str(preset.vt_quality),
                "-profile:v", "main10", "-pix_fmt", "p010le"]
        if "prio_speed" in _encoder_help("hevc_videotoolbox"):
            args += ["-prio_speed", "0"]  # HandBrake VideoPreset "quality"
    else:
        args = ["-c:v", "libx265", "-crf", str(preset.x265_crf), "-preset", "medium",
                "-profile:v", "main10", "-pix_fmt", "yuv420p10le",
                "-x265-params", "log-level=error"]
    return args + ["-tag:v", "hvc1", "-color_range", "tv"]


def audio_args(tracks: list[dict], preset: Preset) -> list[str]:
    """Per-track Opus settings for output audio streams 0..n-1."""
    args: list[str] = []
    for i, track in enumerate(tracks):
        if track["codec"] == "opus":
            args += [f"-c:a:{i}", "copy"]
            continue
        args += [f"-c:a:{i}", "libopus"]
        if track["channels"] > 2:
            # Surround preset: 5.1 mixdown at 256 kbps. aformat also avoids
            # libopus rejecting layouts such as 5.1(side).
            args += [f"-b:a:{i}", f"{preset.surround_kbps}k",
                     f"-filter:a:{i}", "aformat=channel_layouts=5.1"]
        else:
            args += [f"-b:a:{i}", f"{preset.stereo_kbps}k"]
    return args


def hevc_output_args(source_path: str, mp4_path: str, height: int) -> list[str]:
    """Extra ffmpeg output for the HEVC export, or ``[]`` when H.264 is selected.

    Appended after the H.264 output of a command whose input 0 is the cleaned
    raw video and input 1 the original file (for its audio)."""
    # a source above 8 bits keeps its bit depth in this output (videowipe.highbit),
    # which is then written whichever codec was chosen
    from videowipe.highbit import output_args as high_bit_args

    kept = high_bit_args(source_path, mp4_path, height)
    if kept is not None:
        return kept
    if current_codec() != "h265":
        return []
    preset = choose_preset(height)
    tracks = probe_audio(source_path)
    args = ["-map", "0:v"]
    for i in range(len(tracks)):
        args += ["-map", f"1:a:{i}"]
    args += video_args(video_encoder(), preset) + audio_args(tracks, preset)
    return args + [hevc_path(mp4_path)]


def keep_only_hevc(output_dir: str, since: float) -> list[str]:
    """CLI helper: once an HEVC export exists, the H.264 review copy written in
    the same run is redundant (only the web UI plays it). Remove each ``.mp4``
    whose ``.mkv`` sibling was written at or after *since*; return the MKVs."""
    kept = []
    try:
        names = os.listdir(output_dir)
    except OSError:
        return kept
    for name in names:
        if not name.endswith(".mp4"):
            continue
        mp4 = os.path.join(output_dir, name)
        mkv = hevc_path(mp4)
        try:
            if os.path.getmtime(mkv) >= since - 1 and os.path.getsize(mkv) > 0:
                os.remove(mp4)
                kept.append(mkv)
                print(f"Saved H.265 output to {mkv}")
        except OSError:
            continue
    return kept


_EXPORT_JS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "export.js")


def inject_ui(html: str) -> str:
    """Add the output-format control script before ``</body>`` (web UI)."""
    try:
        with open(_EXPORT_JS, encoding="utf-8") as fh:
            script = fh.read()
    except OSError:
        return html
    tag = f"<script>{script}</script>\n"
    at = html.lower().rfind("</body>")
    return html[:at] + tag + html[at:] if at != -1 else html + tag


def install(app) -> None:
    """Add the output-format control to the web UI page served at ``/``.

    Registered from ``server/app.py`` with a single line, as an HTTP
    middleware, so the upstream index route stays untouched."""
    from starlette.responses import Response

    @app.middleware("http")
    async def _videowipe_export_ui(request, call_next):
        response = await call_next(request)
        if (request.method != "GET" or request.url.path != "/" or response.status_code != 200
                or not response.headers.get("content-type", "").startswith("text/html")):
            return response
        body = b"".join([chunk async for chunk in response.body_iterator])
        headers = {k: v for k, v in response.headers.items()
                   if k.lower() not in {"content-length", "etag", "last-modified"}}
        return Response(inject_ui(body.decode("utf-8")), status_code=200,
                        headers=headers, media_type="text/html")
