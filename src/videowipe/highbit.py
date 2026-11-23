"""Keep 10-bit and HDR sources at their bit depth, in an HEVC copy.

Upstream decodes every source to 8-bit BGR, runs the model on it and encodes 8-bit
``yuv420p``. For a 10-bit or HDR source that quantises every pixel to 8 bits (banding
in gradients, PQ and HLG pictures most of all) and drops the tags and the HDR
metadata (``docs/high-bit-depth.md`` has the measurements).

The model still works on 8 bits (it was trained on them, and the region it paints is
small); only the *pipeline around it* changes, and only for sources above 8 bits:

1. The compositor's second frame reader (the one the output frames are composed
   from, :mod:`videowipe.decode`) decodes to 16-bit BGR (``bgr48le``) with the
   source's matrix and range, and hands the compositor its 8-bit rounding as before,
   with the 16-bit frame attached (:class:`HighFrame`).
2. After the compositor has painted the inpainted region into the 8-bit frame,
   :func:`merge` carries only the pixels that changed back into the 16-bit frame
   (each by its 8-bit change, times 256), so everything the model did not touch is
   the source's, to 16 bits, and the painted region is 8-bit precise.
3. The 16-bit frames go to ffmpeg as ``bgr48le``. The MP4 (H.264, 8-bit) stays the
   review copy the web UI plays, from the same frames. A second output, the HEVC
   file :mod:`videowipe.export` already writes (``.mkv``), is encoded 10-bit
   (``yuv420p10le``, Main 10) with the source's colour primaries, transfer and matrix
   and, for HDR10, its mastering display and content light level, which x265 writes
   from explicit parameters because raw frames carry no side data.

``VIDEOWIPE_HIGH_BIT_DEPTH`` is ``auto`` (default) or ``off`` (upstream's behaviour
exactly). An 8-bit source never takes this path, so its output is byte for byte what
it was. Where this ffmpeg cannot do it (checked by encoding known colours first) or
the job cannot use it (dual output, trial runs, a source not read by the ffmpeg
reader) one line says so and the job runs as before.
``VIDEOWIPE_HIGH_BIT_DEPTH_ENCODER`` (``auto``, ``libx265``, ``hevc_videotoolbox``)
overrides the HEVC encoder; ``auto`` is ``export.video_encoder()``'s choice, except
that an HDR10 source with static metadata uses libx265 when it is there, because
VideoToolbox has no way to be given the metadata here.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction

import cv2
import numpy as np

from videowipe import colour, export
from videowipe.source_checks import bit_depth
from videowipe.timing import _run

ENV_VAR = "VIDEOWIPE_HIGH_BIT_DEPTH"
ENCODER_ENV = "VIDEOWIPE_HIGH_BIT_DEPTH_ENCODER"
MODES = ("auto", "off")
ENCODERS = ("auto", "libx265", "hevc_videotoolbox")


def mode() -> str:
    """``auto`` (default) or ``off``, from ``VIDEOWIPE_HIGH_BIT_DEPTH``."""
    value = os.environ.get(ENV_VAR, "auto").strip().lower()
    return value if value in MODES else "auto"


def encoder_override() -> str:
    value = os.environ.get(ENCODER_ENV, "auto").strip().lower()
    return value if value in ENCODERS else "auto"


# -- frames -------------------------------------------------------------------------


#: swscale's 16-bit RGB is the 8-bit value times 256 (white is about 65283, not 65535),
#: so one 8-bit step of a painted pixel is 256 sixteen-bit steps
STEP = 256


class HighFrame(np.ndarray):
    """An 8-bit BGR frame (what the compositor works on) that carries the 16-bit
    frame it was rounded from as ``.high`` and a pristine copy of itself as ``.base``.
    Any array made from it by slicing or arithmetic has neither (``None``)."""

    high = None
    base = None


def make_high_frame(high: np.ndarray) -> HighFrame:
    """The :class:`HighFrame` of a 16-bit BGR frame (rounded to 8 bits)."""
    small = cv2.convertScaleAbs(high, alpha=1 / STEP).view(HighFrame)
    small.high = high
    small.base = np.array(small, copy=True).view(np.ndarray)
    return small


def merge(frame, original):
    """The frame to write: *frame* if *original* carries no 16-bit frame, else the
    16-bit frame with the pixels the compositor changed carried across.

    *frame* is the compositor's 8-bit result and *original* the frame it started
    from. A pixel that differs from the original's has the original's 16-bit value
    plus the change times 256 (swscale's 16-bit scale: 8-bit white is about 65280);
    a pixel that does not is the source's, untouched."""
    high = getattr(original, "high", None)
    if high is None:
        return frame
    base = original.base
    new = np.asarray(frame)
    diff = cv2.absdiff(new, base)
    rows = np.flatnonzero(diff.reshape(diff.shape[0], -1).any(axis=1))
    if not len(rows):
        return high
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    cols = np.flatnonzero(diff[y0:y1].reshape(y1 - y0, -1).any(axis=0))  # three per pixel
    x0, x1 = int(cols[0]) // 3, int(cols[-1]) // 3 + 1
    delta = (new[y0:y1, x0:x1].astype(np.int32) - base[y0:y1, x0:x1].astype(np.int32)) * STEP
    high[y0:y1, x0:x1] = np.clip(high[y0:y1, x0:x1].astype(np.int32) + delta, 0, 65535)
    return high


# -- the source ---------------------------------------------------------------------


@dataclass(frozen=True)
class HighSource:
    """What the plan needs to know about a source above 8 bits."""

    colour: colour.SourceColour
    depth: int
    mastering: tuple | None = None  # (Fraction x 10): rx ry gx gy bx by wx wy max min
    light: tuple | None = None  # (max content, max average)

    @property
    def hdr(self) -> bool:
        return self.colour.hdr


def _fraction(text):
    try:
        return Fraction(str(text))
    except (ValueError, ZeroDivisionError):
        return None


def parse_source(text: str) -> HighSource | None:
    """A :class:`HighSource` from ``ffprobe -show_streams -show_frames`` JSON, or ``None``."""
    try:
        data = json.loads(text)
        stream = data["streams"][0]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    base = colour.parse_probe(json.dumps({"streams": [stream]}))
    if base is None:
        return None
    mastering = light = None
    for frame in data.get("frames", [])[:1]:
        for side in frame.get("side_data_list", []):
            kind = side.get("side_data_type", "")
            if kind == "Mastering display metadata":
                keys = ("red_x", "red_y", "green_x", "green_y", "blue_x", "blue_y",
                        "white_point_x", "white_point_y", "max_luminance", "min_luminance")
                values = tuple(_fraction(side.get(k)) for k in keys)
                mastering = values if all(v is not None for v in values) else None
            elif kind == "Content light level metadata":
                try:
                    light = (int(side["max_content"]), int(side["max_average"]))
                except (KeyError, ValueError, TypeError):
                    light = None
    return HighSource(base, bit_depth(stream.get("pix_fmt"), stream.get("bits_per_raw_sample")),
                      mastering, light)


@functools.lru_cache(maxsize=32)
def _probe(path: str, size: int, mtime_ns: int) -> HighSource | None:
    try:
        out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
                    "-show_frames", "-read_intervals", "%+#1", "-of", "json", path], timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_source(out.stdout.decode("utf-8", "replace")) if out.returncode == 0 else None


def probe_source(path: str) -> HighSource | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return _probe(os.path.realpath(path), stat.st_size, stat.st_mtime_ns)


def x265_mastering(values) -> str:
    """x265's ``master-display`` string from ffprobe's fractions (primaries in
    1/50000, luminance in 1/10000 cd/m2)."""
    rx, ry, gx, gy, bx, by, wx, wy, high, low = values

    def p(v):
        return int(round(v * 50000))

    def lum(v):
        return int(round(v * 10000))

    return (f"G({p(gx)},{p(gy)})B({p(bx)},{p(by)})R({p(rx)},{p(ry)})WP({p(wx)},{p(wy)})"
            f"L({lum(high)},{lum(low)})")


# -- the plan -----------------------------------------------------------------------


@dataclass
class HighBitPlan:
    """What one job does about bit depth. ``active`` means the second reader decodes to
    16 bits and the HEVC output is 10-bit; ``applied`` that the ffmpeg command was
    rewritten to take 16-bit frames (the reader only produces them then)."""

    active: bool = False
    applied: bool = False
    path: str = ""
    matrix: str = "bt709"
    range: str = "tv"
    encoder: str = "libx265"
    tags: dict = field(default_factory=dict)
    x265_params: tuple = ()
    description: str = ""

    def rewrite(self, cmd):
        """*cmd* with its raw input read as ``bgr48le``, or ``None`` when it is not
        the shape this code knows."""
        try:
            at = cmd.index("-i")
            pix = cmd.index("-pix_fmt")
        except ValueError:
            return None
        if not (pix < at and cmd[pix + 1] == "bgr24" and cmd[at + 1] == "-"):
            return None
        out = list(cmd)
        out[pix + 1] = "bgr48le"
        return out

    def command(self, cmd):
        if not self.active:
            return cmd
        rewritten = self.rewrite(cmd)
        if rewritten is None:
            print("Warning: could not keep the source's bit depth with this ffmpeg command; "
                  "the cleaned video is 8-bit.")
            self.active = False
            return cmd
        self.applied = True
        print(f"Keeping the source's bit depth ({self.description}).")
        return rewritten


#: the plan of the job running now, by source path (one job runs at a time); the
#: reader and the HEVC output look it up
_current: dict[str, HighBitPlan] = {}


def current(path: str) -> HighBitPlan | None:
    return _current.get(os.path.realpath(path)) if path else None


def encoder_for(source: HighSource) -> str | None:
    """The HEVC encoder for *source*: ``libx265`` or ``hevc_videotoolbox``, or ``None``."""
    wanted = encoder_override()
    has_x265 = "libx265" in export._encoder_help("libx265")
    if wanted == "libx265":
        return "libx265" if has_x265 else None
    if wanted == "hevc_videotoolbox":
        return "hevc_videotoolbox" if export._videotoolbox_works() else None
    try:
        auto = export.video_encoder()
    except RuntimeError:
        return None
    if auto == "hevc_videotoolbox" and source.mastering is not None and has_x265:
        return "libx265"  # only x265 can be given the HDR10 metadata here
    return auto


def _x265_params(source: HighSource) -> tuple:
    params = []
    c = source.colour
    if source.hdr and c.transfer == "smpte2084":
        params += ["hdr10=1", "hdr10-opt=1", "repeat-headers=1"]
        if source.mastering is not None:
            params.append("master-display=" + x265_mastering(source.mastering))
        if source.light is not None:
            params.append(f"max-cll={source.light[0]},{source.light[1]}")
    return tuple(params)


def video_filter(matrix: str, pix_fmt: str, tags: dict) -> str:
    """The ``-vf`` of the 10-bit output: 16-bit BGR to *pix_fmt* with *matrix* (limited
    range), then the labels (``setparams`` as colour.md explains)."""
    chain = [f"scale=out_color_matrix={matrix}:out_range=tv", f"format={pix_fmt}"]
    label = colour._frame_tags(tags.get("-colorspace"), tags.get("-color_primaries"),
                               tags.get("-color_trc"), "tv")
    if label:
        chain.append(label)
    return ",".join(chain)


def output_options(plan: HighBitPlan, preset) -> list[str]:
    """The video options of the 10-bit HEVC output (without the maps and the file)."""
    args = export.video_args(plan.encoder, preset)
    pix_fmt = args[args.index("-pix_fmt") + 1]
    if plan.encoder == "libx265":
        at = args.index("-x265-params")
        args[at + 1] = ":".join(["log-level=error", *plan.x265_params])
    # video_args ends with "-tag:v hvc1 -color_range tv"; the tags go in before it
    # (-color_range tv stays: the copy is limited range by design)
    tail = args.index("-tag:v")
    args[tail:tail] = ["-vf", video_filter(plan.matrix, pix_fmt, plan.tags),
                       *colour._tag_options(plan.tags.get("-colorspace"),
                                            plan.tags.get("-color_primaries"),
                                            plan.tags.get("-color_trc"), None)]
    return args


def output_args(source_path: str, mp4_path: str, height: int):
    """The second ffmpeg output for *source_path* when this job keeps its bit depth,
    else ``None`` (:func:`videowipe.export.hevc_output_args` then does what it did)."""
    plan = current(source_path)
    if plan is None or not plan.active:
        return None
    preset = export.choose_preset(height)
    tracks = export.probe_audio(source_path)
    args = ["-map", "0:v"]
    for i in range(len(tracks)):
        args += ["-map", f"1:a:{i}"]
    args += output_options(plan, preset) + export.audio_args(tracks, preset)
    return args + [export.hevc_path(mp4_path)]


# -- the capability check -----------------------------------------------------------

#: 16-bit RGB of known colours, and the YCbCr they must come out as (10-bit, any matrix)
_CHECK_COLOURS = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128), (255, 255, 255),
                           (194, 150, 130), (16, 16, 16)], np.uint8)


#: the largest 10-bit YCbCr error :func:`check_pipeline` accepts (12 is three 8-bit codes:
#: another swscale rounds chroma a code or two differently; a wrong matrix is 30 or more out)
CHECK_TOLERANCE = 12.0


def check_pipeline(plan: HighBitPlan, preset) -> bool:
    """Encode known 16-bit colours with this job's output options and read the file
    back: the stream must be 10-bit with the tags asked for, and its YCbCr must be
    the 10-bit values of ``plan.matrix`` (within :data:`CHECK_TOLERANCE` of 1023)."""
    width, height, stripe = 224, 64, 32
    frame = np.zeros((height, width, 3), np.uint16)
    for i, rgb in enumerate(_CHECK_COLOURS):
        frame[:, i * stripe:(i + 1) * stripe] = rgb.astype(np.uint16)[::-1] * 257  # BGR
    options = output_options(plan, preset)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr48le", "-s",
           f"{width}x{height}", "-r", "25", "-i", "-", "-map", "0:v", *options]
    with tempfile.TemporaryDirectory(prefix="videowipe-highbit-") as folder:
        path = os.path.join(folder, "check.mkv")
        try:
            if _run(cmd + [path], input=frame.tobytes() * 2, timeout=180).returncode != 0:
                return False
            info = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=pix_fmt,color_range,color_space,color_primaries,color_transfer",
                         "-of", "json", path], timeout=60)
            planes = _run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1", "-pix_fmt",
                           "yuv420p10le", "-f", "rawvideo", "-"], timeout=60)
            if info.returncode != 0 or planes.returncode != 0:
                return False
            stream = json.loads(info.stdout)["streams"][0]
        except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError):
            return False
    if bit_depth(stream.get("pix_fmt")) != 10 or stream.get("color_range") != "tv":
        return False
    seen = {"-colorspace": stream.get("color_space"), "-color_primaries": stream.get("color_primaries"),
            "-color_trc": stream.get("color_transfer")}
    if any(seen[k] != v for k, v in plan.tags.items() if k in seen):
        return False
    data = np.frombuffer(planes.stdout, np.uint16)
    n = width * height
    if len(data) != n + n // 2:
        return False
    y = data[:n].reshape(height, width)
    cb = data[n:n + n // 4].reshape(height // 2, width // 2)
    cr = data[n + n // 4:].reshape(height // 2, width // 2)
    got = np.array([(y[16:48, i * stripe + 8:i * stripe + 24].mean(),
                     cb[8:24, i * stripe // 2 + 4:i * stripe // 2 + 12].mean(),
                     cr[8:24, i * stripe // 2 + 4:i * stripe // 2 + 12].mean())
                    for i in range(len(_CHECK_COLOURS))])
    expected = colour.rgb_to_yuv(_CHECK_COLOURS, plan.matrix, "tv") * 4
    return float(np.abs(expected - got).max()) <= CHECK_TOLERANCE


@functools.lru_cache(maxsize=16)
def _checked(source: HighSource, matrix: str, rng: str, encoder: str, x265: tuple, tags_key: tuple,
             height: int) -> bool:
    plan = HighBitPlan(True, matrix=matrix, range=rng, encoder=encoder, x265_params=x265,
                       tags=dict(tags_key))
    return check_pipeline(plan, export.choose_preset(height))


# -- deciding -----------------------------------------------------------------------


def describe(source: HighSource, encoder: str) -> str:
    c = source.colour
    kind = {"smpte2084": "HDR10 (PQ)", "arib-std-b67": "HLG"}.get(c.transfer, "SDR")
    return (f"{source.depth}-bit {kind}, {c.space or 'untagged matrix'}; HEVC Main 10 copy by "
            f"{encoder} from 16-bit frames; the MP4 review copy stays 8-bit")


def plan_high_bit_depth(job) -> HighBitPlan:
    """Decide, once per job, whether the cleaned video keeps the source's bit depth,
    and say so when it should but cannot."""
    from videowipe.decode import FfmpegFrameReader

    path = getattr(job, "video_path", None)
    plan = HighBitPlan(path=os.path.realpath(path) if path else "")
    if path:
        _current[plan.path] = plan
    if mode() == "off" or not path or not os.path.isfile(path):
        return plan
    source = probe_source(path)
    if source is None or source.depth <= 8:
        return plan  # 8-bit (or unreadable: colour.py says so): upstream's pipeline, untouched

    def refuse(why: str):
        print(f"Warning: {why}; the cleaned video is 8-bit.")
        return plan

    if getattr(job, "dual", False):
        return refuse("the source is above 8 bits but the dual (side by side) output is 8-bit only")
    if getattr(job, "trial_range", None) is not None:
        return plan  # a trial is a preview: 8-bit, as ever
    reader = getattr(job, "reader", None)
    if not isinstance(reader, FfmpegFrameReader):
        return refuse("the source is above 8 bits but is not read by the ffmpeg reader "
                      "(VIDEOWIPE_DECODE)")
    if not source.colour.pix_fmt.startswith(("yuv", "nv", "p0", "p2")):
        return plan
    encoder = encoder_for(source)
    if encoder is None:
        return refuse("the source is above 8 bits but this ffmpeg has no 10-bit HEVC encoder")
    c = source.colour
    tags = {"-colorspace": c.space, "-color_primaries": c.primaries, "-color_trc": c.transfer}
    tags = {k: v for k, v in tags.items() if v is not None}
    x265 = _x265_params(source) if encoder == "libx265" else ()
    if source.mastering is not None and encoder != "libx265" and c.transfer == "smpte2084":
        print("Warning: this encoder cannot be given the source's HDR10 mastering display and "
              "content light level; the HEVC copy has the other tags only.")
    if not _checked(source, reader.matrix, "tv", encoder, x265, tuple(sorted(tags.items())),
                    c.height):
        return refuse("this ffmpeg did not encode known 16-bit colours as 10-bit with the source's "
                      "tags")
    plan.active = True
    plan.matrix, plan.range, plan.encoder = reader.matrix, reader.range, encoder
    plan.tags, plan.x265_params = tags, x265
    plan.description = describe(source, encoder)
    return plan
