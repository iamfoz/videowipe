#!/usr/bin/env python
"""Record a performance baseline for VideoWipe on the machine running it.

Usage:
    python scripts/perf_baseline.py VIDEO [VIDEO ...] [-o OUTDIR]
        [--trial-seconds N] [--device auto|cpu|mps|cuda]
        [--full | --no-full] [--runs N]

For every video the script records the machine and the video, then drives the
public engine API the CLI uses (``WipeEngine.plan`` and ``WipeEngine.run``, the
same defaults as ``videowipe clean``):

1. planning: detection plus temporal refinement, with the per-phase detector
   counters from ``WipeEngine.last_detection_stats``;
2. optionally (``--trial-seconds N``) a trial of N seconds placed at the END of
   the video, to measure the cost of positioning a trial late in a file;
3. unless ``--no-full``, a full clean, with the inpainting timer split that
   ``benchmark.json`` carries (decode, resize, model, blend, pipe write ...).

Each stage uses its own ``WipeEngine`` (so model loading is visible in the
stage that pays for it, as in a fresh CLI run); process-wide warm-up such as
the first Metal kernels is only paid by run 1, so use ``--runs 3`` for a
steady-state median. Everything is written to ``OUTDIR/perf_baseline.json``
and a readable summary is printed. Anything that cannot be determined (no
torch, no ffprobe, not macOS ...) is recorded as ``"unavailable"``.

Nothing here changes a result: it is the measurement driver for the plan in
docs/PRD-performance-quality.md (P0-3). No dependencies beyond VideoWipe's own.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

try:
    import resource
except ImportError:  # Windows
    resource = None

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

# Measure the code beside this script, not whichever copy happens to be installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2

from videowipe.api import WipeRequest
from videowipe.engine import WipeEngine
from videowipe.plan import JSON_FILENAME

SCHEMA_VERSION = 1
UNAVAILABLE = "unavailable"

PLAN_PHASES = (
    "coarse_sampling", "band_fallback", "refine", "boundary_probe",
    "descender_recovery",
)
_BUILD_KEYWORDS = (
    "Parallel framework", "FFMPEG", "avcodec", "avformat", "avutil", "swscale",
    "VideoToolbox", "AVFoundation", "GStreamer", "OpenCL", "Threads",
)
_ENCODERS = ("libx264", "libx265", "h264_videotoolbox", "hevc_videotoolbox")


# ── machine and video facts ──────────────────────────────────────────────────

def _run(command: list[str], timeout: float = 20) -> str | None:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _sysctl(name: str) -> str | None:
    if platform.system() != "Darwin":
        return None
    out = _run(["sysctl", "-n", name], timeout=5)
    return out.strip() if out and out.strip() else None


def _sysctl_int(name: str):
    value = _sysctl(name)
    try:
        return int(value) if value is not None else UNAVAILABLE
    except ValueError:
        return UNAVAILABLE


def _chip() -> str:
    brand = _sysctl("machdep.cpu.brand_string")
    if brand:
        return brand
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.lower().startswith(("model name", "hardware", "cpu model")):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or UNAVAILABLE


def _total_ram_bytes():
    memsize = _sysctl("hw.memsize")
    if memsize and memsize.isdigit():
        return int(memsize)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return UNAVAILABLE


def _first_line(text: str | None) -> str:
    return text.strip().splitlines()[0] if text and text.strip() else UNAVAILABLE


def _torch_facts() -> dict:
    try:
        import torch
    except Exception as exc:  # ImportError, or a broken binary install
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
    facts: dict = {"available": True, "version": getattr(torch, "__version__", UNAVAILABLE)}
    for key, probe in (
        ("mps_built", lambda: torch.backends.mps.is_built()),
        ("mps_available", lambda: torch.backends.mps.is_available()),
        ("cuda_available", lambda: torch.cuda.is_available()),
        ("threads", lambda: torch.get_num_threads()),
    ):
        try:
            facts[key] = probe()
        except Exception:
            facts[key] = UNAVAILABLE
    try:
        if torch.cuda.is_available():
            facts["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return facts


def _onnxruntime_facts() -> dict:
    try:
        import onnxruntime
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
    try:
        providers = list(onnxruntime.get_available_providers())
    except Exception:
        providers = UNAVAILABLE
    return {"available": True, "version": getattr(onnxruntime, "__version__", UNAVAILABLE),
            "providers": providers}


def _opencv_facts() -> dict:
    try:
        info = cv2.getBuildInformation().splitlines()
    except Exception:
        info = []
    lines = [
        " ".join(line.split()) for line in info
        if any(keyword in line for keyword in _BUILD_KEYWORDS)
    ]
    facts = {"version": cv2.__version__, "build_information": lines or UNAVAILABLE}
    try:
        facts["threads"] = cv2.getNumThreads()
    except Exception:
        pass
    return facts


def _ffmpeg_facts() -> dict:
    version = _run(["ffmpeg", "-version"], timeout=10)
    if version is None:
        return {"available": False, "version": UNAVAILABLE,
                "encoders": {name: UNAVAILABLE for name in _ENCODERS}}
    encoders = _run(["ffmpeg", "-hide_banner", "-encoders"], timeout=20)
    names = {}
    for name in _ENCODERS:
        if encoders is None:
            names[name] = UNAVAILABLE
        else:
            names[name] = any(name in line.split() for line in encoders.splitlines())
    return {
        "available": True,
        "version": _first_line(version),
        "ffprobe": _first_line(_run(["ffprobe", "-version"], timeout=10)),
        "encoders": names,
    }


def collect_machine_facts() -> dict:
    """Machine, runtime and tool facts; every unobtainable item is "unavailable"."""
    perf = _sysctl_int("hw.perflevel0.logicalcpu")
    efficiency = _sysctl_int("hw.perflevel1.logicalcpu")
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or UNAVAILABLE,
        "chip": _chip(),
        "logical_cores": os.cpu_count() or UNAVAILABLE,
        "physical_cores": _sysctl_int("hw.physicalcpu"),
        "performance_cores": perf,
        "efficiency_cores": efficiency,
        "ram_bytes": _total_ram_bytes(),
        "python": platform.python_version(),
        "torch": _torch_facts(),
        "onnxruntime": _onnxruntime_facts(),
        "opencv": _opencv_facts(),
        "ffmpeg": _ffmpeg_facts(),
        "environment": {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith(("VIDEOWIPE_", "OMP_", "MKL_", "PYTORCH_", "ORT_"))
        },
    }


def _rate(text) -> float | None:
    try:
        num, _, den = str(text).partition("/")
        return float(num) / float(den or 1) if float(den or 1) else None
    except ValueError:
        return None


def probe_video(path: str) -> dict:
    """Container, codec, geometry, rate and length; ffprobe when present."""
    facts: dict = {
        "path": os.path.abspath(path),
        "size_bytes": os.path.getsize(path) if os.path.exists(path) else UNAVAILABLE,
    }
    cap = cv2.VideoCapture(path)
    try:
        if cap.isOpened():
            facts["opencv"] = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
                "frame_count": int((cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) + 0.5),
            }
        else:
            facts["opencv"] = UNAVAILABLE
    finally:
        cap.release()
    raw = _run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
               timeout=60)
    facts["ffprobe"] = UNAVAILABLE
    if raw:
        try:
            info = json.loads(raw)
            streams = info.get("streams", [])
            video = next((s for s in streams if s.get("codec_type") == "video"), {})
            fmt = info.get("format", {})
            facts["ffprobe"] = {
                "container": fmt.get("format_name", UNAVAILABLE),
                "container_long_name": fmt.get("format_long_name", UNAVAILABLE),
                "duration_s": _number(fmt.get("duration")),
                "bit_rate": _number(fmt.get("bit_rate")),
                "codec": video.get("codec_name", UNAVAILABLE),
                "profile": video.get("profile", UNAVAILABLE),
                "pix_fmt": video.get("pix_fmt", UNAVAILABLE),
                "width": video.get("width", UNAVAILABLE),
                "height": video.get("height", UNAVAILABLE),
                "r_frame_rate": video.get("r_frame_rate", UNAVAILABLE),
                "avg_frame_rate": video.get("avg_frame_rate", UNAVAILABLE),
                "nb_frames": _number(video.get("nb_frames")),
                "color_transfer": video.get("color_transfer", UNAVAILABLE),
                "audio_streams": [s.get("codec_name") for s in streams
                                  if s.get("codec_type") == "audio"],
            }
        except (ValueError, TypeError, AttributeError):
            facts["ffprobe"] = UNAVAILABLE
    probe = facts["ffprobe"] if isinstance(facts["ffprobe"], dict) else {}
    cv = facts["opencv"] if isinstance(facts["opencv"], dict) else {}
    facts["container"] = probe.get("container", Path(path).suffix.lstrip(".") or UNAVAILABLE)
    facts["codec"] = probe.get("codec", UNAVAILABLE)
    facts["width"] = cv.get("width") or probe.get("width", UNAVAILABLE)
    facts["height"] = cv.get("height") or probe.get("height", UNAVAILABLE)
    facts["fps"] = cv.get("fps") or _rate(probe.get("avg_frame_rate")) or UNAVAILABLE
    facts["frame_count"] = cv.get("frame_count") or probe.get("nb_frames") or UNAVAILABLE
    duration = probe.get("duration_s")
    if not isinstance(duration, (int, float)) and isinstance(facts["fps"], (int, float)) \
            and isinstance(facts["frame_count"], int) and facts["fps"] > 0:
        duration = round(facts["frame_count"] / facts["fps"], 3)
    facts["duration_s"] = duration if isinstance(duration, (int, float)) else UNAVAILABLE
    return facts


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return UNAVAILABLE
    return int(number) if number.is_integer() else round(number, 4)


# ── running the stages ───────────────────────────────────────────────────────

def _peak_rss_mib():
    if resource is None:
        return UNAVAILABLE
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(raw / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(mapping) -> dict:
    return {key: value for key, value in dict(mapping).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)}


class _PhaseClock:
    """Wall time between engine progress events, grouped by event phase."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self._phase = None
        self._since = 0.0

    def __call__(self, event) -> None:
        now = time.perf_counter()
        if self._phase is not None:
            self.seconds[self._phase] = self.seconds.get(self._phase, 0.0) + (now - self._since)
        self._phase, self._since = event.phase, now

    def finish(self) -> dict:
        self(type("Done", (), {"phase": None})())
        return {key: round(value, 4) for key, value in self.seconds.items()}


def _engine(device: str):
    return WipeEngine(task="clean", device=device)


def _cleanup(engine) -> None:
    try:
        engine.cleanup()
    except Exception as exc:  # a failed release must not lose the measurements
        print(f"warning: engine cleanup failed: {exc}", file=sys.stderr)


def _identity(engine) -> dict | str:
    """Backend, device and precision as the engine reports them."""
    try:
        identity = engine._trial_identity()  # what the web server uses for its cache key
    except Exception as exc:
        return f"{UNAVAILABLE}: {type(exc).__name__}: {exc}"[:200]
    keys = ("backend", "device", "precision", "torch", "onnxruntime", "threads", "gap",
            "ref_length", "neighbor_stride", "feather", "input_size", "encoding")
    return {key: identity[key] for key in keys if key in identity}


def run_plan(video: str, out_dir: str, device: str) -> dict:
    """Detection + refinement through ``WipeEngine.plan``."""
    engine = _engine(device)
    clock = _PhaseClock()
    record: dict = {}
    started = time.perf_counter()
    try:
        plan = engine.plan(WipeRequest(video=video, output_dir=out_dir), on_progress=clock)
        record["wall_s"] = round(time.perf_counter() - started, 4)
        record["plan_path"] = os.path.join(out_dir, JSON_FILENAME)
        record["tracks"] = len(plan.tracks)
        record["remove_tracks"] = len(plan.remove_tracks)
        record["remove_segments"] = [
            [track.id, segment.start, segment.end]
            for track in plan.remove_tracks for segment in track.segments
        ]
        record["warnings"] = list(plan.warnings)
    except Exception as exc:
        record["wall_s"] = round(time.perf_counter() - started, 4)
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["progress_phase_s"] = clock.finish()
        record["detection_stats"] = engine.last_detection_stats
        _cleanup(engine)
    return record


def run_inpaint(video: str, out_dir: str, plan_path: str, device: str,
                trial_range=None) -> dict:
    """A full clean or a trial from a saved plan, through ``WipeEngine.run``."""
    engine = _engine(device)
    record: dict = {}
    if trial_range is not None:
        record["range"] = [int(trial_range[0]), int(trial_range[1])]
    started = time.perf_counter()
    try:
        result = engine.run(WipeRequest(
            video=video, output_dir=out_dir, plan=plan_path,
            trial_range=tuple(record["range"]) if trial_range is not None else None,
        ))
        record["wall_s"] = round(time.perf_counter() - started, 4)
        record["timing"] = _numeric(result.timings)
        record["backend_label"] = result.backend
        record["output_path"] = result.output_path
        if os.path.isfile(result.output_path):
            record["output_bytes"] = os.path.getsize(result.output_path)
            record["output_sha256"] = _sha256(result.output_path)
        record["warnings"] = list(result.warnings)
        record["engine_identity"] = _identity(engine)
    except Exception as exc:
        record["wall_s"] = round(time.perf_counter() - started, 4)
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _cleanup(engine)
    return record


def trial_interval(frame_count: int, fps: float, seconds: float) -> tuple[int, int]:
    """``[first, last)`` covering the last ``seconds`` of the video."""
    length = max(1, min(frame_count, round(seconds * fps)))
    return frame_count - length, frame_count


# ── aggregation ──────────────────────────────────────────────────────────────

def median_tree(items: list):
    """Median of every numeric leaf across same-shaped records."""
    first = items[0]
    if isinstance(first, dict):
        keys = list(dict.fromkeys(key for item in items if isinstance(item, dict) for key in item))
        return {key: median_tree([item[key] for item in items
                                  if isinstance(item, dict) and key in item]) for key in keys}
    numbers = [v for v in items if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if numbers and len(numbers) == len(items):
        middle = statistics.median(numbers)
        if all(isinstance(v, int) for v in numbers) and float(middle).is_integer():
            return int(middle)
        return round(float(middle), 4)
    return first


def _ratio(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def derived_ratios(stats: dict) -> dict:
    """Forward passes per refined frame, recovery share and hit rate."""
    out = {}
    for label, rows in (("refine", [stats.get("refine")]),
                        ("all_phases", [stats.get(p) for p in stats])):
        rows = [row for row in rows if row]
        if not rows:
            continue
        total = lambda key: sum(row.get(key, 0) for row in rows)  # noqa: E731
        refined = total("frames_converted")
        out[label] = {
            "frames_converted": refined,
            "forward_passes_per_frame": _ratio(total("forward_passes"), refined),
            "forward_passes_per_detect_call": _ratio(total("forward_passes"), total("detect_calls")),
            "share_of_frames_reaching_recovery": _ratio(total("recovery_frames"), refined),
            "recovery_hit_rate": _ratio(total("recovery_hits"), total("recovery_calls")),
            "recovery_share_of_detect_calls": _ratio(total("recovery_detect_calls"),
                                                     total("detect_calls")),
        }
    return out


def _row(stage, seconds, frames=None, level=0):
    return {"stage": stage, "level": level, "seconds": round(float(seconds), 4),
            "frames": frames,
            "ms_per_frame": _ratio(seconds * 1000, frames) if frames else None}


def inpaint_rows(record: dict, frames: int, label: str) -> list[dict]:
    """Stage rows for a clean or trial record (levels give the nesting)."""
    timing = record.get("timing", {})
    get = lambda key: float(timing.get(key, 0.0))  # noqa: E731
    wall = float(record.get("wall_s", 0.0))
    written = int(timing.get("frames_written", frames))
    rows = [_row(f"{label}: model load", get("model_load_s"))]
    if timing.get("prefix_frames"):
        rows.append(_row(f"{label}: position (skip to start)", get("prefix_skip_s"),
                         int(timing["prefix_frames"])))
    inpaint = get("inpainting_s")
    rows.append(_row(f"{label}: inpainting", inpaint, written))
    prediction, composition = get("prediction_s"), get("composition_s")
    rows.append(_row("prediction", prediction, written, 1))
    for key, name in (("decode_infer_s", "decode (inference reader)"),
                      ("resize_s", "resize to 640x120"),
                      ("cache_read_s", "prediction cache read"),
                      ("model_s", "model inference"),
                      ("cache_write_s", "prediction cache write")):
        rows.append(_row(name, get(key), written, 2))
    rows.append(_row("composition", composition, written, 1))
    for key, name in (("decode_output_s", "decode (output reader)"),
                      ("blend_s", "blend"), ("pipe_write_s", "pipe write")):
        rows.append(_row(name, get(key), written, 2))
    rows.append(_row("mask evaluation", get("mask_eval_s"), written, 1))
    rows.append(_row("ffmpeg finalise", get("encoding_finalize_s"), None, 1))
    rows.append(_row("other", inpaint - prediction - composition - get("mask_eval_s")
                     - get("encoding_finalize_s"), None, 1))
    rows.append(_row(f"{label}: other (plan load, hashing, setup)",
                     wall - get("model_load_s") - get("prefix_skip_s") - inpaint))
    return rows


def plan_rows(record: dict, frames: int) -> list[dict]:
    stats = record.get("detection_stats", {})
    wall = float(record.get("wall_s", 0.0))
    rows = [_row("planning (detect + refine)", wall, frames)]
    covered = 0.0
    for name in PLAN_PHASES:
        row = stats.get(name)
        if not row:
            continue
        covered += row.get("wall_s", 0.0)
        rows.append(_row(name, row.get("wall_s", 0.0),
                         int(row.get("frames_grabbed", 0) + row.get("frames_converted", 0)), 1))
    rows.append(_row("other (candidates, plan build, files)", wall - covered, None, 1))
    return rows


def build_summary(run: dict, frames: int) -> dict:
    """Stage tables with shares of the total, plus the derived detector ratios."""
    summary: dict = {}
    main = []
    if run.get("plan"):
        main += plan_rows(run["plan"], frames)
    if run.get("clean"):
        main += inpaint_rows(run["clean"], frames, "clean")
    # Total is planning wall + clean wall; level-0 rows are exclusive parts of it.
    total = float(run.get("plan", {}).get("wall_s", 0.0)) + float(run.get("clean", {}).get("wall_s", 0.0))
    for row in main:
        row["share"] = _ratio(row["seconds"], total)
    summary["stages"] = main
    summary["total_s"] = round(total, 4)
    if run.get("trial"):
        trial = inpaint_rows(run["trial"], int(run["trial"].get("timing", {}).get("frames_written", 0)
                                               or 0), "trial")
        trial_total = float(run["trial"].get("wall_s", 0.0))
        for row in trial:
            row["share"] = _ratio(row["seconds"], trial_total)
        summary["trial_stages"] = trial
        summary["trial_total_s"] = round(trial_total, 4)
    stats = run.get("plan", {}).get("detection_stats", {})
    summary["detection_ratios"] = derived_ratios(stats) if stats else {}
    return summary


# ── report printing ──────────────────────────────────────────────────────────

def _fmt(value, spec=".2f"):
    if value is None:
        return "-"
    return format(value, "d" if isinstance(value, int) else spec)


def print_stage_table(title: str, rows: list[dict], total_s: float) -> None:
    print(f"\n{title}")
    print(f"{'stage':<46}{'seconds':>10}{'share':>8}{'frames':>9}{'ms/frame':>10}")
    for row in rows:
        indent = "  " * row["level"]
        share = "-" if row.get("share") is None else f"{row['share'] * 100:.1f}%"
        print(f"{indent + row['stage']:<46}{row['seconds']:>10.2f}{share:>8}"
              f"{_fmt(row['frames'], 'd') if row['frames'] else '-':>9}"
              f"{_fmt(row['ms_per_frame']):>10}")
    print(f"{'TOTAL':<46}{total_s:>10.2f}")


def print_counters(stats: dict) -> None:
    columns = (("grab", "frames_grabbed"), ("conv", "frames_converted"),
               ("detect", "detect_calls"), ("fwd", "forward_passes"),
               ("rec.fr", "recovery_frames"), ("rec.call", "recovery_calls"),
               ("rec.det", "recovery_detect_calls"), ("rec.hit", "recovery_hits"),
               ("rec.skip", "recovery_variants_skipped"),
               ("present", "frames_present"),
               ("crops", "band_crops_stored"), ("2nd.pass", "band_second_pass"))
    print("\nDetection counters per planning phase")
    print(f"{'phase':<20}{'wall_s':>8}" + "".join(f"{name:>9}" for name, _ in columns))
    for phase, row in stats.items():
        print(f"{phase:<20}{row.get('wall_s', 0):>8.2f}"
              + "".join(f"{int(row.get(key, 0)):>9d}" for _, key in columns))


def print_summary(video: dict, summary: dict, run: dict) -> None:
    name = os.path.basename(video["path"])
    print(f"\n=== {name}: {video.get('width')}x{video.get('height')} "
          f"{video.get('codec')} {video.get('fps')} fps, {video.get('frame_count')} frames "
          f"({video.get('duration_s')} s) ===")
    for stage in ("plan", "trial", "clean"):
        if run.get(stage, {}).get("error"):
            print(f"  {stage} FAILED: {run[stage]['error']}")
    if summary["stages"]:
        print_stage_table("Stages (share of planning + clean)", summary["stages"], summary["total_s"])
    if "trial_stages" in summary:
        print_stage_table(f"Trial {run['trial'].get('range')} (share of trial)",
                          summary["trial_stages"], summary["trial_total_s"])
    stats = run.get("plan", {}).get("detection_stats", {})
    if stats:
        print_counters(stats)
        for label, ratios in summary["detection_ratios"].items():
            pretty = ", ".join(f"{key}={_fmt(value, '.3f')}" for key, value in ratios.items())
            print(f"  ratios [{label}]: {pretty}")
    identity = run.get("clean", {}).get("engine_identity") or run.get("trial", {}).get("engine_identity")
    if identity:
        print(f"  engine: {identity}")


# ── main ─────────────────────────────────────────────────────────────────────

def baseline_video(path: str, out_root: str, args) -> dict:
    video = probe_video(path)
    frames = video["frame_count"] if isinstance(video["frame_count"], int) else 0
    fps = video["fps"] if isinstance(video["fps"], (int, float)) else 0.0
    stem = Path(path).stem
    runs = []
    for index in range(1, args.runs + 1):
        run_dir = os.path.join(out_root, stem, f"run-{index:03d}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"\n--- {os.path.basename(path)}: run {index}/{args.runs} ---", flush=True)
        run: dict = {"run": index}
        run["plan"] = run_plan(path, os.path.join(run_dir, "plan"), args.device)
        plan_path = run["plan"].get("plan_path")
        plan_ok = "error" not in run["plan"] and plan_path
        if args.trial_seconds and plan_ok and frames and fps:
            first, last = trial_interval(frames, fps, args.trial_seconds)
            run["trial"] = run_inpaint(path, os.path.join(run_dir, "trial"), plan_path,
                                       args.device, (first, last))
            segments = run["plan"].get("remove_segments", [])
            run["trial"]["overlaps_remove_segment"] = any(
                start < last and end > first for _, start, end in segments)
        elif args.trial_seconds and plan_ok:
            run["trial"] = {"error": "frame count or fps unavailable; trial skipped"}
        if args.full and plan_ok:
            run["clean"] = run_inpaint(path, os.path.join(run_dir, "clean"), plan_path, args.device)
        run["peak_rss_mib_so_far"] = _peak_rss_mib()
        run["summary"] = build_summary(run, frames)
        runs.append(run)
        print_summary(video, run["summary"], run)
    result = {"video": video, "runs": runs}
    if len(runs) > 1:
        usable = [{k: v for k, v in run.items() if k not in ("summary", "run")} for run in runs
                  if not any("error" in run.get(s, {}) for s in ("plan", "trial", "clean"))]
        if usable:
            median = median_tree(usable)
            median["summary"] = build_summary(median, frames)
            result["median"] = median
            print(f"\n=== median of {len(usable)} run(s) ===")
            print_summary(video, median["summary"], median)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record a VideoWipe performance baseline (P0-3).",
    )
    parser.add_argument("videos", nargs="+", metavar="VIDEO", help="Video file(s) to measure")
    parser.add_argument("-o", "--output", default="result/perf-baseline",
                        help="Output directory (default: result/perf-baseline)")
    parser.add_argument("--trial-seconds", type=float, default=0, metavar="N",
                        help="Also run an N-second trial placed at the end of each video")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                        help="Torch device (default: auto; ignored by ONNX weights)")
    parser.add_argument("--full", action=argparse.BooleanOptionalAction, default=True,
                        help="Run a full clean after planning (default: yes)")
    parser.add_argument("--runs", type=int, default=1, help="Repeat N times and report the median")
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.trial_seconds < 0:
        parser.error("--trial-seconds must not be negative")
    missing = [v for v in args.videos if not os.path.isfile(v)]
    if missing:
        parser.error(f"video not found: {', '.join(missing)}")

    os.makedirs(args.output, exist_ok=True)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv] if argv is None else ["perf_baseline.py", *argv],
        "options": {"device": args.device, "full": args.full, "runs": args.runs,
                    "trial_seconds": args.trial_seconds},
        "machine": collect_machine_facts(),
        "videos": [],
    }
    report_path = os.path.join(args.output, "perf_baseline.json")

    def write():
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

    machine = report["machine"]
    print(f"Machine: {machine['chip']} | {machine['logical_cores']} logical cores | "
          f"RAM {machine['ram_bytes']} bytes | python {machine['python']}")
    print(f"torch: {machine['torch']} | onnxruntime: {machine['onnxruntime']}")
    failed = False
    for path in args.videos:
        entry = baseline_video(path, args.output, args)
        report["videos"].append(entry)
        write()
        failed |= any("error" in run.get(stage, {}) for run in entry["runs"]
                      for stage in ("plan", "trial", "clean"))
    write()
    print(f"\nReport written to {report_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
