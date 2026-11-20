"""Source timestamps survive cleaning of variable-frame-rate video.

The model is replaced by a stub (as in ``test_frame_identity.py``); everything
else is the real ``STTNInpainter.inpaint`` path with a real ffmpeg.
"""
import json
import shutil
import subprocess
import wave
from fractions import Fraction

import cv2
import numpy as np
import pytest

from videowipe import fidelity, timing
from videowipe.inpainters import sttn
from videowipe.inpainters.base import InpaintJob
from videowipe.tasks.base import read_frame_info

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="requires ffmpeg and ffprobe",
)

REAL_RUN = timing._run
FRAME = 1 / 25
WIDTH, HEIGHT = 160, 100
# 200 frames on a 25 fps grid, of which 30 are dropped early (20-49) and 10
# late (180-189): kept frames sit at irregular times, and an average-rate grid
# puts the flashes 0.1 to 0.5 s from where the audio clicks are.
DROPPED = set(range(20, 50)) | set(range(180, 190))
FLASHES = (10, 100, 179, 198)  # source frame numbers (all kept)


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, capture_output=True, **kwargs)


def packet_times(path):
    """Presentation times of the video packets, as exact fractions."""
    base = Fraction(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=time_base", "-of", "csv=p=0", str(path)], text=True).stdout.strip())
    pts = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "packet=pts", "-of", "csv=p=0", str(path)], text=True).stdout.split()
    return [int(p) * base for p in sorted(pts, key=int)]


def flash_times(path):
    """Times of the frames whose top rows are bright (the flash marker)."""
    times = packet_times(path)
    raw = run(["ffmpeg", "-v", "error", "-i", str(path), "-fps_mode", "passthrough", "-vf",
               "scale=16:20:flags=area,format=gray", "-f", "rawvideo", "-"]).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 20, 16)
    assert len(frames) == len(times)
    return [float(t) for t, top in zip(times, frames[:, :3, :].mean(axis=(1, 2))) if top > 150]


def click_times(path):
    raw = run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "48000",
               "-f", "s16le", "-"]).stdout
    loud = np.nonzero(np.abs(np.frombuffer(raw, np.int16)) > 8000)[0]
    starts = [i for k, i in enumerate(loud) if k == 0 or i - loud[k - 1] > 2400]
    return [i / 48000 for i in starts]


def offsets(path):
    """Flash time minus click time for each marker."""
    flashes, clicks = flash_times(path), click_times(path)
    assert len(flashes) == len(clicks) > 0
    return [f - c for f, c in zip(flashes, clicks)]


def make_clip(path, frames, dropped, flashes, extra_input=()):
    """Video on a 25 fps grid with *dropped* frames removed (so the timestamps are
    irregular), a white band on the *flashes* frames and an audio click at the same
    instants. ``extra_input`` precedes the video input when muxing (e.g. -itsoffset)."""
    n = len(frames)
    video = str(path) + ".video.mp4"
    expression = "+".join(f"eq(n\\,{f})" for f in flashes)
    gaps = "+".join(f"between(n\\,{a}\\,{b})" for a, b in dropped)
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         f"color=c=0x303030:s={WIDTH}x{HEIGHT}:r=25",
         "-vf", f"drawbox=x=0:y=0:w=iw:h=40:color=white:t=fill:enable='{expression}',"
                f"select='not({gaps})'",
         "-frames:v", str(n), "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "18", video])
    audio = np.zeros(48000 * 8, np.int16)
    for f in flashes:
        audio[int(f / 25 * 48000):int(f / 25 * 48000) + 48] = 20000
    wav = str(path) + ".wav"
    with wave.open(wav, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(audio.tobytes())
    run(["ffmpeg", "-y", "-loglevel", "error", *extra_input, "-i", video, "-i", wav,
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", str(path)])


@pytest.fixture(scope="module")
def vfr_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("vfr") / "vfr.mp4"
    kept = [i for i in range(200) if i not in DROPPED]
    make_clip(path, kept, [(20, 49), (180, 189)], FLASHES)
    return path


@pytest.fixture(scope="module")
def constant_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("constant") / "constant.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=s=160x100:r=25",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)])
    return path


@pytest.fixture(autouse=True)
def default_mode(monkeypatch):
    monkeypatch.delenv("VIDEOWIPE_SOURCE_TIMING", raising=False)
    # Audio, metadata and subtitle handling (test_output_tracks.py) is pinned to
    # upstream's here, so these tests compare the video timing path on its own.
    monkeypatch.setenv("VIDEOWIPE_OUTPUT_TRACKS", "off")


@pytest.fixture
def stub_model(monkeypatch):
    monkeypatch.setattr(sttn, "_process_segment",
                        lambda frames, *args: [np.zeros_like(f) for f in frames])


def inpaint(video, output_dir, trial=None, dual=False):
    mask = np.zeros((HEIGHT, WIDTH, 1), np.uint8)
    mask[60:90, 20:140] = 1
    reader, info = read_frame_info(str(video))
    painter = sttn.STTNInpainter()
    painter.backend = object()
    try:
        job = InpaintJob(str(video), mask, str(output_dir), info["fps"], info["len"],
                         info["W_ori"], info["H_ori"], reader=reader, gap=10,
                         trial_range=trial, dual=dual)
        return painter.inpaint(job).output_path
    finally:
        reader.release()


# --- The acceptance test ---------------------------------------------------------------


def test_source_timestamps_are_kept_and_audio_stays_in_sync(vfr_clip, tmp_path, stub_model, capsys):
    out = inpaint(vfr_clip, tmp_path)
    assert "Keeping the source's frame timestamps" in capsys.readouterr().out
    assert packet_times(out) == packet_times(vfr_clip)
    before, after = offsets(vfr_clip), offsets(out)
    assert len(after) == len(FLASHES)
    # Start, middle and end of the output.
    assert all(abs(a - b) <= FRAME for a, b in zip(after, before))
    assert max(abs(a) for a in after) <= FRAME


def test_the_same_check_fails_when_source_timing_is_off(vfr_clip, tmp_path, stub_model,
                                                        monkeypatch, capsys):
    monkeypatch.setenv("VIDEOWIPE_SOURCE_TIMING", "off")
    out = inpaint(vfr_clip, tmp_path)
    assert "frame timestamps" not in capsys.readouterr().out
    assert packet_times(out) != packet_times(vfr_clip)
    drift = [abs(a) for a in offsets(out)]
    # Constant-rate retiming puts the flashes 0.1, 0.5 and 0.29 s from their clicks.
    assert drift[0] > FRAME and drift[1] > FRAME and drift[2] > FRAME


@pytest.mark.parametrize("trial", [(60, 100), (20, 21), (0, 30), (120, 160), (159, 160)])
def test_trial_ranges_start_at_the_real_frame_time(vfr_clip, tmp_path, stub_model, trial):
    out = inpaint(vfr_clip, tmp_path, trial=trial, dual=True)
    source = packet_times(vfr_clip)
    first, last = trial
    assert packet_times(out) == [t - source[first] for t in source[first:last]]
    # The audio was cut at the first frame's real time, so flash and click still agree.
    if flash_times(out):
        assert all(abs(offset) <= FRAME for offset in offsets(out))


def test_dual_output_is_stacked_and_timed(vfr_clip, tmp_path, stub_model):
    out = inpaint(vfr_clip, tmp_path, dual=True)
    shape = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height", "-of", "csv=p=0", out], text=True).stdout.strip()
    assert shape == f"{WIDTH},{HEIGHT * 2}"
    assert packet_times(out) == packet_times(vfr_clip)
    # The original frame is on top, so the flash is seen in both halves' top rows.
    assert max(abs(offset) for offset in offsets(out)) <= FRAME


def test_the_offset_between_video_and_audio_is_kept(tmp_path, stub_model):
    source = tmp_path / "late_video.mp4"
    kept = [i for i in range(200) if i not in DROPPED]
    make_clip(source, kept, [(20, 49), (180, 189)], FLASHES, extra_input=("-itsoffset", "0.5"))
    assert packet_times(source)[0] == Fraction(1, 2)
    out = inpaint(source, tmp_path)
    assert packet_times(out) == packet_times(source)
    assert offsets(out) == pytest.approx(offsets(source), abs=1e-6)


def test_sources_with_off_grid_timestamps_are_not_rounded(tmp_path, stub_model):
    source = tmp_path / "jitter.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x404040:s=160x100:r=25",
         "-vf", "settb=1/1000000,setpts='(N/25+0.013*sin(N*1.7)+0.3)/TB'", "-frames:v", "60",
         "-fps_mode", "passthrough", "-enc_time_base", "1:1000000", "-video_track_timescale", "90000",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)])
    times = packet_times(source)
    assert any((t * 25).denominator != 1 for t in times)  # not on a 25 fps grid
    out = inpaint(source, tmp_path)
    first = times[0]
    assert packet_times(out) == [t - first for t in times]


def test_a_second_output_on_the_same_command_keeps_the_timestamps_too(tmp_path, stub_model, monkeypatch):
    """The optional HEVC copy is appended to the command as another output."""
    source = tmp_path / "jitter.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x404040:s=160x100:r=25",
         "-vf", "settb=1/1000000,setpts='(N/25+0.013*sin(N*1.7)+0.3)/TB'", "-frames:v", "60",
         "-fps_mode", "passthrough", "-enc_time_base", "1:1000000", "-video_track_timescale", "90000",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)])
    second = tmp_path / "second.mkv"
    original = fidelity.EncodePlan.popen
    monkeypatch.setattr(fidelity.EncodePlan, "popen", lambda self, cmd, stderr: original(
        self, cmd + ["-map", "0:v", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(second)], stderr))
    out = inpaint(source, tmp_path)
    times = packet_times(source)
    assert packet_times(out) == [t - times[0] for t in times]
    # Matroska keeps milliseconds.
    assert [abs(a - (t - times[0])) for a, t in zip(packet_times(second), times)] \
        == pytest.approx([0] * len(times), abs=0.0005)


# --- Constant-rate sources are exactly upstream's ---------------------------------------------


def upstream_command(job, output_path, out_w, out_h, first, last):
    """``STTNInpainter.inpaint``'s ffmpeg command as in upstream commit 20872d4."""
    fps = job.fps
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-nostats",
           "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24",
           "-r", str(fps), "-i", "-"]
    if job.video_path:
        if job.trial_range is not None:
            cmd += ["-ss", str(first / fps)]
        cmd += ["-i", job.video_path]
    cmd += ["-map", "0:v"]
    if job.video_path:
        cmd += ["-map", "1:a?"]
    cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]
    if job.video_path:
        cmd += ["-c:a", "aac"]
    if job.trial_range is not None:
        cmd += ["-t", str((last - first) / fps)]
    return cmd + ["-movflags", "+faststart", output_path]


class Tee:
    """Records what is written to ffmpeg's stdin and forwards it."""

    def __init__(self, stream, sink):
        self.stream, self.sink = stream, sink

    def write(self, data):
        self.sink.append(bytes(data))
        return self.stream.write(data)

    def __getattr__(self, name):
        return getattr(self.stream, name)


@pytest.fixture
def spy(monkeypatch):
    seen = {"cmds": [], "stdin": []}
    real = subprocess.Popen

    def popen(cmd, *args, **kwargs):
        pipe = real(cmd, *args, **kwargs)
        if "rawvideo" in cmd:  # the cleaning pipeline's ffmpeg, not the test's helpers
            seen["cmds"].append(list(cmd))
            pipe.stdin = Tee(pipe.stdin, seen["stdin"])
        return pipe

    monkeypatch.setattr(subprocess, "Popen", popen)
    return seen


@pytest.mark.parametrize("variant", ["constant", "off"])
@pytest.mark.parametrize("trial", [None, (10, 40)])
def test_upstream_behaviour_is_unchanged(variant, trial, constant_clip, vfr_clip, tmp_path,
                                         stub_model, spy, monkeypatch):
    frames = []
    blend = sttn._blend_frame_regions
    monkeypatch.setattr(sttn, "_blend_frame_regions", lambda *a: frames.append(blend(*a).copy()) or frames[-1])
    if variant == "off":
        monkeypatch.setenv("VIDEOWIPE_SOURCE_TIMING", "off")
        source = vfr_clip
    else:
        source = constant_clip
    dual = trial is not None
    out = inpaint(source, tmp_path, trial=trial, dual=dual)
    _, info = read_frame_info(str(source))
    first, last = trial or (0, info["len"])
    job = InpaintJob(str(source), None, str(tmp_path), info["fps"], info["len"], WIDTH, HEIGHT,
                     trial_range=trial)
    assert spy["cmds"] == [upstream_command(job, out, WIDTH, HEIGHT * (2 if dual else 1), first, last)]
    if dual:
        expected = [np.vstack([top, f]).tobytes()
                    for top, f in zip(_originals(source, first, last), frames)]
    else:
        expected = [f.tobytes() for f in frames]
    assert len(spy["stdin"]) == last - first
    assert spy["stdin"] == expected


def _originals(source, first, last):
    reader = cv2.VideoCapture(str(source))
    out = []
    for index in range(last):
        ok, image = reader.read()
        assert ok
        if index >= first:
            out.append(image)
    reader.release()
    return out


def test_source_timing_on_applies_to_constant_sources_too(constant_clip, tmp_path, stub_model,
                                                          monkeypatch, capsys):
    monkeypatch.setenv("VIDEOWIPE_SOURCE_TIMING", "on")
    out = inpaint(constant_clip, tmp_path)
    assert "constant-rate source" in capsys.readouterr().out
    assert packet_times(out) == packet_times(constant_clip)


# --- Classification and the decision ---------------------------------------------------------


def make_video(path, codec, rate="25", seconds=8, extra=()):
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=s=160x100:r={rate}",
         "-t", str(seconds), "-c:v", codec, *extra, str(path)])
    return str(path)


def test_constant_mp4_is_constant(constant_clip):
    clock = timing.probe_source(str(constant_clip))
    assert clock.constant and clock.constant_rate == 25 and len(clock.pts) == 75


@pytest.mark.parametrize("suffix,codec,extra", [
    ("webm", "libvpx", ("-b:v", "300k")),
    ("mkv", "libx264", ("-pix_fmt", "yuv420p")),
])
def test_constant_matroska_with_millisecond_timestamps_is_constant(tmp_path, suffix, codec, extra):
    path = make_video(tmp_path / f"ntsc.{suffix}", codec, "30000/1001", extra=extra)
    clock = timing.probe_source(path)
    assert clock.time_base == Fraction(1, 1000)
    assert {b - a for a, b in zip(clock.pts, clock.pts[1:])} == {33, 34}  # rounded to ms
    assert clock.constant and clock.constant_rate == Fraction(30000, 1001)


def test_variable_sources_are_variable(vfr_clip):
    clock = timing.probe_source(str(vfr_clip))
    assert not clock.constant
    assert len(clock.pts) == 160


def test_probe_returns_nothing_when_ffprobe_fails(vfr_clip, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise FileNotFoundError("ffprobe")
    monkeypatch.setattr(timing, "_run", broken)
    assert timing.probe_source(str(vfr_clip)) is None
    monkeypatch.undo()
    assert timing.probe_source(str(tmp_path / "missing.mp4")) is None


@pytest.mark.parametrize("packets", [
    [{"pts": 0}, {"pts": 5}, {"pts": 5}],       # duplicate
    [{"pts": 0}, {}, {"pts": 5}],               # missing
    [{"pts": 0}, {"pts": "N/A"}],               # not available
    [],
])
def test_unusable_timestamps_are_rejected(packets):
    document = {"streams": [{"time_base": "1/1000", "r_frame_rate": "25/1",
                             "avg_frame_rate": "25/1", "start_time": "0.0"}],
                "format": {"start_time": "0.0"}, "packets": packets}
    assert timing.parse_probe(json.dumps(document)) is None


def test_packets_are_sorted_into_presentation_order():
    document = {"streams": [{"time_base": "1/25", "r_frame_rate": "25/1", "avg_frame_rate": "25/1"}],
                "format": {"start_time": "0.0"}, "packets": [{"pts": p} for p in (2, 0, 1, 3)]}
    clock = timing.parse_probe(json.dumps(document))
    assert clock.pts == (0, 1, 2, 3) and clock.constant


def job_for(path, **kwargs):
    reader, info = read_frame_info(str(path))
    reader.release()
    return InpaintJob(str(path), None, "unused", info["fps"], kwargs.pop("frame_count", info["len"]),
                      info["W_ori"], info["H_ori"], **kwargs)


def test_decision_for_each_kind_of_source(vfr_clip, constant_clip, capsys):
    assert timing.plan_source_timing(job_for(constant_clip)).clock is None
    assert capsys.readouterr().out == ""  # constant source: silent, upstream's behaviour
    assert timing.plan_source_timing(job_for(vfr_clip)).active
    assert not timing.plan_source_timing(job_for(vfr_clip, frame_count=161)).active
    assert "cannot be matched" in capsys.readouterr().out
    # A constant source with a different count needs no timestamps and no warning.
    assert not timing.plan_source_timing(job_for(constant_clip, frame_count=80)).active
    assert capsys.readouterr().out == ""


def test_probe_failure_falls_back_with_a_warning(vfr_clip, monkeypatch, capsys):
    monkeypatch.setattr(timing, "probe_source", lambda path: None)
    plan = timing.plan_source_timing(job_for(vfr_clip))
    assert not plan.active
    assert "could not read the source's frame timestamps" in capsys.readouterr().out


def test_the_environment_switch(vfr_clip, constant_clip, monkeypatch):
    monkeypatch.setenv("VIDEOWIPE_SOURCE_TIMING", "off")
    assert not timing.plan_source_timing(job_for(vfr_clip)).active
    monkeypatch.setenv("VIDEOWIPE_SOURCE_TIMING", "on")
    assert timing.plan_source_timing(job_for(constant_clip)).active
    monkeypatch.setenv("VIDEOWIPE_SOURCE_TIMING", "nonsense")
    assert timing.timing_mode() == "auto"


def test_a_count_mismatch_inside_the_pipeline_keeps_upstream_behaviour(vfr_clip, tmp_path,
                                                                       stub_model, spy, capsys):
    reader, info = read_frame_info(str(vfr_clip))
    mask = np.zeros((HEIGHT, WIDTH, 1), np.uint8)
    mask[60:90, 20:140] = 1
    painter = sttn.STTNInpainter()
    painter.backend = object()
    job = InpaintJob(str(vfr_clip), mask, str(tmp_path), info["fps"], info["len"] - 1,
                     info["W_ori"], info["H_ori"], reader=reader, gap=10)
    painter.inpaint(job)
    reader.release()
    assert "rawvideo" in spy["cmds"][0] and "matroska" not in spy["cmds"][0]
    assert "cannot be matched" in capsys.readouterr().out


@pytest.fixture
def fresh_capability_check():
    timing.ffmpeg_timing_options.cache_clear()
    yield
    timing.ffmpeg_timing_options.cache_clear()


def refusing(monkeypatch, *options):
    """Make ffmpeg runs that use any of *options* fail, as an ffmpeg without them would."""
    real = REAL_RUN
    monkeypatch.setattr(timing, "_run", lambda cmd, *a, **k: (
        subprocess.CompletedProcess(cmd, 1, b"", b"Unrecognized option")
        if any(o in cmd for o in options) else real(cmd, *a, **k)))


def test_capability_check_uses_whichever_sync_option_this_ffmpeg_accepts(
        monkeypatch, fresh_capability_check):
    refusing(monkeypatch, "-fps_mode")
    assert timing.ffmpeg_timing_options()[0] == "-vsync"
    timing.ffmpeg_timing_options.cache_clear()
    refusing(monkeypatch)  # nothing refused: the newer spelling is preferred
    assert timing.ffmpeg_timing_options()[0] == "-fps_mode"


def test_without_the_options_the_job_falls_back_with_a_warning(
        vfr_clip, monkeypatch, fresh_capability_check, capsys):
    refusing(monkeypatch, "-fps_mode", "-vsync")
    assert timing.ffmpeg_timing_options() is None
    assert not timing.plan_source_timing(job_for(vfr_clip)).active
    assert "this ffmpeg cannot keep the source's frame timestamps" in capsys.readouterr().out
    timing.ffmpeg_timing_options.cache_clear()
    refusing(monkeypatch, "-enc_time_base")  # rounding to 1 / frame rate would remain
    assert timing.ffmpeg_timing_options() is None


# --- The Matroska writer ---------------------------------------------------------------


@pytest.mark.parametrize("width,height", [(64, 48), (35, 21)])
@pytest.mark.parametrize("style", ["whole", "memoryview", "pieces"])
def test_writer_round_trips_pixels_and_timestamps(tmp_path, width, height, style):
    rng = np.random.default_rng(7)
    frames = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(5)]
    stamps = [(0, 41_000_000), (41_000_000, 79_000_000), (120_000_000, 1_000_000_000),
              (1_120_000_000, 33_000_000), (1_153_000_000, 40_000_000)]
    path = tmp_path / "frames.mkv"
    with open(path, "wb") as handle:
        writer = timing.MatroskaFrameWriter(handle, width, height, width * height * 3, stamps)
        for frame in frames:
            if style == "whole":
                writer.write(frame.tobytes())
            elif style == "memoryview":
                writer.write(memoryview(frame).cast("B"))
            else:
                data = frame.tobytes()
                for at in range(0, len(data), 1000):
                    writer.write(data[at:at + 1000])
        writer.flush()
        assert writer.frames_written == 5
    decoded = run(["ffmpeg", "-v", "error", "-i", str(path), "-fps_mode", "passthrough",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]).stdout
    assert decoded == b"".join(frame.tobytes() for frame in frames)
    shown = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "packet=pts_time,duration_time", "-of", "csv=p=0", str(path)], text=True).stdout.split()
    assert [tuple(row.split(",")) for row in shown] == [
        ("0.000000", "0.041000"), ("0.041000", "0.079000"), ("0.120000", "1.000000"),
        ("1.120000", "0.033000"), ("1.153000", "0.040000")]


def test_writer_refuses_more_frames_than_timestamps(tmp_path):
    with open(tmp_path / "x.mkv", "wb") as handle:
        writer = timing.MatroskaFrameWriter(handle, 2, 2, 12, [(0, 1)])
        writer.write(bytes(12))
        with pytest.raises(RuntimeError):
            writer.write(bytes(12))
