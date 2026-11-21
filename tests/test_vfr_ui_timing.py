"""Seconds <-> frame ordinals for variable-frame-rate sources (web page and server).

Upstream converts with ``frame = seconds * fps``. For a variable-rate source the
server now converts through the source's real frame times instead; a constant
rate source must give exactly upstream's numbers. The ``upstream_*`` functions
below are upstream's originals, copied verbatim, and serve as oracles.
"""
import math
import random
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

from test_server import _post_video, _wait_for_state, client  # noqa: F401  (client is a fixture)
from videowipe import timing
from videowipe.errors import InvalidInputError
from videowipe.plan import Segment
from videowipe.server import app as server_app
from videowipe.server import frame_map as frame_map_module
from videowipe.server.frame_map import FrameMap, frame_map, trial_seconds
from videowipe.server.review import seconds_to_frames

SAMPLES = Path(__file__).resolve().parents[1] / "input" / "detext_examples"
needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="requires ffmpeg and ffprobe"
)
needs_node = pytest.mark.skipif(not shutil.which("node"), reason="requires node")


# --- Upstream's originals (oracles) -----------------------------------------------------


def upstream_seconds_to_frames(start, end, source):
    if not all(type(v) in (int, float) and math.isfinite(v) for v in (start, end)):
        raise InvalidInputError("time must be finite")
    if not 0 <= start < end <= source.frame_count / source.fps:
        raise InvalidInputError("time interval exceeds the video")
    frames = [value * source.fps for value in (start, end)]
    frames = [round(value) if abs(value-round(value)) < 1e-9 else value for value in frames]
    return math.floor(frames[0]), min(source.frame_count, math.ceil(frames[1]))


def upstream_trial_range(plan, start: float, duration: float):
    fps = plan.source.fps
    if start >= plan.source.frame_count / fps:
        raise HTTPException(status_code=400, detail="trial starts outside the video")
    try:
        first, last = upstream_seconds_to_frames(
            start, min(start + duration, plan.source.frame_count / fps), plan.source)
    except InvalidInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not 0 <= first < last <= plan.source.frame_count:
        raise HTTPException(status_code=400, detail="trial starts outside the video")
    if not any(
        track.mask is not None and track.mask.any()
        and any(s.start < last and s.end > first for s in track.segments)
        for track in plan.remove_tracks
    ):
        raise HTTPException(status_code=400, detail="No selected target is active in this interval")
    return first, last


def upstream_recommended_trial(plan):
    intervals = sorted(
        (s.start, s.end) for track in plan.remove_tracks
        if track.mask is not None and track.mask.any() for s in track.segments
    )
    merged: list[list[int]] = []
    for first, last in intervals:
        if merged and first <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], last)
        else:
            merged.append([first, last])
    if not merged:
        return None
    first, last = min(merged, key=lambda row: (-(row[1] - row[0]), row[0]))
    fps = plan.source.fps
    duration = min(3.0, plan.source.frame_count / fps)
    start = max(0.0, min(plan.source.frame_count / fps - duration,
                         (first + last) / (2 * fps) - duration / 2))
    frame_range = upstream_trial_range(plan, start, duration)
    return {"start_seconds": start, "duration_seconds": duration,
            "frame_range": list(frame_range)}


# --- Helpers ----------------------------------------------------------------------------------


def run(cmd, **kwargs):
    return timing._run(cmd, **kwargs)


def sample_facts(path):
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
    cap.release()
    return fps, count


def make_plan(fps, frame_count, segments):
    mask = np.ones((4, 4), dtype=np.uint8)
    track = SimpleNamespace(mask=mask, segments=[Segment(a, b) for a, b in segments])
    return SimpleNamespace(source=SimpleNamespace(fps=fps, frame_count=frame_count),
                           remove_tracks=[track])


def outcome(function, *args):
    try:
        return ("ok", function(*args))
    except HTTPException as exc:
        return ("error", exc.status_code, exc.detail)
    except InvalidInputError as exc:
        return ("invalid", str(exc))


def true_range(times, start, duration):
    """The frames a person means: from the frame showing at *start* up to the
    first frame that starts at or after start + duration (times has n + 1 values)."""
    count = len(times) - 1
    first = max(i for i in range(count) if times[i] <= start + 1e-9)
    end = min(start + duration, times[-1])
    last = next((j for j in range(count) if times[j] >= end - 1e-9), count)
    return first, last


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.delenv("VIDEOWIPE_SOURCE_TIMING", raising=False)
    frame_map_module.clear_cache()
    yield
    frame_map_module.clear_cache()


@pytest.fixture(scope="module")
def vfr_clip(tmp_path_factory):
    """96x64, 150 frames: a 0.2 s gap before frame 10 and a 0.12 s gap before frame 80."""
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("requires ffmpeg and ffprobe")
    path = tmp_path_factory.mktemp("vfr") / "vfr.mp4"
    done = run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=96x64:r=25:d=6",
                "-vf", "drawbox=x=10:y=50:w=76:h=10:color=white:t=fill,"
                       "setpts='PTS+if(gte(N\\,10)\\,5\\,0)+if(gte(N\\,80)\\,3\\,0)'",
                "-fps_mode", "passthrough", "-video_track_timescale", "12800",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])
    assert done.returncode == 0, done.stderr
    return path


# --- FrameMap -------------------------------------------------------------------------------------


def test_frame_map_interpolates_between_frame_times():
    frames = FrameMap([0.0, 0.04, 0.16, 0.20, 0.24])
    assert frames.frame_count == 4 and frames.duration == 0.24
    assert [frames.seconds(i) for i in range(5)] == [0.0, 0.04, 0.16, 0.20, 0.24]
    assert frames.frames(0.0) == 0 and frames.frames(0.04) == 1 and frames.frames(0.16) == 2
    assert frames.frames(0.10) == pytest.approx(1.5)  # half way through the 0.12 s frame
    assert frames.seconds(1.5) == pytest.approx(0.10)
    assert frames.frames(-1) == 0 and frames.frames(99) == 4 and frames.seconds(9) == 0.24
    assert frames.seconds(-3) == 0.0
    for i in range(5):
        assert frames.frames(frames.seconds(i)) == i


def test_frame_map_rejects_unordered_times():
    for bad in ([0.0], [0.0, 0.1, 0.1], [0.0, 0.2, 0.1]):
        with pytest.raises(ValueError):
            FrameMap(bad)


def clock_of(pts, time_base="1/1000", start=0, constant=False):
    from fractions import Fraction
    return SimpleNamespace(pts=tuple(pts), time_base=Fraction(time_base), constant=constant,
                           seconds=lambda i: Fraction(pts[i]) * Fraction(time_base) - start)


def test_frame_map_from_clock_ends_with_the_last_gap():
    frames = FrameMap.from_clock(clock_of([0, 40, 160, 200]), 4)
    assert frames.times == (0.0, 0.04, 0.16, 0.2, 0.24)


def test_frame_map_from_clock_refuses_what_it_cannot_match():
    assert FrameMap.from_clock(clock_of([0, 40, 160]), 4) is None  # packets != frames
    assert FrameMap.from_clock(clock_of([0, 40, 160]), 0) is None
    assert FrameMap.from_clock(clock_of([0, 40, 160], start=1), 3) is None  # first frame before the start
    # Distinct ticks that collapse on rounding to a microsecond cannot be ordered.
    assert FrameMap.from_clock(clock_of([0, 1, 2], time_base="1/10000000"), 3) is None


def test_one_frame_source_uses_the_frame_rate_for_its_length():
    frames = FrameMap.from_clock(clock_of([0]), 1, fps=20)
    assert frames.times == (0.0, 0.05)


# --- Constant-rate sources are exactly upstream ------------------------------------------------------


@needs_ffmpeg
@pytest.mark.parametrize("name", ["english1", "others"])
def test_constant_rate_samples_have_no_map(name):
    path = SAMPLES / f"{name}.mp4"
    fps, count = sample_facts(path)
    assert frame_map(path, count, fps) is None


@needs_ffmpeg
@pytest.mark.parametrize("name", ["english1", "others"])
def test_constant_rate_trial_ranges_equal_upstream(name):
    path = SAMPLES / f"{name}.mp4"
    fps, count = sample_facts(path)
    frames = frame_map(path, count, fps)
    assert frames is None
    total = count / fps
    rng = random.Random(7)
    starts = [i / fps for i in range(count)] + [i * 0.04 for i in range(count)]
    starts += [rng.uniform(0, total) for _ in range(300)] + [0, total, total + 1, 1e-12, total - 1e-12]
    plan = make_plan(fps, count, [(0, count)])
    sparse = make_plan(fps, count, [(count // 3, count // 3 + 40), (count - 30, count)])
    for p in (plan, sparse):
        for start in starts:
            for duration in (3.0, 5.0, 0.25):
                assert outcome(server_app._trial_range, p, start, duration, frames) == \
                    outcome(upstream_trial_range, p, start, duration), (start, duration)
        assert outcome(server_app._recommended_trial, p, frames) == outcome(upstream_recommended_trial, p)
    # The trial's own start and length as the page and the status report them.
    for first, last in [(0, 75), (5, 80), (count - 75, count), (3, 4)]:
        assert trial_seconds((first, last), fps, frames) == (first / fps, (last - first) / fps)


@needs_ffmpeg
@pytest.mark.parametrize("name", ["english1", "others"])
def test_forced_map_on_a_constant_rate_source_still_equals_upstream(name, monkeypatch):
    """``VIDEOWIPE_SOURCE_TIMING=on`` maps a constant source too; the numbers still agree."""
    monkeypatch.setenv("VIDEOWIPE_SOURCE_TIMING", "on")
    path = SAMPLES / f"{name}.mp4"
    fps, count = sample_facts(path)
    frames = frame_map(path, count, fps)
    assert frames is not None and frames.frame_count == count
    total = count / fps
    plan = make_plan(fps, count, [(0, count)])
    starts = [i / fps for i in range(count)] + [(i + 0.5) / fps for i in range(count)]
    for start in starts:
        for duration in (3.0, 5.0):
            assert outcome(server_app._trial_range, plan, start, duration, frames) == \
                outcome(upstream_trial_range, plan, start, duration), (start, duration)
    assert frames.duration == pytest.approx(total)


def test_review_seconds_to_frames_without_a_map_is_upstream():
    for fps in (25, 29.97, 30000 / 1001, 24.9157):
        source = SimpleNamespace(fps=fps, frame_count=1000)
        for first in range(999):
            assert seconds_to_frames(first / fps, (first + 1) / fps, source) == \
                upstream_seconds_to_frames(first / fps, (first + 1) / fps, source)
        for start, end in [(0.1, 0.3), (1.25 / fps, 4.75 / fps), (0, 1000 / fps), (3.3, 9.9)]:
            assert seconds_to_frames(start, end, source) == upstream_seconds_to_frames(start, end, source)
        for bad in [(-1, 1), (2, 1), (0, 1000 / fps + 1), (float("nan"), 1)]:
            assert outcome(lambda: seconds_to_frames(*bad, source)) == \
                outcome(lambda: upstream_seconds_to_frames(*bad, source))


# --- The bundled variable-rate sample ---------------------------------------------------------------------


@needs_ffmpeg
def test_chinese1_gets_a_map_that_matches_ffprobe():
    path = SAMPLES / "chinese1.mp4"
    fps, count = sample_facts(path)
    frames = frame_map(path, count, fps)
    assert frames is not None and frames.frame_count == count == 591
    clock = timing.probe_source(str(path))
    assert [frames.seconds(i) for i in range(count)] == [round(float(clock.seconds(i)), 6) for i in range(count)]
    assert frames.times[1] == 0.12 and frames.times[2] == 0.16  # the 0.12 s gap is after the first frame
    assert frames.duration == pytest.approx(23.72)


@needs_ffmpeg
def test_chinese1_trial_ranges_follow_the_real_frame_times():
    path = SAMPLES / "chinese1.mp4"
    fps, count = sample_facts(path)
    frames = frame_map(path, count, fps)
    plan = make_plan(fps, count, [(0, count)])
    differing = 0
    for i in range(0, count - 1):
        start = frames.times[i]
        for duration in (3.0, 5.0):
            want = true_range(frames.times, start, duration)
            assert server_app._trial_range(plan, start, duration, frames) == want, (i, duration)
            if outcome(upstream_trial_range, plan, start, duration) != ("ok", want):
                differing += 1
    # The constant-rate arithmetic really is wrong for this source (the measurement
    # in scripts/vfr_timing_report.py: most 3 s trials start or end a frame or two off).
    assert differing > 600


@needs_ffmpeg
def test_chinese1_recommended_trial_and_trial_report_agree():
    path = SAMPLES / "chinese1.mp4"
    fps, count = sample_facts(path)
    frames = frame_map(path, count, fps)
    plan = make_plan(fps, count, [(100, 300), (400, 420)])
    recommended = server_app._recommended_trial(plan, frames)
    first, last = recommended["frame_range"]
    assert (first, last) == true_range(frames.times, recommended["start_seconds"], recommended["duration_seconds"])
    # centred on the real time of the longest interval, not on i / fps
    middle = (frames.seconds(100) + frames.seconds(300)) / 2
    assert recommended["start_seconds"] == pytest.approx(middle - 1.5)
    start, length = trial_seconds((first, last), fps, frames)
    assert start == frames.seconds(first) and start + length == pytest.approx(frames.seconds(last))
    # ... and the trial's own recheck of its range returns the same frames
    assert server_app._trial_range(plan, start, length, frames) == (first, last)


@needs_ffmpeg
def test_every_frame_start_time_reads_back_as_its_own_ordinal():
    path = SAMPLES / "chinese1.mp4"
    fps, count = sample_facts(path)
    frames = frame_map(path, count, fps)
    source = SimpleNamespace(fps=fps, frame_count=count)
    for i in range(count):
        assert seconds_to_frames(frames.seconds(i), frames.seconds(i + 1), source, frames) == (i, i + 1)


@needs_ffmpeg
def test_off_switch_restores_upstream(monkeypatch):
    path = SAMPLES / "chinese1.mp4"
    fps, count = sample_facts(path)
    monkeypatch.setenv("VIDEOWIPE_SOURCE_TIMING", "off")
    frame_map_module.clear_cache()
    frames = frame_map(path, count, fps)
    assert frames is None
    plan = make_plan(fps, count, [(0, count)])
    for start in (0.0, 1.0, 5.04, 12.34, 20.0):
        assert outcome(server_app._trial_range, plan, start, 3.0, frames) == \
            outcome(upstream_trial_range, plan, start, 3.0)


# --- Fallbacks and the one-probe cache -----------------------------------------------------------------------


@needs_ffmpeg
def test_failed_probe_falls_back_and_is_cached(monkeypatch, capsys):
    path = SAMPLES / "chinese1.mp4"
    fps, count = sample_facts(path)
    calls = []
    monkeypatch.setattr(frame_map_module, "probe_source", lambda p: calls.append(p))
    assert frame_map(path, count, fps) is None and frame_map(path, count, fps) is None
    assert len(calls) == 1
    assert "constant rate" in capsys.readouterr().out


@needs_ffmpeg
def test_packet_count_mismatch_falls_back(capsys):
    path = SAMPLES / "chinese1.mp4"
    fps, count = sample_facts(path)
    assert frame_map(path, count - 1, fps) is None
    assert "cannot be matched" in capsys.readouterr().out


@needs_ffmpeg
def test_one_probe_per_source(monkeypatch):
    path = SAMPLES / "chinese1.mp4"
    fps, count = sample_facts(path)
    real, calls = frame_map_module.probe_source, []
    monkeypatch.setattr(frame_map_module, "probe_source", lambda p: calls.append(p) or real(p))
    first = frame_map(path, count, fps)
    assert first is not None and frame_map(path, count, fps) is first
    assert len(calls) == 1


def test_missing_file_or_path_is_no_map(tmp_path):
    assert frame_map(tmp_path / "gone.mp4", 10, 25.0) is None
    assert frame_map("", 10, 25.0) is None


def test_source_frames_swallows_unexpected_errors(monkeypatch):
    def broken(*args):
        raise RuntimeError("boom")
    monkeypatch.setattr(server_app, "frame_map", broken)
    plan = make_plan(25.0, 10, [(0, 10)])
    assert server_app._source_frames(SimpleNamespace(video_path="x"), plan) is None


# --- A generated variable-rate clip through the server ---------------------------------------------------------------


def _vfr_preview(test_client, clip):
    job_id = _post_video(test_client, clip).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    return job_id, test_client.get(f"/jobs/{job_id}/preview").json()


@needs_ffmpeg
def test_server_preview_carries_the_frame_times(client, vfr_clip):
    test_client, _ = client
    fps, count = sample_facts(vfr_clip)
    assert count == 150 and abs(fps - 25) > 0.5  # the average rate is not the frame rate
    _, preview = _vfr_preview(test_client, vfr_clip)
    times = preview["source"]["frame_times"]
    assert len(times) == count + 1 and times[9] == 0.36 and times[10] == 0.6 and times[-1] == pytest.approx(6.32)
    assert preview["source"]["frame_count"] == count and preview["source"]["fps"] == fps


@needs_ffmpeg
def test_server_recommended_trial_follows_the_frame_times(client, vfr_clip):
    test_client, _ = client
    _, preview = _vfr_preview(test_client, vfr_clip)
    times, recommended = preview["source"]["frame_times"], preview["recommended_trial"]
    assert tuple(recommended["frame_range"]) == true_range(times, recommended["start_seconds"],
                                                           recommended["duration_seconds"])


@needs_ffmpeg
def test_server_trial_request_is_converted_through_the_frame_times(client, vfr_clip):
    test_client, _ = client
    job_id, preview = _vfr_preview(test_client, vfr_clip)
    times = preview["source"]["frame_times"]
    # a trial asked for at the start time of frame 40, 3 s long
    response = test_client.post(f"/jobs/{job_id}/trial", json={"start_seconds": times[40], "duration_seconds": 3})
    assert response.status_code == 200, response.text
    trial = response.json()["trial"]
    want = true_range(times, times[40], 3.0)
    assert trial["frame_range"] == list(want) and want[0] == 40
    assert trial["start_seconds"] == times[40]
    assert trial["duration_seconds"] == pytest.approx(times[want[1]] - times[40])
    ready = _wait_for_state(test_client, job_id, "preview_ready")["trial"]
    assert ready["ready"] and ready["frame_range"] == list(want)


@needs_ffmpeg
def test_server_trial_recheck_accepts_a_range_the_page_reads_back(client, vfr_clip):
    """The worker re-derives the range from the start and length it reports; on a
    variable-rate source that only comes back as the same frames through the map."""
    test_client, _ = client
    job_id, preview = _vfr_preview(test_client, vfr_clip)
    times = preview["source"]["frame_times"]
    job = server_app.get_job(job_id)
    plan = server_app.load_wipe_plan(str(server_app._plan_path(job)))
    frames = server_app._source_frames(job, plan)
    assert frames is not None
    for first, last in [(0, 76), (11, 87), (75, 150)]:
        start, length = trial_seconds((first, last), plan.source.fps, frames)
        assert server_app._trial_range(plan, start, length, frames) == (first, last)
        assert (start, start + length) == (times[first], pytest.approx(times[last]))


@needs_ffmpeg
def test_server_with_the_switch_off_is_upstream(client, vfr_clip, monkeypatch):
    monkeypatch.setenv("VIDEOWIPE_SOURCE_TIMING", "off")
    test_client, fake = client
    fps, count = sample_facts(vfr_clip)
    job_id = _post_video(test_client, vfr_clip).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    preview = test_client.get(f"/jobs/{job_id}/preview").json()
    assert "frame_times" not in preview["source"]
    plan = server_app.load_wipe_plan(str(server_app._plan_path(server_app.get_job(job_id))))
    assert preview["recommended_trial"] == upstream_recommended_trial(plan)
    response = test_client.post(f"/jobs/{job_id}/trial", json={"start_seconds": 1.0, "duration_seconds": 3})
    assert response.json()["trial"]["frame_range"] == list(upstream_trial_range(plan, 1.0, 3.0))
    assert response.json()["trial"]["start_seconds"] == upstream_trial_range(plan, 1.0, 3.0)[0] / fps


@needs_ffmpeg
def test_server_constant_rate_job_has_no_frame_times(client, tmp_path):
    from test_server import _write_test_video
    test_client, _ = client
    video = tmp_path / "constant.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    preview = test_client.get(f"/jobs/{job_id}/preview").json()
    assert "frame_times" not in preview["source"]
    plan = server_app.load_wipe_plan(str(server_app._plan_path(server_app.get_job(job_id))))
    assert preview["recommended_trial"] == upstream_recommended_trial(plan)


# --- The page's own conversions -----------------------------------------------------------------------------------------


# The page's own position(): seconds of playback, or (while the mask request starts) nominal seconds.
PAGE_POSITION = 'const video={currentTime:0};let view="source",trial=null;'


def page_helpers():
    html = server_app._web_index().read_text(encoding="utf-8")
    conversions = '//' + html.split('    // Frames <-> seconds.', 1)[1].split('    const evidenceImages', 1)[0]
    head, rest = html.split('    const duration = ', 1)[1].split('const stepFrom', 1)
    position = 'const position = ' + html.split('    const position = ', 1)[1].split('\n', 1)[0]
    return conversions + position + 'const duration = ' + head + 'const stepFrom' + rest.split('\n', 1)[0]


def node(code):
    done = subprocess.run(["node", "-e", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


@needs_node
def test_page_conversions_without_frame_times_are_frame_over_fps():
    code = ('const assert=require("node:assert/strict");let source={fps:25,frame_count:1000};'
            + PAGE_POSITION + page_helpers()
            + 'for(const fps of [25,29.97,30000/1001,24.9157]){source={fps,frame_count:1000};'
              'assert.equal(duration(),1000/fps);'
              'for(let n=0;n<=1000;n+=7){assert.equal(f2s(n),n/fps);assert.equal(s2f(n/fps),n/fps*fps);'
              'assert.equal(share(n),n/1000*100);}'
              'video.currentTime=3.3;nominalPositions=true;assert.equal(position(),3.3);nominalPositions=false;}')
    node(code)


@needs_node
@needs_ffmpeg
def test_page_conversions_match_the_server_on_a_variable_rate_source():
    path = SAMPLES / "chinese1.mp4"
    fps, count = sample_facts(path)
    frames = frame_map(path, count, fps)
    import json
    probes = [(0.0, frames.frames(0.0)), (0.12, frames.frames(0.12)), (0.14, frames.frames(0.14)),
              (3.33, frames.frames(3.33)), (23.5, frames.frames(23.5)), (99.0, frames.frames(99.0)),
              (-1.0, frames.frames(-1.0))]
    code = ('const assert=require("node:assert/strict");let source=' + json.dumps(
        {"fps": fps, "frame_count": count, "frame_times": frames.to_list()}) + ';' + PAGE_POSITION + page_helpers()
        + 'assert.equal(duration(),' + json.dumps(frames.duration) + ');'
          'for(let n=0;n<=source.frame_count;n++){assert.equal(f2s(n),source.frame_times[n]);assert.equal(s2f(f2s(n)),n);}'
          'for(const [s,f] of ' + json.dumps(probes) + ')assert.ok(Math.abs(s2f(s)-f)<1e-9,[s,s2f(s),f]);'
          'assert.equal(stepFrom(f2s(10),1),f2s(11));assert.equal(stepFrom(f2s(10),-1),f2s(9));'
          'assert.equal(stepFrom(f2s(0),-1),f2s(0));assert.equal(stepFrom(f2s(590),1),f2s(590));'
          'assert.equal(stepFrom(0.13,1),f2s(2));'
          # the mask request turns a position into a frame with `* source.fps`
          'for(let n=0;n<source.frame_count;n++)for(const extra of [0,0.5,0.99]){'
          'video.currentTime=f2s(n)+extra*(f2s(n+1)-f2s(n));assert.equal(position(),video.currentTime);'
          'nominalPositions=true;assert.equal(Math.floor(position()*source.fps),n);nominalPositions=false;}')
    node(code)


@needs_node
@needs_ffmpeg
def test_page_interval_parsing_round_trips_frames_on_a_variable_rate_source():
    html = server_app._web_index().read_text(encoding="utf-8")
    function = html.split('    function parseIntervals(text) {', 1)[1].split('    function edited()', 1)[0]
    path = SAMPLES / "chinese1.mp4"
    fps, count = sample_facts(path)
    frames = frame_map(path, count, fps)
    import json
    code = ('const assert=require("node:assert/strict");let source=' + json.dumps(
        {"fps": fps, "frame_count": count, "frame_times": frames.to_list()}) + ';' + page_helpers()
        + 'function parseIntervals(text) {' + function
        + 'for(let n=0;n<source.frame_count;n+=3)assert.deepEqual(parseIntervals(`${f2s(n)}, ${f2s(n+1)}`),[[n,n+1]]);'
          'assert.deepEqual(parseIntervals(`${f2s(5)}, ${f2s(source.frame_count)}`),[[5,source.frame_count]]);')
    node(code)
