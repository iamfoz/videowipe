"""P0-3: scripts/perf_baseline.py records facts and stage timings without crashing."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from videowipe.detect import TextBox
from videowipe.engine import WipeEngine
from videowipe.inpainters import sttn

UNAVAILABLE = "unavailable"


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "perf_baseline.py"
    spec = importlib.util.spec_from_file_location("perf_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script():
    return _load_script()


W, H, FRAMES = 96, 64, 60


def _write_video(path, frames=FRAMES):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (W, H))
    for _ in range(frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[50:60, 8:88] = 200
        frame[5:15, 38:58] = 200
        writer.write(frame)
    writer.release()
    return path


class StubDetector:
    def detect(self, frame):
        if frame.shape[:2] != (H, W):
            return []
        return [
            TextBox(np.array([[8, 50], [88, 50], [88, 60], [8, 60]]), 0.9, "subtitle"),
            TextBox(np.array([[38, 5], [58, 5], [58, 15], [38, 15]]), 0.95, "logo"),
        ]


def _stub_engine(device, detector=StubDetector):
    engine = WipeEngine(task="clean", device=device, detector=detector())
    painter = sttn.STTNInpainter()
    painter.backend = SimpleNamespace(
        benchmark_metadata=lambda: {"device": "cpu", "weight_sha256": "test"},
        cleanup=lambda: None,
    )
    engine._task_impl.inpainter = painter
    engine._task_impl.backend = painter.backend
    engine._model_loaded = True
    return engine


@pytest.fixture
def stubbed(script, monkeypatch):
    monkeypatch.setattr(script, "_engine", _stub_engine)
    monkeypatch.setattr(sttn, "_process_segment", lambda frames, *a: [
        np.full((120, 640, 3), 90, np.uint8) for _ in frames])


# ── facts degrade gracefully ────────────────────────────────────────────────

def test_machine_facts_without_any_optional_tool(script, monkeypatch):
    monkeypatch.setattr(script, "_run", lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setattr(script.platform, "system", lambda: "Linux")
    monkeypatch.setattr(script.platform, "processor", lambda: "")
    monkeypatch.setattr(script.os, "cpu_count", lambda: None)

    def no_sysconf(name):
        raise ValueError(name)

    monkeypatch.setattr(script.os, "sysconf", no_sysconf, raising=False)
    monkeypatch.setattr(script, "_chip", lambda: UNAVAILABLE)
    facts = script.collect_machine_facts()

    assert facts["torch"]["available"] is False
    assert facts["onnxruntime"]["available"] is False
    assert facts["ffmpeg"]["available"] is False
    assert facts["ffmpeg"]["version"] == UNAVAILABLE
    assert facts["ffmpeg"]["encoders"] == {
        "libx264": UNAVAILABLE, "libx265": UNAVAILABLE,
        "h264_videotoolbox": UNAVAILABLE, "hevc_videotoolbox": UNAVAILABLE,
    }
    for key in ("chip", "logical_cores", "physical_cores", "performance_cores",
                "efficiency_cores", "ram_bytes"):
        assert facts[key] == UNAVAILABLE, key
    assert facts["opencv"]["version"] == cv2.__version__
    assert facts["python"]
    json.dumps(facts)


def test_machine_facts_on_apple_silicon_use_sysctl_and_probe_encoders(script, monkeypatch):
    answers = {
        "machdep.cpu.brand_string": "Apple M2 Pro",
        "hw.perflevel0.logicalcpu": "8", "hw.perflevel1.logicalcpu": "4",
        "hw.physicalcpu": "12", "hw.memsize": str(32 * 1024 ** 3),
    }

    def run(command, timeout=20):
        if command[0] == "sysctl":
            return answers[command[2]] + "\n"
        if command[:2] == ["ffmpeg", "-version"]:
            return "ffmpeg version 7.1 Copyright\nbuilt with clang\n"
        if command[:2] == ["ffmpeg", "-hide_banner"]:
            return " V....D hevc_videotoolbox    VideoToolbox H.265 Encoder\n" \
                   " V....D libx264              libx264 H.264\n"
        if command[0] == "ffprobe":
            return "ffprobe version 7.1\n"
        return None

    try:  # import before pretending to be macOS, or its platform check warns
        import onnxruntime  # noqa: F401
    except ImportError:
        pass
    monkeypatch.setattr(script, "_run", run)
    monkeypatch.setattr(script.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(script.platform, "machine", lambda: "arm64")
    facts = script.collect_machine_facts()
    assert facts["chip"] == "Apple M2 Pro"
    assert (facts["performance_cores"], facts["efficiency_cores"], facts["physical_cores"]) == (8, 4, 12)
    assert facts["ram_bytes"] == 32 * 1024 ** 3
    assert facts["ffmpeg"]["version"] == "ffmpeg version 7.1 Copyright"
    assert facts["ffmpeg"]["encoders"] == {
        "libx264": True, "libx265": False, "h264_videotoolbox": False, "hevc_videotoolbox": True,
    }


def test_video_facts_fall_back_to_opencv_when_ffprobe_is_missing(script, monkeypatch, tmp_path):
    video = _write_video(tmp_path / "clip.mp4", frames=8)
    monkeypatch.setattr(script, "_run", lambda *a, **k: None)
    facts = script.probe_video(str(video))
    assert facts["ffprobe"] == UNAVAILABLE and facts["codec"] == UNAVAILABLE
    assert (facts["width"], facts["height"], facts["frame_count"]) == (W, H, 8)
    assert facts["fps"] == 4 and facts["duration_s"] == 2.0 and facts["container"] == "mp4"


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="requires ffprobe")
def test_video_facts_use_ffprobe_when_present(script, tmp_path):
    video = _write_video(tmp_path / "clip.mp4", frames=8)
    facts = script.probe_video(str(video))
    assert isinstance(facts["ffprobe"], dict)
    assert "mp4" in facts["container"] and facts["codec"] not in (None, UNAVAILABLE)
    assert facts["frame_count"] == 8


# ── pure helpers ────────────────────────────────────────────────────────────

def test_trial_sits_at_the_end_and_is_clamped(script):
    assert script.trial_interval(100, 25, 2) == (50, 100)
    assert script.trial_interval(100, 25, 0.001) == (99, 100)
    assert script.trial_interval(30, 25, 60) == (0, 30)


def test_median_tree_takes_the_median_of_every_number(script):
    runs = [{"a": 1.0, "n": 3, "s": "x", "d": {"k": 10}},
            {"a": 3.0, "n": 5, "s": "y", "d": {"k": 30}},
            {"a": 2.0, "n": 4, "s": "z", "d": {"k": 20, "extra": 1}}]
    assert script.median_tree(runs) == {"a": 2.0, "n": 4, "s": "x", "d": {"k": 20, "extra": 1}}


def test_derived_ratios(script):
    stats = {"refine": {"frames_converted": 100, "forward_passes": 150, "detect_calls": 120,
                        "recovery_frames": 20, "recovery_calls": 20, "recovery_hits": 5,
                        "recovery_detect_calls": 60},
             "coarse_sampling": {"frames_converted": 50, "forward_passes": 50, "detect_calls": 50}}
    ratios = script.derived_ratios(stats)
    refine = ratios["refine"]
    assert refine["forward_passes_per_frame"] == 1.5
    assert refine["share_of_frames_reaching_recovery"] == 0.2
    assert refine["recovery_hit_rate"] == 0.25
    assert refine["recovery_share_of_detect_calls"] == 0.5
    assert ratios["all_phases"]["forward_passes_per_frame"] == round(200 / 150, 4)
    assert script.derived_ratios({"refine": {"frames_converted": 0}})["refine"]["recovery_hit_rate"] is None


# ── end to end on a tiny clip ───────────────────────────────────────────────

def _check_stage_table(rows, total):
    assert rows and rows[0]["level"] == 0
    for row in rows:
        assert set(row) >= {"stage", "level", "seconds", "frames", "ms_per_frame", "share"}
        assert row["seconds"] >= -0.01
    assert total > 0


def test_planning_only_baseline_writes_the_expected_report(script, stubbed, tmp_path, capsys):
    video = _write_video(tmp_path / "clip.mp4")
    out = tmp_path / "out"
    assert script.main([str(video), "-o", str(out), "--no-full"]) == 0

    report = json.loads((out / "perf_baseline.json").read_text())
    assert report["schema_version"] == 1 and report["options"]["full"] is False
    assert {"platform", "chip", "ram_bytes", "python", "torch", "onnxruntime", "opencv",
            "ffmpeg"} <= report["machine"].keys()
    entry, = report["videos"]
    assert entry["video"]["frame_count"] == FRAMES and entry["video"]["fps"] == 4
    run, = entry["runs"]
    assert "clean" not in run and "trial" not in run
    plan = run["plan"]
    assert plan["wall_s"] > 0 and plan["remove_tracks"] >= 1
    stats = plan["detection_stats"]
    # Fifty samples of sixty frames: fifty converted, the other ten only grabbed.
    assert stats["coarse_sampling"]["frames_converted"] == 50
    assert stats["coarse_sampling"]["frames_grabbed"] == 10
    assert stats["coarse_sampling"]["detect_calls"] == 50
    assert {"detect", "refine"} <= plan["progress_phase_s"].keys()
    _check_stage_table(run["summary"]["stages"], run["summary"]["total_s"])
    assert run["summary"]["detection_ratios"]["refine"]["frames_converted"] == FRAMES
    printed = capsys.readouterr().out
    for text in ("Stages (share of planning + clean)", "Detection counters per planning phase",
                 "coarse_sampling", "forward_passes_per_frame", "ms/frame"):
        assert text in printed


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires ffmpeg")
def test_full_run_with_trial_and_median(script, stubbed, tmp_path):
    video = _write_video(tmp_path / "clip.mp4")
    out = tmp_path / "out"
    code = script.main([str(video), "-o", str(out), "--trial-seconds", "2", "--runs", "2"])
    assert code == 0
    entry, = json.loads((out / "perf_baseline.json").read_text())["videos"]
    assert len(entry["runs"]) == 2 and "median" in entry

    for run in entry["runs"]:
        clean, trial = run["clean"], run["trial"]
        assert "error" not in clean and "error" not in trial
        timing = clean["timing"]
        assert timing["frames_written"] == FRAMES and timing["segments"] == 3
        assert clean["engine_identity"]["backend"] == "STTNInpainter" or clean["engine_identity"]
        assert Path(clean["output_path"]).is_file() and len(clean["output_sha256"]) == 64
        # P0-1: the split timers are in the report next to the lumped ones.
        for key in ("prediction_s", "composition_s", "model_s", "decode_infer_s", "resize_s",
                    "blend_s", "pipe_write_s", "decode_output_s", "inpainting_s"):
            assert key in timing
        # The trial covers the last 2 s (8 frames at 4 fps) and positions past 50 frames.
        assert trial["range"] == [FRAMES - 8, FRAMES]
        assert trial["timing"]["frames_written"] == 8
        assert trial["timing"]["prefix_frames"] == 50 and trial["timing"]["prefix_skip_s"] > 0
        assert trial["overlaps_remove_segment"] is True
        summary = run["summary"]
        _check_stage_table(summary["stages"], summary["total_s"])
        _check_stage_table(summary["trial_stages"], summary["trial_total_s"])
        assert any(row["stage"].startswith("trial: position") for row in summary["trial_stages"])
        names = {row["stage"] for row in summary["stages"]}
        assert {"planning (detect + refine)", "clean: inpainting", "model inference"} <= names
        assert summary["total_s"] == pytest.approx(
            run["plan"]["wall_s"] + clean["wall_s"], abs=1e-3)
    median = entry["median"]
    walls = sorted(run["clean"]["wall_s"] for run in entry["runs"])
    assert median["clean"]["wall_s"] == pytest.approx(sum(walls) / 2, abs=1e-3)
    assert median["summary"]["stages"]


def test_failures_are_recorded_not_raised(script, monkeypatch, tmp_path):
    class Failing:
        def detect(self, frame):
            raise RuntimeError("detector down")

    monkeypatch.setattr(script, "_engine", lambda device: _stub_engine(device, Failing))
    video = _write_video(tmp_path / "clip.mp4", frames=6)
    out = tmp_path / "out"
    assert script.main([str(video), "-o", str(out)]) == 1
    run, = json.loads((out / "perf_baseline.json").read_text())["videos"][0]["runs"]
    assert "detection failed" in run["plan"]["error"].lower() or "detector" in run["plan"]["error"].lower()
    assert "clean" not in run  # nothing to execute without a plan


def test_cli_rejects_bad_arguments(script, tmp_path):
    video = _write_video(tmp_path / "clip.mp4", frames=2)
    with pytest.raises(SystemExit):
        script.main([str(video), "--runs", "0"])
    with pytest.raises(SystemExit):
        script.main([str(tmp_path / "missing.mp4")])
    with pytest.raises(SystemExit):
        script.main([str(video), "--device", "tpu"])
