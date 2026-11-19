"""scripts/compare_clean_outputs.py on generated clips."""
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_clean_outputs.py"
spec = importlib.util.spec_from_file_location("compare_clean_outputs", SCRIPT)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

W, H, FRAMES = 96, 64, 12


def _frames(seed=0, perturb=0, region=None):
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, (H, W, 3), dtype=np.uint8)
    out = []
    for i in range(FRAMES):
        frame = np.clip(base.astype(np.int16) + i, 0, 255).astype(np.uint8)
        if perturb and region is not None:
            y0, y1, x0, x1 = region
            frame[y0:y1, x0:x1] = np.clip(frame[y0:y1, x0:x1].astype(np.int16) + perturb * (1 if i % 2 else -1), 0, 255)
        out.append(frame)
    return out


def _write(path, frames):
    # MJPG in AVI is lossless enough to be predictable and needs no ffmpeg binary.
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 4, (W, H))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4, (W, H))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


@pytest.fixture
def clips(tmp_path):
    region = (40, 56, 10, 80)
    reference = tmp_path / "reference.avi"
    _write(reference, _frames())
    return tmp_path, reference, region


def _mask_provider(region):
    mask = np.zeros((H, W), dtype=bool)
    y0, y1, x0, x1 = region
    mask[y0:y1, x0:x1] = True
    return lambda index: mask


def _check_lossless(path):
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok or not np.array_equal(frame, _frames()[0]):
        pytest.skip("no lossless codec available in this OpenCV build")


def test_identical_videos_pass_with_zero_difference(clips):
    tmp_path, reference, region = clips
    _check_lossless(reference)
    candidate = tmp_path / "candidate.avi"
    shutil.copy(reference, candidate)
    result = tool.compare_videos(str(reference), str(candidate), _mask_provider(region))
    assert result["frames"] == FRAMES and result["size"] == [W, H]
    assert result["whole_frame"]["mean_abs_diff"] == 0 and result["whole_frame"]["max_abs_diff"] == 0
    assert result["mask"]["mean_abs_diff"] == 0 and result["mask"]["frames_with_mask"] == FRAMES
    assert result["flicker"]["temporal_diff_error"] == 0
    assert result["flicker"]["reference_mean_abs_frame_change"] == result["flicker"]["candidate_mean_abs_frame_change"]
    assert tool.judge(result)["verdict"] == "PASS"


def test_perturbation_inside_the_mask_fails(clips):
    tmp_path, reference, region = clips
    _check_lossless(reference)
    candidate = tmp_path / "candidate.avi"
    _write(candidate, _frames(perturb=6, region=region))
    result = tool.compare_videos(str(reference), str(candidate), _mask_provider(region))
    assert result["mask"]["mean_abs_diff"] == pytest.approx(6.0)
    assert result["mask"]["max_abs_diff"] == 6
    assert result["whole_frame"]["mean_abs_diff"] < result["mask"]["mean_abs_diff"]
    verdict = tool.judge(result)
    assert verdict["verdict"] == "FAIL"
    # alternating sign makes the frame-to-frame change differ too: flicker is flagged
    assert result["flicker"]["temporal_diff_error"] > tool.GATE_TEMPORAL_MAE
    assert not verdict["checks"]["temporal_diff_error"]["passed"]


def test_perturbation_outside_the_mask_does_not_fail_the_masked_gate(clips):
    tmp_path, reference, region = clips
    _check_lossless(reference)
    candidate = tmp_path / "candidate.avi"
    _write(candidate, _frames(perturb=6, region=(0, 20, 0, 30)))
    result = tool.compare_videos(str(reference), str(candidate), _mask_provider(region))
    assert result["whole_frame"]["max_abs_diff"] == 6
    assert result["mask"]["mean_abs_diff"] == 0
    assert tool.judge(result)["verdict"] == "PASS"


def test_small_difference_within_the_gate_passes(clips):
    tmp_path, reference, region = clips
    _check_lossless(reference)
    candidate = tmp_path / "candidate.avi"
    frames = _frames()
    for frame in frames:
        frame[40, 10] = np.clip(frame[40, 10].astype(np.int16) + 1, 0, 255)  # one pixel, one level
    _write(candidate, frames)
    result = tool.compare_videos(str(reference), str(candidate), _mask_provider(region))
    assert 0 < result["mask"]["mean_abs_diff"] < tool.GATE_MASKED_MAE
    assert tool.judge(result)["verdict"] == "PASS"


def test_no_plan_gives_whole_frame_figures_and_no_verdict(clips):
    tmp_path, reference, _ = clips
    candidate = tmp_path / "candidate.avi"
    shutil.copy(reference, candidate)
    result = tool.compare_videos(str(reference), str(candidate))
    assert result["mask"] is None and result["flicker"] is None
    verdict = tool.judge(result)
    assert verdict["verdict"] == "NO VERDICT" and "--plan" in verdict["reason"]


def test_frame_count_and_size_mismatches_are_reported(clips):
    tmp_path, reference, _ = clips
    short = tmp_path / "short.avi"
    _write(short, _frames()[:-2])
    with pytest.raises(tool.CompareError, match="frame counts differ"):
        tool.compare_videos(str(reference), str(short))
    other = tmp_path / "other.avi"
    writer = cv2.VideoWriter(str(other), cv2.VideoWriter_fourcc(*"MJPG"), 4, (W + 16, H))
    for _ in range(FRAMES):
        writer.write(np.zeros((H, W + 16, 3), np.uint8))
    writer.release()
    with pytest.raises(tool.CompareError, match="frame sizes differ"):
        tool.compare_videos(str(reference), str(other))
    with pytest.raises(tool.CompareError, match="not found"):
        tool.compare_videos(str(reference), str(tmp_path / "missing.avi"))


def test_main_exit_codes_output_and_json(clips, capsys):
    tmp_path, reference, region = clips
    _check_lossless(reference)
    candidate = tmp_path / "candidate.avi"
    shutil.copy(reference, candidate)
    out = tmp_path / "report.json"
    assert tool.main([str(reference), str(candidate), "--json", str(out)]) == 3  # no plan: no verdict
    printed = capsys.readouterr().out
    assert "NO VERDICT" in printed and "H.264 CRF 18" in printed and "conservative" in printed
    assert json.loads(out.read_text())["gate"]["verdict"] == "NO VERDICT"
    assert tool.main([str(reference), str(tmp_path / "missing.avi")]) == 2
    assert "error:" in capsys.readouterr().err


def _make_plan(directory, region):
    from videowipe.plan import (MaskAsset, Segment, TemporalResolution, Track, WipePlan,
                                compute_source, save_wipe_plan)
    video = directory / "source.avi"
    _write(video, _frames())
    y0, y1, x0, x1 = region
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    source = compute_source(str(video))
    plan = WipePlan("wipe_plan", 1, source, {}, TemporalResolution(0, 0, 0),
                    MaskAsset("wipe_plan_masks.npz", ""),
                    [Track("r", "region", "remove", "remove", (x0, y0, x1 - 1, y1 - 1), 1, 1,
                           "user", [Segment(0, FRAMES)], "r", mask)])
    json_path, _ = save_wipe_plan(plan, str(directory / "plan"))
    return json_path


def test_real_plan_masks_drive_the_comparison(clips, capsys):
    tmp_path, reference, region = clips
    _check_lossless(reference)
    candidate = tmp_path / "candidate.avi"
    _write(candidate, _frames(perturb=6, region=region))
    plan_path = _make_plan(tmp_path, region)
    assert tool.main([str(reference), str(candidate), "--plan", plan_path]) == 1
    assert "Verdict: FAIL" in capsys.readouterr().out
    same = tmp_path / "same.avi"
    shutil.copy(reference, same)
    assert tool.main([str(reference), str(same), "--plan", plan_path]) == 0
    assert "Verdict: PASS" in capsys.readouterr().out


def test_plan_for_other_dimensions_is_reported_not_a_traceback(clips, capsys):
    tmp_path, reference, region = clips
    plan_path = _make_plan(tmp_path, region)
    big = tmp_path / "big.avi"
    writer = cv2.VideoWriter(str(big), cv2.VideoWriter_fourcc(*"MJPG"), 4, (W * 2, H * 2))
    for _ in range(FRAMES):
        writer.write(np.zeros((H * 2, W * 2, 3), np.uint8))
    writer.release()
    assert tool.main([str(big), str(big), "--plan", plan_path]) == 2
    assert "wrong plan" in capsys.readouterr().err
    assert tool.main([str(reference), str(reference), "--plan", str(tmp_path / "no_plan.json")]) == 2
    assert "cannot use plan" in capsys.readouterr().err


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires ffmpeg")
def test_h264_mp4_inputs_are_accepted(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate=4:duration=2",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(clip)], check=True)
    result = tool.compare_videos(str(clip), str(clip), _mask_provider((40, 56, 10, 80)))
    assert result["frames"] == 8 and result["whole_frame"]["max_abs_diff"] == 0
    assert tool.judge(result)["verdict"] == "PASS"
