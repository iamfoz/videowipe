from __future__ import annotations

import subprocess
from types import SimpleNamespace

from videowipe import propainter_wipe


def test_propainter_requires_explicit_checkout(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("VIDEOWIPE_PROPINTER_DIR", raising=False)

    result = propainter_wipe.main([
        str(tmp_path / "video.mp4"),
        str(tmp_path / "mask.png"),
        str(tmp_path / "output"),
    ])

    assert result == 2
    assert "not configured" in capsys.readouterr().err


def test_propainter_discards_child_output_by_default(
    monkeypatch, tmp_path, capsys
):
    checkout = tmp_path / "ProPainter"
    checkout.mkdir()
    (checkout / "inference_propainter.py").write_text("", encoding="utf-8")
    video = tmp_path / "video.mp4"
    mask = tmp_path / "mask.png"
    video.write_bytes(b"video")
    mask.write_bytes(b"mask")

    monkeypatch.setattr(propainter_wipe, "extract_frames", lambda *args: 24)
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(propainter_wipe.subprocess, "run", fake_run)

    result = propainter_wipe.main([
        str(video),
        str(mask),
        str(tmp_path / "output"),
        "--propainter-dir",
        str(checkout),
    ])

    assert result == 1
    assert captured["stdout"] == subprocess.DEVNULL
    assert captured["stderr"] == subprocess.DEVNULL
    assert "capture_output" not in captured
    assert "exited with code 7" in capsys.readouterr().err


def _adapter_run(monkeypatch, tmp_path, fps, make_result):
    """Run the adapter with a fake ProPainter that writes *make_result*'s files."""
    checkout = tmp_path / "ProPainter"
    checkout.mkdir()
    (checkout / "inference_propainter.py").write_text("", encoding="utf-8")
    video = tmp_path / "clip.mp4"
    mask = tmp_path / "mask.png"
    video.write_bytes(b"video")
    mask.write_bytes(b"mask")
    seen = {}

    monkeypatch.setattr(propainter_wipe, "extract_frames", lambda *args: fps)

    def fake_run(command, **kwargs):
        seen["command"] = command
        frames_dir = command[command.index("-i") + 1]
        make_result(command[command.index("-o") + 1], frames_dir)
        return SimpleNamespace(returncode=0)

    def fake_encode(frames_dir, output_path, rate):
        seen["encoded"] = (frames_dir, rate)
        with open(output_path, "wb") as handle:
            handle.write(b"encoded")

    monkeypatch.setattr(propainter_wipe.subprocess, "run", fake_run)
    monkeypatch.setattr(propainter_wipe, "frames_to_video", fake_encode)
    output = tmp_path / "output"
    result = propainter_wipe.main([
        str(video), str(mask), str(output), "--propainter-dir", str(checkout),
    ])
    return result, seen, output / "inpaint_out.mp4"


def test_propainter_gets_a_whole_rate_and_frames_keep_the_real_one(monkeypatch, tmp_path):
    """ProPainter's --save_fps is an integer argument and it names its result
    folder after the frames directory it was given."""
    import os

    def result(pp_output, frames_dir):
        root = os.path.join(pp_output, os.path.basename(frames_dir))
        os.makedirs(os.path.join(root, "frames"))
        with open(os.path.join(root, "inpaint_out.mp4"), "wb") as handle:
            handle.write(b"whole-rate video")

    code, seen, destination = _adapter_run(monkeypatch, tmp_path, 30000 / 1001, result)

    assert code == 0
    command = seen["command"]
    rate = command[command.index("--save_fps") + 1]
    assert rate == "30" and int(rate) == 30  # 29.97 used to be passed as 29
    assert "--save_frames" in command
    assert seen["encoded"][1] == 30000 / 1001  # encoded here at the real rate
    assert destination.read_bytes() == b"encoded"


def test_propainter_video_is_used_when_no_frames_were_saved(monkeypatch, tmp_path, capsys):
    import os

    def result(pp_output, frames_dir):
        root = os.path.join(pp_output, os.path.basename(frames_dir))
        os.makedirs(root)
        with open(os.path.join(root, "inpaint_out.mp4"), "wb") as handle:
            handle.write(b"whole-rate video")

    code, seen, destination = _adapter_run(monkeypatch, tmp_path, 30000 / 1001, result)

    assert code == 0 and "encoded" not in seen
    assert destination.read_bytes() == b"whole-rate video"
    assert "30 fps instead of 29.970" in capsys.readouterr().out


def test_a_result_named_after_the_source_is_still_found(monkeypatch, tmp_path, capsys):
    import os

    def result(pp_output, frames_dir):
        root = os.path.join(pp_output, "clip")
        os.makedirs(root)
        with open(os.path.join(root, "inpaint_out.mp4"), "wb") as handle:
            handle.write(b"video")

    code, seen, destination = _adapter_run(monkeypatch, tmp_path, 25.0, result)

    assert code == 0 and destination.read_bytes() == b"video"
    assert "Warning" not in capsys.readouterr().out


def test_missing_propainter_output_is_an_error(monkeypatch, tmp_path, capsys):
    code, _, destination = _adapter_run(monkeypatch, tmp_path, 25.0, lambda *args: None)

    assert code == 1 and not destination.exists()
    assert "output not found" in capsys.readouterr().err
