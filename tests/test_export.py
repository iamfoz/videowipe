"""H.265 export: preset choice, ffmpeg arguments and codec selection."""
from __future__ import annotations

import pytest

from videowipe import export


@pytest.mark.parametrize("height,quality,crf", [(480, 50, 20), (576, 50, 20), (720, 45, 22), (1080, 45, 22), (2160, 45, 22)])
def test_preset_by_resolution(height, quality, crf):
    preset = export.choose_preset(height)
    assert (preset.vt_quality, preset.x265_crf) == (quality, crf)


def test_videotoolbox_args(monkeypatch):
    monkeypatch.setattr(export, "_encoder_help", lambda name: "Encoder hevc_videotoolbox ... -prio_speed")
    args = export.video_args("hevc_videotoolbox", export.choose_preset(1080))
    assert args[:8] == ["-c:v", "hevc_videotoolbox", "-q:v", "45", "-profile:v", "main10", "-pix_fmt", "p010le"]
    assert "-prio_speed" in args and ["-tag:v", "hvc1"] == args[-4:-2]


def test_videotoolbox_args_without_prio_speed(monkeypatch):
    monkeypatch.setattr(export, "_encoder_help", lambda name: "Encoder hevc_videotoolbox")
    assert "-prio_speed" not in export.video_args("hevc_videotoolbox", export.choose_preset(720))


def test_x265_fallback_args():
    args = export.video_args("libx265", export.choose_preset(576))
    assert args[:4] == ["-c:v", "libx265", "-crf", "20"]
    assert "yuv420p10le" in args and "main10" in args


def test_audio_args_copy_stereo_and_surround():
    tracks = [{"codec": "opus", "channels": 2}, {"codec": "aac", "channels": 2},
              {"codec": "ac3", "channels": 6}, {"codec": "aac", "channels": 1}]
    args = export.audio_args(tracks, export.choose_preset(1080))
    assert args == [
        "-c:a:0", "copy",
        "-c:a:1", "libopus", "-b:a:1", "80k",
        "-c:a:2", "libopus", "-b:a:2", "256k", "-filter:a:2", "aformat=channel_layouts=5.1",
        "-c:a:3", "libopus", "-b:a:3", "80k",
    ]


def test_no_extra_output_for_h264(monkeypatch):
    monkeypatch.delenv("VIDEOWIPE_OUTPUT_CODEC", raising=False)
    assert export.hevc_output_args("in.mp4", "out.mp4", 1080) == []


def test_hevc_output_args(monkeypatch):
    monkeypatch.setattr(export, "probe_audio", lambda path: [{"codec": "aac", "channels": 2}])
    monkeypatch.setattr(export, "video_encoder", lambda: "libx265")
    with export.output_codec("h265"):
        args = export.hevc_output_args("in.webm", "/x/out_clean.mp4", 1080)
    assert args[:4] == ["-map", "0:v", "-map", "1:a:0"]
    assert args[-1] == "/x/out_clean.mkv"
    assert export.current_codec() == "h264"


def test_unknown_codec_rejected():
    with pytest.raises(ValueError):
        with export.output_codec("vp9"):
            pass


def test_env_default(monkeypatch):
    monkeypatch.setenv("VIDEOWIPE_OUTPUT_CODEC", "H265")
    assert export.current_codec() == "h265"


def test_keep_only_hevc(tmp_path):
    (tmp_path / "a_clean.mp4").write_bytes(b"x")
    (tmp_path / "a_clean.mkv").write_bytes(b"y")
    (tmp_path / "b_clean.mp4").write_bytes(b"x")
    kept = export.keep_only_hevc(str(tmp_path), since=0)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a_clean.mkv", "b_clean.mp4"]
    assert kept == [str(tmp_path / "a_clean.mkv")]
