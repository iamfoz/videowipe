"""Audio, metadata, chapters and soft subtitles survive cleaning (full runs).

As in ``test_source_timing.py`` the model is a stub and ffmpeg is real; every
case runs through ``STTNInpainter.inpaint``.
"""
import json
import os
import shutil
import subprocess
from fractions import Fraction

import numpy as np
import pytest
from test_source_timing import (  # shared helpers; this module has its own fixtures
    FLASHES, FRAME, HEIGHT, WIDTH, flash_times, click_times, make_clip, packet_times, run,
    upstream_command,
)

from videowipe import fidelity, tracks
from videowipe.inpainters import sttn
from videowipe.inpainters.base import InpaintJob
from videowipe.tasks.base import read_frame_info

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="requires ffmpeg and ffprobe",
)

META = {"title": "Source title", "artist": "Some Artist", "comment": "A comment",
        "creation_time": "2020-05-06T07:08:09Z"}
SRT = "1\n00:00:00,500 --> 00:00:01,500\nHello world\n\n2\n00:00:01,800 --> 00:00:02,800\nSecond line\n\n"
ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 160
PlayResY: 100

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,12,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,5,5,5,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.50,0:00:01.50,Default,,0,0,0,,Styled hello
"""
CHAPTERS = """;FFMETADATA1
[CHAPTER]
TIMEBASE=1/1000
START=0
END=1000
title=One
[CHAPTER]
TIMEBASE=1/1000
START=1000
END=2000
title=Two
"""


@pytest.fixture(autouse=True)
def default_modes(monkeypatch):
    monkeypatch.delenv("VIDEOWIPE_OUTPUT_TRACKS", raising=False)
    monkeypatch.delenv("VIDEOWIPE_SOURCE_TIMING", raising=False)


@pytest.fixture
def stub_model(monkeypatch):
    monkeypatch.setattr(sttn, "_process_segment",
                        lambda frames, *args: [np.zeros_like(f) for f in frames])


# --- Building sources and cleaning them -----------------------------------------------


def audio_input(layout="stereo", noise=False, seconds=2):
    source = "anoisesrc=a=0.3:" if noise else "sine=frequency=440:"
    return ["-f", "lavfi", "-i", f"{source}d={seconds}:sample_rate=48000,aformat=channel_layouts={layout}"]


def make_source(path, audios, metadata=True, chapters=False, subtitles=(), languages=True, extra=()):
    """A 2 s video with *audios* (each ``(input args, codec args)``), global metadata,
    optional chapters and subtitle files ``(path, codec)``. Per-track language and
    title are set when ``languages``."""
    path = str(path)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=s=160x100:r=25:d=2"]
    for inputs, _ in audios:
        cmd += inputs
    for sub, _ in subtitles:
        cmd += ["-i", sub]
    count = 1 + len(audios) + len(subtitles)
    if chapters:
        cmd += ["-i", chapters]
    cmd += ["-map", "0:v"]
    for i in range(len(audios)):
        cmd += ["-map", f"{1 + i}:a"]
    for i in range(len(subtitles)):
        cmd += ["-map", f"{1 + len(audios) + i}:s"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "25"]
    for i, (_, codec) in enumerate(audios):
        cmd += [f"-c:a:{i}", *codec]
        if languages:
            cmd += [f"-metadata:s:a:{i}", f"language={('fra', 'eng', 'deu')[i]}",
                    f"-metadata:s:a:{i}", f"title=Track {i + 1}"]
    for i, (_, codec) in enumerate(subtitles):
        cmd += [f"-c:s:{i}", codec, f"-metadata:s:s:{i}", f"language={('deu', 'eng')[i]}",
                f"-metadata:s:s:{i}", f"title=Subs {i + 1}"]
    if subtitles:
        cmd += ["-disposition:s:0", "default"]
        if len(subtitles) > 1:
            cmd += ["-disposition:s:1", "forced"]
    if metadata:
        for key, value in META.items():
            cmd += ["-metadata", f"{key}={value}"]
    if chapters:
        cmd += ["-map_chapters", str(count)]
    run([*cmd, *extra, path])
    return path


def clean(source, output_dir, trial=None, dual=False):
    """Run the real inpaint path on *source*; Matroska frame counts are taken by
    reading (OpenCV only estimates them there)."""
    os.makedirs(output_dir, exist_ok=True)
    mask = np.zeros((HEIGHT, WIDTH, 1), np.uint8)
    mask[60:90, 20:140] = 1
    reader, info = read_frame_info(str(source))
    count = info["len"]
    if not str(source).endswith(".mp4"):
        count = 0
        while reader.read()[0]:
            count += 1
        reader.release()
        reader, _ = read_frame_info(str(source))
    painter = sttn.STTNInpainter()
    painter.backend = object()
    try:
        job = InpaintJob(str(source), mask, str(output_dir), info["fps"], count, info["W_ori"],
                         info["H_ori"], reader=reader, gap=10, trial_range=trial, dual=dual)
        return painter.inpaint(job).output_path
    finally:
        reader.release()


def listing(path):
    data = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_chapters",
                           "-show_format", "-of", "json", str(path)], text=True).stdout)
    return data


def streams(path, kind):
    return [s for s in listing(path)["streams"] if s["codec_type"] == kind]


def audio_hashes(path, index=0):
    out = run(["ffprobe", "-v", "error", "-select_streams", f"a:{index}", "-show_entries",
               "packet=data_hash", "-show_data_hash", "sha256", "-of", "csv=p=0", str(path)],
              text=True).stdout.split()
    # Newer ffprobe adds an empty field (so a trailing comma) for a packet that
    # carries side data, such as the end trim; only the payload hash matters.
    return [line.split(",")[0] for line in out]


def format_tags(path):
    ignored = {"encoder", "major_brand", "minor_version", "compatible_brands"}
    return {k.lower(): v for k, v in listing(path)["format"].get("tags", {}).items()
            if k.lower() not in ignored}


def chapters(path):
    return [(c["tags"].get("title"), c["start_time"], c["end_time"]) for c in listing(path)["chapters"]]


def subtitle_text(path, index=0):
    return run(["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:s:{index}", "-f", "srt", "-"],
               text=True).stdout


# --- What the output keeps -----------------------------------------------------------


def compare_tracks(source, output):
    """The output's audio and subtitle listings against the source's.

    Compared per stream, in order: codec (copied tracks), channels, channel layout,
    sample rate, language, forced (and, for subtitles, default) dispositions, and the title (held as the
    MP4 track's handler name). Not compared, because they legitimately differ:
    bit_rate (re-encoded tracks and ffmpeg-derived rates), start_time, duration,
    nb_frames, codec_tag, id, handler/encoder tags and anything of the video stream.
    """
    for kind in ("audio", "subtitle"):
        src, out = streams(source, kind), streams(output, kind)
        if kind == "subtitle":
            src = [s for s in src if s["codec_name"] in tracks.TEXT_SUBTITLES]
        assert len(src) == len(out), kind
        for a, b in zip(src, out):
            ta, tb = a.get("tags", {}), b.get("tags", {})
            assert ta.get("language", "und") == tb.get("language", "und")
            if "title" in ta:
                assert tb.get("handler_name") == ta["title"]
            # The MP4 muxer flags the first audio track "default" whatever the source said.
            for flag in ("forced",) if kind == "audio" else ("default", "forced"):
                assert a["disposition"][flag] == b["disposition"][flag], flag
            if kind == "audio":
                assert (a["channels"], a["sample_rate"]) == (b["channels"], b["sample_rate"])
                if a["codec_name"] in tracks.COPY_CODECS:
                    assert a["codec_name"] == b["codec_name"]
                    assert a.get("channel_layout") == b.get("channel_layout")
                else:
                    assert b["codec_name"] == "aac"


@pytest.mark.parametrize("name,codec,layout,suffix", [
    ("aac stereo", ["aac", "-b:a", "160k"], "stereo", "mp4"),
    ("aac 5.1", ["aac", "-b:a", "384k"], "5.1", "mp4"),
    ("ac3 5.1", ["ac3", "-b:a", "448k"], "5.1", "mp4"),
    ("eac3 5.1", ["eac3", "-b:a", "384k"], "5.1", "mp4"),
    ("mp3", ["libmp3lame", "-b:a", "192k"], "stereo", "mp4"),
    ("mp3 mono", ["libmp3lame", "-b:a", "64k"], "mono", "mp4"),
    ("alac", ["alac"], "stereo", "mp4"),
    ("ac3 5.1 in mkv", ["ac3", "-b:a", "448k"], "5.1", "mkv"),
    ("aac in mkv", ["aac", "-b:a", "128k"], "stereo", "mkv"),
])
def test_copy_codecs_are_stream_copied_bit_for_bit(name, codec, layout, suffix, tmp_path, stub_model, capsys):
    source = make_source(tmp_path / f"in.{suffix}", [(audio_input(layout), codec)])
    out = clean(source, tmp_path)
    assert "1 copied" in capsys.readouterr().out
    assert audio_hashes(out) == audio_hashes(source) != []
    compare_tracks(source, out)
    assert format_tags(out).items() >= {"title": "Source title", "artist": "Some Artist",
                                        "comment": "A comment"}.items()


@pytest.mark.parametrize("name,codec,layout,suffix,channels,out_layout", [
    ("opus", ["libopus", "-b:a", "96k"], "stereo", "mkv", 2, "stereo"),
    ("opus in mp4", ["libopus", "-b:a", "96k"], "stereo", "mp4", 2, "stereo"),
    ("flac", ["flac"], "stereo", "mkv", 2, "stereo"),
    ("flac in mp4", ["flac"], "stereo", "mp4", 2, "stereo"),
    ("pcm", ["pcm_s16le"], "stereo", "mkv", 2, "stereo"),
    ("vorbis", ["libvorbis", "-b:a", "96k"], "stereo", "webm", 2, "stereo"),
    ("flac 5.1(side)", ["flac"], "5.1(side)", "mkv", 6, "5.1"),
    ("flac 7.1", ["flac"], "7.1", "mkv", 8, "7.1"),
    ("opus 5.1", ["libopus", "-b:a", "256k"], "5.1", "mkv", 6, "5.1"),
])
def test_other_codecs_are_encoded_to_aac_keeping_channels_and_layout(
        name, codec, layout, suffix, channels, out_layout, tmp_path, stub_model, capsys):
    source = make_source(tmp_path / f"in.{suffix}", [(audio_input(layout, noise=True), codec)],
                         extra=("-c:v", "libvpx") if suffix == "webm" else ())
    out = clean(source, tmp_path)
    assert "0 copied, 1 encoded to AAC" in capsys.readouterr().out
    (audio,) = streams(out, "audio")
    assert (audio["codec_name"], audio["channels"], audio["channel_layout"]) == ("aac", channels, out_layout)
    compare_tracks(source, out)


def test_the_aac_command_follows_the_bitrate_rule(tmp_path, stub_model, monkeypatch):
    source = make_source(tmp_path / "in.mkv", [(audio_input("stereo", noise=True), ["flac"])])
    seen = []
    real = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, *a, **k: (seen.append(list(cmd)) if "-i" in cmd and "-" in cmd else None) or real(cmd, *a, **k))
    out = clean(source, tmp_path)
    cmd = seen[0]
    assert cmd[cmd.index("-c:a:0") + 1:cmd.index("-c:a:0") + 4] == ["aac", "-b:a:0", "192k"]
    (audio,) = streams(out, "audio")
    assert 170_000 < int(audio["bit_rate"]) < 210_000  # noise: the encoder uses what it is given


@pytest.mark.parametrize("codec,channels,bit_rate,expected", [
    ("mp3", 2, 192_000, 192_000),      # lossy: never below its own rate ...
    ("mp3", 2, 64_000, 128_000),       # ... nor below 64 kb/s per channel
    ("opus", 2, 96_000, 128_000),
    ("vorbis", 2, None, 128_000),      # lossy, rate unknown: 64 kb/s per channel
    ("ac3", 6, 448_000, 448_000),
    ("ac3", 2, 640_000, 256_000),      # ... and not above 128 kb/s per channel
    ("flac", 2, None, 192_000),        # lossless: 96 kb/s per channel
    ("pcm_s16le", 2, 1_536_000, 192_000),
    ("flac", 6, None, 576_000),
    ("opus", 1, 24_000, 64_000),
])
def test_bitrate_rule(codec, channels, bit_rate, expected):
    track = tracks.AudioTrack(0, codec, channels, None, bit_rate, None)
    assert tracks.aac_bitrate(track) == expected


@pytest.mark.parametrize("layout,channels,expected", [
    ("stereo", 2, None), ("5.1", 6, None), ("5.1(side)", 6, "5.1"), ("7.1(wide)", 8, "7.1"),
    ("unknown", 6, "5.1"), (None, 8, "7.1"), (None, 2, None),
])
def test_layout_rule(layout, channels, expected):
    assert tracks.aac_layout(tracks.AudioTrack(0, "flac", channels, layout, None, None)) == expected


# --- Several tracks, subtitles, chapters --------------------------------------------------


def test_two_audio_tracks_keep_order_language_and_title(tmp_path, stub_model, capsys):
    source = make_source(tmp_path / "in.mkv", [
        (audio_input("stereo", noise=True), ["libopus", "-b:a", "96k"]),
        (audio_input("5.1"), ["ac3", "-b:a", "448k"]),
    ])
    out = clean(source, tmp_path)
    assert "(1 copied, 1 encoded to AAC)" in capsys.readouterr().out
    first, second = streams(out, "audio")
    assert (first["codec_name"], first["channels"], first["tags"]["language"], first["tags"]["handler_name"]) \
        == ("aac", 2, "fra", "Track 1")
    assert (second["codec_name"], second["channels"], second["tags"]["language"], second["tags"]["handler_name"]) \
        == ("ac3", 6, "eng", "Track 2")
    assert audio_hashes(out, 1) == audio_hashes(source, 1)
    compare_tracks(source, out)


def test_text_subtitles_are_kept_as_mov_text(tmp_path, stub_model):
    (tmp_path / "a.srt").write_text(SRT)
    (tmp_path / "b.ass").write_text(ASS)
    source = make_source(tmp_path / "in.mkv", [(audio_input(), ["aac", "-b:a", "128k"])],
                         subtitles=[(str(tmp_path / "a.srt"), "srt"), (str(tmp_path / "b.ass"), "ass")])
    out = clean(source, tmp_path)
    subs = streams(out, "subtitle")
    assert [s["codec_name"] for s in subs] == ["mov_text", "mov_text"]
    assert [s["tags"]["language"] for s in subs] == ["deu", "eng"]
    assert [s["tags"]["handler_name"] for s in subs] == ["Subs 1", "Subs 2"]
    assert [(s["disposition"]["default"], s["disposition"]["forced"]) for s in subs] == [(1, 0), (0, 1)]
    text = subtitle_text(out, 0)
    assert "Hello world" in text and "Second line" in text
    assert "Styled hello" in subtitle_text(out, 1)
    compare_tracks(source, out)


def test_mov_text_subtitles_in_an_mp4_source(tmp_path, stub_model):
    (tmp_path / "a.srt").write_text(SRT)
    source = make_source(tmp_path / "in.mp4", [(audio_input(), ["aac", "-b:a", "128k"])],
                         subtitles=[(str(tmp_path / "a.srt"), "mov_text")])
    out = clean(source, tmp_path)
    assert "Hello world" in subtitle_text(out)
    compare_tracks(source, out)


def bitmap_subtitle_source(path):
    """A Matroska file with one hand-made DVD (bitmap) subtitle track."""
    import struct
    from videowipe.timing import _element as el, _uint as uint

    rle = bytes([0x21] * 4)
    commands = bytes([0x01, 0x03, 0x32, 0x10, 0x04, 0xFF, 0xF0, 0x05, 0x00, 0xA0, 0x11, 0x05, 0x00,
                      0x53, 0x06, 0x00, 0x04, 0x00, 0x06, 0xFF])
    first = struct.pack(">HH", 0, 12 + len(commands)) + commands
    second = struct.pack(">HH", 0x0200, 12 + len(commands)) + bytes([0x02, 0xFF])
    body = rle + first + second
    packet = struct.pack(">HH", 4 + len(body), 8) + body
    ebml = el("1A45DFA3", el("4286", uint(1)) + el("42F7", uint(1)) + el("42F2", uint(4))
              + el("42F3", uint(8)) + el("4282", b"matroska") + el("4287", uint(4)) + el("4285", uint(2)))
    palette = b"size: 160x100\npalette: " + b", ".join([b"000000", b"ffffff"] + [b"000000"] * 14) + b"\n"
    track = el("AE", el("D7", uint(1)) + el("73C5", uint(1)) + el("83", uint(0x11))
               + el("86", b"S_VOBSUB") + el("63A2", palette) + el("22B59C", b"jpn"))
    clusters = b"".join(
        el("1F43B675", el("E7", uint(t)) + el("A0", el("A1", b"\x81\x00\x00\x00" + packet)
                                              + el("9B", uint(1000)))) for t in (500, 1800))
    subs = str(path) + ".subs.mks"
    with open(subs, "wb") as handle:
        handle.write(ebml + el("18538067", el("1549A966", el("2AD7B1", uint(1000000)))
                               + el("1654AE6B", track) + clusters))
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=s=160x100:r=25:d=2",
         "-f", "lavfi", "-i", "sine=d=2", "-i", subs, "-map", "0:v", "-map", "1:a", "-map", "2:s",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-c:s", "copy", str(path)])


def test_bitmap_subtitles_are_skipped_with_a_notice(tmp_path, stub_model, capsys):
    source = tmp_path / "in.mkv"
    bitmap_subtitle_source(source)
    assert [s["codec_name"] for s in streams(source, "subtitle")] == ["dvd_subtitle"]
    out = clean(source, tmp_path)
    text = capsys.readouterr().out
    assert "Skipping subtitle track 1 (dvd_subtitle)" in text
    assert streams(out, "subtitle") == []
    assert audio_hashes(out) == audio_hashes(source)


def test_chapters_and_metadata_survive(tmp_path, stub_model):
    (tmp_path / "chapters.ffmeta").write_text(CHAPTERS)
    for suffix in ("mp4", "mkv"):
        source = make_source(tmp_path / f"in.{suffix}", [(audio_input(), ["aac", "-b:a", "128k"])],
                             chapters=str(tmp_path / "chapters.ffmeta"))
        out = clean(source, tmp_path / suffix)
        found = chapters(out)
        assert [c[0] for c in found] == ["One", "Two"]
        if suffix == "mp4":
            assert found == chapters(source)
        assert format_tags(out)["title"] == "Source title"
        assert format_tags(out)["creation_time"].startswith("2020-05-06T07:08:09")


def test_a_source_without_audio_or_metadata_still_works(tmp_path, stub_model):
    source = make_source(tmp_path / "in.mp4", [], metadata=False)
    out = clean(source, tmp_path)
    assert streams(out, "audio") == []


# --- Composes with the source's frame timestamps --------------------------------------------


def vfr_sources(tmp_path):
    kept = [i for i in range(200) if i not in set(range(20, 50)) | set(range(180, 190))]
    base = tmp_path / "vfr.mp4"
    make_clip(base, kept, [(20, 49), (180, 189)], FLASHES)
    # AAC in Matroska: the audio starts 21 ms before the video (encoder delay).
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(base), "-c", "copy", str(tmp_path / "vfr.mkv")])
    late = tmp_path / "late.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-itsoffset", "0.5", "-i", str(base), "-i", str(base),
         "-map", "0:v", "-map", "1:a", "-c", "copy", str(late)])
    return {"mp4": base, "mkv": tmp_path / "vfr.mkv", "video starts after audio": late}


@pytest.fixture(scope="module")
def vfr(tmp_path_factory):
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("requires ffmpeg and ffprobe")
    return vfr_sources(tmp_path_factory.mktemp("vfr_tracks"))


def sync_offsets(path):
    return [float(f) - float(c) for f, c in zip(flash_times(path), click_times(path))]


@pytest.mark.parametrize("kind", ["mp4", "mkv", "video starts after audio"])
def test_copied_audio_stays_in_sync_with_variable_rate_video(kind, vfr, tmp_path, stub_model, capsys):
    source = vfr[kind]
    out = clean(source, tmp_path)
    text = capsys.readouterr().out
    assert "1 copied" in text and "Keeping the source's frame timestamps" in text
    assert audio_hashes(out) == audio_hashes(source)
    # Video timestamps are the source's, relative to the container's start (as the audio is).
    start = Fraction(listing(source)["format"]["start_time"])
    assert packet_times(out) == [t - start for t in packet_times(source)]
    before, after = sync_offsets(source), sync_offsets(out)
    assert len(after) == len(FLASHES)
    assert all(abs(a - b) <= FRAME / 4 for a, b in zip(after, before))
    # The offset between the picture and the sound is the source's (0.5 s when the
    # video starts late), not one frame of it.
    assert max(abs(a - b) for a, b in zip(after, before)) < 0.005


def test_the_acceptance_check_needs_the_new_default(vfr, tmp_path, stub_model, monkeypatch):
    """The same listing check fails with the switch off (audio re-encoded, metadata gone)."""
    out = clean(vfr["mp4"], tmp_path)
    assert audio_hashes(out) == audio_hashes(vfr["mp4"])
    monkeypatch.setenv("VIDEOWIPE_OUTPUT_TRACKS", "off")
    out_off = clean(vfr["mp4"], tmp_path / "off")
    assert audio_hashes(out_off) != audio_hashes(vfr["mp4"])
    assert format_tags(out_off) == {}


# --- Exactness and fallbacks ----------------------------------------------------------------


class Tee:
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
        if "rawvideo" in cmd:
            seen["cmds"].append(list(cmd))
            pipe.stdin = Tee(pipe.stdin, seen["stdin"])
        return pipe

    monkeypatch.setattr(subprocess, "Popen", popen)
    return seen


@pytest.fixture(scope="module")
def constant(tmp_path_factory):
    path = tmp_path_factory.mktemp("constant_tracks") / "constant.mp4"
    make_source(path, [(audio_input(), ["aac", "-b:a", "128k"])])
    return path


def oracle(source, output, first=0, last=None, trial=None):
    reader, info = read_frame_info(str(source))
    reader.release()
    job = InpaintJob(str(source), None, "unused", info["fps"], info["len"], WIDTH, HEIGHT, trial_range=trial)
    out_h = HEIGHT * (2 if trial else 1)
    return upstream_command(job, output, WIDTH, out_h, first, last if last is not None else info["len"])


@pytest.mark.parametrize("mode", ["off", "trial"])
def test_switch_off_and_trials_give_upstreams_audio_handling(mode, constant, tmp_path, stub_model, spy, monkeypatch):
    if mode == "off":
        monkeypatch.setenv("VIDEOWIPE_OUTPUT_TRACKS", "off")
    trial = (10, 30) if mode == "trial" else None
    out = clean(constant, tmp_path, trial=trial, dual=trial is not None)
    assert spy["cmds"] == [oracle(constant, out, *(trial or (0, None)), trial=trial)]
    assert all(len(chunk) == WIDTH * HEIGHT * 3 * (2 if trial else 1) for chunk in spy["stdin"])


def test_the_default_changes_only_audio_and_metadata_options(constant, tmp_path, stub_model, spy):
    out = clean(constant, tmp_path)
    expected = oracle(constant, out)
    i = expected.index("-map", expected.index("-map") + 2)  # "-map 1:a?"
    assert expected[i:i + 2] == ["-map", "1:a?"]
    expected[i:i + 2] = ["-map", "1:a:0"]
    j = expected.index("-c:a")
    expected[j:j + 2] = ["-c:a:0", "copy", "-map_metadata", "1"]
    assert spy["cmds"] == [expected]
    frames = [bytes(chunk) for chunk in spy["stdin"]]
    assert len(frames) == 50 and all(len(f) == WIDTH * HEIGHT * 3 for f in frames)  # raw frames, as before


def test_probe_failure_gives_upstreams_command_and_a_warning(constant, tmp_path, stub_model, spy,
                                                              monkeypatch, capsys):
    monkeypatch.setattr(tracks, "probe_tracks", lambda path: None)
    out = clean(constant, tmp_path)
    assert spy["cmds"] == [oracle(constant, out)]
    assert "could not read the source's audio and subtitle tracks" in capsys.readouterr().out


def test_options_this_ffmpeg_rejects_give_upstreams_command_and_a_warning(constant, tmp_path, stub_model, spy,
                                                                          monkeypatch, capsys):
    monkeypatch.setattr(tracks.OutputTracks, "works_here", lambda self: False)
    out = clean(constant, tmp_path)
    assert spy["cmds"] == [oracle(constant, out)]
    assert "this ffmpeg cannot keep the source's tracks" in capsys.readouterr().out


def test_an_unrecognised_command_is_left_alone(capsys):
    plan = tracks.OutputTracks(tracks.SourceTracks((tracks.AudioTrack(0, "aac", 2, "stereo", None, None),)), "x")
    assert plan.rewrite(["ffmpeg", "-i", "-", "out.mp4"]) is None
    assert plan.rewrite(["ffmpeg", "-map", "1:a?", "-c:a", "aac", "-t", "1", "-movflags", "+faststart", "o.mp4"]) is None
    command = fidelity.EncodePlan(fidelity.plan_source_timing.__globals__["SourceTiming"](), plan)
    assert command.command(["ffmpeg", "-i", "-", "out.mp4"]) == ["ffmpeg", "-i", "-", "out.mp4"]
    assert "could not keep the source's audio, metadata and subtitle tracks" in capsys.readouterr().out


def test_the_switch_values(monkeypatch):
    assert tracks.tracks_mode() == "auto"
    monkeypatch.setenv("VIDEOWIPE_OUTPUT_TRACKS", " OFF ")
    assert tracks.tracks_mode() == "off"
    monkeypatch.setenv("VIDEOWIPE_OUTPUT_TRACKS", "nonsense")
    assert tracks.tracks_mode() == "auto"


def test_a_second_output_gets_the_source_metadata(tmp_path, stub_model, monkeypatch):
    """The optional HEVC copy is another output on the same command; it has its own
    mapping, so it only needs the source's global metadata added."""
    source = make_source(tmp_path / "in.mp4", [(audio_input(), ["aac", "-b:a", "128k"])])
    second = tmp_path / "second.mkv"
    original = fidelity.EncodePlan.popen
    monkeypatch.setattr(fidelity.EncodePlan, "popen", lambda self, cmd, stderr: original(
        self, cmd + ["-map", "0:v", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(second)], stderr))
    out = clean(source, tmp_path)
    assert format_tags(out)["title"] == "Source title"
    assert format_tags(second)["title"] == "Source title"
    assert format_tags(second)["artist"] == "Some Artist"
    assert [s["codec_type"] for s in listing(second)["streams"]] == ["video"]


def test_parse_tracks():
    text = json.dumps({"streams": [
        {"codec_type": "video", "codec_name": "h264"},
        {"codec_type": "audio", "codec_name": "ac3", "channels": 6, "channel_layout": "5.1(side)",
         "bit_rate": "448000", "tags": {"title": "Main"}},
        {"codec_type": "audio", "codec_name": "opus", "channels": 2, "bit_rate": "N/A"},
        {"codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle"},
        {"codec_type": "subtitle", "codec_name": "subrip", "tags": {"title": "English"}},
    ]})
    parsed = tracks.parse_tracks(text)
    assert [(t.index, t.codec, t.bit_rate, t.title, t.copied) for t in parsed.audio] == [
        (0, "ac3", 448000, "Main", True), (1, "opus", None, None, False)]
    assert [(t.index, t.convertible) for t in parsed.subtitles] == [(0, False), (1, True)]
    assert tracks.parse_tracks("not json") is None
