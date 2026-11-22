"""The ffmpeg frame reader: same frames as OpenCV, in the same order, with the right colours.

Real ffmpeg (skipped without it) and real OpenCV. ``scripts/verify_decode_frames.py``
compares the two readers frame for frame (with the reader set to the conversion this
OpenCV measurably uses, the frames must be identical, which also proves their order);
``scripts/decode_clipping.py`` measures the colour error of each decoder on a grid of
known colours.
"""
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from videowipe import colour, decode

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "input" / "detext_examples"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify = load_script("verify_decode_frames")
clipping = load_script("decode_clipping")

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="requires ffmpeg and ffprobe")
pytestmark = needs_ffmpeg


@pytest.fixture(autouse=True)
def default_modes(monkeypatch):
    monkeypatch.delenv("VIDEOWIPE_DECODE", raising=False)
    monkeypatch.delenv("VIDEOWIPE_COLOUR", raising=False)
    monkeypatch.setenv("VIDEOWIPE_OUTPUT_TRACKS", "off")
    decode._decide.cache_clear()
    colour.decoder_conversion.cache_clear()
    colour._reader_plan.cache_clear()
    decode._announced.clear()


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    return verify.generate(tmp_path_factory.mktemp("decode-cases"), frames=60)


# --- Pure functions -------------------------------------------------------------------------


def source(**fields):
    base = dict(width=1280, height=720, pix_fmt="yuv420p", space=None, primaries=None,
                transfer=None, range=None)
    return colour.SourceColour(**{**base, **fields})


@pytest.mark.parametrize("fields, expected", [
    (dict(), ("bt709", "tv")),                                   # untagged HD: players use 709
    (dict(width=1025, height=100), ("bt709", "tv")),             # wider than 1024
    (dict(width=720, height=577), ("bt709", "tv")),              # taller than 576
    (dict(width=1024, height=576), ("bt601", "tv")),             # SD
    (dict(width=720, height=480), ("bt601", "tv")),
    (dict(space="smpte170m"), ("bt601", "tv")),                  # a tag beats the size
    (dict(space="bt470bg", width=720, height=576), ("bt601", "tv")),
    (dict(space="bt709", width=720, height=576), ("bt709", "tv")),
    (dict(space="bt2020nc"), ("bt2020", "tv")),
    (dict(range="pc"), ("bt709", "pc")),
    (dict(space="bt709", range="pc"), ("bt709", "pc")),
    (dict(space="smpte240m"), None),                             # not converted here
    (dict(space="ycgco"), None),
])
def test_conversion_a_player_uses(fields, expected):
    assert decode.source_conversion(source(**fields)) == expected


def test_mode_switch(monkeypatch):
    assert decode.decode_mode() == "auto"
    for value, expected in [("opencv", "opencv"), (" FFMPEG ", "ffmpeg"), ("auto", "auto"),
                            ("on", "auto"), ("", "auto")]:
        monkeypatch.setenv("VIDEOWIPE_DECODE", value)
        assert decode.decode_mode() == expected


def test_reader_command_never_seeks_or_resamples_time():
    cmd = decode.reader_command("in.mp4", "bt709", "tv")
    assert "-ss" not in cmd and "-r" not in cmd and "-vf" in cmd
    assert cmd[cmd.index("-fps_mode") + 1] == "passthrough"
    assert "scale=in_color_matrix=bt709:in_range=tv,format=bgr24" in cmd
    assert cmd[-1] == "-" and cmd[cmd.index("-map") + 1] == "0:v:0"


# --- The reader as a capture ----------------------------------------------------------------


def test_it_behaves_like_a_capture(cases):
    path = str(cases["bframes"])
    ours = decode.FfmpegFrameReader(path, "bt601", "tv")
    theirs = cv2.VideoCapture(path)
    try:
        assert isinstance(ours, decode.CAPTURE_TYPES) and ours.isOpened()
        for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS,
                     cv2.CAP_PROP_FRAME_COUNT):
            assert ours.get(prop) == theirs.get(prop)
        assert ours.set(cv2.CAP_PROP_POS_FRAMES, 5) is False  # never seeks
        # grab() advances exactly as read() does; retrieve() gives the grabbed frame
        for _ in range(3):
            assert ours.grab() and theirs.grab()
        ok_o, a = ours.retrieve()
        ok_t, b = theirs.retrieve()
        assert ok_o and ok_t and np.array_equal(a, b)
        ok_o, a = ours.read()
        ok_t, b = theirs.read()
        assert np.array_equal(a, b)
        assert ours.frames_delivered == 4
        # the frame read() returns belongs to the caller: it can be edited in place
        a[:] = 0
        ok_o, c = ours.read()
        assert c.any()
    finally:
        ours.release()
        theirs.release()


def test_the_end_of_the_stream_is_a_failed_read_and_release_stops_ffmpeg(cases):
    path = str(cases["odd_size"])
    reader = decode.FfmpegFrameReader(path, "bt601", "tv")
    count = 0
    while reader.grab():
        count += 1
    assert count == 40
    assert reader.read() == (False, None) and reader.retrieve()[0] is False
    proc = reader._proc
    reader.release()
    reader.release()  # twice is fine
    assert proc.poll() is not None
    assert reader.read() == (False, None)
    # releasing in the middle stops the process too
    reader = decode.FfmpegFrameReader(path, "bt601", "tv")
    reader.read()
    proc = reader._proc
    reader.release()
    assert proc.poll() is not None


def test_a_missing_file_is_not_opened(tmp_path):
    reader = decode.FfmpegFrameReader(str(tmp_path / "missing.mp4"), "bt601", "tv")
    assert not reader.isOpened()
    assert reader.read() == (False, None)
    reader.release()


# --- Same frames, same order (C1) -----------------------------------------------------------

#: with the reader converting as this OpenCV does, the frames must be identical (they
#: are, bit for bit, for 8-bit sources; 10-bit goes through a different swscale path)
IDENTICAL = ["bframes", "vfr_mp4", "vfr_webm", "vfr_vp8_webm", "editlist_cut", "editlist_delay",
             "editlist_cut_vfr", "hevc_mkv", "rotated", "odd_size", "bt709_tagged", "yuv422p",
             "yuv444p", "full_range"]


@pytest.mark.parametrize("name", IDENTICAL)
def test_generated_sources_yield_opencvs_frames_in_opencvs_order(name, cases):
    if name not in cases:
        pytest.skip("this ffmpeg cannot encode that source")
    result = verify.compare(str(cases[name]))
    assert result["opencv_frames"] == result["ffmpeg_frames"] > 20
    assert not result["ffmpeg_ends_early"] and not result["ffmpeg_has_extra"]
    assert result["shape_mismatch"] == 0 and result["misordered"] == 0
    assert result["max_pixel_diff"] == 0 and result["ok"]


def test_the_generated_awkward_sources_are_awkward(cases):
    """The cases above would prove little if they were not: an edit list that hides
    frames (OpenCV yields fewer than the container holds), variable timestamps, and
    a delayed start."""
    def probe(path, *entries):
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              *entries, "-of", "csv=p=0", str(path)], capture_output=True, text=True)
        return out.stdout.split()

    assert b"elst" in Path(cases["editlist_cut"]).read_bytes()
    cut = verify.compare(str(cases["editlist_cut"]))
    assert cut["opencv_frames"] < 60  # frames before the cut are hidden by the edit list
    assert b"elst" in Path(cases["editlist_delay"]).read_bytes()
    assert float(probe(cases["editlist_delay"], "stream=start_time")[0]) > 1.0
    steps = {round(float(t) - float(u), 3)
             for t, u in zip(*(lambda xs: (xs[1:], xs[:-1]))(
                 sorted(float(x) for x in probe(cases["vfr_mp4"], "packet=pts_time"))))}
    assert len(steps) > 3  # the frame spacing varies


@pytest.mark.parametrize("name", ["english1", "chinese1", "others"])
def test_bundled_samples_in_step_with_opencv(name):
    # the first 120 frames of each (the full runs are in docs/decode.md)
    result = verify.compare(str(SAMPLES / f"{name}.mp4"), max_frames=120)
    assert result["opencv_frames"] == result["ffmpeg_frames"] == 120
    assert result["misordered"] == 0 and result["max_pixel_diff"] == 0 and result["ok"]


def test_ten_bit_frames_correspond_even_though_the_conversion_differs(cases):
    result = verify.compare(str(cases["yuv420p10"]))
    assert result["opencv_frames"] == result["ffmpeg_frames"] == 40
    assert result["misordered"] == 0 and result["shape_mismatch"] == 0


def test_a_dropped_frame_would_be_noticed(cases, monkeypatch):
    """The comparison is only worth something if it can fail: an ffmpeg that dropped
    one frame in the middle shifts every later frame by one."""
    real = decode.FfmpegFrameReader.read
    state = {"n": 0}

    def skipping(self, image=None):
        state["n"] += 1
        if state["n"] == 20:
            real(self)
        return real(self)

    monkeypatch.setattr(decode.FfmpegFrameReader, "read", skipping)
    result = verify.compare(str(cases["bframes"]))
    assert result["misordered"] > 0 and not result["ok"]


def test_the_start_up_check_notices_a_shifted_or_missing_start(cases, monkeypatch):
    path = str(cases["bframes"])
    sync = ("-fps_mode", "passthrough")
    assert decode._matches_opencv(path, "bt601", "tv", sync) is None
    # a different conversion is a different colour, not a different frame
    assert decode._matches_opencv(path, "bt709", "tv", sync) is None
    real = decode.FfmpegFrameReader.read
    state = {"first": True}

    def late(self, image=None):
        if state["first"]:
            state["first"] = False
            real(self)  # the reader's first frame is lost: every frame is one off
        return real(self)

    monkeypatch.setattr(decode.FfmpegFrameReader, "read", late)
    assert "does not correspond" in decode._matches_opencv(path, "bt601", "tv", sync)


# --- Colour -----------------------------------------------------------------------------------


def test_the_reader_gets_back_the_colours_a_wrong_matrix_clips(tmp_path):
    grid = clipping.measure_grid(tmp_path)
    for name in ("untagged HD", "BT.709 tagged HD"):
        row = grid[name]
        assert row["ffmpeg"]["patches_wrong_over_8"] == 0 and max(row["ffmpeg"]["max_err_rgb"]) <= 3
        # a decoder that reads the source as BT.601 loses colours to clipping
        assert row["bt601-only"]["patches_wrong_over_8"] > 300
        assert row["bt601-only"]["patches_clipped"] > 100 and max(row["bt601-only"]["max_err_rgb"]) > 30
    # this OpenCV reads untagged HD as BT.601 (the reason for the reader), and tagged HD as BT.709
    assert grid["untagged HD"]["opencv"]["patches_wrong_over_8"] > 300
    assert grid["BT.709 tagged HD"]["opencv"]["patches_wrong_over_8"] == 0


# --- The decision -----------------------------------------------------------------------------


def make(tmp_path, name, size="1280x720", pix_fmt="yuv420p", *options):
    path = tmp_path / name
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25",
                    "-frames:v", "8", "-c:v", "libx264", "-crf", "10", "-pix_fmt", pix_fmt, *options,
                    str(path)], check=True)
    return str(path)


def test_untagged_hd_is_decoded_by_ffmpeg_and_untagged_sd_by_opencv(tmp_path, capsys):
    hd = make(tmp_path, "hd.mp4")
    sd = make(tmp_path, "sd.mp4", "720x480")
    plan = decode.decide(hd)
    assert plan[:2] == ("bt709", "tv") and "BT.709" in plan[3]
    assert decode.decide(sd) is None  # OpenCV already reads BT.601 SD correctly
    reader = decode.open_capture(hd)
    assert isinstance(reader, decode.FfmpegFrameReader) and reader.matrix == "bt709"
    reader.release()
    capture = decode.open_capture(sd)
    assert type(capture) is cv2.VideoCapture
    capture.release()
    assert "Warning" not in capsys.readouterr().out


def test_where_opencv_already_converts_correctly_it_keeps_decoding(tmp_path, monkeypatch):
    """A decoder that honours the tag (this machine's) needs no replacing; one that
    reads every source as BT.601 (the reference Mac's) does."""
    tagged = make(tmp_path, "tagged.mp4", "1280x720", "yuv420p", "-colorspace", "bt709")
    assert decode.decide(tagged) is None  # honoured here: unchanged
    decode._decide.cache_clear()
    monkeypatch.setattr(colour, "decoder_conversion", lambda *args: ("bt601", "tv"))
    plan = decode.decide(tagged)
    assert plan[:2] == ("bt709", "tv") and "reads the source as BT.601" in plan[3]
    # a full-range tag the decoder honours but a matrix it does not
    full = make(tmp_path, "full.mp4", "1280x720", "yuv420p", "-colorspace", "bt709", "-color_range", "pc")
    decode._decide.cache_clear()
    monkeypatch.setattr(colour, "decoder_conversion", lambda *args: ("bt601", "pc"))
    assert decode.decide(full)[:2] == ("bt709", "pc")


def test_opencv_mode_and_missing_files_decode_with_opencv(tmp_path, monkeypatch):
    hd = make(tmp_path, "hd.mp4")
    monkeypatch.setenv("VIDEOWIPE_DECODE", "opencv")
    assert decode.decide(hd) is None
    assert type(decode.open_capture(hd)) is cv2.VideoCapture
    monkeypatch.setenv("VIDEOWIPE_DECODE", "ffmpeg")
    assert decode.decide(str(tmp_path / "missing.mp4")) is None
    assert decode.decide("") is None


def test_ffmpeg_mode_forces_the_reader_even_where_opencv_is_right(tmp_path, monkeypatch):
    sd = make(tmp_path, "sd.mp4", "720x480")
    monkeypatch.setenv("VIDEOWIPE_DECODE", "ffmpeg")
    plan = decode.decide(sd)
    assert plan[:2] == ("bt601", "tv") and "VIDEOWIPE_DECODE=ffmpeg" in plan[3]


def test_a_high_bit_depth_source_is_read_by_ffmpeg(tmp_path):
    ten = make(tmp_path, "ten.mp4", "320x240", "yuv420p10le", "-colorspace", "bt709")
    plan = decode.decide(ten)
    assert plan[:2] == ("bt709", "tv") and "high-bit-depth" in plan[3]


@pytest.mark.parametrize("name, pix_fmt, options, expected", [
    ("hdr.mp4", "yuv420p10le", ("-color_trc", "smpte2084", "-colorspace", "bt2020nc",
                                 "-color_primaries", "bt2020"), "HDR"),
    ("odd_matrix.mp4", "yuv420p", ("-colorspace", "smpte240m"), "smpte240m"),
    ("rgb.mp4", "rgb24", ("-c:v", "libx264rgb"), "not YCbCr"),
])
def test_what_it_cannot_read_falls_back_with_one_line(tmp_path, monkeypatch, capsys, name, pix_fmt,
                                                      options, expected):
    path = make(tmp_path, name, "1280x720", pix_fmt, *options)
    monkeypatch.setenv("VIDEOWIPE_DECODE", "ffmpeg")
    assert decode.decide(path) is None
    lines = [l for l in capsys.readouterr().out.splitlines() if l]
    assert len(lines) == 1 and lines[0].startswith("Warning:") and expected in lines[0]
    assert lines[0].endswith("decoding with OpenCV.")


def test_colour_off_keeps_opencv_because_the_encode_would_not_match(tmp_path, monkeypatch, capsys):
    hd = make(tmp_path, "hd.mp4")
    monkeypatch.setenv("VIDEOWIPE_COLOUR", "off")
    assert decode.decide(hd) is None
    assert capsys.readouterr().out == ""  # auto: nothing to say, the job is upstream's
    decode._decide.cache_clear()
    monkeypatch.setenv("VIDEOWIPE_DECODE", "ffmpeg")
    assert decode.decide(hd) is None
    assert "needs VIDEOWIPE_COLOUR=auto" in capsys.readouterr().out


@pytest.mark.parametrize("what, expected", [
    ("capability", "did not decode known colours"),
    ("encode", "did not encode with the matrix"),
    ("frames", "do not match OpenCV's (frame 3 does not correspond)"),
])
def test_every_failure_falls_back_with_one_line(tmp_path, monkeypatch, capsys, what, expected):
    hd = make(tmp_path, "hd.mp4")
    if what == "capability":
        monkeypatch.setattr(decode, "_reader_decodes_known_colours", lambda *a: False)
    elif what == "encode":
        monkeypatch.setattr(colour, "reader_encode_plan", lambda *a: None)
    else:
        monkeypatch.setattr(decode, "_matches_opencv", lambda *a: "frame 3 does not correspond")
    assert decode.decide(hd) is None
    lines = [l for l in capsys.readouterr().out.splitlines() if l]
    assert len(lines) == 1 and expected in lines[0] and lines[0].startswith("Warning:")


def test_the_older_vsync_spelling_is_tried_when_fps_mode_is_refused(tmp_path, monkeypatch):
    hd = make(tmp_path, "hd.mp4")
    tried = []

    def fake(matrix, rng, sync):
        tried.append(sync)
        return sync[0] == "-vsync"

    monkeypatch.setattr(decode, "_reader_decodes_known_colours", fake)
    assert decode.decide(hd)[2] == ("-vsync", "passthrough")
    assert [t[0] for t in tried] == ["-fps_mode", "-vsync"]


def test_a_second_reader_of_a_job_converts_as_the_first_does(tmp_path):
    hd = make(tmp_path, "hd.mp4")
    first = decode.FfmpegFrameReader(hd, "bt2020", "pc")
    second = decode.open_capture(hd, like=first)
    assert isinstance(second, decode.FfmpegFrameReader) and (second.matrix, second.range) == ("bt2020", "pc")
    plain = cv2.VideoCapture(hd)
    third = decode.open_capture(hd, like=plain)
    assert type(third) is cv2.VideoCapture  # a job that reads with OpenCV stays with OpenCV
    for reader in (first, second, plain, third):
        reader.release()


def test_the_decision_is_announced_once_when_the_reader_starts(tmp_path, capsys):
    hd = make(tmp_path, "hd.mp4")
    reader = decode.open_capture(hd)
    assert capsys.readouterr().out == ""  # asking is silent; starting says what it does
    reader.read()
    out = capsys.readouterr().out
    assert out.count("Decoding with ffmpeg") == 1
    reader.release()
    second = decode.open_capture(hd)
    second.read()
    second.release()
    assert "Decoding with ffmpeg" not in capsys.readouterr().out


# --- The encode agrees with the decode ------------------------------------------------------


def job_for(path, reader):
    return SimpleNamespace(video_path=path, reader=reader)


def test_the_encode_uses_the_matrix_the_reader_decoded_with(tmp_path):
    hd = make(tmp_path, "hd.mp4")  # untagged HD
    reader = decode.FfmpegFrameReader(hd, "bt709", "tv")
    plan = colour.plan_output_colour(job_for(hd, reader))
    assert plan.active and not plan.tagged
    assert plan.options == ("-vf", "scale=out_color_matrix=bt709:out_range=tv")  # no tags: it had none
    assert plan.second == ("-vf", "scale=out_color_matrix=bt709:out_range=tv")
    cmd = ["ffmpeg", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "o.mp4"]
    assert "scale=out_color_matrix=bt709:out_range=tv" in plan.command(cmd)
    reader.release()


def test_an_sd_source_read_as_bt601_is_encoded_with_ffmpegs_own_default(tmp_path):
    sd = make(tmp_path, "sd.mp4", "720x480")
    reader = decode.FfmpegFrameReader(sd, "bt601", "tv")
    plan = colour.plan_output_colour(job_for(sd, reader))
    assert not plan.active  # upstream's command, exactly
    reader.release()


def test_a_tagged_source_keeps_its_tags_and_follows_the_reader(tmp_path):
    path = make(tmp_path, "t.mp4", "1280x720", "yuv420p", "-colorspace", "bt709", "-color_primaries",
                "bt709", "-color_trc", "bt709")
    reader = decode.FfmpegFrameReader(path, "bt709", "tv")
    plan = colour.plan_output_colour(job_for(path, reader))
    assert plan.active and plan.tagged
    assert "-colorspace" in plan.options and "bt709" in plan.options
    assert "scale=out_color_matrix=bt709" in " ".join(plan.options)
    reader.release()
    # a reader that (as a stand-in for a BT.601-only decoder) decoded BT.709 as BT.601 gets
    # the inverse of that, as colour.py always did
    reader = decode.FfmpegFrameReader(path, "bt601", "tv")
    plan = colour.plan_output_colour(job_for(path, reader))
    assert "scale=out_color_matrix" not in " ".join(plan.options)  # ffmpeg's default is BT.601
    reader.release()


def test_a_plain_capture_is_planned_as_before(tmp_path):
    path = make(tmp_path, "t.mp4", "1280x720", "yuv420p", "-colorspace", "bt709")
    capture = cv2.VideoCapture(path)
    plan = colour.plan_output_colour(job_for(path, capture))
    capture.release()
    assert plan.active and plan.tagged  # the measured OpenCV conversion (709 here) and the tags
    untagged = make(tmp_path, "u.mp4")
    capture = cv2.VideoCapture(untagged)
    assert not colour.plan_output_colour(job_for(untagged, capture)).active  # untouched, as today
    capture.release()


def test_two_plans_for_one_source_do_not_share_state(tmp_path):
    hd = make(tmp_path, "hd.mp4")
    reader = decode.FfmpegFrameReader(hd, "bt709", "tv")
    one = colour.plan_output_colour(job_for(hd, reader))
    one.options = ()
    two = colour.plan_output_colour(job_for(hd, reader))
    assert two.options  # the cached plan was not the object that was changed
    reader.release()


# --- The pipeline, end to end -----------------------------------------------------------------

rr = load_script("colour_roundtrip")


@pytest.mark.parametrize("name", ["hd_untagged"])
def test_untagged_hd_comes_out_right_with_the_reader_and_clipped_without(name, tmp_path, monkeypatch):
    spec = {c.name: c for c in rr.CLASSES}[name]
    monkeypatch.setenv("VIDEOWIPE_DECODE", "opencv")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    before = rr.class_row(spec, 10, str(SAMPLES / "english1.mp4"), tmp_path / "a")
    monkeypatch.setenv("VIDEOWIPE_DECODE", "auto")
    decode._decide.cache_clear()
    after = rr.class_row(spec, 10, str(SAMPLES / "english1.mp4"), tmp_path / "b")
    assert before["delta_e_max"] > 10  # clipped saturated colours (docs/decode.md)
    assert after["delta_e_max"] < 3.0 and after["delta_e_mean"] < 1.6
    assert any("Decoding with ffmpeg" in line for line in after["messages"])
    assert any("Encoding with the matrix the frames were decoded with" in line
               for line in after["messages"])
    assert after["output_tags"]["space"] in ("-", "unknown", "unspecified", None)  # untagged stays untagged
