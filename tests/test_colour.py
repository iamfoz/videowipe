"""The cleaned video carries the source's colour tags, with a matching conversion.

Real ffmpeg (skipped without it), the real ``STTNInpainter.inpaint`` with a stub
model, and ``scripts/colour_roundtrip.py`` for the measurement: sources of known
colour, tagged in each way, go through the pipeline and the output is compared
with the source after both are decoded as a correct player would.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from videowipe import colour, fidelity

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "colour_roundtrip.py"
spec = importlib.util.spec_from_file_location("colour_roundtrip", SCRIPT)
rr = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rr  # dataclasses look their module up here
spec.loader.exec_module(rr)

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="requires ffmpeg and ffprobe",
)
CLASSES = {c.name: c for c in rr.CLASSES}

# Tolerances for a source that is handled correctly, from the measurements in
# docs/colour.md: the 8-bit RGB step between decode and encode, 4:2:0 chroma and
# x264 at crf 18 leave a mean colour difference (CIE76) of about 1 and a worst
# patch of about 2 on flat colours; an uncorrected matrix error is 6 to 25.
MEAN_DELTA_E = 1.6
WORST_DELTA_E = 3.0
SHIFTED_DELTA_E = 3.0  # "off" on a source whose matrix is not what players assume


@pytest.fixture(autouse=True)
def default_mode(monkeypatch):
    monkeypatch.delenv("VIDEOWIPE_COLOUR", raising=False)
    monkeypatch.setenv("VIDEOWIPE_OUTPUT_TRACKS", "off")
    monkeypatch.delenv("VIDEOWIPE_SOURCE_TIMING", raising=False)
    # These tests are about colour.py's own path, which measures what OpenCV does;
    # tests/test_decode.py covers the ffmpeg reader and how colour.py follows it.
    monkeypatch.setenv("VIDEOWIPE_DECODE", "opencv")
    colour.decoder_conversion.cache_clear()


# --- Maths ----------------------------------------------------------------------------------


def test_known_conversions():
    # Pure red, 8-bit: BT.709 limited and BT.601 limited YCbCr (standard values).
    assert np.allclose(colour.rgb_to_yuv([255, 0, 0], "bt709", "tv"), [62.7, 102.4, 240], atol=0.2)
    assert np.allclose(colour.rgb_to_yuv([255, 0, 0], "bt601", "tv"), [81.5, 90.2, 240], atol=0.2)
    assert np.allclose(colour.rgb_to_yuv([255, 255, 255], "bt2020", "pc"), [255, 128, 128])
    assert np.allclose(colour.rgb_to_yuv([0, 0, 0], "bt709", "tv"), [16, 128, 128])


@pytest.mark.parametrize("matrix", sorted(colour.COEFFICIENTS))
@pytest.mark.parametrize("rng", ["tv", "pc"])
def test_the_two_directions_are_inverses(matrix, rng):
    rgb = np.array([[200, 60, 30], [10, 220, 90], [40, 50, 240], [128, 128, 128]], float)
    assert np.allclose(colour.yuv_to_rgb(colour.rgb_to_yuv(rgb, matrix, rng), matrix, rng), rgb, atol=1e-6)


def test_hypotheses_are_told_apart_by_the_probe_colours():
    """Each of the six (matrix, range) readings of the probe patches is its own
    unique best fit, and the next best is clearly worse."""
    patches = np.array(colour.PROBE_YUV)
    for truth in [(m, r) for m in colour.COEFFICIENTS for r in ("tv", "pc")]:
        measured = colour.yuv_to_rgb(patches, *truth)
        errors = {(m, r): float(np.abs(colour.yuv_to_rgb(patches, m, r) - measured).mean())
                  for m in colour.COEFFICIENTS for r in ("tv", "pc")}
        assert colour.best_hypothesis(errors) == truth
        assert sorted(errors.values())[1] > 1.5  # the codec adds up to 1 to the best fit


def test_best_hypothesis_refuses_a_poor_or_ambiguous_fit():
    assert colour.best_hypothesis({("bt601", "tv"): 0.4, ("bt709", "tv"): 2.0}) == ("bt601", "tv")
    assert colour.best_hypothesis({("bt601", "tv"): 3.0, ("bt709", "tv"): 6.0}) is None
    assert colour.best_hypothesis({("bt601", "tv"): 0.9, ("bt709", "tv"): 1.1}) is None


def test_probe_frame_layout():
    w, h = 256, 128
    frame = colour.probe_yuv_frame(w, h)
    assert len(frame) == w * h * 3 // 2
    y = np.frombuffer(frame, np.uint8)[:w * h].reshape(h, w)
    for (yy, _, _), (x0, y0, x1, y1) in zip(colour.PROBE_YUV, colour.probe_patch_rects(w, h)):
        assert (y[y0:y1, x0:x1] == yy).all()


# --- Reading the source ---------------------------------------------------------------------


def probe_json(**stream):
    stream = {"width": 1280, "height": 720, "pix_fmt": "yuv420p", **stream}
    return json.dumps({"streams": [stream]})


def test_parse_probe():
    full = colour.parse_probe(probe_json(color_range="tv", color_space="bt709", color_primaries="bt709",
                                         color_transfer="bt709"))
    assert (full.space, full.primaries, full.transfer, full.range) == ("bt709",) * 3 + ("tv",)
    assert full.tagged and not full.hdr
    bare = colour.parse_probe(probe_json(color_range="unknown", color_space="unknown"))
    assert not bare.tagged
    assert colour.parse_probe(probe_json(pix_fmt="yuvj420p")).range == "pc"
    assert colour.parse_probe(probe_json(color_transfer="smpte2084")).hdr
    assert colour.parse_probe("not json") is None
    assert colour.parse_probe(json.dumps({"streams": []})) is None


def test_mode_switch(monkeypatch):
    assert colour.colour_mode() == "auto"
    for value, expected in [("off", "off"), (" OFF ", "off"), ("auto", "auto"), ("on", "auto"),
                            ("", "auto")]:
        monkeypatch.setenv("VIDEOWIPE_COLOUR", value)
        assert colour.colour_mode() == expected


# --- The command ----------------------------------------------------------------------------


def command(extra=()):
    return ["ffmpeg", "-y", "-f", "rawvideo", "-s", "1280x720", "-pix_fmt", "bgr24", "-i", "-",
            "-i", "in.mp4", "-map", "0:v", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-movflags", "+faststart", "out.mp4", *extra]


OPTIONS = ("-vf", "scale=out_color_matrix=bt709:out_range=tv", "-colorspace", "bt709", "-color_range", "tv")


def test_options_go_before_the_first_output_file():
    plan = colour.OutputColour(OPTIONS, (), "x")
    cmd = plan.rewrite(command())
    at = cmd.index("-movflags")
    assert cmd[at - len(OPTIONS):at] == list(OPTIONS)
    assert cmd[:at - len(OPTIONS)] + cmd[at:] == command()


def test_a_second_output_gets_its_own_options_before_its_file():
    second = ["-map", "0:v", "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-color_range", "tv", "out.mkv"]
    plan = colour.OutputColour(OPTIONS, ("-vf", "scale=out_color_matrix=bt709:out_range=tv",
                                         "-colorspace", "bt709"), "x")
    cmd = plan.rewrite(command(second))
    tail = cmd[cmd.index("out.mp4") + 1:]
    assert tail == second[:-1] + ["-vf", "scale=out_color_matrix=bt709:out_range=tv",
                                  "-colorspace", "bt709", "out.mkv"]


@pytest.mark.parametrize("change", [
    lambda c: [x if x != "libx264" else "libx265" for x in c],
    lambda c: [x if x != "+faststart" else "+frag_keyframe" for x in c],
    lambda c: [x for x in c if x != "-movflags" and x != "+faststart"],
    lambda c: c[:c.index("-c:v")] + ["-vf", "format=yuv420p"] + c[c.index("-c:v"):],
    lambda c: c[:c.index("-c:v")] + ["-color_range", "pc"] + c[c.index("-c:v"):],
])
def test_an_unknown_command_is_left_alone_with_a_warning(change, capsys):
    plan = colour.OutputColour(OPTIONS, (), "x")
    cmd = change(command())
    assert plan.command(cmd) == cmd
    assert "could not keep the source's colour tags" in capsys.readouterr().out
    assert not plan.active


def test_an_inactive_plan_returns_the_command_itself(capsys):
    cmd = command()
    assert colour.OutputColour().command(cmd) is cmd
    assert capsys.readouterr().out == ""


# --- The decision ----------------------------------------------------------------------------


@pytest.fixture
def job(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"x")
    return SimpleNamespace(video_path=str(source))


def stub(monkeypatch, source, decoded=("bt709", "tv"), encode_ok=True):
    monkeypatch.setattr(colour, "probe_source", lambda path: source)
    monkeypatch.setattr(colour, "decoder_conversion", lambda *args: decoded)
    monkeypatch.setattr(colour, "check_encode", lambda *args: encode_ok)


HD709 = colour.SourceColour(1280, 720, "yuv420p", "bt709", "bt709", "bt709", "tv")


def test_switch_off_and_missing_files_change_nothing(job, monkeypatch, capsys):
    stub(monkeypatch, HD709)
    monkeypatch.setenv("VIDEOWIPE_COLOUR", "off")
    assert not colour.plan_output_colour(job).active
    monkeypatch.delenv("VIDEOWIPE_COLOUR")
    assert not colour.plan_output_colour(SimpleNamespace(video_path=None)).active
    assert not colour.plan_output_colour(SimpleNamespace(video_path="/no/such/file.mp4")).active
    assert capsys.readouterr().out == ""


def test_untagged_and_hdr_sources_are_left_alone(job, monkeypatch, capsys):
    stub(monkeypatch, colour.SourceColour(1280, 720, "yuv420p"))
    assert not colour.plan_output_colour(job).active
    stub(monkeypatch, colour.SourceColour(1280, 720, "yuv420p10le", "bt2020nc", "bt2020", "smpte2084", "tv"))
    assert not colour.plan_output_colour(job).active
    assert capsys.readouterr().out == ""


def test_a_tagged_source_gets_a_matching_conversion_and_its_tags(job, monkeypatch, capsys):
    stub(monkeypatch, HD709, ("bt709", "tv"))
    plan = colour.plan_output_colour(job)
    assert plan.options == ("-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setparams=colorspace=bt709:"
                            "color_primaries=bt709:color_trc=bt709:range=tv", "-colorspace", "bt709",
                            "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv")
    # The further (limited-range) output: converted once with the same matrix.
    assert plan.second == ("-vf", "scale=out_color_matrix=bt709:out_range=tv,setparams=colorspace=bt709:"
                           "color_primaries=bt709:color_trc=bt709:range=tv", "-colorspace", "bt709",
                           "-color_primaries", "bt709", "-color_trc", "bt709")
    assert capsys.readouterr().out == ""


def test_a_full_range_source_is_converted_to_full_range(job, monkeypatch):
    stub(monkeypatch, colour.SourceColour(1280, 720, "yuvj420p", "bt709", None, None, "pc"), ("bt709", "pc"))
    plan = colour.plan_output_colour(job)
    assert plan.options == ("-vf", "scale=out_color_matrix=bt709:out_range=pc,format=yuv420p,"
                            "setparams=colorspace=bt709:range=pc", "-colorspace", "bt709", "-color_range", "pc")


def test_where_the_decoder_already_matches_ffmpegs_default_only_tags_are_added(job, monkeypatch):
    """A decoder that always reads BT.601 limited gives BGR that the default encode
    turns back into the source's own YCbCr: only the tags are missing."""
    stub(monkeypatch, HD709, ("bt601", "tv"))
    plan = colour.plan_output_colour(job)
    assert plan.options == ("-vf", "format=yuv420p,setparams=colorspace=bt709:color_primaries=bt709:"
                            "color_trc=bt709:range=tv",
                            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                            "-color_range", "tv")
    assert plan.second == ()  # BGR is not the picture's true colours then


@pytest.mark.parametrize("what, expected", [
    ("matrix", "colour matrix (smpte240m)"),
    ("decoder", "could not tell how this OpenCV"),
    ("probe", "could not read the source's colour tags"),
    ("encode", "did not apply the source's colour tags"),
    ("range", "colour range differently"),
])
def test_every_failure_falls_back_with_one_line(what, expected, job, monkeypatch, capsys):
    source = colour.SourceColour(1280, 720, "yuv420p", "smpte240m", None, None, "tv") \
        if what == "matrix" else HD709
    stub(monkeypatch, None if what == "probe" else source,
         decoded=None if what == "decoder" else ("bt709", "pc" if what == "range" else "tv"),
         encode_ok=what != "encode")
    plan = colour.plan_output_colour(job)
    assert not plan.active
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("Warning:") and expected in lines[0]


# --- Real ffmpeg: the round trip -----------------------------------------------------------------

FRAMES = 6


def cleaned(name, tmp_path, monkeypatch, mode=None, frames=FRAMES):
    """``(metrics, output stream, output path, pipeline log)`` of a source class
    through the real pipeline."""
    if mode:
        monkeypatch.setenv("VIDEOWIPE_COLOUR", mode)
    cls = CLASSES[name]
    ref = rr.reference_frames(*cls.size, frames, None)
    source = tmp_path / f"{name}.mp4"
    rr.make_source(source, cls, ref)
    out_dir = tmp_path / f"out_{name}_{mode}"
    out_dir.mkdir()
    output = rr.run_pipeline(source, out_dir)
    log = list(rr.PIPELINE_LOG)
    src_stream, out_stream = rr.probe_stream(source), rr.probe_stream(output)
    metrics = rr.compare(rr.decode_rgb(source, src_stream), rr.decode_rgb(output, out_stream), cls.size)
    return metrics, out_stream, Path(output), log


def decoder_reads(name):
    cls = CLASSES[name]
    tags = cls.tags or {}
    return colour.decoder_conversion(*cls.size, tags.get("colorspace"), tags.get("color_primaries"),
                                     tags.get("color_trc"), tags.get("color_range"))


def reads_correctly(name):
    """Whether this OpenCV decodes the class with the matrix and range it was made with."""
    cls = CLASSES[name]
    return decoder_reads(name) == (cls.matrix, cls.range)


TAGGED = ["hd709_tv", "sd601_625_tv", "hd709_pc", "sd601_pc", "hd601_tv", "sd709_tv", "hd2020_mtx",
          "hd709_422", "hd709_444"]


@needs_ffmpeg
@pytest.mark.parametrize("name", TAGGED)
def test_tagged_sources_round_trip_within_tolerance(name, tmp_path, monkeypatch):
    metrics, stream, _, log = cleaned(name, tmp_path, monkeypatch, "auto")
    if reads_correctly(name):
        assert metrics["delta_e_mean"] <= MEAN_DELTA_E, metrics
        assert metrics["delta_e_max"] <= WORST_DELTA_E, metrics
    else:
        # The decoder converts with another matrix or range than the source's own
        # (the reference Mac's OpenCV reads BT.709 as BT.601), which clips saturated
        # colours before any of this code runs. That cannot be undone on the way out:
        # the output must only be no worse than upstream's.
        upstream = cleaned(name, tmp_path, monkeypatch, "off")[0]
        assert metrics["delta_e_mean"] <= upstream["delta_e_mean"] + 0.1, (metrics, upstream)
    tags = CLASSES[name].tags
    assert stream.get("color_space") == tags["colorspace"]
    assert stream.get("color_primaries") == tags.get("color_primaries")
    assert stream.get("color_transfer") == tags.get("color_trc")
    assert stream.get("color_range") == tags["color_range"]
    assert any(line.startswith("Keeping the source's colour tags") for line in log)


@needs_ffmpeg
def test_the_acceptance_case_shifts_without_the_switch(tmp_path, monkeypatch):
    """Mutation check: the HD BT.709 source the test above passes is wrong with the
    switch off, on an OpenCV that reads the source's tags (so the BGR frames are
    true colours that upstream's BT.601 encode then mislabels)."""
    if decoder_reads("hd709_tv") != ("bt709", "tv"):
        pytest.skip("this OpenCV does not read the BT.709 tag, so upstream's encode is consistent")
    metrics, stream, _, log = cleaned("hd709_tv", tmp_path, monkeypatch, "off")
    assert metrics["delta_e_mean"] > SHIFTED_DELTA_E
    assert stream.get("color_space") is None
    assert not any("colour tags" in line for line in log)


@needs_ffmpeg
def test_bt2020_with_primaries_is_handled_or_left_alone(tmp_path, monkeypatch):
    """OpenCV 4.14 (ffmpeg 8) converts BT.2020 primaries as well as the matrix, which
    is none of the three conversions this module can invert: the job runs as before
    with one warning. A decoder that only applies the matrix is handled."""
    if decoder_reads("hd2020_tv") is None:
        metrics, stream, output, log = cleaned("hd2020_tv", tmp_path, monkeypatch, "auto")
        assert stream.get("color_space") is None
        assert [line for line in log if line.startswith("Warning")] == [
            "Warning: could not tell how this OpenCV converts the source's colours; the output "
            "carries no colour tags."]
        off = cleaned("hd2020_tv", tmp_path, monkeypatch, "off")
        assert packet_hash(output) == packet_hash(off[2])
    else:
        metrics, stream, _, _ = cleaned("hd2020_tv", tmp_path, monkeypatch, "auto")
        if reads_correctly("hd2020_tv"):
            assert metrics["delta_e_mean"] <= MEAN_DELTA_E
        else:  # read with another matrix (see the round-trip test): no worse than upstream
            upstream = cleaned("hd2020_tv", tmp_path, monkeypatch, "off")[0]
            assert metrics["delta_e_mean"] <= upstream["delta_e_mean"] + 0.1, (metrics, upstream)
        assert stream.get("color_space") == "bt2020nc"


def video_hash(path):
    """Hash of the decoded frames of the first video stream."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "md5", "-"],
                         capture_output=True, text=True, check=True).stdout
    return out.strip()


def packet_hash(path):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-c", "copy",
                          "-bsf:v", "h264_metadata=delete_filler=0", "-f", "md5", "-"],
                         capture_output=True, text=True, check=True).stdout
    return out.strip()


@needs_ffmpeg
def test_an_untagged_source_is_exactly_what_it_was(tmp_path, monkeypatch):
    on = cleaned("hd_untagged", tmp_path, monkeypatch, "auto")
    off = cleaned("hd_untagged", tmp_path, monkeypatch, "off")
    assert not any("colour" in line for line in on[3] + off[3])
    assert on[1].get("color_space") is None
    assert packet_hash(on[2]) == packet_hash(off[2])


@needs_ffmpeg
def test_where_only_tags_were_missing_the_pixels_are_unchanged(tmp_path, monkeypatch):
    """BT.601 SD: the default conversion is already the right one, so only the
    tags differ from today's output; the decoded frames are identical."""
    if decoder_reads("sd601_625_tv") != ("bt601", "tv"):
        pytest.skip("this OpenCV does not decode BT.601 limited range for this source")
    on = cleaned("sd601_625_tv", tmp_path, monkeypatch, "auto")
    off = cleaned("sd601_625_tv", tmp_path, monkeypatch, "off")
    assert on[1].get("color_space") == "bt470bg" and off[1].get("color_space") is None
    assert video_hash(on[2]) == video_hash(off[2])


@needs_ffmpeg
def test_a_decoder_that_ignores_the_matrix_tag_gets_tags_and_todays_pixels(tmp_path, monkeypatch):
    """An OpenCV that decodes a BT.709-tagged source with BT.601 (as measured on the
    reference Mac): the encode must stay BT.601, so the pixels are today's exactly
    and only the tags are added. Labelling BGR frames before converting them would
    convert with BT.709 instead."""
    monkeypatch.setattr(colour, "decoder_conversion", lambda *args: ("bt601", "tv"))
    on = cleaned("hd709_tv", tmp_path, monkeypatch, "auto")
    off = cleaned("hd709_tv", tmp_path, monkeypatch, "off")
    assert any(line.startswith("Keeping the source's colour tags") for line in on[3])
    assert (on[1].get("color_space"), on[1].get("color_primaries"), on[1].get("color_transfer")) == \
        ("bt709", "bt709", "bt709")
    assert video_hash(on[2]) == video_hash(off[2])


# Upstream's command (commit 20872d4), embedded: "off" must reproduce it exactly.
def upstream_command(video, output, w, h, fps):
    return ["ffmpeg", "-y", "-loglevel", "error", "-nostats",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
            "-r", str(fps), "-i", "-", "-i", video, "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-movflags", "+faststart", output]


@needs_ffmpeg
@pytest.mark.parametrize("name, mode", [("hd709_tv", "off"), ("hd_untagged", "auto"), ("hd_untagged", "off")])
def test_the_command_is_upstreams_when_nothing_applies(name, mode, tmp_path, monkeypatch):
    seen = []
    real = subprocess.Popen

    def popen(cmd, *args, **kwargs):
        if "rawvideo" in cmd:
            seen.append(list(cmd))
        return real(cmd, *args, **kwargs)

    cls = CLASSES[name]
    source = tmp_path / "source.mp4"
    rr.make_source(source, cls, rr.reference_frames(*cls.size, 4, None))
    monkeypatch.setenv("VIDEOWIPE_COLOUR", mode)
    monkeypatch.setattr(subprocess, "Popen", popen)
    output = rr.run_pipeline(source, tmp_path)
    info = rr.probe_stream(source)
    fps = 25.0
    assert seen == [upstream_command(str(source), output, int(info["width"]), int(info["height"]), fps)]


@needs_ffmpeg
def test_a_source_smaller_than_the_probe_patches_is_still_probed():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "mean of empty slice"
        small = colour.decoder_conversion(160, 100, "bt709", "bt709", "bt709", "tv")
    assert small is not None
    assert small == colour.decoder_conversion(1280, 720, "bt709", "bt709", "bt709", "tv")


@needs_ffmpeg
def test_the_encode_check_allows_a_few_code_values_and_not_a_wrong_matrix(monkeypatch):
    tags = {"-colorspace": "bt709", "-color_primaries": "bt709", "-color_trc": "bt709", "-color_range": "tv"}
    options = colour._labelled(colour._scale_filters("bt709", "tv"), "bt709", "bt709", "bt709", "tv")
    assert colour.check_encode(options, "bt709", "tv", tags)
    assert not colour.check_encode(options, "bt601", "tv", tags)  # encoded with 709, expected 601
    real = colour.rgb_to_yuv
    for shift, expected in ((3.0, True), (colour.ENCODE_TOLERANCE + 3, False)):
        # an ffmpeg whose conversion is that far out in both chroma planes
        monkeypatch.setattr(colour, "rgb_to_yuv", lambda *a, shift=shift: real(*a) + np.array([0, shift, shift]))
        assert colour.check_encode(options, "bt709", "tv", tags) is expected


@needs_ffmpeg
def test_it_composes_with_the_source_timestamps_of_a_variable_rate_source(tmp_path, monkeypatch):
    """The timestamp rewrite (Matroska frame pipe) and this one touch the same command."""
    source = tmp_path / "jitter.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=25",
         "-vf", "settb=1/1000000,setpts='(N/25+0.013*sin(N*1.7)+0.3)/TB',"
                "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
                "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv",
         "-frames:v", "20", "-fps_mode", "passthrough", "-enc_time_base", "1:1000000",
         "-video_track_timescale", "90000", "-colorspace", "bt709", "-color_primaries", "bt709",
         "-color_trc", "bt709", "-color_range", "tv", "-c:v", "libx264", str(source)],
        check=True, capture_output=True)
    monkeypatch.setenv("VIDEOWIPE_COLOUR", "auto")
    output = rr.run_pipeline(source, tmp_path)
    log = rr.PIPELINE_LOG
    assert any(line.startswith("Keeping the source's frame timestamps") for line in log)
    assert any(line.startswith("Keeping the source's colour tags") for line in log)
    stream = rr.probe_stream(output)
    assert (stream.get("color_space"), stream.get("color_range")) == ("bt709", "tv")

    def times(path):
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "packet=pts_time", "-of", "csv=p=0", str(path)],
                             check=True, capture_output=True, text=True).stdout.split()
        return sorted(float(t) for t in out)

    assert times(output) == pytest.approx([t - times(source)[0] for t in times(source)], abs=1e-4)


@needs_ffmpeg
def test_the_second_output_follows_the_first(tmp_path, monkeypatch):
    """The optional HEVC copy is appended to the command as another output; its
    colours match the main output's, and it is tagged when the plan could tell
    that BGR is the picture's true colours."""
    monkeypatch.setenv("VIDEOWIPE_COLOUR", "auto")
    cls = CLASSES["hd709_tv"]
    source = tmp_path / "source.mp4"
    rr.make_source(source, cls, rr.reference_frames(*cls.size, FRAMES, None))
    second = tmp_path / "second.mkv"
    original = fidelity.EncodePlan.popen
    monkeypatch.setattr(fidelity.EncodePlan, "popen", lambda self, cmd, stderr: original(
        self, cmd + ["-map", "0:v", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-color_range", "tv",
                     str(second)], stderr))
    output = rr.run_pipeline(source, tmp_path)
    first_stream, second_stream = rr.probe_stream(output), rr.probe_stream(second)
    if decoder_reads("hd709_tv") == ("bt709", "tv"):
        assert second_stream.get("color_space") == "bt709"
        a = rr.decode_rgb(output, first_stream)
        b = rr.decode_rgb(second, second_stream)
        assert rr.compare(a, b, cls.size)["delta_e_mean"] <= MEAN_DELTA_E
    else:
        assert second_stream.get("color_space") is None


# --- The script's own maths --------------------------------------------------------------------


def test_script_interpretation_follows_tags_then_conventions():
    assert rr.interpret({"width": 1280, "height": 720, "pix_fmt": "yuv420p"}) == ("bt709", "tv")
    assert rr.interpret({"width": 720, "height": 576, "pix_fmt": "yuv420p"}) == ("bt601", "tv")
    assert rr.interpret({"width": 1280, "height": 720, "pix_fmt": "yuvj420p"}) == ("bt709", "pc")
    assert rr.interpret({"width": 1280, "height": 720, "color_space": "bt470bg",
                         "color_range": "pc", "pix_fmt": "yuv420p"}) == ("bt601", "pc")
    assert rr.interpret({"width": 720, "height": 480, "color_space": "bt2020nc",
                         "pix_fmt": "yuv420p"}) == ("bt2020", "tv")


def test_script_metrics():
    rgb = np.array([[255, 255, 255], [0, 0, 0], [128, 128, 128]])
    assert rr.delta_e76(rgb, rgb).max() == 0
    # Black to white is 100 lightness units.
    assert rr.delta_e76(rgb[0], rgb[1]) == pytest.approx(100, abs=0.5)
    # A grey of equal chroma has no chroma shift; a tint does.
    assert rr.chroma_shift([128, 128, 128], [140, 140, 140]) == pytest.approx(0, abs=0.01)
    assert rr.chroma_shift([128, 128, 128], [150, 128, 128]) > 5
    a = np.zeros((2, 4, 4, 3), np.uint8)
    assert rr.psnr(a, a) == float("inf")
    b = a.copy()
    b[..., 0] += 1
    assert rr.psnr(a, b) == pytest.approx(10 * np.log10(255**2 / (1 / 3)), abs=0.01)
    keep = np.zeros((4, 4), bool)
    keep[0, 0] = True
    assert rr.psnr(a, b, keep) == pytest.approx(rr.psnr(a, b), abs=0.01)


def test_script_layout_keeps_patches_clear_of_the_mask():
    for w, h in (rr.HD, rr.SD625, rr.SD525):
        patches, top, (_, my0, mx1, my1) = rr.layout(w, h)
        assert len(patches) == len(rr.BARS) + len(rr.SWATCHES)
        for x0, y0, x1, y1 in patches.values():
            assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= top
        assert my0 > top and mx1 < w and my1 < h
