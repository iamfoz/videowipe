"""10-bit and HDR sources keep their bit depth, tags and HDR metadata in an HEVC copy.

Real ffmpeg with libx265 (skipped without it) and the real ``STTNInpainter.inpaint``
with a stub model; ``scripts/high_bit_depth_roundtrip.py`` builds the sources (a smooth
10-bit picture, lossless HEVC, tagged SDR, HDR10 and HLG) and measures the result.
"""
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from videowipe import colour, decode, export, fidelity, highbit, source_checks

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "input" / "detext_examples"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hb = load_script("high_bit_depth_roundtrip")
rr = hb.load_roundtrip()


def has_x265():
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return False
    return "libx265" in export._encoder_help("libx265")


needs_x265 = pytest.mark.skipif(not has_x265(), reason="requires ffmpeg with libx265")


@pytest.fixture(autouse=True)
def default_modes(monkeypatch):
    for name in ("VIDEOWIPE_HIGH_BIT_DEPTH", "VIDEOWIPE_HIGH_BIT_DEPTH_ENCODER", "VIDEOWIPE_DECODE",
                 "VIDEOWIPE_COLOUR", "VIDEOWIPE_OUTPUT_CODEC"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VIDEOWIPE_OUTPUT_TRACKS", "off")
    decode._decide.cache_clear()
    colour.decoder_conversion.cache_clear()
    colour._reader_plan.cache_clear()
    highbit._probe.cache_clear()
    highbit._checked.cache_clear()
    highbit._current.clear()
    decode._announced.clear()


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    folder = tmp_path_factory.mktemp("hbd-sources")
    if not has_x265():
        return {}
    out = {}
    for kind in hb.KINDS:
        out[kind] = folder / f"{kind}.mp4"
        hb.make_source(out[kind], kind, 8)
    return out


# --- Pure pieces ----------------------------------------------------------------------------


def test_mode_and_encoder_switches(monkeypatch):
    assert highbit.mode() == "auto" and highbit.encoder_override() == "auto"
    for value, expected in [("off", "off"), (" OFF ", "off"), ("auto", "auto"), ("on", "on"),
                            (" On ", "on"), ("yes", "auto"), ("", "auto")]:
        monkeypatch.setenv("VIDEOWIPE_HIGH_BIT_DEPTH", value)
        assert highbit.mode() == expected
    for value, expected in [("libx265", "libx265"), ("hevc_videotoolbox", "hevc_videotoolbox"),
                            ("x264", "auto")]:
        monkeypatch.setenv("VIDEOWIPE_HIGH_BIT_DEPTH_ENCODER", value)
        assert highbit.encoder_override() == expected


def test_merge_without_a_high_frame_returns_the_frame_itself():
    frame = np.zeros((4, 4, 3), np.uint8)
    assert highbit.merge(frame, np.zeros((4, 4, 3), np.uint8)) is frame
    assert highbit.merge(frame, frame.view(highbit.HighFrame)) is frame  # a slice or view has none


def test_merge_carries_only_the_changed_pixels_across():
    rng = np.random.default_rng(1)
    high = rng.integers(0, 65536, (12, 16, 3), dtype=np.uint16)
    original = highbit.make_high_frame(high.copy())
    assert original.dtype == np.uint8 and original.high is not None
    # the 8-bit frame is the 16-bit one rounded (swscale's scale: 8 bits is 16 bits / 256)
    assert np.abs(original.astype(int) - np.minimum(np.rint(high / 256.0), 255)).max() <= 1
    painted = np.array(original, copy=True)
    painted[3:6, 4:9] = (10, 200, 255)
    out = highbit.merge(painted, original)
    assert out.dtype == np.uint16 and out.shape == high.shape
    untouched = np.ones(high.shape[:2], bool)
    untouched[3:6, 4:9] = False
    assert np.array_equal(out[untouched], high[untouched])  # the source's, bit for bit
    # a changed pixel moves by its 8-bit change times 256
    expected = np.clip(high[3:6, 4:9].astype(int) + (np.array([10, 200, 255]) - original[3:6, 4:9].astype(int)) * 256,
                       0, 65535)
    assert np.array_equal(out[3:6, 4:9], expected)


def test_merge_keeps_white_white_and_survives_an_in_place_edit():
    high = np.full((6, 6, 3), 65283, np.uint16)  # white as swscale writes it
    original = highbit.make_high_frame(high.copy())
    assert (original == 255).all()
    edited = original.view(np.ndarray)
    edited[1:3, 1:3] = 0  # the compositor edited the very frame it was given
    out = highbit.merge(edited, original)
    assert (out[1:3, 1:3] == 3).all() and (out[3:] == 65283).all()  # black, to within the residue
    painted = np.array(original, copy=True)
    painted[0, 0] = 255  # painting white on white changes nothing
    assert highbit.merge(painted, original) is original.high
    # nothing changed: the 16-bit frame itself comes back
    again = highbit.make_high_frame(np.full((6, 6, 3), 1234, np.uint16))
    assert highbit.merge(np.array(again, copy=True), again) is again.high


def test_x265_mastering_string_matches_what_ffprobe_prints():
    from fractions import Fraction as F
    values = (F(34000, 50000), F(16000, 50000), F(13250, 50000), F(34500, 50000), F(7500, 50000),
              F(3000, 50000), F(15635, 50000), F(16450, 50000), F(10000000, 10000), F(1, 10000))
    assert highbit.x265_mastering(values) == \
        "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"


def test_parse_source_reads_depth_tags_and_hdr_side_data():
    text = json.dumps({"streams": [{"width": 640, "height": 360, "pix_fmt": "yuv420p10le",
                                    "color_space": "bt2020nc", "color_primaries": "bt2020",
                                    "color_transfer": "smpte2084", "color_range": "tv"}],
                       "frames": [{"side_data_list": [
                           {"side_data_type": "Mastering display metadata",
                            "red_x": "34000/50000", "red_y": "16000/50000", "green_x": "13250/50000",
                            "green_y": "34500/50000", "blue_x": "7500/50000", "blue_y": "3000/50000",
                            "white_point_x": "15635/50000", "white_point_y": "16450/50000",
                            "min_luminance": "1/10000", "max_luminance": "10000000/10000"},
                           {"side_data_type": "Content light level metadata",
                            "max_content": 1000, "max_average": 400}]}]})
    source = highbit.parse_source(text)
    assert source.depth == 10 and source.hdr and source.light == (1000, 400)
    assert highbit.x265_mastering(source.mastering).startswith("G(13250,34500)")
    params = highbit._x265_params(source)
    assert "hdr10=1" in params and "max-cll=1000,400" in params
    # HLG has no static metadata to write
    hlg = highbit.parse_source(json.dumps({"streams": [{"width": 640, "height": 360,
                                                         "pix_fmt": "yuv420p10le", "color_transfer": "arib-std-b67"}]}))
    assert hlg.hdr and highbit._x265_params(hlg) == ()
    assert highbit.parse_source("nope") is None


def test_the_command_is_rewritten_for_16_bit_input_or_left_alone(capsys):
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "640x360", "-pix_fmt", "bgr24",
           "-r", "25", "-i", "-", "-i", "in.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "o.mp4"]
    plan = highbit.HighBitPlan(active=True, description="x")
    out = plan.command(cmd)
    assert out[out.index("-pix_fmt") + 1] == "bgr48le" and plan.applied
    assert out.count("yuv420p") == 1 and "Keeping the source's bit depth (x)" in capsys.readouterr().out
    odd = highbit.HighBitPlan(active=True)
    assert odd.command(["ffmpeg", "-i", "-", "o.mp4"]) == ["ffmpeg", "-i", "-", "o.mp4"]
    assert not odd.active and not odd.applied
    assert "the cleaned video is 8-bit" in capsys.readouterr().out
    assert highbit.HighBitPlan().command(cmd) == cmd


def test_the_output_options_label_the_frames_and_pass_the_hdr_metadata():
    plan = highbit.HighBitPlan(active=True, matrix="bt2020", encoder="libx265",
                               tags={"-colorspace": "bt2020nc", "-color_primaries": "bt2020",
                                     "-color_trc": "smpte2084"},
                               x265_params=("hdr10=1", "max-cll=1000,400"))
    options = highbit.output_options(plan, export.choose_preset(360))
    vf = options[options.index("-vf") + 1]
    assert vf.startswith("scale=out_color_matrix=bt2020:out_range=tv,format=yuv420p10le,setparams=")
    assert "colorspace=bt2020nc" in vf and "color_trc=smpte2084" in vf and "range=tv" in vf
    assert options[options.index("-x265-params") + 1] == "log-level=error:hdr10=1:max-cll=1000,400"
    assert options[options.index("-colorspace") + 1] == "bt2020nc"
    assert options[options.index("-profile:v") + 1] == "main10" and "hvc1" in options
    assert options[options.index("-color_range") + 1] == "tv"
    assert options.index("-colorspace") < options.index("-tag:v")


def test_source_checks_say_where_the_hevc_copy_is_and_how_to_ask_for_it(monkeypatch):
    sdr, hdr = {"pix_fmt": "yuv420p10le"}, {"pix_fmt": "yuv420p10le", "color_transfer": "smpte2084"}
    # auto: HDR gets the copy, SDR says how to opt in
    assert "HEVC copy keeps the source's bit depth" in source_checks.describe(hdr)[0]
    line = source_checks.describe(sdr)[0]
    assert "10-bit" in line and "VIDEOWIPE_HIGH_BIT_DEPTH=on" in line and "keeps the source's" not in line
    monkeypatch.setenv("VIDEOWIPE_HIGH_BIT_DEPTH", "on")
    assert "HEVC copy keeps the source's bit depth" in source_checks.describe(sdr)[0]
    assert "VIDEOWIPE_HIGH_BIT_DEPTH=on" not in source_checks.describe(sdr)[0]
    monkeypatch.setenv("VIDEOWIPE_HIGH_BIT_DEPTH", "off")
    for stream in (sdr, hdr):
        line = source_checks.describe(stream)[0]
        assert "HEVC copy" not in line and "VIDEOWIPE_HIGH_BIT_DEPTH" not in line
    assert source_checks.describe({"pix_fmt": "yuv420p"}) == []
    for chosen in ("auto", "on", "off"):
        monkeypatch.setenv("VIDEOWIPE_HIGH_BIT_DEPTH", chosen)
        assert source_checks.describe({"pix_fmt": "yuv420p"}) == []


def probe_text(transfer=None, mastering=False):
    stream = {"width": 640, "height": 360, "pix_fmt": "yuv420p10le"}
    if transfer:
        stream["color_transfer"] = transfer
    frames = []
    if mastering:
        frames = [{"side_data_list": [{"side_data_type": "Mastering display metadata",
                                        "red_x": "34000/50000", "red_y": "16000/50000",
                                        "green_x": "13250/50000", "green_y": "34500/50000",
                                        "blue_x": "7500/50000", "blue_y": "3000/50000",
                                        "white_point_x": "15635/50000", "white_point_y": "16450/50000",
                                        "min_luminance": "1/10000", "max_luminance": "10000000/10000"}]}]
    return json.dumps({"streams": [stream], "frames": frames})


@pytest.mark.parametrize("transfer, chosen, expected", [
    ("smpte2084", None, True), ("arib-std-b67", None, True), ("bt709", None, False), (None, None, False),
    ("smpte2084", "on", True), ("bt709", "on", True), (None, "on", True),
    ("smpte2084", "off", False), ("bt709", "off", False),
])
def test_which_sources_get_the_copy(monkeypatch, transfer, chosen, expected):
    """auto: PQ and HLG by their transfer tag; on: everything above 8 bits; off: nothing."""
    if chosen:
        monkeypatch.setenv("VIDEOWIPE_HIGH_BIT_DEPTH", chosen)
    assert highbit.wanted(highbit.parse_source(probe_text(transfer))) is expected


def test_mastering_metadata_without_a_transfer_tag_is_not_taken_for_pq():
    """HDR10 side data says what the mastering display was, not which transfer the picture
    uses (HLG and wide-gamut SDR carry it too), so it is reported, not acted on."""
    source = highbit.parse_source(probe_text(None, mastering=True))
    assert source.mastering is not None and source.untagged_hdr10
    assert not source.hdr and not highbit.wanted(source)
    assert highbit.wanted(highbit.parse_source(probe_text("smpte2084", mastering=True)))
    assert not highbit.parse_source(probe_text("smpte2084", mastering=True)).untagged_hdr10
    assert not highbit.parse_source(probe_text(None)).untagged_hdr10


# --- The 16-bit reader ----------------------------------------------------------------------


@needs_x265
def test_the_16_bit_reader_yields_the_same_frames_as_the_8_bit_one(sources):
    path = str(sources["sdr10"])
    low = decode.FfmpegFrameReader(path, "bt709", "tv")
    high = decode.FfmpegFrameReader(path, "bt709", "tv", bits=16)
    try:
        count = 0
        while True:
            ok_l, a = low.read()
            ok_h, b = high.read()
            assert ok_l == ok_h
            if not ok_l:
                break
            assert isinstance(b, highbit.HighFrame) and b.high.dtype == np.uint16
            assert b.shape == a.shape == b.high.shape
            # the 16-bit frame rounded to 8 bits, against swscale's own 8-bit conversion (which dithers)
            diff = np.abs(b.astype(int) - a.astype(int))
            assert diff.max() <= 4 and diff.mean() < 1.5  # another swscale dithers a little differently
            count += 1
        assert count == 8
    finally:
        low.release()
        high.release()


@needs_x265
def test_a_16_bit_frame_is_the_source_at_16_bits(sources):
    """Neutral grey ramp: 10-bit luma Y becomes grey (Y - 64) / 876 of 65535."""
    reader = decode.FfmpegFrameReader(str(sources["sdr10"]), "bt709", "tv", bits=16)
    ok, frame = reader.read()
    reader.release()
    y, _, _ = hb.picture()
    row = 90  # the ramp
    expected = (y[row].astype(np.float64) - 64) / 876 * 65535
    got = frame.high[row, :, 1].astype(np.float64)
    inside = (expected > 800) & (expected < 64700)
    # swscale's 16-bit RGB tops out at about 65283 (8 bits times 256): a gain 0.4 % under
    # the ideal that the encode undoes exactly (docs/high-bit-depth.md)
    assert np.abs(got - expected * (65283 / 65535))[inside].max() < 8


@needs_x265
def test_grab_and_retrieve_work_on_the_16_bit_reader(sources):
    reader = decode.FfmpegFrameReader(str(sources["pq"]), "bt2020", "tv", bits=16)
    assert reader.grab() and reader.grab()
    ok, frame = reader.retrieve()
    assert ok and frame.high.shape[:2] == (hb.HEIGHT, hb.WIDTH)
    reader.release()


# --- Deciding -------------------------------------------------------------------------------


def job_for(path, reader=None, **extra):
    return SimpleNamespace(video_path=str(path), reader=reader, dual=False, trial_range=None, **extra)


@needs_x265
def test_an_8_bit_source_never_takes_this_path(tmp_path, capsys):
    path = tmp_path / "eight.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25",
                    "-frames:v", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    plan = highbit.plan_high_bit_depth(job_for(path, decode.FfmpegFrameReader(str(path), "bt601", "tv")))
    assert not plan.active and capsys.readouterr().out == ""
    assert highbit.output_args(str(path), "o.mp4", 360) is None
    assert export.hevc_output_args(str(path), "o.mp4", 360) == []  # as ever with H.264 selected


@needs_x265
@pytest.mark.parametrize("kind, label, chosen", [("sdr10", "10-bit SDR", "on"), ("pq", "HDR10 (PQ)", "auto"),
                                                 ("hlg", "HLG", "auto"), ("pq", "HDR10 (PQ)", "on")])
def test_a_high_bit_depth_source_is_planned_with_the_ffmpeg_reader(sources, capsys, monkeypatch, kind, label,
                                                                   chosen):
    monkeypatch.setenv("VIDEOWIPE_HIGH_BIT_DEPTH", chosen)
    path = str(sources[kind])
    reader = decode.open_capture(path)
    assert isinstance(reader, decode.FfmpegFrameReader)  # even HDR (with VIDEOWIPE_HIGH_BIT_DEPTH=auto)
    plan = highbit.plan_high_bit_depth(job_for(path, reader))
    assert plan.active and not plan.applied and label in plan.description
    assert plan.encoder in ("libx265", "hevc_videotoolbox")
    assert plan.matrix == ("bt709" if kind == "sdr10" else "bt2020")
    assert highbit.current(path) is plan
    reader.release()
    assert capsys.readouterr().out == ""  # said when the command is rewritten, not when planned


@needs_x265
def test_auto_leaves_an_sdr_source_alone_and_on_asks_for_the_copy(sources, capsys, monkeypatch):
    path = str(sources["sdr10"])
    reader = decode.open_capture(path)
    assert isinstance(reader, decode.FfmpegFrameReader)  # still read by ffmpeg (decode.md), 8-bit
    plan = highbit.plan_high_bit_depth(job_for(path, reader))
    assert not plan.active and capsys.readouterr().out == ""  # source_checks says how to opt in
    assert highbit.output_args(path, "o.mp4", 360) is None
    assert export.hevc_output_args(path, "o.mp4", 360) == []
    monkeypatch.setenv("VIDEOWIPE_HIGH_BIT_DEPTH", "on")
    assert highbit.plan_high_bit_depth(job_for(path, reader)).active
    reader.release()


@needs_x265
def test_untagged_hdr10_is_left_alone_with_one_line_saying_how_to_get_the_copy(tmp_path, capsys, monkeypatch):
    """An HDR10-looking source with no transfer tag: not PQ by the rule, so not copied in
    auto, and told how to ask."""
    source = tmp_path / "untagged.mkv"
    raw = bytes(640 * 360 * 3) * 4  # yuv420p10le: 640*360*1.5 samples of 2 bytes
    params = ("log-level=error:master-display=G(13250,34500)B(7500,3000)R(34000,16000)"
              "WP(15635,16450)L(10000000,1):max-cll=1000,400")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", "640x360",
                    "-r", "25", "-i", "-", "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", params,
                    str(source)], input=raw, check=True)
    parsed = highbit.probe_source(str(source))
    if parsed is None or parsed.mastering is None:
        pytest.skip("this ffmpeg does not carry the mastering metadata of the test clip")
    assert parsed.colour.transfer is None and parsed.untagged_hdr10
    plan = highbit.plan_high_bit_depth(job_for(source, decode.FfmpegFrameReader(str(source), "bt709", "tv")))
    lines = [l for l in capsys.readouterr().out.splitlines() if l]
    assert not plan.active and len(lines) == 1
    assert "no transfer tag" in lines[0] and "VIDEOWIPE_HIGH_BIT_DEPTH=on" in lines[0]
    monkeypatch.setenv("VIDEOWIPE_HIGH_BIT_DEPTH", "on")
    assert highbit.plan_high_bit_depth(job_for(source, decode.FfmpegFrameReader(str(source), "bt709", "tv"))).active


@needs_x265
def test_the_hevc_output_exists_whichever_codec_is_chosen(sources):
    path = str(sources["pq"])
    reader = decode.open_capture(path)
    highbit.plan_high_bit_depth(job_for(path, reader))
    args = export.hevc_output_args(path, "/out/clip.mp4", 360)
    assert args[:2] == ["-map", "0:v"] and args[-1] == "/out/clip.mkv"
    assert "-x265-params" in args and "master-display=" in args[args.index("-x265-params") + 1]
    reader.release()


@needs_x265
def test_what_it_cannot_do_it_says_once_and_leaves_the_job_8_bit(sources, capsys, monkeypatch):
    monkeypatch.setenv("VIDEOWIPE_HIGH_BIT_DEPTH", "on")
    path = str(sources["sdr10"])
    reader = decode.open_capture(path)
    cases = [
        (dict(dual=True), "dual (side by side) output is 8-bit only"),
        (dict(reader=cv2.VideoCapture(path)), "not read by the ffmpeg reader"),
    ]
    for extra, expected in cases:
        job = job_for(path, reader)
        for key, value in extra.items():
            setattr(job, key, value)
        plan = highbit.plan_high_bit_depth(job)
        lines = [l for l in capsys.readouterr().out.splitlines() if l]
        assert not plan.active and len(lines) == 1 and expected in lines[0]
        assert lines[0].startswith("Warning:") and lines[0].endswith("the cleaned video is 8-bit.")
    monkeypatch.setattr(highbit, "encoder_for", lambda source: None)
    plan = highbit.plan_high_bit_depth(job_for(path, reader))
    assert not plan.active and "no 10-bit HEVC encoder" in capsys.readouterr().out
    monkeypatch.undo()
    monkeypatch.setenv("VIDEOWIPE_HIGH_BIT_DEPTH", "on")
    highbit._checked.cache_clear()
    monkeypatch.setattr(highbit, "check_pipeline", lambda *a: False)
    plan = highbit.plan_high_bit_depth(job_for(path, reader))
    assert not plan.active and "did not encode known 16-bit colours" in capsys.readouterr().out
    reader.release()


@needs_x265
def test_off_and_trial_runs_stay_8_bit_without_a_line(sources, capsys, monkeypatch):
    monkeypatch.setenv("VIDEOWIPE_HIGH_BIT_DEPTH", "on")
    path = str(sources["sdr10"])
    reader = decode.open_capture(path)
    trial = job_for(path, reader)
    trial.trial_range = (0, 4)
    assert not highbit.plan_high_bit_depth(trial).active
    monkeypatch.setenv("VIDEOWIPE_HIGH_BIT_DEPTH", "off")
    assert not highbit.plan_high_bit_depth(job_for(path, reader)).active
    assert capsys.readouterr().out == ""
    reader.release()


@needs_x265
def test_hdr_is_left_to_opencv_when_the_switch_is_off(sources, monkeypatch, capsys):
    path = str(sources["pq"])
    monkeypatch.setenv("VIDEOWIPE_HIGH_BIT_DEPTH", "off")
    assert decode.decide(path) is None  # upstream's behaviour exactly
    assert capsys.readouterr().out == ""  # auto: source_checks already warns about HDR
    assert type(decode.open_capture(path)) is cv2.VideoCapture


def test_hdr_with_moving_content_passes_the_start_up_check(tmp_path):
    """OpenCV converts the primaries and the transfer of a PQ clip, so its pictures are
    nothing like the ffmpeg reader's in value; the check must still see the same frames."""
    path = tmp_path / "pq_testsrc.mp4"
    # tags on the frames (after the conversion) as well as on the output: ffmpeg 9 converts
    # to colour options on the output alone and drops the primaries and transfer
    labels = colour._labelled((), "bt2020nc", "bt2020", "smpte2084", None, pix_fmt="yuv420p10le")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25",
                    "-frames:v", "12", "-c:v", "libx264", "-pix_fmt", "yuv420p10le", *labels,
                    str(path)], check=True)
    plan = decode.decide(str(path))
    assert plan is not None and plan[:2] == ("bt2020", "tv")


@needs_x265
def test_hdr_8_bit_copy_is_converted_but_untagged(sources):
    path = str(sources["pq"])
    reader = decode.FfmpegFrameReader(path, "bt2020", "tv")
    plan = colour.plan_output_colour(job_for(path, reader))
    reader.release()
    assert plan.active and not plan.tagged
    assert plan.options == ("-vf", "scale=out_color_matrix=bt2020:out_range=tv")  # no PQ tags on 8 bits


@needs_x265
def test_the_encode_check_catches_a_wrong_matrix(sources):
    plan = highbit.HighBitPlan(active=True, matrix="bt2020", encoder="libx265",
                               tags={"-colorspace": "bt2020nc", "-color_primaries": "bt2020",
                                     "-color_trc": "smpte2084"})
    preset = export.choose_preset(360)
    assert highbit.check_pipeline(plan, preset)
    plan.tags = {"-colorspace": "bt709"}
    # the stream is tagged as asked, but the check compares values against plan.matrix
    plan.matrix = "bt709"
    assert highbit.check_pipeline(plan, preset)
    real = highbit.video_filter
    highbit.video_filter = lambda matrix, pix_fmt, tags: real("bt709", pix_fmt, tags)
    try:
        plan.matrix, plan.tags = "bt2020", {"-colorspace": "bt2020nc"}
        assert not highbit.check_pipeline(plan, preset)  # converted with 709 but 2020 expected
    finally:
        highbit.video_filter = real


@needs_x265
def test_the_encode_check_allows_a_few_code_values_of_rounding(sources):
    """Another swscale's chroma rounding may be a few 8-bit codes out (12 in 10-bit
    values); the nearest wrong matrix is more than 30 away."""
    plan = highbit.HighBitPlan(active=True, matrix="bt2020", encoder="libx265",
                               tags={"-colorspace": "bt2020nc", "-color_primaries": "bt2020",
                                     "-color_trc": "smpte2084"})
    preset = export.choose_preset(360)
    real = colour.rgb_to_yuv
    try:
        for shift, expected in ((2.0, True), (4.0, False)):  # 8-bit codes, in both chroma planes
            colour.rgb_to_yuv = lambda *a, shift=shift: real(*a) + np.array([0, shift, shift])
            assert highbit.check_pipeline(plan, preset) is expected
    finally:
        colour.rgb_to_yuv = real


# --- The pipeline, end to end -----------------------------------------------------------------


def run(kind, mode, tmp_path, monkeypatch, sources):
    monkeypatch.setenv("VIDEOWIPE_HIGH_BIT_DEPTH", mode)
    decode._decide.cache_clear()
    highbit._current.clear()
    out_dir = tmp_path / f"out_{kind}_{mode}"
    out_dir.mkdir()
    output = Path(rr.run_pipeline(sources[kind], out_dir, hb.MASK_BOX))
    return output, out_dir, list(rr.PIPELINE_LOG)


@needs_x265
@pytest.mark.parametrize("kind, chosen", [("sdr10", "on"), ("pq", "auto"), ("hlg", "auto"), ("pq", "on")])
def test_the_hevc_copy_keeps_the_bit_depth_the_tags_and_the_hdr_metadata(kind, chosen, tmp_path, monkeypatch,
                                                                         sources):
    output, out_dir, log = run(kind, chosen, tmp_path, monkeypatch, sources)
    assert any(line.startswith("Keeping the source's bit depth") for line in log)
    mkv = out_dir / (output.stem + ".mkv")
    assert mkv.exists() and output.exists()
    source = hb.probe(str(sources[kind]))
    kept = hb.probe(str(mkv))
    assert kept["pix_fmt"] == "yuv420p10le"
    for key in ("color_range", "color_space", "color_primaries", "color_transfer",
                "mastering_display", "content_light_level"):
        assert kept[key] == source[key], key
    m = hb.measure(str(sources[kind]), str(mkv))
    # 10 bits: the ramps keep nearly all their levels, with errors of about a code
    assert m["ramp"]["levels"] >= 580 and m["ramp"]["rms"] < 1.5 and m["ramp"]["longest_run"] <= 6
    assert m["shallow"]["levels"] >= 60 and m["shallow"]["rms"] < 1.5
    assert m["chroma"]["cb_levels"] > 150 and m["chroma"]["rms"] < 1.5
    # the MP4 review copy is 8-bit (and says so), as ever
    review = hb.measure(str(sources[kind]), str(output))
    assert review["tags"]["pix_fmt"] == "yuv420p"
    assert review["shallow"]["levels"] <= 17 and review["ramp"]["levels"] < 230


@needs_x265
def test_an_sdr_source_gets_no_hevc_copy_in_auto_and_is_what_off_gives(tmp_path, monkeypatch, sources):
    """SDR above 8 bits is opt-in: in auto the run is the run with the switch off (same MP4
    bytes), with no second file and no bit-depth line."""
    auto, auto_dir, log = run("sdr10", "auto", tmp_path, monkeypatch, sources)
    off, off_dir, _ = run("sdr10", "off", tmp_path, monkeypatch, sources)
    assert not any("bit depth" in line for line in log)
    assert not list(auto_dir.glob("*.mkv")) and not list(off_dir.glob("*.mkv"))
    assert hb.probe(str(auto))["pix_fmt"] == "yuv420p"
    assert content_hash(auto) == content_hash(off)


@needs_x265
def test_without_the_switch_a_high_bit_depth_source_is_8_bit_as_before(tmp_path, monkeypatch, sources):
    output, out_dir, log = run("pq", "off", tmp_path, monkeypatch, sources)
    assert not any("bit depth" in line for line in log)
    assert not list(out_dir.glob("*.mkv"))  # only the MP4, from OpenCV's frames
    assert hb.probe(str(output))["pix_fmt"] == "yuv420p"


@needs_x265
def test_the_painted_region_is_in_the_hevc_copy_and_nothing_else_moved(tmp_path, monkeypatch, sources):
    output, out_dir, log = run("sdr10", "on", tmp_path, monkeypatch, sources)
    mkv = out_dir / (output.stem + ".mkv")
    oy, _, _ = hb.decode_planes(str(mkv))
    x0, y0, x1, y1 = hb.MASK_BOX
    assert oy[y0 + 4:y1 - 4, x0 + 4:x1 - 4].max() < 90  # the stub model painted black (limited range: 64)
    sy, _, _ = (p.astype(int) for p in hb.picture())
    outside = np.ones_like(sy, bool)
    outside[y0 - 12:y1 + 12, x0 - 12:x1 + 12] = False
    assert np.abs(oy - sy)[outside].mean() < 1.5  # only the encode's own noise elsewhere


def content_hash(path):
    """The file's bytes (MP4), or its decoded frames (Matroska writes a random segment
    UID, so two runs never match byte for byte)."""
    if Path(path).suffix == ".mkv":
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "framemd5", "-"],
                             capture_output=True, text=True).stdout
        return hashlib.sha256("\n".join(l for l in out.splitlines() if not l.startswith("#")).encode()).hexdigest()
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="requires ffmpeg")
@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_an_8_bit_source_is_byte_identical_in_every_mode(codec, tmp_path, monkeypatch):
    if codec == "h265" and not has_x265():
        pytest.skip("requires libx265")
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(SAMPLES / "english1.mp4"), "-frames:v", "12",
                    "-an", "-c:v", "libx264", "-crf", "18", str(clip)], check=True)
    monkeypatch.setenv("VIDEOWIPE_OUTPUT_CODEC", codec)
    commands = {}
    real = subprocess.Popen

    def popen(cmd, *args, **kwargs):
        if "rawvideo" in cmd and "-i" in cmd and cmd[cmd.index("-i") + 1] == "-":
            commands.setdefault(mode, []).append(list(cmd))
        return real(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen)
    hashes = {}
    for mode in ("off", "auto", "on"):
        monkeypatch.setenv("VIDEOWIPE_HIGH_BIT_DEPTH", mode)
        decode._decide.cache_clear()
        highbit._current.clear()
        folder = tmp_path / mode
        folder.mkdir()
        rr.run_pipeline(clip, folder, (700, 380, 760, 420))
        hashes[mode] = {p.suffix: content_hash(p) for p in sorted(folder.iterdir())}
    assert hashes["off"]
    for mode in ("auto", "on"):
        assert hashes["off"].keys() == hashes[mode].keys()
        assert hashes["off"] == hashes[mode], mode
    # and the command is the same but for the output folder
    def strip(cmds):
        return [[a.rsplit("/", 1)[-1] if a.startswith(str(tmp_path)) else a for a in c] for c in cmds]

    for mode in ("auto", "on"):
        assert strip(commands["off"]) == strip(commands[mode]), mode
