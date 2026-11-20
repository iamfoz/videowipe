"""Duration-aware sampling: the count rule and its effect on a plan."""
import cv2
import numpy as np
import pytest

from videowipe import sampling
from videowipe.detect import (
    DETECT_MODES,
    TextBox,
    _iter_sample_frames_with_indices,
    detect_clean_candidates,
    resolve_detect_params,
)
from videowipe.plan import predicted_mask_at
from videowipe.planning import finalize, prepare
from videowipe.sampling import (
    DEFAULT_INTERVAL_SECONDS,
    bytes_per_sample,
    effective_sample_count,
    interval_scale,
    planned_sample_count,
    sample_interval_seconds,
    sample_memory_bytes,
)

NO_ENV: dict = {}


@pytest.fixture(autouse=True)
def big_machine(monkeypatch):
    monkeypatch.setattr(sampling, "_available_memory_bytes", lambda: 64 * 1024 ** 3)
    monkeypatch.delenv(sampling.ENV_INTERVAL, raising=False)
    monkeypatch.delenv(sampling.ENV_MEMORY, raising=False)


def count(mode, seconds, fps=25.0, env=None, width=1920, height=1080, fallback=None):
    """Sample count for a mode at its resolved parameters."""
    params = resolve_detect_params(mode)
    return effective_sample_count(
        mode, fallback or params["subtitle_fallback"], params["sample_count"],
        int(round(seconds * fps)), fps, width=width, height=height,
        environ=NO_ENV if env is None else env,
    )


def test_default_interval_is_one_second():
    assert DEFAULT_INTERVAL_SECONDS == 1.0
    assert sample_interval_seconds(NO_ENV) == 1.0


# ── the count rule ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "mode, seconds, expected",
    [
        ("balanced", 1, 50),
        ("balanced", 20, 50),
        ("balanced", 50, 50),       # 50 samples * 1 s: the last length that is unchanged
        ("balanced", 51, 51),
        ("balanced", 120, 120),
        ("balanced", 600, 600),
        ("balanced", 2820, 2820),   # 47 minutes: one sample a second, not one a minute
        ("sensitive", 20, 80),
        ("sensitive", 100, 80),     # 80 samples * 1.25 s
        ("sensitive", 101, 81),
        ("sensitive", 600, 480),
        ("fast", 20, 24),
        ("fast", 600, 24),          # fast never grows
        ("fast", 36000, 24),
    ],
)
def test_count_table(mode, seconds, expected):
    assert count(mode, seconds) == expected


def test_subtitle_request_upgrades_fallback_and_interval():
    # An explicit subtitle target forces the band fallback (seven detector calls
    # per sample instead of three), so the interval is a quarter longer.
    assert interval_scale("balanced", "light") == 1.0
    assert interval_scale("balanced", "force") == 1.25
    assert interval_scale("sensitive", "force") == 1.25
    assert interval_scale("fast", "off") is None
    assert interval_scale("fast", "force") is None
    assert count("balanced", 62, fallback="force") == 50
    assert count("balanced", 63, fallback="force") == 51
    assert count("balanced", 600, fallback="force") == 480


@pytest.mark.parametrize("mode", sorted(DETECT_MODES))
def test_unchanged_up_to_the_boundary_for_every_mode(mode):
    base = DETECT_MODES[mode]["sample_count"]
    # No growth at all for the ~20 s calibration clips.
    for seconds in (0.5, 1, 20, 21):
        assert count(mode, seconds) == base
    scale = interval_scale(mode, DETECT_MODES[mode]["subtitle_fallback"])
    if scale is None:
        assert count(mode, 36000) == base
        return
    limit = base * DEFAULT_INTERVAL_SECONDS * scale
    assert count(mode, limit) == base
    assert count(mode, limit + 2) > base


def test_boundary_tolerates_floating_point():
    # An exact multiple of the interval must not round up to one extra sample.
    assert effective_sample_count("balanced", "light", 50, 50, 1.0, environ=NO_ENV) == 50
    assert effective_sample_count("balanced", "light", 50, 1500, 30.0, environ=NO_ENV) == 50
    assert effective_sample_count("balanced", "light", 50, 1498, 29.97, environ=NO_ENV) == 50
    assert effective_sample_count("sensitive", "force", 80, 2500, 25.0, environ=NO_ENV) == 80


# ── the switch ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "value",
    ["off", "OFF", "0", "0.0", "-3", "false", "none", "no", "disabled", "nan", "inf"],
)
def test_override_off_restores_fixed_counts(value):
    env = {sampling.ENV_INTERVAL: value}
    assert sample_interval_seconds(env) is None
    for mode in DETECT_MODES:
        assert count(mode, 36000, env=env) == DETECT_MODES[mode]["sample_count"]


def test_override_sets_the_interval():
    env = {sampling.ENV_INTERVAL: "10"}
    assert sample_interval_seconds(env) == 10.0
    assert count("balanced", 1000, env=env) == 100
    assert count("balanced", 500, env=env) == 50
    assert count("sensitive", 1000, env=env) == 80  # 1000 s / 12.5 s
    assert count("sensitive", 2000, env=env) == 160
    env = {sampling.ENV_INTERVAL: "4"}
    assert count("balanced", 600, env=env) == 150
    assert count("fast", 600, env=env) == 24


@pytest.mark.parametrize("value", ["", "   ", "soon"])
def test_unusable_interval_means_default(value):
    assert sample_interval_seconds({sampling.ENV_INTERVAL: value}) == DEFAULT_INTERVAL_SECONDS


@pytest.mark.parametrize("fps", [0, 0.0, -25.0, float("nan"), float("inf"), None])
def test_unknown_rate_leaves_count_alone(fps):
    assert effective_sample_count("balanced", "light", 50, 100000, fps, environ=NO_ENV) == 50


def test_unknown_length_leaves_count_alone():
    assert effective_sample_count("balanced", "light", 50, 0, 25.0, environ=NO_ENV) == 50
    assert effective_sample_count("balanced", "light", 50, -1, 25.0, environ=NO_ENV) == 50
    assert effective_sample_count("balanced", "light", 0, 100000, 25.0, environ=NO_ENV) == 0


def test_never_more_samples_than_frames():
    assert effective_sample_count("balanced", "light", 50, 60, 0.01, environ=NO_ENV) == 60
    assert effective_sample_count(
        "balanced", "light", 50, 100, 1.0, environ={sampling.ENV_INTERVAL: "0.01"},
    ) == 100


# ── the memory cap ───────────────────────────────────────────────────────────

def test_memory_cap_scales_with_frame_size():
    landscape = bytes_per_sample(1920, 1080)
    portrait = bytes_per_sample(1080, 1920)
    assert landscape == 320 * 180 + 4096
    assert portrait == 320 * 569 + 4096
    ten_hours = 36000
    wide = count("balanced", ten_hours, width=1920, height=1080)
    tall = count("balanced", ten_hours, width=1080, height=1920)
    assert wide == (256 * 1024 * 1024) // landscape
    assert tall == (256 * 1024 * 1024) // portrait
    assert tall < wide < ten_hours


def test_memory_cap_override_and_floor():
    env = {sampling.ENV_MEMORY: "1"}
    # A budget too small for even the mode's own count never reduces it.
    assert count("balanced", 3600, env=env) == 50
    assert count("sensitive", 3600, env=env) == 80
    assert count("balanced", 36000, env={sampling.ENV_MEMORY: "off"}) == 36000
    env = {sampling.ENV_MEMORY: "20"}
    assert count("balanced", 36000, env=env) == (20 * 1024 * 1024) // bytes_per_sample(1920, 1080)
    assert sample_memory_bytes({sampling.ENV_MEMORY: "junk"}) == 256 * 1024 * 1024
    assert sample_memory_bytes({sampling.ENV_MEMORY: "0"}) is None


def test_default_memory_budget_follows_available_memory(monkeypatch):
    unit = 1024 ** 3
    monkeypatch.setattr(sampling, "_available_memory_bytes", lambda: 64 * unit)
    assert sample_memory_bytes(NO_ENV) == 256 * 1024 * 1024
    monkeypatch.setattr(sampling, "_available_memory_bytes", lambda: 2 * unit)
    assert sample_memory_bytes(NO_ENV) == 2 * unit // 16
    monkeypatch.setattr(sampling, "_available_memory_bytes", lambda: None)
    assert sample_memory_bytes(NO_ENV) == 256 * 1024 * 1024
    # An explicit budget is taken as given, whatever the machine has.
    assert sample_memory_bytes({sampling.ENV_MEMORY: "512"}) == 512 * 1024 * 1024
    # A smaller machine gets a smaller cap on the same video.
    monkeypatch.setattr(sampling, "_available_memory_bytes", lambda: 2 * unit)
    assert count("balanced", 36000) == (2 * unit // 16) // bytes_per_sample(1920, 1080)


def test_unknown_frame_size_still_has_a_cap():
    assert count("balanced", 36000, width=0, height=0) < 36000


# ── files on disk ────────────────────────────────────────────────────────────

FPS = 2
FRAMES = 800            # 400 s
LINE = range(394, 405)  # a 5.5 s line in the middle of a 100 s pause
SPEECH = [*range(0, 300), *range(500, 800)]


def _encode(index):
    frame = np.zeros((100, 160, 3), np.uint8)
    frame[0, 0] = (index % 256, index // 256, 7)
    return frame


def _decode(frame):
    return int(frame[0, 0, 0]) + 256 * int(frame[0, 0, 1])


def _write(path, frames, fps):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (160, 100))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(_encode(index))
    writer.release()
    return str(path)


@pytest.fixture
def pause_video(tmp_path):
    return _write(tmp_path / "pause.avi", FRAMES, FPS)


def test_planned_count_probes_the_file(pause_video, tmp_path):
    assert planned_sample_count(pause_video, "balanced", "light", 50) == 400
    assert planned_sample_count(pause_video, "sensitive", "force", 80) == 320
    assert planned_sample_count(pause_video, "fast", "off", 24) == 24
    short = _write(tmp_path / "short.avi", 160, 8)  # 20 s
    assert planned_sample_count(short, "balanced", "light", 50) == 50
    assert planned_sample_count(str(tmp_path / "missing.avi"), "balanced", "light", 50) == 50
    (tmp_path / "junk.avi").write_bytes(b"not a video")
    assert planned_sample_count(str(tmp_path / "junk.avi"), "balanced", "light", 50) == 50


def test_planned_count_off_does_not_open_the_file(monkeypatch):
    monkeypatch.setenv(sampling.ENV_INTERVAL, "off")

    def boom(*_args, **_kwargs):
        raise AssertionError("the file must not be opened when the rule is off")

    monkeypatch.setattr(cv2, "VideoCapture", boom)
    assert planned_sample_count("anything.mp4", "balanced", "light", 50) == 50


class Stop(Exception):
    pass


@pytest.fixture
def requested_count(monkeypatch):
    """Run prepare() up to the detection call and report the sample_count it asked for."""
    seen: dict = {}

    def capture(*_args, **kwargs):
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr("videowipe.detect.detect_clean_candidates", capture)

    def run(video, **kwargs):
        with pytest.raises(Stop):
            prepare(video, ocr="off", warn=lambda _message: None, **kwargs)
        return seen["sample_count"]

    return run


def test_prepare_requests_the_planned_count(pause_video, requested_count, monkeypatch):
    assert requested_count(pause_video, detect_mode="balanced") == 400
    assert requested_count(pause_video, detect_mode="sensitive") == 320
    assert requested_count(pause_video, detect_mode="fast") == 24
    # An explicit subtitle target upgrades the fallback and lengthens the interval.
    assert requested_count(pause_video, detect_mode="balanced", targets=["subtitle"]) == 320
    monkeypatch.setenv(sampling.ENV_INTERVAL, "off")
    assert requested_count(pause_video, detect_mode="balanced") == 50


def test_prepare_leaves_short_videos_alone(tmp_path, requested_count):
    short = _write(tmp_path / "short.avi", 160, 8)  # 20 s, like the calibration clips
    for mode in DETECT_MODES:
        assert requested_count(short, detect_mode=mode) == DETECT_MODES[mode]["sample_count"]


def test_direct_callers_keep_their_explicit_count(pause_video):
    detector = type("Detector", (), {"detect": lambda self, frame: []})()
    result = detect_clean_candidates(pause_video, detector=detector, sample_count=50)
    assert len(result.sample_indices) == 50
    assert len(list(_iter_sample_frames_with_indices(pause_video, 50))) == 50


# ── pipeline: a subtitle line in a long pause between dialogue ───────────────

def _present():
    # Dialogue is on for 7 frames in every 10 while someone speaks.
    on = {index for index in SPEECH if index % 10 < 7}
    return on | set(LINE)


class StubDetector:
    def __init__(self):
        self.present = _present()
        self.box = TextBox(np.array([[30, 80], [130, 80], [130, 90], [30, 90]]), 0.95)

    def detect(self, frame):
        return [self.box] if _decode(frame) in self.present else []


def _pause_frames_removed(video, monkeypatch, interval):
    monkeypatch.setenv(sampling.ENV_INTERVAL, interval)
    draft = prepare(video, detector=StubDetector(), detect_mode="balanced", ocr="off",
                    warn=lambda _message: None)
    plan = finalize(draft)
    assert plan.remove_tracks, "the dialogue band should have become a candidate"
    removed = {index for index in range(300, 500) if predicted_mask_at(plan, index).any()}
    return len(draft._result.sample_indices), removed


def test_pause_line_is_found_with_the_new_rule_and_missed_without(pause_video, monkeypatch):
    samples_old, removed_old = _pause_frames_removed(pause_video, monkeypatch, "off")
    assert samples_old == 50
    fixed = np.linspace(0, FRAMES - 1, 50, dtype=int)
    assert not any(index in LINE for index in fixed), "premise: no fixed sample lands on the line"
    assert removed_old == set(), "the fixed count marks the whole pause absent"

    samples_new, removed_new = _pause_frames_removed(pause_video, monkeypatch, "4")
    assert samples_new == 100  # 400 s at one sample every 4 s
    # Exactly the line; nothing is inferred for the rest of the pause.
    assert removed_new == set(LINE)

    # The default interval (one second) is denser still and finds it too.
    samples_default, removed_default = _pause_frames_removed(pause_video, monkeypatch, "1")
    assert samples_default == 400
    assert removed_default == set(LINE)


def test_available_memory_is_the_smallest_figure(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 16000000 kB\nMemAvailable: 8000000 kB\n")
    cgroup = tmp_path / "memory.max"
    cgroup.write_text("max\n")
    real_open = open
    mapping = {
        "/proc/meminfo": meminfo,
        "/sys/fs/cgroup/memory.max": cgroup,
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": tmp_path / "absent",
    }

    def fake_open(path, *args, **kwargs):
        return real_open(mapping.get(str(path), path), *args, **kwargs)

    monkeypatch.undo()  # drop the autouse stub so the real function runs
    monkeypatch.setattr(sampling, "open", fake_open, raising=False)
    assert sampling._available_memory_bytes() == 8000000 * 1024
    cgroup.write_text(str(512 * 1024 * 1024))
    assert sampling._available_memory_bytes() == 512 * 1024 * 1024
