"""The long-clip generator's timeline is deterministic and well formed."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gen_long_clip.py"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_long_clip", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timeline_is_deterministic_for_a_seed(gen):
    first = gen.build_timeline(600, seed=5, fps=5)
    again = gen.build_timeline(600, seed=5, fps=5)
    other = gen.build_timeline(600, seed=6, fps=5)
    assert first == again
    assert first != other


@pytest.mark.parametrize("duration, fps", [(120, 10), (300, 5), (1200, 5)])
def test_timeline_shape(gen, duration, fps):
    timeline = gen.build_timeline(duration, seed=3, fps=fps)
    step = 1.0 / fps
    pauses = timeline["pauses"]
    assert pauses, "every clip has at least one long pause"
    no_text = timeline["no_text_stretch"]
    assert no_text in pauses
    for start, end in pauses:
        assert end - start >= 15.0
    for line in timeline["lines"]:
        # Whole frames, so centisecond rounding in the ASS file cannot move a line.
        assert abs(line["start"] / step - round(line["start"] / step)) < 1e-6
        assert 0 < line["end"] - line["start"] <= 5.0 + 1e-6
        assert line["end"] <= duration
        assert "\\N" not in line["text"]
        if line["track"] == "dialogue":
            # Dialogue never speaks inside a pause.
            for start, end in pauses:
                assert line["end"] <= start or line["start"] >= end
        else:
            assert line["end"] <= no_text[0] or line["start"] >= no_text[1]
    dialogue = sorted(
        (l for l in timeline["lines"] if l["track"] == "dialogue"),
        key=lambda l: l["start"],
    )
    for before, after in zip(dialogue, dialogue[1:]):
        assert before["end"] <= after["start"]
    share = sum(l["end"] - l["start"] for l in dialogue) / duration
    assert 0.25 < share < 0.65


def test_ass_times_are_centisecond_exact(gen):
    assert gen._ass_time(61.2) == "0:01:01.20"
    assert gen._ass_time(3600.04) == "1:00:00.04"


def test_fps_must_divide_one_hundred(gen, tmp_path):
    assert gen.main(["--out", str(tmp_path), "--duration", "10", "--fps", "7"]) == 2
