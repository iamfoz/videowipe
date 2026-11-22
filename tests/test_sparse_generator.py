"""The long-clip generator's sparse tracks: timeline, determinism, ASS output."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gen_long_clip.py"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_long_clip_sparse", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("share", [0.05, 0.12, 0.25, 0.38])
@pytest.mark.parametrize("duration, fps", [(600, 5), (300, 10)])
def test_sparse_track_hits_its_share(gen, share, duration, fps):
    lines = gen.build_sparse_lines(duration, 7, "sd", share, fps)
    on = sum(line["end"] - line["start"] for line in lines)
    # Within one line (at most 5 s) of the target, and never short by a whole line.
    assert abs(on / duration - share) <= 5.0 / duration + 1e-9
    step = 1.0 / fps
    previous_end = 0.0
    for line in lines:
        assert line["track"] == "sd"
        assert 1.0 - 1e-6 <= line["end"] - line["start"] <= 5.0 + 1e-6
        assert abs(line["start"] / step - round(line["start"] / step)) < 1e-6
        assert line["start"] >= previous_end and line["end"] <= duration
        previous_end = line["end"]


def test_sparse_track_is_deterministic_and_independent(gen):
    first = gen.build_sparse_lines(600, 5, "sd", 0.2, 5)
    assert first == gen.build_sparse_lines(600, 5, "sd", 0.2, 5)
    assert first != gen.build_sparse_lines(600, 6, "sd", 0.2, 5)
    assert first != gen.build_sparse_lines(600, 5, "tc", 0.2, 5)
    # Adding a sparse track never moves the dialogue or caption timeline.
    plain = gen.build_timeline(600, 5, fps=5)
    with_sparse = gen.add_sparse_tracks(
        gen.build_timeline(600, 5, fps=5), [("sd", "bottom", 0.1), ("as", "aside", 0.2)], 5,
    )
    strip = lambda lines: [{k: v for k, v in l.items() if k != "id"} for l in lines
                           if l["track"] in ("dialogue", "caption")]
    assert strip(with_sparse["lines"]) == strip(plain["lines"])
    assert {l["track"] for l in with_sparse["lines"]} == {"dialogue", "caption", "sd", "as"}
    ids = [l["id"] for l in with_sparse["lines"]]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    assert with_sparse["sparse_tracks"]["as"] == {"position": "aside", "share": 0.2}


def test_sparse_lines_come_in_bursts_with_gaps(gen):
    lines = gen.build_sparse_lines(600, 9, "sd", 0.1, 5)
    gaps = [b["start"] - a["end"] for a, b in zip(lines, lines[1:])]
    assert min(gaps) >= 0.3 - 1e-6
    assert max(gaps) >= 3.0, "separate bursts are well apart"


def test_parse_sparse(gen):
    assert gen.parse_sparse("sd:bottom:0.05") == ("sd", "bottom", 0.05)
    for bad in ("sd:bottom", "sd:nowhere:0.1", "dialogue:bottom:0.1", "x:top:0",
                "x:top:0.95", "x:top:abc", "9x:top:0.1"):
        with pytest.raises(ValueError):
            gen.parse_sparse(bad)


def test_sparse_lines_get_their_own_style(gen, tmp_path):
    timeline = gen.add_sparse_tracks(
        {"duration": 120.0, "seed": 1, "lines": []},
        [("sd", "bottom", 0.1), ("as", "aside", 0.1), ("tc", "top", 0.1)], 5,
    )
    path = tmp_path / "t.ass"
    gen.write_ass(timeline, path)
    text = path.read_text()
    events = [row for row in text.splitlines() if row.startswith("Dialogue:")]
    by_style = {}
    for line in timeline["lines"]:
        by_style.setdefault(gen._style_of(timeline, line["track"]), set()).add(line["track"])
    assert by_style == {"Dialogue": {"sd"}, "Aside": {"as"}, "Caption": {"tc"}}
    assert any(",Aside," in row for row in events)
    # A per-track file holds that track and the watermark only.
    only = tmp_path / "only.ass"
    gen.write_ass(timeline, only, tracks={"as"})
    rows = [row for row in only.read_text().splitlines() if row.startswith("Dialogue:")]
    assert len(rows) == 1 + sum(1 for l in timeline["lines"] if l["track"] == "as")
    assert all(",Aside," in row for row in rows[1:])


def test_main_rejects_bad_sparse_arguments(gen, tmp_path):
    base = ["--out", str(tmp_path), "--duration", "10", "--fps", "5"]
    assert gen.main(base + ["--sparse", "sd:nowhere:0.1"]) == 2
    assert gen.main(base + ["--sparse", "sd:bottom:0.1", "--sparse", "sd:top:0.1"]) == 2
