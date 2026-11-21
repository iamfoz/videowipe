"""OCR crops must come from the sampled frame that the box was detected on."""
import cv2
import numpy as np
import pytest

from videowipe.detect import (
    TextBox,
    _bbox,
    _iter_sample_frames_with_indices,
    _ocr_crop_budget_bytes,
    _ocr_crops_by_box,
    detect_clean_candidates,
)

WIDTH, HEIGHT, FRAMES = 320, 180, 12
POINTS = np.array([[120, 75], [200, 75], [200, 95], [120, 95]])
FIRST_TEXT_FRAME = 6
BACKGROUND = 20


def _level(index):
    """Pixel level of the text area on a frame; each frame has its own level."""
    return 110 + 15 * (index - FIRST_TEXT_FRAME)


def _write_late_text_video(path):
    """Plain background at the box position until frame 6, then a level per frame."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (WIDTH, HEIGHT),
    )
    for i in range(FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
        if i >= FIRST_TEXT_FRAME:
            frame[75:96, 120:201] = _level(i)
        writer.write(frame)
    writer.release()


class _LateTextDetector:
    """Finds the box from the pixels, so it only exists once the text appears."""

    def detect(self, frame):
        if np.median(frame[75:96, 120:201]) > 80:
            return [TextBox(points=POINTS, confidence=0.9)]
        return []


class _Recogniser:
    """Reads ``reading`` from a bright crop and nothing from background."""

    def __init__(self, reading):
        self.reading = reading
        self.crops = []

    def __call__(self, crop):
        self.crops.append(crop.copy())
        return self.reading if np.median(crop) > 80 else None


@pytest.mark.parametrize(
    ("reading", "target_type", "default_remove"),
    [
        ("2026-05-25 12:30:05", "timestamp", True),
        ("@brand", "watermark", True),
    ],
)
def test_recogniser_reads_pixels_from_the_boxes_own_frame(
    tmp_path, reading, target_type, default_remove,
):
    video = tmp_path / "input.mp4"
    _write_late_text_video(video)
    recogniser = _Recogniser(reading)

    result = detect_clean_candidates(
        str(video), detector=_LateTextDetector(), sample_count=FRAMES,
        recognizer=recogniser,
    )

    # Up to three crops, in sample order, each from the frame holding its box.
    assert len(recogniser.crops) == 3
    for crop, frame_index in zip(recogniser.crops, (6, 7, 8)):
        assert crop.shape[:2] == (21, 81)
        assert abs(float(np.median(crop)) - _level(frame_index)) <= 8
    (candidate,) = result.candidates
    assert candidate.text_samples == [reading]
    assert candidate.type == target_type
    assert candidate.default_remove is default_remove


def test_text_less_box_without_a_kept_crop_is_skipped(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    _write_late_text_video(video)
    recogniser = _Recogniser("12:30")
    monkeypatch.setattr("videowipe.detect._ocr_crops_by_box", lambda *args: {})

    result = detect_clean_candidates(
        str(video), detector=_LateTextDetector(), sample_count=FRAMES,
        recognizer=recogniser,
    )

    assert recogniser.crops == []  # no silent fall back to the first frame
    (candidate,) = result.candidates
    assert candidate.text_samples == []
    assert candidate.type == "scene_text"


def test_crop_helper_keeps_only_text_less_boxes_as_copies():
    frame = np.arange(HEIGHT * WIDTH * 3, dtype=np.uint32).astype(np.uint8)
    frame = frame.reshape(HEIGHT, WIDTH, 3)
    plain = TextBox(points=POINTS, confidence=0.9)
    read = TextBox(points=np.array([[10, 10], [60, 10], [60, 30], [10, 30]]),
                   confidence=0.9, text="known")

    crops = _ocr_crops_by_box(frame, [plain, read], WIDTH, HEIGHT)

    assert set(crops) == {id(plain)}
    x1, y1, x2, y2 = _bbox(plain.points, WIDTH, HEIGHT)
    assert np.array_equal(crops[id(plain)], frame[y1:y2 + 1, x1:x2 + 1])
    assert not np.shares_memory(crops[id(plain)], frame)


# Upstream (20872d4) results for the cases this change must leave alone. The
# literals were produced by running these same scenarios on upstream.
def _run_unchanged_case(tmp_path, *, box_text, with_recogniser):
    video = tmp_path / "input.mp4"
    _write_late_text_video(video)
    recogniser = _Recogniser("never wanted")

    class Detector:
        def detect(self, frame):
            if np.median(frame[75:96, 120:201]) > 80:
                return [TextBox(points=POINTS, confidence=0.9, text=box_text)]
            return []

    result = detect_clean_candidates(
        str(video), detector=Detector(), sample_count=FRAMES,
        recognizer=recogniser if with_recogniser else None,
    )
    (candidate,) = result.candidates
    return recogniser, (
        candidate.type, candidate.reason, candidate.default_remove,
        candidate.text_samples, candidate.presence_frames,
    )


def test_boxes_with_text_never_reach_the_recogniser(tmp_path):
    recogniser, outcome = _run_unchanged_case(
        tmp_path, box_text="12:30:05", with_recogniser=True,
    )

    assert recogniser.crops == []
    assert outcome == (
        "timestamp", "time-like text in center", True,
        ["12:30:05"], [6, 7, 8, 9, 10, 11],
    )


def test_no_recogniser_leaves_text_less_boxes_unread(tmp_path):
    _, outcome = _run_unchanged_case(
        tmp_path, box_text="", with_recogniser=False,
    )

    assert outcome == (
        "scene_text", "text in center", False, [], [6, 7, 8, 9, 10, 11],
    )


# Bounding the crops kept while sampling. Only the first three text-less boxes
# of a region are ever read, so the crops held must not grow with the samples.
ROW_COUNT = 40
ROW_BOXES = [
    np.array([[40, 30], [280, 30], [280, 70], [40, 70]]),
    np.array([[40, 100], [280, 100], [280, 140], [40, 140]]),
]


def _write_two_row_video(path):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (WIDTH, HEIGHT),
    )
    for i in range(ROW_COUNT):
        frame = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
        frame[30:71, 40:281] = 100 + 3 * i
        frame[100:141, 40:281] = 200 - 3 * i
        writer.write(frame)
    writer.release()


class _TwoRowDetector:
    def detect(self, frame):
        return [TextBox(points=points, confidence=0.9) for points in ROW_BOXES]


def _spy_crop_bytes(monkeypatch):
    created = []
    original = _ocr_crops_by_box

    def spy(*args):
        crops = original(*args)
        created.extend(crop.nbytes for crop in crops.values())
        return crops

    monkeypatch.setattr("videowipe.detect._ocr_crops_by_box", spy)
    return created


def _run_two_rows(tmp_path, monkeypatch, budget):
    video = tmp_path / "rows.mp4"
    if not video.exists():
        _write_two_row_video(video)
    monkeypatch.setenv("VIDEOWIPE_OCR_CROP_BUDGET_MB", budget)
    created = _spy_crop_bytes(monkeypatch)
    recogniser = _Recogniser("caption")
    result = detect_clean_candidates(
        str(video), detector=_TwoRowDetector(), sample_count=ROW_COUNT,
        recognizer=recogniser,
    )
    return result, recogniser, sum(created)


def test_crops_kept_while_sampling_stay_within_the_budget(tmp_path, monkeypatch):
    _, _, unbounded = _run_two_rows(tmp_path, monkeypatch, "1000")
    one_sample = 2 * 41 * 241 * 3
    assert unbounded == ROW_COUNT * one_sample  # one copy per box per sample

    # Room for about three samples; the crop held is only ever overshot by one.
    _, _, bounded = _run_two_rows(tmp_path, monkeypatch, str(3.5 * one_sample / 2**20))

    assert bounded <= 4 * one_sample


def test_crops_read_after_the_budget_match_the_unbounded_run(tmp_path, monkeypatch):
    kept, read_kept, _ = _run_two_rows(tmp_path, monkeypatch, "1000")
    for budget in ("0", "0.01"):
        refetched, read_refetched, held = _run_two_rows(tmp_path, monkeypatch, budget)

        assert len(read_refetched.crops) == len(read_kept.crops) == 6
        for got, want in zip(read_refetched.crops, read_kept.crops):
            assert np.array_equal(got, want)
        assert [
            (c.type, c.reason, c.default_remove, c.text_samples, c.presence_frames)
            for c in refetched.candidates
        ] == [
            (c.type, c.reason, c.default_remove, c.text_samples, c.presence_frames)
            for c in kept.candidates
        ]
        assert held <= 6 * 41 * 241 * 3  # only the crops that are read


@pytest.mark.parametrize("budget", ["0", "1000"])
def test_late_box_is_read_from_its_own_frame_whatever_the_budget(
    tmp_path, monkeypatch, budget,
):
    video = tmp_path / "input.mp4"
    _write_late_text_video(video)
    monkeypatch.setenv("VIDEOWIPE_OCR_CROP_BUDGET_MB", budget)
    recogniser = _Recogniser("@brand")

    detect_clean_candidates(
        str(video), detector=_LateTextDetector(), sample_count=FRAMES,
        recognizer=recogniser,
    )

    assert len(recogniser.crops) == 3
    for crop, frame_index in zip(recogniser.crops, (6, 7, 8)):
        assert crop.shape[:2] == (21, 81)
        assert abs(float(np.median(crop)) - _level(frame_index)) <= 8


def test_frames_are_read_once_while_crops_fit_in_the_budget(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    _write_late_text_video(video)
    reads = []
    original = _iter_sample_frames_with_indices
    monkeypatch.setattr(
        "videowipe.detect._iter_sample_frames_with_indices",
        lambda *args, **kwargs: reads.append(1) or original(*args, **kwargs),
    )
    monkeypatch.delenv("VIDEOWIPE_OCR_CROP_BUDGET_MB", raising=False)

    detect_clean_candidates(
        str(video), detector=_LateTextDetector(), sample_count=FRAMES,
        recognizer=_Recogniser("12:30"),
    )

    assert len(reads) == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 64 * 2**20), ("", 64 * 2**20), ("lots", 64 * 2**20),
        ("8", 8 * 2**20), ("0.5", 2**19), ("0", 0), ("-3", 0),
    ],
)
def test_budget_override(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("VIDEOWIPE_OCR_CROP_BUDGET_MB", raising=False)
    else:
        monkeypatch.setenv("VIDEOWIPE_OCR_CROP_BUDGET_MB", value)

    assert _ocr_crop_budget_bytes() == expected
