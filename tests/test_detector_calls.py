"""Exactness tests for the P2-4 change: band fallback without a second decode.

``original_detect_clean_candidates`` and ``original_band_fallback_detect``
below are verbatim copies (renamed only) of the functions as they were before
the band-fallback detections moved into the sampling pass; the current code
must reproduce their results exactly, including the order of detector calls
and the detector state each call sees (``DBNetDetector._manual_only`` flips
during the main pass and the fallback has always run with its final value).
"""
from __future__ import annotations

import logging
from itertools import chain
from typing import Callable, Iterable

import cv2
import numpy as np
import pytest

from videowipe import memory, perfstats
from videowipe.detect import (
    _PERSISTENT_OVERLAY_FRACTION,
    CleanCandidate,
    CleanDetectionResult,
    TextBox,
    TextDetector,
    _appearance_stability,
    _appearance_thumbnail,
    _band_fallback_detect,
    _bbox,
    _box_overlaps_bbox,
    _candidate_label,
    _classify_region,
    _default_detector,
    _detect_fixed_logo_candidates,
    _detect_translucent_watermark_candidates,
    _iou_bbox,
    _iter_sample_frames_with_indices,
    _prepare_band_variant,
    _region_candidates,
    _zone,
    detect_clean_candidates,
)
from videowipe.errors import ProcessingCancelledError

logger = logging.getLogger("videowipe.detect")


# ── Oracles: the pre-change implementation, unchanged apart from names ──────

def original_band_fallback_detect(
    frames: Iterable[np.ndarray],
    detector: TextDetector,
    mode: str = "light",
    consistency: float = 0.4,
    frame_shape: tuple[int, int] | None = None,
) -> list[CleanCandidate]:
    """Detect text in subtitle bands missed by the main frequency-map pass.

    For each band (bottom 40%, optionally top 25%), creates image variants
    and runs the detector.  Builds a per-pixel frequency map from fallback
    detections, then extracts connected-component candidates.
    """
    frame_iter = iter(frames)
    if frame_shape is None:
        try:
            first = next(frame_iter)
        except StopIteration:
            return []
        frame_shape = first.shape[:2]
        frame_iter = chain((first,), frame_iter)
    h, w = frame_shape

    bands: list[tuple[str, int, int, int, int]] = []
    bands.append(("bottom", 0, int(h * 0.6), w - 1, h - 1))
    if mode == "force":
        bands.append(("top", 0, 0, w - 1, max(0, int(h * 0.25) - 1)))

    if mode == "light":
        variants = ["original", "contrast"]
    else:
        variants = ["original", "contrast", "inverted"]

    freq = np.zeros((h, w), dtype=np.float32)
    n_valid = 0

    for frame in frame_iter:
        frame_mask = np.zeros((h, w), dtype=np.uint8)
        for _band_name, bx1, by1, bx2, by2 in bands:
            crop = frame[by1:by2 + 1, bx1:bx2 + 1]
            if crop.size == 0:
                continue
            for variant in variants:
                prepared = _prepare_band_variant(crop, variant)
                try:
                    boxes = perfstats.detect(detector, prepared)
                except Exception:
                    continue
                for box in boxes:
                    pts = box.points.copy()
                    pts[:, 0] += bx1
                    pts[:, 1] += by1
                    cv2.fillPoly(frame_mask, [pts.astype(np.int32)], 1)
        freq += frame_mask
        n_valid += 1

    if n_valid == 0:
        return []

    freq /= n_valid

    text_mask = (freq >= consistency).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    text_mask = cv2.dilate(text_mask, kernel, iterations=2)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels = cv2.connectedComponents(text_mask)

    candidates: list[CleanCandidate] = []
    for label_id in range(1, num_labels):
        component = (labels == label_id).astype(np.uint8)
        ys, xs = np.where(component > 0)
        if len(xs) < 10:
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        region_freq = freq[y1:y2 + 1, x1:x2 + 1]
        pos_freq = region_freq[region_freq > 0]
        confidence = float(pos_freq.mean()) if len(pos_freq) > 0 else 0.0
        frame_fraction = float(region_freq.mean())
        zone = _zone((x1, y1, x2, y2), w, h)
        target_type, reason, default_remove = _classify_region(
            (x1, y1, x2, y2), zone, [], w, h,
            presence_fraction=confidence,
        )

        component_mask = (labels[y1:y2 + 1, x1:x2 + 1] == label_id).astype(np.uint8)
        full_mask = np.zeros((h, w, 1), dtype=np.uint8)
        full_mask[y1:y2 + 1, x1:x2 + 1, 0] = component_mask

        candidates.append(
            CleanCandidate(
                id=f"b{len(candidates) + 1}",
                type=target_type,  # type: ignore[arg-type]
                label=_candidate_label(target_type, zone),
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                frame_fraction=frame_fraction,
                reason=f"band fallback: {reason}",
                default_remove=default_remove,
                mask=full_mask,
            )
        )

    return candidates


def original_detect_clean_candidates(
    video_path: str,
    detector: TextDetector | None = None,
    sample_count: int = 50,
    regions: Iterable[str] | None = None,
    detect_text: bool = True,
    include_logo: bool = False,
    include_translucent_watermark: bool = False,
    consistency: float = 0.4,
    subtitle_fallback: str = "off",
    recognizer: object | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> CleanDetectionResult:
    """Detect removable clean targets from sampled video frames.

    Uses a frequency-map approach: builds a per-pixel frequency map of
    text detections across sampled frames, then finds connected regions
    and classifies each by position and content.

    Args:
        recognizer: Optional callable ``(image_crop) -> str | None``.
            When provided and a detected text box has no ``.text`` field,
            the crop is passed to *recognizer* to fill ``text_samples``.
    """
    if detect_text and detector is None:
        detector = _default_detector()

    candidates: list[CleanCandidate] = []
    n_valid = 0
    valid_sample_indices: list[int] = []
    sampled_frame_boxes: dict[int, list[TextBox]] = {}
    failed_frame_indices: set[int] = set()
    all_frame_boxes: list[tuple[int, list[TextBox]]] = []
    appearance_thumbnails: dict[int, np.ndarray] = {}
    first_frame = None
    best_preview = None
    best_preview_score = -1
    logo_frames: list[np.ndarray] = []
    freq = None

    perfstats.begin("coarse_sampling")
    for sample_index, frame in perfstats.counted(_iter_sample_frames_with_indices(
        video_path, sample_count, check_cancelled=check_cancelled,
    )):
        if first_frame is None:
            first_frame = frame
            h, w = frame.shape[:2]
            if detect_text and detector is not None:
                freq = np.zeros((h, w), dtype=np.float32)
        if include_logo and len(logo_frames) < 6:
            logo_frames.append(frame)

        if not detect_text or detector is None:
            continue
        try:
            boxes = perfstats.detect(detector, frame)
        except ProcessingCancelledError:
            raise
        except Exception as exc:
            failed_frame_indices.add(sample_index)
            sampled_frame_boxes[sample_index] = []
            logger.warning("Clean detection failed on sampled frame: %s", exc)
            continue
        n_valid += 1
        valid_sample_indices.append(sample_index)
        sampled_frame_boxes[sample_index] = boxes
        all_frame_boxes.append((sample_index, boxes))
        appearance_thumbnails[sample_index] = _appearance_thumbnail(frame)
        frame_mask = np.zeros((h, w), dtype=np.uint8)
        for box in boxes:
            cv2.fillPoly(frame_mask, [box.points.astype(np.int32)], 1)
        freq += frame_mask

        score = int(frame_mask.sum())
        if score > best_preview_score:
            best_preview_score = score
            best_preview = frame
    perfstats.end()

    if first_frame is None:
        raise ValueError(f"No frames could be read from: {video_path}")
    frame = None
    h, w = first_frame.shape[:2]

    if detect_text and detector is not None:
        if n_valid == 0:
            raise RuntimeError("Target detection failed on all sampled frames.")

        assert freq is not None
        freq /= n_valid

        text_mask = (freq >= consistency).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        text_mask = cv2.dilate(text_mask, kernel, iterations=2)
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels = cv2.connectedComponents(text_mask)

        raw_regions: list[tuple[int, int, int, int, int]] = []  # (label_id, x1, y1, x2, y2)
        for label_id in range(1, num_labels):
            component = (labels == label_id).astype(np.uint8)
            ys, xs = np.where(component > 0)
            if len(xs) < 10:
                continue
            raw_regions.append((label_id, int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))

        raw_regions.sort(key=lambda b: (b[2], b[1]))

        for label_id, x1, y1, x2, y2 in raw_regions:
            text_samples: list[str] = []
            ocr_crops: list[np.ndarray] = []
            presence_frames: list[int] = []
            for sample_index, boxes in all_frame_boxes:
                overlapped = False
                for box in boxes:
                    if _box_overlaps_bbox(box, (x1, y1, x2, y2), w, h):
                        overlapped = True
                        bx1, by1, bx2, by2 = _bbox(box.points, w, h)
                        if box.text:
                            text_samples.append(box.text)
                        elif recognizer is not None:
                            crop = first_frame[by1:by2 + 1, bx1:bx2 + 1]
                            if crop.size > 0:
                                ocr_crops.append(crop)
                if overlapped:
                    presence_frames.append(sample_index)

            if not text_samples and ocr_crops and recognizer is not None:
                for crop in ocr_crops[:3]:
                    try:
                        text = recognizer(crop)
                    except Exception:
                        text = None
                    if text:
                        text_samples.append(text)

            zone = _zone((x1, y1, x2, y2), w, h)
            appearance_stability = _appearance_stability(
                appearance_thumbnails,
                presence_frames,
                (x1, y1, x2, y2),
                (h, w),
            )
            target_type, reason, default_remove = _classify_region(
                (x1, y1, x2, y2), zone, text_samples, w, h,
                presence_fraction=len(presence_frames) / n_valid,
                appearance_stability=appearance_stability,
            )

            region_freq = freq[y1:y2 + 1, x1:x2 + 1]
            pos_freq = region_freq[region_freq > 0]
            confidence = float(pos_freq.mean()) if len(pos_freq) > 0 else 0.0
            frame_fraction = float(region_freq.mean())

            component_mask = (labels[y1:y2 + 1, x1:x2 + 1] == label_id).astype(np.uint8)
            full_mask = np.zeros((h, w, 1), dtype=np.uint8)
            full_mask[y1:y2 + 1, x1:x2 + 1, 0] = component_mask

            candidates.append(
                CleanCandidate(
                    id=f"c{len(candidates) + 1}",
                    type=target_type,  # type: ignore[arg-type]
                    label=_candidate_label(target_type, zone),
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    frame_fraction=frame_fraction,
                    reason=reason,
                    default_remove=default_remove,
                    text_samples=sorted(set(text_samples))[:5],
                    presence_frames=presence_frames,
                    mask=full_mask,
                    detector_backed=True,
                )
            )

    if subtitle_fallback != "off" and detect_text and detector is not None:
        perfstats.begin("band_fallback")
        try:
            fallback_raw = original_band_fallback_detect(
                (
                    frame
                    for _, frame in perfstats.counted(_iter_sample_frames_with_indices(
                        video_path, sample_count, check_cancelled=check_cancelled,
                    ))
                ),
                detector,
                mode=subtitle_fallback,
                consistency=consistency,
                frame_shape=(h, w),
            )
        finally:
            perfstats.end()
        main_bboxes = [c.bbox for c in candidates]
        surviving = [
            fc for fc in fallback_raw
            if not any(_iou_bbox(fc.bbox, mb) > 0.3 for mb in main_bboxes)
        ]
        for idx, fc in enumerate(surviving, 1):
            fc.id = f"b{idx}"
            candidates.append(fc)

    stable_overlays = []
    if n_valid:
        for candidate in candidates:
            x1, y1, x2, y2 = candidate.bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if (
                candidate.detector_backed
                and candidate.type == "scene_text"
                and len(candidate.presence_frames) / n_valid
                >= _PERSISTENT_OVERLAY_FRACTION
                and w * 0.25 <= cx <= w * 0.75
                and h * 0.30 <= cy <= h * 0.82
                and (x2 - x1 + 1) / w <= 0.45
                and (y2 - y1 + 1) / h <= 0.18
            ):
                stable_overlays.append(candidate)

    details = (
        _detect_translucent_watermark_candidates(
            [first_frame], anchors=stable_overlays,
        )
        if stable_overlays else []
    )
    details_by_id = {candidate.id: candidate for candidate in details}
    promoted_overlay_ids: set[str] = set()
    for candidate in stable_overlays:
        detail = details_by_id.get(candidate.id)
        if detail is None:
            continue
        anchor_width = candidate.bbox[2] - candidate.bbox[0] + 1
        horizontal_extension = max(
            candidate.bbox[0] - detail.bbox[0],
            detail.bbox[2] - candidate.bbox[2],
        )
        if horizontal_extension < max(4, int(round(anchor_width * 0.08))):
            continue
        detail_mask = cv2.dilate(
            detail.mask[:, :, 0],
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )[:, :, None]
        candidate.mask = np.maximum(candidate.mask, detail_mask)
        mask_ys, mask_xs = np.where(candidate.mask[:, :, 0] > 0)
        candidate.bbox = (
            int(mask_xs.min()), int(mask_ys.min()),
            int(mask_xs.max()), int(mask_ys.max()),
        )
        candidate.type = "watermark"
        candidate.label = _candidate_label("watermark", _zone(candidate.bbox, w, h))
        candidate.reason = "stable compact center overlay"
        candidate.default_remove = True
        promoted_overlay_ids.add(candidate.id)

    next_index = len(candidates) + 1
    if regions:
        candidates.extend(_region_candidates(regions, (h, w), start_index=next_index))
        next_index = len(candidates) + 1
    if include_logo:
        candidates.extend(
            _detect_fixed_logo_candidates(logo_frames, start_index=next_index)
        )
        next_index = len(candidates) + 1
    if include_translucent_watermark and not promoted_overlay_ids:
        candidates.extend(
            _detect_translucent_watermark_candidates(
                [first_frame], start_index=next_index,
            )
        )

    return CleanDetectionResult(
        candidates=candidates,
        frame_shape=(h, w),
        sample_indices=valid_sample_indices,
        preview_frame=(
            best_preview if best_preview is not None else first_frame
        ).copy(),
        detector=detector,
        sampled_frame_boxes=sampled_frame_boxes,
        failed_frame_indices=failed_frame_indices,
    )


# ── Fixtures ────────────────────────────────────────────────────────────────

W, H, FRAMES = 160, 100, 60
SAMPLE_COUNT = 10  # sampled indices: 0, 6, 13, 19, 26, 32, 39, 45, 52, 59
SWITCH = 30        # text is bright before this frame and dim from it on


def _level(index: int) -> int:
    return 230 if index < SWITCH else 150


def make_frame(index: int) -> np.ndarray:
    # Dark horizontal gradient: nothing below 120 reads as text.
    row = np.linspace(20, 100, W).astype(np.uint8)
    frame = np.repeat(row[None, :, None], H, axis=0).repeat(3, axis=2)
    frame[5:16, 5:45] = _level(index)       # top-left mark, seen by full frames
    frame[70:89, 20:141] = _level(index)    # subtitle in the bottom band
    frame[0, W - 1, :] = index              # frame number, for failure injection
    return frame


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("detector_calls") / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"), 8, (W, H))
    assert writer.isOpened()
    for index in range(FRAMES):
        writer.write(make_frame(index))
    writer.release()
    return path


def _rect(x1, y1, x2, y2, pad=0):
    x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


class FlipDetector:
    """A stand-in with ``DBNetDetector``'s state machine.

    The "high-level" path only fires on bright pixels and returns tight boxes;
    the "manual" path fires on dimmer pixels and returns padded boxes. Like the
    real detector it tries the high-level path first (unless ``_manual_only``),
    falls back to the manual path when that finds nothing, and latches
    ``_manual_only`` the first time the manual path rescues a call. Boxes thus
    depend on the flag's history, which is exactly the hazard for running
    fallback detections in a different order.
    """

    def __init__(self, fail_frames=(), on_crop_call: Callable[[int], None] | None = None):
        self._manual_only = False
        self.log: list[tuple] = []
        self.fail_frames = set(fail_frames)
        self.on_crop_call = on_crop_call
        self.crop_calls = 0

    def _find(self, gray, threshold, pad):
        ys, xs = np.where(gray > threshold)
        if len(xs) < 20:
            return []
        return [TextBox(_rect(xs.min(), ys.min(), xs.max(), ys.max(), pad), 0.9)]

    def detect(self, frame):
        is_full = frame.shape[0] == H
        gray = frame[..., 0].astype(np.int32)
        if is_full:
            number = int(frame[0, -1, 0])
            self.log.append(("full", number, self._manual_only))
            if number in self.fail_frames:
                raise RuntimeError(f"stub failure on frame {number}")
            gray = gray.copy()
            gray[50:, :] = 0   # a full frame only reveals its top half
            gray[0, :] = 0
        else:
            self.crop_calls += 1
            self.log.append(("crop", frame.shape, int(frame.sum()), self._manual_only))
            if self.on_crop_call is not None:
                self.on_crop_call(self.crop_calls)
        tried_hl = not self._manual_only
        if tried_hl:
            boxes = self._find(gray, 200, 0)
            if boxes:
                return boxes
        boxes = self._find(gray, 120, 6)
        if tried_hl and boxes:
            self._manual_only = True
        return boxes


def _capture_counter(monkeypatch):
    """Count capture objects opened and frames advanced through them."""
    real = cv2.VideoCapture
    counts = {"opened": 0, "advanced": 0}

    class Counting:
        def __init__(self, *args, **kwargs):
            counts["opened"] += 1
            self._cap = real(*args, **kwargs)

        def read(self):
            result = self._cap.read()
            counts["advanced"] += 1 if result[0] else 0
            return result

        def grab(self):
            ok = self._cap.grab()
            counts["advanced"] += 1 if ok else 0
            return ok

        def __getattr__(self, name):
            return getattr(self._cap, name)

    monkeypatch.setattr(cv2, "VideoCapture", Counting)
    return counts


def _boxes_equal(a, b):
    assert len(a) == len(b)
    for left, right in zip(a, b):
        assert np.array_equal(left.points, right.points)
        assert left.confidence == right.confidence and left.text == right.text


def assert_same_result(a: CleanDetectionResult, b: CleanDetectionResult) -> None:
    assert a.frame_shape == b.frame_shape
    assert a.sample_indices == b.sample_indices
    assert a.failed_frame_indices == b.failed_frame_indices
    assert np.array_equal(a.preview_frame, b.preview_frame)
    assert [c.to_dict() for c in a.candidates] == [c.to_dict() for c in b.candidates]
    for left, right in zip(a.candidates, b.candidates):
        assert left.text_samples == right.text_samples
        assert left.detector_backed == right.detector_backed
        assert left.temporal_sample_indices == right.temporal_sample_indices
        assert np.array_equal(left.mask, right.mask)
    assert list(a.sampled_frame_boxes) == list(b.sampled_frame_boxes)
    for index in a.sampled_frame_boxes:
        _boxes_equal(a.sampled_frame_boxes[index], b.sampled_frame_boxes[index])


def run_both(video, mode, make_detector=FlipDetector, **kwargs):
    """The pre-change function and the current one, each on a fresh detector."""
    old_detector, new_detector = make_detector(), make_detector()
    common = dict(sample_count=SAMPLE_COUNT, subtitle_fallback=mode, **kwargs)
    old = original_detect_clean_candidates(video, detector=old_detector, **common)
    new = detect_clean_candidates(video, detector=new_detector, **common)
    return old, new, old_detector, new_detector


# ── Identical results ───────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["off", "light", "force", "on"])
def test_results_and_detector_history_match_the_original(video, mode):
    old, new, old_detector, new_detector = run_both(video, mode)
    assert_same_result(old, new)
    # Same detector calls, same order, same flag value at every call.
    assert new_detector.log == old_detector.log
    assert new_detector._manual_only == old_detector._manual_only is True
    if mode != "off":
        assert any(c.id.startswith("b") for c in new.candidates), \
            "the fallback must contribute a candidate or this test proves little"


def test_the_flag_flips_so_an_interleaved_order_would_differ(video):
    """The hazard is real: fallback calls interleaved with sampling see
    ``_manual_only`` still False for the early frames and box them differently."""
    old, _, old_detector, _ = run_both(video, "light")
    crop_flags = {entry[-1] for entry in old_detector.log if entry[0] == "crop"}
    assert crop_flags == {True}          # today every fallback call runs latched
    full_flags = [entry[-1] for entry in old_detector.log if entry[0] == "full"]
    assert full_flags[0] is False and full_flags[-1] is True

    interleaved = FlipDetector()
    first_crop_boxes = None
    for index, frame in _iter_sample_frames_with_indices(video, SAMPLE_COUNT):
        interleaved.detect(frame)
        crop = frame[int(H * 0.6):, :]
        for variant in ("original", "contrast"):
            boxes = interleaved.detect(_prepare_band_variant(crop, variant))
            if first_crop_boxes is None:
                first_crop_boxes = boxes
    assert any(entry[0] == "crop" and entry[-1] is False for entry in interleaved.log)
    # ...and the very first fallback detection really differs in content.
    latched = FlipDetector()
    latched._manual_only = True
    crop = make_frame(0)[int(H * 0.6):, :]
    assert not np.array_equal(first_crop_boxes[0].points, latched.detect(crop)[0].points)


@pytest.mark.parametrize("mode", ["light", "force"])
def test_detector_failures_on_sampled_frames_still_match(video, mode):
    # A failed sampled frame still has its crops examined by the fallback.
    def make():
        return FlipDetector(fail_frames={6, 26, 59})
    old, new, old_detector, new_detector = run_both(video, mode, make_detector=make)
    assert new.failed_frame_indices == {6, 26, 59}
    assert_same_result(old, new)
    assert new_detector.log == old_detector.log


def test_no_text_detection_means_no_fallback_work(video, monkeypatch):
    counts = _capture_counter(monkeypatch)
    detector = FlipDetector()
    result = detect_clean_candidates(
        video, detector=detector, sample_count=SAMPLE_COUNT, detect_text=False,
        regions=["bottom"], subtitle_fallback="light",
    )
    assert detector.log == [] and counts["opened"] == 1
    assert [c.type for c in result.candidates] == ["region"]


def test_band_fallback_detect_wrapper_is_unchanged(video):
    frames = [f for _, f in _iter_sample_frames_with_indices(video, SAMPLE_COUNT)]
    for mode in ("light", "force"):
        old_detector, new_detector = FlipDetector(), FlipDetector()
        old_detector._manual_only = new_detector._manual_only = True
        for frame_shape in (None, (H, W)):
            old = original_band_fallback_detect(
                frames, old_detector, mode=mode, consistency=0.4, frame_shape=frame_shape)
            new = _band_fallback_detect(
                frames, new_detector, mode=mode, consistency=0.4, frame_shape=frame_shape)
            assert [c.to_dict() for c in old] == [c.to_dict() for c in new]
            assert all(np.array_equal(a.mask, b.mask) for a, b in zip(old, new))
    assert _band_fallback_detect([], FlipDetector()) == []


# ── The decode pass is gone ─────────────────────────────────────────────────

def test_light_fallback_decodes_the_file_once(video, monkeypatch):
    counts = _capture_counter(monkeypatch)
    original_detect_clean_candidates(
        video, detector=FlipDetector(), sample_count=SAMPLE_COUNT, subtitle_fallback="light")
    assert counts == {"opened": 2, "advanced": 2 * FRAMES}

    counts.update(opened=0, advanced=0)
    handle = perfstats.start()
    detect_clean_candidates(
        video, detector=FlipDetector(), sample_count=SAMPLE_COUNT, subtitle_fallback="light")
    stats = perfstats.finish(handle)
    assert counts == {"opened": 1, "advanced": FRAMES}
    fallback = stats["band_fallback"]
    assert fallback["frames_converted"] == fallback["frames_grabbed"] == 0
    assert fallback["band_crops_stored"] == SAMPLE_COUNT and fallback["band_second_pass"] == 0
    # Detector work is the same as before: two variants on each sampled frame.
    assert fallback["detect_calls"] == 2 * SAMPLE_COUNT
    assert stats["coarse_sampling"]["detect_calls"] == SAMPLE_COUNT


def test_fallback_off_never_stores_crops(video, monkeypatch):
    counts = _capture_counter(monkeypatch)
    handle = perfstats.start()
    detect_clean_candidates(
        video, detector=FlipDetector(), sample_count=SAMPLE_COUNT, subtitle_fallback="off")
    stats = perfstats.finish(handle)
    assert counts["opened"] == 1 and "band_fallback" not in stats


# ── Memory budget ───────────────────────────────────────────────────────────

# A bottom crop here is 40 x 160 x 3 = 19,200 bytes.
@pytest.mark.parametrize("budget_mb, stored", [
    ("0", False),            # explicitly off
    ("0.01", False),         # smaller than a single frame's crops
    ("0.05", False),         # two frames fit, the third overflows mid-pass
    ("1", True),
])
def test_budget_overflow_falls_back_to_the_second_pass(video, monkeypatch, budget_mb, stored):
    monkeypatch.setenv(memory.BAND_CROP_BUDGET_ENV, budget_mb)
    counts = _capture_counter(monkeypatch)
    handle = perfstats.start()
    old_detector, new_detector = FlipDetector(), FlipDetector()
    new = detect_clean_candidates(
        video, detector=new_detector, sample_count=SAMPLE_COUNT, subtitle_fallback="light")
    stats = perfstats.finish(handle)
    counts_new = dict(counts)
    old = original_detect_clean_candidates(
        video, detector=old_detector, sample_count=SAMPLE_COUNT, subtitle_fallback="light")
    assert_same_result(old, new)
    assert new_detector.log == old_detector.log
    assert counts_new["opened"] == (1 if stored else 2)
    assert stats["band_fallback"]["band_second_pass"] == (0 if stored else 1)
    assert stats["band_fallback"]["band_crops_stored"] == (SAMPLE_COUNT if stored else 0)


def test_force_mode_counts_both_bands_against_the_budget(video, monkeypatch):
    # Bottom 19,200 + top 160 x 24 x 3 = 11,520 bytes: 30,720 per frame.
    monkeypatch.setenv(memory.BAND_CROP_BUDGET_ENV, str(30720 * 9.5 / 1048576))
    counts = _capture_counter(monkeypatch)
    new_detector = FlipDetector()
    new = detect_clean_candidates(
        video, detector=new_detector, sample_count=SAMPLE_COUNT, subtitle_fallback="force")
    # 9.5 frames of budget for 10 samples: overflow on the last one, so the
    # file is read a second time.
    assert counts["opened"] == 2
    monkeypatch.undo()
    old_detector = FlipDetector()
    old = original_detect_clean_candidates(
        video, detector=old_detector, sample_count=SAMPLE_COUNT, subtitle_fallback="force")
    assert_same_result(old, new)
    assert new_detector.log == old_detector.log


def test_budget_helper(monkeypatch):
    monkeypatch.delenv(memory.BAND_CROP_BUDGET_ENV, raising=False)
    monkeypatch.setattr(memory, "available_memory_bytes", lambda: None)
    assert memory.band_crop_budget_bytes() == memory.UNKNOWN_BUDGET_BYTES
    monkeypatch.setattr(memory, "available_memory_bytes", lambda: 8 * 1024 ** 3)
    assert memory.band_crop_budget_bytes() == 512 * 1024 ** 2
    monkeypatch.setattr(memory, "available_memory_bytes", lambda: 64 * 1024 ** 3)
    assert memory.band_crop_budget_bytes() == memory.MAX_BUDGET_BYTES
    monkeypatch.setenv(memory.BAND_CROP_BUDGET_ENV, "12")
    assert memory.band_crop_budget_bytes() == 12 * 1024 ** 2
    monkeypatch.setenv(memory.BAND_CROP_BUDGET_ENV, "0")
    assert memory.band_crop_budget_bytes() == 0
    monkeypatch.setenv(memory.BAND_CROP_BUDGET_ENV, "garbage")
    assert memory.band_crop_budget_bytes() == memory.MAX_BUDGET_BYTES


def test_memory_probe_reports_something_here():
    total, available = memory.total_memory_bytes(), memory.available_memory_bytes()
    assert total is None or total > 0
    assert available is None or available > 0


# ── Cancellation ────────────────────────────────────────────────────────────

class Canceller:
    def __init__(self, after_calls=None):
        self.after_calls = after_calls
        self.calls = 0
        self.flag = False

    def __call__(self):
        self.calls += 1
        if self.flag or (self.after_calls is not None and self.calls > self.after_calls):
            raise ProcessingCancelledError("cancelled")


@pytest.mark.parametrize("after_calls", [3, 40, 59])
def test_cancelling_during_sampling_matches(video, after_calls):
    # The sampling pass makes 60 calls (50 skipped frames, 10 reads).
    old_detector, new_detector = FlipDetector(), FlipDetector()
    with pytest.raises(ProcessingCancelledError):
        original_detect_clean_candidates(
            video, detector=old_detector, sample_count=SAMPLE_COUNT,
            subtitle_fallback="light", check_cancelled=Canceller(after_calls))
    with pytest.raises(ProcessingCancelledError):
        detect_clean_candidates(
            video, detector=new_detector, sample_count=SAMPLE_COUNT,
            subtitle_fallback="light", check_cancelled=Canceller(after_calls))
    assert new_detector.log == old_detector.log


@pytest.mark.parametrize("crop_call", [0, 1, 7, 18])
def test_cancelling_inside_the_fallback_stops_at_the_same_detector_call(video, crop_call):
    """``crop_call`` = number of fallback detections after which cancel is set
    (0: cancelled the moment the sampling pass ends, before any fallback call).
    Cancellation is honoured at frame boundaries, and light mode runs two
    variants per frame, so an odd count finishes its frame first."""
    outcomes = []
    for function in (original_detect_clean_candidates, detect_clean_candidates):
        cancel = Canceller()
        detector = FlipDetector()
        if crop_call:
            detector.on_crop_call = lambda n, c=cancel: setattr(c, "flag", c.flag or n == crop_call)
        else:
            original_detect = detector.detect

            def detect(frame, c=cancel, o=original_detect):
                boxes = o(frame)
                if frame.shape[0] == H and int(frame[0, -1, 0]) == FRAMES - 1:
                    c.flag = True       # the last sampled frame has been detected
                return boxes
            detector.detect = detect
        with pytest.raises(ProcessingCancelledError):
            function(video, detector=detector, sample_count=SAMPLE_COUNT,
                     subtitle_fallback="light", check_cancelled=cancel)
        outcomes.append(detector.log)
    old_log, new_log = outcomes
    assert new_log == old_log
    assert sum(1 for e in new_log if e[0] == "crop") == crop_call + crop_call % 2


def test_all_samples_failing_raises_before_any_fallback(video):
    detector = FlipDetector(fail_frames={0, 6, 13, 19, 26, 32, 39, 45, 52, 59})
    with pytest.raises(RuntimeError, match="failed on all sampled frames"):
        detect_clean_candidates(
            video, detector=detector, sample_count=SAMPLE_COUNT, subtitle_fallback="light")
    assert all(entry[0] == "full" for entry in detector.log)
