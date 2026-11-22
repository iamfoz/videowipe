"""Feed the model a crop around the mask instead of the full-width band.

``VIDEOWIPE_MODEL_CROP=off|auto`` (default ``off``; see ``docs/mask-crop.md``).

Today the model sees each band of rows at full frame width, scaled to 640x120.
On a 1080p frame that is a 3x shrink of text that fills a fraction of the
width. With ``auto`` the model instead sees a 16:3 box fitted around the
mask, still scaled to 640x120 (the same model cost), so the text is shrunk
less. The rule that keeps this safe is that the model's input is never
enlarged: a box narrower than 640 source pixels is not used and the band stays.

Everything here is a pure function of the mask, so the box is decided per
segment (the unit the model and the prediction cache work in) and never
depends on what was processed before. The compositor reaches this module
through two small hooks in ``inpainters/sttn.py``.
"""
from __future__ import annotations

import math
import os
import warnings
from typing import NamedTuple

import numpy as np

MODEL_W, MODEL_H = 640, 120
ASPECT_W, ASPECT_H = 16, 3
# Context kept around the mask, as a fraction of the frame height (measured in
# the Q-9(b) experiment: ``docs/inpaint-quality-experiments.md``).
MARGIN_FRACTION = 0.04
ENV = "VIDEOWIPE_MODEL_CROP"


def crop_mode():
    """``"off"`` (default) or ``"auto"``; anything unrecognised is ``"off"``."""
    value = os.environ.get(ENV, "off").strip().lower()
    if value in ("", "off"):
        return "off"
    if value == "auto":
        return "auto"
    warnings.warn(f"{ENV}={value!r} is not off or auto; the full-width band is used",
                  RuntimeWarning, stacklevel=2)
    return "off"


class ModelCrop(NamedTuple):
    """The source pixels handed to the model: ``frame[top:bottom, left:right]``."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top


def fit_box(union, band, frame_width, frame_height):
    """The crop for a band, or ``None`` to keep the full-width band.

    ``union`` is ``(y0, y1, x0, x1)``, the box of non-zero alpha in the band
    over the whole segment; ``band`` is ``(top, bottom)``. The box is 16:3
    with a width that is a multiple of 16 (so the height is whole too), holds
    the alpha box plus a margin, lies inside the band's rows and the frame,
    and is at least ``MODEL_W`` wide: a smaller box would be enlarged on its
    way into the model, which was measured to lose up to 8.5 dB at 480p.
    """
    y0, y1, x0, x1 = union
    top, bottom = band
    margin = round(MARGIN_FRACTION * frame_height)
    wanted = max((x1 - x0) + 2 * margin, ((y1 - y0) + 2 * margin) * ASPECT_W / ASPECT_H)
    width = -(-math.ceil(wanted) // ASPECT_W) * ASPECT_W
    height = width * ASPECT_H // ASPECT_W
    if width < MODEL_W or width > frame_width or height > bottom - top:
        return None
    left = min(max(round((x0 + x1) / 2 - width / 2), 0), frame_width - width)
    up = min(max(round((y0 + y1) / 2 - height / 2), top), bottom - height)
    crop = ModelCrop(left, up, left + width, up + height)
    if not (crop.left <= x0 and x1 <= crop.right and crop.top <= y0 and y1 <= crop.bottom):
        return None
    if (crop.left, crop.top, crop.right, crop.bottom) == (0, top, frame_width, bottom):
        return None  # that is the band itself
    return crop


class SegmentCropper:
    """Chooses each active band's crop for a segment, from the masks in it."""

    def __init__(self, bands, frame_width, frame_height, job, mask_cache, facts_type):
        self.bands = tuple(bands)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_mask = job.frame_mask
        self.static_mask = job.mask
        self.mask_cache = mask_cache
        self.facts_type = facts_type
        self.segments = self.bands_cropped = self.bands_kept = 0

    def _facts(self, index):
        if self.frame_mask is None:
            return self.mask_cache.get(self.static_mask)
        alpha = np.asarray(self.frame_mask(index))
        if getattr(self.frame_mask, "_videowipe_stable_masks", False):
            return self.mask_cache.get(alpha)
        return self.facts_type(alpha.copy(), self.bands)

    def plan(self, start, end, active, known):
        """``{band: ModelCrop}`` for the bands of ``active`` that get a crop.

        Every frame of the segment counts, not only the emitted ones, so a
        trial range and a full run choose the same box (and hit the same
        cache entry). ``known`` maps frame index to facts already evaluated.
        """
        if not active:
            return {}
        self.segments += 1
        if self.frame_mask is None:
            every = [self._facts(start)]
        else:
            every = [known[i] if i in known else self._facts(i) for i in range(start, end)]
        if not all(facts.simple for facts in every):
            return {}
        crops = {}
        for k in sorted(active):
            boxes = [facts.boxes[k] for facts in every if facts.boxes[k] is not None]
            if not boxes:
                continue
            union = (min(b[0] for b in boxes), max(b[1] for b in boxes),
                     min(b[2] for b in boxes), max(b[3] for b in boxes))
            crop = fit_box(union, self.bands[k], self.frame_width, self.frame_height)
            if crop is None:
                self.bands_kept += 1
            else:
                crops[k] = crop
                self.bands_cropped += 1
        return crops


def model_input(image, from_h, to_h, crop):
    """The pixels the model reads: the band, or the crop inside it."""
    if crop is None:
        return image[from_h:to_h]
    return image[crop.top:crop.bottom, crop.left:crop.right]


class CroppedPrediction:
    """A 640x120 prediction that covers ``crop`` rather than a whole band.

    It stands in for the ndarray the compositor takes for a band, and says
    where the pixels go.
    """

    ndim = 3

    def __init__(self, pixels, crop, band_top):
        self.pixels = pixels
        self.crop = crop
        self.row = crop.top - band_top  # first row of the crop, inside the band

    @property
    def shape(self):
        return self.pixels.shape

    def region(self, upscaler, box):
        """The prediction upscaled over ``box`` (band-relative ``y0, y1, x0, x1``).

        Uses an upscaler for the crop's own width, so the windowed resize and
        its verification apply to a crop exactly as they do to a band.
        """
        by0, by1, x0, x1 = box
        crop = self.crop
        inside = (by0 >= self.row and by1 <= self.row + crop.height
                  and x0 >= crop.left and x1 <= crop.right)
        if not inside:
            raise RuntimeError("the mask reaches outside the model's crop")
        upscalers = upscaler.__dict__.setdefault("_crop_upscalers", {})
        inner = upscalers.get(crop.width)
        if inner is None:
            inner = upscalers[crop.width] = type(upscaler)(crop.width)
        return inner.crop(self.pixels, crop.height,
                          (by0 - self.row, by1 - self.row, x0 - crop.left, x1 - crop.left))


def tag(prediction, crop, band_top):
    """``prediction`` for a band, marked with its crop when it has one."""
    if crop is None:
        return prediction
    return CroppedPrediction(prediction, crop, band_top)


def legacy_resize(prediction, frame_width, band_height, resize):
    """The whole-band resize of a prediction (``resize`` is ``cv2.resize``).

    A cropped prediction is placed in an otherwise empty band: alpha is zero
    everywhere outside the crop, so those pixels are never used.
    """
    if not isinstance(prediction, CroppedPrediction):
        return resize(prediction, (frame_width, band_height))
    crop = prediction.crop
    band = np.zeros((band_height, frame_width, 3), np.uint8)
    band[prediction.row:prediction.row + crop.height, crop.left:crop.right] = resize(
        prediction.pixels, (crop.width, crop.height))
    return band
