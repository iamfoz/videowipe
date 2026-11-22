"""STTN inpainter — the default built-in model.

The segment loop and its helpers were moved verbatim from ``tasks/detext.py``;
only their location changed. STTN's three-stage backend
(``encode``/``transform``/``decode``) is a private implementation detail of
this inpainter, not a public contract.
"""
from __future__ import annotations

import concurrent.futures
import functools
import hashlib
import math
from pathlib import Path
import os
import subprocess
import tempfile
import threading
import time

import cv2
import numpy as np

from videowipe.backends import load_backend
from videowipe.decode import CAPTURE_TYPES, open_capture
from videowipe.inpainters.base import InpaintJob, InpaintOutcome

_VIDEO_CAPTURE_TYPE = CAPTURE_TYPES


def get_ref_index(neighbor_ids, length, ref_length):
    """Select reference frames at regular intervals, excluding neighbors."""
    ref_index = []
    for i in range(0, length, ref_length):
        if i not in neighbor_ids:
            ref_index.append(i)
    return ref_index


def blend_frames(comp_frames, pred_img, neighbor_ids, counts):
    """Blend predicted frames into the composite buffer.

    Pure-numpy replacement for the former @njit version.
    Accumulate repeated frame IDs; normalize once after all windows.
    """
    for i, idx in enumerate(neighbor_ids):
        img = pred_img[i].astype(np.float32)
        comp_frames[idx] += img
        counts[idx] += 1


def _blend_frame_regions(frame_ori, comp_frames, modes, mask):
    """Blend prepared model crops without converting the whole frame to float."""
    active = [
        (mode, comp)
        for mode, comp in zip(modes, comp_frames)
        if comp is not None
    ]
    frame = frame_ori.copy()
    if not active:
        return frame

    work_from = min(mode[0] for mode, _comp in active)
    work_to = max(mode[1] for mode, _comp in active)
    work = frame_ori[work_from:work_to].astype(np.float32)
    for (from_h, to_h), comp in active:
        mask_area = mask[from_h:to_h].astype(np.float32)
        relative = slice(from_h - work_from, to_h - work_from)
        work[relative] = (
            mask_area * comp
            + (1.0 - mask_area) * work[relative]
        )
    frame[work_from:work_to] = np.clip(work, 0, 255).astype(np.uint8)
    return frame


# Compositing only the masked box (P2-6).
#
# ``_blend_frame_regions`` above converts and blends whole bands at full frame
# width. The helpers below produce the same bytes while touching only the
# bounding box of the non-zero alpha in each band:
#
# * With alpha exactly 0.0 the blend is ``0*comp + 1*orig``. ``comp`` is a
#   float32 holding an integer in 0..255, so ``0*comp`` is +-0.0 and
#   ``1.0 * orig`` is exact; the sum, the clip and the uint8 cast return
#   ``orig`` unchanged. Pixels outside a box therefore equal the original
#   frame exactly, as they do today.
# * Inside a box the same float32 operations run in the same order, with the
#   same operands (alpha converted to float32, ``1.0 - alpha`` in float32), so
#   every value is identical. Bands whose boxes overlap share one float work
#   buffer and are blended in band order, because a later band reads the
#   unrounded result of an earlier one.

_REAL_VIDEO_CAPTURE = cv2.VideoCapture
# ``_blend_frame_regions`` stays the module's seam for observing or replacing
# composition: when something has substituted it, frames go through it exactly
# as before instead of through the box path.
_WHOLE_BAND_BLEND = _blend_frame_regions


def _available_memory_bytes():
    """Memory the machine reports as available, or 0 when it cannot tell."""
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES")) // 2
    except (AttributeError, ValueError, OSError):
        return 0


def _blend_cache_budget():
    """Bytes of per-mask float data worth keeping; scales with free memory.

    ``VIDEOWIPE_BLEND_CACHE_MB`` overrides it (``0`` keeps nothing).
    """
    override = os.environ.get("VIDEOWIPE_BLEND_CACHE_MB")
    if override is not None:
        try:
            return max(0, int(float(override) * 1024 * 1024))
        except ValueError:
            pass
    available = _available_memory_bytes()
    if not available:
        return 32 * 1024 * 1024
    return min(available // 64, 512 * 1024 * 1024)


def _alpha_box(alpha, top, bottom):
    """``(y0, y1, x0, x1)`` of the non-zero alpha in rows top:bottom, or None."""
    band = alpha[top:bottom]
    if band.size == 0:
        return None
    nonzero = band != 0
    rows = np.flatnonzero(nonzero.any(axis=(1, 2)))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(nonzero.any(axis=(0, 2)))
    return (top + int(rows[0]), top + int(rows[-1]) + 1,
            int(cols[0]), int(cols[-1]) + 1)


class _MaskFacts:
    """What the compositor needs to know about one alpha array.

    Everything here is a pure function of the array's contents, so it is
    computed once and reused for as long as the same array object recurs.
    The object keeps a reference to the array: while it exists the array
    cannot be freed and its id cannot be handed to a different array.
    """

    def __init__(self, mask, modes, owner=None):
        alpha = np.asarray(mask)
        if alpha.ndim == 2:
            alpha = alpha[:, :, None]
        self.source = mask
        self.alpha = alpha
        # Only (H, W, C) alpha has the box fast path; anything else goes
        # through the unchanged whole-band code.
        self.simple = alpha.ndim == 3
        if self.simple:
            self.boxes = tuple(_alpha_box(alpha, top, bottom) for top, bottom in modes)
        else:
            self.boxes = tuple(
                (top, bottom, 0, 0) if np.any(alpha[top:bottom] != 0) else None
                for top, bottom in modes
            )
        self.active = tuple(k for k, box in enumerate(self.boxes) if box is not None)
        self._derived = {}
        self.derived_bytes = 0
        self._owner = owner

    def float_alpha(self, k):
        """``(alpha, 1 - alpha)`` as float32 over band k's box (read-only)."""
        pair = self._derived.get(k)
        if pair is not None:
            return pair
        y0, y1, x0, x1 = self.boxes[k]
        alpha = self.alpha[y0:y1, x0:x1].astype(np.float32)
        remainder = 1.0 - alpha
        alpha.setflags(write=False)
        remainder.setflags(write=False)
        pair = (alpha, remainder)
        owner = self._owner
        if owner is not None:
            with owner.lock:  # the pipeline overlap composes on another thread
                if owner.admit(self, alpha.nbytes + remainder.nbytes):
                    self._derived[k] = pair
                    self.derived_bytes += alpha.nbytes + remainder.nbytes
        return pair


class _MaskCache:
    """Small bounded LRU of :class:`_MaskFacts` keyed by array identity.

    Entries hold their array, so an id cannot be reused by a different array
    while it is cached, and lookups still confirm identity. Derived float data
    is bounded in bytes as well as in entries.
    """

    def __init__(self, modes, max_entries=8, budget=None):
        self.modes = tuple(modes)
        self.max_entries = max(1, int(max_entries))
        self.budget = _blend_cache_budget() if budget is None else int(budget)
        self.derived_bytes = 0
        self._entries = {}
        self.lock = threading.RLock()

    def __len__(self):
        return len(self._entries)

    def get(self, mask):
        with self.lock:
            return self._get(mask)

    def _get(self, mask):
        entry = self._entries.get(id(mask))
        if entry is not None and entry.source is mask:
            self._entries[id(mask)] = self._entries.pop(id(mask))  # most recent
            return entry
        entry = _MaskFacts(mask, self.modes, self)
        self._entries[id(mask)] = entry
        while len(self._entries) > self.max_entries:
            self._evict_oldest(keep=entry)
        return entry

    def admit(self, facts, nbytes):
        with self.lock:
            return self._admit(facts, nbytes)

    def _admit(self, facts, nbytes):
        while self.derived_bytes + nbytes > self.budget:
            if not self._evict_oldest(keep=facts):
                return False
        self.derived_bytes += nbytes
        return True

    def _evict_oldest(self, keep):
        for key, entry in self._entries.items():
            if entry is not keep:
                del self._entries[key]
                self.derived_bytes -= entry.derived_bytes
                return True
        return False


def _boxes_overlap(a, b):
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def _overlap_groups(items):
    """Group ``(band, box)`` items whose boxes overlap, each in band order."""
    groups = []
    for item in items:
        group = [item]
        for other in [g for g in groups if any(_boxes_overlap(item[1], o[1]) for o in g)]:
            groups.remove(other)
            group.extend(other)
        group.sort(key=lambda entry: entry[0])
        groups.append(group)
    return groups


@functools.lru_cache(maxsize=256)
def _axis_weights_are_stable(src, dst):
    """True when INTER_LINEAR's 11-bit weights cannot depend on window position.

    OpenCV derives each weight from ``fx = float((d + 0.5) * src / dst - 0.5)``
    and rounds ``fx * 2048``. The float32 conversion makes ``fx`` carry a small
    error that depends on how large ``d`` is, which can only change the rounded
    weight if ``fx * 2048`` lies within that error of a rounding boundary. For
    ``src / dst = p / q`` the possible values are ``1024 * r / q`` for integer
    ``r``; require every one of them to stay clear of the boundary by more than
    twice the worst-case error.
    """
    g = math.gcd(src, dst)
    q = dst // g
    error = 2048.0 * 2.0 ** (math.floor(math.log2(max(src, 2))) - 24)
    tolerance = 2.0 * error + 0.02
    return all(
        abs((1024 * r % q) / q - 0.5) > tolerance for r in range(2 * q)
    )


class _BandUpscaler:
    """Upscale a band's prediction and return only the box that is needed.

    ``cv2.resize`` of the prediction to (frame width x band height) is what
    the whole-band code does. For INTER_LINEAR, destination pixel ``d`` samples
    source position ``(d + 0.5) * s - 0.5`` with ``s = src / dst``. A
    destination window ``[c, d)`` therefore reproduces the same samples from
    the source window ``[c*s, d*s)`` at the same scale, provided ``c`` and ``d``
    are multiples of the reduced fraction's denominator (so that ``c*s`` is a
    whole pixel) and one extra unit of margin on each side supplies the
    neighbours the border pixels read. That argument is exact only if the
    fixed-point weights do not depend on position (see
    ``_axis_weights_are_stable``), and it assumes nothing about this OpenCV
    build, so each window is also compared on this machine against the full
    resize over a battery of images before it is used. Any window that is not
    provably and demonstrably identical falls back to resizing the whole band
    and cropping, which is exact by construction.
    """

    PROBES = 6

    def __init__(self, ori_w):
        self.ori_w = ori_w
        self._windows = {}
        self._checked = {}
        # ``VIDEOWIPE_BLEND_WINDOW_RESIZE=off`` always resizes the whole band
        # (exact by construction); anything else lets verified windows be used.
        self._windowed = os.environ.get(
            "VIDEOWIPE_BLEND_WINDOW_RESIZE", "auto").strip().lower() != "off"

    @staticmethod
    def _axis(lo, hi, src, dst):
        """``(d0, d1, s0, s1, core0, core1)`` for rows or columns ``[lo, hi)``."""
        full = (0, dst, 0, src, 0, dst)
        if src <= 0 or dst <= 0 or not _axis_weights_are_stable(src, dst):
            return full
        g = math.gcd(src, dst)
        p, q = src // g, dst // g           # source / destination = p / q
        core0 = lo // q * q
        core1 = min(dst, -(-hi // q) * q)
        c = max(0, core0 - q)
        d = min(dst, core1 + q)
        return c, d, c // q * p, d // q * p, core0, core1

    def _window(self, pred_shape, band_h, box):
        key = (pred_shape, band_h, box)
        try:
            return self._windows[key]
        except KeyError:
            pass
        by0, by1, x0, x1 = box
        ys = self._axis(by0, by1, pred_shape[0], band_h)
        xs = self._axis(x0, x1, pred_shape[1], self.ori_w)
        window = None
        if ys[:2] != (0, band_h) or xs[:2] != (0, self.ori_w):
            candidate = ys[:4] + xs[:4] + (ys[4], ys[5], xs[4], xs[5])
            if self._verified(pred_shape, band_h, candidate):
                window = candidate
        if len(self._windows) > 256:
            self._windows.clear()
        self._windows[key] = window
        return window

    def _verified(self, pred_shape, band_h, window):
        """Compare the window with the full resize on several images."""
        key = (pred_shape, band_h, window)
        known = self._checked.get(key)
        if known is not None:
            return known
        dy0, dy1, sy0, sy1, dx0, dx1, sx0, sx1, cy0, cy1, cx0, cx1 = window
        generator = np.random.RandomState(20240607)
        rows, cols = np.indices(pred_shape[:2])
        probes = [
            generator.randint(0, 256, size=pred_shape, dtype=np.uint8)
            for _ in range(self.PROBES - 2)
        ]
        probes.append(np.where((cols % 2)[:, :, None] == 0, 255, 0).astype(np.uint8)
                      * np.ones(pred_shape, np.uint8))
        probes.append(np.where(((rows + cols) % 2)[:, :, None] == 0, 255, 0).astype(np.uint8)
                      * np.ones(pred_shape, np.uint8))
        ok = True
        for probe in probes:
            full = cv2.resize(probe, (self.ori_w, band_h))[cy0:cy1, cx0:cx1]
            part = cv2.resize(
                np.ascontiguousarray(probe[sy0:sy1, sx0:sx1]), (dx1 - dx0, dy1 - dy0),
            )[cy0 - dy0:cy1 - dy0, cx0 - dx0:cx1 - dx0]
            if not np.array_equal(full, part):
                ok = False
                break
        if len(self._checked) > 256:
            self._checked.clear()
        self._checked[key] = ok
        return ok

    def crop(self, pred, band_h, box):
        """Resized prediction over ``box`` (band-relative), channels reversed
        and converted to float32 exactly as ``cvtColor(BGR2RGB)`` +
        ``astype(float32)`` would."""
        by0, by1, x0, x1 = box
        window = None
        if self._windowed and pred.dtype == np.uint8 and pred.ndim == 3:
            window = self._window(pred.shape, band_h, box)
        if window is None:
            resized = cv2.resize(pred, (self.ori_w, band_h))
            region = resized[by0:by1, x0:x1]
        else:
            dy0, dy1, sy0, sy1, dx0, dx1, sx0, sx1 = window[:8]
            resized = cv2.resize(
                np.ascontiguousarray(pred[sy0:sy1, sx0:sx1]),
                (dx1 - dx0, dy1 - dy0),
            )
            region = resized[by0 - dy0:by1 - dy0, x0 - dx0:x1 - dx0]
        if region.dtype != np.uint8:
            region = region.astype(np.uint8)
        return region[:, :, ::-1].astype(np.float32)


def _blend_patches(frame_ori, predictions, modes, facts, upscaler):
    """Blend each band's prediction inside its box; return ``(box, pixels)`` patches.

    ``predictions`` maps a band index to its (unscaled) prediction. Nothing is
    written to ``frame_ori``; the patches hold final uint8 pixels.
    """
    items = [(k, facts.boxes[k]) for k in facts.active if k in predictions]
    patches = []
    for group in _overlap_groups(items):
        uy0 = min(box[0] for _k, box in group)
        uy1 = max(box[1] for _k, box in group)
        ux0 = min(box[2] for _k, box in group)
        ux1 = max(box[3] for _k, box in group)
        work = frame_ori[uy0:uy1, ux0:ux1].astype(np.float32)
        views = []
        for k, box in group:
            y0, y1, x0, x1 = box
            top, bottom = modes[k]
            comp = upscaler.crop(predictions[k], bottom - top, (y0 - top, y1 - top, x0, x1))
            alpha, remainder = facts.float_alpha(k)
            region = work[y0 - uy0:y1 - uy0, x0 - ux0:x1 - ux0]
            region[...] = alpha * comp + remainder * region
            views.append((box, region))
        for box, region in views:
            patches.append((box, np.clip(region, 0, 255).astype(np.uint8)))
    return patches


def _compose_legacy(frame_ori, predictions, modes, mask, ori_w, dual):
    """The whole-band composition, for inputs the box path does not cover."""
    frame_comps = []
    for k in range(len(modes)):
        if k in predictions:
            mode_height = modes[k][1] - modes[k][0]
            comp = cv2.resize(predictions[k], (ori_w, mode_height))
            comp = cv2.cvtColor(
                np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB
            ).astype(np.float32)
            frame_comps.append(comp)
        else:
            frame_comps.append(None)
    frame = _blend_frame_regions(frame_ori, frame_comps, modes, mask)
    if dual:
        frame = np.vstack([frame_ori, frame])
    return frame


def _compose_frame(frame_ori, predictions, modes, facts, upscaler, ori_w, *,
                   dual, own_frame, legacy_mask):
    """Return the frame to write, byte-identical to the whole-band composition.

    ``own_frame`` says ``frame_ori`` may be edited in place (non-dual only):
    nothing else refers to it. Otherwise it is copied before it is changed,
    and an unchanged frame is returned as is.
    """
    if not (
        _blend_frame_regions is _WHOLE_BAND_BLEND
        and facts.simple
        and np.ndim(legacy_mask) == 3
        and frame_ori.dtype == np.uint8
        and frame_ori.ndim == 3
        and frame_ori.shape[2] == 3
        and frame_ori.shape[1] == ori_w
        and frame_ori.shape[:2] == facts.alpha.shape[:2]
        and all(p.ndim == 3 and p.shape[2] == 3 for p in predictions.values())
    ):
        return _compose_legacy(frame_ori, predictions, modes, legacy_mask, ori_w, dual)
    patches = _blend_patches(frame_ori, predictions, modes, facts, upscaler)
    if dual:
        height = frame_ori.shape[0]
        frame = np.empty((2 * height,) + frame_ori.shape[1:], dtype=np.uint8)
        frame[:height] = frame_ori
        frame[height:] = frame_ori
        target = frame[height:]
    elif not patches:
        return frame_ori
    else:
        frame = frame_ori if own_frame and frame_ori.flags.writeable else frame_ori.copy()
        target = frame
    for (y0, y1, x0, x1), pixels in patches:
        target[y0:y1, x0:x1] = pixels
    return frame


def _write_frame(stream, frame):
    """Write a frame's bytes without an intermediate ``tobytes()`` copy."""
    if frame.flags.c_contiguous and frame.size:
        stream.write(memoryview(frame).cast("B"))
    else:
        stream.write(frame.tobytes())


def _skip_windows_enabled():
    """Whether windows that cannot affect an emitted pixel are skipped.

    ``VIDEOWIPE_SKIP_WINDOWS=off`` computes every window, as before; anything
    else (``auto``, ``on``) skips them.
    """
    return os.environ.get("VIDEOWIPE_SKIP_WINDOWS", "auto").strip().lower() != "off"


def _window_layout(video_length, neighbor_stride):
    """The frame ids covered by each model window of a full run, in order."""
    return [
        list(range(
            max(0, f - neighbor_stride),
            min(video_length, f + neighbor_stride + 1),
        ))
        for f in range(0, video_length, neighbor_stride)
    ]


def _windows_to_run(layout, needed):
    """The windows of ``layout`` that touch a needed frame (all when ``None``)."""
    if needed is None:
        return layout
    wanted = set(needed)
    return [ids for ids in layout if not wanted.isdisjoint(ids)]


def _count_windows(video_length, neighbor_stride, needed=None):
    """``(run, total)`` windows for ``_process_segment`` with the same arguments."""
    layout = _window_layout(video_length, neighbor_stride)
    return len(_windows_to_run(layout, needed)), len(layout)


def _process_segment(frames, backend, w, h, ref_length, neighbor_stride, needed=None):
    """Inpaint a batch of frames through the model (backend-agnostic).

    ``needed`` is a collection of local frame indices whose predictions the
    caller will use; ``None`` means all of them. Every frame is still encoded
    (reference frames come from the whole segment), but a window is skipped
    when none of the frames it predicts is needed. A frame's prediction is the
    mean over every window that covers it, so a frame that some covering window
    skipped would get a different mean from a full run: it is returned as
    ``None``. Every frame returned is therefore bit-identical to the same frame
    of a full run, and every needed frame is among them.
    """
    video_length = len(frames)
    feats = backend.encode(backend.preprocess(frames))

    comp_frames_np = np.zeros((video_length, h, w, 3), dtype=np.float32)
    counts = np.zeros((video_length,), dtype=np.int32)

    layout = _window_layout(video_length, neighbor_stride)
    # Windows covering each frame in a full run.
    coverage = np.zeros((video_length,), dtype=np.int32)
    for neighbor_ids in layout:
        coverage[neighbor_ids] += 1

    for neighbor_ids in _windows_to_run(layout, needed):
        ref_ids = get_ref_index(neighbor_ids, video_length, ref_length)

        ids = neighbor_ids + ref_ids
        selected = feats[ids]
        transformed = backend.transform(selected)
        decoded = backend.decode(transformed[: len(neighbor_ids)])

        blend_frames(comp_frames_np, decoded, np.array(neighbor_ids), counts)

    comp_frames = []
    for idx in range(video_length):
        if counts[idx] and counts[idx] == coverage[idx]:
            comp_frames.append(np.clip(comp_frames_np[idx] / counts[idx], 0, 255).astype(np.uint8))
        else:
            comp_frames.append(None)
    return comp_frames


def get_inpaint_mode(H, h, mask):
    """Determine inpainting segments based on mask position."""
    mode = []
    frontier = H
    while frontier > 0:
        base_start = max(0, frontier - h)
        unprocessed = mask[base_start:frontier, :]
        if not np.all(unprocessed == 0) and np.sum(unprocessed) > 10:
            from_H, to_H = base_start, frontier
            if frontier != H and base_start != 0:
                move = 0
                while frontier + move < H and not np.all(
                    mask[frontier + move, :] == 0
                ):
                    move += 1
                # Shift only when it neither runs off the frame nor drops mask
                # pixels from the still-unprocessed top edge.
                if (
                    0 < move < h
                    and frontier + move < H
                    and np.all(mask[base_start:base_start + move, :] == 0)
                ):
                    to_H += move
                    from_H += move
            mode.append((from_H, to_H))
        frontier = base_start
    return mode


class STTNInpainter:
    """Built-in STTN inpainter.

    Loads weights via :mod:`videowipe.backends` and runs the segment loop over
    the whole video. The loaded :class:`~videowipe.backends.InpaintBackend` is
    exposed as :attr:`backend` so callers (e.g. the engine benchmark) can read
    its class name.
    """

    name = "sttn"

    def __init__(self, ref_length: int = 5, neighbor_stride: int = 5):
        self.ref_length = ref_length
        self.neighbor_stride = neighbor_stride
        self.backend = None

    def load(self, weight_path: str, device: str = "auto") -> None:
        self.backend = load_backend(weight_path, device=device)
        print(
            f"Loaded weight: {weight_path} "
            f"(backend: {type(self.backend).__name__})"
        )

    def inpaint(self, job: InpaintJob) -> InpaintOutcome:
        if self.backend is None:
            raise RuntimeError(
                "STTNInpainter.load() must be called before inpaint()"
            )
        if job.reader is None:
            raise ValueError(
                "STTNInpainter requires job.reader (a cv2.VideoCapture)"
            )

        w, h = 640, 120
        video_length = job.frame_count
        ori_w, ori_h = job.width, job.height
        fps = job.fps
        gap = job.gap
        first, last = job.trial_range or (0, video_length)
        if not 0 <= first < last <= video_length:
            raise ValueError("trial_range exceeds the source video")
        first_segment = first // gap
        last_segment = (last + gap - 1) // gap
        context_start = first_segment * gap
        reader = job.reader

        video_name = (
            os.path.splitext(os.path.basename(job.video_path))[0]
            if job.video_path
            else "output"
        )
        video_out_path = os.path.join(
            job.output_dir, f"{video_name}_{job.output_suffix}{'_trial' if job.trial_range else ''}.mp4"
        )
        out_w = ori_w
        out_h = ori_h * 2 if job.dual else ori_h

        split_h = int(ori_w * 3 / 16)
        mode = get_inpaint_mode(ori_h, split_h, job.mask)
        if not mode:
            raise ValueError(
                "Mask has no inpaintable regions. "
                "The auto-detected mask is empty — the text detector may have "
                "failed to find subtitles. Try providing a mask manually with -m."
            )

        # What to keep from the source: its frame timestamps (variable-rate sources,
        # videowipe/timing.py) and its audio, metadata and subtitles (full runs,
        # videowipe/tracks.py). VIDEOWIPE_SOURCE_TIMING=off and
        # VIDEOWIPE_OUTPUT_TRACKS=off give upstream's command.
        from videowipe import pipeline_overlap
        from videowipe.fidelity import plan_encode
        encode_plan = plan_encode(job)

        cache = None
        if job.prediction_cache_dir is not None:
            from videowipe.inpainters.prediction_cache import PredictionCache
            from videowipe.plan import _sha256_file
            runtime = self.backend.benchmark_metadata()
            implementation = hashlib.sha256()
            package = Path(__file__).resolve().parents[1]
            for name in ('inpainters/sttn.py', 'backends.py', 'models/sttn.py', 'core/spectral_norm.py'):
                implementation.update((package / name).read_bytes())
            identity = dict(runtime, source_sha256=_sha256_file(job.video_path),
                            source_size=[ori_w, ori_h], input_size=[w, h],
                            backend=type(self.backend).__name__,
                            precision=getattr(self.backend, 'precision_label', lambda: 'float16-autocast' if str(runtime['device']).startswith('cuda') else 'float32')(),
                            ref_length=self.ref_length, neighbor_stride=self.neighbor_stride,
                            implementation_sha256=implementation.hexdigest())
            cache = PredictionCache(job.prediction_cache_dir, identity)
        prediction_s = composition_s = 0.0
        output_reader = None
        overlap = pipeline_overlap.OFF

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error", "-nostats",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
        ]
        # Attach the original video as a second input so its audio stream survives
        # (ffmpeg does not carry global metadata or subtitles over from it by
        # itself; videowipe/tracks.py adds them). The trailing "?" on "1:a?" makes
        # ffmpeg tolerate videos with no audio track instead of erroring out.
        if job.video_path:
            if job.trial_range is not None:
                ffmpeg_cmd += ["-ss", str(first / fps)]
            ffmpeg_cmd += ["-i", job.video_path]
        ffmpeg_cmd += [
            "-map", "0:v",
        ]
        if job.video_path:
            ffmpeg_cmd += ["-map", "1:a?"]
        ffmpeg_cmd += [
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p",
        ]
        if job.video_path:
            ffmpeg_cmd += ["-c:a", "aac"]
        if job.trial_range is not None:
            ffmpeg_cmd += ["-t", str((last - first) / fps)]
        ffmpeg_cmd += [
            "-movflags", "+faststart",
            video_out_path,
        ]
        if job.video_path and job.trial_range is None:
            # Optional H.265 export, encoded from the same frames (fork addition).
            from videowipe.export import hevc_output_args
            ffmpeg_cmd += hevc_output_args(job.video_path, video_out_path, ori_h)
        stderr_file = tempfile.TemporaryFile()
        pipe = encode_plan.popen(ffmpeg_cmd, stderr_file)
        stdin_closed = False
        processing_failed = False
        try:
            if job.video_path and isinstance(reader, _VIDEO_CAPTURE_TYPE):
                candidate_reader = open_capture(job.video_path, like=reader)
                if candidate_reader.isOpened():
                    output_reader = candidate_reader
                else:
                    candidate_reader.release()
            clock = time.perf_counter
            prefix_skip_s = 0.0
            if context_start:
                started_prefix = clock()
                from videowipe.video import skip_frames, skip_frames_concurrently

                advanced = 0

                def check_prefix():
                    nonlocal advanced
                    # Phase callbacks also carry cooperative cancellation.
                    if advanced % gap == 0:
                        if job.phase_progress is not None:
                            job.phase_progress("decode", 0, last - first, None)
                        elif job.progress is not None:
                            job.progress(0, last - first)
                    advanced += 1

                if output_reader is not None:
                    skip_frames_concurrently(
                        (reader, output_reader), context_start, check_prefix
                    )
                else:
                    skip_frames(reader, context_start, check_prefix)
                prefix_skip_s = clock() - started_prefix

            # Phase timers (P0-1). Each accumulates one coarse step so that
            # the parts of prediction_s and composition_s can be told apart;
            # they only measure and never change what is decoded or written.
            decode_infer_s = resize_s = cache_read_s = model_s = cache_write_s = 0.0
            decode_output_s = blend_s = pipe_write_s = mask_eval_s = 0.0
            frames_read = frames_grabbed = output_frames_read = frames_written = 0
            segments = segments_cached = segments_inferred = bands_inferred = windows = 0
            windows_skipped = 0
            skip_windows = _skip_windows_enabled()

            # Per-mask facts (bounding boxes, float alpha) are derived once per
            # distinct mask array and kept in a small bounded cache.
            mask_cache = _MaskCache(mode)
            upscaler = _BandUpscaler(ori_w)
            static_facts = None
            # A frame this loop decoded itself can be edited in place: every
            # read() of a real capture returns a new array and nothing else
            # keeps the frame. Anything else is copied before it is changed.
            own_frames = (
                output_reader is not None
                and type(output_reader) is _REAL_VIDEO_CAPTURE
                and not job.dual
            )

            # Optional overlap of decode, compose and write with the model
            # (VIDEOWIPE_PIPELINE_OVERLAP); ``off`` returns these unchanged.
            overlap = pipeline_overlap.attach(
                job, reader, output_reader, _compose_frame, _write_frame,
                frame_bytes=ori_w * ori_h * 3, available_bytes=_available_memory_bytes(),
                capture_type=_REAL_VIDEO_CAPTURE)
            reader, output_reader = overlap.reader, overlap.output_reader

            # Backend instances are shared runtime objects; keep inference
            # serialized until the backend declares a thread-safety contract.
            t_inpaint_start = time.monotonic()
            output_decoded = context_start
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                for i in range(first_segment, last_segment):
                    start_f = i * gap
                    end_f = min((i + 1) * gap, video_length)
                    print(f"Processing frames {start_f + 1}-{end_f}/{video_length}")

                    # Cache only emitted masks. Keep all context frames whenever
                    # any emitted pixel in a band needs a prediction.
                    segments += 1
                    segment_masks = {}
                    active_modes = set(range(len(mode)))
                    segment_len = end_f - start_f
                    emitted = [
                        index - start_f
                        for index in range(max(start_f, first), min(end_f, last))
                    ]
                    # Local frames whose prediction the compositor will use, per
                    # band. A static mask needs every emitted frame in every band;
                    # a per-frame mask needs a frame only in the bands it touches.
                    wanted = {k: emitted for k in active_modes}
                    if job.frame_mask is not None:
                        started_mask = clock()
                        active_modes = set()
                        wanted = {}
                        for index in range(max(start_f, first), min(end_f, last)):
                            alpha = np.asarray(job.frame_mask(index))
                            # Only the plan producer guarantees stable storage.
                            # Even readonly views from custom callbacks can alias
                            # a buffer that the next callback invocation changes.
                            if getattr(job.frame_mask, "_videowipe_stable_masks", False):
                                facts = mask_cache.get(alpha)
                            else:
                                facts = _MaskFacts(alpha.copy(), mode)
                            segment_masks[index] = facts
                            active_modes.update(facts.active)
                            for k in facts.active:
                                wanted.setdefault(k, []).append(index - start_f)
                        mask_eval_s += clock() - started_mask
                    if not skip_windows:
                        wanted = {k: range(segment_len) for k in active_modes}

                    prediction_started = time.monotonic()
                    comps = {}
                    keys = {}
                    if cache is not None:
                        started_cache = clock()
                        for k in active_modes:
                            keys[k] = cache.key(start_f, end_f, *mode[k])
                            cached = cache.read(keys[k], segment_len, need=wanted[k])
                            if cached is not None:
                                comps[k] = cached
                        cache_read_s += clock() - started_cache
                    # Bands to compute, and which of their frames are still
                    # missing. A cached entry may hold only some frames, and it
                    # satisfies a band only when it holds every frame needed.
                    to_predict = {}
                    for k in active_modes:
                        entry = comps.get(k)
                        todo = [j for j in wanted[k] if entry is None or entry[j] is None]
                        if todo:
                            # ``None``: every frame (the encoder sees them all anyway).
                            to_predict[k] = (
                                todo if skip_windows and len(todo) < segment_len else None
                            )
                    missing_modes = set(to_predict)
                    segments_cached += bool(comps.keys() - missing_modes)
                    frames_hr = [] if output_reader is None else None
                    frames = {k: [] for k in range(len(mode))}

                    for local_index in range(end_f - start_f):
                        started_decode = clock()
                        if (output_reader is not None and not missing_modes
                                and hasattr(reader, "grab")):
                            # No band needs these pixels (inactive segment or
                            # all cache hits); advance the decoder only.
                            success, image = reader.grab(), None
                            frames_grabbed += 1
                        else:
                            success, image = reader.read()
                            frames_read += 1
                        started_resize = clock()
                        decode_infer_s += started_resize - started_decode
                        if not success:
                            raise ValueError(
                                f"STTN inference decoded {start_f + local_index} "
                                f"frames; expected {video_length}"
                            )
                        if frames_hr is not None:
                            frames_hr.append(image)
                        for k in missing_modes:
                            from_h, to_h = mode[k]
                            frames[k].append(cv2.resize(
                                image[from_h:to_h], (w, h),
                                interpolation=cv2.INTER_LINEAR,
                            ).astype(np.float32))
                        resize_s += clock() - started_resize

                    started_model = clock()
                    futures = {
                        k: executor.submit(
                            _process_segment,
                            frames[k], self.backend, w, h,
                            self.ref_length, self.neighbor_stride, to_predict[k],
                        )
                        for k in range(len(mode))
                        if frames[k]
                    }
                    for k, future in futures.items():
                        predicted = future.result()
                        entry = comps.get(k)
                        if entry is not None:
                            # Cached frames and new ones are both full-run values,
                            # so keeping the cached ones and adding the new is exact.
                            predicted = [
                                old if old is not None else new
                                for old, new in zip(entry, predicted)
                            ]
                        comps[k] = predicted
                        if cache is not None:
                            started_write = clock()
                            model_s += started_write - started_model
                            cache.write(keys[k], comps[k])
                            started_model = clock()
                            cache_write_s += started_model - started_write
                    model_s += clock() - started_model
                    segments_inferred += bool(futures)
                    bands_inferred += len(futures)
                    for k in futures:
                        run, total = _count_windows(
                            len(frames[k]), self.neighbor_stride, to_predict[k])
                        windows += run
                        windows_skipped += total - run
                    prediction_s += time.monotonic() - prediction_started
                    if cache is not None and job.phase_progress is not None:
                        job.phase_progress('predict', min(end_f, last) - first, last - first,
                                           f'hits={cache.hits};misses={cache.misses}')
                    composition_started = time.monotonic()

                    for j in range(end_f - start_f):
                        started_output = clock()
                        if output_reader is not None:
                            success, frame_ori = output_reader.read()
                            if not success:
                                raise ValueError(
                                    f"STTN output decoded {output_decoded} frames; "
                                    f"expected {video_length}"
                                )
                            output_decoded += 1
                            output_frames_read += 1
                        else:
                            frame_ori = frames_hr[j]
                        started_blend = clock()
                        decode_output_s += started_blend - started_output
                        if not first <= start_f + j < last:
                            continue
                        # Per-frame temporal mask (global index start_f + j) when a
                        # WipePlan supplies one; else the static whole-video mask.
                        if job.frame_mask is not None:
                            facts = segment_masks[start_f + j]
                        else:
                            if static_facts is None:
                                static_facts = mask_cache.get(job.mask)
                            facts = static_facts
                        predictions = {}
                        for k in range(len(mode)):
                            if comps.get(k) and j < len(comps[k]) and comps[k][j] is not None:
                                predictions[k] = comps[k][j]
                        # Soft alpha blend: mask is float32 in [0,1] when
                        # feather_radius > 0, else uint8 in {0,1}.
                        frame = overlap.compose(
                            frame_ori, predictions, mode, facts, upscaler, ori_w,
                            dual=job.dual, own_frame=own_frames,
                            legacy_mask=facts.alpha if job.frame_mask is not None else job.mask,
                        )
                        started_write = clock()
                        blend_s += started_write - started_blend
                        overlap.write(pipe.stdin, frame)
                        pipe_write_s += clock() - started_write
                        frames_written += 1

                    composition_s += time.monotonic() - composition_started
                    if job.phase_progress is not None:
                        job.phase_progress("compose" if cache is not None else "inpaint", min(end_f, last) - first,
                                           last - first, f"bands={len(active_modes)}")
                    elif job.progress is not None:
                        job.progress(min(end_f, last) - first, last - first)

            overlap.finish()
            if job.phase_progress is not None:
                job.phase_progress("encode", 0, 1, None)
            encoding_started = time.monotonic()
            pipe.stdin.close()
            stdin_closed = True
            pipe.wait()

            if isinstance(job.metrics, dict):
                job.metrics.update(prediction_s=round(prediction_s, 4),
                                   composition_s=round(composition_s, 4),
                                   encoding_finalize_s=round(time.monotonic() - encoding_started, 4))
                # P0-1 split. prediction_s ~ cache_read_s + decode_infer_s +
                # resize_s + model_s + cache_write_s; composition_s ~
                # decode_output_s + blend_s + pipe_write_s (pipe_write_s also
                # covers frame.tobytes()). mask_eval_s lies outside both, and
                # prefix_skip_s precedes inpainting_s.
                job.metrics.update(
                    prefix_skip_s=round(prefix_skip_s, 4),
                    mask_eval_s=round(mask_eval_s, 4),
                    cache_read_s=round(cache_read_s, 4),
                    decode_infer_s=round(decode_infer_s, 4),
                    resize_s=round(resize_s, 4),
                    model_s=round(model_s, 4),
                    cache_write_s=round(cache_write_s, 4),
                    decode_output_s=round(decode_output_s, 4),
                    blend_s=round(blend_s, 4),
                    pipe_write_s=round(pipe_write_s, 4),
                    prefix_frames=context_start,
                    frames_read=frames_read, frames_grabbed=frames_grabbed,
                    output_frames_read=output_frames_read,
                    frames_written=frames_written,
                    segments=segments, segments_cached=segments_cached,
                    segments_inferred=segments_inferred,
                    bands_inferred=bands_inferred, windows=windows,
                    windows_skipped=windows_skipped,
                )
                job.metrics.update(overlap.metrics())
                if cache is not None:
                    job.metrics.update(prediction_hits=cache.hits, prediction_misses=cache.misses,
                                       prediction_corrupt=cache.corrupt, prediction_writes=cache.writes)
                job.metrics["inpainting_s"] = round(
                    time.monotonic() - t_inpaint_start, 3
                )
        except Exception:
            processing_failed = True
            raise
        finally:
            overlap.abort()
            if output_reader is not None:
                output_reader.release()
            if not stdin_closed and pipe.stdin is not None:
                pipe.stdin.close()
            if pipe.poll() is None:
                pipe.terminate()
                pipe.wait()
            if processing_failed:
                stderr_file.close()
        if pipe.returncode != 0:
            stderr_file.seek(0)
            stderr = stderr_file.read().decode(errors="replace")
            stderr_file.close()
            raise RuntimeError(
                f"FFmpeg exited with code {pipe.returncode}:\n"
                f"{stderr}"
            )
        stderr_file.close()
        if job.phase_progress is not None:
            job.phase_progress("encode", 1, 1, None)
        print(f"Saved to {video_out_path}")
        return InpaintOutcome(
            output_path=video_out_path,
            backend=type(self.backend).__name__,
        )

    def cleanup(self) -> None:
        if self.backend is not None:
            self.backend.cleanup()
            self.backend = None
