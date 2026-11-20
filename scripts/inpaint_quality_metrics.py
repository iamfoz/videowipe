"""Ground-truth metrics for inpainting experiments (pure numpy/OpenCV).

Every function works on frames that are already cropped to the rows of
interest and on boolean region masks, so the harness can score a synthetic
subtitle clip against the unburned original.  Nothing here imports the
videowipe package.

Regions (see :func:`regions`), all derived from the exact glyph mask:

* ``core``   -- pixels the pipeline fully replaces (feathered alpha == 1).
* ``inner``  -- the 4 px just inside ``core``'s boundary.
* ``seam``   -- the 6 px just outside ``core`` (the feather zone and beyond),
  where the prediction is mixed with the sharp original.
* ``glyph``  -- the burned glyph and outline pixels themselves.
"""
from __future__ import annotations

import cv2
import numpy as np


def luma(frame: np.ndarray) -> np.ndarray:
    """BT.601 luma of a BGR uint8/float frame, float32 in [0, 255]."""
    f = frame.astype(np.float32)
    return 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]


def regions(alpha: np.ndarray, glyph: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean regions from the feathered blend ``alpha`` and the glyph mask."""
    core = alpha >= 0.999
    kernel = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core_u8 = core.astype(np.uint8)
    inner = core & ~cv2.erode(core_u8, kernel(4)).astype(bool)
    seam = cv2.dilate(core_u8, kernel(6)).astype(bool) & ~core
    return {"core": core, "inner": inner, "seam": seam, "glyph": glyph.astype(bool)}


def psnr(a: np.ndarray, b: np.ndarray, region: np.ndarray | None = None) -> float:
    """PSNR in dB over ``region`` (all pixels when None); ``inf`` if identical."""
    diff = a.astype(np.float64) - b.astype(np.float64)
    if region is not None:
        diff = diff[region]
    if diff.size == 0:
        return float("nan")
    mse = float(np.mean(diff ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Standard Gaussian-window SSIM map (11x11, sigma 1.5) of two 2-D images."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)
            / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)))


def ssim(a: np.ndarray, b: np.ndarray, region: np.ndarray | None = None) -> float:
    """Mean SSIM of the luma of two BGR frames, averaged over ``region``."""
    smap = ssim_map(luma(a), luma(b))
    if region is not None:
        smap = smap[region]
    return float(np.mean(smap)) if smap.size else float("nan")


def laplacian_variance(frame: np.ndarray, region: np.ndarray) -> float:
    """Variance of the Laplacian of the luma inside ``region`` (sharpness)."""
    lap = cv2.Laplacian(luma(frame), cv2.CV_32F, ksize=3)
    values = lap[region]
    return float(np.var(values)) if values.size else float("nan")


def gradient_magnitude(frame: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of the luma."""
    y = luma(frame)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def glyph_edge_band(glyph: np.ndarray, core: np.ndarray, radius: int = 2) -> np.ndarray:
    """Pixels within ``radius`` of a glyph boundary, restricted to ``core``."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    g = glyph.astype(np.uint8)
    return (cv2.dilate(g, k).astype(bool) & ~cv2.erode(g, k).astype(bool)) & core


def ghosting(out: np.ndarray, truth: np.ndarray, glyph: np.ndarray, core: np.ndarray) -> dict[str, float]:
    """Residual text energy.

    The excess edge map is ``|grad(out)| - |grad(truth)|``.  Remaining text
    makes it large exactly on the glyph edges, so we report its mean there
    (``energy``) and its Pearson correlation with the glyph-edge indicator
    over the core (``corr``).  Smooth, text-free fills give values near zero;
    positive values mean ghosting.
    """
    excess = gradient_magnitude(out) - gradient_magnitude(truth)
    band = glyph_edge_band(glyph, core)
    if not band.any() or not core.any():
        return {"energy": float("nan"), "corr": float("nan")}
    e = excess[core]
    ind = band[core].astype(np.float32)
    if ind.std() == 0 or e.std() == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(e, ind)[0, 1])
    return {"energy": float(excess[band].mean()), "corr": corr}


def flicker(outs: list[np.ndarray], truths: list[np.ndarray], cores: list[np.ndarray]) -> dict[str, float]:
    """Temporal behaviour inside the removal region.

    ``out``/``truth`` mean |frame_t - frame_{t-1}| of the luma over pixels in
    the core of both frames; ``extra`` is the mean |d(out) - d(truth)|, the
    difference between the output's and the truth's frame-to-frame change.
    """
    d_out, d_truth, extra = [], [], []
    for t in range(1, len(outs)):
        sel = cores[t] & cores[t - 1]
        if not sel.any():
            continue
        do = luma(outs[t]) - luma(outs[t - 1])
        dt = luma(truths[t]) - luma(truths[t - 1])
        d_out.append(np.abs(do[sel]).mean())
        d_truth.append(np.abs(dt[sel]).mean())
        extra.append(np.abs(do[sel] - dt[sel]).mean())
    if not d_out:
        return {"out": float("nan"), "truth": float("nan"), "extra": float("nan")}
    return {"out": float(np.mean(d_out)), "truth": float(np.mean(d_truth)), "extra": float(np.mean(extra))}


def score_clip(outs, truths, alphas, glyphs) -> dict[str, float]:
    """Score a list of frames; every list holds one (cropped) array per frame.

    Returns a flat dict of per-clip means.
    """
    acc: dict[str, list[float]] = {}

    def add(key, value):
        acc.setdefault(key, []).append(value)

    cores = []
    for out, truth, alpha, glyph in zip(outs, truths, alphas, glyphs):
        reg = regions(alpha, glyph)
        cores.append(reg["core"])
        if not reg["core"].any():
            continue
        for name in ("core", "inner", "seam"):
            region = reg[name]
            add(f"{name}_mae", float(np.abs(out.astype(np.float32) - truth.astype(np.float32))[region].mean()))
            add(f"{name}_psnr", psnr(out, truth, region))
        add("core_ssim", ssim(out, truth, reg["core"]))
        add("seam_ssim", ssim(out, truth, reg["seam"]))
        # Signed luma bias just inside the edge: a colour step shows up here.
        add("inner_bias", float((luma(out) - luma(truth))[reg["inner"]].mean()))
        eroded = cv2.erode(reg["core"].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if eroded.any():
            ratio = laplacian_variance(out, eroded) / max(laplacian_variance(truth, eroded), 1e-6)
            add("sharp_ratio", ratio)
        ghost = ghosting(out, truth, reg["glyph"], reg["core"])
        add("ghost_energy", ghost["energy"])
        add("ghost_corr", ghost["corr"])
    result = {}
    for key, values in acc.items():
        finite = [v for v in values if np.isfinite(v)]
        result[key] = float(np.mean(finite)) if finite else float("nan")
    fl = flicker(outs, truths, cores)
    result["flicker_out"], result["flicker_truth"], result["flicker_extra"] = fl["out"], fl["truth"], fl["extra"]
    return result
