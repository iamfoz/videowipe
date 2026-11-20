"""Maths of the inpainting-quality harness (no model, no weights)."""
from pathlib import Path
import runpy
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import inpaint_quality_metrics as M  # noqa: E402


def _texture(seed=1, shape=(48, 64)):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (*shape, 3)).astype(np.uint8)
    return base


def test_psnr_and_ssim_identity_and_ordering():
    a = _texture()
    assert M.psnr(a, a) == float("inf")
    assert abs(M.ssim(a, a) - 1.0) < 1e-9
    noisy = np.clip(a.astype(int) + 10, 0, 255).astype(np.uint8)
    noisier = np.clip(a.astype(int) + 40, 0, 255).astype(np.uint8)
    assert M.psnr(a, noisy) > M.psnr(a, noisier)
    assert M.ssim(a, noisy) > M.ssim(a, noisier)
    # A constant offset of 10 has a PSNR of 20*log10(255/10).
    assert abs(M.psnr(a.astype(np.float32), a.astype(np.float32) + 10) - 20 * np.log10(25.5)) < 1e-6


def test_psnr_respects_region():
    a = np.zeros((4, 4, 3), np.uint8)
    b = a.copy()
    b[0, 0] = 50
    region = np.zeros((4, 4), bool)
    region[1:, 1:] = True
    assert M.psnr(a, b, region) == float("inf")
    region[0, 0] = True
    assert np.isfinite(M.psnr(a, b, region))


def test_regions_are_disjoint_rings_around_the_core():
    alpha = np.zeros((40, 40), np.float32)
    alpha[15:25, 10:30] = 1.0
    alpha[13:15, 10:30] = 0.5  # feather zone is not core
    glyph = alpha >= 1.0
    reg = M.regions(alpha, glyph)
    assert (reg["core"] == (alpha >= 0.999)).all()
    assert not (reg["inner"] & reg["seam"]).any()
    assert reg["inner"].sum() > 0 and reg["seam"].sum() > 0
    assert (reg["inner"] <= reg["core"]).all()
    assert not (reg["seam"] & reg["core"]).any()


def test_sharpness_ratio_drops_when_blurred():
    import cv2

    truth = _texture(3, (60, 80))
    blurred = cv2.GaussianBlur(truth, (0, 0), 2.0)
    region = np.ones((60, 80), bool)
    assert M.laplacian_variance(blurred, region) < 0.2 * M.laplacian_variance(truth, region)
    assert M.laplacian_variance(truth, region) == M.laplacian_variance(truth, region)


def test_ghosting_detects_residual_text_edges():
    truth = np.full((60, 120, 3), 90, np.uint8)
    glyph = np.zeros((60, 120), bool)
    glyph[25:35, 20:100:6] = True  # strokes
    glyph = glyph | np.roll(glyph, 3, axis=1)
    core = np.zeros_like(glyph)
    core[15:45, 10:110] = True
    clean = truth.copy()
    ghost = truth.copy()
    ghost[glyph] = 200  # text left behind
    good = M.ghosting(clean, truth, glyph, core)
    bad = M.ghosting(ghost, truth, glyph, core)
    assert abs(good["energy"]) < 1e-6 and good["corr"] == 0.0
    assert bad["energy"] > 50 and bad["corr"] > 0.3


def test_flicker_separates_temporal_noise_from_a_static_fill():
    rng = np.random.default_rng(0)
    truth = [np.full((20, 20, 3), 100, np.uint8) for _ in range(6)]
    steady = [np.full((20, 20, 3), 110, np.uint8) for _ in range(6)]
    noisy = [np.clip(110 + rng.integers(-8, 9, (20, 20, 3)), 0, 255).astype(np.uint8) for _ in range(6)]
    cores = [np.ones((20, 20), bool)] * 6
    a = M.flicker(steady, truth, cores)
    b = M.flicker(noisy, truth, cores)
    assert a["out"] == 0 and a["extra"] == 0 and a["truth"] == 0
    assert b["extra"] > 3 and b["out"] > 3


def test_score_clip_perfect_output_has_no_error_and_unit_sharpness():
    truth = [_texture(i, (50, 90)) for i in range(3)]
    alpha = np.zeros((50, 90), np.float32)
    alpha[15:35, 20:70] = 1.0
    glyph = np.zeros((50, 90), bool)
    glyph[20:30, 25:65:5] = True
    scores = M.score_clip(truth, truth, [alpha] * 3, [glyph] * 3)
    assert scores["core_mae"] == 0 and scores["seam_mae"] == 0
    assert abs(scores["core_ssim"] - 1) < 1e-9
    assert abs(scores["sharp_ratio"] - 1) < 1e-9
    assert scores["flicker_extra"] == 0
    flat = [np.full_like(t, 128) for t in truth]
    worse = M.score_clip(flat, truth, [alpha] * 3, [glyph] * 3)
    assert worse["core_mae"] > 20 and worse["sharp_ratio"] < 0.01


def test_harness_alpha_matches_the_plan_feather():
    """pipeline_alpha must reproduce plan._project_mask's feather recipe."""
    import cv2

    import inpaint_quality_harness as H
    from videowipe.plan import _project_mask

    glyph = np.zeros((60, 90), bool)
    glyph[25:33, 20:70] = True
    dilated = cv2.dilate(glyph.astype(np.uint8), np.ones((7, 7), np.uint8))
    track = type("T", (), {"action": "remove", "spatial_segments": None})()
    expected = _project_mask([(track, dilated)], (0,), glyph.shape, 4)
    assert np.allclose(H.pipeline_alpha(glyph), expected, atol=1e-6)
