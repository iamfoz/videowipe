# Cropping the model's input around the mask

Status: built, **off by default**. `VIDEOWIPE_MODEL_CROP=auto` turns it on. The
owner decides whether `auto` should become the default after a visual A/B (see
"Measured" below for what the numbers say and what they cannot).

## What it changes

STTN always receives 640x120 pixels. Until now those were a full-width band of
rows around the text (`width * 3 / 16` rows high), so a 1080p subtitle that
fills 45% of the width was shrunk 3x on the way in, and a 4K one 6x. With the
crop, the model receives a 16:3 box fitted around the mask, still scaled to
640x120, so the same text is shrunk about 1.5x (1080p) or 3x (4K). The model
does one 640x120 inference per band and segment either way, so its cost does
not change.

## The switch

| value | behaviour |
|---|---|
| `off` (default, also unset or unrecognised) | The band, exactly as before: the same bytes reach ffmpeg (proved by `tests/test_mask_crop.py`; a digest of the bytes written to the encoder over seven configurations was also identical to `origin/main`'s). |
| `auto` | A crop where the rule below allows it, the band everywhere else. |

## The `auto` rule

For each active band and segment the box is built from the union of the mask's
non-zero alpha over **every frame of the segment** (not only the emitted ones):

1. width = the larger of (mask width + 2 margin) and (mask height + 2 margin) x 16/3,
   rounded up to a multiple of 16 so that the height (width x 3/16) is a whole
   number and the aspect is exactly 16:3. The margin is 4% of the frame height,
   the value used in the Q-9(b) experiment.
2. **The box must be at least 640 source pixels wide.** Otherwise it would be
   enlarged on its way into the 640 wide model input, which was measured to
   make things much worse (a 439 px box at 480p lost 8.5 dB), so the band is
   kept for that segment and band.
3. The box must fit in the frame and in the band's rows (a short top band can
   be too short for it), and must not be the band itself. Otherwise the band
   is kept. Text that spans nearly the whole width therefore always stays on
   the band path and its output does not change.
4. The box is centred on the mask and clamped into the band and the frame.

The geometry is `mask_crop.fit_box`.

## Why per segment and per band

* The model attends across all the frames of a segment, so its input geometry
  must be the same for all of them: a box can change only at a segment
  boundary. A segment is also the unit of the prediction cache.
* The box depends only on the masks of that segment, never on what was processed
  before. A trial range and a full run therefore pick the same box and the same
  cache entry, and the result does not depend on processing order. (Hysteresis
  between segments would smooth box changes but would break that.)
* Text that moves inside a segment is covered by the union over the segment.
* Several boxes in one segment arise as several bands (`get_inpaint_mode` already
  splits distant lines of text into bands); each band gets a box of its own, or
  keeps its band.
* Each segment and band is judged separately, so a subtitle that is wide in one
  segment and short in the next is cropped in the first and left on the band
  in the second.

## How it is wired in

`src/videowipe/inpainters/mask_crop.py` holds the rule, the per-segment choice
(`SegmentCropper`), and the small objects the compositor needs. `sttn.py` gets
one-line hooks:

* a prediction for a cropped band is wrapped in `CroppedPrediction` (it says
  where its pixels go);
* `_BandUpscaler.crop` hands such a prediction to an upscaler for the crop's own
  width, so the windowed resize and its verification apply unchanged, and the
  skipped windows and box compositing are untouched; `_compose_legacy` places the
  upscaled crop in an otherwise empty band (alpha is zero outside the box);
* the model input is `frame[crop.top:crop.bottom, crop.left:crop.right]` instead
  of the band's rows;
* `PredictionCache.key` takes the crop. Without one the key is the old key, with
  one the box (left, top, right, bottom) is part of it, so an entry computed for
  the band can never answer for a crop and vice versa. A segment whose band is
  kept has the old key, and legitimately shares the old entry.

The decisions are summed in `SegmentCropper.segments`, `.bands_cropped` and
`.bands_kept`; the quality harness records the boxes per run.

## Measured

Ground-truth harness (`scripts/inpaint_quality_harness.py`, described in `docs/inpaint-quality-experiments.md`),
run through the real `STTNInpainter.inpaint`, ONNX backend, 50-frame clips, one 25-frame
segment boundary at frame 25. `current` is `VIDEOWIPE_MODEL_CROP=off`, `crop` is `auto`. Scores are
of the frames as they enter the ffmpeg pipe, against the unburned truth. Fourteen clips: the eight of the
first experiment (e1, e2 at 480p; c1, c2, o1, o2 at 1080p; k1, k2 at 4K, which are 2x upscales of
1080p footage) and six new ones: `m1` and `m2` (the text, and so the box, moves and changes size at
the segment boundary), `d1` (text scrolling 0.37 of the frame width inside the clip), `u1` (a caption
at the top and a subtitle, two bands and so two boxes in each segment) and `w1`, `w2` (text across about
90% and 75% of the width).

**The port is exact.** On c1, the harness ported to the windowed compositor on `main` gives, for
`current`, every score bit-for-bit equal to the scores the first experiment stored from the upstream
code (core PSNR 29.494539779846967 in both). On e1 the `crop` run is byte-identical to `current`
(difference 0.0), as it must be: every segment keeps its band.

Per clip, `current -> crop` core PSNR (dB) and paired differences (crop minus current). "Box" is the
model input chosen per segment and band; `band` means the band was kept.

| clip | box (segment 1, segment 2) | core PSNR dB | SSIM | seam MAE | sharpness | ghost corr. | extra flicker |
|---|---|---:|---:|---:|---:|---:|---:|
| e1 480p | band, band | 32.24 -> 32.24 (+0.00) | +0.000 | +0.000 | +0.000 | +0.000 | +0.00 |
| e2 480p | 688x129, band | 32.96 -> 33.13 (+0.17) | +0.001 | -0.006 | +0.067 | -0.009 | +0.01 |
| c1 1080p | 976x183, 928x174 | 29.49 -> 32.50 (+3.00) | +0.026 | +0.080 | +0.095 | -0.001 | -0.52 |
| c2 1080p | 768x144, 768x144 | 31.70 -> 35.00 (+3.31) | +0.022 | -0.029 | +0.215 | +0.002 | -0.53 |
| o1 1080p | 976x183, 928x174 | 35.63 -> 37.04 (+1.41) | +0.001 | +0.150 | +0.458 | -0.000 | -0.42 |
| o2 1080p | 1136x213, 880x165 | 35.04 -> 36.61 (+1.57) | +0.007 | +0.041 | +0.038 | +0.008 | -0.34 |
| k1 4K | 1936x363, 1824x342 | 29.24 -> 31.94 (+2.70) | +0.013 | +0.078 | +0.338 | -0.003 | -0.49 |
| k2 4K | 1440x270, 1440x270 | 30.92 -> 36.06 (+5.14) | +0.017 | -0.116 | +0.445 | +0.019 | -0.67 |
| m1 moving | 976x183, 864x162 | 34.11 -> 35.73 (+1.63) | +0.015 | +0.187 | +0.565 | -0.013 | -0.36 |
| m2 moving | 1136x213, 736x138 | 37.45 -> 38.01 (+0.56) | +0.001 | +0.041 | +0.793 | +0.029 | -0.17 |
| d1 scrolling | 1072x201, 1072x201 | 33.18 -> 37.14 (+3.96) | +0.033 | -0.011 | -0.130 | +0.005 | -1.46 |
| u1 two bands | 976x183 + 864x162 (seg. 1), 928x174 + 864x162 (seg. 2) | 34.45 -> 36.17 (+1.72) | +0.003 | +0.096 | +0.332 | +0.006 | -0.47 |
| w1 90% wide | 1840x345, 1840x345 | 31.02 -> 31.39 (+0.38) | +0.001 | +0.031 | -0.086 | +0.000 | -0.04 |
| w2 75% wide | 1520x285, 1520x285 | 35.80 -> 36.28 (+0.48) | -0.001 | +0.076 | +0.036 | +0.001 | -0.23 |

Paired difference by group, mean (SD across clips) `[clips better / clips]`:

| group | core PSNR dB | core SSIM | seam MAE (lower is better) | sharpness | extra flicker (lower is better) |
|---|---:|---:|---:|---:|---:|
| 480p (e1, e2) | +0.087 (0.123) [1/2] | +0.000 (0.001) [1/2] | -0.003 (0.004) [1/2] | +0.033 (0.047) [1/2] | +0.006 (0.009) [0/2] |
| 1080p (c1 c2 o1 o2) | +2.323 (0.971) [4/4] | +0.014 (0.012) [4/4] | +0.061 (0.075) [1/4] | +0.202 (0.186) [4/4] | -0.452 (0.092) [4/4] |
| 4K (k1 k2) | +3.916 (1.724) [2/2] | +0.015 (0.002) [2/2] | -0.019 (0.137) [1/2] | +0.391 (0.076) [2/2] | -0.583 (0.129) [2/2] |
| moving box (m1 m2) | +1.091 (0.756) [2/2] | +0.008 (0.010) [2/2] | +0.114 (0.103) [0/2] | +0.679 (0.161) [2/2] | -0.266 (0.133) [2/2] |
| scrolling (d1) | +3.963 [1/1] | +0.033 [1/1] | -0.011 [1/1] | -0.130 [0/1] | -1.464 [1/1] |
| two bands (u1) | +1.719 [1/1] | +0.003 [1/1] | +0.096 [0/1] | +0.332 [1/1] | -0.473 [1/1] |
| wide text (w1 w2) | +0.429 (0.075) [2/2] | +0.000 (0.001) [1/2] | +0.054 (0.032) [0/2] | -0.025 (0.086) [1/2] | -0.134 (0.131) [2/2] |

Noise floor: the ONNX CPU backend is deterministic here (a repeated run is byte-identical, shown in the
first experiment), so a repeat adds no noise; the spread that matters is clip to clip, the SD above.

What this says:

* The 1080p and 4K gains reproduce the first experiment: +2.32 dB at 1080p (it was +2.32, SD 1.36; now
  SD 0.97, 4 of 4) and +3.92 dB at 4K (+3.93, 2 of 2). Extra flicker falls by about a quarter.
* The new cases also gain: boxes that change between segments (+1.09 dB, 2 of 2), scrolling text
  (+3.96 dB), two boxes in a segment (+1.72 dB), and text across 75-90% of the width (+0.43 dB, 2 of 2,
  a small gain because the box is nearly the band). Nothing got worse on core PSNR. Where the rule kept the
  band (e1; the second segment of e2) the output is unchanged.
* Core PSNR, SSIM and flicker improve or stay equal everywhere. Not everything does: **seam MAE is
  slightly worse on 9 of the 14 clips** (up to +0.19 on a scale where the values are about 1.7 to 2.2), the
  signed luma bias just inside the edge moves darker by about 0.6 to 0.9 on average, and the sharpness
  ratio rises on most clips, which on the contact sheets shows as a faint fine texture in smooth areas
  (visible on o1 frame 28 and on c1 frames 28 and 45) where `current` is smoother. Whether that texture
  reads as detail or as noise is exactly what the visual A/B is for.
* At 480p the benefit is nil to small (+0.09 dB, SD 0.12, 1 of 2): the band is shrunk only 1.33x there.

**Why the default stays `off`.** The condition was that every measured case improves beyond the noise
floor and none gets worse. Core PSNR improves in 13 of 14 clips and is unchanged in the 14th, but the 480p
gain (+0.09 dB, SD 0.12) is inside the clip-to-clip spread, seam MAE and the sharpness ratio move the
wrong way on several clips, and only synthetic text on a handful of clips was measured (the 4K clips are
upscaled 1080p). That is not "no case gets worse", so the default is not changed; the owner decides after
the visual A/B.

**Model cost.** The model input is 640x120 either way and in every pair the same number of bands and
windows was inferred (bands 2 -> 2, windows 10 -> 10; u1 4 -> 4, 20 -> 20). Wall times from the runs are
useless as a cost measure (the machine was shared and the load average ranged from 3 to 47: one clip's model
time per frame ranged from 2 s to 27 s). Two steadier checks: the CPU time of the whole run, crop over
current, averaged over 13 clips, is 0.98 (SD 0.09, range 0.79 to 1.13), and a direct interleaved
benchmark of `_process_segment` on c1's band input and crop input (25 frames each, four rounds each,
alternating) gives 5.43 s against 5.48 s of CPU time per frame (ratio 1.01) and 2.20 s against 2.15 s of
wall time. The model time per frame is unchanged. Composing a cropped frame is no slower (`blend_s` over the
50 frames: 0.04 to 0.68 s with the band, 0.04 to 0.33 s with the crop).

**Contact sheets** (truth, input, current, crop; frames 4, 28 and 45) are in
`/tmp/claude-0/-home-user-videowipe/47ee2fe6-7bba-547a-befe-0fe44987a6b7/scratchpad/mask-crop/sheets/`.
The five most telling:

1. `mask_crop_c1.png`: the biggest 1080p gain (+3.00 dB); on frame 4 the crop restores the edge of the
   object behind the text; on frames 28 and 45 look for the faint texture.
2. `mask_crop_o1.png`: the one most likely to show a cost (seam MAE +0.15, sharpness +0.46); on frame 28
   the dark area gets a fine pattern that `current` does not have.
3. `mask_crop_m1.png`: the box moves and shrinks at the segment boundary (976x183 then 864x162).
4. `mask_crop_u1.png`: two bands, two boxes at once (caption and subtitle stacked on the sheet).
5. `mask_crop_k2.png`: 4K, the largest gain (+5.14 dB).

Others: `d1` (scrolling), `m2`, `w1`, `w2` (wide), `c2`, `o2`, `k1`, `e2` (480p, one segment cropped).

**Not measured.** Real subtitled footage (no ground truth), native 4K, genuinely full-width text
(over 97% of the width; by the rule it is the band by construction, and the unit tests cover it), text
that straddles two bands, more than two segments, and boxes that change at every segment of a long run.
Nothing was timed on a quiet machine beyond the check above.

## Reproducing

```
S=/path/outside/the/repo
python scripts/inpaint_quality_harness.py build --work $S/work --clips e1,e2,c1,c2,o1,o2,k1,k2,m1,m2,d1,u1,w1,w2
python scripts/inpaint_quality_harness.py run   --work $S/work --clips c1 --variants current,crop
python scripts/inpaint_quality_report.py --runs $S/work/runs --variants current,crop --keys core_psnr,core_ssim,seam_mae,sharp_ratio,flicker_extra,model_ms_per_frame
python scripts/inpaint_quality_harness.py sheets --work $S/work --out $S/sheets --clips c1
```

`current` is `VIDEOWIPE_MODEL_CROP=off` and `crop` is `auto`, both through the
real `STTNInpainter.inpaint`, with an empty prediction cache so that `model_s`
is a real model time. Frames are scored where they enter the ffmpeg pipe.
