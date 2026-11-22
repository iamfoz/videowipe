# Inpainting-quality experiments: Q-7, Q-8, Q-9

Status: measured. Nothing here is built into the pipeline and no default changes.
Q-7 (area downscale), Q-8 (masked model input) and Q-9(a) (seam colour match) are
neutral or worse on the objective metrics, so no switch was added. Q-9(b) (a crop
around the mask instead of a full-width band) is the one clear gain; it is
prototyped in scratch code only, as asked, and described under "Q-9(b)" below.

Update: the crop has since been built behind `VIDEOWIPE_MODEL_CROP=off|auto` (default `off`) and
measured on more cases; see `docs/mask-crop.md`.

Everything below was produced by the scripts in `scripts/`
(`inpaint_quality_harness.py`, `inpaint_quality_metrics.py`,
`inpaint_quality_report.py`, `inpaint_quality_crop_prototype.py`) against the real
`STTNInpainter.inpaint` with the ONNX backend. Upstream files are untouched: the
experiments monkeypatch the `inpainters.sttn` module from the harness.

## How "better" is measured

The true clean background is unknown for real subtitled footage, so the harness
makes its own ground truth:

1. A 2-second (50-frame, 25 fps) clip is cut from a region of the sample videos in
   `input/detext_examples/` that carries no text in the model band (checked against
   the annotation masks and by eye).
2. Subtitle-like text is burned in with ffmpeg `drawtext`: white fill, black
   outline, centred, two different lines that switch at frame 25 (a segment
   boundary), as consecutive subtitles do. Fonts: WenQuanYi Zen Hei (CJK-capable;
   mixed simplified and traditional characters with heavy strokes) and DejaVu Sans
   Bold (Latin). The burned clip is encoded with libx264 CRF 18, as a real source
   would be; the unburned lossless frames are the ground truth.
3. The exact glyph mask comes from a second `drawtext` pass on black. It is dilated
   7x7 (as the detector does) and feathered with the pipeline's own 4 px recipe
   (`plan._project_mask`; a test checks the harness reproduces it), and is passed
   to the inpainter as the per-frame mask (`frame_mask`), the path real plans use.
4. Frames are captured by wrapping `_blend_frame_regions`, i.e. just before
   encoding, so H.264 loss does not enter the scores. Scores use the luma of those
   frames against the truth.

Regions: `core` (alpha = 1, fully replaced), `inner` (4 px inside the core edge),
`seam` (6 px outside the core: the feather and what lies just beyond it). Metrics:
PSNR, SSIM (Gaussian window, own implementation, unit-tested), MAE, the signed luma
bias just inside the edge, sharpness (Laplacian variance of the output divided by
that of the truth in the core), ghosting (correlation of the excess edge map
`|grad(out)| - |grad(truth)|` with the glyph-edge band) and extra flicker
(`mean |d(out) - d(truth)|` frame to frame over pixels that are in the core of both
frames).

| clip | source | size | text | font, outline | band |
|---|---|---|---|---|---|
| e1 | english1 | 852x480 | CJK, regular | 24 px, 2 px | rows 162-321 |
| e2 | english1 | 852x480 | Latin, regular | 24 px, 2 px | rows 162-321 |
| c1 | chinese1 | 1920x1080 | CJK, regular | 54 px, 3 px | rows 360-720 |
| c2 | chinese1 | 1920x1080 | CJK, thin | 39 px, 1 px | rows 360-720 |
| o1 | others | 1920x1080 | CJK, regular | 54 px, 3 px | rows 360-720 |
| o2 | others | 1920x1080 | Latin, thin | 39 px, 1 px | rows 360-720 |
| k1 | chinese1 x2 | 3840x2160 | CJK, regular | 108 px, 7 px | rows 720-1440 |
| k2 | chinese1 x2 | 3840x2160 | CJK, thin | 77 px, 2 px | rows 720-1440 |

The 4K clips are the 1080p footage upscaled 2x with Lanczos (no native 4K source
exists), so they are softer than genuine 4K and understate how much a 6x downscale
of real 4K detail would alias. They did fit in memory and time on the shared
four-core machine. The sample footage in the 1080p clips is fairly smooth and dark
in places, which makes absolute scores optimistic.

### Noise floor

* Run to run: `current` was run twice from scratch (fresh prediction cache, fresh
  inference) on e1 and on c1. The captured frames are byte-identical, so the ONNX
  CPU backend is deterministic here and repeat runs add no noise.
* Clip to clip: absolute scores depend on the footage (core PSNR of `current`
  ranges from 29.2 to 35.6 dB over the eight clips, SD 2.3 dB), so only paired
  differences against `current` on the same clip mean anything. The paired tables
  give the mean difference, the SD of that difference across clips, and `[n/N]`,
  the number of clips where the variant is better. With two to four clips per
  resolution, a mean difference smaller than about one SD, or a split `[n/N]`,
  should not be read as an effect.

All tables: mean over the clips listed, `current -> variant`. Higher is better for
PSNR, SSIM and sharpness (a ratio of 1 matches the truth); lower is better for seam
MAE and extra flicker; the inner-edge luma bias is signed (0 is ideal); ghost
correlation near 0 means no residual text.

## Q-7: area filter for the model-input downscale

Every resize on the inpainting path (upstream `20872d4`):

| site | what | filter |
|---|---|---|
| `inpainters/sttn.py` L336-339 | band `image[from_h:to_h]` to the 640x120 model input | `INTER_LINEAR`, explicit |
| `inpainters/sttn.py` L381-383 | decoded 640x120 prediction to frame width x band height | default `INTER_LINEAR` |

There is no mask resize on the model path: `job.mask`, the per-frame alpha and
`get_inpaint_mode` all work at full resolution, and the feather in
`plan._project_mask` is a `GaussianBlur`. The other `cv2.resize` calls in the
repository belong to text detection (`detect.py`) and a legacy mask-drawing helper
(`core/utils.py`).

On `origin/perf/blend-box` and `origin/perf/skip-windows` the downscale call is
unchanged (explicit `INTER_LINEAR`, sttn.py L792 and L864 respectively). The upscale
was rewritten: `_BandUpscaler.crop` (L379, L383) resizes either the whole band or a
verified window of it, and `_compose_legacy` (L427) still resizes the whole band.
The windowed path is exact only for bilinear weights (`_axis_weights_are_stable`, a
one-pixel margin, a probe against the full resize), so changing the upscale filter
there would also need a wider margin and new probes; the downscale site is a single
line on every branch.

Variants: `area` (`INTER_AREA` down), `area_cubic` and `area_lanczos` (area down,
cubic or Lanczos-4 up), `linear_cubic` (bilinear down, cubic up).

| res | variant | clips | core PSNR dB | core SSIM | seam MAE | inner-edge bias | sharpness | extra flicker |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 480p | area | e1,e2 | 32.601 -> 32.529 | 0.941 -> 0.941 | 2.214 -> 2.198 | -1.994 -> -1.974 | 0.440 -> 0.410 | 1.824 -> 1.817 |
| 480p | area_cubic | e1,e2 | 32.601 -> 32.548 | 0.941 -> 0.940 | 2.214 -> 2.181 | -1.994 -> -1.855 | 0.440 -> 0.526 | 1.824 -> 1.844 |
| 480p | linear_cubic | e1,e2 | 32.601 -> 32.603 | 0.941 -> 0.939 | 2.214 -> 2.197 | -1.994 -> -1.879 | 0.440 -> 0.570 | 1.824 -> 1.855 |
| 1080p | area | c1,c2,o1,o2 | 32.964 -> 32.973 | 0.956 -> 0.960 | 1.871 -> 1.923 | -1.580 -> -2.318 | 0.329 -> 0.262 | 1.871 -> 1.758 |
| 1080p | area_cubic | c1,c2,o1 | 32.272 -> 32.337 | 0.950 -> 0.953 | 1.900 -> 1.942 | -1.424 -> -2.179 | 0.327 -> 0.227 | 1.924 -> 1.823 |
| 1080p | area_lanczos | c1,c2,o1 | 32.272 -> 32.315 | 0.950 -> 0.953 | 1.900 -> 1.946 | -1.424 -> -2.206 | 0.327 -> 0.183 | 1.924 -> 1.827 |
| 1080p | linear_cubic | c1,c2,o1 | 32.272 -> 32.189 | 0.950 -> 0.947 | 1.900 -> 1.890 | -1.424 -> -1.344 | 0.327 -> 0.282 | 1.924 -> 1.968 |
| 4K | area | k1,k2 | 30.080 -> 30.133 | 0.960 -> 0.963 | 1.968 -> 2.047 | -1.271 -> -2.521 | 0.312 -> 0.283 | 1.991 -> 1.873 |
| 4K | area_cubic | k1 | 29.238 -> 29.547 | 0.955 -> 0.957 | 1.873 -> 1.932 | -0.225 -> -1.697 | 0.313 -> 0.284 | 2.073 -> 1.985 |
| 4K | linear_cubic | k1 | 29.238 -> 29.061 | 0.955 -> 0.952 | 1.873 -> 1.867 | -0.225 -> -0.109 | 0.313 -> 0.323 | 2.073 -> 2.128 |

Paired difference of `area` against `current` (mean, SD across clips, clips better):

| res | core PSNR dB | core SSIM | seam MAE | sharpness | extra flicker |
|---|---:|---:|---:|---:|---:|
| 480p | -0.072 (sd 0.077) [0/2] | +0.001 (sd 0.001) [2/2] | -0.016 (sd 0.002) [2/2] | -0.030 (sd 0.013) [0/2] | -0.007 (sd 0.008) [2/2] |
| 1080p | +0.010 (sd 0.554) [1/4] | +0.004 (sd 0.003) [4/4] | +0.052 (sd 0.015) [0/4] | -0.067 (sd 0.017) [0/4] | -0.113 (sd 0.036) [4/4] |
| 4K | +0.053 (sd 0.414) [1/2] | +0.004 (sd 0.001) [2/2] | +0.079 (sd 0.006) [0/2] | -0.029 (sd 0.008) [0/2] | -0.118 (sd 0.010) [2/2] |

Conclusion (confidence moderate to high that it is not worth shipping): `area` is
not better on PSNR beyond noise at any resolution (the 1080p mean is +0.01 dB with
an SD of 0.55 dB; k2 loses 0.24 dB). Its consistent small gains are SSIM
(+0.004, every clip) and extra flicker (about 6% lower, every clip), which is
plausible, since a box filter removes the aliasing that makes bilinear sampling
jitter from frame to frame. Against that, every clip shows a slightly worse seam
MAE, lower sharpness, and a darker inner-edge bias (about 0.7 to 1.2 luma levels at
1080p and 4K). A cubic or Lanczos upscale of the prediction adds nothing at 1080p
and above, is worse than bilinear when paired with bilinear downscale (`linear_cubic`
loses 0.08 dB, and is worse on all three clips), and its only visible effect is a sharper
fill at 480p, where the upscale is just 1.33x and PSNR does not improve. The
`INTER_LINEAR` weights are also what the windowed upscaler on `perf/blend-box` and
`perf/skip-windows` is verified against. Nothing was built.

## Q-8: does the model expect masked input?

Inspection:

* The ONNX graphs take no mask. The encoder input is `frames [T, 3, 120, 640]`
  (float), the transformer `features [T, 256, 30, 160]`, the decoder
  `features [T, 256, 30, 160]`; there is no second input anywhere.
* `models/sttn.py`: `InpaintGenerator.forward(masked_frames)` only encodes, runs
  eight transformer blocks and decodes. `MultiHeadedAttention` and `TransformerBlock`
  take no mask argument. The parameter name is inherited from the original STTN,
  whose training and test code zero the masked pixels before the encoder.
* After `preprocess` the range is `[-1, 1]` (`x / 127.5 - 1`), so "zero" is 0.0,
  which is 127.5 in the 0-255 BGR floats that `_process_segment` receives. That is
  mid-grey, and it is what the variants below insert.

Variants (applied at the downscale site, using the exact per-frame mask resized to
640x120 with `INTER_AREA`, any coverage counting as masked):

* `masked_zero`: masked model pixels set to 127.5 (zero in model space).
* `masked_mean`: set to the mean colour of the unmasked pixels of that band.
* `masked_telea`: filled with `cv2.inpaint` (Telea) before the model (an extra,
  not asked for).

| res | variant | clips | core PSNR dB | core SSIM | seam MAE | sharpness | extra flicker |
|---|---|---|---:|---:|---:|---:|---:|
| 480p | masked_zero | e1 | 32.244 -> 19.373 | 0.942 -> 0.743 | 2.195 -> 2.619 | 0.544 -> 0.128 | 1.961 -> 3.485 |
| 480p | masked_mean | e1 | 32.244 -> 18.102 | 0.942 -> 0.708 | 2.195 -> 2.963 | 0.544 -> 0.089 | 1.961 -> 3.506 |
| 480p | masked_telea | e1 | 32.244 -> 20.424 | 0.942 -> 0.743 | 2.195 -> 2.265 | 0.544 -> 0.876 | 1.961 -> 3.500 |
| 1080p | masked_zero | c1 | 29.495 -> 24.212 | 0.929 -> 0.900 | 1.972 -> 2.209 | 0.294 -> 0.148 | 1.974 -> 2.605 |

(A second 480p clip, e2, was started for `masked_zero` but stopped to save machine
time; the four runs above, on two clips, agree.)

Conclusion (high confidence): masking the model input is far worse, by 5 to 14 dB,
with every variant. Visually the model returns the grey patch as content (see the
contact sheets), so this checkpoint evidently was not trained on grey-masked input and
uses the visible text as its cue, which is consistent with a model trained to remove
text that is present in its input. The training is undocumented, so that is an
inference from the behaviour. Pre-filling the text with Telea is no better. The
unmasked ghosting score of `current` is already about zero on every synthetic clip
(correlation within +/-0.02), so there was no ghosting for a mask to fix. The
prediction-cache key would have needed the mask; since nothing is built, it does
not. Not built. (Frames with real ghosting from the sample videos were not
scored, since they have no ground truth; the synthetic clips show none.)

## Q-9(a): seam colour match

`seam_match` in the harness adds a smooth offset to the upscaled prediction per
frame: the low-pass of (original - prediction) is measured in a ring of untouched
pixels up to 14 px outside the mask (Gaussian sigma 6, normalised convolution),
spread over the mask, and added only where alpha is above zero, before the usual
alpha blend. No temporal smoothing.

| res | clips | core PSNR dB | core SSIM | seam MAE | inner-edge bias | sharpness | extra flicker |
|---|---|---:|---:|---:|---:|---:|---:|
| 480p | e1,e2 | 32.601 -> 32.715 | 0.941 -> 0.941 | 2.214 -> 2.142 | -1.994 -> -1.628 | 0.440 -> 0.451 | 1.824 -> 1.849 |
| 1080p | c1,c2,o1,o2 | 32.964 -> 33.021 | 0.956 -> 0.955 | 1.871 -> 1.812 | -1.580 -> -1.341 | 0.329 -> 0.405 | 1.871 -> 1.908 |
| 4K | k1,k2 | 30.080 -> 30.100 | 0.960 -> 0.957 | 1.968 -> 1.917 | -1.271 -> -1.384 | 0.312 -> 1.072 | 1.991 -> 2.034 |

Paired (1080p): seam MAE -0.059 (sd 0.024) [4/4], core PSNR +0.058 (sd 0.141) [2/4],
core SSIM -0.001 [0/4], extra flicker +0.037 (sd 0.006) [0/4].

Conclusion (moderate to high confidence): the seam is not where the softness is. The
existing step at the edge is about 1 to 2 luma levels of 255 (inner-edge bias), and
the colour match shrinks the seam MAE by about 3% (0.06 of roughly 1.9 levels),
consistently in sign but invisibly small, while core quality does not move beyond
noise and flicker rises slightly on every clip. At 4K thin text it also leaves a
visible round tint around an isolated stroke (k2, frame 28 on the contact sheet), and
the k2 sharpness ratio rises from 0.31 to 1.12 (the output gains edges the truth lacks). Not worth the compute or the risk.
Not built.

## Q-9(b): crop around the mask (scratch prototype only)

`inpaint_quality_crop_prototype.py` drives the same backend and `_process_segment`
from a 16:3 box fitted around the union mask bounding box (margin 4% of the frame
height) instead of the full-width band, resizes it to 640x120, upscales the
prediction back and blends with the same alpha. `--modes band` reproduces the
pipeline's own geometry through this code path as a validity check: on c1 it gave
core PSNR 29.4945, seam MAE 1.9722 and sharpness 0.2938 against the pipeline's
`current` values of 29.495, 1.972 and 0.294.

| clip | full-band shrink | crop box | crop shrink | core PSNR dB, current -> crop |
|---|---:|---|---:|---:|
| o2 (1080p, Latin thin) | 3.00 | 1125x211 | 1.76 | 35.04 -> 35.91 |
| o1 (1080p, CJK regular) | 3.00 | 968x182 | 1.51 | 35.63 -> 37.09 |
| c1 (1080p, CJK regular) | 3.00 | 968x182 | 1.51 | 29.49 -> 32.89 |
| c2 (1080p, CJK thin) | 3.00 | 757x142 | 1.18 | 31.70 -> 35.26 |
| k1 (4K, CJK regular) | 6.00 | 1924x361 | 3.01 | 29.24 -> 32.02 |
| k2 (4K, CJK thin) | 6.00 | 1429x268 | 2.23 | 30.92 -> 35.99 |
| e1 (480p) | 1.33 | 439x82 | 0.69 | 32.24 -> 23.73 (worse: crop enlarged) |
| e1 (480p), box not narrower than 640 | 1.33 | 640x120 | 1.00 | 32.24 -> 32.82 |
| e2 (480p), box not narrower than 640 | 1.33 | 686x129 | 1.07 | 32.96 -> 33.47 |

Mean over clips, `current -> crop`:

| res | clips | core PSNR dB | core SSIM | seam MAE | inner-edge bias | sharpness | extra flicker |
|---|---|---:|---:|---:|---:|---:|---:|
| 1080p | c1,c2,o1,o2 | 32.964 -> 35.287 | 0.956 -> 0.970 | 1.871 -> 1.941 | -1.580 -> -2.184 | 0.329 -> 0.528 | 1.871 -> 1.422 |
| 4K | k1,k2 | 30.080 -> 34.007 | 0.960 -> 0.974 | 1.968 -> 1.959 | -1.271 -> -1.896 | 0.312 -> 0.692 | 1.991 -> 1.399 |

Paired: 1080p core PSNR +2.32 dB (sd 1.36) [4/4], SSIM +0.014 [4/4], extra flicker
-0.45 (sd 0.12) [4/4]; 4K core PSNR +3.93 dB (sd 1.61) [2/2], SSIM +0.014 [2/2],
extra flicker -0.59 (sd 0.11) [2/2]. At 480p a box that is not narrower than the
model input gains +0.55 dB (sd 0.04) [2/2].

Conclusion (moderate confidence on direction, low on size): cropping is the one
experiment with a gain well beyond the noise: 2 to 5 dB at 1080p and 4K, visibly
truer structure on the contact sheets (for example o1 frame 4, where the crop restores the shape of the
bright area behind the text much closer to the truth), less flicker, and the same model
cost (still one 640x120 inference per box). The gain is largest where the shrink factor drops furthest
(c2 from 3.0 to 1.2, k2 from 6.0 to 2.2), smaller where the text is wide (o2), and
small at 480p, where the band is barely shrunk anyway; it is not monotonic in the
shrink factor, so the footage matters too.

Findings that bear on any real implementation:

* Never enlarge the crop. When the crop had to be upscaled into the model (e1, shrink
  0.69) the model left text fragments and artefacts (core PSNR 23.7 against 32.2,
  sharpness ratio 11.9, ghost edge excess +3.1). The model does not behave the same
  on enlarged text and on text that is shrunk; the box must be at least as wide as
  the model input (`cropmin`). That also means the benefit is for frames of 1080p
  and above, or wide boxes.
* Not measured: the box here is one static box per clip (the two lines are centred,
  so the union box is also the per-segment box). A real implementation would pick a
  box per 25-frame segment (and per cluster of mask rows, so one box per distant
  subtitle line), which changes scale at segment boundaries, and the effect of that on
  temporal stability is unmeasured. Text that spans most of the frame width gets
  almost no gain (the box approaches the band).
* Interactions: the prediction cache key is `(start, end, top, bottom)` and would need
  the box's left and right edges; `get_inpaint_mode` and its band-shifting logic would
  give way to box selection; the windowed compositing and skipped windows on
  `perf/blend-box` and `perf/skip-windows` are built around row bands and would have
  to follow. This is a rewrite of the band and window logic, not a hook, which is why
  it stays a prototype.

## Not run or not conclusive

* Real-footage ghosting frames were not scored (no truth); the synthetic clips show none.
* Only two clips per resolution at 480p and 4K and four at 1080p; the SD across clips
  is large against some of the means above, as the tables say.
* Native 4K footage was not available; see above.
* Visual A/B is for the owner: contact sheets per question and clip, listed in the
  index written by `inpaint_quality_harness.py sheets` (kept outside the repository).

## Reproducing

```
python scripts/inpaint_quality_harness.py build --work DIR --clips e1,e2,c1,c2,o1,o2,k1,k2
python scripts/inpaint_quality_harness.py run   --work DIR --clips c1 \
    --variants current,area,area_cubic,area_lanczos,linear_cubic,masked_zero,masked_mean,masked_telea,seam
python scripts/inpaint_quality_crop_prototype.py --work DIR --clips c1 --modes band,crop
python scripts/inpaint_quality_report.py --runs DIR/runs --compact
python scripts/inpaint_quality_harness.py sheets --work DIR --out DIR/sheets --clips c1
```

`DIR` must be outside the repository. The weights are read from
`~/.videowipe/weights/sttn_*.onnx`. A full 8-clip run on a four-core machine takes
several hours when shared.
