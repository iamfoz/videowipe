# Sampling that scales with duration

Planning looks at a handful of sampled frames to decide where each candidate
(a subtitle band, a caption, a mark) is on screen, then checks every frame
inside the windows those samples mark present. Each frame takes the state of
its nearest sample, so a window the samples call absent is never checked.

Until now the sample count was fixed by the detect mode: 24 (`fast`), 50
(`balanced`), 80 (`sensitive`), whatever the length. On a 47 minute video
(70,000 frames) 50 samples are about 57 seconds apart, and a sample that lands
in a pause between two lines marks the whole stretch absent.

## The rule

```
count = max(mode count, ceil(duration / interval))
```

| mode | interval | the fixed count is kept for clips up to |
|------|----------|------------------------------------------|
| `fast` | never grows | any length (24 samples) |
| `balanced` | 1 s | 50 s |
| `balanced`, subtitle target | 1.25 s | 62 s |
| `sensitive` | 1.25 s | 100 s |

* `fast` does not refine, so extra samples buy it little, and it exists to be
  quick (measurements below).
* A mode that runs the forced band fallback (`sensitive`, or any mode asked for
  subtitles) spends seven detector calls per sample against three for the light
  fallback, so its interval is a quarter longer.
* Clips no longer than the table's last column are sampled exactly as before:
  the three ~20 s calibration clips produce the same plans and masks
  (see "Verification").
* Only `planning.prepare` applies the rule. A caller that hands
  `detect_clean_candidates` a `sample_count` gets exactly that many, and the
  legacy `detect_subtitle_mask` keeps its 30 whole frames.
* Duration is `frame_count / fps` from OpenCV, the same two numbers
  `plan.compute_source` records. Frames are still chosen by sequential ordinal;
  nothing is seeked. On a variable-rate source the average rate stands in for
  real time, so "seconds" are nominal there: a stretch where the source is
  slower than its average rate has fewer frames per second, hence fewer samples
  per real second.

### Switches

| variable | effect |
|----------|--------|
| `VIDEOWIPE_SAMPLE_INTERVAL` | seconds between samples for `balanced` (`sensitive` and the subtitle-target case scale it). `off`, `0` or any non-positive number restores the fixed counts exactly. Unset or unreadable: 1. |
| `VIDEOWIPE_SAMPLE_MEMORY_MB` | memory the extra samples may hold. Default: the smaller of 256 MB and a sixteenth of the memory the system reports as available (a container's limit counts). `0` or `off` removes the cap. |

### Caps

Time is capped by the rate itself: never more than one sample per interval, at
three detector calls per sample (seven with the forced fallback). That is at
most 3 / fps of the cost of detecting every frame once (12 % at 25 fps).

Memory is capped explicitly. Every detected sample keeps a greyscale appearance
thumbnail (at most 320 pixels wide) and its boxes; measured at 59 KB per
sample for 640x360 against an estimate of 62 KB. The cap is `budget / bytes per
sample` and never goes below the mode's own count. At the 256 MB default that
is 4,351 samples (72 minutes at one a second) for 16:9 video and 1,441 (24
minutes) for 9:16; beyond that the spacing widens rather than the memory
growing. The estimate counts only the thumbnails and boxes held by upstream
detection; the branches that also keep band crops (`VIDEOWIPE_BAND_CROP_BUDGET_MB`)
bound those separately.

## Evidence

All figures come from **synthetic** clips with exact ground truth, made by
`scripts/gen_long_clip.py` and scored by `scripts/eval_sampling.py`. No real
long clip with known timings was available, so these numbers show the mechanism,
not what any real film would lose. Details of the clips:

* background: moving, textured, text-free (ffmpeg `gradients` plus a `geq`
  plasma); 640x360 at 5 fps (the 2-minute clip at 10 fps), and one 1920x1080
  clip of 3 minutes;
* `dialogue`: lines of 1-5 s, gaps of 0.2-3.5 s, and long pauses of 20-120 s,
  one of which has no subtitle text at all; 44-49 % of the time on screen
  (61 % in the dense clip `d10`);
* `caption`: a top track visible 8-14 % of the time;
* `watermark`: a persistent top-right mark (upstream classes it as a logo to
  keep by default, so it is not scored as removal).

Score: a line is *found* when at least half of its interior frames have the
removal mask over at least half of the line's glyph rectangle.

### Upstream as it is (`VIDEOWIPE_SAMPLE_INTERVAL=off`), balanced

| clip | length | samples (gap) | lines found | frames covered | lost as | detector calls |
|------|--------|---------------|-------------|----------------|---------|----------------|
| c02 | 2 min | 50 (2.5 s) | 18 / 20 (90 %) | 95 % | window absent 2 | 1,567 |
| hd03 (1080p) | 3 min | 50 (3.8 s) | 23 / 26 (88 %) | 90 % | window absent 3 | 1,346 |
| c05 | 5 min | 50 (6.2 s) | 41 / 47 (87 %) | 90 % | window absent 6 | 2,572 |
| c10 | 10 min | 50 (12.4 s) | 82 / 102 (80 %) | 84 % | window absent 20 | 5,049 |
| d10 (dense) | 10 min | 50 (12.4 s) | 109 / 116 (94 %) | 94 % | window absent 7 | 8,802 |
| c20 | 20 min | 50 (24.6 s) | 162 / 198 (82 %) | 84 % | window absent 35, refine dropped 1 | 10,644 |

So the mechanism is real but milder than the worst case suggests: about a fifth
of the lines are lost on the long clips, and almost always at the same stage:
the coarse windows built from the samples call a stretch absent (*window
absent*: a covering remove track exists, its segments miss the line) and
refinement never visits it. The loss does not keep growing with length because
planning also probes a zone of half the widest sample gap around every segment
(`planning._add_spatial_segments`), which is why the detector-call count
grows faster than the length. Refinement itself dropped at most one line per run.

### Interval curve (balanced)

Lines found (detector calls in brackets):

| interval | c05 (5 min) | c10 (10 min) | c20 (20 min) | hd03 (3 min) |
|----------|-------------|--------------|--------------|--------------|
| fixed 50 | 41 / 47 (2,572) | 82 / 102 (5,049) | 162 / 198 (10,644) | 23 / 26 (1,346) |
| 8 s | | 84 / 102 (4,753) | | |
| 4 s | 41 / 47 (2,434) | 81 / 102 (4,216) | 170 / 198 (8,524) | 23 / 26 (1,346) (unchanged: 45 < 50 samples) |
| 2 s | | 93 / 102 (4,018) | | |
| **1 s** | **47 / 47 (2,099)** | **102 / 102 (4,171)** | **197 / 198 (8,309)** | **26 / 26 (1,171)** |
| 0.5 s | | 102 / 102 (5,193) | | |

The 3-5 s interval that first suggests itself does almost nothing: a line lasts
one to five seconds, so a sample is sure to land on it only when the samples
are about a second apart, and a line with no sample is lost unless a
neighbouring segment extends over it. Two seconds finds 91 %, one second finds
every line, half a second adds nothing and costs more. The total number of
detector calls at one second is **lower** than upstream's on every clip (c05
-18 %, c10 -17 %, c20 -22 %, hd03 -13 %, d10 -15 %, c02 -20 %): the extra
samples cost three calls a second of video, but the boundary probes shrink
with the sample gap by more than that.

`sensitive` (80 samples, forced fallback, seven calls per sample):

| | c05 | c10 | d10 |
|--|-----|-----|-----|
| fixed 80 | 43 / 47 (2,629) | 89 / 102 (5,661) | 104 / 116 (10,134) |
| 1.25 s (default) | 47 / 47 (3,175) | 100 / 102 (6,413) | 116 / 116 (9,891) |
| 0.625 s | 47 / 47 (4,322) | 102 / 102 (9,178) | |

so 1.25 s is enough for `sensitive`; going denser only adds calls.

`fast` (no refinement, consistency 0.50): with 45 % of the time on screen the
dialogue does not reach its threshold, so no candidate forms and nothing is
removed, sampled or not. The one exception is the 20 minute clip with the fixed
24 samples, where a candidate formed by chance and the coarse windows around it
(52 s apart) gave 834 false-removal frames; with denser samples it does not
form. On the dense clip (61 %) the
fixed 24 samples find 61 / 116 lines with 418 false-removal frames; 288 samples
(2.2 s) find 69 / 116 with 175; 144 find 53 with 213; 72 find 26 with 230. A
plan made of coarse windows gains too little to be worth the calls, so `fast`
does not grow.

### With the rule on (defaults), balanced

| clip | samples | lines found | frames covered | false-removal frames | detector calls |
|------|---------|-------------|----------------|----------------------|----------------|
| c02 | 120 | 20 / 20 | 99 % | 0 | 1,259 |
| hd03 | 180 | 26 / 26 | 99 % | 0 | 1,171 |
| c05 | 300 | 47 / 47 | 100 % | 0 | 2,099 |
| c10 | 600 | 102 / 102 | 100 % | 0 | 4,171 |
| d10 | 600 | 116 / 116 | 100 % | 93 (81 upstream) | 7,506 |
| c20 | 1,200 | 197 / 198 | 100 % | 0 | 8,309 |

False removals (frames with no text on screen, one frame of margin, where the
mask is non-empty away from the watermark): zero before and after on five
clips. On `d10` there are 81 upstream and 93 with the rule, out of 3,000 frames.
Both lists are runs of one to ten frames, and the 12 extra are isolated detector
false positives in pauses (for example inside the no-text stretch) that the
denser samples happen to meet; nothing is inherited across a gap (the plan only
ever keeps frames with their own positive evidence, as before). The mean
removed area outside on-screen text (a margin of 8 px around each line is
allowed) rises by 0.01-0.06 percentage points of the frame (for example 0.28 %
to 0.34 % on c10), in proportion to the frames now covered: it is the 14 px
side margin that every detection box carries.

### What the sample count does to the `consistency` threshold

The detector keeps pixels seen on at least `consistency` of the samples (0.40
balanced, 0.50 fast, 0.30 sensitive). For a track on screen a share *s* of the
time, more samples do not move the threshold, they sharpen it. Simulation on
the generator's timelines (`scripts/sim_sampling_consistency.py`, ground-truth
presence, so detector noise is excluded), chance that a track forms a
candidate at threshold 0.40:

| true share | 50 samples | 150 | 300 | 600 | 1,200 |
|------------|-----------|-----|-----|-----|-------|
| 0.34-0.38 | 9-13 % | 3-7 % | 0 | 0 | 0 |
| 0.40-0.42 | 35-47 % | 70-74 % | 75-79 % | 70-84 % | 95 % |
| 0.42-0.46 | 66-75 % | 94 % | 97-100 % | 100 % | 100 % |
| 0.46-0.52 | 93-94 % | 100 % | 100 % | 100 % | 100 % |

A rarely visible track becomes *less* likely to be a candidate by luck and a
track just over the line becomes much more reliably one: with 50 samples a
subtitle track on screen 42-46 % of the time is missed altogether a quarter to
a third of the time, with 300 or more it essentially never is. The rule does
not change the threshold itself, so a track below it is still never a
candidate. In the clips the sparse caption track (8-14 % of the time) formed no
candidate with any sample count (0 of 5, 11, 28 and 50 lines, in every
configuration); that is a separate limit of the frequency map, not of sampling
(see [sparse-tracks.md](sparse-tracks.md), which finds those tracks by another
route).

### Memory and time

Peak resident memory of the sampling stage alone, with a stub detector (so only
the thumbnails and boxes count) on the 20 minute 640x360 clip:

| samples | peak RSS |
|---------|----------|
| 50 | 80 MB |
| 300 | 94 MB |
| 1,200 | 148 MB |
| 4,350 (capped by the 256 MB default) | 335 MB |

Decoding is sequential and happens once per pass whatever the count, so the
wall time of the sampling stage with the stub was 10-12 s throughout.

End to end with the real detector, nothing cached, planning only (`prepare` and
`finalize`), on a machine shared with other jobs:

| clip | | detector calls | wall time | peak RSS |
|------|--|----------------|-----------|----------|
| c05 (5 min) | fixed 50 | 2,572 | 286 s | 229 MB |
| | rule (300 samples) | 2,099 | 236 s | 251 MB |
| hd03 (1080p, 3 min) | fixed 50 | 1,346 | 185 s | 327 MB |
| | rule (180 samples) | 1,171 | 164 s | 336 MB |

The other clips were run with a content-addressed detector cache (the evaluator
reports logical calls either way), so only their call counts are comparable.
Detector latency was 0.11-0.28 s per call depending on the load on the machine.

The branches that keep more per sample change the picture. On a tree with all
of them merged, the band crops kept by `perf/detector-calls` (bottom 40 % of
each sampled frame, within `VIDEOWIPE_BAND_CROP_BUDGET_MB`) took the 20 minute
clip's peak up by about 310 MB at 1,200 samples (181 to 492 MB in a cached run);
its own budget bounds them, and past it the file is re-read as before.
`fix/ocr-crop` keeps one crop per text-less box with no budget, and with a
recogniser installed that now scales with the sample count: two subtitle rows at 1080p cost about 250 KB per
sample there (435 MB against 215 MB at 900 samples). `VIDEOWIPE_SAMPLE_MEMORY_MB`
does not see that, so lower it or cap those crops if OCR is in use on long
1080p video.

## Verification

Run on `feat/duration-sampling` (upstream `20872d4` plus this change):

* `python -m pytest -q -p no:cacheprovider`: 560 passed, 6 skipped (upstream
  500 passed, 6 skipped; the 60 new tests are `tests/test_sampling.py` and
  `tests/test_long_clip_generator.py`).
* The three ~20 s calibration clips are sampled as before: for the six decision
  cases (`chinese1` default and watermark, `english1`, `others` default,
  subtitle and watermark) the plans (`wipe_plan.json`) and masks
  (`wipe_plan_masks.npz`) written by upstream and by this branch are
  byte-identical, with the same detector call counts and no call on an image
  upstream had not seen.
* `make fact-baseline` and `make decision-baseline`, upstream then branch:
  every metric identical.

  | fact baseline (macro average) | upstream | branch |
  |-------------------------------|----------|--------|
  | frame_remove_union_region_jaccard | 0.320324 | 0.320324 |
  | frame_remove_union_boundary_f | 0.458288 | 0.458288 |
  | visible_remove_object_region_jaccard | 0.320403 | 0.320403 |
  | visible_remove_object_boundary_f | 0.625185 | 0.625185 |
  | visible_keep_prediction_coverage | 0.0 | 0.0 |
  | no_remove_false_removal_area_ratio | 0.010517 | 0.010517 |
  | visible_annotation_semantic_match_rate | 1.0 | 1.0 |
  | visible_annotation_selection_intent_match_rate | 1.0 | 1.0 |

  All 893 fields of the decision report are equal (case_exact_action_match_rate
  1.0, mixed_object_candidate_rate 0.125, the rest 1.0 or 0.0 as before); in the
  fact report the only differing fields are the random run directory in the
  preview file names (the content hashes after it are equal).
* Merge: the branch merges cleanly (`git merge-tree --write-tree`, exit 0) with
  each of `feat/i18n-english`, `fix/webm-frame-count`, `feat/h265-output`,
  `fix/progress-display`, `perf/refine-exact`, `perf/server-exact`,
  `perf/plan-masks`, `perf/mps`, `perf/instrumentation`, `perf/detector-calls`,
  `perf/blend-box`, `perf/skip-windows`, `perf/recovery-reuse`,
  `fix/output-fidelity` and `fix/ocr-crop`, and with all 15 merged one after
  another in a scratch worktree: 1,295 passed, 7 skipped. On that tree the
  evaluator behaves the same: c05 balanced goes from 41 / 47 lines (fixed) to
  47 / 47 (rule) with 0 false-removal frames in both, c20 finds 197 / 198.

## Reproducing

```
python scripts/gen_long_clip.py --out clips --name c10 --duration 600 --seed 13 --fps 5
PYTHONPATH=src python scripts/eval_sampling.py clips/c10.mp4 --mode balanced \
    --sample-interval off --detector-cache clips/c10.pkl      # upstream's fixed count
PYTHONPATH=src python scripts/eval_sampling.py clips/c10.mp4 --mode balanced \
    --detector-cache clips/c10.pkl                            # the rule
```

(`gen_long_clip.py` options: `--duration`, `--fps`, `--size`, `--seed`,
`--pause-share`, `--two-line-share`. The clips in this note: c02 `--duration 120
--seed 11 --fps 10`; c05 `300`, seed 12; c10 `600`, seed 13; c20 `1200`, seed
14; hd03 `180`, seed 15, `--size 1920x1080`; d10 `600`, seed 21, `--pause-share
0.08`; all at `--fps 5` unless stated.)

## Limits

* Synthetic ground truth only: white outlined text, one font, 22 px at 360p,
  lines of 1-5 s. Real subtitles shorter than the interval can still fall
  between samples.
* One detector (the shipped DBNet weights, CPU).
* Subtitle-light videos pay for the samples they do not need: three detector
  calls per second of video (seven with the forced fallback) whether or not
  there is text. An hour with almost no subtitles costs about 10,800 calls that
  upstream's 150 did not. `VIDEOWIPE_SAMPLE_INTERVAL` trades this against recall.
* Two-row subtitles: in an earlier version of the clips with two-row lines, the
  upper row fell outside the candidate (the frequency map reaches the
  threshold only on the shared bottom row) and the line was only half covered
  however it was sampled. Orthogonal to sampling; not changed here.
* The 0.40 / 0.50 / 0.30 thresholds are unchanged; a subtitle track on screen
  less than that share of the time is still never a candidate of the frequency
  map. [sparse-tracks.md](sparse-tracks.md) adds a second route that admits such
  tracks on recurrence.
