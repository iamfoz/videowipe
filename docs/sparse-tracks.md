# Subtitle and caption tracks that are on screen only rarely

A text region became a candidate only if the detector's boxes covered a pixel
on at least `consistency` of the sampled frames: 0.50 (`fast`), 0.40
(`balanced`), 0.30 (`sensitive`). A track on screen a smaller share of the
time never reached that, at any number of samples, so it was never a candidate
and none of its lines was ever removed. `sparse_tracks` admits such a track on
recurrence alone. Everything below comes from synthetic clips with exact ground
truth and from the three real calibration clips; there is no real long footage
with known timings (see "Limits").

## Result

* Today, balanced: **0 of 609** lines of the 18 sparse tracks measured (5-38 %
  of the time on screen) are found. With the rule: **609 of 609**, and
  false-removal frames are the same on every clip (0 on eleven, 93 on `d10`,
  the same 93 frames as before).
* Sensitive finds tracks from 30 % up already (its consistency): `sD`, 135 of
  144 lines, with or without the rule. Below that it matches balanced: `sA`
  0 of 31 today, 31 of 31 with the rule. `fast` finds none and still finds none:
  it does not refine, and forced on it removes whole windows (see below).
* On by default (`sparse_tracks.DEFAULT_ON`); `VIDEOWIPE_SPARSE_TRACKS=off`
  restores the previous behaviour exactly. The gates held (table below).
* The mean removed area outside on-screen text (an 8 px margin allowed) rises
  because the newly removed lines carry the 14 px side margin every box has: from
  0 to 0.0009-0.0041 of the frame on clips with only sparse tracks, by 0.0003 to
  0.0016 on clips that had a dense track (which already ran at 0.0016-0.0042).
* Cost: more detector calls, because each track's windows are checked frame by
  frame like a dense track's (+10 % to +30 % on clips that already had a dense
  track, up to 5.6 times on a clip that had nothing else).

## What was lost, with today's code

Clips come from `scripts/gen_long_clip.py` (new: `--sparse NAME:POSITION:SHARE`,
`--no-dense`) and are scored by `scripts/eval_sampling.py`: moving textured
text-free background, white outlined dialogue at the bottom, yellow captions at
the top, cyan `as` track in the row above the dialogue (the same subtitle
band), 10 minutes at 640x360, 5 fps (`hd03` is 3 minutes at 1080p). A line is
*found* when half its interior frames carry the mask over half its glyph
rectangle; a *false-removal frame* has no text within a frame of it and a mask
away from the watermark. `sA` to `sD` have only sparse tracks (`sd` at the
dialogue row, `tc` at the top, shares 5/10, 15/20, 25/30, 35/38 %); `sF` and `sG`
add `as` at 10 % and 25 % to dense dialogue and the built-in captions; `c02` to
`d10` and `hd03` are the clips of [sampling.md](sampling.md).

Balanced, lines found by on-screen share (today / with the rule):

| clip | track | on screen | today | with the rule |
|------|-------|-----------|-------|---------------|
| `sA` | `sd` | 5.2 % | 0 / 11 | 11 / 11 |
| `hd03` | captions | 9.4 % | 0 / 5 | 5 / 5 |
| `sF` | `as` | 9.9 % | 0 / 20 | 20 / 20 |
| `sA` | `tc` | 10.1 % | 0 / 20 | 20 / 20 |
| `c02` | captions | 10.8 % | 0 / 5 | 5 / 5 |
| `c05` | captions | 11.1 % | 0 / 11 | 11 / 11 |
| `c20` | captions | 12.2 % | 0 / 50 | 50 / 50 |
| `d10` | captions | 13.1 % | 0 / 26 | 26 / 26 |
| `sF` | captions | 13.1 % | 0 / 26 | 26 / 26 |
| `c10` | captions | 13.8 % | 0 / 28 | 28 / 28 |
| `sB` | `sd` | 14.7 % | 0 / 30 | 30 / 30 |
| `sG` | captions | 15.2 % | 0 / 31 | 31 / 31 |
| `sB` | `tc` | 20.3 % | 0 / 40 | 40 / 40 |
| `sC` | `sd` | 25.3 % | 0 / 50 | 50 / 50 |
| `sG` | `as` | 25.3 % | 0 / 51 | 51 / 51 |
| `sC` | `tc` | 29.8 % | 0 / 61 | 61 / 61 |
| `sD` | `sd` | 35.2 % | 0 / 68 | 68 / 68 |
| `sD` | `tc` | 37.9 % | 0 / 76 | 76 / 76 |
| `c02` ... `c20`, `hd03` | dense dialogue | 44-49 % | 392 / 393 | 392 / 393 |
| `sF` | dense dialogue | 60.8 % | 105 / 123 | 121 / 123 |
| `d10` | dense dialogue | 61.3 % | 116 / 116 | 116 / 116 |
| `sG` | dense dialogue | 62.2 % | 94 / 120 | 120 / 120 |

False-removal frames, balanced, today / with the rule: 0 / 0 on `sA` to `sD`, `sF`, `sG`, `c02`, `c05`, `c10`, `c20` and `hd03`, and 93 / 93 on `d10` (the same 93 frames).
The dense tracks (44-62 %) are as before in every clip but `sF` and `sG`, where
libass pushes dialogue lines up into the `as` row while an `as` line is on
screen: 18 and 26 dialogue lines sat outside the dense candidate and were lost
(`no_candidate`), and the new `as` candidate now covers them.

Sensitive and fast, same scoring:

* `sensitive` (480 samples, seven detector calls each): `sA` 0 / 31 lines today,
  31 / 31 with the rule, 0 false-removal frames either way; `sD` (tracks at 35 and
  38 %, above its 0.30) 67 / 68 and 71 / 76, the same with the rule (the five
  lines lost are `refine_dropped` and `window_absent`, not this work);
  `sF` captions 0 / 26 to 26 / 26, `as` 0 / 20 to 20 / 20, dense dialogue 105 / 123
  to 122 / 123 (the lines pushed into the `as` row), no false-removal frames
  either way.
* `fast` (24 samples, no refinement): none of the 12 sparse tracks of `sA` to
  `sD`, `sF` and `sG` is found, today or with the rule (it is excluded); `sF`'s
  dense dialogue is found 58 / 123 with 346 false-removal frames, `sG`'s
  0 / 120, today and with the rule alike.

## Where each missed track fails

Not the detector: on every sample on which a sparse line is on screen it put a
box on it (100 % in the table). Not the connected-component step, `_classify_region`
or the decision: a candidate made from the same boxes is classified as a
subtitle (`wide bottom text`, `transient wide top text`) with `default_remove`
set and is then removed. The evaluator agrees: every lost line is
`no_candidate`, never `not_removed`, `window_absent` or `refine_dropped`. The
loss is the frequency threshold alone; the peak per-pixel frequency of a track
is about its share, since lines of different lengths share only their middle:

| track | share | samples on | detected | peak pixel frequency | clears 0.50 / 0.40 / 0.30 |
|-------|-------|------------|----------|----------------------|----------------------------|
| `sA` `sd` | 5.2 % | 32 | 100 % | 0.053 | no / no / no |
| `sA` `tc` | 10.1 % | 60 | 100 % | 0.100 | no / no / no |
| `d10` caption | 13.1 % | 81 | 100 % | 0.143 | no / no / no |
| `sB` `sd` | 14.7 % | 88 | 100 % | 0.147 | no / no / no |
| `sB` `tc` | 20.3 % | 123 | 100 % | 0.205 | no / no / no |
| `sD` `sd` | 35.2 % | 210 | 100 % | 0.350 | no / no / yes |
| `sD` `tc` | 37.9 % | 226 | 100 % | 0.377 | no / no / yes |
| `c10` dialogue (dense) | 48.0 % | 290 | 100 % | 0.483 | no / yes / yes |

More samples do not help ([sampling.md](sampling.md#what-the-sample-count-does-to-the-consistency-threshold)):
the peak converges on the share. Tracks at 35-38 % pass only in `sensitive`.

## The rule

`src/videowipe/sparse_tracks.py`, called from `detect_clean_candidates` after
the frequency-map candidates, before the band fallback (whose duplicate filter
then sees them). It uses only the boxes the detector reported on the sampled
frames.

1. Take every box except those inside an existing non-subtitle candidate or on
   the rows of an existing subtitle candidate (the halves of a long dense line
   fall outside its narrow region), and those narrower than 6 % of the frame or
   shorter than 6 px.
2. Two boxes *agree* when their vertical centres are within 3 px or a quarter of
   their height, their heights within a factor of 1.43, and the shorter lies at
   least half inside the other's horizontal extent (lines of different lengths in
   one row overlap in the middle; text in different places does not agree).
3. From the box that agrees with the most distinct samples, take its agreeing
   boxes as one track; repeat on the rest.
4. A track qualifies with boxes on at least `max(8, ceil(sqrt(n)))` of the `n`
   samples, in at least three separate stretches of consecutive samples, and on
   more samples than the same boxes spread evenly over the frame would put in
   that row plus four standard deviations. Noise recurs in proportion to the
   number of samples, so the threshold grows with `n`; the last term makes a
   video on which the detector reports many boxes demand more of a row.
5. It becomes a candidate only if `_classify_region` calls it a subtitle with
   `default_remove` (the test a dense track passes) and its appearance changes
   from sample to sample (`_appearance_stability` under 0.80): a static graphic
   that keeps coming back is a logo or watermark, left to the overlay rules.

For `n` = 600 that is 25 samples (4.2 % of the time), for 1,200 it is 35.
Presence is the samples on which one of the track's own boxes was found
(`presence_frames`). The same evidence rule (`CleanCandidate.evidence_filter`)
decides which boxes count in refinement (`refine_temporal_presence`) and which
extend a segment or make up its mask (`_add_spatial_segments`): a box of another
height or row inside the region is not evidence. Before the rule was applied
in `_add_spatial_segments` as well, `d10` gained 9 false-removal frames next to
caption lines (93 to 102); with it, none. Nothing is carried across a gap (C2); every frame is marked on a
box of the track found on that frame, and frames are still identified by
sequential read (C1).

`fast` never uses it (`planning.prepare` passes `sparse_tracks_on=False`): it
does not refine, so the coarse windows of a track found from 24 samples would be
removed whole. Forced on (24 samples, `fast` today / forced): `sD` 0 / 457
false-removal frames, `c05` 0 / 185 (it removes the 45 % dialogue, 32 of 47
lines, but over whole windows), `sC` and `d10` unchanged.

Switch: `VIDEOWIPE_SPARSE_TRACKS`, read on every call. `off` (`0`, `false`,
`no`) restores the previous behaviour exactly, `on` (`1`, `true`, `yes`) enables
it, unset or anything else is `sparse_tracks.DEFAULT_ON`.

## Gate

| check | result |
|-------|--------|
| three calibration clips, six decision cases (`chinese1` default and watermark, `english1`, `others` default, subtitle, watermark): plans (`wipe_plan.json`) and masks (`wipe_plan_masks.npz`) against `origin/main` | byte-identical in `fast`, `balanced` and `sensitive`, with the switch unset, `off` and `on`, with the same detector calls and none on an unseen image |
| `make fact-baseline` | all 684 fields equal `origin/main` (macro average: frame union Jaccard 0.320324, boundary F 0.458288, visible remove Jaccard 0.320403, boundary F 0.625185, keep coverage 0.0, no-remove false area 0.010517, semantic and intent match 1.0) |
| `make decision-baseline` | all 893 fields equal (exact action match 1.0, mixed object rate 0.125, the rest 1.0 or 0.0 as before) |
| false-removal frames | equal on every clip: 0 on eleven, 93 on `d10` (the same frames) |
| `off` | the same plans, masks, baselines and detector calls as `origin/main` |
| real footage with no truth | a text-free noisy clip (`n10`, 10 minutes) gives no track; see below for the spliced real clip |

A spliced clip of the three real calibration clips repeated seven times
(`real7`, 7 minutes, 1280x720, so each clip's rows are on screen a third of the
time) gives six recurring rows: the three real subtitle rows are admitted, the
three static overlays (appearance stability 0.96-0.99, against 0.005-0.10 for
the subtitle rows) are not. Only the candidates were made for it (planning that clip frame by frame was not run).

## Tests

`tests/test_sparse_tracks.py` (the rule on boxes; candidates, their evidence
rule and the static-graphic and classification exclusions; the whole pipeline on
a small lossless video with a stub detector: only the frames with a box are
removed, the one-off box and the strays are not, `off` is exactly today, `fast`
never uses it) and `tests/test_sparse_generator.py` (shares, determinism,
independence from the dialogue draws, styles). Mutation check: each of 18
changes (episode requirement, support growth, floor, chance term, width filter,
horizontal agreement, skipping existing candidates, presence on every sample,
no per-frame rule, switch default and both explicit settings, classification,
row position, static graphics, refinement ignoring the rule, `fast` not
excluded, boundary extension ignoring the rule) makes at least one of them fail.

## Limits and risks

* The evidence is synthetic apart from three 20 s real clips (unchanged by the
  rule) and one spliced clip without ground truth. A real film with a busy
  textured background may report wide boxes the synthetic noise (boxes under
  6 % of the width) does not; the `chance_support` term and the episode
  requirement are the guard, and `VIDEOWIPE_SPARSE_TRACKS=off` the escape.
* A track needs about 4 % of the time at 600 samples (8 samples at the least,
  so at 50 samples a track needs about 16 %) and at least three separate
  appearances; a caption shown twice is not admitted.
* A logo or watermark whose appearance changes (animated, or over changing
  video) and that is on screen only part of the time is a "transient wide top
  text" and is removed, as a dense overlay on screen 40-79 % of the time is
  today.
* A track the classifier would not call a subtitle is not admitted: a region
  under 15 % of the frame wide at the top or bottom, text in the middle of the
  frame unless it is wide and thin, scene text, and anything that reads as a
  timestamp or watermark.
* The review path (`refine_review`) rebuilds candidates from saved evidence,
  which does not carry the evidence rule, so a sparse track selected there is
  refined with the plain overlap test.
* Detector calls: sparse tracks are refined frame by frame like dense ones
  (`d10` 7,221 to 7,912; `c20` 8,039 to 10,440; `sA` 1,800 to 3,618; `sD`
  1,800 to 10,142).

## Reproducing

```
python scripts/gen_long_clip.py --out clips --name sA --duration 600 --fps 5 --seed 31 \
    --no-dense --sparse sd:bottom:0.05 --sparse tc:top:0.10
python scripts/gen_long_clip.py --out clips --name sF --duration 600 --fps 5 --seed 36 \
    --pause-share 0.08 --sparse as:aside:0.10
for switch in off on; do
  VIDEOWIPE_SPARSE_TRACKS=$switch PYTHONPATH=src python scripts/eval_sampling.py clips/sA.mp4 \
      --mode balanced --detector-cache clips/sA.pkl --output sA_$switch.json
done
```

Seeds: `sA` 31, `sB` 32, `sC` 33, `sD` 34 (`--no-dense`, shares as above), `sF` 36
and `sG` 37 (`--pause-share 0.08`, `as:aside:0.10` and `0.25`). The report lists
every track with its `on_screen_share` and the false-removal frames as runs
(`false_removal_runs`).
