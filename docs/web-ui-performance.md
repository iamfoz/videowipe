# Web UI responsiveness (PRD items U-1 to U-4)

Every change here is a speed-up only. The page shows the same thing, the
server sends the same states in the same order, and the translator returns the
same strings. Section 5 is the evidence for that; sections 1 to 4 are the
measurements.

| Item | What was slow | Shipped | Branch |
| --- | --- | --- | --- |
| U-1 | `renderTimeline` rebuilt one element per segment on every call | Rebuild only when the segments change; skip same-value text writes | `perf/web-ui` |
| U-2 | Overlay refresh looped over 2 million pixels in JavaScript | Tint with a canvas colour-matrix filter, guarded by a probe | `perf/web-ui` |
| U-3 | Full job snapshot sent every 0.5 s, every one re-run through the page | Send on change, with a 15 s keep-alive comment; skip same-value text writes in `renderActions` | `perf/web-ui` |
| U-4 | `translate()` rebuilt and recompiled the pattern list on every call | Compile once per catalogue, memoise results | `perf/i18n-memo` (on `feat/i18n-english`) |

## Method

* Headless Chromium through Playwright 1.63, 1440 x 1000, device pixel ratio 1.
* A harness serves the real `index.html` and the real `/jobs/current`,
  `/jobs/{id}` and `/jobs/{id}/progress` routes, with the review data stubbed:
  a 1920 x 1080 source of 90 minutes at 25 fps (135,000 frames), three tracks,
  and a feathered 1080p mask PNG. "551 segments" is 400 + 150 + 1 across the
  tracks; "5,001" is 3,000 + 2,000 + 1.
* Main-thread cost is the Chrome DevTools `TaskDuration` metric per call, with
  one frame between calls, so it includes style, layout and paint, not only
  script. An idle frame costs 0.3 to 0.5 ms.
* Each figure is the median of five runs, alternating before and after. The
  machine has four cores shared with other jobs, so expect about 20 % scatter;
  differences of that size are not claimed.

## 1. U-1: timeline

`renderTimeline` runs on every slider input, every bounding-box drag step, and
every edit. Script time was small (1 to 2 ms for 551 segments); the cost was
style, layout and paint of one `<span>` per segment.

| Main-thread ms per call | 551 segments | 5,001 segments |
| --- | --- | --- |
| `renderTimeline()`, segments unchanged: before | 10.3 | 59.9 |
| after | 0.7 | 0.7 |
| `start` slider input event: before | 11.3 | 59.2 |
| after | 3.1 | 9.6 |

What was done:

* The presence bars are rebuilt only when the frame count or the selected
  tracks' segments change. They depend on nothing else, and the elements
  produced are the same ones as before.
* The two timeline labels and the action-panel text are written only when the
  value differs from the last one written. Writing the same string again still
  replaces the text node, re-lays-out the element and repaints, and that
  repaint grows with the number of bars: 8 ms per call at 5,001 segments came
  from two label writes alone.

A slider event still costs 3 to 10 ms; the rest is `renderActions`, `scheduleMask`
and the browser's own input handling.

A real edit to the segments (apply, reset, undo, a track ticked or unticked)
still rebuilds the bars and costs what it did before: about 10 ms at 551
segments and 60 ms at 5,001. It now happens once per edit rather than on every
input event.

**Not shipped: one `<canvas>` or merged intervals.** Each bar is drawn at 35 %
opacity, so where tracks overlap the bars accumulate and look darker. Merging
the intervals would change that picture, and a canvas would have to reproduce
Blink's pixel snapping and anti-aliasing of 2 px rounded bars to stay
identical. The rebuild is now rare, so the saving would be small.

## 2. U-2: mask overlay

Each refresh drew the mask PNG, read the 1080p canvas back, set 2 million
pixels to orange with the mask level as alpha in a JavaScript loop, and wrote
it back. A canvas colour-matrix filter (`ctx.filter` with an SVG
`feColorMatrix`, A' = R, RGB constant) does the same while the mask is drawn.

| `refreshMask()` end to end | before | after |
| --- | --- | --- |
| Main-thread ms per refresh | 26.4 to 28.2 | 6.7 to 7.8 |
| CPU in all browser processes, ms | 35.5 to 37.7 | 25.0 to 26.7 |
| The tint step alone (micro-benchmark) | 17 ms | 4 ms |

Main thread is the number that matters for responsiveness; total CPU falls less
because the filter still runs, in the compositor. With a GPU the saving should
be larger, since the read-back disappears.

The result is byte-identical, not merely close: over the whole 1080p mask and
over a strip holding every level 0 to 255, `getImageData` is the same for the
loop and the filter (0 differing bytes). Because browsers differ, the first
draw runs a probe: the 256 levels are tinted both ways and compared, and the
original loop is used if the filter is missing or gives any different byte.

## 3. U-3: progress stream

### Server

The stream serialised and sent the whole job snapshot every 0.5 s. It now
sends a snapshot only when its serialised form differs from the last one sent,
and writes an SSE comment (`: keep-alive`) after 15 s of silence so the
connection and any proxy stay alive. The poll interval is unchanged, so a
change still reaches the page within 0.5 s, and the final state is still the
last event.

Idle job (state `running`, nothing changing), 60 s, real route, 1.9 KB snapshot:

| | before | after |
| --- | --- | --- |
| Data messages per minute | 120 | 1 |
| Keep-alive comments per minute | 0 | 3 |
| Bytes per minute | 223,680 | 1,906 |

The page also stops receiving 120 messages a minute, each of which cost about
3.1 ms of main thread at 551 segments (about 9 ms at 5,001).

Comparing serialised text rather than the dictionaries is deliberate: parts of
the snapshot (`overrides`, `previous_review`) are shared with the job and can be
changed in place, which an equality test on the same object would not notice.

### Page

A progress message made 44 DOM mutations. Only the text writes were expensive;
assigning an unchanged boolean or attribute is free.
`renderActions` now skips repeat text writes and `applyStatus` skips an
unchanged file name:

| Main-thread ms per progress message | 551 segments | 5,001 segments |
| --- | --- | --- |
| before | 3.1 | 8.7 |
| after | 2.4 | 6.5 |
| `renderActions()` alone, 551 segments | 3.9 | 1.0 |

The remaining cost is the `busyHint` lines in `applyStatus`, which are written
several times per message. They were left alone because `fix/progress-display`
rewrites exactly those lines; that branch could take the same treatment once
both are merged. With the idle stream silent this matters only while a job is
actually progressing.

The `put()` helper remembers the last value written rather than comparing with
the element's current text. Under the English overlay the element no longer
holds the Chinese source text, so a comparison would never match and the skip
would be lost (`renderActions` stayed at 2.4 ms with a comparison, and fell to
0.6 ms with the memory). It is only valid for elements that nothing else
writes, which is true of the ones it is used for; the file name, which several
functions set, is compared with the element's own text instead.

## 4. U-4: translation overlay

`translateCore` rebuilt and sorted the pattern list from the whole catalogue
(252 messages, 26 patterns) and compiled a regular expression for every pattern
it tried, on every call, and the page translates every text change. On
`perf/i18n-memo` the list, the compiled expressions and the fragment-pass
candidates are built once per catalogue (`WeakMap`), and results are memoised
(bounded to 2,000 entries, because counters rarely repeat).

Per call, in Chromium, English:

| String | before | after |
| --- | --- | --- |
| Exact catalogue hit (`检测目标`) | about 0.3 us | about 0.3 us |
| Pattern, seen before (`当前阶段 25%`) | 40 us | 0.3 us |
| Pattern, new each time (`当前阶段 17%`) | 38 to 42 us | 4 us |
| Counters (`1,234 / 4,000 帧`) | 74 to 90 us | 14 us |
| Fragment (`前缀 预测 3/12 帧 后缀`) | 249 to 265 us | 17 to 27 us |

Per UI update, English against the unmodified page, 551 segments:

| Main-thread ms | before | after |
| --- | --- | --- |
| `renderTargets()`, JavaScript only, including the observer | 10.5 to 12.8 | 1.3 to 1.5 |
| `renderTargets()`, with layout and paint | 19.3 | 8.6 |
| One progress message, JavaScript only | 0.7 to 0.9 | 0.18 to 0.24 |

Time to a loaded page did not change measurably (223 against 266 ms in one
pairing, 217 against 213 in the other; within the scatter). The Python
`translate()` costs 130 to 850 us per call but is only used for command-line
messages, once each, so it was left as it was.

## 5. Evidence that nothing visible changed

* **DOM.** A 34-step trace (15 status messages covering every state, then 19
  local changes: mask on and off, result stale, selection, segment edits, frame
  count, source cleared, review unloaded) records the full DOM after each step.
  It is identical before and after in all 34 steps, in Chinese and in English
  (English compares the original page and translator with the changed ones).
* **Mask.** The overlay canvas pixels hash the same in all seven screenshot
  states, before and after.
* **Screenshots.** Seven states, three runs each side. The stage, the actions
  panel and the target list are pixel-identical, apart from one-level (1/255)
  noise on a handful of pixels that also appears between two runs of the
  original. The timeline strip differs by
  at most 6 of 255 on a few hundred pixels in two states. The same difference
  appears in the original page when the *same* bars are re-inserted without any
  change (338 pixels, maximum 6): Chrome's raster of the anti-aliased edges of
  the 2 px rounded bars depends on the history of invalidation. The first
  rendering compared like for like differs by 3 pixels, maximum 1, the same as
  two runs of the original.
* **Progress stream.** `tests/test_progress_stream.py` checks that every state
  change reaches a consumer in order and equals the snapshot at that moment,
  that no two consecutive events are identical, that the final state is last
  and ends the stream (for each of the five final states), that an idle job gets
  one event and then comments only, and that a change after a quiet spell still
  arrives.
* **Translator.** `tests/test_i18n_memo.py` runs the translator from before the
  change (`tests/fixtures/i18n_unmemoised.js`) beside the new one over the whole
  catalogue (plain, padded, punctuated, doubled), every pattern instantiated
  with samples and with other catalogue entries, concatenations, fragment
  cases, progress-style strings and the shared vectors, three passes each so the
  later ones come from the cache, and over three pathological catalogues. About
  7,800 comparisons, no differences. It fails if the pattern sort is reversed.

## Not covered

* Firefox and Safari. Only Chromium was run; the probe in section 2 exists so
  that another browser falls back instead of differing.
* GPU rasterisation: headless Chromium here rasterises in software.
* A real engine run end to end; the harness stubs the review data, though it
  uses the real job, snapshot and stream code.
* The `busyHint` writes in `applyStatus` (see section 3).
