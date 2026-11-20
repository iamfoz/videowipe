# PRD: performance and quality improvements

Status: proposal · Scope: whole application · Focus: Apple Silicon, but not only

This document records an audit of VideoWipe's detection, planning, inpainting,
encoding, server and web UI, and turns the findings into requirements that can
be implemented one at a time. Each requirement states the evidence, the change,
how to prove it worked, and how confident the finding is.

## 1. Summary

There **are** worthwhile improvements. In order of expected value:

1. **Long videos wait too long before a trial is possible.** The "confirming
   appearance times" (refine) pass decodes the whole file, runs up to 14
   detector passes on every text-free frame inside a candidate window, and has
   quadratic Python bookkeeping. Most of this can be removed without changing
   results (§6, P1-1, P1-2) and more with baseline-checked changes (P2-3, P2-4).
2. **Inpainting is the dominant cost of a clean** (75–84% of a run in our
   profile). On Apple Silicon it runs in 32-bit float; half precision on the
   Metal (MPS) backend is the one large untried lever (P2-1). Skipping model
   windows that cannot affect any output pixel is a second, exact one (P2-2).
3. **Output fidelity has real gaps**: variable-frame-rate sources are written
   at a constant rate (audio drift risk — relevant to YouTube WebM), audio is
   always re-encoded, metadata/chapters/subtitle tracks are dropped, colour
   tags are not carried, and 10-bit/HDR is silently reduced to 8-bit (§8).
4. **A likely quality problem on long videos**: only 50 frames are sampled
   whatever the length, so on a 47-minute file each sample decides ~57 seconds
   of video and subtitle lines in "absent" windows are never examined (Q-6).
5. **The web UI does avoidable server work during playback** (a full mask
   render about four times a second, under the lock that progress updates
   need) and re-decodes from frame 0 for each target thumbnail (P1-3, P1-4).

Several plausible-sounding ideas are **not** recommended (§10): Core ML for the
inpainting model, `torch.compile`, hardware video decode, VideoToolbox for the
preview copy, and tuning CPU threads. Upstream has already measured some of
them, and the profile shows others cannot matter.

## 2. Evidence base and its limits

| Source | What it gives | Limit |
|---|---|---|
| Code reading of every module (four parallel audits) | What the code does, with file and line | No execution on a Mac |
| cProfile of real cleans on a 150-frame 1080p clip (Linux, 4-core Xeon, ONNX Runtime on CPU) | The shape of the cost profile | Absolute numbers do not transfer to MPS |
| Micro-benchmark of the refine bookkeeping | Confirms quadratic growth | Synthetic |
| Upstream's own measurements in `plans/STTN_INFERENCE_SPEED.md` | MPS timings, things already tried | Upstream's hardware |
| A user report: 70,199-frame 1080p VP9 WebM on Apple Silicon, MPS | The motivating case | One file |

Line numbers refer to branch `feat/i18n-english`; in `server/app.py` they are
two higher than upstream `20872d4`, elsewhere identical. Every claim marked
**Verified** was read in the code; **Hypothesis** needs measurement before work
starts. Nothing here was measured on Apple Silicon — §4 defines the baseline
run that must come first.

### Where the time goes (measured, CPU ONNX, 150 frames at 1080p)

| Stage | Share of run |
|---|---|
| STTN inference (transformer ≈ 98 s, decoder ≈ 90 s, encoder ≈ 2 s) | 75–84% |
| Text detection (245 OpenCV DNN forward passes, ~90 ms each) | ~12% |
| Video decode (1,023 frame reads for a 150-frame clip: 8 readers) | ~1% |
| Compositing | ~1–2% |
| Pipe write and ffmpeg wait | <1% |
| SHA-256 hashing | ~0% on a 2 MB clip |

Two caveats when carrying this to the Mac. First, MPS inference is roughly
15–20× faster than this CPU (upstream: ~1.75 s per 25-frame segment), so the
non-inference stages grow from ~15% to a much larger share and are worth more
there than this table suggests. Second, decode was H.264 at 3 ms a frame; VP9
at 1080p is slower, and a 70,199-frame file multiplies every extra pass.

One further measured fact: `--gap 50` is **43% slower per frame** than the
default 25 (1.76 s against 1.23 s), because reference frames scale with segment
length. The default is right; do not raise it for speed.

## 3. Goals, non-goals and constraints

**Goals**

- G1. Cut the wait between upload and first possible trial on long videos.
- G2. Cut full-clean time on Apple Silicon.
- G3. Stop the output differing from the source in ways the user did not ask
  for (timing, audio, colour, metadata).
- G4. Keep the web UI responsive on long videos.

**Non-goals**

- Replacing the STTN model or adding a second inpainting model.
- Changing upstream's review workflow (trial and export wait for refinement).
  Upstream explicitly decided not to "refine only the trial region"
  (`plans/PRODUCT_EXPERIENCE_DELIVERIES.md:89`): it affects global candidate
  state and the inference crop bands. This PRD makes the refine pass faster
  instead of bypassing it.

**Constraints every change must respect**

- C1. **Frame identity.** Frames are addressed by sequential-read ordinal;
  `CAP_PROP_POS_FRAMES` seeks are not trusted (`video.py:4-9`,
  `tests/test_frame_identity.py`). `grab()` is allowed: it advances the decoder
  by exactly one frame, as `read()` does.
- C2. **No inferred presence.** Empty, failed and unobserved frames never
  inherit neighbouring boxes (`planning.py:688-689`).
- C3. **Baselines.** Any change to detector behaviour must pass the fact and
  decision baselines (`make fact-baseline`, `make decision-baseline`).
- C4. **Inpainting quality gate.** Upstream's gate: mean absolute error in the
  removal region ≤ 0.5/255 against the current output, plus a flicker check
  (`plans/STTN_INFERENCE_SPEED.md`, gate section).
- C5. **Fork hygiene.** One branch per workstream from upstream, each
  mergeable with the others in any order, minimal edits to upstream files.
- C6. **Capability-gated, never machine-gated.** The reference machine is an
  M2 Pro with 32 GB, but nothing may be tuned to it. A feature is chosen by
  what the running machine reports or demonstrably does — a backend
  capability, a short self-test, measured memory and core counts — never by
  chip name or a fixed constant sized for the reference machine. Anything that
  trades memory for speed scales with available memory; anything that depends
  on a newer chip or library is used when present and skipped when absent.
  Every such choice has an override so a user can force it on or off.

Requirements are tagged: **[exact]** output is bit-identical and only tests for
equality are needed; **[gated]** output changes and C3 or C4 applies.

## 4. Phase 0 — measure first on the target Mac

**P0-1. Split the inpainting timers.** `prediction_s` currently includes the
inference reader's decode, the resize and cache I/O; `composition_s` includes
the second reader's decode and the blocking pipe write
(`inpainters/sttn.py:325-354`, `360-400`). Record decode, resize, inference,
second decode, compose and pipe write separately in `benchmark.json`.
Acceptance: the six numbers sum to within 2% of `inpainting_s`.

**P0-2. Count detector work.** Add counters to refine: frames decoded, frames
detected, `net.forward` calls, frames reaching contrast recovery, recovery
calls that returned any box. Emit them in the plan timings.

**P0-3. Record a Mac baseline.** On the target machine, for the three sample
clips and one long real file: chip model, torch version, phase timings from
P0-1/P0-2, time from upload to `preview_ready`, and full-clean time. Use
`scripts/verify_inference_speed.py --device mps` for the inference part.
Every later requirement reports against this baseline.

Effort: small. No behaviour change. Do this before anything in §6–§7.

**Status.** P0-1 and P0-2 are implemented on `perf/instrumentation`, which is
stacked on `perf/refine-exact` because it instruments the loops that branch
rewrote. The new numbers appear in `benchmark.json` (`timing`), in
`result.timings` and in the web job's `timings` (`detection_stats`,
`review_stats`); `VIDEOWIPE_PERF_STATS=0` turns the counters off. Measured on
a real clean, the timer parts sum to within 0.01% of the existing totals and
the output video is byte-identical with and without instrumentation.

**P0-3 is run with** `scripts/perf_baseline.py`:

```
python scripts/perf_baseline.py VIDEO [VIDEO ...] -o OUTDIR \
    [--runs 3] [--trial-seconds 3] [--no-full] [--device auto|cpu|mps|cuda]
```

It records the machine (chip, cores, memory, torch/MPS, OpenCV build, ffmpeg
encoders) without assuming any, the video's properties, per-stage timings and
the detection counters, prints a summary and writes `perf_baseline.json`.
`--trial-seconds N` also times a trial placed at the end of the file, which
exposes the positioning cost. Use `--runs 3` for a steady-state median, and
`--no-full` to measure planning alone on a long file. Not captured: time from
upload to `preview_ready` in the web flow (the script times the engine
stages). The reference Mac run is recorded below.

### Reference baseline (P0-3): M2 Pro, 12 cores, 32 GB, torch 2.15 on MPS, float32

Recorded with Phase 1 and the instrumentation merged.

**Short clip** — `chinese1.mp4`, 1080p H.264, 591 frames, two active bands
(top-left watermark and bottom subtitle), median of 3 runs, 135.5 s in total:

| Stage | Seconds | Share | Per frame |
|---|---|---|---|
| Planning | 44.2 | 32.6% | 74.8 ms |
| · coarse sampling | 2.6 | 1.9% | |
| · band fallback | 4.6 | 3.4% | |
| · refine | 26.5 | 19.5% | |
| · boundary probes | 9.5 | 7.0% | |
| · descender recovery | 0.8 | 0.6% | |
| Inpainting | 90.9 | 67.0% | 153.8 ms |
| · model inference | 81.1 | 59.8% | 137.1 ms |
| · blend | 6.3 | 4.7% | 10.7 ms |
| · pipe write | 1.5 | 1.1% | 2.6 ms |
| · both decodes | 1.4 | 1.0% | |

A 3-second trial at the end of the clip: 16.5 s, of which positioning (500
frames skipped) 0.29 s and model inference 13.3 s. First model load in a
process: 7.4 s; warm: 0.3 s.

Detector work: 954 `detect()` calls, one forward pass each (the double
forward in P2-3 does **not** occur here). Contrast recovery ran on 28 refine
frames and 23 probe frames: 306 of the 804 detector calls in those two phases
(38%), and **found nothing in any of them**. At roughly 45 ms a call that is
about 13 s of the 44 s of planning.

**Long file** — 1080p AV1, 70,819 frames (49 minutes), planning only: 102 s,
all of it two sequential decode passes at ~0.6 ms a frame (coarse sampling
44 s, band fallback 58 s) to run the detector on 100 frames. Refine had
nothing to confirm and, with P1-2, read no frames.

**What the baseline changes in this plan**

- Inference is 60% of a run on MPS, not the 75–84% seen on CPU, and planning
  is a third. Planning work is worth more than §2 implied.
- P2-1 remains the largest single lever: at 137 ms a frame, a two-band clean
  of the long file is about 2.7 hours of model time.
- P2-3: recovery is 38% of detector calls in refine and probes with a 0% hit
  rate on this clip, so gating it is the main planning saving. Its
  double-forward part can be dropped: it was not observed.
- P2-4: merging band fallback into sampling halves planning on the long file
  (58 s of 102 s) and saves 10% of planning on the short one.
- P2-6: blending is 6.3 s (4.7%); worth doing, after the above.
- Decode, resize, cache and positioning are each under 1%: leave them
  (P2-5's overlap would gain at most a few percent).

Earlier data point (Linux, 4 cores, ONNX on CPU, 150 frames at 1080p): planning
25 s, of which band fallback 9.3 s (100 detector calls on 50 sampled frames),
refine 7.6 s, coarse sampling 5.4 s, boundary probes 1.9 s; contrast recovery
fired on 1 of 108 refined frames. That clip is dense dialogue, so it says
little about pauses — the long real file will show whether P2-3 matters.

## 5. Requirement index

| ID | Requirement | Type | Effort | Confidence | Expected value |
|---|---|---|---|---|---|
| P1-1 | Linear bookkeeping in refine | exact | S | High | High on long videos |
| P1-2 | `grab()` for unneeded frames; stop at last segment | exact | S | High | Medium |
| P1-3 | Target thumbnails without per-request decode from 0 | exact | S–M | High | High (UI) |
| P1-4 | Mask endpoint: no per-tick render, cache, no lock | exact | M | High | High (UI) |
| P1-5 | Hash the source once per job | exact | S | High | Low–medium on large files |
| P1-6 | `execution_masks`: bisect rows, blur only the box | exact | M | High | Medium–high on long videos |
| P1-7 | Free the MPS allocator on cleanup | exact | S | High | Low (server memory) |
| P1-8 | Faster positioning for a late trial | exact | S | High | Medium |
| P2-1 | Half precision on MPS | gated | S–M | Medium | High if it works |
| P2-2 | Skip model windows that cannot matter | exact | M | Medium | Workload-dependent |
| P2-3 | Fewer detector passes in refine | gated | M | Medium | High on dialogue |
| P2-4 | Fewer full decode passes in planning | exact | M | High | Medium on long videos |
| P2-5 | Overlap decode, inference, compose and write | exact | M–L | Medium | 3–10%, after P2-1 |
| P2-6 | Blend only inside the mask's box | exact | M | High | Small–medium |
| P2-7 | Prefer Torch when CUDA is present | gated | S | High | Non-Apple only |
| Q-1 | Correct timing for variable-frame-rate sources | gated | M–L | High | High for affected files |
| Q-2 | Keep audio, metadata, chapters, subtitle tracks | gated | M | High | High |
| Q-3 | Carry colour tags and range | gated | M | Medium | Unknown until measured |
| Q-4 | 10-bit and HDR: warn everywhere, then preserve | gated | S / L | High | High for affected files |
| Q-5 | OCR reads the right frame | gated | S | High | Medium |
| Q-6 | Sampling that scales with duration | gated | M | Medium | High on long videos |
| Q-7 | Area filter for the model's downscale | gated | S | Medium | Small, free |
| Q-8 | Does the model expect masked input? | research | S | Low | Possibly large |
| Q-9 | Seam colour match; tighter crop | gated | M | Low–medium | Unknown |
| Q-10 | ProPainter adapter frame-rate truncation | exact | S | High | Small |
| U-1…4 | Web UI responsiveness | exact | S–M | Medium | Medium on long videos |

## 6. Phase 1 — exact changes (no change to results)

### P1-1. Linear bookkeeping in the refine loop

- **Problem.** For every frame and every active candidate the loop rebuilds and
  sorts a growing list and rescans another:
  `candidate.temporal_sample_indices = sorted(set(...) | {frame_index})` and
  `candidate.presence_frames = [i for i in ... if i != frame_index]`
  (`detect.py:1654-1660`). **Verified.**
- **Measured.** A stand-alone reproduction takes 0.8 s at 5,000 frames, 2.9 s
  at 10,000 and 11.8 s at 20,000 — quadratic. Extrapolated, a candidate active
  on all 70,199 frames (a persistent watermark) costs about 2–3 minutes of
  pure Python on the test CPU, per candidate.
- **Change.** Keep per-candidate sets during the loop; sort once after it (the
  function already ends with `sorted(set(...))`).
- **Acceptance.** `temporal_sample_indices` and `presence_frames` equal the
  current implementation's on the three samples and on a synthetic 70k-frame
  full-length candidate; refine time on that synthetic case with a stub
  detector is linear in frame count.

### P1-2. Do not convert frames nobody looks at; stop when nothing is left

- **Problem.** Refine calls `cap.read()` for every frame before deciding
  whether any candidate is active, and continues to the end of the file
  (`detect.py:1639-1652`). `skip_frames` also uses `read()` (`video.py:14`).
  **Verified.**
- **Change.** (a) Decide `active` first; `grab()` when it is empty, `retrieve()`
  otherwise. (b) Exit after the last candidate segment ends, still emitting the
  final progress event. (c) `skip_frames` uses `grab()`. (d) The same in
  `_recover_descenders` for frames with no boxes.
- **Benefit.** `grab()` skips colour conversion and the frame copy. Measured
  gain on H.264 1080p is modest (0.33–0.37 s against 0.40–0.47 s per 150
  frames); expect more on 4K. Early exit removes the tail of the pass entirely
  and matters most for `refine_review`, where a newly selected target may end
  early. **Hypothesis** on magnitude for VP9.
- **Acceptance.** Identical plans; `tests/test_frame_identity.py` updated for a
  fake reader with `grab()`; frame count read is unchanged.

### P1-3. Target thumbnails without decoding from frame 0 each time

- **Problem.** `/jobs/{id}/frame` opens a new capture and grabs from 0 to the
  requested frame, under `job.media_lock`, once per target
  (`server/app.py:1076-1086`). The comment there already anticipates the cost.
  It also parses the plan and the full candidates file before checking the
  cache. **Verified.** For a target first seen 45 minutes into a file this is a
  long sequential decode per thumbnail.
- **Change.** In order of effort: check `evidence-{n}.jpg` before parsing
  anything; produce all thumbnails in one ascending pass; best, write them
  during detection, which already has those frames in hand.
- **Acceptance.** Thumbnails are byte-identical to today's; opening the review
  page of a long job triggers no decode pass longer than one sequential read.

### P1-4. Mask endpoint: stop rendering on every playback tick

- **Problem.** The page calls `/mask` on every `timeupdate` (about 4 Hz), even
  with "show mask" off, only to read the `X-Has-Removal` header
  (`web/index.html`, `refreshMask`). The server, inside `job.lock`, reloads the
  plan with masks, rasterises, blurs and PNG-encodes a full-resolution mask
  (`server/app.py:375-397`). `job.lock` is the lock progress snapshots need, so
  playback stalls progress updates. **Verified.**
- **Change.** Expose "has removal" per review revision in the job snapshot;
  fetch `/mask` only when the overlay is on; cache `(plan, masks)` per
  `(job, revision)` and PNG bytes per `(revision, active set)`; compute outside
  the lock; send `ETag`.
- **Acceptance.** Playing a video with the overlay off makes no `/mask`
  requests; with it on, repeated frames in the same mask state are served from
  cache; mask PNGs are identical to today's.

### P1-5. Hash the source once per job

- **Problem.** The full source is SHA-256 hashed at upload
  (`server/app.py:481`, inside an `async` handler, so it blocks the event
  loop), again in `compute_source` (`plan.py:298`), and three more times per
  trial or full run (`server/app.py:651`, plan load in `engine.run`,
  `inpainters/sttn.py:210`). **Verified.** Negligible on small clips; several
  full reads of a multi-gigabyte file otherwise.
- **Change.** Hash during the upload copy loop, off the event loop; pass the
  known digest down (`job.source_sha256` → plan → `InpaintJob`) instead of
  re-reading. Keep one re-hash per run as the "input changed" check.
- **Acceptance.** One hash at upload plus one per run; the "source sha256
  mismatch" error still fires when the file is replaced.

### P1-6. `execution_masks`: find the row by bisect, blur only the box

- **Problem.** `_spatial_mask` scans every spatial row for each frame
  (`plan.py:790-799`); the one-entry cache misses whenever the row changes,
  and each miss runs a full-frame Gaussian blur (`plan.py:840-862`). With a
  protected region, `_union_projection` blurs once per row boundary
  (`plan.py:942-954`). On a long subtitle track this is quadratic in rows.
  **Verified.**
- **Change.** Bisect on row starts (rows are ordered and disjoint by
  validation); blur only the mask's bounding box padded by the kernel radius.
- **Acceptance.** `frame_mask(i)` bit-identical for every frame of the samples
  and of a synthetic 30,000-row plan; 1,000 calls on that plan at least 10×
  faster.

### P1-7. Free the MPS allocator on cleanup

- **Problem.** `TorchBackend.cleanup` only empties the CUDA cache
  (`backends.py:133-139`). On Apple Silicon memory is shared with the system.
  **Verified** in code; allocator behaviour **Hypothesis**.
- **Change.** `gc.collect()` then `torch.mps.empty_cache()` when on MPS.
- **Acceptance.** After a job in the long-lived server, process memory returns
  close to its pre-job level.

### P1-8. Faster positioning for a trial late in the file

- **Problem.** A trial starting at frame N reads frames 0…N on two readers, one
  after the other, with full conversion (`inpainters/sttn.py:263-280`).
  Progress shows `0/total` throughout. **Verified.**
- **Change.** With P1-2(c) both skips use `grab()`; run the two skips on two
  threads (decode releases the GIL); report skip progress. In the inpaint loop,
  `grab()` instead of `read()` on the inference reader when no band needs
  pixels (inactive segment or all cache hits).
- **Acceptance.** Trial output identical; time to first trial frame at the end
  of a long file roughly halves.

## 7. Phase 2 — larger performance work

### P2-1. Half precision on MPS **(the main Apple Silicon lever)**

- **Problem.** Mixed precision is enabled for CUDA only:
  `torch.cuda.amp.autocast(enabled=t.is_cuda)` (`backends.py:119`). On MPS the
  transformer — about 70% of the model's arithmetic — runs in 32-bit float.
  Upstream deferred this rather than rejecting it
  (`plans/STTN_INFERENCE_SPEED.md:71`). **Verified.**
- **Change.** Run `model.infer` under `torch.autocast("mps",
  dtype=torch.float16)` (torch ≥ 2.5), or convert the transformer to half
  precision, on MPS only; keep encoder and decoder in float32 as the CUDA path
  does. Make the `precision` label honest (`inpainters/sttn.py:213`,
  `engine.py:858`), which also invalidates stale prediction-cache entries.
- **Risk.** Attention scores are divided by `sqrt(d_k·h·w)` with products up
  to 76,800 (`models/sttn.py:142-143`): check for half-precision overflow.
  Speed-up on M1–M4 comes from memory traffic, not doubled throughput.
- **Selection (C6).** Do not decide by chip. Enable half precision when the
  installed torch supports MPS autocast and a one-off self-test on the loaded
  model passes the numeric check; record the choice in the benchmark metadata
  and the cache identity. Provide an explicit precision setting (`auto`,
  `float32`, `float16`). On the M2 Pro the gain may be modest; chips with
  faster half-precision paths then benefit without a code change, and a
  machine where it is slower or inaccurate keeps float32.
- **Acceptance.** C4 quality gate passes on the three samples; hot median of
  three runs at least 15% faster than P0-3 (upstream's own bar for keeping a
  change). If it fails either test, close as "tried, not adopted" and record
  the numbers. **Hypothesis**: 1.2–2× on the transformer.

### P2-2. Skip model windows that cannot influence an emitted pixel

- **Problem.** Within an active segment all five windows run even when the
  mask is active on a few frames only; predictions for the rest are discarded
  by the compositor (`inpainters/sttn.py:73-103`, `296-310`). **Verified.**
- **Change.** Pass the set of needed frames into `_process_segment`; skip a
  window when none of its neighbour frames is needed. Needed frames stay
  bit-identical because every window covering them still runs.
- **Must also do.** The prediction-cache key excludes the mask
  (`prediction_cache.py:30-32`); add the needed-frame set to the key or cache
  only fully computed segments, otherwise a later run could reuse a segment
  with holes.
- **Acceptance.** Pixel-identical output on the sparse fixture; no benefit
  expected for static masks.

### P2-3. Fewer detector passes in refine

Three verified multipliers stack on text-free frames inside a candidate window
(dialogue pauses):

1. `_needs_contrast_recovery` is true for every frame with no box or a box
   under half the candidate width (`detect.py:1526`); recovery then runs the
   detector on six image variants with no early exit (`detect.py:1576`).
2. `DBNetDetector.detect` runs the high-level path and, if it finds nothing,
   a second full forward pass (`detect.py:284-300`); recovery restores the
   flag that would skip the first, so the double pass recurs per variant.
3. The detector input is a fixed pixel budget (`detect.py:264`), so a band
   crop costs as much as a full frame.

Worst case: 2 + 6×2 = **14 forward passes** for one empty frame.

- **Change, in increasing risk.** (a) Stop recovery when the remaining
  variants cannot produce the two-family agreement it requires. (b) Avoid the
  double forward inside recovery without mutating shared state. (c) Scale crop
  inputs by the full-frame scale factor instead of a fixed budget (~4× cheaper
  per band call). (d) A cheap contrast pre-gate so recovery only runs on the
  flash/dissolve frames it was written for.
- **Decide from data.** P0-2's counters show how often recovery fires and how
  often it returns anything; do (c) and (d) only if that shows they are safe.
- **Acceptance.** C3 baselines unchanged for (a); within agreed tolerance for
  (b)–(d); forward passes per refined frame reported before and after.

### P2-4. Fewer full decode passes in planning

- **Problem.** A balanced job decodes the file five times before a trial:
  coarse sampling (the last sample is the last frame, so it reads everything),
  band fallback (a second sampling pass, `detect.py:1374-1386`), refine,
  boundary probes (`planning.py:574-583`) and descender recovery
  (`planning.py:483-497`). **Verified**; the profile counted 8 readers and
  1,023 reads for 150 frames.
- **Change.** Run band-fallback detection inside the sampling loop (snapshot
  the detector's `_manual_only` flag so results match today's); combine the
  probe and descender passes into one. Five passes become three.
- **Acceptance.** Identical candidates and plans; reader and read counts drop
  as stated.

### P2-5. Overlap decode, inference, compose and write

- **Problem.** The inpaint loop is strictly serial; the one-worker executor is
  joined immediately (`inpainters/sttn.py:286-353`). **Verified.**
- **Change.** A bounded three-stage pipeline: decode and resize segment N+1
  while N is in the model; compose and write N−1 on a writer thread.
- **Why later.** Non-inference work is under 10% today on MPS (upstream's
  resident-feature prototype gained 4.3%). Its share rises after P2-1.
  Cancellation, progress order and trial slicing must be preserved.
- **Acceptance.** Identical output; gain measured with P0-1's split timers.

### P2-6. Blend only inside the mask's bounding box

- **Problem.** Per frame, the compositor copies the frame, converts whole
  bands to float, allocates several band-sized temporaries, and upscales the
  full band, even when the mask is a small box (`inpainters/sttn.py:48-70`,
  `377-398`). **Verified.**
- **Change.** Compute the alpha bounding box once per distinct mask; blend and
  resize only there; precompute `alpha` and `1−alpha`; write
  `memoryview(frame)` instead of `frame.tobytes()`.
- **Acceptance.** Bit-identical frames; `composition_s` per frame reduced.

### P2-7. Prefer Torch when CUDA is available (non-Apple)

- **Problem.** Off Apple Silicon, ONNX is chosen whenever `onnxruntime` is
  importable, even with a CUDA-capable Torch installed (`engine.py:395-401`),
  so the half-precision CUDA path is never used. **Verified.**
- **Change.** Choose Torch when `torch.cuda.is_available()`.

## 8. Phase 3 — output quality and fidelity

The encode command is fixed (`inpainters/sttn.py:220-251`): raw `bgr24` on a
pipe at `-r {fps}`, `libx264 -crf 18 -preset medium -pix_fmt yuv420p`,
`-map 0:v -map 1:a?`, `-c:a aac`, `+faststart`.

### Q-1. Variable-frame-rate sources

- **Problem.** Frames are re-timed at one constant rate while audio keeps the
  source's real timing; nothing carries source timestamps. The web path only
  warns (`server/app.py:271-274`); the CLI does not. **Verified.** YouTube
  WebM downloads often have irregular timing, so the motivating file is
  plausibly affected. Size of the drift: **Hypothesis**.
- **Change.** Minimum: pass the exact rational rate from ffprobe and show the
  warning in the CLI. Proper: take video timestamps from the source (ffmpeg
  decodes the source for timing, or frames are written with per-frame PTS).
- **Acceptance.** On a synthetic VFR clip with a click track, audio/video
  offset at the start, middle and end of the output matches the input within
  one frame.

### Q-2. Audio, metadata, chapters and subtitle tracks

- **Problem.** Audio is always re-encoded to AAC at ffmpeg's default bitrate,
  including 5.1 tracks; there is no `-map_metadata`/`-map_chapters`, so title,
  creation time and chapters are lost; soft subtitle tracks are not mapped.
  The code comment claims metadata survives; the command does not do that.
  **Verified.**
- **Change.** Stream-copy audio when the codec fits the container, else encode
  at a bitrate derived from the source; add `-map_metadata 1 -map_chapters 1`;
  map subtitle tracks where convertible. (The fork's `feat/h265-output` branch
  already handles audio per track for its MKV output; this is the equivalent
  for the default MP4.)
- **Acceptance.** `ffprobe` stream, chapter and format listings match the
  source except for the video stream; copied audio is bit-identical.

### Q-3. Colour tags and range

- **Problem.** No colour primaries, transfer, matrix or range are set on input
  or output. **Verified.** Whether this shifts colours today depends on which
  matrix OpenCV used to decode; if decode and encode both assume BT.601 the
  errors cancel for untouched pixels. **Hypothesis.**
- **Change.** Measure first (colour bars through a no-op run; compare mean
  chroma and PSNR outside the mask for BT.709, BT.601 and full-range sources).
  Then set output tags to the source's *together with* a matching conversion.
  Do not add tags on the encode side alone — that can introduce a shift.

### Q-4. 10-bit and HDR sources

- **Problem.** Decode is 8-bit BGR and output is `yuv420p`; HDR is only warned
  about in the web path. **Verified.**
- **Change.** Short term: warn for bit depth above 8 or an HDR transfer in the
  CLI as well. Long term: see §9.

### Q-5. OCR crops the wrong frame

- **Problem.** For classification, the crop is taken from `first_frame` using
  box coordinates from a different sampled frame
  (`detect.py:1320`: `crop = first_frame[by1:by2 + 1, bx1:bx2 + 1]`), so OCR
  often reads background. That text decides timestamp/watermark typing and the
  default selection. **Verified.**
- **Change.** Crop from the frame the box came from (run OCR inside the
  sampling loop for the best box per region).
- **Acceptance.** Decision baseline re-run and reviewed; classification on the
  samples no worse.

### Q-6. Sampling that scales with duration

- **Problem.** Balanced mode samples 50 frames whatever the length
  (`detect.py:1165-1169`). Each frame takes the state of its nearest sample
  (`plan.py:306-346`) and refine only examines frames inside positive windows
  (`detect.py:1648-1653`). On 70,199 frames one sample speaks for ~1,400
  frames (~57 s): a sample that lands in a dialogue pause marks that whole
  window absent and its subtitle lines are never checked. A track visible less
  than 40% of the time may not become a candidate at all. Upstream already
  emits a warning when the gap exceeds two seconds (`plan.py:501-506`), which
  is every video over ~100 s. Mechanism **Verified**; how often it loses lines
  in practice **Hypothesis**.
- **Change.** Sample by time (for example one sample every 3–5 s, capped).
  Sampling is a sequential read, so decode cost is unchanged; the extra cost is
  detector calls only, and smaller gaps shrink the probe step.
- **Risk.** Changes documented defaults and every baseline. Needs a long clip
  with ground-truth subtitle times.
- **Acceptance.** On that clip, subtitle-line recall improves with no rise in
  false removals; C3 baselines re-recorded and reviewed.

### Q-7. Area filter when shrinking the model input

- **Problem.** Each band is reduced 3× at 1080p (6× at 4K) with bilinear
  interpolation, which aliases thin strokes (`inpainters/sttn.py:336-338`).
  **Verified.**
- **Change.** `INTER_AREA` for the downscale (same cost); optionally cubic for
  the upscale. Gate with C4 and a visual A/B, since the weights may prefer the
  filter they were trained with.

### Q-8. Research: does the model expect masked input?

- **Finding.** The encoder receives the frames with the text still in them; no
  mask is applied (`inpainters/sttn.py:76`), though the model's parameter is
  named `masked_frames` (`models/sttn.py:101`). Standard STTN zeroes masked
  pixels first. How `detext_trial.pth` was trained is not documented.
- **Action.** Ask the model's author or A/B on frames with visible ghosting.
  If masking helps, the prediction-cache key must include the mask. Low
  confidence; potentially the largest quality gain at zero compute cost.

### Q-9. Seam and softness at 1080p and above (experiments)

The fill is a 3× upscaled prediction pasted beside sharp pixels with a 4-pixel
feather (`plan.py:854-858`). Two experiments: a low-frequency colour match
across the feather ring; cropping around the mask's box rather than a
full-width band to raise effective resolution. Both need visual A/B.

### Q-10. ProPainter adapter

`--save_fps int(fps)` truncates 29.97 to 29 (`propainter_wipe.py:129`), a 3%
duration error. Pass the float. **Verified.**

## 9. Longer-term direction (not scheduled)

**Compose in ffmpeg, in the source's pixel format.** Let ffmpeg decode the
source and overlay only the inpainted patch and its alpha. This would solve
Q-1 (source timestamps), Q-3 and Q-4 (no 8-bit BGR round trip), shrink pipe
traffic from a whole frame to a small crop, and allow hardware decode. It is a
rewrite of the inpainter's I/O and should only start once Q-1…Q-4 show how
often real files are affected.

## 10. Considered and not recommended

| Idea | Why not |
|---|---|
| Core ML execution provider for the STTN ONNX models | Seven-dimensional reshapes are unsupported on the Neural Engine; upstream lists conversion as deprioritised. Torch on MPS is the Apple path. |
| Moving the text detector to ONNX Runtime/Core ML | The detector is OpenCV DNN, not ONNX Runtime (`detect.py:223`): this is a re-platforming with new numerics. Reducing calls (P2-3) is cheaper and larger. |
| `torch.compile` on MPS | Experimental; window sizes vary per call. |
| Keeping features on the device between stages | Upstream prototyped it: 4.3% (`plans/STTN_INFERENCE_SPEED.md:23`). |
| Hardware video decode | Unproven in the OpenCV wheel; pixels can differ from software decode, which breaks cache identity and detection reproducibility; software decode is ~1% of a run. |
| VideoToolbox for the H.264 preview copy | Trials are at most five seconds; worse quality per bit would blur whether an artefact is the encoder's or the inpainter's. |
| Raising `--gap` | Measured 43% slower per frame at 50. |
| Fewer overlapping windows as the default | Up to 2× faster but removes the averaging that hides flicker; at most an opt-in "fast" preset after A/B. |
| CPU thread tuning | Upstream measured the optimum; ONNX Runtime already uses all cores. |
| Refining only the trial region | Rejected by upstream (see §3). |
| Stream-copying untouched GOPs | Concatenating differently encoded H.264 is fragile; §9 is the safer route. |
| Seeking by a keyframe index | Would make sampling and thumbnails O(GOP), but it is exactly the area upstream's frame-identity fix addressed. Reconsider only if decode still dominates after P1 and P2-4. |

Checked and found fine: device selection order, segment-level skipping of
inactive bands, mask feather caching within a run, prediction-cache I/O cost,
Range and ETag serving of videos, trial-result reuse, ffmpeg stderr handling,
cancellation granularity.

## 11. Apple Silicon summary

- Use a native install with the `torch` extra: inference then runs on MPS.
  Docker on macOS is a Linux VM with no GPU, so the image is CPU-only there.
- Apple-specific requirements: **P2-1** (half precision on MPS), **P1-7**
  (free the allocator), P0-3 (baseline). Everything in §6 helps at least as
  much on a Mac, because fast inference makes the other stages a larger share.
- VideoToolbox is useful for the optional H.265 export (the fork's
  `feat/h265-output` branch), not for the pipeline's internal H.264 copy.
- Unknown until measured on the target Mac: the half-precision speed-up, VP9
  decode cost, and detector latency per call.
- Reference machine: M2 Pro, 32 GB. Memory-for-speed options (keeping a
  segment's full-resolution frames instead of decoding twice, prediction-cache
  limits, pipeline queue depth in P2-5) must size themselves from available
  memory, so 16 GB machines stay safe and 64 GB+ machines gain more (C6).

## 12. Web UI responsiveness

- **U-1.** Timeline: one `<canvas>` or merged intervals instead of one element
  per segment, rebuilt on every edit (`web/index.html`, `renderTimeline`).
- **U-2.** Mask overlay: the page recolours every pixel in JavaScript per
  refresh (~2 M iterations at 1080p); have the server return a pre-tinted PNG
  or tint with compositing.
- **U-3.** Progress stream: a full job snapshot is sent every 0.5 s whether or
  not it changed (`server/app.py:1008-1014`), and each message rewrites ~40
  DOM properties; send on change and skip unchanged writes.
- **U-4.** Translation overlay (fork): memoise `translate()` results and
  compile the pattern list once per language. Cheap now (240 keys); it scales
  with catalogue size.
- Also: `job.json` embeds the candidate list with per-frame presence and is
  fsynced on every edit (`server/jobs.py:109-124`); move the immutable
  candidates to a side file. Old run directories are never pruned.

## 13. Delivery plan

One branch per group, each from upstream, each independently mergeable (C5).

| Order | Branch | Contents | Gate |
|---|---|---|---|
| 1 | `perf/instrumentation` (on `perf/refine-exact`) | P0-1, P0-2, P0-3 (done; baseline in §4) | — |
| 2 | `perf/refine-exact` | P1-1, P1-2, P1-8 (done) | equality tests |
| 3 | `perf/server-exact` | P1-3, P1-4, P1-5 (done) | equality tests |
| 4 | `perf/plan-masks` | P1-6 (done) | equality tests |
| 5 | `perf/mps` | P1-7 (done), P2-1 | C4 + 15% bar |
| 6 | `perf/detector-calls` | P2-3, P2-4 | C3 baselines |
| 7 | `fix/output-fidelity` | Q-1 (done, the proper fix), Q-2 (done), Q-4 warning (done), Q-10 (done) | ffprobe comparisons |
| 8 | `fix/ocr-crop` | Q-5 | decision baseline |
| 9 | `feat/duration-sampling` | Q-6 | ground-truth clip + baselines |
| 10 | `perf/skip-windows`, `perf/recovery-reuse` | P2-2, recovery reuse (done) | equality tests |
| 11 | later | P2-5, P2-7, Q-3, Q-7…Q-9, U-1…U-4 | as stated |

Steps 2–4 are safe to start immediately after step 1; nothing in them changes
results. Steps 5, 6 and 9 must not be merged without their gate numbers.

### Phase 1 status

Implemented on the four branches above (P1-7 on `perf/mps`). Verified by
oracle tests that embed the original functions, and by an end-to-end clean of
a 150-frame 1080p clip with upstream and with every fork branch merged: the
plan, mask file and output video are byte-identical. Not yet measured on the
reference Mac — step 1 (instrumentation and the P0-3 baseline) is still open
and is needed before Phase 2.

Known differences from upstream behaviour, all outside the results:

- Refine no longer raises for a file that is short only after the last
  candidate segment, and sends one final progress event instead of per-frame
  events for that unread tail.
- The windowed mask blur (P1-6) is used only when a one-off self-test proves
  it bit-identical on the running machine; `VIDEOWIPE_MASK_WINDOW_BLUR`
  (`auto`, `on`, `off`) overrides it. Its alignment constants were found on
  x86; on Apple Silicon the self-test decides.
- `PATCH /review` also computes whether anything is removed, so the page need
  not request a mask on every playback tick; after a reload one `/mask`
  request per revision remains.
- The source hash is remembered per process, keyed by path, size,
  modification time and inode.
- The first thumbnail request decodes to the last evidence frame and caches
  all of them.

### Phase 2 status

Built after the baseline, each on its own branch; verified with all fork
branches merged (1,020 tests) and by an end-to-end clean whose plan, masks
and output video are byte-identical to upstream's.

| Item | Branch | Outcome |
|---|---|---|
| P2-1 half precision on MPS | `perf/mps` | Implemented behind `VIDEOWIPE_PRECISION` (`auto`, `float32`, `float16`), guarded by an on-device numeric self-test with a float32 fallback. **Measured on the reference Mac and not adopted: `auto` stays float32 on MPS.** See the result below. The setting remains for chips where half precision is faster (C6). |
| P2-4 band fallback without a second decode | `perf/detector-calls` (on `perf/instrumentation`) | Done, exact. The fallback runs from band crops kept while sampling, within a memory budget (`VIDEOWIPE_BAND_CROP_BUDGET_MB`), otherwise the old second pass. Probe and descender passes were not merged: the descender frame set depends on the probe results. |
| P2-3(a) recovery pruning | `perf/detector-calls` | Done, exact for returned boxes. Only the sixth variant is provably redundant: 33 of 954 detector calls on the reference clip (3%). A detector error that only that call would raise is no longer seen. |
| P2-3(b) recovery pre-gate | — | **Not built.** A brightness gate separated hits from misses on the three real samples but would have discarded 12 genuine recoveries on synthetic fades to black and mid-grey dissolves — the cases recovery exists for. No statistic separated all 275 recorded recoveries with a margin. |
| P2-6 blend inside the mask's box | `perf/blend-box` (on `perf/instrumentation`) | Done, byte-identical pipe output. Blend time in a real clean fell from 2.2 s to 0.2–0.3 s on the Linux test machine (the reference Mac spent 6.3 s). Cropped upscaling is used only where a rounding argument and an on-machine check both agree; `VIDEOWIPE_BLEND_WINDOW_RESIZE=off` disables it. |
| P2-2 skip model windows | `perf/skip-windows` (on `perf/blend-box`) | Done, exact. A window runs only if a frame it predicts reaches an emitted pixel in that band; a frame is returned only when every window covering it ran, so no partly averaged frame is used or cached. A cache entry may hold only some frames and satisfies a band only when it holds every frame needed; missing frames are computed and merged in. `VIDEOWIPE_SKIP_WINDOWS=off` restores the old behaviour. **Saving on the samples is small** (see below). |
| Recovery reuse | `perf/recovery-reuse` (on `perf/detector-calls`) | Done, exact. Boundary probes repeat recovery calls refine already made (same frame, same box); a per-run memo keyed on crop pixels, box, frame size and detector state answers them. 6–8% of planning's detector calls on two samples, under 1% on the third; on the reference Mac boundary probes run 2.5–2.9 s faster on those two (see below). `VIDEOWIPE_RECOVERY_REUSE=off` disables it. |
| P2-3(c) cheaper crop inputs | — | **Measured, not built.** 10–15× faster per recovery call, but it changes plans on two of three samples and loses real recoveries. See below. |

**P2-1 result on the reference Mac** (M2 Pro, torch 2.15, `chinese1.mp4`,
median of 3 runs each; the float16 run used MPS autocast and passed the
on-device self-test):

| | float32 | float16 | Change |
|---|---|---|---|
| Model inference | 80.38 s (136.0 ms/frame) | 79.26 s (134.1 ms/frame) | −1.4% |
| Whole run | 127.49 s | 125.32 s | −1.7% |

That is within run-to-run variation and far below the 15% bar. The output
comparison also failed the gate: mean absolute difference inside the mask
0.63 (limit 0.5) and temporal-difference error 0.52 (limit 0.5), with the
caveat that both figures include H.264 encoder noise (0.16 over the whole
frame). No speed for a measurable change in output: closed as tried, not
adopted on this chip. As predicted in §7, M1–M4 class GPUs gain little from
half precision; the setting stays available for later chips.

**Phase 1 + 2 result on the reference Mac** (same clip, float32, median of 3,
against the §4 baseline):

| Stage | Baseline | Now | Change |
|---|---|---|---|
| Whole run | 135.54 s | 127.49 s | −5.9% |
| Planning | 44.22 s | 41.46 s | −6.2% |
| · refine | 26.48 s | 24.86 s | |
| · boundary probes | 9.48 s | 8.74 s | |
| · band fallback | 4.59 s | 4.35 s | (the removed decode pass is small on a 591-frame clip) |
| Blend | 6.30 s | 0.74 s | −88% |
| Composition | 8.54 s | 3.79 s | −56% |
| Model inference | 81.05 s | 80.38 s | unchanged |

Model inference is now 63% of the run and unchanged at 136 ms a frame. The
remaining levers on it are P2-2 (now built; small on these clips, see below)
and fewer overlapping windows (faster, but changes
output; opt-in only). The long-file planning re-run (expected to drop the
58 s band-fallback pass) is still to be recorded.

**P2-2 result** (window counts from real plans, `scripts/skip_windows_estimate.py`;
exact and machine-independent; model time is at most proportional):

| Clip | Band-segments | Windows before | Windows now | Saved |
|---|---|---|---|---|
| `chinese1.mp4` | 24 | 119 | 106 | 10.9% |
| `english1.mp4` | 20 | 100 | 99 | 1.0% |
| `others.mp4` | 21 | 104 | 104 | 0.0% |

Subtitles are on screen almost throughout these clips, so nearly every window
touches a needed frame. The change pays on sparse text, on trial ranges that
do not fill their edge segments, and on re-runs over a partly filled cache;
on dense dialogue it is negligible. Proven against the original
`_process_segment` embedded as an oracle and by an end-to-end clean (4 of 30
windows skipped on the 6-second clip, output byte-identical to upstream's).

Measured on the reference Mac (M2 Pro, torch 2.15 on MPS, float32, median of
3 runs, `VIDEOWIPE_SKIP_WINDOWS` on against `off` on the same branch):

| Clip | Windows | Model inference off → on | Whole run off → on |
|---|---|---|---|
| `chinese1.mp4` | 119 → 106 | 79.98 s → 70.96 s (−11.3%) | 129.66 s → 120.09 s (−7.4%) |
| `english1.mp4` | 100 → 99 | 69.79 s → 66.81 s (−4.3%, mostly noise: 1 window) | 121.89 s → 117.00 s |
| `others.mp4` | 104 → 104 | 71.13 s → 69.43 s (one clean run; no change expected) | — |

Model time follows the window count on `chinese1`. The output videos are
byte-identical with the switch on and off on all three clips on MPS. Three of
the nine "on" runs (`english1` run 3, `others` runs 1 and 2) ran about 2.1×
slower in model inference with identical work, so something else loaded the
machine then; the `others` median is unusable and its one undisturbed run is
quoted. The Mac's plans also have one inpainting band per clip, so the "two
active bands" in §4 describes candidates, not model bands.

**Recovery reuse result** (`scripts/recovery_reuse_probe.py`, counts exact):

| Clip | Planning detector calls | Recovery calls (detector calls inside) | Repeats | Detector calls saved |
|---|---|---|---|---|
| `chinese1.mp4` | 921 | 51 (273) | 13 | 70 (7.6%) |
| `english1.mp4` | 1,135 | 93 (541) | 13 | 73 (6.4%) |
| `others.mp4` | 784 | 28 (168) | 1 | 6 (0.8%) |

24 of the 27 repeats are boundary probes asking about a gap frame refine had
already tried with the same box, always after an empty answer (a refine hit
adds evidence and the probe then skips the frame). A one-entry "previous
frame" memo would have caught 3 of 27, so the memo is keyed. Whole-frame
detection is never repeated. Plans and masks are byte-identical with the memo
on and off on all three clips. `refine_review` is covered by tests with stubs
only.

Measured on the reference Mac (M2 Pro, torch 2.15 on MPS, float32, planning
only, median of 3 runs, `VIDEOWIPE_RECOVERY_REUSE` on against `off` on the
same branch):

| Clip | Forward passes off → on | Boundary probe off → on | Refine off → on | Planning off → on |
|---|---|---|---|---|
| `chinese1.mp4` | 921 → 851 (−70, 7.6%) | 8.83 s → 5.93 s (−2.90 s) | 25.02 s → 26.57 s (same 568 calls) | 41.71 s → 40.51 s (−1.20 s, −2.9%) |
| `english1.mp4` | 1,134 → 1,062 (−72, 6.3%) | 6.28 s → 3.76 s (−2.52 s) | 35.35 s → 33.97 s (835 → 823 calls) | 49.00 s → 44.85 s (−4.15 s, −8.5%) |

The boundary-probe saving is the reliable figure: about 2.5–2.9 s a clip, or
41–42 ms for each detector call avoided, steady to within 0.16 s across the
three runs of each setting. It matches the estimate (about 3 s on
`chinese1`). The whole-planning figures are less clean because refine moved
by more than its work explains: on `chinese1` refine did identical work in
both settings (568 calls) yet took 26.81, 25.01 and 26.57 s with the memo on
against 24.74, 25.11 and 25.02 s with it off, which hides 1.5 s of the saving
in the median; on `english1` refine was 1.4 s faster for 12 fewer calls
(about 0.5 s expected). This is taken to be run-to-run variation, not memo
overhead, because one "on" run matches the "off" runs and `english1` moves
the other way; that reading is not proven. No run was disturbed in the way
seen in the skip-windows measurement. Every counter other than the detector
calls saved (frames converted, recovery frames and calls, recovery hits,
frames found present) is identical on and off, as an exact change requires.
`english1` shows one detector call fewer here than in the probe table above
(1,134 and 72 saved against 1,135 and 73, counted on the Linux test machine);
the difference was not investigated. `others.mp4` was not timed (6 calls,
about 0.25 s expected).

**P2-3(c), measured and not built.** Sizing recovery's detector input by the
full-frame scale factor (for the `chinese1` band, about 1664×256 → 256×64)
makes each recovery detector call 10–15× faster on the test CPU. It changes
results: `chinese1` unchanged (no hits either way); `english1` keeps all 57
hits but with different boxes on 57 frames; `others` loses 8 of 28 hits, so a
track loses 8 removal frames. On 183 synthetic fades and dissolves where
normal detection fails, the fixed budget recovered 178 and the scaled input
181 (2 lost, 5 gained). Fact baseline: region Jaccard 0.3203 → 0.3190,
boundary F 0.4583 → 0.4537, false-removal area 0.0105 → 0.0049; decision
baseline unchanged. Not adopted as is; a variant that falls back to the full
input when the scaled pass finds nothing might keep the hits and most of the
speed, and would need its own evaluation.

Still open in recovery: per-variant reuse (blank or saturated variants repeat
across frames; a little more than the above) — not built.

### Phase 3 status

| Item | Branch | Outcome |
|---|---|---|
| Q-1 variable-frame-rate sources | `fix/output-fidelity` | Done (the proper fix, not the minimum). Constant-rate sources are exact: same ffmpeg arguments and pipe bytes, output byte-identical to upstream's. Variable-rate sources keep each frame's own timestamp. `VIDEOWIPE_SOURCE_TIMING=auto\|off\|on`. **Not yet run on the reference Mac's ffmpeg** (built and tested on 6.1.1). See below. |
| Q-2 audio, metadata, chapters, subtitles | `fix/output-fidelity` | Done for full runs; trials unchanged. Audio is copied per track where MP4 carries the codec (bit-identical packets), otherwise AAC at a rate derived from the source; global metadata and text subtitles are carried over. `VIDEOWIPE_OUTPUT_TRACKS=auto\|off`; `off` is byte-identical to upstream. **Not checked in QuickTime or any player, nor on the reference Mac's ffmpeg.** See below. |
| Q-4 warning for more than 8 bits or HDR | `fix/output-fidelity` | Done (the short-term item only). One line when the source is first read for a plan, in the CLI and on the server console; nothing in the output changes. |
| Q-10 ProPainter adapter frame rate | `fix/output-fidelity` | Done, but not as the PRD said: passing the float would have stopped ProPainter. **Not run against a real ProPainter.** See below. |

**Q-1 evidence.** The hypothesis is confirmed on a bundled sample:
`chinese1.mp4` is variable-rate (589 frame gaps of 0.04 s and one of 0.12 s
after the first frame; `r_frame_rate` 25/1, `avg_frame_rate` 14775/593;
OpenCV reports 24.9157 fps). Upstream re-times it on the average-rate grid,
which puts the picture up to 0.08 s (two frames) from its real time while the
audio keeps true timing. `english1.mp4` and `others.mp4` are exactly constant.

**Q-1 change** (`src/videowipe/timing.py`, two small hooks in
`STTNInpainter.inpaint`, `docs/source-timing.md`). One ffprobe call lists the
packet timestamps (demux only). A source is constant when one of its declared
rates fits every frame within one time-base tick, so millisecond-rounded
Matroska at 30000/1001 counts as constant. For a variable source whose packet
count equals the job's frame count (C1: frames are matched by ordinal only),
frames are piped as a minimal Matroska stream carrying each frame's timestamp
instead of raw video; trials cut the audio at the real time of their first
frame; an offset between the video and audio streams is kept. Anything else
(probe failure, count mismatch, a first frame before the file start, an
ffmpeg that cannot do it, an unrecognised command) falls back to upstream's
behaviour with a one-line warning. The ffmpeg options are chosen by running
ffmpeg once on a small off-grid stream and checking the timestamps come out
exact, not by version (C6).

Two facts found on the way, both measured on ffmpeg 6.1.1: MP4 output
defaults to constant-rate (frames are duplicated to fill gaps) unless
`-fps_mode passthrough` is given; and even then every frame is rounded up to
a grid of one over the guessed frame rate unless `-enc_time_base -1` is also
given (a frame at 0.54 s came out at 0.56 s). The same applies to the second
(HEVC/MKV) output of `feat/h265-output`, so the options are inserted there
too, from this branch, without editing that one.

**Q-1 results.**

- Acceptance test (generated clip, 160 frames with two multi-frame gaps,
  flashes and audio clicks at the same instants, run through the real
  `inpaint` path with a stub model): output packet timestamps equal the
  source's for every frame and the flash-to-click offsets are exact at the
  start, middle and end. With the switch `off` the flashes land 0.1, 0.5 and
  0.29 s from their clicks. Making `auto` behave as `off` fails 14 of the 39
  new tests.
- Real clean of a 3-second cut of `chinese1.mp4` with its gap and audio:
  maximum video timestamp error 0.0 s (0.079 s with the switch off), audio
  unchanged.
- Constant-rate exactness: a 6-second `english1` clip cleaned with upstream
  and with the branch gives byte-identical `clip_clean.mp4` and masks and an
  equal plan; the ffmpeg command is checked against upstream's embedded as an
  oracle.
- Suite: 539 passed on the branch (500 on upstream, plus 39 new); 1,146
  passed with all fourteen fork branches merged; `git merge-tree` is clean
  against each of the other thirteen.
- Probe cost: 0.07 s for `chinese1`; 0.64 s in total for a generated
  70,000-frame MP4. Not measured on a multi-gigabyte Matroska file, where the
  cost is reading the file.
- The PRD's "minimum" (pass the exact rational rate) was measured and is not
  needed: at 24000/1001, 30000/1001 and 60000/1001 the outputs with the
  OpenCV float and with the rational are byte-identical.

**Q-1 limits and open points.**

- ffmpeg 7 and 8 are untested. If the self-test fails there the job falls
  back with a warning, so the failure is safe; the owner should run
  `python -m pytest -q tests/test_source_timing.py` on the Mac.
- In timestamp mode the encoder sees different rate information, so the
  cleaned pixels differ slightly from a constant-rate encode of the same
  frames (about 45 dB on the real clip). A full clean of `chinese1.mp4` is
  therefore no longer byte-identical to upstream's, by design. The 6-second
  reference clip in SAVEPOINT is constant-rate (the re-encode fills the gap)
  and still is.
- Without `fix/webm-frame-count`, OpenCV's frame count for Matroska is an
  estimate, so variable-rate WebM/MKV falls back with a warning; with that
  branch merged it works (tested).
- **Not changed, still constant-rate arithmetic:** seconds-to-frame
  conversion in the web UI (`web/index.html`: overlay visibility, mask fetch,
  mark-issue, seek step), `server/app.py` (`_trial_range`,
  `_recommended_trial`, the trial handler's `start_seconds` and
  `duration_seconds`) and `server/review.py` (`seconds_to_frames`). On
  `chinese1.mp4` these are 0 to 2 frames off, worst at the start; the bound
  grows with larger or more gaps. Cleaned and trial videos used to sit on the
  same average-rate grid as that arithmetic; they now follow the source
  timeline, so the mismatch there is the same as in the source view. The web
  warning about variable-rate positioning was left as it is, because
  positioning is still approximate. A fix needs the timestamp table in the
  plan or served to the UI: a separate item.
- The messages are English lines on stdout; they are not shown in the web UI
  and have no catalogue entries on `feat/i18n-english`.

**Q-2 evidence** (generated sources through the real encode path before the
change, ffmpeg 6.1.1; the raw and Matroska pipes behave alike). The PRD's
problem statement was partly wrong: chapters and track languages already
survived.

| Item | Before |
|---|---|
| Global metadata (title, artist, comment, creation time) | Lost: ffmpeg takes it from the first input, which is the frame pipe |
| Chapters | Survive (ffmpeg copies them from the first input that has any). Left alone |
| Track language | Survives |
| Track title (MKV/WebM audio) | Lost |
| Audio | Always re-encoded to AAC at ffmpeg's default: about 128k stereo, 70k mono, 275k for 5.1 |
| Channel layout | `5.1(side)` from AC-3 or E-AC-3 came out as six channels with no layout |
| Soft subtitles (mov_text, SubRip, ASS, bitmap) | All dropped |

**Q-2 change** (`src/videowipe/tracks.py`; `src/videowipe/fidelity.py` applies
it and then Q-1's timing to the finished command; `docs/output-tracks.md`).

- Audio, per track and in order: AAC, MP3, AC-3, E-AC-3 and ALAC are
  stream-copied. Opus, FLAC, PCM, Vorbis and anything else are encoded to
  AAC. ffmpeg will write Opus and FLAC into MP4, so leaving them out is a
  player-compatibility judgement, **not tested on QuickTime**.
- AAC rate: 64 kb/s a channel as the base; a lossy source keeps its own rate
  between 64 and 128 kb/s a channel; a lossless source gets 96 kb/s a
  channel. ffmpeg's AAC encoder tops out near 275 kb/s for 5.1 and 7.1
  whatever is asked, so the rule changes little for multichannel.
  `5.1(side)` and `7.1(wide)` are converted to `5.1` and `7.1`.
- `-map_metadata` from the source. A track title is stored as the MP4 handler
  name, the only per-track name the muxer keeps (shown by ffprobe as
  `handler_name`; not checked in a player).
- SubRip, ASS, SSA, WebVTT and mov_text become `mov_text` with language,
  title and default/forced flags; ASS styling is lost. Bitmap subtitles are
  skipped with a notice.
- The planned options are tried on the first half second of the source
  before the job starts (C6). A failed probe or trial run gives upstream's
  handling for the whole job, with a warning, not per track.
- The second (HEVC/MKV) output of `feat/h265-output` gets the global
  metadata from this branch; it already had chapters and its own audio
  handling. It still has no subtitles: not built.

**Q-2 results.**

- Acceptance: source and output ffprobe listings match per stream (codec for
  copied tracks, channels, layout, sample rate, language, flags, title as
  handler name), as do format tags and chapters; copied audio is
  bit-identical packet for packet for AAC (mono, stereo, 5.1), AC-3, E-AC-3,
  MP3 and ALAC, from MP4 and MKV. Not compared, because they legitimately
  differ: bit rate, start time, duration, frame counts, codec tag, encoder
  and handler tags, the video stream, and the audio default flag (the MP4
  muxer marks the first audio track default regardless).
- With Q-1: the click-track test holds with copied audio, including an MKV
  whose audio starts 21 ms early and a source whose video starts 0.5 s late.
- Real clean of a 6-second `english1` clip with audio and added metadata:
  283 audio packets bit-identical to the source, video timing unchanged,
  metadata carried. With `VIDEOWIPE_OUTPUT_TRACKS=off` the output is
  byte-identical to upstream's. The no-audio reference clip is byte-identical
  to upstream's by default.
- Suite: 594 passed on the branch (55 new); 1,201 passed with all fourteen
  fork branches merged; `git merge-tree` clean against each of the other
  thirteen. Making `auto` behave as `off` fails 33 of the 55 new tests.

**Q-2 limits and open points.**

- By default a source with audio, metadata or subtitles no longer gives a
  file byte-identical to upstream's (the video stream is produced the same
  way); set `VIDEOWIPE_OUTPUT_TRACKS=off` for byte comparisons.
- Nothing was tried in a player, and only ffmpeg 6.1.1 was used.
- A no-audio, no-metadata source prints "Keeping source metadata." though
  nothing is mapped.
- An MKV whose audio starts before zero has its chapters shifted by that
  amount (21 ms in the test). This is ffmpeg's behaviour and was there
  before.
- One upstream test (`tests/test_boundaries.py`,
  `test_sttn_inpaint_preserves_audio_and_reports_progress`) now pins the
  switch to `off`, because it asserts upstream's audio arguments.
- Trials still re-encode audio and carry no metadata or subtitles.

**Q-4 (warning).** `src/videowipe/source_checks.py`, called from
`plan.compute_source`: ffprobe gives the pixel format, bits per raw sample
and transfer; a PQ or HLG transfer, or more than 8 bits, prints one warning
per source per process. A source that cannot be probed is not reported. The
web UI's own HDR warning is unchanged and still says nothing about 10-bit SDR
sources (a UI string, which would also need a catalogue entry on
`feat/i18n-english`). Keeping the depth or the HDR signal is the long-term
item in §9 and is not started.

**Q-10.** Read against ProPainter's current `inference_propainter.py`
(fetched from its repository; other versions not checked):

- `--save_fps` is declared `type=int`, so the PRD's fix ("pass the float")
  would make ProPainter exit on its arguments. The adapter now passes the
  rounded rate, asks for the frames as well (`--save_frames`) and encodes
  them itself at the source's real rate.
- ProPainter names its result folder after its input, which is the adapter's
  `frames` directory, while the adapter looked in a folder named after the
  source video. Unless the video happened to be called `frames`, the adapter
  would have ended with "ProPainter output not found" after ProPainter had
  finished. Both locations are now accepted. **This is from reading the two
  programs, not from a run**: ProPainter needs its own checkout, weights and
  in practice a GPU, none of which were available.
- If no frames come back, ProPainter's own video is used, with a warning when
  the rate is not a whole number.
- Not changed: the adapter's output has no audio and a constant rate (its
  input rate is whatever imageio reports), and all frames are held in memory
  to encode.

Suite after Q-4 and Q-10: 622 passed on the branch; 1,229 with all fourteen
fork branches merged; `git merge-tree` clean against the other thirteen.

## 14. Open questions

1. ~~Which Apple Silicon chip and how much memory is the target machine?~~
   **Answered: M2 Pro, 32 GB.** This is the reference machine for P0-3 only;
   see C6 — later chips and larger memory must benefit without retuning.
2. Is a long clip with known subtitle timings available for Q-6?
3. Should these go upstream (they are mostly upstream bugs and limits) or stay
   on the fork?
4. Are 10-bit, HDR or 5.1 sources in regular use, or is a clear warning
   enough for now?
5. How was `detext_trial.pth` trained — on masked or unmasked frames (Q-8)?
