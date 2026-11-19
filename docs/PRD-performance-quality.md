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
| 1 | `perf/instrumentation` | P0-1, P0-2; then record P0-3 on the Mac | — |
| 2 | `perf/refine-exact` | P1-1, P1-2, P1-8 | equality tests |
| 3 | `perf/server-exact` | P1-3, P1-4, P1-5 | equality tests |
| 4 | `perf/plan-masks` | P1-6 | equality tests |
| 5 | `perf/mps` | P1-7 (done), P2-1 | C4 + 15% bar |
| 6 | `perf/detector-calls` | P2-3, P2-4 | C3 baselines |
| 7 | `fix/output-fidelity` | Q-1 (minimum), Q-2, Q-4 (warning), Q-10 | ffprobe comparisons |
| 8 | `fix/ocr-crop` | Q-5 | decision baseline |
| 9 | `feat/duration-sampling` | Q-6 | ground-truth clip + baselines |
| 10 | later | P2-2, P2-5, P2-6, P2-7, Q-3, Q-7…Q-9, U-1…U-4 | as stated |

Steps 2–4 are safe to start immediately after step 1; nothing in them changes
results. Steps 5, 6 and 9 must not be merged without their gate numbers.

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
