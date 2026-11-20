# Save point

A hand-over for a new session. Read this, then `docs/PRD-performance-quality.md`
(same branch), which holds the full plan, evidence and measured results.

## Repository

`iamfoz/videowipe` is a fork of `KKenny0/videowipe` (local removal of burned-in
subtitles and watermarks from video). Upstream's last commit is `20872d4`.
`main` is upstream plus two fork-only commits (`f1d326d`: `.mcp.json` and the
vendored Switchyard MCP servers in `.switchyard-mcp/`; `a708c67`: those
servers updated to 1.24.0).

## Branches

Every branch starts from upstream `20872d4` unless marked as stacked. Any
combination merges without conflicts (66 of 66 pairs checked earlier; the
three newest branches checked against every other), and with all of them
merged the suite passes (1,355 tests) and an end-to-end clean of a
constant-rate clip is byte-identical to upstream's output. With
`fix/output-fidelity` merged, a variable-rate source (the full
`chinese1.mp4` is one) is deliberately no longer identical: its frames keep
their own timestamps; and any source with audio, metadata or subtitles
differs in those tracks unless `VIDEOWIPE_OUTPUT_TRACKS=off` is set (with it
off the output is byte-identical to upstream's, measured).

| Branch | Head | Contents |
|---|---|---|
| `feat/i18n-english` | `e083cfb` | Translation overlay (source-text-keyed catalogue, `web/i18n.js`, CLI wrapper `videowipe.i18n_cli`), English (en-GB), `scripts/i18n_sync.py`, `docs/i18n.md` |
| `fix/webm-frame-count` | `e3758a3` | Exact frame count for WebM/Matroska via ffprobe |
| `feat/h265-output` | `90e2a7b` | Optional 10-bit HEVC + Opus MKV (`--codec h265`, web "Output format"), VideoToolbox or libx265, `docs/h265-output.md` |
| `fix/progress-display` | `360aecb` | Processing progress on stable separate lines |
| `perf/refine-exact` | `fa1ded3` | Linear refine bookkeeping, `grab()` for unused frames, early exit, concurrent trial skip |
| `perf/server-exact` | `918bc40` | Thumbnails in one pass, cached mask rendering off the job lock, source hashed once |
| `perf/plan-masks` | `727893e` | Bisected mask rows, windowed blur behind a self-test |
| `perf/mps` | `a9413f6` | MPS allocator cleanup; `VIDEOWIPE_PRECISION` (`auto`/`float32`/`float16`), `scripts/compare_clean_outputs.py`, `docs/precision.md` |
| `perf/instrumentation` | `f805b92` | **Stacked on `perf/refine-exact`.** Split timers, detection counters (`perfstats.py`), `scripts/perf_baseline.py` |
| `perf/detector-calls` | `1e37cb5` | **Stacked on `perf/instrumentation`.** Band fallback from stored crops (no second decode), recovery pruning |
| `perf/blend-box` | `4133f14` | **Stacked on `perf/instrumentation`.** Compositing only inside the mask's box |
| `perf/skip-windows` | `8c451f7` | **Stacked on `perf/blend-box`.** Model windows that cannot reach an emitted pixel are skipped; partial prediction-cache entries; `scripts/skip_windows_estimate.py` |
| `perf/recovery-reuse` | `fa4503c` | **Stacked on `perf/detector-calls`.** Per-run memo for repeated contrast-recovery calls; `scripts/recovery_reuse_probe.py` |
| `fix/output-fidelity` | `291da43` | Q-1: source timestamps kept for variable-rate video (`timing.py`, `VIDEOWIPE_SOURCE_TIMING`, `docs/source-timing.md`). Q-2: audio copied, metadata and text subtitles kept (`tracks.py`, `fidelity.py`, `VIDEOWIPE_OUTPUT_TRACKS`, `docs/output-tracks.md`). Q-4: warning for sources above 8 bits or HDR (`source_checks.py`). Q-10: ProPainter adapter encodes at the real frame rate |
| `fix/ocr-crop` | `faf4746` | Q-5: OCR reads each box's own frame, not the first sampled frame (`detect.py`) |
| `fix/colour-tags` | `0b98964` | **Stacked on `fix/output-fidelity`.** Q-3: output carries the source's colour tags with a matching conversion (`colour.py`, `VIDEOWIPE_COLOUR`, `scripts/colour_roundtrip.py`, `docs/colour.md`) |
| `feat/inpaint-quality` | `14c6d98` | Q-7, Q-8, Q-9: ground-truth harness and results only (`scripts/inpaint_quality_*.py`, `docs/inpaint-quality-experiments.md`); no `src/` change |
| `feat/duration-sampling` | `4ab23be` | Q-6: sample count grows with duration for balanced and sensitive (`sampling.py`, one call in `planning.prepare`, `VIDEOWIPE_SAMPLE_INTERVAL`, `VIDEOWIPE_SAMPLE_MEMORY_MB`, `docs/sampling.md`) |
| `docs/performance-prd` | this branch | The PRD and this file |
| `main` | `a708c67` | Upstream plus the Switchyard MCP setup only; no feature branch is merged into it |
| `chore/switchyard-mcp` | `f1d326d` | Merged to `main`; can be deleted |
| `chore/switchyard-mcp-1.24` | `a708c67` | Merged to `main` (fast-forward); can be deleted |
| `iamfoz/beautiful-ramanujan-cw1ggy` | `f9b21f3` | Stale early branch; the owner should delete it (a cloud session cannot) |

The owner also keeps a local `combined` branch that merges the feature
branches for day-to-day use; it is not on GitHub.

## Measured on the reference machine

Apple M2 Pro, 12 cores, 32 GB, torch 2.15 on MPS. Clip `chinese1.mp4` (1080p,
591 frames, two active bands), medians of three runs:

- Whole run 135.5 s → 127.5 s after Phases 1 and 2 (−5.9%); 120.1 s with
  skip windows as well.
- Blend 6.30 s → 0.74 s. Planning 44.2 s → 41.5 s. With recovery reuse as
  well, planning is 40.5 s (boundary probes 8.8 s → 5.9 s). Model inference
  unchanged at ~80 s (136 ms a frame) and now 63% of the run.
- Half precision on MPS: 1.4% faster and failed the output gate. **Not
  adopted; `auto` stays float32 on MPS.** The setting remains for later chips.
- Long file (1080p AV1, 70,819 frames), planning only: 102 s, all of it two
  decode passes. The band-fallback pass (58 s) should now be gone; **the
  re-run that confirms this has not been recorded.**

## Last work finished

Both items that were in hand are done and pushed; details and numbers are in
the PRD's Phase 2 status.

1. **P2-2 skip windows** (`perf/skip-windows`): exact, but the saving is small
   on the samples (windows saved: 10.9%, 1.0%, 0.0%). It pays on sparse text,
   trials and partly cached re-runs.
2. **Recovery reuse** (`perf/recovery-reuse`): exact; 6–8% of planning's
   detector calls on two samples, under 1% on the third. Timed on the
   reference Mac: 2.5–2.9 s off boundary probes on those two. P2-3(c) (cheaper
   crop inputs) was measured and not built: 10–15× faster per call but it
   changes plans and loses real recoveries.

Skip windows is measured on the reference Mac: `chinese1` model inference
79.98 s → 70.96 s (−11.3%), whole run 129.66 s → 120.09 s; nothing
measurable on the other two; output byte-identical on MPS with the switch on
and off.

Recovery reuse is measured on the reference Mac too (planning only, median of
three runs, on against `VIDEOWIPE_RECOVERY_REUSE=off`): boundary probes
8.83 s → 5.93 s on `chinese1` and 6.28 s → 3.76 s on `english1`, about 41 ms
for each of the 70 and 72 detector calls avoided, as estimated. Whole
planning 41.71 s → 40.51 s (`chinese1`) and 49.00 s → 44.85 s (`english1`);
those two are blurred by refine varying about 1.5 s between runs with
identical work, so quote the boundary-probe figure. All other counters are
identical on and off. No disturbed runs. Nothing from Phases 1 and 2 is left
to measure except the long-file planning re-run noted above.

## Phase 3 (authorised by the owner; Q-1, Q-2, Q-4 warning, Q-5 and Q-10 done)

**Q-1 is done and pushed** on `fix/output-fidelity`; details in the PRD's
Phase 3 status. `chinese1.mp4` turned out to be variable-rate (one 0.12 s
gap), so upstream put its picture up to two frames from the sound. The branch
pipes frames to ffmpeg as a minimal Matroska stream with each frame's own
timestamp when the source is variable; constant-rate sources are untouched
and byte-identical.

Outstanding on Q-1:

- **Not yet run on the Mac's ffmpeg** (built on 6.1.1). The owner runs, on
  `fix/output-fidelity`: `python -m pytest -q tests/test_source_timing.py`,
  and optionally a clean of `chinese1.mp4`, which should print "Keeping the
  source's frame timestamps". A failed self-test falls back with a warning.
- The web UI and server still convert seconds to frames with one constant
  rate (0 to 2 frames off on `chinese1`). Not authorised or started; it needs
  the timestamp table in the plan or served to the UI.

**Q-2 is done and pushed** on the same branch. Evidence first: chapters and
track languages already survived; global metadata, track titles and soft
subtitles were lost and audio was always re-encoded. Now, for full runs,
AAC, MP3, AC-3, E-AC-3 and ALAC are copied (bit-identical packets), other
codecs go to AAC at a rate derived from the source, metadata is mapped from
the source and text subtitles become `mov_text`. Trials are unchanged.

Outstanding on Q-2:

- **Not played in QuickTime or any player, and only ffmpeg 6.1.1 was used.**
  The owner runs `python -m pytest -q tests/test_output_tracks.py` on the Mac
  and opens one cleaned file with audio in QuickTime. Leaving Opus and FLAC
  out of the copy list is a compatibility judgement, not a measurement.
- The HEVC/MKV output of `feat/h265-output` gets the metadata but still no
  subtitles. Not built.

**Q-4 (warning) and Q-10 are done and pushed** on the same branch, written
by the orchestrating session (both small). Q-4 prints one line for a source
above 8 bits or with an HDR transfer. Q-10 was not what the PRD said:
ProPainter's `--save_fps` only takes a whole number, so the adapter now asks
for frames and encodes them at the real rate; and it looked for ProPainter's
result in the wrong folder, which is fixed too. **Q-10 has not been run
against a real ProPainter** (no checkout, weights or GPU here); if the owner
uses the adapter, one run would confirm it.

**Q-5 is done and pushed** on `fix/ocr-crop`. The bug is real but only live
when the optional OCR extra (`rapidocr-onnxruntime`) is installed, and on
the three samples it changes recognised text in 2 of 9 regions and no
classification. The Makefile baselines run with OCR off and cannot see it;
they were also run with `--ocr rapidocr` and are unchanged. Open: the kept
crops have no memory budget.

**Q-3 is done and pushed** on `fix/colour-tags` (stacked on
`fix/output-fidelity`). Measured first: tagged BT.709 HD sources shifted by
up to 40 code values on saturated colours because OpenCV decodes with the
tagged matrix and ffmpeg encodes with BT.601 and writes no tags. The fix
measures what the installed OpenCV does and makes the encode its inverse,
then copies the tags. Untagged sources (all three samples) are
byte-identical. **Owner's Mac check:** `python scripts/colour_roundtrip.py
--probe decode` (seconds), then `python -m pytest -q tests/test_colour.py`,
and look at one cleaned tagged HD file in QuickTime. Open: untagged HD is
wrong upstream and not fixed; full-range sources stay full range.

**Q-7, Q-8 and Q-9 are measured and nothing was built** (`feat/inpaint-quality`
holds the harness and write-up). Area downscale: no gain. Masked model
input: 5 to 14 dB worse, the checkpoint paints the fill back. Seam colour
match: not worth it. The one real gain is cropping the model's input around
the mask instead of the full-width band (+2.3 dB at 1080p, +3.9 dB at 4K in
a scratch prototype); it needs the band and window logic rewritten and is
the best next quality workstream if the owner wants one.

`fix/output-fidelity` now holds everything the PRD's delivery plan gave it.

**Q-6 is done and pushed** on `feat/duration-sampling`: balanced samples
about once a second on videos longer than 50 s (sensitive every 1.25 s past
100 s; fast unchanged). On generated clips with known timings the fixed 50
samples missed about a fifth of lines at 10 to 20 minutes; the rule finds
all but 1 of 198 with fewer detector calls. The three bundled samples are
unchanged. **Evidence is synthetic only**; a real long clip run on the Mac
would be worth having, and the owner has not yet chosen between the 1 s
default and 2 s (cheaper on long videos with no text).

**Phase 3 is complete.** Not started, and not in Phase 3: the long-term
half of Q-4, the web UI items (U-1 to U-4), P2-5 and P2-7, the
constant-rate seconds-to-frames arithmetic in the UI and server found
during Q-1, and the two follow-ups the experiments suggest (cropping the
model's input around the mask; a memory budget for the OCR crops).

Gotcha for new probes: several test suites replace `subprocess.run` or
`subprocess.Popen` and count calls. Run ffprobe/ffmpeg helpers through
`videowipe.timing._run`, which uses the `Popen` bound at import.

## Working rules agreed with the owner

- UK English. Give the actual thing asked for; state assumptions and carry
  on; say how confident each claim is; be brief.
- Commits and branches: author and committer `Martyn Forryan
  <iamfoz@users.noreply.github.com>`; no AI or co-author trailers and no AI
  mention anywhere; branch names `feat/`, `fix/`, `perf/`, `docs/`, `chore/`;
  linear history; never change git config. No pull requests unless asked.
- One branch per workstream from upstream; each must merge with every other
  in any order (check pairs with `git merge-tree --write-tree A B`). Stack a
  branch only when it edits code another fork branch rewrote, and say so.
- Minimal edits to upstream files; prefer new files and hooks.
- Performance changes are either **exact** (prove equality against the
  original functions embedded in tests as oracles, and by an end-to-end clean
  compared byte for byte) or **gated** (measured on the reference Mac and
  checked against the PRD's gates before becoming a default).
- Capability-gated, never machine-gated: choose features by what the running
  machine demonstrates (self-tests, available memory), with an override, so
  later chips and more memory benefit. Nothing is tuned to the M2 Pro.
- Heavy work goes to cheaper models with an explicit effort setting, in
  isolated worktrees; the orchestrating session reviews diffs, re-runs tests
  and the merge checks, and pushes. Agents can be confidently wrong: verify.
- Evidence before code for anything that might not pay: "measured, not worth
  shipping" is an acceptable outcome.

## Practical notes

- Tests: `python -m pytest -q` from the repository root (pytest adds `src`).
- End-to-end reference (the clip is constant-rate even when cut from
  `chinese1.mp4`, because the re-encode fills its gap): make a 6-second clip
  (`ffmpeg -i input/detext_examples/chinese1.mp4 -t 6 -c:v libx264 -crf 18 -an clip.mp4`),
  clean it with upstream `20872d4` (`PYTHONPATH=src python -m videowipe.cli
  clean clip.mp4 -o base`), and compare `clip_clean.mp4`, `wipe_plan.json`
  (ignoring `created_at`) and `wipe_plan_masks.npz` with a clean from the
  branches under test. About 3.5 minutes per clean on a 4-core CPU.
- Mac measurements are run by the owner:
  `python scripts/perf_baseline.py VIDEO -o OUT [--runs 3] [--no-full] [--trial-seconds 3]`,
  inside the project's virtual environment (`source .venv/bin/activate`).
- Cloud sessions: the Default environment allows `switchyard.forryan.net`
  and holds its API key. A fast-forward push to `main` from a cloud session
  worked when the owner asked for it explicitly; do not push to `main`
  otherwise. Deleting remote
  branches is also refused.
- Switchyard MCP (`switchyard_code`, `switchyard_handles`) loads from
  `.mcp.json` only in sessions that start on a branch containing it (`main`).
  Without `SWITCHYARD_URL` in the session's environment the tools work
  locally and report nothing to Switchyard.
- Environment switches added by the fork: `VIDEOWIPE_LANG`,
  `VIDEOWIPE_OUTPUT_CODEC`, `VIDEOWIPE_PRECISION`, `VIDEOWIPE_PERF_STATS`,
  `VIDEOWIPE_MASK_WINDOW_BLUR`, `VIDEOWIPE_BLEND_WINDOW_RESIZE`,
  `VIDEOWIPE_BLEND_CACHE_MB`, `VIDEOWIPE_BAND_CROP_BUDGET_MB`,
  `VIDEOWIPE_SKIP_WINDOWS`, `VIDEOWIPE_RECOVERY_REUSE`,
  `VIDEOWIPE_SOURCE_TIMING`, `VIDEOWIPE_OUTPUT_TRACKS`.
