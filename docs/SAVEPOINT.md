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
merged the suite passes (1,146 tests) and an end-to-end clean of a
constant-rate clip is byte-identical to upstream's output. With
`fix/output-fidelity` merged, a variable-rate source (the full
`chinese1.mp4` is one) is deliberately no longer identical: its frames keep
their own timestamps.

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
| `fix/output-fidelity` | `48cc050` | Q-1: source timestamps kept for variable-rate video (`src/videowipe/timing.py`, `VIDEOWIPE_SOURCE_TIMING`, `docs/source-timing.md`). Q-2, Q-4 (warning) and Q-10 are to follow on this branch |
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

## Phase 3 (authorised by the owner; Q-1 first)

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

Not started: the rest of Phase 3 (audio copy, metadata and chapters, colour
tags, 10-bit/HDR, the OCR wrong-frame crop, duration-scaled sampling) and the
web UI items. The PRD's plan puts Q-2, Q-4 (warning) and Q-10 on
`fix/output-fidelity` after Q-1.

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
  `VIDEOWIPE_SOURCE_TIMING`.
