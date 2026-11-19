# Save point

A hand-over for a new session. Read this, then `docs/PRD-performance-quality.md`
(same branch), which holds the full plan, evidence and measured results.

## Repository

`iamfoz/videowipe` is a fork of `KKenny0/videowipe` (local removal of burned-in
subtitles and watermarks from video). Upstream's last commit is `20872d4`.
`main` is upstream plus one fork-only commit (`f1d326d`: `.mcp.json` and the
vendored Switchyard MCP servers in `.switchyard-mcp/`).

## Branches

Every branch starts from upstream `20872d4` unless marked as stacked. Any
combination merges without conflicts (66 of 66 pairs checked), and with all
of them merged the suite passes (1,020 tests) and an end-to-end clean is
byte-identical to upstream's output.

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
| `docs/performance-prd` | this branch | The PRD and this file |
| `chore/switchyard-mcp` | `f1d326d` | Merged to `main`; can be deleted |
| `iamfoz/beautiful-ramanujan-cw1ggy` | `f9b21f3` | Stale early branch; the owner should delete it (a cloud session cannot) |

The owner also keeps a local `combined` branch that merges the feature
branches for day-to-day use; it is not on GitHub.

## Measured on the reference machine

Apple M2 Pro, 12 cores, 32 GB, torch 2.15 on MPS. Clip `chinese1.mp4` (1080p,
591 frames, two active bands), medians of three runs:

- Whole run 135.5 s → 127.5 s after Phases 1 and 2 (−5.9%).
- Blend 6.30 s → 0.74 s. Planning 44.2 s → 41.5 s. Model inference unchanged
  at ~80 s (136 ms a frame) and now 63% of the run.
- Half precision on MPS: 1.4% faster and failed the output gate. **Not
  adopted; `auto` stays float32 on MPS.** The setting remains for later chips.
- Long file (1080p AV1, 70,819 frames), planning only: 102 s, all of it two
  decode passes. The band-fallback pass (58 s) should now be gone; **the
  re-run that confirms this has not been recorded.**

## Work in hand when this was written

Two items were started and then stopped unfinished; nothing from them is
committed. Both are specified in the PRD.

1. **P2-2, skip model windows that cannot affect any emitted pixel.** New
   branch `perf/skip-windows` stacked on `perf/blend-box`. Needed frames must
   be bit-identical; the prediction cache must never serve a partly computed
   segment as complete. Benefit is workload-dependent — measure it.
2. **Reuse contrast-recovery results.** New branch `perf/recovery-reuse`
   stacked on `perf/detector-calls`. Evidence first: count how often a
   recovery call's inputs are bit-identical to an earlier one, including
   boundary probes repeating work refine already did; ship code only if a real
   saving exists. Also measure, without shipping, P2-3(c) (cheaper crop inputs
   for recovery calls). Recovery is still about a third of detector calls in
   refine and probes on the reference clip, with no hits there. A brightness
   pre-gate was investigated and rejected (it would discard genuine recoveries
   on fades and dissolves).

Not started: Phase 3 of the PRD (output fidelity: variable-frame-rate timing,
audio copy, metadata and chapters, colour tags, 10-bit/HDR, the OCR
wrong-frame crop, duration-scaled sampling) and the web UI items.

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
- End-to-end reference: make a 6-second clip
  (`ffmpeg -i input/detext_examples/chinese1.mp4 -t 6 -c:v libx264 -crf 18 -an clip.mp4`),
  clean it with upstream `20872d4` (`PYTHONPATH=src python -m videowipe.cli
  clean clip.mp4 -o base`), and compare `clip_clean.mp4`, `wipe_plan.json`
  (ignoring `created_at`) and `wipe_plan_masks.npz` with a clean from the
  branches under test. About 3.5 minutes per clean on a 4-core CPU.
- Mac measurements are run by the owner:
  `python scripts/perf_baseline.py VIDEO -o OUT [--runs 3] [--no-full] [--trial-seconds 3]`,
  inside the project's virtual environment (`source .venv/bin/activate`).
- Cloud sessions: the Default environment allows `switchyard.forryan.net`
  and holds its API key. Pushes to `main` from a cloud session are refused by
  an automatic safety check; the owner pushes to `main`. Deleting remote
  branches is also refused.
- Switchyard MCP (`switchyard_code`, `switchyard_handles`) loads from
  `.mcp.json` only in sessions that start on a branch containing it (`main`).
  Without `SWITCHYARD_URL` in the session's environment the tools work
  locally and report nothing to Switchyard.
- Environment switches added by the fork: `VIDEOWIPE_LANG`,
  `VIDEOWIPE_OUTPUT_CODEC`, `VIDEOWIPE_PRECISION`, `VIDEOWIPE_PERF_STATS`,
  `VIDEOWIPE_MASK_WINDOW_BLUR`, `VIDEOWIPE_BLEND_WINDOW_RESIZE`,
  `VIDEOWIPE_BLEND_CACHE_MB`, `VIDEOWIPE_BAND_CROP_BUDGET_MB`.
