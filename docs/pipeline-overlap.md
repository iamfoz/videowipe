# Pipeline overlap (PRD item P2-5)

The clean loop used to do one thing at a time: decode a segment and resize it
for the model, run the model, decode the segment again, compose each frame and
write it to ffmpeg. While the model ran, the CPU cores that decode, blend and
the ffmpeg encode need sat idle, and while ffmpeg was encoding, the model sat
idle. The overlap lets those stages run at the same time. It changes timing
only: the bytes written, and their order, are the same.

`VIDEOWIPE_PIPELINE_OVERLAP` selects it:

| Value | Meaning |
| --- | --- |
| `auto` (default) | On when this process may use at least 3 CPUs and the machine reports enough free memory for two frames per queue; otherwise off. |
| `on` | Always on (even on one core or with little free memory; the queues shrink to one frame). |
| `off` | Today's serial code path: the original `_compose_frame` and `_write_frame` are called on the loop's thread and the readers are not wrapped. |

`VIDEOWIPE_PIPELINE_DEPTH=N` sets the frames each queue holds (default: one
segment, at most 32, reduced to fit the memory budget below).

## What runs where

```
reader thread      source reader.read() ───────▶ queue ─┐
                                                        ├▶ main thread: resize, mask facts, cache,
output-reader thread   second reader.read() ──▶ queue ─┘      model (one worker), progress
                                                        │
                                                        ▼ queue
writer thread      compose frame  ─▶  write to ffmpeg stdin
```

* **Reader threads** decode ahead of the loop. They only call `read()` in order
  and never skip, seek or reorder, so the loop sees exactly the frames a direct
  read would give, including the end of stream and any exception (raised by the
  `read()` that reaches it, after the frames before it). `grab()` is offered
  only when the reader has it, so the loop chooses between `grab` and `read` as
  before (a read-ahead reader does decode the frames of segments that need no
  pixels with `read()`, so those cost the colour conversion that `grab()` would
  have skipped, on the reader thread). The caller's own reader is never
  released by the overlap. Only the real `cv2.VideoCapture`, or an object with a
  true `videowipe_prefetch_safe` attribute, is read ahead; test doubles and any
  other reader object are used as they are, on the loop's thread.
* **The main thread** still evaluates masks, reads and writes the prediction
  cache, resizes the model input, runs the model and reports progress, in the
  same order and with the same arguments as before.
* **The writer thread** calls the original `_compose_frame` and `_write_frame`,
  one frame at a time in submission order, so ffmpeg (and the Matroska wrapper
  used for variable-rate sources) receives the same stream. It is the only
  thread that touches the pipe until the loop has finished.

The per-mask cache (`_MaskCache`) is now used from two threads, so its
operations (`get`, `admit` and the accounting in `float_alpha`) hold a lock. The
cache only ever decides what is kept in memory, never a value.

## Bounds

* Three bounded queues, each at most `depth` frames, so at most
  `3 x depth` full-resolution frames wait in queues, plus one in hand in each
  thread. `depth` is the segment length (25 by default), at most 32.
* The three queues together are limited to
  `min(available memory / 8, 1 GiB)`; on a machine that reports less memory,
  `depth` falls. 1080p frames are 6.2 MB: depth 25 is about 470 MB at most
  (measured +380 MiB peak resident on `chinese1.mp4`); 852 x 480 frames are
  1.2 MB (+115 MiB measured).
* A full queue blocks its producer, so the loop can never run further ahead
  than the queues allow.

## Errors, cancellation and progress

* An exception in the writer (a failed compose, a broken pipe) is stored, the
  queue is aborted so that a blocked producer wakes at once, and the original
  exception is raised in the main thread at its next frame read or write, or
  when the loop finishes. It is the same exception object a serial run would
  have raised, a little later: by at most the frames already queued.
* An exception in the main thread (including `ProcessingCancelledError` raised
  by a progress callback, which is how cancellation arrives) aborts every
  stage in the `finally` block: queued frames are dropped, not written, readers
  are stopped and joined, and the writer is given two seconds to leave a write
  that is already under way before ffmpeg is terminated as before.
* Progress callbacks are called by the main thread at the same points with the
  same values. One difference: a segment is reported when its frames have been
  handed to the writer, so the figure can run ahead of the encoder by up to
  `depth` frames. The `encode` phase starts only after the writer has drained.

## `benchmark.json`

With the overlap on, `blend_s`, `pipe_write_s` and `decode_output_s` measure
only what the loop itself spent (queueing, not blending or writing), so the
six-part split no longer sums to `inpainting_s`. The work is reported under
`overlap_compose_s`, `overlap_write_s` (the writer thread's time inside
ffmpeg's pipe, i.e. waiting for the encoder), `overlap_submit_wait_s` (the loop
waiting for room in the writer queue), `overlap_writer_idle_s`,
`overlap_reader_wait_s`, `overlap_output_reader_wait_s` and each queue's
`..._high_water`, plus `pipeline_overlap` and `pipeline_depth`. With the
overlap off none of these keys exist.

## Mac timing command

Run from the repository root on the Mac, once with the overlap off and once
with the default. Each run plans, then cleans; by the PRD's baseline that is 4 to 5 minutes a run,
so about half an hour per video.

```sh
for mode in off auto; do
  VIDEOWIPE_PIPELINE_OVERLAP=$mode python scripts/perf_baseline.py \
      input/detext_examples/chinese1.mp4 input/detext_examples/english1.mp4 \
      -o result/overlap-$mode --runs 3
done
python - <<'EOF'
import json, statistics
for mode in ("off", "auto"):
    report = json.load(open(f"result/overlap-{mode}/perf_baseline.json"))
    for video in report["videos"]:
        runs = [run["clean"] for run in video["runs"]]
        name = video["video"]["path"].rsplit("/", 1)[-1]
        print(mode.ljust(5), name.ljust(14),
              "inpainting_s median", statistics.median(r["timing"]["inpainting_s"] for r in runs),
              "wall_s median", statistics.median(r["wall_s"] for r in runs),
              "sha256", {r["output_sha256"][:12] for r in runs})
EOF
```

The two `sha256` sets for each video must be the same single value (the output
is byte-identical); the medians are the comparison.

## Measurements

Machine: 4 cores (a container), 16 GB, ONNX Runtime 1.29 on CPU, OpenCV 4.14,
ffmpeg with libx264 and libx265. **The machine was shared with three other
workers throughout; its load average was 20 to 43 on 4 cores, so the model and
ffmpeg ran at a fraction of a core each and every timing below carries a
scatter of 10 to 25 %.** All timings are medians of 3 runs, off and on
interleaved, each in its own process, from `inpainting_s` in `benchmark.json`
(peak resident memory is that process's `ru_maxrss`).

### Where the time goes (overlap off)

`english1.mp4` (852 x 480, 522 frames), model replaced by a sleep of 100 ms
per frame and band (see below), seconds: model 50.2, decode for the model 3.4,
resize 0.2, second decode 1.4, blend 0.7, write to ffmpeg 8.8, finalise 1.6.
`chinese1.mp4` (1080p, 591 frames, variable rate), sleep 40 ms: model 24.4,
decode 9.5, second decode 4.1, blend 4.3, **write 41.6** (libx264 `medium`
on a loaded CPU encodes at about 14 frames a second and the write blocks on
it), finalise 8.1. Every stage except the model can run beside it, and so can
the encoder, which only works while frames are being written.

### Identity

Byte for byte against `off`, SHA-256 of the output file, on this machine:

| Run | Result |
| --- | --- |
| `english1.mp4` full clean, model replaced by a deterministic stub | identical (`d1d4dda2b28c` with the first stub; `9322b28e8522` with the cheap stub, in every `english1` run of the wall-time table below, 18 runs) |
| `chinese1.mp4` full clean (variable rate: `MatroskaFrameWriter`), stub | identical (`406096450924` with the first stub; `6dcf1ab16f0c` with the cheap stub; `9501923b52d2` with the cheap stub and the fast-encoder shim) |
| `english1.mp4` trial, frames 125 to 175, **real ONNX model** | identical (`fc0bd3ca8080`) |
| `chinese1.mp4` trial, frames 200 to 225, **real ONNX model** | identical (`bbad359d2277`, 8 runs, 4 off and 4 on, one hash) |
| trial ranges 125 to 175 and 200 to 300, stub | identical (`76eabb429b12`, `2c14f3b42d70`) |
| `english1.mp4` with `VIDEOWIPE_OUTPUT_CODEC=h265`, stub | the H.264 MP4 is identical (`9322b28e8522`, equal to the run without HEVC); the HEVC `.mkv` differs in 60 bytes, all in its first 330 bytes (segment UID and date, which ffmpeg randomises: two plain remuxes of one file differ in the same 60 bytes), and its video packets (`-c copy -f md5`) and decoded frames are identical |

What was **not** run: a full clean of either sample with the real model. At
this machine's load the real model needed about 12 minutes for 50 frames, so a
full clean was several hours per run. The full cleans above use a stub in place
of the model, with real decoding, composing, blending and ffmpeg encoding; the
stub is a deterministic function of the decoded frames, so a frame that was
decoded, resized, composed or written out of order or differently shows up in
the bytes. The real model is exercised by the two trials. The unit tests also
compare the exact bytes written to the pipe, off against on, over static and
per-frame masks, `dual`, a trial range that does not start on a segment, queue
depths 1, 2 and the default, and real and fake readers.

### Wall time

The model cannot be hidden by anything on this machine's CPU backend, so the
gain is measured two ways.

**Real model (ONNX on 4 shared cores), `chinese1.mp4` trial 200 to 225:**
`inpainting_s` off 251.2 (runs 251.2, 322.6, 197.6), on 235.7 (281.7, 235.7,
178.5); process CPU time 155.6 s against 159.5 s; peak resident memory 3,560
against 3,957 MiB. The difference (6 %) is smaller than the scatter between
runs of the same mode (up to 50 %), so **nothing is claimed for the CPU
backend**: the model uses every core, so there is little for the overlap to
use. It does no harm that can be seen (CPU time within 3 %).

**Model replaced by a sleep that releases the GIL**, as an accelerator does
(real decoding, composing and ffmpeg; the prediction is the resized input, so it
still depends on every decoded frame). This shows what the overlap hides when
the cores are free while the model works, as on an Apple GPU. It is a
simulation of that case, not a measurement of it:

| Case | off | on | change | peak memory off / on |
| --- | --- | --- | --- | --- |
| `english1`, no model time (encoder bound) | 21.4 | 20.3 | -5 % | 259 / 376 MiB |
| `english1`, 30 ms per frame and band | 29.2 | 20.5 | -30 % | 259 / 373 MiB |
| `english1`, 100 ms (model is 76 % of off) | 66.1 | 54.2 | -18 % | 257 / 368 MiB |
| `chinese1`, 40 ms, libx264 `medium` (encoder bound: write 41.6 s) | 93.3 | 86.5 | -7 % | 641 / 1,034 MiB |
| `chinese1`, 70 ms, encoder preset forced to `ultrafast`\* | 68.3 | 47.3 | -31 % | 432 / 927 MiB |

\* A shim on `PATH` that rewrites `-preset medium` to `ultrafast` for both
modes, standing in for an encoder that is cheap next to the model. The model
is 61 % of the off time here; decode, blend and write are 33 %, which is much
more than the 10 % or so the PRD's M2 Pro baseline shows (blend 4.7 %, decode
under 1 % each), so this row is an upper bound for the Mac, not an estimate.

What the overlap can gain is the part of the run that is not the model:
decode, blend, the write to ffmpeg and the encoder's own time, minus the one
segment at the start and the drain at the end that cannot overlap. When the
encoder is slower than the model (the encoder-bound rows) the run is limited by
it, and the gain is the model time that now hides behind encoding.

Components, `chinese1`, 70 ms, fast encoder, median of 3: off 59.6 s; writer
thread only 54.6 (-8 %, 536 MiB); plus read-ahead of the second decode 53.0
(-11 %, 796 MiB); plus read-ahead of the model's decode 45.7 (-23 %, 947 MiB).
All three are kept: the memory is bounded by the queue depth and falls back
with the available memory.

Peak memory with the overlap on is 100 to 400 MiB higher on the CPU runs
(`depth x 3 x frame size`: 25 x 3 x 1.2 MB = 92 MB for 852 x 480, 25 x 3 x 6.2
MB = 467 MB for 1080p). `VIDEOWIPE_PIPELINE_OVERLAP=off` removes it.

### Tests

`tests/test_pipeline_overlap.py`: ordering (queue, reader, writer), error
propagation (compose, write, decoder, a producer blocked on a full queue),
cancellation (a progress callback that raises; a writer stuck mid-write), the
queue bound (unit and in a real run), the decision (`off`, `on`, `auto` by
cores and memory, depth from memory), the exact bytes off against on, and the
mask cache's lock. Mutation check: each of the following was applied to the
code in turn and made at least one test fail: a queue that ignores its bound,
LIFO instead of FIFO, a swallowed writer error, an abort that does not drop
the queue, dropped decoder errors, a reader that skips every other frame,
`auto` ignoring the core count, `off` still overlapping, arguments reordered
in the compose token, a blocked producer that sees a bare abort instead of the
stage's error, every reader read ahead whether safe or not, and the mask
cache's lock replaced by a no-op.
