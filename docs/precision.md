# Numeric precision (`VIDEOWIPE_PRECISION`)

The inpainting model has three stages: an encoder, a transformer and a decoder.
The transformer is about 70% of the model's arithmetic. The PyTorch backend can
run it in half precision (16-bit floating point), which cuts memory traffic and
can be faster. The encoder and decoder always run in 32-bit float.

## The setting

Set the environment variable before starting VideoWipe:

| Value | Behaviour |
|---|---|
| `auto` (default) | Exactly what earlier versions did: half-precision autocast on CUDA, 32-bit float on Apple's MPS and on the CPU. |
| `float32` | Never use reduced precision, on any device (this includes CUDA, which could not be switched off before). |
| `float16` | Use half precision for the transformer on devices that support it: CUDA and MPS. The CPU has no benefit from it and stays in float32. |

Anything else, including an empty value, behaves as `auto`. The ONNX backend
ignores the setting.

Half precision on MPS is **not** enabled by `auto` yet. It has to be measured on
a real Mac first (see below). When it has been, the change is one line, marked
with a comment in `wants_reduced_precision` in `src/videowipe/backends.py`.

## How half precision is chosen on MPS

Nothing depends on the chip name. When `float16` is requested on MPS:

1. If the installed PyTorch supports `torch.autocast("mps", dtype=torch.float16)`
   (detected by using it once on a tiny matrix product, not by version number),
   the transformer runs under autocast.
2. Otherwise a float16 copy of the transformer alone is made and used, with the
   features cast to half on the way in and back to float32 on the way out.
3. Before the first real use, a self-test runs once per backend instance. It
   passes a fixed pseudo-random batch (twelve 640x120 frames, seeded, through
   the real encoder) through the transformer in float32 and in half precision,
   then through the decoder, and requires every value to be finite, the mean
   absolute pixel difference to be at most 0.5 (on the 0-255 scale, the quality
   gate) and the largest difference to be at most 16. The model is not changed
   by the test.
4. If the self-test fails or raises, one warning is logged and the backend
   carries on in float32.

## What is reported

The precision actually in use is recorded in `benchmark_metadata()`, in the
`engine:` identity printed by `scripts/perf_baseline.py`, and in the
prediction-cache identity, so a cache built in one precision is never reused in
another.

| Label | Meaning |
|---|---|
| `float32` | 32-bit float (any device) |
| `float16-autocast` | CUDA autocast (the existing behaviour) |
| `float16-autocast-mps` | MPS autocast |
| `float16-half-mps` | MPS with the explicit half-precision transformer copy |
| `float32-mps-fallback` | Half precision was requested on MPS but failed the self-test |

When the request could not be honoured (`float16` on the CPU, or a failed
self-test) the metadata also carries a `precision_note` saying why.

## Measuring speed and quality on a Mac

`scripts/perf_baseline.py` lives on the `perf/instrumentation` branch. With that
branch checked out, run the same clip in each precision, three runs each:

```
VIDEOWIPE_PRECISION=float32 python scripts/perf_baseline.py CLIP -o out-f32 --runs 3
VIDEOWIPE_PRECISION=float16 python scripts/perf_baseline.py CLIP -o out-f16 --runs 3
```

Check the `precision` value in each run's printed `engine:` line: it should read
`float32` and `float16-autocast-mps` (or `float16-half-mps`). If the second says
`float32-mps-fallback`, the self-test failed and the log has the numbers.

Compare the speed using the median model inference time of the three runs in
each summary. The PRD asks for at least 15% faster.

Then compare the quality of the two cleaned videos, using the plan from the
float32 run for both:

```
python scripts/compare_clean_outputs.py \
    out-f32/<clip>/run-001/clean/<clip>_clean.mp4 \
    out-f16/<clip>/run-001/clean/<clip>_clean.mp4 \
    --plan out-f32/<clip>/run-001/plan/wipe_plan.json
```

`<clip>` is the video's file name without its extension. The tool prints the
mean and maximum difference over the whole frame and inside the plan's removal
mask, a flicker figure for each video, and `PASS` or `FAIL` against the gate:
mean absolute difference inside the mask of at most 0.5 (0-255 scale) and a
temporal-difference error of at most 0.5. Add `--json report.json` to keep the
figures. The exit status is 0 for PASS, 1 for FAIL, 2 if the files cannot be
compared and 3 if no plan was given (so there is no mask to judge).

Both files have been through H.264 CRF 18 encoding, so encoder noise is part of
every figure and the gate is conservative: a result that is identical before
encoding can still show a small non-zero difference. Watch the two videos side
by side as well; a number cannot replace looking for text remnants, edge
smearing and flicker.

If either the quality gate or the 15% speed bar fails, leave `auto` as it is.
