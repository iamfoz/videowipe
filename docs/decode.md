# Decoding with the right colour matrix

This follows the colour work in [colour.md](colour.md). That change made the encode
the inverse of whatever matrix OpenCV decoded with, and said so in the output's tags.
It could not restore what the decode had already lost. This one fixes the decode.

## The problem

OpenCV converts the source's YCbCr to BGR with a matrix of its own choosing, and
the inpainter, the model and the output all work on those BGR frames. What
`scripts/colour_roundtrip.py --probe decode` measured (see colour.md):

* **the reference Mac's OpenCV** (4.14.0, avcodec 61.19) uses BT.601 for every source.
  It ignores BT.709 and BT.2020 matrix tags and honours the full-range tag;
* **this machine's OpenCV** (4.14.0, avcodec 62.28) honours the BT.709 tag, but still
  reads an *untagged* HD source as BT.601, where every player assumes BT.709.

A BT.709 picture read as BT.601 has its saturated colours pushed outside the RGB
cube, and the decoder clips them. Clipped values cannot be restored by anything
later, so the encode's matrix (colour.md) can only make the damage consistent.

### Measured clipping

`scripts/decode_clipping.py` converts a grid of 512 known colours (8 x 8 x 8 steps of
the RGB cube, as flat 32-pixel patches) to BT.709 YCbCr, stores it losslessly, and
decodes it four ways. Errors are in 8-bit code values, over the 512 patches
(OpenCV 4.14.0, ffmpeg 6.1.1):

| source | decoder | mean abs error R, G, B | largest R, G, B | patches off by more than 8 |
|---|---|---|---|---|
| untagged HD | this machine's OpenCV | 8.0, 9.3, 3.8 | 23, 36, 13 | 344 of 512 |
| untagged HD | ffmpeg reader (BT.709) | 0.9, 1.3, 0.9 | 3, 3, 3 | 0 |
| BT.709-tagged HD | this machine's OpenCV | 0.9, 1.3, 0.9 | 3, 3, 3 | 0 |
| BT.709-tagged HD | a BT.601-only decoder (simulated) | 8.0, 9.3, 3.8 | 23, 36, 13 | 344 |

The BT.601-only decoder is simulated by the same ffmpeg reader told to use BT.601;
on an untagged HD source it matches this machine's OpenCV exactly, which is why it is
a fair stand-in for the Mac. 128 of the 512 colours lie outside the RGB cube under
that reading (by up to 44 code values) and are clipped. The 0.9 to 1.3 floor is the
8-bit rounding of the test itself.

Real pictures have few saturated colours, so the cost is smaller than the grid
suggests. Comparing the BT.601 reading with the correct one over every frame of the
bundled samples (`verify_decode_frames.py --correct`): `english1.mp4` (SD, BT.601 by
convention) is identical; `chinese1.mp4` differs by at most 15 code values in a
pixel, 0.5 on average in its worst frame; `others.mp4` by at most 30, 4.3 on
average in its worst frame. The decision baselines and previews do not depend on it
(detection is not changed, see below).

### What the pipeline does with it

`scripts/colour_roundtrip.py` (flat patches and real content through the real
`STTNInpainter.inpaint` with a stub model; "dE" is the worst-patch colour difference,
about 2.3 is just visible). `--decode sim601` replaces the decoder by the BT.601-only
one, as the Mac's:

| class | OpenCV here (before) | BT.601-only decoder, as on the Mac (before) | after, `auto` |
|---|---|---|---|
| untagged HD | 4.2 mean / 17.6 worst | 4.2 / 17.6 | 1.05 / 1.96 |
| BT.709 HD, tagged | 1.05 / 1.96 | 4.2 / 17.6 | 1.05 / 1.96 |
| BT.709 HD, full range | 0.37 / 1.54 | 3.5 / 17.7 | 0.37 / 1.54 |
| BT.709 SD, tagged | 1.05 / 1.96 | 4.2 / 17.6 | 1.05 / 1.96 |
| BT.709 4:2:2 HD | 1.06 / 1.96 | 4.2 / 17.6 | 1.06 / 1.96 |
| BT.709 10-bit HD | 0.73 / 1.42 | 3.8 / 17.7 | 0.73 / 1.42 |
| BT.2020 matrix, HD | 1.10 / 2.14 | 3.4 / 13.8 | 1.10 / 2.14 |
| BT.2020 with 2020 primaries and transfer tags, HD | 25.1 / 91.2 | 3.4 / 13.8 | 1.10 / 2.14 |
| BT.601 HD and SD, untagged SD | 1.18 / 2.14 | 1.18 / 2.14 | 1.18 / 2.14 |

"After" is the same on both machines because it no longer depends on the decoder
(on this machine `auto` leaves OpenCV decoding wherever it is already right, see
below, and the numbers are the same). The last-but-one row was the class
colour.md called uncorrectable; the ffmpeg reader converts it with the matrix tag
alone and so does not suffer from OpenCV's primaries conversion.

## What it does

`videowipe/decode.py` provides `FfmpegFrameReader`, which stands in for the
`cv2.VideoCapture` the inpainting path reads from. ffmpeg decodes the source and
converts to `bgr24` with `setparams=colorspace=T:range=R,scale=in_color_matrix=M:in_range=R,
format=bgr24` (`T` is the tag that stands for `M`: smpte170m, bt709 or bt2020nc), and the
frames come down a pipe. The `setparams` relabel comes first so that the frames carry the
matrix and range the reader converts with: ffmpeg 6.1.1 takes the scaler's options over the
frames' own colour properties, and newer ffmpeg (9.x on the reference Mac) may follow the
frames, so a source whose tags disagree with `M` and `R` would otherwise be converted with
its own. With both in agreement it does not matter which wins; on 6.1.1 the bytes are
identical to the command without the relabel (every bundled sample, the generated cases
of `scripts/verify_decode_frames.py`, each of the six matrix and range pairs). `M` and `R`
are:

1. the source's tags (`color_space`, `color_range`, or a `yuvj` pixel format);
2. where there is no matrix tag, the convention players use: BT.709 for a picture
   wider than 1024 or taller than 576, BT.601 otherwise; limited range.

`tasks/base.py: read_frame_info` opens the reader (it decides as below) and
`STTNInpainter` opens its second reader (the one the output frames are composed
from) as the same kind, so the model and the compositor never see different
colours. `videowipe/colour.py` takes the matrix and range from the reader itself (not
from a probe), so the encode is exactly the inverse of the decode, and an untagged
source stays untagged. For untagged HD the only change to the encode command is
`-vf scale=out_color_matrix=bt709:out_range=tv`, which the player's own BT.709
assumption then undoes. For untagged SD there is no change: the command is upstream's.

### Frames are the same frames, in the same order

The inpainter identifies frames by their sequential-read ordinal and never seeks.
The reader yields one frame per decoded frame in decode order. It never seeks
(`set` is refused), and uses `-fps_mode passthrough` so ffmpeg does not duplicate or
drop frames to fit a frame rate (`-vsync passthrough` on an ffmpeg that has no
`-fps_mode`; the first that decodes known colours correctly is used).

`scripts/verify_decode_frames.py` reads a source with OpenCV and the reader in
lockstep, with the reader converting as this OpenCV measurably does for that source.
Then the frames must be identical: they are, bit for bit, for every 8-bit source
tried (largest pixel difference 0), so frame *i* of one is frame *i* of the other.
It also reports any frame for which a neighbour of the other reader matches better
than the frame itself (a dropped, repeated or shifted frame).

| source | frames, OpenCV / ffmpeg | largest pixel difference | misordered |
|---|---|---|---|
| `english1.mp4` (852x480) | 522 / 522 | 0 | 0 |
| `chinese1.mp4` (1920x1080, variable rate: 591 frames, 24.9 fps average) | 591 / 591 | 0 | 0 |
| `others.mp4` (1920x1080) | 516 / 516 | 0 | 0 |
| generated: B-frames, variable frame rate MP4 (irregular gaps and bursts) | 90 / 90 | 0 | 0 |
| generated: variable-rate WebM, VP9 and VP8 | 90 / 90 | 0 | 0 |
| generated: edit list cutting inside a GOP (frames hidden before the cut) | 65 / 65 (the container holds 90) | 0 | 0 |
| generated: edit list with an empty edit (start delayed 1.5 s); the same on a variable-rate source | 90 / 90; 60 / 60 | 0 | 0 |
| generated: HEVC in Matroska with B-frames; rotated by display matrix; odd size; 4:2:2; 4:4:4; full range; BT.709 tagged | equal | 0 | 0 |
| generated: 10-bit | 40 / 40 | up to 100 (see below) | 0 |

(`python scripts/verify_decode_frames.py --generate DIR`; the sources are
generated, the table is from one run. `tests/test_decode.py` repeats it on shorter
versions and also checks that the comparison would notice a dropped frame.)

What this does not prove: a source whose OpenCV and ffmpeg frame sequences differ in
a way the first few frames do not show. At the start of every run the reader is
compared with OpenCV on the first six frames (picture, size and order); a mismatch
falls back to OpenCV with one line. If ffmpeg ended early the run stops with the
existing "decoded N frames; expected M" error. `verify_decode_frames.py FILE` is the
whole-file check; run it on a source you doubt.

### Where ffmpeg's conversion differs from OpenCV's

For 8-bit 4:2:0, 4:2:2 and 4:4:4 sources with the same matrix and range the two are
bit-identical (both use libswscale's default bicubic conversion). For 10-bit sources
they are not: on a generated 10-bit test pattern (hard colour edges) OpenCV is off by
a mean of 1.5 and up to 91 code values from a 16-bit reference conversion, and the
ffmpeg reader by 0.26 and up to 3. Real pictures are far gentler, but because the
8-bit probe says nothing about OpenCV's high-bit-depth path, `auto` decodes sources
above 8 bits with ffmpeg.

## The switch

`VIDEOWIPE_DECODE` (environment variable):

| value | behaviour |
|-------|-----------|
| `auto` (default) | measure what this OpenCV does for a stream tagged like the source (the probe colour.md already runs). OpenCV keeps decoding when it uses the correct matrix and range, so nothing changes; otherwise, and for sources above 8 bits, the ffmpeg reader |
| `ffmpeg` | the ffmpeg reader wherever it can be used |
| `opencv` | upstream's decoder |

Any other value means `auto`. `VIDEOWIPE_COLOUR=off` keeps OpenCV (the encode would
be ffmpeg's BT.601 default whatever the decode did; `ffmpeg` with colour off says so).

### Why `auto` is the default

It is chosen from capability (C6), not from the machine: the same rule picks OpenCV
on a build that honours the tags and ffmpeg on one that does not. On this machine
that leaves tagged BT.709, BT.601 and SD sources, and every 8-bit source OpenCV
reads correctly, on OpenCV with **identical pixels** to before, and switches untagged
HD, BT.2020 and 10-bit sources. On the Mac (OpenCV that ignores BT.709 tags) it
switches every BT.709 and BT.2020 source. Always using ffmpeg (`ffmpeg`) would give
the same pixels where OpenCV is right (identical, above), so the only costs of the
default are the reader's speed (below) and a second process; `auto` avoids both where
they buy nothing.

### Falling back

One line each, then OpenCV decodes as before (`Warning: ...; decoding with OpenCV.`):

* the source cannot be probed; its pixel format is not YCbCr (RGB, palette); its matrix
  tag is not BT.601, BT.709 or BT.2020 (smpte240m, fcc, ycgco ...);
* HDR (PQ or HLG): `source_checks` already warns about it, so `auto` is silent and
  `ffmpeg` says so; [high-bit-depth.md](high-bit-depth.md) handles it;
* this ffmpeg does not decode known colours as asked (a clip of known YCbCr patches,
  tagged with the *wrong* matrix and range, must come out as the matrix and range given),
  or does not encode with the matching matrix (colour.md's check);
* the reader's first frames are not OpenCV's;
* `VIDEOWIPE_DECODE=ffmpeg` with `VIDEOWIPE_COLOUR=off`.

When the reader starts it says so once per source:
`Decoding with ffmpeg (BT.709, limited range); this OpenCV reads the source as BT.601, limited range.`

## What does not change

* **Detection and planning keep decoding with OpenCV** (`detect.py`, `planning.py`,
  `sampling.py`, the web previews). They find text, which does not depend on a few
  code values of saturation, and they identify frames by the same ordinals (checked
  above), so a mask found on OpenCV's frame *i* applies to the reader's frame *i*.
  Switching them would change the detection baselines (`make fact-baseline`,
  `make decision-baseline`), which were not re-run: the detector is untouched, so they
  are unchanged by construction. Changing the detector's input would need those run
  on the reference machine first.
* The model sees the reader's frames, so for untagged HD it now works on the true
  colours rather than BT.601-clipped ones.
* HDR sources, and sources whose matrix tag is not handled, decode as before.
* Hardware decoding and sources whose first video stream is not the one `-map 0:v:0`
  picks: not tested.

## Speed

Reading 150 frames of `chinese1.mp4` (1080p) sequentially, three runs, on a
4-core machine that was heavily loaded by other jobs (load average about 30), so the
ratio is indicative only:

| | OpenCV | ffmpeg reader |
|---|---|---|
| `read()` | 98 to 146 frames/s | 62 to 93 frames/s |
| `grab()` (skipping) | 137 to 322 frames/s | 85 to 119 frames/s |

The reader is about 0.6 times as fast, and skipping into a trial range converts every
skipped frame (OpenCV's `grab` does not). The model runs at a few frames per second,
so decoding is not what a run waits for. Each job runs two readers (model input and
compositing), as before.

## Mac checks

On the reference Mac:

    python scripts/decode_clipping.py --content input/detext_examples/others.mp4
    python scripts/colour_roundtrip.py --probe decode          # which matrix OpenCV uses there
    python scripts/colour_roundtrip.py --probe table --decode auto --classes hd709_tv,hd_untagged
    python scripts/verify_decode_frames.py --generate /tmp/decode-cases --correct

The first shows the clipping; `colour_roundtrip` with `auto` should switch every BT.709
class to the ffmpeg reader (look for `Decoding with ffmpeg`) and give the dE of the
"after" column. The last must say `OK` for every source (ffmpeg 9 and its `-fps_mode`
are the thing to confirm). The reader checks `-fps_mode passthrough` against known
colours before it is used, so an ffmpeg that refuses it falls back with one line
rather than decoding wrongly.

The clips that script (and the tests) generate carry their colour tags on the frames
(`setparams`, after any conversion) as well as on the output, because ffmpeg 9 converts
the pixels to colour options given on the output alone. `tests/test_decode.py` checks
that labelling does not change the pixels (the same clip made with and without the
tags decodes to the same YCbCr), which would fail on an ffmpeg that converts.

The `vs correct` column of `--correct` is the colour correction itself, not a measure of
how the clips were made: where OpenCV honours the source's matrix tag (this machine's
4.14.0 does) it is 0.00 for every 8-bit clip, and where OpenCV decodes every source as
BT.601 (the reference Mac's does) it is the BT.601 against BT.709 difference, several
code values, for exactly the clips tagged BT.709 (`bt709_tagged`, `yuv422p`, `yuv444p`,
`full_range`, `yuv420p10`). `yuv420p10` differs further by OpenCV's own 10-bit
conversion (1.60 here).
