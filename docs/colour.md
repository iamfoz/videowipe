# Colour tags and range in the cleaned video

This is the third part of the output-fidelity work, after the frame timestamps
([source-timing.md](source-timing.md)) and the audio, metadata and subtitle tracks
([output-tracks.md](output-tracks.md)). It has its own switch and falls back in the
same way.

## What upstream does

OpenCV decodes the source to 8-bit BGR. The frames go to ffmpeg as `bgr24`, which
encodes `libx264 -pix_fmt yuv420p`. Nothing sets colour primaries, transfer, matrix
or range on the way in or out, so the output file carries no colour tags and:

* OpenCV converts YCbCr to BGR with a matrix and range of **its own choosing**.
  That depends on the OpenCV build: the one measured here (4.14.0, which bundles
  an ffmpeg 8 libswscale) reads the source's matrix and range tags, an older build
  always uses BT.601 limited range.
* ffmpeg converts BGR to YCbCr with swscale's default, **BT.601 limited range**.
* Players read an untagged stream as BT.709 when it is wider than 1024 or taller
  than 576, BT.601 below, limited range.

Whether the picture comes out right depends on whether those three agree. They do
for an untagged or BT.601 SD source. They do not for a tagged BT.709 HD source on an
OpenCV that honours the tag (BGR is then the true colour, but ffmpeg encodes it as
BT.601 and the player reads it as BT.709), nor for BT.601 HD, BT.709 SD, full-range
or BT.2020 sources.

## Measurements

`scripts/colour_roundtrip.py` builds sources of known colour (SMPTE 75 % bars,
eleven saturated flat patches, and a clip of `input/detext_examples/english1.mp4`
as real content), converts and tags them correctly, runs them through the real
`STTNInpainter.inpaint` (stub model whose only effect is in a small mask; real
OpenCV decode, `bgr24` pipe and ffmpeg encode), then decodes source and output to
RGB as a correct player would (their tags; conventions where there are none) and
compares them. Machine: OpenCV 4.14.0 (bundled avcodec 62.28 / swscale 9.5), system
ffmpeg 6.1.1, 12 to 50 frames per class; sources are lossless H.264 (`-crf 0`).

`dE` is the CIE76 colour difference between the mean source and output colour of
each of the 18 flat patches (about 2.3 is a just-noticeable difference side by
side, 1 is not visible); "worst" is the worst patch. `err` is the mean absolute
difference per channel in 8-bit code values over the patches; PSNR is over the real
content (the part of the picture that is not flat patches).

| source class | before: dE mean / worst, err R,G,B, PSNR | after: dE mean / worst, err R,G,B, PSNR | output tags after |
|---|---|---|---|
| BT.709 limited HD, tagged | 6.6 / 19.1 (green), 2.1 16.0 1.3, 38.1 | 1.1 / 1.9, 1.6 1.0 1.5, 38.9 | bt709, tv |
| BT.601 limited SD 625, tagged | 1.2 / 2.1, 1.7 0.7 1.7, 38.4 | 1.2 / 2.1, 1.7 0.7 1.7, 38.4 | bt470bg, tv |
| BT.601 limited SD 525, tagged | 1.2 / 2.1, 1.7 0.7 1.7, 38.3 | 1.2 / 2.1, 1.7 0.7 1.7, 38.3 | smpte170m, tv |
| BT.709 full range HD | 6.7 / 19.2 (green), 2.4 16.2 1.2, 39.3 | 0.4 / 1.5, 0.3 0.6 0.3, 42.9 | bt709, pc (yuvj420p) |
| BT.601 full range SD | 0.9 / 2.1, 1.1 0.6 1.3, 40.3 | 0.2 / 1.0, 0.2 0.3 0.3, 42.1 | bt470bg, pc |
| BT.601 tagged, HD frame size | 6.5 / 19.1 (green), 2.3 15.9 1.2, 37.8 | 1.2 / 2.1, 1.7 0.7 1.7, 38.7 | bt470bg, tv |
| BT.709 tagged, SD frame size | 1.2 / 2.1, 1.7 0.7 1.7, 38.7 | 1.1 / 2.0, 1.6 1.0 1.5, 38.6 | bt709, tv |
| BT.2020 matrix tag only, HD | 6.6 / 19.6 (green), 2.1 15.9 1.3, 38.1 | 1.1 / 2.1, 1.4 1.1 1.7, 39.4 | bt2020nc, tv |
| BT.709 4:2:2 / 4:4:4 / 10-bit, HD | 6.6 / 19.1; 6.4 / 19.1; 6.5 / 19.2 | 1.1 / 2.0; 0.8 / 1.9; 0.7 / 1.4 | bt709, tv |
| untagged HD (BT.709 by convention) | 4.2 / 17.6 (green), 2.1 11.6 1.6, 38.5 | unchanged | none |
| untagged SD (BT.601 by convention) | 1.2 / 2.1, 1.7 0.7 1.7, 38.4 | unchanged | none |
| BT.2020 with 2020 primaries and transfer, HD | 25.1 / 91.2 (blue), 9.0 37.2 21.9, 29.1 | unchanged (falls back, see below) | none |

The "before" row for an unaffected class and the "after" row of a corrected one
are the same, about 1.1 dE and 1.5 code values: that is the noise floor of the
pipeline itself (an 8-bit RGB step between decode and encode, 4:2:0 chroma, x264 at
crf 18), including a constant bias of 3 code values on R and B (not G) on white and
grey. The change neither adds to it nor removes it. Where the floor is lower
(full-range rows) the encode has finer steps.

What the three probes in the script say, on this machine:

* **decode** (BGR from known YCbCr patches): with no tag or a BT.601 tag
  (smpte170m, bt470bg) OpenCV uses BT.601 whatever the size; BT.709 and BT.2020
  matrix tags are honoured; so is a full-range tag (`pc`). With BT.2020 primaries
  *and* a BT.2020 transfer tag it matches none of BT.601, BT.709 or BT.2020 (mean
  error 15 code values; it also converts the primaries), which this change cannot
  undo.
* **encode** (YCbCr from known BGR patches, the pipeline's own options): BT.601
  limited range, whatever the size, and no tags written.
* **pass-through** (YCbCr patches through the whole pipeline): BT.601 sources move
  by at most 2 code values in Y, Cb and Cr; BT.709 sources by up to 11, 7 and 4 (and
  full-range ones are re-scaled to limited range).

Which sources are wrong today (on this OpenCV): every tagged source that is not
BT.601-limited-and-SD, that is BT.709 HD and full range (saturated colours move by
up to 41 code values: pure green's G channel comes out 40 lower), BT.601 HD (read
as BT.709), 4:2:2, 4:4:4 and 10-bit BT.709, and BT.2020. Those errors are plainly
visible on saturated colours. On the real-content part of the picture the shift
costs little PSNR (38.1 dB against 38.9 after, for BT.709 HD) because that content
has few saturated colours, so the flat patches' dE is the better indicator. Which
are fine: untagged and tagged SD BT.601, and BT.709 SD (whose output is read as
BT.601, which is what the BGR frames hold).

Untagged HD is wrong in a different way and is **not** changed: OpenCV decodes it
as BT.601, but players (and the source's own colours) are BT.709, so highly
saturated colours are clipped on decode (up to 39 code values of G lost on the saturated patches).
Mid-saturation real content is unaffected (38.5 dB, the same as the floor). The
output reads as BT.709 exactly as the source does, so tagging it would add nothing;
fixing the clipping would need a different decoder path.

Other OpenCV builds: an OpenCV that ignores the tags and always converts with
BT.601 limited range was not available to measure here. The change measures what the
installed build does (below) rather than assuming, so on such a build it would only
add tags and the pixels would be today's (this case is covered by a unit test that
supplies that decoder result, and by the BT.601 SD rows above, where the measured
conversion is the default). The reference machine (a Mac with a newer ffmpeg and a
possibly different OpenCV, perhaps hardware decoding) may give different decode
rows: run the script there.

## What changes

`videowipe/colour.py`, applied last by `videowipe/fidelity.py` (after audio,
metadata and subtitle tracks, and after the timestamp rewrite). For a source that
has any colour tags and is not HDR:

1. The source's tags are read with ffprobe (`color_space`, `color_primaries`,
   `color_transfer`, `color_range`; a `yuvj` pixel format means full range).
2. The matrix and range OpenCV converts with for a stream carrying those tags is
   **measured on this machine**: a small H.264 clip of known YCbCr patches with the
   same tags and size (at least 256 by 256) is decoded with `cv2.VideoCapture` and
   compared with the six readings (BT.601, BT.709, BT.2020; limited, full). If none fits (error above 1.6
   code values, or the best is not 0.5 better than the next), nothing is changed.
3. ffmpeg is told to convert BGR to YCbCr the same way, and to tag the output with
   the source's tags: `-vf scale=out_color_matrix=<m>:out_range=<r> -colorspace ..
   -color_primaries .. -color_trc .. -color_range ..` before the first output file.
   The scale option is left out when the measured conversion is ffmpeg's own default
   (BT.601 limited), so the pixels are then exactly today's and only the tags are
   added. The decode and encode matrices are the same, so the source's YCbCr values
   survive for untouched pixels and the tags describe them truthfully.
4. Before the encode starts, those exact options are run on known colours with this
   ffmpeg: the output must carry the requested tags and its YCbCr values must be
   those of the chosen matrix and range.

When all of that holds, one line is printed:
`Keeping the source's colour tags (BT.709, limited range; the source is bt709).`

A full-range source gives a full-range output (`yuvj420p`, tagged `pc`), as the
source was. A player that ignores the range flag would show it as limited range; no
player was tested.

The optional second output of `feat/h265-output` (the HEVC MKV) is limited range by
design. It is converted once, with the same matrix and `out_range=tv`, and given the
matrix, primaries and transfer tags only where BGR is already the picture's true
colours (the decoder read the source's tags as a player does). Otherwise it is left
as before.

## The switch

`VIDEOWIPE_COLOUR` (environment variable):

| value | behaviour |
|-------|-----------|
| `auto` (default) | as above |
| `off` | upstream's command exactly (no tags, ffmpeg's default conversion) |

Any other value means `auto`.

## When it does nothing

* `VIDEOWIPE_COLOUR=off`; no source file; a source with no colour tags at all
  (untagged HD and SD output is exactly what it was, byte for byte); HDR (PQ or HLG
  transfer: `videowipe.source_checks` already warns that these cannot be right).
* A warning and today's command, one line each:
  * the source cannot be probed ("could not read the source's colour tags");
  * the matrix tag is not BT.601, BT.709 or BT.2020 (smpte240m, fcc, ycgco, gbr ...);
  * the decoder's conversion matches none of the six readings ("could not tell how
    this OpenCV converts the source's colours"; this is what BT.2020 with 2020
    primaries and transfer tags does with OpenCV 4.14);
  * the decoder read the source with a different range from its tag ("read the
    source's colour range differently from its tag"; not seen with OpenCV 4.14);
  * this ffmpeg does not apply the options to the check ("did not apply the source's
    colour tags with a matching conversion");
  * the ffmpeg command is not the shape expected ("could not keep the source's
    colour tags with this ffmpeg command").

## Limits

* 8-bit output only; a source with more bits (10-bit BT.709 was measured and
  behaves like 8-bit) is decoded to 8-bit as before. HDR is not handled.
* The decoder probe uses an H.264 clip. A source in another codec or with its tags
  only in the container is assumed to be decoded the same way.
* Chroma stays 4:2:0 (upstream's `-pix_fmt yuv420p`): 4:2:2 and 4:4:4 sources lose
  chroma resolution as before (PSNR in the table).
* BT.2020 with 2020 primaries and transfer is not corrected (see above), and untagged
  sources are left as they are.
* The model sees the BGR frames OpenCV produced; for a source that OpenCV decodes
  "wrongly" (untagged HD) it works on those colours, as before.

## Reproducing the measurements

    python scripts/colour_roundtrip.py --colour off     # today's behaviour
    python scripts/colour_roundtrip.py --colour auto    # with this change
    python scripts/colour_roundtrip.py --probe decode   # one section; --classes a,b --frames 25

The first lines say which OpenCV and ffmpeg were used. `--json file` keeps the
numbers; `--keep dir` keeps the generated clips.
