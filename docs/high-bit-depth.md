# Keeping 10-bit and HDR sources at their bit depth

This follows [decode.md](decode.md) (the frames are decoded with the right matrix by an
ffmpeg reader) and builds on it: it is stacked on `fix/decode-colour`. Until now a
source above 8 bits was only warned about (`source_checks.py`): every frame was
reduced to 8-bit BGR, the model worked on it, and the output was 8-bit `yuv420p`.

## What is lost today

`scripts/high_bit_depth_roundtrip.py` builds lossless HEVC (Main 10) clips of a smooth
picture, tags them as 10-bit SDR (BT.709), HDR10 (BT.2020, PQ, with a mastering display
and a content light level) and HLG (BT.2020, ARIB STD-B67), cleans them with the real
`STTNInpainter.inpaint` and a stub model (a mask box is blanked, everything else goes
through decode, pipe and encode), and compares the first frame of every output file with
the source outside the mask, in 10-bit code values.

The picture has a luma ramp over the whole range (876 codes across 640 pixels), a
*shallow* ramp of 64 codes (16 eight-bit levels) where banding is plainest, and Cb and
Cr ramps. "Levels" is the number of distinct values left along the middle row of a ramp;
"run" the longest stretch of equal values in pixels (the source's is 1 on the full ramp
and 10 on the shallow one). Machine: ffmpeg 6.1.1 with libx265, OpenCV 4.14.0.

**Today** (origin/main; the MP4 is the only output):

| source | ramp: rms / max error, levels of 640, longest run | shallow: rms / max, levels of 65, run | tags in the MP4 |
|---|---|---|---|
| 10-bit SDR, BT.709 | 7.0 / 14, 217, 8 px | 7.2 / 18, 18, 71 px | pix_fmt yuv420p; BT.709 tags kept (colour.md) |
| HDR10, PQ | 161 / 362, 204, 268 px | 80 / 121, 37, 31 px | none; mastering display and content light level gone |
| HLG | 87 / 215, 216, 161 px | 43 / 51, 19, 71 px | none |

The 10-bit row says the 8-bit pipeline keeps about a third of the levels and turns the
shallow ramp into 18 bands 71 pixels wide. The two HDR rows are far worse than
quantisation: OpenCV (4.14.0) also converts the primaries and the transfer of a BT.2020
picture that is tagged so, and the output carries no tags, so a player shows the
PQ or HLG signal as SDR (the warning `source_checks` already prints). The HDR metadata
is not carried at all.

## What it does

`VIDEOWIPE_HIGH_BIT_DEPTH=auto` (default; `off` is upstream's behaviour exactly), for a
source above 8 bits that the ffmpeg reader decodes (all of them in `auto`, HDR included
now: BT.2020 matrix, no tone mapping):

1. The model works on 8 bits as before: the first reader (model input) is the 8-bit
   ffmpeg reader of decode.md. It was trained on 8-bit pictures and the region it
   paints is small.
2. The compositor's reader (the second one, which the output frames are composed from)
   decodes to **16-bit BGR** (`bgr48le`) with the source's matrix and range, and hands
   the compositor the frame's 8-bit rounding, as before, with the 16-bit frame attached
   (`highbit.HighFrame`).
3. When the compositor has painted the 8-bit frame, `highbit.merge` carries **only the
   changed pixels** back: each is the original's 16-bit value plus its 8-bit change times
   256 (swscale's 16-bit scale, see below). Every pixel the model did not change is the
   source's, bit for bit. The painted region is 8-bit precise.
4. The 16-bit frames go to ffmpeg as `bgr48le`. The **MP4 stays the 8-bit H.264 review
   copy** the web UI plays, from the same frames. A **second output**, the HEVC MKV that
   `export.py` already writes for `VIDEOWIPE_OUTPUT_CODEC=h265`, is written for these
   sources whichever codec is chosen: HEVC Main 10 (`yuv420p10le`, or `p010le` on
   VideoToolbox), converted with the source's matrix to limited range, with the
   source's `color_primaries`, `color_trc` and `colorspace` (on the frames with
   `setparams` as colour.md explains, and on the stream), and for HDR10 the **mastering
   display and content light level**, written by x265 from explicit parameters (the
   raw frames carry no side data): `hdr10=1:hdr10-opt=1:repeat-headers=1:master-display=...:max-cll=...`.
   The audio and metadata follow `export.py`'s presets (Opus).

Planned once per job (`fidelity.plan_encode`, the fourth part after tracks, timing and
colour); the command is rewritten to read `bgr48le` and a line says so:
`Keeping the source's bit depth (10-bit HDR10 (PQ), bt2020nc; HEVC Main 10 copy by libx265 from 16-bit frames; the MP4 review copy stays 8-bit).`

### Gate: 8-bit sources are byte-identical

A source of 8 bits or less never takes this path: `plan_high_bit_depth` returns an
inactive plan without a line, no command is rewritten, the reader yields plain frames,
`merge` returns its argument, and `export.hevc_output_args` is what it was.
`tests/test_high_bit_depth.py::test_an_8_bit_source_is_byte_identical_with_the_switch_on_or_off`
runs a bundled sample with the switch off and on, with H.264 and with H.265 selected,
and compares the MP4 bytes and the MKV's decoded frames (Matroska writes a random
segment UID, so its bytes never repeat).

### When it does nothing (one line each, then 8-bit as before)

* `VIDEOWIPE_HIGH_BIT_DEPTH=off`;
* dual (side by side) output, because the frame is twice as tall; a trial run (a
  preview: no second output); the source not read by the ffmpeg reader
  (`VIDEOWIPE_DECODE=opencv`, or the reader fell back);
* no 10-bit HEVC encoder (libx265 or hevc_videotoolbox) in this ffmpeg;
* this ffmpeg does not encode known 16-bit colours as 10-bit with the source's tags and
  matrix. Before the job starts, the exact output options are run on seven known
  colours: the stream must be 10-bit, limited range and carry the tags asked for, and its
  YCbCr must be the 10-bit values of the matrix within 6 of 1023 (so an ffmpeg that
  converts or tags differently, as 9.0.2 may, falls back rather than writes a wrong file);
* the command is not the shape expected ("could not keep the source's bit depth with
  this ffmpeg command").

`VIDEOWIPE_HIGH_BIT_DEPTH_ENCODER` (`auto`, `libx265`, `hevc_videotoolbox`) overrides the
HEVC encoder. `auto` is `export.video_encoder()`'s choice, except that an **HDR10 source
with static metadata uses libx265 when it is there**: VideoToolbox cannot be given the
metadata from raw frames here, so with it the file has the other tags only (a line says
so).

## After

Same sources, same harness, `VIDEOWIPE_HIGH_BIT_DEPTH=auto` (this machine, libx265):

| source | output | ramp: rms / max, levels of 640, run | shallow: rms / max, levels of 65, run | tags |
|---|---|---|---|---|
| 10-bit SDR | MKV, 10-bit | 0.86 / 9, 601, 3 px | 0.77 / 5, 64, 16 px | yuv420p10le, tv, bt709 all three |
| 10-bit SDR | MP4 review copy | 1.7 / 7, 215, 6 px | 1.5 / 10, 17, 48 px | yuv420p, as before |
| HDR10 | MKV, 10-bit | 0.86 / 7, 593, 5 px | 0.78 / 4, 63, 17 px | yuv420p10le, tv, bt2020nc, bt2020, smpte2084, **mastering display and content light level present** |
| HDR10 | MP4 review copy | 1.7 / 7, 215, 6 px | 1.5 / 10, 17, 48 px | 8-bit, no tags (as before) |
| HLG | MKV, 10-bit | 0.86 / 9, 601, 3 px | 0.77 / 5, 64, 16 px | yuv420p10le, tv, bt2020nc, bt2020, arib-std-b67 |

The MKV's errors are the encode's own noise (x265 at the export preset's CRF, about one
10-bit code) and 4:2:0; the chroma ramps (not in the table) keep 4:2:0's error of 2.6
codes rms, the same as 8-bit gets. Where the 8-bit MP4 had 18 bands on the shallow ramp
the MKV has 64 levels. A lossless trial of the 16-bit decode and the 10-bit encode with
no model in between (swscale both ways) gave an error of exactly 0 on both luma
ramps, so the conversions cancel; the MKV's 0.8 rms is the codec's. The 8-bit MP4 of an
HDR source is better than today's (rms 1.7 against 161) only because the matrix is now
consistent: it is still the PQ signal viewed as SDR, untagged.

Checked by `tests/test_high_bit_depth.py` with these thresholds, and run for SDR, PQ and
HLG: the painted region (the stub paints black) is in the MKV, and outside it the MKV is
the source within 1.5 codes rms.

## Details worth knowing

* **swscale's 16-bit RGB is the 8-bit value times 256**: white is 65283, not 65535 (it
  has a 0.4 % gain under the ideal; `zscale`, which is exact, was measured and rejected:
  its chroma resampling is twice as noisy and not every ffmpeg has it). Its decode and
  encode cancel exactly (above), so untouched pixels are unaffected, and a painted pixel
  moves by 256 sixteen-bit steps per 8-bit step, which is what keeps painted white level
  with the surrounding white.
* Pixels the model changed have **8-bit precision** (a region that is plain text over a
  gradient is painted from the 8-bit model output), each plus the original's sub-8-bit
  residue. A feathered edge is smooth.
* Chroma stays 4:2:0, as upstream: 4:2:2 and 4:4:4 sources lose chroma resolution as
  before; 12-bit and 16-bit sources are written as 10-bit.
* Speed and memory: a 16-bit 1080p frame is 12 MB (6 MB in 8-bit); the pipe carries
  twice the bytes; `merge` finds the changed rows and columns with `cv2.absdiff` and
  costs about 10 ms per 1080p frame (the machine was heavily loaded). The HEVC encode by
  libx265 is the slow part; with VideoToolbox it is not.
* Not handled: Dolby Vision (the RPU/enhancement layers are dropped; the HDR10 or HLG
  base layer is what is kept), HDR10+ dynamic metadata, a tone-mapped SDR review copy
  (the MP4 of an HDR source is the PQ or HLG signal viewed as SDR, as before), 10-bit
  H.264 (HEVC only: browsers and VideoToolbox do not decode High 10).

## What was verified, and what needs an HDR display

Run here (ffmpeg 6.1.1, libx265, OpenCV 4.14.0): the tables above; the stream's pixel
format, range, matrix, primaries and transfer; the mastering display and content light
level read back from the MKV with ffprobe; the capability check against a wrong matrix;
the byte-identical 8-bit gate; the command rewrite and its failure lines.

**Not verified, needs the owner**:

* that the picture *looks* right on an HDR display: no HDR display was available. The
  numbers say the YCbCr values and the tags are the source's within the encode's noise;
  whether the player applies them (HDR badge, brightness) is for a screen to say. Open
  the MKV in IINA, mpv or VLC on the Mac (QuickTime does not play Matroska), beside the
  source, and look at a PQ or HLG scene and a smooth sky or gradient;
* `hevc_videotoolbox` (not built here): the options are the export's own plus the colour
  tags, but whether this encoder writes the tags and keeps a Main 10 profile with the
  `setparams` labels was not run. Run the harness on the Mac with
  `VIDEOWIPE_HIGH_BIT_DEPTH_ENCODER=hevc_videotoolbox` and with `libx265`
  (if installed) and compare `tags` in the output;
* ffmpeg 9.0.2: the capability check is there for it (it encodes known colours with the
  exact options and compares tags and values before the job); if it falls back, the line
  says which check failed;
* a real HDR clip (an iPhone HLG or HDR10 file): mastering metadata and Dolby Vision
  side data differ from the generated clips.

On the Mac:

    python scripts/high_bit_depth_roundtrip.py --modes off,auto --json /tmp/hbd.json
    VIDEOWIPE_HIGH_BIT_DEPTH_ENCODER=hevc_videotoolbox python scripts/high_bit_depth_roundtrip.py --kinds pq --modes auto
    ffprobe -v error -show_frames -read_intervals %+#1 -select_streams v:0 -of json OUTPUT.mkv
