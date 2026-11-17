# H.265 output

VideoWipe can write a 10-bit H.265 (HEVC) file with Opus audio instead of the
default H.264/AAC MP4.

## Choosing it

- **Web UI:** pick **Output format (输出格式) → H.265 (MKV)** next to "Clean the whole
  video now" before running the full clean. The choice is remembered in the
  browser. **Download** then serves the `.mkv`.
- **CLI:** `videowipe clean input.webm --codec h265` (also on `detext`).
- **Default for both:** set `VIDEOWIPE_OUTPUT_CODEC=h265`.

The cleaned frames are encoded straight to HEVC. In the web UI an H.264 copy
is still written alongside it, because trial wipes and the in-app review play
in the browser and most browsers cannot play HEVC in MKV. The CLI removes that
copy, leaving only the `.mkv`.

## Encoder settings

Taken from the HandBrake presets marked `[VTB]`, chosen automatically:

| Source height | Video | Audio per track |
|---|---|---|
| ≤ 576 (DVD) | quality 50 | Opus 80 kbps stereo |
| ≤ 720 | quality 45 | Opus 80 kbps stereo |
| larger | quality 45 | Opus 80 kbps stereo, or 256 kbps 5.1 for tracks with more than two channels |

- Video is 10-bit HEVC (Main 10), limited colour range, tagged `hvc1`.
- **VideoToolbox** (`hevc_videotoolbox`) is used when ffmpeg has it and it can
  encode 10-bit HEVC on this Mac (constant quality needs Apple Silicon).
  Otherwise **libx265** is used with CRF 22 (CRF 20 at 576p and below) as an
  approximate equivalent.
- Audio tracks that are already Opus are copied unchanged.
- Resolution is kept: the presets' autocrop and maximum size are not applied.

Only the built-in STTN inpainter supports this; external commands and
ProPainter are unchanged.
