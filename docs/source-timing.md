# Source timing for variable-frame-rate video

## The problem

STTN sends the cleaned frames to ffmpeg as raw `bgr24` with a single constant
rate (`-r <fps>`, where `fps` is OpenCV's average rate) and takes the audio
from the original file. For a constant-rate source that is exact. A
variable-rate source (screen recordings, phone video, many downloads) has
frames at irregular times, so every output frame is placed on an average-rate
grid while the audio keeps its real timing: the picture drifts away from the
sound, by up to a few frames and more across a long gap.

## What changes

`videowipe/timing.py` looks at the source before ffmpeg starts and chooses one
of two behaviours.

**Constant-rate source (the common case): the video path does not change.** The
ffmpeg arguments that concern the video and the bytes written to its standard input
are exactly what they were before this module existed. With
`VIDEOWIPE_OUTPUT_TRACKS=off` (see [output-tracks.md](output-tracks.md)) the whole
command is upstream's and the output is byte-identical; by default the audio,
metadata and subtitle handling described there also applies.

**Variable-rate source: its own timestamps are kept.** The frames are written as
a minimal Matroska stream (one cluster per frame, each stamped with the source
frame's presentation time and duration) and ffmpeg reads that instead of raw
video. The encoder keeps its input time base and the MP4 track timescale is the
source's, so each output frame has the same presentation time as the source
frame it came from. Output packet timestamps equal the source's (relative to the
source container's start, as the audio is). A trial cuts the audio at the real
time of its first frame rather than at `first / fps`, and its video timestamps
start at that frame. The stacked original-and-cleaned output (`--dual`, and all
trials) is timed the same way, as is the optional HEVC copy.

A source's start offset is respected: if the video begins 0.5 s after the audio
in the source, it does in the output.

## How a source is classified

ffprobe lists the video stream's packet timestamps (container index or packet
headers only; nothing is decoded). The source is *constant* if, for its
`r_frame_rate` or else its `avg_frame_rate`, every frame lies within one
time-base tick of `first + i / rate`. One tick is the resolution the container
can express, so a Matroska or WebM file whose timestamps were rounded to
milliseconds (30000/1001 fps shows gaps of 33 and 34 ms) is still constant.
Anything else is variable.

Listing the packets took 0.26 to 0.39 s for 70,000 frames (MP4, Matroska,
MPEG-TS; payloads of 19 to 37 MB), and 0.64 s including the Python side. For
containers without an index the cost is that of reading the file.

## The switch

`VIDEOWIPE_SOURCE_TIMING` (environment variable):

| value | behaviour |
|-------|-----------|
| `auto` (default) | constant source: unchanged; variable source: keep its timestamps when possible |
| `off` | always the constant-rate behaviour |
| `on` | keep the source's timestamps whenever they are usable, even for a constant source (useful for testing) |

Any other value means `auto`.

## When it falls back

Timestamps are matched to frames by sequential ordinal only, as everywhere else
in the pipeline, so they are used only when the number of packets equals the
job's frame count. Otherwise the job runs exactly as before and prints one line
saying so:

- ffprobe is missing or fails, or a timestamp is missing, `N/A` or duplicated:
  "could not read the source's frame timestamps".
- A variable source whose packet count differs from the frame count (for
  example Matroska, whose frame count OpenCV only estimates): "cannot be matched
  to the decoded frames".
- The first frame precedes the container's start time (a stream-copied cut whose
  edit list hides leading frames; in the file measured OpenCV decoded 130 of
  150 packets, and the job fails regardless, as it does without this module).
- This ffmpeg cannot do it. The check is a capability test, not a version test:
  the first time it is needed the module runs ffmpeg on a small off-grid Matroska
  stream and confirms that the output timestamps come out exact, trying
  `-fps_mode` and then `-vsync` for the "do not retime" option, and requiring
  an encoder time base taken from the input: `-enc_time_base demux`, or `-1` on
  builds older than that spelling (ffmpeg 9 refuses `-1` and has no `-vsync`).
  Without it ffmpeg rounds each frame to a grid of 1 / frame rate, which hides
  the problem on sources that sit on such a grid.

When timestamps are kept, one line says so ("Keeping the source's frame
timestamps ...").

## Limits

- Only the written video is retimed. Detection, planning, masks and the plan
  format are untouched, and anything that converts between seconds and frame
  numbers with a single constant rate (the web UI's time controls, trial ranges
  chosen by seconds, review windows) is still approximate on a variable-rate
  source, as before.
- Matroska output (the HEVC copy) holds millisecond timestamps, so off-grid
  frames there are rounded to the nearest millisecond.
- The last frame of a file lasts as long as the frame before it (a one-frame
  source lasts one nominal frame), which is what ffmpeg itself assumes.
- The ffmpeg command is rewritten after upstream builds it; if its shape is not
  the one this module expects, the job falls back to the old behaviour with a
  warning rather than guessing.
