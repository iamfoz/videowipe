# Seconds and frames on variable-frame-rate video (web page and server)

This builds on [source timing](source-timing.md): the cleaned output keeps each
frame's own source timestamp, and the page and server now agree with it.

## The problem

The web page and the server turn seconds into frame ordinals with
`frame = seconds * fps`, where `fps` is OpenCV's average rate. For a
constant-rate source that is exact. For a variable-rate source the frames are
not on that grid, so the trial range the person asks for, the recommended
trial, the times shown for each target and the positions used when reviewing a
result are off by up to a couple of frames (and by the length of any long gap).

Frame identity is the sequential-read ordinal everywhere (frame 0, 1, 2 ... in
decode order); nothing here seeks by `CAP_PROP_POS_FRAMES`.

## Measurements

`python scripts/vfr_timing_report.py` reads each frame's presentation timestamp
with ffprobe and compares it with the constant-rate arithmetic. On the bundled
samples:

| sample | frames | gaps | seconds to frame at each frame's own start | frame to seconds (`i / fps`) | 3 s trials that get a different range |
|--------|--------|------|-----------|-----------|----|
| `chinese1.mp4` (OpenCV 24.9157 fps) | 591 | 589 of 0.04 s, one of 0.12 s | 294 of 591 frames mis-numbered, by 0 to +1 | 0 to 2 frames (up to 80 ms) early | 516 of 591 (first frame 0 to +1, last frame 0 to +2) |
| `english1.mp4` | 522 | 521 of 0.04 s | none | exact | none |
| `others.mp4` | 516 | 515 of 0.04 s | none | exact | none |

`chinese1.mp4` has one 0.12 s gap, after its first frame, so the average rate
(24.9157) differs from the true 25 and the error is largest early in the video
and shrinks to zero at the end.

## What changes

`videowipe/server/frame_map.py` holds a `FrameMap`: the start time of every
frame, plus the end of the last (`frame_count + 1` values, in seconds on the
cleaned output's timeline, rounded to a microsecond). It is built from the
source's real timestamps, using `probe_source` from `videowipe/timing.py`, once
per source file and cached (a failed probe is cached too).

* `server/app.py`: `_trial_range`, `_recommended_trial`, the trial request, the
  trial worker's recheck and the preview response use the map when there is one.
  The trial's `start_seconds` and `duration_seconds` are the real times of its
  first and last frames. `seconds_to_frames` in `server/review.py` takes the map
  as an optional argument.
* The preview response carries the map to the page as `source.frame_times`.
  The page converts with `f2s` (frame to seconds) and `s2f` (seconds to
  frame) instead of dividing and multiplying by `fps`; they use `frame_times`
  when it is present. It is used for the trial window, the target and review
  times, the segment and protection editors, the overlay and mask frame, the
  review windows and the arrow-key frame step.
* The overlay's mask request (`refreshMask`) still computes its frame as
  `position() * source.fps`. So that its text stays as upstream's, the page
  sets a flag around the call (`nominalPositions`) and `position()` then returns
  the seconds at which that product is the right frame; without `frame_times`
  nothing differs.
* Between two frame times the ordinal is interpolated, so the rest of the
  arithmetic (snap to a whole frame within 1e-9, then floor for a start and
  ceil for an end) is upstream's.

A **constant-rate source has no map**: nothing is sent, and the page and server
use upstream's `seconds * fps` code exactly. The tests embed upstream's
original functions as oracles and check that the numbers are identical.

## The switch

It follows `VIDEOWIPE_SOURCE_TIMING`, as in [source timing](source-timing.md):

| value | behaviour |
|-------|-----------|
| `auto` (default) | variable-rate source: use the real frame times; constant-rate: upstream's arithmetic |
| `off` | always upstream's `seconds * fps` arithmetic, as before |
| `on` | build the map for a constant-rate source too (the numbers still agree with upstream's) |

## When it falls back

To upstream's arithmetic, with one line on the console, when ffprobe fails or
its timestamps are unusable, when the number of packets differs from the
decoded frame count, when the first frame precedes the container's start, or
when two frame times collapse to the same microsecond.

## Limits

* The times are those of the container's timeline (`SourceClock.seconds`, the
  same one the cleaned MP4 uses). They are the browser's `currentTime` as long
  as the browser plays the file from the container's start, which holds for the
  bundled samples (start time 0); a browser that normalises a late first frame
  differently would show a constant offset.
* `frame_times` is sent with every preview response, about 6 to 7 bytes a frame
  as JSON (3.8 KB for `chinese1.mp4`; an hour at 30 fps would be about 0.7 MB),
  and only for a variable-rate source.
* The first request for a source needs one ffprobe pass over its packets
  (0.10 s for `chinese1.mp4`, which has 591 frames; it grows with the packet
  count).
* Detection, sampling and the cleaning itself are unchanged; the review
  windows' padding (`round(fps)` frames) is still a frame count.
* The CLI has no seconds-based option, so nothing there changes.
