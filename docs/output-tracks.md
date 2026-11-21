# Audio, metadata, chapters and subtitles in the cleaned MP4

This is the second half of the output-fidelity work; the first half (the video's
frame timestamps) is in [source-timing.md](source-timing.md). The two are
independent and have separate switches.

## What was lost (measured on ffmpeg 6.1, constant-rate sources, MP4 output)

The cleaned MP4 is made by one ffmpeg command whose input 0 is the frame pipe and
input 1 is the original file. Upstream maps `-map 0:v -map 1:a?` and encodes
`-c:a aac`.

| | before |
|---|---|
| Global metadata (title, artist, comment, creation time) | lost (ffmpeg copies it from input 0, the pipe) |
| Chapters | **kept**, with MP4 and MKV sources (ffmpeg takes them from the first input that has any) |
| Track language | kept |
| Track title | lost (MP4 has no field for it other than the track's handler name) |
| Audio | every track re-encoded to AAC at ffmpeg's default (stereo 128 kb/s, 5.1 about 275 kb/s: the native encoder tops out there) |
| 5.1(side) layout (AC-3, E-AC-3, FLAC 5.1 ...) | the AAC track has no channel layout |
| Soft subtitles (text or bitmap) | dropped |
| Default and forced flags | audio default kept; subtitles dropped with the tracks |

The same holds with the Matroska frame pipe of source-timing.md. Chapters of an MKV
source whose AAC audio starts 21 ms early come out shifted by 21 ms, because ffmpeg
moves the whole source timeline to start at zero (so the chapters follow the audio).

## What it does now (full runs)

`videowipe/tracks.py` rewrites the finished command; `videowipe/fidelity.py` applies
it before the timestamp rewrite of `videowipe/timing.py` and starts ffmpeg.

* **Audio**, every track, in the source's order. Stream-copied when the codec is
  AAC, MP3, AC-3, E-AC-3 or ALAC. Anything else (Opus, FLAC, PCM, Vorbis, DTS ...)
  is encoded to AAC.
  * Why that list: those five are registered MP4 audio types (ALAC is Apple's own)
    and the players people actually use for MP4 (QuickTime, Safari, phones, TVs)
    decode them. Opus and FLAC are left out: ffmpeg 6.1 writes both into MP4 without
    complaint, but support for them in MP4 is patchy outside ffmpeg-based players.
    That is a judgement; it was not tested on QuickTime. PCM is excluded because
    it is huge and not an ISO MP4 type.
  * AAC bitrate: 64 kb/s per channel is the base (128k stereo, 384k for 5.1). A lossy
    source is not squeezed below its own rate: the target is the source's bit rate
    (when ffprobe knows it) kept between the base and 128 kb/s per channel. A
    lossless source (FLAC, PCM ...) gets 96 kb/s per channel (192k stereo). The AAC
    encoder in ffmpeg 6.1 produces about 275 kb/s at most for 5.1, so the rule only
    changes 5.1 and 7.1 tracks when they are below that.
  * Channel layout is kept; `5.1(side)` and `7.1(wide)` (which AAC cannot write) are
    converted to `5.1` and `7.1` with `aformat`.
* **Global metadata**: `-map_metadata 1` (the source). Track titles are written as
  the MP4 track's handler name, the only per-track name the muxer stores. Languages
  were already kept.
* **Chapters**: nothing to do, they already survive.
* **Subtitles**: text tracks (SubRip, ASS/SSA, WebVTT, `mov_text`, plain text) are
  converted to `mov_text`, with language, title, and default and forced flags.
  Styling of ASS is lost (MP4 text is plain). Bitmap tracks (DVD, PGS, DVB ...) are
  skipped, with a line saying which.
* Before the encode starts, the new options are tried on the first half second of the
  source with this machine's ffmpeg. If that fails (or the source cannot be probed)
  the job runs as upstream does and prints one warning.

## Not changed

* **Trial previews** (`trial_range` set) keep upstream's AAC re-encode with no
  metadata, chapters or subtitles. A preview is a throw-away cut, and its `-ss`/`-t`
  would also cut the subtitles and metadata for little use.
* The video stream's own tags (language, title) are not carried over.
* The optional HEVC/MKV copy (`feat/h265-output`) keeps its own audio handling
  (`export.py`): per-track Opus copy or encode, track titles and languages (MKV can hold
  them) and chapters (ffmpeg's default copies them to every output). The only thing
  added to it is the source's global metadata (one `-map_metadata 1` ahead of its
  mapping); it still has no subtitle tracks.

## The switch

`VIDEOWIPE_OUTPUT_TRACKS`: `auto` (default) or `off`. `off` gives upstream's audio,
metadata and subtitle handling exactly. Any other value means `auto`.

## What is identical to upstream by default

| source and settings | video bytes sent to ffmpeg | ffmpeg arguments | output file |
|---|---|---|---|
| `VIDEOWIPE_OUTPUT_TRACKS=off`, constant-rate source | identical | identical | byte-identical (measured on a 6 s clip with audio) |
| trial preview | identical | identical | identical |
| default, constant rate, no audio and no global metadata (the 6 s `-an` reference clip) | identical | identical except `-map_metadata 1` | byte-identical (measured; nothing to map) |
| default, constant rate, with audio, metadata or subtitles | identical | audio, metadata and subtitle arguments differ | video identical in content; audio (copied or re-encoded), metadata and subtitle tracks differ |
| default, variable rate | frames in a Matroska stream (source-timing.md) | differs | video timing differs |

"Identical" for the video content means the same frames go into the same encoder
settings; whether an MP4 is byte-identical also depends on the audio, metadata and
subtitle tracks written beside it.

## Limits

* Copied audio keeps the source's timing exactly (including encoder delay and a video
  that starts after its audio), but a source whose audio and video start at different
  times is still shifted as a whole to start at zero, as ffmpeg always does.
* The source's end trim is not carried when audio is copied: ffmpeg drops it, so the
  copied track can run up to one audio frame longer at the very end (measured: an
  AC-3 track of 1.994 s became 2.010 s; 32 ms is the most for AC-3). The start, the
  sync and every packet's bytes are unchanged. AAC in MP4 was not affected in the
  tests.
* Dispositions: the MP4 muxer marks the first audio track as default whatever the
  source said.
* Tracks that this ffmpeg cannot copy or convert make the preflight fail and the job
  fall back to upstream's handling for everything, not track by track.

## Sources the output cannot represent

Frames are decoded to 8-bit BGR and encoded as 8-bit `yuv420p`, with no tone
mapping. A source with more than 8 bits per sample, or with an HDR transfer
(PQ or HLG), is still cleaned, and one line says what is lost, as soon as the
source is read (`src/videowipe/source_checks.py`):

- `Warning: the source is 10-bit; the cleaned video is made from 8-bit frames, so gradients may show banding.`
- `Warning: the source is HDR (PQ); the cleaned video is 8-bit SDR with no tone mapping, so its colours will be wrong.`

It is printed once per source in a process and changes nothing in the output.
A source that ffprobe cannot read is not reported. Keeping the depth or the
HDR signal needs a different frame path and is not done.
