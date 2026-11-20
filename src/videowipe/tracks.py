"""Keep the source's audio tracks, metadata and soft subtitles in the cleaned MP4.

Upstream's encode command maps ``-map 0:v -map 1:a?`` and re-encodes every audio
track with ``-c:a aac`` at ffmpeg's default bitrate, maps no subtitle tracks and
no global metadata. Measured on ffmpeg 6.1 (see ``docs/output-tracks.md``):

* Global metadata (title, artist, comment, creation time) is lost: ffmpeg copies it
  from the *first input*, which is the frame pipe.
* Chapters survive (ffmpeg takes them from the first input that has any), and each
  track's language survives; a track's title does not (MP4 has no field for it).
* Every audio track is re-encoded, so lossless or surround audio is degraded.
  A 5.1(side) layout (AC-3, E-AC-3) comes out as an AAC track with no channel layout.
* Soft subtitles (text or bitmap) are dropped.

For a full run this module rewrites the finished command so that

* audio tracks that MP4 carries reliably (:data:`COPY_CODECS`) are stream-copied,
  and the others are encoded to AAC at a bitrate derived from the source
  (:func:`aac_bitrate`), keeping the channel layout and the track order;
* the source's global metadata is mapped, and a track's title is kept as its
  handler name;
* text subtitle tracks (:data:`TEXT_SUBTITLES`) are converted to ``mov_text``;
  bitmap tracks are skipped with a one-line notice.

Trial previews are left exactly as upstream builds them. ``VIDEOWIPE_OUTPUT_TRACKS``
is ``auto`` (default) or ``off`` (upstream's audio and metadata handling). Before
the encode starts, the new options are tried on the first half second of the
source; if that fails, or the source cannot be probed, the job runs as upstream
does and says so in one line.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field

from videowipe.timing import _run

ENV_VAR = "VIDEOWIPE_OUTPUT_TRACKS"
MODES = ("auto", "off")

# Audio that an MP4 carries reliably and that common players (QuickTime, Safari,
# phones, TVs) decode: AAC, MP3, AC-3 and E-AC-3 are registered MP4 audio types
# and ALAC is Apple's own lossless codec for MP4/MOV. Opus, FLAC and PCM are left
# out: ffmpeg writes them into MP4, but support in MP4 is patchy outside ffmpeg-
# based players (decided by judgement, not tested on QuickTime), so they are
# encoded to AAC as upstream does.
COPY_CODECS = ("aac", "mp3", "ac3", "eac3", "alac")
LOSSLESS_CODECS = ("flac", "alac", "truehd", "mlp", "wavpack", "ape", "tta")
# Subtitle formats ffmpeg converts to MP4's mov_text. Everything else (DVD, PGS,
# DVB and other bitmaps, or formats ffmpeg may not convert) is skipped.
TEXT_SUBTITLES = ("subrip", "ass", "ssa", "webvtt", "mov_text", "text")

# AAC cannot write these layouts as they are; the equivalent it can write.
AAC_LAYOUTS = {
    "5.1(side)": "5.1", "5.0(side)": "5.0", "7.1(wide)": "7.1", "7.1(wide-side)": "7.1",
}
_AAC_BY_CHANNELS = {1: "mono", 2: "stereo", 6: "5.1", 8: "7.1"}


def tracks_mode() -> str:
    """``auto`` (default) or ``off``, from ``VIDEOWIPE_OUTPUT_TRACKS``."""
    value = os.environ.get(ENV_VAR, "auto").strip().lower()
    return value if value in MODES else "auto"


@dataclass(frozen=True)
class AudioTrack:
    index: int  # among the source's audio streams
    codec: str
    channels: int
    layout: str | None
    bit_rate: int | None
    title: str | None

    @property
    def copied(self) -> bool:
        return self.codec in COPY_CODECS


@dataclass(frozen=True)
class SubtitleTrack:
    index: int  # among the source's subtitle streams
    codec: str
    title: str | None

    @property
    def convertible(self) -> bool:
        return self.codec in TEXT_SUBTITLES


@dataclass(frozen=True)
class SourceTracks:
    audio: tuple = ()
    subtitles: tuple = ()


def _positive_int(value) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def parse_tracks(text: str) -> SourceTracks | None:
    """The audio and subtitle streams of ``ffprobe -show_streams -of json``."""
    try:
        streams = json.loads(text)["streams"]
        audio, subtitles = [], []
        for stream in streams:
            kind, tags = stream.get("codec_type"), stream.get("tags") or {}
            title = tags.get("title") or None
            if kind == "audio":
                audio.append(AudioTrack(
                    len(audio), str(stream["codec_name"]), int(stream.get("channels") or 2),
                    stream.get("channel_layout") or None, _positive_int(stream.get("bit_rate")),
                    title,
                ))
            elif kind == "subtitle":
                subtitles.append(SubtitleTrack(len(subtitles), str(stream["codec_name"]), title))
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    return SourceTracks(tuple(audio), tuple(subtitles))


def probe_tracks(path: str) -> SourceTracks | None:
    """Probe *path*; ``None`` on any failure."""
    try:
        result = _run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", path],
                      timeout=120, text=True)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not isinstance(result.stdout, str):
        return None
    return parse_tracks(result.stdout)


def aac_bitrate(track: AudioTrack) -> int:
    """Bits per second for re-encoding *track* to AAC.

    The base is 64 kb/s per channel (128k stereo, as ffmpeg's default; 384k for 5.1).
    A lossy source is not squeezed below its own bitrate: the target is the source
    bitrate when known, kept between the base and 128 kb/s per channel (a stereo
    MP3 at 192k stays 192k; AC-3 5.1 at 448k stays 448k; Opus at 96k gets 128k).
    A lossless source (FLAC, PCM, ...) gets 96 kb/s per channel (192k stereo).
    """
    base = 64_000 * track.channels
    if track.codec in LOSSLESS_CODECS or track.codec.startswith("pcm_"):
        return 96_000 * track.channels
    if track.bit_rate is None:
        return base
    return round(min(max(track.bit_rate, base), 128_000 * track.channels), -3)


def aac_layout(track: AudioTrack) -> str | None:
    """The AAC-writable layout to convert to, or ``None`` when ffmpeg already
    keeps the source's (stereo, mono, 5.1, ...)."""
    if track.layout in AAC_LAYOUTS:
        return AAC_LAYOUTS[track.layout]
    if track.layout is None or track.layout == "unknown":
        return _AAC_BY_CHANNELS.get(track.channels) if track.channels > 2 else None
    return None


def _kbps(bits: int) -> str:
    return f"{bits // 1000}k"


@dataclass
class OutputTracks:
    """What one job does about audio, metadata and subtitles. ``source`` is set
    when the new options apply; otherwise the command is upstream's."""

    source: SourceTracks | None = None
    video_path: str = ""
    skipped: list = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.source is not None

    @property
    def subtitles(self) -> list:
        return [s for s in self.source.subtitles if s.convertible]

    def options(self, input_index: int):
        """``(audio maps, subtitle maps, audio codecs, subtitle codecs, metadata)``
        for source input *input_index*; audio keeps its order."""
        audio_maps, sub_maps, audio_codecs, sub_codecs = [], [], [], []
        meta = ["-map_metadata", str(input_index)]
        for track in self.source.audio:
            n = track.index
            audio_maps += ["-map", f"{input_index}:a:{n}"]
            if track.copied:
                audio_codecs += [f"-c:a:{n}", "copy"]
            else:
                audio_codecs += [f"-c:a:{n}", "aac", f"-b:a:{n}", _kbps(aac_bitrate(track))]
                layout = aac_layout(track)
                if layout:
                    audio_codecs += [f"-filter:a:{n}", f"aformat=channel_layouts={layout}"]
            if track.title:
                meta += [f"-metadata:s:a:{n}", f"handler_name={track.title}"]
        for position, track in enumerate(self.subtitles):
            sub_maps += ["-map", f"{input_index}:s:{track.index}"]
            if track.title:
                meta += [f"-metadata:s:s:{position}", f"handler_name={track.title}"]
        if sub_maps:
            sub_codecs = ["-c:s", "mov_text"]
        return audio_maps, sub_maps, audio_codecs, sub_codecs, meta

    def works_here(self) -> bool:
        """Whether this ffmpeg accepts the options on the start of the source."""
        audio_maps, sub_maps, audio_codecs, sub_codecs, meta = self.options(0)
        with tempfile.TemporaryDirectory() as tmp:
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-nostdin", "-t", "0.5",
                   "-i", self.video_path, *audio_maps, *sub_maps, *audio_codecs, *sub_codecs, *meta,
                   os.path.join(tmp, "probe.mp4")]
            try:
                return _run(cmd, timeout=120).returncode == 0
            except (OSError, ValueError, subprocess.SubprocessError):
                return False

    def summary(self) -> str:
        audio = self.source.audio
        copied = sum(t.copied for t in audio)
        parts = []
        if audio:
            parts.append(f"{len(audio)} audio track(s) ({copied} copied, {len(audio) - copied} "
                         "encoded to AAC)")
        if self.subtitles:
            parts.append(f"{len(self.subtitles)} subtitle track(s)")
        parts.append("source metadata")
        return "Keeping " + ", ".join(parts) + "."

    def rewrite(self, cmd):
        """*cmd* with audio, metadata and subtitle options replaced, or ``None``
        when it is not the command upstream builds for a full run."""
        cmd = list(cmd)
        try:
            maps_at = [i for i in range(len(cmd) - 1) if cmd[i:i + 2] == ["-map", "1:a?"]]
            codec_at = [i for i in range(len(cmd) - 1) if cmd[i:i + 2] == ["-c:a", "aac"]]
            movflags = cmd.index("-movflags")
            if len(maps_at) != 1 or len(codec_at) != 1 or "-ss" in cmd or "-t" in cmd:
                return None
            if cmd[movflags + 1] != "+faststart" or not maps_at[0] < codec_at[0] < movflags:
                return None
        except ValueError:
            return None
        audio_maps, sub_maps, audio_codecs, sub_codecs, meta = self.options(1)
        # A later output (the optional HEVC copy) has its own mapping; it only gets
        # the source's global metadata, which ffmpeg would take from the frame pipe.
        later = movflags + 3
        if later < len(cmd):
            cmd[later:later] = ["-map_metadata", "1"]
        # Without audio the upstream options stay (they do nothing).
        cmd[codec_at[0]:codec_at[0] + 2] = (audio_codecs or ["-c:a", "aac"]) + sub_codecs + meta
        cmd[maps_at[0]:maps_at[0] + 2] = (audio_maps or ["-map", "1:a?"]) + sub_maps
        return cmd


def plan_output_tracks(job) -> OutputTracks:
    """Decide how this job treats the source's tracks (and say so)."""
    plan = OutputTracks(video_path=job.video_path or "")
    if tracks_mode() == "off" or job.trial_range is not None:
        return plan
    if not job.video_path or not os.path.isfile(job.video_path):
        return plan
    source = probe_tracks(job.video_path)
    if source is None:
        print("Warning: could not read the source's audio and subtitle tracks; "
              "audio is re-encoded and metadata and subtitles are not kept.")
        return plan
    candidate = OutputTracks(source, job.video_path)
    if not candidate.works_here():
        print("Warning: this ffmpeg cannot keep the source's tracks as planned; "
              "audio is re-encoded and metadata and subtitles are not kept.")
        return plan
    for track in source.subtitles:
        if not track.convertible:
            print(f"Skipping subtitle track {track.index + 1} ({track.codec}): "
                  "bitmap subtitles cannot be converted to MP4 text.")
    print(candidate.summary())
    return candidate
