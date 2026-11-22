"""One place where the cleaned video's ffmpeg command is adapted to its source.

``STTNInpainter.inpaint`` builds upstream's command and hands it to
:meth:`EncodePlan.popen`, which applies, in order,

1. :mod:`videowipe.tracks` (audio, metadata, subtitles; full runs only), then
2. :mod:`videowipe.timing` (the source's frame timestamps), then
3. :mod:`videowipe.colour` (the source's colour tags and a matching conversion), then
4. :mod:`videowipe.highbit` (16-bit frames in, for a source above 8 bits),

and starts ffmpeg. Each part decides for itself whether it applies and falls back
to upstream's behaviour, with a one-line warning, when it cannot; each has its own
environment switch (``VIDEOWIPE_OUTPUT_TRACKS``, ``VIDEOWIPE_SOURCE_TIMING``,
``VIDEOWIPE_COLOUR``, ``VIDEOWIPE_HIGH_BIT_DEPTH``).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from videowipe.colour import OutputColour, plan_output_colour
from videowipe.highbit import HighBitPlan, plan_high_bit_depth
from videowipe.timing import SourceTiming, plan_source_timing
from videowipe.tracks import OutputTracks, plan_output_tracks


@dataclass
class EncodePlan:
    timing: SourceTiming
    tracks: OutputTracks
    colour: OutputColour | None = None
    high_bit: HighBitPlan | None = None

    def command(self, cmd):
        if self.tracks.active:
            rewritten = self.tracks.rewrite(cmd)
            if rewritten is None:
                print("Warning: could not keep the source's audio, metadata and subtitle "
                      "tracks with this ffmpeg command; audio is re-encoded and metadata "
                      "and subtitles are not kept.")
            else:
                cmd = rewritten
        cmd = self.timing.command(cmd)
        cmd = cmd if self.colour is None else self.colour.command(cmd)
        return cmd if self.high_bit is None else self.high_bit.command(cmd)

    def popen(self, cmd, stderr_file):
        pipe = subprocess.Popen(self.command(cmd), stdin=subprocess.PIPE, stderr=stderr_file)
        return self.timing.attach(pipe)


def plan_encode(job) -> EncodePlan:
    """Decide, once per job, what to keep from the source."""
    # the bit-depth plan first: the colour plan asks whether the 10-bit output exists
    high_bit = plan_high_bit_depth(job)
    return EncodePlan(plan_source_timing(job), plan_output_tracks(job), plan_output_colour(job),
                      high_bit)
