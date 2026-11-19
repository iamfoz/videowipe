"""Memory facts for choosing between a memory-hungry and a re-decoding path.

Nothing here is tuned to a machine: the figures come from what the running
system reports, with a conservative default when it reports nothing, and every
budget has an environment override (``0`` switches the memory-hungry path off).
No third-party dependency (no ``psutil``).
"""
from __future__ import annotations

import os
import subprocess

#: Used when the system reports no memory figure at all.
UNKNOWN_BUDGET_BYTES = 256 * 1024 * 1024
#: Never plan to hold more than this for a speed-up, whatever the machine has.
MAX_BUDGET_BYTES = 1024 * 1024 * 1024
#: Fraction of the memory that is available that a speed-up may claim.
BUDGET_FRACTION = 1 / 16

BAND_CROP_BUDGET_ENV = "VIDEOWIPE_BAND_CROP_BUDGET_MB"


def total_memory_bytes() -> int | None:
    """Physical memory of the machine, or ``None`` when it cannot be read."""
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        if total > 0:
            return int(total)
    except (AttributeError, ValueError, OSError):
        pass
    try:  # macOS reports hw.memsize even where sysconf lacks SC_PHYS_PAGES
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
            timeout=5, check=True,
        ).stdout.strip()
        if out.isdigit() and int(out) > 0:
            return int(out)
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _read_int(path: str) -> int | None:
    try:
        with open(path, encoding="ascii") as handle:
            text = handle.read().strip()
    except (OSError, UnicodeDecodeError):
        return None
    return int(text) if text.isdigit() else None


def available_memory_bytes() -> int | None:
    """Memory a new allocation can expect, or ``None`` when unknown.

    Linux reports ``MemAvailable`` and may confine a container with a cgroup
    limit; the smaller of the two wins. Elsewhere (macOS) only the physical
    total is known, which is the best figure available without ``psutil``.
    """
    figures: list[int] = []
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    figures.append(int(line.split()[1]) * 1024)
                    break
    except (OSError, ValueError, IndexError, UnicodeDecodeError):
        pass
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        limit = _read_int(path)
        # An unlimited cgroup reports "max" (v2) or a huge sentinel (v1).
        if limit is not None and limit < (1 << 60):
            figures.append(limit)
    if figures:
        return min(figures)
    return total_memory_bytes()


def band_crop_budget_bytes() -> int:
    """How many bytes of subtitle-band crops one detection may keep in memory.

    ``VIDEOWIPE_BAND_CROP_BUDGET_MB`` overrides the figure (``0`` keeps none, so
    detection re-reads the file as it always did). Otherwise a sixteenth of the
    memory that is available, capped at 1 GiB, or 256 MiB when it is unknown.
    """
    override = os.environ.get(BAND_CROP_BUDGET_ENV, "").strip()
    if override:
        try:
            return max(0, int(float(override) * 1024 * 1024))
        except ValueError:
            pass
    available = available_memory_bytes()
    if not available:
        return UNKNOWN_BUDGET_BYTES
    return int(min(MAX_BUDGET_BYTES, available * BUDGET_FRACTION))
