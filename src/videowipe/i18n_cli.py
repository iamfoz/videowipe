"""``videowipe`` command entry point that translates CLI messages.

Wraps :func:`videowipe.cli.main` so upstream's ``cli.py`` stays untouched:
each complete line written to stderr is passed through
:func:`videowipe.i18n.translate_message` (whole messages only, so file names
are never rewritten). The language follows ``VIDEOWIPE_LANG`` / the locale.
"""
from __future__ import annotations

import sys

from videowipe.i18n import translate_message


class _TranslatingStream:
    def __init__(self, stream):
        self._stream = stream
        self._pending = ""

    def write(self, text):
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._stream.write(translate_message(line) + "\n")
        return len(text)

    def flush(self):
        if self._pending:
            self._stream.write(translate_message(self._pending))
            self._pending = ""
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def main():
    from videowipe import cli

    original = sys.stderr
    sys.stderr = _TranslatingStream(original)
    try:
        return cli.main()
    finally:
        sys.stderr.flush()
        sys.stderr = original


if __name__ == "__main__":
    main()
