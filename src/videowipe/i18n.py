"""Runtime translation (presentation-time i18n) for videowipe.

Upstream source files keep their zh-CN text; translation is applied at
presentation time over those source strings. A catalogue is a JSON file in
``src/videowipe/locales/<lang>.json`` whose *key* is the exact zh-CN source
string and whose *value* is the translation (an empty value means
"untranslated"). ``translate()`` below is the shared algorithm; the
identical implementation in ``web/i18n.js`` must pass the same vectors in
``tests/fixtures/i18n_vectors.json``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path

LOG = logging.getLogger("videowipe.i18n")

SOURCE_LANGUAGE = "zh-CN"
DEFAULT_LANGUAGE = "en"
_LOCALES_DIR = Path(__file__).resolve().parent / "locales"
# code -> loaded message map (or None if the catalogue is missing/invalid).
_CATALOGUE_CACHE: dict[str, dict | None] = {}

# U+3000-U+303F (CJK punctuation) | U+3400-U+9FFF (CJK) | U+FF00-U+FFEF (fullwidth)
_CJK = re.compile("[\u3000-\u303f\u3400-\u9fff\uff00-\uffef]")
_PLACEHOLDER = re.compile(r"\{(\d+)\}")
_HEAD_TAG = re.compile(r"<head[^>]*>", re.IGNORECASE)
# Bounds recursive group translation against a pathological catalogue.
_MAX_DEPTH = 8


def _is_pattern_key(key: str) -> bool:
    """True if *key* is a pattern key, i.e. contains a ``{n}`` placeholder."""
    return _PLACEHOLDER.search(key) is not None


def _build_pattern(key: str):
    """Build an anchored regex for a pattern key: escape literal parts, each
    ``{n}`` becomes a lazy capture group (ordinal 1..K in order of appearance).

    Returns ``(regex, number_to_group)`` where ``number_to_group`` maps a
    placeholder number (from ``{n}``) to its capture-group ordinal, so a value
    may reference or reorder placeholders by their ``{n}`` number."""
    pieces = re.split(r"(\{\d+\})", key)
    pattern = []
    number_to_group = {}
    ordinal = 0
    for piece in pieces:
        if piece.startswith("{"):
            ordinal += 1
            number_to_group[int(piece[1:-1])] = ordinal
            pattern.append(r"(.+?)")
        else:
            pattern.append(re.escape(piece))
    return re.compile("^" + "".join(pattern) + "$"), number_to_group


def translate_core(text: str, messages: dict[str, str], _depth: int = 0) -> str:
    """Translate the trimmed *text* against *messages*. Shared algorithm —
    see ``web/i18n.js`` and ``tests/fixtures/i18n_vectors.json``.

    ``_depth`` bounds recursive group translation so a pathological catalogue
    (e.g. a bare ``{0}`` pattern key) cannot recurse forever."""
    if _CJK.search(text) is None:
        return text

    # 3. Exact match.
    exact = messages.get(text)
    if exact:
        return exact

    # 4. Pattern match, in order of descending literal (non-placeholder) length.
    if _depth < _MAX_DEPTH:
        pattern_keys = []
        for key, value in messages.items():
            if value and _is_pattern_key(key):
                literal_len = sum(
                    len(part)
                    for part in re.split(r"(\{\d+\})", key)
                    if not part.startswith("{")
                )
                pattern_keys.append((literal_len, key, value))
        pattern_keys.sort(key=lambda item: item[0], reverse=True)
        for _, key, value in pattern_keys:
            regex, number_to_group = _build_pattern(key)
            match = regex.match(text)
            if match:
                return _apply_value(value, match, number_to_group, messages, _depth)

    # 5. Fragment fallback: non-pattern keys with non-empty values, longest
    #    first, scanning left to right without re-replacing produced output.
    fragments = [
        (len(key), key, value)
        for key, value in messages.items()
        if value and not _is_pattern_key(key) and key in text
    ]
    if not fragments:
        return text

    result = []
    start = 0
    while start < len(text):
        best = None
        for length, key, value in fragments:
            at = text.find(key, start)
            if at == -1:
                continue
            if best is None or at < best[0] or (at == best[0] and length > best[1]):
                best = (at, length, key, value)
        if best is None:
            break
        at, _, key, value = best
        result.append(translate_core(text[start:at], messages, _depth))
        result.append(value)
        start = at + len(key)
    result.append(translate_core(text[start:], messages, _depth))
    return "".join(result)


def _apply_value(
    value: str, match, number_to_group: dict, messages: dict, _depth: int
) -> str:
    """Substitute each ``{n}`` in *value* with the recursively-translated
    captured group, preserving any literal text in the value (reordering is
    allowed because values reference groups by their ``{n}`` number)."""
    pieces = re.split(r"(\{\d+\})", value)
    result = []
    for piece in pieces:
        if piece.startswith("{"):
            ordinal = number_to_group.get(int(piece[1:-1]))
            if ordinal is not None:
                result.append(
                    translate_core(match.group(ordinal), messages, _depth + 1)
                )
            else:
                result.append(piece)
        else:
            result.append(piece)
    return "".join(result)


def translate_with(text: str, messages: dict[str, str]) -> str:
    """Translate *text* against *messages* (no language resolution).

    Preserves leading/trailing whitespace; a string without CJK, or with an
    empty core, is returned unchanged. This is the function the shared vectors
    exercise, mirroring ``translateWith`` in ``web/i18n.js``."""
    if not text:
        return text
    if _CJK.search(text) is None:
        return text
    match = re.match(r"^(\s*)(.*?)(\s*)$", text, re.DOTALL)
    if not match:
        return text
    leading, core, trailing = match.group(1), match.group(2), match.group(3)
    if not core:
        return text
    return leading + translate_core(core, messages) + trailing


def translate(text: str, lang: str | None = None) -> str:
    """Translate *text* into *lang* (default: the resolved language).

    Source language (``zh-CN``) and any unavailable/empty catalogue are the
    identity."""
    if not text:
        return text
    code = lang if lang is not None else resolve_language()
    if code == SOURCE_LANGUAGE:
        return text
    catalogue = load_catalogue(code)
    if not catalogue:
        return text
    return translate_with(text, catalogue)


def available_locales() -> list[str]:
    """Sorted codes of catalogue files present on disk that load cleanly.

    A broken/invalid catalogue is treated as absent (see ``load_catalogue``),
    so it does not appear here or in the injected payload."""
    if not _LOCALES_DIR.is_dir():
        return []
    stems = (path.stem for path in _LOCALES_DIR.glob("*.json") if path.stem != "ignore")
    return sorted(code for code in stems if load_catalogue(code))


def load_catalogue(code: str) -> dict | None:
    """Load and cache one catalogue; return None if missing or invalid."""
    if code in _CATALOGUE_CACHE:
        return _CATALOGUE_CACHE[code]
    catalogue = None
    path = _LOCALES_DIR / f"{code}.json"
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("messages"), dict):
                catalogue = {
                    k: v
                    for k, v in loaded["messages"].items()
                    if isinstance(k, str) and isinstance(v, str) and v
                }
        except (OSError, ValueError) as error:
            LOG.warning("ignoring invalid i18n catalogue %s: %s", path.name, error)
            catalogue = None
    _CATALOGUE_CACHE[code] = catalogue
    return catalogue


def all_catalogues() -> dict[str, dict]:
    """{code: {"name":..., "messages": {non-empty only}}} for every catalogue."""
    result = {}
    for code in available_locales():
        catalogue = load_catalogue(code)
        if not catalogue:
            continue
        meta = catalogue_meta(code)
        result[code] = {
            "name": meta.get("name") or code,
            "messages": {k: v for k, v in catalogue.items() if v},
        }
    return result


def catalogue_meta(code: str) -> dict:
    """The ``_meta`` block of a catalogue ({} if unavailable)."""
    path = _LOCALES_DIR / f"{code}.json"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    meta = loaded.get("_meta") if isinstance(loaded, dict) else None
    return meta if isinstance(meta, dict) else {}


def _match_lang(raw: str, available: list[str]) -> str | None:
    """Prefix-match *raw* (e.g. "en-GB") against *available* codes; any
    ``zh`` prefix maps to the source language."""
    if not raw:
        return None
    lower = raw.lower()
    if lower.startswith("zh"):
        return SOURCE_LANGUAGE
    for code in available:
        if lower.startswith(code.lower()):
            return code
    return None


def resolve_language(env: Mapping[str, str] | None = None) -> str:
    """Resolve the CLI/Python language: VIDEOWIPE_LANG, then
    LC_ALL/LC_MESSAGES/LANG, then the default."""
    env = env if env is not None else os.environ
    available = available_locales()
    for name in ("VIDEOWIPE_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        raw = env.get(name, "")
        if raw:
            code = _match_lang(raw, available)
            if code:
                return code
    return DEFAULT_LANGUAGE


def render_index(html: str) -> str:
    """Inject the i18n runtime into *html*.

    Immediately after the opening ``<head>`` tag an inline ``<script>`` is
    inserted that sets ``window.__VIDEOWIPE_I18N__`` and then runs the
    contents of ``web/i18n.js``. Never fails: on any problem the original
    HTML is returned unchanged."""
    try:
        payload = {
            "source": SOURCE_LANGUAGE,
            "default": DEFAULT_LANGUAGE,
            "locales": all_catalogues(),
        }
        payload_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        payload_text = payload_text.replace("</", "<\\/")
        script_src = _read_i18n_js()
        injection = f"<script>window.__VIDEOWIPE_I18N__={payload_text};</script>\n<script>{script_src}</script>"
        match = _HEAD_TAG.search(html)
        if match is None:
            return html
        insert_at = match.end()
        return html[:insert_at] + "\n" + injection + html[insert_at:]
    except Exception as error:  # noqa: BLE001 - "never fails" is the contract
        LOG.warning("i18n injection failed; serving unmodified HTML: %s", error)
        return html


_I18N_JS_PATH = Path(__file__).resolve().parent / "web" / "i18n.js"


def _read_i18n_js() -> str:
    return _I18N_JS_PATH.read_text(encoding="utf-8")
