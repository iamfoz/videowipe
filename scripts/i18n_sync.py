#!/usr/bin/env python3
"""Catalogue tooling for the VideoWipe i18n overlay.

Upstream source files keep their Simplified Chinese (zh-CN) strings; this
tool extracts those strings and maintains per-language catalogue files in
src/videowipe/locales/<lang>.json that overlay translations at presentation
time. An empty value or a missing key means "untranslated" and the original
Chinese is shown, so new upstream strings never break anything.

Commands (run from the repository root, or pass --root):

    python scripts/i18n_sync.py extract [--json]
    python scripts/i18n_sync.py report
    python scripts/i18n_sync.py update
    python scripts/i18n_sync.py check
    python scripts/i18n_sync.py new-locale <code> <name>
    python scripts/i18n_sync.py translate --locale <code> [--model M] [--batch N] [--dry-run]

Standard library only.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import re
import sys
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

SOURCE_LANGUAGE = "zh-CN"
LOCALES_RELPATH = Path("src") / "videowipe" / "locales"
DEFAULT_MODEL = "claude-sonnet"
API_TIMEOUT = 180
GLOSSARY_SAMPLE_SIZE = 12

# CJK ranges from the shared spec: U+3000-U+303F, U+3400-U+9FFF, U+FF00-U+FFEF
CJK_RE = re.compile("[　-〿㐀-鿿＀-￯]")
PLACEHOLDER_RE = re.compile(r"\{(\d+)\}")

# Attribute values that carry user-visible text in the web UI.
_HTML_TEXT_ATTRS = {
    "title",
    "placeholder",
    "aria-label",
    "alt",
    "value",
    "content",
    "data-screen-label",
}


# ── Extraction ───────────────────────────────────────────────────────────────


def _has_cjk(text: str) -> bool:
    return CJK_RE.search(text) is not None


def _placeholders(text: str) -> set:
    return set(PLACEHOLDER_RE.findall(text))


# --- JavaScript literal scanner ----------------------------------------------

_JS_SIMPLE_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
    "v": "\v",
    "0": "\0",
    "'": "'",
    '"': '"',
    "`": "`",
    "\\": "\\",
}


def _unescape_js(raw: str) -> str:
    """Decode JS escape sequences in a raw literal body (without quotes)."""
    out = []
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= n:
            out.append(ch)
            break
        nxt = raw[i + 1]
        if nxt == "\n":  # line continuation: dropped
            i += 2
            continue
        if nxt == "x":
            if re.fullmatch(r"[0-9a-fA-F]{2}", raw[i + 2 : i + 4]):
                out.append(chr(int(raw[i + 2 : i + 4], 16)))
                i += 4
                continue
            i += 2
            continue
        if nxt == "u":
            if raw[i + 2 : i + 3] == "{":
                end = raw.find("}", i + 3)
                if end != -1 and re.fullmatch(r"[0-9a-fA-F]+", raw[i + 3 : end]):
                    try:
                        out.append(chr(int(raw[i + 3 : end], 16)))
                        i = end + 1
                        continue
                    except (ValueError, OverflowError):
                        pass
            if re.fullmatch(r"[0-9a-fA-F]{4}", raw[i + 2 : i + 6]):
                out.append(chr(int(raw[i + 2 : i + 6], 16)))
                i += 6
                continue
            i += 2
            continue
        out.append(_JS_SIMPLE_ESCAPES.get(nxt, nxt))
        i += 2
    return "".join(out)


def _skip_interpolation(code: str, start: int) -> int:
    """Return the index just past the `}` closing the ${...} opened at start.

    *start* points at the first character after `${`. Nested braces, string
    literals and template literals inside the expression are skipped.
    """
    n = len(code)
    i = start
    depth = 1
    while i < n:
        ch = code[i]
        if ch in "\"'`":
            quote = ch
            i += 1
            while i < n:
                if code[i] == "\\":
                    i += 2
                    continue
                if code[i] == quote:
                    break
                i += 1
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _read_template(code: str, start: int, spans: list | None = None) -> tuple[str | None, int]:
    """Read a template literal whose opening backtick precedes *start*.

    Returns (value, close_index) where every ${...} interpolation became
    {0}, {1}, ... in order. Returns (None, len(code)) if unterminated.
    If *spans* is given, the (start, end) offsets of each interpolation's
    expression are appended to it so nested literals can be scanned too.
    """
    n = len(code)
    i = start
    parts: list[str] = []
    current: list[str] = []
    placeholders = 0
    while i < n:
        ch = code[i]
        if ch == "\\":
            if i + 1 < n:
                current.append(code[i : i + 2])
                i += 2
                continue
            current.append(ch)
            i += 1
            continue
        if ch == "$" and i + 1 < n and code[i + 1] == "{":
            parts.append(_unescape_js("".join(current)))
            current = []
            parts.append("{%d}" % placeholders)
            placeholders += 1
            end = _skip_interpolation(code, i + 2)
            if spans is not None:
                spans.append((i + 2, end - 1))
            i = end
            continue
        if ch == "`":
            parts.append(_unescape_js("".join(current)))
            return "".join(parts), i
        current.append(ch)
        i += 1
    return None, n


def js_literals(code: str, base_line: int = 1):
    """Yield (line, value) for every JS string literal in *code*.

    A small hand-written scanner: single- and double-quoted literals with
    escape decoding, and template literals whose ${...} interpolations become
    {0}, {1}, ... . Skips ``//`` and ``/* */`` comments. It is not a full JS
    parser: regex literals are not recognised, so a quote character inside a
    regex literal can confuse it. Keep such constructs out of the UI file.
    """
    n = len(code)
    i = 0
    while i < n:
        ch = code[i]
        if ch == "/" and i + 1 < n and code[i + 1] == "/":
            end = code.find("\n", i)
            i = n if end == -1 else end
            continue
        if ch == "/" and i + 1 < n and code[i + 1] == "*":
            end = code.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if ch in "\"'":
            quote = ch
            j = i + 1
            while j < n:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == quote or code[j] == "\n":
                    break
                j += 1
            if j < n and code[j] == quote:
                yield code.count("\n", 0, i) + base_line, _unescape_js(code[i + 1 : j])
                i = j + 1
                continue
            i += 1
            continue
        if ch == "`":
            spans: list = []
            value, close = _read_template(code, i + 1, spans)
            if value is not None:
                yield code.count("\n", 0, i) + base_line, value
                # Literals nested in ${...} (e.g. `${ok ? "是" : "否"}`) are
                # shown text too: scan each interpolation expression.
                for s, e in spans:
                    yield from js_literals(code[s:e], code.count("\n", 0, s) + base_line)
                i = close + 1
                continue
            i += 1
            continue
        i += 1


# --- HTML --------------------------------------------------------------------


def _html_text_strings(html: str):
    """Yield (line, value) for translatable HTML text nodes and attributes."""

    class _TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self._in_script = 0
            self._in_style = 0
            self.items: list[tuple[int, str]] = []

        def _scan_attrs(self, attrs):
            line, _ = self.getpos()
            for name, value in attrs:
                if value is not None and name in _HTML_TEXT_ATTRS:
                    self.items.append((line, value))

        def handle_starttag(self, tag, attrs):
            if tag == "script":
                self._in_script += 1
            elif tag == "style":
                self._in_style += 1
            self._scan_attrs(attrs)

        def handle_startendtag(self, tag, attrs):
            self._scan_attrs(attrs)

        def handle_endtag(self, tag):
            if tag == "script" and self._in_script:
                self._in_script -= 1
            elif tag == "style" and self._in_style:
                self._in_style -= 1

        def handle_data(self, data):
            if self._in_script or self._in_style:
                return  # JS and CSS `content:` strings are scanned separately
            line, _ = self.getpos()
            self.items.append((line, data))

    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.items


_SCRIPT_BLOCK_RE = re.compile(r"<script\b[^>]*>(.*?)</script>", re.S | re.I)
_STYLE_BLOCK_RE = re.compile(r"<style\b[^>]*>(.*?)</style>", re.S | re.I)
# CSS generated content, e.g. `.stage.is-seeking::after { content:"正在定位画面…"; }`
_CSS_CONTENT_RE = re.compile(r"""\bcontent\s*:\s*(["'])((?:\\.|(?!\1).)*)\1""", re.S)


def css_content_strings(css: str, base_line: int = 1):
    """Yield (line, value) for every quoted CSS `content:` string in *css*."""
    for match in _CSS_CONTENT_RE.finditer(css):
        value = re.sub(r"\\(.)", r"\1", match.group(2))
        yield css.count("\n", 0, match.start()) + base_line, value


def _html_strings(html: str):
    """Yield (line, value) for HTML text/attr values and JS literals."""
    for line, value in _html_text_strings(html):
        yield line, value
    for match in _SCRIPT_BLOCK_RE.finditer(html):
        base_line = html.count("\n", 0, match.start(1)) + 1
        for line, value in js_literals(match.group(1), base_line):
            yield line, value
    for match in _STYLE_BLOCK_RE.finditer(html):
        base_line = html.count("\n", 0, match.start(1)) + 1
        yield from css_content_strings(match.group(1), base_line)


# --- Python -------------------------------------------------------------------


def _is_compile_regex(parent) -> bool:
    """True if *parent* is a ``re.compile(...)`` call.

    Its first argument is a regular-expression pattern, not UI text, and a CJK
    range written with literal characters (as in videowipe.i18n) would
    otherwise leak into the catalogue.
    """
    if not isinstance(parent, ast.Call):
        return False
    func = parent.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "compile"
        and isinstance(func.value, ast.Name)
        and func.value.id == "re"
    )


def _python_strings(source: str):
    """Yield (line, value) for str constants and f-strings.

    Skips docstrings and string literals that are ``re.compile`` patterns.
    Every non-constant part of an f-string becomes {0}, {1}, ... in order,
    matching the catalogue placeholder convention.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    parents: dict = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in ast.walk(tree):
        parent = parents.get(node)
        if isinstance(node, ast.JoinedStr):
            if isinstance(parent, ast.Expr):
                continue  # f-string docstring
            parts = []
            count = 0
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                else:
                    parts.append("{%d}" % count)
                    count += 1
            yield node.lineno, "".join(parts)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # parent Expr: docstring / bare expression; JoinedStr: literal part
            # of an f-string, already handled above.
            if isinstance(parent, (ast.Expr, ast.JoinedStr)):
                continue
            if _is_compile_regex(parent):
                continue
            yield node.lineno, node.value


# --- Driver -------------------------------------------------------------------


def default_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_ignore(root: Path) -> dict:
    data = {"paths": [], "strings": set()}
    path = root / LOCALES_RELPATH / "ignore.json"
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        data["paths"] = list(raw.get("paths", []))
        data["strings"] = set(raw.get("strings", []))
    return data


def extract_all(root: Path) -> dict[str, tuple[str, int]]:
    """Return {source string: (relative file, line)} for every CJK string."""
    ignore = load_ignore(root)
    found: dict[str, tuple[str, int]] = {}

    def add(raw, rel, line):
        if not raw:
            return
        key = raw.strip()
        if not key or not _has_cjk(key) or key in ignore["strings"]:
            return
        found.setdefault(key, (rel, line))

    html_path = root / "src" / "videowipe" / "web" / "index.html"
    if html_path.is_file():
        rel = html_path.relative_to(root).as_posix()
        html = html_path.read_text(encoding="utf-8")
        for line, value in _html_strings(html):
            add(value, rel, line)

    web = root / "src" / "videowipe" / "web"
    if web.is_dir():
        for js_path in sorted(web.glob("*.js")):
            rel = js_path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(rel, pattern) for pattern in ignore["paths"]):
                continue
            for line, value in js_literals(js_path.read_text(encoding="utf-8")):
                add(value, rel, line)

    package = root / "src" / "videowipe"
    if package.is_dir():
        for py_path in sorted(package.rglob("*.py")):
            rel = py_path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(rel, pattern) for pattern in ignore["paths"]):
                continue
            for line, value in _python_strings(py_path.read_text(encoding="utf-8")):
                add(value, rel, line)
    return found


# ── Catalogues ───────────────────────────────────────────────────────────────


def _locale_paths(root: Path):
    locales_dir = root / LOCALES_RELPATH
    if not locales_dir.is_dir():
        return []
    return sorted(p for p in locales_dir.glob("*.json") if p.name != "ignore.json")


def load_catalogue(path: Path) -> tuple[dict, dict, dict]:
    meta: dict = {}
    messages: dict = {}
    obsolete: dict = {}
    if path.is_file():
        doc = json.loads(path.read_text(encoding="utf-8"))
        meta = dict(doc.get("_meta") or {})
        messages = {k: v for k, v in (doc.get("messages") or {}).items()}
        obsolete = {k: v for k, v in (doc.get("obsolete") or {}).items()}
    meta.setdefault("language", path.stem)
    meta.setdefault("name", path.stem)
    return meta, messages, obsolete


def _existing_extra(path: Path) -> dict:
    """Hand-added fragment keys ("extra") are curated by translators for text
    upstream only builds by concatenation; sync never adds, moves or drops them."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    extra = doc.get("extra") if isinstance(doc, dict) else None
    return {k: extra[k] for k in sorted(extra)} if isinstance(extra, dict) else {}


def write_catalogue(path: Path, language: str, name: str, messages: dict, obsolete: dict,
                    extra: dict | None = None) -> None:
    doc = {
        "_meta": {
            "language": language,
            "name": name,
            "source_language": SOURCE_LANGUAGE,
        },
        "messages": {k: messages[k] for k in sorted(messages)},
        "obsolete": {k: obsolete[k] for k in sorted(obsolete)},
    }
    extra = _existing_extra(path) if extra is None else {k: extra[k] for k in sorted(extra)}
    if extra:
        doc["extra"] = extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ── Commands ─────────────────────────────────────────────────────────────────


def _root(args) -> Path:
    return Path(args.root).resolve() if getattr(args, "root", None) else default_root()


def run_extract(args) -> int:
    root = _root(args)
    entries = extract_all(root)
    rows = [(f, l, k) for k, (f, l) in entries.items()]
    rows.sort()
    if args.json:
        payload = {
            "strings": [{"key": k, "file": f, "line": l} for f, l, k in rows],
            "count": len(rows),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for f, l, k in rows:
            print(f"{f}:{l}\t{k}")
    return 0


def run_report(args) -> int:
    root = _root(args)
    entries = extract_all(root)
    if not entries:
        print("warning: no translatable strings extracted; check --root")
    for path in _locale_paths(root):
        meta, messages, obsolete = load_catalogue(path)
        untranslated = sorted(k for k in entries if not messages.get(k))
        translated = len(entries) - len(untranslated)
        rel = path.relative_to(root).as_posix()
        print(
            f"{rel}: total={len(entries)} translated={translated} "
            f"untranslated={len(untranslated)} obsolete={len(obsolete)}"
        )
        for key in untranslated:
            f, l = entries[key]
            print(f"  untranslated: {key}\t{f}:{l}")
    return 0


def run_update(args) -> int:
    root = _root(args)
    entries = extract_all(root)
    changed = 0
    for path in _locale_paths(root):
        meta, messages, obsolete = load_catalogue(path)
        extra = _existing_extra(path)
        new_messages: dict = {}
        for key in entries:
            if key in messages:
                new_messages[key] = messages[key]
            elif key in obsolete:
                new_messages[key] = obsolete.pop(key)  # string reappeared: restore
            elif extra.get(key):
                new_messages[key] = extra.pop(key)  # pre-translated: adopt it
            else:
                new_messages[key] = ""  # new upstream string: untranslated
        for key in list(messages):
            if key not in entries:
                obsolete.setdefault(key, messages.pop(key))  # no longer found upstream
        changed_now = _write_if_changed(path, meta, new_messages, obsolete, extra)
        changed += changed_now
        print(("updated " if changed_now else "unchanged ") + path.relative_to(root).as_posix())
    return 0


def _write_if_changed(path, meta, messages, obsolete, extra=None) -> bool:
    before = path.read_text(encoding="utf-8") if path.is_file() else None
    doc = {
        "_meta": {
            "language": meta.get("language", path.stem),
            "name": meta.get("name", path.stem),
            "source_language": SOURCE_LANGUAGE,
        },
        "messages": {k: messages[k] for k in sorted(messages)},
        "obsolete": {k: obsolete[k] for k in sorted(obsolete)},
    }
    extra = _existing_extra(path) if extra is None else {k: extra[k] for k in sorted(extra)}
    if extra:
        doc["extra"] = extra
    after = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
    if before == after:
        return False
    write_catalogue(path, meta.get("language", path.stem), meta.get("name", path.stem), messages, obsolete, extra)
    return True


def run_check(args) -> int:
    root = _root(args)
    entries = extract_all(root)
    if not entries:
        print("warning: no translatable strings extracted; check --root")
    failures = 0
    for path in _locale_paths(root):
        meta, messages, obsolete = load_catalogue(path)
        rel = path.relative_to(root).as_posix()
        untranslated = sorted(k for k in entries if not messages.get(k))
        # Only translated (non-empty) values can drift from their key's
        # placeholders; empty values are simply untranslated.
        mismatches = [
            (k, messages[k])
            for k in sorted(messages)
            if messages[k] and _placeholders(k) != _placeholders(messages[k])
        ]
        stale = sorted(k for k in messages if k not in entries)
        ok = not untranslated and not mismatches
        if not ok:
            failures += 1
        print(
            f"{rel}: {'OK' if ok else 'FAIL'} (untranslated={len(untranslated)}, "
            f"placeholder_mismatches={len(mismatches)}, obsolete={len(obsolete)})"
        )
        for key in untranslated:
            f, l = entries[key]
            print(f"  untranslated: {key}\t{f}:{l}")
        for key, value in mismatches:
            print(f"  placeholder mismatch: {key!r} -> {value!r}")
        for key in stale:
            print(f"  stale (run update): {key}")
    if failures:
        print(f"i18n check failed for {failures} locale(s)")
        return 1
    print("i18n check passed")
    return 0


def run_new_locale(args) -> int:
    root = _root(args)
    code = args.code
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", code):
        print(f"error: invalid locale code {code!r}", file=sys.stderr)
        return 2
    path = root / LOCALES_RELPATH / f"{code}.json"
    if path.exists():
        print(f"error: {path.relative_to(root).as_posix()} already exists", file=sys.stderr)
        return 2
    write_catalogue(path, code, args.name, {}, {})
    print(f"created {path.relative_to(root).as_posix()} ({args.name})")
    return run_update(args)


# --- translate (Anthropic Messages API) ---------------------------------------


def _api_base() -> str:
    return os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")


def _api_headers() -> dict:
    headers = {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        headers["x-api-key"] = key
        return headers
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if token:
        headers["authorization"] = f"Bearer {token}"
        return headers
    raise RuntimeError("set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN")


def _api_payload(model: str, prompt: str) -> dict:
    return {
        "model": model,
        "max_tokens": 16000,
        "messages": [{"role": "user", "content": prompt}],
    }


def _api_call(base: str, headers: dict, payload: dict) -> str:
    request = urllib.request.Request(
        base + "/v1/messages",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=API_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:500]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code} from API: {detail}".strip()) from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach API: {exc.reason}") from None
    return "".join(
        block.get("text", "")
        for block in body.get("content", [])
        if isinstance(block, dict)
    )


def _parse_model_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in response")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("response JSON is not an object")
    return data


def _build_prompt(args, meta, glossary: dict, batch: list[str]) -> str:
    name = meta.get("name", args.locale)
    lines = [
        "You are translating the user-interface strings of VideoWipe, a local tool "
        "for removing subtitles and watermarks from video files.",
        f"Translate each source string (Simplified Chinese) into {name} ({args.locale}).",
        "Rules:",
        "- Keep every {n} placeholder exactly as written; never translate, drop or renumber one.",
        "- Use a concise UI register: short labels and sentences.",
    ]
    if args.locale.lower().startswith("en"):
        lines.append("- Use UK English spellings (colour, cancelled, organise).")
    lines.append(
        "- Return ONLY a JSON object mapping each source string to its translation. "
        "No prose, no markdown."
    )
    if glossary:
        lines.append("Already translated, for consistent terminology:")
        lines.append(json.dumps(glossary, ensure_ascii=False))
    lines.append("Translate these entries:")
    lines.append(json.dumps({key: "" for key in batch}, ensure_ascii=False))
    return "\n".join(lines)


def run_translate(args) -> int:
    root = _root(args)
    path = root / LOCALES_RELPATH / f"{args.locale}.json"
    if not path.is_file():
        print(
            f"error: no catalogue for locale {args.locale!r} "
            f"({path.relative_to(root).as_posix()}); create it with new-locale first",
            file=sys.stderr,
        )
        return 2
    entries = extract_all(root)
    meta, messages, obsolete = load_catalogue(path)
    pending = sorted(k for k in entries if not messages.get(k))
    if not pending:
        print(f"{args.locale}: nothing to translate")
        return 0
    if not args.dry_run:
        _api_headers()  # fail fast on missing credentials

    translated_values = {k: messages[k] for k in sorted(messages) if messages[k]}
    glossary = dict(list(translated_values.items())[:GLOSSARY_SAMPLE_SIZE])
    batches = [pending[i : i + args.batch] for i in range(0, len(pending), max(1, args.batch))]
    translated = 0
    rejected: list[tuple[str, str]] = []
    failed_batches = 0
    for index, batch in enumerate(batches, 1):
        prompt = _build_prompt(args, meta, glossary, batch)
        if args.dry_run:
            print(f"[dry-run] batch {index}/{len(batches)}: {len(batch)} strings")
            continue
        try:
            response_text = _api_call(_api_base(), _api_headers(), _api_payload(args.model, prompt))
            translations = _parse_model_json(response_text)
        except (RuntimeError, ValueError, KeyError) as exc:
            print(f"batch {index}/{len(batches)} failed: {exc}", file=sys.stderr)
            failed_batches += 1
            continue
        for key in batch:
            value = translations.get(key)
            if not isinstance(value, str) or not value.strip():
                rejected.append((key, "missing or empty in response"))
                continue
            if _placeholders(key) != _placeholders(value):
                rejected.append((key, f"placeholder mismatch: {value!r}"))
                continue
            messages[key] = value  # never overwrites: pending were empty
            translated += 1
    if not args.dry_run and translated:
        write_catalogue(path, meta["language"], meta["name"], messages, obsolete)
    for key, why in rejected:
        print(f"rejected: {key}\t{why}", file=sys.stderr)
    if args.dry_run:
        print(
            f"dry run: {len(pending)} strings pending in {len(batches)} batch(es); "
            "nothing written"
        )
        return 0
    print(
        f"{args.locale}: translated {translated}/{len(pending)}, "
        f"rejected {len(rejected)}, failed batches {failed_batches}"
    )
    return 1 if (failed_batches or rejected) else 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="i18n_sync.py",
        description="Extract, sync, check and translate VideoWipe locale catalogues.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_root(p):
        p.add_argument(
            "--root",
            default=None,
            help="repository root (default: the repo containing this script)",
        )

    p = sub.add_parser("extract", help="print every translatable source string")
    add_root(p)
    p.add_argument("--json", action="store_true", help="emit JSON instead of text")

    p = sub.add_parser("report", help="per-locale translation status")
    add_root(p)

    p = sub.add_parser("update", help="sync catalogues with the extracted strings")
    add_root(p)

    p = sub.add_parser("check", help="fail on untranslated strings or placeholder drift")
    add_root(p)

    p = sub.add_parser("new-locale", help="create an empty catalogue and populate it")
    add_root(p)
    p.add_argument("code")
    p.add_argument("name")

    p = sub.add_parser("translate", help="fill empty catalogue values via the Anthropic API")
    add_root(p)
    p.add_argument("--locale", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "extract": run_extract,
        "report": run_report,
        "update": run_update,
        "check": run_check,
        "new-locale": run_new_locale,
        "translate": run_translate,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
