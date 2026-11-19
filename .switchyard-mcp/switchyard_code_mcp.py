#!/usr/bin/env python3
"""Read code by outline and symbol instead of whole files.

An MCP server with read-only code-navigation tools: code_find and code_context
(ranked symbol search, and the best symbols packed into a token budget), code_index,
code_tree, code_outline, code_symbol, code_search, code_refs, code_read, code_changed,
code_tests, code_history and code_describe. It keeps an index of each root's files and their symbols
(functions, classes, methods, headings, keys, ...) in SQLite under
~/.switchyard/code-index (SWITCHYARD_CODE_INDEX_DIR), one file per root named by the
SHA-1 of the root's path, so a git worktree is its own root with its own index. Every
tool indexes its root on first use and re-parses only files whose mtime or size
changed. Search reads the files themselves, streaming and stopping at its caps.
One more tool, report_model, only passes the model's own name to the gateway's
activity report (switchyard_gateway_client).

The default root is the git toplevel of SWITCHYARD_CODE_ROOT (else the current
directory), or that directory itself outside git. SWITCHYARD_CODE_ALLOWED
(directories separated by ':') limits which roots may be opened. An installed
third-party package can be indexed as a separate read-only root, `dep:<name>`.

Speaks MCP over stdio (newline-delimited JSON-RPC). Standard library only,
Python 3.9+; run it with `python3 -I switchyard_code_mcp.py`. From a shell:
`python3 -I switchyard_code_mcp.py outline '{"path": "server.py"}'`.
"""

import array
import ast
import bisect
import datetime
import fnmatch
import glob as globlib
import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python3 -I leaves the script's folder off sys.path
try:
    import switchyard_gateway_client as gateway_client
except ImportError:  # this file copied somewhere on its own: environment only, no activity report
    gateway_client = None


def gateway_settings():
    """(url or None, key or None) of the Switchyard gateway: SWITCHYARD_URL and SWITCHYARD_API_KEY from the
    environment, else from the gateway file (~/.switchyard/gateway.env)."""
    if gateway_client is not None:
        return gateway_client.gateway_settings()
    url = (os.environ.get("SWITCHYARD_URL") or "").strip().rstrip("/")
    key = (os.environ.get("SWITCHYARD_API_KEY") or "").strip()
    return url or None, key or None


SCHEMA_VERSION = "2"
MAX_FILE_BYTES = 2 * 1024 * 1024   # bigger files are listed but not parsed or searched
BINARY_SNIFF = 8192                # a NUL in the first 8k bytes means binary
MAX_FILES = 50000                  # most files one root indexes
MAX_SIG = 200                      # longest signature kept
MAX_DOC = 120                      # longest doc line kept
MAX_IDENTS = 20000                 # most distinct identifiers remembered per file
SEARCH_MAX_CHARS = 6000            # default output cap of code_search and code_refs
SEARCH_LINE_CHARS = 240            # longer matching lines are cut
READ_MAX_LINES = 500               # hard cap of code_read, whatever max_lines says
READ_LINE_CHARS = 1000
MAX_OUTPUT_CHARS = 40000           # most text any call returns
SYMBOL_MAX_LINES = 400             # longest body code_symbol prints
OUTLINE_MAX_CHARS = 20000
TREE_MAX_ENTRIES = 400
FIND_LIMIT = 20                    # default rows of code_find
CONTEXT_BUDGET = 4000              # default token budget of code_context
TESTS_LIMIT = 50                   # default rows of code_tests
HISTORY_LIMIT = 10                 # default commits of code_history
GIT_TIMEOUT = 20                  # seconds any git command may take
DOCX_MAX_XML = 20 * 1024 * 1024    # most bytes of word/document.xml read

SKIP_DIRS = {"node_modules", ".venv", "venv", ".git", ".hg", ".svn", "dist", "build", "target",
             "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".next"}
# Secret-looking files are never indexed or read (an example/sample/template copy is fine)
SECRET_RE = re.compile(r"(^|/)(\.env(\.[^/]*)?|[^/]*\.(pem|p12|pfx)|id_(rsa|dsa|ecdsa|ed25519)[^/]*|"
                       r"\.netrc|\.npmrc|\.pypirc|credentials\.json)$")
SECRET_OK_RE = re.compile(r"\.(example|sample|template|dist|defaults?)$")

LANGUAGES = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".cs": "csharp", ".swift": "swift", ".rb": "ruby", ".rake": "ruby",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".c++": "cpp", ".hpp": "cpp",
    ".hh": "cpp", ".hxx": "cpp", ".ipp": "cpp", ".inl": "cpp",
    ".php": "php", ".phtml": "php", ".sql": "sql",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".md": "markdown", ".markdown": "markdown", ".mdx": "markdown",
    ".rst": "rst", ".adoc": "asciidoc", ".asciidoc": "asciidoc", ".asc": "asciidoc",
    ".ipynb": "notebook", ".docx": "docx",
    ".yml": "yaml", ".yaml": "yaml", ".json": "json", ".toml": "toml",
    ".html": "html", ".htm": "html", ".css": "css", ".scss": "css", ".less": "css",
}
FILENAMES = {"Rakefile": "ruby", "Gemfile": "ruby", "Dockerfile": "text", "Makefile": "text"}
# Files shown through a text view (cells, paragraphs) rather than their raw bytes
VIEW_LANGUAGES = {"notebook", "docx"}
# Languages whose identifiers count as references (code_find's inbound count, code_refs' prefilter)
CODE_LANGUAGES = {"python", "javascript", "typescript", "go", "rust", "java", "kotlin", "csharp", "swift",
                  "ruby", "c", "cpp", "php", "shell", "sql", "html", "css", "notebook"}

class ToolError(Exception):
    """A problem to report to the model as tool output (isError), not a crash."""


# --- small helpers ------------------------------------------------------------------------

def human_size(n) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB"):
        if n < 1024 or unit == "MB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def when(timestamp) -> str:
    try:
        return datetime.datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OverflowError, OSError):
        return "never"


def integer(args: dict, name: str, default=None, low=None, high=None):
    value = args.get(name, default)
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ToolError(f"Argument {name} must be an integer, not {value!r}.")
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def boolean(args: dict, name: str, default=False) -> bool:
    value = args.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "0", "no", "off")
    return bool(value)


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]} ... [{len(text) - limit:,} more characters]"


def short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def finish(lines: list, limit: int = 0) -> str:
    limit = limit or MAX_OUTPUT_CHARS
    text = "\n".join(lines)
    if len(text) > limit:
        text = text[:limit].rsplit("\n", 1)[0] + "\n... [output cut to fit; ask for less]"
    return text


def is_secret(rel: str) -> bool:
    return bool(SECRET_RE.search(rel)) and not SECRET_OK_RE.search(rel)


def language_of(rel: str, head: bytes = b"") -> str:
    name = os.path.basename(rel)
    ext = os.path.splitext(name)[1].lower()
    if ext in LANGUAGES:
        return LANGUAGES[ext]
    if name in FILENAMES:
        return FILENAMES[name]
    if head.startswith(b"#!"):
        first = head.split(b"\n", 1)[0]
        if re.search(rb"\b(ba|z|k|da)?sh\b", first):
            return "shell"
        if b"python" in first:
            return "python"
        if b"ruby" in first:
            return "ruby"
        if b"node" in first:
            return "javascript"
    return "text"

# --- comment and string blanking for the brace-based parsers ------------------------------

C_LIKE = {"javascript", "typescript", "go", "rust", "java", "kotlin", "csharp", "swift", "css", "c", "cpp", "php"}
CHAR_LITERAL = re.compile(r"'(?:\\.[^']{0,8}|[^\\'\n])'")
REGEX_PREV = set("(,=:[!&|?{};+-*%~^") | {""}  # not < or >: </tag> and <br/> in JSX
SINGLE_IS_STRING = {"javascript", "typescript", "shell", "ruby", "python", "css", "php", "sql", "yaml", "toml"}
HASH_COMMENTS = {"shell", "ruby", "yaml", "toml", "python", "php"}
NESTED_BLOCK_COMMENTS = {"rust", "swift", "kotlin"}
MULTILINE_STRINGS = {"shell", "ruby", "php", "sql", "toml"}
HEREDOC = {
    "shell": re.compile(r"<<(-?)[ \t]*(['\"]?)([A-Za-z_][\w.-]*)\2"),
    "ruby": re.compile(r"<<([~-]?)(['\"]?)([A-Z_][A-Z0-9_]*)\2|<<([~-])(['\"]?)([A-Za-z_]\w*)\5"),
    "php": re.compile(r"<<<[ \t]*(['\"]?)([A-Za-z_]\w*)\1"),
}
# interpolation that holds code (with its own strings) inside a double-quoted string
INTERPOLATION = {"kotlin": ("${", "}"), "ruby": ("#{", "}"), "php": ("{$", "}")}


def blank_code(text: str, lang: str) -> str:
    """text with comments and string literals replaced by spaces (newlines kept), so
    braces and keywords inside them don't count. A small lexer per language family:
    nested template literals and interpolations, heredocs, raw and verbatim strings,
    nested block comments. Not a real lexer; good enough."""
    n = len(text)
    out = list(text)
    slash_line = lang in C_LIKE and lang != "css"  # url(//x) in CSS is not a comment
    slash_block = lang in C_LIKE or lang == "sql"
    hash_line = lang in HASH_COMMENTS
    dash_line = lang == "sql"
    triple = lang in ("kotlin", "swift", "java", "csharp", "python")
    single_is_string = lang in SINGLE_IS_STRING
    backtick = lang in ("javascript", "typescript", "go", "shell", "ruby")
    multiline = lang in MULTILINE_STRINGS
    heredoc = HEREDOC.get(lang)
    jsx = lang in ("javascript", "typescript")
    pending = []  # heredoc terminators waiting for the end of this line

    def blank(a, b):
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    def heredoc_bodies(i):
        """i is at a newline: skip the bodies of the heredocs opened on this line."""
        j = i + 1
        for strip, word in pending:
            while j < n:
                end = text.find("\n", j)
                end = n if end < 0 else end
                line = text[j:end]
                body = line.strip() if strip else line.rstrip("\r")
                if body == word or (lang == "php" and re.match(re.escape(word) + r"\b", line.strip())):
                    blank(i + 1, end)
                    j = end
                    break
                j = end + 1
            else:
                blank(i + 1, n)
                j = n
        pending.clear()
        return j

    def comment_end(i):
        ch = text[i]
        two = text[i:i + 2]
        if slash_line and two == "//" and not (jsx and i > 0 and text[i - 1] == ":"):  # http:// in JSX text
            j = text.find("\n", i)
            return n if j < 0 else j
        if slash_block and two == "/*":
            if lang in NESTED_BLOCK_COMMENTS:
                depth, j = 1, i + 2
                while j < n and depth:
                    if text.startswith("/*", j):
                        depth, j = depth + 1, j + 2
                    elif text.startswith("*/", j):
                        depth, j = depth - 1, j + 2
                    else:
                        j += 1
                return j
            j = text.find("*/", i + 2)
            return n if j < 0 else j + 2
        if dash_line and two == "--":
            j = text.find("\n", i)
            return n if j < 0 else j
        if hash_line and ch == "#":
            if lang == "shell" and i > 0 and text[i - 1] not in " \t\n;|&(":
                return 0  # $#, ${#x}, a#b
            if lang == "php" and text.startswith("#[", i):
                return 0  # an attribute
            j = text.find("\n", i)
            return n if j < 0 else j
        if lang == "ruby" and ch == "=" and (i == 0 or text[i - 1] == "\n") and text.startswith("=begin", i):
            m = re.compile(r"^=end\b.*$", re.M).search(text, i)
            return m.end() if m else n
        return 0

    def after_arrow(i):
        k = i - 1
        while k > 0 and text[k] in " \t\n":
            k -= 1
        return text[k - 1:k + 1] == "=>"

    def string_end(i, last_sig):
        """End (exclusive) of the string literal starting at i, or 0 when none starts there."""
        ch = text[i]
        if lang == "rust" and ch in "rb" and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            m = re.compile(r'b?r(#*)"').match(text, i)
            if m:
                close = '"' + m.group(1)
                j = text.find(close, m.end())
                return n if j < 0 else j + len(close)
        if lang == "cpp" and ch == "R" and text.startswith('R"', i) and (i == 0 or not text[i - 1].isalnum()):
            m = re.compile(r'R"([^ ()\\\n]{0,16})\(').match(text, i)
            if m:
                close = ")" + m.group(1) + '"'
                j = text.find(close, m.end())
                return n if j < 0 else j + len(close)
        if lang == "csharp" and ch in "@$" and re.compile(r'(?:\$@|@\$|@)"').match(text, i):
            j = text.find('"', i) + 1
            while j < n:
                if text[j] == '"':
                    if text.startswith('""', j):
                        j += 2
                        continue
                    return j + 1
                j += 1
            return n
        if lang == "sql" and ch == "$" and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            m = re.compile(r"\$([A-Za-z_]\w*)?\$").match(text, i)
            if m:
                j = text.find(m.group(0), m.end())
                return n if j < 0 else j + len(m.group(0))
        if triple and text.startswith('"""', i):
            j = text.find('"""', i + 3)
            return n if j < 0 else j + 3
        if ch == "`" and backtick:
            return quoted_end(i, "`")
        if jsx and ch in "'\"" and i > 0 and (text[i - 1].isalnum()):
            return 0  # an apostrophe in JSX text (don't), never a string in code
        if ch == '"':
            return quoted_end(i, '"')
        if ch == "'":
            if single_is_string:
                return quoted_end(i, "'")
            m = CHAR_LITERAL.match(text, i)
            return m.end() if m else 0
        if ch == "/" and lang in ("javascript", "typescript") and \
                (last_sig in REGEX_PREV or after_arrow(i)) and \
                text[i:i + 2] not in ("//", "/*", "/>"):  # /> closes a JSX tag
            # a regular-expression literal: up to the closing / on this line
            j, in_class = i + 1, False
            while j < n and text[j] != "\n":
                c = text[j]
                if c == "\\":
                    j += 2
                    continue
                if c == "[":
                    in_class = True
                elif c == "]":
                    in_class = False
                elif c == "/" and not in_class:
                    return j + 1
                j += 1
        return 0

    def quoted_end(i, q):
        j = i + 1
        escapes = not ((q == "`" and lang == "go") or (q == "'" and lang in ("shell", "sql")) or
                       (q == '"' and lang == "sql"))
        spans_lines = q == "`" or multiline
        while j < n:
            c = text[j]
            if q == "`" and lang in ("javascript", "typescript") and text.startswith("${", j):
                j = run(j + 2, "}", False)
                continue
            if q == '"' and lang == "shell" and c == "$" and text[j + 1:j + 2] in ("(", "{"):
                j = run(j + 2, ")" if text[j + 1] == "(" else "}", False)
                continue
            if q == '"' and lang == "swift" and text.startswith("\\(", j):
                j = run(j + 2, ")", False)
                continue
            if q == '"' and lang in INTERPOLATION and text.startswith(INTERPOLATION[lang][0], j):
                opener = INTERPOLATION[lang][0]
                j = run(j + len(opener) - (1 if lang == "php" else 0), "}", False)
                continue
            if c == "\\" and escapes:
                j += 2
                continue
            if lang == "sql" and c == q and text[j + 1:j + 2] == q:
                j += 2
                continue
            if c == q:
                return j + 1
            if c == "\n" and not spans_lines:
                return j
            j += 1
        return n

    def run(i, closer, top):
        """Lex from i; at the top level blank comments and strings. With a closer, stop
        after the closer that balances (for code inside an interpolation)."""
        opener = {"}": "{", ")": "("}.get(closer)
        depth = 0
        last_sig = ""
        while i < n:
            ch = text[i]
            if ch == "\n" and pending and top:
                i = heredoc_bodies(i)
                continue
            if closer is not None:
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    if depth == 0:
                        return i + 1
                    depth -= 1
            j = comment_end(i)
            if j:
                if top:
                    blank(i, j)
                i = j
                continue
            j = string_end(i, last_sig)
            if j:
                if top:
                    blank(i, j)
                i = j
                last_sig = '"'
                continue
            if heredoc is not None and ch == "<" and top:
                if lang == "shell" and text.startswith("<<<", i):
                    i += 3  # a here-string, not a heredoc
                    continue
                m = heredoc.match(text, i)
                if m:
                    groups = [g for g in m.groups()]
                    if lang == "php":
                        strip, word = True, groups[1]
                    elif lang == "ruby":
                        strip, word = bool(groups[0] or groups[3]), groups[2] or groups[5]
                    else:
                        strip, word = bool(groups[0]), groups[2]
                    if word:
                        pending.append((strip, word))
                        i = m.end()
                        last_sig = '"'
                        continue
            if not ch.isspace():
                last_sig = ch
            i += 1
        return n

    run(0, None, True)
    return "".join(out)


# --- symbols --------------------------------------------------------------------------------

def symbol(kind, name, start, end, line, depth, sig, parent="", doc=""):
    return {"kind": kind, "name": name, "qual": f"{parent}.{name}" if parent else name, "parent": parent,
            "start": start, "end": end, "line": line, "depth": depth,
            "sig": short(sig, MAX_SIG), "doc": short(doc, MAX_DOC) if doc else ""}


COMMENT_MARKERS = re.compile(r"^\s*(?:///?!?|/\*[*!]?|\*/?|#+|--|<!--|;+)\s*")


def comment_text(line: str) -> str:
    text = COMMENT_MARKERS.sub("", line).strip()
    text = re.sub(r"\s*(\*/|-->)\s*$", "", text)
    return text if re.search(r"[A-Za-z0-9]", text) else ""


def is_comment(stripped: str) -> bool:
    return stripped.startswith(("//", "/*", "*", "#", "--", "<!--")) and not stripped.startswith(("#[", "#!"))


def comment_above(lines: list, start: int) -> str:
    """First line of the comment block just above line `start` (1-based), skipping
    annotations and attributes."""
    j = start - 2
    while j >= 0 and (lines[j].strip().startswith(("@", "#[")) or re.fullmatch(r"\[[A-Z][^\]]*\]", lines[j].strip())):
        j -= 1  # decorators, annotations and attributes
    block = []
    while j >= 0 and len(block) < 30:
        s = lines[j].strip()
        if not s or not is_comment(s):
            break
        block.append(s)
        j -= 1
    for s in reversed(block):
        text = comment_text(s)
        if text:
            return text
    return ""


def leading_doc(lines: list) -> str:
    """The first line of the comment at the top of a file (after a shebang)."""
    for raw in lines[:40]:
        s = raw.strip()
        if not s or s.startswith("#!"):
            continue
        if is_comment(s) or s.startswith(('"""', "'''")):
            text = comment_text(s.strip("\"'"))
            if text:
                return text
            continue
        return ""
    return ""


def nest(symbols: list, dotted: bool = True) -> list:
    """Set depth (and the dotted parent for code) from line ranges."""
    symbols.sort(key=lambda s: (s["start"], -s["end"]))
    stack = []
    for s in symbols:
        while stack and stack[-1]["end"] < s["start"]:
            stack.pop()
        s["depth"] = len(stack)
        if dotted and stack:
            s["parent"] = stack[-1]["qual"]
            s["qual"] = f"{stack[-1]['qual']}.{s['name']}"
        stack.append(s)
    return symbols


def balanced_end(s: str, i: int, open_ch: str, close_ch: str) -> int:
    """Index just after the bracket that closes the one at s[i] (len(s) if none does)."""
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return j + 1
    return len(s)


GENERIC = re.compile(r"<[^<>(){};=]*>")


def strip_generics(s: str) -> str:
    """s without its <...> type parameters, innermost first."""
    for _ in range(8):
        t = GENERIC.sub("", s)
        if t == s:
            break
        s = t
    return s


AT_DECORATORS = {"javascript", "typescript", "java", "kotlin", "swift"}
DECO_AT = re.compile(r"@(?!interface\b)[A-Za-z_][\w.$]*(?::[\w.$]+)?")


def skip_decorators(s: str, lang: str):
    """(start, decl) offsets in a declaration header: where the symbol starts (its
    decorators and attributes included, file prefixes and inner attributes left out) and
    where the declaration itself starts."""
    i, n, start = 0, len(s), None
    while True:
        while i < n and s[i].isspace():
            i += 1
        if lang == "php" and s.startswith("<?php", i):
            i += 5
            continue
        if lang == "rust" and s.startswith("#![", i):
            i = balanced_end(s, i + 2, "[", "]")
            continue
        if start is None:
            start = i
        if lang in AT_DECORATORS and s.startswith("@", i):
            m = DECO_AT.match(s, i)
            if m:
                j = k = m.end()
                while k < n and s[k] in " \t":
                    k += 1
                if k < n and s[k] == "(":
                    j = balanced_end(s, k, "(", ")")
                i = j
                continue
        if (lang in ("rust", "php") and s.startswith("#[", i)) or (lang == "csharp" and s.startswith("[", i)) or \
                (lang in ("c", "cpp") and s.startswith("[[", i)):
            i = balanced_end(s, i + (1 if s[i] == "#" else 0), "[", "]")
            continue
        break
    return (start if start is not None else i), i


# --- brace languages: one scanner, a classifier per language --------------------------------

TRANSPARENT = {"transparent": True}  # a block whose members belong to the enclosing scope
TOKEN = re.compile(r"[{}()\[\];\n]|[^\s{}()\[\];]+")
CONT_END = re.compile(r"(?:,|\(|\[|(?<![=!<>])=|:|\||&&|\.|\\|=>|->|\b(?:extends|implements|where|throws))\s*$")
NEXT_START = re.compile(r"\s*(?:[{.?:,|=+]|&&|=>|->|\b(?:where|extends|implements|throws)\b)")


class Ctx:
    """What a classifier is told about a header: the language, the scope it is in
    (top, members, body or group), its container symbol, the group kind (Go's
    type/const/var blocks) and the character that ended it ({, ;, a newline or ))."""

    def __init__(self, lang, scope, container, term, group=None):
        self.lang, self.scope, self.container, self.term, self.group = lang, scope, container, term, group

    @property
    def container_name(self):
        return self.container["name"] if self.container else ""

    @property
    def container_qual(self):
        return self.container["qual"] if self.container else ""


class Rules:
    def __init__(self, classify, newline_ends=False, inline=None, group=None, prepass=None):
        self.classify, self.newline_ends, self.inline, self.group, self.prepass = \
            classify, newline_ends, inline, group, prepass


def found(kind, name, scope="none", start=0, decl=None, parent=None, decl_only=False, typedef=False):
    """A classifier's answer: the symbol's kind and name, the scope of its body, the
    offsets of its start (decorators included) and of the declaration in the header, an
    explicit parent qual (else the container's), and whether it is only a declaration."""
    r = {"kind": kind, "name": name, "scope": scope, "start": start, "decl": start if decl is None else decl,
         "decl_only": decl_only, "typedef": typedef}
    if parent is not None:
        r["parent"] = parent
    return r


def scan_blocks(text: str, lines: list, lang: str, rules: Rules) -> list:
    """Symbols of a brace-delimited language. Walks the code (comments and strings
    blanked) token by token, keeping a stack of blocks; the text since the last ';', '{'
    or '}' (and, for languages without semicolons, the last line end that finishes a
    statement) is a header, which the language's classifier turns into a symbol when it
    ends at a '{' (a body that closes at its matching '}') or at a ';' or line end (a
    declaration without a body). Headers are only classified directly inside the top
    level, a namespace, a class-like body or (for nested definitions) a function body,
    so calls and control flow inside functions are never taken for definitions."""
    code = blank_code(text, lang)
    symbols = []
    if rules.prepass:
        code, symbols = rules.prepass(code, lines)
    code_lines = code.split("\n")
    next_nonblank, upcoming = [None] * len(code_lines), None
    for k in range(len(code_lines) - 1, -1, -1):
        next_nonblank[k] = upcoming
        if code_lines[k].strip():
            upcoming = k
    stack = [{"type": "block", "sym": None, "scope": "top", "container": None}]
    st = {"hstart": None, "hline": 1, "paren": 0}
    typedef = [None]  # (symbol, frame): an anonymous typedef'd struct waiting for its name
    line = 1

    def frame():
        f = stack[-1]
        return None if f["type"] == "inline" else f

    def classify(header, f, term):
        try:
            return rules.classify(header, Ctx(lang, f["scope"], f["container"], term, f.get("group")))
        except (re.error, IndexError, ValueError):
            return None

    def make(r, header, f, term, here):
        hs_line = st["hline"]
        start = hs_line + header.count("\n", 0, r["start"])
        decl_line = hs_line + header.count("\n", 0, r["decl"])
        if term == "{":
            sig_end = here
        else:
            rest = header[r["decl"]:]
            p = rest.find("(")
            sig_end = decl_line + (rest.count("\n", 0, balanced_end(rest, p, "(", ")")) if p >= 0 else 0)
            sig_end = min(max(sig_end, decl_line), here)
        sig = " ".join(l.strip() for l in lines[decl_line - 1:sig_end])
        container = f["container"]
        parent = r["parent"] if "parent" in r else (container["qual"] if container else "")
        depth = container["depth"] + 1 if container else 0
        sym = symbol(r["kind"], r["name"], start, here, decl_line, depth, sig, parent, comment_above(lines, start))
        if r["decl_only"]:
            sym["decl"] = 1
        return sym

    def finish(term, at, here):
        """The header ends at a ';', a line end or a group's ')': a bodiless declaration."""
        hs = st["hstart"]
        st["hstart"] = None
        f = frame()
        if hs is None or f is None:
            return
        header = code[hs:at]
        pending = typedef[0]
        typedef[0] = None
        if pending is not None and pending[1] is f:
            m = re.match(r"\s*[*\s]*([A-Za-z_]\w*)", header)
            if m:
                sym = pending[0]
                sym["name"] = m.group(1)
                sym["qual"] = f"{sym['parent']}.{sym['name']}" if sym["parent"] else sym["name"]
                sym["sig"] = short(f"typedef {sym['sig']} {m.group(1)}", MAX_SIG)
                sym["end"] = here
                symbols.append(sym)
            return
        if f["scope"] == "none" or not header.strip():
            return
        r = classify(header, f, term)
        if r and r is not TRANSPARENT:
            if term not in (";", "{"):  # ends on its last line of text, not where the next one starts
                here = st["hline"] + header.rstrip().count("\n")
            symbols.append(make(r, header, f, term, here))

    def ends_here(pos):
        """At a line end with balanced parentheses: does the statement end here?"""
        cur = code_lines[line - 1].rstrip()
        tail = lines[line - 1][len(cur):].strip() if line - 1 < len(lines) else ""
        ends_in_literal = bool(tail) and not tail.startswith(("//", "/*", "#", "--"))
        if not ends_in_literal and CONT_END.search(cur):
            return False
        nxt = next_nonblank[line - 1]
        if nxt is not None and NEXT_START.match(code_lines[nxt]):
            return False
        header = code[st["hstart"]:pos]
        start, decl = skip_decorators(header, lang)
        if decl > start and not header[decl:].strip():
            return False  # decorators only: they belong to the next line's declaration
        return True

    for m in TOKEN.finditer(code):
        tok, pos = m.group(0), m.start()
        if tok == "\n":
            if rules.newline_ends and st["hstart"] is not None and st["paren"] == 0 and \
                    stack[-1]["type"] != "inline" and ends_here(pos):
                finish("\n", pos, line)
            line += 1
            continue
        if st["hstart"] is None and tok not in ("}", ")", "]", ";"):
            st["hstart"], st["hline"] = pos, line
        if tok == "(" or tok == "[":
            if tok == "(" and st["paren"] == 0 and rules.group and frame() is not None:
                f = frame()
                header = code[st["hstart"]:pos]
                g = rules.group(header, f, lang) if f["scope"] != "none" else None
                if g is not None:
                    new = {"type": "group", "sym": None, "scope": "group", "container": f["container"]}
                    if "kind" in g:  # a definition whose body is in parentheses (a shell subshell function)
                        sym = make(g, header, f, "{", line)
                        symbols.append(sym)
                        new.update(sym=sym, scope=g["scope"], container=sym)
                    else:
                        new["group"] = g["group"]
                    stack.append(new)
                    st["hstart"] = None
                    continue
            st["paren"] += 1
        elif tok == ")" or tok == "]":
            if st["paren"] > 0:
                st["paren"] -= 1
            elif tok == ")" and stack[-1]["type"] == "group":
                finish(")", pos, line)
                closed = stack.pop()
                if closed["sym"] is not None:
                    closed["sym"]["end"] = line
                st["hstart"] = None
        elif tok == "{":
            header = code[st["hstart"]:pos] if st["hstart"] is not None else ""
            if st["paren"] > 0 or (rules.inline and header.strip() and rules.inline(header)):
                stack.append({"type": "inline", "saved": (st["hstart"], st["hline"], st["paren"])})
                st["hstart"], st["paren"] = None, 0
                continue
            f = frame()
            typedef[0] = None
            new = {"type": "block", "sym": None, "scope": "none", "container": f["container"] if f else None}
            if f is not None and f["scope"] != "none" and header.strip():
                r = classify(header, f, "{")
                if r is TRANSPARENT:
                    new.update(scope=f["scope"], container=f["container"], group=f.get("group"))
                elif r:
                    sym = make(r, header, f, "{", line)
                    new.update(sym=sym, scope=r["scope"], container=sym, typedef=r["typedef"])
                    if not r["typedef"]:
                        symbols.append(sym)
            stack.append(new)
            st["hstart"], st["paren"] = None, 0
        elif tok == "}":
            if len(stack) > 1:
                closed = stack.pop()
                if closed["type"] == "inline":
                    st["hstart"], st["hline"], st["paren"] = closed["saved"]
                    continue
                if closed["sym"] is not None:
                    closed["sym"]["end"] = line
                    if closed.get("typedef"):
                        typedef[0] = (closed["sym"], frame())
            st["hstart"], st["paren"] = None, 0
        elif tok == ";":
            if st["paren"] == 0:
                finish(";", pos, line)
        elif st["hstart"] is not None and line - st["hline"] > 80 and not rules.newline_ends:
            st["hstart"], st["hline"] = pos, line  # a runaway header (a macro without ';'): start again
    for f in stack:
        if f.get("sym") is not None and f["type"] != "inline" and f["sym"]["end"] <= f["sym"]["start"]:
            f["sym"]["end"] = len(lines)
    symbols.sort(key=lambda s: (s["start"], -s["end"]))
    return merge_overloads(symbols)


def merge_overloads(symbols: list) -> list:
    """Bodiless overload signatures directly above the definition they describe
    (TypeScript) become part of it."""
    out = []
    for s in symbols:
        prev = out[-1] if out else None
        if prev is not None and prev.get("decl") and prev["qual"] == s["qual"] and prev["kind"] == s["kind"] and \
                0 <= s["start"] - prev["end"] <= 3:
            s["start"] = prev["start"]
            out[-1] = s
            continue
        out.append(s)
    return out


def ends_with(header: str, endings) -> bool:
    return header.rstrip().endswith(endings)


# --- JavaScript and TypeScript ---

JS_HEAD = r"(?:export\s+)?(?:default\s+)?(?:declare\s+)?"
JS_FUNC = re.compile(r"\s*" + JS_HEAD + r"(?:async\s+)?function\b\s*\*?\s*([A-Za-z_$][\w$]*)")
JS_CLASS = re.compile(r"\s*" + JS_HEAD + r"(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)")
JS_IFACE = re.compile(r"\s*" + JS_HEAD + r"interface\s+([A-Za-z_$][\w$]*)")
JS_ENUM = re.compile(r"\s*" + JS_HEAD + r"(?:const\s+)?enum\s+([A-Za-z_$][\w$]*)")
JS_NS = re.compile(r"\s*" + JS_HEAD + r"(?:namespace|module)\s+([A-Za-z_$][\w$.]*)\s*$")
JS_ANON_NS = re.compile(r"\s*" + JS_HEAD + r"(?:module|global)\s*$")
JS_TYPE = re.compile(r"\s*" + JS_HEAD + r"type\s+([A-Za-z_$][\w$]*)\s*(?:<.*?>)?\s*=", re.S)
JS_VAR = re.compile(r"\s*(export\s+)?(?:declare\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*[!]?\s*"
                    r"(?::(?:[^=]|=>)*)?=(?![=>])\s*(.*)$", re.S)
JS_ASSIGN_FN = re.compile(r"\s*((?:[A-Za-z_$][\w$]*\.)+)([A-Za-z_$][\w$]*)\s*=(?![=>])\s*(.*)$", re.S)
JS_WRAPPED_FN = re.compile(r"\s*[\w$.]+\s*(?:<[^()]*?>)?\s*\(\s*(?:async\s*)?(?:\([^()]*\)|[\w$]+)\s*(?::[^=()]*)?=>")
JS_MEMBER_MODS = r"(?:(?:public|private|protected|static|async|readonly|override|abstract|declare|accessor)\s+)*"
JS_METHOD = re.compile(r"\s*" + JS_MEMBER_MODS + r"(?:(get|set)\s+(?=[#\w$]))?\*?\s*(#?[A-Za-z_$][\w$]*)\s*[?!]?\s*"
                       r"(?:<.*?>)?\s*\(", re.S)
JS_FIELD = re.compile(r"\s*" + JS_MEMBER_MODS + r"(#?[A-Za-z_$][\w$]*)\s*[?!]?\s*(?::(?:[^=]|=>)*)?=(?![=>])\s*(.*)$",
                      re.S)
JS_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "function", "new", "await", "typeof",
               "else", "do", "try", "with", "yield", "throw", "delete", "void", "super", "import"}


def js_function_value(rhs: str) -> bool:
    """Is this right-hand side a function (function expression or arrow function)?"""
    r = re.sub(r"^\s*async\b\s*", "", rhs).lstrip()
    if r.startswith("function"):
        return True
    if r.startswith("<"):
        r = r[balanced_end(r, 0, "<", ">"):].lstrip()
    if r.startswith("("):
        rest = r[balanced_end(r, 0, "(", ")"):].lstrip()
        return rest.startswith("=>") or (rest.startswith(":") and "=>" in rest)
    return bool(re.match(r"[A-Za-z_$][\w$]*\s*=>", r))


JS_TYPED = re.compile(r"\s*(?:export\s+)?(?:declare\s+)?(?:(?:const|let|var)\s+|" + JS_MEMBER_MODS +
                      r")#?[\w$]+\s*[!?]?\s*:(?:[^=]|=>)*$", re.S)


def js_inline(header: str) -> bool:
    """A '{' that opens an object literal or a type, part of the statement around it."""
    h = header.rstrip()
    if h.endswith(("=", ":", "(", ",", "?", "[", "||", "&&", "??", "...")) or re.search(r"\b(return|import|export|in|of)$", h):
        return True
    if JS_TYPED.match(h):  # const x: readonly { a: string }[] = ... (a type, not a body)
        return True
    return h.endswith("extends") and h.count("<") > h.count(">")


def js_classify(header: str, ctx: Ctx):
    start, decl = skip_decorators(header, ctx.lang)
    s = header[decl:].replace("\n", " ")
    term = ctx.term
    if ctx.scope == "top":
        m = JS_FUNC.match(s)
        if m:
            return found("function", m.group(1), "body", start, decl, decl_only=term != "{")
        m = JS_CLASS.match(s)
        if m:
            return found("class", m.group(1), "members", start, decl)
        m = JS_IFACE.match(s)
        if m:
            return found("interface", m.group(1), "none", start, decl)
        m = JS_ENUM.match(s)
        if m:
            return found("enum", m.group(1), "none", start, decl)
        if term == "{":
            m = JS_NS.match(s)
            if m:
                return found("namespace", m.group(1), "top", start, decl)
            if JS_ANON_NS.match(s):
                return TRANSPARENT
        m = JS_TYPE.match(s)
        if m:
            return found("type", m.group(1), "none", start, decl)
        m = JS_VAR.match(s)
        if m:
            rhs = m.group(3)
            if js_function_value(rhs):
                return found("function", m.group(2), "body", start, decl)
            if re.match(r"class\b", rhs.strip()):
                return found("class", m.group(2), "members", start, decl)
            if JS_WRAPPED_FN.match(rhs):  # a function made by a call: memo(() => ...), lazy(() => import(...))
                return found("function", m.group(2), "body", start, decl)
            if m.group(1) or re.fullmatch(r"[A-Z][A-Za-z0-9_]*", m.group(2)):
                return found("constant", m.group(2), "none", start, decl)
            return None
        m = JS_ASSIGN_FN.match(s)
        if m and js_function_value(m.group(3)) and not m.group(1).startswith(("this.", "module.", "exports.")):
            owner = m.group(1)[:-1]
            if owner.endswith(".prototype"):
                return found("method", m.group(2), "body", start, decl, parent=owner[:-len(".prototype")])
            return found("function", m.group(2), "body", start, decl, parent=owner)
        if m and js_function_value(m.group(3)):
            return found("function", m.group(2), "body", start, decl)
        return None
    if ctx.scope == "members":
        m = JS_FIELD.match(s)
        if m:
            if js_function_value(m.group(2)):
                return found("method", m.group(1), "body", start, decl)
            return None
        m = JS_METHOD.match(s)
        if m and m.group(2) not in JS_KEYWORDS:
            kind = {"get": "getter", "set": "setter"}.get(m.group(1), "method")
            return found(kind, m.group(2), "body", start, decl, decl_only=term != "{")
        return None
    if ctx.scope == "body" and term == "{":
        m = JS_FUNC.match(s)
        if m:
            return found("function", m.group(1), "body", start, decl)
        m = JS_CLASS.match(s)
        if m:
            return found("class", m.group(1), "members", start, decl)
    return None


# --- Go ---

GO_METHOD = re.compile(r"\s*func\s*\(\s*(?:[A-Za-z_]\w*\s+)?\*?\s*([A-Za-z_]\w*)(?:\[[^\]]*\])?\s*\)\s*([A-Za-z_]\w*)")
GO_FUNC = re.compile(r"\s*func\s+([A-Za-z_]\w*)")
GO_TYPE = re.compile(r"\s*type\s+([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*(?:=\s*)?(struct|interface)?\b")
GO_SPEC = re.compile(r"\s*([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*(?:=\s*)?(struct|interface)?\b")
GO_VALUE = re.compile(r"\s*(const|var)\s+([A-Za-z_]\w*)")


def go_type(name, what, start, decl):
    if what == "interface":
        return found("interface", name, "members", start, decl)
    return found(what or "type", name, "none", start, decl)


def go_classify(header: str, ctx: Ctx):
    s = header.replace("\n", " ")
    if ctx.scope == "group":
        if ctx.group == "type":
            m = GO_SPEC.match(s)
            return go_type(m.group(1), m.group(2), 0, 0) if m else None
        if ctx.group in ("const", "var"):
            m = re.match(r"\s*([A-Za-z_]\w*)", s)
            if m and m.group(1) != "_":
                return found("constant" if ctx.group == "const" else "variable", m.group(1))
        return None
    if ctx.scope == "members":  # an interface's method set
        m = re.match(r"\s*([A-Za-z_]\w*)\s*\(", s)
        return found("method", m.group(1), decl_only=True) if m else None
    if ctx.scope != "top":
        return None
    m = GO_METHOD.match(s)
    if m:
        return found("method", m.group(2), "body", parent=m.group(1), decl_only=ctx.term != "{")
    m = GO_FUNC.match(s)
    if m:
        return found("function", m.group(1), "body", decl_only=ctx.term != "{")
    m = GO_TYPE.match(s)
    if m:
        return go_type(m.group(1), m.group(2), 0, 0)
    m = GO_VALUE.match(s)
    if m and m.group(2) != "_":
        return found("constant" if m.group(1) == "const" else "variable", m.group(2))
    return None


def go_group(header: str, f: dict, lang: str):
    if f["scope"] != "top":
        return None
    m = re.fullmatch(r"\s*(type|const|var|import)\s*", header)
    return {"group": m.group(1)} if m else None


# --- Rust ---

RS_VIS = r"\s*(?:pub(?:\s*\([^)]*\))?\s+)?"
RS_FN = re.compile(RS_VIS + r"(?:default\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s*(?:\"[^\"]*\"\s*)?)?"
                   r"fn\s+([A-Za-z_]\w*)")
RS_ITEM = re.compile(RS_VIS + r"(?:unsafe\s+)?(struct|enum|trait|union|mod|type|const|static)\s+(?:mut\s+)?([A-Za-z_]\w*)")
RS_IMPL = re.compile(r"\s*(?:unsafe\s+)?impl\b\s*(.*?)\s*(?:\bwhere\b.*)?$", re.S)
RS_MACRO = re.compile(r"\s*macro_rules!\s*([A-Za-z_]\w*)")
RS_KINDS = {"const": "constant", "static": "static", "type": "type", "struct": "struct", "enum": "enum",
            "union": "union", "trait": "trait", "mod": "mod"}


def rust_classify(header: str, ctx: Ctx):
    start, decl = skip_decorators(header, "rust")
    s = header[decl:].replace("\n", " ")
    scope, term = ctx.scope, ctx.term
    in_impl = ctx.container is not None and ctx.container["kind"] in ("impl", "trait")
    m = RS_FN.match(s)
    if m:
        if scope == "body" and term != "{":
            return None
        return found("method" if in_impl else "function", m.group(1), "body", start, decl, decl_only=term != "{")
    if scope not in ("top", "members"):
        return None
    if term == "{" and re.match(r"\s*(?:unsafe\s+)?extern\s*$", s):
        return TRANSPARENT
    m = RS_IMPL.match(strip_generics(s))
    if m and term == "{":
        target = m.group(1)
        if " for " in f" {target} ":
            target = target.split(" for ", 1)[1]
        target = target.strip(" &").split("::")[-1].strip() or "impl"
        return found("impl", target, "members", start, decl)
    m = RS_ITEM.match(s)
    if m:
        kind = RS_KINDS[m.group(1)]
        scope = "members" if kind == "trait" else "top" if kind == "mod" else "none"
        return found(kind, m.group(2), scope, start, decl)
    m = RS_MACRO.match(s)
    if m:
        return found("macro", m.group(1), "none", start, decl)
    return None


# --- Java, Kotlin, C# and Swift ---

MODS = (r"(?:(?:public|private|protected|internal|static|final|abstract|sealed|open|data|inner|partial|"
        r"readonly|fileprivate|override|unsafe|virtual|async|extern|synchronized|native|default|strictfp|"
        r"transient|volatile|const|lateinit|suspend|inline|infix|operator|tailrec|external|actual|expect|"
        r"value|annotation|companion|enum|mutating|nonmutating|convenience|required|dynamic|lazy|weak|"
        r"indirect|new|ref|record|fun|nonisolated|isolated|non-sealed)\s+)*")
CLASSLIKE = re.compile(r"\s*" + MODS +
                       r"(class|interface|enum|struct|record|object|protocol|extension|namespace|actor|trait|@interface)"
                       r"\s+([A-Za-z_][\w.]*)")
COMPANION = re.compile(r"\s*(?:(?:private|internal|public|protected)\s+)?companion\s+object\b\s*([A-Za-z_]\w*)?")
FUN = re.compile(r"\s*" + MODS + r"(?:fun|func)\s+(?:[\w.?]+\.)?([A-Za-z_]\w*|`[^`]+`)")
SWIFT_INIT = re.compile(r"\s*" + MODS + r"(init|deinit|subscript)\s*[?!]?\s*[(<{]?")
SWIFT_PROP = re.compile(r"\s*" + MODS + r"(?:var|let)\s+([A-Za-z_]\w*)\s*:")
KOTLIN_CONST = re.compile(r"\s*(?:(?:private|internal|public|protected)\s+)?const\s+val\s+([A-Za-z_]\w*)")
KOTLIN_CTOR = re.compile(r"\s*" + MODS + r"constructor\s*\(")
JAVA_METHOD = re.compile(r"\s*" + MODS + r"([\w\[\]?.]+)\s+([A-Za-z_]\w*)\s*\(")
JAVA_CTOR = re.compile(r"\s*" + MODS + r"([A-Z]\w*)\s*\(")
CS_DTOR = re.compile(r"\s*~\s*([A-Za-z_]\w*)\s*\(")
CS_PROP = re.compile(r"\s*" + MODS + r"([\w\[\]?.]+)\s+([A-Za-z_]\w*)\s*(=>|$)")
NOT_TYPES = {"return", "new", "throw", "else", "case", "await", "yield", "goto", "using", "var", "val", "let",
             "import", "package", "event", "delegate"}
NOT_NAMES = {"if", "for", "while", "switch", "catch", "synchronized", "using", "lock", "foreach", "return", "when"}
CLASS_KINDS = {"class", "interface", "enum", "struct", "record", "object", "protocol", "extension", "actor", "trait",
               "annotation"}


def jvm_classify(header: str, ctx: Ctx):
    lang, scope, term = ctx.lang, ctx.scope, ctx.term
    start, decl = skip_decorators(header, lang)
    s = strip_generics(header[decl:].replace("\n", " "))
    if scope == "none" or scope == "group":
        return None
    in_class = scope == "members"
    if scope in ("top", "members") or (scope == "body" and term == "{"):
        m = CLASSLIKE.match(s)
        if m:
            kind = "annotation" if m.group(1) == "@interface" else m.group(1)
            name = m.group(2)
            if kind == "extension":
                name = name.split(".")[-1]
            if kind == "namespace":
                return found(kind, name, "top", start, decl) if term in ("{", ";") else None
            return found(kind, name, "members", start, decl)
        m = COMPANION.match(s) if lang == "kotlin" else None
        if m:
            return found("object", m.group(1) or "Companion", "members", start, decl)
    decl_only = term != "{" and "=>" not in s and not re.search(r"\)\s*(?::[^=]*)?=", s)
    if lang in ("kotlin", "swift"):
        m = FUN.match(s)
        if m and (scope in ("top", "members") or term == "{" or "=" in s):
            return found("method" if in_class else "function", m.group(1).strip("`"), "body", start, decl,
                         decl_only=decl_only)
        if lang == "kotlin":
            m = KOTLIN_CONST.match(s) if scope in ("top", "members") else None
            if m:
                return found("constant", m.group(1), "none", start, decl)
            m = KOTLIN_CTOR.match(s) if in_class else None
            if m:
                return found("constructor", "constructor", "body", start, decl)
        if lang == "swift" and in_class:
            m = SWIFT_INIT.match(s)
            if m and (m.group(1) != "subscript" or term == "{" or "(" in s):
                return found("method", m.group(1), "body", start, decl, decl_only=decl_only)
            m = SWIFT_PROP.match(s)
            if m and term == "{":
                return found("property", m.group(1), "none", start, decl)
        return None
    if not in_class:
        return None
    m = JAVA_CTOR.match(s)
    if m and m.group(1) == ctx.container_name:
        return found("constructor", m.group(1), "body", start, decl, decl_only=decl_only)
    m = CS_DTOR.match(s) if lang == "csharp" else None
    if m:
        return found("destructor", "~" + m.group(1), "body", start, decl)
    m = JAVA_METHOD.match(s)
    if m and m.group(1) not in NOT_TYPES and m.group(2) not in NOT_NAMES:
        return found("method", m.group(2), "body", start, decl, decl_only=decl_only)
    if lang == "csharp":
        m = CS_PROP.match(s)
        if m and m.group(1) not in NOT_TYPES and ((term == "{" and not m.group(3)) or (term == ";" and m.group(3))):
            return found("property", m.group(2), "none", start, decl)
    return None


def jvm_inline(header: str) -> bool:
    return ends_with(header, ("=", ",", "(", "->")) or bool(re.search(r"\breturn$", header.rstrip()))


# --- C and C++ ---

C_ACCESS = re.compile(r"\s*(?:public|protected|private|signals|slots|Q_SLOTS|Q_SIGNALS)(?:\s+slots)?\s*:(?!:)")
C_MACRO_LINE = re.compile(r"[ \t]*[A-Z_][A-Z0-9_]*[ \t]*(?:\([^()\n]*\))?[ \t]*;?[ \t]*\n")
C_TEMPLATE = re.compile(r"\s*template\s*<")
C_RECORD = re.compile(r"\s*(typedef\s+)?(struct|class|union|enum(?:\s+(?:class|struct))?)\b\s*"
                      r"(?:alignas\s*\([^)]*\)\s*|__attribute__\s*\(\(.*?\)\)\s*|__declspec\s*\([^)]*\)\s*|"
                      r"[A-Z_][A-Z0-9_]*\s+(?=[A-Za-z_]))*([A-Za-z_]\w*)?(.*)$", re.S)
C_OPERATOR = re.compile(r"\boperator\s*(\(\s*\)|\[\s*\]|new\b(?:\s*\[\s*\])?|delete\b(?:\s*\[\s*\])?|[^\w\s(]{1,3}|"
                        r"[A-Za-z_]\w*)\s*\(")
C_QUALIFIER = re.compile(r"((?:[A-Za-z_]\w*\s*(?:<[^()]*?>)?\s*::\s*)+)$")
C_NOT_NAMES = {"if", "for", "while", "switch", "return", "sizeof", "catch", "defined", "alignof", "decltype",
               "static_assert", "_Static_assert", "__attribute__", "typeof", "__typeof__", "case", "do", "else", "new",
               "delete", "throw", "alignas", "noexcept", "__declspec", "asm", "__asm__", "_Generic"}
C_TYPE_WORDS = {"int", "char", "void", "long", "short", "unsigned", "signed", "float", "double", "bool", "auto",
                "const", "volatile", "struct", "union", "enum", "class", "typename", "_Bool"}
C_STORAGE = {"static", "inline", "extern", "virtual", "explicit", "constexpr", "consteval", "friend", "__inline",
             "__inline__", "__forceinline", "register", "thread_local", "typedef"}


def c_prepass(code: str, lines: list):
    """Blank preprocessor lines out of the code (they don't end in ';') and make
    #define macros symbols."""
    rows = code.split("\n")
    syms = []
    i = 0
    while i < len(rows):
        if rows[i].lstrip().startswith("#"):
            j = i
            while j + 1 < len(rows) and rows[j].rstrip().endswith("\\"):
                j += 1
            m = re.match(r"\s*#\s*define\s+([A-Za-z_]\w*)", rows[i])
            if m:
                syms.append(symbol("macro", m.group(1), i + 1, j + 1, i + 1, 0, lines[i].strip().rstrip("\\"),
                                   doc=comment_above(lines, i + 1)))
            for k in range(i, j + 1):
                rows[k] = " " * len(rows[k])
            i = j + 1
            continue
        i += 1
    return "\n".join(rows), syms


def c_inline(header: str) -> bool:
    h = header.rstrip()
    if ends_with(h, ("=", ",", "(")) or re.search(r"\breturn$", h):
        return True
    # a brace initialiser in a constructor's member-initialiser list: Foo() : a{1} {
    return bool(re.search(r"\)\s*(?:const\s*)?(?:noexcept\s*)?:(?!:)", h) and re.search(r"[\w>]$", h))


def c_classify(header: str, ctx: Ctx):
    if ctx.scope not in ("top", "members"):
        return None
    term, lang = ctx.term, ctx.lang
    i = 0
    while True:  # access specifiers and macro-only lines (Q_OBJECT) before the declaration
        m = C_ACCESS.match(header, i) or C_MACRO_LINE.match(header, i)
        if not m or m.end() == i:
            break
        i = m.end()
    start, decl = skip_decorators(header[i:], lang)
    start, decl = start + i, decl + i
    while True:
        m = C_TEMPLATE.match(header, decl)
        if not m:
            break
        decl = balanced_end(header, m.end() - 1, "<", ">")
    while decl < len(header) and header[decl].isspace():
        decl += 1
    s = header[decl:].replace("\n", " ")
    m = re.match(r"\s*(?:inline\s+)?namespace\b\s*([\w:]*)\s*(=)?", s)
    if m:
        if m.group(2) or term != "{":
            return None
        return found("namespace", m.group(1).replace("::", "."), "top", start, decl) if m.group(1) else TRANSPARENT
    if re.match(r"\s*extern\s*$", s):
        return TRANSPARENT if term == "{" else None
    m = re.match(r"\s*using\s+([A-Za-z_]\w*)\s*=", s)
    if m:
        return found("type", m.group(1), "none", start, decl) if term == ";" else None
    if re.match(r"\s*(?:using|friend|static_assert|_Static_assert|return)\b", s):
        return None
    m = C_RECORD.match(s)
    if m and "(" not in m.group(4).split(":")[0] and "(" not in (m.group(4) if term == "{" else ""):
        typedef, kind, name, rest = m.group(1), m.group(2).split()[0], m.group(3), m.group(4)
        if term == "{":
            if name:
                return found(kind, name, "none" if kind == "enum" else "members", start, decl)
            return found(kind, "", "none", start, decl, typedef=bool(typedef)) if typedef else None
        if typedef:
            t = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)*$", rest.strip())
            return found("type", t.group(1), "none", start, decl) if t else None
        return None
    if s.lstrip().startswith("typedef"):
        if term != ";":
            return None
        t = re.search(r"\(\s*\*\s*([A-Za-z_]\w*)\s*\)", s) or re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)*$", s.strip())
        return found("type", t.group(1), "none", start, decl) if t else None
    op = C_OPERATOR.search(s)
    if op:
        name = "operator" + (" " + op.group(1) if op.group(1)[0].isalpha() else re.sub(r"\s+", "", op.group(1)))
        prefix, paren = s[:op.start()], op.end() - 1
    else:
        paren = s.find("(")
        if paren < 0:
            return None
        n = re.search(r"(~?\s*[A-Za-z_]\w*)\s*$", s[:paren])
        if not n:
            return None
        name = re.sub(r"\s+", "", n.group(1))
        prefix = s[:n.start()]
    q = C_QUALIFIER.search(prefix)
    qualifier = q.group(1) if q else ""
    before = prefix[:q.start()] if q else prefix
    if name.lstrip("~") in C_NOT_NAMES or name in C_TYPE_WORDS or "=" in before or "(" in before or ")" in before:
        return None
    words = re.findall(r"[A-Za-z_]\w*", before)
    has_type = any(w not in C_STORAGE for w in words) or "*" in before or "&" in before
    after = s[balanced_end(s, paren, "(", ")"):].strip()
    if term == ";" and after.startswith("(") or re.match(r"[A-Za-z_]\w*\s*\(", after or "-"):
        return None  # a call, a function pointer, or a macro wrapped around the declaration
    decl_only = term != "{"
    if qualifier:
        parts = [strip_generics(p).strip() for p in qualifier.split("::") if p.strip()]
        parent = ".".join(([ctx.container_qual] if ctx.container_qual else []) + parts)
        kind = "constructor" if name == parts[-1] else "destructor" if name.startswith("~") else "method"
        return found(kind, name, "body", start, decl, parent=parent, decl_only=decl_only)
    if ctx.scope == "members":
        if name == ctx.container_name:
            kind = "constructor"
        elif name.startswith("~"):
            kind = "destructor"
        elif has_type:
            kind = "method"
        else:
            return None
        return found(kind, name, "body", start, decl, decl_only=decl_only)
    if not has_type or name.startswith("~"):
        return None
    return found("function" if term == "{" else "prototype", name, "body", start, decl, decl_only=decl_only)


# --- PHP ---

PHP_CLASS = re.compile(r"\s*(?:(?:abstract|final|readonly)\s+)*(class|interface|trait|enum)\s+([A-Za-z_]\w*)")
PHP_FUNC = re.compile(r"\s*(?:(?:public|private|protected|static|abstract|final)\s+)*function\s+&?\s*([A-Za-z_]\w*)")
PHP_NS = re.compile(r"\s*namespace\s+([A-Za-z_][\w\\]*)\s*$")


def php_classify(header: str, ctx: Ctx):
    start, decl = skip_decorators(header, "php")
    s = header[decl:].replace("\n", " ")
    scope, term = ctx.scope, ctx.term
    if scope == "top":
        m = PHP_NS.match(s)
        if m:
            return found("namespace", m.group(1), "top", start, decl)
        m = PHP_CLASS.match(s)
        if m:
            return found(m.group(1), m.group(2), "members", start, decl)
    m = PHP_FUNC.match(s)
    if m and (scope != "body" or term == "{"):
        return found("method" if scope == "members" else "function", m.group(1), "body", start, decl,
                     decl_only=term != "{")
    return None


# --- shell ---

SH_FUNC = re.compile(r"\s*(?:function\s+([A-Za-z_][\w:.-]*)\s*(?:\(\s*\))?|([A-Za-z_][\w:.-]*)\s*\(\s*\))\s*$")
SH_PAREN_FUNC = re.compile(r"\s*(?:function\s+)?([A-Za-z_][\w:.-]*)\s*\(\s*\)\s*$")


def shell_classify(header: str, ctx: Ctx):
    if ctx.scope not in ("top", "body") or ctx.term != "{":
        return None
    m = SH_FUNC.match(header.replace("\n", " "))
    return found("function", m.group(1) or m.group(2), "body") if m else None


def shell_group(header: str, f: dict, lang: str):
    if f["scope"] not in ("top", "body"):
        return None
    m = SH_PAREN_FUNC.match(header.replace("\n", " "))
    return found("function", m.group(1), "body") if m else None


def shell_inline(header: str) -> bool:
    return header.endswith("$")


BRACE_RULES = {
    "javascript": Rules(js_classify, newline_ends=True, inline=js_inline),
    "typescript": Rules(js_classify, newline_ends=True, inline=js_inline),
    "go": Rules(go_classify, newline_ends=True, group=go_group),
    "rust": Rules(rust_classify, inline=lambda h: ends_with(h, ("=",))),
    "java": Rules(jvm_classify, inline=jvm_inline),
    "csharp": Rules(jvm_classify, inline=jvm_inline),
    "kotlin": Rules(jvm_classify, newline_ends=True, inline=jvm_inline),
    "swift": Rules(jvm_classify, newline_ends=True, inline=jvm_inline),
    "c": Rules(c_classify, inline=c_inline, prepass=c_prepass),
    "cpp": Rules(c_classify, inline=c_inline, prepass=c_prepass),
    "php": Rules(php_classify, inline=lambda h: ends_with(h, ("=", ",", "(", "=>"))),
    "shell": Rules(shell_classify, newline_ends=True, inline=shell_inline, group=shell_group),
}


def parse_brace_language(text, lines, lang):
    return scan_blocks(text, lines, lang, BRACE_RULES[lang]), leading_doc(lines), ""


# --- Ruby (end depth) -----------------------------------------------------------------------

RB_DEF = re.compile(r"^\s*(?:(?:private|protected|public|module_function|private_class_method)\s+)?"
                    r"def\s+(self\.)?([A-Za-z_]\w*[?!=]?|[^\s(;.]+)")
RB_ENDLESS = re.compile(r"^\s*(?:\w+\s+)?def\s+[^\s(;]+(?:\([^)]*\))?\s*=(?!=)")
RB_CLASS = re.compile(r"^\s*(class|module)\s+(<<\s*\w+|[A-Z][\w:]*)")
RB_OPEN = re.compile(r"^\s*(?:if|unless|while|until|case|begin|for)\b")
RB_ASSIGN_OPEN = re.compile(r"(?:[^=!<>]=|\|\|=)\s*(?:if|unless|case|begin|while|until)\b")
RB_DO = re.compile(r"(?<![\w.:])do\b(?![?!:])")
RB_END = re.compile(r"(?<![\w.:])end\b(?![?!:])")
SINGLETON = {"singleton": True}


def parse_ruby(text, lines, lang="ruby"):
    code_lines = blank_code(text, "ruby").split("\n")
    out, stack = [], []  # stack entries: a symbol, None (a block) or SINGLETON (class << self)
    for i, line in enumerate(lines):
        code = code_lines[i] if i < len(code_lines) else ""
        if not code.strip():
            continue
        syms = [s for s in stack if s and s is not SINGLETON]
        parent = syms[-1] if syms else None
        singleton = False
        for e in reversed(stack):  # a class << self between here and the enclosing class
            if e is SINGLETON:
                singleton = True
                break
            if e is not None:
                break
        m = RB_DEF.match(code)
        c = RB_CLASS.match(code)
        if m:
            in_class = parent is not None and parent["kind"] in ("class", "module")
            kind = "class method" if (m.group(1) or singleton) and in_class else \
                "method" if in_class else "function"
            name = m.group(2) if not (m.group(1) and not in_class) else "self." + m.group(2)
            sym = symbol(kind, name, i + 1, i + 1, i + 1, len(syms), line.strip(),
                         parent["qual"] if parent else "", comment_above(lines, i + 1))
            out.append(sym)
            if not RB_ENDLESS.match(code):
                stack.append(sym)
        elif c:
            if c.group(2).startswith("<<"):
                stack.append(SINGLETON)
            else:
                name = c.group(2)
                sym = symbol(c.group(1), name.split("::")[-1], i + 1, i + 1, i + 1, len(syms), line.strip(),
                             ".".join(([parent["qual"]] if parent else []) + name.split("::")[:-1]),
                             comment_above(lines, i + 1))
                out.append(sym)
                stack.append(sym)
        elif RB_OPEN.match(code):
            stack.append(None)  # while x do ... is one block, not two
        else:
            if RB_ASSIGN_OPEN.search(code):
                stack.append(None)
            for _ in RB_DO.finditer(code):
                stack.append(None)
        for _ in RB_END.finditer(code):
            if stack:
                entry = stack.pop()
                if entry and entry is not SINGLETON:
                    entry["end"] = i + 1
    for entry in stack:
        if entry and entry is not SINGLETON:
            entry["end"] = len(lines)
    return out, leading_doc(lines), ""


# --- SQL (statements) -----------------------------------------------------------------------

SQL_NAME = r"(?:[\w$]+|\"[^\"]+\"|`[^`]+`|\[[^\]]+\])"
SQL_CREATE = re.compile(
    r"create\s+(?:or\s+replace\s+|or\s+alter\s+)?(?:(?:temp|temporary|unlogged|materialized|unique|global|local|"
    r"recursive|definer\s*=\s*\S+|algorithm\s*=\s*\w+|sql\s+security\s+\w+)\s+)*"
    r"(table|view|function|procedure|index|trigger|type|schema|sequence|domain|extension|policy|aggregate|"
    r"rule|macro|stage|stream|task)\s+(?:if\s+not\s+exists\s+)?(?:concurrently\s+)?"
    r"(" + SQL_NAME + r"(?:\s*\.\s*" + SQL_NAME + r")*)", re.I)


def parse_sql(text, lines, lang="sql"):
    code = blank_code(text, "sql")
    lm = LineMap(code)
    out = []
    start = None
    for m in re.finditer(r";|[^\s;]+|\n", code):
        tok = m.group(0)
        if tok == "\n":
            continue
        if start is None and tok != ";":
            start = m.start()
        line_text = lines[lm.line(m.start()) - 1] if lm.line(m.start()) <= len(lines) else ""
        batch_end = tok.upper() == "GO" and line_text.strip().upper() == "GO"
        if tok == ";" or batch_end:
            if start is not None:
                c = SQL_CREATE.match(text, start)
                if c:
                    first = lm.line(start)
                    name = ".".join(p.strip().strip('"`[]') for p in re.split(r"\s*\.\s*", c.group(2)))
                    out.append(symbol(c.group(1).lower(), name, first, lm.line(m.start()), first, 0,
                                      lines[first - 1].strip(), doc=comment_above(lines, first)))
            start = None
    return out, leading_doc(lines), ""


# --- Python (ast) ---------------------------------------------------------------------------

def python_header(node, lines: list) -> str:
    start = node.lineno
    body_line = node.body[0].lineno if getattr(node, "body", None) else start
    parts = []
    for i in range(start, max(start, body_line - 1) + 1):
        if i > len(lines):
            break
        s = lines[i - 1].strip()
        parts.append(s)
        if re.sub(r"\s+#.*$", "", s).endswith(":"):
            break
    decorators = []
    for d in getattr(node, "decorator_list", []):
        if 0 < d.lineno <= len(lines):
            decorators.append(lines[d.lineno - 1].strip())
    return " ".join(decorators + parts)


def parse_python(text: str, lines: list):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError) as e:
        line = getattr(e, "lineno", None)
        note = f"parse error{f' at line {line}' if line else ''}: {getattr(e, 'msg', e)}; outline is approximate"
        return parse_python_fallback(lines), leading_doc(lines), note
    out = []

    def doc_of(node):
        try:
            doc = ast.get_docstring(node)
        except Exception:
            return ""
        return doc.strip().split("\n", 1)[0] if doc else ""

    def visit(body, parent, depth, in_class, assignments=True):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if isinstance(node, ast.ClassDef):
                    kind = "class"
                else:
                    kind = "method" if in_class else "function"
                    if isinstance(node, ast.AsyncFunctionDef):
                        kind = "async " + kind
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                end = getattr(node, "end_lineno", None) or node.lineno
                sym = symbol(kind, node.name, start, end, node.lineno, depth,
                             python_header(node, lines), parent, doc_of(node))
                out.append(sym)
                visit(node.body, sym["qual"], depth + 1, isinstance(node, ast.ClassDef))
            elif depth == 0 and assignments and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        kind = "constant" if re.fullmatch(r"_*[A-Z][A-Z0-9_]*", t.id) else "variable"
                        end = getattr(node, "end_lineno", None) or node.lineno
                        out.append(symbol(kind, t.id, node.lineno, end, node.lineno, depth,
                                          lines[node.lineno - 1].strip(), parent))
            else:
                # compound statements (if TYPE_CHECKING:, try:, with ...) can hold definitions;
                # module constants count only from if/try blocks, not loops or __main__ blocks
                keep = assignments and isinstance(node, (ast.If, ast.Try)) and not (
                    isinstance(node, ast.If) and any(isinstance(c, ast.Constant) and c.value == "__main__"
                                                     for c in ast.walk(node.test)))
                for field in ("body", "orelse", "finalbody"):
                    inner = getattr(node, field, None)
                    if isinstance(inner, list) and inner and isinstance(inner[0], ast.AST):
                        visit(inner, parent, depth, in_class, keep)
                for handler in getattr(node, "handlers", None) or []:
                    visit(handler.body, parent, depth, in_class, keep)

    visit(tree.body, "", 0, False)
    return out, doc_of(tree) or leading_doc(lines), ""


PY_DEF = re.compile(r"^(\s*)(?:async\s+)?(def|class)\s+([A-Za-z_]\w*)")


def parse_python_fallback(lines: list) -> list:
    """def and class lines by indentation, for a file ast can't parse."""
    out, stack = [], []  # stack of (indent, symbol)
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        while stack and indent <= stack[-1][0]:
            stack.pop()
        m = PY_DEF.match(line)
        if not m:
            continue
        parent = stack[-1][1] if stack else None
        kind = "class" if m.group(2) == "class" else ("method" if parent and parent["kind"] == "class" else "function")
        sym = symbol(kind, m.group(3), i + 1, i + 1, i + 1, len(stack), line.strip(),
                     parent["qual"] if parent else "")
        out.append(sym)
        stack.append((indent, sym))
    # end: the line before the next line indented at or above the symbol
    for k, sym in enumerate(out):
        indent = len(lines[sym["start"] - 1]) - len(lines[sym["start"] - 1].lstrip())
        end = len(lines)
        for j in range(sym["start"], len(lines)):
            s = lines[j]
            if s.strip() and len(s) - len(s.lstrip()) <= indent:
                end = j
                break
        while end > sym["start"] and not lines[end - 1].strip():
            end -= 1
        sym["end"] = end
    return out

# --- documents and data ---------------------------------------------------------------------

MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
MD_FENCE = re.compile(r"^\s*(```|~~~)")


def parse_markdown(text, lines, lang="markdown"):
    heads, fence = [], None
    for i, line in enumerate(lines):
        f = MD_FENCE.match(line)
        if f:
            fence = None if fence == f.group(1) else (fence or f.group(1))
            continue
        if fence:
            continue
        m = MD_HEADING.match(line)
        if m:
            if m.group(2):
                heads.append((len(m.group(1)), m.group(2), i))
            continue
        # a setext heading: a text line underlined with === (h1) or --- (h2)
        prev = lines[i - 1].strip() if i > 0 else ""
        if prev and not MD_HEADING.match(lines[i - 1]) and not prev.startswith(("-", "*", ">", "|", "<")):
            if re.fullmatch(r"=+\s*", line):
                heads.append((1, prev, i - 1))
            elif re.fullmatch(r"-{2,}\s*", line):
                heads.append((2, prev, i - 1))
    out = []
    for k, (level, title, i) in enumerate(heads):
        end = len(lines)
        for level2, _, j in heads[k + 1:]:
            if level2 <= level:
                end = j
                break
        while end > i + 1 and not lines[end - 1].strip():
            end -= 1
        out.append(symbol(f"h{level}", title, i + 1, end, i + 1, 0, lines[i].strip() if lines[i].strip().startswith("#") else f"{'#' * level} {title}"))
    nest(out, dotted=False)
    first = next((l.strip().lstrip("#").strip() for l in lines if l.strip()), "")
    return out, short(first, MAX_DOC), ""


def until_next(symbols: list, lines: list, last_end: int):
    """End each symbol at the line before the next one starts, minus trailing blank
    and comment lines."""
    for k, sym in enumerate(symbols):
        end = symbols[k + 1]["start"] - 1 if k + 1 < len(symbols) else last_end
        while end > sym["start"] and (not lines[end - 1].strip() or lines[end - 1].lstrip().startswith("#")):
            end -= 1
        sym["end"] = max(sym["start"], end)


YAML_KEY = re.compile(r"""^(?![-#\s])(["']?[^:#\s"'][^:#]*?["']?)\s*:(?:\s|$)""")


def parse_yaml(text, lines, lang="yaml"):
    out = []
    for i, line in enumerate(lines):
        if line.startswith(("---", "...")):
            continue
        m = YAML_KEY.match(line)
        if m:
            out.append(symbol("key", m.group(1).strip("\"'"), i + 1, i + 1, i + 1, 0, line.strip(),
                              doc=comment_above(lines, i + 1)))
    until_next(out, lines, len(lines))
    return out, leading_doc(lines), ""


TOML_TABLE = re.compile(r"^\s*(\[\[?)\s*([^\]]+?)\s*\]\]?")
TOML_KEY = re.compile(r"^([A-Za-z0-9_.-]+|\"[^\"]+\")\s*=")


def parse_toml(text, lines, lang="toml"):
    out, in_table = [], False
    for i, line in enumerate(lines):
        m = TOML_TABLE.match(line)
        if m:
            in_table = True
            out.append(symbol("array table" if m.group(1) == "[[" else "table", m.group(2), i + 1, i + 1,
                              i + 1, 0, line.strip(), doc=comment_above(lines, i + 1)))
            continue
        k = TOML_KEY.match(line)
        if k and not in_table:
            out.append(symbol("key", k.group(1).strip('"'), i + 1, i + 1, i + 1, 0, line.strip()))
    until_next(out, lines, len(lines))
    for sym in out:
        if sym["kind"] == "table" and "." in sym["name"]:
            sym["depth"] = sym["name"].count(".")
    return out, leading_doc(lines), ""


JSON_TOKEN = re.compile(r'"(?:\\.|[^"\\\n])*"|[{}\[\]:,\n]')


def parse_json(text, lines, lang="json"):
    out = []
    first = text.lstrip()[:1]
    if first != "{":
        return out, "", ""
    depth, line, prev_string, prev_line = 0, 1, None, 1
    closing_line = len(lines)
    for m in JSON_TOKEN.finditer(text):
        tok = m.group(0)
        if tok == "\n":
            line += 1
            continue
        if tok[0] == '"':
            prev_string, prev_line = tok, line
            continue
        if tok == ":" and depth == 1 and prev_string is not None:
            try:
                name = json.loads(prev_string)
            except ValueError:
                name = prev_string.strip('"')
            out.append(symbol("key", str(name), prev_line, prev_line, prev_line, 0,
                              lines[prev_line - 1].strip() if prev_line <= len(lines) else name))
        elif tok in "{[":
            depth += 1
        elif tok in "}]":
            depth -= 1
            if depth == 0:
                closing_line = line
                break
        prev_string = None
    for k, sym in enumerate(out):
        end = out[k + 1]["start"] - 1 if k + 1 < len(out) else closing_line - 1
        while end > sym["start"] and not lines[end - 1].strip():
            end -= 1
        sym["end"] = max(sym["start"], end)
    return out, "", ""


class LineMap:
    def __init__(self, text):
        self.starts = [0] + [m.end() for m in re.finditer("\n", text)]

    def line(self, offset):
        return bisect.bisect_right(self.starts, offset)


HTML_ID = re.compile(r"<([A-Za-z][\w-]*)\b[^>]*?\bid\s*=\s*[\"']([^\"']+)[\"'][^>]*>", re.S)
HTML_BLOCK = re.compile(r"<(script|style)\b([^>]*)>", re.I)
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


def parse_html(text, lines, lang="html"):
    lm, out, blocked = LineMap(text), [], []
    counts = {"script": 0, "style": 0}
    for m in HTML_BLOCK.finditer(text):
        tag = m.group(1).lower()
        close = re.compile(rf"</{tag}\s*>", re.I).search(text, m.end())
        stop = close.end() if close else len(text)
        blocked.append((m.start(), stop))
        counts[tag] += 1
        attrs = m.group(2)
        ident = re.search(r"\bid\s*=\s*[\"']([^\"']+)", attrs)
        src = re.search(r"\b(?:src|href)\s*=\s*[\"']([^\"']+)", attrs)
        name = ident.group(1) if ident else (os.path.basename(src.group(1)) if src else f"{tag}{counts[tag]}")
        out.append(symbol(tag, name, lm.line(m.start()), lm.line(max(m.start(), stop - 1)), lm.line(m.start()), 0,
                          short(m.group(0), MAX_SIG)))
    for m in HTML_ID.finditer(text):
        if any(a <= m.start() < b for a, b in blocked):
            continue
        if len(out) > 2000:
            break
        tag = m.group(1).lower()
        end_offset = m.end()
        if tag not in VOID and not m.group(0).endswith("/>"):
            depth = 1
            for t in re.finditer(rf"<(/?){re.escape(tag)}\b[^>]*>", text[m.end():m.end() + 500000], re.I):
                depth += -1 if t.group(1) else (0 if t.group(0).endswith("/>") else 1)
                if depth == 0:
                    end_offset = m.end() + t.end()
                    break
        out.append(symbol(f"<{tag}>", m.group(2), lm.line(m.start()), lm.line(max(m.start(), end_offset - 1)),
                          lm.line(m.start()), 0, short(m.group(0), MAX_SIG)))
    nest(out, dotted=False)
    title = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
    return out, short(title.group(1), MAX_DOC) if title else "", ""


CSS_GROUPS = ("@media", "@supports", "@layer", "@container", "@document")


def parse_css(text, lines, lang="css"):
    code = blank_code(text, "css")
    lm, out, stack = LineMap(text), [], []  # stack: symbol or None per open brace
    sel_start = None
    for i, ch in enumerate(code):
        if ch == "{":
            parent = stack[-1] if stack else None
            top = not stack or (parent and parent["name"].startswith(CSS_GROUPS) and len(stack) == 1)
            selector = " ".join(text[sel_start:i].split()) if sel_start is not None else ""
            if top and selector:
                line = lm.line(sel_start)
                sym = symbol("at-rule" if selector.startswith("@") else "rule", short(selector, 100), line, line,
                             line, len(stack), selector)
                out.append(sym)
                stack.append(sym)
            else:
                stack.append(None)
            sel_start = None
        elif ch == "}":
            if stack:
                entry = stack.pop()
                if entry:
                    entry["end"] = lm.line(i)
            sel_start = None
        elif ch == ";":
            if not stack and sel_start is not None:
                selector = " ".join(text[sel_start:i].split())
                if selector.startswith("@"):
                    line = lm.line(sel_start)
                    out.append(symbol("at-rule", short(selector, 100), line, line, line, 0, selector))
            sel_start = None
        elif not ch.isspace() and sel_start is None:
            sel_start = i
    return out, leading_doc(lines), ""


def parse_json_doc(text, lines, lang="json"):
    """Top-level keys, plus an OpenAPI/Swagger document's paths, operations and schemas."""
    symbols, doc, note = parse_json(text, lines, lang)
    keys = {s["name"] for s in symbols}
    if "paths" in keys and ("openapi" in keys or "swagger" in keys):
        try:
            data = json.loads(text)
        except ValueError:
            data = {}
        spans = json_spans(text)
        values = {}
        if isinstance(data, dict):
            for p, item in (data.get("paths") or {}).items():
                for method, op in (item.items() if isinstance(item, dict) else []):
                    if isinstance(op, dict):
                        values[("paths", p, method, "summary")] = op.get("summary") or ""
                        values[("paths", p, method, "operationId")] = op.get("operationId") or ""
        symbols += openapi_symbols(spans, values, lines)
        nest(symbols, dotted=False)
    return symbols, doc, note


def json_spans(text: str, max_depth: int = 4) -> dict:
    """{key path: [first line, last line]} for the object keys of a JSON text, up to
    max_depth keys deep (array elements count as a '[]' step)."""
    spans = {}
    stack = []  # [bracket, path, current key]
    line, prev, prev_line, last_line = 1, None, 1, 1

    def close_value(fr):
        if fr[0] == "{" and fr[2] is not None:
            path = fr[1] + (fr[2],)
            if path in spans and spans[path][1] < last_line:
                spans[path][1] = last_line

    for m in JSON_TOKEN.finditer(text):
        tok = m.group(0)
        if tok == "\n":
            line += 1
            continue
        if tok[0] == '"':
            prev, prev_line, last_line = tok, line, line
            continue
        if tok == ":":
            if stack and stack[-1][0] == "{" and prev is not None:
                try:
                    key = json.loads(prev)
                except ValueError:
                    key = prev.strip('"')
                stack[-1][2] = str(key)
                path = stack[-1][1] + (str(key),)
                if len(path) <= max_depth:
                    spans[path] = [prev_line, prev_line]
        elif tok in "{[":
            path = ()
            if stack:
                fr = stack[-1]
                path = fr[1] + ((fr[2],) if fr[0] == "{" else ("[]",))
            stack.append([tok, path, None])
        elif tok in "}]":
            if stack:
                fr = stack.pop()
                close_value(fr)
                if fr[1] in spans:
                    spans[fr[1]][1] = line
        elif tok == ",":
            if stack:
                close_value(stack[-1])
        prev, last_line = None, line
    return spans


HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")


def openapi_symbols(spans: dict, values: dict, lines: list) -> list:
    """Path items, operations (GET /pets) and schemas of an OpenAPI or Swagger document."""
    out = []
    for path, (a, b) in sorted(spans.items(), key=lambda kv: kv[1][0]):
        sig = lines[a - 1].strip() if a <= len(lines) else ""
        if len(path) == 2 and path[0] == "paths":
            out.append(symbol("path", path[1], a, b, a, 0, sig))
        elif len(path) == 3 and path[0] == "paths" and path[2].lower() in HTTP_METHODS:
            doc = values.get(path + ("summary",)) or values.get(path + ("operationId",)) or ""
            out.append(symbol("operation", f"{path[2].upper()} {path[1]}", a, b, a, 0, sig, doc=str(doc)))
        elif (len(path) == 3 and path[:2] == ("components", "schemas")) or (len(path) == 2 and path[0] == "definitions"):
            out.append(symbol("schema", path[-1], a, b, a, 0, sig))
    return out


YAML_ANY_KEY = re.compile(r"""^(\s*(?:-\s+)?)("(?:[^"\\]|\\.)*"|'[^']*'|[^\s#'"{\[\-][^#]*?|-[^\s#][^#]*?)\s*:(?:[ \t]+(.*?))?\s*$""")


def yaml_spans(lines: list, max_depth: int = 4):
    """({key path: [first line, last line]}, {key path: inline value}) from the block
    mappings of a YAML text, by indentation. Not a YAML parser: flow collections and
    multi-line plain scalars are not followed, block scalars (| and >) are skipped."""
    spans, values, stack = {}, {}, []  # stack: (indent, path, start)
    block, last = None, 0

    def close(indent):
        while stack and stack[-1][0] >= indent:
            _, path, start = stack.pop()
            if len(path) <= max_depth:
                spans[path] = [start, max(start, last)]

    for i, raw in enumerate(lines, 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        if block is not None:
            if indent > block:
                last = i
                continue
            block = None
        if indent == 0 and raw.startswith(("---", "...")):
            continue
        m = YAML_ANY_KEY.match(raw)
        if not m:
            last = i
            continue
        key_indent = len(m.group(1))
        close(key_indent)
        key = m.group(2).strip()
        if key[:1] in "\"'":
            key = key[1:-1]
        path = (stack[-1][1] if stack else ()) + (key,)
        value = m.group(3) or ""
        if re.fullmatch(r"[|>][-+0-9]*", value):
            block = key_indent
        elif value:
            values[path] = value.strip().strip("\"'")
        stack.append((key_indent, path, i))
        last = i
    close(-1)
    return spans, values


def parse_yaml_doc(text, lines, lang="yaml"):
    symbols, doc, note = parse_yaml(text, lines, lang)
    if re.search(r"^(openapi|swagger)\s*:", text[:5000], re.M) and re.search(r"^paths\s*:", text, re.M):
        spans, values = yaml_spans(lines)
        symbols += openapi_symbols(spans, values, lines)
        nest(symbols, dotted=False)
    return symbols, doc, note


def sections(heads: list, lines: list, dotted: bool = False) -> list:
    """Heading symbols from (level, title, first line index, heading line index): each
    runs until the next heading of the same or a higher level."""
    out = []
    for k, (level, title, first, at) in enumerate(heads):
        end = len(lines)
        for level2, _, first2, _ in heads[k + 1:]:
            if level2 <= level:
                end = first2
                break
        while end > first + 1 and not lines[end - 1].strip():
            end -= 1
        out.append(symbol(f"h{level}", title, first + 1, end, at + 1, 0, f"{'#' * level} {title}"))
    return nest(out, dotted=dotted)


RST_ADORN = re.compile(r"^([=\-`:'\"~^_*+#<>.])\1{2,}\s*$")


def parse_rst(text, lines, lang="rst"):
    """reStructuredText section titles: a title underlined (and maybe overlined) with a
    punctuation character; levels in the order the styles first appear."""
    heads, styles, i, n = [], [], 0, len(lines)
    while i < n:
        line = lines[i]
        style = None
        if RST_ADORN.match(line) and i + 2 < n and lines[i + 1].strip() and not RST_ADORN.match(lines[i + 1]) \
                and lines[i + 2].strip()[:1] == line.strip()[:1] and RST_ADORN.match(lines[i + 2]):
            style, title, first, at, i = (line.strip()[0], True), lines[i + 1].strip(), i, i + 1, i + 3
        elif line.strip() and not line[0].isspace() and not RST_ADORN.match(line) and i + 1 < n and \
                RST_ADORN.match(lines[i + 1]) and len(lines[i + 1].rstrip()) * 2 >= len(line.rstrip()):
            style, title, first, at, i = (lines[i + 1].strip()[0], False), line.strip(), i, i, i + 2
        else:
            i += 1
            continue
        if style not in styles:
            styles.append(style)
        heads.append((min(6, styles.index(style) + 1), title, first, at))
    first_doc = next((h[1] for h in heads), "")
    return sections(heads, lines), short(first_doc, MAX_DOC), ""


ADOC_HEAD = re.compile(r"^(={1,6}|#{1,6})\s+(\S.*?)\s*$")
ADOC_DELIM = re.compile(r"^(-{4,}|\.{4,}|_{4,}|\*{4,}|={4,}|\+{4,}|/{4,})\s*$")


def parse_asciidoc(text, lines, lang="asciidoc"):
    """AsciiDoc section titles (= Title, == Section, ...), skipping delimited blocks."""
    heads, block = [], None
    for i, line in enumerate(lines):
        d = ADOC_DELIM.match(line)
        if d:
            block = None if block == d.group(1) else (block or d.group(1))
            continue
        if block or line.startswith("//"):
            continue
        m = ADOC_HEAD.match(line)
        if m:
            heads.append((len(m.group(1)), re.sub(r"\s+=+$", "", m.group(2)), i, i))
    return sections(heads, lines), short(heads[0][1], MAX_DOC) if heads else "", ""


# --- views: files read through a text rendering (notebooks, Word documents) --------------

NOTEBOOK_CELL = re.compile(r"^# %% \[(\w+)\] cell (\d+)$")
VIEW_MAX_BYTES = 32 * 1024 * 1024


def render_notebook(data: bytes) -> str:
    """A Jupyter notebook as text: each cell's source under a '# %% [kind] cell N' line.
    Outputs are left out."""
    nb = json.loads(data.decode("utf-8", errors="replace"))
    cells = nb.get("cells")
    if cells is None:  # nbformat 3
        cells = [c for ws in nb.get("worksheets") or [] for c in ws.get("cells") or []]
    out = []
    for k, cell in enumerate(cells or [], 1):
        if not isinstance(cell, dict):
            continue
        src = cell.get("source", cell.get("input", ""))
        if isinstance(src, list):
            src = "".join(str(s) for s in src)
        body = str(src).replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if body and body[-1] == "":
            body.pop()
        out.append(f"# %% [{re.sub(r'[^A-Za-z]', '', str(cell.get('cell_type') or 'code'))}] cell {k}")
        out.extend(body)
        out.append("")
    return "\n".join(out)


def parse_notebook(text, lines, lang="notebook"):
    """Cells as symbols; a markdown cell's headings and a code cell's functions and
    classes nested under it."""
    marks = [(i, m.group(1), int(m.group(2))) for i, line in enumerate(lines) for m in [NOTEBOOK_CELL.match(line)] if m]
    out = []
    for k, (i, kind, number) in enumerate(marks):
        stop = marks[k + 1][0] if k + 1 < len(marks) else len(lines)
        body = lines[i + 1:stop]
        while body and not body[-1].strip():
            body.pop()
        first = next((b.strip() for b in body if b.strip()), "")
        cell = symbol(f"{kind} cell", f"cell {number}", i + 1, i + 1 + len(body), i + 1, 0,
                      short(first, 100) or "(empty)", doc=first.lstrip("#").strip() if kind == "markdown" else "")
        out.append(cell)
        if kind == "markdown":
            for j, b in enumerate(body):
                h = MD_HEADING.match(b)
                if h and h.group(2):
                    out.append(symbol(f"h{len(h.group(1))}", h.group(2), i + 2 + j, i + 2 + j, i + 2 + j, 0, b.strip()))
        elif kind == "code" and body:
            # IPython magics and shell escapes are not Python; blank them (line numbers kept)
            src = "\n".join("" if b.lstrip().startswith(("%", "!")) else b for b in body)
            syms, _, _ = parse_python(src, src.split("\n"))
            for s in syms:
                for key in ("start", "end", "line"):
                    s[key] += i + 1
                out.append(s)
    for s in out:
        if s["kind"].endswith("cell"):
            continue
        s["depth"] += 1
    return out, short(next((s["doc"] for s in out if s["doc"]), ""), MAX_DOC), ""


W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def render_docx(data: bytes) -> str:
    """A Word document's text, one line per paragraph: headings as Markdown headings (from
    the paragraph style or its outline level), list items as '- ', table rows as
    '| a | b |'. Only word/document.xml and word/styles.xml are read."""
    import xml.etree.ElementTree as ET
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        def part(name):
            try:
                with z.open(name) as f:
                    raw = f.read(DOCX_MAX_XML + 1)
            except KeyError:
                return None
            if len(raw) > DOCX_MAX_XML:
                raise ValueError(f"{name} is over {DOCX_MAX_XML // (1024 * 1024)} MB")
            if b"<!DOCTYPE" in raw[:4096] or b"<!ENTITY" in raw[:65536]:
                raise ValueError(f"{name} declares a DTD or entities; not read")
            return ET.fromstring(raw)
        document = part("word/document.xml")
        styles = part("word/styles.xml")
    if document is None:
        raise ValueError("no word/document.xml")
    levels = {}
    for st in (styles.iter(W_NS + "style") if styles is not None else []):
        sid = st.get(W_NS + "styleId") or ""
        name_el = st.find(W_NS + "name")
        name = (name_el.get(W_NS + "val") if name_el is not None else "") or sid
        m = re.match(r"(?i)heading\s*(\d)$", name) or re.match(r"(?i)heading(\d)$", sid)
        if m:
            levels[sid] = int(m.group(1))
        elif name.lower() == "title" or sid.lower() == "title":
            levels[sid] = 1
    out = []

    def text_of(el):
        parts = []
        for t in el.iter():
            if t.tag == W_NS + "t" and t.text:
                parts.append(t.text)
            elif t.tag == W_NS + "tab":
                parts.append("\t")
            elif t.tag in (W_NS + "br", W_NS + "cr"):
                parts.append(" ")
        return " ".join("".join(parts).split())

    body = document.find(W_NS + "body")
    for el in (list(body) if body is not None else []):
        if el.tag == W_NS + "p":
            text = text_of(el)
            ppr = el.find(W_NS + "pPr")
            level, listed = 0, False
            if ppr is not None:
                ps = ppr.find(W_NS + "pStyle")
                if ps is not None:
                    level = levels.get(ps.get(W_NS + "val") or "", 0)
                ol = ppr.find(W_NS + "outlineLvl")
                if not level and ol is not None and (ol.get(W_NS + "val") or "").isdigit():
                    level = int(ol.get(W_NS + "val")) + 1
                listed = ppr.find(W_NS + "numPr") is not None
            if not text:
                if out and out[-1]:
                    out.append("")
                continue
            if level:
                out.append(f"{'#' * min(level, 6)} {text}")
            elif listed:
                out.append(f"- {text}")
            elif text.startswith("#") or re.fullmatch(r"[=-]+", text):
                out.append(" " + text)  # not a heading in the view
            else:
                out.append(text)
        elif el.tag == W_NS + "tbl":
            for row in el.iter(W_NS + "tr"):
                cells = [text_of(c) for c in row.iter(W_NS + "tc")]
                out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def render_view(data: bytes, lang: str) -> str:
    return render_notebook(data) if lang == "notebook" else render_docx(data)


def parse_none(text, lines, lang="text"):
    return [], "", ""


PARSERS = {
    "python": lambda text, lines, lang: parse_python(text, lines),
    "javascript": parse_brace_language,
    "typescript": parse_brace_language,
    "go": parse_brace_language,
    "rust": parse_brace_language,
    "java": parse_brace_language,
    "kotlin": parse_brace_language,
    "csharp": parse_brace_language,
    "swift": parse_brace_language,
    "c": parse_brace_language,
    "cpp": parse_brace_language,
    "php": parse_brace_language,
    "shell": parse_brace_language,
    "ruby": parse_ruby,
    "sql": parse_sql,
    "markdown": parse_markdown,
    "docx": parse_markdown,
    "rst": parse_rst,
    "asciidoc": parse_asciidoc,
    "notebook": parse_notebook,
    "yaml": parse_yaml_doc,
    "toml": parse_toml,
    "json": parse_json_doc,
    "html": parse_html,
    "css": parse_css,
}
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def split_lines(text: str) -> list:
    """Lines the way a file is read (\\n, \\r\\n or \\r), so line numbers agree with code_read."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def parse_text(text: str, lang: str):
    """(symbols, leading doc line, note, lines) of a text in a language."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = split_lines(text)
    symbols, doc, note = PARSERS.get(lang, parse_none)(text, lines, lang)
    for s in symbols:
        s["end"] = max(s["start"], min(s["end"], len(lines) or 1))
    return symbols, doc, note, lines


def parse_source(text: str, path: str) -> list:
    """The symbols of a source text, its language taken from the file name."""
    return parse_text(text, language_of(path, text[:200].encode("utf-8", "replace")))[0]


_VIEWS = {}


def view_text(full: str, lang: str, st=None) -> str:
    """The text rendering of a notebook or Word document, cached by mtime and size."""
    st = st or os.stat(full)
    key = (full, st.st_mtime_ns, st.st_size)
    cached = _VIEWS.get(full)
    if cached and cached[0] == key:
        return cached[1]
    if st.st_size > VIEW_MAX_BYTES:
        raise ValueError(f"over {VIEW_MAX_BYTES // (1024 * 1024)} MB")
    with open(full, "rb") as f:
        text = render_view(f.read(), lang)
    if len(_VIEWS) > 64:
        _VIEWS.clear()
    _VIEWS[full] = (key, text)
    return text


def index_file(full: str, rel: str, st) -> dict:
    """The index entry for one file: language, symbols, line count, doc, a note and the
    identifiers it mentions."""
    lang = language_of(rel)
    entry = {"mtime": st.st_mtime_ns, "size": st.st_size, "language": lang,
             "symbols": [], "note": "", "lines": 0, "doc": "", "idents": ""}
    limit = VIEW_MAX_BYTES if lang in VIEW_LANGUAGES else MAX_FILE_BYTES
    if st.st_size > limit:
        entry["note"] = f"over {limit // (1024 * 1024)} MB: not parsed or searched"
        return entry
    if lang in VIEW_LANGUAGES:
        try:
            text = view_text(full, lang, st)
        except Exception as e:  # a broken notebook or document is noted, never fatal
            entry["note"] = short(f"unreadable {lang}: {type(e).__name__}: {e}", 200)
            return entry
    else:
        try:
            with open(full, "rb") as f:
                data = f.read(MAX_FILE_BYTES + 1)
        except OSError as e:
            entry["note"] = f"unreadable: {e.strerror or e}"
            return entry
        if b"\0" in data[:BINARY_SNIFF]:
            entry["language"], entry["note"] = "binary", "binary: not parsed or searched"
            return entry
        if lang == "text":
            entry["language"] = lang = language_of(rel, data[:200])
        text = data.decode("utf-8", errors="replace")
    try:
        symbols, doc, note, lines = parse_text(text, lang)
        entry["symbols"], entry["doc"], entry["note"] = symbols, short(doc or "", MAX_DOC), note
    except Exception as e:  # a parser must never take the index down
        lines = split_lines(text)
        entry["note"] = short(f"parse error: {type(e).__name__}: {e}", 200)
    entry["lines"] = len(lines)
    if lang in CODE_LANGUAGES:
        idents = set(IDENT.findall(text))
        if len(idents) <= MAX_IDENTS:
            entry["idents"] = " ".join(sorted(idents))
        else:
            entry["idents"] = "*"  # too many to keep: always scanned
    return entry


# --- file lists -----------------------------------------------------------------------------

def git(root: str, *args, timeout: int = GIT_TIMEOUT, text: bool = True):
    """A read-only git command in root: (returncode, stdout). Optional index locks and
    the file-system monitor are off, so nothing in the repository is written."""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    try:
        p = subprocess.run(["git", "-c", "core.fsmonitor=false", "-C", root, *args], capture_output=True,
                           text=text, timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError):
        return 1, "" if text else b""
    return p.returncode, p.stdout


def git_toplevel(path: str):
    code, out = git(path, "rev-parse", "--show-toplevel", timeout=10)
    top = out.strip()
    return os.path.realpath(top) if code == 0 and top else None


def gitignore_rules(root: str) -> list:
    rules = []
    try:
        with open(os.path.join(root, ".gitignore"), encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", "!")):
                    continue
                dir_only = line.endswith("/")
                line = line.strip("/")
                if line:
                    rules.append((line, dir_only, "/" in line))
    except OSError:
        pass
    return rules


def ignored(rel: str, is_dir: bool, rules: list) -> bool:
    parts = rel.split("/")
    for pattern, dir_only, anchored in rules:
        if dir_only and not is_dir:
            # a directory rule still hides files below that directory, which the walk prunes
            continue
        if anchored:
            if fnmatch.fnmatchcase(rel, pattern) or rel.startswith(pattern + "/"):
                return True
        elif fnmatch.fnmatchcase(parts[-1], pattern):
            return True
    return False


def vendored_folder(root: str) -> str:
    """"<folder>/" (relative to root) when this script was copied into the project it serves
    (`switchyard.sh clients vendor`: the folder holds VENDORED.txt), else "". A project's index
    leaves that folder out, so searches find the project's code, not this server's."""
    here = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
    top = os.path.realpath(root)
    if here == top or not inside(here, top) or not os.path.isfile(os.path.join(here, "VENDORED.txt")):
        return ""
    return os.path.relpath(here, top).replace(os.sep, "/") + "/"


def list_files(root: str):
    """(relative paths, how they were found, what was left out): git's view of the tree
    when root is in a work tree (tracked plus untracked, minus ignored), else a walk
    that skips the usual dependency and build directories and the root .gitignore's
    patterns. Secret-looking files are always left out (and listed as such)."""
    skipped = {"secret": [], "ignored": [], "dirs": []}
    own = vendored_folder(root)   # this server's own files, when it was copied into the project
    code, out = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", timeout=60, text=False)
    if code == 0:
        names = sorted(set(n for n in out.decode("utf-8", "surrogateescape").split("\0") if n))
        keep = []
        for n in names:
            if own and n.startswith(own):
                continue
            (skipped["secret"] if is_secret(n) else keep).append(n)
        if own:
            skipped["dirs"].append(own)
        return keep, "git ls-files", skipped
    rules = gitignore_rules(root)
    keep = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        rel_dir = "" if rel_dir == "." else rel_dir.replace(os.sep, "/")
        kept_dirs = []
        for d in sorted(dirnames):
            rel = f"{rel_dir}/{d}".lstrip("/")
            if d in SKIP_DIRS or d.endswith(".egg-info") or rel + "/" == own:
                skipped["dirs"].append(rel + "/")
            elif ignored(rel, True, rules):
                skipped["ignored"].append(rel + "/")
            else:
                kept_dirs.append(d)
        dirnames[:] = kept_dirs
        for name in sorted(filenames):
            rel = f"{rel_dir}/{name}".lstrip("/")
            if is_secret(rel):
                skipped["secret"].append(rel)
            elif ignored(rel, False, rules):
                skipped["ignored"].append(rel)
            else:
                keep.append(rel)
        if len(keep) > MAX_FILES:
            break
    return keep, "directory walk", skipped


def walk_package(base: str):
    """The files of an installed package directory (a dependency root): everything but
    nested node_modules, VCS directories and caches; build output is kept."""
    out = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in ("node_modules", ".git", "__pycache__", ".hg"))
        rel_dir = os.path.relpath(dirpath, base)
        rel_dir = "" if rel_dir == "." else rel_dir.replace(os.sep, "/")
        for name in sorted(filenames):
            if not name.endswith((".map", ".pyc")):
                out.append(f"{rel_dir}/{name}".lstrip("/"))
        if len(out) > MAX_FILES:
            break
    return out


# --- the index ------------------------------------------------------------------------------

def index_dir() -> str:
    return os.path.expanduser(os.environ.get("SWITCHYARD_CODE_INDEX_DIR") or "~/.switchyard/code-index")


def db_path_for(key: str) -> str:
    return os.path.join(index_dir(), hashlib.sha1(key.encode("utf-8")).hexdigest() + ".sqlite3")


FILE_COLUMNS = "path, mtime, size, language, symbols_json, note, lines, doc, idents"


class Index:
    """One root's files and symbols, kept in memory and mirrored to SQLite. A
    dependency root (dep) lists the files of one installed package instead of a tree."""

    def __init__(self, root: str, key: str = None, dep: dict = None):
        self.root = root
        self.key = key or root
        self.dep = dep
        self.path = db_path_for(self.key)
        self.files = {}
        self.mode = ""
        self.skipped = {"secret": [], "ignored": [], "dirs": []}
        self.last_refresh = None
        self.last_seconds = 0.0
        self.last_changed = set()
        self.truncated = 0
        self.generation = 0
        self.caches = {}
        self._df = None
        self._ignored = (0.0, [])
        self.db = None
        self.db_error = ""
        self._open()

    def _open(self):
        try:
            os.makedirs(os.path.dirname(self.path), mode=0o700, exist_ok=True)
            db = sqlite3.connect(self.path, timeout=10)
            try:
                db.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:
                pass
            db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
            if meta.get("schema") != SCHEMA_VERSION or meta.get("root") not in (None, self.key):
                db.execute("DROP TABLE IF EXISTS files")
            db.execute("CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, mtime INTEGER, size INTEGER, "
                       "language TEXT, symbols_json TEXT, note TEXT, lines INTEGER, doc TEXT, idents TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS embeddings (hash TEXT PRIMARY KEY, vec BLOB)")
            db.execute("INSERT OR REPLACE INTO meta VALUES ('schema', ?)", (SCHEMA_VERSION,))
            db.execute("INSERT OR REPLACE INTO meta VALUES ('root', ?)", (self.key,))
            if self.dep:
                db.execute("INSERT OR REPLACE INTO meta VALUES ('dependency', ?)", (json.dumps(self.dep_summary()),))
            db.commit()
            for path, mtime, size, language, symbols_json, note, lines, doc, idents in db.execute(
                    f"SELECT {FILE_COLUMNS} FROM files"):
                try:
                    symbols = json.loads(symbols_json or "[]")
                except ValueError:
                    continue
                self.files[path] = {"mtime": mtime, "size": size, "language": language, "symbols": symbols,
                                    "note": note or "", "lines": lines or 0, "doc": doc or "", "idents": idents or ""}
            self.db = db
        except (sqlite3.Error, OSError) as e:
            self.db, self.db_error = None, f"{type(e).__name__}: {e}"

    def dep_summary(self) -> dict:
        d = self.dep or {}
        return {k: d.get(k) for k in ("name", "version", "ecosystem", "root", "project")}

    def label(self) -> str:
        return f"dep:{self.dep['name']}" if self.dep else self.root

    def list(self):
        if self.dep:
            files = self.dep.get("files")
            rels = sorted(files) if files is not None else walk_package(self.root)
            return rels, f"{self.dep['ecosystem']} package files", {"secret": [], "ignored": [], "dirs": []}
        return list_files(self.root)

    def refresh(self, force: bool = False) -> dict:
        t0 = time.time()
        rels, self.mode, self.skipped = self.list()
        self.truncated = max(0, len(rels) - MAX_FILES)
        rels = rels[:MAX_FILES]
        seen, changed, edited = set(), [], set()
        for rel in rels:
            full = os.path.join(self.root, rel)
            try:
                st = os.lstat(full)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            seen.add(rel)
            old = self.files.get(rel)
            if force or old is None or old["mtime"] != st.st_mtime_ns or old["size"] != st.st_size:
                self._replace(rel, index_file(full, rel, st))
                changed.append(rel)
                if old is not None and not force:
                    edited.add(rel)  # changed on disk since it was indexed (not just new to the index)
        removed = [rel for rel in self.files if rel not in seen]
        for rel in removed:
            self._replace(rel, None)
        self.last_refresh = time.time()
        self.last_seconds = self.last_refresh - t0
        self.last_changed = edited
        self._save(changed, removed, force)
        return {"changed": len(changed), "removed": len(removed), "seconds": self.last_seconds}

    def _replace(self, rel: str, entry):
        old = self.files.get(rel)
        if self._df is not None:
            if old is not None:
                for name in self._idents(old):
                    self._df[name] = self._df.get(name, 1) - 1
            if entry is not None:
                for name in self._idents(entry):
                    self._df[name] = self._df.get(name, 0) + 1
        if entry is None:
            self.files.pop(rel, None)
        else:
            self.files[rel] = entry
        self.generation += 1

    def _save(self, changed, removed, force=False):
        if self.db is None:
            return
        try:
            with self.db:
                if changed or removed or force:
                    self.db.executemany("DELETE FROM files WHERE path = ?", [(r,) for r in removed])
                    self.db.executemany(
                        "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [(rel, e["mtime"], e["size"], e["language"], json.dumps(e["symbols"]), e["note"],
                          e["lines"], e["doc"], e.get("idents", "")) for rel, e in
                         ((r, self.files[r]) for r in changed if r in self.files)])
                for key, value in (("last_refresh", str(self.last_refresh)), ("files", str(len(self.files))),
                                   ("symbols", str(self.symbol_count()))):
                    self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))
        except sqlite3.Error as e:
            self.db_error = f"{type(e).__name__}: {e}"

    def update_file(self, rel: str) -> bool:
        """Re-index one file now (it changed since the last refresh); False if it is gone."""
        full = os.path.join(self.root, rel)
        try:
            st = os.lstat(full)
        except OSError:
            self._replace(rel, None)
            self._save([], [rel])
            return False
        self._replace(rel, index_file(full, rel, st))
        self._save([rel], [])
        return True

    def symbol_count(self) -> int:
        return sum(len(e["symbols"]) for e in self.files.values())

    def languages(self) -> dict:
        counts = {}
        for e in self.files.values():
            files, syms = counts.get(e["language"], (0, 0))
            counts[e["language"]] = (files + 1, syms + len(e["symbols"]))
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1][0], kv[0])))

    def searchable(self):
        for rel in sorted(self.files):
            e = self.files[rel]
            if e["language"] != "binary" and not e["note"].startswith(("over", "unreadable")):
                yield rel, e

    @staticmethod
    def _idents(entry) -> list:
        idents = entry.get("idents") or ""
        return [] if idents in ("", "*") else idents.split(" ")

    def df(self) -> dict:
        """{identifier: number of files that mention it}, built on first use and kept
        up to date as files change."""
        if self._df is None:
            df = {}
            for e in self.files.values():
                for name in self._idents(e):
                    df[name] = df.get(name, 0) + 1
            self._df = df
        return self._df

    def may_mention(self, rel: str, word: str) -> bool:
        """False only when the file certainly doesn't contain word as an identifier."""
        e = self.files.get(rel)
        if e is None or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", word):
            return True
        idents = e.get("idents") or ""
        if idents == "*" or not idents:
            return True
        names = e.get("_identset")
        if names is None:
            names = e["_identset"] = frozenset(idents.split(" "))
        return word in names

    def ignored_sample(self) -> list:
        """Paths git ignores under this root (directories collapsed), for 'not covered'
        notes; cached for a minute."""
        if self.dep:
            return []
        if self.mode != "git ls-files":
            return self.skipped.get("ignored", []) + self.skipped.get("dirs", [])
        at, sample = self._ignored
        if time.time() - at < 60:
            return sample
        code, out = git(self.root, "ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z",
                        timeout=10, text=False)
        names = sorted(n for n in out.decode("utf-8", "surrogateescape").split("\0") if n) if code == 0 else []
        dirs = [n for n in names if n.endswith("/")]
        names = [n for n in names if not any(n != d and n.startswith(d) for d in dirs)]
        sample = sorted(names, key=lambda n: (not n.endswith("/"), n))[:500]  # directories first
        self._ignored = (time.time(), sample)
        return sample


_INDEXES = {}
_DEFAULT_ROOT = None


def start_dir() -> str:
    return os.path.realpath(os.path.expanduser(os.environ.get("SWITCHYARD_CODE_ROOT") or os.getcwd()))


def default_root() -> str:
    global _DEFAULT_ROOT
    if _DEFAULT_ROOT is None:
        base = start_dir()
        _DEFAULT_ROOT = git_toplevel(base) or base
    return _DEFAULT_ROOT


def inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def allowed_dirs() -> list:
    return [os.path.realpath(os.path.expanduser(p))
            for p in (os.environ.get("SWITCHYARD_CODE_ALLOWED") or "").split(os.pathsep) if p.strip()]


def check_allowed(root: str):
    allowed = allowed_dirs()
    if allowed and not any(inside(root, a) for a in allowed):
        raise ToolError(f"{root} is outside the directories this server may read "
                        f"({', '.join(allowed)}).")


def check_not_too_wide(root: str):
    """Refuse a root that is the filesystem root, the home folder or a folder that holds the home
    folder: a server started by an app in the wrong folder would otherwise walk the whole disk.
    SWITCHYARD_CODE_ALLOW_WIDE=1 allows it."""
    if (os.environ.get("SWITCHYARD_CODE_ALLOW_WIDE") or "").strip().lower() in ("1", "true", "yes", "on"):
        return
    home = os.path.realpath(os.path.expanduser("~"))
    if root == os.path.sep:
        what = "the whole disk"
    elif root == home:
        what = "your home folder"
    elif home != os.path.sep and inside(home, root):
        what = "a folder that holds your home folder"
    else:
        return
    raise ToolError(f"{root} is {what}, which is too wide to index. Pass root with a project folder "
                    "(e.g. root=\"/path/to/repo\"), or set SWITCHYARD_CODE_ROOT for this server. "
                    "SWITCHYARD_CODE_ALLOW_WIDE=1 allows it.")


def resolve_root(args: dict, path_hint: str = "") -> str:
    raw = args.get("root")
    if raw:
        root = os.path.realpath(os.path.join(start_dir(), os.path.expanduser(str(raw))))
        if not os.path.isdir(root):
            raise ToolError(f"Root {raw!r} is not a directory.")
    else:
        root = default_root()
        if path_hint and os.path.isabs(os.path.expanduser(path_hint)):
            # an absolute path outside the default root: use the root that holds it
            target = os.path.realpath(os.path.expanduser(path_hint))
            if not inside(target, root):
                folder = target if os.path.isdir(target) else os.path.dirname(target)
                if os.path.isdir(folder):
                    root = git_toplevel(folder) or folder
    check_allowed(root)
    check_not_too_wide(root)
    return root


def get_index(args: dict, path_hint: str = "", refresh: bool = True):
    """The root's index (a project directory, or dep:<name> for an installed package),
    refreshed (files whose mtime or size changed are re-parsed) unless refresh is False."""
    raw = str(args.get("root") or "").strip()
    if raw.startswith("dep:"):
        idx = dependency_index(raw[4:].strip(), default_root())
    else:
        root = resolve_root(args, path_hint)
        idx = _INDEXES.get(root)
        if idx is None:
            idx = _INDEXES[root] = Index(root)
    if refresh:
        idx.refresh()
    return idx


def relative(idx, raw, must_exist: bool = True) -> str:
    """A path argument as a path relative to the root ('' for the root itself)."""
    raw = os.path.expanduser(str(raw or "").strip())
    if raw in ("", ".", "./"):
        return ""
    full = os.path.realpath(os.path.join(idx.root, raw))
    if not inside(full, idx.root):
        raise ToolError(f"{raw} is outside the root {idx.root}; pass root to read another directory.")
    if must_exist and not os.path.exists(full):
        raise ToolError(f"No such file or directory: {raw} (under {idx.label()}).")
    rel = os.path.relpath(full, idx.root).replace(os.sep, "/")
    rel = "" if rel == "." else rel
    if idx.dep and rel and rel not in idx.files and not any(r.startswith(rel + "/") for r in idx.files):
        raise ToolError(f"{rel} is not part of {idx.label()} (its files: code_tree root=dep:{idx.dep['name']}).")
    return rel


# --- dependency roots -----------------------------------------------------------------------

def normalize_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def project_site_packages(project: str) -> list:
    """site-packages directories of the project's own virtualenv (.venv or venv)."""
    out = []
    for env in (".venv", "venv", "env"):
        base = os.path.join(project, env)
        if not os.path.isfile(os.path.join(base, "pyvenv.cfg")):
            continue
        for pattern in ("lib/python*/site-packages", "lib64/python*/site-packages", "Lib/site-packages"):
            out += sorted(globlib.glob(os.path.join(base, pattern)))
    return [os.path.realpath(p) for p in out if os.path.isdir(p)]


def interpreter_venv() -> str:
    """The virtualenv this server runs in, if any (or $VIRTUAL_ENV)."""
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return os.path.realpath(sys.prefix)
    env = os.environ.get("VIRTUAL_ENV")
    return os.path.realpath(env) if env and os.path.isdir(env) else ""


def find_python_dist(name: str, sites: list):
    """(distribution, site-packages dir) for a distribution or top-level module name.
    Only metadata files are read; no package code is imported."""
    import importlib.metadata as md
    wanted = normalize_dist(name)
    for site in sites:
        try:
            dists = list(md.distributions(path=[site]))
        except Exception:
            continue
        for d in dists:
            try:
                if normalize_dist(d.metadata["Name"] or "") == wanted:
                    return d, site
            except Exception:
                continue
        for d in dists:  # an import name: yaml -> PyYAML
            try:
                top = (d.read_text("top_level.txt") or "").split()
            except Exception:
                top = []
            if name in top:
                return d, site
    return None, None


def python_dependency(name: str, project: str):
    sites = project_site_packages(project)
    where = "the project's virtualenv"
    if not sites:
        sites = [os.path.realpath(p) for p in sys.path if p.endswith(("site-packages", "dist-packages")) and os.path.isdir(p)]
        where = "this interpreter's site-packages"
    dist, site = find_python_dist(name, sites)
    if dist is None:
        return None, f"no Python distribution {name!r} in {where} ({', '.join(sites) or 'none found'})"
    files = []
    for p in dist.files or []:
        rel = str(p).replace(os.sep, "/")
        if rel.startswith("../") or "/__pycache__/" in f"/{rel}" or ".dist-info/" in rel or rel.endswith((".pyc", ".pyo")):
            continue
        full = os.path.realpath(os.path.join(site, rel))
        if inside(full, site) and os.path.isfile(full):
            files.append(os.path.relpath(full, site).replace(os.sep, "/"))
    if not files:  # no RECORD (an egg or editable install): the top-level packages' files
        try:
            top = (dist.read_text("top_level.txt") or "").split()
        except Exception:
            top = []
        for t in top or [name]:
            base = os.path.join(site, t)
            if os.path.isdir(base):
                files += [f"{t}/{r}" for r in walk_package(base)]
            elif os.path.isfile(base + ".py"):
                files.append(t + ".py")
    return {"name": dist.metadata["Name"] or name, "version": dist.version or "?", "ecosystem": "python",
            "root": site, "files": sorted(set(files))}, ""


def node_dependency(name: str, project: str):
    if not re.fullmatch(r"(@[\w.-]+/)?[\w.-]+", name):
        return None, f"{name!r} is not an npm package name"
    base = os.path.join(project, "node_modules", *name.split("/"))
    pkg = os.path.join(base, "package.json")
    if not os.path.isfile(pkg):
        return None, f"no node_modules/{name}/package.json under {project}"
    try:
        with open(pkg, encoding="utf-8", errors="replace") as f:
            version = str(json.load(f).get("version") or "?")
    except (OSError, ValueError):
        version = "?"
    return {"name": name, "version": version, "ecosystem": "npm", "root": os.path.realpath(base), "files": None}, ""


def resolve_dependency(name: str, project: str) -> dict:
    """Where an installed package lives. The rule: its files must be inside the project
    (its .venv or node_modules) or inside the virtualenv this server runs in, and inside
    SWITCHYARD_CODE_ALLOWED when that is set."""
    name = name.strip()
    if not name:
        raise ToolError("Missing dependency name, e.g. dependency=requests or dep:react.")
    kind = ""
    if ":" in name and name.split(":", 1)[0] in ("py", "python", "npm", "node"):
        kind, name = name.split(":", 1)
    tries = [node_dependency, python_dependency]
    if kind in ("py", "python"):
        tries = [python_dependency]
    elif kind in ("npm", "node"):
        tries = [node_dependency]
    reasons = []
    for resolve in tries:
        info, why = resolve(name, project)
        if info is None:
            reasons.append(why)
            continue
        root = info["root"]
        venv = interpreter_venv()
        if not (inside(root, project) or (venv and inside(root, venv))):
            raise ToolError(f"{info['ecosystem']} package {name!r} is at {root}, outside the project {project} "
                            f"and its virtualenv{f' ({venv})' if venv else ''}; dependency roots must live inside "
                            "the project (.venv, node_modules) or the virtualenv this server runs in.")
        check_allowed(root)
        info["project"] = project
        return info
    raise ToolError(f"Can't find dependency {name!r}: " + "; ".join(reasons) + ".")


def dependency_index(name: str, project: str) -> "Index":
    kind, bare = (name.split(":", 1) if ":" in name else ("", name))
    ecosystem = {"py": "python", "python": "python", "npm": "npm", "node": "npm"}.get(kind, "")
    for key, idx in _INDEXES.items():
        if idx.dep and idx.dep.get("project") == project and (not ecosystem or idx.dep["ecosystem"] == ecosystem) \
                and normalize_dist(idx.dep["name"]) == normalize_dist(bare):
            return idx
    info = resolve_dependency(name, project)
    key = f"dep:{info['ecosystem']}:{info['name']}@{project}"
    idx = _INDEXES.get(key)
    if idx is None:
        idx = _INDEXES[key] = Index(info["root"], key=key, dep=info)
    remember_dependency(project, info)
    return idx


def remember_dependency(project: str, info: dict):
    """Record an indexed dependency in the project's index, for code_tree."""
    proj = _INDEXES.get(project) or _INDEXES.setdefault(project, Index(project))
    if proj.db is None:
        return
    try:
        row = proj.db.execute("SELECT value FROM meta WHERE key = 'dependencies'").fetchone()
        deps = json.loads(row[0]) if row else {}
        deps[info["name"]] = {k: info[k] for k in ("version", "ecosystem", "root")}
        with proj.db:
            proj.db.execute("INSERT OR REPLACE INTO meta VALUES ('dependencies', ?)", (json.dumps(deps, sort_keys=True),))
    except (sqlite3.Error, ValueError):
        pass


def known_dependencies(idx) -> dict:
    if idx.db is None or idx.dep:
        return {}
    try:
        row = idx.db.execute("SELECT value FROM meta WHERE key = 'dependencies'").fetchone()
        return json.loads(row[0]) if row else {}
    except (sqlite3.Error, ValueError):
        return {}


# --- trust notes ----------------------------------------------------------------------------

def few(paths, n: int = 3) -> str:
    paths = list(paths)
    shown = ", ".join(paths[:n])
    return shown + (f" (+{len(paths) - n} more)" if len(paths) > n else "")


def freshness(idx, rels=()) -> str:
    """One short line on how current the index was for the files an answer came from:
    'index fresh', which of them were re-parsed on this call because they had changed on
    disk, or which changed while the answer was being made (re-indexed now; re-read them)."""
    rels = [r for r in dict.fromkeys(rels) if r in idx.files]
    edited = []
    for rel in rels[:300]:
        e = idx.files[rel]
        try:
            st = os.lstat(os.path.join(idx.root, rel))
            changed = st.st_mtime_ns != e["mtime"] or st.st_size != e["size"]
        except OSError:
            changed = True
        if changed:
            edited.append(rel)
            idx.update_file(rel)
    label = f"dep:{idx.dep['name']} {idx.dep['version']}, read-only; " if idx.dep else ""
    if edited:
        return f"[{label}edited since index → re-read now: {few(edited)}]"
    reparsed = [r for r in rels if r in idx.last_changed] if rels else []
    if reparsed:
        return f"[{label}index fresh; re-parsed on this call (changed on disk): {few(reparsed)}]"
    return f"[{label}index fresh]"


def in_scope(rel: str, prefix: str = "", glob: str = "") -> bool:
    return (not prefix or rel == prefix or rel.startswith(prefix + "/")) and (not glob or glob_matches(rel, glob))


def coverage(idx, prefix: str = "", glob: str = "", symbols: bool = False) -> str:
    """For an empty or 'not found' answer: what it could not see, so 'not found' is not
    read as 'does not exist'."""
    scoped = [(r, e) for r, e in idx.files.items() if in_scope(r, prefix, glob)]
    looked = [r for r, e in scoped if e["language"] != "binary" and not e["note"].startswith(("over", "unreadable"))]
    parts = []
    for label, test in (("binary", lambda e: e["language"] == "binary"),
                        ("too big", lambda e: e["note"].startswith("over")),
                        ("unreadable", lambda e: e["note"].startswith("unreadable")),
                        ("parse errors", lambda e: e["note"].startswith("parse error"))):
        hit = sorted(r for r, e in scoped if test(e))
        if hit:
            parts.append(f"{len(hit)} {label} ({few(hit)})")
    if symbols:
        bare = sorted(r for r, e in scoped if e["language"] not in PARSERS and e["language"] != "binary")
        if bare:
            parts.append(f"{len(bare)} without a symbol extractor ({few(bare)}; try code_search)")
    secret = sorted(r for r in idx.skipped.get("secret", []) if in_scope(r, prefix, glob))
    if secret:
        parts.append(f"{len(secret)} secret-looking ({few(secret)})")
    ignored_paths = [r for r in idx.ignored_sample() if in_scope(r.rstrip("/"), prefix) or (prefix and prefix.startswith(r))]
    if ignored_paths:
        parts.append(f"ignored: {few(ignored_paths, 4)}")
    if idx.truncated:
        parts.append(f"{idx.truncated:,} dropped by the {MAX_FILES:,}-file cap")
    where = idx.label() + (f"/{prefix}" if prefix else "") + (f" matching {glob}" if glob else "")
    what = "symbols of" if symbols else "searched"
    text = f"[{what} {len(looked):,} files under {where}"
    if parts:
        text += "; not covered: " + "; ".join(parts)
    if any(d.rstrip("/").split("/")[-1] in ("node_modules", ".venv", "venv", "site-packages") for d in ignored_paths +
           idx.skipped.get("dirs", [])):
        text += "; installed packages: code_index dependency=<name>"
    return text + "]"


def answer(text: str, idx, rels=(), empty: bool = False, prefix: str = "", glob: str = "",
           symbols: bool = False) -> str:
    out = [text, freshness(idx, rels)]
    if empty:
        out.append(coverage(idx, prefix, glob, symbols))
    return "\n".join(out)


# --- answer formats -------------------------------------------------------------------------

FORMATS = ("auto", "plain", "compact")


def output_format(args: dict, default: str) -> str:
    value = str(args.get("format") or os.environ.get("SWITCHYARD_CODE_FORMAT") or default).strip().lower()
    if value not in FORMATS:
        raise ToolError(f"format must be one of {', '.join(FORMATS)}, not {value!r}.")
    return value


def pick(fmt: str, plain: list, compact: list, limit: int = 0) -> str:
    """plain or compact text; auto takes compact only when it is at least 15% shorter."""
    p = finish(plain, limit)
    if fmt == "plain":
        return p
    c = finish(compact, limit)
    if fmt == "compact" or len(c) <= 0.85 * len(p):
        return c
    return p


def span(s) -> str:
    return f"{s['start']}-{s['end']}" if s["end"] != s["start"] else f"{s['start']}"


# --- tools ----------------------------------------------------------------------------------

def tool_index(args: dict) -> str:
    dep = str(args.get("dependency") or "").strip()
    if dep:
        idx = dependency_index(dep, resolve_root({k: v for k, v in args.items() if k == "root"}))
    else:
        idx = get_index(args, refresh=False)
    stats = idx.refresh(force=boolean(args, "refresh"))
    langs = ", ".join(f"{lang} {files}" for lang, (files, _) in idx.languages().items())
    notes = sum(1 for e in idx.files.values() if e["note"].startswith("parse error"))
    if idx.dep:
        d = idx.dep
        first = (f"Indexed dep:{d['name']} {d['version']} ({d['ecosystem']}, read-only) from {idx.root}: "
                 f"{len(idx.files):,} files ({stats['changed']:,} parsed, {stats['removed']:,} removed), "
                 f"{idx.symbol_count():,} symbols in {stats['seconds']:.2f} s. Use root=dep:{d['name']} with the other tools.")
    else:
        first = (f"Indexed {idx.root}: {len(idx.files):,} files ({stats['changed']:,} parsed, "
                 f"{stats['removed']:,} removed), {idx.symbol_count():,} symbols in {stats['seconds']:.2f} s "
                 f"(files from {idx.mode}).")
    out = [first, f"Languages: {langs or 'none'}."]
    if notes:
        out.append(f"{notes} file(s) had parse errors; their outlines are approximate (code_describe lists them).")
    if idx.truncated:
        out.append(f"Only the first {MAX_FILES:,} files are indexed ({idx.truncated:,} left out).")
    embedded = embed_refresh(idx)
    if embedded:
        out.append(embedded)
    out.append(f"Index: {idx.path}" + (f" (not saved: {idx.db_error})" if idx.db_error else ""))
    return "\n".join(out)


def tool_tree(args: dict) -> str:
    idx = get_index(args, str(args.get("path") or ""))
    sub = relative(idx, args.get("path"))
    depth = integer(args, "depth", 2, 1, 12)
    fmt = output_format(args, "plain")
    if sub in idx.files:
        e = idx.files[sub]
        return answer(f"{sub}  {human_size(e['size'])}  {e['language']}, {len(e['symbols'])} symbols", idx, [sub])
    prefix = sub + "/" if sub else ""
    tree = {"dirs": {}, "files": {}, "count": 0, "size": 0, "symbols": 0}
    for rel, e in idx.files.items():
        if not rel.startswith(prefix):
            continue
        parts = rel[len(prefix):].split("/")
        node = tree
        for part in parts[:-1]:
            node["count"] += 1
            node["size"] += e["size"]
            node["symbols"] += len(e["symbols"])
            node = node["dirs"].setdefault(part, {"dirs": {}, "files": {}, "count": 0, "size": 0, "symbols": 0})
        node["count"] += 1
        node["size"] += e["size"]
        node["symbols"] += len(e["symbols"])
        node["files"][parts[-1]] = e
    if not tree["count"]:
        return answer(f"No indexed files under {sub or idx.label()}.", idx, empty=True, prefix=sub)
    top = sub or idx.root
    plain = [f"{top}/  ({tree['count']:,} files, {tree['symbols']:,} symbols, {human_size(tree['size'])})"]
    compact = [f"{top}/ {tree['count']}f {tree['symbols']}s {compact_size(tree['size'])}",
               "# dir/ Nf=files Ns=symbols size | file size symbols (bin=binary, big=not parsed); indent=1 space/level"]
    shown = [0]

    def walk(node, level):
        for name in sorted(node["dirs"]):
            if shown[0] >= TREE_MAX_ENTRIES:
                return
            child = node["dirs"][name]
            shown[0] += 1
            plain.append(f"{'  ' * level}{name}/  ({child['count']:,} files, {child['symbols']:,} symbols, "
                         f"{human_size(child['size'])})")
            compact.append(f"{' ' * level}{name}/ {child['count']}f {child['symbols']}s {compact_size(child['size'])}")
            if level < depth:
                walk(child, level + 1)
        for name in sorted(node["files"]):
            if shown[0] >= TREE_MAX_ENTRIES:
                return
            e = node["files"][name]
            shown[0] += 1
            extra, mark = (f"  {len(e['symbols'])} sym", f" {len(e['symbols'])}") if e["symbols"] else ("", "")
            if e["language"] == "binary":
                extra, mark = "  binary", " bin"
            elif e["note"].startswith("over"):
                extra, mark = "  too big to parse", " big"
            plain.append(f"{'  ' * level}{name}  {human_size(e['size'])}{extra}")
            compact.append(f"{' ' * level}{name} {compact_size(e['size'])}{mark}")

    walk(tree, 1)
    if shown[0] >= TREE_MAX_ENTRIES:
        note = f"... [stopped at {TREE_MAX_ENTRIES} entries; pass a path or a smaller depth]"
        plain.append(note)
        compact.append(note)
    deps = known_dependencies(idx) if not sub else {}
    if deps:
        line = "Dependency roots: " + ", ".join(f"dep:{n} {d['version']} ({d['ecosystem']})" for n, d in sorted(deps.items()))
        plain.append(line)
        compact.append(line)
    return answer(pick(fmt, plain, compact), idx)


def compact_size(n) -> str:
    return human_size(n).replace(" ", "")


def file_entry(args: dict):
    raw = args.get("path")
    if not raw:
        raise ToolError("Missing argument path (a file, relative to the root).")
    idx = get_index(args, str(raw))
    rel = relative(idx, raw)
    if os.path.isdir(os.path.join(idx.root, rel)):
        raise ToolError(f"{rel or idx.root} is a directory; use code_tree for directories.")
    entry = idx.files.get(rel)
    if entry is None:
        if is_secret(rel):
            raise ToolError(f"{rel} looks like a secrets file; this server doesn't read those.")
        st = os.lstat(os.path.join(idx.root, rel))
        entry = index_file(os.path.join(idx.root, rel), rel, st)  # e.g. a git-ignored file
        entry["note"] = (entry["note"] + "; " if entry["note"] else "") + "not in the index (ignored)"
    return idx, rel, entry


def tool_outline(args: dict) -> str:
    idx, rel, e = file_entry(args)
    fmt = output_format(args, "plain")
    head = [f"{rel}  {e['language']}, {e['lines']:,} lines, {human_size(e['size'])}, {len(e['symbols'])} symbols"]
    if e["doc"]:
        head.append(f"  {e['doc']}")
    if e["note"]:
        head.append(f"  note: {e['note']}")
    if e["language"] in VIEW_LANGUAGES:
        head.append("  (lines are those of its text view: code_read and code_search show the same view)")
    if not e["symbols"]:
        head.append("  (no symbols; use code_read for its lines or code_search for text in it)")
    plain, compact = list(head), list(head)
    if e["symbols"]:
        compact.append("# lines kind: signature  -- doc; one space of indent per nesting level")
    for s in e["symbols"]:
        doc = f"  -- {s['doc']}" if s["doc"] else ""
        plain.append(f"{span(s):>11} {'  ' * s['depth']}{s['kind']}: {s['sig']}{doc}")
        compact.append(f"{' ' * s['depth']}{span(s)} {s['kind']}: {s['sig']}{doc}")
    if rel in idx.files:
        stands_in_for(idx, [rel])
    return answer(pick(fmt, plain, compact, OUTLINE_MAX_CHARS), idx, [rel] if rel in idx.files else [])


def numbered(lines, first: int, width: int = 6, limit: int = READ_LINE_CHARS) -> list:
    return [f"{first + i:>{width}}  {clip(line, limit)}" for i, line in enumerate(lines)]


def read_lines(full: str, start: int, end: int):
    """(lines start..end, total line count) streamed from the file."""
    picked, total = [], 0
    with open(full, encoding="utf-8", errors="replace", newline=None) as f:
        for total, line in enumerate(f, 1):
            if start <= total <= end:
                picked.append(line.rstrip("\r\n"))
    return picked, total


def file_lines(idx, rel: str) -> list:
    """All lines of a file as the index sees them (a notebook or document: its text view)."""
    full = os.path.join(idx.root, rel)
    lang = language_of(rel)
    if lang in VIEW_LANGUAGES:
        return split_lines(view_text(full, lang))
    with open(full, encoding="utf-8", errors="replace", newline=None) as f:
        return split_lines(f.read())


def lines_of(idx, rel: str, start: int, end: int):
    """(lines start..end, total) of a file, through its text view when it has one."""
    if language_of(rel) in VIEW_LANGUAGES:
        lines = file_lines(idx, rel)
        return lines[start - 1:end], len(lines)
    return read_lines(os.path.join(idx.root, rel), start, end)


def match_symbols(idx, name: str, path: str = "", kind: str = ""):
    """(matches, how): exact name or dotted name, then case-insensitive, then a
    dotted suffix (Class.method) or the last part of a dotted name."""
    prefix = path.rstrip("/")
    kind = kind.lower().strip()
    pool = []
    for rel, e in idx.files.items():
        if prefix and not (rel == prefix or rel.startswith(prefix + "/")):
            continue
        for s in e["symbols"]:
            if kind and not (s["kind"].lower() == kind or kind in s["kind"].lower()):
                continue
            pool.append((rel, s))
    name = name.replace("::", ".").replace("#", ".")
    lowered = name.lower()
    stages = [
        ("exact", lambda s: s["name"] == name or s["qual"] == name),
        ("case-insensitive", lambda s: s["name"].lower() == lowered or s["qual"].lower() == lowered),
        ("dotted suffix", lambda s: s["qual"].lower().endswith("." + lowered)),
    ]
    for how, test in stages:
        found_ = [(rel, s) for rel, s in pool if test(s)]
        if found_:
            found_.sort(key=lambda rs: (rs[0], rs[1]["start"]))
            return found_, how
    return [], ""


def tool_symbol(args: dict) -> str:
    name = str(args.get("name") or "").strip()
    if not name:
        raise ToolError("Missing argument name (a symbol name, or Class.method).")
    idx = get_index(args, str(args.get("path") or ""))
    path = relative(idx, args.get("path")) if args.get("path") else ""
    found_, how = match_symbols(idx, name, path, str(args.get("kind") or ""))
    if not found_:
        lowered = name.lower().rsplit(".", 1)[-1]
        similar = sorted({s["qual"] for e in idx.files.values() for s in e["symbols"] if lowered in s["name"].lower()})
        if not similar:
            similar = [q for q, _ in suggest_names(idx, lowered)]
        hint = f" Similar: {', '.join(similar[:12])}." if similar else ""
        raise ToolError(f"No symbol named {name!r}{f' in {path}' if path else ''}.{hint} "
                        "Try code_find for a ranked search, code_search for text, or code_outline for a file's "
                        "symbols.\n" + coverage(idx, path, symbols=True))
    if len(found_) > 1:
        out = [f"{len(found_)} symbols match {name!r} ({how}); pass path, kind or a dotted name "
               "(Class.method) to pick one:"]
        for rel, s in found_[:60]:
            out.append(f"  {rel}:{s['start']}-{s['end']}  {s['kind']}  {s['qual']}  -- {s['sig']}")
        if len(found_) > 60:
            out.append(f"  ... and {len(found_) - 60} more")
        return answer(finish(out), idx, [rel for rel, _ in found_[:60]])
    rel, s = found_[0]
    end = s["end"]
    capped = end - s["start"] + 1 > SYMBOL_MAX_LINES
    if capped:
        end = s["start"] + SYMBOL_MAX_LINES - 1
    lines, total = lines_of(idx, rel, s["start"], end)
    out = [f"{rel}:{s['start']}-{s['end']}  {s['kind']}  {s['qual']}"]
    if how != "exact":
        out[0] += f"  (matched {how})"
    out += numbered(lines, s["start"])
    if capped:
        out.append(f"... [{s['end'] - end:,} more lines; continue with code_read path={rel} start={end + 1}]")
    stands_in_for(idx, [rel])
    return answer(finish(out), idx, [rel])


def compile_query(args: dict, key: str = "query"):
    query = args.get(key)
    if not isinstance(query, str) or query == "":
        raise ToolError(f"Missing argument {key}.")
    flags = re.IGNORECASE if boolean(args, "ignore_case") else 0
    note = ""
    if boolean(args, "regex"):
        try:
            return re.compile(query, flags), note
        except re.error as e:
            note = f"(Not a valid regular expression: {e}. Searched for it as plain text.)"
    return re.compile(re.escape(query), flags), note


def glob_matches(rel: str, pattern: str) -> bool:
    patterns = [p.strip() for p in pattern.split(",") if p.strip()]
    for p in patterns:
        candidates = [p] + ([p[3:]] if p.startswith("**/") else [])
        for c in candidates:
            if fnmatch.fnmatchcase(rel, c) or ("/" not in c and fnmatch.fnmatchcase(rel.rsplit("/", 1)[-1], c)):
                return True
            if c.endswith("/**") and (rel.startswith(c[:-3] + "/")):
                return True
    return False


def scan(idx, regex, glob: str, max_results: int, context: int, max_chars: int, skip=None, word: str = ""):
    """Matching lines grouped by file, streamed, stopping at max_results matching lines
    or max_chars of output: (plain lines, compact lines, matches, files, truncated,
    capped, the files with hits). With word, files that can't mention it are skipped."""
    plain, compact, size = [], [], 0
    matches, files, truncated, capped = 0, 0, False, False
    hit_files = []
    for rel, e in idx.searchable():
        if glob and not glob_matches(rel, glob):
            continue
        if word and not idx.may_mention(rel, word):
            continue
        try:
            lines = file_lines(idx, rel)
        except (OSError, ValueError):
            continue
        hits = [i for i, line in enumerate(lines) if regex.search(line) and not (skip and skip(rel, i + 1))]
        if not hits:
            continue
        if matches + len(hits) > max_results:
            hits = hits[:max_results - matches]
            capped = True
        block, cblock = [f"{rel}"], [f"{rel}"]
        hit_set, previous = set(hits), -1
        for i in hits:
            lo, hi = max(0, i - context, previous + 1), min(len(lines), i + context + 1)
            if context and previous >= 0 and lo > previous + 1:
                block.append("    --")
                cblock.append("--")
            for j in range(lo, hi):
                mark = ":" if j in hit_set else "-"
                text = clip(lines[j].rstrip(), SEARCH_LINE_CHARS)
                block.append(f"{j + 1:>6}{mark} {text}")
                cblock.append(f"{j + 1}{mark}{text}")
            previous = max(previous, hi - 1)
        kept = 0
        for line in block:
            if size + len(line) + 1 > max_chars:
                truncated = True
                break
            plain.append(line)
            size += len(line) + 1
            kept += 1
        compact.extend(cblock[:kept])
        files += 1
        matches += len(hits)
        hit_files.append(rel)
        if truncated or capped:
            break
    return plain, compact, matches, files, truncated, capped, hit_files


SEARCH_LEGEND = "# file path, then line:text for a match and line-text for context; -- marks a gap"


def tool_search(args: dict) -> str:
    idx = get_index(args)
    regex, note = compile_query(args)
    glob = str(args.get("glob") or "")
    max_results = integer(args, "max_results", 50, 1, 1000)
    context = integer(args, "context", 0, 0, 10)
    max_chars = integer(args, "max_chars", SEARCH_MAX_CHARS, 500, MAX_OUTPUT_CHARS)
    fmt = output_format(args, "plain")
    plain, compact, matches, files, truncated, capped, hit_files = scan(idx, regex, glob, max_results, context, max_chars)
    kind = "regex" if boolean(args, "regex") and not note else "text"
    where = f" in files matching {glob}" if glob else ""
    head = [note] if note else []
    if not matches:
        head.append(f"No matches for {kind} {args['query']!r}{where} in {len(idx.files):,} indexed files under "
                    f"{idx.label()}.")
        return answer("\n".join(head), idx, empty=True, glob=glob)
    head.append(f"{matches:,} matching lines in {files:,} files for {kind} {args['query']!r}{where}:")
    tail = []
    if truncated:
        tail.append(f"... [output truncated at {max_chars:,} characters; narrow the query, add a glob, or raise max_chars]")
    elif capped:
        tail.append(f"... [stopped at max_results={max_results}; there may be more]")
    text = pick(fmt, head + plain + tail, head + [SEARCH_LEGEND] + compact + tail)
    return answer(text, idx, hit_files)


def tool_refs(args: dict) -> str:
    name = str(args.get("name") or "").strip()
    if not name:
        raise ToolError("Missing argument name (the symbol whose references to find).")
    idx = get_index(args)
    word = name.rsplit(".", 1)[-1] if re.fullmatch(r"[\w$.]+", name) else name
    regex = re.compile(r"(?<![\w$])" + re.escape(word) + r"(?![\w$])")
    definitions = set()
    for rel, e in idx.files.items():
        for s in e["symbols"]:
            if s["name"] == word:
                definitions.add((rel, s["line"]))
    max_results = integer(args, "max_results", 100, 1, 2000)
    max_chars = integer(args, "max_chars", SEARCH_MAX_CHARS, 500, MAX_OUTPUT_CHARS)
    fmt = output_format(args, "plain")
    glob = str(args.get("glob") or "")
    plain, compact, matches, files, truncated, capped, hit_files = scan(
        idx, regex, glob, max_results, 0, max_chars, skip=lambda rel, line: (rel, line) in definitions, word=word)
    if not matches:
        return answer(f"No references to {word!r} outside its definition{'s' if len(definitions) != 1 else ''} "
                      f"({len(definitions)} found) under {idx.label()}.", idx, empty=True, glob=glob)
    head = [f"{matches:,} lines reference {word!r} in {files:,} files (definition lines left out):"]
    tail = []
    if truncated:
        tail.append(f"... [output truncated at {max_chars:,} characters; add a glob or raise max_chars]")
    elif capped:
        tail.append(f"... [stopped at max_results={max_results}; there may be more]")
    return answer(pick(fmt, head + plain + tail, head + [SEARCH_LEGEND] + compact + tail), idx, hit_files)


def tool_read(args: dict) -> str:
    raw = args.get("path")
    if not raw:
        raise ToolError("Missing argument path.")
    idx = get_index(args, str(raw), refresh=False)
    rel = relative(idx, raw)
    full = os.path.join(idx.root, rel)
    if os.path.isdir(full):
        raise ToolError(f"{rel or idx.root} is a directory; use code_tree.")
    if is_secret(rel):
        raise ToolError(f"{rel} looks like a secrets file; this server doesn't read those.")
    view = language_of(rel) in VIEW_LANGUAGES
    if not view:
        try:
            with open(full, "rb") as f:
                if b"\0" in f.read(BINARY_SNIFF):
                    raise ToolError(f"{rel} is a binary file.")
        except OSError as e:
            raise ToolError(f"Can't read {rel}: {e.strerror or e}")
    max_lines = integer(args, "max_lines", 200, 1, READ_MAX_LINES)
    start = integer(args, "start", 1, 1)
    end = integer(args, "end", None)
    wanted_end = end if end is not None else start + max_lines - 1
    if wanted_end < start:
        raise ToolError(f"end {wanted_end} is before start {start}.")
    capped = wanted_end - start + 1 > max_lines
    stop = start + max_lines - 1 if capped else wanted_end
    try:
        lines, total = lines_of(idx, rel, start, stop)
    except (OSError, ValueError, zipfile.BadZipFile) as e:
        raise ToolError(f"Can't read {rel}: {e}")
    if total == 0:
        return f"{rel} is empty."
    if start > total:
        raise ToolError(f"start {start} is past the end: {rel} has {total:,} lines.")
    stop = min(stop, total)
    out = [f"{rel}  lines {start}-{stop} of {total:,}" + ("  (text view)" if view else "")] + numbered(lines, start)
    if capped:
        out.append(f"... [capped at {max_lines} lines; continue with start={stop + 1}]")
    elif stop < total:
        out.append(f"... [{total - stop:,} more lines; continue with start={stop + 1}]")
    return finish(out)


def tool_describe(args: dict) -> str:
    idx = get_index(args)
    langs = idx.languages()
    out = [f"Index directory: {index_dir()}",
           f"Root: {idx.label()}" + (f" ({idx.dep['ecosystem']} {idx.dep['version']}, files under {idx.root})"
                                     if idx.dep else ""),
           f"  index file: {idx.path}" + (f" (not saved: {idx.db_error})" if idx.db_error else ""),
           f"  files: {len(idx.files):,} (from {idx.mode}), symbols: {idx.symbol_count():,}",
           f"  last refresh: {when(idx.last_refresh)} ({idx.last_seconds:.2f} s)",
           "  languages: " + (", ".join(f"{lang} {f} files/{s} symbols" for lang, (f, s) in langs.items()) or "none")]
    skipped = idx.skipped.get("secret", [])
    if skipped:
        out.append(f"  secret-looking files left out: {few(sorted(skipped), 5)}")
    deps = known_dependencies(idx)
    if deps:
        out.append("  dependency roots: " + ", ".join(f"dep:{n} {d['version']} ({d['ecosystem']})" for n, d in sorted(deps.items())))
    out.append("  meaning-based ranking: " + embed_status(idx))
    notes = [(rel, e["note"]) for rel, e in sorted(idx.files.items()) if e["note"]]
    if notes:
        out.append(f"  notes ({len(notes)}):")
        for rel, note in notes[:20]:
            out.append(f"    {rel}: {note}")
        if len(notes) > 20:
            out.append(f"    ... and {len(notes) - 20} more")
    out.append("Roots known in this index directory:")
    try:
        names = sorted(n for n in os.listdir(index_dir()) if n.endswith(".sqlite3"))
    except OSError:
        names = []
    for name in names[:50]:
        try:
            db = sqlite3.connect(f"file:{os.path.join(index_dir(), name)}?mode=ro", uri=True, timeout=2)
            meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
            db.close()
        except sqlite3.Error:
            continue
        out.append(f"  {meta.get('root', '?')}  {int(meta.get('files') or 0):,} files, "
                   f"{int(meta.get('symbols') or 0):,} symbols, refreshed {when(meta.get('last_refresh'))}")
    return finish(out)


# --- ranked symbol search -------------------------------------------------------------------

WORD_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOPWORDS = {"a", "an", "the", "is", "are", "was", "be", "how", "what", "where", "when", "which", "who", "why",
             "does", "do", "did", "of", "to", "in", "on", "for", "and", "or", "it", "its", "by", "with", "from",
             "at", "as", "this", "that", "there", "get", "gets", "into", "out", "i", "we", "you"}
DATA_KINDS = {"key", "table", "array table", "rule", "at-rule", "path"}
DOC_KINDS = {"h1", "h2", "h3", "h4", "h5", "h6", "markdown cell", "code cell", "raw cell"}


def stem(word: str) -> str:
    """A light suffix stripper so excluded/excludes/exclude and windows/window meet."""
    for suffix, keep in (("ing", 5), ("ed", 4), ("es", 4), ("s", 3)):
        if word.endswith(suffix) and len(word) - len(suffix) >= keep - 1 and not word.endswith("ss"):
            word = word[:-len(suffix)]
            break
    return word[:-1] if len(word) > 4 and word.endswith("e") else word


def words_of(text: str) -> list:
    """Lowercase word parts of identifiers and prose: priceWindow_active -> price, window, active."""
    return [p.lower() for token in re.findall(r"[A-Za-z0-9]+", text) for p in WORD_PART.findall(token)]


def symbol_terms(s: dict):
    """(name terms, {term: weight}, length) of one symbol: name parts weigh 3, the
    container's 1.5, signature and doc words 1, all stemmed."""
    name_terms = [stem(w) for w in words_of(s["name"])]
    weights = {}
    for t in name_terms:
        weights[t] = weights.get(t, 0) + 3.0
    for t in words_of(s.get("parent") or ""):
        t = stem(t)
        weights[t] = weights.get(t, 0) + 1.5
    for t in words_of(s["sig"]) + words_of(s.get("doc") or ""):
        if t not in STOPWORDS:
            t = stem(t)
            weights[t] = weights.get(t, 0) + 1.0
    return name_terms, weights, sum(weights.values()) or 1.0


def find_table(idx, rel: str, e: dict) -> list:
    """The symbols' search terms, cached on the file's entry (rebuilt when it changes)."""
    table = e.get("_terms")
    if table is None:
        table = e["_terms"] = [symbol_terms(s) for s in e["symbols"]]
    return table


def osa_distance(a: str, b: str, limit: int) -> int:
    """Edit distance with adjacent transpositions, or limit + 1 once it is over limit."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


def vocabulary(idx) -> dict:
    """{name term: number of symbols with it}, cached per index generation."""
    cached = idx.caches.get("vocab")
    if cached and cached[0] == idx.generation:
        return cached[1]
    vocab = {}
    for rel, e in idx.files.items():
        for name_terms, _, _ in find_table(idx, rel, e):
            for t in set(name_terms):
                vocab[t] = vocab.get(t, 0) + 1
    idx.caches["vocab"] = (idx.generation, vocab)
    return vocab


def spelled(idx, term: str, like: str = "") -> str:
    """A name term as it is written in the code (the stem 'pric' -> 'price'): the
    spelling closest to like (what was typed), else the most common."""
    cached = idx.caches.get("spellings")
    if not cached or cached[0] != idx.generation:
        counts = {}
        for e in idx.files.values():
            for s in e["symbols"]:
                for w in words_of(s["name"]):
                    counts[w] = counts.get(w, 0) + 1
        forms = {}
        for w, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            forms.setdefault(stem(w), []).append(w)
        cached = idx.caches["spellings"] = (idx.generation, forms)
    forms = cached[1].get(term) or [term]
    if like:
        return min(forms, key=lambda w: (osa_distance(like, w, 4), forms.index(w)))
    return forms[0]


def nearest_term(idx, term: str):
    """The most common name term within a small edit distance of term (a typo)."""
    limit = 1 if len(term) <= 6 else 2
    best = None
    for cand, count in vocabulary(idx).items():
        if cand == term or abs(len(cand) - len(term)) > limit or (cand[0] != term[0] and cand[:2] != term[1::-1]):
            continue
        d = osa_distance(term, cand, limit)
        if d <= limit:
            key = (d, sorted(cand) != sorted(term), -count, cand)
            if best is None or key < best[0]:
                best = (key, cand)
    return best[1] if best else None


def suggest_names(idx, text: str):
    """Symbol names close to text (for a symbol that wasn't found)."""
    rows, note = rank_symbols(idx, text, limit=8)
    return [(r["sym"]["qual"], r["rel"]) for r in rows]


def query_terms(query: str) -> list:
    raw = words_of(query)
    kept = [w for w in raw if w not in STOPWORDS] or raw
    return list(dict.fromkeys(stem(w) for w in kept))


def path_filter(path: str):
    path = path.strip().strip("/")
    if not path:
        return lambda rel: True
    if any(c in path for c in "*?[,"):
        return lambda rel: glob_matches(rel, path)
    return lambda rel: rel == path or rel.startswith(path + "/")


def rank_symbols(idx, query: str, kind: str = "", language: str = "", path: str = "", limit: int = FIND_LIMIT,
                 typo: bool = True):
    """(rows, note): symbols ranked for a query. Each row is {score, rel, sym}. BM25 over
    each symbol's name parts, container, signature and doc words, plus boosts for an
    exact or prefix name match, every query word in the name, definitions over
    declarations and data keys, shallower paths, and names used from many files."""
    terms = query_terms(query)
    if not terms:
        return [], ""
    compact_query = "".join(words_of(query))
    kind = kind.lower().strip()
    in_path = path_filter(path)
    k1, b = 1.2, 0.6
    candidates, df, total, length_sum = [], dict.fromkeys(terms, 0), 0, 0.0
    for rel, e in idx.files.items():
        if not e["symbols"] or (language and e["language"] != language) or not in_path(rel):
            continue
        table = find_table(idx, rel, e)
        for s, (name_terms, weights, length) in zip(e["symbols"], table):
            if kind and kind not in s["kind"].lower():
                continue
            total += 1
            length_sum += length
            tf = None
            for t in terms:
                w = weights.get(t, 0.0)
                if not w and len(t) >= 3:
                    for nt in name_terms:
                        if nt.startswith(t):
                            w = 1.5
                            break
                if w:
                    if tf is None:
                        tf = {}
                    tf[t] = w
                    df[t] += 1
            if tf:
                candidates.append((rel, s, tf, length, name_terms))
    note = ""
    if typo:
        missing = [t for t in terms if df.get(t, 0) == 0 and len(t) >= 4]
        fixes = {t: nearest_term(idx, t) for t in missing}
        fixes = {t: f for t, f in fixes.items() if f}
        if fixes:
            fixed_query = " ".join(spelled(idx, fixes[stem(w)], w) if stem(w) in fixes else w for w in words_of(query))
            rows, _ = rank_symbols(idx, fixed_query, kind, language, path, limit, typo=False)
            if rows:
                said = {stem(w): w for w in words_of(query)}
                return rows, "did you mean: " + ", ".join(f"{spelled(idx, f, said.get(t, t))} (for {said.get(t, t)!r})"
                                                         for t, f in fixes.items())
    if not candidates:  # nothing shares a word: only meaning (when configured) can help
        rows = embed_blend(idx, query, [])
        rows.sort(key=lambda r: (-round(r["score"], 6), r["rel"], r["sym"]["start"]))
        return rows[:limit], note
    avg = length_sum / max(1, total)
    df_names = idx.df()
    rows = []
    for rel, s, tf, length, name_terms in candidates:
        score = 0.0
        for t, w in tf.items():
            idf = math.log(1 + (total - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * (w * (k1 + 1)) / (w + k1 * (1 - b + b * length / avg))
        name_l = re.sub(r"[^a-z0-9]", "", s["name"].lower())
        if name_l == compact_query:
            score += 8.0
        elif len(compact_query) >= 3 and name_l.startswith(compact_query):
            score += 3.0
        if len(tf) == len(terms) and len(terms) > 1:
            score *= 1.3
        if all(t in name_terms for t in terms):
            score += 2.0
        score *= len(tf) / len(terms) if len(terms) > 1 else 1.0
        if s.get("decl"):
            score *= 0.75
        elif s["kind"] in DATA_KINDS or s["kind"].startswith("<"):
            score *= 0.7
        elif s["kind"] in DOC_KINDS:
            score *= 0.85
        score /= 1 + 0.05 * rel.count("/")
        if TEST_PATH.search(rel):
            score *= 0.8
        uses = df_names.get(s["name"], 0) - 1
        if uses > 0:
            score *= 1 + 0.08 * math.log1p(uses)
        rows.append({"score": score, "rel": rel, "sym": s})
    rows = embed_blend(idx, query, rows)
    rows.sort(key=lambda r: (-round(r["score"], 6), r["rel"], r["sym"]["start"]))
    return rows[:limit], note


FIND_LEGEND = "# file, then its rows: score kind name lines | signature (higher score = better match)"


def tool_find(args: dict) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ToolError("Missing argument query (words to look for in symbol names, signatures and docs).")
    idx = get_index(args)
    limit = integer(args, "limit", FIND_LIMIT, 1, 100)
    fmt = output_format(args, "auto")
    kind, language, path = (str(args.get(k) or "") for k in ("kind", "language", "path"))
    rows, note = rank_symbols(idx, query, kind, language, path, limit)
    filters = ", ".join(f"{k}={v}" for k, v in (("kind", kind), ("language", language), ("path", path)) if v)
    filters = f" ({filters})" if filters else ""
    if not rows:
        return answer(f"No symbols match {query!r}{filters}. Try other words, code_search for text, or "
                      "code_tree to look around.", idx, empty=True, prefix=path if path and not any(
                          c in path for c in "*?[,") else "", glob=path if any(c in path for c in "*?[,") else "",
                      symbols=True)
    head = [f"{len(rows)} symbols for {query!r}{filters}, best first" + (f"; {note}" if note else "") + ":"]
    plain, compact, groups = list(head), head + [FIND_LEGEND], {}
    for r in rows:
        s = r["sym"]
        plain.append(f"{r['score']:>6.1f}  {s['kind']}  {s['qual']}  {r['rel']}:{s['start']}-{s['end']}  {s['sig']}")
        groups.setdefault(r["rel"], []).append(r)
    for rel, group in groups.items():  # files in order of their best row; the scores keep the ranking
        compact.append(rel)
        for r in group:
            s = r["sym"]
            compact.append(f"{r['score']:.1f} {s['kind']} {s['qual']} {s['start']}-{s['end']} | {s['sig']}")
    return answer(pick(fmt, plain, compact), idx, [r["rel"] for r in rows])


# --- one-call context -----------------------------------------------------------------------

IMPORT_LINE = re.compile(r"^\s*(?:import\s|from\s+\S+\s+import\s|#\s*include\s|use\s+[\w:{}]|using\s+[\w.]+\s*;|"
                         r"require(?:_relative)?[\s(]|(?:const|let|var)\s+.*=\s*require\(|package\s+[\w.]+|"
                         r"@import\s|extern\s+crate\s)")
CONTEXT_PARTS = ("bodies", "imports", "callers", "callees")


def import_lines(lines: list, limit: int = 6):
    """(numbered import lines from the head of a file, how many more there were)."""
    found_, in_block = [], False
    for i, line in enumerate(lines[:300]):
        s = line.strip()
        if in_block:
            found_.append(i)
            if s.startswith(")") or s.endswith(")") or s.endswith("}"):
                in_block = False
            continue
        if IMPORT_LINE.match(line):
            found_.append(i)
            if s.endswith(("(", "{")) or (s.startswith("import") and s.endswith("(")):
                in_block = True
    return found_[:limit], max(0, len(found_) - limit)


def defined_names(idx) -> dict:
    """{symbol name: (file, symbol)} of the first definition of each name; per generation."""
    cached = idx.caches.get("names")
    if cached and cached[0] == idx.generation:
        return cached[1]
    names = {}
    for rel in sorted(idx.files):
        for s in idx.files[rel]["symbols"]:
            if s["kind"] not in DATA_KINDS and s["kind"] not in DOC_KINDS and len(s["name"]) > 2:
                names.setdefault(s["name"], (rel, s))
    idx.caches["names"] = (idx.generation, names)
    return names


GENERIC_METHODS = {"get", "set", "put", "pop", "add", "run", "close", "open", "read", "write", "send", "update",
                   "items", "keys", "values", "append", "extend", "join", "split", "strip", "format", "copy", "clear",
                   "remove", "insert", "index", "count", "start", "stop", "next", "call", "apply", "map", "filter"}
CALLEE_SKIP = {"self", "this", "super", "print", "len", "str", "int", "dict", "list", "set", "tuple", "return", "None",
               "True", "False", "null", "true", "false", "new", "await", "async", "function", "def", "class"}


def callees_of(idx, rel: str, body: list, s: dict, limit: int = 10) -> list:
    """Names the body calls that are defined in the index (strings and comments left
    out; obj.method() only when that method is defined in the same file)."""
    names = defined_names(idx)
    lang = idx.files[rel]["language"] if rel in idx.files else "text"
    code = blank_code("\n".join(body), lang) if lang in C_LIKE or lang in HASH_COMMENTS else "\n".join(body)
    local = {x["name"] for x in idx.files.get(rel, {}).get("symbols", [])}
    seen = []
    for m in re.finditer(r"(\.?)\s*([A-Za-z_]\w*)\s*\(", code):
        n = m.group(2)
        if m.group(1) and (n not in local or n in GENERIC_METHODS):
            continue
        if n in names and n != s["name"] and n not in CALLEE_SKIP and n not in seen:
            seen.append(n)
    return seen[:limit]


def callers_of(idx, rel: str, s: dict, limit: int = 8, max_files: int = 12, cache=None) -> list:
    """Symbols whose lines call s by name, from the files that mention the name. cache
    ({path: lines}, with a "_reads" budget) is shared by one code_context call."""
    cache = {} if cache is None else cache
    name = s["name"]
    if not re.fullmatch(r"[A-Za-z_]\w*", name) or len(name) < 3:
        return []
    call = re.compile(r"(?<![\w$])" + re.escape(name) + r"\s*\(")
    files = [rel] + sorted(r for r, e in idx.files.items() if r != rel and e["symbols"] and idx.may_mention(r, name)
                           and e.get("idents"))
    out = []
    for other in files[:max_files]:
        lines = cache.get(other)
        if lines is None:
            if cache.get("_reads", 0) >= 60:
                break  # this call has read enough files; callers stay partial
            cache["_reads"] = cache.get("_reads", 0) + 1
            try:
                lines = cache[other] = file_lines(idx, other)
            except (OSError, ValueError):
                continue
        syms = idx.files[other]["symbols"]
        for i, line in enumerate(lines, 1):
            if not call.search(line) or (other == rel and s["start"] <= i <= s["end"]):
                continue
            inner = [x for x in syms if x["start"] <= i <= x["end"] and x["kind"] not in DOC_KINDS]
            where = min(inner, key=lambda x: x["end"] - x["start"])["qual"] if inner else "(top level)"
            label = f"{where} ({other}:{i})"
            if label not in out and not any(o.startswith(where + " (" + other) for o in out):
                out.append(label)
            if len(out) >= limit:
                return out
    return out


def render_symbol(idx, rel: str, s: dict, lines: list, include: set, room: int, short_only: bool = False,
                  cache=None) -> list:
    """One symbol's part of a context answer, fitted to room characters: the whole body
    when small, else its signature and head, an elided middle with the lines to read,
    and its tail."""
    out = [f"### {s['kind']} {s['qual']}  {rel}:{s['start']}-{s['end']}" + (f"  -- {s['doc']}" if s["doc"] else "")]
    body = lines[s["start"] - 1:s["end"]]
    width = len(str(s["end"]))

    def row(n):
        return f"{n:>{width}}  {clip(lines[n - 1], 300)}" if n - 1 < len(lines) else ""

    if "bodies" in include and not short_only:
        full = [row(n) for n in range(s["start"], s["end"] + 1)]
        if sum(len(x) + 1 for x in full) <= room * 0.9:
            out += full
        else:
            sig_end = s.get("line", s["start"])
            for n in range(sig_end, min(s["end"], sig_end + 8) + 1):
                sig_end = n
                if re.search(r"[{:]\s*(#.*|//.*)?$", lines[n - 1] if n - 1 < len(lines) else ""):
                    break
            head_n, tail_n = 12, 3
            while head_n >= 0:
                head = [row(n) for n in range(s["start"], min(s["end"], sig_end + head_n) + 1)]
                first_elided = s["start"] + len(head)
                tail_start = max(first_elided + 1, s["end"] - tail_n + 1)
                tail = [row(n) for n in range(tail_start, s["end"] + 1)] if s["end"] > first_elided else []
                mark = (f"{'':>{width}}  ... [lines {first_elided}-{tail_start - 1} elided; "
                        f"code_read path={rel} start={first_elided} end={tail_start - 1}]")
                block = head + ([mark] if tail_start - 1 >= first_elided else []) + tail
                if sum(len(x) + 1 for x in block) <= room * 0.9 or head_n == 0:
                    out += block
                    break
                head_n = head_n - 4 if head_n > 4 else head_n - 1
    else:
        out.append(f"{'':>{width}}  {s['sig']}  [body: code_symbol name={s['qual']} path={rel}]")
    if "callees" in include and not short_only:
        calls = callees_of(idx, rel, body, s)
        if calls:
            out.append("calls: " + ", ".join(calls))
    if "callers" in include and not short_only:
        callers = callers_of(idx, rel, s, cache=cache)
        if callers:
            out.append("called from: " + ", ".join(callers))
    return out


def tool_context(args: dict) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ToolError("Missing argument query (what you want to understand, in words or names).")
    budget = integer(args, "budget_tokens", CONTEXT_BUDGET, 200, 32000)
    raw_include = args.get("include")
    if isinstance(raw_include, str):
        raw_include = [p.strip() for p in raw_include.split(",") if p.strip()]
    include = set(raw_include or CONTEXT_PARTS)
    unknown = include - set(CONTEXT_PARTS)
    if unknown:
        raise ToolError(f"include takes {', '.join(CONTEXT_PARTS)}; not {', '.join(sorted(unknown))}.")
    idx = get_index(args)
    kind, language, path = (str(args.get(k) or "") for k in ("kind", "language", "path"))
    rows, note = rank_symbols(idx, query, kind, language, path, limit=40)
    if not rows:
        return answer(f"No symbols match {query!r}; try code_search for text.", idx, empty=True, symbols=True)
    budget_chars = budget * 4
    picked, left, files_shown, cache = [], [], {}, {}
    trailer_room = 200

    def text_of(parts, left_rows):
        head = f"Context for {query!r}: {len(parts)} of {len(rows)} ranked symbols in ~{budget:,} tokens" + \
               (f" ({note})" if note else "") + ":"
        body = [line for part in parts for line in part["lines"]]
        foot = []
        if left_rows:
            foot.append(f"Left out ({len(left_rows)}; fetch with code_symbol name=... path=...):")
            for r in left_rows[:8]:
                s = r["sym"]
                foot.append(f"  {s['qual']}  {r['rel']}:{s['start']}-{s['end']}")
            if len(left_rows) > 8:
                foot.append(f"  ... and {len(left_rows) - 8} more (code_find {query!r} limit=40)")
        return "\n".join([head] + body + foot)

    for r in rows:
        rel, s = r["rel"], r["sym"]
        spans_in_file = [p for p in picked if p["rel"] == rel]
        if any(p["sym"]["start"] <= s["start"] and s["end"] <= p["sym"]["end"] for p in spans_in_file):
            continue  # already inside a symbol that is shown
        if rel not in cache:
            try:
                cache[rel] = file_lines(idx, rel)
            except (OSError, ValueError):
                left.append(r)
                continue
        lines = cache[rel]
        used = len(text_of(picked, left + [r])) + trailer_room
        room = budget_chars - used
        part = []
        if rel not in files_shown:
            e = idx.files[rel]
            part.append(f"## {rel}  ({e['language']}, {e['lines']:,} lines)")
            if "imports" in include:
                imps, more = import_lines(lines)
                part += [f"{i + 1:>4}  {clip(lines[i], 200)}" for i in imps]
                if more:
                    part.append(f"      ... [{more} more import lines]")
        for short_only in (False, True):
            block = part + render_symbol(idx, rel, s, lines, include, max(0, room - sum(len(x) + 1 for x in part)),
                                         short_only, cache)
            if sum(len(x) + 1 for x in block) <= room:
                picked.append({"rel": rel, "sym": s, "lines": block})
                files_shown[rel] = True
                break
        else:
            left.append(r)
    while picked and len(text_of(picked, left)) + trailer_room > budget_chars:
        dropped = picked.pop()
        left.insert(0, {"rel": dropped["rel"], "sym": dropped["sym"]})
    used_files = sorted({p["rel"] for p in picked})
    result = answer(text_of(picked, left), idx, used_files)
    while len(result) > budget_chars and picked:
        dropped = picked.pop()
        left.insert(0, {"rel": dropped["rel"], "sym": dropped["sym"]})
        result = answer(text_of(picked, left), idx, sorted({p["rel"] for p in picked}))
    stands_in_for(idx, sorted({p["rel"] for p in picked}))
    return result[:budget_chars]


# --- changed symbols ------------------------------------------------------------------------

EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # git's empty tree
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def default_base(root: str):
    """(commit, how): the merge-base with the default branch, else HEAD."""
    code, out = git(root, "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD", timeout=10)
    candidates = ([out.strip()] if code == 0 and out.strip() else []) + ["origin/main", "origin/master", "main", "master"]
    head_code, head = git(root, "rev-parse", "--verify", "--quiet", "HEAD^{commit}", timeout=10)
    for ref in dict.fromkeys(candidates):
        code, sha = git(root, "rev-parse", "--verify", "--quiet", ref + "^{commit}", timeout=10)
        if code != 0:
            continue
        code, mb = git(root, "merge-base", "HEAD", ref, timeout=10)
        if code == 0 and mb.strip():
            return mb.strip(), f"merge-base with {ref}"
    if head_code == 0:
        return head.strip(), "HEAD"
    return None, "no commits yet"


def parse_diff(text: str) -> list:
    """Files of a `git diff -U0` text: {old, new, status, binary, hunks: [(os, oc, ns, nc)]}."""
    files, cur = [], None
    for line in text.split("\n"):
        if line.startswith("diff --git "):
            cur = {"old": None, "new": None, "status": "modified", "binary": False, "hunks": []}
            files.append(cur)
            m = re.match(r'diff --git "?a/(.*?)"? "?b/(.*?)"?$', line)
            if m:
                cur["old"], cur["new"] = m.group(1), m.group(2)
        elif cur is None:
            continue
        elif line.startswith("new file mode"):
            cur["status"] = "added"
        elif line.startswith("deleted file mode"):
            cur["status"] = "deleted"
        elif line.startswith("rename from "):
            cur["old"], cur["status"] = line[len("rename from "):], "renamed"
        elif line.startswith("rename to "):
            cur["new"] = line[len("rename to "):]
        elif line.startswith("Binary files "):
            cur["binary"] = True
        elif line.startswith("--- "):
            cur["old"] = None if line[4:] == "/dev/null" else line[4:].split("a/", 1)[-1] if line[4:].startswith("a/") else cur["old"]
        elif line.startswith("+++ "):
            cur["new"] = None if line[4:] == "/dev/null" else line[6:] if line[4:].startswith("b/") else cur["new"]
        else:
            m = HUNK.match(line)
            if m:
                os_, oc, ns, nc = (int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1))
                cur["hunks"].append((os_, oc, ns, nc))
    return files


def ranges(numbers, most: int = 0) -> str:
    """1,2,3,7 -> '1-3,7'; with most, at most that many ranges and a '+N more' count."""
    numbers = sorted(set(numbers))
    out, start, prev = [], None, None
    for n in numbers:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if prev != start else f"{start}")
    if most and len(out) > most:
        return ",".join(out[:most]) + f",+{len(out) - most} more"
    return ",".join(out)


def symbols_of_blob(root: str, spec: str, rel: str):
    """(symbols, lines) of a file as git has it (spec is '<commit>:path' or ':path')."""
    code, data = git(root, "show", spec, text=False)
    if code != 0 or not data:
        return [], []
    lang = language_of(rel)
    try:
        if lang in VIEW_LANGUAGES:
            text = render_view(data, lang)
        else:
            if b"\0" in data[:BINARY_SNIFF] or len(data) > MAX_FILE_BYTES:
                return [], []
            if lang == "text":
                lang = language_of(rel, data[:200])
            text = data.decode("utf-8", errors="replace")
        symbols, _, _, lines = parse_text(text, lang)
        return symbols, lines
    except Exception:
        return [], []


def innermost(symbols: list, line: int):
    best = None
    for s in symbols:
        if s["start"] <= line <= s["end"]:
            if best is None or (s["end"] - s["start"]) < (best["end"] - best["start"]):
                best = s
    return best


CHANGED_LEGEND = "# file (status), then: A added / M modified / D removed, kind, name, lines, [changed lines]"


def tool_changed(args: dict) -> str:
    idx = get_index(args)
    if idx.dep:
        raise ToolError("code_changed works on a git work tree, not on a dependency root.")
    root = git_toplevel(idx.root)
    if not root:
        raise ToolError(f"{idx.root} is not inside a git work tree.")
    staged = boolean(args, "staged")
    fmt = output_format(args, "auto")
    raw_paths = args.get("paths") or []
    if isinstance(raw_paths, str):
        raw_paths = [p.strip() for p in raw_paths.split(",") if p.strip()]
    paths = [relative(idx, p, must_exist=False) or "." for p in raw_paths]
    base = str(args.get("base") or "").strip()
    if base:
        if base.startswith("-"):
            raise ToolError(f"base {base!r} is not a commit.")
        code, sha = git(root, "rev-parse", "--verify", "--quiet", base + "^{commit}", timeout=10)
        if code != 0:
            raise ToolError(f"base {base!r} is not a commit in {root}.")
        sha, how = sha.strip(), base
    elif staged:
        code, sha = git(root, "rev-parse", "--verify", "--quiet", "HEAD^{commit}", timeout=10)
        sha, how = (sha.strip(), "HEAD") if code == 0 else (None, "no commits yet")
    else:
        sha, how = default_base(root)
    if sha is None:  # no commits yet: everything is new
        sha, how = EMPTY_TREE, "no commits yet"
        if staged:
            sha = EMPTY_TREE
    diff_args = ["diff", "--no-color", "--no-ext-diff", "--no-textconv", "-U0", "--find-renames"]
    diff_args += (["--cached", sha] if staged else [sha]) + ["--"] + (paths or [])
    code, out = git(root, *diff_args, timeout=GIT_TIMEOUT)
    if code != 0:
        raise ToolError(f"git diff failed in {root}.")
    files = parse_diff(out)
    if not staged:
        code, untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z", "--", *(paths or []))
        for rel in sorted(n for n in untracked.split("\0") if n and not is_secret(n)):
            files.append({"old": None, "new": rel, "status": "added", "binary": False, "hunks": [], "untracked": True})
    what = "staged changes" if staged else "working tree"
    desc = f"{what} vs {sha[:10]} ({how})"
    rows = []  # (file label, [(status, kind, qual, span, changed)])
    count = 0
    for f in files[:300]:
        new_rel, old_rel = f["new"], f["old"]
        rel = new_rel or old_rel
        if not rel or is_secret(rel):
            continue
        label = rel + (f" (renamed from {old_rel})" if f["status"] == "renamed" else
                       f" ({'untracked, ' if f.get('untracked') else ''}{f['status']})")
        if f["binary"]:
            rows.append((label, [("binary", "", "", "", "")]))
            continue
        new_syms, new_lines = [], []
        if new_rel and not staged:
            e = idx.files.get(new_rel)
            full = os.path.join(root, new_rel)
            try:
                if e is None:
                    e = index_file(full, new_rel, os.lstat(full))
                new_syms = e["symbols"]
                new_lines = file_lines(idx, new_rel) if os.path.isfile(full) and e["language"] != "binary" else []
            except (OSError, ValueError):
                pass
        elif new_rel:
            new_syms, new_lines = symbols_of_blob(root, ":" + new_rel, new_rel)
        old_syms = symbols_of_blob(root, f"{sha}:{old_rel}", old_rel)[0] if old_rel and f["status"] != "added" else []
        old_keys = {(s["qual"], s["kind"]) for s in old_syms}
        new_keys = {(s["qual"], s["kind"]) for s in new_syms}
        touched, loose = {}, []
        if f["status"] == "added":
            for s in new_syms:
                touched.setdefault(id(s), (s, []))
        for os_, oc, ns, nc in f["hunks"]:
            lines = range(ns, ns + nc) if nc else [max(1, ns)]
            for n in lines:
                s = innermost(new_syms, n)
                if s is None:
                    if nc and n - 1 < len(new_lines) and new_lines[n - 1].strip():  # blank lines don't count
                        loose.append(n)
                    continue
                touched.setdefault(id(s), (s, []))[1].append(n)
        items = []
        for s, changed in sorted(touched.values(), key=lambda t: t[0]["start"]):
            status = "modified" if (s["qual"], s["kind"]) in old_keys else "added"
            items.append((status, s["kind"], s["qual"], span(s), ranges(changed, 12) if status == "modified" else ""))
        for s in old_syms:
            if (s["qual"], s["kind"]) not in new_keys:
                items.append(("removed", s["kind"], s["qual"], "was " + span(s), ""))
        if loose:
            items.append(("added" if f["status"] == "added" else "modified", "module level", "", "",
                          ranges(loose, 12)))
        if not items and f["status"] == "deleted":
            items.append(("removed", "file", "", "", ""))
        count += sum(1 for it in items if it[2])
        if len(items) > 40:
            more = len(items) - 40
            items = items[:40] + [("more", "", "", "", f"{more} more in this file; code_changed paths={json.dumps([rel])} "
                                                         "or code_outline for all of them")]
        rows.append((label, items))
    if not rows:
        return answer(f"No changes in the {what} vs {sha[:10]} ({how})" + (f" under {', '.join(paths)}" if paths else "")
                      + ".", idx)
    head = [f"{count} changed symbols in {len(rows)} files, {desc}:"]
    plain, compact = list(head), head + [CHANGED_LEGEND]
    letters = {"added": "A", "modified": "M", "removed": "D", "binary": "B"}
    for label, items in rows:
        plain.append(label)
        compact.append(label)
        for status, kind, qual, where, changed in items:
            if status == "binary":
                plain.append("  binary file changed")
                compact.append("B binary file changed")
                continue
            if status == "more":
                plain.append(f"  ... {changed}")
                compact.append(f"... {changed}")
                continue
            extra = f"  (lines {changed})" if changed else ""
            plain.append("  " + "  ".join(x for x in (f"{status:<8}", kind, qual, where) if x) + extra)
            compact.append(" ".join(x for x in (letters[status], kind.replace(" ", "_"), qual, where.replace("was ", "was:"))
                                    if x) + (f" [{changed}]" if changed else ""))
    if len(files) > 300:
        plain.append(f"... [{len(files) - 300} more files; pass paths to narrow]")
        compact.append(plain[-1])
    used = [label.split(" (")[0] for label, _ in rows]
    return answer(pick(fmt, plain, compact), idx, [u for u in used if u in idx.files])


# --- tests and history ----------------------------------------------------------------------

TEST_PATH = re.compile(r"(^|/)(tests?|spec|__tests__)/|(^|/)test_|_test\.|\.(spec|test)\.")
TEST_FILES = "under tests/, test/, spec/ or __tests__/, or named test_*, *_test.*, *.test.* or *.spec.*"
TEST_NAME = re.compile(r"^test|test$", re.I)
TEST_BLOCK = re.compile(r"^[ \t]*(describe|context|it|test)((?:\.(?:only|skip|todo|concurrent))*)\s*\(\s*"
                        r"(['\"`])(.*?)\3", re.M)
TESTS_LEGEND = "# file, then its rows: call (calls it) or use, kind, name, lines, [lines that mention it]"
HISTORY_LEGEND = "# hash date +added -removed subject, newest first"


def test_blocks(text: str, code: str, lang: str) -> list:
    """describe/it/test('title', ...) calls of a JavaScript or TypeScript test file as
    symbols, each ending at the parenthesis that closes the call (code is the text with
    comments and strings blanked)."""
    if lang not in ("javascript", "typescript"):
        return []
    where = LineMap(text)
    lines = text.split("\n")
    out = []
    for m in TEST_BLOCK.finditer(text):
        at = text.index("(", m.end(2))
        if code[m.start(1):at + 1] != text[m.start(1):at + 1]:
            continue  # inside a comment or a string
        first = where.line(m.start(1))
        last = where.line(max(at, balanced_end(code, at, "(", ")") - 1))
        kind = "describe" if m.group(1) in ("describe", "context") else "test"
        out.append(symbol(kind, m.group(3) + m.group(4) + m.group(3), first, last, first, 0, lines[first - 1].strip()))
    return out


def enclosing_test(symbols: list, blocks: list, line: int):
    """The test a line belongs to: the innermost it/test block, else the innermost function
    or method named like a test, else the innermost function or method (a helper or a
    fixture), else the innermost describe block; None at module level."""
    def around(pool):
        return sorted((s for s in pool if s["start"] <= line <= s["end"]), key=lambda s: s["end"] - s["start"])
    tests = around(b for b in blocks if b["kind"] == "test")
    if tests:
        return tests[0]
    funcs = around(s for s in symbols if s["kind"].split()[-1] in ("function", "method"))
    named = [s for s in funcs if TEST_NAME.search(s["name"])]
    if named or funcs:
        return (named or funcs)[0]
    groups = around(b for b in blocks if b["kind"] == "describe")
    return groups[0] if groups else None


def tool_tests(args: dict) -> str:
    name = str(args.get("name") or "").strip()
    if not name:
        raise ToolError("Missing argument name (the symbol whose tests to find).")
    idx = get_index(args, str(args.get("path") or ""))
    prefix = relative(idx, args.get("path")) if args.get("path") else ""
    max_results = integer(args, "max_results", TESTS_LIMIT, 1, 500)
    fmt = output_format(args, "auto")
    word = name.rsplit(".", 1)[-1] if re.fullmatch(r"[\w$.]+", name) else name
    mention = re.compile(r"(?<![\w$])" + re.escape(word) + r"(?![\w$])")
    call = re.compile(r"(?<![\w$])" + re.escape(word) + r"\s*\(")
    definitions = {(rel, s["line"]) for rel, e in idx.files.items() for s in e["symbols"] if s["name"] == word}
    files = [(rel, e) for rel, e in idx.searchable() if TEST_PATH.search(rel) and in_scope(rel, prefix)]
    found_, loose = {}, 0
    for rel, e in files:
        if not idx.may_mention(rel, word):
            continue
        try:
            lines = file_lines(idx, rel)
        except (OSError, ValueError):
            continue
        hits = [i + 1 for i, line in enumerate(lines) if mention.search(line) and (rel, i + 1) not in definitions]
        if not hits:
            continue
        text = "\n".join(lines)
        code = blank_code(text, e["language"]) if e["language"] in CODE_LANGUAGES else text
        code_lines = code.split("\n")  # a call in a comment or a string is not a call
        blocks = test_blocks(text, code, e["language"])
        for n in hits:
            s = enclosing_test(e["symbols"], blocks, n)
            if s is None:
                loose += 1
                continue
            row = found_.setdefault((rel, s["start"], s["qual"]), {"rel": rel, "sym": s, "lines": [], "calls": 0})
            row["lines"].append(n)
            row["calls"] += 1 if call.search(code_lines[n - 1]) else 0
    where = idx.label() + (f"/{prefix}" if prefix else "")
    outside = (f"; {loose} line{'s' if loose != 1 else ''} outside any function left out (imports, module-level code)"
               if loose else "")
    if not found_:
        return answer(f"No test functions reference {word!r}: searched {len(files):,} test files under {where} "
                      f"(files {TEST_FILES}){outside}.", idx, empty=True, prefix=prefix)
    rows = sorted(found_.values(), key=lambda r: (-r["calls"], -len(r["lines"]), r["rel"], r["sym"]["start"]))
    shown = rows[:max_results]
    used = len({r["rel"] for r in rows})
    head = [f"{len(rows)} test functions reference {word!r} in {used:,} of {len(files):,} test files under {where}, "
            f"direct calls first{outside}:"]
    plain, compact, groups = list(head), head + [TESTS_LEGEND], {}
    for r in shown:
        s, how = r["sym"], "call" if r["calls"] else "use"
        plain.append(f"  {how:<4}  {s['kind']}  {s['qual']}  {r['rel']}:{span(s)}  (lines {ranges(r['lines'], 12)})")
        groups.setdefault(r["rel"], []).append(r)
    for rel, group in groups.items():  # files in order of their best row
        compact.append(rel)
        for r in group:
            s = r["sym"]
            compact.append(f"{'call' if r['calls'] else 'use'} {s['kind'].replace(' ', '_')} {s['qual']} {span(s)} "
                           f"[{ranges(r['lines'], 12)}]")
    if len(rows) > max_results:
        plain.append(f"... [stopped at max_results={max_results}; {len(rows) - max_results} more]")
        compact.append(plain[-1])
    return answer(pick(fmt, plain, compact), idx, [r["rel"] for r in shown])


def parse_log(text: str, numstat: bool) -> list:
    """Commits of a `git log --format=%x1e%h%x1f%ad%x1f%s` text, with lines added and
    removed counted from --numstat or, for `git log -L`, from the patch."""
    commits = []
    for chunk in text.split("\x1e")[1:]:
        head, _, body = chunk.partition("\n")
        parts = head.split("\x1f")
        if len(parts) < 3:
            continue
        c = {"hash": parts[0], "date": parts[1], "subject": parts[2], "added": 0, "removed": 0, "binary": False}
        in_hunk = False
        for line in body.split("\n"):
            if numstat:
                m = re.match(r"^(\d+|-)\t(\d+|-)\t", line)
                if m and m.group(1) == "-":
                    c["binary"] = True
                elif m:
                    c["added"] += int(m.group(1))
                    c["removed"] += int(m.group(2))
            elif line.startswith("diff --git "):
                in_hunk = False
            elif line.startswith("@@"):
                in_hunk = True
            elif in_hunk and line.startswith("+"):
                c["added"] += 1
            elif in_hunk and line.startswith("-"):
                c["removed"] += 1
        commits.append(c)
    return commits


def tool_history(args: dict) -> str:
    raw = args.get("path")
    if not raw:
        raise ToolError("Missing argument path (a file, relative to the root).")
    idx = get_index(args, str(raw))
    if idx.dep:
        raise ToolError("code_history works on a git work tree, not on a dependency root.")
    rel = relative(idx, raw, must_exist=False)
    full = os.path.join(idx.root, rel)
    if not rel or os.path.isdir(full):
        raise ToolError(f"{rel or idx.root} is a directory; code_history takes a file.")
    if is_secret(rel):
        raise ToolError(f"{rel} looks like a secrets file; this server doesn't read those.")
    if shutil.which("git") is None:
        return "No history: git is not installed (not found on PATH)."
    top = git_toplevel(idx.root)
    if not top:
        return f"No history: {idx.root} is not inside a git repository."
    limit = integer(args, "limit", HISTORY_LIMIT, 1, 100)
    fmt = output_format(args, "auto")
    path = os.path.relpath(full, top).replace(os.sep, "/")
    used = [rel] if rel in idx.files else []
    name = str(args.get("symbol") or "").strip()
    log = ["log", "--no-color", "--no-show-signature", f"-n{limit + 1}", "--date=short", "--format=%x1e%h%x1f%ad%x1f%s"]
    if name:
        if not os.path.isfile(full):
            raise ToolError(f"No such file: {rel} (under {idx.label()}).")
        found_, _ = match_symbols(idx, name, rel)
        exact = [(r, s) for r, s in found_ if s["qual"] == name]
        found_ = exact if len(exact) == 1 else found_
        if not found_:
            similar = sorted({s["qual"] for s in (idx.files.get(rel) or {}).get("symbols", [])
                              if name.lower().rsplit(".", 1)[-1] in s["name"].lower()})
            hint = f" Similar: {', '.join(similar[:12])}." if similar else ""
            raise ToolError(f"No symbol named {name!r} in {rel}.{hint} code_outline lists its symbols.")
        if len(found_) > 1:
            raise ToolError(f"{len(found_)} symbols in {rel} match {name!r}; pass a dotted name (Class.method): "
                            + ", ".join(f"{s['kind']} {s['qual']} {span(s)}" for _, s in found_[:20]) + ".")
        s = found_[0][1]
        # git log -L counts lines in HEAD's version of the file, so take the symbol's lines from there
        at_head = [h for h in symbols_of_blob(top, f"HEAD:{path}", rel)[0]
                   if h["qual"] == s["qual"] and h["kind"] == s["kind"]]
        if not at_head:
            return answer(f"No history for {s['kind']} {s['qual']}: it is not in {rel} at HEAD (not committed yet).",
                          idx, used)
        start, end = at_head[0]["start"], at_head[0]["end"]
        what = f"{s['kind']} {s['qual']} ({rel}:{start}-{end} at HEAD)"
        log += ["--no-ext-diff", "--no-textconv", f"-L{start},{end}:{path}"]
    else:
        what = rel
        log += ["--numstat", "--follow", "--", path]
    code, out = git(top, *log)
    if code != 0:
        code, _ = git(top, "rev-parse", "--verify", "--quiet", "HEAD^{commit}", timeout=10)
        if code != 0:
            return answer(f"No history: {top} has no commits yet.", idx, used)
        raise ToolError(f"git log failed in {top}.")
    commits = parse_log(out, numstat=not name)
    if not commits:
        return answer(f"No commits touch {what}.", idx, used)
    more = len(commits) > limit
    commits = commits[:limit]
    head = [f"{len(commits)} commit{'s' if len(commits) != 1 else ''} touched {what}, newest first:"]
    plain, compact = list(head), head + [HISTORY_LEGEND]
    for c in commits:
        change = "binary" if c["binary"] else f"+{c['added']} -{c['removed']}"
        subject = short(c["subject"], 120)
        plain.append(f"  {c['hash']}  {c['date']}  {change:<11}  {subject}")
        compact.append(f"{c['hash']} {c['date']} {change} {subject}")
    if more:
        plain.append(f"... [stopped at limit={limit}; there are older commits]")
        compact.append(plain[-1])
    return answer(pick(fmt, plain, compact), idx, used)


# --- optional meaning-based ranking (embeddings through the gateway) ----------------------

EMBED_BATCH = 64
EMBED_TIMEOUT = 20      # seconds per request
EMBED_MAX_CARD = 300    # characters of one symbol card
EMBED_MAX_SYMBOLS = 100000


def embed_config():
    """(url, key, model, dims) when SWITCHYARD_CODE_EMBED=1 and the gateway is configured, else None."""
    if os.environ.get("SWITCHYARD_CODE_EMBED", "").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    url, key = gateway_settings()
    if not url or not key:
        return None
    model = (os.environ.get("SWITCHYARD_CODE_EMBED_MODEL") or "Qwen3-Embedding").strip()
    try:
        dims = max(16, min(4096, int(os.environ.get("SWITCHYARD_CODE_EMBED_DIMS") or 256)))
    except ValueError:
        dims = 256
    return url, key, model, dims


def symbol_card(s: dict) -> str:
    """What is sent for one symbol: kind, qualified name, signature line and first doc
    line. Never the body."""
    return f"{s['kind']} {s['qual']}\n{s['sig']}\n{s.get('doc') or ''}".strip()[:EMBED_MAX_CARD]


def card_hash(model: str, card: str) -> str:
    return hashlib.sha1(f"{model}\0{card}".encode("utf-8")).hexdigest()


def embed_request(texts: list, cfg) -> list:
    """Unit vectors (array('f'), cut to dims) for texts from POST /v1/embeddings."""
    url, key, model, dims = cfg
    body = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(url + "/v1/embeddings", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
        data = json.loads(resp.read(64 * 1024 * 1024).decode("utf-8"))
    rows = sorted(data.get("data") or [], key=lambda r: r.get("index", 0))
    if len(rows) != len(texts):
        raise ValueError(f"asked for {len(texts)} embeddings, got {len(rows)}")
    out = []
    for r in rows:
        vec = [float(x) for x in (r.get("embedding") or [])][:dims]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        out.append(array.array("f", (x / norm for x in vec)))
    return out


def embed_vectors(idx) -> dict:
    """{card hash: vector} loaded from the index once and kept in memory."""
    vectors = idx.caches.get("vectors")
    if vectors is None:
        vectors = {}
        if idx.db is not None:
            try:
                for h, blob in idx.db.execute("SELECT hash, vec FROM embeddings"):
                    v = array.array("f")
                    v.frombytes(blob)
                    vectors[h] = v
            except sqlite3.Error:
                pass
        idx.caches["vectors"] = vectors
    return vectors


def embed_refresh(idx, seconds: float = 60.0) -> str:
    """Embed the cards of symbols that have no vector yet, in batches, within a time
    budget; on any error stop quietly (lexical ranking still works). Returns a status line."""
    cfg = embed_config()
    if cfg is None:
        return ""
    vectors = embed_vectors(idx)
    todo, seen = [], set()
    for rel in sorted(idx.files):
        for s in idx.files[rel]["symbols"]:
            card = symbol_card(s)
            h = card_hash(cfg[2], card)
            if h not in vectors and h not in seen:
                seen.add(h)
                todo.append((h, card))
            if len(seen) >= EMBED_MAX_SYMBOLS:
                break
    if not todo:
        return f"Embeddings: {len(vectors):,} symbol cards embedded with {cfg[2]}."
    t0, done = time.time(), 0
    for i in range(0, len(todo), EMBED_BATCH):
        if time.time() - t0 > seconds:
            break
        batch = todo[i:i + EMBED_BATCH]
        try:
            vecs = embed_request([c for _, c in batch], cfg)
        except Exception as e:  # any failure: keep lexical ranking, say why in code_describe
            idx.caches["embed_error"] = short(f"{type(e).__name__}: {e}", 200)
            break
        idx.caches.pop("embed_error", None)
        rows = []
        for (h, _), v in zip(batch, vecs):
            vectors[h] = v
            rows.append((h, v.tobytes()))
        done += len(batch)
        if idx.db is not None:
            try:
                with idx.db:
                    idx.db.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", rows)
            except sqlite3.Error:
                pass
    left = len(todo) - done
    status = f"Embeddings: {done:,} new symbol cards embedded with {cfg[2]}" + (f", {left:,} still to do." if left else ".")
    if idx.caches.get("embed_error"):
        status += f" Stopped: {idx.caches['embed_error']} (ranking stays lexical)."
    return status


def embed_status(idx) -> str:
    cfg = embed_config()
    if cfg is None:
        return "off (SWITCHYARD_CODE_EMBED=1 with SWITCHYARD_URL and SWITCHYARD_API_KEY turns it on)"
    vectors = embed_vectors(idx)
    error = idx.caches.get("embed_error")
    return f"on, {cfg[2]}, {len(vectors):,} cards embedded" + (f"; last error: {error}" if error else "")


def embed_blend(idx, query: str, rows: list) -> list:
    """Blend cosine similarity between the query and symbol cards into lexical scores, and
    add close matches the words missed. Off unless configured; any error leaves rows as
    they are."""
    cfg = embed_config()
    if cfg is None:
        return rows
    try:
        embed_refresh(idx, seconds=3.0)
        vectors = embed_vectors(idx)
        if not vectors:
            return rows
        cache = idx.caches.setdefault("query_vectors", {})
        q = cache.get(query)
        if q is None:
            q = cache[query] = embed_request([query[:EMBED_MAX_CARD]], cfg)[0]
        top = max((r["score"] for r in rows), default=1.0) or 1.0
        by_key = {(r["rel"], r["sym"]["start"], r["sym"]["qual"]): r for r in rows}
        scored, t0 = [], time.time()
        for rel in sorted(idx.files):
            for s in idx.files[rel]["symbols"]:
                v = vectors.get(card_hash(cfg[2], symbol_card(s)))
                if v is None or len(v) != len(q):
                    continue
                cos = sum(a * b for a, b in zip(q, v))
                scored.append((cos, rel, s))
            if time.time() - t0 > 2.0:
                break
        scored.sort(key=lambda t: -t[0])
        cosines = {(rel, s["start"], s["qual"]): cos for cos, rel, s in scored}
        out = []
        for key, r in by_key.items():
            cos = max(0.0, cosines.get(key, 0.0))
            out.append(dict(r, score=top * (0.65 * r["score"] / top + 0.35 * cos)))
        for cos, rel, s in scored[:20]:
            key = (rel, s["start"], s["qual"])
            if key not in by_key and cos >= 0.45:
                out.append({"rel": rel, "sym": s, "score": top * 0.35 * cos})
        return out
    except Exception as e:
        idx.caches["embed_error"] = short(f"{type(e).__name__}: {e}", 200)
        return rows


# --- protocol -------------------------------------------------------------------------------

ROOT_PROPERTY = {"type": "string", "description": "Directory to work in (default: the git toplevel of the "
                 "current directory, else the current directory). A git worktree is its own root. "
                 "dep:<package> reads an installed dependency (see code_index dependency)."}
FORMAT_PROPERTY = {"type": "string", "enum": list(FORMATS),
                   "description": "plain (one labelled row per item), compact (a legend line, shared paths "
                                  "written once, unpadded rows) or auto (compact when it is at least 15% shorter)."}

TOOLS = [
    {
        "name": "code_find",
        "description": (
            "Ranked symbol search: start here when you know roughly what you want but not where it is. "
            "Words are matched against symbol names (camelCase and snake_case split), qualified names, "
            "signatures and doc lines, ranked by relevance with exact and prefix name matches first, "
            "definitions over declarations, shallower paths and widely used names higher. Tolerates a "
            "typo (says 'did you mean'). Returns score, kind, qualified name, file:lines and the signature. "
            "Then code_symbol for one body, or code_context for several bodies in one call."),
        "inputSchema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Words or a name, e.g. 'price window' or parseArgs."},
            "kind": {"type": "string", "description": "Only this kind, e.g. function, class, method (optional)."},
            "language": {"type": "string", "description": "Only this language, e.g. python, typescript (optional)."},
            "path": {"type": "string", "description": "Only under this directory or file, or matching this glob (optional)."},
            "limit": {"type": "integer", "description": f"Most rows (default {FIND_LIMIT}).", "default": FIND_LIMIT},
            "format": dict(FORMAT_PROPERTY, default="auto"),
            "root": ROOT_PROPERTY,
        }, "required": ["query"]},
    },
    {
        "name": "code_context",
        "description": (
            "One call to understand a feature or question: ranks symbols like code_find, then packs the "
            "best ones into a token budget: each file's import lines, whole bodies when small (signature, "
            "head and tail with the elided lines named when large), and the names they call and are called "
            "from. Ends with what was left out and how to fetch it. Use it before reading files for "
            "'how does X work' questions."),
        "inputSchema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What you want to understand, in words or names."},
            "budget_tokens": {"type": "integer", "description": f"Size of the answer in tokens, estimated as "
                              f"characters/4 (default {CONTEXT_BUDGET}).", "default": CONTEXT_BUDGET},
            "include": {"type": "array", "items": {"type": "string", "enum": list(CONTEXT_PARTS)},
                        "description": "Parts to include (default all): bodies, imports, callers, callees."},
            "kind": {"type": "string", "description": "Only symbols of this kind (optional)."},
            "language": {"type": "string", "description": "Only this language (optional)."},
            "path": {"type": "string", "description": "Only under this directory or matching this glob (optional)."},
            "root": ROOT_PROPERTY,
        }, "required": ["query"]},
    },
    {
        "name": "code_index",
        "description": (
            "Build or refresh the code index for a root and report files, symbols, languages and the "
            "time taken. Optional: every other code_ tool indexes its root on first use and re-parses "
            "files whose mtime or size changed. Pass refresh=true to re-parse everything. Pass "
            "dependency=<package> to index an installed third-party package (Python from the project's "
            ".venv, npm from node_modules; py:/npm: prefixes force one) as the read-only root dep:<package>."),
        "inputSchema": {"type": "object", "properties": {
            "dependency": {"type": "string", "description": "An installed package to index, e.g. requests or npm:react."},
            "root": ROOT_PROPERTY,
            "refresh": {"type": "boolean", "description": "Re-parse every file (default false).", "default": False},
        }},
    },
    {
        "name": "code_tree",
        "description": (
            "Directory tree with file sizes and symbol counts, honouring .gitignore. Directories below "
            "the depth show totals only. Lists indexed dependency roots. Use it to find your way around "
            "before outlining files."),
        "inputSchema": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory under the root (default: the root)."},
            "depth": {"type": "integer", "description": "Levels to expand (default 2).", "default": 2},
            "format": dict(FORMAT_PROPERTY, default="plain"),
            "root": ROOT_PROPERTY,
        }},
    },
    {
        "name": "code_outline",
        "description": (
            "A file's symbols in order (functions, classes, methods, constants, headings, keys, notebook "
            "cells, API operations, ...) with kind, signature line, start-end lines and nesting, plus the "
            "file's leading doc line. Much cheaper than reading the file: outline first, then code_symbol "
            "or code_read for the part you need. Word documents and notebooks are outlined through a text view."),
        "inputSchema": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path, relative to the root (or absolute)."},
            "format": dict(FORMAT_PROPERTY, default="plain"),
            "root": ROOT_PROPERTY,
        }, "required": ["path"]},
    },
    {
        "name": "code_symbol",
        "description": (
            "The full source (with line numbers) of the symbol with this name: exact match first, then "
            "case-insensitive, then a dotted suffix such as Class.method. When several match, lists "
            "them with paths and lines instead; narrow with path, kind or a dotted name. Also reads one "
            "section of a document by its heading."),
        "inputSchema": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Symbol name, e.g. parse_args or Router.handle."},
            "path": {"type": "string", "description": "Only this file or directory (optional)."},
            "kind": {"type": "string", "description": "Only this kind, e.g. class, function, method (optional)."},
            "root": ROOT_PROPERTY,
        }, "required": ["name"]},
    },
    {
        "name": "code_search",
        "description": (
            "Search the indexed files for text (or a regular expression with regex=true) and show "
            "matching lines with line numbers, grouped by file. Ignored, binary and very large files are "
            f"skipped (an empty answer says which). Output is capped (default {SEARCH_MAX_CHARS:,} "
            "characters) and says when it was cut."),
        "inputSchema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Text to find (a Python regex with regex=true)."},
            "regex": {"type": "boolean", "description": "Treat query as a regular expression (default false).", "default": False},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive (default false).", "default": False},
            "glob": {"type": "string", "description": "Only files matching this glob, e.g. *.py or src/**/*.ts (comma-separated for several)."},
            "max_results": {"type": "integer", "description": "Most matching lines (default 50).", "default": 50},
            "context": {"type": "integer", "description": "Lines of context around each match (default 0).", "default": 0},
            "max_chars": {"type": "integer", "description": f"Output cap in characters (default {SEARCH_MAX_CHARS}).", "default": SEARCH_MAX_CHARS},
            "format": dict(FORMAT_PROPERTY, default="plain"),
            "root": ROOT_PROPERTY,
        }, "required": ["query"]},
    },
    {
        "name": "code_refs",
        "description": (
            "Lines that mention a symbol name as a whole word, grouped by file, leaving out the lines "
            "that define it. Use it to find callers and users before changing a function."),
        "inputSchema": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Symbol name (for Class.method, the method name is searched)."},
            "glob": {"type": "string", "description": "Only files matching this glob (optional)."},
            "max_results": {"type": "integer", "description": "Most lines (default 100).", "default": 100},
            "format": dict(FORMAT_PROPERTY, default="plain"),
            "root": ROOT_PROPERTY,
        }, "required": ["name"]},
    },
    {
        "name": "code_read",
        "description": (
            f"Lines start..end of a file with line numbers, at most max_lines (default 200, never more "
            f"than {READ_MAX_LINES}), so a long file is never pulled in whole by accident. Says how to "
            "continue when there is more. Notebooks and Word documents are read through their text view."),
        "inputSchema": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path, relative to the root (or absolute)."},
            "start": {"type": "integer", "description": "First line, 1-based (default 1).", "default": 1},
            "end": {"type": "integer", "description": "Last line, inclusive (default start + max_lines - 1)."},
            "max_lines": {"type": "integer", "description": f"Most lines to return (default 200, at most {READ_MAX_LINES}).", "default": 200},
            "root": ROOT_PROPERTY,
        }, "required": ["path"]},
    },
    {
        "name": "code_changed",
        "description": (
            "Which symbols a change touches: maps `git diff` hunks to the functions, classes and other "
            "symbols they fall in, as added, modified (with the changed lines) or removed. Compares the "
            "working tree (or, with staged=true, the staged changes) with base: default the merge-base "
            "with the default branch, else HEAD. Untracked files count as added. Read-only git only."),
        "inputSchema": {"type": "object", "properties": {
            "base": {"type": "string", "description": "Commit, branch or tag to compare with (default: merge-base with the default branch, else HEAD)."},
            "staged": {"type": "boolean", "description": "Only staged changes (default false).", "default": False},
            "paths": {"type": "array", "items": {"type": "string"}, "description": "Only these files or directories (optional)."},
            "format": dict(FORMAT_PROPERTY, default="auto"),
            "root": ROOT_PROPERTY,
        }},
    },
    {
        "name": "code_tests",
        "description": (
            "The test functions that use a name: searches test files (under tests/, test/, spec/ or "
            "__tests__/, or named test_*, *_test.*, *.test.* or *.spec.*) for the name as a whole word "
            "and lists the enclosing test function, method or it/test block with file and lines, the "
            "ones that call it directly first. Says what it searched when it finds none."),
        "inputSchema": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Symbol name (for Class.method, the method name is searched)."},
            "path": {"type": "string", "description": "Only test files under this directory or file (optional)."},
            "max_results": {"type": "integer", "description": f"Most test functions listed (default {TESTS_LIMIT}).", "default": TESTS_LIMIT},
            "format": dict(FORMAT_PROPERTY, default="auto"),
            "root": ROOT_PROPERTY,
        }, "required": ["name"]},
    },
    {
        "name": "code_history",
        "description": (
            "Recent commits that touched a file, or with symbol one function or class in it (git log -L "
            "over the symbol's lines at HEAD): short hash, date, lines added and removed, and subject, "
            f"newest first, at most limit (default {HISTORY_LIMIT}). Outside git, or without git, it says "
            "so in one line. Read-only git only."),
        "inputSchema": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path, relative to the root (or absolute)."},
            "symbol": {"type": "string", "description": "A symbol in that file, e.g. parse_args or Router.handle (optional)."},
            "limit": {"type": "integer", "description": f"Most commits (default {HISTORY_LIMIT}, at most 100).", "default": HISTORY_LIMIT},
            "format": dict(FORMAT_PROPERTY, default="auto"),
            "root": ROOT_PROPERTY,
        }, "required": ["path"]},
    },
    {
        "name": "code_describe",
        "description": (
            "Describe the code index: where it is kept, the roots it knows, when this root was last "
            "refreshed, its languages, how many symbols, dependency roots, whether meaning-based ranking "
            "is on, and files with parse notes."),
        "inputSchema": {"type": "object", "properties": {"root": ROOT_PROPERTY}},
    },
]
TOOL_NAMES = {tool["name"] for tool in TOOLS}

# report_model passes the model's own name to the activity report (switchyard_gateway_client). Copied here
# for a copy of this file on its own, where it answers the same and sends nothing.
REPORT_MODEL = "report_model"
REPORT_MODEL_TOOL = gateway_client.REPORT_MODEL_TOOL if gateway_client is not None else {
    "name": REPORT_MODEL,
    "description": (
        "Tell Switchyard which model you are, once per conversation, before your first other Switchyard "
        "tool call. Give your own model name exactly as you know it (for example claude-opus-5-5). If "
        "you do not know it, do not call this and do not guess. Calling it on any one Switchyard server "
        "is enough."),
    "inputSchema": {
        "type": "object",
        "properties": {"model": {"type": "string", "description": "Your model name, e.g. claude-opus-5-5."}},
        "required": ["model"],
    },
}
REPORT_MODEL_INSTRUCTION = gateway_client.REPORT_MODEL_INSTRUCTION if gateway_client is not None else (
    "Once per conversation, before your first other Switchyard tool call, call report_model with your own "
    "model name if you know it; do not guess.")
SERVER_TOOLS = TOOLS + [REPORT_MODEL_TOOL]  # what tools/list answers


def report_model(args) -> dict:
    """The report_model call: hands the name to the reporter, answers one line, and is not reported."""
    if gateway_client is not None:
        return gateway_client.answer_report_model(REPORTER, args)
    name = args.get("model") if isinstance(args, dict) else None
    name = name.strip()[:80] if isinstance(name, str) else ""
    if not name:
        return {"content": [{"type": "text", "text": "No model name given; nothing was noted."}], "isError": True}
    return {"content": [{"type": "text", "text": f"Noted: {name}."}], "isError": False}


HANDLERS = {
    "code_find": tool_find,
    "code_context": tool_context,
    "code_index": tool_index,
    "code_tree": tool_tree,
    "code_outline": tool_outline,
    "code_symbol": tool_symbol,
    "code_search": tool_search,
    "code_refs": tool_refs,
    "code_read": tool_read,
    "code_changed": tool_changed,
    "code_tests": tool_tests,
    "code_history": tool_history,
    "code_describe": tool_describe,
}

INSTRUCTIONS = (
    "Code navigation: read code by symbol instead of whole files. Start with code_find (ranked symbol "
    "search by words or a name) when you don't know where something is, or code_context to get the best "
    "matching symbols' bodies, imports and callers in one budgeted answer. code_outline shows a file's "
    "symbols with line ranges; code_symbol prints one symbol's source; code_search and code_refs find text "
    "and uses; code_read reads a capped line range; code_changed lists the symbols a diff touches; "
    "code_tests lists the tests that use a name; code_history lists the commits that touched a file or "
    "symbol. Each "
    "answer ends with a short [index ...] note: 'edited since index' means re-read, and an empty answer "
    "says what it could not see. The index is built on first use. " + REPORT_MODEL_INSTRUCTION)


# Set by main() when it serves MCP: tells the gateway about each tool call
# (switchyard_gateway_client.Reporter). None for the command line and when imported.
REPORTER = None

# --- what a call saved (for the activity report) ----------------------------------------------
#
# An estimate of the context a call kept out of the conversation, against the usual alternative of
# reading the whole file: code_outline, a code_symbol body and code_context stand in for the files
# they show. Searches, listings, code_read and the rest stand in for a grep or a ranged read, which
# cost about the same, so they claim nothing.

SAVING_FILE_CAP = 25000        # tokens: about what one whole-file read returns at most
_STANDS_IN = None              # (idx, [rel, ...]) declared by the tool that is answering, per call


def stands_in_for(idx, rels) -> None:
    """The answer being built replaces reading these files whole (saving_ledger counts it)."""
    global _STANDS_IN
    _STANDS_IN = (idx, list(rels))


class SavingLedger:
    """Tokens saved per call, without counting a file twice. A file is credited the first time an
    answer stands in for it: its size (characters / 4, at most SAVING_FILE_CAP) less the answer.
    A later answer about the same file saves nothing more and costs its own size, which is taken
    off that file's credit and off the next saving reported, so the running total stays
    (files stood in for) - (answers about them), never below zero."""

    def __init__(self):
        self.credit = {}     # (root, rel) -> tokens of saving still standing for that file
        self.debt = 0        # reported too much so far by this many tokens

    def saved(self, files, answer_tokens: int) -> int:
        """files: [((root, rel), file tokens)] the answer stands in for."""
        files = list(dict(files).items())
        new = [(key, size) for key, size in files if key not in self.credit]
        old = [key for key, _ in files if key in self.credit]
        raw = sum(min(size, SAVING_FILE_CAP) for _, size in new) - max(0, int(answer_tokens))
        if raw >= 0:
            total = sum(min(size, SAVING_FILE_CAP) for _, size in new) or 1
            for key, size in new:
                self.credit[key] = raw * min(size, SAVING_FILE_CAP) // total
        else:
            excess, raw = -raw, 0
            for key, _ in new:
                self.credit[key] = 0
            for key in old:
                take = min(excess, self.credit[key])
                self.credit[key] -= take
                self.debt += take
                excess -= take
        saved = max(0, raw - self.debt)
        self.debt -= raw - saved
        return saved


SAVING_LEDGER = SavingLedger()


def saved_tokens_of(answer_tokens: int, is_error: bool) -> int:
    """The estimate for the call just answered (0 for a tool that declared nothing, or an error)."""
    declared = _STANDS_IN
    if declared is None or is_error:
        return 0
    idx, rels = declared
    files = []
    for rel in rels:
        entry = idx.files.get(rel)
        if not entry or entry.get("language") in VIEW_LANGUAGES:
            continue   # a document's size on disk is not the size of its text
        files.append(((idx.root, rel), int(entry.get("size") or 0) // 4))
    return SAVING_LEDGER.saved(files, answer_tokens) if files else 0


def call_tool(name: str, args: dict) -> dict:
    global _STANDS_IN
    _STANDS_IN = None
    try:
        text, is_error = HANDLERS[name](args if isinstance(args, dict) else {}), False
    except ToolError as e:
        text, is_error = str(e), True
    except Exception as e:  # report it rather than drop the connection
        text, is_error = f"{name} failed: {type(e).__name__}: {e}", True
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def handle(request: dict):
    """The JSON-RPC response for one request, or None for a notification."""
    method, rid = request.get("method"), request.get("id")
    if rid is None:
        return None
    if method == "initialize":
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        version = params.get("protocolVersion") or "2025-06-18"
        if REPORTER is not None:
            info = params.get("clientInfo") if isinstance(params.get("clientInfo"), dict) else {}
            REPORTER.set_client(info.get("name"), info.get("version"))
        result = {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "switchyard-code", "version": "2.0"},
            "instructions": INSTRUCTIONS,
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": SERVER_TOOLS}
    elif method == "tools/call":
        params = request.get("params") or {}
        name = params.get("name")
        if name == REPORT_MODEL:
            return {"jsonrpc": "2.0", "id": rid, "result": report_model(params.get("arguments") or {})}
        if name not in TOOL_NAMES:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": f"unknown tool {name}"}}
        started = time.monotonic()
        result = call_tool(name, params.get("arguments") or {})
        if REPORTER is not None:
            size = gateway_client.tokens_of(sum(len(c.get("text") or "") for c in result["content"]))
            saved = saved_tokens_of(size, result["isError"])
            # result_tokens: what reading the files whole would have put in the conversation
            REPORTER.call(name, (time.monotonic() - started) * 1000, size + saved, size,
                          is_error=result["isError"], saved_tokens=saved if saved else None)
    else:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"no method {method}"}}
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def write(response: dict) -> None:
    try:
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    except (OSError, ValueError):
        pass  # the client has gone away


def cli(argv: list) -> int:
    """`switchyard_code_mcp.py TOOL [JSON-ARGS]`: run one tool and print its text."""
    if argv[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        print("\nTools: " + ", ".join(t["name"] for t in TOOLS))
        return 0
    name = argv[0] if argv[0].startswith("code_") else "code_" + argv[0]
    if name not in TOOL_NAMES:
        print(f"Unknown tool {argv[0]!r}. Tools: {', '.join(t['name'] for t in TOOLS)}", file=sys.stderr)
        return 2
    try:
        args = json.loads(argv[1]) if len(argv) > 1 else {}
    except ValueError as e:
        print(f"Arguments must be a JSON object: {e}", file=sys.stderr)
        return 2
    if not isinstance(args, dict):
        print("Arguments must be a JSON object.", file=sys.stderr)
        return 2
    result = call_tool(name, args)
    text = result["content"][0]["text"]
    print(text, file=sys.stderr if result["isError"] else sys.stdout)
    return 1 if result["isError"] else 0


def main() -> int:
    global REPORTER
    if len(sys.argv) > 1:
        return cli(sys.argv[1:])
    if gateway_client is not None:
        REPORTER = gateway_client.Reporter("code", "switchyard_code")
    try:
        return serve()
    finally:
        if REPORTER is not None:
            REPORTER.close()


def serve() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            continue
        if not isinstance(request, dict):
            continue
        response = handle(request)
        if response is not None:
            write(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
