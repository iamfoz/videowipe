#!/usr/bin/env python3
"""Handles for oversized tool results: keep the full text outside the prompt.

Long Claude Code sessions fill the context with old tool output (whole files
read, long test logs, big grep listings) that the model rarely looks at again.
apply_handles() replaces large, old tool results in an Anthropic Messages
request with a short stub: a summary plus a handle id. The full text goes into
a SQLite store (HandleStore), and the switchyard_handles MCP server
(switchyard_handles_mcp.py) lets the model read it back with
peek / slice / search / describe / list.

It works on the Anthropic side of the proxy (the MessagesRequest, as a dict),
before conversion, so one implementation covers every upstream.

Built for prompt caching. Claude Code re-sends the whole conversation every
turn, and the upstream's cache only hits while every earlier message is
byte-identical, so:
  - the handle id is a hash of (tool_use_id, content), and the stub is built
    only from the content and the tool call, then stored and reused: the same
    result becomes exactly the same bytes on every request (and after a
    restart or an upgrade of this module, for as long as the row is kept);
  - results leave the recency window `recency_batch` at a time, so the set of
    replaced results changes once per batch of tool calls, not every turn;
  - nothing is ever put back: a result replaced once stays replaced, even if
    the token estimate later dips under the threshold;
  - nothing happens until the (unreplaced) request is over context_fraction of
    the model's window, so short sessions are forwarded untouched.

Large fresh results are the exception to the last two rules: a result over
fresh_min_tokens (HANDLES_FRESH_MIN_TOKENS, 8000; 0 or off disables) from a
tool matching fresh_tools (HANDLES_FRESH_TOOLS globs) is replaced on arrival,
whatever its age, by a stub with its first fresh_head_lines and last
fresh_tail_lines lines. It never reaches the model whole, so it never costs a
cache miss later either; and it stays that stub on every later request.

Standard library only, and Python 3.9 compatible (the MCP server runs under
macOS's /usr/bin/python3).


INTEGRATION NOTE (for the server.py maintainer)
================================================

1. Settings and store, next to the TRIM_* settings (~line 172):

       import switchyard_handles as handles
       HANDLES_SETTINGS = handles.HandleSettings.from_env()   # HANDLES=on enables
       if HANDLES_SETTINGS.enabled:
           TRIM_TOOL_RESULTS = False   # handles replace the old trimming

   Env: HANDLES (off), HANDLES_CONTEXT_FRACTION (0.5), HANDLES_RECENCY_WINDOW (6),
   HANDLES_RECENCY_BATCH (4), HANDLES_MIN_RESULT_TOKENS (1500),
   HANDLES_TTL_DAYS (7), SWITCHYARD_HANDLES_DB (~/.switchyard/handles.sqlite3),
   HANDLES_REQUIRE_TOOLS (on: only replace when the request shows the
   switchyard_handles MCP tools, so the model can actually read a handle),
   HANDLES_FRESH_MIN_TOKENS (8000), HANDLES_FRESH_TOOLS
   (Read,Grep,Glob,Bash,WebFetch,mcp__*), HANDLES_FRESH_HEAD_LINES (40),
   HANDLES_FRESH_TAIL_LINES (10).
   The store opens lazily on first use (handles.open_store), so the proxy
   starts fine even if the path is unwritable; apply_handles then returns the
   request unchanged with stats["skipped"] = "store_error: ...".

   HANDLES=on should switch TRIM_TOOL_RESULTS off: both rewrite old tool
   results, and running both would move the cut-off twice and stub a stub.
   Keep the TRIM_* code for HANDLES=off (it needs no MCP server).

2. The call, at the top of convert_anthropic_to_litellm (~line 849), before
   convert_messages_to_openai, so create_message AND count_tokens see the
   same, smaller prompt (count_tokens must agree, or Claude Code auto-compacts
   on the unreplaced size):

       handle_stats = None
       if HANDLES_SETTINGS.enabled and not anthropic_request.model.startswith("anthropic/"):
           capabilities = get_model_capabilities(anthropic_request.model)
           window = handles.context_window_for(anthropic_request.model, capabilities)
           ratio = token_calibration.chars_per_token(anthropic_request.model)
           new_request, handle_stats = handles.apply_handles(
               anthropic_request.model_dump(exclude_none=True),
               model_window=window,
               settings=HANDLES_SETTINGS,
               session=claude_code_session(anthropic_request),
               estimator=lambda r: handles.estimate_tokens(r, chars_per_token=ratio),
           )
           if handle_stats["results_replaced"]:
               # model_copy skips the validators, so the mapped model and
               # original_model are kept as they are
               anthropic_request = anthropic_request.model_copy(update={"messages": [
                   Message.model_validate(m) for m in new_request["messages"]]})
       ...
       litellm_request["_proxy_handles"] = handle_stats   # popped like _proxy_trim

   context_window_for() takes MODEL_CAPABILITIES' context_window when it is
   set (262144 for qwen3.8 and deepseek-v4-flash) and otherwise falls back to
   MODEL_CONTEXT_WINDOWS here, then 128000. The estimator is optional
   (default: 4 chars per token); passing the calibrated ratio is fine
   because replaced results are sticky, so a drifting estimate can delay a
   batch but never undo one. apply_handles never mutates its input and never
   raises on store errors. It does a few small SQLite reads/writes per call;
   if that ever shows up in latency, run it via starlette's run_in_threadpool.

3. Recording, in create_message right after convert_anthropic_to_litellm
   (next to record_trim):

       request_log.record_handles(litellm_request.pop("_proxy_handles", None))

   with RequestLog.record_handles setting
       self.handles_created  = stats["handles_created"]
       self.results_replaced = stats["results_replaced"]
       self.tokens_saved     = stats["tokens_saved"]
   and a note such as f"{n} old tool results as handles (-{saved:,} tokens)".
   stats["new_replaced_ids"] non-empty means earlier messages changed this
   request (one cache miss), which is what TrimSteps detects today, without
   needing per-conversation state. In count_tokens just pop and drop it.

   UsageStore: add ("handles_created", "INTEGER"), ("results_replaced",
   "INTEGER"), ("tokens_saved", "INTEGER") to the CREATE TABLE and to the
   ALTER TABLE loop (~line 2950), and to the INSERT in record(), written as
   `log.handles_created or None` etc. like the trimmed_* columns.

4. Workers (switchyard-worker.sh): give every worker the tools, not only
   with -t. After the approve-mcp.json block, write a second config:

       handles_config="$scratch/handles-mcp.json"
       /usr/bin/python3 -I - "$handles_config" "$PROJECT_DIR/switchyard_handles_mcp.py" <<'EOF'
       import json, sys
       path, script = sys.argv[1:]
       config = {"mcpServers": {"switchyard_handles": {
           "type": "stdio", "command": "/usr/bin/python3", "args": ["-I", script],
           "env": {}}}}
       with open(path, "w") as f:
           json.dump(config, f)
       EOF

   pass it with `--mcp-config "$handles_config"` (--mcp-config takes several
   files, so with -t it is `--mcp-config "$mcp_config" "$handles_config"`),
   and add "mcp__switchyard_handles" to DEFAULT_ALLOW so a headless worker
   can call the read-only tools without a permission prompt. If the proxy
   runs with a non-default SWITCHYARD_HANDLES_DB, put the same value in
   "env" so both sides open the same file.

5. Interactive users register the server once:

       claude mcp add --scope user switchyard_handles -- \\
           python3 -I /Users/forryanmp/code/switchyard/switchyard_handles_mcp.py

   (add `-e SWITCHYARD_HANDLES_DB=/path` before `--` for a non-default store).

6. Remote store. With SWITCHYARD_URL set, open_store() returns a
   RemoteHandleStore that talks to a Switchyard gateway over HTTP instead of
   opening SQLite (x-api-key: SWITCHYARD_API_KEY), so the MCP tools and the
   MCP proxy (switchyard_mcp_proxy.py) work on another machine. Each of the
   two is read from the environment, else from the gateway file
   (~/.switchyard/gateway.env; switchyard_gateway_client.py):

       GET  {base}/handles/{id}            -> record with "text", or 404
       GET  {base}/handles?session=&limit= -> {"handles": [records without text]}
       POST {base}/handles {"tool_name", "tool_use_id"?, "text", "session"?,
                            "tool_input"?}  -> {"id", "stub", "created"}

   The gateway itself must open its store with open_store(remote=False) (or
   HandleStore directly), so it never calls itself. Its POST handler should
   answer with store_tool_result(text, tool_name=..., tool_use_id=...,
   tool_input=..., session=..., store=<local store>): that is the same code
   path the MCP proxy uses locally, so stubs are byte-identical either way.
"""

from __future__ import annotations

import csv
import difflib
import fnmatch
import hashlib
import io
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    import switchyard_gateway_client as gateway_client
except ImportError:  # imported from somewhere that doesn't have this folder on sys.path
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import switchyard_gateway_client as gateway_client

DEFAULT_DB_PATH = "~/.switchyard/handles.sqlite3"
HANDLE_PREFIX = "h_"
HANDLE_HEX_CHARS = 12
CHARS_PER_TOKEN = 4.0
TOKENS_PER_IMAGE = 1600

MCP_SERVER_NAME = "switchyard_handles"
# How each client spells this server's tools: Claude Code mcp__<server>__<tool>
# (Hermes 0.20 too); Codex sends the namespace "mcp__switchyard_handles" with
# "handles_peek" (the Responses ingress joins them with "__"); older Hermes
# mcp_<server>_<tool>, single underscores
MCP_TOOL_PREFIX = f"mcp__{MCP_SERVER_NAME}__"
MCP_TOOL_PREFIXES = (MCP_TOOL_PREFIX, f"mcp_{MCP_SERVER_NAME}_")
# The same tools merged into a wrapped MCP server by switchyard_mcp_proxy.py
# show up as mcp__<server>__handles_peek etc. (mcp_<server>_handles_peek in
# older Hermes)
HANDLES_TOOL_RE = re.compile(
    r"\bmcp_[A-Za-z0-9_-]+?_handles_(?:peek|slice|search|describe|list|table|aggregate)\b")

# Context windows (tokens) by upstream model name, lower case, without the
# "openai/" provider prefix. Longest matching prefix wins. Mirrors
# server.py MODEL_CAPABILITIES, which context_window_for() prefers.
MODEL_CONTEXT_WINDOWS = {
    "qwen3.8": 262144,
    "qwen3": 262144,
    "deepseek-v4-flash": 262144,
    "deepseek-v4": 262144,
}
DEFAULT_CONTEXT_WINDOW = 128000

# Lines longer than this are cut in stubs; JSON whose lines are longer than
# VIEW_PRETTY_LINE_CHARS is pretty-printed for line-based reading.
STUB_LINE_CHARS = 160
VIEW_PRETTY_LINE_CHARS = 1000

# Large fresh results (see HandleSettings.fresh_*). Their stubs keep lines
# verbatim, except that a line longer than FRESH_LINE_CHARS is cut (so one
# minified line can't keep the stub as large as the result).
DEFAULT_FRESH_MIN_TOKENS = 8000
# Claude Code's read, search, shell and fetch tools; Hermes's (read_file,
# search_files, terminal, execute_code, process, web_extract); Codex's shell
# tools (exec_command and write_stdin, older shell / shell_command); and MCP
# tools in every client's spelling (mcp_* covers mcp__* too)
DEFAULT_FRESH_TOOLS = ("Read", "Grep", "Glob", "Bash", "WebFetch",
                       "read_file", "search_files", "terminal", "execute_code", "process", "web_extract",
                       "exec_command", "write_stdin", "shell", "shell_command", "mcp_*")
FRESH_LINE_CHARS = 500
# The first line of every stub this module writes; a tool result that starts
# with one is already a stub and is never replaced again
STUB_PREFIX = "[Tool result kept outside the conversation by Switchyard as handle "
FRESH_STUB_PREFIX = "[Large tool result kept outside the conversation by Switchyard as handle "
SOURCE_ARG_KEYS = ("file_path", "path", "command", "cmd", "pattern", "url", "query", "description")
# Shell tools other than Claude Code's Bash: Hermes's terminal, Codex's
# exec_command (argument cmd) and its older shell / shell_command
SHELL_TOOLS = ("terminal", "exec_command", "shell", "shell_command")
SHELL_EXIT_RES = (re.compile(r"^Exit code (-?\d+)", re.M),
                  re.compile(r"^Process exited with code (-?\d+)", re.M),  # Codex exec_command
                  re.compile(r'"exit_code":\s*(-?\d+)'))  # Hermes terminal (JSON)


# ---------------------------------------------------------------- settings


def _env_flag(value: Optional[str], default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("on", "true", "1", "yes")


def _env_number(env: Dict[str, str], name: str, default, kind=int):
    try:
        raw = env.get(name)
        return kind(raw) if raw not in (None, "") else default
    except ValueError:
        return default


@dataclass(frozen=True)
class HandleSettings:
    enabled: bool = False
    context_fraction: float = 0.5
    recency_window: int = 6
    recency_batch: int = 4
    min_result_tokens: int = 1500
    ttl_days: float = 7.0
    db_path: Optional[str] = None  # None: SWITCHYARD_HANDLES_DB or DEFAULT_DB_PATH
    require_tools: bool = True
    head_lines: int = 5
    tail_lines: int = 3
    # Large fresh results: a result over fresh_min_tokens (0: off) from a tool
    # matching one of fresh_tools (globs; empty: any tool) is replaced on
    # arrival, whatever its age, by a stub keeping its first and last lines.
    fresh_min_tokens: int = DEFAULT_FRESH_MIN_TOKENS
    fresh_tools: Tuple[str, ...] = DEFAULT_FRESH_TOOLS
    fresh_head_lines: int = 40
    fresh_tail_lines: int = 10

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "HandleSettings":
        env = dict(os.environ if env is None else env)
        return cls(
            enabled=_env_flag(env.get("HANDLES"), False),
            context_fraction=_env_number(env, "HANDLES_CONTEXT_FRACTION", 0.5, float),
            recency_window=_env_number(env, "HANDLES_RECENCY_WINDOW", 6),
            recency_batch=max(1, _env_number(env, "HANDLES_RECENCY_BATCH", 4)),
            min_result_tokens=_env_number(env, "HANDLES_MIN_RESULT_TOKENS", 1500),
            ttl_days=_env_number(env, "HANDLES_TTL_DAYS", 7.0, float),
            db_path=env.get("SWITCHYARD_HANDLES_DB") or None,
            require_tools=_env_flag(env.get("HANDLES_REQUIRE_TOOLS"), True),
            fresh_min_tokens=parse_fresh_min_tokens(env.get("HANDLES_FRESH_MIN_TOKENS")),
            fresh_tools=parse_fresh_tools(env.get("HANDLES_FRESH_TOOLS")),
            fresh_head_lines=max(0, _env_number(env, "HANDLES_FRESH_HEAD_LINES", 40)),
            fresh_tail_lines=max(0, _env_number(env, "HANDLES_FRESH_TAIL_LINES", 10)),
        )


def parse_fresh_min_tokens(value: Any, default: int = DEFAULT_FRESH_MIN_TOKENS) -> int:
    """HANDLES_FRESH_MIN_TOKENS (or a consumer's fresh_handles_min_tokens):
    a whole number of tokens; 0, a negative number, "off" or "false"
    turn fresh compaction off; unset or unreadable gives `default`."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0, int(value))
    text = str(value).strip().lower()
    if text == "":
        return default
    if text in ("off", "false", "no", "none"):
        return 0
    try:
        return max(0, int(float(text)))
    except ValueError:
        return default


def parse_fresh_tools(value: Optional[str]) -> Tuple[str, ...]:
    """HANDLES_FRESH_TOOLS: comma-separated globs of tool names. Unset gives
    the default list; set but empty means any tool."""
    if value is None:
        return DEFAULT_FRESH_TOOLS
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def default_db_path(path: Optional[str] = None) -> str:
    return os.path.expanduser(path or os.environ.get("SWITCHYARD_HANDLES_DB") or DEFAULT_DB_PATH)


def default_ttl_days() -> float:
    return _env_number(dict(os.environ), "HANDLES_TTL_DAYS", 7.0, float)


def context_window_for(
    model: Optional[str],
    capabilities: Optional[Dict[str, Any]] = None,
    default: int = DEFAULT_CONTEXT_WINDOW,
) -> int:
    """The model's context window in tokens: capabilities["context_window"]
    (server.py get_model_capabilities) when set, else MODEL_CONTEXT_WINDOWS by
    longest prefix of the name without its provider prefix, else `default`."""
    if capabilities and capabilities.get("context_window"):
        return int(capabilities["context_window"])
    if not model:
        return default
    name = model.split("/", 1)[-1].lower()
    if name in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[name]
    best = None
    for prefix, window in MODEL_CONTEXT_WINDOWS.items():
        if name.startswith(prefix) and (best is None or len(prefix) > best[0]):
            best = (len(prefix), window)
    return best[1] if best else default


# ---------------------------------------------------------------- content


def tool_result_text(content: Any) -> Optional[str]:
    """A tool_result's content as one string, or None when it holds an image
    (those are left alone: the model needs to see them)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "image":
                    return None
                if item.get("type") == "text" or "text" in item:
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(json.dumps(item, sort_keys=True, ensure_ascii=False, default=str))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def handle_id(tool_use_id: str, text: str) -> str:
    digest = hashlib.sha256(f"{tool_use_id or ''}\0{text}".encode("utf-8")).hexdigest()
    return HANDLE_PREFIX + digest[:HANDLE_HEX_CHARS]


def normalize_handle(handle: Any) -> str:
    """Accept "h_abc…", "abc…", or either wrapped in quotes or brackets."""
    text = str(handle or "").strip().strip("[]()<>\"'`").strip().lower()
    if text and not text.startswith(HANDLE_PREFIX):
        text = HANDLE_PREFIX + text
    return text


READ_LINE_RE = re.compile(r"^\s*(\d+)(?:\t|→)")
DIFF_HUNK_RE = re.compile(r"^(diff --git |@@ -\d+(?:,\d+)? \+\d+)", re.M)
DIFF_FILE_RE = re.compile(r"^(\+\+\+|---) ", re.M)
LOG_LINE_RE = re.compile(
    r"(\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}|\b\d{2}:\d{2}:\d{2}\b|"
    r"\b(ERROR|WARN(ING)?|INFO|DEBUG|TRACE|FATAL|CRITICAL)\b|^\s*(PASSED|FAILED|ok|FAIL)\b)"
)
CODE_LINE_RE = re.compile(
    r"^\s*(def |class |import |from \S+ import|function |const |let |var |return\b|if \(|"
    r"for \(|#include|package |public |private |fn |func |struct |impl |@\w+)|[;{}]\s*$"
)
CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".swift", ".c",
    ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".sh", ".bash", ".zsh", ".sql",
    ".scala", ".lua", ".m", ".mm", ".vue", ".svelte", ".css", ".scss", ".html", ".toml",
    ".yaml", ".yml", ".ini", ".cfg", ".mk", ".gradle", ".r", ".jl", ".ex", ".exs", ".dart",
}


def strip_read_line_numbers(text: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    """Claude Code's Read output prefixes lines with "   12\\t" (or "12→").
    Return the text without them, and the (first, last) line numbers."""
    lines = text.split("\n")
    matches = [READ_LINE_RE.match(line) for line in lines]
    hits = [m for m in matches if m]
    non_empty = sum(1 for line in lines if line.strip())
    if not hits or len(hits) < 0.8 * non_empty:
        return text, None
    stripped = "\n".join(line[m.end():] if m else line for line, m in zip(lines, matches))
    return stripped, (int(hits[0].group(1)), int(hits[-1].group(1)))


def _parse_json(text: str) -> Tuple[bool, Any]:
    body = text.strip()
    if body[:1] not in ("{", "["):
        return False, None
    try:
        return True, json.loads(body)
    except (ValueError, RecursionError):
        return False, None


def _fraction(lines: List[str], pattern) -> float:
    sample = [line for line in lines if line.strip()][:300]
    if not sample:
        return 0.0
    return sum(1 for line in sample if pattern.search(line)) / len(sample)


def detect_kind(text: str, tool_name: Optional[str] = None, tool_input: Optional[dict] = None) -> str:
    """One of json, diff, log, code, text."""
    is_json, _ = _parse_json(text)
    if is_json:
        return "json"
    if DIFF_HUNK_RE.search(text) and DIFF_FILE_RE.search(text):
        return "diff"
    path = ""
    if isinstance(tool_input, dict):
        path = str(tool_input.get("file_path") or tool_input.get("path") or "")
    extension = os.path.splitext(path)[1].lower()
    if extension in (".diff", ".patch"):
        return "diff"
    if extension == ".log":
        return "log"
    if tool_name == "Read" and extension in CODE_EXTENSIONS:
        return "code"
    lines = text.split("\n")
    if _fraction(lines, LOG_LINE_RE) > 0.3:
        return "log"
    if _fraction(lines, CODE_LINE_RE) > 0.25:
        return "code"
    return "text"


def json_outline(value: Any, limit: int = 12) -> str:
    """Top-level shape of a JSON value, e.g. "object with keys: a, b (+3 more)"."""
    def names(keys: List[str]) -> str:
        shown = ", ".join(keys[:limit])
        return shown + (f" (+{len(keys) - limit} more)" if len(keys) > limit else "")

    if isinstance(value, dict):
        return f"object with {len(value)} keys: {names([str(k) for k in value])}" if value else "empty object"
    if isinstance(value, list):
        text = f"array of {len(value):,} items"
        if value and isinstance(value[0], dict):
            text += f"; first item keys: {names([str(k) for k in value[0]])}"
        return text
    return f"{type(value).__name__} value"


def json_keys(value: Any) -> List[str]:
    if isinstance(value, dict):
        return [str(k) for k in value]
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return [str(k) for k in value[0]]
    return []


def view_lines(content: str, kind: Optional[str] = None) -> List[str]:
    """The lines that stubs, peek, slice and search count: the content's own
    lines, except that JSON with very long lines (minified) is pretty-printed."""
    lines = content.split("\n")
    if (kind in (None, "json")) and max((len(line) for line in lines), default=0) > VIEW_PRETTY_LINE_CHARS:
        is_json, value = _parse_json(content)
        if is_json:
            return json.dumps(value, indent=2, ensure_ascii=False).split("\n")
    return lines


def result_view(text: str, tool_name: Optional[str] = None, kind: Optional[str] = None) -> List[str]:
    """view_lines() of a stored result, the same way for stubs and the MCP
    tools: a Read of a JSON file is viewed as JSON (without line numbers)."""
    if tool_name == "Read" and kind == "json":
        body, _ = strip_read_line_numbers(text)
        if _parse_json(body)[0]:
            return view_lines(body, kind)
    return view_lines(text, kind)


def _clip(text: str, limit: int = STUB_LINE_CHARS) -> str:
    text = text.replace("\r", "")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _quote(value: Any, limit: int = 160) -> str:
    return json.dumps(_clip(str(value), limit), ensure_ascii=False)


def describe_source(
    tool_name: Optional[str],
    tool_input: Optional[dict],
    text: str,
    is_error: bool = False,
    read_range: Optional[Tuple[int, int]] = None,
) -> str:
    """One line on where a result came from, using the tool call's input."""
    name = tool_name or "tool"
    args = tool_input if isinstance(tool_input, dict) else {}
    non_empty = [line for line in text.split("\n") if line.strip()]
    if name == "Read":
        source = f"Read {_quote(args.get('file_path') or args.get('path') or '?')}"
        if read_range:
            source += f", lines {read_range[0]}-{read_range[1]}"
        elif args.get("offset") or args.get("limit"):
            start = int(args.get("offset") or 1)
            source += f", lines {start}-{start + int(args.get('limit') or 0) - 1}" if args.get("limit") else f", from line {start}"
        return source
    if name == "Bash":
        source = f"Bash command {_quote(args.get('command') or '?', 200)}"
        match = re.search(r"^Exit code (\d+)", text, re.M)
        if match:
            source += f", exit code {match.group(1)}"
        elif is_error:
            source += ", failed"
        else:
            source += ", succeeded"
        return source
    if name == "Grep":
        source = f"Grep pattern {_quote(args.get('pattern') or '?')}"
        if args.get("path"):
            source += f" in {_quote(args['path'])}"
        mode = args.get("output_mode") or "files_with_matches"
        found = re.match(r"\s*Found (\d+) (\w+)", text)
        if found:
            source += f", {int(found.group(1)):,} {found.group(2)}"
        elif mode == "count":
            total = 0
            for line in non_empty:
                tail = line.rsplit(":", 1)[-1].strip()
                total += int(tail) if tail.isdigit() else 0
            source += f", {total:,} matches in {len(non_empty):,} files"
        elif mode == "content":
            source += f", {len(non_empty):,} matching lines"
        else:
            source += f", {len(non_empty):,} files with matches"
        return source
    if name == "Glob":
        source = f"Glob pattern {_quote(args.get('pattern') or '?')}"
        if args.get("path"):
            source += f" in {_quote(args['path'])}"
        paths = [line for line in non_empty if not line.startswith("(")]
        return source + f", {len(paths):,} paths"
    if name in SHELL_TOOLS:
        command = args.get("command") if args.get("command") is not None else args.get("cmd")
        if isinstance(command, list):  # Codex's older shell tool: an argv list
            command = " ".join(str(part) for part in command)
        source = f"{name} command {_quote(command or '?', 200)}"
        for pattern in SHELL_EXIT_RES:
            match = pattern.search(text)
            if match:
                return source + f", exit code {match.group(1)}"
        return source + (", failed" if is_error else ", succeeded")
    if name == "read_file":  # Hermes: path, offset (1-based) and limit
        source = f"read_file {_quote(args.get('path') or '?')}"
        if args.get("offset") or args.get("limit"):
            start = int(args.get("offset") or 1)
            source += f", lines {start}-{start + int(args.get('limit') or 0) - 1}" if args.get("limit") else f", from line {start}"
        return source
    for key in SOURCE_ARG_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value:
            return f"{name} {key}={_quote(value)}" + (", failed" if is_error else "")
    return name + (", failed" if is_error else "")


def analyse_result(
    text: str,
    tool_name: Optional[str] = None,
    tool_input: Optional[dict] = None,
    is_error: bool = False,
) -> Dict[str, Any]:
    """Kind, size, shape and source of one tool result (all deterministic)."""
    body, read_range = (strip_read_line_numbers(text) if tool_name == "Read" else (text, None))
    kind = detect_kind(body, tool_name, tool_input)
    is_json, value = _parse_json(body) if kind == "json" else (False, None)
    try:
        shape = content_shape(body, None, kind, value=value, parsed=is_json)
    except Exception:  # never let shape detection stop a result being stored
        shape = {"kind": "text"}
    # JSON Lines is checked line by line, so it beats "code" (lines ending in })
    if (shape["kind"] == "table" and kind not in ("json", "diff", "code")) or \
            (shape["kind"] == "jsonl" and kind not in ("json", "diff")):
        kind = shape["kind"]
    lines = result_view(text, tool_name, kind)
    return {
        "hint": shape_hint(shape),
        "kind": kind,
        "bytes": len(text.encode("utf-8")),
        "lines": len(lines),
        "tokens": int(len(text) / CHARS_PER_TOKEN),
        "keys": json_keys(value) if is_json else [],
        "outline": json_outline(value) if is_json else "",
        "source": describe_source(tool_name, tool_input, text, is_error, read_range),
        "view": lines,
    }


def build_stub(hid: str, info: Dict[str, Any], head_lines: int = 5, tail_lines: int = 3) -> str:
    """The text that replaces a tool result. Built only from the result and
    its tool call, so the same result always gives the same bytes."""
    lines = info["view"]
    out = [
        f"[Tool result kept outside the conversation by Switchyard as handle {hid}, to save context.]",
        f"Source: {info['source']}",
        f"Size: {info['lines']:,} lines, {info['bytes']:,} bytes, about {info['tokens']:,} tokens. "
        f"Kind: {info['kind']}.",
    ]
    if info.get("outline"):
        out.append(f"JSON: {info['outline']}")
    if info.get("hint"):
        out.append(info["hint"])
    head = lines[:head_lines]
    tail = lines[max(len(head), len(lines) - tail_lines):]
    out.append("First lines:")
    out.extend("  " + _clip(line) for line in head)
    if tail:
        skipped = len(lines) - len(head) - len(tail)
        if skipped > 0:
            out.append(f"  ... {skipped:,} more lines ...")
        out.append("Last lines:")
        out.extend("  " + _clip(line) for line in tail)
    out.append(
        f"Use the switchyard_handles tools (peek/slice/search/describe) with handle {hid} to read more."
    )
    return "\n".join(out)


def one_line_summary(info: Dict[str, Any]) -> str:
    return f"{_clip(info['source'], 200)} · {info['kind']} · {info['lines']:,} lines · {info['bytes']:,} bytes"


def _fresh_line(line: str) -> str:
    if len(line) <= FRESH_LINE_CHARS:
        return line
    return line[:FRESH_LINE_CHARS] + f" [... {len(line) - FRESH_LINE_CHARS:,} more characters]"


def build_fresh_stub(hid: str, info: Dict[str, Any], tool_name: Optional[str],
                     head_lines: int = 40, tail_lines: int = 10) -> str:
    """The text that replaces a large fresh tool result: its size, and its
    first and last lines verbatim (line numbers are the ones handles_slice
    takes). Built only from the result and its tool call, like build_stub."""
    lines = info["view"]
    head = lines[:max(0, head_lines)]
    tail = lines[max(len(head), len(lines) - max(0, tail_lines)):]
    skipped = len(lines) - len(head) - len(tail)
    total = len(lines)
    out = [
        f"{FRESH_STUB_PREFIX}{hid}, to save context; only its first and last lines are shown.]",
        f"Tool: {tool_name or 'tool'}",
        f"Source: {info['source']}",
        f"Size: {info['lines']:,} lines, about {info['tokens']:,} tokens, {info['bytes']:,} bytes.",
    ]
    if head:
        out.append(f"Lines 1-{len(head)} of {total}:")
        out.extend(_fresh_line(line) for line in head)
    if tail:
        if skipped > 0:
            out.append(f"[... lines {len(head) + 1}-{total - len(tail)} not shown ...]")
        out.append(f"Lines {total - len(tail) + 1}-{total} of {total}:")
        out.extend(_fresh_line(line) for line in tail)
    if info.get("hint"):
        out.append(info["hint"])
    out.append(
        f"For the rest, call handles_slice with handle {hid} and a line range, "
        f"or handles_search with handle {hid} and a pattern."
    )
    return "\n".join(out)


def is_stub_text(text: Optional[str]) -> bool:
    """True if a tool result's text is already a Switchyard stub."""
    if not text:
        return False
    start = text.lstrip()[:len(FRESH_STUB_PREFIX)]
    return start.startswith(STUB_PREFIX) or start.startswith(FRESH_STUB_PREFIX)


def is_fresh_stub(stub: Optional[str]) -> bool:
    return bool(stub) and stub.startswith(FRESH_STUB_PREFIX)


def is_handles_tool(name: Optional[str]) -> bool:
    """The switchyard_handles tools themselves, in any client's spelling
    (also when merged into a wrapped MCP server): their output is never put
    behind another handle."""
    name = str(name or "")
    return name.startswith(MCP_TOOL_PREFIXES) or bool(HANDLES_TOOL_RE.fullmatch(name))


def fresh_tool_matches(name: Optional[str], patterns: Iterable[str]) -> bool:
    """Whether a result of tool `name` may be compacted fresh: the name
    matches one of the globs (case-insensitively), or there are none."""
    if is_handles_tool(name):
        return False
    patterns = [p for p in patterns if p]
    if not patterns:
        return True
    lowered = str(name or "").lower()
    return bool(lowered) and any(fnmatch.fnmatchcase(lowered, p.lower()) for p in patterns)


# ---------------------------------------------------------------- request walking


def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def iter_tool_results(messages: List[dict]) -> List[Tuple[int, int, dict]]:
    """(message index, block index, block) of every tool_result, oldest first."""
    found = []
    for mi, message in enumerate(messages):
        content = _get(message, "content")
        if not isinstance(content, list):
            continue
        for bi, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                found.append((mi, bi, block))
    return found


def tool_uses(messages: List[dict]) -> Dict[str, Tuple[str, dict]]:
    """tool_use id -> (tool name, input)."""
    uses = {}
    for message in messages:
        content = _get(message, "content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                uses[block.get("id")] = (block.get("name") or "tool", block.get("input") or {})
    return uses


def _system_texts(request: dict) -> Iterable[str]:
    system = request.get("system")
    if isinstance(system, str):
        yield system
    elif isinstance(system, list):
        for part in system:
            if isinstance(part, dict):
                yield str(part.get("text") or "")
    for message in request.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    yield str(block.get("text") or "")


def has_handles_tools(request: dict) -> bool:
    """True if the model can call the switchyard_handles tools: they are in
    the request's tools, or named in the prompt (Claude Code lists MCP tools
    it defers, to be loaded by tool search, in a system reminder). The tools
    of an MCP server wrapped by switchyard_mcp_proxy.py count too."""
    for tool in request.get("tools") or []:
        name = str(_get(tool, "name") or "")
        if name.startswith(MCP_TOOL_PREFIXES) or HANDLES_TOOL_RE.fullmatch(name):
            return True
    return any(any(prefix in text for prefix in MCP_TOOL_PREFIXES) or HANDLES_TOOL_RE.search(text)
               for text in _system_texts(request))


def request_size(request: dict) -> Tuple[int, int]:
    """(characters of text, number of images) in an Anthropic request."""
    chars = images = 0
    for text in _system_texts({"system": request.get("system")}):
        chars += len(text)
    for message in request.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str):
            chars += len(content)
            continue
        for block in content or []:
            if not isinstance(block, dict):
                chars += len(str(block))
                continue
            kind = block.get("type")
            if kind == "text":
                chars += len(str(block.get("text") or ""))
            elif kind == "image":
                images += 1
            elif kind == "tool_use":
                chars += len(json.dumps(block.get("input") or {}, default=str)) + len(str(block.get("name")))
            elif kind == "tool_result":
                inner = block.get("content")
                if isinstance(inner, list):
                    for item in inner:
                        if isinstance(item, dict) and item.get("type") == "image":
                            images += 1
                        else:
                            chars += len(tool_result_text([item]) or "")
                else:
                    chars += len(tool_result_text(inner) or "")
            else:
                chars += len(json.dumps(block, default=str))
    chars += len(json.dumps(request.get("tools") or [], default=str))
    return chars, images


def estimate_tokens(request: dict, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Rough prompt tokens of an Anthropic request: characters / chars_per_token,
    plus a flat rate per image. server.py can pass its calibrated ratio."""
    chars, images = request_size(request)
    return int(chars / (chars_per_token or CHARS_PER_TOKEN)) + images * TOKENS_PER_IMAGE


def recency_protected_count(total: int, recency_window: int, recency_batch: int = 1) -> int:
    """How many of the newest `total` tool results stay verbatim. Results
    leave the window `recency_batch` at a time, so the rest changes rarely."""
    if total <= 0:
        return 0
    window = max(0, recency_window)
    batch = max(1, recency_batch)
    compressible = ((total - window) // batch) * batch if total > window else 0
    return total - compressible


# ---------------------------------------------------------------- store

PIN_MAX = 200  # most handles pinned at once (HandleStore.pin)


class PinLimitError(ValueError):
    """Pinning one more handle would go over PIN_MAX; the message says what to do."""


class HandleStore:
    """Full tool-result text by handle id, in SQLite. Rows unused for
    ttl_days are deleted when the store opens (and at most hourly after),
    except pinned ones (pin), which stay until they are unpinned.

    The gateway's store runs in the background (start_background): its
    writes are queued for one writer thread (switchyard_db.BatchWriter) and
    committed in batches, reads see what is queued (an overlay) before it is
    committed, and handles already replaced (whose stub never changes again)
    are kept in memory, so a request's handles pass reads the DB only for ids
    it has not seen and never waits for a commit or another connection's
    lock. batch() gathers a request's writes into one transaction."""

    COLUMNS = ("id", "tool_use_id", "tool_name", "tool_input", "kind", "bytes", "lines",
               "tokens", "keys", "summary", "stub", "session", "created", "last_seen", "replaced",
               "pinned")
    META_MAX = 20000  # Replaced handles' metadata kept in memory (background)

    def __init__(self, path: Optional[str] = None, ttl_days: Optional[float] = None,
                 clock: Callable[[], float] = time.time, background: bool = False):
        self.path = default_db_path(path)
        self.ttl_days = default_ttl_days() if ttl_days is None else ttl_days
        self._clock = clock
        self._lock = threading.Lock()
        self._mem_lock = threading.Lock()
        self._writer = None
        self._meta: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()  # id -> metadata, replaced ones
        self._overlay: Dict[str, Dict[str, Any]] = {}  # id -> full row queued, not yet committed
        self._collect = threading.local()  # batch(): this thread's writes being gathered
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._db = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS handles (
                    id TEXT PRIMARY KEY, tool_use_id TEXT, tool_name TEXT, tool_input TEXT,
                    kind TEXT, bytes INTEGER, lines INTEGER, tokens INTEGER, keys TEXT,
                    summary TEXT, stub TEXT, content TEXT, session TEXT,
                    created REAL, last_seen REAL, replaced INTEGER DEFAULT 0, pinned INTEGER DEFAULT 0)"""
            )
            columns = {row[1] for row in self._db.execute("PRAGMA table_info(handles)")}
            if "pinned" not in columns:  # A store made before pinning
                try:
                    self._db.execute("ALTER TABLE handles ADD COLUMN pinned INTEGER DEFAULT 0")
                except sqlite3.OperationalError as e:  # Another process added it first
                    if "duplicate column" not in str(e).lower():
                        raise
            self._db.execute("CREATE INDEX IF NOT EXISTS handles_last_seen ON handles (last_seen)")
            self._db.execute("CREATE INDEX IF NOT EXISTS handles_session ON handles (session)")
            # Counting pins must not read every row (content comes before pinned)
            self._db.execute("CREATE INDEX IF NOT EXISTS handles_pinned ON handles (pinned) WHERE pinned = 1")
            self._db.commit()
        self._last_prune = 0.0
        if background:
            self.start_background()
        self.prune()

    @property
    def background(self) -> bool:
        return self._writer is not None

    def start_background(self) -> None:
        """Queue writes for a writer thread from now on (the gateway's store)."""
        if self._writer is None:
            import switchyard_db  # Only the gateway's store needs it

            self._writer = switchyard_db.BatchWriter(self.path, "handles-writer")

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait for the queued writes (tests, shutdown); whether they landed in time."""
        return self._writer.flush(timeout) if self._writer is not None else True

    def writer_stats(self) -> Optional[Dict[str, Any]]:
        return self._writer.stats() if self._writer is not None else None

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        with self._lock:
            self._db.close()

    # Writes: run at once, gathered by batch(), or queued for the writer

    def _execute(self, sql: str, params, many: bool = False) -> int:
        """Run one write statement now or as part of the batch/queue; its
        rowcount when run now (-1 otherwise)."""
        ops = getattr(self._collect, "ops", None)
        if self._writer is not None:
            if ops is not None:
                ops.append((sql, params, many))
            else:
                self._writer.submit(_run_ops, [(sql, params, many)])
            return -1
        with self._lock:
            cursor = (self._db.executemany if many else self._db.execute)(sql, params)
            if ops is None:
                self._db.commit()
            else:
                ops.append(None)  # Committed with the batch
            return cursor.rowcount

    @contextmanager
    def batch(self):
        """Gather this thread's writes in the block and write them as one
        transaction at its end (queued for the writer in the background)."""
        if getattr(self._collect, "ops", None) is not None:
            yield  # Already gathering
            return
        ops: List[tuple] = []
        self._collect.ops = ops
        try:
            yield
        finally:
            # Written even when the block failed part way, as the writes made
            # one by one always were
            self._collect.ops = None
            if ops and self._writer is not None:
                ids = [params[0] for sql, params, many in ops if sql.startswith("INSERT")]
                self._writer.submit(_run_ops, ops, after=lambda _r: self._landed(ids))
            elif ops:
                with self._lock:
                    self._db.commit()

    def _landed(self, ids: List[str]) -> None:
        """Queued handles were committed: the DB has them now."""
        with self._mem_lock:
            for hid in ids:
                row = self._overlay.pop(hid, None)
                if row is not None and row.get("replaced"):
                    self._remember(hid, {k: row[k] for k in self.COLUMNS})

    def _remember(self, hid: str, meta: Dict[str, Any]) -> None:
        """Keep a replaced handle's metadata (caller holds _mem_lock)."""
        self._meta[hid] = meta
        self._meta.move_to_end(hid)
        while len(self._meta) > self.META_MAX:
            self._meta.popitem(last=False)

    def prune(self, ttl_days: Optional[float] = None) -> int:
        """Delete rows not seen for ttl_days (<= 0 keeps everything), except
        pinned ones. In the background it is queued, and returns -1."""
        ttl = self.ttl_days if ttl_days is None else ttl_days
        self._last_prune = self._clock()
        if not ttl or ttl <= 0:
            return 0
        cutoff = self._clock() - ttl * 86400
        sql = "DELETE FROM handles WHERE last_seen < ? AND NOT COALESCE(pinned, 0)"
        if self._writer is not None:
            with self._mem_lock:
                # Only a cache: a pinned row dropped here is read again from the DB
                for hid in [h for h, m in self._meta.items() if (m.get("last_seen") or 0) < cutoff]:
                    del self._meta[hid]
            self._writer.submit(_run_ops, [(sql, (cutoff,), False)])
            return -1
        with self._lock:
            cursor = self._db.execute(sql, (cutoff,))
            self._db.commit()
            return cursor.rowcount

    def maybe_prune(self, every_seconds: float = 3600) -> None:
        if self._clock() - self._last_prune >= every_seconds:
            self.prune()

    def _row(self, row: sqlite3.Row, content: bool = False) -> Dict[str, Any]:
        data = {k: row[k] for k in self.COLUMNS}
        data["replaced"] = bool(data["replaced"])
        data["pinned"] = bool(data["pinned"])
        data["keys"] = json.loads(data["keys"] or "[]")
        try:
            data["tool_input"] = json.loads(data["tool_input"] or "{}")
        except ValueError:
            pass
        if content:
            data["content"] = row["content"]
        return data

    def lookup(self, ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        """Metadata (no content) for the ids that exist."""
        ids = list(dict.fromkeys(ids))
        found: Dict[str, Dict[str, Any]] = {}
        if self._writer is not None:
            with self._mem_lock:
                for hid in ids:
                    row = self._overlay.get(hid)
                    if row is not None:
                        found[hid] = {k: row[k] for k in self.COLUMNS}
                    elif hid in self._meta:
                        self._meta.move_to_end(hid)
                        found[hid] = dict(self._meta[hid])
        missing = [hid for hid in ids if hid not in found]
        read: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for start in range(0, len(missing), 500):
                chunk = missing[start:start + 500]
                marks = ",".join("?" * len(chunk))
                for row in self._db.execute(
                        f"SELECT {', '.join(self.COLUMNS)} FROM handles WHERE id IN ({marks})", chunk):
                    read[row["id"]] = self._row(row)
        if self._writer is not None and read:
            with self._mem_lock:
                for hid, row in read.items():
                    if hid in self._overlay:  # Queued meanwhile: newer than the DB
                        read[hid] = {k: self._overlay[hid][k] for k in self.COLUMNS}
                    elif row.get("replaced"):
                        self._remember(hid, dict(row))
        found.update(read)
        return {hid: found[hid] for hid in ids if hid in found}

    def get(self, hid: str) -> Optional[Dict[str, Any]]:
        """Everything about one handle, content included, or None."""
        hid = normalize_handle(hid)
        if self._writer is not None:
            with self._mem_lock:
                row = self._overlay.get(hid)
                if row is not None:
                    return dict(row)
        with self._lock:
            row = self._db.execute(
                f"SELECT {', '.join(self.COLUMNS)}, content FROM handles WHERE id = ?", (hid,)).fetchone()
        return self._row(row, content=True) if row else None

    def put(self, record: Dict[str, Any]) -> bool:
        """Insert a handle unless it exists; True if inserted (in the
        background: unless this store knows it, from the DB or queued)."""
        now = self._clock()
        params = (
            record["id"], record.get("tool_use_id"), record.get("tool_name"),
            json.dumps(record.get("tool_input") or {}, ensure_ascii=False, default=str),
            record.get("kind"), record.get("bytes"), record.get("lines"), record.get("tokens"),
            json.dumps(record.get("keys") or [], ensure_ascii=False), record.get("summary"),
            record.get("stub"), record["content"], record.get("session"),
            now, now, int(bool(record.get("replaced"))),
        )
        sql = ("INSERT OR IGNORE INTO handles (id, tool_use_id, tool_name, tool_input, kind, bytes,"
               " lines, tokens, keys, summary, stub, content, session, created, last_seen, replaced)"
               " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
        if self._writer is not None:
            hid = record["id"]
            with self._mem_lock:
                if hid in self._overlay or hid in self._meta:
                    return False
                self._overlay[hid] = {
                    "id": hid, "tool_use_id": record.get("tool_use_id"), "tool_name": record.get("tool_name"),
                    "tool_input": record.get("tool_input") or {}, "kind": record.get("kind"),
                    "bytes": record.get("bytes"), "lines": record.get("lines"), "tokens": record.get("tokens"),
                    "keys": list(record.get("keys") or []), "summary": record.get("summary"),
                    "stub": record.get("stub"), "session": record.get("session"), "created": now,
                    "last_seen": now, "replaced": bool(record.get("replaced")), "pinned": False,
                    "content": record["content"],
                }
            if getattr(self._collect, "ops", None) is not None:
                self._collect.ops.append((sql, params, False))
            else:
                self._writer.submit(_run_ops, [(sql, params, False)], after=lambda _r: self._landed([hid]))
            return True
        return self._execute(sql, params) > 0

    def _note(self, ids: Iterable[str], **changes) -> None:
        """In the background: what this store knows of `ids` changes now."""
        if self._writer is None:
            return
        with self._mem_lock:
            for hid in ids:
                if hid in self._overlay:
                    self._overlay[hid].update(changes)
                elif hid in self._meta:
                    self._meta[hid].update(changes)

    def mark_replaced(self, ids: Iterable[str]) -> None:
        ids = list(ids)
        if not ids:
            return
        self._note(ids, replaced=True)
        self._execute("UPDATE handles SET replaced = 1 WHERE id = ?", [(i,) for i in ids], many=True)

    def set_stub(self, hid: str, stub: str) -> None:
        """Mark a handle replaced with this stub (a stored but not yet
        replaced result that is now compacted fresh), so later requests send
        the same bytes."""
        self._note([hid], stub=stub, replaced=True)
        self._execute("UPDATE handles SET stub = ?, replaced = 1 WHERE id = ?", (stub, hid))

    def touch(self, ids: Iterable[str], min_age: float = 3600) -> None:
        """Refresh last_seen (what the TTL counts from), at most hourly per row."""
        ids = list(ids)
        if not ids:
            return
        now = self._clock()
        if self._writer is not None:
            with self._mem_lock:
                for hid in ids:
                    meta = self._overlay.get(hid) or self._meta.get(hid)
                    if meta is not None and (meta.get("last_seen") or 0) < now - min_age:
                        meta["last_seen"] = now
        self._execute("UPDATE handles SET last_seen = ? WHERE id = ? AND last_seen < ?",
                      [(now, i, now - min_age) for i in ids], many=True)

    def pin(self, hid: str, pinned: bool = True) -> Optional[Dict[str, Any]]:
        """Pin a handle, so prune() keeps it whatever its age, or unpin it.
        Either way last_seen becomes now, so an unpinned handle gets a whole
        TTL from the unpinning. Returns {"id", "pinned", "changed", "count"}
        (count: handles pinned now), or None if there is no such handle.
        Pinning one more than PIN_MAX raises PinLimitError."""
        hid = normalize_handle(hid)
        self.flush()  # A queued handle must be in the DB to be updated
        now = self._clock()
        with self._lock:
            if not self._db.in_transaction:
                self._db.execute("BEGIN IMMEDIATE")  # Count and update with no other writer between
            try:
                row = self._db.execute("SELECT pinned FROM handles WHERE id = ?", (hid,)).fetchone()
                if row is None:
                    self._db.rollback()
                    return None
                was = bool(row[0])
                count = self._db.execute("SELECT COUNT(*) FROM handles WHERE pinned = 1").fetchone()[0]
                if pinned and not was and count >= PIN_MAX:
                    raise PinLimitError(
                        f"Can't pin {hid}: {count:,} handles are pinned already, and {PIN_MAX:,} is the "
                        "most allowed. Unpin one first (handles_pin with pin false); handles_list and "
                        "handles_describe say which are pinned.")
                self._db.execute("UPDATE handles SET pinned = ?, last_seen = ? WHERE id = ?",
                                 (int(bool(pinned)), now, hid))
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise
        self._note([hid], pinned=bool(pinned), last_seen=now)
        changed = was != bool(pinned)
        if changed:
            count += 1 if pinned else -1
        return {"id": hid, "pinned": bool(pinned), "changed": changed, "count": count}

    def recent(self, session: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Newest first, without content."""
        query = f"SELECT {', '.join(self.COLUMNS)} FROM handles"
        params: List[Any] = []
        if session:
            query += " WHERE session = ?"
            params.append(session)
        query += " ORDER BY created DESC, id LIMIT ?"
        params.append(max(1, int(limit)))
        with self._lock:
            rows = [self._row(row) for row in self._db.execute(query, params)]
        if self._writer is None:
            return rows
        with self._mem_lock:
            queued = [{k: row[k] for k in self.COLUMNS} for row in self._overlay.values()
                      if not session or row.get("session") == session]
        if not queued:
            return rows
        seen = {row["id"] for row in rows}
        rows += [row for row in queued if row["id"] not in seen]
        rows.sort(key=lambda r: (-(r.get("created") or 0), r["id"]))
        return rows[:max(1, int(limit))]

    def delete_session(self, session: str) -> int:
        self.flush()
        with self._mem_lock:
            for store in (self._meta, self._overlay):
                for hid in [h for h, m in store.items() if m.get("session") == session]:
                    del store[hid]
        with self._lock:
            cursor = self._db.execute("DELETE FROM handles WHERE session = ?", (session,))
            self._db.commit()
            return cursor.rowcount


def _run_ops(db, ops: List[tuple]) -> None:
    """A writer-thread job: the queued statements, in order (no commit)."""
    for sql, params, many in ops:
        (db.executemany if many else db.execute)(sql, params)


class RemoteStoreError(OSError):
    """The Switchyard gateway could not be used; the message says why, in
    words fit to show the model or the user. An OSError, so apply_handles
    and other callers that handle store errors handle this one too."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: it would carry the x-api-key to another URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _epoch(value: Any) -> Optional[float]:
    """A created_at value (epoch seconds, or an ISO 8601 string) as epoch seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        import datetime
        return datetime.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


class RemoteHandleStore:
    """HandleStore's interface over HTTP to a Switchyard gateway (see the
    integration note, point 6). Reads come back in the same shape as
    HandleStore rows ("content" holds the text); the gateway decides ids,
    stubs and expiry. Transport problems raise RemoteStoreError with a clear
    message; a handle the gateway doesn't have is None, as locally."""

    TIMEOUT = 10.0

    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 ttl_days: Optional[float] = None, timeout: Optional[float] = None):
        self.base_url = str(base_url).strip().rstrip("/")
        self.path = self.base_url  # what error messages call "the store"
        self.api_key = api_key
        self.ttl_days = default_ttl_days() if ttl_days is None else ttl_days
        self.timeout = self.TIMEOUT if timeout is None else timeout
        self._opener = urllib.request.build_opener(_NoRedirect())

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "RemoteHandleStore":
        """From SWITCHYARD_URL and SWITCHYARD_API_KEY, each read from the environment, else from the
        gateway file (switchyard_gateway_client.gateway_settings)."""
        url, key = gateway_client.gateway_settings(env)
        return cls(url or "", key)

    def _request(self, method: str, path: str, body: Optional[dict] = None,
                 missing_ok: bool = False) -> Optional[Any]:
        url = self.base_url + path
        headers = {"Accept": "application/json", "User-Agent": "switchyard-handles"}
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["x-api-key"] = self.api_key
        where = f"the Switchyard gateway at {self.base_url}"
        try:
            with self._opener.open(urllib.request.Request(url, data=data, headers=headers, method=method),
                                   timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 and missing_ok:
                return None
            if e.code in (401, 403):
                raise RemoteStoreError(
                    f"{where} refused the request (HTTP {e.code}); check SWITCHYARD_API_KEY.") from None
            detail = ""
            try:
                detail = e.read(300).decode("utf-8", "replace").strip()
            except Exception:
                pass
            raise RemoteStoreError(
                f"{where} answered HTTP {e.code} to {method} {path.split('?')[0]}"
                + (f": {detail}" if detail else ".")) from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            reason = getattr(e, "reason", None) or e
            raise RemoteStoreError(f"Can't reach {where}: {reason}. Is SWITCHYARD_URL right and the "
                                   "gateway running?") from None
        try:
            return json.loads(raw.decode("utf-8") or "null")
        except ValueError:
            raise RemoteStoreError(f"{where} sent a response that isn't JSON.") from None

    def _row(self, record: Dict[str, Any], content: bool = False) -> Dict[str, Any]:
        text = record.get("text")
        row = {
            "id": record.get("id"),
            "tool_use_id": record.get("tool_use_id"),
            "tool_name": record.get("tool_name"),
            "tool_input": record.get("tool_input") or {},
            "kind": record.get("kind"),
            "bytes": record.get("bytes"),
            "lines": record.get("lines"),
            "tokens": record.get("tokens"),
            "keys": record.get("keys") or [],
            "summary": record.get("summary"),
            "stub": record.get("stub"),
            "session": record.get("session"),
            "created": _epoch(record.get("created_at", record.get("created"))),
            "last_seen": _epoch(record.get("last_seen")),
            "replaced": record.get("replaced"),  # None: the gateway doesn't say
            "pinned": record.get("pinned"),  # None: the gateway doesn't say
        }
        if isinstance(text, str):
            if row["tokens"] is None:
                row["tokens"] = int(len(text) / CHARS_PER_TOKEN)
            if not row["keys"] and row["kind"] == "json":
                row["keys"] = json_keys(_parse_json(text)[1])
        if content:
            row["content"] = text if isinstance(text, str) else ""
        return row

    def get(self, hid: str) -> Optional[Dict[str, Any]]:
        hid = normalize_handle(hid)
        if not re.fullmatch(r"[a-z0-9_]{1,64}", hid):
            return None
        record = self._request("GET", "/handles/" + urllib.parse.quote(hid), missing_ok=True)
        if not isinstance(record, dict):
            return None
        return self._row(record, content=True)

    def lookup(self, ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        found = {}
        for hid in dict.fromkeys(ids):
            row = self.get(hid)
            if row is not None:
                row.pop("content", None)
                found[row["id"] or hid] = row
        return found

    def recent(self, session: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        query = {"limit": max(1, int(limit))}
        if session:
            query["session"] = session
        data = self._request("GET", "/handles?" + urllib.parse.urlencode(query))
        records = data.get("handles") if isinstance(data, dict) else None
        if not isinstance(records, list):
            raise RemoteStoreError(f"the Switchyard gateway at {self.base_url} sent no handle list.")
        return [self._row(r) for r in records if isinstance(r, dict)]

    def create(self, text: str, tool_name: Optional[str] = None, tool_use_id: Optional[str] = None,
               session: Optional[str] = None, tool_input: Optional[dict] = None) -> Dict[str, Any]:
        """POST the text; the gateway's {"id", "stub", "created"}."""
        body: Dict[str, Any] = {"tool_name": tool_name, "text": text}
        if tool_use_id:
            body["tool_use_id"] = tool_use_id
        if session:
            body["session"] = session
        if tool_input:
            body["tool_input"] = tool_input
        data = self._request("POST", "/handles", body)
        if not isinstance(data, dict) or not data.get("id") or not isinstance(data.get("stub"), str):
            raise RemoteStoreError(f"the Switchyard gateway at {self.base_url} sent no handle id and stub.")
        return {"id": data["id"], "stub": data["stub"], "created": bool(data.get("created"))}

    def put(self, record: Dict[str, Any]) -> bool:
        """Store a record built for HandleStore; True if the gateway created it."""
        return self.create(record["content"], record.get("tool_name"), record.get("tool_use_id"),
                           record.get("session"), record.get("tool_input"))["created"]

    def pin(self, hid: str, pinned: bool = True) -> Optional[Dict[str, Any]]:
        """The gateway's /handles API has no way to pin a handle."""
        raise RemoteStoreError(
            f"The Switchyard gateway at {self.base_url} can't pin handles: its /handles API has no pin "
            "route. Pinning works with a local handle store (SWITCHYARD_URL unset).")

    # The gateway keeps its own last-use times and expiry: nothing to do here.
    def close(self) -> None:
        pass

    def prune(self, ttl_days: Optional[float] = None) -> int:
        return 0

    def maybe_prune(self, every_seconds: float = 3600) -> None:
        pass

    def mark_replaced(self, ids: Iterable[str]) -> None:
        pass

    def set_stub(self, hid: str, stub: str) -> None:
        pass

    def touch(self, ids: Iterable[str], min_age: float = 3600) -> None:
        pass


AnyStore = Any  # HandleStore or RemoteHandleStore

_stores: Dict[Tuple[str, float], HandleStore] = {}
_stores_lock = threading.Lock()


def open_store(path: Optional[str] = None, ttl_days: Optional[float] = None,
               remote: Optional[bool] = None, background: bool = False) -> AnyStore:
    """The handle store for this process. With SWITCHYARD_URL set (and no
    explicit path), a RemoteHandleStore for that gateway; else one shared
    SQLite HandleStore per (path, ttl). remote=False always opens SQLite
    (the gateway itself must, or it would call itself). `background`: its
    writes go to a writer thread from now on (the running gateway's store).
    SWITCHYARD_URL and SWITCHYARD_API_KEY come from the environment, else
    from the gateway file (switchyard_gateway_client.gateway_settings)."""
    if remote is False:
        url, key = "", None
    else:
        url, key = gateway_client.gateway_settings()
        url = url or ""
    if remote is None:
        remote = bool(url) and path is None
    if remote:
        if not url:
            raise RemoteStoreError("SWITCHYARD_URL is not set, so there is no gateway to use.")
        with _stores_lock:
            store = _stores.get(("remote", url, key))
            if store is None:
                store = _stores[("remote", url, key)] = RemoteHandleStore(url, key, ttl_days)
            return store
    resolved = default_db_path(path)
    ttl = default_ttl_days() if ttl_days is None else ttl_days
    with _stores_lock:
        store = _stores.get((resolved, ttl))
        if store is None:
            store = _stores[(resolved, ttl)] = HandleStore(resolved, ttl, background=background)
        elif background and not store.background:
            store.start_background()
        return store


def store_tool_result(
    text: str,
    *,
    tool_name: Optional[str],
    tool_use_id: Optional[str] = None,
    tool_input: Optional[dict] = None,
    session: Optional[str] = None,
    store: Optional[AnyStore] = None,
    settings: Optional[HandleSettings] = None,
) -> Dict[str, Any]:
    """Store one tool result's full text and return {"id", "stub", "created"}.
    The stub is built exactly as apply_handles builds it (and reused if the
    handle exists), so the gateway's POST /handles and the MCP proxy give the
    same bytes. A RemoteHandleStore hands the work to the gateway."""
    settings = settings or HandleSettings()
    store = store or open_store(settings.db_path, settings.ttl_days)
    if isinstance(store, RemoteHandleStore):
        return store.create(text, tool_name, tool_use_id, session, tool_input)
    hid = handle_id(tool_use_id or "", text)
    row = store.lookup([hid]).get(hid)
    if row and row.get("stub"):
        with _batch(store):
            if not row.get("replaced"):
                store.mark_replaced([hid])
            store.touch([hid])
        return {"id": hid, "stub": row["stub"], "created": False}
    block = {"tool_use_id": tool_use_id or ""}
    record = _make_record(hid, block, text, {block["tool_use_id"]: (tool_name, tool_input or {})},
                          settings, session)
    record["replaced"] = True
    return {"id": hid, "stub": record["stub"], "created": store.put(record)}


# ---------------------------------------------------------------- apply


def _new_stats() -> Dict[str, Any]:
    return {
        "handles_created": 0,      # new rows written to the store
        "results_replaced": 0,     # tool results sent as stubs in this request
        "tokens_saved": 0,         # estimated, chars / 4
        "replaced_ids": [],        # handle ids sent as stubs in this request
        "new_replaced_ids": [],    # replaced for the first time: earlier messages changed
        "fresh_replaced": 0,       # of results_replaced, those sent as large fresh stubs
        "fresh_tokens_saved": 0,   # of tokens_saved, by those
        "tool_results": 0,
        "request_tokens": 0,
        "threshold_tokens": 0,
        "gate_open": False,
        "skipped": None,
    }


def _make_record(hid: str, block: dict, text: str, uses: Dict[str, Tuple[str, dict]],
                 settings: HandleSettings, session: Optional[str], fresh: bool = False) -> Dict[str, Any]:
    tool_name, tool_input = uses.get(block.get("tool_use_id"), (None, {}))
    info = analyse_result(text, tool_name, tool_input, bool(block.get("is_error")))
    if fresh:
        stub = build_fresh_stub(hid, info, tool_name, settings.fresh_head_lines, settings.fresh_tail_lines)
    else:
        stub = build_stub(hid, info, settings.head_lines, settings.tail_lines)
    return {
        "id": hid, "tool_use_id": block.get("tool_use_id"), "tool_name": tool_name,
        "tool_input": tool_input, "kind": info["kind"], "bytes": info["bytes"],
        "lines": info["lines"], "tokens": info["tokens"], "keys": info["keys"],
        "summary": one_line_summary(info),
        "stub": stub,
        "content": text, "session": session,
    }


def _fresh_eligible(block: dict, text: str, uses: Dict[str, Tuple[str, dict]],
                    settings: HandleSettings) -> bool:
    """Whether a tool result is compacted fresh: fresh compaction is on, the
    text is over fresh_min_tokens (at a fixed 4 characters per token, so the
    decision never changes between requests), it is not an error, and its
    tool matches fresh_tools (never the handles tools themselves)."""
    if settings.fresh_min_tokens <= 0 or block.get("is_error"):
        return False
    if len(text) <= settings.fresh_min_tokens * CHARS_PER_TOKEN:
        return False
    tool_name = uses.get(block.get("tool_use_id"), (None, {}))[0]
    return fresh_tool_matches(tool_name, settings.fresh_tools)


def _large_results(messages: List[dict], settings: HandleSettings):
    """(position, message index, block index, block, text, handle id) of every
    text tool result over min_result_tokens; position counts all results."""
    min_chars = settings.min_result_tokens * CHARS_PER_TOKEN
    large = []
    for position, (mi, bi, block) in enumerate(iter_tool_results(messages)):
        text = tool_result_text(block.get("content"))
        if text is None or len(text) <= min_chars or is_stub_text(text):
            continue
        large.append((position, mi, bi, block, text, handle_id(block.get("tool_use_id") or "", text)))
    return large


def register_tool_results(
    request: Any,
    *,
    settings: Optional[HandleSettings] = None,
    store: Optional[HandleStore] = None,
    session: Optional[str] = None,
) -> List[str]:
    """Store every large tool result in `request` without replacing anything,
    so the MCP tools can already read it. Returns the handle ids. (apply_handles
    stores what it replaces itself; this is for callers that want more.)"""
    settings = settings or HandleSettings()
    if hasattr(request, "model_dump"):
        request = request.model_dump(exclude_none=True)
    messages = request.get("messages") or []
    large = _large_results(messages, settings)
    if not large:
        return []
    store = store or open_store(settings.db_path, settings.ttl_days)
    known = store.lookup(hid for *_, hid in large)
    uses = tool_uses(messages)
    with _batch(store):
        for _, _, _, block, text, hid in large:
            if hid not in known:
                store.put(_make_record(hid, block, text, uses, settings, session))
    return [hid for *_, hid in large]


def apply_handles(
    request: Any,
    *,
    model_window: int,
    settings: Optional[HandleSettings] = None,
    store: Optional[HandleStore] = None,
    estimator: Optional[Callable[[dict], int]] = None,
    session: Optional[str] = None,
) -> Tuple[dict, Dict[str, Any]]:
    """Replace large tool results in an Anthropic Messages request (a dict,
    or a pydantic model) with handle stubs. Returns (request, stats); the
    input is never modified, and the request comes back unchanged when
    nothing is replaced or the store can't be used.

    Two passes, fresh first, both behind the same gate (require_tools):
      - fresh: a result over fresh_min_tokens from a tool matching
        fresh_tools (not an error) is replaced whatever its age and the
        request's size, by a stub with its first and last lines;
      - old: a result over min_result_tokens is replaced when the request is
        over context_fraction * model_window tokens and the result sits
        before the recency window (which releases results recency_batch at a
        time).
    Error results (is_error) follow one rule in both passes: never
    compacted while recent, stubbed like any other result once old. The
    fresh pass skips them because the model has to read the failure it just
    got in full; once a result has left the recency window its turn has
    been answered, and a failed command's log (Claude Code marks every
    non-zero Bash exit as an error, so a failing test run is one) is as
    stale as any other old output, often the largest. Their stub's Source
    line says the call failed. (The OpenAI shapes carry no is_error, so
    there every result is treated as a success.)
    Replacements are sticky: a result replaced once (by either pass) keeps
    its stored stub on every later request, and a result that already is a
    stub is never replaced again."""
    settings = settings or HandleSettings()
    stats = _new_stats()
    if hasattr(request, "model_dump"):
        request = request.model_dump(exclude_none=True)
    messages = request.get("messages") or []
    results = iter_tool_results(messages)
    total = len(results)
    stats["tool_results"] = total
    if not total:
        stats["skipped"] = "no_tool_results"
        return request, stats
    if settings.require_tools and not has_handles_tools(request):
        stats["skipped"] = "handles_tools_missing"
        return request, stats

    stats["request_tokens"] = int((estimator or estimate_tokens)(request))
    fraction = settings.context_fraction
    stats["threshold_tokens"] = int(fraction * model_window) if fraction > 0 else 0
    stats["gate_open"] = fraction <= 0 or stats["request_tokens"] > stats["threshold_tokens"]
    compressible = total - recency_protected_count(total, settings.recency_window, settings.recency_batch)

    uses = tool_uses(messages)
    min_chars = settings.min_result_tokens * CHARS_PER_TOKEN
    # (position, message index, block index, block, text, handle id, large, fresh)
    candidates = []
    for position, (mi, bi, block) in enumerate(results):
        text = tool_result_text(block.get("content"))
        if text is None or is_stub_text(text):
            continue
        large = len(text) > min_chars
        fresh = _fresh_eligible(block, text, uses, settings)
        if large or fresh:
            candidates.append((position, mi, bi, block, text,
                               handle_id(block.get("tool_use_id") or "", text), large, fresh))
    if not candidates:
        stats["skipped"] = "no_large_results"
        return request, stats
    try:
        store = store or open_store(settings.db_path, settings.ttl_days)
        store.maybe_prune()
        known = store.lookup(item[5] for item in candidates)
    except (OSError, sqlite3.Error) as e:
        stats["skipped"] = f"store_error: {e}"
        return request, stats

    chosen = []
    for item in candidates:
        position, _, _, _, _, hid, large, fresh = item
        row = known.get(hid) or {}
        if row.get("replaced"):
            chosen.append((item, "sticky"))      # its stored stub, whichever pass made it
        elif fresh:
            chosen.append((item, "fresh"))       # fresh before old
        elif large and stats["gate_open"] and position < compressible:
            chosen.append((item, "old"))
    if not chosen:
        stats["skipped"] = "nothing_to_replace"
        return request, stats

    replacements: Dict[Tuple[int, int], str] = {}
    created, newly = [], []
    try:
        with _batch(store):  # One transaction for this request's writes
            _write_chosen(store, chosen, known, uses, settings, session, stats, replacements, created, newly)
    except (OSError, sqlite3.Error) as e:
        stats.update(_new_stats(), tool_results=total, skipped=f"store_error: {e}")
        return request, stats
    return _with_replacements(request, messages, replacements, stats, created, newly)


def _batch(store: Any):
    """store.batch() when the store has one (HandleStore), else nothing."""
    batch = getattr(store, "batch", None)
    return batch() if batch is not None else nullcontext()


def _write_chosen(store, chosen, known, uses, settings, session, stats, replacements, created, newly) -> None:
    """apply_handles' writes: new handles stored, stubs set, replaced ids
    marked and touched; the stubs to send gathered in `replacements`."""
    for (_, mi, bi, block, text, hid, _, _), how in chosen:
        row = known.get(hid)
        fresh = how == "fresh"
        if row is None:
            record = _make_record(hid, block, text, uses, settings, session, fresh=fresh)
            record["replaced"] = True
            if store.put(record):
                created.append(hid)
            stub = record["stub"]
            newly.append(hid)
        elif fresh and not is_fresh_stub(row.get("stub")):
            # Stored earlier (register_tool_results) but never replaced:
            # store the fresh stub, so later requests send the same bytes
            stub = _make_record(hid, block, text, uses, settings, session, fresh=True)["stub"]
            store.set_stub(hid, stub)
            newly.append(hid)
        else:
            stub = row.get("stub") or _make_record(hid, block, text, uses, settings, session, fresh=fresh)["stub"]
            if not row.get("replaced"):
                newly.append(hid)
        if len(stub) >= len(text):
            continue
        replacements[(mi, bi)] = stub
        stats["replaced_ids"].append(hid)
        saved = int((len(text) - len(stub)) / CHARS_PER_TOKEN)
        stats["tokens_saved"] += saved
        if is_fresh_stub(stub):
            stats["fresh_replaced"] += 1
            stats["fresh_tokens_saved"] += saved
    store.mark_replaced([h for h in newly if h in stats["replaced_ids"] and h not in created])
    store.touch(stats["replaced_ids"])


def _with_replacements(request, messages, replacements, stats, created, newly):
    """apply_handles' result: the request with the stubs in place, and its stats."""
    new_messages = list(messages)
    for (mi, bi), stub in replacements.items():
        message = new_messages[mi]
        if message is messages[mi]:
            message = dict(message)
            message["content"] = list(message["content"])
            new_messages[mi] = message
        block = dict(message["content"][bi])
        block["content"] = stub
        message["content"][bi] = block
    new_request = dict(request)
    new_request["messages"] = new_messages
    stats["handles_created"] = len(created)
    stats["results_replaced"] = len(stats["replaced_ids"])
    stats["new_replaced_ids"] = [h for h in newly if h in stats["replaced_ids"]]
    return new_request, stats


# ---------------------------------------------------------------- content shape
#
# What a stored result holds (JSON, JSON Lines, a table, a log, plain text),
# and the queries the handles tools run on it: JSON paths, and table queries
# in an in-memory SQLite database. Everything is bounded: detection reads a
# sample from the start of the text; parsing, loading and querying stop at
# the caps below, and an answer cut short by a cap says so. docs/handles.md
# documents the verbs, the path syntax and the where grammar.

SHAPE_SAMPLE_CHARS = 128 * 1024    # detection reads at most this much
SHAPE_SAMPLE_LINES = 300           # ... and at most this many non-empty lines
JSON_MAX_CHARS = 16 * 1024 * 1024  # larger JSON is not parsed for shape or path
TABLE_MAX_CHARS = 32 * 1024 * 1024  # table queries parse at most this much text
TABLE_MAX_ROWS = 200000
TABLE_MAX_COLUMNS = 100
TABLE_MAX_CELLS = 4000000          # rows are capped to keep rows * columns under this
QUERY_SECONDS = 5.0                # loading plus one query
QUERY_DEFAULT_LIMIT = 50
QUERY_MAX_LIMIT = 500
PATH_MAX_SHOWN = 100
PATH_MAX_VISITS = 1000000          # nodes a ..key search may visit
CELL_CHARS = 200                   # longer cells are cut in answers
WHERE_MAX_CHARS = 2000
WHERE_MAX_DEPTH = 20
WHERE_MAX_TERMS = 64

_INT_RE = re.compile(r"[-+]?(?:0|[1-9]\d{0,17})\Z")
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d+|\.\d+|\d+(?=[eE]))(?:[eE][-+]?\d+)?\Z")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\Z")
_TIME_RE = re.compile(r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\Z")
_HEX_RE = re.compile(r"(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,64}\Z")
_MODE_RE = re.compile(r"[-dlcbps][-rwxsStT]{9}[@+.]?\Z")
_SIZE_RE = re.compile(r"\d+(?:\.\d+)?[KMGTP]i?B?\Z", re.I)
_WORD_RE = re.compile(r"[A-Za-z_][\w.@:/-]{0,39}\Z")
_STRONG_CLASSES = ("int", "float", "bool", "date", "datetime", "time", "hex", "mode", "size")


def cell_type(value: Any) -> str:
    """The type of one table cell: null, int, float, bool, date, datetime,
    time, hex, mode (ls permissions), size (4.0K), or text."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, (dict, list)):
        return "json"
    v = str(value).strip()
    if not v:
        return "null"
    if _INT_RE.match(v):
        return "int"
    if _FLOAT_RE.match(v):
        return "float"
    if v.lower() in ("true", "false"):
        return "bool"
    if _DATE_RE.match(v):
        return "datetime" if len(v) > 10 else "date"
    if _TIME_RE.match(v):
        return "time"
    if _HEX_RE.match(v):
        return "hex"
    if _MODE_RE.match(v):
        return "mode"
    if _SIZE_RE.match(v):
        return "size"
    return "text"


def column_type(values: Iterable[Any]) -> str:
    """One type for a column: the cells' common type (int and float give
    float) when 95% of the non-empty cells have it, else text."""
    counts: Dict[str, int] = {}
    for value in values:
        kind = cell_type(value)
        if kind != "null":
            counts[kind] = counts.get(kind, 0) + 1
    total = sum(counts.values())
    if not total:
        return "empty"
    numbers = counts.get("int", 0) + counts.get("float", 0)
    if numbers >= 0.95 * total:
        return "float" if counts.get("float") else "int"
    top = max(counts, key=lambda k: counts[k])
    return top if counts[top] >= 0.95 * total else "text"


def content_body(text: str, tool_name: Optional[str] = None) -> str:
    """The text a result's shape is read from: a Read result without its
    line-number prefixes, else the text itself."""
    if tool_name == "Read":
        return strip_read_line_numbers(text)[0]
    return text


def _sample(body: str) -> Tuple[str, bool]:
    """(start of body, whether that is all of it), cut at a line end."""
    if len(body) <= SHAPE_SAMPLE_CHARS:
        return body, True
    cut = body.rfind("\n", 0, SHAPE_SAMPLE_CHARS)
    return body[:cut if cut > 0 else SHAPE_SAMPLE_CHARS], False


def _nonempty_lines(text: str, limit: Optional[int] = None) -> List[str]:
    out = []
    for line in text.split("\n"):
        line = line.rstrip()
        if line.strip():
            out.append(line)
            if limit is not None and len(out) >= limit:
                break
    return out


# ---- markdown pipe tables

_MD_SEP_RE = re.compile(r"\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?\s*\Z")


def _md_cells(line: str) -> List[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", s)]


def _markdown_spec(lines: List[str]) -> Optional[Dict[str, Any]]:
    """A markdown table that makes up most of the lines."""
    for i in range(min(len(lines) - 1, 5)):
        if "|" not in lines[i] or not _MD_SEP_RE.match(lines[i + 1]):
            continue
        header = _md_cells(lines[i])
        if len(header) < 2 or len(_md_cells(lines[i + 1])) != len(header):
            continue
        body = [line for line in lines[i + 2:] if "|" in line]
        fit = [line for line in body if len(_md_cells(line)) == len(header)]
        if len(fit) < 2 or (len(fit) + 2) < 0.6 * len(lines):
            continue
        return {"format": "markdown", "header_line": lines[i], "columns": header}
    return None


def _markdown_rows(body: str, spec: Dict[str, Any]):
    started = seen_header = False
    width = len(spec["columns"])
    for line in body.split("\n"):
        if not line.strip():
            continue
        if not started:
            if seen_header and _MD_SEP_RE.match(line):
                started = True
            else:
                seen_header = line.rstrip() == spec["header_line"]
            continue
        cells = _md_cells(line) if "|" in line else []
        yield cells if len(cells) == width else None


# ---- delimited text (CSV, TSV, ...)

_DELIMITERS = (("\t", "tsv"), (",", "csv"), (";", "semicolon-separated"), ("|", "pipe-separated"))
_HEADER_BAD_RE = re.compile(r"[{}\[\]<>\\=]")


def _header_ok(cells: List[str]) -> bool:
    named = [c for c in cells if c]
    if len(named) < 0.9 * len(cells) or len(named) < 2:
        return False
    for cell in named:
        if len(cell) > 64 or _HEADER_BAD_RE.search(cell) or cell.count("(") != cell.count(")"):
            return False
        if cell_type(cell) not in ("text", "hex"):
            return False
    return len({c.lower() for c in named}) == len(named)


def _delimited_spec(sample: str, complete: bool) -> Optional[Dict[str, Any]]:
    first = sample.lstrip("\n").split("\n", 1)[0]
    best = None
    for delimiter, fmt in _DELIMITERS:
        if delimiter not in first:
            continue
        records = []
        try:
            for record in csv.reader(io.StringIO(sample), delimiter=delimiter):
                if any(c.strip() for c in record):
                    records.append(record)
                    if len(records) > SHAPE_SAMPLE_LINES:
                        break
        except csv.Error:
            continue
        if not complete and records:
            records = records[:-1]  # may be cut short by the sample
        if len(records) < 3:
            continue
        width = len(records[0])
        if width < 2:
            continue
        data = records[1:]
        fit = [r for r in data if len(r) == width]
        if len(fit) < 2 or len(fit) < 0.9 * len(data):
            continue
        cells = [c for r in fit for c in r]
        if sum(len(c) for c in cells) / max(1, len(cells)) > 60:
            continue
        header = [c.strip() for c in records[0]]
        types = [column_type(r[i] for r in fit) for i in range(width)]
        typed = any(t in _STRONG_CLASSES for t in types)
        if _header_ok(header):
            has_header = True
        elif all(cell_type(h) == t or (cell_type(h) == "int" and t == "float")
                 for h, t in zip(header, types)) and typed:
            has_header = False  # the first record is data like the rest
        else:
            continue
        if not has_header and width == 2 and _looks_like_line_numbers([r[0] for r in records]):
            continue  # cat -n: line numbers, a tab, then the line
        if delimiter != "\t" and not typed:
            if width < 3:
                continue
            # prose that happens to have the same number of commas per line
            sentences = sum(1 for r in fit if r[-1].strip().endswith((".", "!", "?")) and " " in r[-1].strip())
            if sentences > 0.3 * len(fit):
                continue
        columns = header if has_header else [f"c{i + 1}" for i in range(width)]
        candidate = {"format": fmt, "delimiter": delimiter, "has_header": has_header, "columns": columns}
        if delimiter == "\t":
            return candidate
        if best is None or width > len(best["columns"]):
            best = candidate
    return best


def _delimited_rows(body: str, spec: Dict[str, Any]):
    width = len(spec["columns"])
    reader = csv.reader(io.StringIO(body), delimiter=spec["delimiter"])
    first = spec["has_header"]
    try:
        for record in reader:
            if not any(c.strip() for c in record):
                continue
            if first:
                first = False
                continue
            yield record if len(record) == width else None
    except csv.Error:
        yield None


# ---- aligned CLI tables (docker ps, ps aux, kubectl get, ls -l, git log --oneline)

_HEADER_WORD_RE = re.compile(r"[A-Za-z%#][\w%#./()+:-]*\Z")
_DASH_LINE_RE = re.compile(r"\s*[-=]{2,}(?:\s+[-=]{2,})+\s*\Z")


def _header_tokens(line: str, wide: bool) -> List[Tuple[int, str]]:
    pattern = r"\S+(?: \S+)*" if wide else r"\S+"
    # with wide, single spaces join words: "CONTAINER ID" is one column
    return [(m.start(), m.group()) for m in re.finditer(pattern, line)]


def _positional_cells(line: str, starts: List[int]) -> Optional[List[str]]:
    for start in starts[1:]:
        if start < len(line) and line[start - 1] != " ":
            return None
    cells = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(line)
        cells.append(line[start:end].strip() if start < len(line) else "")
    return cells


def _aligned_header_spec(lines: List[str]) -> Optional[Dict[str, Any]]:
    if len(lines) < 3:
        return None
    header = lines[0]
    dashed = _DASH_LINE_RE.match(lines[1]) is not None
    data = lines[2:] if dashed else lines[1:]
    if len(data) < 2:
        return None
    tried = []
    for wide in (True, False):
        tokens = _header_tokens(header, wide)
        names = [t for _, t in tokens]
        if len(tokens) < 2 or names in [n for n, _ in tried]:
            continue
        words = [w for t in names for w in t.split(" ")]
        if not all(_HEADER_WORD_RE.match(w) and len(w) <= 30 for w in words):
            continue
        if not dashed and not all(t == t.upper() for t in names):
            continue  # without a ---- line, only an UPPER CASE header counts
        tried.append((names, tokens))
    for names, tokens in tried:
        starts = [s for s, _ in tokens]
        fit = sum(1 for line in data if _positional_cells(line, starts) is not None)
        if fit >= 0.9 * len(data):
            return {"format": "aligned", "mode": "positions", "starts": starts, "columns": names,
                    "header_line": header, "dashed": dashed}
    for names, tokens in tried:
        if any(" " in n for n in names):
            continue
        fit = sum(1 for line in data if len(line.split(None, len(names) - 1)) == len(names))
        if fit >= 0.9 * len(data):
            return {"format": "aligned", "mode": "split", "columns": names,
                    "header_line": header, "dashed": dashed}
    return None


def _aligned_rows(body: str, spec: Dict[str, Any]):
    width = len(spec["columns"])
    skip = 2 if spec.get("dashed") else 1
    for line in body.split("\n"):
        line = line.rstrip()
        if not line.strip():
            continue
        if skip:
            skip -= 1
            continue
        if spec["mode"] == "positions":
            yield _positional_cells(line, spec["starts"])
        else:
            cells = line.split(None, width - 1)
            yield cells if len(cells) == width else None


def _token_class(token: str) -> str:
    kind = cell_type(token)
    if kind == "text" and _WORD_RE.match(token):
        return "word"
    return kind


def _classes_match(token: str, expected: str) -> bool:
    if expected == "hex" and re.fullmatch(r"[0-9a-f]{7,64}", token):
        return True  # a hash that happens to be all digits
    got = _token_class(token)
    return got == expected or {got, expected} <= {"int", "float"} or {got, expected} <= {"int", "hex"}


def _headerless_spec(lines: List[str]) -> Optional[Dict[str, Any]]:
    """Whitespace-separated rows with no header: leading columns of a
    consistent type (the first one typed: a hash, a mode, a count, ...),
    then the rest of the line as one last column. One preamble line that
    doesn't fit ("total 48" before ls -l) is allowed."""
    return _headerless_rows_spec(lines) or (_headerless_rows_spec(lines[1:]) if len(lines) > 3 else None)


def _looks_like_line_numbers(values: List[str]) -> bool:
    numbers = [int(v) for v in values[:50] if _INT_RE.match(v.strip())]
    return len(numbers) > 2 and all(b - a == 1 for a, b in zip(numbers, numbers[1:]))


def _headerless_rows_spec(lines: List[str]) -> Optional[Dict[str, Any]]:
    if len(lines) < 3:
        return None
    tokens = [line.split() for line in lines]
    classes: List[str] = []
    for pos in range(12):
        counts: Dict[str, int] = {}
        for t in tokens:
            if len(t) > pos:
                kind = _token_class(t[pos])
                counts[kind] = counts.get(kind, 0) + 1
        if not counts:
            break
        kind = max(counts, key=lambda k: counts[k])
        if kind in ("text", "json"):
            break
        # count tokens compatible with the class (an all-digit hash in a hash column)
        if sum(1 for t in tokens if len(t) > pos and _classes_match(t[pos], kind)) < 0.9 * len(lines):
            break
        classes.append(kind)
    # the first column must be typed (a hash, a mode, a count, a size, ...)
    # and take more than one value: "result 3 line 0: ..." is prose
    if not classes or classes[0] not in _STRONG_CLASSES:
        return None
    if len({t[0] for t in tokens[:100]}) < 2:
        return None
    while classes[-1] not in _STRONG_CLASSES:
        classes.pop()  # trailing plain words stay in the rest column
    width = len(classes)
    rest = any(len(t) > width for t in tokens)
    if width + rest < 2:
        return None
    fit = [t for t in tokens if len(t) >= width and all(_classes_match(t[i], c) for i, c in enumerate(classes))]
    if len(fit) < 3 or len(fit) < 0.9 * len(lines):
        return None
    if width == 1 and classes[0] == "int" and _looks_like_line_numbers([t[0] for t in fit]):
        return None  # line numbers (cat -n), not a table
    columns = [f"c{i + 1}" for i in range(width)] + (["rest"] if rest else [])
    return {"format": "aligned, no header", "classes": classes, "rest": rest, "columns": columns}


def _headerless_rows(body: str, spec: Dict[str, Any]):
    classes = spec["classes"]
    width = len(classes)
    for line in body.split("\n"):
        line = line.strip()
        if not line:
            continue
        cells = line.split(None, width) if spec["rest"] else line.split()
        if len(cells) < width or not all(_classes_match(cells[i], c) for i, c in enumerate(classes)):
            yield None
            continue
        if spec["rest"]:
            cells = cells + [""] * (width + 1 - len(cells))
        elif len(cells) != width:
            yield None
            continue
        yield cells


# ---- logs

_LEVEL_RE = re.compile(
    r"\b(TRACE|DEBUG|INFO|NOTICE|WARN(?:ING)?|ERROR|ERR|FATAL|CRITICAL|PANIC|SEVERE)\b"
    r"|\blevel=\"?([A-Za-z]+)|\[(trace|debug|info|notice|warn|warning|error|fatal|critical)\]")
_LEVEL_NAMES = {"WARNING": "WARN", "ERR": "ERROR", "SEVERE": "ERROR", "PANIC": "FATAL"}
_LEVEL_ORDER = ("FATAL", "CRITICAL", "ERROR", "WARN", "NOTICE", "INFO", "DEBUG", "TRACE")
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
LOG_MAX_LINES = 500000


def log_levels(body: str, max_lines: int = LOG_MAX_LINES) -> Tuple[Dict[str, int], int]:
    """({level: lines}, lines scanned): each line counts once, for the
    first level word in it."""
    counts: Dict[str, int] = {}
    scanned = 0
    for line in body.split("\n", max_lines)[:max_lines]:
        scanned += 1
        m = _LEVEL_RE.search(line)
        if not m:
            continue
        level = (m.group(1) or m.group(2) or m.group(3)).upper()
        level = _LEVEL_NAMES.get(level, level)
        counts[level] = counts.get(level, 0) + 1
    ordered = {k: counts[k] for k in _LEVEL_ORDER if k in counts}
    ordered.update({k: v for k, v in sorted(counts.items()) if k not in ordered})
    return ordered, scanned


def _levels_text(levels: Dict[str, int]) -> str:
    return ", ".join(f"{k} {v:,}" for k, v in levels.items())


# ---- JSON

def _parse_json_bounded(body: str) -> Tuple[bool, Any]:
    if len(body) > JSON_MAX_CHARS:
        return False, None
    try:
        return _parse_json(body)
    except RecursionError:
        return False, None


def _jsonl_spec(sample: str, complete: bool) -> Optional[Dict[str, Any]]:
    lines = _nonempty_lines(sample, SHAPE_SAMPLE_LINES + 1)
    if not complete or len(lines) > SHAPE_SAMPLE_LINES:
        lines = lines[:-1]
    if len(lines) < 2:
        return None
    good = 0
    for line in lines:
        s = line.strip()
        if s[:1] != "{":
            continue
        try:
            if isinstance(json.loads(s), dict):
                good += 1
        except (ValueError, RecursionError):
            pass
    return {"format": "JSON Lines"} if good >= 2 and good >= 0.95 * len(lines) else None


def parse_jsonl(body: str) -> Tuple[List[Any], int]:
    """(parsed lines, lines that aren't JSON), reading at most JSON_MAX_CHARS."""
    values, bad = [], 0
    for line in body[:JSON_MAX_CHARS].split("\n"):
        s = line.strip()
        if not s:
            continue
        try:
            values.append(json.loads(s))
        except (ValueError, RecursionError):
            bad += 1
    return values, bad


_PATH_SIMPLE_KEY_RE = re.compile(r"[A-Za-z_$][\w$-]*\Z")


def path_child(path: str, key: Any) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    key = str(key)
    if _PATH_SIMPLE_KEY_RE.match(key):
        return f"{path}.{key}"
    return f"{path}[{json.dumps(key, ensure_ascii=False)}]"


def _short_path(path: str) -> str:
    """$.items -> items (how a path is passed back as an argument)."""
    if path.startswith("$."):
        return path[2:]
    return path[1:] if path.startswith("$") and path[1:2] == "[" else path


def _records_like(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    head = value[:50]
    return sum(1 for item in head if isinstance(item, dict)) >= 0.8 * len(head)


def flatten_record(obj: Dict[str, Any], prefix: str = "", depth: int = 0,
                   out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """{"a": {"b": 1}} -> {"a.b": 1}; two levels deep, lists kept whole."""
    out = {} if out is None else out
    for key, value in obj.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value and depth < 2:
            flatten_record(value, name + ".", depth + 1, out)
        else:
            out[name] = value
    return out


def _record_columns(items: List[Any], sample: int = 200) -> List[str]:
    columns: Dict[str, None] = {}
    for item in items[:sample]:
        if isinstance(item, dict):
            for key in flatten_record(item):
                columns.setdefault(key, None)
        else:
            columns.setdefault("value", None)
    return list(columns)


def json_tables(value: Any, max_depth: int = 3, limit: int = 5) -> List[Tuple[str, int, int]]:
    """(path, rows, columns) of arrays of objects within max_depth, largest first."""
    found = []
    stack = [("$", value, 0)]
    visits = 0
    while stack and visits < 20000:
        path, node, depth = stack.pop()
        visits += 1
        if isinstance(node, list):
            if _records_like(node):
                found.append((path, len(node), len(_record_columns(node))))
        elif isinstance(node, dict) and depth < max_depth:
            for key, child in node.items():
                stack.append((path_child(path, key), child, depth + 1))
    found.sort(key=lambda t: (-t[1], t[0]))
    return found[:limit]


def _schema(value: Any, depth: int, max_depth: int) -> Any:
    """A merged type description of a JSON value, for json_skeleton."""
    if isinstance(value, dict):
        if depth >= max_depth:
            return ("deep", "object", len(value))
        return ("obj", {str(k): (_schema(v, depth + 1, max_depth), 1) for k, v in list(value.items())[:40]},
                len(value), 1)
    if isinstance(value, list):
        if depth >= max_depth or not value:
            return ("arr", len(value), None)
        item = None
        for x in value[:20]:
            item = _merge_schema(item, _schema(x, depth + 1, max_depth))
        return ("arr", len(value), item)
    return ("scalar", frozenset([_scalar_name(value)]))


def _scalar_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def _schema_names(s: Any) -> frozenset:
    if s[0] in ("scalar", "union"):
        return s[1]
    return frozenset(["object" if s[0] in ("obj", "deep") else "array"])


def _merge_schema(a: Any, b: Any) -> Any:
    if a is None:
        return b
    if a[0] == "obj" and b[0] == "obj":
        keys = dict(a[1])
        for k, (sb, nb) in b[1].items():
            if k in keys:
                sa, na = keys[k]
                keys[k] = (_merge_schema(sa, sb), na + nb)
            else:
                keys[k] = (sb, nb)
        return ("obj", keys, max(a[2], b[2]), a[3] + b[3])
    if a[0] == "arr" and b[0] == "arr":
        item = b[2] if a[2] is None else (a[2] if b[2] is None else _merge_schema(a[2], b[2]))
        return ("arr", -1 if a[1] != b[1] else a[1], item)
    if a[0] == "scalar" and b[0] == "scalar":
        return ("scalar", a[1] | b[1])
    if a == b:
        return a
    # mixed: an object and a string, say; null merges into the other side
    if a[0] == "scalar" and a[1] == frozenset(["null"]):
        return b
    if b[0] == "scalar" and b[1] == frozenset(["null"]):
        return a
    return ("union", _schema_names(a) | _schema_names(b))


def _render_schema(s: Any, indent: str, depth: int, inline: bool) -> str:
    kind = s[0]
    if kind == "scalar" or kind == "union":
        names = sorted(s[1], key=lambda n: (n == "null", n))
        return "|".join(names)
    if kind == "deep":
        return f"{{…{s[2]} keys}}"
    if kind == "arr":
        count = f"{s[1]:,} × " if s[1] >= 0 else ""
        if s[2] is None:
            return "[]" if s[1] == 0 else (f"[{s[1]:,} items]" if s[1] > 0 else "[…]")
        return f"[{count}{_render_schema(s[2], indent, depth + 1, True)}]"
    # object
    items = list(s[1].items())
    parts = []
    for k, (child, n) in items:
        label = k if _PATH_SIMPLE_KEY_RE.match(k) else json.dumps(k, ensure_ascii=False)
        optional = "?" if n < s[3] else ""
        parts.append(f"{label}{optional}: {_render_schema(child, indent + '  ', depth + 1, inline)}")
    more = s[2] - len(items)
    if inline or depth >= 2:
        kept: List[str] = []
        for part in parts:
            if len(", ".join(kept + [part])) > 200 and kept:
                break
            kept.append(part)
        more += len(parts) - len(kept)
        return "{" + ", ".join(kept + ([f"…+{more} keys"] if more > 0 else [])) + "}"
    if more > 0:
        parts.append(f"…+{more} keys")
    return "{\n" + ",\n".join(f"{indent}  {p}" for p in parts) + f"\n{indent}}}"


def json_skeleton(value: Any, max_depth: int = 4, max_chars: int = 3000) -> str:
    """A depth-limited outline of a JSON value: keys with types, arrays as
    [count × item shape] merged over their first 20 items, optional keys
    marked "?"."""
    text = _render_schema(_schema(value, 0, max_depth), "", 0, False)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n… [outline cut; read parts with handles_slice path=…]"
    return text


# ---- detection

def content_shape(text: str, tool_name: Optional[str] = None, kind: Optional[str] = None,
                  value: Any = None, parsed: bool = False) -> Dict[str, Any]:
    """What a result is: {"kind": json | jsonl | table | log | text, ...}.
    Reads a sample from the start (SHAPE_SAMPLE_CHARS), except that JSON is
    parsed whole when it is at most JSON_MAX_CHARS (pass value and
    parsed=True when the caller parsed it already). Conservative: anything
    that isn't clearly one of the others is text. `kind` is detect_kind's
    answer; code and diffs are never taken for tables."""
    body = content_body(text, tool_name)
    stripped = body.strip()
    lines_total = body.count("\n") + 1
    if stripped[:1] in ("{", "["):
        if not parsed:
            parsed, value = _parse_json_bounded(body)
        if parsed:
            shape: Dict[str, Any] = {"kind": "json", "value": value}
            if isinstance(value, dict):
                shape["top"] = f"object, {len(value):,} key{'s' if len(value) != 1 else ''}"
            elif isinstance(value, list):
                shape["top"] = f"array, {len(value):,} item{'s' if len(value) != 1 else ''}"
            else:
                shape["top"] = _scalar_name(value)
            shape["tables"] = json_tables(value)
            return shape
        if len(body) > JSON_MAX_CHARS:
            return {"kind": "text", "note": f"looks like JSON but is over {JSON_MAX_CHARS // 2**20} MB, "
                                            "so it was not parsed", "lines": lines_total}
    sample, complete = _sample(body)
    spec = _jsonl_spec(sample, complete) if stripped[:1] == "{" else None
    if spec:
        records = sum(1 for line in body.split("\n") if line.strip())
        parsed_lines = [_safe_loads(line) for line in _nonempty_lines(sample, 50)]
        columns = _record_columns([v for v in parsed_lines if isinstance(v, dict)])
        return {"kind": "jsonl", "format": "JSON Lines", "rows": records, "columns": columns}
    if kind not in ("code", "diff"):
        lines = _nonempty_lines(sample, SHAPE_SAMPLE_LINES + (0 if complete else 1))
        if not complete and lines:
            lines = lines[:-1]
        is_log = _fraction(lines, LOG_LINE_RE) > 0.3
        spec = _markdown_spec(lines) or _delimited_spec(sample, complete)
        if spec is not None and is_log and not spec.get("has_header", True):
            spec = None  # log lines with a steady comma count ("10:00:00,123 INFO ...")
        spec = spec or _aligned_header_spec(lines)
        if spec is None and not is_log:
            spec = _headerless_spec(lines)
        if spec is not None:
            shape = {"kind": "table", "spec": spec, "format": spec["format"], "columns": list(spec["columns"])}
            sample_rows = [r for r in _iter_text_rows(sample, spec, limit=SHAPE_SAMPLE_LINES) if r is not None]
            shape["sample_rows"] = sample_rows[:2]
            shape["types"] = [column_type(r[i] for r in sample_rows) for i in range(len(spec["columns"]))]
            if complete:
                rows = list(_iter_text_rows(body, spec))
                shape["rows"] = sum(1 for r in rows if r is not None)
                shape["skipped"] = len(rows) - shape["rows"]
                shape["estimated"] = False
            else:
                tried = list(_iter_text_rows(sample, spec))
                ratio = (sum(1 for r in tried if r is not None) / len(tried)) if tried else 1.0
                nonblank = body.count("\n") + (0 if body.endswith("\n") else 1) - body.count("\n\n")
                header = 0 if spec.get("has_header") is False or "classes" in spec else 1
                header += 1 if spec["format"] == "markdown" or spec.get("dashed") else 0
                shape["rows"] = max(0, int((nonblank - header) * ratio))
                shape["estimated"] = True
            return shape
        if is_log:
            levels, scanned = log_levels(body)
            shape = {"kind": "log", "lines": lines_total, "levels": levels, "scanned": scanned}
            stamps = [m.group() for m in (_TIMESTAMP_RE.search(l) for l in lines[:20]) if m]
            if stamps:
                shape["first_time"] = stamps[0]
                tail = [m.group() for m in (_TIMESTAMP_RE.search(l) for l in body[-4000:].split("\n")) if m]
                if tail:
                    shape["last_time"] = tail[-1]
            return shape
    return {"kind": "text", "lines": lines_total}


def _safe_loads(line: str) -> Any:
    try:
        return json.loads(line)
    except (ValueError, RecursionError):
        return None


def _iter_text_rows(body: str, spec: Dict[str, Any], limit: Optional[int] = None):
    fmt = spec["format"]
    if fmt == "markdown":
        rows = _markdown_rows(body, spec)
    elif "delimiter" in spec:
        rows = _delimited_rows(body, spec)
    elif fmt == "aligned":
        rows = _aligned_rows(body, spec)
    else:
        rows = _headerless_rows(body, spec)
    for n, row in enumerate(rows):
        if limit is not None and n >= limit:
            return
        yield row


def shape_hint(shape: Dict[str, Any]) -> Optional[str]:
    """One line naming what the content is and the verb to read it with,
    or None for plain text."""
    kind = shape.get("kind")
    if kind == "table":
        cols = shape["columns"]
        names = ", ".join(cols[:6]) + (", …" if len(cols) > 6 else "")
        rows = f"{'~' if shape.get('estimated') else ''}{shape['rows']:,}"
        return (f"Shape: table ({shape['format']}), {rows} rows × {len(cols)} cols ({_clip(names, 100)})"
                f" — query with handles_table / handles_aggregate.")
    if kind == "jsonl":
        return (f"Shape: JSON Lines, {shape['rows']:,} records × {len(shape['columns'])} keys"
                " — query with handles_table / handles_aggregate, or handles_slice path=[0].")
    if kind == "json":
        tables = shape.get("tables") or []
        value = shape.get("value")
        if tables and tables[0][0] == "$":
            _, rows, cols = tables[0]
            return (f"Shape: JSON array of {rows:,} objects × {cols} keys — query with handles_table / "
                    "handles_aggregate, or read with handles_slice path=[0].")
        if tables:
            path, rows, cols = tables[0]
            p = _short_path(path)
            return (f"Shape: JSON {shape['top']}; {p} is {rows:,} rows × {cols} cols — handles_table "
                    f"path={p}, or handles_slice path={p}[0].")
        if isinstance(value, (dict, list)):
            return f"Shape: JSON {shape['top']} — handles_describe for its outline, handles_slice path=… for parts."
        return None
    if kind == "log":
        levels = shape.get("levels") or {}
        if levels:
            worst = [k for k in ("FATAL", "CRITICAL", "ERROR", "WARN") if k in levels]
            find = "|".join(worst) if worst else next(iter(levels))
            return (f"Shape: log, {shape['lines']:,} lines ({_clip(_levels_text(levels), 80)}) — "
                    f"handles_search pattern={find}.")
        return f"Shape: log, {shape['lines']:,} lines — handles_search for errors or times."
    return None


def describe_shape(shape: Dict[str, Any]) -> List[str]:
    """The lines handles_describe adds for the content's shape."""
    kind = shape.get("kind")
    out: List[str] = []
    if kind == "json":
        out.append(f"content: JSON {shape['top']}")
        if isinstance(shape.get("value"), (dict, list)):
            out.append("outline:")
            out.extend(json_skeleton(shape["value"]).split("\n"))
        for path, rows, cols in shape.get("tables") or []:
            out.append(f"table at {_short_path(path) or '$'}: {rows:,} rows × {cols} cols")
    elif kind in ("table", "jsonl"):
        rows = f"{'~' if shape.get('estimated') else ''}{shape['rows']:,}"
        out.append(f"content: {'table (' + shape['format'] + ')' if kind == 'table' else 'JSON Lines'}, "
                   f"{rows} rows × {len(shape['columns'])} columns")
        if shape.get("types"):
            out.append("columns: " + ", ".join(f"{c} ({t})" for c, t in zip(shape["columns"], shape["types"])))
        else:
            out.append("columns: " + ", ".join(shape["columns"][:60]))
        if shape.get("skipped"):
            out.append(f"lines that don't fit the columns (left out of queries): {shape['skipped']:,}")
        if shape.get("sample_rows"):
            out.append("sample rows:")
            out.append("  " + " | ".join(shape["columns"]))
            out.extend("  " + " | ".join(_clip(c, 60) for c in row) for row in shape["sample_rows"])
    elif kind == "log":
        out.append(f"content: log, {shape['lines']:,} lines")
        levels = shape.get("levels") or {}
        scanned = "" if shape.get("scanned", 0) >= shape["lines"] else f" (first {shape['scanned']:,} lines)"
        out.append(f"levels{scanned}: {_levels_text(levels) if levels else 'none recognised'}")
        if shape.get("first_time"):
            out.append(f"time: {shape['first_time']} … {shape.get('last_time', '?')}")
    else:
        out.append(f"content: text, {shape.get('lines', 0):,} lines" + (f" ({shape['note']})" if shape.get("note") else ""))
    hint = shape_hint(shape)
    if hint:
        out.append("next: " + hint.split(" — ", 1)[-1])
    return out


# ---------------------------------------------------------------- JSON paths

class QueryError(ValueError):
    """A bad path, where, sort or measure: the message says what and where."""


_PATH_KEY_RE = re.compile(r"[^.\[\]\s]+")


def _quoted(s: str, i: int, what: str) -> Tuple[str, int]:
    """A '...' or "..." string starting at s[i]; (value, index after it)."""
    quote = s[i]
    j, out = i + 1, []
    while j < len(s):
        c = s[j]
        if c == "\\" and j + 1 < len(s):
            out.append(s[j + 1])
            j += 2
            continue
        if c == quote:
            if s[j + 1:j + 2] == quote:  # doubled quote
                out.append(quote)
                j += 2
                continue
            return "".join(out), j + 1
        out.append(c)
        j += 1
    raise QueryError(f"{what}: string starting at character {i + 1} is not closed")


def parse_path(path: Any) -> List[Tuple[Any, ...]]:
    """Steps of a JSON path (see docs/handles.md):
    $ (optional root), .key or key (at the start), ["any key"], [n] (negative
    counts from the end), [a:b], [*] or .*, ..key (that key at any depth)."""
    s = str(path if path is not None else "").strip()
    if not s:
        raise QueryError("path is empty; e.g. items[0].name, or $ for the whole document")
    steps: List[Tuple[Any, ...]] = []
    i = 1 if s.startswith("$") else 0
    start = i
    while i < len(s):
        c = s[i]
        if s.startswith("..", i):
            i += 2
            if s[i:i + 1] == "[" and s[i + 1:i + 2] in ("'", '"'):
                key, i = _quoted(s, i + 1, "path")
                if s[i:i + 1] != "]":
                    raise QueryError(f"path: expected ] at character {i + 1}")
                i += 1
            else:
                m = _PATH_KEY_RE.match(s, i)
                if not m or m.group() == "*":
                    raise QueryError(f"path: expected a key after '..' at character {i + 1}, e.g. ..name")
                key, i = m.group(), m.end()
            steps.append(("find", key))
        elif c == ".":
            i += 1
            m = _PATH_KEY_RE.match(s, i)
            if not m:
                raise QueryError(f"path: expected a key after '.' at character {i + 1}")
            steps.append(("all",) if m.group() == "*" else ("key", m.group()))
            i = m.end()
        elif c == "[":
            if s[i + 1:i + 2] in ("'", '"'):
                key, i = _quoted(s, i + 1, "path")
                if s[i:i + 1] != "]":
                    raise QueryError(f"path: expected ] at character {i + 1}")
                steps.append(("key", key))
                i += 1
                continue
            end = s.find("]", i)
            if end < 0:
                raise QueryError(f"path: [ at character {i + 1} is not closed")
            inner = s[i + 1:end].strip()
            if inner == "*":
                steps.append(("all",))
            elif re.fullmatch(r"-?\d+", inner):
                steps.append(("index", int(inner)))
            elif re.fullmatch(r"(-?\d+)?\s*:\s*(-?\d+)?", inner):
                a, b = (part.strip() for part in inner.split(":"))
                steps.append(("slice", int(a) if a else None, int(b) if b else None))
            elif inner.count(":") > 1:
                raise QueryError(f"path: [{inner}] — slices take no step; use [a:b]")
            else:
                raise QueryError(f"path: [{inner}] is not an index, a:b slice, * or quoted key; "
                                 f"for a key with odd characters use [\"{inner}\"]")
            i = end + 1
        elif i == start:
            m = _PATH_KEY_RE.match(s, i)
            if not m:
                raise QueryError(f"path: unexpected {c!r} at character {i + 1}")
            steps.append(("all",) if m.group() == "*" else ("key", m.group()))
            i = m.end()
        else:
            raise QueryError(f"path: unexpected {c!r} at character {i + 1}; keys follow '.' or go in [\"...\"]")
    return steps


def eval_path(root: Any, steps: List[Tuple[Any, ...]]) -> Tuple[List[Tuple[str, Any]], bool]:
    """([(path, value)] in document order, whether a cap stopped the walk)."""
    nodes: List[Tuple[str, Any]] = [("$", root)]
    capped = False
    budget = PATH_MAX_VISITS
    for step in steps:
        out: List[Tuple[str, Any]] = []
        kind = step[0]
        for path, value in nodes:
            if kind == "key":
                key = step[1]
                if isinstance(value, dict):
                    if key in value:
                        out.append((path_child(path, key), value[key]))
                elif isinstance(value, list) and re.fullmatch(r"-?\d+", key):
                    n = int(key)
                    if -len(value) <= n < len(value):
                        out.append((path_child(path, n % len(value)), value[n]))
            elif kind == "index":
                if isinstance(value, list) and -len(value) <= step[1] < len(value):
                    n = step[1] % len(value)
                    out.append((path_child(path, n), value[n]))
            elif kind == "slice":
                if isinstance(value, list):
                    for n in range(*slice(step[1], step[2]).indices(len(value))):
                        out.append((path_child(path, n), value[n]))
            elif kind == "all":
                if isinstance(value, dict):
                    out.extend((path_child(path, k), v) for k, v in value.items())
                elif isinstance(value, list):
                    out.extend((path_child(path, n), v) for n, v in enumerate(value))
            elif kind == "find":
                key = step[1]
                stack = [(path, value)]
                while stack:
                    budget -= 1
                    if budget < 0:
                        capped = True
                        break
                    p, v = stack.pop()
                    if isinstance(v, dict):
                        if key in v:
                            out.append((path_child(p, key), v[key]))
                        stack.extend(reversed([(path_child(p, k), c) for k, c in v.items()
                                               if isinstance(c, (dict, list))]))
                    elif isinstance(v, list):
                        stack.extend(reversed([(path_child(p, n), c) for n, c in enumerate(v)
                                               if isinstance(c, (dict, list))]))
            if capped:
                break
        nodes = out
        if capped:
            break
    return nodes, capped


def json_root(text: str, tool_name: Optional[str] = None) -> Tuple[Any, str]:
    """(parsed document, "json" or "jsonl") of a stored result, or QueryError."""
    body = content_body(text, tool_name)
    stripped = body.strip()
    if stripped[:1] in ("{", "["):
        if len(body) > JSON_MAX_CHARS:
            raise QueryError(f"this JSON is over {JSON_MAX_CHARS // 2**20} MB, the most a path query parses; "
                             "use handles_search or a line range")
        ok, value = _parse_json_bounded(body)
        if ok:
            return value, "json"
        sample, complete = _sample(body)
        if _jsonl_spec(sample, complete):
            values, _ = parse_jsonl(body)
            return values, "jsonl"
    raise QueryError("path works on JSON (or JSON Lines), and this result isn't JSON")


def _node_keys(value: Any) -> str:
    if isinstance(value, dict):
        return f"an object with keys {_clip(', '.join(map(str, list(value)[:30])), 300)}"
    if isinstance(value, list):
        return f"an array of {len(value):,} items (use [n], [a:b] or [*])"
    return f"a {_scalar_name(value)} value"


def _path_miss(root: Any, steps: List[Tuple[Any, ...]]) -> str:
    """Where a path that matched nothing stopped matching, and what is there."""
    for n in range(len(steps) - 1, -1, -1):
        found, _ = eval_path(root, steps[:n])
        if found:
            where = "the top level" if n == 0 else f"{found[0][0]}" + (
                f" (and {len(found) - 1:,} more like it)" if len(found) > 1 else "")
            return f"It matches as far as {where}, which is {_node_keys(found[0][1])}."
    return ""


def format_value(value: Any, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f" … [{len(text) - limit:,} more characters; select a deeper path]"


def path_query(text: str, path: Any, tool_name: Optional[str] = None) -> List[str]:
    """The lines of a handles_slice answer for a JSON path."""
    steps = parse_path(path)
    root, form = json_root(text, tool_name)
    matches, capped = eval_path(root, steps)
    out = []
    shown = matches[:PATH_MAX_SHOWN]
    label = "JSON Lines (as an array of records)" if form == "jsonl" else "JSON"
    if not matches:
        out.append(f"No nodes match path {str(path).strip()} in this {label}. {_path_miss(root, steps)}")
        return out
    total = f"{len(matches):,}" + ("+" if capped else "")
    out.append(f"{total} nodes match {str(path).strip()}" +
               (f"; showing 1-{len(shown)}" if len(shown) < len(matches) else "") + ":")
    per_value = max(300, min(50000, 50000 // max(1, len(shown))))
    for p, value in shown:
        out.append(f"{p} = {format_value(value, per_value)}")
    if len(matches) > len(shown):
        out.append(f"… {len(matches) - len(shown):,} more; narrow the path, or page with a slice such as "
                   f"[{len(shown)}:{len(shown) * 2}] on the array.")
    if capped:
        out.append(f"(the ..key search stopped after visiting {PATH_MAX_VISITS:,} nodes; "
                   "start it from a deeper path)")
    return out


# ---------------------------------------------------------------- tables

class Table:
    """A stored result as rows in an in-memory SQLite table t, two columns
    per field: c<i> (the value, a number when the cell is one) and t<i> (the
    original text). Caller column names are only ever looked up in
    self.columns, never put into SQL."""

    def __init__(self, columns: List[str], types: List[str], rows: Iterable[List[Any]],
                 source: str, notes: Optional[List[str]] = None):
        self.columns = columns
        self.types = types
        self.source = source
        self.notes = list(notes or [])
        self.lock = threading.Lock()
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        width = len(columns)
        fields = ", ".join(f"c{i}, t{i}" for i in range(width))
        self.db.execute(f"CREATE TABLE t ({fields})")
        marks = ", ".join("?" * (2 * width))
        insert = f"INSERT INTO t VALUES ({marks})"
        batch: List[Tuple[Any, ...]] = []
        self.rows = 0
        for row in rows:
            batch.append(tuple(x for cell in row for x in _sql_cell(cell)))
            if len(batch) >= 5000:
                self.db.executemany(insert, batch)
                self.rows += len(batch)
                batch = []
        if batch:
            self.db.executemany(insert, batch)
            self.rows += len(batch)
        # columns where at least half the cells are numbers: min/max compare those as numbers
        counts = self.db.execute("SELECT " + ", ".join(
            f"sum(typeof(c{i}) IN ('integer','real')), count(t{i})" for i in range(width)) + " FROM t").fetchone()
        self.numeric = [bool(counts[2 * i]) and (counts[2 * i] or 0) * 2 >= (counts[2 * i + 1] or 0)
                        for i in range(width)]

    def close(self) -> None:
        self.db.close()


def _sql_cell(value: Any) -> Tuple[Any, Any]:
    """(value, text) of one cell for SQLite: numbers as numbers, empty as NULL."""
    if value is None:
        return None, None
    if isinstance(value, bool):
        text = "true" if value else "false"
        return text, text
    if isinstance(value, int):
        if abs(value) > 2 ** 62:
            return float(value), str(value)
        return value, str(value)
    if isinstance(value, float):
        return value, repr(value)
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        return text, text
    text = str(value).strip()
    if not text:
        return None, None
    if _INT_RE.match(text):
        return int(text), text
    if _FLOAT_RE.match(text):
        try:
            return float(text), text
        except ValueError:
            pass
    return text, text


def _cap_rows(rows: Iterable[Optional[List[Any]]], width: int, deadline: float,
              notes: List[str]) -> Iterable[List[Any]]:
    """Rows that fit, at most the row/cell caps and the time limit; notes
    say what was left out."""
    limit = min(TABLE_MAX_ROWS, max(1, TABLE_MAX_CELLS // max(1, width)))
    kept = skipped = extra = 0
    for n, row in enumerate(rows):
        if row is None:
            skipped += 1
            continue
        if kept >= limit:
            extra += 1
            continue
        if n % 2000 == 0 and time.monotonic() > deadline:
            notes.append(f"loading stopped after {QUERY_SECONDS:g} s (time cap) at row {kept:,}; "
                         "answers cover those rows only")
            break
        kept += 1
        yield row[:width]
    if extra:
        notes.append(f"only the first {limit:,} of {kept + extra:,} rows were loaded (row cap); "
                     "answers cover those rows only")
    if skipped:
        notes.append(f"{skipped:,} line{' that doesn' if skipped == 1 else 's that don'}'t fit the columns "
                     f"{'was' if skipped == 1 else 'were'} left out")


def load_table(text: str, tool_name: Optional[str] = None, path: Any = None,
               kind: Optional[str] = None) -> Table:
    """A stored result as a Table: a delimited/markdown/aligned text table,
    JSON Lines, or a JSON array of objects (the one at `path`, else the
    whole document when it is one, else the largest one inside)."""
    deadline = time.monotonic() + QUERY_SECONDS
    notes: List[str] = []
    body = content_body(text, tool_name)
    if len(body) > TABLE_MAX_CHARS:
        cut = body.rfind("\n", 0, TABLE_MAX_CHARS)
        body = body[:cut if cut > 0 else TABLE_MAX_CHARS]
        notes.append(f"only the first {TABLE_MAX_CHARS // 2**20} MB were read (size cap)")
    stripped = body.lstrip()
    if path not in (None, "") or stripped[:1] in ("{", "["):
        try:
            root, form = json_root(body)
        except QueryError:
            if path not in (None, ""):
                raise
            root = None
        if root is not None:
            return _json_table(root, form, path, notes, deadline)
    shape = content_shape(body, None, kind)
    if shape["kind"] != "table":
        raise QueryError(f"this result isn't a table (it looks like {shape['kind']}); handles_table and "
                         "handles_aggregate read CSV/TSV, markdown or aligned tables, JSON Lines and JSON "
                         "arrays of objects. handles_describe shows what it is.")
    spec = shape["spec"]
    columns = _dedupe(spec["columns"])
    if len(columns) > TABLE_MAX_COLUMNS:
        notes.append(f"only the first {TABLE_MAX_COLUMNS} of {len(columns)} columns were loaded (column cap)")
        columns = columns[:TABLE_MAX_COLUMNS]
    types = (shape.get("types") or [])[:len(columns)]
    rows = _cap_rows(_iter_text_rows(body, spec), len(columns), deadline, notes)
    table = Table(columns, types, rows, f"table ({spec['format']})", [])
    table.notes = notes
    return table


def _dedupe(names: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out = []
    for i, name in enumerate(names):
        name = str(name).strip() or f"c{i + 1}"
        key = name.lower()
        if key in seen:
            seen[key] += 1
            name = f"{name}_{seen[key]}"
        else:
            seen[key] = 1
        out.append(name)
    return out


def _json_table(root: Any, form: str, path: Any, notes: List[str], deadline: float) -> Table:
    where = "$"
    if path not in (None, ""):
        matches, _ = eval_path(root, parse_path(path))
        if not matches:
            raise QueryError(f"no nodes match path {path}")
        if len(matches) == 1 and isinstance(matches[0][1], list):
            where, items = matches[0][0], matches[0][1]
        else:
            where, items = f"{len(matches):,} nodes matching {path}", [v for _, v in matches]
    elif isinstance(root, list):
        items = root
    else:
        tables = json_tables(root)
        if not tables:
            raise QueryError("this JSON holds no array of objects to query; use handles_slice path=… to "
                             "read it, or give path= pointing at an array")
        where = tables[0][0]
        items, _ = eval_path(root, parse_path(where))
        items = items[0][1]
    if not isinstance(items, list):
        raise QueryError(f"path {path} is not an array")
    columns = _dedupe(_record_columns(items))
    if not columns:
        raise QueryError(f"the array at {_short_path(where) or '$'} is empty")
    if len(columns) > TABLE_MAX_COLUMNS:
        notes.append(f"only the first {TABLE_MAX_COLUMNS} of {len(columns)}+ keys were loaded (column cap)")
        columns = columns[:TABLE_MAX_COLUMNS]
    flat = []
    for item in items[:200]:
        flat.append(flatten_record(item) if isinstance(item, dict) else {"value": item})
    types = [column_type(f.get(c) for f in flat) for c in columns]

    def rows():
        for item in items:
            record = flatten_record(item) if isinstance(item, dict) else {"value": item}
            yield [record.get(c) for c in columns]

    source = "JSON Lines" if form == "jsonl" else f"JSON array at {_short_path(where) or '$'}"
    table = Table(columns, types, _cap_rows(rows(), len(columns), deadline, notes), source)
    table.notes = notes
    return table


# ---- where grammar

_WHERE_TOKEN_RE = re.compile(r"""\s*(?:
    (?P<num>[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)(?![\w.:-])
  | (?P<str>'(?:[^'\\]|\\.|'')*'|"(?:[^"\\]|\\.|"")*")
  | (?P<ident>`[^`]+`)
  | (?P<op><=|>=|!=|<>|==|=|<|>)
  | (?P<punct>[(),])
  | (?P<word>[^\s()',"`=<>!]+)
  )""", re.X)
_KEYWORDS = {"and", "or", "not", "in", "contains", "startswith", "endswith", "is", "null", "true", "false"}
_OPS = {"=": "=", "==": "=", "!=": "!=", "<>": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}


class _Value:
    __slots__ = ("text", "num", "kind")

    def __init__(self, text: Optional[str], num: Optional[float], kind: str):
        self.text, self.num, self.kind = text, num, kind  # kind: text, number, bool, null

    def __repr__(self):
        return f"_Value({self.text!r}, {self.num!r}, {self.kind})"


def _tokens(source: str, what: str) -> List[Tuple[str, str, int]]:
    out = []
    i = 0
    while i < len(source):
        if source[i:].strip() == "":
            break
        m = _WHERE_TOKEN_RE.match(source, i)
        if not m or m.end() == i:
            at = i + len(source[i:]) - len(source[i:].lstrip())
            ch = source[at]
            if ch in "'\"":
                raise QueryError(f"{what}: string starting at character {at + 1} is not closed")
            raise QueryError(f"{what}: unexpected {ch!r} at character {at + 1}")
        kind = m.lastgroup
        out.append((kind, m.group(kind), m.start(kind)))
        i = m.end()
    return out


def _unquote(token: str) -> str:
    return _quoted(token, 0, "where")[0]


class _WhereParser:
    """Recursive descent over the where grammar (docs/handles.md):
        expr       := term ("or" term)*
        term       := factor ("and" factor)*
        factor     := "not" factor | "(" expr ")" | comparison
        comparison := column op value | column ["not"] "in" "(" value ("," value)* ")"
                    | column ["not"] (contains | startswith | endswith) value
                    | column "is" ["not"] "null"
        op         := = | == | != | <> | < | <= | > | >=
        value      := 'text' | "text" | number | true | false | null | bareword
        column     := name | `any name` | "any name"
    """

    def __init__(self, source: str):
        if len(source) > WHERE_MAX_CHARS:
            raise QueryError(f"where is longer than {WHERE_MAX_CHARS} characters")
        self.source = source
        self.toks = _tokens(source, "where")
        self.i = 0
        self.terms = 0

    def peek(self, offset: int = 0):
        j = self.i + offset
        return self.toks[j] if j < len(self.toks) else None

    def word(self, offset: int = 0) -> Optional[str]:
        tok = self.peek(offset)
        return tok[1].lower() if tok and tok[0] == "word" else None

    def where_at(self) -> str:
        tok = self.peek()
        return f"at character {tok[2] + 1}" if tok else "at the end"

    def fail(self, expected: str):
        tok = self.peek()
        got = f"{tok[1]!r}" if tok else "the end"
        raise QueryError(f"where: expected {expected} {self.where_at()}, got {got}")

    def parse(self):
        if not self.toks:
            raise QueryError("where is empty")
        node = self.expr(0)
        if self.peek() is not None:
            self.fail("and / or, or the end")
        return node

    def expr(self, depth: int):
        if depth > WHERE_MAX_DEPTH:
            raise QueryError(f"where nests deeper than {WHERE_MAX_DEPTH} levels")
        parts = [self.term(depth)]
        while self.word() == "or":
            self.i += 1
            parts.append(self.term(depth))
        return parts[0] if len(parts) == 1 else ("or", parts)

    def term(self, depth: int):
        parts = [self.factor(depth)]
        while self.word() == "and":
            self.i += 1
            parts.append(self.factor(depth))
        return parts[0] if len(parts) == 1 else ("and", parts)

    def factor(self, depth: int):
        if self.word() == "not":
            self.i += 1
            return ("not", self.factor(depth + 1))
        tok = self.peek()
        if tok and tok[0] == "punct" and tok[1] == "(":
            self.i += 1
            node = self.expr(depth + 1)
            tok = self.peek()
            if not tok or tok[1] != ")":
                self.fail("')'")
            self.i += 1
            return node
        return self.comparison()

    def column(self) -> str:
        tok = self.peek()
        if tok is None:
            self.fail("a column name")
        kind, text, _ = tok
        if kind == "ident":
            name = text[1:-1]
        elif kind == "str":
            name = _unquote(text)
        elif kind == "word" and text.lower() not in _KEYWORDS:
            name = text
        elif kind == "num":
            name = text  # a column named 2024, say
        else:
            self.fail("a column name")
        self.i += 1
        return name

    def value(self) -> _Value:
        tok = self.peek()
        if tok is None:
            self.fail("a value")
        kind, text, _ = tok
        if kind == "num":
            self.i += 1
            return _Value(text, float(text), "number")
        if kind == "str":
            self.i += 1
            inner = _unquote(text)
            num = float(inner) if _FLOAT_RE.match(inner.strip()) or _INT_RE.match(inner.strip()) else None
            return _Value(inner, num, "text")
        if kind == "word":
            low = text.lower()
            if low in ("true", "false"):
                self.i += 1
                return _Value(low, None, "bool")
            if low == "null":
                self.i += 1
                return _Value(None, None, "null")
            if low in _KEYWORDS:
                self.fail("a value")
            self.i += 1
            return _Value(text, None, "text")  # a bare word is text
        if kind == "ident":
            raise QueryError(f"where: {text} {self.where_at()} — values can't be columns; quote text 'like this'")
        self.fail("a value")

    def comparison(self):
        self.terms += 1
        if self.terms > WHERE_MAX_TERMS:
            raise QueryError(f"where has more than {WHERE_MAX_TERMS} comparisons")
        column = self.column()
        tok = self.peek()
        if tok is None:
            self.fail("an operator (= != < <= > >= contains startswith endswith in is)")
        if tok[0] == "op":
            self.i += 1
            return ("cmp", column, _OPS[tok[1]], self.value())
        negate = False
        word = self.word()
        if word == "not":
            negate, self.i = True, self.i + 1
            word = self.word()
            if word not in ("in", "contains", "startswith", "endswith"):
                self.fail("in / contains / startswith / endswith after 'not'")
        if word in ("contains", "startswith", "endswith"):
            self.i += 1
            node = ("cmp", column, word, self.value())
        elif word == "in":
            self.i += 1
            tok = self.peek()
            if not tok or tok[1] != "(":
                self.fail("'(' after in")
            self.i += 1
            values = [self.value()]
            while self.peek() and self.peek()[1] == ",":
                self.i += 1
                values.append(self.value())
                if len(values) > 200:
                    raise QueryError("where: in (...) takes at most 200 values")
            tok = self.peek()
            if not tok or tok[1] != ")":
                self.fail("',' or ')'")
            self.i += 1
            node = ("in", column, values)
        elif word == "is":
            self.i += 1
            is_not = self.word() == "not"
            if is_not:
                self.i += 1
            if self.word() != "null":
                self.fail("null after is")
            self.i += 1
            return ("null", column, not is_not)
        else:
            self.fail("an operator (= != < <= > >= contains startswith endswith in is)")
        return ("not", node) if negate else node


def parse_where(source: Any):
    """The where expression as a tree of tuples; QueryError on bad input."""
    return _WhereParser(str(source)).parse()


def resolve_column(name: str, columns: List[str]) -> int:
    """A caller's column name -> its index: exact, then ignoring case."""
    if name in columns:
        return columns.index(name)
    lowered = [c.lower() for c in columns]
    if lowered.count(name.lower()) == 1:
        return lowered.index(name.lower())
    shown = ", ".join(columns[:40]) + (" …" if len(columns) > 40 else "")
    raise QueryError(f"no column {name!r}; columns are: {shown}")


_NUMERIC = "typeof(c{i}) IN ('integer','real')"


def compile_where(node, columns: List[str]) -> Tuple[str, List[Any]]:
    """SQL for a parsed where: only c<i>/t<i> names from resolve_column,
    fixed operators, and ? parameters for every value."""
    kind = node[0]
    if kind in ("and", "or"):
        parts = [compile_where(n, columns) for n in node[1]]
        sql = f" {kind.upper()} ".join(f"({p[0]})" for p in parts)
        return sql, [x for p in parts for x in p[1]]
    if kind == "not":
        sql, params = compile_where(node[1], columns)
        return f"NOT coalesce(({sql}), 0)", params
    if kind == "null":
        i = resolve_column(node[1], columns)
        return (f"t{i} IS NULL" if node[2] else f"t{i} IS NOT NULL"), []
    if kind == "in":
        i = resolve_column(node[1], columns)
        parts = [_compare(i, "=", v) for v in node[2]]
        return " OR ".join(f"({p[0]})" for p in parts), [x for p in parts for x in p[1]]
    i = resolve_column(node[1], columns)
    return _compare(i, node[2], node[3])


def _compare(i: int, op: str, value: _Value) -> Tuple[str, List[Any]]:
    t, c, numeric = f"t{i}", f"c{i}", _NUMERIC.format(i=i)
    if value.kind == "null":
        if op in ("=", "!="):
            return (f"{t} IS NULL" if op == "=" else f"{t} IS NOT NULL"), []
        raise QueryError("where: null only goes with =, != or is [not] null")
    if op in ("contains", "startswith", "endswith"):
        needle = (value.text or "").lower()
        if needle == "":
            return f"{t} IS NOT NULL", []
        if op == "contains":
            return f"instr(lower({t}), ?) > 0", [needle]
        if op == "startswith":
            return f"substr(lower({t}), 1, ?) = ?", [len(needle), needle]
        return f"(length({t}) >= ? AND substr(lower({t}), -?) = ?)", [len(needle), len(needle), needle]
    if value.kind == "bool":
        if op not in ("=", "!="):
            raise QueryError(f"where: {op} doesn't go with true/false; use = or !=")
        sql = f"lower({t}) = ?"
        return (sql, [value.text]) if op == "=" else (f"NOT coalesce({sql}, 0)", [value.text])
    if op in ("=", "!="):
        if value.num is not None:
            sql, params = f"({t} = ? OR ({numeric} AND {c} = ?))", [value.text, value.num]
        elif value.text == "":
            sql, params = f"({t} IS NULL OR {t} = '')", []
        else:
            sql, params = f"{t} = ?", [value.text]
        return (sql, params) if op == "=" else (f"NOT coalesce({sql}, 0)", params)
    if value.num is not None:
        return f"({numeric} AND {c} {op} ?)", [value.num]
    return f"{t} {op} ?", [value.text]


def _split_list(value: Any, what: str) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(part).strip() for part in value]
    else:
        raise QueryError(f"{what} must be a list of names or a comma-separated string")
    return [item for item in items if item]


def _sort_terms(sort: Any) -> List[Tuple[str, bool]]:
    """"-size, name" or ["size desc", "name"] -> [(name, descending)]."""
    terms = []
    for item in _split_list(sort, "sort"):
        desc = False
        if item.startswith("-"):
            desc, item = True, item[1:].strip()
        elif item.startswith("+"):
            item = item[1:].strip()
        words = item.rsplit(None, 1)
        if len(words) == 2 and words[1].lower() in ("asc", "desc"):
            item, desc = words[0], words[1].lower() == "desc"
        if item.startswith("`") and item.endswith("`") and len(item) > 1:
            item = item[1:-1]
        terms.append((item, desc))
    return terms


def _run(table: Table, sql: str, params: List[Any]) -> List[Tuple[Any, ...]]:
    deadline = time.monotonic() + QUERY_SECONDS
    with table.lock:
        table.db.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 2000)
        try:
            return table.db.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            if "interrupt" in str(e):
                raise QueryError(f"the query stopped after {QUERY_SECONDS:g} s (time cap); narrow it with "
                                 "where, or group by fewer columns") from None
            raise QueryError(f"the query failed: {e}") from None
        except OverflowError:
            raise QueryError("a sum overflowed; try avg, or narrow the rows with where") from None
        finally:
            table.db.set_progress_handler(None, 0)


def _limit(value: Any, default: int = QUERY_DEFAULT_LIMIT) -> Tuple[int, Optional[str]]:
    try:
        n = int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        raise QueryError(f"limit must be a whole number, not {value!r}") from None
    if n > QUERY_MAX_LIMIT:
        return QUERY_MAX_LIMIT, f"limit capped at {QUERY_MAX_LIMIT}"
    return max(1, n), None


def _rows(n: int) -> str:
    return f"{n:,} row{'s' if n != 1 else ''}"


def _show(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{round(value, 6):.15g}"
    text = str(value).replace("\r", "").replace("\n", "\\n")
    return text if len(text) <= CELL_CHARS else text[:CELL_CHARS - 1] + "…"


def _where_sql(table: Table, where: Any) -> Tuple[str, List[Any], str]:
    if where in (None, ""):
        return "", [], ""
    sql, params = compile_where(parse_where(where), table.columns)
    return f" WHERE {sql}", params, f" where {str(where).strip()}"


def table_query(table: Table, columns: Any = None, where: Any = None, sort: Any = None,
                limit: Any = None, offset: Any = None) -> List[str]:
    """The lines of a handles_table answer."""
    names = _split_list(columns, "columns")
    picked = [resolve_column(n, table.columns) for n in names] or list(range(len(table.columns)))
    where_sql, params, where_text = _where_sql(table, where)
    order = []
    for name, desc in _sort_terms(sort):
        i = resolve_column(name, table.columns)
        order.append(f"(c{i} IS NULL), c{i}{' DESC' if desc else ''}")
    order.append("rowid")
    n, note = _limit(limit)
    try:
        skip = max(0, int(offset or 0))
    except (TypeError, ValueError):
        raise QueryError(f"offset must be a whole number, not {offset!r}") from None
    total = _run(table, f"SELECT count(*) FROM t{where_sql}", params)[0][0]
    rows = _run(table, f"SELECT {', '.join(f't{i}' for i in picked)} FROM t{where_sql} "
                       f"ORDER BY {', '.join(order)} LIMIT ? OFFSET ?", params + [n, skip])
    out = [f"{table.source}, {table.rows:,} rows × {len(table.columns)} cols{where_text}: {_rows(total)} match"]
    if rows:
        out.append(f"rows {skip + 1:,}-{skip + len(rows):,} of {total:,}:")
        out.append(" | ".join(table.columns[i] for i in picked))
        out.extend(" | ".join(_show(v) for v in row) for row in rows)
    elif total:
        out.append(f"offset {skip:,} is past the last matching row")
    if skip + len(rows) < total:
        out.append(f"… {total - skip - len(rows):,} more; next page offset={skip + len(rows)}")
    out.extend(f"note: {x}" for x in table.notes + ([note] if note else []))
    return out


_MEASURE_RE = re.compile(r"(count|sum|avg|min|max|distinct)\s*(?:[:(]\s*(.*?)\s*\)?)?\Z", re.I)


def _measures(measures: Any, table: Table) -> List[Tuple[str, str, Optional[int]]]:
    """[(label, SQL, numeric column or None)] for count | sum:col | avg:col |
    min:col | max:col | distinct:col."""
    items = _split_list(measures, "measures") or ["count"]
    out = []
    for item in items:
        m = _MEASURE_RE.match(item.strip())
        if not m:
            raise QueryError(f"measure {item!r}: use count, sum:col, avg:col, min:col, max:col or distinct:col")
        fn, col = m.group(1).lower(), (m.group(2) or "").strip().strip("`")
        if fn == "count" and col in ("", "*"):
            out.append(("count", "count(*)", None))
            continue
        if not col:
            raise QueryError(f"measure {item!r} needs a column, e.g. {fn}:size")
        i = resolve_column(col, table.columns)
        label = f"{fn}({table.columns[i]})"
        numeric = f"CASE WHEN {_NUMERIC.format(i=i)} THEN c{i} END"
        uses_numbers: Optional[int] = None
        if fn == "count":
            sql = f"count(t{i})"
        elif fn == "distinct":
            sql = f"count(DISTINCT t{i})"
        elif fn in ("sum", "avg") or table.numeric[i]:
            sql, uses_numbers = f"{fn}({numeric})", i
        else:
            sql = f"{fn}(t{i})"
        out.append((label, sql, uses_numbers))
    return out


def table_aggregate(table: Table, group_by: Any = None, measures: Any = None, where: Any = None,
                    sort: Any = None, limit: Any = None) -> List[str]:
    """The lines of a handles_aggregate answer."""
    groups = [resolve_column(n, table.columns) for n in _split_list(group_by, "group_by")]
    measure_list = _measures(measures, table)
    where_sql, params, where_text = _where_sql(table, where)
    labels = [table.columns[i] for i in groups] + [m[0] for m in measure_list]
    select = [f"{sql} AS k{k}" for k, sql in enumerate([f"t{i}" for i in groups] + [m[1] for m in measure_list])]
    group_sql = f" GROUP BY {', '.join(f't{i}' for i in groups)}" if groups else ""
    order = []
    for name, desc in _sort_terms(sort):
        key = name.replace(" ", "")
        match = None
        for k, label in enumerate(labels):
            if key.lower() in (label.lower().replace(" ", ""), label.lower().replace("(", ":").rstrip(")")):
                match = k
                break
        if match is None:
            raise QueryError(f"sort {name!r}: sort by one of {', '.join(labels)} (or sum:col style)")
        order.append(f"(k{match} IS NULL), k{match}{' DESC' if desc else ''}")
    if not order and groups:
        order = [f"k{len(groups)} DESC"] + [f"(k{k} IS NULL), k{k}" for k in range(len(groups))]
    n, note = _limit(limit)
    sql = f"SELECT {', '.join(select)} FROM t{where_sql}{group_sql}"
    if order:
        sql += f" ORDER BY {', '.join(order)}"
    matched = _run(table, f"SELECT count(*) FROM t{where_sql}", params)[0][0]
    total_groups = (_run(table, f"SELECT count(*) FROM (SELECT 1 FROM t{where_sql}{group_sql})", params)[0][0]
                    if groups else 1)
    rows = _run(table, sql + " LIMIT ?", params + [n])
    head = f"{table.source}, {_rows(table.rows)}"
    if where_text:
        head += f";{where_text}: {_rows(matched)} match"
    if groups:
        head += f"; {total_groups:,} group{'s' if total_groups != 1 else ''} by {', '.join(table.columns[i] for i in groups)}"
    out = [head, " | ".join(labels)]
    out.extend(" | ".join(_show(v) if v is not None else ("(empty)" if k < len(groups) else "")
                          for k, v in enumerate(row)) for row in rows)
    if len(rows) < total_groups:
        out.append(f"… {total_groups - len(rows):,} more groups; raise limit or add where")
    numeric_cols = sorted({m[2] for m in measure_list if m[2] is not None})
    if numeric_cols:
        counts = _run(table, "SELECT " + ", ".join(
            f"sum(t{i} IS NOT NULL AND NOT {_NUMERIC.format(i=i)})" for i in numeric_cols) + f" FROM t{where_sql}",
            params)[0]
        for i, n in zip(numeric_cols, counts):
            if n:
                out.append(f"note: {n:,} {table.columns[i]} cell{'s' if n != 1 else ''} that "
                           f"{'aren' if n != 1 else 'isn'}'t a number {'were' if n != 1 else 'was'} left out "
                           "of its sum/avg/min/max")
    out.extend(f"note: {x}" for x in table.notes + ([note] if note else []))
    return out


# ---- diff (handles_diff)

DIFF_DEFAULT_LINES = 200    # diff lines shown unless max_lines says otherwise
DIFF_CONTEXT = 3            # unchanged lines shown around each change
DIFF_MAX_COMPARED = 20000   # longer changed parts are shown as one block, not compared line by line
DIFF_NAMES_SHOWN = 20       # keys named per list in the JSON line


def diff_view(row: Dict[str, Any]) -> List[str]:
    """A stored result's lines for handles_diff, numbered as handles_slice
    numbers them, but a Read result without its line-number prefixes, so a
    line added near the top doesn't make every later line differ."""
    text = row.get("content") or ""
    if row.get("tool_name") == "Read":
        text = strip_read_line_numbers(text)[0]
    return view_lines(text, row.get("kind"))


def _diff_opcodes(a: List[str], b: List[str]) -> Tuple[List[tuple], bool]:
    """difflib opcodes turning a into b, and whether the changed part was
    too long to compare line by line (it is then one block). The common
    start and end are matched first, so a small change in a long result
    costs little."""
    head = 0
    while head < len(a) and head < len(b) and a[head] == b[head]:
        head += 1
    tail = 0
    while tail < len(a) - head and tail < len(b) - head and a[-1 - tail] == b[-1 - tail]:
        tail += 1
    a_mid, b_mid = a[head:len(a) - tail], b[head:len(b) - tail]
    whole = len(a_mid) > DIFF_MAX_COMPARED or len(b_mid) > DIFF_MAX_COMPARED
    if whole:
        tag = "replace" if a_mid and b_mid else ("delete" if a_mid else "insert")
        middle = [(tag, 0, len(a_mid), 0, len(b_mid))]
    else:
        middle = difflib.SequenceMatcher(None, a_mid, b_mid).get_opcodes()
    codes = [("equal", 0, head, 0, head)] if head else []
    codes += [(tag, i1 + head, i2 + head, j1 + head, j2 + head) for tag, i1, i2, j1, j2 in middle]
    if tail:
        codes.append(("equal", len(a) - tail, len(a), len(b) - tail, len(b)))
    return codes, whole


def _diff_hunks(codes: List[tuple], context: int):
    """The opcodes in hunks with `context` unchanged lines around each
    change (as difflib's get_grouped_opcodes)."""
    codes = list(codes)
    if codes and codes[0][0] == "equal":
        tag, i1, i2, j1, j2 = codes[0]
        codes[0] = tag, max(i1, i2 - context), i2, max(j1, j2 - context), j2
    if codes and codes[-1][0] == "equal":
        tag, i1, i2, j1, j2 = codes[-1]
        codes[-1] = tag, i1, min(i2, i1 + context), j1, min(j2, j1 + context)
    group: List[tuple] = []
    for tag, i1, i2, j1, j2 in codes:
        if tag == "equal" and i2 - i1 > 2 * context:
            group.append((tag, i1, min(i2, i1 + context), j1, min(j2, j1 + context)))
            yield group
            group = []
            i1, j1 = max(i1, i2 - context), max(j1, j2 - context)
        group.append((tag, i1, i2, j1, j2))
    if group and not (len(group) == 1 and group[0][0] == "equal"):
        yield group


def _diff_range(start: int, stop: int) -> str:
    """A hunk header's range, as unified diffs write it."""
    length = stop - start
    if length == 1:
        return str(start + 1)
    return f"{start + 1 if length else start},{length}"


def _unified(a: List[str], b: List[str], codes: List[tuple], context: int, a_label: str, b_label: str):
    yield f"--- {a_label}"
    yield f"+++ {b_label}"
    for group in _diff_hunks(codes, context):
        yield (f"@@ -{_diff_range(group[0][1], group[-1][2])} "
               f"+{_diff_range(group[0][3], group[-1][4])} @@")
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                yield from (" " + line for line in a[i1:i2])
                continue
            yield from ("-" + line for line in a[i1:i2])
            yield from ("+" + line for line in b[j1:j2])


def _json_type(value: Any) -> str:
    return "object" if isinstance(value, dict) else "array" if isinstance(value, list) else "value"


def json_changes(old: Any, new: Any) -> Optional[str]:
    """One line on what changed between two JSON documents: for two
    objects, the top-level keys added, removed and changed (with row counts
    of arrays); for two arrays, the row counts. None unless both are
    objects or arrays."""
    def names(items: List[str]) -> str:
        shown = ", ".join(items[:DIFF_NAMES_SHOWN])
        return shown + (f" (+{len(items) - DIFF_NAMES_SHOWN} more)" if len(items) > DIFF_NAMES_SHOWN else "")

    def rows(a: list, b: list) -> str:
        return f"{len(a):,} → {len(b):,} rows" if len(a) != len(b) else f"same {len(a):,} rows, some changed"

    if isinstance(old, dict) and isinstance(new, dict):
        changed = []
        for key in old:
            if key in new and old[key] != new[key]:
                both = isinstance(old[key], list) and isinstance(new[key], list)
                changed.append(f"{key} ({rows(old[key], new[key])})" if both else key)
        parts = [f"{label}: {names(items)}" for label, items in (
            ("keys added", [k for k in new if k not in old]),
            ("keys removed", [k for k in old if k not in new]),
            ("changed", changed)) if items]
        return "JSON object: " + ("; ".join(parts) if parts else "the same top-level keys and values")
    if isinstance(old, list) and isinstance(new, list):
        return "JSON array: " + ("the same rows" if old == new else rows(old, new))
    if isinstance(old, (dict, list)) and isinstance(new, (dict, list)):
        return f"JSON: the shapes differ ({_json_type(old)} → {_json_type(new)}), so keys aren't compared"
    return None


def handle_diff(old: Dict[str, Any], new: Dict[str, Any], max_lines: int = DIFF_DEFAULT_LINES,
                context: int = DIFF_CONTEXT) -> List[str]:
    """The lines of a handles_diff answer for two stored rows (content
    included): a summary line; for two JSON documents, what changed at the
    top level; then a unified diff of their lines (old to new), cut after
    max_lines lines with a line saying how many more there were."""
    a, b = diff_view(old), diff_view(new)
    codes, whole = _diff_opcodes(a, b)
    removed = sum(i2 - i1 for tag, i1, i2, _, _ in codes if tag != "equal")
    added = sum(j2 - j1 for tag, _, _, j1, j2 in codes if tag != "equal")
    head = f"{old['id']} → {new['id']}: "
    if not added and not removed:
        layout = "" if (old.get("content") or "") == (new.get("content") or "") else \
            " (the stored texts differ only in layout or Read line numbers)"
        return [head + f"identical, {len(a):,} line{'s' if len(a) != 1 else ''}{layout}."]
    out = [head + f"{added:,} line{'s' if added != 1 else ''} added, {removed:,} removed "
                  f"({len(a):,} → {len(b):,} lines)."]
    is_json, old_value = _parse_json_bounded(content_body(old.get("content") or "", old.get("tool_name")))
    if is_json:
        is_json, new_value = _parse_json_bounded(content_body(new.get("content") or "", new.get("tool_name")))
        line = json_changes(old_value, new_value) if is_json else None
        if line:
            out.append(line)
    if whole:
        out.append(f"note: the changed part is over {DIFF_MAX_COMPARED:,} lines, so it is shown as one "
                   "block (all its old lines removed, then all its new lines added)")

    def label(row: Dict[str, Any]) -> str:
        return f"{row['id']}  {row.get('summary') or row.get('tool_name') or 'tool result'}"

    shown, more = [], 0
    for line in _unified(a, b, codes, max(0, context), label(old), label(new)):
        if len(shown) < max_lines:
            shown.append(line)
        else:
            more += 1
    out += shown
    if more:
        out.append(f"… {more:,} more diff lines cut (max_lines={max_lines}); read the rest with a larger "
                   "max_lines, or handles_slice either handle at the line numbers in the @@ headers")
    return out
