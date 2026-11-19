#!/usr/bin/env python3
"""Read tool results that Switchyard moved out of the conversation.

When a conversation gets long, the Switchyard proxy replaces large, old tool
results with a short stub naming a handle (h_ followed by 12 hex digits) and
keeps the full text in ~/.switchyard/handles.sqlite3 (SWITCHYARD_HANDLES_DB),
or on a Switchyard gateway when SWITCHYARD_URL is set (SWITCHYARD_API_KEY; both
read from the environment, else from ~/.switchyard/gateway.env). With a gateway,
each tool call is reported to it (switchyard_gateway_client.Reporter; never the
arguments or the text; SWITCHYARD_MCP_REPORT=0 turns that off).
switchyard_mcp_proxy.py stores big MCP results the same way, as they arrive.
This MCP server gives the model tools to get it back:
handles_peek, handles_slice (lines, or a JSON path), handles_search,
handles_describe (the content's shape), handles_list, handles_diff (two
handles compared), and for tables and JSON arrays of objects handles_table
and handles_aggregate. All of them only read, except handles_pin, which
keeps a handle past its expiry until it is unpinned. docs/handles.md has
the path syntax, the where grammar and the caps. It also lists report_model
(switchyard_gateway_client.REPORT_MODEL_TOOL), which only passes the model's
own name to the activity report.

Speaks MCP over stdio (newline-delimited JSON-RPC). Standard library only;
run it with `python3 -I switchyard_handles_mcp.py`.
"""

import datetime
import json
import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python3 -I leaves the script's folder off sys.path

import switchyard_gateway_client as gateway_client  # noqa: E402
import switchyard_handles as handles  # noqa: E402

# Set by main(): tells the gateway about each tool call (switchyard_gateway_client.Reporter). None when
# this module is imported, so switchyard_mcp_proxy.py, which serves the same tools, reports them itself.
REPORTER = None

MAX_LINES = 400          # most lines one peek or slice returns
MAX_LINE_CHARS = 2000     # longer lines are cut
MAX_OUTPUT_CHARS = 60000  # most text one call returns
MAX_MATCHES = 200
MAX_CONTEXT = 20

HANDLE_PROPERTY = {"type": "string", "description": "Handle id from the stub (h_...)."}
INT = {"type": "integer"}
STR = {"type": "string"}
LIST = {"type": "array", "items": {"type": "string"}}
WHERE = {"type": "string", "description": "e.g. status = 'failed' and (size > 1000 or name contains 'tmp')"}
JSON_PATH = {"type": "string", "description": "JSON: which array (default: the largest)."}

TOOLS = [
    {
        "name": "handles_peek",
        "description": (
            "Start and end of a tool result Switchyard replaced with a handle (stub: 'kept outside "
            "the conversation by Switchyard as handle h_...'). Faster and exact, so prefer it to "
            "re-running the tool."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": HANDLE_PROPERTY,
                "head_lines": {"type": "integer", "description": "Default 40."},
                "tail_lines": {"type": "integer", "description": "Default 10."},
            },
            "required": ["handle"],
        },
    },
    {
        "name": "handles_slice",
        "description": (
            f"Read a handle's lines start_line..end_line (1-based, inclusive, at most {MAX_LINES}), or, "
            "for JSON, the nodes at path: a.b, [0], [2:5], [*], ..key (any depth)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": HANDLE_PROPERTY,
                "start_line": INT,
                "end_line": INT,
                "path": {"type": "string", "description": "JSON path, e.g. items[0].name"},
            },
            "required": ["handle"],
        },
    },
    {
        "name": "handles_search",
        "description": "Regex search of a handle's text; matching lines with numbers and context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": HANDLE_PROPERTY,
                "pattern": {"type": "string", "description": "Python regex."},
                "context": {"type": "integer", "description": "Lines around each match (default 2)."},
                "max_matches": {"type": "integer", "description": "Default 50."},
                "ignore_case": {"type": "boolean", "description": "Default true."},
            },
            "required": ["handle", "pattern"],
        },
    },
    {
        "name": "handles_describe",
        "description": (
            "What a handle holds and how to read it: size, source tool, and the shape (JSON outline, "
            "table columns with types and sample rows, log levels)."
        ),
        "inputSchema": {"type": "object", "properties": {"handle": HANDLE_PROPERTY}, "required": ["handle"]},
    },
    {
        "name": "handles_list",
        "description": "Recent handles, newest first, one line each; for when you lack the id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": {"type": "string", "description": "Only this Switchyard session."},
                "limit": {"type": "integer", "description": "Default 20, max 100."},
            },
        },
    },
    {
        "name": "handles_table",
        "description": (
            "Rows of a handle holding a table (CSV/TSV, markdown, aligned CLI output) or JSON array "
            "of objects: pick columns, filter, sort ('-col' descending), page. where: col = != < <= > >= "
            "contains startswith endswith in (...), is null; and/or/not, parentheses; quote text."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": HANDLE_PROPERTY,
                "columns": LIST,
                "where": WHERE,
                "sort": STR,
                "limit": {"type": "integer", "description": f"Default {handles.QUERY_DEFAULT_LIMIT}, max {handles.QUERY_MAX_LIMIT}."},
                "offset": INT,
                "path": JSON_PATH,
            },
            "required": ["handle"],
        },
    },
    {
        "name": "handles_aggregate",
        "description": (
            "Group and total a handle's table (as handles_table): measures count, sum:col, avg:col, "
            "min:col, max:col, distinct:col. Sorted by the first measure, descending, unless sort is given."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": HANDLE_PROPERTY,
                "group_by": LIST,
                "measures": LIST,
                "where": WHERE,
                "sort": STR,
                "limit": INT,
                "path": JSON_PATH,
            },
            "required": ["handle"],
        },
    },
    {
        "name": "handles_diff",
        "description": "Unified diff of handle to other:lines added/removed; for JSON, changed keys or row counts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": HANDLE_PROPERTY,
                "other": STR,
                "max_lines": {"type": "integer", "description": f"Default {handles.DIFF_DEFAULT_LINES}."},
                "context": INT,
            },
            "required": ["handle", "other"],
        },
    },
    {
        "name": "handles_pin",
        "description": "Keep a handle past expiry; pin false unpins.",
        "inputSchema": {
            "type": "object",
            "properties": {"handle": HANDLE_PROPERTY, "pin": {"type": "boolean"}},
            "required": ["handle"],
        },
    },
]
TOOL_NAMES = {tool["name"] for tool in TOOLS}
# What this server lists: the handles_* tools (which switchyard_mcp_proxy.py also adds) and report_model.
SERVER_TOOLS = TOOLS + [gateway_client.REPORT_MODEL_TOOL]


class ToolError(Exception):
    """A problem to report to the model as tool output (isError), not a crash."""


_store = None


def store():
    """The shared store: SQLite (SWITCHYARD_HANDLES_DB), or the Switchyard
    gateway at SWITCHYARD_URL when that is set (handles.open_store)."""
    global _store
    if _store is None:
        try:
            _store = handles.open_store()
        except Exception as e:  # unwritable path, corrupt file, ...
            where = gateway_client.gateway_settings()[0] or handles.default_db_path()
            raise ToolError(f"Can't open the handle store at {where}: {e}")
    return _store


def integer(args: dict, name: str, default=None, low=None, high=None) -> int:
    value = args.get(name, default)
    if value is None:
        raise ToolError(f"Missing argument {name}.")
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ToolError(f"Argument {name} must be an integer, not {value!r}.")
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def boolean(args: dict, name: str, default: bool) -> bool:
    value = args.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off")
    return bool(value)


def handle_arg(args: dict, name: str = "handle"):
    """(as given, normalized id) of the handle in args[name]."""
    raw = args.get(name)
    if not raw:
        raise ToolError(f"Missing argument {name} (the h_... id from the stub).")
    return raw, handles.normalize_handle(raw)


def unknown(raw, hid: str) -> ToolError:
    days = store().ttl_days
    kept = f"handles are kept {days:g} days after last use" if days and days > 0 else "it was deleted"
    return ToolError(
        f"Unknown handle {raw!r}: nothing is stored under {hid}. It may be mistyped, or it "
        f"has expired ({kept}). Call handles_list to see recent handles, or re-run the "
        "original tool to get the output again.")


def load(args: dict, view: bool = True, name: str = "handle"):
    """(row, view lines) for the handle in args[name] (no lines with view=False)."""
    raw, hid = handle_arg(args, name)
    try:
        row = store().get(hid)
    except handles.RemoteStoreError as e:
        raise ToolError(str(e))
    if row is None:
        raise unknown(raw, hid)
    if not view:
        return row, None
    return row, handles.result_view(row["content"], row.get("tool_name"), row.get("kind"))


def clip_line(line: str) -> str:
    if len(line) <= MAX_LINE_CHARS:
        return line
    return (f"{line[:MAX_LINE_CHARS]} … [{len(line) - MAX_LINE_CHARS:,} more characters on this "
            "line; use handles_search to find text in it]")


def numbered(lines, start: int, marker: str = " ") -> list:
    """Lines prefixed with their 1-based number; `start` is the first's index."""
    return [f"{start + i + 1:>6}{marker} {clip_line(line)}" for i, line in enumerate(lines)]


def header(row: dict, total_lines: int) -> str:
    return (f"Handle {row['id']}: {row.get('summary') or row.get('tool_name') or 'tool result'}"
            f" ({total_lines:,} lines)")


def sentence(error) -> str:
    text = str(error).strip()
    return text if text.endswith((".", "?", "!")) else text + "."


def finish(lines: list) -> str:
    text = "\n".join(lines)
    if len(text) > MAX_OUTPUT_CHARS:
        text = (text[:MAX_OUTPUT_CHARS]
                + "\n… [output cut to fit; ask for fewer lines with handles_slice]")
    return text


def peek(args: dict) -> str:
    row, lines = load(args)
    head = integer(args, "head_lines", 40, 0, MAX_LINES)
    tail = integer(args, "tail_lines", 10, 0, MAX_LINES - head)
    out = [header(row, len(lines))]
    if len(lines) <= head + tail:
        out += numbered(lines, 0)
        return finish(out)
    out += numbered(lines[:head], 0)
    skipped = len(lines) - head - tail
    out.append(f"   ... {skipped:,} lines not shown (lines {head + 1}-{head + skipped}; "
               "use handles_slice or handles_search) ...")
    if tail:
        out += numbered(lines[len(lines) - tail:], len(lines) - tail)
    return finish(out)


def slice_(args: dict) -> str:
    if args.get("path") not in (None, ""):
        row, _ = load(args, view=False)
        try:
            body = handles.path_query(row["content"], args["path"], row.get("tool_name"))
        except handles.QueryError as e:
            raise ToolError(sentence(e))
        return finish([header(row, row.get("lines") or 0)] + body)
    row, lines = load(args)
    total = len(lines)
    start = integer(args, "start_line", low=1)
    end = integer(args, "end_line", low=1)
    if start > total:
        raise ToolError(f"start_line {start} is past the end: handle {row['id']} has {total:,} lines.")
    if end < start:
        raise ToolError(f"end_line {end} is before start_line {start}.")
    end = min(end, total)
    out = [header(row, total)]
    capped = end - start + 1 > MAX_LINES
    if capped:
        end = start + MAX_LINES - 1
    out += numbered(lines[start - 1:end], start - 1)
    if capped:
        out.append(f"   ... capped at {MAX_LINES} lines; continue with start_line={end + 1} ...")
    elif end < total:
        out.append(f"   ... {total - end:,} more lines after line {end} ...")
    return finish(out)


def search(args: dict) -> str:
    row, lines = load(args)
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or pattern == "":
        raise ToolError("Missing argument pattern (a regular expression).")
    context = integer(args, "context", 2, 0, MAX_CONTEXT)
    max_matches = integer(args, "max_matches", 50, 1, MAX_MATCHES)
    flags = re.IGNORECASE if boolean(args, "ignore_case", True) else 0
    note = None
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        regex = re.compile(re.escape(pattern), flags)
        note = f"(Not a valid regular expression: {e}. Searched for it as plain text.)"
    hits = [i for i, line in enumerate(lines) if regex.search(line)]
    out = [header(row, len(lines))]
    if note:
        out.append(note)
    if not hits:
        out.append(f"No lines match {pattern!r}.")
        return finish(out)
    shown = hits[:max_matches]
    more = f" (showing the first {len(shown)})" if len(hits) > len(shown) else ""
    out.append(f"{len(hits):,} matching lines{more}:")
    matched = set(shown)
    previous_end = -1
    for i in shown:
        start, end = max(0, i - context), min(len(lines), i + context + 1)
        start = max(start, previous_end)
        if start > previous_end and previous_end >= 0:
            out.append("    --")
        for j in range(start, end):
            out.append(f"{j + 1:>6}{':' if j in matched else '-'} {clip_line(lines[j])}")
        previous_end = max(previous_end, end)
    return finish(out)


def when(timestamp) -> str:
    try:
        return datetime.datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


def describe(args: dict) -> str:
    row, lines = load(args)
    tool_input = json.dumps(row.get("tool_input") or {}, ensure_ascii=False, default=str)
    if len(tool_input) > 800:
        tool_input = tool_input[:797] + "..."
    info = [
        f"handle: {row['id']}",
        f"kind: {row.get('kind')}",
        f"lines: {len(lines):,}",
        f"bytes: {row.get('bytes') or 0:,}",
        f"tokens (estimated): {row.get('tokens') or 0:,}",
        f"source tool: {row.get('tool_name') or 'unknown'}",
        f"tool input: {tool_input}",
        f"summary: {row.get('summary') or ''}",
        f"stored: {when(row.get('created'))}",
        f"session: {row.get('session') or 'unknown'}",
    ]
    if row.get("replaced") is not None:  # a remote gateway may not say
        info.append(f"replaced in the conversation: {'yes' if row.get('replaced') else 'not yet'}")
    if row.get("pinned") is not None:  # nor this
        info.append("pinned: yes (kept until unpinned)" if row.get("pinned") else "pinned: no")
    if row.get("keys"):
        keys = row["keys"]
        shown = ", ".join(keys[:50]) + (f" (+{len(keys) - 50} more)" if len(keys) > 50 else "")
        info.insert(2, f"JSON keys: {shown}")
    try:
        shape = handles.content_shape(row["content"] or "", row.get("tool_name"), row.get("kind"))
        info += handles.describe_shape(shape)
    except Exception as e:  # the basic facts above are still worth sending
        info.append(f"content: (shape not worked out: {type(e).__name__})")
    return finish(info)


# The last few tables loaded, so paging or re-grouping a big result doesn't
# parse it again: (handle id, content length, path) -> Table.
_tables = {}
_tables_order = []
_tables_lock = threading.Lock()
TABLE_CACHE = 3


def table_for(args: dict):
    row, _ = load(args, view=False)
    path = args.get("path") or None
    key = (row["id"], len(row["content"] or ""), path)
    with _tables_lock:
        table = _tables.get(key)
    if table is None:
        try:
            table = handles.load_table(row["content"] or "", row.get("tool_name"), path, row.get("kind"))
        except handles.QueryError as e:
            raise ToolError(f"{row['id']}: {sentence(e)}")
        with _tables_lock:
            if key not in _tables:
                _tables[key] = table
                _tables_order.append(key)
            while len(_tables_order) > TABLE_CACHE:
                # not closed here: another thread may still be querying it; it
                # closes when the last reference goes
                _tables.pop(_tables_order.pop(0), None)
    return row, table


def table(args: dict) -> str:
    row, data = table_for(args)
    try:
        lines = handles.table_query(data, args.get("columns"), args.get("where"), args.get("sort"),
                                    args.get("limit"), args.get("offset"))
    except handles.QueryError as e:
        raise ToolError(sentence(e))
    return finish([f"Handle {row['id']}: " + lines[0]] + lines[1:])


def aggregate(args: dict) -> str:
    row, data = table_for(args)
    try:
        lines = handles.table_aggregate(data, args.get("group_by"), args.get("measures"), args.get("where"),
                                        args.get("sort"), args.get("limit"))
    except handles.QueryError as e:
        raise ToolError(sentence(e))
    return finish([f"Handle {row['id']}: " + lines[0]] + lines[1:])


def list_(args: dict) -> str:
    session = args.get("session") or None
    limit = integer(args, "limit", 20, 1, 100)
    try:
        rows = store().recent(session=session, limit=limit)
    except handles.RemoteStoreError as e:
        raise ToolError(str(e))
    if not rows:
        return "No handles stored" + (f" for session {session}." if session else ".")
    out = [f"{len(rows)} most recent handles" + (f" for session {session}" if session else "") + ":"]
    for row in rows:
        out.append(f"{row['id']}  {when(row.get('created'))}  {row.get('kind') or '?':<5} "
                   f"{'[pinned] ' if row.get('pinned') else ''}{row.get('summary') or ''}")
    return finish(out)


def diff(args: dict) -> str:
    old, _ = load(args, view=False)
    new, _ = load(args, view=False, name="other")
    max_lines = integer(args, "max_lines", handles.DIFF_DEFAULT_LINES, 1, MAX_LINES)
    context = integer(args, "context", handles.DIFF_CONTEXT, 0, MAX_CONTEXT)
    return finish([clip_line(line) for line in handles.handle_diff(old, new, max_lines, context)])


def pin(args: dict) -> str:
    raw, hid = handle_arg(args)
    pinned = boolean(args, "pin", True)
    try:
        result = store().pin(hid, pinned)
    except (handles.PinLimitError, handles.RemoteStoreError) as e:
        raise ToolError(sentence(e))
    if result is None:
        raise unknown(raw, hid)
    in_use = f"{result['count']:,} of {handles.PIN_MAX:,} pins in use."
    if pinned:
        done = "Pinned" if result["changed"] else "Already pinned"
        return f"{done} {hid}: it is kept until unpinned, whatever its age. {in_use}"
    days = store().ttl_days
    expiry = (f"it expires {days:g} days after its last use, counted from now" if days and days > 0
              else "expiry is off, so it is kept anyway")
    return f"{'Unpinned' if result['changed'] else 'Not pinned'} {hid}: {expiry}. {in_use}"


HANDLERS = {
    "handles_peek": peek,
    "handles_slice": slice_,
    "handles_search": search,
    "handles_describe": describe,
    "handles_list": list_,
    "handles_table": table,
    "handles_aggregate": aggregate,
    "handles_diff": diff,
    "handles_pin": pin,
}


def call_tool(name: str, args: dict) -> dict:
    """The tools/call result: text, with isError for a problem to report.
    switchyard_mcp_proxy.py serves the same tools (TOOLS, TOOL_NAMES) through
    this function, merged into the server it wraps."""
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
            "serverInfo": {"name": "switchyard-handles", "version": "1.0"},
            "instructions": (
                "Switchyard replaces large, old tool results with a stub naming a handle "
                "(h_...). Use these tools to read the full result back instead of re-running "
                "the original tool. " + gateway_client.REPORT_MODEL_INSTRUCTION
            ),
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": SERVER_TOOLS}
    elif method == "tools/call":
        params = request.get("params") or {}
        name = params.get("name")
        if name == gateway_client.REPORT_MODEL:  # not a tool call worth reporting: only names the model
            return {"jsonrpc": "2.0", "id": rid,
                    "result": gateway_client.answer_report_model(REPORTER, params.get("arguments") or {})}
        if name not in TOOL_NAMES:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": f"unknown tool {name}"}}
        started = time.monotonic()
        result = call_tool(name, params.get("arguments") or {})
        if REPORTER is not None:
            size = gateway_client.tokens_of(sum(len(c.get("text") or "") for c in result["content"]))
            REPORTER.call(name, (time.monotonic() - started) * 1000, size, size, is_error=result["isError"])
    else:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"no method {method}"}}
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def write(response: dict) -> None:
    try:
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    except (OSError, ValueError):
        pass  # the client has gone away


def main() -> int:
    global REPORTER
    REPORTER = gateway_client.Reporter("handles", "switchyard_handles")
    try:
        return serve()
    finally:
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
        write_response = handle(request)
        if write_response is not None:
            write(write_response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
