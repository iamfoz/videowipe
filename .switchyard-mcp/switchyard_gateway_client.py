#!/usr/bin/env python3
"""Find the Switchyard gateway, and tell it what Switchyard's MCP servers did.

Used by switchyard_handles.py, switchyard_handles_mcp.py, switchyard_code_mcp.py
and switchyard_mcp_proxy.py.

Gateway settings. A desktop app starts MCP servers without the user's shell
environment and does not expand ${VAR} in its config, so each setting is taken
from the environment variable (SWITCHYARD_URL, SWITCHYARD_API_KEY) when it is
set and not empty, else from the gateway file: ~/.switchyard/gateway.env
(SWITCHYARD_GATEWAY_FILE moves it), plain NAME=value lines, # comments.
`switchyard.sh clients gateway --url URL` writes it (mode 0600), `--show` says
what it holds without the key, `--remove` deletes it.

Activity report. Reporter posts each tool call (tool name, duration, result
and sent sizes in tokens, the handle id, isError) to
POST {SWITCHYARD_URL}/v1/mcp/activity from a background thread, at most every
2 seconds, with a heartbeat every 30 seconds once the client has initialized.
Tool arguments, results, file paths and conversation text are never sent.
The report_model tool (REPORT_MODEL_TOOL, served by each of those servers) lets
the model name itself; Reporter.set_model adds that name to every report from
then on, so the gateway can price the tokens saved. Without a gateway URL, or with SWITCHYARD_MCP_REPORT=0, it does nothing. It
never raises into its caller, never delays a tool call and never writes to
stdout (the MCP channel).

Standard library only, Python 3.9 compatible; runs under python3 -I.
"""

import atexit
import json
import os
import secrets
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

URL_VAR = "SWITCHYARD_URL"
KEY_VAR = "SWITCHYARD_API_KEY"
FILE_VAR = "SWITCHYARD_GATEWAY_FILE"
REPORT_VAR = "SWITCHYARD_MCP_REPORT"
DEFAULT_FILE = os.path.join("~", ".switchyard", "gateway.env")
MAX_FILE_BYTES = 64 * 1024
ACTIVITY_PATH = "/v1/mcp/activity"


# ---------------------------------------------------------------- the gateway file


def _home(env):
    return env.get("HOME") or os.path.expanduser("~")


def gateway_file_path(environ=None):
    """The gateway file this process reads: SWITCHYARD_GATEWAY_FILE, else ~/.switchyard/gateway.env."""
    env = os.environ if environ is None else environ
    override = (env.get(FILE_VAR) or "").strip()
    if override:
        if override.startswith("~"):
            override = _home(env) + override[1:]
        return override
    return os.path.join(_home(env), ".switchyard", "gateway.env")


def _unquote(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def read_gateway_file(path):
    """{SWITCHYARD_URL, SWITCHYARD_API_KEY} found in the file (only those two). A missing or unreadable file
    gives {}, silently."""
    found = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(MAX_FILE_BYTES)
    except (OSError, ValueError):
        return found
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name.startswith("export "):
            name = name[len("export "):].strip()
        if name in (URL_VAR, KEY_VAR):
            found[name] = _unquote(value)
    return found


def gateway_settings(environ=None):
    """(url or None, key or None): each from its environment variable when set and not empty, else from the
    gateway file. The URL loses a trailing slash."""
    env = os.environ if environ is None else environ
    url = (env.get(URL_VAR) or "").strip()
    key = (env.get(KEY_VAR) or "").strip()
    if not url or not key:
        stored = read_gateway_file(gateway_file_path(env))
        url = url or (stored.get(URL_VAR) or "").strip()
        key = key or (stored.get(KEY_VAR) or "").strip()
    return (url.rstrip("/") or None), (key or None)


class GatewayFileError(Exception):
    """The gateway file cannot be written as asked."""


def write_gateway_file(url, key, path=None, environ=None):
    """Write URL and key to the gateway file atomically, mode 0600; the folder is made (0700) when missing.
    Returns the path."""
    url = (url or "").strip().rstrip("/")
    key = (key or "").strip()
    if not url.startswith(("http://", "https://")):
        raise GatewayFileError("the URL must start with https:// or http:// (got %r)" % url)
    if any(c in url for c in "\r\n \t"):
        raise GatewayFileError("the URL must not contain spaces or line breaks")
    if not key:
        raise GatewayFileError("the key is empty")
    if any(c in key for c in "\r\n"):
        raise GatewayFileError("the key must be one line")
    path = os.path.abspath(path or gateway_file_path(environ))
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder, mode=0o700)
        os.chmod(folder, 0o700)  # makedirs' mode is masked by the umask
    text = _merged_text(path, {URL_VAR: url, KEY_VAR: key})
    _replace_file(path, text)
    return path


_NEW_FILE_HEADER = ("# Switchyard gateway for the MCP scripts; written by switchyard.sh clients gateway.\n"
                    "# Keep this file private (mode 0600). --remove takes the two settings out again.\n")


def _our_line(line):
    """(prefix, name) when the line sets one of our two variables (with or without `export`), else None."""
    stripped = line.lstrip()
    indent = line[:len(line) - len(stripped)]
    prefix = ""
    if stripped.startswith("export ") or stripped.startswith("export\t"):
        prefix = stripped[:7]
        stripped = stripped[7:].lstrip()
    if "=" not in stripped:
        return None
    name = stripped.split("=", 1)[0].strip()
    if name in (URL_VAR, KEY_VAR):
        return indent + prefix, name
    return None


def _existing_lines(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read(MAX_FILE_BYTES).splitlines()
    except (OSError, ValueError):
        return None


def _merged_text(path, values):
    """The file's text with our two variables set to `values`. Every other line of an existing file is kept
    as it is (comments, other variables, an `export` in front of ours), because the file may also be read
    by the user's shell; a variable it does not hold yet is added at the end."""
    lines = _existing_lines(path)
    if lines is None:
        return _NEW_FILE_HEADER + "".join("%s=%s\n" % (name, values[name]) for name in (URL_VAR, KEY_VAR))
    left = dict(values)
    out = []
    for line in lines:
        ours = _our_line(line)
        if ours is None:
            out.append(line)
        elif ours[1] in left:
            out.append("%s%s=%s" % (ours[0], ours[1], left.pop(ours[1])))
        # a second line for the same variable is dropped
    out += ["%s=%s" % (name, left[name]) for name in (URL_VAR, KEY_VAR) if name in left]
    return "\n".join(out) + "\n"


def _replace_file(path, text):
    folder = os.path.dirname(path)
    fd, temp = tempfile.mkstemp(prefix=".gateway.", suffix=".tmp", dir=folder)  # created 0600
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, 0o600)
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


def key_hint(key):
    """What may be shown of a key: its last 4 characters when it is long enough, else nothing."""
    if not key:
        return ""
    return "..." + key[-4:] if len(key) >= 12 else ""


def show_gateway_file(path=None, environ=None):
    """{path, exists, url, key_set, key_hint} for the gateway file. Never the key itself."""
    path = os.path.abspath(path or gateway_file_path(environ))
    stored = read_gateway_file(path)
    key = (stored.get(KEY_VAR) or "").strip()
    return {"path": path, "exists": os.path.isfile(path),
            "url": (stored.get(URL_VAR) or "").strip().rstrip("/") or None,
            "key_set": bool(key), "key_hint": key_hint(key)}


def describe_gateway_file(info):
    """Lines for printing what show_gateway_file found."""
    if not info["exists"]:
        return ["%s: no gateway file" % info["path"]]
    key = "not set"
    if info["key_set"]:
        key = "set" + (" (ends %s)" % info["key_hint"][3:] if info["key_hint"] else "")
    lines = ["%s:" % info["path"], "  %s  %s" % (URL_VAR, info["url"] or "not set"), "  %s  %s" % (KEY_VAR, key)]
    try:
        mode = os.stat(info["path"]).st_mode & 0o777
        if mode & 0o077:
            lines.append("  warning: the file is readable by others (mode %04o); chmod 600 it" % mode)
    except OSError:
        pass
    return lines


def remove_gateway_file(path=None, environ=None):
    """Take the gateway settings out; True when there was a file. A file that holds nothing else is deleted;
    one with other settings (it may be read by the user's shell too) only loses our two lines."""
    path = os.path.abspath(path or gateway_file_path(environ))
    lines = _existing_lines(path)
    if lines is None:
        return False
    kept = [line for line in lines if _our_line(line) is None]
    if any(line.strip() and not line.strip().startswith("#") for line in kept):
        _replace_file(path, "\n".join(kept) + "\n")
        return True
    try:
        os.unlink(path)
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------- activity report


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # a 3xx becomes an HTTPError: the key is never sent anywhere else


def tokens_of(chars):
    """Characters / 4 as an int >= 0, as the rest of Switchyard estimates tokens."""
    try:
        return max(0, int(chars) // 4)
    except (TypeError, ValueError):
        return 0


def _off(value):
    return str(value or "").strip().lower() in ("0", "off", "false", "no")


class Reporter:
    """Posts this server's tool calls to the gateway (contract: POST /v1/mcp/activity). Inert without a
    gateway URL or with SWITCHYARD_MCP_REPORT=0."""

    POST_INTERVAL = 2.0     # seconds between posts at most
    HEARTBEAT = 30.0        # an empty post this often once the client has initialized
    RETRY_AFTER = 60.0      # after a failure, nothing is sent for this long
    TIMEOUT = 5.0
    FLUSH_WAIT = 1.0        # how long close() waits for the last post
    MAX_QUEUE = 500
    MAX_BATCH = 200

    def __init__(self, kind, server, environ=None, opener=None):
        self.kind = str(kind or "")
        self.server = str(server or "")
        self.client = ""
        self.client_version = ""
        self.session = secrets.token_hex(6)
        self.enabled = False
        try:
            env = os.environ if environ is None else environ
            url, key = gateway_settings(env)
            self.endpoint = (url + ACTIVITY_PATH) if url else None
            self.key = key
            self.enabled = bool(url) and not _off(env.get(REPORT_VAR))
        except Exception:
            self.endpoint, self.key = None, None
        self._opener = opener or urllib.request.build_opener(_NoRedirect())
        self._cond = threading.Condition()
        self._queue = []
        self._initialized = False
        self._closed = False
        self._thread = None
        self._last_post = None      # monotonic time of the last post attempt
        self._retry_at = 0.0
        self._warned = False
        self._post_now = False      # set_model: post at the next chance, whatever the interval
        self._in_flight = False     # the sender thread is posting events it has taken off the queue
        self.model = ""             # the model's own name, from report_model; "" until then
        self.posts = 0              # posts that the gateway accepted (for tests and debugging)
        self.failures = 0
        if self.enabled:
            atexit.register(self.close)

    # ------------------------------------------------------------ what the servers call

    def set_client(self, name, version=None):
        """The client app named in MCP initialize (clientInfo); starts the heartbeat."""
        try:
            self.client = str(name or "")[:80]
            self.client_version = str(version or "")[:80]
            if not self.enabled:
                return
            with self._cond:
                self._initialized = True
                self._start()
                self._cond.notify_all()
        except Exception:
            pass

    def set_model(self, name):
        """The model's name as it reported it (report_model): added to every report from now on, and a
        report carrying it is posted at once. An empty name is ignored."""
        try:
            name = str(name or "").strip()[:80]
            if not name:
                return
            self.model = name
            if not self.enabled:
                return
            with self._cond:
                if self._closed:
                    return
                self._post_now = True
                self._start()
                self._cond.notify_all()
        except Exception:
            pass

    def call(self, tool, duration_ms, result_tokens, sent_tokens, handle=None, is_error=False, saved_tokens=None):
        """Queue one finished tool call. Never raises, never blocks on the network."""
        if not self.enabled:
            return
        try:
            event = {"at": round(time.time(), 3), "tool": str(tool or "")[:80],
                     "duration_ms": max(0, int(duration_ms or 0)),
                     "result_tokens": max(0, int(result_tokens or 0)),
                     "sent_tokens": max(0, int(sent_tokens or 0)),
                     "is_error": bool(is_error)}
            if saved_tokens is not None:
                event["saved_tokens"] = max(0, int(saved_tokens))
            if handle:
                event["handle"] = str(handle)[:80]
            with self._cond:
                if self._closed:
                    return
                self._queue.append(event)
                if len(self._queue) > self.MAX_QUEUE:
                    del self._queue[:len(self._queue) - self.MAX_QUEUE]   # the oldest go
                self._start()
                self._cond.notify_all()
        except Exception:
            pass

    def close(self):
        """Stop the thread and send what is queued, waiting at most FLUSH_WAIT seconds."""
        try:
            with self._cond:
                if self._closed:
                    return
                self._closed = True
                events = self._queue[:self.MAX_BATCH]
                self._queue = []
                thread = self._thread if self._in_flight else None
                self._cond.notify_all()
            if not self.enabled:
                return
            deadline = time.monotonic() + self.FLUSH_WAIT
            if thread is not None and thread is not threading.current_thread():
                thread.join(self.FLUSH_WAIT)    # a post already under way carries events too: let it land
            if not events:
                return
            left = max(0.05, deadline - time.monotonic())
            sender = threading.Thread(target=self._post, args=(events, left), daemon=True)
            sender.start()
            sender.join(left)
        except Exception:
            pass

    # ------------------------------------------------------------ the thread

    def _start(self):
        """Start the sender thread (caller holds the lock)."""
        if self._thread is None and not self._closed:
            self._thread = threading.Thread(target=self._run, name="switchyard-mcp-report", daemon=True)
            self._thread.start()

    def _due(self, now):
        """When the next post may go (monotonic), or None for 'wait to be woken'."""
        if now < self._retry_at:
            return self._retry_at
        if self._post_now:
            return now
        last = self._last_post
        if self._queue:
            return now if last is None else last + self.POST_INTERVAL
        if self._initialized:
            return now if last is None else last + self.HEARTBEAT
        return None

    def _run(self):
        while True:
            try:
                with self._cond:
                    while True:
                        if self._closed:
                            return
                        now = time.monotonic()
                        due = self._due(now)
                        if due is not None and due <= now:
                            break
                        self._cond.wait(None if due is None else max(0.01, due - now))
                    events = self._queue[:self.MAX_BATCH]
                    del self._queue[:len(events)]
                    self._last_post = now
                    self._post_now = False
                    self._in_flight = True
                try:
                    ok, keep = self._post(events, self.TIMEOUT)
                finally:
                    self._in_flight = False
                if not ok:
                    with self._cond:
                        self._retry_at = time.monotonic() + self.RETRY_AFTER
                        if keep and events:
                            self._queue[:0] = events
                            if len(self._queue) > self.MAX_QUEUE:
                                del self._queue[:len(self._queue) - self.MAX_QUEUE]
            except Exception:
                time.sleep(1.0)  # never spin, never die quietly in a tight loop

    def _body(self, events):
        body = {"client": self.client, "client_version": self.client_version, "server": self.server,
                "kind": self.kind, "session": self.session, "events": events}
        if self.model:
            body["model"] = self.model
        return body

    def _post(self, events, timeout):
        """(sent, keep the events for a retry)."""
        try:
            data = json.dumps(self._body(events), ensure_ascii=False).encode("utf-8")
            headers = {"Content-Type": "application/json", "Accept": "application/json",
                       "User-Agent": "switchyard-mcp-report"}
            if self.key:
                headers["x-api-key"] = self.key
            request = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
            with self._opener.open(request, timeout=timeout) as response:
                response.read(4096)
            self.posts += 1
            return True, False
        except urllib.error.HTTPError as e:
            # the gateway refused these events (bad key, a body it can't take): sending them again won't help
            self._failed("HTTP %s" % e.code)
            return False, e.code >= 500 or e.code in (408, 429)
        except Exception as e:
            self._failed(type(e).__name__)
            return False, True

    def _failed(self, why):
        self.failures += 1
        if self._warned:
            return
        self._warned = True
        try:
            sys.stderr.write("switchyard: couldn't send MCP activity to the gateway (%s); tool calls are not "
                             "affected, trying again in %d seconds. SWITCHYARD_MCP_REPORT=0 turns this off.\n"
                             % (why, int(self.RETRY_AFTER)))
            sys.stderr.flush()
        except Exception:
            pass


# ---------------------------------------------------------------- the report_model tool

REPORT_MODEL = "report_model"
REPORT_MODEL_TOOL = {
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
# One sentence for a server's MCP instructions.
REPORT_MODEL_INSTRUCTION = ("Once per conversation, before your first other Switchyard tool call, call "
                            "report_model with your own model name if you know it; do not guess.")


def answer_report_model(reporter, args):
    """The tools/call result of report_model: hands the name to `reporter` (which may be None or inert) and
    answers one short line. Not reported as a tool call."""
    name = args.get("model") if isinstance(args, dict) else None
    name = name.strip()[:80] if isinstance(name, str) else ""
    if not name:
        return {"content": [{"type": "text", "text": "No model name given; nothing was noted."}],
                "isError": True}
    if reporter is not None:
        try:
            reporter.set_model(name)
        except Exception:
            pass
    return {"content": [{"type": "text", "text": "Noted: %s." % name}], "isError": False}
