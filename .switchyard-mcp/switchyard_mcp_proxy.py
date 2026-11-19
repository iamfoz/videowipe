#!/usr/bin/env python3
"""Wrap another stdio MCP server so its big results arrive as handles.

    switchyard_mcp_proxy.py [--name NAME] [--threshold-tokens 1500]
                            [--no-handle-tools] -- <command> [args...]

Claude Code starts this instead of the real server (switchyard_mcp_cli.py
init rewrites the config). It starts <command>, and passes MCP's
newline-delimited JSON-RPC through both ways untouched: requests, responses,
notifications and cancellations keep their ids and bytes, the child's stderr
goes to ours, and lines that aren't JSON are forwarded as they are. It
changes three things:

  - a tools/call result whose text is over --threshold-tokens (characters / 4)
    is stored in the handle store (switchyard_handles.open_store: SQLite, or
    the gateway at SWITCHYARD_URL) as "<name>/<tool>", and its text blocks are
    replaced by the same stub the Switchyard proxy writes for old results
    (switchyard_handles.store_tool_result). Images and other blocks stay.
    structuredContent is dropped from a stubbed result (it repeats the text),
    so outputSchema is dropped from the tools list: a client that checks
    results against it would otherwise reject the stubbed one;
  - tools/list gets the handles_* tools of switchyard_handles_mcp.py and
    report_model added (unless --no-handle-tools), so the model can read a
    handle back through the same server; this proxy answers those calls itself;
  - initialize's capabilities get "tools" if the child didn't list it, and
    one sentence asking for report_model is added after the child's own
    instructions, when the handle tools are on.

With a gateway (SWITCHYARD_URL, from the environment or ~/.switchyard/gateway.env)
each tools/call is reported to it as it passes back (switchyard_gateway_client.
Reporter): the tool's name, how long it took, the size of the real result and of
what the client got, the handle id and isError; never the arguments or the text.
The client's name comes from initialize's clientInfo, the model's name from a
report_model call (which is itself not reported). SWITCHYARD_MCP_REPORT=0
turns this off.

When the child exits, this exits with its code. When stdin closes, the
child's stdin is closed too, and the child is stopped if it doesn't exit
within a few seconds. --threshold-tokens 0 turns stubbing off.

Standard library only, Python 3.9 compatible; run it with python3 -I.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python3 -I leaves the script's folder off sys.path

import switchyard_gateway_client as gateway_client  # noqa: E402
import switchyard_handles as handles  # noqa: E402
import switchyard_handles_mcp as handle_tools  # noqa: E402

DEFAULT_THRESHOLD_TOKENS = 1500
EXIT_GRACE_SECONDS = 3.0


def log(message: str) -> None:
    try:
        sys.stderr.write(f"switchyard-mcp-proxy: {message}\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


def parse_args(argv):
    """(options, child command). Everything after the first "--" is the child."""
    if "--" in argv:
        split = argv.index("--")
        own, command = argv[:split], argv[split + 1:]
    else:
        own, command = argv, []
    parser = argparse.ArgumentParser(
        prog="switchyard_mcp_proxy.py",
        usage="%(prog)s [--name NAME] [--threshold-tokens N] [--no-handle-tools] -- <command> [args...]",
        description="Wrap a stdio MCP server; store its big tool results as Switchyard handles.")
    parser.add_argument("--name", help="Server name for stored handles (default: the command's name).")
    parser.add_argument("--threshold-tokens", type=int, default=DEFAULT_THRESHOLD_TOKENS,
                        help="Store results over this many tokens, estimated as characters / 4 "
                             f"(default {DEFAULT_THRESHOLD_TOKENS}; 0 turns stubbing off).")
    parser.add_argument("--no-handle-tools", action="store_true",
                        help="Don't add the handles_* tools and report_model to the wrapped server.")
    options = parser.parse_args(own)
    if not command:
        parser.error("give the MCP server to wrap after --, e.g. -- npx -y some-mcp-server")
    if not options.name:
        options.name = os.path.basename(command[0]) or "mcp"
    return options, command


def rid_key(rid):
    """A hashable key for a JSON-RPC id that keeps 1 and "1" apart."""
    return json.dumps(rid, sort_keys=True)


class Proxy:
    def __init__(self, options, command, stdin=None, stdout=None):
        self.name = options.name
        self.threshold_chars = max(0, options.threshold_tokens) * handles.CHARS_PER_TOKEN
        self.handle_tools = not options.no_handle_tools
        self.command = command
        self.stdin = stdin or sys.stdin.buffer
        self.stdout = stdout or sys.stdout.buffer
        self.child = None
        self.out_lock = threading.Lock()
        self.lock = threading.Lock()
        self.pending = {}     # rid_key -> (method, params, monotonic start) of client requests sent to the child
        self.local = set()    # rid_keys of handles_* calls this proxy is answering
        self.workers = []
        self.stopped_by_us = False
        self.reporter = gateway_client.Reporter("proxy", self.name)

    # ------------------------------------------------------------ output

    def write_client(self, data: bytes) -> None:
        if not data.endswith(b"\n"):
            data += b"\n"
        with self.out_lock:
            try:
                self.stdout.write(data)
                self.stdout.flush()
            except (OSError, ValueError):
                pass  # the client has gone away

    def send_client(self, message) -> None:
        self.write_client(json.dumps(message, ensure_ascii=False).encode("utf-8"))

    def spawn(self, target, *args) -> None:
        thread = threading.Thread(target=target, args=args, daemon=True)
        with self.lock:
            self.workers = [t for t in self.workers if t.is_alive()]
            self.workers.append(thread)
        thread.start()

    # ------------------------------------------------------------ client -> child

    def pump_client(self) -> None:
        try:
            for line in iter(self.stdin.readline, b""):
                if self.from_client(line):
                    try:
                        self.child.stdin.write(line if line.endswith(b"\n") else line + b"\n")
                        self.child.stdin.flush()
                    except (OSError, ValueError):
                        break  # the child is gone; main() sees it exit
        except (OSError, ValueError):
            pass
        self.client_closed()

    def from_client(self, line: bytes) -> bool:
        """Handle one line from the client; True to forward it to the child."""
        try:
            message = json.loads(line)
        except ValueError:
            return True
        if not isinstance(message, dict):
            return True
        method, rid = message.get("method"), message.get("id")
        if not isinstance(method, str):
            return True  # a response to a request the child made
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        if method == "notifications/cancelled":
            target = rid_key(params.get("requestId"))
            with self.lock:
                if target in self.local:
                    return False  # ours: the answer is cheap, let it finish
                self.pending.pop(target, None)
            return True
        if rid is None:
            return True
        if method == "tools/call" and self.handle_tools and params.get("name") == gateway_client.REPORT_MODEL:
            # answered here, at once; it only names the model, so it is not reported as a call
            result = gateway_client.answer_report_model(self.reporter, params.get("arguments") or {})
            self.send_client({"jsonrpc": "2.0", "id": rid, "result": result})
            return False
        if (method == "tools/call" and self.handle_tools
                and params.get("name") in handle_tools.TOOL_NAMES):
            key = rid_key(rid)
            with self.lock:
                self.local.add(key)
            self.spawn(self.serve_handle_tool, rid, params)
            return False
        if method == "initialize":
            info = params.get("clientInfo") if isinstance(params.get("clientInfo"), dict) else {}
            self.reporter.set_client(info.get("name"), info.get("version"))
        if method in ("tools/call", "tools/list", "initialize"):
            with self.lock:
                self.pending[rid_key(rid)] = (method, params, time.monotonic())
        return True

    def serve_handle_tool(self, rid, params) -> None:
        started = time.monotonic()
        try:
            result = handle_tools.call_tool(params.get("name"), params.get("arguments") or {})
        finally:
            with self.lock:
                self.local.discard(rid_key(rid))
        self.send_client({"jsonrpc": "2.0", "id": rid, "result": result})
        size = gateway_client.tokens_of(len(self.all_text(result)))
        self.reporter.call(params.get("name"), self.elapsed_ms(started), size, size,
                           is_error=result.get("isError") is True, saved_tokens=0)

    @staticmethod
    def elapsed_ms(started) -> float:
        return (time.monotonic() - started) * 1000

    def client_closed(self) -> None:
        """stdin closed: close the child's, give it a moment, then stop it."""
        try:
            self.child.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            self.child.wait(timeout=EXIT_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            self.stop_child()

    def stop_child(self) -> None:
        self.stopped_by_us = True
        try:
            self.child.terminate()
            self.child.wait(timeout=EXIT_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            self.child.kill()
        except OSError:
            pass

    # ------------------------------------------------------------ child -> client

    def pump_child(self) -> None:
        try:
            for line in iter(self.child.stdout.readline, b""):
                self.from_child(line)
        except (OSError, ValueError):
            pass

    def from_child(self, line: bytes) -> None:
        try:
            message = json.loads(line)
        except ValueError:
            self.write_client(line)
            return
        if (not isinstance(message, dict) or "method" in message or "id" not in message
                or not isinstance(message.get("result"), dict)):
            request = None
            if isinstance(message, dict) and "error" in message and "method" not in message:
                with self.lock:
                    request = self.pending.pop(rid_key(message.get("id")), None)
            self.write_client(line)
            if request is not None and request[0] == "tools/call":
                error = message.get("error")
                size = gateway_client.tokens_of(len(str(error.get("message") or "")) if isinstance(error, dict) else 0)
                self.reporter.call(request[1].get("name"), self.elapsed_ms(request[2]), size, size, is_error=True)
            return
        with self.lock:
            request = self.pending.pop(rid_key(message["id"]), None)
        if request is None:
            self.write_client(line)
            return
        method, params, started = request
        result = message["result"]
        if method == "tools/list":
            if self.edit_tools_list(result, params):
                self.send_client(message)
                return
        elif method == "initialize":
            if self.edit_initialize(result):
                self.send_client(message)
                return
        elif method == "tools/call" and self.big_text(result) is not None:
            # storing can take a moment (a remote gateway): don't hold up other replies
            self.spawn(self.stub_and_send, message, params, started)
            return
        self.write_client(line)
        if method == "tools/call":
            size = gateway_client.tokens_of(len(self.all_text(result)))
            self.reporter.call(params.get("name"), self.elapsed_ms(started), size, size,
                               is_error=result.get("isError") is True)

    def edit_initialize(self, result: dict) -> bool:
        """With the handle tools on: add "tools" to the capabilities when missing, and the report_model
        sentence after the child's own instructions (their text is kept as it is); True if changed."""
        if not self.handle_tools:
            return False
        changed = False
        if isinstance(result.get("capabilities"), dict) and "tools" not in result["capabilities"]:
            result["capabilities"]["tools"] = {}
            changed = True
        own = result.get("instructions")
        if own is None or isinstance(own, str):
            result["instructions"] = (own + "\n\n" if own else "") + gateway_client.REPORT_MODEL_INSTRUCTION
            changed = True
        return changed

    def edit_tools_list(self, result: dict, params: dict) -> bool:
        """Add the handle tools and report_model (first page only) and drop outputSchema; True if changed."""
        tools = result.get("tools")
        if not isinstance(tools, list):
            return False
        changed = False
        if self.threshold_chars:
            for tool in tools:
                if isinstance(tool, dict) and "outputSchema" in tool:
                    del tool["outputSchema"]
                    changed = True
        if self.handle_tools and not params.get("cursor"):
            ours = handle_tools.TOOL_NAMES | {gateway_client.REPORT_MODEL}
            kept = [t for t in tools if not (isinstance(t, dict) and t.get("name") in ours)]
            result["tools"] = kept + [dict(tool) for tool in handle_tools.SERVER_TOOLS]
            changed = True
        return changed

    @staticmethod
    def text_blocks(result: dict) -> list:
        content = result.get("content")
        if not isinstance(content, list):
            return []
        return [block.get("text") for block in content
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)]

    def all_text(self, result) -> str:
        """The result's text blocks joined (what the activity report measures)."""
        return "\n".join(self.text_blocks(result)) if isinstance(result, dict) else ""

    def big_text(self, result: dict):
        """The result's text blocks joined, if over the threshold; else None."""
        if not self.threshold_chars:
            return None
        texts = self.text_blocks(result)
        if not texts:
            return None
        text = "\n".join(texts)
        return text if len(text) > self.threshold_chars else None

    def stub_and_send(self, message: dict, params: dict, started: float) -> None:
        result = message["result"]
        text = self.big_text(result)
        tool = str(params.get("name") or "tool")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        full = gateway_client.tokens_of(len(text))
        is_error = result.get("isError") is True
        try:
            stored = handles.store_tool_result(text, tool_name=f"{self.name}/{tool}", tool_input=arguments,
                                               session=os.environ.get("SWITCHYARD_SESSION") or None)
        except Exception as e:  # never lose the result: send it whole
            log(f"{self.name}/{tool}: couldn't store a {len(text):,}-character result, "
                f"sent it whole: {type(e).__name__}: {e}")
            self.send_client(message)
            self.reporter.call(tool, self.elapsed_ms(started), full, full, is_error=is_error)
            return
        stub = stored["stub"]
        if len(stub) >= len(text):
            self.send_client(message)
            self.reporter.call(tool, self.elapsed_ms(started), full, full, is_error=is_error)
            return
        content, placed = [], False
        for block in result["content"]:
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                if not placed:
                    content.append({"type": "text", "text": stub})
                    placed = True
                continue
            content.append(block)
        result["content"] = content
        result.pop("structuredContent", None)
        log(f"{self.name}/{tool}: stored {len(text):,} characters as {stored['id']}")
        self.send_client(message)
        self.reporter.call(tool, self.elapsed_ms(started), full, gateway_client.tokens_of(len(stub)),
                           handle=stored.get("id"), is_error=is_error)

    # ------------------------------------------------------------ run

    def run(self) -> int:
        try:
            self.child = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=None, bufsize=0)
        except OSError as e:
            log(f"can't start {self.command[0]}: {e}")
            return 127

        def on_signal(signum, frame):
            # main() is waiting on the child: stop it (and kill it later if it
            # ignores that), and the wait ends
            self.stopped_by_us = True
            try:
                self.child.terminate()
            except OSError:
                pass
            killer = threading.Timer(EXIT_GRACE_SECONDS, lambda: self.child.poll() is None and self.child.kill())
            killer.daemon = True
            killer.start()

        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            try:
                signal.signal(signum, on_signal)
            except (OSError, ValueError, AttributeError):
                pass
        threading.Thread(target=self.pump_client, daemon=True).start()
        reader = threading.Thread(target=self.pump_child, daemon=True)
        reader.start()
        code = self.child.wait()
        reader.join(timeout=5)  # everything the child wrote before exiting
        with self.lock:
            workers = list(self.workers)
        for thread in workers:
            thread.join(timeout=15)
        if self.stopped_by_us:
            return 0
        return code if code >= 0 else 128 - code


def main(argv=None) -> int:
    options, command = parse_args(sys.argv[1:] if argv is None else argv)
    proxy = Proxy(options, command)
    try:
        return proxy.run()
    finally:
        proxy.reporter.close()  # os._exit below skips atexit


if __name__ == "__main__":
    code = main()
    # A reader thread may still be blocked on stdin; a normal interpreter
    # shutdown would abort on its buffer lock, so flush and leave directly.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    os._exit(code)
