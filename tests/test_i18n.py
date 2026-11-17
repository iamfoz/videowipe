"""Tests for the runtime translation framework (videowipe.i18n).

The shared algorithm is verified two ways:
  * Python ``translate_with`` against every vector in
    ``tests/fixtures/i18n_vectors.json``;
  * the JS implementation (``web/i18n.js``) run under node against the same
    vectors (skipped when node is absent), mirroring test_server.py's node usage.
The remaining tests cover catalogue loading, language resolution, the
``render_index`` injection, the index route, and tolerance of a broken
catalogue.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from videowipe import i18n
from videowipe.server import app as server_app

_VECTORS_PATH = Path(__file__).resolve().parent / "fixtures" / "i18n_vectors.json"
_I18N_JS_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "videowipe" / "web" / "i18n.js"
)
_NODE = shutil.which("node")


def _vectors() -> list[dict]:
    return json.loads(_VECTORS_PATH.read_text(encoding="utf-8"))


def _node_has_playwright() -> bool:
    if not _NODE:
        return False
    probe = 'try { require("playwright"); console.log("1"); } catch (e) { console.log("0"); }'
    result = subprocess.run(
        ["node", "-e", probe], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() == "1"


_HAS_PLAYER = _node_has_playwright()


# ---------------------------------------------------------------------------
# Shared vectors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("vector", _vectors(), ids=lambda v: v.get("name", ""))
def test_python_vectors(vector):
    assert (
        i18n.translate_with(vector["input"], vector["messages"]) == vector["expected"]
    )


def test_vectors_fixture_well_formed():
    for vector in _vectors():
        assert set(vector) == {"name", "messages", "input", "expected"}
        assert isinstance(vector["messages"], dict)


@pytest.mark.skipif(not _NODE, reason="requires node")
@pytest.mark.parametrize(
    "href,expected",
    [
        ("http://x/?lang=en&foo=1#frag", "http://x/?foo=1#frag"),
        ("http://x/?foo=1&lang=en", "http://x/?foo=1"),
        ("http://x/?lang=en", "http://x/"),
        ("http://x/?a=1&lang=en&b=2", "http://x/?a=1&b=2"),
        ("http://x/noquery", "http://x/noquery"),
        ("http://x/?x=1", "http://x/?x=1"),
    ],
)
def test_js_strip_lang_param(href, expected):
    """setLanguage reloads without ?lang=; verify the URL is left clean."""
    harness = f"""
global.window = global;
global.document = {{ readyState: "complete", addEventListener(){{}}, documentElement: {{ lang: "" }} }};
global.navigator = {{ languages: [] }};
require({json.dumps(str(_I18N_JS_PATH))});
const got = window.VideoWipeI18n.stripLangParam({json.dumps(href)});
if (got !== {json.dumps(expected)}) {{ console.error("got", got, "want", {json.dumps(expected)}); process.exit(1); }}
console.log("ok");
"""
    subprocess.run(["node", "-e", harness], check=True, capture_output=True, text=True)


# A small fake-DOM harness (no external dependency) so the DOM layer of
# web/i18n.js can be exercised under node: recursive text translation,
# attribute/button-input translation, script/textarea/contenteditable pruning,
# the injected language <select>, and the MutationObserver path (including the
# idempotency that prevents infinite loops).
_DOM_HARNESS = r"""
const TEXT_NODE = 3, ELEMENT_NODE = 1;
function el(t) {
  const o = { nodeType: ELEMENT_NODE, nodeName: t.toUpperCase(), tagName: t.toUpperCase(),
    children: [], childNodes: [], attrs: {}, value: "", type: "", isContentEditable: false,
    className: "", id: "", classList: { toggle(){}, add(){}, remove(){} },
    style: { setProperty(){}, cssText: "" }, parentElement: null, parentNode: null,
    appendChild(c) { this.children.push(c); this.childNodes.push(c); c.parentElement = this; c.parentNode = this; return c; },
    after(n) { if (this.parentElement) { const i = this.parentElement.children.indexOf(this);
      this.parentElement.children.splice(i + 1, 0, n); this.parentElement.childNodes.splice(i + 1, 0, n);
      n.parentElement = this.parentElement; n.parentNode = this.parentElement; } },
    hasAttribute(a) { return a in this.attrs; }, getAttribute(a) { return a in this.attrs ? this.attrs[a] : null; },
    setAttribute(a, v) { this.attrs[a] = v; }, removeAttribute(a) { delete this.attrs[a]; }, addEventListener() {},
    get options() { return this.children.filter(c => c.nodeName === "OPTION"); },
    querySelector(sel) { const key = sel.replace(/^\./, "");
      (function r(e) { for (const c of e.children) if (c.nodeType === ELEMENT_NODE) {
        if (c.className === key || c.id === key) return c; const f = r(c); if (f) return f; } })(this); return null; },
    querySelectorAll() { const o = []; (function r(e) { for (const c of e.children)
      if (c.nodeType === ELEMENT_NODE) { o.push(c); r(c); } })(this); return o; },
    closest() { return null; } };
  Object.defineProperty(o, "textContent", { set(v) { o._tc = v; }, get() { return o._tc || ""; } });
  return o;
}
function text(s) { return { nodeType: TEXT_NODE, nodeValue: s, parentNode: null, parentElement: null }; }
const body = el("body"), html = el("html"), head = el("head");
const topbar = el("header"); topbar.className = "topbar";
const spacer = el("div"); spacer.className = "spacer"; topbar.appendChild(spacer);
const deep = el("main"), inner = el("div"); inner.appendChild(text("检测目标")); deep.appendChild(inner);
const btn = el("button"); btn.type = "button";
const input = el("input"); input.type = "text"; input.attrs["placeholder"] = "选择视频";
const submit = el("input"); submit.type = "submit"; submit.value = "重试";
const script = el("script"); script.appendChild(text("中文代码"));
const ta = el("textarea"); ta.appendChild(text("用户输入"));
body.appendChild(topbar); body.appendChild(deep); body.appendChild(btn);
body.appendChild(input); body.appendChild(submit); body.appendChild(script); body.appendChild(ta);
html.appendChild(head); html.appendChild(body);
let observerCb = null;
global.MutationObserver = class { constructor(cb) { observerCb = cb; } observe() {} };
global.window = global;
global.Node = { TEXT_NODE, ELEMENT_NODE };
global.NodeFilter = { SHOW_ELEMENT: 1, SHOW_TEXT: 4, FILTER_ACCEPT: 1, FILTER_REJECT: 2, FILTER_SKIP: 3 };
global.URLSearchParams = URLSearchParams;
global.location = { href: "http://x/", search: "?lang=en", replace(u) { global.__reloaded = u; } };
global.localStorage = { _d: {}, getItem(k) { return this._d[k] ?? null; }, setItem(k, v) { this._d[k] = v; }, removeItem(k) { delete this._d[k]; } };
global.document = {
  readyState: "complete", addEventListener() {}, documentElement: html, body, createTreeWalker: () => ({ nextNode() { return null; } }),
  getElementById(id) { return (function f(e) { for (const c of e.children) { if (c.id === id) return c;
    if (c.nodeType === ELEMENT_NODE) { const r = f(c); if (r) return r; } } })(body) || null; },
  querySelector(sel) { return sel === ".topbar" ? topbar : null; },
  createElement(t) { return el(t); }
};
window.__VIDEOWIPE_I18N__ = { source: "zh-CN", default: "en",
  locales: { en: { name: "English", messages: { "检测目标": "Detect targets", "选择视频": "Choose video", "重试": "Retry" } } } };
require(__I18N_JS__);
const assert = require("node:assert/strict");
assert.equal(window.VideoWipeI18n.lang, "en", "lang resolved to en");
assert.equal(html.lang, "en", "documentElement.lang set");
assert.equal(inner.childNodes[0].nodeValue, "Detect targets", "nested text translated");
assert.equal(input.attrs["placeholder"], "Choose video", "placeholder translated");
assert.equal(submit.value, "Retry", "submit button value translated");
assert.equal(script.childNodes[0].nodeValue, "中文代码", "script untouched");
assert.equal(ta.childNodes[0].nodeValue, "用户输入", "textarea untouched");
assert.equal(topbar.children.length, 2, "select added to topbar");
const sel = topbar.children[1];
assert.equal(sel.nodeName, "SELECT", "injected node is a SELECT");
assert.equal(sel.options.length, 2, "options: 中文 + English");
assert.equal(sel.options[0].textContent, "中文");
assert.equal(sel.options[0].value, "zh-CN");
assert.equal(sel.options[1].textContent, "English");
assert.equal(sel.options[1].value, "en");
const dyn = text("重试"); btn.appendChild(dyn);
assert.ok(observerCb, "observer installed");
observerCb([{ type: "childList", addedNodes: [dyn], removedNodes: [], target: btn }]);
assert.equal(dyn.nodeValue, "Retry", "observer translated added text");
observerCb([{ type: "childList", addedNodes: [dyn], removedNodes: [], target: btn }]);
assert.equal(dyn.nodeValue, "Retry", "idempotent, no loop");
console.log("dom-ok");
"""


@pytest.mark.skipif(not _NODE, reason="requires node")
def test_js_dom_runtime():
    """The DOM layer translates text/attributes/inputs, prunes code blocks,
    injects the language select, and observes later mutations idempotently."""
    harness = _DOM_HARNESS.replace("__I18N_JS__", json.dumps(str(_I18N_JS_PATH)))
    subprocess.run(["node", "-e", harness], check=True, capture_output=True, text=True)


@pytest.mark.skipif(not _NODE, reason="requires node")
def test_js_vectors_match_python():
    """The JS translateWith must pass the same vectors as the Python one."""
    harness = f"""
const fs = require("node:fs");
global.window = global;
global.document = {{ readyState: "complete", addEventListener(){{}}, documentElement: {{ lang: "" }} }};
global.navigator = {{ languages: [] }};
require({json.dumps(str(_I18N_JS_PATH))});
const api = window.VideoWipeI18n;
const vectors = JSON.parse(fs.readFileSync({json.dumps(str(_VECTORS_PATH))}, "utf-8"));
let fail = 0;
for (const v of vectors) {{
  const got = api.translateWith(v.input, v.messages);
  if (got !== v.expected) {{ fail++; console.error("FAIL", v.name, JSON.stringify({{got, want: v.expected}})); }}
}}
if (fail) {{ console.error(fail + " vector failures"); process.exit(1); }}
console.log("js-vectors-ok");
"""
    subprocess.run(["node", "-e", harness], check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Catalogue loading
# ---------------------------------------------------------------------------
def test_load_catalogue_returns_messages():
    catalogue = i18n.load_catalogue("en")
    assert catalogue is not None
    assert catalogue.get("检测目标")
    # Empty values are dropped from the loaded message map.
    assert "" not in catalogue.values()


def test_available_locales_contains_en():
    assert "en" in i18n.available_locales()


def _clear_cache(monkeypatch):
    monkeypatch.setattr(i18n, "_CATALOGUE_CACHE", {})


def test_broken_catalogue_is_tolerated(tmp_path, monkeypatch):
    """A missing or invalid-JSON catalogue must not crash; it is skipped."""
    _clear_cache(monkeypatch)
    fake_dir = tmp_path / "locales"
    fake_dir.mkdir()
    (fake_dir / "en.json").write_text("{ this is not valid json", encoding="utf-8")
    monkeypatch.setattr(i18n, "_LOCALES_DIR", fake_dir)

    assert i18n.load_catalogue("en") is None
    assert "en" not in i18n.available_locales()
    monkeypatch.setenv("VIDEOWIPE_LANG", "en")
    # Unavailable catalogue -> identity, never an exception.
    assert i18n.translate("检测目标", lang="en") == "检测目标"


def test_empty_catalogue_directory_is_tolerated(tmp_path, monkeypatch):
    _clear_cache(monkeypatch)
    fake_dir = tmp_path / "locales"
    fake_dir.mkdir()
    monkeypatch.setattr(i18n, "_LOCALES_DIR", fake_dir)
    assert i18n.available_locales() == []


# ---------------------------------------------------------------------------
# translate() behaviour (through the resolved/en catalogue)
# ---------------------------------------------------------------------------
def test_source_language_is_identity(monkeypatch):
    assert i18n.translate("检测目标", lang="zh-CN") == "检测目标"
    # A catalogue language with a real entry translates.
    assert i18n.translate("检测目标", lang="en") == "Detect targets"


def test_translate_missing_key_is_identity(monkeypatch):
    assert i18n.translate("未收录的字符串", lang="en") == "未收录的字符串"


def test_translate_preserves_whitespace(monkeypatch):
    assert i18n.translate("  检测目标  ", lang="en") == "  Detect targets  "


def test_translate_empty_string_and_ascii_are_identity(monkeypatch):
    assert i18n.translate("", lang="en") == ""
    assert i18n.translate("plain ascii", lang="en") == "plain ascii"


# ---------------------------------------------------------------------------
# Language resolution (Python/CLI)
# ---------------------------------------------------------------------------
def _resolve(env: dict) -> str:
    merged = dict(os.environ)
    for key in ("VIDEOWIPE_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        merged.pop(key, None)
    merged.update(env)
    return i18n.resolve_language(merged)


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, "en"),
        ({"VIDEOWIPE_LANG": "en"}, "en"),
        ({"VIDEOWIPE_LANG": "zh-CN"}, "zh-CN"),
        ({"VIDEOWIPE_LANG": "zh"}, "zh-CN"),
        ({"LANG": "en_GB.UTF-8"}, "en"),
        ({"LC_ALL": "en_US.UTF-8"}, "en"),
        ({"LC_MESSAGES": "en"}, "en"),
        ({"LANG": "zh_CN.UTF-8"}, "zh-CN"),
        # VIDEOWIPE_LANG wins over the locale environment variables.
        ({"VIDEOWIPE_LANG": "en", "LANG": "zh_CN.UTF-8"}, "en"),
        # LC_ALL outranks LC_MESSAGES outranks LANG.
        ({"LC_ALL": "zh_CN.UTF-8", "LANG": "en"}, "zh-CN"),
    ],
)
def test_resolve_language(env, expected):
    assert _resolve(env) == expected


# ---------------------------------------------------------------------------
# render_index injection
# ---------------------------------------------------------------------------
def _payload(rendered: str) -> dict:
    raw = rendered.split("window.__VIDEOWIPE_I18N__=", 1)[1].split(";</script>", 1)[0]
    return json.loads(raw)


def test_render_index_injects_after_head():
    html = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"></head><body>x</body></html>'
    rendered = i18n.render_index(html)
    assert rendered != html
    # Injected right after the opening <head> tag.
    head_end = html.index("<head>") + len("<head>")
    assert rendered[head_end:].lstrip().startswith("<script>window.__VIDEOWIPE_I18N__=")
    # The JS runtime and the catalogue are both embedded.
    assert "VideoWipeI18n" in rendered
    data = _payload(rendered)
    assert data["source"] == "zh-CN"
    assert data["default"] == "en"
    assert "en" in data["locales"]
    assert data["locales"]["en"]["name"] == "English"
    assert data["locales"]["en"]["messages"]["检测目标"] == "Detect targets"
    # Only non-empty entries ship.
    assert "" not in data["locales"]["en"]["messages"].values()


def test_render_index_escaping_of_script_close():
    # The payload must not embed a raw "</" that could terminate the <script>.
    rendered = i18n.render_index("<head></head>")
    raw = rendered.split("window.__VIDEOWIPE_I18N__=", 1)[1].split(";</script>", 1)[0]
    assert "</" not in raw
    # And it still parses.
    assert json.loads(raw)


def test_render_index_no_head_is_unchanged():
    html = "<body>no head here</body>"
    assert i18n.render_index(html) == html


def test_render_index_with_attributes_on_head():
    html = '<html><head data-x="y">text</head><body></body>'
    rendered = i18n.render_index(html)
    assert "window.__VIDEOWIPE_I18N__=" in rendered
    assert rendered.index('<head data-x="y">') < rendered.index(
        "window.__VIDEOWIPE_I18N__="
    )


# ---------------------------------------------------------------------------
# Index route
# ---------------------------------------------------------------------------
def test_index_route_returns_injected_page(monkeypatch):
    monkeypatch.setenv("VIDEOWIPE_JOBS_DIR", "/tmp/videowipe-i18n-jobs")
    with TestClient(server_app.app) as test_client:
        response = test_client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "window.__VIDEOWIPE_I18N__=" in body
    assert "VideoWipeI18n" in body
    # The original page content survives the injection.
    assert 'main class="shell"' in body


# ---------------------------------------------------------------------------
# Headless MutationObserver check (optional; needs node + playwright + chromium)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_PLAYER, reason="requires node + playwright")
def test_dom_translation_headless(tmp_path):
    """A headless browser translates text set after load via MutationObserver."""
    rendered = i18n.render_index(
        '<!doctype html><html lang="zh-CN"><head><title>检测目标</title></head><body>'
        '<header class="topbar"></header>'
        '<p id="static">检测目标</p>'
        '<button id="dyn" type="button"></button>'
        '<input id="ph" type="text" placeholder="选择视频">'
        "</body></html>"
    )
    page_path = tmp_path / "page.html"
    page_path.write_text(rendered, encoding="utf-8")
    script = f"""
(async () => {{
  const {{ chromium }} = require("playwright");
  const browser = await chromium.launch();
  const page = await browser.newPage();
  page.on("console", (msg) => console.error("PAGE", msg.type(), msg.text()));
  await page.goto("file://{page_path}");
  await page.waitForFunction(() => window.VideoWipeI18n && window.VideoWipeI18n.lang === "en", null, {{timeout: 5000}});
  const staticText = await page.textContent("#static");
  await page.evaluate(() => {{ document.querySelector("#dyn").textContent = "下载成片"; }});
  await page.waitForFunction(
    () => document.querySelector("#dyn").textContent === "Download the finished video",
    null, {{ timeout: 5000 }}
  );
  // An element inserted with its attribute already set is translated too.
  await page.evaluate(() => {{
    const d = document.createElement("div"); d.id = "box";
    d.setAttribute("aria-label", "下载成片"); document.body.appendChild(d);
  }});
  await page.waitForFunction(
    () => document.querySelector("#box").getAttribute("aria-label") === "Download the finished video",
    null, {{ timeout: 5000 }}
  );
  // Setting an already-translated text node back to Chinese re-translates it.
  await page.evaluate(() => {{ document.querySelector("#dyn").firstChild.nodeValue = "选择视频"; }});
  await page.waitForFunction(
    () => document.querySelector("#dyn").textContent === "Choose video", null, {{ timeout: 5000 }}
  );
  await page.evaluate(() => {{ document.querySelector("#dyn").firstChild.nodeValue = "下载成片"; }});
  await page.waitForFunction(
    () => document.querySelector("#dyn").textContent === "Download the finished video",
    null, {{ timeout: 5000 }}
  );
  const title = await page.title();
  if (title !== "Detect targets") {{ console.error("title", title); process.exit(1); }}
  const placeholder = await page.getAttribute("#ph", "placeholder");
  const selectPresent = await page.$("#videowipe-lang");
  await browser.close();
  if (staticText !== "Detect targets") {{ console.error("static", staticText); process.exit(1); }}
  if (placeholder !== "Choose video") {{ console.error("placeholder", placeholder); process.exit(1); }}
  if (!selectPresent) {{ console.error("language select missing"); process.exit(1); }}
  console.log("dom-ok");
}})().catch((e) => {{ console.error(e); process.exit(1); }});
"""
    subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)


def test_extra_fragments_are_merged_into_catalogue(tmp_path, monkeypatch):
    (tmp_path / "xx.json").write_text(
        json.dumps({"_meta": {"name": "X"}, "messages": {"调整": "Adjust", "区域": ""},
                    "extra": {"区域": "area", "边缘": "edge"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(i18n, "_LOCALES_DIR", tmp_path)
    monkeypatch.setattr(i18n, "_CATALOGUE_CACHE", {})
    catalogue = i18n.load_catalogue("xx")
    # An empty extracted entry does not mask a curated extra fragment.
    assert catalogue == {"调整": "Adjust", "区域": "area", "边缘": "edge"}
    assert i18n.translate("调整区域", "xx") == "Adjust area"
