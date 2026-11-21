"""The browser translator caches compiled patterns and results; its output must not change.

``tests/fixtures/i18n_unmemoised.js`` is web/i18n.js exactly as it was before
the caches were added. Both files are run under node over the whole catalogue,
instantiated patterns, fragments and progress-style strings, and must agree
string for string, on a first call and on a repeat that is answered from the cache.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from videowipe import i18n

_ROOT = Path(__file__).resolve().parents[1]
_NEW = _ROOT / "src" / "videowipe" / "web" / "i18n.js"
_OLD = _ROOT / "tests" / "fixtures" / "i18n_unmemoised.js"
_VECTORS = _ROOT / "tests" / "fixtures" / "i18n_vectors.json"
_NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(not _NODE, reason="requires node")

_HARNESS = r"""
const fs = require("node:fs");
const vm = require("node:vm");
const [oldPath, newPath, vectorsPath] = process.argv.slice(1);
const catalogue = JSON.parse(fs.readFileSync(0, "utf8"));
const vectors = JSON.parse(fs.readFileSync(vectorsPath, "utf8"));

function load(path) {
  const window = { __VIDEOWIPE_I18N__: catalogue, location: { search: "?lang=en", href: "http://x/" } };
  const document = { readyState: "complete", title: "", documentElement: {}, styleSheets: null, body: null,
                     addEventListener() {}, querySelector() { return null; } };
  const ctx = vm.createContext({ window, document, navigator: { languages: ["en"] }, URLSearchParams, console,
                                 Node: { TEXT_NODE: 3, ELEMENT_NODE: 1 } });
  vm.runInContext(fs.readFileSync(path, "utf8"), ctx);
  return window.VideoWipeI18n;
}
const oldApi = load(oldPath), newApi = load(newPath);
const messages = catalogue.locales.en.messages, keys = Object.keys(messages);
const patternKeys = keys.filter(k => /\{\d+\}/.test(k));
const samples = ["12", "3.5", "1,234", "00:12.50", "字幕", "目标 2", "底部字幕", "检测目标", "未知文字", "abc", "100%", " "];
let seed = 12345; const rnd = n => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed % n; };
const fill = (key, pick) => key.replace(/\{\d+\}/g, pick);

const inputs = new Set();
for (const k of keys) { inputs.add(k); inputs.add("  " + k + "\n"); inputs.add(k + "。"); inputs.add("（" + k + "）"); inputs.add(k + " " + k); }
for (const k of patternKeys) {
  for (const s of samples) inputs.add(fill(k, () => s));
  for (let i = 0; i < 6; i++) inputs.add(fill(k, () => samples[rnd(samples.length)]));
  for (let i = 0; i < 3; i++) inputs.add(fill(k, () => keys[rnd(keys.length)]));
  for (let i = 0; i < 2; i++) inputs.add("前缀 " + fill(k, () => samples[rnd(samples.length)]) + " 后缀");
}
for (let i = 0; i < 600; i++) {
  let t = "";
  for (let j = 0, n = 1 + rnd(4); j < n; j++) t += (rnd(3) === 0 ? samples[rnd(samples.length)] : fill(keys[rnd(keys.length)], () => samples[rnd(samples.length)])) + (rnd(2) ? " " : "");
  inputs.add(t);
}
for (let i = 0; i < 200; i++) {
  inputs.add(`当前阶段 ${i % 100}%`); inputs.add(`${i * 7} / 4,000 帧`); inputs.add(`处理约剩 ${i}–${i + 20} 秒`);
  inputs.add(`预测 ${i}/${i + 9} 帧`); inputs.add(`试擦 00:${String(i % 60).padStart(2, "0")}.00–00:${String((i + 3) % 60).padStart(2, "0")}.50`);
}
for (const v of vectors) inputs.add(v.input);
for (const odd of ["hello world", "", null, undefined, 5]) inputs.add(odd);

const bad = []; let compared = 0;
for (let pass = 1; pass <= 3; pass++) {            // pass 1 fills the caches, passes 2 and 3 are served from them
  for (const t of inputs) {
    compared++;
    const a = oldApi.translate(t), b = newApi.translate(t);
    if (a !== b) bad.push({ pass, t, old: a, new: b });
  }
}
// Each shared vector brings its own catalogue, including pathological ones (bare {0}, adjacent placeholders).
for (const v of vectors) for (let rep = 0; rep < 3; rep++) {
  compared++;
  const a = oldApi.translateWith(v.input, v.messages), b = newApi.translateWith(v.input, v.messages);
  if (a !== b || b !== v.expected) bad.push({ vector: v.name, old: a, new: b, expected: v.expected });
}
// Pathological catalogues: runaway recursion is cut by the depth limit exactly as before.
const nasty = [
  { "{0}": "<{0}>", "甲{0}": "A{0}", "{0}乙": "{0}B" },
  { "{0}{1}": "{1}{0}", "你好{0}": "Hi {0}", "{0}世界": "{0} world" },
  { "甲": "a", "乙{0}丙": "b{0}c", "乙{0}": "x{0}", "{0}丙": "{0}y", "{0}和{1}": "{1} and {0}" },
];
for (const m of nasty) for (const t of ["甲乙甲丙", "乙甲丙", "你好世界", "你好甲", "甲和乙", "甲和乙和丙", "乙乙乙丙丙丙", "你好你好你好"]) for (let rep = 0; rep < 2; rep++) {
  compared++;
  const a = oldApi.translateWith(t, m), b = newApi.translateWith(t, m);
  if (a !== b) bad.push({ nasty: JSON.stringify(m), t, old: a, new: b });
}
console.log(JSON.stringify({ compared, distinct: inputs.size, bad: bad.slice(0, 5), badCount: bad.length }));
process.exit(bad.length ? 1 : 0);
"""


def _catalogue_json() -> str:
    return json.dumps(
        {"source": i18n.SOURCE_LANGUAGE, "default": i18n.DEFAULT_LANGUAGE, "locales": i18n.all_catalogues()},
        ensure_ascii=False,
    )


def test_cached_translations_equal_uncached_over_the_catalogue_and_dynamic_strings():
    result = subprocess.run(
        ["node", "-e", _HARNESS, str(_OLD), str(_NEW), str(_VECTORS)],
        input=_catalogue_json(), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["badCount"] == 0
    assert report["compared"] > 7000          # the sample is wide, not a handful of strings
    assert report["distinct"] > 1500


_CACHE_HARNESS = r"""
const vm = require("node:vm"), fs = require("node:fs");
const window = { __VIDEOWIPE_I18N__: JSON.parse(fs.readFileSync(0, "utf8")), location: { search: "?lang=en", href: "http://x/" } };
const document = { readyState: "complete", title: "", documentElement: {}, styleSheets: null, body: null, addEventListener() {}, querySelector() { return null; } };
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), vm.createContext({ window, document, navigator: { languages: ["en"] }, URLSearchParams, console, Node: { TEXT_NODE: 3, ELEMENT_NODE: 1 } }));
const api = window.VideoWipeI18n;
const assert = require("node:assert/strict");

// Two catalogues must never share cached answers.
const a = { "检测": "Detect", "试擦 {0}": "Trial {0}" }, b = { "检测": "Scan", "试擦 {0}": "Wipe {0}" };
assert.equal(api.translateWith("试擦 3", a), "Trial 3");
assert.equal(api.translateWith("试擦 3", b), "Wipe 3");
assert.equal(api.translateWith("前缀检测后缀", a), "前缀Detect后缀");   // fragment pass
assert.equal(api.translateWith("前缀检测后缀", b), "前缀Scan后缀");
assert.equal(api.translateWith("试擦 3", a), "Trial 3");          // still a's answer after b was used
// A translation that is not a catalogue hit is cached too, and stays the identity.
for (let i = 0; i < 3; i++) assert.equal(api.translateWith("完全未知的文字", a), "完全未知的文字");
// The result cache is bounded: thousands of distinct strings do not grow it without limit and stay correct.
for (let i = 0; i < 6000; i++) assert.equal(api.translateWith("试擦 " + i, a), "Trial " + i);
assert.equal(api.translateWith("试擦 17", a), "Trial 17");
console.log("ok");
"""


def test_catalogues_do_not_share_caches_and_many_strings_stay_correct():
    result = subprocess.run(
        ["node", "-e", _CACHE_HARNESS, str(_NEW)],
        input=_catalogue_json(), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "ok"


def test_the_unmemoised_reference_is_the_original_file():
    """Guards the fixture: it is the pre-cache translator, so it must not itself contain a cache."""
    text = _OLD.read_text(encoding="utf-8")
    assert "compiledFor" not in text and "MEMO_LIMIT" not in text
    assert "function translateCore(text, messages, depth)" in text
