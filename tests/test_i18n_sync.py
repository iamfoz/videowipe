"""Tests for scripts/i18n_sync.py.

The extraction logic is exercised on small in-memory strings and on fixture
roots built in a pytest tmp dir, so the tests never depend on the real
index.html or the real locale catalogues.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    path = REPO_ROOT / "scripts" / "i18n_sync.py"
    spec = importlib.util.spec_from_file_location("i18n_sync_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def _ns(root, **kw):
    base = {
        "root": str(root),
        "json": False,
        "locale": "en",
        "model": mod.DEFAULT_MODEL,
        "batch": 10,
        "dry_run": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


# ── JS literal scanning ──────────────────────────────────────────────────────


def test_js_simple_literals_with_escapes():
    code = 'var a = "line\\ntab\\u4e2d\\x61"; var b = \'单引号\';'
    values = [v for _, v in mod.js_literals(code)]
    assert values == ["line\ntab中a", "单引号"]


def test_js_skips_line_and_block_comments():
    code = "// 注释中文\n/* 块中文 */\nvar x = '保留中文';"
    values = [v for _, v in mod.js_literals(code)]
    assert values == ["保留中文"]
    assert all("注释" not in v and "块" not in v for v in values)


def test_js_template_literals_become_placeholders():
    code = "const s = `前 ${name} 后 ${a+b} 尾`;"
    values = [v for _, v in mod.js_literals(code)]
    assert values == ["前 {0} 后 {1} 尾"]


def test_js_template_nested_interpolation():
    # Nested template and string inside ${...} must not confuse the scanner.
    code = "const s = `a ${'inner ${x}'} b`;"
    values = [v for _, v in mod.js_literals(code)]
    # The outer literal has one interpolation -> {0}; the string inside the
    # interpolation is shown text too, so it is emitted after the outer one.
    assert values == ["a {0} b", "inner ${x}"]


def test_js_line_numbers():
    code = "\n\nvar x = '中文';"
    lines = [l for l, _ in mod.js_literals(code)]
    assert lines == [3]  # 1-based within the snippet


def test_js_unterminated_string_is_skipped():
    code = "var a = 'broken\nvar b = '好';"
    values = [v for _, v in mod.js_literals(code)]
    assert "好" in values
    assert "broken" not in values


# ── Python AST extraction ────────────────────────────────────────────────────


def test_python_strings_and_fstrings_skip_docstrings():
    src = (
        'def f():\n'
        '    """docstring 中文"""\n'
        '    a = "常量中文"\n'
        '    b = f"前{1 + 1}后{2 * 2}"\n'
        '    return b\n'
    )
    got = list(mod._python_strings(src))
    # line 1 is `def f():`, line 2 the docstring, so strings start at line 3
    assert got == [(3, "常量中文"), (4, "前{0}后{1}")]


def test_python_module_docstring_skipped():
    src = '"""模块 中文 docstring"""\nX = "字符串中文"\n'
    got = list(mod._python_strings(src))
    assert got == [(2, "字符串中文")]


def test_python_fstring_docstring_skipped():
    src = "def f():\n    f'f-string doc 中文 {1}'\n    x = '真 中文'\n"
    got = list(mod._python_strings(src))
    assert got == [(3, "真 中文")]


def test_python_multiple_fstring_parts_share_counter():
    src = 'M = f"{a}{literal}中文{b}"\n'
    got = list(mod._python_strings(src))
    assert got == [(1, "{0}{1}中文{2}")]


def test_python_skips_re_compile_pattern():
    # A CJK range written with literal characters in re.compile(...) is a
    # regex pattern, not UI text, and must not leak into the catalogue.
    src = (
        "import re\n"
        "CJK = re.compile('[　-〿㐀-鿿＀-￯]')\n"
        "UI = '真实界面中文'\n"
    )
    got = [v for _, v in mod._python_strings(src)]
    assert "真实界面中文" in got
    assert not any("〿" in v for v in got)


# ── HTML text / attributes ───────────────────────────────────────────────────


def test_html_text_and_attributes():
    html = (
        '<html><head><title>标题中文</title></head>'
        "<body><button title='按钮说明' placeholder='占位中文'>正文中文</button>"
        '<img alt="图片说明"><meta name="x" content="内容中文">'
        "</body></html>"
    )
    values = {v for _, v in mod._html_strings(html)}
    assert {"标题中文", "按钮说明", "占位中文", "正文中文", "图片说明", "内容中文"} <= values


def test_html_ignores_css_and_script_as_css():
    html = "<style>.a { color: 中文; }</style><p>可见中文</p>"
    values = [v for _, v in mod._html_strings(html)]
    assert "可见中文" in values
    assert "color: 中文" not in values


def test_html_scans_script_blocks_as_js():
    html = "<script>var m = '脚本中文';</script><p>正文中文</p>"
    values = [v for _, v in mod._html_strings(html)]
    assert "脚本中文" in values
    assert "正文中文" in values


# ── extraction driver + CJK filtering ───────────────────────────────────────


def test_extract_all_filters_non_cjk_and_honours_ignore(tmp_path):
    web = tmp_path / "src" / "videowipe" / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_text(
        "<html><body><p>保留中文</p><span>English only</span></body></html>",
        encoding="utf-8",
    )
    pkg = tmp_path / "src" / "videowipe"
    (pkg / "kept.py").write_text('A = "保留的中文"\n', encoding="utf-8")
    (pkg / "ignored.py").write_text('B = "被忽略的中文"\n', encoding="utf-8")
    loc = tmp_path / "src" / "videowipe" / "locales"
    loc.mkdir()
    (loc / "ignore.json").write_text(
        json.dumps({"paths": ["src/videowipe/ignored.py"], "strings": ["保留的中文"]}),
        encoding="utf-8",
    )
    got = mod.extract_all(tmp_path)
    assert "保留中文" in got
    assert "English only" not in got
    assert "被忽略的中文" not in got  # file on ignore paths
    assert "保留的中文" not in got  # on the string ignore list
    # first location is recorded
    assert got["保留中文"][0].endswith("index.html")


# ── update semantics ─────────────────────────────────────────────────────────


def _make_root(tmp_path, html, locale=None, py=None):
    web = tmp_path / "src" / "videowipe" / "web"
    web.mkdir(parents=True, exist_ok=True)
    (web / "index.html").write_text(html, encoding="utf-8")
    if py is not None:
        (tmp_path / "src" / "videowipe" / "mod.py").write_text(py, encoding="utf-8")
    loc = tmp_path / "src" / "videowipe" / "locales"
    loc.mkdir(exist_ok=True)
    path = loc / "en.json"
    if locale is not None:
        path.write_text(
            json.dumps(locale, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return path


def test_update_adds_new_obsoletes_gone_and_never_overwrites(tmp_path):
    path = _make_root(
        tmp_path,
        "<html><body><p>仍存在的串</p></body></html>",
        locale={
            "_meta": {"language": "en", "name": "English", "source_language": "zh-CN"},
            "messages": {
                "仍存在的串": "Still here",
                "已删除的串": "Removed long ago",
            },
            "obsolete": {},
        },
    )
    rc = mod.run_update(_ns(tmp_path))
    assert rc == 0
    doc = json.loads(path.read_text(encoding="utf-8"))
    # kept the existing translation untouched (never overwritten)
    assert doc["messages"]["仍存在的串"] == "Still here"
    # the string no longer found upstream moved to obsolete, value preserved
    assert doc["obsolete"]["已删除的串"] == "Removed long ago"
    assert "已删除的串" not in doc["messages"]


def test_update_restores_from_obsolete_when_string_reappears(tmp_path):
    path = _make_root(
        tmp_path,
        "<html><body><p>复活的串</p></body></html>",
        locale={
            "_meta": {"language": "en", "name": "English", "source_language": "zh-CN"},
            "messages": {},
            "obsolete": {"复活的串": "Revived"},
        },
    )
    mod.run_update(_ns(tmp_path))
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["messages"]["复活的串"] == "Revived"
    assert "复活的串" not in doc["obsolete"]


def test_update_is_idempotent_when_nothing_changed(tmp_path):
    path = _make_root(
        tmp_path,
        "<html><body><p>稳定的串</p></body></html>",
        locale={
            "_meta": {"language": "en", "name": "English", "source_language": "zh-CN"},
            "messages": {"稳定的串": "Stable"},
            "obsolete": {},
        },
    )
    before = path.read_text(encoding="utf-8")
    mod.run_update(_ns(tmp_path))
    assert path.read_text(encoding="utf-8") == before


# ── check exit codes + placeholder validation ───────────────────────────────


def test_check_fails_on_untranslated(tmp_path, capsys):
    _make_root(
        tmp_path,
        "<html><body><p>未翻译串</p></body></html>",
        locale={
            "_meta": {"language": "en", "name": "English", "source_language": "zh-CN"},
            "messages": {"未翻译串": ""},
            "obsolete": {},
        },
    )
    rc = mod.run_check(_ns(tmp_path))
    assert rc == 1
    assert "untranslated" in capsys.readouterr().out


def test_check_passes_when_fully_translated(tmp_path, capsys):
    _make_root(
        tmp_path,
        "<html><body><p>已翻译串</p></body></html>",
        locale={
            "_meta": {"language": "en", "name": "English", "source_language": "zh-CN"},
            "messages": {"已翻译串": "Translated"},
            "obsolete": {},
        },
    )
    rc = mod.run_check(_ns(tmp_path))
    assert rc == 0
    assert "i18n check passed" in capsys.readouterr().out


def test_check_flags_placeholder_mismatch(tmp_path, capsys):
    _make_root(
        tmp_path,
        "<html><body><p>甲 {0} 乙</p></body></html>",
        locale={
            "_meta": {"language": "en", "name": "English", "source_language": "zh-CN"},
            "messages": {"甲 {0} 乙": "A {1} B"},
            "obsolete": {},
        },
    )
    rc = mod.run_check(_ns(tmp_path))
    assert rc == 1
    assert "placeholder mismatch" in capsys.readouterr().out


def test_check_ignores_placeholder_drift_on_empty_value(tmp_path, capsys):
    # An untranslated value (empty) must not be flagged for placeholder drift.
    _make_root(
        tmp_path,
        "<html><body><p>甲 {0} 乙</p></body></html>",
        locale={
            "_meta": {"language": "en", "name": "English", "source_language": "zh-CN"},
            "messages": {"甲 {0} 乙": ""},
            "obsolete": {},
        },
    )
    rc = mod.run_check(_ns(tmp_path))
    out = capsys.readouterr().out
    assert rc == 1  # still fails: untranslated
    assert "placeholder mismatch" not in out


# ── new-locale ───────────────────────────────────────────────────────────────


def test_new_locale_creates_and_populates(tmp_path):
    web = tmp_path / "src" / "videowipe" / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_text("<html><body><p>新语言串</p></body></html>", encoding="utf-8")
    rc = mod.run_new_locale(_ns(tmp_path, code="fr", name="Français"))
    assert rc == 0
    path = tmp_path / "src" / "videowipe" / "locales" / "fr.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["_meta"]["language"] == "fr"
    assert doc["messages"]["新语言串"] == ""


def test_new_locale_rejects_existing(tmp_path):
    _make_root(
        tmp_path,
        "<html><body><p>x 串</p></body></html>",
        locale={"_meta": {"language": "en", "name": "English", "source_language": "zh-CN"},
                "messages": {}, "obsolete": {}},
    )
    rc = mod.run_new_locale(_ns(tmp_path, code="en", name="English"))
    assert rc == 2


# ── translate (mocked HTTP, no network) ─────────────────────────────────────


def _translate_fixture(tmp_path):
    # 甲 already translated; 乙 {0} and 丙 pending
    path = _make_root(
        tmp_path,
        "<html><body><p>甲</p><p>乙 {0}</p><p>丙</p></body></html>",
        locale={
            "_meta": {"language": "en", "name": "English", "source_language": "zh-CN"},
            "messages": {"甲": "Alpha", "乙 {0}": "", "丙": ""},
            "obsolete": {},
        },
    )
    return path


def test_translate_fills_blanks_and_keeps_existing(tmp_path, monkeypatch, capsys):
    path = _translate_fixture(tmp_path)

    def fake_call(base, headers, payload):
        # The prompt asks for the pending batch; return translations for both.
        return json.dumps({"乙 {0}": "Beta {0}", "丙": "bad placeholder {9}"})

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(mod, "_api_call", fake_call)
    monkeypatch.setattr(mod, "_api_headers", lambda: {"x-api-key": "test-key"})

    rc = mod.run_translate(_ns(tmp_path, locale="en", batch=10))
    doc = json.loads(path.read_text(encoding="utf-8"))
    # existing translation untouched (never overwritten)
    assert doc["messages"]["甲"] == "Alpha"
    # valid placeholder-preserving translation applied
    assert doc["messages"]["乙 {0}"] == "Beta {0}"
    # placeholder-mismatch entry rejected, left empty, and reported
    assert doc["messages"]["丙"] == ""
    assert rc == 1
    assert "rejected" in capsys.readouterr().err


def test_translate_dry_run_writes_nothing(tmp_path, monkeypatch):
    path = _translate_fixture(tmp_path)
    before = path.read_text(encoding="utf-8")

    def fake_call(base, headers, payload):
        raise AssertionError("dry run must not call the API")

    monkeypatch.setattr(mod, "_api_call", fake_call)
    rc = mod.run_translate(_ns(tmp_path, locale="en", batch=10, dry_run=True))
    assert rc == 0
    assert path.read_text(encoding="utf-8") == before


def test_translate_no_network_on_missing_credentials(tmp_path, monkeypatch):
    _translate_fixture(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    def fake_call(base, headers, payload):
        raise AssertionError("must fail fast before the network")

    monkeypatch.setattr(mod, "_api_call", fake_call)
    with pytest.raises(RuntimeError):
        mod.run_translate(_ns(tmp_path, locale="en", batch=10))


def test_translate_handles_http_error_gracefully(tmp_path, monkeypatch, capsys):
    _translate_fixture(tmp_path)

    def fake_call(base, headers, payload):
        raise RuntimeError("HTTP 500 from API: boom")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(mod, "_api_headers", lambda: {"x-api-key": "test-key"})
    monkeypatch.setattr(mod, "_api_call", fake_call)
    rc = mod.run_translate(_ns(tmp_path, locale="en", batch=10))
    # batch failed, nothing translated, but no exception escaped
    assert rc == 1
    assert "failed" in capsys.readouterr().err


def test_api_call_posts_expected_shape(tmp_path, monkeypatch):
    """Verify the request path, headers and body shape without any network."""
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {"content": [{"type": "text", "text": "{\"甲\": \"A\"}"}]}
            ).encode("utf-8")

    import urllib.request

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    result = mod._api_call(
        "https://api.anthropic.com",
        {"x-api-key": "k", "anthropic-version": "2023-06-01",
         "content-type": "application/json"},
        {"model": "m", "max_tokens": 10, "messages": []},
    )
    assert result == '{"甲": "A"}'
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    # urllib stores header names in their title-cased form
    assert captured["headers"].get("X-api-key") == "k"
    assert captured["headers"].get("Anthropic-version") == "2023-06-01"
    assert captured["body"]["model"] == "m"
    assert captured["body"]["messages"] == []


def test_parse_model_json_strips_fences():
    assert mod._parse_model_json('```json\n{"a":"b"}\n```') == {"a": "b"}
    assert mod._parse_model_json('noise {"a":"b"} noise') == {"a": "b"}
    with pytest.raises(ValueError):
        mod._parse_model_json("no json here")


def test_js_literals_inside_template_interpolations_are_extracted():
    code = 'x=`${a?"调整":"查看"}${b}区域 ${c?" 帧":""}`;'
    values = [v for _, v in mod.js_literals(code)]
    assert values[0] == "{0}{1}区域 {2}"
    assert {"调整", "查看", " 帧"} <= set(values)


def test_update_preserves_hand_added_extra_fragments(tmp_path):
    path = _make_root(
        tmp_path,
        "<html><body><p>仍存在的串</p></body></html>",
        locale={
            "_meta": {"language": "en", "name": "English", "source_language": "zh-CN"},
            "messages": {"仍存在的串": "Still here"},
            "obsolete": {},
            "extra": {"区域": "area"},
        },
    )
    assert mod.run_update(_ns(tmp_path)) == 0
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["extra"] == {"区域": "area"}
    assert "区域" not in doc["messages"] and "区域" not in doc["obsolete"]
    # A second run is a no-op.
    before = path.read_text(encoding="utf-8")
    mod.run_update(_ns(tmp_path))
    assert path.read_text(encoding="utf-8") == before


def test_css_content_strings_are_extracted():
    html = (
        "<html><head><style>\n.a::after { content:\"正在定位画面…\"; }\n"
        ".b::before { content: '' }\n.c { content:'左\\'右'; }</style></head><body></body></html>"
    )
    values = [v for _, v in mod._html_strings(html)]
    assert "正在定位画面…" in values
    assert "左'右" in values


def test_update_adopts_pre_translated_extra_when_string_appears(tmp_path):
    path = _make_root(
        tmp_path,
        "<html><body><p>仍存在的串</p><p>新出现的串</p></body></html>",
        locale={
            "_meta": {"language": "en", "name": "English", "source_language": "zh-CN"},
            "messages": {"仍存在的串": "Still here"},
            "obsolete": {},
            "extra": {"新出现的串": "Newly arrived", "区域": "area"},
        },
    )
    assert mod.run_update(_ns(tmp_path)) == 0
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["messages"]["新出现的串"] == "Newly arrived"
    assert doc["extra"] == {"区域": "area"}
    assert mod.run_check(_ns(tmp_path)) == 0


def test_check_counts_extra_translations_as_covered(tmp_path):
    _make_root(
        tmp_path,
        "<html><body><p>新出现的串</p></body></html>",
        locale={
            "_meta": {"language": "en", "name": "English", "source_language": "zh-CN"},
            "messages": {"新出现的串": ""},
            "obsolete": {},
            "extra": {"新出现的串": "Newly arrived"},
        },
    )
    assert mod.run_check(_ns(tmp_path)) == 0
