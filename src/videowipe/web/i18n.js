/* VideoWipe runtime translation (presentation-time i18n).
 *
 * Upstream source keeps its zh-CN text; this script translates it in the
 * browser from the catalogue in window.__VIDEOWIPE_I18N__ (injected by the
 * server right after <head>). The translate() algorithm is byte-for-byte the
 * same as videowipe/i18n.py and must pass tests/fixtures/i18n_vectors.json.
 *
 * Vanilla JS, no dependencies. Exposes window.VideoWipeI18n = {
 *   lang, translate, setLanguage, locales
 * }. Degrades to a no-op (the page keeps showing zh-CN) if the catalogue is
 * absent, and never throws on its own errors.
 */
(function () {
  "use strict";

  var STORE_KEY = "videowipe.lang";
  var SOURCE = "zh-CN";
  var DEFAULT = "en";
  var MAX_DEPTH = 8;

  var config = (typeof window !== "undefined" && window.__VIDEOWIPE_I18N__) || null;
  var source = (config && config.source) || SOURCE;
  var def = (config && config.default) || DEFAULT;
  // code -> {name, messages}. `messages` holds only non-empty entries.
  var catalogues = (config && config.locales) || {};
  var available = Object.keys(catalogues).sort();

  // U+3000-U+303F | U+3400-U+9FFF | U+FF00-U+FFEF
  var CJK = new RegExp("[\\u3000-\\u303f\\u3400-\\u9fff\\uff00-\\uffef]");
  var PLACEHOLDER = /\{(\d+)\}/;
  var PLACEHOLDER_SPLIT = /(\{\d+\})/;

  function hasCJK(text) {
    return CJK.test(text);
  }
  function isPatternKey(key) {
    return PLACEHOLDER.test(key);
  }
  function literalLength(key) {
    var parts = key.split(PLACEHOLDER_SPLIT);
    var total = 0;
    for (var i = 0; i < parts.length; i++) if (parts[i].charAt(0) !== "{") total += parts[i].length;
    return total;
  }
  function escapeRegex(s) {
    // Mirror Python's re.escape() for the strings we actually embed.
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }
  function buildPattern(key) {
    // Returns { regex, numberToGroup } where numberToGroup maps a {n} number
    // to its capture-group ordinal (1-based, in order of appearance).
    var parts = key.split(PLACEHOLDER_SPLIT);
    var pattern = "";
    var numberToGroup = {};
    var ordinal = 0;
    for (var i = 0; i < parts.length; i++) {
      var piece = parts[i];
      if (piece.charAt(0) === "{") {
        ordinal += 1;
        numberToGroup[parseInt(piece.slice(1, -1), 10)] = ordinal;
        pattern += "(.+?)";
      } else {
        pattern += escapeRegex(piece);
      }
    }
    return { regex: new RegExp("^" + pattern + "$"), numberToGroup: numberToGroup };
  }

  function applyValue(value, match, numberToGroup, messages, depth) {
    var parts = value.split(PLACEHOLDER_SPLIT);
    var out = "";
    for (var i = 0; i < parts.length; i++) {
      var piece = parts[i];
      if (piece.charAt(0) === "{") {
        var ordinal = numberToGroup[parseInt(piece.slice(1, -1), 10)];
        out += (ordinal ? translateCore(match[ordinal], messages, depth + 1) : piece);
      } else {
        out += piece;
      }
    }
    return out;
  }

  function translateCore(text, messages, depth) {
    depth = depth || 0;
    if (!hasCJK(text)) return text;

    // 3. Exact match.
    if (Object.prototype.hasOwnProperty.call(messages, text) && messages[text]) return messages[text];

    // 4. Pattern match, descending literal length.
    if (depth < MAX_DEPTH) {
      var patterns = [];
      for (var key in messages) {
        if (!messages[key] || !isPatternKey(key)) continue;
        patterns.push({ literal: literalLength(key), key: key, value: messages[key] });
      }
      patterns.sort(function (a, b) { return b.literal - a.literal; });
      for (var i = 0; i < patterns.length; i++) {
        var built = buildPattern(patterns[i].key);
        var match = built.regex.exec(text);
        if (match) return applyValue(patterns[i].value, match, built.numberToGroup, messages, depth);
      }
    }

    // 5. Fragment fallback: non-pattern keys, longest first, left to right,
    //    without re-replacing produced output.
    var fragments = [];
    for (var fkey in messages) {
      if (!messages[fkey] || isPatternKey(fkey)) continue;
      if (text.indexOf(fkey) !== -1) fragments.push({ len: fkey.length, key: fkey, value: messages[fkey] });
    }
    if (!fragments.length) return text;

    var out = "";
    var start = 0;
    while (start < text.length) {
      var best = null;
      for (var j = 0; j < fragments.length; j++) {
        var at = text.indexOf(fragments[j].key, start);
        if (at === -1) continue;
        if (!best || at < best.at || (at === best.at && fragments[j].len > best.len)) {
          best = { at: at, len: fragments[j].len, key: fragments[j].key, value: fragments[j].value };
        }
      }
      if (!best) break;
      out += translateCore(text.slice(start, best.at), messages, depth);
      out += best.value;
      start = best.at + best.key.length;
    }
    out += translateCore(text.slice(start), messages, depth);
    return out;
  }

  function messagesFor(code) {
    var cat = catalogues[code];
    return (cat && cat.messages) || {};
  }

  // translateWith: preserve surrounding whitespace and translate the trimmed
  // core against the given messages. This is the function the shared vectors
  // exercise, mirroring translate_with in videowipe/i18n.py.
  function translateWith(text, messages) {
    if (text == null || text === "") return text;
    if (typeof text !== "string") return text;
    if (!hasCJK(text)) return text;
    var m = /^\s*([\s\S]*?)\s*$/.exec(text);
    if (!m) return text;
    var core = m[1];
    if (!core) return text;
    var idx = text.indexOf(core);
    return text.slice(0, idx) + translateCore(core, messages, 0) + text.slice(idx + core.length);
  }
  // Public translate(text): resolve the active language and translate.
  function translate(text) {
    if (activeLang === source) return text;
    return translateWith(text, messagesFor(activeLang));
  }

  // ---- Language selection --------------------------------------------------
  function storeGet(key) {
    try { return window.localStorage.getItem(key); } catch (e) { return null; }
  }
  function storeSet(key, value) {
    try { window.localStorage.setItem(key, value); } catch (e) { /* ignore */ }
  }
  function matchLang(raw) {
    if (!raw) return null;
    var lower = String(raw).toLowerCase();
    if (lower.indexOf("zh") === 0) return source;
    for (var i = 0; i < available.length; i++) {
      if (lower.indexOf(available[i].toLowerCase()) === 0) return available[i];
    }
    return null;
  }
  function pickLanguage() {
    try {
      var params = new URLSearchParams(window.location.search);
      var q = params.get("lang");
      if (q) { var c = matchLang(q); if (c) return c; }
    } catch (e) { /* ignore */ }
    var stored = storeGet(STORE_KEY);
    if (stored) { var c2 = matchLang(stored); if (c2) return c2; }
    if (typeof navigator !== "undefined" && Array.isArray(navigator.languages)) {
      for (var i = 0; i < navigator.languages.length; i++) {
        var c3 = matchLang(navigator.languages[i]);
        if (c3) return c3;
      }
    }
    return available.indexOf(def) !== -1 ? def : (available[0] || source);
  }

  var activeLang = pickLanguage();
  // Remove the ?lang= query param from an absolute URL, leaving other params,
  // a trailing slash, and the fragment untouched. Pure, so it is unit-tested.
  function stripLangParam(href) {
    var hashIndex = href.indexOf("#");
    var base = hashIndex === -1 ? href : href.slice(0, hashIndex);
    var hash = hashIndex === -1 ? "" : href.slice(hashIndex);
    base = base.replace(/([?&])lang=[^&#]*/i, "");
    if (base.charAt(base.length - 1) === "?") base = base.slice(0, -1);
    if (base.charAt(base.length - 1) === "&") base = base.slice(0, -1);
    // A removed leading param can leave a dangling "&"; turn it into "?".
    if (base.indexOf("?") === -1) {
      var amp = base.indexOf("&");
      if (amp !== -1) base = base.slice(0, amp) + "?" + base.slice(amp + 1);
    }
    return base + hash;
  }

  function setLanguage(code) {
    if (!code) return activeLang;
    var resolved = matchLang(code);
    if (!resolved) resolved = code; // keep even an unknown code; it just maps to source text
    storeSet(STORE_KEY, resolved);
    activeLang = resolved;
    applyDocLang();
    // A change of catalogue requires the injected __VIDEOWIPE_I18N__ data for
    // the new language; the simplest correct behaviour is to reload without
    // the ?lang= param so the stored value (and the fresh page) win.
    try {
      window.location.replace(stripLangParam(window.location.href));
    } catch (e) { /* ignore */ }
    return activeLang;
  }
  function applyDocLang() {
    try {
      document.documentElement.lang = activeLang === source ? source : activeLang;
    } catch (e) { /* ignore */ }
  }
  applyDocLang();

  // ---- DOM translation -----------------------------------------------------
  var ATTRIBUTE_TARGETS = ["title", "placeholder", "aria-label", "alt"];
  var wrote = new WeakMap(); // node/element -> {attrName -> lastValue}

  // Skip whole subtrees that hold code, styles, or user-typed content.
  function shouldSkipElement(el) {
    if (!el || el.nodeType !== 1) return false;
    var tag = el.nodeName;
    if (tag === "SCRIPT" || tag === "STYLE" || tag === "TEXTAREA") return true;
    if (el.isContentEditable) return true;
    return false;
  }
  function isButtonInput(el) {
    return el && el.tagName === "INPUT" && (el.type === "button" || el.type === "submit");
  }
  function attrRecord(node) {
    var rec = wrote.get(node);
    if (!rec) { rec = {}; wrote.set(node, rec); }
    return rec;
  }
  function translateTextNode(node) {
    var value = node.nodeValue;
    if (!value || !hasCJK(value)) return;
    var translated = translate(value);
    if (translated === value) return;
    var rec = attrRecord(node); // remember per node to keep the observer quiet
    if (rec.text === translated) return;
    rec.text = translated;
    node.nodeValue = translated;
  }
  function translateElement(el) {
    for (var i = 0; i < ATTRIBUTE_TARGETS.length; i++) {
      var attr = ATTRIBUTE_TARGETS[i];
      if (el.hasAttribute && el.hasAttribute(attr)) {
        var current = el.getAttribute(attr);
        if (current && hasCJK(current)) {
          var value = translate(current);
          var rec = attrRecord(el);
          if (rec[attr] !== value) {
            if (value !== current) el.setAttribute(attr, value);
            rec[attr] = value;
          }
        }
      }
    }
    if (isButtonInput(el)) {
      var btnValue = el.value;
      if (btnValue && hasCJK(btnValue)) {
        var tv = translate(btnValue);
        var brec = attrRecord(el);
        if (brec.value !== tv) {
          if (tv !== btnValue) el.value = tv;
          brec.value = tv;
        }
      }
    }
  }
  // Recursively translate text and target attributes under *node*. A plain
  // childNodes walk (no TreeWalker) so nested text is never skipped, and
  // script/style/textarea/contenteditable subtrees are pruned.
  function walkNode(node) {
    var children = node.childNodes;
    if (children) {
      for (var i = 0; i < children.length; i++) {
        var child = children[i];
        if (child.nodeType === Node.TEXT_NODE) {
          translateTextNode(child);
        } else if (child.nodeType === Node.ELEMENT_NODE) {
          if (shouldSkipElement(child)) continue;
          translateElement(child);
          walkNode(child);
        }
      }
    }
  }
  function walk(root) {
    walkNode(root);
  }

  function scan() {
    if (!document.body) return;
    walk(document.body);
  }

  function observe() {
    if (typeof MutationObserver === "undefined") return;
    var observer = new MutationObserver(function (mutations) {
      for (var i = 0; i < mutations.length; i++) {
        var mutation = mutations[i];
        if (mutation.type === "childList") {
          for (var j = 0; j < mutation.addedNodes.length; j++) {
            var added = mutation.addedNodes[j];
            if (added.nodeType === Node.TEXT_NODE) translateTextNode(added);
            else if (added.nodeType === Node.ELEMENT_NODE) walk(added);
          }
        } else if (mutation.type === "characterData") {
          translateTextNode(mutation.target);
        } else if (mutation.type === "attributes") {
          var target = mutation.target;
          if (target.nodeType === 1) translateElement(target);
        }
      }
    });
    observer.observe(document.body, {
      childList: true, characterData: true, attributes: true, subtree: true,
      attributeFilter: ATTRIBUTE_TARGETS.concat(["value"])
    });
  }

  // Wrap alert/confirm/prompt so their message text is translated.
  function wrapDialogs() {
    if (!window) return;
    ["alert", "confirm", "prompt"].forEach(function (name) {
      var original = window[name];
      if (typeof original !== "function" || original.__i18nWrapped) return;
      var wrapped = function (message) {
        try { message = translate(message); } catch (e) { /* keep original */ }
        var args = [message];
        for (var i = 1; i < arguments.length; i++) args.push(arguments[i]);
        return original.apply(this, args);
      };
      wrapped.__i18nWrapped = true;
      try { window[name] = wrapped; } catch (e) { /* ignore */ }
    });
  }

  // Inject a small accessible language <select> into the top bar (falling back
  // to a fixed corner control), styled with the page's CSS variables.
  var SELECT_ID = "videowipe-lang";
  function injectSelect() {
    if (!document.body) return;
    if (document.getElementById(SELECT_ID)) return;
    var select = document.createElement("select");
    select.id = SELECT_ID;
    select.className = "videowipe-lang";
    select.setAttribute("aria-label", "Language");
    var optionFor = function (code, label, value) {
      var option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      return option;
    };
    // Each available locale by its own name; the source language ("中文") first.
    var sourceOption = optionFor(source, "中文", source);
    select.appendChild(sourceOption);
    for (var i = 0; i < available.length; i++) {
      var code = available[i];
      if (code === source) continue;
      select.appendChild(optionFor(code, (catalogues[code] && catalogues[code].name) || code, code));
    }
    // Inline styling using the page's CSS variables.
    select.style.cssText =
      "font:inherit;color:var(--ink);background:var(--paper);border:1px solid var(--line);" +
      "border-radius:6px;padding:4px 6px;min-height:30px;";
    select.addEventListener("change", function () { setLanguage(select.value); });
    var target = document.querySelector(".topbar");
    if (target) {
      var spacer = target.querySelector(".spacer");
      if (spacer) spacer.after(select); else target.appendChild(select);
    } else {
      select.style.position = "fixed";
      select.style.top = "8px";
      select.style.right = "8px";
      select.style.zIndex = "1000";
      document.body.appendChild(select);
    }
    // Reflect the active language in the control.
    var shown = activeLang === source ? source : activeLang;
    for (var k = 0; k < select.options.length; k++) {
      if (select.options[k].value === shown) { select.value = shown; break; }
    }
    if (!select.value && select.options.length) select.value = select.options[0].value;
  }

  function start() {
    try {
      wrapDialogs();
      scan();
      injectSelect();
      observe();
    } catch (e) { /* never break the page */ }
  }

  // Runs in <head>: if the DOM is not ready yet, wait for it.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }

  window.VideoWipeI18n = {
    get lang() { return activeLang; },
    translate: translate,
    translateWith: translateWith,
    setLanguage: setLanguage,
    stripLangParam: stripLangParam,
    locales: available.slice()
  };
})();
