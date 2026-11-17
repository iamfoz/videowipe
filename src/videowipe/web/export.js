/* VideoWipe output-format choice (fork addition, injected by the server).
 *
 * Adds an "output format" select next to the full-run controls and attaches
 * the choice to POST /jobs/<id>/confirm as `output_codec`. H.264 (MP4) is the
 * default; H.265 (MKV) additionally writes a 10-bit HEVC + Opus export that
 * the download button serves. In-browser previews always use the H.264 copy.
 * Kept out of index.html so upstream changes merge cleanly.
 */
(function () {
  "use strict";

  var KEY = "videowipe.outputCodec";
  var CODECS = { h264: "H.264 (MP4)", h265: "H.265 (MKV)" };

  function stored() {
    try {
      var value = window.localStorage.getItem(KEY);
      return CODECS[value] ? value : "h264";
    } catch (e) { return "h264"; }
  }
  function store(value) {
    try { window.localStorage.setItem(KEY, value); } catch (e) { /* ignore */ }
  }

  function inject() {
    var host = document.getElementById("extra");
    if (!host || document.getElementById("outputCodec")) return;
    var label = document.createElement("label");
    label.htmlFor = "outputCodec";
    label.style.cssText = "display:flex;align-items:center;gap:7px;font-size:11px;color:var(--muted);";
    label.appendChild(document.createTextNode("输出格式"));
    var select = document.createElement("select");
    select.id = "outputCodec";
    select.style.cssText = "background:var(--paper);color:var(--ink);border:1px solid var(--line);padding:4px 6px;font-size:11px;";
    for (var code in CODECS) {
      var option = document.createElement("option");
      option.value = code;
      option.textContent = CODECS[code];
      select.appendChild(option);
    }
    select.value = stored();
    select.addEventListener("change", function () { store(select.value); });
    host.appendChild(label);
    label.appendChild(select);
  }

  // Attach the choice to full-run requests without touching the page's code.
  var originalFetch = window.fetch;
  if (typeof originalFetch === "function") {
    window.fetch = function (input, init) {
      try {
        var url = typeof input === "string" ? input : (input && input.url) || "";
        if (/\/jobs\/[^/?#]+\/confirm(?:[?#]|$)/.test(url) && init && typeof init.body === "string") {
          var body = JSON.parse(init.body);
          body.output_codec = stored();
          init = Object.assign({}, init, { body: JSON.stringify(body) });
        }
      } catch (e) { /* send the request unchanged */ }
      return originalFetch.call(this, input, init);
    };
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", inject);
  else inject();
})();
