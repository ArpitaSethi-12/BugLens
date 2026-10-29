/* BugLens front end. No external libraries, so it works offline. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v, d = 0) => (v == null ? "n/a" : (v * 100).toFixed(d) + "%");

const COLORS = { codebert: "#2456E8", codet5: "#E0307F" };
const NAMES = { codebert: "CodeBERT", codet5: "CodeT5-small" };

const state = {
  samples: [],
  status: null,
  sampleId: null,
  model: "codebert",
  detect: null,
  review: null,
  heat: false,
  flagged: new Set(),
  focus: new Set(),
  busy: false,
  labLoaded: false,
};

async function api(path, body) {
  const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  let data = null;
  try { data = await res.json(); } catch { /* not json */ }
  if (!res.ok) throw new Error((data && data.detail) || `Request failed (${res.status})`);
  return data;
}

/* =================================================================== tabs */
$$(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    $$(".tab").forEach((x) => x.setAttribute("aria-selected", String(x === t)));
    $$(".view").forEach((v) => (v.hidden = v.id !== "view-" + t.dataset.view));
    if (t.dataset.view === "lab" && !state.labLoaded) loadLab();
  })
);

/* =================================================================== status */
async function loadStatus() {
  try {
    state.status = await api("/api/status");
  } catch {
    $("#engines").innerHTML = `<span class="engine off"><i></i>Server not reachable</span>`;
    return;
  }
  const d = state.status.detector, r = state.status.reviewer;
  const chips = ["codebert", "codet5"].map((m) => {
    const ok = d[m].loaded;
    return `<span class="engine ${ok ? "ok" : "off"}" title="${ok ? "Fine-tuned model loaded" : esc(d[m].error || "No trained model in models/" + m)}"><i></i>${NAMES[m]} ${ok ? "ready" : "missing"}</span>`;
  });
  chips.push(`<span class="engine ${r.configured ? (r.last_error ? "warn" : "ok") : "warn"}" title="${esc(r.last_error || (r.configured ? r.model : "Add GEMINI_API_KEY to .env"))}"><i></i>Gemini ${r.configured ? "ready" : "offline"}</span>`);
  $("#engines").innerHTML = chips.join("");

  // model picker: disable models that are not loaded
  const loaded = ["codebert", "codet5"].filter((m) => d[m].loaded);
  $$("#modelPick button").forEach((b) => {
    const m = b.dataset.model;
    b.disabled = m === "both" ? loaded.length < 2 : !d[m].loaded;
  });
  if (!loaded.includes(state.model)) state.model = loaded[0] || "codebert";
  setModel(state.model);
}

function setModel(m) {
  state.model = m;
  $$("#modelPick button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.model === m)));
}
$$("#modelPick button").forEach((b) => b.addEventListener("click", () => !b.disabled && setModel(b.dataset.model)));

function detectorReady() {
  return state.status && ["codebert", "codet5"].some((m) => state.status.detector[m].loaded);
}

/* =================================================================== samples */
async function loadSamples() {
  try { state.samples = await api("/api/samples"); } catch { state.samples = []; }
  $("#language").value = "python";
  renderDrawer("all");
  // start with a sample so the page is never blank in a demo
  if (state.samples.length && !$("#code").value) pickSample(state.samples[0].id);
}

function renderDrawer(lang) {
  const list = state.samples.filter((s) => lang === "all" || s.language === lang);
  $("#drawer").innerHTML = list.map((s) => `
    <button class="specimen" data-id="${s.id}" aria-pressed="${s.id === state.sampleId}">
      <span class="sp-meta"><span class="lang-chip">${s.language === "c" ? "C" : "Py"}</span><span class="sev sev-${s.severity}">${s.severity}</span></span>
      <span class="sp-title">${esc(s.title)}</span>
    </button>`).join("");
  $$(".specimen").forEach((b) => b.addEventListener("click", () => pickSample(b.dataset.id)));
}
$$("#langFilter button").forEach((b) =>
  b.addEventListener("click", () => {
    $$("#langFilter button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    renderDrawer(b.dataset.lang);
  })
);

function pickSample(id) {
  const s = state.samples.find((x) => x.id === id);
  if (!s) return;
  state.sampleId = id;
  $$(".specimen").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === id)));
  setCode(s.buggy, s.language);
}

function setCode(code, lang) {
  $("#code").value = code;
  if (lang) $("#language").value = lang === "c" ? "c" : lang;
  resetResults();
  renderEditor();
  $("#code").scrollTop = 0;
  syncScroll();
}

/* =================================================================== editor */
const KW = {
  python: "False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield self print len range",
  c: "auto break case char const continue default do double else enum extern float for goto if inline int long register return short signed sizeof static struct switch typedef union unsigned void volatile while NULL size_t uint8_t uint16_t uint32_t int64_t bool true false include define",
  java: "abstract boolean break byte case catch char class const continue default do double else enum extends final finally float for if implements import instanceof int interface long new null package private protected public return short static super switch this throw throws try void while true false String",
  javascript: "async await break case catch class const continue default delete do else export extends false finally for function if import in instanceof let new null return super switch this throw true try typeof undefined var void while yield of",
};
const kwSets = Object.fromEntries(Object.entries(KW).map(([k, v]) => [k, new Set(v.split(" "))]));

function syntaxClasses(code, lang) {
  const cls = new Array(code.length).fill("");
  const isPy = lang === "python";
  const re = isPy
    ? /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|[rbfuRBFU]{0,2}"(?:\\.|[^"\\\n])*"|[rbfuRBFU]{0,2}'(?:\\.|[^'\\\n])*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_]\w*)|([-+*/%=<>!&|^~]+)/g
    : /(\/\/[^\n]*|\/\*[\s\S]*?\*\/|^\s*#\s*\w+)|("(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`[^`]*`)|(\b(?:0x[\da-fA-F]+|\d+(?:\.\d+)?)\b)|([A-Za-z_]\w*)|([-+*/%=<>!&|^~]+)/gm;
  const kws = kwSets[lang] || kwSets.python;
  let m;
  while ((m = re.exec(code))) {
    let c = "";
    if (m[1]) c = "t-com";
    else if (m[2]) c = "t-str";
    else if (m[3]) c = "t-num";
    else if (m[4]) {
      if (kws.has(m[4])) c = "t-kw";
      else if (/^\s*\(/.test(code.slice(re.lastIndex, re.lastIndex + 3))) c = "t-fn";
    } else if (m[5]) c = "t-op";
    if (c) for (let i = m.index; i < re.lastIndex; i++) cls[i] = c;
  }
  return cls;
}

function heatColor(s) {
  return `rgba(255, 210, 63, ${(0.12 + 0.83 * s).toFixed(3)})`;
}

function currentAttention() {
  if (!state.heat || !state.detect) return null;
  const first = Object.values(state.detect.results)[0];
  return first ? first.attention : null;
}

function renderEditor() {
  const code = $("#code").value;
  const lang = $("#language").value;
  const cls = syntaxClasses(code, lang);
  const att = new Float32Array(code.length);
  const spans = currentAttention();
  if (spans) for (const [a, b, s] of spans) for (let i = a; i < Math.min(b, code.length); i++) att[i] = Math.max(att[i], s);

  const lines = code.split("\n");
  const lineHeat = [];
  let pos = 0;
  const html = lines.map((line, li) => {
    let out = "", i = 0, maxHeat = 0;
    while (i < line.length) {
      const g = pos + i, c = cls[g], a = att[g];
      let j = i + 1;
      while (j < line.length && cls[pos + j] === c && att[pos + j] === a) j++;
      const text = esc(line.slice(i, j));
      maxHeat = Math.max(maxHeat, a);
      if (a > 0.04) {
        const dark = a > 0.4 ? "color:#1C1A3A;" : "";
        out += `<span class="att ${c}" style="background:${heatColor(a)};${dark}">${text}</span>`;
      } else out += c ? `<span class="${c}">${text}</span>` : text;
      i = j;
    }
    pos += line.length + 1;
    lineHeat.push(maxHeat);
    const n = li + 1;
    const lc = state.focus.has(n) ? "line focus" : state.flagged.has(n) ? "line flagged" : "line";
    return `<span class="${lc}">${out || " "}</span>`;
  }).join("");
  $("#highlight").innerHTML = html + `<span class="line"> </span>`;

  $("#gutter").innerHTML = lines.map((_, li) => {
    const n = li + 1, h = lineHeat[li];
    const heat = spans && h > 0.04 ? `<span class="heat" style="background:${heatColor(h)}" data-tip="Line ${n}: attention ${(h * 100).toFixed(0)}% of peak"></span>` : "";
    return `<div class="${state.flagged.has(n) ? "flag" : ""}">${heat}${n}</div>`;
  }).join("") + "<div></div>";
}

function syncScroll() {
  const ta = $("#code");
  $("#highlight").scrollTop = ta.scrollTop;
  $("#highlight").scrollLeft = ta.scrollLeft;
  $("#gutter").scrollTop = ta.scrollTop;
}

const ta = $("#code");
ta.addEventListener("input", () => {
  if (state.detect || state.review) resetResults();
  state.sampleId = null;
  $$(".specimen").forEach((b) => b.setAttribute("aria-pressed", "false"));
  renderEditor();
  syncScroll();
});
ta.addEventListener("scroll", syncScroll);
ta.addEventListener("keydown", (e) => {
  if (e.key === "Tab" && !e.shiftKey) {
    e.preventDefault();
    const { selectionStart: s, selectionEnd: en } = ta;
    ta.setRangeText("    ", s, en, "end");
    ta.dispatchEvent(new Event("input"));
  }
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); scan(); }
});
$("#language").addEventListener("change", renderEditor);
$("#heatToggle").addEventListener("click", () => {
  state.heat = !state.heat;
  $("#heatToggle").setAttribute("aria-pressed", String(state.heat));
  renderEditor();
});

function focusLines(lines) {
  state.focus = new Set(lines);
  renderEditor();
  if (lines.length) {
    const lh = parseFloat(getComputedStyle($("#editor")).lineHeight) || 24;
    ta.scrollTop = Math.max(0, (Math.min(...lines) - 3) * lh);
    syncScroll();
  }
}

/* =================================================================== pipeline rail */
function station(step, cls, value) {
  const el = $(`.station[data-step="${step}"]`);
  el.className = "station " + (cls || "");
  if (value != null) $(".st-val", el).textContent = value;
}
function resetRail() {
  ["code", "tokens", "encoder", "verdict", "review"].forEach((s) => station(s, "", "waiting"));
}

function resetResults() {
  state.detect = null;
  state.review = null;
  state.flagged = new Set();
  state.focus = new Set();
  state.heat = false;
  $("#heatToggle").setAttribute("aria-pressed", "false");
  $("#heatToggle").disabled = true;
  resetRail();
  $("#verdict").innerHTML = `<div class="empty"><p class="empty-title">Ready to scan</p><p>Press Scan code to run the two-stage pipeline on this function.</p></div>`;
  $("#review").innerHTML = "";
}

/* =================================================================== scan */
$("#scanBtn").addEventListener("click", scan);

async function scan() {
  const code = ta.value;
  if (state.busy) return;
  if (!code.trim()) {
    $("#verdict").innerHTML = `<div class="empty"><p class="empty-title">Nothing to scan</p><p>Paste a function into the editor or pick a specimen from the drawer.</p></div>`;
    return;
  }
  state.busy = true;
  $("#scanBtn").disabled = true;
  $("#scanBtn").textContent = "Scanning";
  resetResults();
  $("#verdict").innerHTML = `<div class="loading-card" style="border:0;padding:0"><span class="spinner"></span>Running the detector on your code</div>`;
  const lang = $("#language").value;
  const nLines = code.replace(/\n+$/, "").split("\n").length;

  try {
    station("code", "done", `${nLines} line${nLines > 1 ? "s" : ""} of ${$("#language").selectedOptions[0].text}`);
    await sleep(250);

    let detection = null;
    if (detectorReady()) {
      station("tokens", "active", "splitting into tokens");
      const models = state.model === "both" ? ["codebert", "codet5"] : [state.model];
      const req = api("/api/detect", { code, models });
      await sleep(450);
      const res = await req;
      state.detect = res;
      const results = Object.values(res.results);
      const first = results[0];
      station("tokens", "done", `${first.tokens} tokens${first.truncated ? ", cut to fit" : ""}`);
      station("encoder", "active", "encoding");
      await sleep(450);
      station("encoder", "done", results.map((r) => `${r.display} in ${Math.round(r.latency_ms)} ms`).join(", "));

      const prob = results.reduce((a, r) => a + r.probability, 0) / results.length;
      const label = prob >= 0.5 ? "buggy" : "clean";
      detection = {
        display: results.length > 1 ? "CodeBERT + CodeT5 ensemble" : first.display,
        probability: prob,
        label,
      };
      state.detection = detection;
      station("verdict", "done " + (label === "buggy" ? "bad" : "good"), `${pct(prob)} likely buggy`);
      renderVerdict(res, detection);
      $("#heatToggle").disabled = false;
    } else {
      ["tokens", "encoder", "verdict"].forEach((s) => station(s, "skipped", "model not loaded"));
      $("#verdict").innerHTML = `<div class="empty"><p class="empty-title">Detector not loaded</p><p>No fine-tuned model was found in the models folder, so the code goes straight to Gemini.</p></div>`;
    }

    if (!detection || detection.label === "buggy" || $("#alwaysReview").checked) {
      await runReview(code, lang, detection);
    } else {
      station("review", "skipped", "not needed");
      $("#review").innerHTML = `
        <div class="skip-card">
          <p><b>Gemini skipped.</b> The detector rated this code clean, so no review was requested.</p>
          <button class="btn" id="askAnyway">Ask Gemini anyway</button>
        </div>`;
      $("#askAnyway").addEventListener("click", () => runReview(code, lang, detection));
    }
  } catch (err) {
    station("tokens", "", "");
    $("#verdict").innerHTML = `<div class="notice err"><b>Scan failed.</b> ${esc(err.message)}</div>`;
    resetRail();
  } finally {
    state.busy = false;
    $("#scanBtn").disabled = false;
    $("#scanBtn").textContent = "Scan code";
  }
}

/* =================================================================== verdict + gauge */
function gaugeSVG(p, color) {
  // semicircle from 180deg (left, 0%) to 0deg (right, 100%)
  const r = 100, cx = 130, cy = 120;
  const arc = (a0, a1) => {
    const p0 = [cx + r * Math.cos(Math.PI * (1 - a0)), cy - r * Math.sin(Math.PI * (1 - a0))];
    const p1 = [cx + r * Math.cos(Math.PI * (1 - a1)), cy - r * Math.sin(Math.PI * (1 - a1))];
    return `M${p0[0].toFixed(1)} ${p0[1].toFixed(1)} A${r} ${r} 0 0 1 ${p1[0].toFixed(1)} ${p1[1].toFixed(1)}`;
  };
  const len = Math.PI * r;
  return `
  <svg viewBox="0 0 260 140" role="img" aria-label="Bug probability ${pct(p)}">
    <path class="g-zone-clean" d="${arc(0, 0.5)}"/>
    <path class="g-zone-bug" d="${arc(0.5, 1)}"/>
    <path class="g-fill" d="${arc(0, 1)}" stroke="${color}" stroke-dasharray="${len}" stroke-dashoffset="${len}" data-target="${len * (1 - p)}"/>
    <line class="g-tick" x1="${cx}" y1="${cy - r - 15}" x2="${cx}" y2="${cy - r + 15}"/>
    <text class="g-label" x="${cx - r}" y="${cy + 18}" text-anchor="middle">0</text>
    <text class="g-label" x="${cx}" y="${cy - r - 20}" text-anchor="middle">threshold 0.5</text>
    <text class="g-label" x="${cx + r}" y="${cy + 18}" text-anchor="middle">1</text>
    <line class="g-needle" x1="${cx}" y1="${cy}" x2="${cx - r + 18}" y2="${cy}" style="transform-origin:${cx}px ${cy}px; transform: rotate(0deg)" data-rot="${p * 180}"/>
    <circle class="g-hub" cx="${cx}" cy="${cy}" r="8"/>
  </svg>`;
}

function renderVerdict(res, detection) {
  const results = Object.values(res.results);
  const gauges = results.map((r) => `
    <div class="gauge">
      ${gaugeSVG(r.probability, COLORS[r.model])}
      <div class="g-pct">${pct(r.probability)}</div>
      <div class="g-model">${r.display}</div>
      <div class="g-meta">chance of a defect, ${r.tokens} tokens, ${Math.round(r.latency_ms)} ms</div>
    </div>`).join("");
  const first = results[0];
  const toks = first.top_tokens.filter((t) => t.text.trim()).map((t) =>
    `<span class="tok" style="background:${heatColor(t.score)}">${esc(t.text)}</span>`).join("");
  const truncated = results.some((r) => r.truncated)
    ? `<div class="notice">This function is longer than the model's 400-token window, so only the first part was scored.</div>` : "";
  const domain = res.language !== "c"
    ? `<div class="notice">The detector was trained on C functions. On ${esc(res.language)} code its score is a weaker signal, so Gemini double-checks it.</div>` : "";

  $("#verdict").innerHTML = `
    <div class="gauges">${gauges}</div>
    <div style="text-align:center">
      <span class="stamp ${detection.label}">${detection.label === "buggy" ? "Bug suspected" : "Looks clean"}</span>
      ${results.length > 1 ? `<div class="g-meta" style="margin-top:6px">Average of both models: ${pct(detection.probability)}</div>` : ""}
    </div>
    <div class="focus-tokens">
      <h3>Where ${first.display} looked hardest</h3>
      <div class="tok-row">${toks}</div>
    </div>
    ${truncated}${domain}`;

  requestAnimationFrame(() => requestAnimationFrame(() => {
    $$("#verdict .g-fill").forEach((el) => (el.style.strokeDashoffset = el.dataset.target));
    $$("#verdict .g-needle").forEach((el) => (el.style.transform = `rotate(${el.dataset.rot}deg)`));
  }));
}

/* =================================================================== review */
async function runReview(code, lang, detection) {
  station("review", "active", "asking Gemini");
  $("#review").innerHTML = `<div class="loading-card"><span class="spinner"></span>Gemini is reading the code and writing a review</div>`;
  try {
    const review = await api("/api/review", { code, language: lang, detection });
    state.review = review;
    const n = review.issues.length;
    station("review", "done " + (n ? "bad" : "good"), review.verdict === "buggy" ? `${n} issue${n === 1 ? "" : "s"} found` : "no defect found");
    renderReview(review, detection, code);
  } catch (err) {
    station("review", "skipped", "failed");
    $("#review").innerHTML = `<div class="notice err"><b>Review failed.</b> ${esc(err.message)}</div>`;
  }
  loadStatus(); // refresh the Gemini chip so it reflects the latest call
}

function renderReview(rv, detection, code) {
  state.flagged = new Set(rv.issues.flatMap((i) => i.lines || []));
  renderEditor();

  const src = rv.source === "gemini"
    ? `<span class="src-chip">${esc(rv.model)}, ${(rv.latency_ms / 1000).toFixed(1)} s</span>`
    : `<span class="src-chip offline">Offline reference</span>`;
  let agree = "";
  if (detection) {
    const same = detection.label === rv.verdict;
    agree = `<div class="agree">Detector said <b>${detection.label}</b> (${pct(detection.probability)}), Gemini says <b>${rv.verdict}</b>. ${same ? "They agree." : "They disagree, so trust the reasoning, not just the score."}</div>`;
  }
  const head = `
    <div class="review-head">
      <div>
        <h2>${rv.verdict === "buggy" ? `Gemini found ${rv.issues.length} issue${rv.issues.length === 1 ? "" : "s"}` : "Gemini found no defect"}</h2>
        <p>${esc(rv.summary || "")}</p>
        ${agree}
      </div>
      ${src}
    </div>`;

  const cards = rv.issues.map((it, k) => `
    <button class="pin-card sev-${it.severity}" data-lines="${(it.lines || []).join(",")}" style="animation-delay:${0.08 * k}s" title="Show these lines in the editor">
      <span class="pin-top"><span class="pin-title">${esc(it.title)}</span><span class="sev sev-${it.severity}">${it.severity}</span></span>
      <span class="pin-label"><span><b>Type</b> ${esc(it.bug_type)}</span>${it.lines && it.lines.length ? `<span><b>Line${it.lines.length > 1 ? "s" : ""}</b> ${it.lines.join(", ")}</span>` : ""}</span>
      <p>${esc(it.explanation)}</p>
      ${it.fix_hint ? `<span class="pin-fix">Fix: ${esc(it.fix_hint)}</span>` : ""}
    </button>`).join("");

  let fix = "";
  if (rv.verdict === "buggy" && rv.fixed_code && rv.fixed_code.trim() !== code.trim()) {
    fix = `
    <div class="fix-card">
      <div class="fix-head">
        <div><h3>Suggested fix</h3><p>${esc(rv.fix_explanation || "")}</p></div>
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          <button class="btn" id="copyFix">Copy</button>
          <button class="btn primary" id="useFix">Use fixed code</button>
        </div>
      </div>
      <pre class="diff">${diffHTML(code, rv.fixed_code)}</pre>
    </div>`;
  }
  const notice = rv.notice ? `<div class="notice">${esc(rv.notice)}</div>` : "";
  $("#review").innerHTML = head + cards + fix + notice;

  $$(".pin-card").forEach((c) =>
    c.addEventListener("click", () => {
      const active = c.classList.toggle("active");
      $$(".pin-card").forEach((o) => o !== c && o.classList.remove("active"));
      focusLines(active && c.dataset.lines ? c.dataset.lines.split(",").map(Number) : []);
    })
  );
  const use = $("#useFix");
  if (use) use.addEventListener("click", () => {
    setCode(rv.fixed_code, $("#language").value);
    $("#verdict").innerHTML = `<div class="empty"><p class="empty-title">Fixed code loaded</p><p>Scan it again to check that the detector now rates it clean.</p></div>`;
  });
  const copy = $("#copyFix");
  if (copy) copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(rv.fixed_code); copy.textContent = "Copied"; }
    catch { copy.textContent = "Copy failed"; }
    setTimeout(() => (copy.textContent = "Copy"), 1500);
  });
}

function diffHTML(a, b) {
  const A = a.replace(/\n+$/, "").split("\n"), B = b.replace(/\n+$/, "").split("\n");
  const n = A.length, m = B.length;
  const L = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    L[i][j] = A[i].trimEnd() === B[j].trimEnd() ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const out = [];
  let i = 0, j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && A[i].trimEnd() === B[j].trimEnd()) { out.push(`<span class="same">${esc(A[i])}</span>`); i++; j++; }
    else if (i < n && (j >= m || L[i + 1][j] >= L[i][j + 1])) { out.push(`<span class="del">${esc(A[i])}</span>`); i++; }
    else { out.push(`<span class="add">${esc(B[j])}</span>`); j++; }
  }
  return out.join("");
}

/* =================================================================== tooltip */
const tip = $("#tooltip");
function showTip(html, x, y) {
  tip.innerHTML = html;
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(window.innerWidth - w - 8, x + 14) + "px";
  tip.style.top = Math.max(8, y - h - 12) + "px";
}
function hideTip() { tip.hidden = true; }
document.addEventListener("mouseover", (e) => {
  const t = e.target.closest("[data-tip]");
  if (t) { const r = t.getBoundingClientRect(); showTip(esc(t.dataset.tip), r.right, r.top + 10); }
});
document.addEventListener("mouseout", (e) => { if (e.target.closest("[data-tip]")) hideTip(); });

/* =================================================================== model lab */
async function loadLab() {
  state.labLoaded = true;
  let data;
  try { data = await api("/api/results"); }
  catch (e) { $("#lab").innerHTML = `<div class="notice err">Could not load results: ${esc(e.message)}</div>`; return; }
  const names = ["codebert", "codet5"].filter((m) => data.models[m]);
  if (!names.length) {
    $("#lab").innerHTML = `
      <div class="lab-empty">
        <h2>No training results yet</h2>
        <p>The charts here are drawn from the files that training writes. To fill them:</p>
        <ol>
          <li>Open <code>training/BugLens_training.ipynb</code> in Google Colab with a T4 GPU.</li>
          <li>Run all cells. It trains CodeBERT and CodeT5 and downloads <code>buglens_trained.zip</code>.</li>
          <li>Unzip it into this project folder so you get <code>models/</code> and <code>results/</code>, then restart the app.</li>
        </ol>
      </div>`;
    return;
  }
  const M = data.models, base = data.baselines;
  const any = M[names[0]];
  const legend = names.map((m) => `<span><i style="background:${COLORS[m]}"></i>${NAMES[m]}</span>`).join("");

  $("#lab").innerHTML = `
    <div class="lab-intro">
      <div>
        <h2>Model lab</h2>
        <p>All scores are on the CodeXGLUE test split: ${any.config.test_size.toLocaleString()} C functions that neither model saw during training.</p>
      </div>
      <div class="legend">${legend}</div>
    </div>
    <div class="lab-grid">
      <div class="panel span-7">
        <h3>Head to head on the test set</h3>
        <p class="sub">Dashed marks show the accuracy published for each model in the CodeT5 paper.</p>
        <div id="scoreboard"></div>
        <button class="btn table-toggle" id="tableToggle" aria-expanded="false">Show as table</button>
        <div id="scoreTable" hidden></div>
      </div>
      <div class="panel span-5">
        <h3>Size and speed</h3>
        <p class="sub">Latency is one 400-token function on a plain CPU (the Colab machine), no GPU.</p>
        <div class="eff-list" id="eff"></div>
      </div>
      <div class="panel span-4 chart">
        <h3>ROC curve</h3>
        <p class="sub">Higher and further left is better. The diagonal is random guessing.</p>
        <div id="rocChart"></div>
      </div>
      <div class="panel span-4 chart">
        <h3>Validation accuracy by epoch</h3>
        <p class="sub">The best epoch is kept. Training stops after 2 epochs without gains.</p>
        <div id="valChart"></div>
      </div>
      <div class="panel span-4 chart">
        <h3>Training loss by epoch</h3>
        <p class="sub">Average binary cross-entropy over each epoch.</p>
        <div id="lossChart"></div>
      </div>
      <div class="panel span-6">
        <h3>Confusion matrix</h3>
        <p class="sub">Rows are the true label, columns are what the model predicted.</p>
        <div class="cm-wrap" id="cms"></div>
      </div>
      <div class="panel span-6 chart">
        <h3>Cross-language probe</h3>
        <p class="sub">Accuracy on the hand-written bug and fix pairs. Training data was C only.</p>
        <div id="probeChart"></div>
        <div id="probeItems"></div>
      </div>
    </div>`;

  // ---- scoreboard
  const metrics = [
    ["accuracy", "Accuracy", "share correct"],
    ["precision", "Precision", "flagged that were buggy"],
    ["recall", "Recall", "bugs that were caught"],
    ["f1", "F1 score", "balance of the two"],
    ["auc", "ROC AUC", "ranking quality"],
    ["mcc", "MCC", "-1 to 1, 0 is chance"],
  ];
  $("#scoreboard").innerHTML = metrics.map(([k, label, hint]) => `
    <div class="score-row">
      <div class="score-name">${label}<small>${hint}</small></div>
      <div class="bars">${names.map((m) => {
        const v = M[m].test[k];
        const b = k === "accuracy" && base[m] ? `<span class="bar-base" style="left:${base[m] * 100}%" data-tip="${NAMES[m]} published: ${pct(base[m], 2)}"></span>` : "";
        return `<div class="bar-line"><div class="bar-track">${b}<div class="bar-fill" style="width:0;background:${COLORS[m]}" data-w="${Math.max(0, v || 0) * 100}" data-tip="${NAMES[m]} ${label}: ${k === "mcc" ? (v ?? 0).toFixed(3) : pct(v, 2)}"></div></div><span class="bar-val">${k === "mcc" ? (v ?? 0).toFixed(3) : pct(v, 1)}</span></div>`;
      }).join("")}</div>
    </div>`).join("");
  requestAnimationFrame(() => requestAnimationFrame(() => $$(".bar-fill").forEach((b) => (b.style.width = b.dataset.w + "%"))));
  $("#scoreTable").innerHTML = `<table class="data-table"><thead><tr><th>Metric</th>${names.map((m) => `<th>${NAMES[m]}</th>`).join("")}<th>Published accuracy</th></tr></thead><tbody>${
    metrics.map(([k, label]) => `<tr><td>${label}</td>${names.map((m) => `<td>${(M[m].test[k] ?? 0).toFixed(4)}</td>`).join("")}<td>${k === "accuracy" ? names.map((m) => base[m] ? `${NAMES[m]} ${base[m].toFixed(4)}` : "").join(", ") : ""}</td></tr>`).join("")}</tbody></table>`;
  $("#tableToggle").addEventListener("click", (e) => {
    const t = $("#scoreTable"); t.hidden = !t.hidden;
    e.target.setAttribute("aria-expanded", String(!t.hidden));
    e.target.textContent = t.hidden ? "Show as table" : "Hide table";
  });

  // ---- efficiency
  const fmtParams = (n) => (n / 1e6).toFixed(0) + "M";
  const fmtMin = (s) => (s < 120 ? Math.round(s) + " s" : (s / 60).toFixed(0) + " min");
  const effRows = [
    ["Parameters", (r) => fmtParams(r.parameters), "weights in the model"],
    ["CPU time per function", (r) => Math.round(r.cpu_latency_ms) + " ms", "400 tokens, no GPU"],
    ["Training time", (r) => fmtMin(r.train_seconds), `${any.config.device}`],
    ["Epochs run", (r) => `${r.config.epochs_run} of ${r.config.max_epochs}`, "stops early after 2 flat epochs"],
  ];
  $("#eff").innerHTML = effRows.map(([title, f, lab]) => `
    <div><h4>${title}</h4><div class="stat-pair">${names.map((m) => `
      <div class="stat"><div class="s-model"><i style="background:${COLORS[m]}"></i>${NAMES[m]}</div><div class="s-val">${f(M[m])}</div><div class="s-lab">${esc(lab)}</div></div>`).join("")}</div></div>`).join("");

  // ---- charts
  lineChart("#rocChart", {
    series: names.map((m) => ({ key: m, name: `${NAMES[m]} (AUC ${(M[m].test.auc ?? 0).toFixed(3)})`, short: NAMES[m], color: COLORS[m], points: M[m].roc })),
    x: [0, 1], y: [0, 1], xTitle: "False positive rate", yTitle: "True positive rate", diagonal: true,
    fx: (v) => v.toFixed(1), fy: (v) => v.toFixed(1), tipX: (v) => `FPR ${v.toFixed(2)}`, tipY: (v) => `TPR ${v.toFixed(2)}`,
  });
  const maxEp = Math.max(...names.map((m) => M[m].history.length));
  const valVals = names.flatMap((m) => M[m].history.map((h) => h.val_accuracy));
  lineChart("#valChart", {
    series: names.map((m) => ({ key: m, name: NAMES[m], short: NAMES[m], color: COLORS[m], points: M[m].history.map((h) => [h.epoch, h.val_accuracy]), markers: true })),
    x: [1, Math.max(2, maxEp)], y: [Math.floor(Math.min(...valVals) * 20) / 20 - 0.05, Math.min(1, Math.ceil(Math.max(...valVals) * 20) / 20 + 0.02)],
    xTitle: "Epoch", yTitle: "Accuracy", fx: (v) => (Number.isInteger(v) ? v : ""), fy: (v) => pct(v), integerX: true,
    tipX: (v) => `Epoch ${v}`, tipY: (v) => pct(v, 1),
  });
  const lossVals = names.flatMap((m) => M[m].history.map((h) => h.train_loss));
  lineChart("#lossChart", {
    series: names.map((m) => ({ key: m, name: NAMES[m], short: NAMES[m], color: COLORS[m], points: M[m].history.map((h) => [h.epoch, h.train_loss]), markers: true })),
    x: [1, Math.max(2, maxEp)], y: [Math.max(0, Math.min(...lossVals) - 0.05), Math.max(...lossVals) + 0.05],
    xTitle: "Epoch", yTitle: "Loss", fx: (v) => (Number.isInteger(v) ? v : ""), fy: (v) => v.toFixed(2), integerX: true,
    tipX: (v) => `Epoch ${v}`, tipY: (v) => v.toFixed(3),
  });

  // ---- confusion matrices (sequential, one hue per model)
  $("#cms").innerHTML = names.map((m) => {
    const cm = M[m].confusion_matrix, total = cm.flat().reduce((a, b) => a + b, 0), mx = Math.max(...cm.flat());
    const cell = (v, what) => {
      const a = 0.1 + 0.85 * (v / mx);
      const rgb = m === "codebert" ? "36,86,232" : "224,48,127";
      return `<div class="cm-cell" style="background:rgba(${rgb},${a.toFixed(2)});color:${a > 0.55 ? "#fff" : "#1C1A3A"}" data-tip="${what}: ${v} of ${total} (${pct(v / total, 1)})">${v}<small>${what}</small></div>`;
    };
    return `<div class="cm"><h4><i style="background:${COLORS[m]}"></i>${NAMES[m]}</h4>
      <div class="cm-grid">
        <span></span><span class="h">Predicted clean</span><span class="h">Predicted buggy</span>
        <span class="r">Clean</span>${cell(cm[0][0], "True negative")}${cell(cm[0][1], "False alarm")}
        <span class="r">Buggy</span>${cell(cm[1][0], "Missed bug")}${cell(cm[1][1], "Caught bug")}
      </div></div>`;
  }).join("") + `<div class="cm-note">${names.map((m) => {
    const cm = M[m].confusion_matrix, bugs = cm[1][0] + cm[1][1], clean = cm[0][0] + cm[0][1];
    return `<p><b>${NAMES[m]}</b> catches ${cm[1][1].toLocaleString()} of ${bugs.toLocaleString()} buggy functions (${pct(cm[1][1] / bugs)}) and raises a false alarm on ${pct(cm[0][1] / clean)} of clean ones.</p>`;
  }).join("")}<p>A missed bug is the costly mistake here: code rated clean is never sent to Gemini for review.</p></div>`;

  // ---- probe
  const probeNames = names.filter((m) => M[m].probe);
  if (probeNames.length) {
    const langs = Object.keys(M[probeNames[0]].probe.by_language);
    barChart("#probeChart", {
      groups: langs.map((l) => ({ label: `${l === "c" ? "C" : l[0].toUpperCase() + l.slice(1)} (${M[probeNames[0]].probe.by_language[l].n} functions)`, values: probeNames.map((m) => ({ key: m, color: COLORS[m], name: NAMES[m], v: M[m].probe.by_language[l].accuracy })) })),
    });
    $("#probeItems").innerHTML = probeNames.map((m) => `
      <div style="margin-top:10px"><div class="s-model" style="font-size:.85rem;font-weight:700;display:flex;gap:6px;align-items:center"><i style="width:10px;height:10px;border-radius:2px;background:${COLORS[m]}"></i>${NAMES[m]}, each pair</div>
      <div class="probe-items">${M[m].probe.items.map((it) => {
        const ok = (it.prob >= 0.5 ? 1 : 0) === it.target;
        return `<span class="probe-dot" style="background:${ok ? "#D3F5E6" : "#FFDDEC"}" data-tip="${esc(it.id)}: ${it.target ? "buggy" : "fixed"} version scored ${pct(it.prob)}, ${ok ? "correct" : "wrong"}">${ok ? "✓" : "✗"}</span>`;
      }).join("")}</div></div>`).join("");
  } else {
    $("#probeChart").innerHTML = `<p class="sub">No probe results in the training output.</p>`;
  }
}

/* ---- line chart with crosshair tooltip */
function lineChart(sel, o) {
  const W = 420, H = 280, m = { l: 52, r: o.diagonal ? 16 : 92, t: 14, b: 44 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const sx = (v) => m.l + ((v - o.x[0]) / (o.x[1] - o.x[0])) * iw;
  const sy = (v) => m.t + ih - ((v - o.y[0]) / (o.y[1] - o.y[0])) * ih;
  const xt = o.integerX ? range(o.x[0], o.x[1]) : ticks(o.x[0], o.x[1], 5);
  const yt = ticks(o.y[0], o.y[1], 5);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.yTitle)} by ${esc(o.xTitle)}">`;
  s += `<g class="grid">${yt.map((v) => `<line x1="${m.l}" x2="${W - m.r}" y1="${sy(v)}" y2="${sy(v)}"/>`).join("")}</g>`;
  s += `<g class="axis">${yt.map((v) => `<text x="${m.l - 8}" y="${sy(v) + 4}" text-anchor="end">${o.fy(v)}</text>`).join("")}
        ${xt.map((v) => `<text x="${sx(v)}" y="${H - m.b + 18}" text-anchor="middle">${o.fx(v)}</text>`).join("")}</g>`;
  s += `<text class="axis-title" x="${m.l + iw / 2}" y="${H - 6}" text-anchor="middle">${esc(o.xTitle)}</text>`;
  s += `<text class="axis-title" transform="translate(13 ${m.t + ih / 2}) rotate(-90)" text-anchor="middle">${esc(o.yTitle)}</text>`;
  if (o.diagonal) s += `<line class="baseline" x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}"/>`;
  o.series.forEach((se) => {
    const d = se.points.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)} ${sy(p[1]).toFixed(1)}`).join(" ");
    s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>`;
    if (se.markers) s += se.points.map((p) => `<circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="4.5" fill="${se.color}" stroke="#fff" stroke-width="2"/>`).join("");
  });
  // direct labels at line ends, nudged apart if they collide
  const ends = o.series.map((se) => { const p = se.points[se.points.length - 1]; return { se, x: sx(p[0]), y: sy(p[1]) }; }).sort((a, b) => a.y - b.y);
  ends.forEach((e) => (e.ly = e.y + 4));
  for (let i = 1; i < ends.length; i++) if (ends[i].ly - ends[i - 1].ly < 15) ends[i].ly = ends[i - 1].ly + 15;
  if (!o.diagonal) ends.forEach((e) => { s += `<text class="dir-label" x="${e.x + 9}" y="${e.ly}" text-anchor="start">${esc(e.se.short)}</text>`; });
  else ends.forEach((e, i) => { s += `<text class="dir-label" x="${m.l + iw - 4}" y="${m.t + ih - 10 - i * 16}" text-anchor="end" fill="${e.se.color}">${esc(e.se.name)}</text>`; });
  s += `<line class="xhair" y1="${m.t}" y2="${m.t + ih}" x1="-10" x2="-10" visibility="hidden"/>`;
  s += `<rect class="hit" x="${m.l}" y="${m.t}" width="${iw}" height="${ih}"/></svg>`;
  const host = $(sel);
  host.innerHTML = s;
  const svg = $("svg", host), hit = $(".hit", host), xh = $(".xhair", host);
  hit.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    let xv = o.x[0] + ((px - m.l) / iw) * (o.x[1] - o.x[0]);
    if (o.integerX) xv = Math.round(xv);
    const rows = o.series.map((se) => {
      let best = se.points[0];
      for (const p of se.points) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
      return `<div><span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${se.color};margin-right:6px"></span>${esc(se.short)}: <b>${o.tipY(best[1])}</b></div>`;
    });
    xh.setAttribute("x1", sx(xv)); xh.setAttribute("x2", sx(xv)); xh.setAttribute("visibility", "visible");
    showTip(`<div style="margin-bottom:3px">${o.tipX(xv)}</div>${rows.join("")}`, e.clientX, e.clientY);
  });
  hit.addEventListener("mouseleave", () => { hideTip(); xh.setAttribute("visibility", "hidden"); });
}

/* ---- grouped bar chart */
function barChart(sel, o) {
  const W = 620, H = 250, m = { l: 44, r: 10, t: 18, b: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const sy = (v) => m.t + ih - v * ih;
  const gw = iw / o.groups.length, bw = Math.min(46, (gw - 30) / o.groups[0].values.length);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Probe accuracy by language">`;
  s += `<g class="grid">${[0, .25, .5, .75, 1].map((v) => `<line x1="${m.l}" x2="${W - m.r}" y1="${sy(v)}" y2="${sy(v)}"/>`).join("")}</g>`;
  s += `<g class="axis">${[0, .25, .5, .75, 1].map((v) => `<text x="${m.l - 8}" y="${sy(v) + 4}" text-anchor="end">${pct(v)}</text>`).join("")}</g>`;
  s += `<line class="baseline" x1="${m.l}" x2="${W - m.r}" y1="${sy(.5)}" y2="${sy(.5)}"/><text class="axis" x="${W - m.r}" y="${sy(.5) - 5}" text-anchor="end" style="font:12px var(--font);fill:var(--ink-3)">coin flip</text>`;
  o.groups.forEach((g, gi) => {
    const x0 = m.l + gi * gw + (gw - bw * g.values.length - 2 * (g.values.length - 1)) / 2;
    g.values.forEach((b, bi) => {
      const x = x0 + bi * (bw + 2), y = sy(b.v), h = m.t + ih - y, r = Math.min(4, h);
      s += `<path d="M${x} ${m.t + ih} V${y + r} Q${x} ${y} ${x + r} ${y} H${x + bw - r} Q${x + bw} ${y} ${x + bw} ${y + r} V${m.t + ih} Z" fill="${b.color}" data-tip="${esc(b.name)} on ${esc(g.label)}: ${pct(b.v, 1)}"/>`;
      s += `<text x="${x + bw / 2}" y="${y - 6}" text-anchor="middle" style="font:700 12px var(--font);fill:var(--ink)">${pct(b.v)}</text>`;
    });
    s += `<text class="axis" x="${m.l + gi * gw + gw / 2}" y="${H - m.b + 20}" text-anchor="middle" style="font:600 12px var(--font);fill:var(--ink-2)">${esc(g.label)}</text>`;
  });
  $(sel).innerHTML = s + "</svg>";
}

function ticks(a, b, n) {
  const step = niceStep((b - a) / n), out = [];
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
}
function range(a, b) { const o = []; for (let i = a; i <= b; i++) o.push(i); return o; }

/* =================================================================== boot */
(async function boot() {
  resetRail();
  await loadStatus();
  await loadSamples();
  renderEditor();
})();
