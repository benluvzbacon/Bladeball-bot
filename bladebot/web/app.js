"use strict";
/* BladeBot control menu - vanilla JS, talks to the local Python server. */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = {
  schema: null,
  values: {},
  status: null,
  tab: "control",
  history: [],
  lastParries: null,
  lastEventKey: "",
  lastArenaKey: "",
  clickMode: "anchor",
  pending: {},
  pendingTimer: null,
  trainTimer: null,
  connected: true,
  parryFlashUntil: 0,
};

// ------------------------------------------------------------------ helpers
async function api(path, body) {
  const opts = body === undefined
    ? { cache: "no-store" }
    : { method: "POST", headers: { "Content-Type": "application/json", "X-BladeBot": "1" }, body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch (_) { data = null; }
  if (!res.ok) {
    const err = new Error((data && data.error) || res.statusText || "request failed");
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

let toastTimer = null;
function toast(msg, isError = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (isError ? " error" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, isError ? 5000 : 2500);
}

function fmtMs(sec) {
  if (sec === null || sec === undefined || !isFinite(sec)) return "-";
  return Math.round(sec * 1000) + " ms";
}

function isTyping() {
  const a = document.activeElement;
  return a && (a.tagName === "INPUT" || a.tagName === "SELECT" || a.tagName === "TEXTAREA");
}

// ------------------------------------------------------------------ tabs
function setTab(name) {
  if (!$("#tab-" + name)) name = "control";
  state.tab = name;
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  if (name === "vision") visionLoop.start();
  if (name === "arena") {
    arenaLoop.start();
    const st = state.status;
    if (state.values.source !== "arena" && !(st && st.enabled)) {
      setSetting("source", "arena", true);
      toast("Switched the source to the practice arena");
    }
  }
  if (name === "train") { refreshModel(); refreshRecordings(); refreshTrain(); }
}
$$(".tab").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));

// ------------------------------------------------------------------ preview loops
function previewLoop(img, tabs) {
  let running = false;
  const next = () => {
    if (!tabs.includes(state.tab) || document.hidden) { running = false; return; }
    running = true;
    img.src = "api/preview.png?t=" + Date.now();
  };
  img.addEventListener("load", () => setTimeout(next, 70));
  img.addEventListener("error", () => setTimeout(next, 1000));
  return { start() { if (!running) next(); } };
}
const visionLoop = previewLoop($("#visionPreview"), ["vision"]);
const arenaLoop = previewLoop($("#arenaPreview"), ["arena"]);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) { visionLoop.start(); arenaLoop.start(); }
});

// ------------------------------------------------------------------ settings forms
const FORM_FOR_GROUP = { timing: "#timingForm", input: "#inputForm", capture: "#visionForm", vision: "#visionForm", gate: "#visionForm", arena: "#arenaForm" };
const fields = {};

function decimals(step) {
  const s = String(step || 1);
  return s.includes(".") ? s.split(".")[1].length : 0;
}

function fmtValue(s, v) {
  if (s.kind === "int") return `${Math.round(v)}${s.unit ? " " + s.unit : ""}`;
  if (s.kind === "float") return `${Number(v).toFixed(Math.min(decimals(s.step), 3))}${s.unit ? " " + s.unit : ""}`;
  return String(v);
}

function keyLabel(k) { return String(k || "").replace("_", " ").toUpperCase(); }

function buildForms() {
  Object.values(FORM_FOR_GROUP).forEach((sel) => { $(sel).innerHTML = ""; });
  for (const [group, title] of Object.entries(state.schema.groups)) {
    const box = $(FORM_FOR_GROUP[group]);
    if (!box) continue;
    if (group !== "arena") box.append(el("h3", "", title));
    for (const s of state.schema.settings.filter((x) => x.group === group)) box.append(buildField(s));
  }
  syncForms();
}

function buildField(s) {
  const wrap = el("div", "field");
  const top = el("div", "field-top");
  const label = el("label", "", s.label);
  const id = "set-" + s.key;
  label.htmlFor = id;
  let update;
  if (s.kind === "int" || s.kind === "float") {
    const val = el("span", "val");
    const input = el("input");
    Object.assign(input, { type: "range", id, min: s.min, max: s.max, step: s.step || (s.kind === "int" ? 1 : 0.01) });
    input.addEventListener("input", () => {
      const v = s.kind === "int" ? parseInt(input.value, 10) : parseFloat(input.value);
      val.textContent = fmtValue(s, v);
      setSetting(s.key, v);
    });
    top.append(label, val);
    wrap.append(top, input);
    update = (v) => { if (document.activeElement !== input) input.value = v; val.textContent = fmtValue(s, v); };
  } else if (s.kind === "bool") {
    const input = el("input");
    Object.assign(input, { type: "checkbox", id });
    input.addEventListener("change", () => setSetting(s.key, input.checked, true));
    const lab = el("label", "check");
    lab.append(input, document.createTextNode(s.label));
    wrap.append(lab);
    update = (v) => { input.checked = !!v; };
  } else if (s.kind === "choice") {
    const sel = el("select");
    sel.id = id;
    for (const [value, text] of s.choices) {
      const o = el("option", "", text);
      o.value = value;
      sel.append(o);
    }
    sel.addEventListener("change", () => setSetting(s.key, sel.value, true));
    top.append(label, sel);
    wrap.append(top);
    update = (v) => { sel.value = v; };
  } else if (s.kind === "key") {
    const btn = el("button", "btn keybtn");
    btn.type = "button";
    btn.id = id;
    let listening = false;
    const onKey = (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      window.removeEventListener("keydown", onKey, true);
      listening = false;
      btn.classList.remove("listening");
      if (ev.code === "Escape") { update(state.values[s.key]); return; }
      const name = keyName(ev);
      if (!name) { toast("That key is not supported", true); update(state.values[s.key]); return; }
      setSetting(s.key, name, true);
      update(name);
    };
    btn.addEventListener("click", () => {
      if (listening) return;
      listening = true;
      btn.classList.add("listening");
      btn.textContent = "press a key...";
      window.addEventListener("keydown", onKey, true);
    });
    top.append(label, btn);
    wrap.append(top);
    update = (v) => { if (!listening) btn.textContent = keyLabel(v); };
  }
  if (s.help) wrap.append(el("div", "help", s.help));
  fields[s.key] = update;
  return wrap;
}

function syncForms() {
  for (const [key, update] of Object.entries(fields)) {
    if (key in state.values && !(key in state.pending)) update(state.values[key]);
  }
  $("#dryRunQuick").checked = !!state.values.dry_run;
  $("#hotkeyHint").textContent = keyLabel(state.values.hotkey || "f6");
  $$("#sourceSwitch button").forEach((b) => b.classList.toggle("active", b.dataset.source === state.values.source));
  $$("#arenaMode button").forEach((b) => b.classList.toggle("active", b.dataset.mode === state.values.arena_mode));
}

function setSetting(key, value, immediate = false) {
  state.pending[key] = value;
  state.values[key] = value;
  clearTimeout(state.pendingTimer);
  state.pendingTimer = setTimeout(flushSettings, immediate ? 0 : 160);
}

async function flushSettings() {
  const changes = state.pending;
  state.pending = {};
  if (!Object.keys(changes).length) return;
  try {
    const r = await api("api/settings", { changes });
    state.values = r.values;
    syncForms();
  } catch (e) {
    toast("Could not save setting: " + e.message, true);
  }
}

$$("[data-reset]").forEach((b) => b.addEventListener("click", async () => {
  const group = b.dataset.reset;
  const groups = group === "vision" ? ["capture", "vision", "gate"] : [group];
  try {
    for (const g of groups) {
      const r = await api("api/settings/reset", { group: g });
      state.values = r.values;
    }
    syncForms();
    toast("Defaults restored");
  } catch (e) { toast(e.message, true); }
}));

function keyName(ev) {
  const c = ev.code || "";
  if (/^Key[A-Z]$/.test(c)) return c.slice(3).toLowerCase();
  if (/^Digit[0-9]$/.test(c)) return c.slice(5);
  if (/^F([1-9]|1[0-2])$/.test(c)) return c.toLowerCase();
  const map = {
    Space: "space", Tab: "tab", Enter: "enter", ShiftLeft: "shift", ShiftRight: "shift", ControlLeft: "ctrl",
    ControlRight: "ctrl", AltLeft: "alt", AltRight: "alt", CapsLock: "caps_lock", Insert: "insert", Delete: "delete",
    Home: "home", End: "end", PageUp: "page_up", PageDown: "page_down", ArrowUp: "up", ArrowDown: "down",
    ArrowLeft: "left", ArrowRight: "right", Pause: "pause", ScrollLock: "scroll_lock", NumLock: "num_lock",
    PrintScreen: "print_screen", ContextMenu: "menu",
  };
  return map[c] || null;
}

// ------------------------------------------------------------------ practice notice
function requireAck() {
  return new Promise((resolve) => {
    const modal = $("#ackModal");
    const check = $("#ackCheck");
    const accept = $("#ackAccept");
    const cancel = $("#ackCancel");
    check.checked = false;
    accept.disabled = true;
    modal.hidden = false;
    const done = async (ok) => {
      modal.hidden = true;
      check.onchange = accept.onclick = cancel.onclick = null;
      if (ok) {
        try {
          await api("api/ack", { accepted: true });
          state.values.practice_ack = true;
        } catch (e) { toast(e.message, true); ok = false; }
      }
      resolve(ok);
    };
    check.onchange = () => { accept.disabled = !check.checked; };
    accept.onclick = () => done(true);
    cancel.onclick = () => done(false);
  });
}

$("#withdrawAck").addEventListener("click", async () => {
  try {
    await api("api/ack", { accepted: false });
    state.values.practice_ack = false;
    toast("Acceptance withdrawn - the bot will ask again before running on Roblox");
  } catch (e) { toast(e.message, true); }
});

// ------------------------------------------------------------------ control
async function togglePower() {
  const st = state.status;
  const turnOn = !(st && st.enabled);
  if (turnOn && state.values.source === "screen" && !state.values.practice_ack) {
    const ok = await requireAck();
    if (!ok) return;
  }
  try {
    await api("api/toggle", { on: turnOn });
  } catch (e) {
    if (e.data && /practice-only/i.test(e.message)) {
      if (await requireAck()) return togglePower();
      return;
    }
    toast(e.message, true);
  }
}
$("#powerBtn").addEventListener("click", togglePower);

$$("#sourceSwitch button").forEach((b) => b.addEventListener("click", () => setSetting("source", b.dataset.source, true)));
$("#dryRunQuick").addEventListener("change", (ev) => setSetting("dry_run", ev.target.checked, true));

// ------------------------------------------------------------------ vision clicks
$$("#clickMode button").forEach((b) => b.addEventListener("click", () => {
  state.clickMode = b.dataset.mode;
  $$("#clickMode button").forEach((x) => x.classList.toggle("active", x === b));
}));

$("#visionPreview").addEventListener("click", async (ev) => {
  const img = ev.currentTarget;
  const r = img.getBoundingClientRect();
  const x = (ev.clientX - r.left) / r.width;
  const y = (ev.clientY - r.top) / r.height;
  try {
    if (state.clickMode === "anchor") {
      const res = await api("api/anchor", { x, y });
      Object.assign(state.values, { anchor_x: res.anchor_x, anchor_y: res.anchor_y });
      toast("Character position set");
    } else {
      const res = await api("api/pick_color", { x, y });
      Object.assign(state.values, { hue_center: res.hue_center, sat_min: res.sat_min, val_min: res.val_min });
      toast(`Ball colour set - hue ${Math.round(res.hue_center)} deg (RGB ${res.rgb.join(", ")})`);
    }
    syncForms();
  } catch (e) { toast(e.message, true); }
});

// ------------------------------------------------------------------ arena
$$("#arenaMode button").forEach((b) => b.addEventListener("click", () => setSetting("arena_mode", b.dataset.mode, true)));
$("#arenaReset").addEventListener("click", async () => {
  try { await api("api/arena/reset", {}); toast("Arena restarted"); } catch (e) { toast(e.message, true); }
});
async function humanBlock() {
  try { await api("api/arena/parry", {}); } catch (e) { toast(e.message, true); }
}
$("#blockBtn").addEventListener("mousedown", (ev) => { ev.preventDefault(); humanBlock(); });
window.addEventListener("keydown", (ev) => {
  if (state.tab !== "arena" || isTyping() || ev.repeat) return;
  if (ev.code === "Space") {
    ev.preventDefault();
    if (state.values.arena_mode === "human") humanBlock();
  }
});

// ------------------------------------------------------------------ charts
function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || cv.width;
  const h = cv.clientHeight || cv.height;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function drawProb(st) {
  const { ctx, w, h } = setupCanvas($("#probChart"));
  ctx.clearRect(0, 0, w, h);
  const L = 40, R = 22, T = 22, B = 24;
  const pw = w - L - R, ph = h - T - B;
  const horizons = (st && st.horizons) || [1.0];
  const hmax = horizons[horizons.length - 1];
  const X = (t) => L + (t / hmax) * pw;
  const Y = (p) => T + (1 - p) * ph;
  ctx.font = "11px system-ui, sans-serif";
  ctx.strokeStyle = "#2c3444";
  ctx.fillStyle = "#8f99ab";
  ctx.lineWidth = 1;
  for (let p = 0; p <= 1.001; p += 0.25) {
    ctx.beginPath(); ctx.moveTo(L, Y(p)); ctx.lineTo(L + pw, Y(p)); ctx.stroke();
    ctx.fillText(p.toFixed(2), 6, Y(p) + 4);
  }
  ctx.textAlign = "center";
  for (let t = 0; t <= hmax + 1e-6; t += 0.1) {
    ctx.fillText(Math.round(t * 1000) + "", X(t), h - 7);
  }
  ctx.textAlign = "right";
  ctx.fillText("time until impact (ms)", L + pw, 13);
  ctx.textAlign = "left";
  const lead = (state.values.lead_ms || 300) / 1000;
  const conf = state.values.confidence || 0.6;
  // confidence line
  ctx.setLineDash([5, 4]);
  ctx.strokeStyle = "#3fd6ff";
  ctx.beginPath(); ctx.moveTo(L, Y(conf)); ctx.lineTo(L + pw, Y(conf)); ctx.stroke();
  ctx.strokeStyle = "#ffd23f";
  ctx.beginPath(); ctx.moveTo(X(lead), T); ctx.lineTo(X(lead), T + ph); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#ffd23f";
  ctx.fillText(`lead ${Math.round(lead * 1000)} ms`, Math.min(X(lead) + 4, L + pw - 80), 13);
  ctx.fillStyle = "#3fd6ff";
  ctx.fillText(`threshold ${conf.toFixed(2)}`, L + 6, Y(conf) - 5);
  const probs = st && st.nn && st.nn.probs;
  if (!probs) {
    const msg = st && st.enabled ? "No ball coming at you" : "Bot is off - no ball tracked";
    ctx.font = "13px system-ui, sans-serif";
    const tw = ctx.measureText(msg).width;
    const cx = L + pw * 0.62, cy = T + ph * 0.42;
    ctx.fillStyle = "rgba(22, 26, 35, .92)";
    ctx.fillRect(cx - tw / 2 - 10, cy - 16, tw + 20, 26);
    ctx.fillStyle = "#8f99ab";
    ctx.textAlign = "center";
    ctx.fillText(msg, cx, cy + 2);
    ctx.textAlign = "left";
    return;
  }
  const grad = ctx.createLinearGradient(0, T, 0, T + ph);
  grad.addColorStop(0, "rgba(255, 122, 168, .45)");
  grad.addColorStop(1, "rgba(255, 122, 168, .02)");
  ctx.beginPath();
  ctx.moveTo(X(0), Y(0));
  horizons.forEach((t, i) => ctx.lineTo(X(t), Y(probs[i])));
  ctx.lineTo(X(hmax), Y(0));
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(X(0), Y(0));
  horizons.forEach((t, i) => ctx.lineTo(X(t), Y(probs[i])));
  ctx.strokeStyle = "#ff7aa8";
  ctx.lineWidth = 2;
  ctx.stroke();
  const p = st.nn.p_lead;
  ctx.beginPath();
  ctx.arc(X(lead), Y(p), 6, 0, Math.PI * 2);
  ctx.fillStyle = p >= conf ? "#ff3d5a" : "#e7eaf0";
  ctx.fill();
}

function drawTimeline() {
  const { ctx, w, h } = setupCanvas($("#timeline"));
  ctx.clearRect(0, 0, w, h);
  const now = performance.now() / 1000;
  const span = 6;
  const X = (t) => ((t - (now - span)) / span) * w;
  const Y = (p) => 6 + (1 - p) * (h - 12);
  const hist = state.history;
  // targeted shading
  ctx.fillStyle = "rgba(255, 61, 90, .22)";
  for (let i = 1; i < hist.length; i++) {
    if (hist[i - 1].targ) ctx.fillRect(X(hist[i - 1].t), 0, Math.max(X(hist[i].t) - X(hist[i - 1].t), 1), h);
  }
  const conf = state.values.confidence || 0.6;
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = "#3fd6ff";
  ctx.beginPath(); ctx.moveTo(0, Y(conf)); ctx.lineTo(w, Y(conf)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  hist.forEach((pt, i) => (i ? ctx.lineTo(X(pt.t), Y(pt.p)) : ctx.moveTo(X(pt.t), Y(pt.p))));
  ctx.strokeStyle = "#ff7aa8";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.strokeStyle = "#ffd23f";
  hist.forEach((pt) => {
    if (pt.parry) { ctx.beginPath(); ctx.moveTo(X(pt.t), 0); ctx.lineTo(X(pt.t), h); ctx.stroke(); }
  });
}
window.addEventListener("resize", () => { drawProb(state.status); drawTimeline(); });

// ------------------------------------------------------------------ rendering status
function setAlerts(st) {
  const list = [];
  if (!state.connected) list.push(["error", "Lost connection to BladeBot. Is the program still running?"]);
  if (st) {
    if (st.model && !st.model.loaded) list.push(["error", (st.model.error || "No neural network loaded.") + " Open the Train tab to train one."]);
    if (st.source === "screen") {
      if (st.capture_error) list.push(["error", "Screen capture: " + st.capture_error]);
      if (st.input && !st.input.backend && st.input.error) list.push(["warn", "Can't send key presses: " + st.input.error + " (observe-only still works)."]);
      if (st.input && st.input.last_error) list.push(["warn", st.input.last_error]);
    }
    if (st.hotkey && st.hotkey.error) list.push(["info", "Hotkey: " + st.hotkey.error + " - use the button instead."]);
  }
  const key = JSON.stringify(list);
  if (key === setAlerts.last) return;
  setAlerts.last = key;
  const box = $("#alerts");
  box.innerHTML = "";
  for (const [kind, text] of list) box.append(el("div", "alert " + kind, text));
}

function renderEvents(listEl, events, key, emptyText) {
  const k = JSON.stringify(events.slice(-1)) + events.length;
  if (state[key] === k) return;
  state[key] = k;
  listEl.innerHTML = "";
  if (!events.length && emptyText) listEl.append(el("li", "empty", emptyText));
  for (const ev of [...events].reverse()) {
    const li = el("li", ev.kind || "");
    li.append(el("span", "time", ev.time || (ev.t !== undefined ? ev.t.toFixed(1) + "s" : "")), el("span", "", ev.text));
    listEl.append(li);
  }
}

function render(st) {
  const on = st.enabled;
  const dry = on && st.dry_run;
  const pill = $("#topState");
  pill.textContent = dry ? "OBSERVING" : on ? "ON" : "OFF";
  pill.className = "pill " + (dry ? "dry" : on ? "on" : "off");
  $("#topSource").textContent = st.source === "arena" ? "Practice arena" : "Roblox (screen)";
  $("#topFps").textContent = st.idle ? "idle" : `${Math.round(st.fps)} fps`;
  const power = $("#powerBtn");
  power.className = "power " + (dry ? "dry" : on ? "on" : "off");
  power.setAttribute("aria-pressed", on ? "true" : "false");
  $("#powerLabel").textContent = on ? "Turn bot OFF" : "Turn bot ON";
  let hint = "";
  if (!on && st.source === "screen" && !st.practice_ack) hint = "You'll be asked to accept the practice-only notice first.";
  else if (on && st.source === "screen") hint = dry ? "Watching Roblox - not pressing anything." : "Watching Roblox. Keep the Blade Ball window visible.";
  else if (on && st.source === "arena") hint = st.arena_mode === "human" ? "Arena in 'I play' mode - the network only shows when it would press." : "The network is playing the practice arena.";
  $("#powerHint").textContent = hint;
  if (st.practice_ack !== undefined) state.values.practice_ack = st.practice_ack;

  const v = st.vision || {};
  $("#dotTarget").className = "dot" + (v.targeted ? " red" : "");
  $("#stTarget").textContent = st.idle ? "-" : v.targeted ? "YES" : "no";
  $("#dotBall").className = "dot" + (v.ball ? " green" : "");
  $("#stBall").textContent = st.idle ? "-" : v.ball ? (st.nn && st.nn.tracking ? "tracking" : "seen") : "none";
  $("#stEta").textContent = st.nn && st.nn.eta !== null && st.nn.probs ? fmtMs(st.nn.eta) : "-";
  $("#stP").textContent = st.nn && st.nn.probs ? st.nn.p_lead.toFixed(2) : "-";
  $("#stParries").textContent = st.decision ? st.decision.parries : 0;
  $("#stProc").textContent = st.idle ? "-" : `${st.proc_ms.toFixed(1)} ms`;

  const parries = st.decision ? st.decision.parries : 0;
  const newParry = state.lastParries !== null && parries > state.lastParries;
  state.lastParries = parries;
  if (newParry) state.parryFlashUntil = performance.now() + 400;
  const would = $("#wouldPress");
  const flash = performance.now() < state.parryFlashUntil;
  would.textContent = flash ? "PARRY!" : "WOULD PARRY";
  would.classList.toggle("show", flash || !!(st.decision && st.decision.would_press));

  const now = performance.now() / 1000;
  state.history.push({ t: now, p: st.nn && st.nn.probs ? st.nn.p_lead : 0, targ: !!v.targeted, parry: newParry });
  while (state.history.length && state.history[0].t < now - 6.5) state.history.shift();
  if (state.tab === "control") { drawProb(st); drawTimeline(); }

  renderEvents($("#eventLog"), st.events || [], "lastEventKey", "Nothing yet - turn the bot on, or open the practice arena to watch the network play.");
  setAlerts(st);

  // vision readout
  if (state.tab === "vision") {
    const parts = [];
    parts.push(`targeted: ${v.targeted ? "YES" : "no"} (red in box ${(100 * (v.gate_frac || 0)).toFixed(1)}%, need ${(100 * (state.values.gate_min_frac || 0)).toFixed(1)}%)`);
    parts.push(v.ball ? `ball: x ${v.ball[0].toFixed(3)}, y ${v.ball[1].toFixed(3)}, r ${v.ball[2].toFixed(3)}` : "ball: none");
    parts.push(`red pixels: ${(100 * (v.red_frac || 0)).toFixed(2)}%, blobs: ${v.blobs || 0}`);
    if (st.region) parts.push(`capture: ${st.region.width}x${st.region.height} at (${st.region.left}, ${st.region.top}), monitors: ${st.monitors}`);
    if (st.source === "arena") parts.push("(showing the practice arena - switch the source to Roblox on the Control tab to calibrate for the game)");
    $("#visionReadout").textContent = parts.join("  |  ");
  }

  // arena
  const a = st.arena;
  if (a) {
    $("#scBlocks").textContent = a.stats.blocks;
    $("#scHits").textContent = a.stats.hits;
    $("#scWhiffs").textContent = a.stats.whiffs;
    $("#scStreak").textContent = a.stats.streak;
    $("#scBest").textContent = a.stats.best_streak;
    $("#scSpeed").textContent = Math.round(a.speed);
    const ov = $("#arenaOverlay");
    ov.innerHTML = "";
    const phaseText = { countdown: "Get ready...", incoming: "Ball incoming!", away: "Ball going to another player", eliminated: "Eliminated" }[a.phase] || a.phase;
    ov.append(el("span", "", phaseText));
    if (a.shield > 0) ov.append(el("span", "shield", `SHIELD ${Math.round(a.shield * 1000)} ms`));
    if (a.cooldown > 0) ov.append(el("span", "cool", `COOLDOWN ${a.cooldown.toFixed(1)} s`));
    const fb = $("#arenaFeedback");
    if (a.feedback && st.arena_mode === "human") {
      fb.textContent = a.feedback.text;
      fb.className = "feedback " + a.feedback.result;
    } else if (st.arena_mode === "human") {
      fb.textContent = "Press Space (or the BLOCK button) just before the red ball reaches you.";
      fb.className = "feedback";
    } else {
      fb.textContent = "The neural network is blocking. Switch to 'I play' to practise your own timing.";
      fb.className = "feedback";
    }
    renderEvents($("#arenaLog"), a.events || [], "lastArenaKey", "Blocks, hits and early presses will show up here.");
  }
  const human = st.arena_mode === "human";
  $("#blockBtn").style.display = human ? "" : "none";
  let ahint = "";
  if (st.source !== "arena") ahint = "The source is Roblox (screen). Switch it to the practice arena on the Control tab.";
  else if (!human && !on) ahint = "The bot is off - turn it on (Control tab or the hotkey) to let the network play.";
  $("#arenaHint").textContent = ahint;

  // recording + about
  const rec = st.recording || {};
  const rb = $("#recordBtn");
  rb.textContent = rec.active ? "Stop recording" : "Start recording";
  rb.classList.toggle("recording", !!rec.active);
  $("#recordInfo").textContent = rec.active ? `${rec.path} - ${rec.frames} frames` : `${rec.files || 0} recording(s) saved`;
  $("#ackState").textContent = st.practice_ack ? "You accepted the practice-only notice." : "You have not accepted the practice-only notice yet (needed before using the bot on Roblox).";
  $("#withdrawAck").style.display = st.practice_ack ? "" : "none";
  $("#aboutVersion").textContent = `BladeBot ${st.version}`;
}

async function poll() {
  try {
    const st = await api("api/state");
    state.status = st;
    state.connected = true;
    render(st);
  } catch (e) {
    state.connected = false;
    setAlerts(state.status);
  } finally {
    setTimeout(poll, document.hidden ? 1000 : 100);
  }
}

// ------------------------------------------------------------------ training + model
function pct(v) { return v === null || v === undefined ? "-" : (100 * v).toFixed(1) + "%"; }

async function refreshModel() {
  try {
    const m = await api("api/model");
    const box = $("#modelInfo");
    if (!m.loaded) { box.textContent = m.error || "No model loaded."; return; }
    const met = m.metrics || {};
    const tr = m.trained_on || {};
    box.innerHTML = "";
    const lines = [
      `file        ${m.path || "-"}`,
      `layers      ${m.layers.join(" -> ")}  (${m.activation}, ${m.parameters.toLocaleString()} weights)`,
      `outputs     P(impact within ${m.horizons[0].toFixed(2)} ... ${m.horizons[m.horizons.length - 1].toFixed(2)} s), ${m.horizons.length} time horizons`,
      `trained     ${(m.created || "-").replace("T", " ")} on ${tr.approaches ? tr.approaches.toLocaleString() + " simulated approaches" : "-"}${tr.recording_frames ? " + " + tr.recording_frames.toLocaleString() + " recorded frames" : ""}`,
      `sim test    network ${pct(met.nn_success)} vs classic looming ${pct(met.heuristic_success)} parry success`,
      `ETA error   ${met.eta_mae_s ? Math.round(met.eta_mae_s * 1000) + " ms" : "-"} (mean, last 0.8 s before impact)`,
    ];
    box.append(el("pre", "pre", lines.join("\n")));
  } catch (e) { toast(e.message, true); }
}

async function refreshRecordings() {
  try {
    const r = await api("api/recordings");
    const list = $("#recList");
    list.innerHTML = "";
    if (!r.files.length) list.append(el("li", "", "No recordings yet."));
    for (const f of r.files.slice(-20).reverse()) list.append(el("li", "", `${f.name}  ${f.kb} KB  ${f.modified}`));
  } catch (_) { /* ignore */ }
}

function showTrain(s) {
  $("#trainBar").style.width = Math.round((s.progress || 0) * 100) + "%";
  $("#trainMsg").textContent = s.state === "idle" ? "" : `${s.state}: ${s.message || ""}${s.error ? " - " + s.error : ""}`;
  if (s.log) $("#trainLog").textContent = s.log.join("\n");
  const running = s.state === "running";
  $("#trainStart").disabled = running;
  $("#trainCancel").disabled = !running;
}

async function refreshTrain() {
  try {
    const s = await api("api/train/status");
    showTrain(s);
    clearTimeout(state.trainTimer);
    if (s.state === "running") state.trainTimer = setTimeout(refreshTrain, 500);
    else if (refreshTrain.wasRunning) {
      refreshModel();
      if (s.state === "done") toast("Training finished - the new network is now active");
    }
    refreshTrain.wasRunning = s.state === "running";
  } catch (_) { /* ignore */ }
}

$("#trainForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  const body = Object.fromEntries(f.entries());
  body.use_recordings = f.get("use_recordings") === "on";
  try {
    const r = await api("api/train", body);
    showTrain(r.status);
    refreshTrain.wasRunning = true;
    setTimeout(refreshTrain, 400);
  } catch (e) { toast(e.message, true); }
});
$("#trainCancel").addEventListener("click", async () => {
  try { await api("api/train/cancel", {}); } catch (e) { toast(e.message, true); }
});
$("#useDefaultModel").addEventListener("click", async () => {
  try { await api("api/model/use_default", {}); toast("Bundled model loaded"); refreshModel(); } catch (e) { toast(e.message, true); }
});
$("#reloadModel").addEventListener("click", async () => {
  try { await api("api/model/reload", {}); toast("Model reloaded"); refreshModel(); } catch (e) { toast(e.message, true); }
});
$("#recordBtn").addEventListener("click", async () => {
  const active = state.status && state.status.recording && state.status.recording.active;
  try {
    await api("api/record", { on: !active });
    if (active) setTimeout(refreshRecordings, 300);
    toast(active ? "Recording saved" : "Recording started");
  } catch (e) { toast(e.message, true); }
});

// ------------------------------------------------------------------ start
async function init() {
  try {
    const sch = await api("api/schema");
    state.schema = sch;
    state.values = sch.values;
    buildForms();
  } catch (e) {
    toast("Could not load settings: " + e.message, true);
  }
  setTab((location.hash || "#control").slice(1));
  poll();
}
init();
