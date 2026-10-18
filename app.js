import { AudioClassifier, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@0.10.14/audio_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite";

// YAMNet (AudioSet) のクラス名 → 反応グループ
const GROUPS = [
  { id: "voice", label: "🗣 人の声・話しかけ", level: "soft", cooldown: 20, on: true,
    classes: ["Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue"] },
  { id: "shout", label: "📢 叫び声・大声", level: "strong", cooldown: 8, on: true,
    classes: ["Shout", "Yell", "Screaming", "Children shouting", "Bellow"] },
  { id: "door", label: "🚪 インターホン・ノック", level: "strong", cooldown: 8, on: true,
    classes: ["Doorbell", "Ding-dong", "Knock"] },
  { id: "phone", label: "📞 電話・着信音", level: "strong", cooldown: 10, on: true,
    classes: ["Telephone", "Telephone bell ringing", "Ringtone"] },
  { id: "alarm", label: "🚨 アラーム・警報・サイレン", level: "strong", cooldown: 8, on: true,
    classes: ["Alarm", "Alarm clock", "Smoke detector, smoke alarm", "Fire alarm", "Siren",
      "Civil defense siren", "Buzzer", "Car alarm", "Emergency vehicle", "Police car (siren)",
      "Ambulance (siren)", "Fire engine, fire truck (siren)"] },
  { id: "beep", label: "⏲ 家電のピピッ音", level: "soft", cooldown: 10, on: true,
    classes: ["Beep, bleep", "Microwave oven", "Ding", "Chime"] },
  { id: "baby", label: "👶 赤ちゃん・泣き声", level: "strong", cooldown: 10, on: true,
    classes: ["Baby cry, infant cry", "Crying, sobbing", "Whimper"] },
  { id: "animal", label: "🐶 犬・猫の鳴き声", level: "soft", cooldown: 15, on: false,
    classes: ["Dog", "Bark", "Howl", "Cat", "Meow"] },
  { id: "crash", label: "💥 割れる・ぶつかる音", level: "strong", cooldown: 8, on: true,
    classes: ["Glass", "Shatter", "Smash, crash", "Breaking", "Slam"] },
  { id: "loud", label: "🔊 急な大きい音（種類問わず）", level: "soft", cooldown: 8, on: true, classes: [] },
];
const classToGroup = new Map();
for (const g of GROUPS) for (const c of g.classes) classToGroup.set(c, g);

const JA = {
  "Speech": "話し声", "Child speech, kid speaking": "子どもの声", "Conversation": "会話", "Narration, monologue": "話し声",
  "Shout": "叫び声", "Yell": "大声", "Screaming": "悲鳴", "Children shouting": "子どもの叫び声", "Bellow": "怒鳴り声",
  "Doorbell": "インターホン", "Ding-dong": "ピンポン", "Knock": "ノック",
  "Telephone": "電話", "Telephone bell ringing": "電話のベル", "Ringtone": "着信音",
  "Alarm": "アラーム", "Alarm clock": "目覚まし", "Smoke detector, smoke alarm": "火災報知器", "Fire alarm": "火災警報",
  "Siren": "サイレン", "Civil defense siren": "警報サイレン", "Buzzer": "ブザー", "Car alarm": "車のアラーム",
  "Emergency vehicle": "緊急車両", "Police car (siren)": "パトカー", "Ambulance (siren)": "救急車", "Fire engine, fire truck (siren)": "消防車",
  "Beep, bleep": "ピー音", "Microwave oven": "電子レンジ", "Ding": "チン", "Chime": "チャイム",
  "Baby cry, infant cry": "赤ちゃんの泣き声", "Crying, sobbing": "泣き声", "Whimper": "すすり泣き",
  "Dog": "犬", "Bark": "犬の吠え声", "Howl": "遠吠え", "Cat": "猫", "Meow": "猫の鳴き声",
  "Glass": "ガラス", "Shatter": "割れる音", "Smash, crash": "衝突音", "Breaking": "壊れる音", "Slam": "ドンという音",
  "Music": "音楽", "Silence": "静か", "Inside, small room": "室内", "Typing": "タイピング",
  "Computer keyboard": "キーボード", "Vehicle": "乗り物", "White noise": "ノイズ",
};
const ja = (n) => JA[n] || n;

// ---------- 設定 ----------
const DEFAULTS = {
  groups: Object.fromEntries(GROUPS.map((g) => [g.id, g.on])),
  keywords: "すみません,すいません,ねえ,ちょっと,おーい,もしもし",
  score: 0.35, loud: 20, vol: 0.6, useNotify: true, useSpeech: true,
};
let settings = structuredClone(DEFAULTS);
try {
  const saved = JSON.parse(localStorage.getItem("mimi-alert") || "null");
  if (saved) settings = { ...settings, ...saved, groups: { ...settings.groups, ...saved.groups } };
} catch {}
const save = () => { try { localStorage.setItem("mimi-alert", JSON.stringify(settings)); } catch {} };

const $ = (id) => document.getElementById(id);

function buildSettingsUI() {
  const box = $("groups");
  for (const g of GROUPS) {
    const l = document.createElement("label");
    l.className = "check";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !!settings.groups[g.id];
    cb.onchange = () => { settings.groups[g.id] = cb.checked; save(); };
    l.append(cb, g.label);
    box.append(l);
  }
  $("keywords").value = settings.keywords;
  $("keywords").oninput = (e) => { settings.keywords = e.target.value; save(); };
  const bindRange = (id, fmt) => {
    const el = $(id), val = $(id + "Val");
    el.value = settings[id];
    val.textContent = fmt(settings[id]);
    el.oninput = () => { settings[id] = +el.value; val.textContent = fmt(+el.value); save(); };
  };
  bindRange("score", (v) => v.toFixed(2));
  bindRange("loud", (v) => v);
  bindRange("vol", (v) => Math.round(v * 100) + "%");
  for (const id of ["useNotify", "useSpeech"]) {
    $(id).checked = settings[id];
    $(id).onchange = () => {
      settings[id] = $(id).checked; save();
      if (id === "useNotify" && settings.useNotify) askNotify();
      if (id === "useSpeech" && running) settings.useSpeech ? startRecognition() : stopRecognition();
    };
  }
  $("testStrong").onclick = () => alertUser("テスト（強）", "strong");
  $("testSoft").onclick = () => alertUser("テスト（弱）", "soft");
}

// ---------- アラート ----------
let ctx = null;
const getCtx = () => (ctx ??= new AudioContext());

function beep(freqs, step = 0.16) {
  const c = getCtx();
  const out = c.createGain();
  out.gain.value = settings.vol;
  out.connect(c.destination);
  let t = c.currentTime + 0.02;
  for (const f of freqs) {
    const o = c.createOscillator(), g = c.createGain();
    o.type = "sine";
    o.frequency.value = f;
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(1, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.001, t + step * 1.6);
    o.connect(g).connect(out);
    o.start(t);
    o.stop(t + step * 1.7);
    t += step;
  }
}

function flash(color) {
  const f = $("flash");
  f.style.background = color;
  f.classList.add("on");
  setTimeout(() => f.classList.remove("on"), 250);
}

function askNotify() {
  if ("Notification" in window && Notification.permission === "default") Notification.requestPermission();
}

function alertUser(text, level) {
  if (level === "strong") {
    beep([880, 1320, 880, 1320, 1760], 0.14);
    flash("#ff5d5d");
    navigator.vibrate?.([200, 100, 200, 100, 300]);
  } else {
    beep([1046, 1568], 0.18);
    flash("#ffb547");
    navigator.vibrate?.(150);
  }
  logEvent(text, level);
  if (settings.useNotify && "Notification" in window && Notification.permission === "granted" && document.hidden) {
    try { new Notification("👂 みみアラート", { body: text, tag: "mimi-alert", renotify: true }); } catch {}
  }
}

const lastFired = {};
function trigger(group, text) {
  if (!settings.groups[group.id]) return;
  const now = performance.now();
  if (now - (lastFired[group.id] ?? -1e9) < group.cooldown * 1000) return;
  lastFired[group.id] = now;
  alertUser(text, group.level);
}

// ---------- ログ ----------
const hhmmss = () => new Date().toLocaleTimeString("ja-JP", { hour12: false });
function addLog(listId, html, cls = "") {
  const ul = $(listId);
  ul.querySelector(".empty")?.remove();
  const li = document.createElement("li");
  li.innerHTML = `<time>${hhmmss()}</time><span class="${cls}">${html}</span>`;
  ul.prepend(li);
  while (ul.children.length > 100) ul.lastChild.remove();
}
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const logEvent = (text, level) => addLog("events", esc(text), level);
for (const id of ["events", "transcripts"]) $(id).innerHTML = `<li class="empty">まだありません</li>`;

// ---------- 音声認識（文字起こし） ----------
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null, recWanted = false;

function keywordList() {
  return settings.keywords.split(/[,、，]/).map((s) => s.trim()).filter(Boolean);
}

function startRecognition() {
  if (!SR) { $("interim").textContent = "このブラウザは文字起こしに未対応です（Chrome / Edge / Safari 推奨）"; return; }
  recWanted = true;
  if (rec) return;
  rec = new SR();
  rec.lang = "ja-JP";
  rec.continuous = true;
  rec.interimResults = true;
  rec.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i];
      const text = r[0].transcript.trim();
      if (!text) continue;
      if (r.isFinal) onFinalText(text);
      else interim += text;
    }
    const el = $("interim");
    if (interim) { el.textContent = interim; el.classList.add("live"); checkKeywords(interim); }
  };
  rec.onerror = (e) => {
    if (e.error === "not-allowed" || e.error === "service-not-allowed") {
      recWanted = false;
      $("interim").textContent = "音声認識が許可されていません";
    }
  };
  rec.onend = () => {
    rec = null;
    if (recWanted && running) setTimeout(startRecognition, 300);
  };
  try { rec.start(); } catch { rec = null; }
}

function stopRecognition() {
  recWanted = false;
  try { rec?.stop(); } catch {}
  rec = null;
}

let lastKeywordHit = 0;
function checkKeywords(text) {
  const hits = keywordList().filter((k) => text.includes(k));
  if (!hits.length) return null;
  const now = performance.now();
  if (now - lastKeywordHit > 6000) {
    lastKeywordHit = now;
    alertUser(`呼ばれたかも：「${text}」`, "strong");
  }
  return hits;
}

function onFinalText(text) {
  const el = $("interim");
  el.textContent = text;
  el.classList.remove("live");
  const hits = checkKeywords(text) || [];
  let html = esc(text);
  for (const k of hits) html = html.split(esc(k)).join(`<span class="hit">${esc(k)}</span>`);
  addLog("transcripts", html, hits.length ? "strong" : "");
}

// ---------- マイク + 音の分類 ----------
let classifier = null, stream = null, running = false, wakeLock = null;
let ring, ringPos = 0, sinceClassify = 0, floorDb = -60, sampleRate = 48000;

const WORKLET = `
class Tap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(2048); this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = ch[i];
      if (this.n === this.buf.length) { this.port.postMessage(this.buf.slice()); this.n = 0; }
    }
    return true;
  }
}
registerProcessor("tap", Tap);`;

async function loadClassifier() {
  if (classifier) return;
  const fileset = await FilesetResolver.forAudioTasks(WASM_URL);
  classifier = await AudioClassifier.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL },
    maxResults: 5,
    scoreThreshold: 0.1,
  });
}

function setStatus(text, cls) {
  const s = $("status");
  s.textContent = text;
  s.className = "status " + cls;
}

async function start() {
  const btn = $("toggle");
  btn.disabled = true;
  try {
    setStatus("準備中…", "loading");
    const c = getCtx();
    await c.resume();
    askNotify();
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false },
    });
    setStatus("音声モデル読み込み中…", "loading");
    await loadClassifier();

    sampleRate = c.sampleRate;
    ring = new Float32Array(Math.round(sampleRate * 0.975));
    ringPos = 0; sinceClassify = 0; floorDb = -60;

    const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
    await c.audioWorklet.addModule(url).catch(() => {}); // 2回目以降は登録済み
    const src = c.createMediaStreamSource(stream);
    const tap = new AudioWorkletNode(c, "tap");
    const mute = c.createGain();
    mute.gain.value = 0;
    src.connect(tap).connect(mute).connect(c.destination);
    tap.port.onmessage = (e) => onChunk(e.data);
    stream._nodes = [src, tap, mute];

    running = true;
    if (settings.useSpeech) startRecognition();
    await requestWakeLock();
    setStatus("聞き取り中", "on");
    btn.textContent = "停止";
    btn.classList.add("stop");
  } catch (err) {
    console.error(err);
    stop();
    setStatus(err.name === "NotAllowedError" ? "マイクが許可されていません" : "エラー: " + err.message, "error");
  } finally {
    btn.disabled = false;
  }
}

function stop() {
  running = false;
  stopRecognition();
  if (stream) {
    stream._nodes?.forEach((n) => n.disconnect());
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  wakeLock?.release?.().catch(() => {});
  wakeLock = null;
  setStatus("停止中", "idle");
  const btn = $("toggle");
  btn.textContent = "聞き取り開始";
  btn.classList.remove("stop");
  $("meterBar").style.width = "0";
  $("dbNow").textContent = "-- dB";
}

async function requestWakeLock() {
  try {
    wakeLock = await navigator.wakeLock?.request("screen");
    wakeLock?.addEventListener("release", () => { wakeLock = null; });
  } catch {}
}

const dbToPct = (db) => Math.max(0, Math.min(100, ((db + 80) / 80) * 100));

function onChunk(chunk) {
  if (!running) return;
  // 音量（dBFS）と周囲ノイズの基準
  let sum = 0;
  for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i];
  const db = 10 * Math.log10(sum / chunk.length + 1e-12);
  const prevFloor = floorDb;
  floorDb = db < floorDb ? floorDb + (db - floorDb) * 0.3 : floorDb + (db - floorDb) * 0.01;
  if (db > prevFloor + settings.loud && db > -50) {
    trigger(GROUPS.find((g) => g.id === "loud"), `急な大きい音（+${Math.round(db - prevFloor)} dB）`);
  }
  $("meterBar").style.width = dbToPct(db) + "%";
  $("floorMark").style.left = dbToPct(floorDb) + "%";
  $("dbNow").textContent = `${Math.round(db)} dB（周囲 ${Math.round(floorDb)} dB）`;

  // リングバッファに貯めて約0.5秒ごとに分類
  for (let i = 0; i < chunk.length; i++) {
    ring[ringPos] = chunk[i];
    ringPos = (ringPos + 1) % ring.length;
  }
  sinceClassify += chunk.length;
  if (sinceClassify >= sampleRate * 0.5) {
    sinceClassify = 0;
    classify();
  }
}

function classify() {
  const buf = new Float32Array(ring.length);
  buf.set(ring.subarray(ringPos));
  buf.set(ring.subarray(0, ringPos), ring.length - ringPos);
  let results;
  try { results = classifier.classify(buf, sampleRate); } catch (e) { console.error(e); return; }
  const cats = results?.[0]?.classifications?.[0]?.categories ?? [];
  if (cats[0]) $("topClass").textContent = `${ja(cats[0].categoryName)} ${Math.round(cats[0].score * 100)}%`;
  for (const cat of cats) {
    const g = classToGroup.get(cat.categoryName);
    if (g && cat.score >= settings.score) {
      trigger(g, `${ja(cat.categoryName)}（${Math.round(cat.score * 100)}%）`);
    }
  }
}

$("toggle").onclick = () => (running ? stop() : start());
document.addEventListener("visibilitychange", async () => {
  if (running && !document.hidden && !wakeLock) await requestWakeLock();
});
buildSettingsUI();
