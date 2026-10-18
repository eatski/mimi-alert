import { AudioClassifier, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@0.10.14/audio_bundle.mjs";
import { AutoProcessor, AutoModelForAudioClassification } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1";

// 1段目：YAMNet（軽量）で常時監視して「候補」を拾う
const YAMNET_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@0.10.14/wasm";
const YAMNET_MODEL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite";
// 2段目：AST（高精度）で候補の音を再判定して確定する
const AST_MODEL = "Xenova/ast-finetuned-audioset-10-10-0.4593";

const HOP_SEC = 0.15;         // YAMNet を回す間隔
const CANDIDATE_SCORE = 0.15; // YAMNet：取りこぼさないよう低め
const INSTANT_SCORE = 0.6;    // YAMNet がここまで自信があれば AST を待たずに鳴らす
const PRE_SEC = 1.5;          // 再生時、音が始まる前に遡る秒数
const MAX_CLIP_SEC = 5;       // 再生する最大の長さ

// AudioSet のクラス名 → 知らせる音の種類
const GROUPS = [
  { label: "話しかけ・人の声", confirm: 0.5, cooldown: 20,
    classes: ["Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue"] },
  { label: "叫び声・大声", confirm: 0.3, cooldown: 8,
    classes: ["Shout", "Yell", "Screaming", "Children shouting", "Bellow"] },
  { label: "インターホン・ノック", confirm: 0.3, cooldown: 8,
    classes: ["Doorbell", "Ding-dong", "Knock"] },
  { label: "電話・着信音", confirm: 0.3, cooldown: 10,
    classes: ["Telephone", "Telephone bell ringing", "Ringtone"] },
  { label: "アラーム・警報", confirm: 0.3, cooldown: 8,
    classes: ["Alarm", "Alarm clock", "Smoke detector, smoke alarm", "Fire alarm", "Siren",
      "Civil defense siren", "Buzzer", "Car alarm", "Emergency vehicle", "Police car (siren)",
      "Ambulance (siren)", "Fire engine, fire truck (siren)"] },
  { label: "家電のピピッ音", confirm: 0.3, cooldown: 10,
    classes: ["Beep, bleep", "Microwave oven", "Ding", "Chime"] },
  { label: "泣き声", confirm: 0.3, cooldown: 10,
    classes: ["Baby cry, infant cry", "Crying, sobbing", "Whimper"] },
  { label: "割れる・ぶつかる音", confirm: 0.35, cooldown: 8,
    classes: ["Glass", "Shatter", "Smash, crash", "Breaking", "Slam"] },
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
  "Glass": "ガラス", "Shatter": "割れる音", "Smash, crash": "衝突音", "Breaking": "壊れる音", "Slam": "ドンという音",
};
const ja = (n) => JA[n] || n;

const $ = (id) => document.getElementById(id);

// ---------- ログ ----------
const hhmmss = () => new Date().toLocaleTimeString("ja-JP", { hour12: false });
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
function addLog(listId, text, cls = "") {
  const ul = $(listId);
  ul.querySelector(".empty")?.remove();
  const li = document.createElement("li");
  li.innerHTML = `<time>${hhmmss()}</time><span class="${cls}">${esc(text)}</span>`;
  ul.prepend(li);
  while (ul.children.length > 100) ul.lastChild.remove();
}
for (const id of ["events", "transcripts"]) $(id).innerHTML = `<li class="empty">まだありません</li>`;

// ---------- アラート音 + 録音した音の再生 ----------
let ctx = null;
const getCtx = () => (ctx ??= new AudioContext());

// 鳴らし終わる時刻を返す
function chime() {
  const c = getCtx();
  let t = c.currentTime + 0.02;
  for (const f of [1046, 1568]) {
    const o = c.createOscillator(), g = c.createGain();
    o.frequency.value = f;
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.6, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.3);
    o.connect(g).connect(c.destination);
    o.start(t);
    o.stop(t + 0.32);
    t += 0.18;
  }
  return t + 0.35;
}

// 小さい音でも聞こえるようにピークを揃えて再生。再生が終わる時刻を返す
function playClip(clip, at) {
  const c = getCtx();
  let peak = 0;
  for (const v of clip) peak = Math.max(peak, Math.abs(v));
  const buf = c.createBuffer(1, clip.length, sampleRate);
  buf.copyToChannel(clip, 0);
  const src = c.createBufferSource(), g = c.createGain();
  src.buffer = buf;
  g.gain.value = Math.min(0.8 / (peak || 1), 20);
  src.connect(g).connect(c.destination);
  src.start(at);
  return at + buf.duration;
}

// ---------- 音声認識（文字起こし） ----------
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null, recBlocked = false;

function startRecognition() {
  if (!SR) { $("interim").textContent = "このブラウザは文字起こしに未対応です（Chrome / Edge 推奨）"; return; }
  if (rec || recBlocked) return;
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
      if (r.isFinal) {
        addLog("transcripts", text);
        $("interim").textContent = "";
      } else interim += text;
    }
    if (interim) $("interim").textContent = interim;
  };
  rec.onerror = (e) => {
    if (e.error === "not-allowed" || e.error === "service-not-allowed") {
      recBlocked = true;
      $("interim").textContent = "音声認識が許可されていません";
    }
  };
  rec.onend = () => {
    rec = null;
    if (running) setTimeout(startRecognition, 300);
  };
  try { rec.start(); } catch { rec = null; }
}

function stopRecognition() {
  const r = rec;
  rec = null;
  try { r?.stop(); } catch {}
}

// ---------- モデル ----------
let yamnet = null, ast = null;

async function loadYamnet() {
  const fileset = await FilesetResolver.forAudioTasks(YAMNET_WASM);
  return AudioClassifier.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: YAMNET_MODEL },
    maxResults: -1,
    scoreThreshold: 0.05,
  });
}

async function loadAst() {
  const processor = await AutoProcessor.from_pretrained(AST_MODEL);
  let model;
  try {
    model = await AutoModelForAudioClassification.from_pretrained(AST_MODEL, { dtype: "fp16", device: "webgpu" });
  } catch {
    model = await AutoModelForAudioClassification.from_pretrained(AST_MODEL, { dtype: "q8", device: "wasm" });
  }
  return { processor, model, labels: model.config.id2label };
}

// AST は 16kHz 入力
async function to16k(clip) {
  const oc = new OfflineAudioContext(1, Math.ceil(clip.length * 16000 / sampleRate), 16000);
  const b = oc.createBuffer(1, clip.length, sampleRate);
  b.copyToChannel(clip, 0);
  const s = oc.createBufferSource();
  s.buffer = b;
  s.connect(oc.destination);
  s.start();
  return (await oc.startRendering()).getChannelData(0);
}

// 各クラスのスコア（マルチラベルなので sigmoid）
async function astScores(clip) {
  const inputs = await ast.processor(await to16k(clip));
  const { logits } = await ast.model(inputs);
  const scores = new Map();
  logits.data.forEach((v, i) => scores.set(ast.labels[i], 1 / (1 + Math.exp(-v))));
  return scores;
}

// ---------- マイク ----------
let stream = null, running = false, nodes = [];
let sampleRate = 48000, ring, ringPos = 0, sinceClassify = 0;
let totalSamples = 0;
let busy = false;     // 再判定・再生中は新しい候補を拾わない
const lastFired = new Map();

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
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false },
    });
    if (!yamnet || !ast) {
      setStatus("音声モデル読み込み中…", "loading");
      [yamnet, ast] = await Promise.all([yamnet ?? loadYamnet(), ast ?? loadAst()]);
    }

    sampleRate = c.sampleRate;
    ring = new Float32Array(Math.round(sampleRate * (MAX_CLIP_SEC + 1)));
    ringPos = 0; totalSamples = 0; sinceClassify = 0; busy = false;
    await astScores(new Float32Array(sampleRate)); // 初回推論は遅いので先に済ませておく

    const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
    await c.audioWorklet.addModule(url).catch(() => {}); // 2回目以降は登録済み
    const src = c.createMediaStreamSource(stream);
    const tap = new AudioWorkletNode(c, "tap");
    const mute = c.createGain();
    mute.gain.value = 0;
    src.connect(tap).connect(mute).connect(c.destination);
    tap.port.onmessage = (e) => onChunk(e.data);
    nodes = [src, tap, mute];

    running = true;
    startRecognition();
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
  nodes.forEach((n) => n.disconnect());
  nodes = [];
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  setStatus("停止中", "idle");
  const btn = $("toggle");
  btn.textContent = "聞き取り開始";
  btn.classList.remove("stop");
}

// リングバッファの直近 n サンプルを取り出す
function lastSamples(n) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = ring[(ringPos - n + i + ring.length) % ring.length];
  return out;
}

function onChunk(chunk) {
  if (!running) return;
  for (let i = 0; i < chunk.length; i++) {
    ring[ringPos] = chunk[i];
    ringPos = (ringPos + 1) % ring.length;
  }
  totalSamples += chunk.length;
  sinceClassify += chunk.length;
  if (sinceClassify >= sampleRate * HOP_SEC && !busy) {
    sinceClassify = 0;
    detectCandidate();
  }
}

// 種類ごとにクールダウン中でなく、しきい値を超えた中で一番スコアの高いもの
function pickBest(entries, threshold) {
  const now = performance.now();
  let best = null;
  for (const [name, score] of entries) {
    const g = classToGroup.get(name);
    if (!g || score < threshold(g)) continue;
    if (now - (lastFired.get(g) ?? -1e9) < g.cooldown * 1000) continue;
    if (!best || score > best.score) best = { g, name, score };
  }
  return best;
}

function detectCandidate() {
  const win = Math.round(sampleRate * 0.975);
  let results;
  try { results = yamnet.classify(lastSamples(win), sampleRate); } catch (e) { console.error(e); return; }
  const cats = results?.[0]?.classifications?.[0]?.categories ?? [];
  const best = pickBest(cats.map((c) => [c.categoryName, c.score]), () => CANDIDATE_SCORE);
  if (!best) return;
  const mark = totalSamples - win; // 音が始まったあたり
  if (best.score >= INSTANT_SCORE) return fire(best, mark); // 確実なものは再判定せず即鳴らす
  busy = true;
  verify(lastSamples(Math.round(sampleRate * PRE_SEC) + win), mark);
}

async function verify(clip, mark) {
  try {
    const best = pickBest(await astScores(clip), (g) => g.confirm);
    if (best && running) return fire(best, mark);
  } catch (e) {
    console.error(e);
  }
  busy = false;
}

// アラート音を鳴らし、続けて「音が始まる少し前〜今」までの音を再生する
function fire(best, mark) {
  busy = true;
  lastFired.set(best.g, performance.now());
  addLog("events", `${ja(best.name)}（${best.g.label}）`, "alert");
  const c = getCtx();
  const chimeEnd = chime();
  setTimeout(() => {
    if (!running) { busy = false; return; }
    const from = mark - Math.round(sampleRate * PRE_SEC);
    const n = Math.min(totalSamples - from, Math.round(sampleRate * MAX_CLIP_SEC), ring.length);
    const end = playClip(lastSamples(n), c.currentTime);
    // 再生した音を自分で拾わないよう、再生が終わるまで候補探しを止める
    setTimeout(() => { busy = false; }, (end - c.currentTime) * 1000 + 500);
  }, Math.max(0, (chimeEnd - c.currentTime) * 1000));
}

$("toggle").onclick = () => (running ? stop() : start());
