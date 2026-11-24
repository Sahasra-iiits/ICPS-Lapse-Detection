// Runs in the extension's hidden offscreen page: opens the camera, runs MediaPipe
// Face Landmarker and the detection logic (lapse_core.js), and sends the live
// state + alerts to background.js. Nothing leaves this computer.
import { FaceLandmarker, FilesetResolver } from "./lib/vision_bundle.mjs";

const core = globalThis.LapseCore;
const SKIN_W = 320;                 // frames are downscaled to this width for the skin-colour (heart rate) signal

let landmarkerP = null, modelP = null;
let session = null;                 // {stream, reader, monitor, t0, lastT}
let sendTimer = null;
const canvas = new OffscreenCanvas(SKIN_W, 180);
const ctx = canvas.getContext("2d", { willReadFrequently: true });

function loadLandmarker() {
  landmarkerP ??= (async () => {
    const fileset = await FilesetResolver.forVisionTasks(chrome.runtime.getURL("lib/wasm"));
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: chrome.runtime.getURL("model/face_landmarker.task"), delegate },
      runningMode: "VIDEO", numFaces: 1,
      outputFaceBlendshapes: false, outputFacialTransformationMatrixes: true,
    });
    try { return await FaceLandmarker.createFromOptions(fileset, opts("GPU")); }
    catch (e) { console.warn("GPU unavailable, using CPU:", e); return FaceLandmarker.createFromOptions(fileset, opts("CPU")); }
  })();
  return landmarkerP;
}
function loadModel() {
  modelP ??= fetch(chrome.runtime.getURL("model/zoning_model.json"))
    .then((r) => r.json()).then((j) => new core.ZoningModel(j))
    .catch((e) => { console.warn("No zoning model:", e); return null; });
  return modelP;
}

const send = (msg) => chrome.runtime.sendMessage(msg).catch(() => {});
function sendState() {
  if (session && session.monitor) send({ type: "det-state", state: session.monitor.status });
}
function fail(code, message) {
  stop();
  send({ type: "det-error", error: { code, message } });
}

async function openCamera() {
  try {
    return await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 360 }, frameRate: { ideal: 30 } }, audio: false,
    });
  } catch (e) {
    const name = e && e.name;
    if (name === "NotAllowedError" || name === "SecurityError")
      throw { code: "camera-permission", message: "Camera access needed" };
    if (name === "NotFoundError" || name === "OverconstrainedError")
      throw { code: "runtime", message: "No camera found" };
    if (name === "NotReadableError")
      throw { code: "runtime", message: "The camera is being used by another app" };
    throw { code: "runtime", message: String((e && e.message) || e) };
  }
}

function processFrame(frame, s, landmarker) {
  const W = frame.displayWidth, H = frame.displayHeight;
  // camera capture time (µs), not processing time: processing delays would distort the pulse signal
  if (s.ts0 == null) s.ts0 = frame.timestamp;
  let tMs = Number.isFinite(frame.timestamp) ? Math.round((frame.timestamp - s.ts0) / 1000) : Math.round(performance.now() - s.t0);
  if (tMs <= s.lastT) tMs = s.lastT + 1;               // MediaPipe needs increasing timestamps
  s.lastT = tMs;
  const res = landmarker.detectForVideo(frame, tMs);
  const lm = res.faceLandmarks && res.faceLandmarks[0];
  const f = core.faceFeatures(lm, res.facialTransformationMatrixes && res.facialTransformationMatrixes[0], W, H);
  let rgb = null;
  if (lm) {
    const cw = SKIN_W, ch = Math.max(1, Math.round((SKIN_W * H) / W));
    if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
    const box = core.skinRoiBox(lm, cw, ch);
    if (box) {
      ctx.drawImage(frame, 0, 0, cw, ch);
      const img = ctx.getImageData(box.x, box.y, box.w, box.h);
      rgb = core.skinRGB(lm, cw, ch, { width: box.w, height: box.h, data: img.data, x0: box.x, y0: box.y });
    }
  }
  s.monitor.frame(tMs, f, rgb);
}

async function start(msg) {
  if (session) { applySettings(msg.settings); return; }
  const token = {};
  session = token;                                       // claim; stop() clears it
  try {
    const [stream, landmarker, model] = await Promise.all([openCamera(), loadLandmarker(), loadModel()]);
    if (session !== token) { stream.getTracks().forEach((t) => t.stop()); return; }
    const monitor = new core.Monitor(model, msg.calib_s || 20);
    Object.assign(token, { stream, monitor, t0: performance.now(), lastT: -1 });
    applySettings(msg.settings);
    clearInterval(sendTimer);
    sendTimer = setInterval(sendState, 500);
    sendState();
    const track = stream.getVideoTracks()[0];
    track.addEventListener("ended", () => session === token && fail("runtime", "The camera was disconnected"));
    const reader = new MediaStreamTrackProcessor({ track }).readable.getReader();
    token.reader = reader;
    while (session === token) {
      const { value: frame, done } = await reader.read();
      if (done) break;
      try { if (session === token) processFrame(frame, token, landmarker); }
      catch (e) { console.error(e); }
      finally { frame.close(); }
    }
  } catch (e) {
    if (session === token) fail(e.code || "runtime", e.message || String(e));
  }
}

function applySettings(st) {
  if (session && session.monitor && st) session.monitor.setSettings(st.sensitivity, st.enabled);
}

function stop() {
  const s = session;
  session = null;
  clearInterval(sendTimer);
  if (!s) return;
  if (s.reader) s.reader.cancel().catch(() => {});
  if (s.stream) s.stream.getTracks().forEach((t) => t.stop());   // camera light goes off
}

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  if (msg.target !== "detector") return;
  if (msg.cmd === "start") start(msg);
  else if (msg.cmd === "stop") { stop(); send({ type: "det-state", state: null }); }
  else if (msg.cmd === "settings") applySettings(msg.settings);
  else if (msg.cmd === "ping") { reply({ ok: true, running: !!session }); return; }
  reply({ ok: true });
});

// Load MediaPipe and the model straight away, so switching on is quick.
loadLandmarker().catch((e) => console.error(e));
loadModel();
send({ type: "det-ready" });
