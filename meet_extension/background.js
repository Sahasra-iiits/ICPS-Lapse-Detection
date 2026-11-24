// Starts / stops the in-browser detector (offscreen.html + detector.js) and shares
// its live state with the Meet page and the popup.
const DEFAULTS = {
  enabled: { dozing: true, drowsy: true, zoning: true, stress: true },
  sensitivity: "normal", muted: false, showPill: true, autoStart: false, calibS: 20,
};
const OFFSCREEN = "offscreen.html";

let on = false;            // monitoring on
let detState = null;       // last state from the detector
let error = null;          // {code: "camera-permission" | "runtime", message}
let pendingStart = false;  // start again once the camera is allowed
let cameraTab = null, returnTab = null;
let lastAlertId = 0, alertUntil = 0;
const callTabs = new Set();

async function getSettings() {
  const s = await chrome.storage.sync.get(DEFAULTS);
  return { ...DEFAULTS, ...s, enabled: { ...DEFAULTS.enabled, ...(s.enabled || {}) } };
}
const enabledList = (s) => Object.keys(s.enabled).filter((k) => s.enabled[k]);
const detSettings = (s) => ({ sensitivity: s.sensitivity, enabled: enabledList(s) });

// ---------- the hidden detector page ----------
let creating = null;
async function ensureOffscreen() {
  if (await chrome.offscreen.hasDocument()) return;
  creating ??= chrome.offscreen.createDocument({
    url: OFFSCREEN, reasons: ["USER_MEDIA"],
    justification: "Reads the webcam on this computer to detect dozing off and lost focus during Google Meet calls.",
  }).finally(() => { creating = null; });
  await creating;
}
async function closeOffscreen() {
  try { if (await chrome.offscreen.hasDocument()) await chrome.offscreen.closeDocument(); } catch (e) {}
}
async function toDetector(msg, tries = 40) {
  for (let i = 0; ; i++) {
    try { return await chrome.runtime.sendMessage({ target: "detector", ...msg }); }
    catch (e) {
      if (i >= tries || !/Receiving end does not exist/i.test(String(e))) throw e;
      await new Promise((r) => setTimeout(r, 100));       // page still loading
    }
  }
}

async function start() {
  const s = await getSettings();
  error = null; on = true; detState = null; pendingStart = false;
  updateBadge();
  try {
    await ensureOffscreen();
    await toDetector({ cmd: "start", settings: detSettings(s), calib_s: s.calibS });
  } catch (e) {
    on = false;
    error = { code: "runtime", message: String((e && e.message) || e) };
  }
  updateBadge();
}

function stop() {
  on = false; detState = null; pendingStart = false;
  toDetector({ cmd: "stop" }, 0).catch(() => {});          // camera released
  updateBadge();
}

async function openCameraTab() {
  if (cameraTab != null) {
    try { await chrome.tabs.update(cameraTab, { active: true }); return; } catch (e) { cameraTab = null; }
  }
  const t = await chrome.tabs.create({ url: "camera.html", active: true });
  cameraTab = t.id;
}

chrome.storage.onChanged.addListener(async () => {
  if (on) toDetector({ cmd: "settings", settings: detSettings(await getSettings()) }, 0).catch(() => {});
});

// ---------- state for the Meet page and the popup ----------
async function view() {
  const settings = await getSettings();
  const state = on ? detState || { running: true, started: false, calibrating: true, alerts: [] } : null;
  if (state && state.alerts && state.alerts.length) {
    const newest = state.alerts[state.alerts.length - 1].id;
    if (newest > lastAlertId) { if (lastAlertId) alertUntil = Date.now() + 8000; lastAlertId = newest; }
  }
  return { settings, on, state, error };
}

function updateBadge() {
  let text = "", color = "#b3001b";
  if (on && Date.now() < alertUntil) text = "!";
  else if (on && (!detState || detState.calibrating)) { text = "…"; color = "#e8858f"; }
  else if (on) text = "on";
  else if (error) { text = "!"; color = "#80868b"; }
  chrome.action.setBadgeText({ text });
  chrome.action.setBadgeBackgroundColor({ color });
}

chrome.tabs.onRemoved.addListener((tabId) => {
  if (tabId === cameraTab) cameraTab = null;
  if (callTabs.delete(tabId) && callTabs.size === 0) { if (on) stop(); closeOffscreen(); }   // last Meet call closed
});

chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  if (msg.target === "detector") return;                   // meant for the offscreen page
  (async () => {
    switch (msg.type) {
      case "det-state":
        if (on && msg.state) detState = msg.state;
        updateBadge();
        return reply({});
      case "det-error":
        on = false; detState = null; error = msg.error;
        if (error.code === "camera-permission") { pendingStart = true; await openCameraTab(); }
        updateBadge();
        return reply({});
      case "det-ready":
        return reply({});
      case "camera-granted":
        if (sender.tab) chrome.tabs.remove(sender.tab.id).catch(() => {});
        cameraTab = null;
        if (returnTab != null) chrome.tabs.update(returnTab, { active: true }).catch(() => {});
        error = null;
        if (pendingStart) await start();
        return reply({});
      case "open-camera":
        await openCameraTab();
        return reply({});
      case "view": { const v = await view(); updateBadge(); return reply(v); }
      case "toggle":
        if (sender.tab) returnTab = sender.tab.id;
        if (msg.on) await start(); else stop();
        return reply(await view());
      case "in-call": {
        if (sender.tab) { callTabs.add(sender.tab.id); returnTab = sender.tab.id; }
        ensureOffscreen().catch(() => {});                  // load MediaPipe now, so switching on is quick
        const s = await getSettings();
        if (s.autoStart && !on && !(error && error.code === "camera-permission")) await start();
        return reply(await view());
      }
      case "left-call":
        if (sender.tab) callTabs.delete(sender.tab.id);
        if (callTabs.size === 0) { if (on) stop(); closeOffscreen(); }
        return reply(await view());
      default:
        return reply({});
    }
  })();
  return true;
});

updateBadge();
