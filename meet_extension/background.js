// Starts / stops the local detector (native helper) and shares its live state
// with the Meet page and the popup.
const HOST = "com.icps.focus_alerts";
const HTTP_HELPER = "http://127.0.0.1:47813";   // optional: helper started by hand (live_monitor.py --meet)
const DEFAULTS = {
  enabled: { dozing: true, drowsy: true, zoning: true, stress: true },
  sensitivity: "normal", muted: false, showPill: true, autoStart: false,
};

let port = null;          // native messaging connection
let on = false;           // user wants monitoring on
let nativeState = null;   // last state from the helper
let error = null;         // {code: "not-installed" | "runtime", message}
let lastAlertId = 0;
let alertUntil = 0;
const callTabs = new Set();

async function getSettings() {
  const s = await chrome.storage.sync.get(DEFAULTS);
  return { ...DEFAULTS, ...s, enabled: { ...DEFAULTS.enabled, ...(s.enabled || {}) } };
}
const enabledList = (s) => Object.keys(s.enabled).filter((k) => s.enabled[k]);
const helperSettings = (s) => ({ sensitivity: s.sensitivity, enabled: enabledList(s) });

function connect() {
  try {
    port = chrome.runtime.connectNative(HOST);
  } catch (e) {
    error = { code: "not-installed", message: String(e) };
    return false;
  }
  port.onMessage.addListener((m) => {
    if (m.type === "state") nativeState = m;
    else if (m.type === "error") error = { code: "runtime", message: m.message };
    updateBadge();
  });
  port.onDisconnect.addListener(() => {
    const msg = chrome.runtime.lastError ? chrome.runtime.lastError.message : "";
    port = null;
    nativeState = null;
    if (on) {
      on = false;
      error = /not found|forbidden|not installed/i.test(msg)
        ? { code: "not-installed", message: msg }
        : { code: "runtime", message: error && error.message ? error.message : msg || "The detector stopped." };
    }
    updateBadge();
  });
  return true;
}

async function start() {
  const s = await getSettings();
  error = null;
  on = true;
  if (!port && !connect()) { on = false; updateBadge(); return; }
  port.postMessage({ cmd: "start", settings: helperSettings(s) });
  updateBadge();
}

function stop() {
  on = false;
  nativeState = null;
  if (port) { try { port.disconnect(); } catch (e) {} port = null; }   // helper exits, camera released
  updateBadge();
}

chrome.storage.onChanged.addListener(async () => {
  if (port) port.postMessage({ cmd: "settings", settings: helperSettings(await getSettings()) });
});

async function httpState(s) {
  try {
    const r = await fetch(`${HTTP_HELPER}/state?sens=${s.sensitivity}&en=${enabledList(s).join(",")}`, { cache: "no-store" });
    return r.ok ? await r.json() : null;
  } catch (e) { return null; }
}

async function view() {
  const settings = await getSettings();
  let state = null, mode = "off";
  if (on) { mode = "native"; state = nativeState || { running: true, started: false, calibrating: true, alerts: [] }; }
  else {
    const h = await httpState(settings);
    if (h && h.running) { mode = "manual"; state = h; }
  }
  if (state && state.alerts && state.alerts.length) {
    const newest = state.alerts[state.alerts.length - 1].id;
    if (newest > lastAlertId) { if (lastAlertId) alertUntil = Date.now() + 8000; lastAlertId = newest; }
  }
  return { settings, on: on || mode === "manual", mode, state, error };
}

function updateBadge() {
  let text = "", color = "#b3001b";
  if (on && Date.now() < alertUntil) text = "!";
  else if (on && nativeState && nativeState.calibrating) { text = "…"; color = "#e8858f"; }
  else if (on) text = "on";
  else if (error) { text = "!"; color = "#80868b"; }
  chrome.action.setBadgeText({ text });
  chrome.action.setBadgeBackgroundColor({ color });
}

chrome.tabs.onRemoved.addListener((tabId) => {
  callTabs.delete(tabId);
  if (on && callTabs.size === 0) stop();          // last Meet call closed -> stop the camera
});

chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  (async () => {
    if (msg.type === "view") { const v = await view(); updateBadge(); return reply(v); }
    if (msg.type === "toggle") { (msg.on ? await start() : stop()); return reply(await view()); }
    if (msg.type === "in-call") {
      if (sender.tab) callTabs.add(sender.tab.id);
      const s = await getSettings();
      if (s.autoStart && !on && !(error && error.code === "not-installed")) await start();
      return reply(await view());
    }
    if (msg.type === "left-call") {
      if (sender.tab) callTabs.delete(sender.tab.id);
      if (on && callTabs.size === 0) stop();
      return reply(await view());
    }
    reply({});
  })();
  return true;
});

updateBadge();
