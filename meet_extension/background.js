// Talks to the local Python helper (live_monitor.py --meet) and keeps the toolbar badge updated.
const HELPER = "http://127.0.0.1:47813";
const DEFAULTS = {
  enabled: { dozing: true, drowsy: true, zoning: true, stress: true },
  sensitivity: "normal", muted: false, showPill: true,
};
let lastAlertId = 0;
let alertUntil = 0;

async function getSettings() {
  const s = await chrome.storage.sync.get(DEFAULTS);
  return { ...DEFAULTS, ...s, enabled: { ...DEFAULTS.enabled, ...(s.enabled || {}) } };
}

async function helper(path) {
  const r = await fetch(HELPER + path, { cache: "no-store" });
  if (!r.ok) throw new Error("helper " + r.status);
  return r.json();
}

function setBadge(state) {
  if (!state || state.offline || !state.running) {
    chrome.action.setBadgeText({ text: "off" });
    chrome.action.setBadgeBackgroundColor({ color: "#9aa0a6" });
  } else if (Date.now() < alertUntil) {
    chrome.action.setBadgeText({ text: "!" });
    chrome.action.setBadgeBackgroundColor({ color: "#d93025" });
  } else if (state.calibrating) {
    chrome.action.setBadgeText({ text: "cal" });
    chrome.action.setBadgeBackgroundColor({ color: "#f9ab00" });
  } else {
    chrome.action.setBadgeText({ text: "ok" });
    chrome.action.setBadgeBackgroundColor({ color: "#1e8e3e" });
  }
}

async function poll() {
  const s = await getSettings();
  const en = Object.keys(s.enabled).filter((k) => s.enabled[k]).join(",");
  try {
    const state = await helper(`/state?sens=${s.sensitivity}&en=${en}`);
    const newest = state.alerts.length ? state.alerts[state.alerts.length - 1].id : 0;
    if (newest > lastAlertId) {
      if (lastAlertId) alertUntil = Date.now() + 10000;
      lastAlertId = newest;
    }
    setBadge(state);
    return { ...state, settings: s };
  } catch (e) {
    const off = { offline: true, settings: s };
    setBadge(off);
    return off;
  }
}

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  (async () => {
    if (msg.type === "poll") return reply(await poll());
    if (msg.type === "test") {
      try { await helper(`/test?kind=${msg.kind}`); } catch (e) {}
      return reply(await poll());
    }
    if (msg.type === "sim") {
      try { await helper(`/sim?on=${msg.on ? 1 : 0}`); } catch (e) {}
      return reply(await poll());
    }
    reply({});
  })();
  return true; // async reply
});
