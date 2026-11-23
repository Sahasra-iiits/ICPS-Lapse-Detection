const DEFAULTS = {
  enabled: { dozing: true, drowsy: true, zoning: true, stress: true },
  sensitivity: "normal", muted: false, showPill: true,
};
const $ = (s) => document.querySelector(s);
let settings = { ...DEFAULTS };

async function load() {
  const s = await chrome.storage.sync.get(DEFAULTS);
  settings = { ...DEFAULTS, ...s, enabled: { ...DEFAULTS.enabled, ...(s.enabled || {}) } };
  document.querySelectorAll("[data-alert]").forEach((cb) => (cb.checked = settings.enabled[cb.dataset.alert]));
  document.querySelectorAll("#sens button").forEach((b) => b.classList.toggle("on", b.dataset.v === settings.sensitivity));
  $("#sound").checked = !settings.muted;
  $("#pill").checked = settings.showPill;
}

async function save() {
  const enabled = {};
  document.querySelectorAll("[data-alert]").forEach((cb) => (enabled[cb.dataset.alert] = cb.checked));
  settings = { ...settings, enabled, muted: !$("#sound").checked, showPill: $("#pill").checked };
  await chrome.storage.sync.set(settings);
  refresh();
}

function render(st) {
  const dot = $("#dot"), grid = $("#grid");
  dot.className = "dot";
  if (!st || st.offline || !st.running) {
    $("#headline").textContent = "Helper not running";
    grid.innerHTML = `<span>Start it in a terminal:</span><b></b><span><code>python live_monitor.py --meet</code></span><b></b>`;
    return;
  }
  if (st.calibrating) { dot.classList.add("cal"); $("#headline").textContent = `Calibrating${st.calib_left_s != null ? ` (${st.calib_left_s} s)` : ""}`; }
  else if (!st.face) { dot.classList.add("cal"); $("#headline").textContent = "Face not visible"; }
  else { dot.classList.add("live"); $("#headline").textContent = "Monitoring"; }
  const pct = (v) => (v == null ? "-" : Math.round(v * 100) + "%");
  const hr = st.hr ? `${Math.round(st.hr)} bpm (${st.hr_source}${st.sim_hr ? ", sim" : ""})` : "-";
  const zs = st.zoning_score == null ? "-" : st.zoning_score.toFixed(2) + (st.zoning_threshold != null ? ` / ${Number(st.zoning_threshold).toFixed(2)}` : "");
  grid.innerHTML = `<span>Heart rate</span><b>${hr}</b><span>Eyes closed, last minute</span><b>${pct(st.perclos)}</b>` +
    `<span>Zoning score / alert level</span><b>${zs}</b><span>Alerts this session</span><b>${st.alerts ? st.alerts.length : 0}</b>`;
  $("#sim").checked = !!st.sim_hr;
}

async function refresh() { render(await chrome.runtime.sendMessage({ type: "poll" })); }

async function toMeet(msg) {
  const tabs = await chrome.tabs.query({ url: "https://meet.google.com/*" });
  const tab = tabs.find((t) => t.active) || tabs[0];
  if (!tab) { $("#note").textContent = "Open a Google Meet tab first."; return; }
  $("#note").textContent = "";
  try { await chrome.tabs.sendMessage(tab.id, msg); await chrome.tabs.update(tab.id, { active: true }); }
  catch (e) { $("#note").textContent = "Reload the Meet tab once so the extension can attach."; }
}

document.querySelectorAll("[data-alert], #sound, #pill").forEach((x) => x.addEventListener("change", save));
document.querySelectorAll("#sens button").forEach((b) => b.addEventListener("click", () => {
  settings.sensitivity = b.dataset.v;
  document.querySelectorAll("#sens button").forEach((x) => x.classList.toggle("on", x === b));
  save();
}));
document.querySelectorAll("[data-demo]").forEach((b) => b.addEventListener("click", () => toMeet({ type: "demo", kind: b.dataset.demo })));
$("#demoAll").addEventListener("click", () => toMeet({ type: "demo-all" }));
$("#sim").addEventListener("change", async (e) => render(await chrome.runtime.sendMessage({ type: "sim", on: e.target.checked })));
load().then(refresh);
setInterval(refresh, 1000);
