const DEFAULTS = {
  enabled: { dozing: true, drowsy: true, zoning: true, stress: true },
  sensitivity: "normal", muted: false, showPill: true,
};
const $ = (s) => document.querySelector(s);

async function load() {
  const s = await chrome.storage.sync.get(DEFAULTS);
  const en = { ...DEFAULTS.enabled, ...(s.enabled || {}) };
  document.querySelectorAll("[data-alert]").forEach((cb) => (cb.checked = en[cb.dataset.alert]));
  $("#sens").value = s.sensitivity;
  $("#muted").checked = s.muted;
  $("#pill").checked = s.showPill;
}

async function save() {
  const enabled = {};
  document.querySelectorAll("[data-alert]").forEach((cb) => (enabled[cb.dataset.alert] = cb.checked));
  await chrome.storage.sync.set({ enabled, sensitivity: $("#sens").value, muted: $("#muted").checked, showPill: $("#pill").checked });
  refresh();
}

function render(st) {
  const el = $("#status");
  if (!st || st.offline || !st.running) {
    el.innerHTML = "<b>⚪ Helper not running</b>Start it in a terminal:<br><code>python live_monitor.py --meet</code>";
    return;
  }
  const pct = (v) => (v == null ? "–" : Math.round(v * 100) + "%");
  const head = st.calibrating ? `🟡 Calibrating… ${st.calib_left_s ?? ""} s` : st.face ? "🟢 Monitoring" : "🟡 Face not visible";
  el.innerHTML = `<b>${head}</b>` +
    `Heart rate: ${st.hr ? Math.round(st.hr) + " bpm (" + st.hr_source + (st.sim_hr ? ", simulated" : "") + ")" : "–"}<br>` +
    `Eyes closed (last min): ${pct(st.perclos)}<br>` +
    `Zoning score: ${st.zoning_score == null ? "–" : st.zoning_score.toFixed(2)}` +
    (st.zoning_threshold != null ? ` (alert ≥ ${Number(st.zoning_threshold).toFixed(2)})` : "") +
    `<br>Alerts so far: ${st.alerts ? st.alerts.length : 0}`;
  $("#sim").checked = !!st.sim_hr;
}

async function refresh() { render(await chrome.runtime.sendMessage({ type: "poll" })); }

document.querySelectorAll("input, select").forEach((x) => x.id !== "sim" && x.addEventListener("change", save));
document.querySelectorAll("[data-test]").forEach((b) =>
  b.addEventListener("click", async () => render(await chrome.runtime.sendMessage({ type: "test", kind: b.dataset.test }))));
$("#sim").addEventListener("change", async (e) => render(await chrome.runtime.sendMessage({ type: "sim", on: e.target.checked })));
load().then(refresh);
setInterval(refresh, 1000);
