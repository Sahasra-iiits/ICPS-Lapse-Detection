const DEFAULTS = {
  enabled: { dozing: true, drowsy: true, zoning: true, stress: true },
  sensitivity: "normal", muted: false, showPill: true, autoStart: false,
};
const $ = (s) => document.querySelector(s);
let settings = { ...DEFAULTS };
let current = null;

async function load() {
  const s = await chrome.storage.sync.get(DEFAULTS);
  settings = { ...DEFAULTS, ...s, enabled: { ...DEFAULTS.enabled, ...(s.enabled || {}) } };
  document.querySelectorAll("[data-alert]").forEach((cb) => (cb.checked = settings.enabled[cb.dataset.alert]));
  document.querySelectorAll("#sens button").forEach((b) => b.classList.toggle("on", b.dataset.v === settings.sensitivity));
  $("#sound").checked = !settings.muted;
  $("#pill").checked = settings.showPill;
  $("#auto").checked = settings.autoStart;
}

async function save() {
  const enabled = {};
  document.querySelectorAll("[data-alert]").forEach((cb) => (enabled[cb.dataset.alert] = cb.checked));
  settings = { ...settings, enabled, muted: !$("#sound").checked, showPill: $("#pill").checked, autoStart: $("#auto").checked };
  await chrome.storage.sync.set(settings);
}

function render(v) {
  if (!v) return;
  current = v;
  const st = v.state, dot = $("#dot"), grid = $("#grid");
  const notInstalled = v.error && v.error.code === "not-installed" && !v.on;
  $("#setup").style.display = notInstalled ? "block" : "none";
  $("#power").classList.toggle("on", !!v.on);
  $("#power").disabled = v.mode === "manual";
  dot.className = "dot";
  grid.innerHTML = "";
  let head, sub = "";
  if (notInstalled) { head = "Setup needed"; sub = "Finish the one-time setup below."; }
  else if (v.error && !v.on) { head = "Couldn't start"; sub = v.error.message || ""; }
  else if (!v.on) { head = "Off"; sub = "Turn it on here or from the switch in Google Meet."; }
  else if (!st || st.started === false) { dot.classList.add("warm"); head = "Starting camera"; sub = "This takes a few seconds."; }
  else if (st.calibrating) { dot.classList.add("warm"); head = `Calibrating${st.calib_left_s != null ? ` · ${st.calib_left_s} s` : ""}`; sub = "Look at the screen and blink normally."; }
  else if (!st.face) { dot.classList.add("warm"); head = "Face not visible"; sub = "Make sure your face is in view of the camera."; }
  else { dot.classList.add("live"); head = "Monitoring"; sub = v.mode === "manual" ? "Running from a terminal helper." : "Alerts appear in your Meet tab."; }
  if (v.on && st && st.started !== false && !st.calibrating) {
    const pct = (x) => (x == null ? "-" : Math.round(x * 100) + "%");
    grid.innerHTML = `<span>Heart rate</span><b>${st.hr ? Math.round(st.hr) + " bpm" : "-"}</b>` +
      `<span>Eyes closed, last minute</span><b>${pct(st.perclos)}</b>` +
      `<span>Alerts this session</span><b>${st.alerts ? st.alerts.length : 0}</b>`;
  }
  $("#headline").textContent = head;
  $("#sub").textContent = sub;
  $("#hsub").textContent = v.on ? "On" : "Off";
}

async function refresh() { try { render(await chrome.runtime.sendMessage({ type: "view" })); } catch (e) {} }

$("#power").addEventListener("click", async () => {
  const want = !(current && current.on);
  $("#power").classList.toggle("on", want);
  render(await chrome.runtime.sendMessage({ type: "toggle", on: want }));
});
document.querySelectorAll("[data-alert], #sound, #pill, #auto").forEach((x) => x.addEventListener("change", save));
document.querySelectorAll("#sens button").forEach((b) => b.addEventListener("click", () => {
  settings.sensitivity = b.dataset.v;
  document.querySelectorAll("#sens button").forEach((x) => x.classList.toggle("on", x === b));
  save();
}));
load().then(refresh);
setInterval(refresh, 1000);
