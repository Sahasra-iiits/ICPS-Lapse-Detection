// Runs inside Google Meet: shows alert banners, plays sounds, shows the breathing guide.
(() => {
  if (window.__icpsFocusAlerts) return;
  window.__icpsFocusAlerts = true;

  const STYLE = {
    dozing: { color: "#d93025", icon: "😴", beep: [[1200, .15], [0, .05], [1200, .15], [0, .05], [1200, .15], [0, .05], [1200, .15]] },
    drowsy: { color: "#e8710a", icon: "🥱", beep: [[900, .2], [0, .1], [900, .2]] },
    zoning: { color: "#1a73e8", icon: "💭", beep: [[660, .25]] },
    stress: { color: "#9334e6", icon: "🫁", beep: [[440, .4]] },
  };
  const BREATH_IN = 4, BREATH_OUT = 6, BREATH_CYCLES = 6;

  // ---------- UI in a shadow root so Meet's styles don't interfere ----------
  const host = document.createElement("div");
  host.style.cssText = "position:fixed;inset:0;pointer-events:none;z-index:2147483647;";
  document.documentElement.appendChild(host);
  const root = host.attachShadow({ mode: "open" });
  root.innerHTML = `
    <style>
      * { box-sizing: border-box; font-family: "Google Sans", Roboto, Arial, sans-serif; }
      .banner { position: fixed; top: 16px; left: 50%; transform: translate(-50%, -150%);
        min-width: 320px; max-width: 90vw; padding: 14px 18px 14px 16px; border-radius: 12px; color: #fff;
        box-shadow: 0 6px 24px rgba(0,0,0,.35); display: flex; gap: 12px; align-items: center;
        transition: transform .35s ease; pointer-events: auto; }
      .banner.show { transform: translate(-50%, 0); }
      .banner .icon { font-size: 30px; line-height: 1; }
      .banner .title { font-size: 18px; font-weight: 700; }
      .banner .msg { font-size: 14px; opacity: .95; margin-top: 2px; }
      .banner .x { margin-left: auto; background: transparent; border: 0; color: #fff; font-size: 20px; cursor: pointer; }
      .pill { position: fixed; left: 16px; bottom: 96px; padding: 6px 12px; border-radius: 999px;
        background: rgba(32,33,36,.85); color: #fff; font-size: 12px; display: flex; gap: 8px; align-items: center; }
      .dot { width: 8px; height: 8px; border-radius: 50%; background: #9aa0a6; }
      .breath { position: fixed; right: 24px; bottom: 110px; width: 200px; height: 230px; border-radius: 16px;
        background: rgba(32,33,36,.9); color: #fff; display: none; flex-direction: column; align-items: center;
        justify-content: center; gap: 12px; pointer-events: auto; }
      .breath.show { display: flex; }
      .ring { width: 140px; height: 140px; border-radius: 50%; border: 2px solid rgba(255,255,255,.5);
        display: flex; align-items: center; justify-content: center; }
      .ball { width: 40px; height: 40px; border-radius: 50%; background: #9334e6; transition: width 1s linear, height 1s linear; }
      .breath .label { font-size: 16px; font-weight: 600; }
      .breath .x { position: absolute; top: 6px; right: 10px; background: transparent; border: 0; color: #fff; font-size: 18px; cursor: pointer; }
    </style>
    <div class="banner" id="banner"><div class="icon" id="bicon"></div>
      <div><div class="title" id="btitle"></div><div class="msg" id="bmsg"></div></div>
      <button class="x" id="bclose" title="Dismiss">✕</button></div>
    <div class="pill" id="pill"><span class="dot" id="dot"></span><span id="ptext">Focus alerts: starting…</span></div>
    <div class="breath" id="breath"><button class="x" id="brclose" title="Close">✕</button>
      <div class="ring"><div class="ball" id="ball"></div></div><div class="label" id="blabel">Breathe in</div></div>`;
  const $ = (id) => root.getElementById(id);
  let bannerTimer = null;

  function showBanner(a) {
    const st = STYLE[a.kind] || STYLE.zoning;
    $("banner").style.background = st.color;
    $("bicon").textContent = st.icon;
    $("btitle").textContent = a.title;
    $("bmsg").textContent = a.msg;
    $("banner").classList.add("show");
    clearTimeout(bannerTimer);
    bannerTimer = setTimeout(() => $("banner").classList.remove("show"), a.kind === "dozing" ? 8000 : 6000);
  }
  $("bclose").onclick = () => $("banner").classList.remove("show");

  // ---------- sound (Web Audio; unlocked by your first click in Meet) ----------
  let ctx = null;
  const unlock = () => { try { ctx = ctx || new AudioContext(); ctx.resume(); } catch (e) {} };
  window.addEventListener("pointerdown", unlock, true);
  window.addEventListener("keydown", unlock, true);
  function beep(pattern) {
    try {
      ctx = ctx || new AudioContext();
      let t = ctx.currentTime + 0.02;
      for (const [f, d] of pattern) {
        if (f > 0) {
          const o = ctx.createOscillator(), g = ctx.createGain();
          o.frequency.value = f;
          g.gain.setValueAtTime(0.0001, t);
          g.gain.exponentialRampToValueAtTime(0.25, t + 0.01);
          g.gain.exponentialRampToValueAtTime(0.0001, t + d);
          o.connect(g).connect(ctx.destination);
          o.start(t); o.stop(t + d + 0.02);
        }
        t += d;
      }
    } catch (e) {}
  }

  // ---------- breathing guide (4 s in, 6 s out, 6 cycles) ----------
  let breathTimer = null;
  function startBreathing() {
    clearInterval(breathTimer);
    const start = Date.now();
    $("breath").classList.add("show");
    const tick = () => {
      const el = (Date.now() - start) / 1000, cyc = BREATH_IN + BREATH_OUT;
      if (el > cyc * BREATH_CYCLES) return stopBreathing();
      const ph = el % cyc, inhale = ph < BREATH_IN;
      const frac = inhale ? ph / BREATH_IN : 1 - (ph - BREATH_IN) / BREATH_OUT;
      const size = Math.round(40 + 90 * frac);
      $("ball").style.width = $("ball").style.height = size + "px";
      $("blabel").textContent = inhale ? "Breathe in…" : "Breathe out…";
    };
    tick();
    breathTimer = setInterval(tick, 250);
  }
  function stopBreathing() { clearInterval(breathTimer); $("breath").classList.remove("show"); }
  $("brclose").onclick = stopBreathing;

  // ---------- polling the local helper (through the extension's background) ----------
  let lastSeen = null;
  function pill(state) {
    const s = state.settings || {};
    $("pill").style.display = s.showPill === false ? "none" : "flex";
    let color = "#9aa0a6", text;
    if (state.offline || !state.running) text = "Focus alerts: helper not running (python live_monitor.py --meet)";
    else if (state.calibrating) { color = "#f9ab00"; text = `Focus alerts: calibrating… ${state.calib_left_s ?? ""} s — look at the screen, blink normally`; }
    else if (!state.face) { color = "#f9ab00"; text = "Focus alerts: face not visible"; }
    else {
      color = "#1e8e3e";
      text = "Focus alerts: on" + (state.hr ? ` · HR ${Math.round(state.hr)}${state.sim_hr ? " (sim)" : ""}` : "") +
        (state.perclos != null ? ` · eyes closed ${Math.round(state.perclos * 100)}%` : "");
    }
    if (s.muted) text += " · muted";
    $("dot").style.background = color;
    $("ptext").textContent = text;
  }

  async function poll() {
    let state;
    try { state = await chrome.runtime.sendMessage({ type: "poll" }); }
    catch (e) { return; }               // extension reloaded: stop quietly
    if (!state) return;
    pill(state);
    if (state.offline || !state.alerts) return;
    const s = state.settings;
    if (lastSeen === null) { lastSeen = state.alerts.length ? state.alerts[state.alerts.length - 1].id : 0; return; }
    for (const a of state.alerts) {
      if (a.id <= lastSeen) continue;
      lastSeen = a.id;
      if (!s.enabled[a.kind]) continue;
      showBanner(a);
      if (!s.muted) beep((STYLE[a.kind] || STYLE.zoning).beep);
      if (a.kind === "stress") startBreathing();
    }
  }
  setInterval(poll, 700);
  poll();
})();
