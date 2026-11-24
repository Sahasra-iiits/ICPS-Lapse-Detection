// Runs inside Google Meet: the Focus Alerts switch, alert banners, sounds and
// the breathing guide. All alerts come from the in-browser detector (detector.js).
(() => {
  if (window.__icpsFocusAlerts) return;
  window.__icpsFocusAlerts = true;

  const ICONS = {
    dozing: '<path d="M3 11c2.6 3.2 5.6 4.8 9 4.8s6.4-1.6 9-4.8"/><path d="M6.2 14.6 4.8 16.8M12 15.8v2.6M17.8 14.6l1.4 2.2"/><path d="M15.5 4.5h3l-3 3.5h3"/>',
    drowsy: '<path d="M2.5 12s3.6-6 9.5-6 9.5 6 9.5 6-3.6 6-9.5 6-9.5-6-9.5-6z"/><path d="M3.6 10.6C6.4 9 9.2 8.2 12 8.2s5.6.8 8.4 2.4"/><path d="M9.4 12.4a2.6 2.6 0 0 0 5.2 0"/>',
    zoning: '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="3.5"/><path d="M14.8 9.2 19.5 4.5" stroke-dasharray="1.6 2.2"/><circle cx="20" cy="4" r="1.2" fill="currentColor"/>',
    stress: '<path d="M12 20s-7-4.4-9-8.6C1.7 8.4 3.6 5 7 5c2 0 3.3 1.1 5 3 1.7-1.9 3-3 5-3 3.4 0 5.3 3.4 4 6.4-2 4.2-9 8.6-9 8.6z"/><path d="M5.5 12h3l1.5-2.6 2.6 5.2 1.6-2.6h4.3"/>',
  };
  const KIND = {
    dozing: { eyebrow: "Focus alert", grad: "linear-gradient(120deg,#4a000e 0%,#8f0018 45%,#d2042d 100%)", ms: 8000, pulse: true,
              beep: [[1200, .15], [0, .05], [1200, .15], [0, .05], [1200, .15], [0, .05], [1200, .15]] },
    drowsy: { eyebrow: "Focus alert", grad: "linear-gradient(120deg,#5e0014 0%,#a3001c 50%,#e0243f 100%)", ms: 7000,
              beep: [[900, .2], [0, .1], [900, .2]] },
    zoning: { eyebrow: "Focus check", grad: "linear-gradient(120deg,#6b0f1f 0%,#b51b36 50%,#ea4c64 100%)", ms: 6000,
              beep: [[660, .25]] },
    stress: { eyebrow: "Wellbeing", grad: "linear-gradient(120deg,#5a0b1e 0%,#a51d47 50%,#e4587d 100%)", ms: 7000,
              beep: [[440, .4]] },
  };
  const BREATH_IN = 4, BREATH_OUT = 6, BREATH_CYCLES = 6;
  const X = '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>';

  const host = document.createElement("div");
  host.style.cssText = "position:fixed;inset:0;pointer-events:none;z-index:2147483647;";
  document.documentElement.appendChild(host);
  const root = host.attachShadow({ mode: "open" });
  root.innerHTML = `
  <style>
    * { box-sizing: border-box; margin: 0; font-family: "Google Sans", "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    .wrap { position: fixed; top: 18px; left: 50%; width: min(560px, 92vw); transform: translate(-50%, -140%);
      opacity: 0; transition: transform .45s cubic-bezier(.2,.9,.25,1.15), opacity .3s ease; pointer-events: auto; }
    .wrap.show { transform: translate(-50%, 0); opacity: 1; }
    .card { position: relative; overflow: hidden; display: flex; align-items: center; gap: 16px;
      padding: 16px 18px 18px 16px; border-radius: 18px; color: #fff; border: 1px solid rgba(255,255,255,.18);
      box-shadow: 0 22px 48px rgba(110,0,22,.45), 0 4px 12px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.28); }
    .card::before { content: ""; position: absolute; inset: 0; pointer-events: none;
      background: radial-gradient(120% 140% at 100% 0%, rgba(255,255,255,.22) 0%, rgba(255,255,255,0) 55%); }
    .card.pulse { animation: glow 1.4s ease-in-out infinite; }
    @keyframes glow { 0%,100% { box-shadow: 0 22px 48px rgba(110,0,22,.45), 0 0 0 0 rgba(210,4,45,.55), inset 0 1px 0 rgba(255,255,255,.28); }
                      50% { box-shadow: 0 22px 48px rgba(110,0,22,.45), 0 0 0 10px rgba(210,4,45,0), inset 0 1px 0 rgba(255,255,255,.28); } }
    .badge { flex: none; width: 48px; height: 48px; border-radius: 14px; display: grid; place-items: center;
      background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.22); }
    .badge svg { width: 28px; height: 28px; fill: none; stroke: #fff; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
    .text { min-width: 0; flex: 1; }
    .eyebrow { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: rgba(255,255,255,.78); font-weight: 600; }
    .title { font-size: 19px; font-weight: 700; margin-top: 3px; letter-spacing: -.01em; }
    .msg { font-size: 14px; margin-top: 3px; color: rgba(255,255,255,.92); line-height: 1.35; }
    .close { flex: none; width: 32px; height: 32px; border-radius: 50%; border: 0; cursor: pointer;
      background: rgba(255,255,255,.14); display: grid; place-items: center; transition: background .2s; }
    .close:hover { background: rgba(255,255,255,.28); }
    .close svg { width: 14px; height: 14px; stroke: #fff; stroke-width: 2.2; stroke-linecap: round; fill: none; }
    .bar { position: absolute; left: 0; bottom: 0; height: 3px; width: 100%; background: rgba(255,255,255,.75); transform-origin: left; }
    .wrap:hover .bar { animation-play-state: paused; }
    @keyframes shrink { from { transform: scaleX(1); } to { transform: scaleX(0); } }

    .pill { position: fixed; left: 16px; bottom: 96px; display: flex; align-items: center; gap: 10px;
      padding: 6px 8px 6px 12px; border-radius: 999px; font-size: 12.5px; color: #fff; pointer-events: auto;
      background: rgba(26,8,12,.86); border: 1px solid rgba(210,4,45,.42);
      box-shadow: 0 8px 22px rgba(0,0,0,.32); backdrop-filter: blur(8px); max-width: min(560px, 80vw); }
    .brand { font-weight: 700; letter-spacing: .01em; white-space: nowrap; }
    .sep { width: 1px; height: 14px; background: rgba(255,255,255,.22); }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: #7d7d7d; flex: none; }
    .dot.live { background: #ff4d6d; animation: live 2s infinite; }
    .dot.warm { background: #ffb4a2; }
    @keyframes live { 0% { box-shadow: 0 0 0 0 rgba(255,77,109,.6); } 70% { box-shadow: 0 0 0 7px rgba(255,77,109,0); } 100% { box-shadow: 0 0 0 0 rgba(255,77,109,0); } }
    .ptext { color: rgba(255,255,255,.86); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .switch { position: relative; width: 36px; height: 20px; flex: none; cursor: pointer; border: 0; border-radius: 999px;
      background: rgba(255,255,255,.22); transition: background .2s; }
    .switch::after { content: ""; position: absolute; top: 2px; left: 2px; width: 16px; height: 16px; border-radius: 50%;
      background: #fff; transition: transform .2s; box-shadow: 0 1px 3px rgba(0,0,0,.35); }
    .switch.on { background: #d2042d; }
    .switch.on::after { transform: translateX(16px); }
    .switch:disabled { opacity: .45; cursor: default; }

    .breath { position: fixed; right: 24px; bottom: 110px; width: 220px; padding: 18px 16px 16px; border-radius: 20px;
      color: #fff; text-align: center; pointer-events: auto; display: none;
      background: linear-gradient(160deg, rgba(58,6,16,.95), rgba(26,4,9,.95));
      border: 1px solid rgba(255,77,109,.35); box-shadow: 0 20px 44px rgba(90,0,20,.5); }
    .breath.show { display: block; animation: fadein .4s ease; }
    @keyframes fadein { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
    .breath .head { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: rgba(255,255,255,.7); font-weight: 600; }
    .ring { width: 150px; height: 150px; margin: 14px auto 12px; border-radius: 50%; display: grid; place-items: center;
      border: 1.5px solid rgba(255,255,255,.22); }
    .ball { width: 46px; height: 46px; border-radius: 50%; transition: width .25s linear, height .25s linear;
      background: radial-gradient(circle at 35% 30%, #ff8fa3, #d2042d 60%, #8f0018); box-shadow: 0 0 28px rgba(255,77,109,.55); }
    .breath .label { font-size: 17px; font-weight: 700; }
    .breath .sub { font-size: 12px; color: rgba(255,255,255,.65); margin-top: 3px; }
    .breath .close { position: absolute; top: 10px; right: 10px; width: 26px; height: 26px; }
  </style>
  <div class="wrap" id="wrap"><div class="card" id="card">
    <div class="badge"><svg viewBox="0 0 24 24" id="icon"></svg></div>
    <div class="text"><div class="eyebrow" id="eyebrow"></div><div class="title" id="title"></div><div class="msg" id="msg"></div></div>
    <button class="close" id="close" title="Dismiss">${X}</button><div class="bar" id="bar"></div></div></div>
  <div class="pill" id="pill">
    <span class="dot" id="dot"></span><span class="brand">Focus Alerts</span><span class="sep"></span>
    <span class="ptext" id="ptext">Off</span>
    <button class="switch" id="switch" title="Turn Focus Alerts on or off" aria-label="Focus Alerts"></button>
  </div>
  <div class="breath" id="breath"><button class="close" id="bclose" title="Close">${X}</button>
    <div class="head">Breathing guide</div><div class="ring"><div class="ball" id="ball"></div></div>
    <div class="label" id="blabel">Breathe in</div><div class="sub" id="bsub"></div></div>`;
  const $ = (id) => root.getElementById(id);

  // ---------- alert banner ----------
  let hideTimer = null;
  function showBanner(a) {
    const k = KIND[a.kind] || KIND.zoning;
    $("card").style.background = k.grad;
    $("card").classList.toggle("pulse", !!k.pulse);
    $("icon").innerHTML = ICONS[a.kind] || ICONS.zoning;
    $("eyebrow").textContent = k.eyebrow;
    $("title").textContent = a.title;
    $("msg").textContent = a.msg;
    const bar = $("bar");
    bar.style.animation = "none"; void bar.offsetWidth;
    bar.style.animation = `shrink ${k.ms}ms linear forwards`;
    $("wrap").classList.add("show");
    clearTimeout(hideTimer);
    hideTimer = setTimeout(hideBanner, k.ms);
  }
  function hideBanner() { $("wrap").classList.remove("show"); }
  $("close").onclick = hideBanner;
  $("wrap").addEventListener("mouseenter", () => clearTimeout(hideTimer));
  $("wrap").addEventListener("mouseleave", () => { hideTimer = setTimeout(hideBanner, 1500); });

  // ---------- sound (allowed after your first click in Meet) ----------
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
          g.gain.exponentialRampToValueAtTime(0.22, t + 0.012);
          g.gain.exponentialRampToValueAtTime(0.0001, t + d);
          o.connect(g).connect(ctx.destination);
          o.start(t); o.stop(t + d + 0.02);
        }
        t += d;
      }
    } catch (e) {}
  }

  // ---------- breathing guide ----------
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
      $("ball").style.width = $("ball").style.height = Math.round(46 + 92 * frac) + "px";
      $("blabel").textContent = inhale ? "Breathe in" : "Breathe out";
      $("bsub").textContent = `Cycle ${Math.min(BREATH_CYCLES, Math.floor(el / cyc) + 1)} of ${BREATH_CYCLES}`;
    };
    tick();
    breathTimer = setInterval(tick, 200);
  }
  function stopBreathing() { clearInterval(breathTimer); $("breath").classList.remove("show"); }
  $("bclose").onclick = stopBreathing;

  // ---------- status bar with the on/off switch ----------
  let current = null;
  function render(v) {
    current = v;
    const s = v.settings || {};
    $("pill").style.display = s.showPill === false ? "none" : "flex";
    const dot = $("dot"), sw = $("switch"), st = v.state;
    dot.className = "dot";
    sw.classList.toggle("on", !!v.on);
    let text;
    if (v.error && v.error.code === "camera-permission" && !v.on) text = "Allow camera access in the tab that opened, then it starts";
    else if (v.error && !v.on) text = "Couldn't start: " + (v.error.message || "unknown error");
    else if (!v.on) text = "Off";
    else if (!st || st.started === false) { dot.classList.add("warm"); text = "Starting camera…"; }
    else if (st.calibrating) { dot.classList.add("warm"); text = `Calibrating${st.calib_left_s != null ? ` · ${st.calib_left_s} s` : ""} · look at the screen`; }
    else if (!st.face) { dot.classList.add("warm"); text = "Face not visible"; }
    else {
      dot.classList.add("live");
      text = "On" + (st.hr ? ` · HR ${Math.round(st.hr)}` : "");
    }
    if (v.on && s.muted) text += " · muted";
    $("ptext").textContent = text;
  }
  $("switch").onclick = async () => {
    const want = !(current && current.on);
    $("switch").classList.toggle("on", want);
    unlock();
    try { render(await chrome.runtime.sendMessage({ type: "toggle", on: want })); } catch (e) {}
  };

  // ---------- live updates ----------
  let lastSeen = null;
  async function poll() {
    let v;
    try { v = await chrome.runtime.sendMessage({ type: "view" }); } catch (e) { return; }
    if (!v) return;
    render(v);
    const alerts = (v.state && v.state.alerts) || [];
    const newest = alerts.length ? alerts[alerts.length - 1].id : 0;
    if (lastSeen === null || newest < lastSeen) { lastSeen = newest; return; }   // new session: don't replay old alerts
    for (const a of alerts) {
      if (a.id <= lastSeen) continue;
      lastSeen = a.id;
      if (v.settings.enabled[a.kind] === false) continue;
      showBanner(a);
      if (!v.settings.muted) beep((KIND[a.kind] || KIND.zoning).beep);
      if (a.kind === "stress") startBreathing();
    }
  }
  setInterval(poll, 700);
  poll();

  // ---------- call detection (for auto-start, and stopping when you leave) ----------
  const inCall = () => /^\/[a-z]{3}-[a-z]{4}-[a-z]{3}/i.test(location.pathname);
  let wasInCall = false;
  async function checkCall() {
    const now = inCall();
    if (now !== wasInCall) {
      wasInCall = now;
      try { render(await chrome.runtime.sendMessage({ type: now ? "in-call" : "left-call" })); } catch (e) {}
    }
  }
  setInterval(checkCall, 1500);
  checkCall();
  window.addEventListener("pagehide", () => { try { chrome.runtime.sendMessage({ type: "left-call" }); } catch (e) {} });
})();
