// One-time camera permission for the extension. The detector runs in a hidden
// page, which can't show Chrome's permission prompt, so it's asked for here.
const $ = (id) => document.getElementById(id);

async function ask() {
  $("denied").hidden = true;
  $("state").className = "state";
  $("state").innerHTML = "Click <b>Allow</b> in Chrome's camera prompt at the top of this window.";
  try {
    const s = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    s.getTracks().forEach((t) => t.stop());
    $("state").className = "state ok";
    $("state").textContent = "Camera allowed. Taking you back to Meet…";
    setTimeout(() => chrome.runtime.sendMessage({ type: "camera-granted" }), 700);
  } catch (e) {
    $("state").textContent = e && e.name === "NotFoundError" ? "No camera was found on this computer." : "Camera access was not allowed.";
    $("denied").hidden = e && e.name === "NotFoundError";
  }
}
$("retry").onclick = ask;
ask();
