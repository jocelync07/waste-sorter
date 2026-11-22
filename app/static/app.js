const CLASS_LABELS = {
  AluCan: "Aluminium can",
  Glass: "Glass bottle",
  HDPEM: "HDPE milk bottle",
  PET: "PET plastic bottle",
};

const video = document.getElementById("video");
const preview = document.getElementById("preview");
const startButton = document.getElementById("start");
const snapButton = document.getElementById("snap");
const liveToggle = document.getElementById("live");
const uploadInput = document.getElementById("upload");
const labelEl = document.getElementById("label");
const metaEl = document.getElementById("meta");
const barsEl = document.getElementById("bars");
const errorEl = document.getElementById("error");
const uncertainEl = document.getElementById("uncertain");
const untrainedEl = document.getElementById("untrained");

let stream = null;
let liveTimer = null;
let busy = false;

function show(el, visible) {
  el.classList.toggle("hidden", !visible);
}

function showError(message) {
  errorEl.textContent = message;
  show(errorEl, Boolean(message));
}

function renderResult(result) {
  showError("");
  labelEl.textContent = result.display_name;
  metaEl.textContent = `${(result.confidence * 100).toFixed(1)}% confident · ${result.latency_ms} ms`;
  show(uncertainEl, result.low_confidence);

  barsEl.replaceChildren();
  const sorted = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  for (const [name, p] of sorted) {
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("span");
    label.textContent = CLASS_LABELS[name] || name;
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${(p * 100).toFixed(1)}%`;
    track.append(fill);
    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = `${(p * 100).toFixed(0)}%`;
    row.append(label, track, value);
    barsEl.append(row);
  }
}

async function classify(blob, source) {
  if (busy) return; // skip a frame rather than queue requests
  busy = true;
  try {
    const form = new FormData();
    form.append("file", blob, "frame.jpg");
    form.append("source", source);
    const response = await fetch("/api/predict", { method: "POST", body: form });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
    renderResult(body);
  } catch (err) {
    showError(err.message);
  } finally {
    busy = false;
  }
}

function captureFrame() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
}

async function classifyFrame() {
  if (!stream || !video.videoWidth) return;
  const blob = await captureFrame();
  if (blob) await classify(blob, "webcam");
}

async function startCamera() {
  if (stream) {
    stopCamera();
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    video.style.display = "block";
    preview.style.display = "none";
    startButton.textContent = "Stop camera";
    snapButton.disabled = false;
    liveToggle.disabled = false;
    showError("");
  } catch (err) {
    showError("Could not open the camera. Allow camera access, or upload an image instead.");
  }
}

function stopCamera() {
  stream.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  startButton.textContent = "Start camera";
  snapButton.disabled = true;
  liveToggle.checked = false;
  liveToggle.disabled = true;
  setLive(false);
}

function setLive(on) {
  clearInterval(liveTimer);
  liveTimer = on ? setInterval(classifyFrame, 1000) : null;
}

startButton.addEventListener("click", startCamera);
snapButton.addEventListener("click", classifyFrame);
liveToggle.addEventListener("change", () => setLive(liveToggle.checked));

uploadInput.addEventListener("change", () => {
  const file = uploadInput.files[0];
  if (!file) return;
  if (stream) stopCamera();
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
  video.style.display = "none";
  classify(file, "upload");
  uploadInput.value = "";
});

fetch("/api/health")
  .then((r) => r.json())
  .then((health) => show(untrainedEl, !health.model_loaded))
  .catch(() => {});
