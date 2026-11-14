'use strict';

const SIZE = 28;
const BOX = 20;
const BRUSH = 20;

const activations = {
  relu: v => (v > 0 ? v : 0),
  tanh: Math.tanh,
  sigmoid: v => 1 / (1 + Math.exp(-v)),
  none: v => v,
};

// ---- model ----

function parseModel(model, buffer) {
  const { layers } = model;
  const floats = layers.reduce((n, l) => n + l.in * l.out + l.out, 0);
  if (buffer.byteLength !== floats * 4) {
    throw new Error(`weights.bin is ${buffer.byteLength} bytes, expected ${floats * 4}`);
  }
  if (layers[0].in !== SIZE * SIZE) throw new Error(`first layer takes ${layers[0].in} inputs, expected 784`);

  const view = new DataView(buffer);
  let offset = 0;
  const read = n => {
    const arr = new Float32Array(n);
    for (let i = 0; i < n; i++, offset += 4) arr[i] = view.getFloat32(offset, true);
    return arr;
  };

  return layers.map((l, k) => {
    if (!activations[l.activation]) throw new Error(`unknown activation "${l.activation}"`);
    if (k > 0 && layers[k - 1].out !== l.in) throw new Error(`layer ${k} size mismatch`);
    return { inSize: l.in, outSize: l.out, activation: l.activation, W: read(l.in * l.out), b: read(l.out) };
  });
}

function forward(layers, input) {
  let x = input;
  for (const { inSize, outSize, activation, W, b } of layers) {
    const z = Float32Array.from(b);
    for (let i = 0; i < inSize; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      const row = i * outSize;
      for (let j = 0; j < outSize; j++) z[j] += xi * W[row + j];
    }
    x = z.map(activations[activation]);
  }
  return x;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = Array.from(logits, v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

// ---- preprocessing ----

function inkBounds(img, threshold = 10) {
  const { width, height, data } = img;
  let x0 = width, y0 = height, x1 = -1, y1 = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (data[(y * width + x) * 4] > threshold) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  return x1 < 0 ? null : { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

function makeCanvas(w, h) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  return Object.assign(document.createElement('canvas'), { width: w, height: h });
}

// MNIST digits were size-normalised to fit a 20x20 box (keeping aspect ratio)
// and then placed in a 28x28 image so that their centre of mass is in the middle.
// Doing the same here makes a big difference for hand-drawn input.
function preprocess(img) {
  const box = inkBounds(img);
  if (!box) return null;

  const scale = BOX / Math.max(box.w, box.h);
  const w = Math.max(1, Math.round(box.w * scale));
  const h = Math.max(1, Math.round(box.h * scale));

  const src = makeCanvas(img.width, img.height);
  src.getContext('2d').putImageData(img, 0, 0);
  const small = makeCanvas(w, h);
  const ctx = small.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, box.x, box.y, box.w, box.h, 0, 0, w, h);

  return centerByMass(ctx.getImageData(0, 0, w, h).data, w, h);
}

function centerByMass(rgba, w, h) {
  let total = 0, sx = 0, sy = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = rgba[(y * w + x) * 4] / 255;
      total += v;
      sx += v * x;
      sy += v * y;
    }
  }
  const out = new Float32Array(SIZE * SIZE);
  if (total === 0) return out;

  // shift so the weighted centre lands on pixel (14, 14)
  const dx = Math.round(SIZE / 2 - sx / total);
  const dy = Math.round(SIZE / 2 - sy / total);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const tx = x + dx, ty = y + dy;
      if (tx >= 0 && tx < SIZE && ty >= 0 && ty < SIZE) {
        out[ty * SIZE + tx] = rgba[(y * w + x) * 4] / 255;
      }
    }
  }
  return out;
}

// ---- page ----

let network = null;

const pad = document.getElementById('pad');
const padCtx = pad.getContext('2d', { willReadFrequently: true });
const preview = document.getElementById('preview').getContext('2d');
const digitEl = document.getElementById('digit');
const confidenceEl = document.getElementById('confidence');
const statusEl = document.getElementById('status');
const announceEl = document.getElementById('announce');
const bars = buildBars(document.getElementById('bars'));

function buildBars(list) {
  return Array.from({ length: 10 }, (_, d) => {
    const li = document.createElement('li');
    li.innerHTML = `<span class="num">${d}</span><span class="track"><span class="fill"></span></span><span class="pct">0%</span>`;
    list.appendChild(li);
    return { li, fill: li.querySelector('.fill'), pct: li.querySelector('.pct') };
  });
}

function clearPad() {
  padCtx.fillStyle = '#000';
  padCtx.fillRect(0, 0, pad.width, pad.height);
  preview.clearRect(0, 0, SIZE, SIZE);
  showResult(null);
  announceEl.textContent = '';
}

function showResult(probs) {
  const best = probs ? probs.indexOf(Math.max(...probs)) : -1;
  digitEl.textContent = probs ? best : '–';
  confidenceEl.textContent = probs ? `${(probs[best] * 100).toFixed(1)}% confident` : ' ';
  bars.forEach((bar, d) => {
    const p = probs ? probs[d] : 0;
    bar.fill.style.transform = `scaleX(${p})`;
    bar.pct.textContent = `${(p * 100).toFixed(p >= 0.995 || p === 0 ? 0 : 1)}%`;
    bar.li.classList.toggle('best', d === best);
  });
}

function drawPreview(input) {
  const img = preview.createImageData(SIZE, SIZE);
  for (let i = 0; i < input.length; i++) {
    const v = input[i] * 255;
    img.data.set([v, v, v, 255], i * 4);
  }
  preview.putImageData(img, 0, 0);
}

function predict() {
  const input = preprocess(padCtx.getImageData(0, 0, pad.width, pad.height));
  if (!input) {
    preview.clearRect(0, 0, SIZE, SIZE);
    showResult(null);
    return null;
  }
  drawPreview(input);
  if (!network) return null;
  const probs = softmax(forward(network, input));
  showResult(probs);
  return probs;
}

let framePending = false;
function schedulePredict() {
  if (framePending) return;
  framePending = true;
  requestAnimationFrame(() => {
    framePending = false;
    predict();
  });
}

let activePointer = null;
let last = null;

function toCanvas(e) {
  const rect = pad.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) * (pad.width / rect.width),
    y: (e.clientY - rect.top) * (pad.height / rect.height),
  };
}

function strokeTo(p) {
  padCtx.beginPath();
  padCtx.moveTo(last.x, last.y);
  padCtx.lineTo(p.x, p.y);
  padCtx.stroke();
  last = p;
}

pad.addEventListener('pointerdown', e => {
  if (activePointer !== null || e.button > 0) return;
  e.preventDefault();
  activePointer = e.pointerId;
  last = toCanvas(e);
  try { pad.setPointerCapture(e.pointerId); } catch {}
  strokeTo(last);
  schedulePredict();
});

pad.addEventListener('pointermove', e => {
  if (e.pointerId !== activePointer) return;
  const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
  for (const ev of events.length ? events : [e]) strokeTo(toCanvas(ev));
  schedulePredict();
});

function endStroke(e) {
  if (e.pointerId !== activePointer) return;
  activePointer = null;
  last = null;
  const probs = predict();
  if (probs) {
    const best = probs.indexOf(Math.max(...probs));
    announceEl.textContent = `Looks like a ${best}, ${Math.round(probs[best] * 100)} percent confident.`;
  }
}

pad.addEventListener('pointerup', endStroke);
pad.addEventListener('pointercancel', endStroke);
document.getElementById('clear').addEventListener('click', clearPad);

async function loadModel() {
  try {
    const res = await fetch('model.json');
    if (!res.ok) throw new Error(`model.json: HTTP ${res.status}`);
    const model = await res.json();
    const bin = await fetch(model.weights || 'weights.bin');
    if (!bin.ok) throw new Error(`${model.weights}: HTTP ${bin.status}`);
    network = parseModel(model, await bin.arrayBuffer());

    document.getElementById('depth').textContent = network.length;
    if (typeof model.test_accuracy === 'number') {
      const acc = document.getElementById('accuracy');
      acc.textContent = `Test accuracy: ${(model.test_accuracy * 100).toFixed(1)}%.`;
      acc.hidden = false;
    }
    statusEl.textContent = '';
    predict();
  } catch (err) {
    console.warn(err);
    statusEl.classList.add('error');
    statusEl.textContent = location.protocol === 'file:'
      ? 'Could not load the model. Browsers block fetch() on file:// pages, so serve this folder over HTTP (for example python -m http.server).'
      : `Could not load the model weights (${err.message}). Drawing still works, but there is nothing to predict with.`;
  }
}

padCtx.lineWidth = BRUSH;
padCtx.lineCap = 'round';
padCtx.lineJoin = 'round';
padCtx.strokeStyle = '#fff';
clearPad();
loadModel();
