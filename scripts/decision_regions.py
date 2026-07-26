#!/usr/bin/env python3
"""
Render the trained ColorCLIP color encoder's decision regions as an
interactive 3D view of the RGB cube: every grid point is classified by the
model and painted its own true color, so each class's territory (and any
class with no territory at all) becomes visible.

Usage:
    python scripts/decision_regions.py --checkpoint <path/to/model.pth>
    python scripts/decision_regions.py --checkpoint <...>.pth --grid 48 --prior 1.0
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from ML.data_prep import DataLoader
from ML.embeddings import rgb_to_oklch
from ML.models.clip_models import ColorCLIPModel
from ML.utils import get_device

MAX_GRID = 64  # 64^3 = 262,144 points; above this the HTML payload gets unwieldy
BATCH = 8192


def build_grid(grid_n: int, color_space: str) -> np.ndarray:
    """(grid_n^3, 3) model-input array covering the RGB cube, in [0, 1]."""
    axis = np.linspace(0.0, 1.0, grid_n, dtype=np.float32)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    rgb01 = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1)

    if color_space == "oklch":
        return rgb_to_oklch(rgb01 * 255.0, normalize=True).astype(np.float32)
    return rgb01


def encode_in_batches(model, inputs: torch.Tensor, device) -> torch.Tensor:
    chunks = []
    with torch.no_grad():
        for i in range(0, len(inputs), BATCH):
            chunks.append(model.encode_color(inputs[i:i + BATCH].to(device)).cpu())
    return torch.cat(chunks)


def main():
    parser = argparse.ArgumentParser(description="Visualize ColorCLIP decision regions in 3D")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved *_color_clip_model.pth")
    parser.add_argument("--grid", type=int, default=32, help="Grid points per RGB axis (default 32)")
    parser.add_argument("--prior", type=float, default=0.0,
                        help="Prior-correction strength a: subtract a*log(train_freq) from the "
                             "logits before taking top-1 (0 = raw model, default)")
    parser.add_argument("--out", default=None, help="Output HTML path (default: next to --checkpoint)")
    args = parser.parse_args()

    if not (2 <= args.grid <= MAX_GRID):
        print(f"Error: --grid must be between 2 and {MAX_GRID} "
              f"({MAX_GRID}^3 = {MAX_GRID**3:,} points is already a lot for one HTML file).")
        return 1

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]

    print("Loading data bundle for class names, BoW table, and train frequencies...")
    bundle = DataLoader(config).load()
    label_encoder = bundle["label_encoder"]
    class_names = list(label_encoder.classes_)
    k = len(class_names)

    model_cfg = config["model"]
    model = ColorCLIPModel(
        vocab_size=bundle["vocab_size"],
        embed_dim=model_cfg["embed_dim"],
        color_hidden_dims=model_cfg.get("color_hidden_dims"),
        text_hidden_dims=model_cfg.get("text_hidden_dims"),
        text_encoder_type=model_cfg.get("text_encoder", "bow"),
        num_classes=k,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    class_bow = torch.FloatTensor(bundle["bow_embedder"].transform(class_names)).to(device)
    with torch.no_grad():
        prototypes = model.encode_text(class_bow).cpu()  # (K, D)

    train_freq = np.bincount(np.asarray(bundle["y_train"]), minlength=k).astype(np.float64)
    log_prior = torch.tensor(np.log(train_freq / train_freq.sum()), dtype=torch.float32)

    color_space = config["data"].get("color_space", "oklch")
    print(f"Building {args.grid}^3 = {args.grid**3:,} grid points ({color_space} input)...")
    grid_input = torch.FloatTensor(build_grid(args.grid, color_space))

    print("Classifying grid...")
    grid_embeds = encode_in_batches(model, grid_input, device)
    logits = grid_embeds @ prototypes.t()  # (N, K)

    top1_raw = logits.argmax(dim=1)
    top1_adjusted = (logits - args.prior * log_prior).argmax(dim=1)
    changed = (top1_adjusted != top1_raw)

    rgb255 = (build_grid(args.grid, "rgb") * 255).round().astype(np.uint8)  # true display color
    points = np.concatenate([
        rgb255,
        top1_adjusted.numpy().astype(np.int16)[:, None],
        changed.numpy().astype(np.int8)[:, None],
    ], axis=1).tolist()

    out_path = args.out or os.path.join(os.path.dirname(args.checkpoint), "decision_regions.html")
    write_html(out_path, points, class_names, args.prior)
    print(f"Wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return 0


def write_html(out_path: str, points: list, class_names: list, prior: float):
    payload = json.dumps({"points": points, "names": class_names, "prior": prior},
                         separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__PAYLOAD__", payload)
    with open(out_path, "w") as f:
        f.write(html)


HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>ColorCLIP decision regions</title>
<style>
  body { font-family: sans-serif; margin: 0; padding: 1rem; }
  #controls { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin-bottom: 8px; font-size: 13px; }
  canvas { width: 100%; max-width: 900px; display: block; cursor: grab; touch-action: none; }
  #hoverName { font-weight: 600; min-width: 120px; }
</style></head>
<body>
<div id="controls">
  <label>class: <select id="classSel"><option value="-1">All</option></select></label>
  <label><input type="checkbox" id="cbDiff"> diff mode (prior-adjusted vs raw)</label>
  <label><input type="checkbox" id="cbSpin" checked> spin</label>
  <span>zoom <input type="range" id="zoom" min="60" max="220" value="120" step="1"></span>
  <span id="hoverName"></span>
</div>
<canvas id="cv"></canvas>
<script>
const DATA = __PAYLOAD__;
const PRIOR = DATA.prior;
document.getElementById('cbDiff').parentElement.title =
    PRIOR === 0 ? 'no prior correction was applied (--prior 0) -- nothing will change' : '';

const sel = document.getElementById('classSel');
DATA.names.forEach((n, i) => {
  const o = document.createElement('option'); o.value = i; o.textContent = n; sel.appendChild(o);
});

const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
let W, H, DPR = Math.min(2, window.devicePixelRatio || 1);
function resize() { W = cv.clientWidth; H = Math.round(W * 0.72); cv.width = W * DPR; cv.height = H * DPR; cv.style.height = H + 'px'; ctx.setTransform(DPR, 0, 0, DPR, 0, 0); }

let yaw = 0.6, pitch = -0.35, zoom = 1.2, spin = true;
function project(x, y, z) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
  const y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
  const f = 320 / (320 + z2);
  return [W / 2 + x1 * zoom * f * (W / 340), H / 2 + y1 * zoom * f * (W / 340), z2, f];
}

let hoverIdx = -1;
function visible(p) {
  const classSel = parseInt(sel.value);
  if (classSel >= 0 && p[3] !== classSel) return false;
  if (document.getElementById('cbDiff').checked && p[4] === 0) return false;
  return true;
}

function draw() {
  ctx.clearRect(0, 0, W, H);
  const items = [];
  for (let i = 0; i < DATA.points.length; i++) {
    const p = DATA.points[i];
    if (!visible(p)) continue;
    const pos = [(p[0] / 255 - 0.5) * 200, (p[1] / 255 - 0.5) * 200, (p[2] / 255 - 0.5) * 200];
    const pj = project(...pos);
    items.push({ i, z: pj[2], x: pj[0], y: pj[1], f: pj[3], r: p[0], g: p[1], b: p[2] });
  }
  items.sort((a, b) => b.z - a.z);
  for (const it of items) {
    ctx.beginPath();
    ctx.arc(it.x, it.y, 2.6 * it.f, 0, 6.2832);
    ctx.fillStyle = `rgb(${it.r},${it.g},${it.b})`;
    ctx.fill();
  }
  return items;
}

let lastItems = [];
let dragging = false, lx = 0, ly = 0;
cv.addEventListener('pointerdown', e => { dragging = true; lx = e.clientX; ly = e.clientY; cv.style.cursor = 'grabbing'; cv.setPointerCapture(e.pointerId); document.getElementById('cbSpin').checked = false; spin = false; });
cv.addEventListener('pointermove', e => {
  if (dragging) { yaw += (e.clientX - lx) * 0.008; pitch += (e.clientY - ly) * 0.008; pitch = Math.max(-1.5, Math.min(1.5, pitch)); lx = e.clientX; ly = e.clientY; lastItems = draw(); }
  else {
    const rect = cv.getBoundingClientRect(); const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let best = -1, bd = 36;
    for (const it of lastItems) { const dx = it.x - mx, dy = it.y - my; const dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; best = it.i; } }
    const name = best >= 0 ? DATA.names[DATA.points[best][3]] : '';
    if (name !== document.getElementById('hoverName').textContent) document.getElementById('hoverName').textContent = name;
  }
});
cv.addEventListener('pointerup', () => { dragging = false; cv.style.cursor = 'grab'; });
document.getElementById('zoom').addEventListener('input', e => { zoom = e.target.value / 100; lastItems = draw(); });
document.getElementById('cbSpin').addEventListener('change', e => { spin = e.target.checked; });
sel.addEventListener('change', () => { lastItems = draw(); });
document.getElementById('cbDiff').addEventListener('change', () => { lastItems = draw(); });
function tick() { if (spin && !dragging) { yaw += 0.004; lastItems = draw(); } requestAnimationFrame(tick); }
window.addEventListener('resize', () => { resize(); lastItems = draw(); });
resize(); lastItems = draw(); tick();
</script>
</body></html>
"""


if __name__ == "__main__":
    sys.exit(main())
