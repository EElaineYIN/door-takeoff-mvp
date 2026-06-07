"""VLM semantic layer for the demo (Approach: YOLO locates, Claude understands).

For each YOLO door box we crop the region and ask Claude to describe what a
human estimator would read off the symbol:
  - door type      (swing / double / sliding / pocket / other)
  - swing direction (left / right / up / down / n/a)
  - is_door         (sanity check: is this actually a door?)

These attributes are NOT in the training GT, so they are qualitative — they
showcase the reasoning a detection model cannot do on its own.

Calls go through the same OpenAI-compatible inference proxy the project's
experiments used (model.service-inference.ai), NOT the native Anthropic SDK,
because the project's keys (sk-inf-v1-...) are proxy keys.
"""
from __future__ import annotations

import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

# Same proxy + model id the A1–A5 experiments used (see src/model_comparison.py).
BASE_URL = os.environ.get("INFERENCE_BASE_URL", "https://model.service-inference.ai/v1")
MODEL = os.environ.get("DEMO_VLM_MODEL", "claude-opus-4-8")
ENV_VAR = "ANTHROPIC_API_KEY"
MARGIN = 0.35          # expand crop by this fraction on each side for context
MAX_WORKERS = 6

_PROMPT = (
    "You are assisting an AEC quantity estimator. The image is a small crop from "
    "an architectural floor plan, centered on ONE door symbol that a detector found.\n"
    "Identify the door from the symbol (a quarter-circle swing arc = swing door, "
    "two facing arcs = double door, parallel lines sliding in a track = sliding, "
    "dashed rectangle in wall = pocket).\n"
    "Reply with ONLY a compact JSON object, no prose:\n"
    '{"is_door": true|false, "type": "swing|double|sliding|pocket|other", '
    '"direction": "left|right|up|down|n/a", "note": "<=6 words"}'
)


def _client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get(ENV_VAR, ""), base_url=BASE_URL)


def _crop_b64(img: Image.Image, box: dict) -> str:
    w, h = img.size
    bw = box["x2"] - box["x1"]
    bh = box["y2"] - box["y1"]
    mx, my = bw * MARGIN, bh * MARGIN
    x1 = max(0, int(box["x1"] - mx))
    y1 = max(0, int(box["y1"] - my))
    x2 = min(w, int(box["x2"] + mx))
    y2 = min(h, int(box["y2"] + my))
    crop = img.crop((x1, y1, x2, y2))
    longest = max(crop.size)
    if longest < 256:  # upscale tiny crops so the symbol is legible
        scale = 256 / longest
        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _classify_one(client, b64: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64}", "detail": "high"}},
                ],
            }],
            response_format={"type": "json_object"},
            max_tokens=200,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        data = json.loads(text)
        return {
            "is_door": bool(data.get("is_door", True)),
            "type": str(data.get("type", "swing")),
            "direction": str(data.get("direction", "n/a")),
            "note": str(data.get("note", "")),
        }
    except Exception as e:  # demo: degrade gracefully
        return {"is_door": True, "type": "swing", "direction": "n/a",
                "note": "vlm error", "error": str(e)[:120]}


def classify(image_path: str, boxes: list[dict]) -> list[dict]:
    """Classify each box. Returns list aligned with boxes input."""
    if not boxes:
        return []
    img = Image.open(image_path).convert("RGB")
    crops = [_crop_b64(img, b) for b in boxes]
    client = _client()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = list(ex.map(lambda c: _classify_one(client, c), crops))
    return results
