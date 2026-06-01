"""Few-shot variant of swing-door detection (Step 3 of 1+2+3).

Bumps prompt to ``swing-v3-fs`` and prepends a synthetic example image
that visually demonstrates the arc-only bbox convention. The example
is shown to the VLM BEFORE the query image in a single multi-image
chat-completion call.

Why: phase-4 analysis showed anthropic / openai consistently produce
L-shape (leaf + arc) bboxes (h/w ≈ 1.5-1.7) even after the swing-v2
text-only "arc-only" instruction. A visual demonstration of the
convention is a much stronger signal than text alone.

The synthetic example PNG is generated once with PIL and cached at
``data/examples/few_shot_swing_v3.png`` so the bytes hash to the same
value across runs (essential for the disk cache in ``model_comparison``).

Usage::

    from src.swing_v3 import run_plan_swing_v3_model
    run = run_plan_swing_v3_model("anthropic", image_bytes)
"""

from __future__ import annotations

import base64
import io
import math
import os
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.model_comparison import (
    INFERENCE_BASE_URL,
    MODELS,
    PLAN_TILE_COLUMNS,
    ModelRun,
    ModelSpec,
    _door_plan_tile_rows,
    _run_model_raw,
)


DOOR_PLAN_SWING_V3_PROMPT_VERSION = "swing-v3-fs"

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "data" / "examples" / "few_shot_swing_v3.png"


# --------------------------------------------------------- few-shot example


def _draw_swing_door(draw: ImageDraw.ImageDraw, hinge_xy: tuple[int, int],
                     door_width: int, leaf_orient: str, swing_quadrant: str,
                     wall_thickness: int = 4, line_width: int = 2,
                     leaf_offset: int = 6) -> tuple[int, int, int, int]:
    """Draw one swing-door symbol; return the arc-only bbox (x1, y1, x2, y2).

    ``leaf_orient``  — "horizontal" or "vertical"; direction the closed leaf points.
    ``swing_quadrant`` — which quadrant the arc sweeps INTO:
                       "tl", "tr", "bl", "br" relative to the hinge.
    ``leaf_offset`` — pixels to nudge the leaf line OUTSIDE the arc bbox so it
                     remains visually distinct from the red bbox border (the
                     real convention has the leaf along the bbox edge; we cheat
                     a few px to make the example clearer to the VLM).
    """
    hx, hy = hinge_xy
    r = door_width

    # Per-quadrant geometry. The leaf is drawn slightly outside the arc bbox
    # along the edge "opposite" the arc curve so the L-shape is visually clear.
    if swing_quadrant == "br":
        arc_bbox = (hx, hy, hx + r, hy + r)
        # Leaf horizontal at top of bbox, nudged up by leaf_offset.
        ly = hy - leaf_offset
        leaf = [(hx, ly), (hx + r, ly)]
        wall_a = [(hx - 40, ly), (hx - 6, ly)]
        wall_b = [(hx + r + 6, ly), (hx + r + 40, ly)]
        a0, a1 = 0, 90
    elif swing_quadrant == "bl":
        arc_bbox = (hx - r, hy, hx, hy + r)
        # Leaf vertical at right of bbox, nudged right by leaf_offset.
        lx = hx + leaf_offset
        leaf = [(lx, hy), (lx, hy + r)]
        wall_a = [(lx, hy - 40), (lx, hy - 6)]
        wall_b = [(lx, hy + r + 6), (lx, hy + r + 40)]
        a0, a1 = 90, 180
    elif swing_quadrant == "tl":
        arc_bbox = (hx - r, hy - r, hx, hy)
        ly = hy + leaf_offset
        leaf = [(hx - r, ly), (hx, ly)]
        wall_a = [(hx - r - 40, ly), (hx - r - 6, ly)]
        wall_b = [(hx + 6, ly), (hx + 40, ly)]
        a0, a1 = 180, 270
    else:  # tr
        arc_bbox = (hx, hy - r, hx + r, hy)
        lx = hx - leaf_offset
        leaf = [(lx, hy - r), (lx, hy)]
        wall_a = [(lx, hy - r - 40), (lx, hy - r - 6)]
        wall_b = [(lx, hy + 6), (lx, hy + 40)]
        a0, a1 = 270, 360

    # Walls (thicker black) — give the door context.
    draw.line(wall_a, fill="black", width=wall_thickness)
    draw.line(wall_b, fill="black", width=wall_thickness)

    # Door leaf (thin black line, nudged slightly outside the arc bbox).
    draw.line(leaf, fill="black", width=line_width)

    # Quarter-circle arc.
    draw.arc((hx - r, hy - r, hx + r, hy + r),
             start=a0, end=a1, fill="black", width=line_width)

    return arc_bbox


def _build_few_shot_example_image() -> bytes:
    """Synthesize a single PNG with 3 swing doors + their CORRECT (arc-only) red bboxes.

    Layout: 720x520 white canvas, 3 doors arranged in a row, each with a
    different orientation. Each door has a thin red square drawn TIGHTLY
    around just the quarter-circle arc — demonstrating the convention.
    """
    W, H = 720, 520
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    door_w = 80  # door width in px (typical of a 700-wide tile)

    # Three doors at different orientations.
    bboxes: list[tuple[int, int, int, int]] = []

    # Door 1 — top-left area, leaf horizontal, swing into bottom-right quadrant.
    b = _draw_swing_door(draw, hinge_xy=(80, 130), door_width=door_w,
                         leaf_orient="horizontal", swing_quadrant="br")
    bboxes.append(b)

    # Door 2 — middle, leaf vertical, swing into bottom-left quadrant.
    b = _draw_swing_door(draw, hinge_xy=(380, 130), door_width=door_w,
                         leaf_orient="vertical", swing_quadrant="bl")
    bboxes.append(b)

    # Door 3 — right side, leaf horizontal, swing into top-right quadrant.
    b = _draw_swing_door(draw, hinge_xy=(560, 270), door_width=door_w,
                         leaf_orient="horizontal", swing_quadrant="tr")
    bboxes.append(b)

    # Draw the red arc-only bboxes (the convention demo).
    for (x1, y1, x2, y2) in bboxes:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    # Caption.
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    draw.text((20, 380), "EXAMPLE — Three swing doors with the CORRECT arc-only bbox in RED.",
              fill="black", font=font)
    draw.text((20, 410), "Each bbox encloses ONLY the quarter-circle arc (not the leaf line).",
              fill="red", font=font)
    draw.text((20, 435), "Bbox shape: SQUARE (h/w ≈ 1.0), about one door-width on each side.",
              fill="black", font=small)
    draw.text((20, 460), "When labelling the query image, follow the SAME convention.",
              fill="black", font=small)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def few_shot_example_bytes() -> bytes:
    """Return the cached PNG bytes; build + persist on first call."""
    if EXAMPLE_PATH.exists():
        return EXAMPLE_PATH.read_bytes()
    EXAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _build_few_shot_example_image()
    EXAMPLE_PATH.write_bytes(data)
    return data


# ------------------------------------------------------------- prompt + call

def build_swing_v3_prompt() -> str:
    return """You are looking at an architectural FLOOR PLAN (or a tile cropped from one). Find every SWING DOOR visible in the image.

There are TWO IMAGES attached to this prompt:
  - FIRST IMAGE = EXAMPLE. It shows three swing doors with their CORRECT bounding boxes drawn in RED. Each red bbox tightly encloses ONLY the quarter-circle arc — NOT the straight door-leaf line. The bboxes are SQUARE (h/w ≈ 1.0), about one door-width on each side.
  - SECOND IMAGE = QUERY. Apply the SAME bbox convention to this image and return every swing door you can find.

DEFINITION OF A SWING DOOR
A swing door is drawn as TWO graphical elements that always appear together inside a wall opening:
  (a) STRAIGHT LINE = the door leaf (panel itself, one door-width long).
  (b) QUARTER-CIRCLE ARC = the swing path, from hinge to leaf tip.

CRITICAL — your bbox MUST tightly enclose ONLY the arc (the curve), NOT the leaf line. The bbox should be SQUARE, approximately one door-width on each side (typically ~25-35 px on a 700-px-wide plan, ~50-70 px on a 1700-px-wide plan). See the red boxes in the EXAMPLE.

DO NOT report any of these (no arc):
  - Bifold doors (zigzag of two short panels)
  - Bypass / sliding doors (two parallel lines)
  - Pocket doors (single line in a thickened wall)
  - Stair direction arrows, furniture arcs (chairs, fans, sinks), window sashes

For a DOUBLE swing door (two leaves drawn mirrored), report TWO detections — one per leaf.

Return ONLY this JSON object (no commentary, no markdown fences):

{
  "doors": [
    {
      "mark": "door",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.85
    }
  ]
}

Where:
  - ``mark`` is always the literal string ``"door"``.
  - ``bbox`` is normalized to [0, 1] of the QUERY IMAGE (the SECOND image), tightly enclosing ONLY the arc curve — h/w ≈ 1.0.
  - ``confidence`` ∈ [0, 1].

If no swing doors are visible in the QUERY image, return {"doors": []}.
Output must be valid JSON only — no leading or trailing text.
"""


def _call_few_shot_proxy(
    image_bytes: bytes, prompt: str, model_id: str, env_var: str
) -> str:
    """Multi-image call: [example image, query image] in one user message."""
    from openai import OpenAI  # type: ignore
    api_key = os.environ.get(env_var, "")
    client = OpenAI(api_key=api_key, base_url=INFERENCE_BASE_URL)
    example_b64 = base64.b64encode(few_shot_example_bytes()).decode("ascii")
    query_b64 = base64.b64encode(image_bytes).decode("ascii")
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{example_b64}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{query_b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=16384,
    )
    return resp.choices[0].message.content or ""


# Register v3 specs (= same model IDs / keys, but use few-shot call).
MODELS_V3: dict[str, ModelSpec] = {
    k: ModelSpec(
        key=spec.key,
        label=spec.label,
        model_id=spec.model_id,
        env_var=spec.env_var,
        call=_call_few_shot_proxy,
    )
    for k, spec in MODELS.items()
}


def run_plan_swing_v3_model(
    model_key: str,
    image_bytes: bytes,
    use_cache: bool = True,
) -> ModelRun:
    """Call one model with the swing-v3 few-shot prompt.

    Cache key includes ``swing-v3-fs`` so it is independent of swing-v2.
    Output DataFrame mirrors ``run_plan_swing_model``.
    """
    if model_key not in MODELS_V3:
        return ModelRun(
            spec=MODELS.get(
                model_key,
                ModelSpec(model_key, model_key, "", "", _call_few_shot_proxy),
            ),
            df=pd.DataFrame(columns=PLAN_TILE_COLUMNS),
            error=f"Unknown model {model_key!r}",
        )
    spec = MODELS_V3[model_key]
    prompt = build_swing_v3_prompt()
    raw, cached, elapsed, err = _run_model_raw(
        spec,
        image_bytes,
        prompt,
        DOOR_PLAN_SWING_V3_PROMPT_VERSION,
        kind="door_plan_swing",
        use_cache=use_cache,
    )
    if err:
        return ModelRun(
            spec=spec, df=pd.DataFrame(columns=PLAN_TILE_COLUMNS),
            error=err, cached=cached, elapsed_s=elapsed,
        )
    rows = _door_plan_tile_rows(raw, spec.label)
    return ModelRun(
        spec=spec, df=pd.DataFrame(rows, columns=PLAN_TILE_COLUMNS),
        cached=cached, elapsed_s=elapsed, raw_response=raw,
    )


if __name__ == "__main__":
    # Eagerly create the example image so it can be inspected.
    data = few_shot_example_bytes()
    print(f"Few-shot example PNG: {EXAMPLE_PATH} ({len(data)} bytes)")
