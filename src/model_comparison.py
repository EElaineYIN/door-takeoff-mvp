"""Multi-model extraction for the door takeoff.

Two extraction kinds share the same API/cache machinery:

* ``run_door_schedule_model`` — given the A10.2 left-third crop, ask each
  model to extract every door row (mark, type, size, material, fire rating,
  hardware set, notes). Used by Tab 1.
* ``run_plan_detection_model`` — given an entire floor plan image, ask
  each model to find every door tag and report its bounding box (normalized
  0..1). Used by Tab 2.

Both use OpenAI gpt-4o, Anthropic Claude Sonnet 4.6, and Google Gemini 2.5
Flash. Each model's output is parsed into a standardised pandas DataFrame.

Design notes
------------
- API key absence is graceful: ``ModelRun.error`` carries a human message
  and ``ModelRun.df`` is empty.
- Disk cache under ``outputs/.cache/llm/`` keyed by
  ``(model, kind, image_sha256, prompt_version)`` so repeated clicks of
  "Run all models" cost nothing and the two prompt kinds never collide.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


# --------------------------------------------------------------------- prompts

DOOR_SCHEDULE_PROMPT_VERSION = "v1"

DOOR_SCHEDULE_PROMPT = """You are analyzing the DOOR SCHEDULE table from an architectural construction drawing.

Extract every door row visible in the schedule in this image. Return ONLY a single
JSON object (no commentary, no markdown fences) with this shape:

{
  "doors": [
    {
      "mark": "string — the door mark/tag (e.g. '01', '101A', 'D1')",
      "type": "string — door type code or name (e.g. 'A', 'TYPE 1', 'FLUSH')",
      "size": "string — width x height as printed (e.g. \\"3'-0\\\" x 7'-0\\\"\\")",
      "material": "string — material code or name (e.g. 'WD', 'HM', 'WOOD', 'HOLLOW METAL')",
      "fire_rating": "string — fire rating (e.g. '20 MIN', '90 MIN'). Empty string if not rated.",
      "hardware_set": "string — hardware set code (e.g. 'HW-1', 'A'). Empty if not shown.",
      "notes": "string — any remaining text from the row"
    }
  ]
}

Rules:
- Only include doors you can actually see in the schedule. Do NOT invent doors.
- Skip header rows ('MARK', 'TYPE', 'SIZE', etc.).
- Use an empty string ("") for fields you cannot read clearly.
- The schedule may contain dozens of doors; list every one in order.
- Output must be valid JSON only — no leading or trailing text.
"""


DOOR_PLAN_PROMPT_VERSION = "v1"


def build_door_plan_prompt(vocab: list[str] | None = None) -> str:
    """Build the floor-plan detection prompt, optionally biasing with vocab."""
    if vocab:
        vocab_block = (
            "The KNOWN door marks for this project are:\n  "
            + ", ".join(sorted(set(vocab)))
            + "\n\nOnly report marks from this list (or close OCR variants — "
            "e.g. 'O1' should be reported as '01'). Drop any tag that does "
            "not match the list.\n"
        )
    else:
        vocab_block = (
            "Door marks typically look like 1–3 digits optionally followed "
            "by a letter (e.g. '01', '01A', '101A', 'D-101').\n"
        )

    return f"""You are looking at an architectural FLOOR PLAN. Each door on the plan is labelled with a small text tag (e.g. '01', '01A', 'D-101') placed next to the door symbol.

Find every door tag visible on this plan. List EVERY occurrence — the same mark CAN appear many times (one per door on the building); each occurrence is its own JSON entry.

{vocab_block}
Return ONLY this JSON (no commentary, no markdown fences):

{{
  "doors": [
    {{
      "mark": "01",
      "bbox": [x1, y1, x2, y2]
    }}
  ]
}}

Where ``bbox`` is normalized to [0, 1] with:
  - x1, y1 = top-left corner (x1 < x2, y1 < y2)
  - x2, y2 = bottom-right corner

If you truly cannot determine a tight bbox, still list the mark and use [0, 0, 0, 0].
Output must be valid JSON only — no leading or trailing text.
"""


# Build the version-1 prompt once at import for caching purposes. The function
# above is still available for callers that want to pass a vocab hint.
DOOR_PLAN_PROMPT_DEFAULT = build_door_plan_prompt(vocab=None)


# --------------------------------------------------- tile-based plan detection

# A separate prompt + cache namespace for the tile pipeline. Each tile is a
# small (~2000×2000 px) crop of one floor plan, sent to the VLM at full
# resolution so individual tags stay legible. The prompt is much stricter
# than the whole-page one — a frequent failure mode there was VLMs treating
# room numbers / dimension callouts / keynote bubbles as door tags.
#
# v2: confirmed from the drawing's own legend — door tags in A2.x are
# HEXAGONAL symbols containing the mark. v1 was symbol-agnostic and let
# GPT match room rectangles ("MAIL 108") while Claude matched wall-type
# squares ("32 T"). v2 describes the hexagon explicitly and spells out the
# four shape-lookalikes that must be ignored.
#
# v3: added per-sheet prompt routing. A7.1.1 (Signage – Roof Deck) uses
# a totally different convention — each door is marked with a SIGNAGE
# label like "1.11" / "7.15" printed next to a black filled circle (•)
# at the door swing arc, with NO enclosing hexagon. v3 lets the caller
# pass ``sheet_no`` so the right symbol description goes to the model.
DOOR_PLAN_TILE_PROMPT_VERSION = "v3"


def _door_plan_tile_prompt_hexagon(vocab_block: str) -> str:
    """A2.x convention — hexagon-enclosed alphanumeric marks."""
    return f"""You are looking at a TILE cropped from an architectural floor plan. Find every DOOR visible on this tile.

CRITICAL — DOOR TAG SHAPE
In this drawing set, a door tag is a **HEXAGON** symbol containing an alphanumeric mark (e.g. a hexagon with "101" inside). The drawing's own legend states: "<hexagon>101 — DOOR — SEE DOOR SCHEDULE". The hexagon IS what makes it a door tag.

Only report a detection when ALL of these are true:
  1. The mark is enclosed in a hexagonal symbol (six sides — NOT a square, NOT a circle, NOT a diamond).
  2. The hexagon sits next to a door swing arc (90° curved line) inside a wall opening.
  3. The text inside the hexagon matches the schedule vocabulary below.

DO NOT report any of these — they often look numeric but are NOT door tags:
  - **Wall type tags**: small SQUARE / RECTANGLE containing a number plus a trailing letter (e.g. "32 T", "31 T", "39 A").
  - **Room numbers**: numbers inside a RECTANGLE that ALSO carries a room name (e.g. "MAIL / 108", "WC / 107").
  - **Detail / section references**: numbers inside a CIRCLE with a horizontal cross-hair, paired with a sheet ref (e.g. "1 / A4.6").
  - **Grid line markers**: single letters or numbers inside CIRCLES at the page margins (A, B, E.7, 1, 2, 5', 9, …).
  - **Keynote bubbles**: small numbers in plain circles referencing the KEYNOTES table.
  - **Dimension callouts**: numbers along dashed dimension lines.

Rule of thumb: if the enclosing shape is NOT a hexagon, it is NOT a door tag — even if the number matches a schedule mark.

{vocab_block}
Return ONLY this JSON object (no commentary, no markdown fences):

{{
  "doors": [
    {{
      "mark": "01",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.85
    }}
  ]
}}

Where:
  - ``bbox`` is normalized to [0, 1] of THIS TILE (not the full plan):
      x1, y1 = top-left of the door HEXAGON, x1 < x2, y1 < y2
      x2, y2 = bottom-right of the door HEXAGON
  - ``confidence`` ∈ [0, 1] — your subjective certainty this is a real door (a hexagon next to a swing arc).

If no hexagonal door tags are visible, return {{"doors": []}}.
Output must be valid JSON only — no leading or trailing text.
"""


def _door_plan_tile_prompt_signage(vocab_block: str) -> str:
    """A7.1.1 convention — text label next to filled dot + door swing arc.

    Sheet A7.1.1 ("SIGNAGE – ROOF DECK PLAN") marks every door with a
    plain text **SIGNAGE label** of the form ``<digits>.<digits>`` such
    as ``1.11``, ``7.15``, ``3.3``, ``6.1``, ``6.2``, ``7.16``. The
    label is printed right next to a small **black filled circle (•)**
    that sits at the hinge end of a door swing arc (a quarter-circle
    drawn from the wall opening).
    """
    return f"""You are looking at a TILE cropped from sheet A7.1.1 — a "SIGNAGE" plan. Find every DOOR visible on this tile.

CRITICAL — DOOR TAG CONVENTION (A7.1.1 signage plan)
On this sheet, every door is marked by THREE visual elements clustered together:
  (a) a small **black FILLED CIRCLE (•)** roughly the size of a period, sitting at the hinge end of a door swing,
  (b) a **door swing arc** — a thin quarter-circle drawn from the wall opening,
  (c) a plain text **SIGNAGE LABEL** printed next to the dot, of the form ``<digits>.<digits>`` (examples: ``1.11``, ``7.15``, ``7.16``, ``3.3``, ``6.1``, ``6.2``).

Report ONE detection per door cluster. The mark is the signage label text. There is NO enclosing hexagon, NO square, NO circle around the label — the label is plain text on the drawing.

DO NOT report any of these (common false positives on this sheet):
  - **Dimension callouts**: numbers along thin dashed dimension lines, often with a tick or arrow.
  - **Detail / section references**: numbers inside a CIRCLE with a cross-hair (e.g. ``3`` inside a circle pointing at ``A7.1.1`` at the bottom of the page).
  - **Grid markers** at the drawing margins (letters or numbers inside circles).
  - **Generic black dots** that are NOT next to a curved door swing arc (planter dots, gravel pattern, fixture symbols).
  - Any label that is NOT in the ``<digits>.<digits>`` format above.

Rule of thumb: a real door cluster = filled dot + quarter-circle arc + ``N.NN`` label, all three within ~50 px of each other.

{vocab_block}
Return ONLY this JSON object (no commentary, no markdown fences):

{{
  "doors": [
    {{
      "mark": "7.15",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.85
    }}
  ]
}}

Where:
  - ``bbox`` is normalized to [0, 1] of THIS TILE (not the full plan), and should tightly enclose the FILLED DOT plus its adjacent SIGNAGE LABEL (you can ignore the arc itself when sizing the box):
      x1, y1 = top-left,  x1 < x2, y1 < y2
      x2, y2 = bottom-right
  - ``confidence`` ∈ [0, 1] — your subjective certainty this is a real door cluster (dot + arc + N.NN label).

If no door clusters are visible, return {{"doors": []}}.
Output must be valid JSON only — no leading or trailing text.
"""


def build_door_plan_tile_prompt(
    vocab: list[str] | None = None,
    sheet_no: str | None = None,
) -> str:
    """Build the per-tile floor-plan detection prompt with negative examples.

    The symbol convention is sheet-specific:
      * A2.x — hexagon-enclosed alphanumeric door marks
      * A7.1.1 — filled dot + arc + signage label (``N.NN`` format)

    Pass ``sheet_no`` so the right prompt body is selected. If unknown,
    we default to the hexagon convention (preserves prior behavior).
    """
    if vocab:
        vocab_block = (
            "The KNOWN marks for this sheet are:\n  "
            + ", ".join(sorted(set(vocab)))
            + "\n\nOnly report a mark if it appears in this list (or is a "
            "close OCR variant). Drop anything else.\n"
        )
    else:
        if sheet_no == "A7.1.1":
            vocab_block = (
                "Signage labels on this sheet follow the format "
                "``<digits>.<digits>`` (e.g. ``1.11``, ``7.15``, ``6.2``).\n"
            )
        else:
            vocab_block = (
                "Door marks typically look like 1–3 digits optionally followed "
                "by a letter (e.g. '01', '12A', '101A', 'D-101').\n"
            )

    if sheet_no == "A7.1.1":
        return _door_plan_tile_prompt_signage(vocab_block)
    return _door_plan_tile_prompt_hexagon(vocab_block)


# Output schema for tile detections (whole-page schema + a confidence column).
PLAN_TILE_COLUMNS = [
    "mark", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "confidence", "model",
]


def _door_plan_tile_rows(text: str, model_label: str) -> list[dict]:
    """Parse the tile-prompt JSON response into rows for a DataFrame."""
    obj = _extract_json_object(text)
    if not obj:
        return []
    doors = obj.get("doors", obj.get("detections", obj.get("rows", [])))
    if not isinstance(doors, list):
        return []
    out: list[dict] = []
    for raw in doors:
        if not isinstance(raw, dict):
            continue
        bbox = _coerce_bbox(raw.get("bbox") or raw.get("box"))
        if bbox is None:
            bbox = (0.0, 0.0, 0.0, 0.0)
        x1, y1, x2, y2 = bbox
        try:
            conf = float(raw.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        out.append(
            {
                "mark": str(raw.get("mark", "")).strip(),
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "confidence": conf,
                "model": model_label,
            }
        )
    return out


# ------------------------------------------------------------------- registry

@dataclass(frozen=True)
class ModelSpec:
    key: str            # short id used as cache prefix and dict key
    label: str          # human label for UI ("OpenAI gpt-4o" etc.)
    model_id: str       # exact API model identifier
    env_var: str        # env var name holding the API key
    call: Callable[[bytes, str, str], str]  # (image_bytes, prompt, model_id) -> raw text


def _call_openai(image_bytes: bytes, prompt: str, model_id: str) -> str:
    from openai import OpenAI  # type: ignore
    client = OpenAI()
    b64 = base64.b64encode(image_bytes).decode("ascii")
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
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=4096,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(image_bytes: bytes, prompt: str, model_id: str) -> str:
    from anthropic import Anthropic  # type: ignore
    client = Anthropic()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    resp = client.messages.create(
        model=model_id,
        max_tokens=4096,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "\n".join(parts)


def _call_gemini(image_bytes: bytes, prompt: str, model_id: str) -> str:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    client = genai.Client()
    resp = client.models.generate_content(
        model=model_id,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
        config={
            "response_mime_type": "application/json",
            "temperature": 0.0,
            "max_output_tokens": 4096,
        },
    )
    return getattr(resp, "text", "") or ""


MODELS: dict[str, ModelSpec] = {
    "openai": ModelSpec(
        key="openai",
        label="OpenAI gpt-4o",
        model_id="gpt-4o",
        env_var="OPENAI_API_KEY",
        call=_call_openai,
    ),
    "anthropic": ModelSpec(
        key="anthropic",
        label="Anthropic Claude Sonnet 4.6",
        model_id="claude-sonnet-4-6",
        env_var="ANTHROPIC_API_KEY",
        call=_call_anthropic,
    ),
    "gemini": ModelSpec(
        key="gemini",
        label="Google Gemini 2.5 Flash",
        model_id="gemini-2.5-flash",
        env_var="GOOGLE_API_KEY",
        call=_call_gemini,
    ),
}


# ------------------------------------------------------------- response parsing

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    lo, hi = text.find("{"), text.rfind("}")
    if 0 <= lo < hi:
        try:
            return json.loads(text[lo : hi + 1])
        except Exception:
            pass
    return None


def _door_schedule_rows(text: str, model_label: str) -> list[dict]:
    obj = _extract_json_object(text)
    if not obj:
        return []
    doors = obj.get("doors", obj.get("rows", []))
    if not isinstance(doors, list):
        return []
    out = []
    for raw in doors:
        if not isinstance(raw, dict):
            continue
        out.append(
            {
                "mark": str(raw.get("mark", "")).strip(),
                "type": str(raw.get("type", "")).strip(),
                "size": str(raw.get("size", "")).strip(),
                "material": str(raw.get("material", "")).strip(),
                "fire_rating": str(raw.get("fire_rating", "")).strip(),
                "hardware_set": str(raw.get("hardware_set", "")).strip(),
                "notes": str(raw.get("notes", "")).strip(),
                "model": model_label,
            }
        )
    return out


def _coerce_bbox(raw_bbox) -> tuple[float, float, float, float] | None:
    """Accept ``[x1, y1, x2, y2]`` (normalized 0..1) or a dict with the same keys."""
    if raw_bbox is None:
        return None
    if isinstance(raw_bbox, dict):
        try:
            x1 = float(raw_bbox.get("x1", raw_bbox.get("left", 0)))
            y1 = float(raw_bbox.get("y1", raw_bbox.get("top", 0)))
            x2 = float(raw_bbox.get("x2", raw_bbox.get("right", 0)))
            y2 = float(raw_bbox.get("y2", raw_bbox.get("bottom", 0)))
        except (TypeError, ValueError):
            return None
    elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
        try:
            x1, y1, x2, y2 = (float(v) for v in raw_bbox[:4])
        except (TypeError, ValueError):
            return None
    else:
        return None
    # Clamp to [0, 1] and normalize order.
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return (x1, y1, x2, y2)


def _door_plan_rows(text: str, model_label: str) -> list[dict]:
    obj = _extract_json_object(text)
    if not obj:
        return []
    doors = obj.get("doors", obj.get("detections", obj.get("rows", [])))
    if not isinstance(doors, list):
        return []
    out = []
    for raw in doors:
        if not isinstance(raw, dict):
            continue
        bbox = _coerce_bbox(raw.get("bbox") or raw.get("box"))
        if bbox is None:
            bbox = (0.0, 0.0, 0.0, 0.0)
        x1, y1, x2, y2 = bbox
        out.append(
            {
                "mark": str(raw.get("mark", "")).strip(),
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "model": model_label,
            }
        )
    return out


# ------------------------------------------------------------------- caching

CACHE_DIR = Path(__file__).resolve().parent.parent / "outputs" / ".cache" / "llm"


def _cache_path(model_key: str, kind: str, image_bytes: bytes, version: str) -> Path:
    h = hashlib.sha256(image_bytes).hexdigest()[:16]
    return CACHE_DIR / f"{model_key}__{kind}__{h}__{version}.json"


def _read_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


# ------------------------------------------------------------------- public API

DOOR_COLUMNS = [
    "mark", "type", "size", "material", "fire_rating",
    "hardware_set", "notes", "model",
]

PLAN_COLUMNS = [
    "mark", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "model",
]


@dataclass
class ModelRun:
    spec: ModelSpec
    df: pd.DataFrame
    error: str = ""
    cached: bool = False
    elapsed_s: float = 0.0
    raw_response: str = ""


def _run_model_raw(
    spec: ModelSpec,
    image_bytes: bytes,
    prompt: str,
    prompt_version: str,
    kind: str,
    use_cache: bool,
) -> tuple[str, bool, float, str]:
    """Run one model, returning ``(raw_text, cached, elapsed_s, error)``.

    The cache filename includes ``kind`` so the door-schedule and plan-detection
    prompts don't collide on the same image hash.
    """
    cache_path = _cache_path(spec.key, kind, image_bytes, prompt_version)
    if use_cache:
        cached = _read_cache(cache_path)
        if cached and cached.get("prompt_version") == prompt_version:
            return (
                cached.get("raw_response", ""),
                True,
                float(cached.get("elapsed_s", 0.0)),
                "",
            )

    if not os.environ.get(spec.env_var):
        return ("", False, 0.0, f"{spec.env_var} is not set — add it to .env and restart.")

    t0 = time.time()
    try:
        raw = spec.call(image_bytes, prompt, spec.model_id)
    except Exception as exc:  # noqa: BLE001
        return ("", False, time.time() - t0, f"{type(exc).__name__}: {exc}")
    elapsed = time.time() - t0

    _write_cache(
        cache_path,
        {
            "model": spec.key,
            "model_id": spec.model_id,
            "kind": kind,
            "prompt_version": prompt_version,
            "raw_response": raw,
            "elapsed_s": elapsed,
        },
    )
    return raw, False, elapsed, ""


def run_door_schedule_model(
    model_key: str, image_bytes: bytes, use_cache: bool = True
) -> ModelRun:
    """Call one model with the A10.2 door-schedule prompt."""
    if model_key not in MODELS:
        return ModelRun(
            spec=MODELS.get(model_key, ModelSpec(model_key, model_key, "", "", _call_openai)),
            df=pd.DataFrame(columns=DOOR_COLUMNS),
            error=f"Unknown model {model_key!r}",
        )
    spec = MODELS[model_key]
    raw, cached, elapsed, err = _run_model_raw(
        spec, image_bytes,
        DOOR_SCHEDULE_PROMPT, DOOR_SCHEDULE_PROMPT_VERSION,
        kind="door_schedule", use_cache=use_cache,
    )
    if err:
        return ModelRun(
            spec=spec, df=pd.DataFrame(columns=DOOR_COLUMNS),
            error=err, cached=cached, elapsed_s=elapsed,
        )
    rows = _door_schedule_rows(raw, spec.label)
    return ModelRun(
        spec=spec, df=pd.DataFrame(rows, columns=DOOR_COLUMNS),
        cached=cached, elapsed_s=elapsed, raw_response=raw,
    )


# Back-compat alias — Tab 1 callers used to import ``run_model``.
run_model = run_door_schedule_model


def run_plan_detection_model(
    model_key: str,
    image_bytes: bytes,
    vocab: list[str] | None = None,
    use_cache: bool = True,
) -> ModelRun:
    """Call one model with the floor-plan detection prompt.

    Output DataFrame uses ``PLAN_COLUMNS`` (mark + normalized bbox + model).
    """
    if model_key not in MODELS:
        return ModelRun(
            spec=MODELS.get(model_key, ModelSpec(model_key, model_key, "", "", _call_openai)),
            df=pd.DataFrame(columns=PLAN_COLUMNS),
            error=f"Unknown model {model_key!r}",
        )
    spec = MODELS[model_key]
    prompt = build_door_plan_prompt(vocab=vocab) if vocab else DOOR_PLAN_PROMPT_DEFAULT
    raw, cached, elapsed, err = _run_model_raw(
        spec, image_bytes,
        prompt, DOOR_PLAN_PROMPT_VERSION,
        kind="door_plan", use_cache=use_cache,
    )
    if err:
        return ModelRun(
            spec=spec, df=pd.DataFrame(columns=PLAN_COLUMNS),
            error=err, cached=cached, elapsed_s=elapsed,
        )
    rows = _door_plan_rows(raw, spec.label)
    return ModelRun(
        spec=spec, df=pd.DataFrame(rows, columns=PLAN_COLUMNS),
        cached=cached, elapsed_s=elapsed, raw_response=raw,
    )


def run_all_door_schedule(
    image_bytes: bytes, use_cache: bool = True
) -> dict[str, ModelRun]:
    return {k: run_door_schedule_model(k, image_bytes, use_cache=use_cache) for k in MODELS}


def run_all_plan_detection(
    image_bytes: bytes, vocab: list[str] | None = None, use_cache: bool = True
) -> dict[str, ModelRun]:
    return {
        k: run_plan_detection_model(k, image_bytes, vocab=vocab, use_cache=use_cache)
        for k in MODELS
    }


def run_plan_tile_model(
    model_key: str,
    image_bytes: bytes,
    vocab: list[str] | None = None,
    use_cache: bool = True,
    sheet_no: str | None = None,
) -> ModelRun:
    """Call one model with the per-tile floor-plan detection prompt.

    The output DataFrame uses ``PLAN_TILE_COLUMNS`` (mark + normalized bbox
    of the TILE + confidence + model). Convert with ``tile_detections_as_pixel_df``
    to translate into tile-pixel coords, then ``plan_tiling.shift_pixel_detections``
    to land in plan-region coords.

    ``sheet_no`` selects the per-sheet prompt body (A2.x: hexagon;
    A7.1.1: filled dot + arc + signage label).
    """
    if model_key not in MODELS:
        return ModelRun(
            spec=MODELS.get(model_key, ModelSpec(model_key, model_key, "", "", _call_openai)),
            df=pd.DataFrame(columns=PLAN_TILE_COLUMNS),
            error=f"Unknown model {model_key!r}",
        )
    spec = MODELS[model_key]
    prompt = build_door_plan_tile_prompt(vocab=vocab, sheet_no=sheet_no)
    raw, cached, elapsed, err = _run_model_raw(
        spec, image_bytes,
        prompt, DOOR_PLAN_TILE_PROMPT_VERSION,
        kind="door_plan_tile", use_cache=use_cache,
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


def run_all_plan_tile_detection(
    image_bytes: bytes,
    vocab: list[str] | None = None,
    use_cache: bool = True,
    sheet_no: str | None = None,
) -> dict[str, ModelRun]:
    """Run every registered model against a single tile."""
    return {
        k: run_plan_tile_model(
            k, image_bytes, vocab=vocab, use_cache=use_cache, sheet_no=sheet_no,
        )
        for k in MODELS
    }


# Back-compat alias.
run_all_models = run_all_door_schedule


def tesseract_baseline_as_df(tesseract_df: pd.DataFrame) -> pd.DataFrame:
    """Re-shape the Tesseract door extraction into the door-schedule comparison schema."""
    if tesseract_df is None or tesseract_df.empty:
        return pd.DataFrame(columns=DOOR_COLUMNS)
    out = pd.DataFrame()
    out["mark"] = tesseract_df.get("mark", "")
    out["type"] = tesseract_df.get("type", "")
    out["size"] = tesseract_df.get("size", "")
    notes = tesseract_df.get("notes", pd.Series([""] * len(tesseract_df))).fillna("")
    out["material"] = ""
    out["fire_rating"] = ""
    out["hardware_set"] = ""
    out["notes"] = notes
    out["model"] = "Tesseract (baseline)"
    return out[DOOR_COLUMNS]


def plan_detections_as_pixel_df(
    plan_model_df: pd.DataFrame,
    source_sheet: str,
    source_page: int,
    image_width: int,
    image_height: int,
) -> pd.DataFrame:
    """Convert a VLM's per-detection rows (normalized bbox) into the same pixel
    schema produced by ``plan_detector.detect_marks_on_plan``.

    Lets the reconciliation logic consume VLM + Tag-OCR detections uniformly.
    """
    cols = [
        "mark", "source_sheet", "source_page",
        "bbox_left", "bbox_top", "bbox_width", "bbox_height",
        "image_width", "image_height", "confidence", "detector",
    ]
    if plan_model_df is None or plan_model_df.empty:
        return pd.DataFrame(columns=cols)

    out_rows: list[dict] = []
    for _, r in plan_model_df.iterrows():
        mark = str(r.get("mark", "")).strip()
        if not mark:
            continue
        x1 = float(r.get("bbox_x1", 0.0))
        y1 = float(r.get("bbox_y1", 0.0))
        x2 = float(r.get("bbox_x2", 0.0))
        y2 = float(r.get("bbox_y2", 0.0))
        left = int(round(x1 * image_width))
        top = int(round(y1 * image_height))
        width = max(0, int(round((x2 - x1) * image_width)))
        height = max(0, int(round((y2 - y1) * image_height)))
        out_rows.append(
            {
                "mark": mark,
                "source_sheet": source_sheet,
                "source_page": source_page,
                "bbox_left": left,
                "bbox_top": top,
                "bbox_width": width,
                "bbox_height": height,
                "image_width": image_width,
                "image_height": image_height,
                "confidence": 100.0,  # VLMs don't return a confidence — assume 100.
                "detector": str(r.get("model", "vlm")),
            }
        )
    return pd.DataFrame(out_rows, columns=cols)


def tile_detections_as_pixel_df(
    tile_model_df: pd.DataFrame,
    source_sheet: str,
    source_page: int,
    tile_width: int,
    tile_height: int,
    region_label: str = "",
) -> pd.DataFrame:
    """Convert a tile prompt's per-detection rows into the standard pixel schema.

    The bbox is in TILE coordinates (still relative to the tile's own size).
    Use ``plan_tiling.shift_pixel_detections`` afterwards to push the result
    into plan-region coordinates, and again to push into full-page coordinates
    if you want to overlay on the original page render.

    ``confidence`` is preserved from the model (0..1, scaled to 0..100 to match
    Tesseract's confidence column convention).
    """
    cols = [
        "mark", "source_sheet", "source_page",
        "bbox_left", "bbox_top", "bbox_width", "bbox_height",
        "image_width", "image_height", "confidence", "detector",
        "region",
    ]
    if tile_model_df is None or tile_model_df.empty:
        return pd.DataFrame(columns=cols)

    out_rows: list[dict] = []
    for _, r in tile_model_df.iterrows():
        mark = str(r.get("mark", "")).strip()
        if not mark:
            continue
        x1 = float(r.get("bbox_x1", 0.0))
        y1 = float(r.get("bbox_y1", 0.0))
        x2 = float(r.get("bbox_x2", 0.0))
        y2 = float(r.get("bbox_y2", 0.0))
        left = int(round(x1 * tile_width))
        top = int(round(y1 * tile_height))
        width = max(0, int(round((x2 - x1) * tile_width)))
        height = max(0, int(round((y2 - y1) * tile_height)))
        try:
            conf01 = float(r.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf01 = 0.5
        out_rows.append(
            {
                "mark": mark,
                "source_sheet": source_sheet,
                "source_page": source_page,
                "bbox_left": left,
                "bbox_top": top,
                "bbox_width": width,
                "bbox_height": height,
                "image_width": tile_width,
                "image_height": tile_height,
                "confidence": max(0.0, min(1.0, conf01)) * 100.0,
                "detector": str(r.get("model", "vlm-tile")),
                "region": region_label,
            }
        )
    return pd.DataFrame(out_rows, columns=cols)
