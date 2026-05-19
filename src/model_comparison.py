"""Multi-model door-schedule extraction for the Tab 6 comparison feature.

Calls OpenAI (gpt-4o), Anthropic (claude-sonnet-4-6), and Google (gemini-2.5-flash)
in parallel on the *same* cropped A10.2 door-schedule image, and returns each
model's output in a standardised pandas DataFrame so they can be compared
side-by-side against the existing Tesseract baseline.

Design notes
------------
- The prompt is identical across all three models so the comparison is fair.
- Each model is asked to return a single JSON object ``{"doors": [...]}`` with a
  fixed per-row schema (mark, type, size, material, fire_rating, hardware_set,
  notes). Empty strings are used for fields the model cannot read.
- Image bytes are sent as base64 PNG. The caller is responsible for cropping /
  resizing if the page is huge.
- API responses are cached on disk under ``outputs/.cache/llm/`` keyed by
  ``(model, image_sha256, prompt_version)``. This means repeated clicks of
  "Run all models" cost nothing.
- Missing API keys or transport errors return an empty DataFrame + an error
  string so the UI can surface them without crashing the rest of the app.
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


# --------------------------------------------------------------------- prompt

# Bump this when the prompt or schema changes so caches invalidate.
PROMPT_VERSION = "v1"

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


# ------------------------------------------------------------------- registry

@dataclass(frozen=True)
class ModelSpec:
    key: str            # short id used as cache prefix and dict key
    label: str          # human label for UI ("OpenAI gpt-4o" etc.)
    model_id: str       # exact API model identifier
    env_var: str        # env var name holding the API key
    call: Callable[[bytes, str, str], str]  # (image_bytes, prompt, model_id) -> raw text


# Late-binding to avoid importing SDKs unless the model is actually called.
def _call_openai(image_bytes: bytes, prompt: str, model_id: str) -> str:
    from openai import OpenAI  # type: ignore
    client = OpenAI()  # picks up OPENAI_API_KEY from env
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
    client = Anthropic()  # picks up ANTHROPIC_API_KEY from env
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
    # content is a list of blocks; for a JSON-only prompt the first text block
    # holds the whole response.
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "\n".join(parts)


def _call_gemini(image_bytes: bytes, prompt: str, model_id: str) -> str:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    client = genai.Client()  # picks up GOOGLE_API_KEY / GEMINI_API_KEY
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


# ------------------------------------------------------------- response parser

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json_object(text: str) -> dict | None:
    """Robustly find a JSON object in the model's response."""
    if not text:
        return None
    text = text.strip()
    # 1. Try the whole text as-is.
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2. Try inside a ```json ... ``` fence.
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 3. Try the substring from first '{' to last '}'.
    lo, hi = text.find("{"), text.rfind("}")
    if 0 <= lo < hi:
        try:
            return json.loads(text[lo : hi + 1])
        except Exception:
            pass
    return None


def _rows_from_response(text: str, model_label: str) -> list[dict]:
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


# ------------------------------------------------------------------- caching

CACHE_DIR = Path(__file__).resolve().parent.parent / "outputs" / ".cache" / "llm"


def _cache_path(model_key: str, image_bytes: bytes) -> Path:
    h = hashlib.sha256(image_bytes).hexdigest()[:16]
    return CACHE_DIR / f"{model_key}__{h}__{PROMPT_VERSION}.json"


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


@dataclass
class ModelRun:
    spec: ModelSpec
    df: pd.DataFrame
    error: str = ""
    cached: bool = False
    elapsed_s: float = 0.0
    raw_response: str = ""


def run_model(
    model_key: str, image_bytes: bytes, use_cache: bool = True
) -> ModelRun:
    """Call one model with the door-schedule prompt.

    Returns a ``ModelRun`` whose ``df`` is empty when the model errored; in
    that case ``error`` carries a human-readable message for the UI.
    """
    if model_key not in MODELS:
        return ModelRun(
            spec=MODELS.get(model_key, ModelSpec(model_key, model_key, "", "", _call_openai)),
            df=pd.DataFrame(columns=DOOR_COLUMNS),
            error=f"Unknown model {model_key!r}",
        )
    spec = MODELS[model_key]

    cache_path = _cache_path(spec.key, image_bytes)
    if use_cache:
        cached = _read_cache(cache_path)
        if cached and cached.get("prompt_version") == PROMPT_VERSION:
            rows = _rows_from_response(cached.get("raw_response", ""), spec.label)
            df = pd.DataFrame(rows, columns=DOOR_COLUMNS)
            return ModelRun(
                spec=spec, df=df, cached=True,
                raw_response=cached.get("raw_response", ""),
                elapsed_s=float(cached.get("elapsed_s", 0.0)),
            )

    if not os.environ.get(spec.env_var):
        return ModelRun(
            spec=spec,
            df=pd.DataFrame(columns=DOOR_COLUMNS),
            error=f"{spec.env_var} is not set — add it to .env and restart Streamlit.",
        )

    t0 = time.time()
    try:
        raw = spec.call(image_bytes, DOOR_SCHEDULE_PROMPT, spec.model_id)
    except Exception as exc:  # noqa: BLE001
        return ModelRun(
            spec=spec,
            df=pd.DataFrame(columns=DOOR_COLUMNS),
            error=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.time() - t0,
        )
    elapsed = time.time() - t0

    rows = _rows_from_response(raw, spec.label)
    df = pd.DataFrame(rows, columns=DOOR_COLUMNS)
    _write_cache(
        cache_path,
        {
            "model": spec.key,
            "model_id": spec.model_id,
            "prompt_version": PROMPT_VERSION,
            "raw_response": raw,
            "n_rows": len(rows),
            "elapsed_s": elapsed,
        },
    )
    return ModelRun(spec=spec, df=df, elapsed_s=elapsed, raw_response=raw)


def run_all_models(
    image_bytes: bytes, use_cache: bool = True
) -> dict[str, ModelRun]:
    """Convenience wrapper — runs every registered model sequentially."""
    return {key: run_model(key, image_bytes, use_cache=use_cache) for key in MODELS}


def tesseract_baseline_as_df(tesseract_df: pd.DataFrame) -> pd.DataFrame:
    """Re-shape the existing tesseract door extraction into the same schema.

    The existing ``extract_door_schedule`` returns columns ``mark, type, size,
    notes, ...``. We synthesise the missing comparison fields from ``notes``
    so the Tab 6 table aligns column-for-column with the LLM outputs.
    """
    if tesseract_df is None or tesseract_df.empty:
        return pd.DataFrame(columns=DOOR_COLUMNS)
    out = pd.DataFrame()
    out["mark"] = tesseract_df.get("mark", "")
    out["type"] = tesseract_df.get("type", "")
    out["size"] = tesseract_df.get("size", "")
    notes = tesseract_df.get("notes", pd.Series([""] * len(tesseract_df))).fillna("")
    out["material"] = ""        # baseline didn't split this out
    out["fire_rating"] = ""     # baseline didn't split this out
    out["hardware_set"] = ""    # baseline didn't split this out
    out["notes"] = notes
    out["model"] = "Tesseract (baseline)"
    return out[DOOR_COLUMNS]
