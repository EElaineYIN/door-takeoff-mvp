"""Door-schedule reading + reconciliation for the demo.

Two jobs:
  1. find + read the DOOR SCHEDULE table out of a construction-set PDF, using
     the VLM (Claude via the inference proxy) the same way the experiments did.
  2. reconcile what we detected on a plan sheet against that schedule.

The VLM reading the schedule table is itself the AEC-AI story: a human reads
the schedule the same way, and now the model does it.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

BASE_URL = os.environ.get("INFERENCE_BASE_URL", "https://model.service-inference.ai/v1")
MODEL = os.environ.get("DEMO_VLM_MODEL", "claude-opus-4-8")
ENV_VAR = "ANTHROPIC_API_KEY"

SCHEDULE_KEYWORDS = ("door schedule", "door & frame schedule", "door and frame schedule")


def _client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get(ENV_VAR, ""), base_url=BASE_URL)


def find_schedule_page(pdf_bytes: bytes) -> int | None:
    """Return the 0-based page index whose text layer mentions a door schedule.

    Cheap heuristic first (text search). Returns None if no text layer match;
    the caller can then fall back to asking the VLM.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    best = None
    for i in range(doc.page_count):
        txt = (doc[i].get_text() or "").lower()
        if any(k in txt for k in SCHEDULE_KEYWORDS):
            # prefer the page with the most door-mark-ish content
            score = txt.count("door")
            if best is None or score > best[1]:
                best = (i, score)
    doc.close()
    return best[0] if best else None


def _page_png_b64(image_path: str, max_side: int = 2000) -> str:
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


_SCHED_PROMPT = (
    "This is an architectural DOOR SCHEDULE sheet. Read the schedule table like a "
    "quantity estimator. Extract every door row.\n"
    "For each row return: mark (the door tag/number), type (swing/double/sliding/"
    "pocket/other — infer from description if needed), width, height, and qty if a "
    "quantity column exists (else null).\n"
    "Reply with ONLY compact JSON:\n"
    '{"doors":[{"mark":"...","type":"swing","width":"3\'-0\\"","height":"7\'-0\\"",'
    '"qty":null}], "note":"<=10 words about the schedule"}'
)


def read_schedule(image_path: str) -> dict:
    """Send the schedule page image to the VLM and parse the door catalog."""
    b64 = _page_png_b64(image_path)
    client = _client()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _SCHED_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64}", "detail": "high"}},
                ],
            }],
            response_format={"type": "json_object"},
            max_tokens=4096,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        data = json.loads(text)
        doors = data.get("doors", []) or []
        # normalize
        for d in doors:
            d["mark"] = str(d.get("mark", "")).strip()
            d["type"] = str(d.get("type", "other")).strip().lower()
            d["width"] = str(d.get("width", "") or "")
            d["height"] = str(d.get("height", "") or "")
        return {"doors": doors, "note": str(data.get("note", ""))}
    except Exception as e:
        return {"doors": [], "note": "", "error": str(e)[:160]}


def reconcile(detected: list[dict], schedule_doors: list[dict]) -> dict:
    """Compare detected door type-counts against the schedule's type-counts.

    Returns per-type rows: scheduled qty, detected qty, status.
    """
    from collections import Counter

    det = Counter((b.get("type") or "swing") for b in detected)
    sched = Counter()
    for d in schedule_doors:
        q = d.get("qty")
        n = int(q) if isinstance(q, (int, float)) or (isinstance(q, str) and q.isdigit()) else 1
        sched[d.get("type") or "other"] += n

    types = sorted(set(det) | set(sched))
    rows = []
    for t in types:
        s, dcount = sched.get(t, 0), det.get(t, 0)
        if s == 0:
            status = "unscheduled"           # detected but not in schedule
        elif dcount == s:
            status = "match"
        elif dcount < s:
            status = "missing"               # fewer found than scheduled
        else:
            status = "extra"                 # more found than scheduled
        rows.append({"type": t, "scheduled": s, "detected": dcount, "status": status})
    return {
        "rows": rows,
        "scheduled_total": sum(sched.values()),
        "detected_total": sum(det.values()),
    }
