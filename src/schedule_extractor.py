"""Door-schedule extractor over OCR output.

Reads A10.2 (left ~33% crop) and reconstructs door rows: mark, type, size,
notes, schedule count. The page hosts four side-by-side schedules; cropping
restricts OCR to the door schedule only.

Future multimodal extractors can drop in alongside this baseline (see
``model_comparison.py``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import pdf_utils


# --------------------------------------------------------- row reconstruction

def _cluster_rows(words: list[dict], y_tolerance: int | None = None) -> list[list[dict]]:
    """Group words by their vertical position into row-like clusters.

    Each cluster is sorted left-to-right. ``y_tolerance`` defaults to a third
    of the median word height (tight enough to keep dense schedule rows apart).
    """
    if not words:
        return []
    if y_tolerance is None:
        median_h = sorted(w["height"] for w in words)[len(words) // 2]
        y_tolerance = max(6, median_h // 3)

    sorted_words = sorted(words, key=lambda w: (w["top"], w["left"]))
    rows: list[list[dict]] = []
    current_row: list[dict] = []
    current_top: int | None = None  # anchor for the cluster — set once

    for w in sorted_words:
        if current_top is None or abs(w["top"] - current_top) <= y_tolerance:
            current_row.append(w)
            if current_top is None:
                current_top = w["top"]
        else:
            current_row.sort(key=lambda x: x["left"])
            rows.append(current_row)
            current_row = [w]
            current_top = w["top"]

    if current_row:
        current_row.sort(key=lambda x: x["left"])
        rows.append(current_row)
    return rows


def _clean_token(tok: str) -> str:
    """Strip OCR column-border characters that bleed into adjacent tokens."""
    return (tok or "").lstrip("|_[").rstrip("|_]").strip()


def _row_mean_conf(row: list[dict]) -> float:
    confs = [w["conf"] for w in row if w.get("conf", -1) >= 0]
    return float(sum(confs) / len(confs)) if confs else 0.0


# ----------------------------------------------------------- Door schedule

# Door marks tend to be like "01", "01A", "1A", "D1", "P-101". Tighten conservatively.
_DOOR_MARK_RE = re.compile(r"^[A-Z]?[A-Z]?[-]?\d{1,4}[A-Z]?$")
_DOOR_MIN_TOKENS = 3   # mark + at least two more fields
_DOOR_MAX_MARK_LEN = 6


def _looks_like_door_mark(tok: str) -> bool:
    if not tok or len(tok) > _DOOR_MAX_MARK_LEN:
        return False
    return bool(_DOOR_MARK_RE.match(tok))


def extract_door_schedule(pdf_path: Path, page_index: int) -> pd.DataFrame:
    """Extract door rows from A10.2 (left-third crop only).

    A10.2 hosts four side-by-side schedules; cropping to the left ~33% restricts
    OCR to the door schedule.
    """
    ocr = pdf_utils.ocr_page(pdf_path, page_index, crop=(0.0, 0.0, 0.33, 1.0))
    rows = _cluster_rows(ocr["words"])

    out: list[dict] = []
    seen_marks: set[str] = set()
    for row in rows:
        tokens = [w["text"] for w in row]
        if len(tokens) < _DOOR_MIN_TOKENS:
            continue
        first = tokens[0]
        if not _looks_like_door_mark(first):
            continue
        # Skip header-like rows (e.g. "MARK" alone).
        if first.upper() in {"MARK", "TYPE", "SIZE", "DOOR"}:
            continue
        mark = first.strip()
        if mark in seen_marks:
            continue
        seen_marks.add(mark)

        # Tokens after the mark form the description (type/size/material/hw).
        rest = [_clean_token(t) for t in tokens[1:]]
        rest = [t for t in rest if t and t not in {"|", "_|", "|_", "_"}]
        type_field = rest[0] if rest else ""
        size_field = " ".join(rest[1:4]) if len(rest) > 1 else ""
        notes = " ".join(rest[4:]) if len(rest) > 4 else ""

        out.append(
            {
                "mark": mark,
                "type": type_field,
                "size": size_field,
                "notes": notes,
                "count_schedule": 1,
                "source_sheet": "A10.2",
                "source_page": page_index + 1,
                "confidence": round(_row_mean_conf(row), 1),
                "flag": "",
                "review_action": "Needs check",
                "source": "tesseract",
            }
        )

    return pd.DataFrame(
        out,
        columns=[
            "mark", "type", "size", "notes", "count_schedule",
            "source_sheet", "source_page", "confidence", "flag",
            "review_action", "source",
        ],
    )


# ---------------------------------------------------------------- dispatch

def run_schedule_extraction(index_df: pd.DataFrame, pdf_path: Path) -> pd.DataFrame:
    """Run the door-schedule extractor on the primary schedule sheet (A10.2)."""
    target = index_df[index_df["sheet_no"] == "A10.2"]
    if target.empty:
        return extract_door_schedule(pdf_path, 0).iloc[0:0]
    page = int(target.iloc[0]["pdf_page"])
    return extract_door_schedule(pdf_path, page - 1)
