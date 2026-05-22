"""Door-tag detection on architectural floor plans.

Two extractors are provided here, both with the same output schema so they
can be union-ed and fed to ``matching.reconcile``:

* ``detect_marks_on_plan`` — **Tag OCR baseline.** Renders a plan at higher
  DPI than the schedule (default 300), runs Tesseract, and keeps every word
  whose text matches a known door mark from the schedule vocabulary.
* ``detect_marks_on_all_plans`` — convenience wrapper over each plan sheet.

Output schema (one row per detected occurrence):

    mark, source_sheet, source_page, bbox_left, bbox_top,
    bbox_width, bbox_height, image_width, image_height, confidence,
    detector

Where bbox values are pixel coordinates of the rendered plan image at the
extractor's DPI (so overlays drawn at the same DPI line up).

A small drawing utility ``draw_detections_on_image`` overlays the bboxes
onto a rendered plan image — used by Tab 2 for the visual review.
"""

from __future__ import annotations

import colorsys
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from . import pdf_utils


# --------------------------------------------------------------- vocab match

# Strip punctuation that Tesseract sometimes appends to mark tokens.
_TRIM_CHARS = "()[]{}.,:;\"'"


def _normalize_mark(text: str) -> str:
    """Trim outer punctuation and uppercase a candidate mark token."""
    if not text:
        return ""
    t = text.strip().strip(_TRIM_CHARS).upper()
    return t


def _build_vocab(schedule_marks: Iterable[str]) -> set[str]:
    """Return a normalized set of known door marks."""
    vocab: set[str] = set()
    for m in schedule_marks or []:
        if m is None:
            continue
        n = _normalize_mark(str(m))
        if n:
            vocab.add(n)
    return vocab


# Door marks are short — drop tokens that obviously aren't one.
_MAX_MARK_LEN = 6
_MARK_SHAPE_RE = re.compile(r"^[A-Z]?[A-Z]?[-]?\d{1,4}[A-Z]?$")


def _looks_like_mark(token: str) -> bool:
    if not token or len(token) > _MAX_MARK_LEN:
        return False
    return bool(_MARK_SHAPE_RE.match(token))


# ------------------------------------------------------------- per-plan OCR

PLAN_DPI = 300  # Higher than the schedule pass — tags on plans are tiny.


def detect_marks_on_plan(
    pdf_path: Path,
    page_index: int,
    vocab: set[str],
    dpi: int = PLAN_DPI,
    sheet_no: str = "",
) -> pd.DataFrame:
    """Return every word on the plan whose normalized text is in ``vocab``.

    Each detected occurrence becomes one row. The same mark may legitimately
    appear many times on a single plan (one per door on the building) — those
    are kept as separate rows so ``matching.reconcile`` can sum them.
    """
    ocr = pdf_utils.ocr_page(pdf_path, page_index, dpi=dpi)

    out: list[dict] = []
    img_w, img_h = _ocr_image_dims(pdf_path, page_index, dpi)
    for w in ocr.get("words", []):
        token = _normalize_mark(w.get("text", ""))
        if not token or not _looks_like_mark(token):
            continue
        if token not in vocab:
            continue
        out.append(
            {
                "mark": token,
                "source_sheet": sheet_no,
                "source_page": page_index + 1,
                "bbox_left": int(w.get("left", 0)),
                "bbox_top": int(w.get("top", 0)),
                "bbox_width": int(w.get("width", 0)),
                "bbox_height": int(w.get("height", 0)),
                "image_width": img_w,
                "image_height": img_h,
                "confidence": float(w.get("conf", 0.0)),
                "detector": "tesseract_tag_ocr",
            }
        )

    return pd.DataFrame(
        out,
        columns=[
            "mark", "source_sheet", "source_page",
            "bbox_left", "bbox_top", "bbox_width", "bbox_height",
            "image_width", "image_height", "confidence", "detector",
        ],
    )


def detect_marks_on_all_plans(
    pdf_path: Path,
    plan_pages: list[tuple[str, int]],
    vocab: set[str],
    dpi: int = PLAN_DPI,
) -> pd.DataFrame:
    """Run the Tag OCR baseline on each ``(sheet_no, pdf_page)`` plan."""
    frames: list[pd.DataFrame] = []
    for sheet_no, pdf_page in plan_pages:
        df = detect_marks_on_plan(
            pdf_path, page_index=pdf_page - 1, vocab=vocab,
            dpi=dpi, sheet_no=sheet_no,
        )
        frames.append(df)
    if not frames:
        return _empty_plan_df()
    return pd.concat(frames, ignore_index=True)


def _empty_plan_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "mark", "source_sheet", "source_page",
            "bbox_left", "bbox_top", "bbox_width", "bbox_height",
            "image_width", "image_height", "confidence", "detector",
        ]
    )


def _ocr_image_dims(pdf_path: Path, page_index: int, dpi: int) -> tuple[int, int]:
    """Return the (width, height) of the rendered image used for OCR."""
    img = pdf_utils.render_page(pdf_path, page_index, dpi=dpi)
    return img.size


# --------------------------------------------------------------- bbox overlay

def _color_for_mark(mark: str) -> tuple[int, int, int]:
    """Stable HSV-based color per mark so detections of the same mark match."""
    h = (abs(hash(mark)) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.9)
    return (int(r * 255), int(g * 255), int(b * 255))


_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _load_font(size: int):
    """Best-effort TrueType font, falling back to PIL's default bitmap font."""
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10
    except Exception:
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    """Backwards-compatible text size measurement (works on Pillow <10 and >=10)."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except AttributeError:
        return draw.textsize(text, font=font)


def draw_detections_on_image(
    img: Image.Image,
    detections: pd.DataFrame,
    show_labels: bool = True,
    line_width: int = 12,
    font_size: int = 36,
    pad: int = 6,
) -> Image.Image:
    """Return a copy of ``img`` with each detection drawn as a colored rectangle.

    ``detections`` is expected to be in the schema produced by
    ``detect_marks_on_plan`` (pixel-space bboxes at the same DPI as ``img``).

    The defaults are tuned for plans rendered at ~300 DPI (≈7000×10000 px) so
    boxes remain visible after the browser downscales the image to fit a column.
    Callers rendering at lower DPI for display should override ``line_width``,
    ``font_size``, and ``pad`` proportionally.
    """
    annotated = img.copy().convert("RGB")
    if detections is None or detections.empty:
        return annotated

    draw = ImageDraw.Draw(annotated)
    font = _load_font(font_size) if show_labels else None

    for _, row in detections.iterrows():
        mark = str(row.get("mark", "")).strip()
        if not mark:
            continue
        l = int(row.get("bbox_left", 0))
        t = int(row.get("bbox_top", 0))
        w = int(row.get("bbox_width", 0))
        h = int(row.get("bbox_height", 0))
        if w <= 0 or h <= 0:
            continue
        color = _color_for_mark(mark)
        # Inflate so the box is comfortably visible around small tags.
        rect = [l - pad, t - pad, l + w + pad, t + h + pad]
        draw.rectangle(rect, outline=color, width=line_width)
        if show_labels and font is not None:
            tw, th = _text_size(draw, mark, font)
            label_x = rect[0]
            label_y = max(0, rect[1] - th - 8)
            # Filled background so the label is readable on a busy plan.
            draw.rectangle(
                [label_x, label_y, label_x + tw + 10, label_y + th + 6],
                fill=color,
            )
            draw.text((label_x + 5, label_y + 3), mark, fill="white", font=font)

    return annotated


# --------------------------------------------------------------- summaries

def per_mark_counts(detections: pd.DataFrame) -> pd.DataFrame:
    """Aggregate detections to one row per ``(mark, source_sheet)``.

    Used by Tab 2 to show "mark X appeared N times on A2.1".
    """
    if detections is None or detections.empty:
        return pd.DataFrame(columns=["mark", "source_sheet", "count"])
    return (
        detections.groupby(["mark", "source_sheet"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["mark", "source_sheet"])
        .reset_index(drop=True)
    )
