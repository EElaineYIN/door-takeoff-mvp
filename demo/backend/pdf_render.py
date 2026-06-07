"""Render an uploaded PDF (or image) into page PNGs for the demo.

Uses PyMuPDF (fitz) so we don't depend on poppler/pdf2image.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

DPI = 200

# Sheet numbers in a title block look like: A2.2, A-201, S1.1, M2.0, E101, AD1.1, T-1.
_SHEET_RE = re.compile(r"^[A-Z]{1,3}[-.]?\d{1,3}(?:\.\d{1,2})?[A-Z]?$")
# Words that mark a drawing title we can show next to the sheet number.
_TITLE_KEYS = ("PLAN", "ELEVATION", "SECTION", "DETAIL", "SCHEDULE",
               "DIAGRAM", "COVER", "ROOF")
# Words that signal a *non*-title line (wall legends, general notes) to skip.
_TITLE_SKIP = ("WALL", "RATED", "PARTITION", "LEGEND", "GENERAL", "TYPE")


def _sheet_label(page) -> str | None:
    """Pull a sheet number (+ short title) from a page's title block text.

    The sheet number is usually the largest text in the bottom-right corner.
    Returns e.g. "A2.2 · Third & Fourth Floor Plan" or just "A2.2", or None.
    """
    try:
        d = page.get_text("dict")
    except Exception:
        return None
    W, H = page.rect.width, page.rect.height
    spans = []
    for blk in d.get("blocks", []):
        for line in blk.get("lines", []):
            for s in line.get("spans", []):
                t = (s.get("text") or "").strip()
                if t:
                    x0, y0 = s["bbox"][0], s["bbox"][1]
                    spans.append((s.get("size", 0), x0, y0, t))

    # candidate sheet numbers: match the pattern, prefer large font + bottom-right
    def corner_bonus(x, y):
        return (1.0 if x > 0.6 * W else 0.0) + (1.0 if y > 0.7 * H else 0.0)

    best = None
    for sz, x, y, t in spans:
        if _SHEET_RE.match(t.replace(" ", "")):
            score = sz + 6 * corner_bonus(x, y)
            if best is None or score > best[0]:
                best = (score, sz, x, y, t.replace(" ", ""))
    if best is None:
        return None
    num = best[4]

    # drawing title: restrict to the bottom-right title block, require a title
    # keyword, and reject wall-legend / general-note lines.
    title = None
    cand = [(y, t) for sz, x, y, t in spans
            if x > 0.6 * W and y > 0.6 * H
            and any(k in t.upper() for k in _TITLE_KEYS)
            and not any(k in t.upper() for k in _TITLE_SKIP)
            and not _SHEET_RE.match(t.replace(" ", ""))]
    if cand:
        title = max(cand, key=lambda yt: len(yt[1]))[1].strip()
        title = re.sub(r"\s+", " ", title).title()
    return f"{num} · {title}" if title else num


def render_pdf(pdf_bytes: bytes, out_dir: Path) -> list[dict]:
    """Render every page of a PDF to PNG at DPI. Returns page metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pages: list[dict] = []
    for i in range(doc.page_count):
        label = _sheet_label(doc[i])
        pix = doc[i].get_pixmap(matrix=mat, alpha=False)
        fname = f"page_{i + 1}.png"
        fpath = out_dir / fname
        pix.save(str(fpath))
        pages.append({
            "index": i,
            "name": label or f"Sheet {i + 1}",
            "sheet": label.split(" · ")[0] if label else f"Sheet {i + 1}",
            "file": fname,
            "width": pix.width,
            "height": pix.height,
        })
    doc.close()
    return pages


def render_image(img_bytes: bytes, out_dir: Path) -> list[dict]:
    """Treat an uploaded raster image as a single 'page'."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    fpath = out_dir / "page_1.png"
    img.save(str(fpath))
    return [{
        "index": 0,
        "name": "Sheet 1",
        "sheet": "Sheet 1",
        "file": "page_1.png",
        "width": img.width,
        "height": img.height,
    }]
