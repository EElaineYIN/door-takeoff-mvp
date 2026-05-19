"""PDF rendering, OCR, and disk caching.

PDFs in this project are image-based (no selectable text), so all extraction
flows through ``pytesseract`` over PyMuPDF-rendered page rasters. We cache both
the raster (PNG) and the OCR result (JSON with per-word bbox+confidence) on
disk so repeated runs are near-instant.

Cache layout:
    outputs/.cache/{stem}__p{page:03d}__d{dpi}{__crop_LRTB}.{png|ocr.json}
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image


CACHE_DIR = Path(__file__).resolve().parent.parent / "outputs" / ".cache"
DEFAULT_DPI = 200
MAX_DPI = 300
LOW_CONF_THRESHOLD = 60.0  # exposed for matching.py


# ---------------------------------------------------------------- environment

def _check_tesseract() -> bool:
    """Return True if the ``tesseract`` binary is on PATH."""
    return shutil.which("tesseract") is not None


def tesseract_version() -> str:
    """Return tesseract's version string, or 'missing' if not installed."""
    if not _check_tesseract():
        return "missing"
    try:
        out = subprocess.run(
            ["tesseract", "--version"], capture_output=True, text=True, check=False
        )
        first = (out.stdout or out.stderr).splitlines()[0]
        return first.strip()
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


# -------------------------------------------------------------------- caching

def _safe_stem(pdf_path: Path) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(pdf_path).stem)
    return s.strip("_") or "pdf"


def _crop_suffix(crop: Optional[tuple[float, float, float, float]]) -> str:
    if crop is None:
        return ""
    l, t, r, b = crop
    return f"__crop_{l:.2f}_{t:.2f}_{r:.2f}_{b:.2f}"


def _cache_path(pdf_path: Path, page_index: int, dpi: int, kind: str,
                crop: Optional[tuple[float, float, float, float]] = None) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(pdf_path)
    name = f"{stem}__p{page_index:03d}__d{dpi}{_crop_suffix(crop)}.{kind}"
    return CACHE_DIR / name


# -------------------------------------------------------------- pdf rendering

def get_pdf_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF."""
    with fitz.open(str(pdf_path)) as doc:
        return doc.page_count


def render_page(pdf_path: Path, page_index: int, dpi: int = DEFAULT_DPI) -> Image.Image:
    """Render a single PDF page as a PIL image, cached to PNG on disk."""
    if dpi > MAX_DPI:
        dpi = MAX_DPI
    cache = _cache_path(pdf_path, page_index, dpi, "png")
    if cache.exists():
        return Image.open(cache).copy()

    with fitz.open(str(pdf_path)) as doc:
        page = doc.load_page(page_index)
        # PyMuPDF default is 72dpi; use Matrix scale = dpi / 72.
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    img.save(cache, format="PNG", optimize=False)
    return img


def _crop_image(img: Image.Image,
                crop: tuple[float, float, float, float]) -> Image.Image:
    """Crop a PIL image using normalized 0..1 coordinates (l, t, r, b)."""
    w, h = img.size
    l, t, r, b = crop
    box = (int(l * w), int(t * h), int(r * w), int(b * h))
    return img.crop(box)


# -------------------------------------------------------------------- ocr

def ocr_page(
    pdf_path: Path,
    page_index: int,
    dpi: int = DEFAULT_DPI,
    crop: Optional[tuple[float, float, float, float]] = None,
) -> dict:
    """Run tesseract on a (cropped) PDF page; cache result to JSON.

    Returns a dict::

        {
            "text": str,                    # full page text
            "words": [                      # per-word records
                {"text": str, "left": int, "top": int,
                 "width": int, "height": int, "conf": float},
                ...
            ],
            "mean_conf": float,             # average confidence over words with conf >= 0
            "tesseract_version": str,
            "dpi": int,
            "crop": tuple | None,
            "rendered_at": str (ISO 8601),
        }
    """
    if not _check_tesseract():
        raise RuntimeError(
            "Tesseract not found on PATH. Install with `brew install tesseract`."
        )

    cache = _cache_path(pdf_path, page_index, dpi, "ocr.json", crop=crop)
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except json.JSONDecodeError:
            pass  # corrupt cache; regenerate

    img = render_page(pdf_path, page_index, dpi=dpi)
    if crop is not None:
        img = _crop_image(img, crop)

    # Lazy import — only needed when tesseract is on the path
    import pytesseract
    from pytesseract import Output

    data = pytesseract.image_to_data(img, output_type=Output.DICT)

    words: list[dict] = []
    text_buf: list[str] = []
    confs: list[float] = []
    for i in range(len(data["text"])):
        txt = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not txt:
            continue
        words.append(
            {
                "text": txt,
                "left": int(data["left"][i]),
                "top": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
                "conf": conf,
            }
        )
        text_buf.append(txt)
        if conf >= 0:
            confs.append(conf)

    full_text = pytesseract.image_to_string(img)
    mean_conf = float(sum(confs) / len(confs)) if confs else 0.0

    result = {
        "text": full_text,
        "words": words,
        "mean_conf": mean_conf,
        "tesseract_version": tesseract_version(),
        "dpi": dpi,
        "crop": list(crop) if crop is not None else None,
        "rendered_at": datetime.now().isoformat(timespec="seconds"),
    }
    cache.write_text(json.dumps(result))
    return result


# --------------------------------------------------------------------- prewarm

def prewarm(pdf_path: Path, page_indices: list[int], dpi: int = DEFAULT_DPI) -> None:
    """Render + OCR each of ``page_indices`` so subsequent reads are cache hits."""
    for p in page_indices:
        render_page(pdf_path, p, dpi=dpi)
        try:
            ocr_page(pdf_path, p, dpi=dpi)
        except RuntimeError:
            # Tesseract missing — render only.
            pass


# --------------------------------------------------------------- CLI for prewarm

def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Door takeoff MVP — pdf_utils")
    sub = p.add_subparsers(dest="cmd", required=True)
    pw = sub.add_parser(
        "prewarm",
        help="Pre-render + OCR the door schedule (A10.2) and floor plans (A2.1–A2.4)",
    )
    pw.add_argument("--data", default="data", help="Directory containing PDFs")
    pw.add_argument("--dpi", default=DEFAULT_DPI, type=int)

    args = p.parse_args()

    if args.cmd == "prewarm":
        from .index_parser import load_index
        from .sheet_selector import expected_pages

        data_dir = Path(args.data)
        index_df, _ = load_index(data_dir)
        pages = expected_pages(index_df)
        pdf_path = data_dir / "1522-CONSTRUCTION SET-11.19.18.pdf"
        if not pdf_path.exists():
            raise SystemExit(f"PDF not found: {pdf_path}")
        # PyMuPDF uses 0-based page indices; index lists 1-based PDF pages.
        page_indices = [p - 1 for p in pages if p >= 1]
        print(f"Pre-warming {len(page_indices)} pages from {pdf_path.name} at {args.dpi} dpi…")
        for i, idx in enumerate(page_indices, 1):
            print(f"  [{i}/{len(page_indices)}] page {idx+1}…")
            render_page(pdf_path, idx, dpi=args.dpi)
            try:
                ocr_page(pdf_path, idx, dpi=args.dpi)
            except RuntimeError as e:
                print(f"    skipped OCR: {e}")
        print("Done.")


if __name__ == "__main__":
    _cli()
