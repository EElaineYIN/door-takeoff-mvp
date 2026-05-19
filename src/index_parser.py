"""Parse the Drawing Index into a structured DataFrame.

The primary source is ``Drawing Index.xlsx`` (selectable text, 4 columns:
``PDF PAGE | SHEET # | TITLE | DATE``). Rows after the blank separator belong
to a separate Fire Protection PDF that is **not** in our /data construction set,
so their ``PDF PAGE`` numbers restart at 1 — we flag them ``in_main_set=False``.

If the xlsx is missing, ``parse_index_pdf_fallback`` OCR-extracts the PDF
index, but expect lower fidelity.
"""

from __future__ import annotations

import re
from pathlib import Path

import openpyxl
import pandas as pd


_DISCIPLINE_PREFIXES: list[tuple[str, str]] = [
    # Order matters: longest prefix first.
    ("FP", "Fire Protection"),
    ("LR", "Landscape"),
    ("LP", "Landscape"),
    ("A", "Architectural"),
    ("S", "Structural"),
    ("M", "Mechanical"),
    ("E", "Electrical"),
    ("P", "Plumbing"),
    ("C", "Civil"),
    ("L", "Landscape"),
    ("T", "Telecom"),
]


def infer_discipline(sheet_no: str | None) -> str:
    """Map a sheet number prefix to a discipline name."""
    if not sheet_no or not isinstance(sheet_no, str):
        return "Other"
    s = sheet_no.strip().upper()
    if not s or s == "NA":
        return "Other"
    for prefix, name in _DISCIPLINE_PREFIXES:
        if s.startswith(prefix) and (len(s) == len(prefix) or not s[len(prefix)].isalpha()):
            return name
    return "Other"


_SCOPE_TAG_RULES: list[tuple[str, str]] = [
    # (regex over sheet_no, tag)
    (r"^A10\.2$", "door_schedule"),
    (r"^A2\.[1-4]$", "floor_plan"),
]


def infer_scope_tags(sheet_no: str | None, title: str | None) -> list[str]:
    """Return scope tags for a sheet: door_schedule, floor_plan, schedule, plan."""
    sn = (sheet_no or "").strip().upper()
    t = (title or "").strip().upper()
    tags: list[str] = []

    for pattern, tag in _SCOPE_TAG_RULES:
        if re.match(pattern, sn) and tag not in tags:
            tags.append(tag)

    if "DOOR" in t and "door" not in tags:
        tags.append("door")
    if "SCHEDULE" in t and "schedule" not in tags:
        tags.append("schedule")
    if "PLAN" in t and "plan" not in tags:
        tags.append("plan")

    return tags


def _normalize_title(title) -> str:
    if title is None:
        return ""
    return re.sub(r"\s+", " ", str(title)).strip()


def parse_index_xlsx(xlsx_path: Path) -> pd.DataFrame:
    """Parse the Drawing Index xlsx into a DataFrame.

    Output columns:
        pdf_page (int) | sheet_no (str) | title (str) | discipline (str)
        | scope_tags (list[str]) | source_file (str) | in_main_set (bool)

    Rows after the blank separator (Fire Protection appendix) have
    ``in_main_set=False`` because their page numbers refer to a different PDF.
    """
    xlsx_path = Path(xlsx_path)
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return _empty_index_df()

    # Detect header (row 0).
    header = [str(c).strip().upper() if c is not None else "" for c in rows[0]]
    if "PDF PAGE" not in [h.replace("\xa0", "") for h in header]:
        data_rows = rows
    else:
        data_rows = rows[1:]

    parsed: list[dict] = []
    in_main_set = True

    for row in data_rows:
        if not row or all(c is None for c in row):
            in_main_set = False
            continue

        pdf_page = row[0] if len(row) > 0 else None
        sheet_no = row[1] if len(row) > 1 else None
        title = row[2] if len(row) > 2 else None

        # Section header like ('FIRE PROTECTION', None, None, None) — skip but flag.
        if pdf_page is not None and sheet_no is None and title is None:
            in_main_set = False
            continue

        try:
            pdf_page_int = int(pdf_page)
        except (TypeError, ValueError):
            continue
        if not sheet_no:
            continue

        sheet_no_str = str(sheet_no).strip()
        title_str = _normalize_title(title)
        parsed.append(
            {
                "pdf_page": pdf_page_int,
                "sheet_no": sheet_no_str,
                "title": title_str,
                "discipline": infer_discipline(sheet_no_str),
                "scope_tags": infer_scope_tags(sheet_no_str, title_str),
                "source_file": xlsx_path.name,
                "in_main_set": in_main_set,
            }
        )

    return pd.DataFrame(parsed)


def parse_index_pdf_fallback(pdf_path: Path) -> pd.DataFrame:
    """Fallback OCR-based parser for the Drawing Index PDF."""
    from . import pdf_utils  # local import — avoids circular dep at module load

    rows: list[dict] = []
    page_count = pdf_utils.get_pdf_page_count(pdf_path)
    sheet_re = re.compile(r"^[A-Z]{1,3}[0-9]+(?:\.[0-9]+[A-Z]?)*$")

    for page_index in range(page_count):
        ocr = pdf_utils.ocr_page(pdf_path, page_index)
        text_lines = [ln.strip() for ln in ocr["text"].splitlines() if ln.strip()]
        for i, line in enumerate(text_lines):
            tokens = line.split()
            if not tokens:
                continue
            if sheet_re.match(tokens[0]):
                sheet_no = tokens[0]
                title = " ".join(tokens[1:]) if len(tokens) > 1 else ""
                rows.append(
                    {
                        "pdf_page": None,
                        "sheet_no": sheet_no,
                        "title": title,
                        "discipline": infer_discipline(sheet_no),
                        "scope_tags": infer_scope_tags(sheet_no, title),
                        "source_file": Path(pdf_path).name,
                        "in_main_set": True,
                    }
                )

    return pd.DataFrame(rows)


def _empty_index_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "pdf_page", "sheet_no", "title", "discipline",
            "scope_tags", "source_file", "in_main_set",
        ]
    )


def load_index(data_dir: Path) -> tuple[pd.DataFrame, str]:
    """Convenience loader.

    Returns ``(df, source)`` where ``source`` is one of ``"xlsx"``, ``"pdf"``, ``"none"``.
    """
    data_dir = Path(data_dir)
    xlsx_path = data_dir / "Drawing Index.xlsx"
    pdf_path = data_dir / "Drawing Index.pdf"
    if xlsx_path.exists():
        return parse_index_xlsx(xlsx_path), "xlsx"
    if pdf_path.exists():
        return parse_index_pdf_fallback(pdf_path), "pdf"
    return _empty_index_df(), "none"
