"""Door-takeoff sheet selection.

Two kinds of sheet matter for door takeoff:

* **Schedule sheet** — ``A10.2`` (the catalog of door types/sizes/marks)
* **Plan sheets**  — ``A2.1`` … ``A2.4`` (each floor's plan, where door tags
  show occurrence counts on the building)

A complete takeoff = (schedule spec) × (plan occurrences). This module
returns each set independently.

Patterns are anchored so e.g. ``A0.2A`` doesn't accidentally match ``^A0\\.2``.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


# Primary schedule sheet for the door takeoff.
SCHEDULE_SHEET = "A10.2"

# Floor-plan sheets to look for door occurrences on.
#
# A2.1–A2.4 are the dense architectural floor plans (hexagonal door tags
# crowded among room numbers, wall-type squares, and grid bubbles — hard
# for VLMs to disambiguate without prompt engineering).
#
# A7.1.1 is the **Signage** sheet (Roof Deck / etc.): same building, but
# the plan is sparse and each door is marked with a SIGNAGE label such
# as ``1.11`` or ``7.15`` printed next to a black filled dot at the door
# swing arc. Much cleaner to detect — used as the polished demo target.
PLAN_SHEET_PATTERNS = [r"^A2\.[1-4]$", r"^A7\.1\.1$"]


def _matches_any(sheet_no: str, patterns: Iterable[str]) -> bool:
    if not isinstance(sheet_no, str):
        return False
    return any(re.match(p, sheet_no) for p in patterns)


def _in_main_set(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["in_main_set"]] if "in_main_set" in df.columns else df


def schedule_row(index_df: pd.DataFrame) -> pd.DataFrame:
    """Return the index row(s) for the door schedule sheet (A10.2)."""
    df = _in_main_set(index_df)
    return df[df["sheet_no"] == SCHEDULE_SHEET].sort_values("pdf_page").reset_index(drop=True)


def plan_rows(index_df: pd.DataFrame) -> pd.DataFrame:
    """Return the floor-plan sheet rows (A2.1…A2.4) sorted by sheet number."""
    df = _in_main_set(index_df)
    mask = df["sheet_no"].apply(lambda s: _matches_any(s, PLAN_SHEET_PATTERNS))
    return df[mask].sort_values("sheet_no").reset_index(drop=True)


def relevant_sheets(index_df: pd.DataFrame) -> pd.DataFrame:
    """Schedule + plan rows combined, sorted by PDF page."""
    return (
        pd.concat([schedule_row(index_df), plan_rows(index_df)], ignore_index=True)
        .drop_duplicates(subset=["sheet_no"])
        .sort_values("pdf_page")
        .reset_index(drop=True)
    )


def schedule_page(index_df: pd.DataFrame) -> int | None:
    """Return the PDF page number of the door schedule sheet (1-based)."""
    rows = schedule_row(index_df)
    if rows.empty:
        return None
    return int(rows.iloc[0]["pdf_page"])


def plan_pages(index_df: pd.DataFrame) -> list[tuple[str, int]]:
    """Return ``[(sheet_no, pdf_page), ...]`` for the plan sheets."""
    rows = plan_rows(index_df)
    return [(r["sheet_no"], int(r["pdf_page"])) for _, r in rows.iterrows()]


def expected_pages(index_df: pd.DataFrame) -> list[int]:
    """All schedule + plan page numbers (used for prewarm)."""
    return relevant_sheets(index_df)["pdf_page"].astype(int).tolist()
