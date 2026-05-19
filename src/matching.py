"""Cross-checking and confidence flagging for the door takeoff.

Two flagging passes:

1. ``add_confidence_flags`` — OCR confidence + mark-shape sanity, applied
   to any DataFrame with a ``confidence`` and/or ``mark`` column.
2. ``reconcile`` — schedule × plan reconciliation: for each unique mark,
   compares the schedule spec to the plan-side occurrence count and tags
   each as **matched / mismatch / orphan / stray**.

Glossary:
  * **orphan** — mark exists in the schedule but never appears on a plan
    (likely an unused spec, or the plan detector missed it).
  * **stray**  — mark detected on a plan but absent from the schedule
    (likely a tag the schedule doesn't cover, or an OCR/VLM hallucination).
  * **mismatch** — appears on both sides but the plan count is unexpected
    (only meaningful once the user supplies an "expected count" per mark).
"""

from __future__ import annotations

import re

import pandas as pd


LOW_CONF_THRESHOLD = 60.0


def _add_flag(existing: str, new: str) -> str:
    if not existing:
        return new
    if new in existing:
        return existing
    return f"{existing}; {new}"


def add_confidence_flags(
    df: pd.DataFrame, low_conf_threshold: float = LOW_CONF_THRESHOLD
) -> pd.DataFrame:
    """Add confidence-based flags to every row in-place (returns the same df).

    Rules:
      * confidence < threshold        → low_ocr_confidence
      * mark contains non-alphanumeric → mark_unparseable
      * duplicate mark                → duplicate_mark
    """
    if df.empty:
        return df

    if "confidence" in df.columns:
        low = df["confidence"].fillna(0) < low_conf_threshold
        df.loc[low, "flag"] = df.loc[low, "flag"].apply(
            lambda f: _add_flag(f, "low_ocr_confidence")
        )

    if "mark" in df.columns:
        bad = df["mark"].astype(str).apply(
            lambda m: bool(re.search(r"[^A-Za-z0-9._-]", m))
        )
        df.loc[bad, "flag"] = df.loc[bad, "flag"].apply(
            lambda f: _add_flag(f, "mark_unparseable")
        )
        dup_mask = df["mark"].duplicated(keep=False)
        df.loc[dup_mask, "flag"] = df.loc[dup_mask, "flag"].apply(
            lambda f: _add_flag(f, "duplicate_mark")
        )

    return df


# ---------------------------------------------------- schedule × plan reconcile

def reconcile(
    schedule_df: pd.DataFrame,
    plan_df: pd.DataFrame,
    expected_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Combine the schedule spec and plan-side occurrences per mark.

    ``schedule_df`` columns expected: ``mark, type, size, notes, count_schedule``.
    ``plan_df``     columns expected: ``mark, source_sheet`` (one row per
                                       detection). Rows are aggregated to a
                                       per-mark plan count.
    ``expected_counts`` (optional): override per-mark expected counts; defaults
                                    to ``count_schedule`` from the schedule
                                    (each schedule mark = 1 spec entry).

    Output columns:
        mark, type, size, notes, count_schedule, plan_count, plan_sheets,
        status (matched / mismatch / orphan / stray), flag
    """
    sched = schedule_df.copy() if not schedule_df.empty else _empty_schedule_df()
    plan = plan_df.copy() if not plan_df.empty else _empty_plan_df()

    # Aggregate plan detections to one row per mark.
    if not plan.empty:
        plan_agg = (
            plan.groupby("mark", as_index=False)
            .agg(
                plan_count=("mark", "size"),
                plan_sheets=("source_sheet", lambda s: ", ".join(sorted(set(s.dropna())))),
            )
        )
    else:
        plan_agg = pd.DataFrame(columns=["mark", "plan_count", "plan_sheets"])

    sched_marks = set(sched["mark"]) if not sched.empty else set()
    plan_marks = set(plan_agg["mark"]) if not plan_agg.empty else set()

    merged = sched.merge(plan_agg, on="mark", how="outer")
    merged["plan_count"] = merged["plan_count"].fillna(0).astype(int)
    merged["plan_sheets"] = merged["plan_sheets"].fillna("")
    for col in ("type", "size", "notes"):
        if col in merged.columns:
            merged[col] = merged[col].fillna("")
    if "count_schedule" in merged.columns:
        merged["count_schedule"] = merged["count_schedule"].fillna(0).astype(int)
    else:
        merged["count_schedule"] = 0
    if "flag" not in merged.columns:
        merged["flag"] = ""
    merged["flag"] = merged["flag"].fillna("")

    expected_counts = expected_counts or {}

    statuses: list[str] = []
    flags: list[str] = []
    for _, row in merged.iterrows():
        mark = row["mark"]
        in_sched = mark in sched_marks
        in_plan = mark in plan_marks
        plan_n = int(row["plan_count"])
        flag = row["flag"]

        if in_sched and not in_plan:
            statuses.append("orphan")
            flag = _add_flag(flag, "no_plan_occurrence")
        elif in_plan and not in_sched:
            statuses.append("stray")
            flag = _add_flag(flag, "not_in_schedule")
        else:
            expected = expected_counts.get(mark)
            if expected is None:
                # Without an expected count, only "present on both sides" is asserted.
                statuses.append("matched")
            elif plan_n == expected:
                statuses.append("matched")
            else:
                statuses.append("mismatch")
                flag = _add_flag(flag, f"expected_{expected}_got_{plan_n}")

        flags.append(flag)

    merged["status"] = statuses
    merged["flag"] = flags

    cols = [
        "mark", "type", "size", "notes",
        "count_schedule", "plan_count", "plan_sheets",
        "status", "flag",
    ]
    for c in cols:
        if c not in merged.columns:
            merged[c] = ""
    return merged[cols].sort_values("mark").reset_index(drop=True)


def _empty_schedule_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "mark", "type", "size", "notes", "count_schedule",
    ])


def _empty_plan_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["mark", "source_sheet"])
