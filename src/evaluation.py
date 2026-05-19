"""Door-takeoff metrics."""

from __future__ import annotations

import pandas as pd


def compute_metrics(
    schedule_df: pd.DataFrame,
    plan_df: pd.DataFrame,
    reconciled_df: pd.DataFrame,
) -> dict:
    """Return summary metrics for the door takeoff.

    Keys:
        items_in_schedule    — unique marks with a schedule entry
        items_on_plan        — unique marks detected on any plan sheet
        items_matched        — marks present on both sides (status == matched)
        items_orphan         — in schedule, missing from plans
        items_stray          — on plans, missing from schedule
        items_mismatch       — present both sides but count != expected
        plan_detections      — total plan-side detection rows (sum of counts)
        pct_needs_review     — fraction of reconciled rows that aren't 'matched'
    """
    n_schedule = int(schedule_df["mark"].nunique()) if not schedule_df.empty else 0
    n_plan = int(plan_df["mark"].nunique()) if not plan_df.empty else 0
    plan_detections = int(len(plan_df))

    if reconciled_df.empty:
        return {
            "items_in_schedule": n_schedule,
            "items_on_plan": n_plan,
            "items_matched": 0,
            "items_orphan": 0,
            "items_stray": 0,
            "items_mismatch": 0,
            "plan_detections": plan_detections,
            "pct_needs_review": 0.0,
        }

    counts = reconciled_df["status"].value_counts().to_dict()
    matched = int(counts.get("matched", 0))
    orphan = int(counts.get("orphan", 0))
    stray = int(counts.get("stray", 0))
    mismatch = int(counts.get("mismatch", 0))
    total = int(len(reconciled_df))
    needs = total - matched
    pct = (needs / total * 100.0) if total else 0.0

    return {
        "items_in_schedule": n_schedule,
        "items_on_plan": n_plan,
        "items_matched": matched,
        "items_orphan": orphan,
        "items_stray": stray,
        "items_mismatch": mismatch,
        "plan_detections": plan_detections,
        "pct_needs_review": round(pct, 1),
    }
