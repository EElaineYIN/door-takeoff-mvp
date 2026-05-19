"""Sampling-based human review + accuracy estimation for Tab 6.

Workflow:
    1. Each model returns its own DataFrame of door rows.
    2. We collect the union of door marks across all models, sample ``N`` of
       them, and build a "long" review DataFrame with one row per
       (door_mark × model) so the estimator can grade each one ✓ / ✗ / Partial.
    3. After grading, ``compute_accuracy`` aggregates by model:
         - row_accuracy = (correct + 0.5 * partial) / graded
         - count_vs_expected = 1 - |n_rows - expected| / expected (if expected given)
         - review_rate = (partial + wrong) / graded
    4. Optional: the user types an "expected total door count" from the
       schedule to enable count accuracy.

This is intentionally per-row (not per-field) for the MVP — finer-grained
field accuracy would need a wider grading UI.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from .model_comparison import DOOR_COLUMNS


GRADE_OPTIONS = ["—", "Correct", "Wrong", "Partial", "Skip"]
DEFAULT_GRADE = "—"


def collect_marks(model_dfs: dict[str, pd.DataFrame]) -> list[str]:
    """Union of non-empty ``mark`` values across all model outputs."""
    marks: set[str] = set()
    for df in model_dfs.values():
        if df is None or df.empty or "mark" not in df.columns:
            continue
        for m in df["mark"].astype(str):
            m = m.strip()
            if m:
                marks.add(m)
    return sorted(marks)


def sample_marks(
    model_dfs: dict[str, pd.DataFrame], n: int, seed: int = 42
) -> list[str]:
    """Random sample of door marks from the pooled union."""
    all_marks = collect_marks(model_dfs)
    if not all_marks:
        return []
    n = min(n, len(all_marks))
    rng = random.Random(seed)
    return sorted(rng.sample(all_marks, n))


def build_review_table(
    model_dfs: dict[str, pd.DataFrame], sampled_marks: list[str]
) -> pd.DataFrame:
    """Long-format review table — one row per (mark × model).

    A model that didn't extract a sampled mark gets a blank row whose grade
    defaults to "—" (i.e. ungraded). That itself is informative — a missing
    row implies the model failed to recall that door.
    """
    rows: list[dict] = []
    for mark in sampled_marks:
        for model_label, df in model_dfs.items():
            if df is None or df.empty or "mark" not in df.columns:
                row = {c: "" for c in DOOR_COLUMNS}
                row["mark"] = mark
                row["model"] = model_label
                row["present"] = False
            else:
                hit = df[df["mark"].astype(str).str.strip() == mark]
                if hit.empty:
                    row = {c: "" for c in DOOR_COLUMNS}
                    row["mark"] = mark
                    row["model"] = model_label
                    row["present"] = False
                else:
                    r = hit.iloc[0].to_dict()
                    row = {c: str(r.get(c, "")) for c in DOOR_COLUMNS}
                    row["model"] = model_label
                    row["present"] = True
            row["grade"] = DEFAULT_GRADE
            rows.append(row)
    cols = ["mark", "model", "present"] + [
        c for c in DOOR_COLUMNS if c not in ("mark", "model")
    ] + ["grade"]
    return pd.DataFrame(rows, columns=cols)


@dataclass
class ModelAccuracy:
    model: str
    graded: int           # rows the user actually graded (Correct/Wrong/Partial)
    correct: int
    partial: int
    wrong: int
    present_rows: int     # rows where the model did extract this mark
    missing_rows: int     # sampled marks the model didn't extract at all
    row_accuracy: float | None
    recall: float | None  # present / sampled
    review_rate: float | None
    count_vs_expected: float | None


def compute_accuracy(
    review_df: pd.DataFrame,
    n_sampled: int,
    model_row_counts: dict[str, int],
    expected_count: int | None = None,
) -> pd.DataFrame:
    """Aggregate grades by model and return a summary DataFrame.

    ``model_row_counts`` maps model label → total rows that model extracted
    (used to compute ``count_vs_expected``).
    """
    if review_df is None or review_df.empty:
        return pd.DataFrame()

    summaries: list[ModelAccuracy] = []
    for model_label, group in review_df.groupby("model", sort=False):
        graded_mask = group["grade"].isin({"Correct", "Wrong", "Partial"})
        graded = int(graded_mask.sum())
        correct = int((group["grade"] == "Correct").sum())
        wrong = int((group["grade"] == "Wrong").sum())
        partial = int((group["grade"] == "Partial").sum())
        present_rows = int(group["present"].sum()) if "present" in group else 0
        missing_rows = max(0, n_sampled - present_rows)

        row_acc = (
            (correct + 0.5 * partial) / graded if graded > 0 else None
        )
        recall = (present_rows / n_sampled) if n_sampled > 0 else None
        review_rate = (
            (wrong + partial) / graded if graded > 0 else None
        )

        total_extracted = int(model_row_counts.get(model_label, 0))
        if expected_count and expected_count > 0:
            err = abs(total_extracted - expected_count) / expected_count
            count_vs_expected = max(0.0, 1.0 - err)
        else:
            count_vs_expected = None

        summaries.append(
            ModelAccuracy(
                model=model_label,
                graded=graded,
                correct=correct,
                partial=partial,
                wrong=wrong,
                present_rows=present_rows,
                missing_rows=missing_rows,
                row_accuracy=row_acc,
                recall=recall,
                review_rate=review_rate,
                count_vs_expected=count_vs_expected,
            )
        )

    out = pd.DataFrame(
        [
            {
                "Model": s.model,
                "Sampled doors found / N": f"{s.present_rows} / {n_sampled}",
                "Recall": f"{s.recall:.0%}" if s.recall is not None else "—",
                "Graded": s.graded,
                "Correct": s.correct,
                "Partial": s.partial,
                "Wrong": s.wrong,
                "Row accuracy": (
                    f"{s.row_accuracy:.0%}" if s.row_accuracy is not None else "—"
                ),
                "Review rate": (
                    f"{s.review_rate:.0%}" if s.review_rate is not None else "—"
                ),
                "Total rows extracted": int(model_row_counts.get(s.model, 0)),
                "Count vs expected": (
                    f"{s.count_vs_expected:.0%}"
                    if s.count_vs_expected is not None
                    else "—"
                ),
            }
            for s in summaries
        ]
    )
    return out
