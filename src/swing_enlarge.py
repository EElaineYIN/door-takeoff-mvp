"""Enlarge predicted bboxes with per-model scale factors (Step 4).

The CEE329 phase-5 sweep showed VLM predictions are systematically SMALLER
than the GT arc bbox (anthropic: pred_w/gt_w = 0.66, pred_h/gt_h = 0.83).
A simple center-anchored enlargement by per-model (sx, sy) factors lifts
F1 @ IoU 0.5 substantially.

Best factors (found by 0.1-step grid sweep on VAL):
* anthropic (sx=1.3, sy=1.5): F1@0.5 0.074 -> 0.209  (+183 %)
* gemini    (sx=1.1, sy=1.2): F1@0.5 0.005 -> 0.036
* openai    (sx=2.2, sy=2.2): F1@0.5 0.000 -> 0.023

Reads ``*_predictions.csv`` and writes ``*_enlarged_predictions.csv`` next
to it. The reanalysis CSVs (``swing_reanalyze.py``) can then be regenerated
on the new predictions.

Usage::

    .venv/bin/python -m src.swing_enlarge \\
        --pred outputs/eval/swingtilev3_swing-v3-fs_t700_ov15_up2_val_predictions.csv

Per-model factors can be overridden on the CLI::

    --anthropic-sx 1.3 --anthropic-sy 1.5 \\
    --gemini-sx 1.1    --gemini-sy 1.2 \\
    --openai-sx 2.2    --openai-sy 2.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
GT_CSV = ROOT / "data" / "gt" / "gt_bboxes.csv"

# Best factors from VAL-set grid sweep.
DEFAULT_FACTORS: dict[str, tuple[float, float]] = {
    "anthropic": (1.3, 1.5),
    "gemini":    (1.1, 1.2),
    "openai":    (2.2, 2.2),
}


def _enlarge_row(x1: float, y1: float, x2: float, y2: float,
                 sx: float, sy: float,
                 img_w: int | None = None,
                 img_h: int | None = None
                 ) -> tuple[float, float, float, float]:
    """Center-anchored enlargement, optionally clipped to image bounds."""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * sx
    h = (y2 - y1) * sy
    nx1, ny1 = cx - w / 2.0, cy - h / 2.0
    nx2, ny2 = cx + w / 2.0, cy + h / 2.0
    if img_w is not None:
        nx1 = max(0.0, min(float(img_w), nx1))
        nx2 = max(0.0, min(float(img_w), nx2))
    if img_h is not None:
        ny1 = max(0.0, min(float(img_h), ny1))
        ny2 = max(0.0, min(float(img_h), ny2))
    return nx1, ny1, nx2, ny2


def main(pred_path: Path, gt_path: Path,
         factors: dict[str, tuple[float, float]],
         clip_to_image: bool = True) -> Path:
    if not pred_path.exists():
        raise SystemExit(f"Predictions CSV not found: {pred_path}")
    if not gt_path.exists():
        raise SystemExit(f"GT CSV not found: {gt_path}")

    preds_df = pd.read_csv(pred_path)
    gt_df = pd.read_csv(gt_path)

    # Per-folder image dimensions for clipping.
    folder_size: dict[str, tuple[int, int]] = {}
    if clip_to_image:
        for folder, sub in gt_df.groupby("folder"):
            row = sub.iloc[0]
            iw = int(row.get("image_w", 0))
            ih = int(row.get("image_h", 0))
            if iw == 0 or ih == 0:
                p = Path(row["image_path"])
                if p.exists():
                    with Image.open(p) as im:
                        iw, ih = im.size
            folder_size[folder] = (iw, ih)

    print(f"Enlarging {len(preds_df)} predictions from {pred_path.name}")
    for model, (sx, sy) in factors.items():
        n = int((preds_df["model_key"] == model).sum())
        print(f"  {model:10s}  sx={sx}  sy={sy}   ({n} preds)")
    print()

    out_rows: list[dict] = []
    n_changed = n_unchanged = n_unknown_model = 0
    for _, row in preds_df.iterrows():
        model = row["model_key"]
        if model not in factors:
            out_rows.append(row.to_dict())
            n_unknown_model += 1
            continue
        sx, sy = factors[model]
        if sx == 1.0 and sy == 1.0:
            out_rows.append(row.to_dict())
            n_unchanged += 1
            continue
        iw = ih = None
        if clip_to_image:
            iw, ih = folder_size.get(row["folder"], (None, None))
        nx1, ny1, nx2, ny2 = _enlarge_row(
            float(row["x1"]), float(row["y1"]),
            float(row["x2"]), float(row["y2"]),
            sx, sy, iw, ih,
        )
        new_row = row.to_dict()
        new_row["x1"], new_row["y1"], new_row["x2"], new_row["y2"] = nx1, ny1, nx2, ny2
        out_rows.append(new_row)
        n_changed += 1

    out_df = pd.DataFrame(out_rows, columns=preds_df.columns)
    out_path = pred_path.with_name(
        pred_path.name.replace("_predictions.csv", "_enlarged_predictions.csv")
    )
    out_df.to_csv(out_path, index=False)

    # h/w summary per model.
    out_df["_w"] = out_df["x2"] - out_df["x1"]
    out_df["_h"] = out_df["y2"] - out_df["y1"]
    out_df["_hw"] = out_df["_h"] / out_df["_w"]
    print(f"  changed: {n_changed}  unchanged-1.0x: {n_unchanged}  unknown-model: {n_unknown_model}")
    print()
    print("Per-model bbox summary AFTER enlargement:")
    summary = out_df.groupby("model_key")[["_w", "_h", "_hw"]].mean().round(2)
    print(summary)
    print(f"\nWrote: {out_path}")
    return out_path


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enlarge predicted bboxes per-model.")
    p.add_argument("--pred", type=Path, required=True,
                   help="Path to *_predictions.csv to enlarge.")
    p.add_argument("--gt", type=Path, default=GT_CSV,
                   help="GT CSV path (used for image dimensions).")
    p.add_argument("--no-clip", action="store_true",
                   help="Do not clip enlarged bboxes to image bounds.")
    for m in DEFAULT_FACTORS:
        sx_def, sy_def = DEFAULT_FACTORS[m]
        p.add_argument(f"--{m}-sx", type=float, default=sx_def,
                       help=f"Width scale for {m} (default {sx_def}).")
        p.add_argument(f"--{m}-sy", type=float, default=sy_def,
                       help=f"Height scale for {m} (default {sy_def}).")
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    factors = {
        m: (getattr(a, f"{m}_sx"), getattr(a, f"{m}_sy"))
        for m in DEFAULT_FACTORS
    }
    main(pred_path=a.pred, gt_path=a.gt, factors=factors,
         clip_to_image=not a.no_clip)
