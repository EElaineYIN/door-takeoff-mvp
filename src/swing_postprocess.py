"""Shrink L-shape predicted bboxes to arc-only squares (Step 2 of 1+2+3).

The CEE329 phase-4 analysis showed that anthropic / openai predict bboxes
that enclose BOTH the door-leaf line AND the quarter-circle arc — an
L-shape with mean h/w ≈ 1.5-1.7. Ground-truth convention is arc-only
(h/w ≈ 1.0). This mismatch costs ~3-4x of IoU.

This script reads any ``*_predictions.csv`` from ``swing_eval.py`` or
``swing_tile_eval.py``, applies an image-content-based shrink to every
prediction whose aspect ratio is L-shape-y, and writes a new predictions
CSV that downstream eval (``swing_reanalyze.py``, ``viz_*``) can consume.

Strategy (per prediction):
1. Read the predicted bbox region from the original image.
2. Threshold dark pixels (door strokes are typically dark on white bg).
3. If h/w ∈ [0.85, 1.15], the bbox is already square — leave alone.
4. Else compute the centroid of dark pixels. Set side = min(h, w).
   Anchor a side×side square at the centroid, clipped to the original
   prediction bbox. This biases the new bbox toward wherever the dark
   pixels concentrate (arc curve + leaf intersection), which empirically
   centers on the arc region.

Usage::

    .venv/bin/python -m src.swing_postprocess \\
        --pred outputs/eval/swingtile_swing-v2_t700_ov15_val_predictions.csv

Writes ``<original_stem>_post_predictions.csv`` in the same directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
GT_CSV = ROOT / "data" / "gt" / "gt_bboxes.csv"

# Aspect-ratio band considered "already square enough" — skip shrinking.
SQUARE_LO = 0.85
SQUARE_HI = 1.15

# Grayscale threshold for "dark pixel" (door stroke).
DARK_THRESH = 128


def _shrink_to_arc(image: Image.Image, x1: float, y1: float,
                   x2: float, y2: float) -> tuple[float, float, float, float]:
    """Image-content-based shrink of an L-shape bbox to an arc-only square.

    Geometric model: the model's L-shape bbox encloses both the door-leaf
    (a straight line along one edge) and the quarter-circle arc. The GT
    arc-only bbox is square (side ≈ door width). The leaf strip is
    perpendicular to the wall and adds extent in one direction.

    Heuristic:
    1. If aspect ratio is already ≈ 1 → return unchanged.
    2. Else side = min(h, w). Identify which end of the long dimension
       contains the LEAF (= a thin strip with most dark pixels concentrated
       in a single row/column) and anchor the side×side square at the
       OPPOSITE end (so the new bbox keeps the arc and drops the leaf).
    3. The "leaf concentration" test: in the leaf strip, dark pixels
       collapse to a small number of rows/cols (straight line);
       in the arc strip, dark pixels are spread over many rows/cols.
       The strip with fewer ACTIVE rows/cols is the leaf side.

    Returns the new (x1, y1, x2, y2) in original image pixel coords.
    """
    ix1, iy1 = int(round(x1)), int(round(y1))
    ix2, iy2 = int(round(x2)), int(round(y2))
    iw_full, ih_full = image.size
    ix1 = max(0, min(iw_full, ix1))
    ix2 = max(0, min(iw_full, ix2))
    iy1 = max(0, min(ih_full, iy1))
    iy2 = max(0, min(ih_full, iy2))
    if ix2 <= ix1 or iy2 <= iy1:
        return (x1, y1, x2, y2)

    h = iy2 - iy1
    w = ix2 - ix1
    aspect = h / w if w > 0 else 1.0

    if SQUARE_LO <= aspect <= SQUARE_HI:
        return (float(ix1), float(iy1), float(ix2), float(iy2))

    crop = image.crop((ix1, iy1, ix2, iy2)).convert("L")
    arr = np.asarray(crop)
    dark = arr < DARK_THRESH
    if not dark.any():
        return (float(ix1), float(iy1), float(ix2), float(iy2))

    side = min(h, w)

    if h > w:
        # Leaf is horizontal — adds extra height. Decide top vs bottom.
        excess = h - side
        # Examine equal-sized strips at the two ends (size = excess px tall).
        top_strip = dark[:excess, :]
        bot_strip = dark[-excess:, :]
        # Number of rows in each strip that contain any dark pixel.
        # Leaf side has fewer active rows (concentrated in one line).
        top_active = int(top_strip.any(axis=1).sum())
        bot_active = int(bot_strip.any(axis=1).sum())
        # Tie-break: also consider total dark pixels (leaf has fewer overall
        # because it's a single line, while arc has more curve pixels).
        if top_active == bot_active:
            top_dense = int(top_strip.sum())
            bot_dense = int(bot_strip.sum())
            leaf_at_top = top_dense < bot_dense
        else:
            leaf_at_top = top_active < bot_active
        if leaf_at_top:
            # Leaf at top → arc at bottom.
            sy1, sy2 = h - side, h
        else:
            sy1, sy2 = 0, side
        sx1, sx2 = 0, w
    else:
        # Leaf is vertical — adds extra width. Decide left vs right.
        excess = w - side
        left_strip = dark[:, :excess]
        right_strip = dark[:, -excess:]
        left_active = int(left_strip.any(axis=0).sum())
        right_active = int(right_strip.any(axis=0).sum())
        if left_active == right_active:
            left_dense = int(left_strip.sum())
            right_dense = int(right_strip.sum())
            leaf_at_left = left_dense < right_dense
        else:
            leaf_at_left = left_active < right_active
        if leaf_at_left:
            sx1, sx2 = w - side, w
        else:
            sx1, sx2 = 0, side
        sy1, sy2 = 0, h

    return (float(ix1 + sx1), float(iy1 + sy1),
            float(ix1 + sx2), float(iy1 + sy2))


def main(pred_path: Path, gt_path: Path = GT_CSV) -> None:
    if not pred_path.exists():
        raise SystemExit(f"Predictions CSV not found: {pred_path}")
    if not gt_path.exists():
        raise SystemExit(f"GT CSV not found: {gt_path}")

    preds_df = pd.read_csv(pred_path)
    gt_df = pd.read_csv(gt_path)

    # Pull image_path per folder from GT (predictions CSV doesn't store it).
    folder_image: dict[str, Path] = {}
    for folder, sub in gt_df.groupby("folder"):
        folder_image[folder] = Path(sub["image_path"].iloc[0])

    print(f"Post-processing {len(preds_df)} predictions from {pred_path.name}")
    print(f"  square band: [{SQUARE_LO}, {SQUARE_HI}], dark threshold: <{DARK_THRESH}\n")

    # Cache loaded images per folder.
    image_cache: dict[str, Image.Image] = {}
    out_rows: list[dict] = []
    n_changed = n_kept = n_skipped = 0
    for i, row in preds_df.iterrows():
        folder = row["folder"]
        image_path = folder_image.get(folder)
        if image_path is None or not image_path.exists():
            # No image — pass through unchanged.
            out_rows.append(row.to_dict())
            n_skipped += 1
            continue
        if folder not in image_cache:
            image_cache[folder] = Image.open(image_path).convert("RGB")
        image = image_cache[folder]

        x1, y1, x2, y2 = (float(row["x1"]), float(row["y1"]),
                          float(row["x2"]), float(row["y2"]))
        nx1, ny1, nx2, ny2 = _shrink_to_arc(image, x1, y1, x2, y2)
        if (abs(nx1 - x1) > 0.5 or abs(ny1 - y1) > 0.5
                or abs(nx2 - x2) > 0.5 or abs(ny2 - y2) > 0.5):
            n_changed += 1
        else:
            n_kept += 1
        new_row = row.to_dict()
        new_row["x1"], new_row["y1"], new_row["x2"], new_row["y2"] = nx1, ny1, nx2, ny2
        out_rows.append(new_row)

    out_df = pd.DataFrame(out_rows, columns=preds_df.columns)

    out_path = pred_path.with_name(
        pred_path.name.replace("_predictions.csv", "_post_predictions.csv")
    )
    out_df.to_csv(out_path, index=False)

    # Print before/after aspect-ratio summary.
    before_w = preds_df["x2"] - preds_df["x1"]
    before_h = preds_df["y2"] - preds_df["y1"]
    before_hw = (before_h / before_w).replace([np.inf, -np.inf], np.nan).dropna()
    after_w = out_df["x2"] - out_df["x1"]
    after_h = out_df["y2"] - out_df["y1"]
    after_hw = (after_h / after_w).replace([np.inf, -np.inf], np.nan).dropna()

    print(f"  changed: {n_changed}  unchanged-square: {n_kept}  skipped: {n_skipped}")
    print(f"  h/w  before: mean={before_hw.mean():.2f} median={before_hw.median():.2f}")
    print(f"  h/w  after:  mean={after_hw.mean():.2f} median={after_hw.median():.2f}")
    print(f"\nPer-model h/w summary AFTER post-processing:")
    out_df["_hw"] = after_h / after_w
    print(out_df.groupby("model_key")["_hw"].describe()[["mean", "50%", "25%", "75%"]].round(2))

    print(f"\nWrote: {out_path}")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shrink L-shape bboxes to arc squares.")
    p.add_argument("--pred", type=Path, required=True,
                   help="Path to *_predictions.csv produced by swing_eval / swing_tile_eval.")
    p.add_argument("--gt", type=Path, default=GT_CSV,
                   help="GT CSV path (used only to look up image_path per folder).")
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    main(pred_path=a.pred, gt_path=a.gt)
