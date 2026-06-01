"""Overlay GT vs tile-based predictions.

Same as ``viz_eval`` but reads the predictions CSV produced by
``swing_tile_eval`` (filename pattern ``swingtile_<ver>_t<size>_ov<pct>_<split>_predictions.csv``).

Usage::

    .venv/bin/python -m src.viz_tile_eval                                    # default
    .venv/bin/python -m src.viz_tile_eval --split val --ver swing-v2 --tile 700 --overlap 15
    .venv/bin/python -m src.viz_tile_eval --pred outputs/eval/swingtile_swing-v2_t700_ov15_val_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from src.viz_eval import (
    GT_COLOR,
    GT_WIDTH,
    PRED_COLOR,
    PRED_WIDTH,
    _draw_box,
    _draw_legend,
)


ROOT = Path(__file__).resolve().parents[1]
GT_CSV = ROOT / "data" / "gt" / "gt_bboxes.csv"
EVAL_DIR = ROOT / "outputs" / "eval"
VIZ_DIR = EVAL_DIR / "viz"


def main(pred_csv: Path, suffix: str) -> None:
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    if not pred_csv.exists():
        raise SystemExit(f"Missing predictions CSV: {pred_csv}")
    preds = pd.read_csv(pred_csv)

    if "split" not in preds.columns:
        raise SystemExit("predictions CSV missing 'split' column")
    splits = preds["split"].unique()
    if len(splits) != 1:
        raise SystemExit(f"expected 1 split in predictions, got {splits!r}")
    split = splits[0]

    gt = pd.read_csv(GT_CSV)
    gt = gt[gt["split"] == split]
    if gt.empty:
        raise SystemExit(f"No GT rows for split={split!r}")

    models = sorted(preds["model_key"].unique())
    print(f"Predictions CSV: {pred_csv.name}")
    print(f"Models: {models}")
    print(f"Folders: {gt['folder'].nunique()}\n")

    for folder in sorted(gt["folder"].unique()):
        fg = gt[gt["folder"] == folder]
        image_path = Path(fg["image_path"].iloc[0])
        if not image_path.exists():
            print(f"SKIP {folder}: image not found at {image_path}")
            continue

        base = Image.open(image_path).convert("RGBA")
        gt_boxes = list(zip(fg["x1"], fg["y1"], fg["x2"], fg["y2"]))
        fp = preds[preds["folder"] == folder]

        for mkey in models:
            mp = fp[fp["model_key"] == mkey]
            pred_boxes = list(zip(mp["x1"], mp["y1"], mp["x2"], mp["y2"]))

            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            for b in gt_boxes:
                _draw_box(draw, b, GT_COLOR, GT_WIDTH)
            for b in pred_boxes:
                _draw_box(draw, b, PRED_COLOR, PRED_WIDTH)
            _draw_legend(draw, f"{mkey} (tile)", len(gt_boxes), len(pred_boxes))

            comp = Image.alpha_composite(base, overlay).convert("RGB")
            safe_folder = folder.replace(" ", "_")
            out_path = VIZ_DIR / f"{safe_folder}__{mkey}__{suffix}.png"
            comp.save(out_path)
            print(f"  {folder} / {mkey}: GT={len(gt_boxes)} pred={len(pred_boxes)} -> {out_path.name}")
        print()

    print(f"\nDone — tile viz PNGs in {VIZ_DIR}")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render GT + tile-pred overlays.")
    p.add_argument("--pred", type=Path, default=None,
                   help="Predictions CSV (default: build from --ver/--tile/--overlap/--split).")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--ver", default="swing-v2")
    p.add_argument("--tile", type=int, default=700)
    p.add_argument("--overlap", type=int, default=15, help="overlap percent (default 15)")
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    if a.pred is not None:
        pred = a.pred
        suffix = pred.stem.replace(f"_{a.split}_predictions", "").replace("swingtile_", "tile_")
    else:
        pred = EVAL_DIR / f"swingtile_{a.ver}_t{a.tile}_ov{a.overlap}_{a.split}_predictions.csv"
        suffix = f"tile_{a.ver}_t{a.tile}_ov{a.overlap}"
    main(pred, suffix)
