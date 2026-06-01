"""Overlay GT vs model predictions on the original floor-plan PNG.

For each (image, model) combo in the chosen split, draws:
  - GT bboxes in RED
  - That model's prediction bboxes in BLUE

Saves to ``outputs/eval/viz/{folder}__{model}.png`` — 18 PNGs for VAL
(6 images × 3 models). Use this to diagnose:
  - Is the model hallucinating bboxes where no doors exist?
  - Is GT incomplete (real doors not labelled)?
  - Are preds close to GT but mis-sized / mis-centered?

Usage::

    .venv/bin/python -m src.viz_eval                       # default: val, swing-v2
    .venv/bin/python -m src.viz_eval --split test
    .venv/bin/python -m src.viz_eval --ver swing-v1        # earlier prompt version
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
GT_CSV = ROOT / "data" / "gt" / "gt_bboxes.csv"
EVAL_DIR = ROOT / "outputs" / "eval"
VIZ_DIR = EVAL_DIR / "viz"

GT_COLOR = (220, 20, 20, 255)     # red
PRED_COLOR = (20, 90, 230, 255)   # blue
GT_WIDTH = 3
PRED_WIDTH = 2


def _draw_box(draw: ImageDraw.ImageDraw, box, color, width):
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def _draw_legend(draw: ImageDraw.ImageDraw, model_key: str,
                 n_gt: int, n_pred: int) -> None:
    """Top-left legend with colour swatches + counts."""
    pad = 10
    line_h = 24
    sw = 22  # swatch size
    text_offset = sw + 10
    w, h = 320, pad * 2 + line_h * 3
    # background
    draw.rectangle([8, 8, 8 + w, 8 + h], fill=(255, 255, 255, 235),
                   outline=(0, 0, 0, 255), width=1)
    y = 8 + pad
    draw.text((8 + pad, y), f"model: {model_key}", fill=(0, 0, 0, 255))
    y += line_h
    draw.rectangle([8 + pad, y, 8 + pad + sw, y + sw], outline=GT_COLOR, width=GT_WIDTH)
    draw.text((8 + pad + text_offset, y + 4), f"GT ({n_gt})", fill=(0, 0, 0, 255))
    y += line_h
    draw.rectangle([8 + pad, y, 8 + pad + sw, y + sw], outline=PRED_COLOR, width=PRED_WIDTH)
    draw.text((8 + pad + text_offset, y + 4), f"pred ({n_pred})", fill=(0, 0, 0, 255))


def main(split: str = "val", ver: str = "swing-v2") -> None:
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    gt = pd.read_csv(GT_CSV)
    gt = gt[gt["split"] == split]
    if gt.empty:
        raise SystemExit(f"No GT rows for split={split!r}")

    preds_csv = EVAL_DIR / f"swing_{ver}_{split}_predictions.csv"
    if not preds_csv.exists():
        raise SystemExit(f"Missing predictions CSV: {preds_csv}")
    preds = pd.read_csv(preds_csv)

    models = sorted(preds["model_key"].unique())
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
            _draw_legend(draw, mkey, len(gt_boxes), len(pred_boxes))

            comp = Image.alpha_composite(base, overlay).convert("RGB")
            safe_folder = folder.replace(" ", "_")
            out_path = VIZ_DIR / f"{safe_folder}__{mkey}.png"
            comp.save(out_path)
            print(f"  {folder} / {mkey}: GT={len(gt_boxes)} pred={len(pred_boxes)} -> {out_path.name}")
        print()

    print(f"\nDone — viz PNGs in {VIZ_DIR}")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render GT+pred overlay PNGs.")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--ver", default="swing-v2", help="prompt version (matches CSV name)")
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    main(split=a.split, ver=a.ver)
