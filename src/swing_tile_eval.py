"""Tile-based swing-door detection eval.

Slices each image into a small grid of overlapping tiles, runs the swing
prompt per tile, converts tile-relative bboxes back to full-image pixel
coordinates, NMS-merges across tile boundaries, then runs the same IoU
eval as ``swing_eval.py``.

Why tiling helps: on a 688-px-wide A2.x plan, a swing-door arc occupies
~4% of image width. A VLM at modest resolution has trouble placing a
pixel-tight bbox at that scale. Splitting the plan into a 2×2 grid makes
each door ~8% of tile width, doubling the effective resolution budget
the VLM has for localization.

Tile grid is chosen automatically per image so each tile is roughly
``tile_target`` px on a side (default 700). For A2.x plans this gives
2×2 = 4 tiles. For the 1728×4032 A7.1.1 Typical plan, 2×6 = 12 tiles.

Usage::

    .venv/bin/python -m src.swing_tile_eval                       # default: val
    .venv/bin/python -m src.swing_tile_eval --split test
    .venv/bin/python -m src.swing_tile_eval --tile 600 --overlap 0.2

Cache: each tile is hashed independently so partial reruns hit the
same ``swing-v2`` cache namespace as ``swing_eval.py``.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from PIL import Image

from src.model_comparison import (
    DOOR_PLAN_SWING_PROMPT_VERSION,
    MODELS,
    run_plan_swing_model,
)


ROOT = Path(__file__).resolve().parents[1]
GT_CSV = ROOT / "data" / "gt" / "gt_bboxes.csv"
OUT_DIR = ROOT / "outputs" / "eval"

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.0
TILE_TARGET = 700           # target tile size px per side
OVERLAP = 0.15              # fractional overlap between adjacent tiles
NMS_IOU = 0.4               # cross-tile dedup threshold
MIN_TILES_PER_DIM = 2       # force ≥ 2 tiles per dim if dim > 500 (else 1)


# --------------------------------------------------------------------- tiling

def _grid_origins_1d(img_dim: int, n_tiles: int, overlap: float):
    """Return list of (origin, tile_dim) so n tiles cover img_dim with
    the requested fractional overlap. Last tile is anchored at img_dim."""
    if n_tiles <= 1:
        return [(0, img_dim)]
    tile_dim = int(round(img_dim / (1 + (n_tiles - 1) * (1 - overlap))))
    step = (img_dim - tile_dim) / (n_tiles - 1)
    return [(int(round(i * step)), tile_dim) for i in range(n_tiles)]


def _grid_for_image(img_w: int, img_h: int, tile_target: int,
                    overlap: float):
    n_cols = max(MIN_TILES_PER_DIM if img_w > 500 else 1,
                 round(img_w / tile_target))
    n_rows = max(MIN_TILES_PER_DIM if img_h > 500 else 1,
                 round(img_h / tile_target))
    xs = _grid_origins_1d(img_w, n_cols, overlap)
    ys = _grid_origins_1d(img_h, n_rows, overlap)
    return [(x, y, w, h) for (y, h) in ys for (x, w) in xs]


def _slice_image(image_bytes: bytes, tiles) -> list[bytes]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    out = []
    for x, y, w, h in tiles:
        crop = img.crop((x, y, x + w, y + h))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


# ------------------------------------------------------------------ IoU utils

def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _nms(preds: list[dict], iou_thresh: float) -> list[dict]:
    """Confidence-sorted greedy NMS to dedupe overlapping cross-tile dets."""
    out: list[dict] = []
    for p in sorted(preds, key=lambda q: -q["confidence"]):
        keep = True
        for k in out:
            if _iou(
                (p["x1"], p["y1"], p["x2"], p["y2"]),
                (k["x1"], k["y1"], k["x2"], k["y2"]),
            ) >= iou_thresh:
                keep = False
                break
        if keep:
            out.append(p)
    return out


def _greedy_match(preds, gts, iou_thresh):
    preds_sorted = sorted(preds, key=lambda p: -p["confidence"])
    matched_gt: set[int] = set()
    matches, fps = [], []
    for p in preds_sorted:
        best_g, best_iou = None, 0.0
        for g in gts:
            if g["idx"] in matched_gt:
                continue
            iou = _iou(
                (p["x1"], p["y1"], p["x2"], p["y2"]),
                (g["x1"], g["y1"], g["x2"], g["y2"]),
            )
            if iou >= iou_thresh and iou > best_iou:
                best_g, best_iou = g, iou
        if best_g is not None:
            matched_gt.add(best_g["idx"])
            matches.append({"pred_idx": p["idx"], "gt_idx": best_g["idx"], "iou": best_iou})
        else:
            fps.append(p["idx"])
    fns = [g["idx"] for g in gts if g["idx"] not in matched_gt]
    return matches, fps, fns


def _prf1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# ------------------------------------------------------------------- main

def main(split: str = "val", iou_thresh: float = IOU_THRESHOLD,
         conf_thresh: float = CONF_THRESHOLD, tile_target: int = TILE_TARGET,
         overlap: float = OVERLAP, nms_iou: float = NMS_IOU) -> None:
    load_dotenv(ROOT / ".env", override=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt = pd.read_csv(GT_CSV)
    split_gt = gt[gt["split"] == split].copy()
    if split_gt.empty:
        raise SystemExit(f"No GT rows for split={split!r}")

    print(
        f"\n{split.upper()} TILE eval: {split_gt['folder'].nunique()} images, "
        f"{len(split_gt)} GT — IoU≥{iou_thresh}, tile_target={tile_target}px, "
        f"overlap={overlap:.0%}, NMS IoU={nms_iou}\n"
    )

    pred_rows: list[dict] = []
    match_rows: list[dict] = []

    for folder, folder_gt in split_gt.groupby("folder"):
        image_path = Path(folder_gt["image_path"].iloc[0])
        sheet = folder_gt["sheet"].iloc[0]
        floor = folder_gt["floor"].iloc[0]
        img_w = int(folder_gt["image_w"].iloc[0])
        img_h = int(folder_gt["image_h"].iloc[0])

        if not image_path.exists():
            print(f"SKIP {folder}: image not found")
            continue

        tiles = _grid_for_image(img_w, img_h, tile_target, overlap)
        image_bytes = image_path.read_bytes()
        tile_bytes_list = _slice_image(image_bytes, tiles)

        gts_list = [
            {"idx": int(i), "x1": float(r["x1"]), "y1": float(r["y1"]),
             "x2": float(r["x2"]), "y2": float(r["y2"])}
            for i, r in folder_gt.reset_index(drop=True).iterrows()
        ]

        print(f"=== {folder} ({sheet}/{floor}) — {img_w}x{img_h}, "
              f"{len(tiles)} tiles, {len(gts_list)} GT ===")

        for model_key in MODELS:
            all_preds: list[dict] = []
            cache_hits = 0
            api_calls = 0
            total_t = 0.0

            for ti, ((x0, y0, tw, th), tb) in enumerate(zip(tiles, tile_bytes_list)):
                run = run_plan_swing_model(model_key, tb, use_cache=True)
                if run.cached:
                    cache_hits += 1
                else:
                    api_calls += 1
                    total_t += run.elapsed_s
                if run.error:
                    continue
                for j, r in run.df.reset_index(drop=True).iterrows():
                    try:
                        conf = float(r.get("confidence", 0.5))
                    except (TypeError, ValueError):
                        conf = 0.5
                    if conf < conf_thresh:
                        continue
                    fx1 = x0 + float(r["bbox_x1"]) * tw
                    fy1 = y0 + float(r["bbox_y1"]) * th
                    fx2 = x0 + float(r["bbox_x2"]) * tw
                    fy2 = y0 + float(r["bbox_y2"]) * th
                    if fx2 <= fx1 or fy2 <= fy1:
                        continue
                    all_preds.append({
                        "tile_idx": ti,
                        "x1": fx1, "y1": fy1, "x2": fx2, "y2": fy2,
                        "confidence": conf,
                    })

            merged = _nms(all_preds, iou_thresh=nms_iou)
            for k, p in enumerate(merged):
                p["idx"] = k
                pred_rows.append({
                    "split": split, "folder": folder, "sheet": sheet, "floor": floor,
                    "model_key": model_key, "pred_idx": k,
                    "x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"],
                    "confidence": p["confidence"],
                })

            matches, fps, fns = _greedy_match(merged, gts_list, iou_thresh)
            tp, fp, fn = len(matches), len(fps), len(fns)
            p_, r_, f1_ = _prf1(tp, fp, fn)
            tag = "cache" if api_calls == 0 else f"{total_t:5.1f}s"
            print(
                f"  {model_key:10s} [{tag:>6}] {cache_hits}/{len(tiles)} cache  "
                f"raw={len(all_preds):3d} merged={len(merged):3d}  "
                f"TP={tp:3d} FP={fp:3d} FN={fn:3d}  "
                f"P={p_:.3f} R={r_:.3f} F1={f1_:.3f}"
            )

            for m in matches:
                match_rows.append({"split": split, "folder": folder, "sheet": sheet,
                                   "model_key": model_key, "status": "TP",
                                   "pred_idx": m["pred_idx"], "gt_idx": m["gt_idx"],
                                   "iou": m["iou"]})
            for fp_idx in fps:
                match_rows.append({"split": split, "folder": folder, "sheet": sheet,
                                   "model_key": model_key, "status": "FP",
                                   "pred_idx": fp_idx, "gt_idx": None, "iou": None})
            for fn_idx in fns:
                match_rows.append({"split": split, "folder": folder, "sheet": sheet,
                                   "model_key": model_key, "status": "FN",
                                   "pred_idx": None, "gt_idx": fn_idx, "iou": None})
        print()

    pred_df = pd.DataFrame(pred_rows)
    match_df = pd.DataFrame(match_rows)

    if match_df.empty:
        print("No predictions produced — check API keys / models.")
        return

    def _agg(group: pd.DataFrame) -> pd.Series:
        tp = int((group["status"] == "TP").sum())
        fp = int((group["status"] == "FP").sum())
        fn = int((group["status"] == "FN").sum())
        p_, r_, f1_ = _prf1(tp, fp, fn)
        return pd.Series({"TP": tp, "FP": fp, "FN": fn,
                          "P": round(p_, 4), "R": round(r_, 4), "F1": round(f1_, 4)})

    by_sheet = (match_df.groupby(["model_key", "sheet"], as_index=False)
                .apply(_agg, include_groups=False).reset_index(drop=True))
    overall = (match_df.groupby(["model_key"], as_index=False)
               .apply(_agg, include_groups=False).reset_index(drop=True))

    ver = DOOR_PLAN_SWING_PROMPT_VERSION
    suffix = f"{ver}_t{tile_target}_ov{int(overlap*100)}"
    pred_csv = OUT_DIR / f"swingtile_{suffix}_{split}_predictions.csv"
    match_csv = OUT_DIR / f"swingtile_{suffix}_{split}_matches.csv"
    sheet_csv = OUT_DIR / f"swingtile_{suffix}_{split}_metrics_by_sheet.csv"
    overall_csv = OUT_DIR / f"swingtile_{suffix}_{split}_metrics_overall.csv"

    pred_df.to_csv(pred_csv, index=False)
    match_df.to_csv(match_csv, index=False)
    by_sheet.to_csv(sheet_csv, index=False)
    overall.to_csv(overall_csv, index=False)

    print("\n=== Metrics by (model, sheet) ===")
    print(by_sheet.to_string(index=False))
    print("\n=== Overall metrics (across split) ===")
    print(overall.to_string(index=False))
    print(f"\nWrote:\n  {pred_csv}\n  {match_csv}\n  {sheet_csv}\n  {overall_csv}")


# -------------------------------------------------------------------- cli

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tile-based swing-door eval.")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    p.add_argument("--tile", type=int, default=TILE_TARGET,
                   help="Target tile size px per side (default 700).")
    p.add_argument("--overlap", type=float, default=OVERLAP,
                   help="Fractional overlap between adjacent tiles (default 0.15).")
    p.add_argument("--nms", type=float, default=NMS_IOU,
                   help="NMS IoU threshold for cross-tile dedup (default 0.4).")
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    main(split=a.split, iou_thresh=a.iou, conf_thresh=a.conf,
         tile_target=a.tile, overlap=a.overlap, nms_iou=a.nms)
