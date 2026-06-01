"""Phase 3 — swing-only detection evaluation against GT bboxes.

Reads ``data/gt/gt_bboxes.csv`` (produced by ``gt_parser.py``), runs the
swing-only prompt on every image in the chosen split (default ``val``)
across all 3 models in :data:`model_comparison.MODELS`, and writes:

* ``swing_{ver}_{split}_predictions.csv`` — every model prediction in
  pixel-space (one row per pred).
* ``swing_{ver}_{split}_matches.csv`` — per-prediction TP / FP and
  per-GT FN, with the matched IoU.
* ``swing_{ver}_{split}_metrics_by_sheet.csv`` — Precision / Recall / F1
  per (model, sheet).
* ``swing_{ver}_{split}_metrics_overall.csv`` — Precision / Recall / F1
  per model across the whole split.

Matching is greedy at IoU ≥ 0.5 (configurable). Predictions are sorted
by confidence descending; the first prediction to clear the IoU bar with
an unmatched GT consumes that GT.

Usage::

    .venv/bin/python -m src.swing_eval                # default: val split
    .venv/bin/python -m src.swing_eval --split test   # final eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.model_comparison import (
    DOOR_PLAN_SWING_PROMPT_VERSION,
    MODELS,
    run_plan_swing_model,
)


ROOT = Path(__file__).resolve().parents[1]
GT_CSV = ROOT / "data" / "gt" / "gt_bboxes.csv"
OUT_DIR = ROOT / "outputs" / "eval"

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.0  # keep all detections for baseline; tune later


# ----------------------------------------------------------------- IoU helpers

def _iou(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    """IoU between two pixel bboxes ``(x1, y1, x2, y2)``."""
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


def _greedy_match(preds: list[dict], gts: list[dict],
                  iou_thresh: float) -> tuple[list[dict], list[int], list[int]]:
    """Greedy bbox matching, predictions consumed in confidence-desc order.

    Returns ``(matches, fp_pred_idxs, fn_gt_idxs)`` where ``matches`` is a
    list of ``{pred_idx, gt_idx, iou}`` dicts.
    """
    preds_sorted = sorted(preds, key=lambda p: -p["confidence"])
    matched_gt: set[int] = set()
    matches: list[dict] = []
    fps: list[int] = []
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


# ------------------------------------------------------------------- main loop

def _gt_for_image(folder_gt: pd.DataFrame) -> list[dict]:
    return [
        {
            "idx": int(i),
            "x1": float(r["x1"]),
            "y1": float(r["y1"]),
            "x2": float(r["x2"]),
            "y2": float(r["y2"]),
        }
        for i, r in folder_gt.reset_index(drop=True).iterrows()
    ]


def _preds_for_image(model_df: pd.DataFrame, img_w: int, img_h: int,
                     conf_threshold: float) -> list[dict]:
    out: list[dict] = []
    for i, r in model_df.reset_index(drop=True).iterrows():
        try:
            conf = float(r.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        if conf < conf_threshold:
            continue
        out.append({
            "idx": int(i),
            "x1": float(r["bbox_x1"]) * img_w,
            "y1": float(r["bbox_y1"]) * img_h,
            "x2": float(r["bbox_x2"]) * img_w,
            "y2": float(r["bbox_y2"]) * img_h,
            "confidence": conf,
        })
    return out


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main(split: str = "val", iou_thresh: float = IOU_THRESHOLD,
         conf_thresh: float = CONF_THRESHOLD) -> None:
    load_dotenv(ROOT / ".env", override=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt = pd.read_csv(GT_CSV)
    split_gt = gt[gt["split"] == split].copy()
    if split_gt.empty:
        raise SystemExit(f"No GT rows for split={split!r} in {GT_CSV}")

    print(
        f"\n{split.upper()} set: {split_gt['folder'].nunique()} images, "
        f"{len(split_gt)} GT bboxes — IoU≥{iou_thresh}, conf≥{conf_thresh}\n"
    )

    pred_rows: list[dict] = []
    match_rows: list[dict] = []

    # group preserves folder ordering so the console log reads in a stable order
    for folder, folder_gt in split_gt.groupby("folder"):
        image_path = Path(folder_gt["image_path"].iloc[0])
        sheet = folder_gt["sheet"].iloc[0]
        floor = folder_gt["floor"].iloc[0]
        img_w = int(folder_gt["image_w"].iloc[0])
        img_h = int(folder_gt["image_h"].iloc[0])

        gts = _gt_for_image(folder_gt)
        print(f"=== {folder} ({sheet} / {floor}) — {img_w}x{img_h}, {len(gts)} GT ===")

        if not image_path.exists():
            print(f"  SKIP: image not found at {image_path}\n")
            continue
        image_bytes = image_path.read_bytes()

        for model_key in MODELS:
            run = run_plan_swing_model(model_key, image_bytes, use_cache=True)
            if run.error:
                print(f"  {model_key:10s} ERROR — {run.error}")
                continue

            preds = _preds_for_image(run.df, img_w, img_h, conf_thresh)
            for p in preds:
                pred_rows.append({
                    "split": split, "folder": folder, "sheet": sheet, "floor": floor,
                    "model_key": model_key, "model": run.spec.label,
                    "pred_idx": p["idx"],
                    "x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"],
                    "confidence": p["confidence"],
                })

            matches, fps, fns = _greedy_match(preds, gts, iou_thresh)
            tp, fp, fn = len(matches), len(fps), len(fns)
            p_, r_, f1_ = _prf1(tp, fp, fn)
            tag = "cache" if run.cached else f"{run.elapsed_s:4.1f}s"
            print(
                f"  {model_key:10s} [{tag:>6}] "
                f"preds={len(preds):3d}  TP={tp:3d} FP={fp:3d} FN={fn:3d}  "
                f"P={p_:.3f} R={r_:.3f} F1={f1_:.3f}"
            )

            for m in matches:
                match_rows.append({
                    "split": split, "folder": folder, "sheet": sheet,
                    "model_key": model_key, "status": "TP",
                    "pred_idx": m["pred_idx"], "gt_idx": m["gt_idx"], "iou": m["iou"],
                })
            for fp_idx in fps:
                match_rows.append({
                    "split": split, "folder": folder, "sheet": sheet,
                    "model_key": model_key, "status": "FP",
                    "pred_idx": fp_idx, "gt_idx": None, "iou": None,
                })
            for fn_idx in fns:
                match_rows.append({
                    "split": split, "folder": folder, "sheet": sheet,
                    "model_key": model_key, "status": "FN",
                    "pred_idx": None, "gt_idx": fn_idx, "iou": None,
                })
        print()

    pred_df = pd.DataFrame(pred_rows)
    match_df = pd.DataFrame(match_rows)

    # ----------------------------------------------------------- aggregations

    def _agg(group: pd.DataFrame) -> pd.Series:
        tp = int((group["status"] == "TP").sum())
        fp = int((group["status"] == "FP").sum())
        fn = int((group["status"] == "FN").sum())
        p_, r_, f1_ = _prf1(tp, fp, fn)
        return pd.Series({"TP": tp, "FP": fp, "FN": fn,
                          "P": round(p_, 4), "R": round(r_, 4), "F1": round(f1_, 4)})

    if match_df.empty:
        print("No predictions produced — likely all API keys missing. "
              "Add them to .env and rerun.")
        return

    by_sheet = (
        match_df.groupby(["model_key", "sheet"], as_index=False)
                .apply(_agg, include_groups=False)
                .reset_index(drop=True)
    )
    overall = (
        match_df.groupby(["model_key"], as_index=False)
                .apply(_agg, include_groups=False)
                .reset_index(drop=True)
    )

    # ----------------------------------------------------------------- write

    ver = DOOR_PLAN_SWING_PROMPT_VERSION
    pred_csv = OUT_DIR / f"swing_{ver}_{split}_predictions.csv"
    match_csv = OUT_DIR / f"swing_{ver}_{split}_matches.csv"
    sheet_csv = OUT_DIR / f"swing_{ver}_{split}_metrics_by_sheet.csv"
    overall_csv = OUT_DIR / f"swing_{ver}_{split}_metrics_overall.csv"

    pred_df.to_csv(pred_csv, index=False)
    match_df.to_csv(match_csv, index=False)
    by_sheet.to_csv(sheet_csv, index=False)
    overall.to_csv(overall_csv, index=False)

    print("\n=== Metrics by (model, sheet) ===")
    print(by_sheet.to_string(index=False))
    print("\n=== Overall metrics (across split) ===")
    print(overall.to_string(index=False))

    print(f"\nWrote:\n  {pred_csv}\n  {match_csv}\n  {sheet_csv}\n  {overall_csv}")


# -------------------------------------------------------------------------- cli

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Swing-door detection eval vs GT.")
    p.add_argument("--split", default="val", choices=["val", "test"],
                   help="GT split to evaluate (default: val).")
    p.add_argument("--iou", type=float, default=IOU_THRESHOLD,
                   help=f"IoU matching threshold (default: {IOU_THRESHOLD}).")
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                   help=f"Min confidence to keep a prediction "
                        f"(default: {CONF_THRESHOLD}).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(split=args.split, iou_thresh=args.iou, conf_thresh=args.conf)
