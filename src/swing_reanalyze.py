"""Multi-threshold + counting + center-distance re-analysis.

Takes an existing ``*_predictions.csv`` from ``swing_eval.py`` or
``swing_tile_eval.py`` and reports the same TP/FP/FN/F1 numbers at
**multiple IoU thresholds** plus two threshold-free metrics:

* **Counting accuracy** — per (model, sheet): predicted count vs GT count.
  This is the metric that actually matters for door takeoff in practice
  (the contractor needs the right *number* of doors per sheet, not pixel
  IoU). Reported as ``count_err = |pred_count - gt_count|`` and
  ``count_pct_err = count_err / max(gt_count, 1)``.

* **Center-distance F1** — pred and GT centers are matched greedily by
  Euclidean distance (in pixels). A match is a TP if center distance
  ≤ ``dist_thresh`` (default ~one mean-GT-bbox-width). This relaxes IoU
  while still requiring spatial alignment, and reflects "did the model
  point at the right door, even if its bbox is mis-shaped?".

Reads the predictions CSV from ``outputs/eval/`` and the GT CSV from
``data/gt/gt_bboxes.csv``. Writes a Markdown summary +
``*_reanalysis.csv`` next to the input.

Usage::

    .venv/bin/python -m src.swing_reanalyze \\
        --pred outputs/eval/swing_swing-v2_val_predictions.csv

    .venv/bin/python -m src.swing_reanalyze \\
        --pred outputs/eval/swingtile_swing-v2_t700_ov15_val_predictions.csv \\
        --iou 0.1 0.2 0.3 0.5
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GT_CSV_DEFAULT = ROOT / "data" / "gt" / "gt_bboxes.csv"

DEFAULT_IOUS = [0.1, 0.2, 0.3, 0.5]


# ------------------------------------------------------------------ geometry

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


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _euclid(c1, c2) -> float:
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


# ----------------------------------------------------------------- matching

def _match_iou(preds, gts, iou_thresh):
    preds_sorted = sorted(preds, key=lambda p: -p["confidence"])
    matched_gt: set[int] = set()
    tp = fp = 0
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
            tp += 1
        else:
            fp += 1
    fn = len(gts) - tp
    return tp, fp, fn


def _match_distance(preds, gts, dist_thresh):
    """Greedy center-distance matching, conf-desc."""
    preds_sorted = sorted(preds, key=lambda p: -p["confidence"])
    matched_gt: set[int] = set()
    tp = fp = 0
    matched_dists: list[float] = []
    for p in preds_sorted:
        pc = _center((p["x1"], p["y1"], p["x2"], p["y2"]))
        best_g, best_d = None, float("inf")
        for g in gts:
            if g["idx"] in matched_gt:
                continue
            gc = _center((g["x1"], g["y1"], g["x2"], g["y2"]))
            d = _euclid(pc, gc)
            if d <= dist_thresh and d < best_d:
                best_g, best_d = g, d
        if best_g is not None:
            matched_gt.add(best_g["idx"])
            tp += 1
            matched_dists.append(best_d)
        else:
            fp += 1
    fn = len(gts) - tp
    return tp, fp, fn, matched_dists


def _prf1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# ------------------------------------------------------------------- main

def main(pred_path: Path, gt_path: Path, iou_list: list[float],
         dist_thresh_factor: float = 1.0) -> None:
    if not pred_path.exists():
        raise SystemExit(f"Predictions CSV not found: {pred_path}")
    if not gt_path.exists():
        raise SystemExit(f"GT CSV not found: {gt_path}")

    preds_df = pd.read_csv(pred_path)
    gt_df = pd.read_csv(gt_path)

    if "split" not in preds_df.columns:
        raise SystemExit("predictions CSV missing 'split' column")
    splits = preds_df["split"].unique()
    if len(splits) != 1:
        raise SystemExit(f"expected 1 split in predictions, got {splits!r}")
    split = splits[0]
    gt_df = gt_df[gt_df["split"] == split].copy()

    # mean GT bbox edge length (px) — used for default distance threshold
    gt_w = (gt_df["x2"] - gt_df["x1"]).mean()
    gt_h = (gt_df["y2"] - gt_df["y1"]).mean()
    mean_gt_edge = float((gt_w + gt_h) / 2.0)
    dist_thresh = dist_thresh_factor * mean_gt_edge

    print(f"\nReanalyzing {pred_path.name}")
    print(f"  split={split}  GT={len(gt_df)} bboxes across {gt_df['folder'].nunique()} images")
    print(f"  IoU thresholds: {iou_list}")
    print(f"  center-distance threshold: {dist_thresh:.1f}px "
          f"(={dist_thresh_factor:.2f} × mean-GT-edge {mean_gt_edge:.1f}px)\n")

    # ----------------------------------------------- per (model, image) accum

    iou_rows: list[dict] = []      # one row per (model, folder, iou)
    count_rows: list[dict] = []    # one row per (model, folder)
    dist_rows: list[dict] = []     # one row per (model, folder)

    for model_key in sorted(preds_df["model_key"].unique()):
        model_preds = preds_df[preds_df["model_key"] == model_key]

        for folder in sorted(gt_df["folder"].unique()):
            folder_gt = gt_df[gt_df["folder"] == folder]
            sheet = folder_gt["sheet"].iloc[0]
            n_gt = len(folder_gt)

            gts = [
                {"idx": int(i), "x1": float(r["x1"]), "y1": float(r["y1"]),
                 "x2": float(r["x2"]), "y2": float(r["y2"])}
                for i, r in folder_gt.reset_index(drop=True).iterrows()
            ]

            fp_preds = model_preds[model_preds["folder"] == folder]
            preds = []
            for i, r in fp_preds.reset_index(drop=True).iterrows():
                try:
                    conf = float(r.get("confidence", 0.5))
                except (TypeError, ValueError):
                    conf = 0.5
                preds.append({"idx": int(i), "x1": float(r["x1"]), "y1": float(r["y1"]),
                              "x2": float(r["x2"]), "y2": float(r["y2"]),
                              "confidence": conf})
            n_pred = len(preds)

            # IoU sweep
            for iou_t in iou_list:
                tp, fp, fn = _match_iou(preds, gts, iou_t)
                p_, r_, f1_ = _prf1(tp, fp, fn)
                iou_rows.append({
                    "model_key": model_key, "folder": folder, "sheet": sheet,
                    "iou": iou_t,
                    "n_gt": n_gt, "n_pred": n_pred,
                    "TP": tp, "FP": fp, "FN": fn,
                    "P": round(p_, 4), "R": round(r_, 4), "F1": round(f1_, 4),
                })

            # counting metric
            count_err = abs(n_pred - n_gt)
            count_rows.append({
                "model_key": model_key, "folder": folder, "sheet": sheet,
                "n_gt": n_gt, "n_pred": n_pred,
                "count_err": count_err,
                "count_pct_err": round(count_err / max(n_gt, 1), 4),
                "over_under": ("over" if n_pred > n_gt else
                               ("under" if n_pred < n_gt else "exact")),
            })

            # center-distance metric
            dtp, dfp, dfn, dlist = _match_distance(preds, gts, dist_thresh)
            dp_, dr_, df1_ = _prf1(dtp, dfp, dfn)
            dist_rows.append({
                "model_key": model_key, "folder": folder, "sheet": sheet,
                "n_gt": n_gt, "n_pred": n_pred,
                "TP": dtp, "FP": dfp, "FN": dfn,
                "P": round(dp_, 4), "R": round(dr_, 4), "F1": round(df1_, 4),
                "median_dist_px": round(
                    sorted(dlist)[len(dlist) // 2] if dlist else float("nan"), 1),
            })

    iou_df = pd.DataFrame(iou_rows)
    count_df = pd.DataFrame(count_rows)
    dist_df = pd.DataFrame(dist_rows)

    # ----------------------------------------------- aggregations across split

    def _agg(group):
        tp = int(group["TP"].sum())
        fp = int(group["FP"].sum())
        fn = int(group["FN"].sum())
        p_, r_, f1_ = _prf1(tp, fp, fn)
        return pd.Series({"TP": tp, "FP": fp, "FN": fn,
                          "P": round(p_, 4), "R": round(r_, 4), "F1": round(f1_, 4)})

    iou_overall = (iou_df.groupby(["model_key", "iou"], as_index=False)
                   .apply(_agg, include_groups=False).reset_index(drop=True))
    dist_overall = (dist_df.groupby(["model_key"], as_index=False)
                    .apply(_agg, include_groups=False).reset_index(drop=True))

    count_overall = count_df.groupby("model_key", as_index=False).agg(
        n_gt=("n_gt", "sum"),
        n_pred=("n_pred", "sum"),
        sum_abs_err=("count_err", "sum"),
        mean_pct_err=("count_pct_err", "mean"),
    )
    count_overall["mean_pct_err"] = count_overall["mean_pct_err"].round(4)
    count_overall["overall_pct_err"] = (
        count_overall["sum_abs_err"] / count_overall["n_gt"].clip(lower=1)
    ).round(4)

    # --------------------------------------------------------------- output

    print("=== F1 by IoU threshold (across split) ===")
    pivot = iou_overall.pivot(index="model_key", columns="iou", values="F1")
    print(pivot.round(3).to_string())

    print("\n=== Counting accuracy (across split) ===")
    print(count_overall.to_string(index=False))

    print(f"\n=== Center-distance F1 (≤{dist_thresh:.0f}px, across split) ===")
    print(dist_overall.to_string(index=False))

    out_dir = pred_path.parent
    stem = pred_path.stem.replace("_predictions", "")
    iou_out = out_dir / f"{stem}_reanalysis_iou.csv"
    count_out = out_dir / f"{stem}_reanalysis_counting.csv"
    dist_out = out_dir / f"{stem}_reanalysis_distance.csv"
    iou_overall.to_csv(iou_out, index=False)
    count_df.to_csv(count_out, index=False)
    dist_df.to_csv(dist_out, index=False)
    print(f"\nWrote:\n  {iou_out}\n  {count_out}\n  {dist_out}")


# -------------------------------------------------------------------- cli

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-threshold / counting / "
                                            "distance re-analysis of swing preds.")
    p.add_argument("--pred", type=Path, required=True,
                   help="Path to a *_predictions.csv from swing_eval.py "
                        "or swing_tile_eval.py.")
    p.add_argument("--gt", type=Path, default=GT_CSV_DEFAULT,
                   help="GT CSV path (default: data/gt/gt_bboxes.csv).")
    p.add_argument("--iou", type=float, nargs="+", default=DEFAULT_IOUS,
                   help="IoU thresholds to sweep (default: 0.1 0.2 0.3 0.5).")
    p.add_argument("--dist-factor", type=float, default=1.0,
                   help="center-distance threshold as multiple of mean GT edge "
                        "(default: 1.0 ≈ one door-width).")
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    main(pred_path=a.pred, gt_path=a.gt,
         iou_list=a.iou, dist_thresh_factor=a.dist_factor)
