"""Tile-based swing-door eval with UPSCALE + FEW-SHOT (Steps 1+3 of 1+2+3).

Stacks two phase-4 follow-up improvements on top of ``swing_tile_eval.py``:

1. **Upscale (Step 1)**: each input image is upsampled by ``--upscale`` (default 2)
   via PIL bicubic before tiling. Tiles are now 2× their normal size, giving
   the VLM 2× the pixels per door symbol. Tile-relative bboxes are converted
   back to ORIGINAL pixel coords by dividing by the upscale factor, so the
   resulting predictions stay in the SAME coordinate frame as GT.

2. **Few-shot (Step 3)**: each per-tile call uses ``run_plan_swing_v3_model``
   (from ``swing_v3``), which prepends a synthetic example PNG showing three
   swing doors with their CORRECT arc-only red bboxes. Cache namespace is
   ``swing-v3-fs`` so it does not collide with swing-v2.

A separate ``--postprocess`` flag, when given, also applies the
content-based L→square shrinker from ``swing_postprocess.py`` after NMS
(this is Step 2 — the third leg of the 1+2+3 combo).

Usage::

    .venv/bin/python -m src.swing_tile_eval_v3                            # val, upscale=2, no postprocess
    .venv/bin/python -m src.swing_tile_eval_v3 --postprocess               # 1+2+3 combined
    .venv/bin/python -m src.swing_tile_eval_v3 --upscale 1 --postprocess   # 2+3 only (no upscale)
    .venv/bin/python -m src.swing_tile_eval_v3 --models anthropic          # only anthropic
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from PIL import Image

from src.swing_tile_eval import (
    CONF_THRESHOLD, IOU_THRESHOLD, MIN_TILES_PER_DIM,
    NMS_IOU, OVERLAP, OUT_DIR, ROOT, TILE_TARGET,
    _greedy_match, _grid_for_image, _nms, _prf1,
)
from src.swing_v3 import (
    DOOR_PLAN_SWING_V3_PROMPT_VERSION,
    MODELS_V3,
    run_plan_swing_v3_model,
)
from src.swing_postprocess import _shrink_to_arc


GT_CSV = ROOT / "data" / "gt" / "gt_bboxes.csv"


# ---------------------------------------------------------------- upscale tile slicing

def _upscale_image(img: Image.Image, factor: float) -> Image.Image:
    if factor == 1.0:
        return img
    nw = int(round(img.size[0] * factor))
    nh = int(round(img.size[1] * factor))
    return img.resize((nw, nh), Image.BICUBIC)


def _slice_pil(img: Image.Image, tiles) -> list[bytes]:
    out = []
    for x, y, w, h in tiles:
        crop = img.crop((x, y, x + w, y + h))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


# ------------------------------------------------------------------- main

def main(split: str = "val", iou_thresh: float = IOU_THRESHOLD,
         conf_thresh: float = CONF_THRESHOLD, tile_target: int = TILE_TARGET,
         overlap: float = OVERLAP, nms_iou: float = NMS_IOU,
         upscale: float = 2.0, postprocess: bool = False,
         models: list[str] | None = None) -> None:
    load_dotenv(ROOT / ".env", override=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt = pd.read_csv(GT_CSV)
    split_gt = gt[gt["split"] == split].copy()
    if split_gt.empty:
        raise SystemExit(f"No GT rows for split={split!r}")

    sel_models = models or list(MODELS_V3.keys())
    for m in sel_models:
        if m not in MODELS_V3:
            raise SystemExit(f"Unknown model {m!r}; available: {list(MODELS_V3.keys())}")

    print(
        f"\n{split.upper()} TILE-V3 eval: {split_gt['folder'].nunique()} images, "
        f"{len(split_gt)} GT — IoU≥{iou_thresh}, tile_target={tile_target}px, "
        f"overlap={overlap:.0%}, NMS={nms_iou}, upscale={upscale}x, "
        f"postprocess={postprocess}, models={sel_models}\n"
    )

    pred_rows: list[dict] = []
    match_rows: list[dict] = []

    for folder, folder_gt in split_gt.groupby("folder"):
        image_path = Path(folder_gt["image_path"].iloc[0])
        sheet = folder_gt["sheet"].iloc[0]
        floor = folder_gt["floor"].iloc[0]
        img_w_orig = int(folder_gt["image_w"].iloc[0])
        img_h_orig = int(folder_gt["image_h"].iloc[0])

        if not image_path.exists():
            print(f"SKIP {folder}: image not found")
            continue

        # Load + upscale.
        orig_img = Image.open(image_path).convert("RGB")
        up_img = _upscale_image(orig_img, upscale)
        up_w, up_h = up_img.size

        # Tile in upscaled space.
        tiles_up = _grid_for_image(up_w, up_h, tile_target, overlap)
        tile_bytes_list = _slice_pil(up_img, tiles_up)

        gts_list = [
            {"idx": int(i), "x1": float(r["x1"]), "y1": float(r["y1"]),
             "x2": float(r["x2"]), "y2": float(r["y2"])}
            for i, r in folder_gt.reset_index(drop=True).iterrows()
        ]

        print(f"=== {folder} ({sheet}/{floor}) — orig {img_w_orig}x{img_h_orig}, "
              f"up {up_w}x{up_h}, {len(tiles_up)} tiles, {len(gts_list)} GT ===")

        for model_key in sel_models:
            all_preds: list[dict] = []
            cache_hits = 0
            api_calls = 0
            total_t = 0.0

            for ti, ((x0, y0, tw, th), tb) in enumerate(zip(tiles_up, tile_bytes_list)):
                run = run_plan_swing_v3_model(model_key, tb, use_cache=True)
                if run.cached:
                    cache_hits += 1
                else:
                    api_calls += 1
                    total_t += run.elapsed_s
                if run.error:
                    if api_calls <= 1:  # only print first error per model
                        print(f"    err [{model_key} tile {ti}]: {run.error}")
                    continue
                for j, r in run.df.reset_index(drop=True).iterrows():
                    try:
                        conf = float(r.get("confidence", 0.5))
                    except (TypeError, ValueError):
                        conf = 0.5
                    if conf < conf_thresh:
                        continue
                    # Tile-relative [0,1] -> upscaled-pixel coords.
                    fx1_up = x0 + float(r["bbox_x1"]) * tw
                    fy1_up = y0 + float(r["bbox_y1"]) * th
                    fx2_up = x0 + float(r["bbox_x2"]) * tw
                    fy2_up = y0 + float(r["bbox_y2"]) * th
                    # Upscaled-pixel -> original-pixel (divide by upscale factor).
                    fx1 = fx1_up / upscale
                    fy1 = fy1_up / upscale
                    fx2 = fx2_up / upscale
                    fy2 = fy2_up / upscale
                    if fx2 <= fx1 or fy2 <= fy1:
                        continue
                    all_preds.append({
                        "tile_idx": ti,
                        "x1": fx1, "y1": fy1, "x2": fx2, "y2": fy2,
                        "confidence": conf,
                    })

            merged = _nms(all_preds, iou_thresh=nms_iou)

            if postprocess:
                # Apply content-based L→square shrink in original coords.
                shrunk = []
                for p in merged:
                    nx1, ny1, nx2, ny2 = _shrink_to_arc(
                        orig_img, p["x1"], p["y1"], p["x2"], p["y2"]
                    )
                    p2 = dict(p)
                    p2["x1"], p2["y1"], p2["x2"], p2["y2"] = nx1, ny1, nx2, ny2
                    shrunk.append(p2)
                merged = shrunk

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
                f"  {model_key:10s} [{tag:>6}] {cache_hits}/{len(tiles_up)} cache  "
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

    ver = DOOR_PLAN_SWING_V3_PROMPT_VERSION
    suffix = f"{ver}_t{tile_target}_ov{int(overlap*100)}_up{int(upscale)}"
    if postprocess:
        suffix += "_post"
    pred_csv = OUT_DIR / f"swingtilev3_{suffix}_{split}_predictions.csv"
    match_csv = OUT_DIR / f"swingtilev3_{suffix}_{split}_matches.csv"
    sheet_csv = OUT_DIR / f"swingtilev3_{suffix}_{split}_metrics_by_sheet.csv"
    overall_csv = OUT_DIR / f"swingtilev3_{suffix}_{split}_metrics_overall.csv"

    pred_df.to_csv(pred_csv, index=False)
    match_df.to_csv(match_csv, index=False)
    by_sheet.to_csv(sheet_csv, index=False)
    overall.to_csv(overall_csv, index=False)

    print("\n=== Metrics by (model, sheet) ===")
    print(by_sheet.to_string(index=False))
    print("\n=== Overall metrics (across split) ===")
    print(overall.to_string(index=False))
    print(f"\nWrote:\n  {pred_csv}\n  {match_csv}\n  {sheet_csv}\n  {overall_csv}")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tile + upscale + few-shot + (opt) postprocess swing-door eval.")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    p.add_argument("--tile", type=int, default=TILE_TARGET)
    p.add_argument("--overlap", type=float, default=OVERLAP)
    p.add_argument("--nms", type=float, default=NMS_IOU)
    p.add_argument("--upscale", type=float, default=2.0,
                   help="Image upscale factor before tiling (default 2.0).")
    p.add_argument("--postprocess", action="store_true",
                   help="Also apply content-based L→square bbox shrink (Step 2).")
    p.add_argument("--models", nargs="+", default=None,
                   help="Subset of model keys (default: all).")
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    main(split=a.split, iou_thresh=a.iou, conf_thresh=a.conf,
         tile_target=a.tile, overlap=a.overlap, nms_iou=a.nms,
         upscale=a.upscale, postprocess=a.postprocess, models=a.models)
