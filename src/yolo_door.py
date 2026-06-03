"""Method 4b — YOLO fine-tuning on the ByteTrooper floor-plan door dataset.

This module is self-contained (does NOT touch any existing pipeline code) and
implements the three steps the advisor recommended:

  1. download   — pull the Roboflow Universe dataset to ``data/yolo_doors/``
  2. train      — fine-tune a small YOLO checkpoint on it (Apple-Silicon MPS)
  3. predict    — run the fine-tuned model on the project's VAL/TEST sheets,
                  convert outputs to the same predictions.csv schema as the
                  4 VLMs use, then re-use ``swing_reanalyze.py`` to compare.

Usage::

    # one-time download from Roboflow (needs ROBOFLOW_API_KEY in .env)
    .venv/bin/python -m src.yolo_door download

    # fine-tune
    .venv/bin/python -m src.yolo_door train --epochs 50 --imgsz 640

    # predict on VAL split (writes outputs/eval/yolo_swing_val_predictions.csv)
    .venv/bin/python -m src.yolo_door predict --split val

The dataset details come from the user's Roboflow workspace; override at the CLI
if you want a different fork::

    --workspace yins-workspace-egmni  --project doors-detection-in-floor-plans-h8ma9 --version 1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "yolo_doors"
GT_CSV = ROOT / "data" / "gt" / "gt_bboxes.csv"
EVAL_DIR = ROOT / "outputs" / "eval"
WEIGHTS_DIR = ROOT / "outputs" / "yolo_weights"

# User's Roboflow workspace (per chat 2026-06-01).
DEFAULT_WORKSPACE = "yins-workspace-egmni"
DEFAULT_PROJECT = "doors-detection-in-floor-plans-h8ma9"


# ----------------------------------------------------------------------------
#  download
# ----------------------------------------------------------------------------

def download(workspace: str, project: str, version: int, fmt: str = "yolov8") -> Path:
    load_dotenv(override=False)
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "ERROR: ROBOFLOW_API_KEY is not set in .env.\n"
            "Get a free key at https://app.roboflow.com/settings/api"
        )

    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("Run: .venv/bin/pip install roboflow")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    versions = proj.versions()
    if not versions:
        raise SystemExit(f"No versions found for {workspace}/{project}")
    if version is None or version == 0:
        # use the latest version
        v_obj = versions[-1]
        print(f"Using latest version: {v_obj.version}")
    else:
        v_obj = proj.version(version)

    print(f"Downloading {workspace}/{project} v{v_obj.version} as {fmt} -> {DATA_DIR}")
    ds = v_obj.download(fmt, location=str(DATA_DIR))
    print(f"OK. Dataset at: {ds.location}")
    print(f"data.yaml at: {Path(ds.location) / 'data.yaml'}")
    return Path(ds.location)


# ----------------------------------------------------------------------------
#  train
# ----------------------------------------------------------------------------

def train(model: str, data_yaml: Path, epochs: int, imgsz: int, batch: int,
          device: str, project_name: str = "yolo_door") -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Run: .venv/bin/pip install ultralytics")

    if not data_yaml.exists():
        raise SystemExit(
            f"data.yaml not found at {data_yaml}. "
            "Did you run `python -m src.yolo_door download` first?"
        )

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Training {model} for {epochs} epochs on {device} (imgsz={imgsz}, batch={batch})")
    yolo = YOLO(model)
    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(WEIGHTS_DIR),
        name=project_name,
        exist_ok=True,
    )
    best = WEIGHTS_DIR / project_name / "weights" / "best.pt"
    print(f"Done. Best weights: {best}")
    return best


# ----------------------------------------------------------------------------
#  predict (on the project's VAL/TEST sheets, write predictions.csv)
# ----------------------------------------------------------------------------

def predict(weights: Path, split: str, conf: float, imgsz: int) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Run: .venv/bin/pip install ultralytics")
    if not weights.exists():
        raise SystemExit(f"Weights not found at {weights}. Train first.")
    if not GT_CSV.exists():
        raise SystemExit(f"GT CSV not found at {GT_CSV}.")

    gt = pd.read_csv(GT_CSV)
    sheets = gt[gt["split"] == split].drop_duplicates(subset=["folder"])

    yolo = YOLO(str(weights))
    rows: list[dict] = []
    for _, srow in sheets.iterrows():
        folder = srow["folder"]
        img_path = Path(srow["image_path"])
        if not img_path.exists():
            print(f"  WARN: image not found: {img_path} (folder={folder})")
            continue
        print(f"  predict {folder}: {img_path}")
        res = yolo.predict(
            source=str(img_path), conf=conf, imgsz=imgsz, verbose=False,
        )[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
        scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            rows.append({
                "folder": folder,
                "sheet": srow["sheet"],
                "floor": srow["floor"],
                "split": split,
                "image_path": str(img_path),
                "image_w": int(srow["image_w"]),
                "image_h": int(srow["image_h"]),
                "model_key": "yolo",
                "class": "door",
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "score": float(score),
            })

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"yolo_swing_{split}_predictions.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} preds -> {out}")
    return out


# ----------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO fine-tune for swing doors.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="Download Roboflow dataset.")
    d.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    d.add_argument("--project", default=DEFAULT_PROJECT)
    d.add_argument("--version", type=int, default=0,
                   help="0 = latest version (default).")
    d.add_argument("--format", default="yolov8")

    t = sub.add_parser("train", help="Fine-tune a YOLO model.")
    t.add_argument("--model", default="yolov8n.pt",
                   help="Base checkpoint (yolov8n/s/m/l/x.pt).")
    t.add_argument("--data", type=Path,
                   default=DATA_DIR / "data.yaml")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--device", default="mps",
                   help="mps (Apple Silicon), cpu, or 0 for first CUDA gpu.")
    t.add_argument("--project-name", default="yolo_door")

    pr = sub.add_parser("predict", help="Run trained model on VAL/TEST sheets.")
    pr.add_argument("--weights", type=Path,
                    default=WEIGHTS_DIR / "yolo_door" / "weights" / "best.pt")
    pr.add_argument("--split", default="val", choices=["val", "test"])
    pr.add_argument("--conf", type=float, default=0.25)
    pr.add_argument("--imgsz", type=int, default=640)

    return p.parse_args()


def main() -> None:
    a = _args()
    if a.cmd == "download":
        download(a.workspace, a.project, a.version, a.format)
    elif a.cmd == "train":
        train(a.model, a.data, a.epochs, a.imgsz, a.batch,
              a.device, a.project_name)
    elif a.cmd == "predict":
        predict(a.weights, a.split, a.conf, a.imgsz)
    else:
        sys.exit(f"unknown subcommand: {a.cmd}")


if __name__ == "__main__":
    main()
