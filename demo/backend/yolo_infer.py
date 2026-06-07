"""YOLO door detection for the demo.

Loads the fine-tuned checkpoint once (outputs/yolo_weights/yolo_door/weights/best.pt)
and exposes detect() returning door bounding boxes in pixel coords.

Construction sheets render very large (e.g. 7200x4800). The model was trained on
~640px single floor-plan tiles, so feeding a whole sheet at imgsz=640 shrinks every
door to a few pixels and recall collapses. We therefore run *sliced / tiled*
inference (SAHI-style): cut the sheet into overlapping tiles near the training
scale, detect on each, map boxes back to full-image coords, then NMS-merge.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "outputs" / "yolo_weights" / "yolo_door" / "weights" / "best.pt"

# Class 0 == door in data.yaml (names: ['0', 'stairs']). We only surface doors.
DOOR_CLASS = 0

# A whole sheet larger than this on its long side gets tiled.
TILE_TRIGGER = 1200
# Tile window size in source pixels (~2x the 640 train size keeps doors crisp).
TILE = 1280
# Fractional overlap between adjacent tiles so doors on a seam aren't lost.
OVERLAP = 0.25

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from ultralytics import YOLO
                if not WEIGHTS.exists():
                    raise FileNotFoundError(f"YOLO weights not found at {WEIGHTS}")
                _model = YOLO(str(WEIGHTS))
    return _model


def _predict_array(arr, conf: float, imgsz: int) -> list[list[float]]:
    """Run the model on a numpy RGB array. Returns [[x1,y1,x2,y2,score], ...] for doors."""
    model = _get_model()
    res = model.predict(source=arr, conf=conf, imgsz=imgsz, verbose=False)[0]
    if res.boxes is None:
        return []
    xyxy = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    classes = res.boxes.cls.cpu().numpy().astype(int)
    out = []
    for (x1, y1, x2, y2), score, cls in zip(xyxy, scores, classes):
        if int(cls) != DOOR_CLASS:
            continue
        out.append([float(x1), float(y1), float(x2), float(y2), float(score)])
    return out


def _nms(boxes: list[list[float]], iou_thr: float = 0.45) -> list[list[float]]:
    """Greedy NMS on [x1,y1,x2,y2,score] rows."""
    if not boxes:
        return []
    b = np.array(boxes, dtype=float)
    x1, y1, x2, y2, sc = b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = sc.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return [boxes[i] for i in keep]


def detect(image_path: str, conf: float = 0.25, imgsz: int = 640,
           region: list[float] | None = None) -> list[dict]:
    """Run YOLO on a page image. Returns a list of door boxes.

    Each box: {x1, y1, x2, y2, score} in pixel coordinates of the full image.
    Large sheets are processed with overlapping tiled inference + NMS.
    If `region`=[x1,y1,x2,y2] is given, only that crop is scanned (and boxes
    are mapped back to full-image coords) — lets the user isolate one drawing.
    """
    img = Image.open(image_path).convert("RGB")

    ox0 = oy0 = 0
    if region and len(region) == 4:
        fx1, fy1, fx2, fy2 = region
        x1, y1 = max(0, int(min(fx1, fx2))), max(0, int(min(fy1, fy2)))
        x2, y2 = int(max(fx1, fx2)), int(max(fy1, fy2))
        if x2 - x1 > 8 and y2 - y1 > 8:
            img = img.crop((x1, y1, x2, y2))
            ox0, oy0 = x1, y1

    W, H = img.size

    # Small images: single pass, matches training scale.
    if max(W, H) <= TILE_TRIGGER:
        boxes = _predict_array(np.asarray(img), conf, imgsz)
    else:
        step = int(TILE * (1 - OVERLAP))
        xs = list(range(0, max(1, W - TILE) + 1, step)) or [0]
        ys = list(range(0, max(1, H - TILE) + 1, step)) or [0]
        if xs[-1] + TILE < W:
            xs.append(W - TILE)
        if ys[-1] + TILE < H:
            ys.append(H - TILE)
        arr = np.asarray(img)
        boxes = []
        for oy in ys:
            for ox in xs:
                tile = arr[oy:oy + TILE, ox:ox + TILE]
                for x1, y1, x2, y2, s in _predict_array(tile, conf, imgsz=TILE):
                    boxes.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy, s])
        boxes = _nms(boxes, iou_thr=0.45)

    return [
        {"x1": b[0] + ox0, "y1": b[1] + oy0, "x2": b[2] + ox0, "y2": b[3] + oy0, "score": b[4]}
        for b in boxes
    ]
