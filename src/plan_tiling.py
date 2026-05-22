"""Tile-based plan analysis for VLM door detection.

The whole-page approach in Tab 2's "Compare with multimodal models" expander
fails on dense floor-plan sheets like A2.1: the page is rendered at 300 DPI
(~9000×6800 px), then ``_plan_image_bytes_for_vlm`` downsamples to max_dim
2048 to fit inside VLM input limits. After that downsample, individual door
tags (originally ~30×20 px) become 6×4 px — too small for any VLM to read.
Models then hallucinate plausible-looking marks at made-up positions.

This module supports a smarter pipeline:

1. **Region crop.** A2.1 contains TWO floor plans on one page (Second + Ground)
   plus a title block and wall-type legend. ``PLAN_REGIONS`` holds normalized
   crops per sheet so we discard the non-plan strip before doing anything else.

2. **Grid tiling.** Each plan region is split into an N×M grid of overlapping
   tiles. At 300 DPI a tile is ~2000×2000 px, which fits in VLM context
   without resizing — the model sees tags at native resolution.

3. **Per-tile detection.** Each tile is sent to each VLM with a stricter
   prompt (see ``model_comparison.build_door_plan_tile_prompt``) that
   explicitly tells the model NOT to confuse room numbers / dimension
   callouts / keynote bubbles with door tags.

4. **Merge.** Detection bboxes returned in tile-relative coords are shifted
   back to plan-region coords (and on to full-page coords if needed). Per-mark
   NMS deduplicates the same door reported on adjacent tiles' overlap zones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from PIL import Image


# --------------------------------------------------------------- plan regions

# Per-sheet plan-region crops in normalized (0..1) page coordinates.
# Empirically chosen for the 2067 University Avenue sheet layout where a
# single A2.x page carries TWO floor plans plus a right-edge title block /
# wall-type legend. Bounds are intentionally a bit generous so we never clip
# a door swing; the title-block strip on the right is what we want to drop.
PLAN_REGIONS: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = {
    "A2.1": [
        ("Second Floor", (0.04, 0.04, 0.48, 0.97)),
        ("Ground Floor", (0.48, 0.04, 0.83, 0.97)),
    ],
    # A7.1.1 = "Signage – Roof Deck Plan". Single sparse plan, portrait
    # orientation, drawing occupies the upper ~92% of the page. The bottom
    # ~8% carries a small title block we want to drop. Generous side
    # margins so we don't clip the planter band at the top of the roof.
    "A7.1.1": [
        ("Roof Deck", (0.02, 0.02, 0.98, 0.92)),
    ],
    # A2.2, A2.3, A2.4 share the same template; they can be added when we're
    # confident the A2.1 pipeline is producing sensible detections.
}


# Per-sheet (n_cols, n_rows) tile grid. Dense plans (A2.x) get a 2×3 grid so
# individual hexagon tags survive at native DPI. Sparse plans (A7.1.1, only
# ~6 signage tags total) work fine as a single tile — fewer API calls, no
# overlap-merging noise.
TILE_CONFIG: dict[str, tuple[int, int]] = {
    "A2.1": (2, 3),
    "A7.1.1": (1, 1),
}


def tile_config_for_sheet(sheet_no: str) -> tuple[int, int]:
    """Return the ``(n_cols, n_rows)`` tile grid for ``sheet_no``."""
    return TILE_CONFIG.get(sheet_no, (2, 3))


def regions_for_sheet(sheet_no: str) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Return ``[(label, (l, t, r, b))]`` for ``sheet_no`` (empty if unknown)."""
    return list(PLAN_REGIONS.get(sheet_no, []))


def crop_region(
    img: Image.Image, region: tuple[float, float, float, float]
) -> Image.Image:
    """Crop a normalized (l, t, r, b) ∈ [0, 1] region from a full-page image."""
    w, h = img.size
    l, t, r, b = region
    box = (
        max(0, int(round(l * w))),
        max(0, int(round(t * h))),
        min(w, int(round(r * w))),
        min(h, int(round(b * h))),
    )
    return img.crop(box)


# --------------------------------------------------------------- tile grid

@dataclass(frozen=True)
class Tile:
    """One tile cropped from a plan region.

    ``offset_x`` / ``offset_y`` are the tile's top-left position inside the
    PARENT region (not the full page). Detections returned by a VLM in
    tile-pixel coordinates are shifted by these offsets to land in
    region-pixel coordinates.
    """

    image: Image.Image
    offset_x: int
    offset_y: int
    width: int
    height: int
    col: int
    row: int

    @property
    def label(self) -> str:
        return f"r{self.row}c{self.col}"


def tile_image(
    img: Image.Image,
    n_cols: int = 2,
    n_rows: int = 3,
    overlap: float = 0.15,
) -> list[Tile]:
    """Split ``img`` into an ``n_cols × n_rows`` grid of overlapping tiles.

    Each tile is sized so that it overlaps its neighbours by roughly
    ``overlap`` of a tile width/height. The last column and last row are
    bottom/right-aligned so they don't clip the image edge.
    """
    if n_cols < 1 or n_rows < 1:
        raise ValueError("n_cols and n_rows must be >= 1")
    w, h = img.size

    # Tile size = nominal stride * (1 + overlap).
    tile_w = int(round(w / n_cols * (1 + overlap)))
    tile_h = int(round(h / n_rows * (1 + overlap)))
    tile_w = min(tile_w, w)
    tile_h = min(tile_h, h)

    # Step between tile origins so the last tile lands flush with the edge.
    step_x = (w - tile_w) // (n_cols - 1) if n_cols > 1 else 0
    step_y = (h - tile_h) // (n_rows - 1) if n_rows > 1 else 0

    tiles: list[Tile] = []
    for r in range(n_rows):
        for c in range(n_cols):
            if c == n_cols - 1:
                x = max(0, w - tile_w)
            else:
                x = c * step_x
            if r == n_rows - 1:
                y = max(0, h - tile_h)
            else:
                y = r * step_y
            x2 = min(w, x + tile_w)
            y2 = min(h, y + tile_h)
            sub = img.crop((x, y, x2, y2))
            tiles.append(
                Tile(
                    image=sub,
                    offset_x=x,
                    offset_y=y,
                    width=x2 - x,
                    height=y2 - y,
                    col=c,
                    row=r,
                )
            )
    return tiles


# --------------------------------------------------------------- coord shift

def shift_pixel_detections(
    pixel_df: pd.DataFrame,
    offset_x: int,
    offset_y: int,
) -> pd.DataFrame:
    """Translate ``bbox_left``/``bbox_top`` by an offset (other cols untouched)."""
    if pixel_df is None or pixel_df.empty:
        return pixel_df
    out = pixel_df.copy()
    if "bbox_left" in out.columns:
        out["bbox_left"] = out["bbox_left"].astype(int) + int(offset_x)
    if "bbox_top" in out.columns:
        out["bbox_top"] = out["bbox_top"].astype(int) + int(offset_y)
    return out


# --------------------------------------------------------------- NMS

def _iou(a: dict, b: dict) -> float:
    """Intersection over union of two pixel-space bboxes (dict form)."""
    ax1 = int(a.get("bbox_left", 0))
    ay1 = int(a.get("bbox_top", 0))
    ax2 = ax1 + int(a.get("bbox_width", 0))
    ay2 = ay1 + int(a.get("bbox_height", 0))
    bx1 = int(b.get("bbox_left", 0))
    by1 = int(b.get("bbox_top", 0))
    bx2 = bx1 + int(b.get("bbox_width", 0))
    by2 = by1 + int(b.get("bbox_height", 0))
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    return (inter / union) if union > 0 else 0.0


def nms_per_mark(
    detections: pd.DataFrame, iou_threshold: float = 0.3
) -> pd.DataFrame:
    """Per-mark non-max suppression.

    Tiles overlap, so the same door can appear in two adjacent tiles. We
    suppress duplicates ONLY within the same mark (a real door labelled '01'
    can sit close to a different door labelled '02', and both should be kept).
    """
    if detections is None or detections.empty:
        return detections

    kept: list[dict] = []
    for _, group in detections.groupby("mark", sort=False):
        items = sorted(
            group.to_dict("records"),
            key=lambda x: -float(x.get("confidence", 0)),
        )
        suppressed = [False] * len(items)
        for i, det in enumerate(items):
            if suppressed[i]:
                continue
            kept.append(det)
            for j in range(i + 1, len(items)):
                if suppressed[j]:
                    continue
                if _iou(det, items[j]) > iou_threshold:
                    suppressed[j] = True

    return (
        pd.DataFrame(kept, columns=detections.columns)
        .reset_index(drop=True)
    )


def filter_by_vocab(
    detections: pd.DataFrame, vocab: Iterable[str]
) -> pd.DataFrame:
    """Drop detections whose mark isn't in the (case-insensitive) vocab set."""
    if detections is None or detections.empty:
        return detections
    norm_vocab = {str(v).strip().upper() for v in vocab if v}
    if not norm_vocab:
        return detections
    mask = (
        detections["mark"]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(norm_vocab)
    )
    return detections.loc[mask].reset_index(drop=True)


def filter_by_mark_pattern(
    detections: pd.DataFrame, pattern: str
) -> pd.DataFrame:
    """Drop detections whose mark doesn't match the given regex.

    Used when the schedule vocab doesn't apply (e.g. A7.1.1 carries
    SIGNAGE codes like ``1.11`` / ``7.15`` rather than door schedule
    marks). Pattern is anchored implicitly by ``re.fullmatch``.
    """
    import re as _re
    if detections is None or detections.empty:
        return detections
    rx = _re.compile(pattern)
    mask = (
        detections["mark"]
        .astype(str)
        .str.strip()
        .apply(lambda s: bool(rx.fullmatch(s)))
    )
    return detections.loc[mask].reset_index(drop=True)
