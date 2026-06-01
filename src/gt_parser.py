"""Ground-truth parser: VOC XML labels → unified CSVs.

Reads PASCAL VOC XML annotations from ``~/Desktop/a7_1_1_gt/label data/`` —
one folder per labelled sheet, each containing a ``*.png`` and a ``*.xml`` —
and produces two CSVs under ``data/gt/``:

* ``gt_bboxes.csv``  — one row per bbox (class normalised to lowercase)
* ``gt_summary.csv`` — one row per (split, sheet, floor, folder)

Floor mapping and val/test split are hardcoded based on the user's confirmation:

  Folder            Sheet     Floor             Split
  -------------     -------   ---------------   -----
  A2.1-1            A2.1      Ground            val
  A2.1-2            A2.1      Floor 2           val
  A2.2-1            A2.2      Floor 3           val
  A2.2-2            A2.2      Floor 4           test
  A2.3-1            A2.3      Floor 5           val
  A2.3-2            A2.3      Floor 6           test
  A2.4              A2.4      Floor 7           test
  A7.1.1 Ground     A7.1.1    Ground (signage)  val
  A7.1.1 Typical    A7.1.1    Typical (×6)      val
  A7.1.1 RoofDeck   A7.1.1    RoofDeck          test

The A2.x folders are the actual floor plans used to derive the building
swing-door total. A7.1.1 is the signage-plan view; ``Typical`` represents
floors 2–7 and would need to be multiplied by 6 if used for a building total.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from PIL import Image


# ----------------------------------------------------------------- config

LABEL_ROOT = Path("/Users/kinelaine/Desktop/a7_1_1_gt/label data")
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "gt"

FLOOR_MAP: dict[str, tuple[str, str]] = {
    "A2.1-1":          ("A2.1", "Ground"),
    "A2.1-2":          ("A2.1", "Floor 2"),
    "A2.2-1":          ("A2.2", "Floor 3"),
    "A2.2-2":          ("A2.2", "Floor 4"),
    "A2.3-1":          ("A2.3", "Floor 5"),
    "A2.3-2":          ("A2.3", "Floor 6"),
    "A2.4":            ("A2.4", "Floor 7"),
    "A7.1.1 Ground":   ("A7.1.1", "Ground (signage)"),
    "A7.1.1 Typical":  ("A7.1.1", "Typical (x6)"),
    "A7.1.1 RoofDeck": ("A7.1.1", "RoofDeck"),
}

VAL_FOLDERS = {
    "A2.1-1", "A2.1-2", "A2.2-1", "A2.3-1",
    "A7.1.1 Ground", "A7.1.1 Typical",
}
TEST_FOLDERS = {
    "A2.2-2", "A2.3-2", "A2.4", "A7.1.1 RoofDeck",
}


# ---------------------------------------------------------------- helpers

def _normalize_class(name: str) -> str:
    """Lowercase + strip so 'Door' and 'door' canonicalise to 'door'."""
    return (name or "").strip().lower()


def _parse_xml(xml_path: Path) -> tuple[int, int, list[dict]]:
    """Return ``(img_w, img_h, [bbox_dict, ...])`` from a VOC XML."""
    root = ET.parse(xml_path).getroot()
    sz = root.find("size")
    img_w = int(sz.find("width").text)
    img_h = int(sz.find("height").text)
    boxes: list[dict] = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        boxes.append({
            "class": _normalize_class(obj.findtext("name", "")),
            "x1": int(b.find("xmin").text),
            "y1": int(b.find("ymin").text),
            "x2": int(b.find("xmax").text),
            "y2": int(b.find("ymax").text),
        })
    return img_w, img_h, boxes


def _split_for(folder_name: str) -> str:
    if folder_name in VAL_FOLDERS:
        return "val"
    if folder_name in TEST_FOLDERS:
        return "test"
    return "unassigned"


# ------------------------------------------------------------------ main

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for folder in sorted(LABEL_ROOT.iterdir()):
        if not folder.is_dir() or folder.name not in FLOOR_MAP:
            if folder.is_dir():
                print(f"skip (unmapped folder): {folder.name}")
            continue

        sheet, floor = FLOOR_MAP[folder.name]
        split = _split_for(folder.name)

        pngs = list(folder.glob("*.png"))
        xmls = list(folder.glob("*.xml"))
        if not pngs or not xmls:
            print(f"skip (missing png or xml): {folder.name}")
            continue
        png_path, xml_path = pngs[0], xmls[0]

        img_w, img_h, boxes = _parse_xml(xml_path)
        with Image.open(png_path) as im:
            actual_w, actual_h = im.size
        if (actual_w, actual_h) != (img_w, img_h):
            print(
                f"WARN dimension mismatch {folder.name}: "
                f"png={actual_w}x{actual_h}, xml={img_w}x{img_h}"
            )

        for b in boxes:
            rows.append({
                "folder": folder.name,
                "sheet": sheet,
                "floor": floor,
                "split": split,
                "image_path": str(png_path),
                "image_w": img_w,
                "image_h": img_h,
                "class": b["class"],
                "x1": b["x1"], "y1": b["y1"],
                "x2": b["x2"], "y2": b["y2"],
                "bbox_w": b["x2"] - b["x1"],
                "bbox_h": b["y2"] - b["y1"],
            })

    df = pd.DataFrame(rows)
    bboxes_csv = OUT_DIR / "gt_bboxes.csv"
    df.to_csv(bboxes_csv, index=False)
    print(f"\nwrote {len(df)} bbox rows -> {bboxes_csv}")

    summary = (
        df.groupby(["split", "sheet", "floor", "folder"], as_index=False)
          .agg(
              bbox_count=("class", "count"),
              image_w=("image_w", "first"),
              image_h=("image_h", "first"),
          )
          .sort_values(["split", "folder"])
          .reset_index(drop=True)
    )
    summary_csv = OUT_DIR / "gt_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"wrote {len(summary)} summary rows -> {summary_csv}\n")

    # Pretty-print
    print("=== Per-folder summary ===")
    print(summary.to_string(index=False))

    a2_total = df[df["sheet"].str.startswith("A2.")]["class"].count()
    a7_total = df[df["sheet"] == "A7.1.1"]["class"].count()

    print("\n=== Building totals ===")
    print(f"  From A2.x (actual floor plans):   {a2_total} swing doors")
    print(f"  From A7.1.1 (raw, single layer):  {a7_total}")
    print(
        "  (A7.1.1 Typical represents floors 2-7; multiply by 6 if used for "
        "building total)"
    )

    print("\n=== Split totals ===")
    print(df.groupby("split")["class"].count().to_string())


if __name__ == "__main__":
    main()
