# Phase 4 — Error Analysis & Findings

> Door-takeoff MVP / CEE329 — VAL split, 6 images, 167 GT bboxes (swing-door arcs)
> Models: `anthropic` (claude-opus-4-8), `openai` (gpt-4.1), `gemini` (gemini-2.5-pro)

## TL;DR

**Frontier VLMs cannot produce IoU-precise bboxes for swing-door arcs out of the box
(F1 ≤ 0.02 at IoU 0.5 on full-image), but they CAN locate doors and approximate the door
count. Tiling lifts claude-opus-4-8 to F1 = 0.077 across the split and F1 = 0.27 on the
one high-resolution sheet (A7.1.1 Typical, 1728×4032).** The gap between IoU-strict
and counting / center-distance metrics localizes the failure to *bbox shape and image
resolution*, not *visual recognition*.

| Metric (VAL, swing-v2)          | anthropic | openai | gemini |
|---------------------------------|-----------|--------|--------|
| F1 @ IoU 0.5 (full-image)       | 0.023     | 0.006  | 0.000  |
| F1 @ IoU 0.5 (**tiled**)        | **0.077** | 0.000  | 0.005  |
| F1 @ IoU 0.1 (tiled)            | **0.340** | 0.085  | 0.160  |
| Center-distance F1 (tiled, ≤42 px) | **0.42** | 0.18 | 0.28   |
| Counting error (full-image)     | 44 %      | 14 %   | **8 %** |

**Headline:** tiling triples anthropic's F1 @ IoU 0.5 (0.023 → 0.077) and doubles its
distance F1 (0.32 → 0.42). Gemini and OpenAI are flat-or-worse under tiling.

## 1. Method recap

* **GT** — VOC XMLs hand-labeled in CVAT, parsed to `data/gt/gt_bboxes.csv`. Convention:
  bbox tightly encloses **only the quarter-circle arc**, not the door leaf.
* **Prompts** — `swing-v2` asks the VLM for normalized bboxes around "ONLY the
  quarter-circle arc, approximately square, ~one door-width". Earlier `swing-v1` asked
  for the L-shape (leaf + arc) and produced taller bboxes.
* **Eval** — greedy IoU matching, conf-desc. Multi-IoU sweep + counting + center-distance
  added in `swing_reanalyze.py` to triangulate the failure mode.

## 2. Headline numbers (VAL)

### Full-image, swing-v1 baseline

| model     | F1 @ 0.1 | F1 @ 0.5 | preds | counting err |
|-----------|----------|----------|-------|--------------|
| anthropic | 0.203    | 0.000    | 109   | 41 %         |
| gemini    | 0.020    | 0.000    | 28    | 84 %         |
| openai    | 0.093    | 0.011    | 198   | 39 %         |

### Full-image, swing-v2 (current best)

| model     | F1 @ 0.1 | F1 @ 0.5 | preds | counting err | center-dist F1 |
|-----------|----------|----------|-------|--------------|----------------|
| anthropic | 0.153    | 0.023    | 94    | 44 %         | **0.32**       |
| gemini    | 0.155    | 0.000    | 169   | **8 %**      | 0.31           |
| openai    | 0.088    | 0.006    | 153   | 14 %         | 0.19           |

### Tile-based, swing-v2 (2×2 / 2×6 grid, 700 px target, 15 % overlap)

| model     | F1 @ 0.1 | F1 @ 0.5 | preds | counting err | center-dist F1 |
|-----------|----------|----------|-------|--------------|----------------|
| anthropic | **0.340** | **0.077** | 92    | 49 %         | **0.42** (P 0.59) |
| gemini    | 0.160    | 0.005    | 196   | 17 %         | 0.28           |
| openai    | 0.085    | 0.000    | 161   | 13 %         | 0.18           |

By sheet (anthropic only — others ~ 0):

| sheet         | image px       | doors px     | tiles | F1 @ 0.5 |
|---------------|----------------|--------------|-------|----------|
| A2.1          | 608–688 × 1138 | 25–27 × 26–27 | 4     | 0.000    |
| A2.2          | 754 × 1286     | 28 × 27      | 4     | 0.000    |
| A2.3          | 718 × 1288     | 30 × 27      | 4     | 0.044    |
| A7.1.1 Ground | 672 × 1304     | 45 × 45      | 4     | 0.048    |
| A7.1.1 Typical| **1728 × 4032** | **107 × 98** | **12**| **0.271** |

The big win is on A7.1.1 Typical — the only sheet where doors are large in absolute pixels.
Tiling preserves their absolute size, full-image VLM input downsizes them past the
detectable threshold.

## 3. What changed v1 → v2

* **Prompt body**: "L-shape (leaf + arc)" → "ONLY the quarter-circle arc, approximately
  square, one door-width".
* **Gemini model**: `gemini-3.1-pro-preview` (returned `{"doors":[]}` 4 / 6 times,
  pixel-coord garbage on the rest) → `gemini-2.5-pro` (well-formed, normalized output).
* **Cache namespace**: `swing-v1` → `swing-v2`, fresh API calls everywhere.

The only model that lost ground was anthropic at IoU 0.1 (0.203 → 0.153), because v2
made it ~14 % more conservative (109 → 94 preds, the only model under-predicting). Net:
+precision, -recall. At IoU 0.5 anthropic gained (0.000 → 0.023) — v2 boxes are the
right *shape* even if not always right *placement*.

## 4. Why F1 @ IoU 0.5 ≈ 0

Visual inspection of `outputs/eval/viz/*.png` (18 PNGs, GT red, pred blue):

1. **Convention mismatch** — even after the v2 prompt, models tend to draw a bbox that
   encloses the door-leaf line *and* the arc (an L-shape with h/w ≈ 1.4–1.9). GT bboxes
   are the arc alone (h/w ≈ 1.0). The two boxes are spatially adjacent (the leaf shares
   one edge with the arc) but only overlap by ~30–40 % of the arc — below IoU 0.5.

2. **Anthropic placement is close, shape is wrong**. 42 of 94 predictions land within one
   door-width of a real GT door (center-distance TP), but only 4 of those clear IoU 0.5.

3. **Gemini noise outside the drawing extents.** ~30 % of gemini's 169 preds land in
   the title block, north arrow, or door-schedule region — visible in the viz PNGs.
   Counting accuracy is great (8 % error) only because over- and under-counts roughly
   cancel across sheets; per-sheet error is higher.

4. **OpenAI is the worst on every metric** despite being middle-of-the-road on counting.
   Bboxes are systematically too wide and shifted up-left of the arc.

## 5. Counting & center-distance — the metrics that actually work

For door takeoff in practice (the contractor needs N doors per sheet, not pixel-perfect
arcs), the more useful metrics are:

* **Counting error (gemini ≈ 8 %)** — viable for first-pass takeoff with human review.
* **Center-distance F1 (anthropic ≈ 0.32, gemini ≈ 0.31)** — viable for "highlight where
  to look" UX, not viable for billable measurements.

Anthropic's center-distance precision (0.45) is the highest of the three: when claude
predicts a door, it's right ~45 % of the time. Gemini predicts more aggressively
(P 0.31, R 0.31).

## 6. What tiling buys us (or doesn't)

Tiling slices each plan into a 2×2 grid (or 2×6 for the 1728×4032 A7.1.1 Typical),
runs the same swing-v2 prompt per tile, and NMS-merges across tile boundaries. Total
96 API calls per VAL run (32 tiles × 3 models).

**Anthropic gains substantially**:
* F1 @ IoU 0.5: 0.023 → **0.077** (+ 235 %)
* F1 @ IoU 0.1: 0.153 → **0.340** (+ 122 %)
* Distance F1:  0.32  → **0.42**  (P jumps to 0.59 — anthropic finds doors and is
  right 59 % of the time when it does)

**Gemini and OpenAI do not gain**, and slightly regress at the strict thresholds.

The anthropic gain is concentrated on the one large image (A7.1.1 Typical, 1728×4032,
107×98 px doors). On the smaller A2.x plans, tiling does not help because the doors
were *already* small in absolute pixels (~27 px) — splitting the image doesn't make
them bigger, and the convention mismatch dominates.

This means: tiling helps when the failure is **resolution** (anthropic on big images),
not when the failure is **convention** (everyone on small images). The two failure modes
co-occur in this dataset, which is why tiling pushes anthropic from "convention-only
failure on A7.1.1" to "real F1=0.27 on A7.1.1".

## 7. Is the GT good enough?

**Adequate for CEE329, weak for an ML paper.**

* 6 VAL + 4 TEST images, 261 bboxes — small but enough to estimate a recall ceiling
  to within ±0.07 (binomial 95 % CI).
* Single annotator, no inter-rater agreement metric. Visual spot-check during this
  phase suggests GT recall on real doors is ~95 % — a few small swings inside
  bathrooms / closets were missed.
* The arc-only convention is consistent across all 167 VAL bboxes (mean h/w = 1.03,
  std = 0.08). That's a tight specification, which is precisely *why* the convention
  mismatch with model output is so visible.

For the class deliverable: the GT supports the negative-finding story. If we wanted
to publish, we'd want ≥ 50 images, two annotators, and a dual-convention GT (arc-only
+ leaf+arc) so we could measure IoU under each.

## 8. Recommendation for the report

Frame as a **negative result with mechanistic explanation, plus a partial positive
finding from tiling**:

1. *Frontier VLMs are not yet ready for pixel-precise takeoff in general* — F1 @ IoU 0.5
   is essentially zero on small floor-plan images across three top labs.
2. *Tiling + claude-opus-4-8 is workable for high-resolution sheets* — F1 0.27 on the
   A7.1.1 Typical sheet (1728×4032, 107×98 px doors). Distance F1 reaches 0.42 across the
   split with 0.59 precision: when anthropic+tile predicts a door, it's right 59 % of the
   time.
3. *Counting and door-localization UX work today* — gemini single-digit % counting error
   on full-image, anthropic+tile single-digit FP-rate on A7.1.1 Typical.
4. *The dominant failure mode is annotation-convention drift, then resolution* —
   evidenced by (a) the 4× drop from F1 @ IoU 0.1 to F1 @ IoU 0.5, (b) the per-sheet
   A2.x vs A7.1.1 split where only the large-pixel doors benefit from tiling.
5. *Best config of the four we tested:* **claude-opus-4-8 + swing-v2 prompt + 700 px
   tiles + 15 % overlap + NMS @ IoU 0.4**. This is the config to run on TEST.

### Open questions for follow-up

* Does dual-convention GT (arc-only + leaf+arc) recover IoU 0.5 F1 on A2.x?
* Does a finer tile (e.g. 400 px target → 4×4 grid on A2.x) help small-pixel doors?
* Does a model-specific bbox-shape post-processing step (e.g. shrink anthropic bboxes
  to a square anchored at the leaf endpoint) help?

## 9. Reproducibility

```bash
# baseline
.venv/bin/python -m src.swing_eval --split val
.venv/bin/python -m src.swing_reanalyze --pred outputs/eval/swing_swing-v2_val_predictions.csv

# tile-based
.venv/bin/python -m src.swing_tile_eval --split val
.venv/bin/python -m src.swing_reanalyze --pred outputs/eval/swingtile_swing-v2_t700_ov15_val_predictions.csv

# viz
.venv/bin/python -m src.viz_eval
.venv/bin/python -m src.viz_tile_eval
```
