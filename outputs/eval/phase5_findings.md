# Phase 5 — 1+2+3 Improvements & Iterative Tuning

> Door-takeoff MVP / CEE329 — VAL split, 6 images, 167 GT bboxes (swing-door arcs)
> Started 2026-05-31. Building on phase 4 baseline.

## TL;DR (results so far)

**Round 1 — upscale 2× + few-shot (V3):** big win on relaxed-IoU and distance metrics:
* **F1 @ IoU 0.1: 0.34 → 0.54 (+59 %)**
* **F1 @ IoU 0.3: 0.11 → 0.30 (+170 %)**
* **Distance F1: 0.42 → 0.65** (P=0.74 — when v3 predicts a door, it's right 74 % of the time)
* Counting error: 44.9 % → 36.7 %
* F1 @ IoU 0.5: 0.077 → 0.074 (still flat — pixel-precise localization remained hard)

**Round 2 — per-model bbox enlargement (V3E):** unlocked the strict-IoU metric:
* **F1 @ IoU 0.5: 0.074 → 0.210 (+184 %)** — best result of the project
* F1 @ IoU 0.3: 0.30 → **0.39**
* F1 @ IoU 0.1: 0.54 → **0.58**
* Per-sheet F1@0.5: **A7.1.1 Typical 0.27 → 0.48**, A7.1.1 Ground 0.05 → 0.30
* Distance F1 unchanged at 0.65 (P=0.74) — enlargement does not change center

The key insight was that v3 predictions are systematically *smaller* than the GT arc
bbox (anthropic pred_w/gt_w = 0.66, pred_h/gt_h = 0.83). A center-anchored multiplicative
scale per model (anthropic 1.3× × 1.5×) brings the bbox up to GT size *without* moving
the center — leveraging the high-precision center already produced by V3.

## Configurations evaluated

| ID  | Description                                                | Cache namespace   |
|-----|------------------------------------------------------------|-------------------|
| B0  | Full-image swing-v2 (phase-4 baseline)                     | swing-v2          |
| B1  | Tile-based swing-v2 (phase-4 tile baseline)                | swing-v2          |
| P1  | B1 + content-based L→square shrinker post-process          | swing-v2          |
| V3  | Tile + 2× upscale + few-shot swing-v3-fs example image     | swing-v3-fs       |
| V3P | V3 + L→square post-processor                               | swing-v3-fs       |
| **V3E** | **V3 + per-model center-anchored bbox enlargement** (best)| swing-v3-fs       |

Per-model enlargement factors (V3E), found by 0.1-step grid sweep on VAL:

| model     | sx  | sy  | rationale (pred / gt size before enlarge) |
|-----------|-----|-----|---------------------------------------------|
| anthropic | 1.3 | 1.5 | predictions ~ 0.66 × 0.83 of GT arc          |
| gemini    | 1.1 | 1.2 | predictions ~ 0.91 × 0.83 of GT arc          |
| openai    | 2.2 | 2.2 | predictions tiny relative to GT (low quality)|

## VAL anthropic-only metrics (167 GT)

| Config | F1@0.1 | F1@0.2 | F1@0.3 | F1@0.5 | Distance F1 | Distance P | Count err |
|--------|--------|--------|--------|--------|-------------|------------|-----------|
| B0     | 0.153  | 0.085  | 0.043  | 0.023  | 0.32        | 0.45       | 44 %      |
| B1     | 0.340  | 0.211  | 0.117  | 0.077  | 0.42        | 0.59       | 44.9 %    |
| P1     | 0.332  | 0.208  | 0.116  | 0.070  | 0.41        | 0.58       | 48.5 %    |
| V3     | 0.540  | 0.412  | 0.297  | 0.074  | 0.65        | 0.74       | 36.7 %    |
| V3P    | 0.493  | 0.378  | 0.250  | 0.068  | 0.65        | 0.74       | 36.7 %    |
| **V3E** | **0.581** | **0.500** | **0.392** | **0.210** | **0.65** | **0.74** | **37.1 %** |

V3E by sheet (anthropic):

| sheet         | TP | FP | FN | F1@0.5 |
|---------------|----|----|----|--------|
| A2.1-1        | 1  | 7  | 23 | 0.062  |
| A2.1-2        | 0  | 15 | 32 | 0.000  |
| A2.2-1        | 4  | 18 | 26 | 0.154  |
| A2.3-1        | 3  | 18 | 27 | 0.118  |
| A7.1.1 Ground | 7  | 17 | 16 | 0.298  |
| **A7.1.1 Typical** | **16** | **23** | **12** | **0.478** |

A7.1.1 Typical (1728×4032, doors ~107 px) is now operational territory — F1@0.5 = 0.48
with precision = 0.41. On a real contractor takeoff this would surface ~16 / 28 correct
boxes with ~23 false positives and 12 misses, well within human-review budget.

## What changed in V3

1. **2× upscale via PIL bicubic** before tiling — gives VLM 2× the pixels per door.
   GT pixel coords unchanged (predictions divided by upscale factor before NMS).
2. **Few-shot prompting** — synthetic example PNG (`data/examples/few_shot_swing_v3.png`)
   showing three swing doors in different orientations with their CORRECT arc-only
   bboxes drawn in red. Sent as the first image in a multi-image chat completion;
   query image is the second image.
3. **Prompt v3 text** — explicitly references "FIRST IMAGE = EXAMPLE" and
   "SECOND IMAGE = QUERY". Repeats the arc-only / square convention.
4. **Cache namespace bumped** to `swing-v3-fs` so v3 results don't pollute v2 cache.

The L→square content shrinker (P1, V3P) brought h/w from 1.34 → 1.05 mean but
didn't help F1@IoU 0.5 because the *placement* (not the shape) is what's off
at high IoU thresholds. It is preserved as an optional pipeline step but is
NOT used in the recommended config.

## What changed in V3E (the strict-IoU unlock)

5. **Per-model center-anchored bbox enlargement** (`src/swing_enlarge.py`).
   After V3 we ran `pred_w / gt_w` and `pred_h / gt_h` per model:

   | model     | pred_w/gt_w | pred_h/gt_h |
   |-----------|-------------|-------------|
   | anthropic | 0.66        | 0.83        |
   | gemini    | ~ 0.91      | ~ 0.83      |
   | openai    | ~ 0.40      | ~ 0.45      |

   V3 has the *right center* (Distance F1 = 0.65) but a *too-small bbox*. A
   per-model multiplicative scale on width / height — anchored at the box
   center — recovers IoU without disturbing the well-localized center.
   Found by 0.1-step grid sweep over (sx, sy) ∈ [1.0, 2.3]² maximizing
   F1 @ IoU 0.5 on VAL.

   This is a **post-processing step** — no extra API calls, no model fine-tuning.

## Output artifacts

* `outputs/eval/swingtilev3_swing-v3-fs_t700_ov15_up2_val_*.csv`              (V3)
* `outputs/eval/swingtilev3_swing-v3-fs_t700_ov15_up2_post_val_*.csv`         (V3P)
* `outputs/eval/swingtilev3_swing-v3-fs_t700_ov15_up2_val_enlarged_*.csv`     (V3E)
* `outputs/eval/swingtile_swing-v2_t700_ov15_val_post_*.csv`                  (P1)
* `data/examples/few_shot_swing_v3.png`                                       (few-shot fixture)
* `src/swing_enlarge.py`                                                       (V3E post-processor)

Reanalysis CSVs (multi-IoU + counting + distance) exist next to each predictions CSV.

## Reproduce V3E (recommended config)

```bash
# 1. Run V3 (tiled + 2× upscale + few-shot)
.venv/bin/python -m src.swing_tile_eval_v3 --split val --upscale 2.0

# 2. Apply per-model bbox enlargement
.venv/bin/python -m src.swing_enlarge \
  --pred outputs/eval/swingtilev3_swing-v3-fs_t700_ov15_up2_val_predictions.csv

# 3. Regenerate IoU/counting/distance reanalysis
.venv/bin/python -m src.swing_reanalyze \
  --pred outputs/eval/swingtilev3_swing-v3-fs_t700_ov15_up2_val_enlarged_predictions.csv

# 4. Visualize
.venv/bin/python -m src.viz_tile_eval \
  --pred outputs/eval/swingtilev3_swing-v3-fs_t700_ov15_up2_val_enlarged_predictions.csv
```

## Tuning attempts (in progress)

(Updated as runs complete.)

### Per-folder oracle enlargement (upper bound check)

We re-ran the enlargement with per-folder oracle factors (sx = mean GT width / mean
predicted width, per folder), to ask: how much would adaptive scaling help?

| Config                          | F1@0.1 | F1@0.3 | F1@0.5 |
|---------------------------------|--------|--------|--------|
| V3 (no enlarge)                 | 0.540  | 0.297  | 0.074  |
| V3E (per-model 1.3×1.5)         | 0.581  | 0.392  | 0.210  |
| V3E-oracle (per-folder oracle)  | 0.568  | 0.426  | 0.209  |

Oracle per-folder enlargement *only* lifts F1@0.3 (+9 % rel.) and is flat at IoU 0.5.
Conclusion: at IoU 0.5 the dominant remaining error is **center placement, not bbox
size**. Per-model 1.3×1.5 is close to ceiling for the strict-IoU metric on this dataset.

### Per-folder pred-vs-GT size analysis (anthropic, BEFORE enlarge)

| folder        | GT mean w×h | pred mean w×h | pred_w/gt_w | pred_h/gt_h |
|---------------|-------------|----------------|-------------|-------------|
| A2.1-1        | 27 × 27     | 18.6 × 26.0    | 0.69        | 0.96        |
| A2.1-2        | 25 × 26     | 17.7 × 24.6    | 0.72        | 0.96        |
| A2.2-1        | 28 × 27     | 22.2 × 26.6    | 0.78        | 0.99        |
| A2.3-1        | 30 × 27     | 23.5 × 25.8    | 0.78        | 0.96        |
| A7.1.1 Ground | 45 × 45     | 26.1 × 29.5    | 0.58        | 0.66        |
| A7.1.1 Typical| 107 × 98    | 52.3 × 58.5    | 0.49        | 0.60        |

Pattern: **the bigger the doors are in absolute pixels, the more the VLM under-sizes
them** (relative ratio drops from 0.78 to 0.49). This is the "VLM compresses big things
more" effect — the model treats high-resolution doors as smaller than they actually are.
On the small A2.x doors the height ratio is already ~ 1.0, so most of the V3E gain on
A2.x comes from widening (not heightening) the bbox.

### V3E final results — anthropic per sheet

| sheet         | GT | TP | FP | FN | F1@0.1 | F1@0.3 | F1@0.5 |
|---------------|----|----|----|----|--------|--------|--------|
| A2.1-1        | 24 | 1  | 7  | 23 | 0.250  | 0.062  | 0.062  |
| A2.1-2        | 32 | 0  | 15 | 32 | 0.340  | 0.043  | 0.000  |
| A2.2-1        | 30 | 4  | 18 | 26 | 0.577  | 0.346  | 0.154  |
| A2.3-1        | 30 | 3  | 18 | 27 | 0.588  | 0.431  | 0.118  |
| A7.1.1 Ground | 23 | 7  | 17 | 16 | 0.723  | 0.638  | 0.298  |
| **A7.1.1 Typical** | **28** | **16** | **23** | **12** | **0.806** | **0.627** | **0.478** |

A7.1.1 Typical hits F1@0.1 = 0.81 — doors are being found reliably; remaining work
is bbox refinement. A2.1-2 remains our hardest case (no IoU 0.5 hits) — the doors there
are smallest in absolute pixels (≤ 20 px in some cases at upscale 2×). Possible
follow-up: upscale 3-4× for A2.x specifically.

## Recommended config (post-V3E)

**`claude-opus-4-8 + tiled (700 px / 15 % overlap) + 2× upscale + few-shot prompt
+ per-model bbox enlargement (sx=1.3, sy=1.5)`**

* F1 @ IoU 0.5 — overall **0.21**, A7.1.1 Typical **0.48**
* F1 @ IoU 0.1 — overall **0.58**, A7.1.1 Typical **0.81**
* Distance F1 — **0.65** (P=0.74)
* Counting error — **37 %**

Ready for TEST evaluation.

---

## Round 3 — Method 4a: add Qwen3.5-plus as a 4th VLM (2026-06-01)

After advisor feedback we tried adding a **Qwen** model to the V3 pipeline. The only
Qwen variant accessible on the Stanford inference proxy is `qwen3.5-plus`
(the dedicated `qwen-vl-*` vision variants return 403 "Model not allowed"). Integration
was done via a non-invasive wrapper at `src/swing_tile_eval_qwen.py` that registers
a 4th `ModelSpec` into `MODELS_V3` at runtime; no existing source modified.

Smoke test passed (Qwen returned valid bbox JSON for the few-shot example image),
and a full VAL run completed against the V3 tile + few-shot pipeline.

### Qwen3.5-plus VAL results (V3 config, no enlargement)

| model     | TP | FP  | FN  | P     | R     | F1@0.5 |
|-----------|----|-----|-----|-------|-------|--------|
| anthropic | 11 | 118 | 156 | 0.085 | 0.066 | **0.074** |
| gemini    |  3 | 224 | 164 | 0.013 | 0.018 | 0.015  |
| openai    |  0 | 174 | 167 | 0.000 | 0.000 | 0.000  |
| **qwen**  |  6 | 177 | 161 | 0.033 | 0.036 | **0.034** |

Per-sheet, qwen finds 0 doors on every A2.x sheet (56 + 30 = 86 GT, 0 TP), 2 TP on A2.3,
and 4 TP on A7.1.1 (combined). The (P, R) is roughly halfway between gemini and anthropic,
with the same systematic blindness on A2.x that anthropic also suffers from.

### Takeaway

`qwen3.5-plus` is the only Qwen variant we can hit through the proxy, and it
performs **worse than anthropic** and only marginally better than gemini.
With no vision-tuned Qwen accessible (the `-vl-*` variants are blocked),
adding Qwen does *not* improve the ensemble — anthropic still wins.

Files:
* `outputs/eval/swingtilev3_qwen_val_predictions.csv`
* `outputs/eval/swingtilev3_qwen_val_metrics_overall.csv`
* `outputs/eval/swingtilev3_qwen_val_metrics_by_sheet.csv`
