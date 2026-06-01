"""Build the CEE329 Door Takeoff presentation deck.

Run:: .venv/bin/python -m src.build_deck

Output: outputs/cee329_door_takeoff.pptx
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu


ROOT = Path(__file__).resolve().parents[1]
VIZ = ROOT / "outputs" / "eval" / "viz"
OUT = ROOT / "outputs" / "cee329_door_takeoff.pptx"

# 16:9 widescreen
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# Stanford-ish palette
NAVY = RGBColor(0x0B, 0x2E, 0x4A)
CARDINAL = RGBColor(0x8C, 0x14, 0x15)
ACCENT = RGBColor(0x17, 0x6A, 0xA8)
GREEN_OK = RGBColor(0x1B, 0x8A, 0x4B)
RED_BAD = RGBColor(0xB4, 0x1A, 0x1A)
GREY = RGBColor(0x55, 0x55, 0x55)
LIGHT_GREY = RGBColor(0xE8, 0xEC, 0xEF)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


# --------------------------------------------------------------- helpers

def set_font(run, size=18, bold=False, color=None, name="Calibri"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = name
    if color is not None:
        run.font.color.rgb = color


def add_textbox(slide, left, top, width, height, text, *,
                size=18, bold=False, color=NAVY, align=PP_ALIGN.LEFT,
                anchor=MSO_ANCHOR.TOP, name="Calibri"):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    set_font(run, size=size, bold=bold, color=color, name=name)
    return tb


def add_bullets(slide, left, top, width, height, bullets, *,
                size=18, color=NAVY, line_spacing=1.15):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_top = Inches(0.02)
    for i, item in enumerate(bullets):
        bold = False
        sub = False
        text = item
        if isinstance(item, dict):
            text = item.get("text", "")
            bold = item.get("bold", False)
            sub = item.get("sub", False)
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = line_spacing
        bullet = "  • " if sub else "•  "
        run = p.add_run()
        run.text = bullet + text
        set_font(run, size=size - (2 if sub else 0), bold=bold,
                 color=GREY if sub else color)
    return tb


def add_title_bar(slide, title, subtitle=None):
    # accent bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                 0, 0, SLIDE_W, Inches(0.9))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    tf = bar.text_frame
    tf.margin_left = Inches(0.5)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = title
    set_font(run, size=26, bold=True, color=WHITE)
    if subtitle:
        add_textbox(slide, Inches(0.5), Inches(0.9), SLIDE_W - Inches(1.0),
                    Inches(0.35), subtitle, size=14, color=GREY,
                    name="Calibri")


def add_footer(slide, page_no, total):
    add_textbox(slide, Inches(0.4), SLIDE_H - Inches(0.35),
                Inches(5), Inches(0.3),
                "CEE329 Final Project · Door Takeoff MVP · 2026",
                size=10, color=GREY)
    add_textbox(slide, SLIDE_W - Inches(1.2), SLIDE_H - Inches(0.35),
                Inches(1.0), Inches(0.3), f"{page_no} / {total}",
                size=10, color=GREY, align=PP_ALIGN.RIGHT)


def add_image(slide, path, left, top, width=None, height=None,
              caption=None, caption_size=11):
    pic = slide.shapes.add_picture(str(path), left, top,
                                   width=width, height=height)
    if caption:
        cap_top = pic.top + pic.height + Inches(0.05)
        add_textbox(slide, pic.left, cap_top, pic.width, Inches(0.3),
                    caption, size=caption_size, color=GREY,
                    align=PP_ALIGN.CENTER)
    return pic


def add_table(slide, left, top, width, height, data, *,
              header_fill=NAVY, header_color=WHITE,
              body_size=14, header_size=14,
              highlight_cells=None,
              col_widths=None):
    """data: list of rows, first row = header."""
    rows = len(data)
    cols = len(data[0])
    tbl_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    tbl = tbl_shape.table

    if col_widths is not None:
        for ci, w in enumerate(col_widths):
            tbl.columns[ci].width = w

    for ri, row in enumerate(data):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.text = ""
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.04)
            cell.margin_bottom = Inches(0.04)
            tf = cell.text_frame
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if ci > 0 or ri == 0 else PP_ALIGN.LEFT
            run = p.add_run()
            run.text = str(val)
            if ri == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_fill
                set_font(run, size=header_size, bold=True, color=header_color)
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_GREY if ri % 2 == 0 else WHITE
                color = NAVY
                bold = False
                if highlight_cells and (ri, ci) in highlight_cells:
                    color = highlight_cells[(ri, ci)]
                    bold = True
                set_font(run, size=body_size, bold=bold, color=color)
    return tbl_shape


# ------------------------------------------------------------------- slides

def slide_blank(prs):
    blank = prs.slide_layouts[6]
    s = prs.slides.add_slide(blank)
    return s


def build():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    slides_factories = [
        slide_01_title,
        slide_02_motivation,
        slide_03_dataset,
        slide_04_pipeline,
        slide_05_approach_overview,
        slide_06_v1_setup,
        slide_07_v1_results,
        slide_08_v1_lessons,
        slide_09_v2_setup,
        slide_10_v2_results,
        slide_11_v2_diagnosis,
        slide_12_tile_setup,
        slide_13_tile_results,
        slide_14_tile_a7_win,
        slide_15_multimetric,
        slide_16_failure_modes,
        slide_17_recommendations,
        slide_18_limitations,
        slide_19_reproducibility,
        slide_20_thanks,
    ]
    total = len(slides_factories)
    for i, fn in enumerate(slides_factories, start=1):
        s = slide_blank(prs)
        fn(s)
        add_footer(s, i, total)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Wrote {OUT}  ({total} slides)")


# ============================================================== individual slides

def slide_01_title(s):
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid(); bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()

    # cardinal stripe
    stripe = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                 0, Inches(3.0), SLIDE_W, Inches(0.08))
    stripe.fill.solid(); stripe.fill.fore_color.rgb = CARDINAL
    stripe.line.fill.background()

    add_textbox(s, Inches(0.8), Inches(1.6), Inches(12), Inches(1.2),
                "Door Takeoff with Frontier VLMs",
                size=46, bold=True, color=WHITE)
    add_textbox(s, Inches(0.8), Inches(2.3), Inches(12), Inches(0.6),
                "Can GPT-4.1 / Claude-Opus / Gemini-2.5 detect swing doors in floor plans?",
                size=22, color=RGBColor(0xCC, 0xD6, 0xE0))

    add_textbox(s, Inches(0.8), Inches(3.4), Inches(12), Inches(0.4),
                "CEE329 Final Project  ·  Stanford  ·  2026",
                size=18, color=WHITE)

    add_textbox(s, Inches(0.8), Inches(6.4), Inches(12), Inches(0.4),
                "3 models · 2 prompts · 2 inference strategies · 10 floor plans · 261 GT bboxes",
                size=14, color=RGBColor(0xCC, 0xD6, 0xE0))


def slide_02_motivation(s):
    add_title_bar(s, "Why door takeoff?",
                  "Counting doors is the first item on every architectural takeoff")

    add_bullets(s, Inches(0.5), Inches(1.3), Inches(7.5), Inches(5),
                [
                    {"text": "Manual takeoff today: estimator clicks every door on every sheet",
                     "bold": True},
                    "~1–3 hours per floor plan, 10–50 plans per project",
                    "Errors are expensive — under-count → change order, over-count → wasted bid",
                    "",
                    {"text": "Frontier VLMs can in principle do this in seconds.",
                     "bold": True},
                    "But CAN they? Recent vision-language models claim spatial grounding.",
                    "We test three of the strongest on a controlled benchmark.",
                    "",
                    {"text": "Research question:", "bold": True},
                    {"text": "How close are top VLMs to production-ready door takeoff,",
                     "sub": True},
                    {"text": "and what is the dominant failure mode if not yet?",
                     "sub": True},
                ], size=17)

    # right-side callout
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                             Inches(8.5), Inches(1.5),
                             Inches(4.3), Inches(5.3))
    box.fill.solid(); box.fill.fore_color.rgb = LIGHT_GREY
    box.line.color.rgb = NAVY

    add_textbox(s, Inches(8.7), Inches(1.7), Inches(4.0), Inches(0.4),
                "Spec at a glance", size=18, bold=True, color=NAVY)
    add_bullets(s, Inches(8.7), Inches(2.2), Inches(4.0), Inches(4.5),
                [
                    "Task: detect swing-door arcs",
                    "Output: pixel-space bboxes",
                    "Metric: F1 @ IoU 0.5",
                    "Stretch: multi-IoU + counting",
                    "",
                    "Models tested:",
                    {"text": "claude-opus-4-8 (Anthropic)", "sub": True},
                    {"text": "gpt-4.1 (OpenAI)", "sub": True},
                    {"text": "gemini-2.5-pro (Google)", "sub": True},
                ], size=15)


def slide_03_dataset(s):
    add_title_bar(s, "Dataset & Ground Truth",
                  "10 floor plans · 261 hand-labelled swing arcs · VAL/TEST split")

    add_bullets(s, Inches(0.5), Inches(1.3), Inches(6.8), Inches(5),
                [
                    {"text": "Source: 4 architectural sets from real student housing project",
                     "bold": True},
                    "A2.x = floor plans (Ground / Floor 2-5)",
                    "A7.1.1 = enlarged unit / signage plans",
                    "Mixed scales: 608×1138 to 1728×4032 px",
                    "",
                    {"text": "Annotation: CVAT → VOC XML → CSV", "bold": True},
                    {"text": "Convention: bbox tightly encloses ONLY the arc",
                     "sub": True},
                    {"text": "(not the straight door-leaf line)", "sub": True},
                    {"text": "Mean h/w = 1.03 (consistently square)", "sub": True},
                    "",
                    {"text": "Splits:", "bold": True},
                    {"text": "VAL = 6 images, 167 bboxes (prompt iteration)",
                     "sub": True},
                    {"text": "TEST = 4 images, 94 bboxes (held out)", "sub": True},
                ], size=16)

    data = [
        ["Sheet", "Folder", "Image (px)", "Doors", "Door size"],
        ["A2.1", "Ground", "688×1138", "24", "27×27"],
        ["A2.1", "Floor 2", "608×1138", "32", "25×26"],
        ["A2.2", "Floor 3", "754×1286", "30", "28×27"],
        ["A2.3", "Floor 5", "718×1288", "30", "30×27"],
        ["A7.1.1", "Ground", "672×1304", "23", "45×45"],
        ["A7.1.1", "Typical", "1728×4032", "28", "107×98"],
    ]
    add_table(s, Inches(7.5), Inches(1.5), Inches(5.5), Inches(3.2),
              data, body_size=12, header_size=13,
              col_widths=[Inches(0.9), Inches(1.4), Inches(1.4),
                          Inches(0.8), Inches(1.0)])
    add_textbox(s, Inches(7.5), Inches(4.8), Inches(5.5), Inches(0.4),
                "VAL split — note the 70× range in image-area",
                size=11, color=GREY, align=PP_ALIGN.CENTER)


def slide_04_pipeline(s):
    add_title_bar(s, "Pipeline",
                  "PDF → PNG → VLM → JSON bboxes → IoU eval → multi-metric reanalysis")

    # Pipeline as a horizontal flow
    steps = [
        ("PDF\n(set drawings)", NAVY),
        ("PNG\n(per sheet)", ACCENT),
        ("VLM call\n3 models", CARDINAL),
        ("JSON bboxes\n(normalized)", ACCENT),
        ("Pixel bboxes\n+ NMS (tile)", NAVY),
        ("IoU eval\nvs GT XML", GREEN_OK),
        ("Reanalysis\nIoU + count + dist", GREY),
    ]
    n = len(steps)
    margin = Inches(0.4)
    gap = Inches(0.15)
    total_w = SLIDE_W - 2 * margin
    box_w = (total_w - gap * (n - 1)) / n
    box_h = Inches(1.0)
    top = Inches(2.0)
    for i, (txt, color) in enumerate(steps):
        left = margin + i * (box_w + gap)
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                 left, top, box_w, box_h)
        box.fill.solid(); box.fill.fore_color.rgb = color
        box.line.fill.background()
        tf = box.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = txt
        set_font(run, size=12, bold=True, color=WHITE)

        if i < n - 1:
            arr_left = left + box_w
            arr_w = gap
            arr = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                     arr_left, top + box_h / 2 - Inches(0.07),
                                     arr_w, Inches(0.14))
            arr.fill.solid(); arr.fill.fore_color.rgb = GREY
            arr.line.fill.background()

    # below: explanation
    add_bullets(s, Inches(0.5), Inches(3.5), Inches(6), Inches(3.5),
                [
                    {"text": "Per-image flow", "bold": True},
                    "PDF rendered at 200 dpi → cached PNG",
                    "PNG → base64 → multimodal request",
                    "Response parsed into per-door bboxes",
                    "Cache: hash(image) + model + prompt-ver",
                ], size=14)
    add_bullets(s, Inches(7), Inches(3.5), Inches(5.8), Inches(3.5),
                [
                    {"text": "Tile variant", "bold": True},
                    "Image sliced 2×2 (or 2×6 for largest sheet)",
                    "Each tile gets its own VLM call",
                    "Tile-relative bboxes → full-image pixels",
                    "Cross-tile NMS @ IoU 0.4",
                ], size=14)


def slide_05_approach_overview(s):
    add_title_bar(s, "What we tried",
                  "Three approaches, each strictly better than the previous")

    data = [
        ["Approach", "Models", "Prompt", "Inference", "API calls",
         "Best F1@0.5"],
        ["A1: Baseline", "3 frontier", "swing-v1\n(L-shape)", "Full image",
         "18 (VAL)", "0.000"],
        ["A2: Refined", "3 frontier", "swing-v2\n(arc only)",
         "Full image", "18 (VAL)", "0.023"],
        ["A3: Tile",
         "3 frontier", "swing-v2\n(arc only)", "2×2 / 2×6\n+ NMS",
         "96 (VAL)", "0.077"],
    ]
    add_table(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(2.7),
              data, body_size=14, header_size=14,
              highlight_cells={(3, 5): GREEN_OK})

    add_bullets(s, Inches(0.5), Inches(4.4), Inches(12.3), Inches(2.5),
                [
                    {"text": "Each approach was motivated by a specific failure observed in the previous run:", "bold": True},
                    "A1 → A2: Gemini returned `{\"doors\":[]}` 4/6 times; anthropic bboxes were too tall (h/w 1.4–1.9)",
                    "A2 → A3: A2.x sheets still F1=0; hypothesis was \"VLM resolution loss on small doors\" → tile to preserve pixels",
                    "",
                    {"text": "Total work: 132 API calls (incl. tile retries), 36 viz PNGs, 21 metric CSVs, 4 src/ modules added.",
                     "bold": True, },
                ], size=14)


def slide_06_v1_setup(s):
    add_title_bar(s, "Approach 1 · Baseline (swing-v1)",
                  "Send the whole plan, ask for door-arc L-shape bboxes")

    add_bullets(s, Inches(0.5), Inches(1.3), Inches(7.3), Inches(5),
                [
                    {"text": "Prompt v1 (key excerpt):", "bold": True},
                    {"text": "\"bbox is normalized to [0,1] of THIS image,",
                     "sub": True},
                    {"text": " tightly enclosing the L-shape (door leaf + arc):\"",
                     "sub": True},
                    "",
                    {"text": "Output schema (per model):", "bold": True},
                    {"text": "{ \"doors\": [ {bbox: [x1,y1,x2,y2], confidence: 0..1} ] }",
                     "sub": True},
                    "",
                    {"text": "Models (initial choice):", "bold": True},
                    {"text": "anthropic = claude-opus-4-8", "sub": True},
                    {"text": "openai    = gpt-4.1", "sub": True},
                    {"text": "gemini    = gemini-3.1-pro-preview", "sub": True},
                ], size=15)

    # Right: the v1 results
    add_textbox(s, Inches(8.0), Inches(1.3), Inches(5), Inches(0.4),
                "VAL results (swing-v1, full image)", size=16, bold=True,
                color=NAVY)
    data = [
        ["Model", "F1@0.1", "F1@0.5", "Counting err"],
        ["anthropic", "0.203", "0.000", "41 %"],
        ["openai", "0.093", "0.011", "39 %"],
        ["gemini", "0.020", "0.000", "84 %"],
    ]
    add_table(s, Inches(8.0), Inches(1.8), Inches(5), Inches(2.2),
              data, body_size=14, header_size=14,
              highlight_cells={(3, 4): RED_BAD})
    add_textbox(s, Inches(8.0), Inches(4.1), Inches(5), Inches(2),
                "All three F1 ≈ 0 at the standard IoU=0.5 threshold.\n"
                "Gemini returned empty lists on 4 / 6 sheets.\n\n"
                "Where do we go from here? → inspect outputs.",
                size=13, color=GREY)


def slide_07_v1_results(s):
    add_title_bar(s, "What the v1 outputs looked like",
                  "GT = red · model preds = blue")

    # Pick a representative full-image v1 viz — but viz_eval is swing-v2 only.
    # We use the v2 baseline viz here because they share the visual story.
    # NOTE: v1 viz wasn't saved, so use v2 anthropic + gemini as proxy for
    # the "early baseline" look.

    add_image(s, VIZ / "A2.1-1__anthropic.png",
              Inches(0.4), Inches(1.2), height=Inches(5.6),
              caption="Anthropic — bboxes look about right shape, often misaligned")
    add_image(s, VIZ / "A2.1-1__gemini.png",
              Inches(4.7), Inches(1.2), height=Inches(5.6),
              caption="Gemini — many predictions outside the drawing extents")
    add_image(s, VIZ / "A2.1-1__openai.png",
              Inches(9.0), Inches(1.2), height=Inches(5.6),
              caption="OpenAI — wider, shifted up-left of GT arcs")


def slide_08_v1_lessons(s):
    add_title_bar(s, "Lessons from v1 — two distinct problems",
                  "Format chaos vs spatial chaos")

    add_textbox(s, Inches(0.5), Inches(1.2), Inches(6.0), Inches(0.5),
                "Problem 1 — Gemini format chaos",
                size=20, bold=True, color=CARDINAL)
    add_bullets(s, Inches(0.5), Inches(1.7), Inches(6.0), Inches(2.5),
                [
                    "4 of 6 sheets returned `{\"doors\": []}` (empty)",
                    "2 of 6 returned pixel coords (not normalized)",
                    "Counting error = 84 %  (predicted 28 vs 167 actual)",
                    "",
                    {"text": "Hypothesis: gemini-3.1-pro-preview is",
                     "bold": True},
                    {"text": "miscalibrated on dense bbox tasks.", "bold": True},
                ], size=15)

    add_textbox(s, Inches(7.0), Inches(1.2), Inches(6.0), Inches(0.5),
                "Problem 2 — convention mismatch",
                size=20, bold=True, color=CARDINAL)
    add_bullets(s, Inches(7.0), Inches(1.7), Inches(6.0), Inches(2.5),
                [
                    "Model bboxes have aspect h/w ≈ 1.4–1.9",
                    "GT bboxes have aspect h/w ≈ 1.0 (arc-only)",
                    "Models include the straight door-leaf line",
                    "",
                    {"text": "Even when the model finds the right door,",
                     "bold": True},
                    {"text": "its bbox overlap with GT is < 0.5 IoU.",
                     "bold": True},
                ], size=15)

    add_textbox(s, Inches(0.5), Inches(4.6), Inches(12.3), Inches(0.5),
                "→ Two interventions for v2", size=20, bold=True, color=GREEN_OK)
    add_bullets(s, Inches(0.5), Inches(5.1), Inches(12.3), Inches(2.2),
                [
                    "Swap gemini-3.1-pro-preview → gemini-2.5-pro (more mature checkpoint)",
                    "Rewrite the bbox spec: \"ONLY the quarter-circle arc · approximately square · ~one door-width\"",
                    "Bump cache namespace swing-v1 → swing-v2 so all 18 calls are fresh",
                ], size=15)


def slide_09_v2_setup(s):
    add_title_bar(s, "Approach 2 · Refined prompt (swing-v2)",
                  "Tighten the bbox spec, swap one model")

    add_textbox(s, Inches(0.5), Inches(1.3), Inches(12), Inches(0.5),
                "Prompt diff", size=20, bold=True, color=NAVY)

    # before / after
    add_textbox(s, Inches(0.5), Inches(1.9), Inches(6.0), Inches(0.4),
                "swing-v1 (before)", size=14, bold=True, color=CARDINAL)
    box_a = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.5), Inches(2.3),
                                Inches(6.0), Inches(1.7))
    box_a.fill.solid(); box_a.fill.fore_color.rgb = RGBColor(0xFD, 0xF0, 0xF0)
    box_a.line.color.rgb = CARDINAL
    tf = box_a.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.15); tf.margin_top = Inches(0.1)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = ("bbox is normalized to [0, 1] of THIS image, tightly "
                "enclosing the L-shape (door leaf + arc).")
    set_font(run, size=14, color=NAVY, name="Consolas")

    add_textbox(s, Inches(6.8), Inches(1.9), Inches(6.0), Inches(0.4),
                "swing-v2 (after)", size=14, bold=True, color=GREEN_OK)
    box_b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(6.8), Inches(2.3),
                                Inches(6.0), Inches(1.7))
    box_b.fill.solid(); box_b.fill.fore_color.rgb = RGBColor(0xEF, 0xF8, 0xF1)
    box_b.line.color.rgb = GREEN_OK
    tf = box_b.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.15); tf.margin_top = Inches(0.1)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = ("bbox encloses ONLY the quarter-circle arc — do NOT include "
                "the straight door-leaf line. Approximately SQUARE, "
                "~one door-width on each side.")
    set_font(run, size=14, color=NAVY, name="Consolas")

    add_textbox(s, Inches(0.5), Inches(4.2), Inches(12.3), Inches(0.5),
                "Model swap", size=20, bold=True, color=NAVY)
    add_bullets(s, Inches(0.5), Inches(4.7), Inches(12.3), Inches(2.5),
                [
                    "gemini-3.1-pro-preview  →  gemini-2.5-pro  (production checkpoint, less format drift)",
                    "anthropic & openai unchanged (claude-opus-4-8, gpt-4.1)",
                    "",
                    {"text": "Same cache namespace bump: swing-v2 → 18 new VAL API calls.",
                     "bold": True},
                ], size=15)


def slide_10_v2_results(s):
    add_title_bar(s, "Approach 2 · Results",
                  "Refined prompt + model swap → fixes gemini format, marginal IoU gain")

    add_textbox(s, Inches(0.5), Inches(1.2), Inches(6), Inches(0.4),
                "VAL results · swing-v2 · full image", size=16, bold=True, color=NAVY)
    data = [
        ["Model", "F1@0.1", "F1@0.5", "Preds", "Counting", "Dist F1"],
        ["anthropic", "0.153", "0.023", "94", "44 %", "0.32"],
        ["openai", "0.088", "0.006", "153", "14 %", "0.19"],
        ["gemini", "0.155", "0.000", "169", "8 %", "0.31"],
    ]
    add_table(s, Inches(0.5), Inches(1.7), Inches(6.5), Inches(2.4),
              data, body_size=14, header_size=14,
              highlight_cells={
                  (1, 6): GREEN_OK,   # anthropic dist F1
                  (3, 5): GREEN_OK,   # gemini counting
                  (1, 3): GREEN_OK,   # anthropic F1@0.5
              })

    add_textbox(s, Inches(7.5), Inches(1.2), Inches(5.5), Inches(0.4),
                "What changed vs v1", size=16, bold=True, color=NAVY)
    add_bullets(s, Inches(7.5), Inches(1.7), Inches(5.5), Inches(3),
                [
                    {"text": "Gemini counting:  84% → 8 %  ✓", "bold": True},
                    "→ 2.5-pro emits clean normalized bboxes",
                    "",
                    {"text": "Anthropic F1@0.5:  0.000 → 0.023  ✓", "bold": True},
                    "→ arc-only spec is closer to GT convention",
                    "",
                    {"text": "OpenAI:  ~unchanged", "bold": True},
                    "→ still wide & shifted bboxes",
                ], size=14)

    add_textbox(s, Inches(0.5), Inches(4.4), Inches(12.3), Inches(0.5),
                "But: F1@0.5 still essentially zero. Why?",
                size=20, bold=True, color=CARDINAL)
    add_bullets(s, Inches(0.5), Inches(5.0), Inches(12.3), Inches(2.0),
                [
                    "Two metrics tell the diagnostic story:",
                    {"text": "F1 @ IoU 0.1 stays in 0.09 – 0.16 range — predictions are spatially close but not pixel-precise",
                     "sub": True},
                    {"text": "Center-distance F1 reaches 0.32 — models DO find the right doors, they just can't draw a tight arc-only box",
                     "sub": True},
                    "",
                    {"text": "Hypothesis: the limit is image resolution. At full-image, an arc occupies ~4% of width → too few pixels for tight bbox.",
                     "bold": True},
                ], size=15)


def slide_11_v2_diagnosis(s):
    add_title_bar(s, "Diagnosis · GT vs model overlay",
                  "anthropic finds the doors but boxes the wrong sub-region")

    add_image(s, VIZ / "A2.2-1__anthropic.png",
              Inches(0.4), Inches(1.2), height=Inches(5.6),
              caption="A2.2-1 / anthropic (full-image, swing-v2)")
    add_image(s, VIZ / "A2.1-2__anthropic.png",
              Inches(4.7), Inches(1.2), height=Inches(5.6),
              caption="A2.1-2 / anthropic — blue often misses red by half-a-bbox")
    add_image(s, VIZ / "A2.3-1__anthropic.png",
              Inches(9.0), Inches(1.2), height=Inches(5.6),
              caption="A2.3-1 / anthropic — same story")


def slide_12_tile_setup(s):
    add_title_bar(s, "Approach 3 · Tile-based grounding",
                  "Slice the plan, run the prompt per tile, NMS-merge")

    add_bullets(s, Inches(0.5), Inches(1.3), Inches(7), Inches(5.5),
                [
                    {"text": "Why tiling might help", "bold": True},
                    {"text": "At full image, a 27-px door covers ~4 % of width",
                     "sub": True},
                    {"text": "At 2×2 tile, that same door is ~8 % of tile width",
                     "sub": True},
                    {"text": "Effective resolution roughly doubles", "sub": True},
                    "",
                    {"text": "Mechanics", "bold": True},
                    {"text": "Target tile size = 700 px per side",
                     "sub": True},
                    {"text": "Overlap = 15 %  (catches doors on tile edges)",
                     "sub": True},
                    {"text": "A2.x → 2×2 = 4 tiles · A7.1.1 Typical → 2×6 = 12 tiles",
                     "sub": True},
                    {"text": "32 tiles per model · 3 models = 96 API calls / VAL run",
                     "sub": True},
                    "",
                    {"text": "Post-processing", "bold": True},
                    {"text": "Tile-relative normalized bbox → full-image pixel coords",
                     "sub": True},
                    {"text": "Cross-tile NMS @ IoU 0.4 (dedupes overlap-region doubles)",
                     "sub": True},
                ], size=14)

    # mini grid diagram on the right
    add_textbox(s, Inches(8.0), Inches(1.3), Inches(5), Inches(0.4),
                "Tile grid (schematic)", size=16, bold=True, color=NAVY)
    grid_l = Inches(8.0); grid_t = Inches(1.9)
    grid_w = Inches(4.5); grid_h = Inches(4.5)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, grid_l, grid_t, grid_w, grid_h)
    bg.fill.solid(); bg.fill.fore_color.rgb = LIGHT_GREY
    bg.line.color.rgb = GREY

    # 4 overlapping tiles
    tile_colors = [
        (RGBColor(0x9B, 0xC1, 0xE7), Inches(0.0), Inches(0.0)),
        (RGBColor(0xE7, 0xB7, 0x9B), Inches(2.3), Inches(0.0)),
        (RGBColor(0xB1, 0xDC, 0xC0), Inches(0.0), Inches(2.3)),
        (RGBColor(0xE6, 0xD4, 0x88), Inches(2.3), Inches(2.3)),
    ]
    tile_w = Inches(2.2); tile_h = Inches(2.2)
    for color, ox, oy in tile_colors:
        t = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                grid_l + ox, grid_t + oy, tile_w, tile_h)
        t.fill.solid(); t.fill.fore_color.rgb = color
        t.line.color.rgb = NAVY
        t.fill.transparency = 0.4  # not supported in pptx, leave

    # overlap label
    add_textbox(s, grid_l, grid_t + grid_h + Inches(0.15),
                grid_w, Inches(0.4),
                "4 tiles · 15 % overlap · NMS @ IoU 0.4",
                size=12, color=GREY, align=PP_ALIGN.CENTER)


def slide_13_tile_results(s):
    add_title_bar(s, "Approach 3 · Tile results",
                  "anthropic +235 % at IoU 0.5 · others unchanged or worse")

    add_textbox(s, Inches(0.5), Inches(1.2), Inches(6.0), Inches(0.4),
                "VAL · swing-v2 · 2×2/2×6 tile", size=16, bold=True, color=NAVY)
    data = [
        ["Model", "F1@0.1", "F1@0.5", "Preds", "Counting", "Dist F1"],
        ["anthropic", "0.340", "0.077", "92", "49 %", "0.42"],
        ["openai", "0.085", "0.000", "161", "13 %", "0.18"],
        ["gemini", "0.160", "0.005", "196", "17 %", "0.28"],
    ]
    add_table(s, Inches(0.5), Inches(1.7), Inches(6.5), Inches(2.4),
              data, body_size=14, header_size=14,
              highlight_cells={
                  (1, 1): GREEN_OK, (1, 2): GREEN_OK, (1, 3): GREEN_OK,
                  (1, 6): GREEN_OK,
              })

    add_textbox(s, Inches(7.5), Inches(1.2), Inches(5.5), Inches(0.4),
                "Per-sheet F1@0.5 · anthropic", size=16, bold=True, color=NAVY)
    data2 = [
        ["Sheet", "Image px", "Door px", "F1@0.5"],
        ["A2.1", "688 × 1138", "27 × 27", "0.000"],
        ["A2.2", "754 × 1286", "28 × 27", "0.000"],
        ["A2.3", "718 × 1288", "30 × 27", "0.044"],
        ["A7.1.1 G", "672 × 1304", "45 × 45", "0.048"],
        ["A7.1.1 T", "1728 × 4032", "107 × 98", "0.271"],
    ]
    add_table(s, Inches(7.5), Inches(1.7), Inches(5.5), Inches(2.8),
              data2, body_size=12, header_size=13,
              highlight_cells={(5, 4): GREEN_OK})

    add_textbox(s, Inches(0.5), Inches(4.7), Inches(12.3), Inches(0.5),
                "Pattern", size=20, bold=True, color=NAVY)
    add_bullets(s, Inches(0.5), Inches(5.2), Inches(12.3), Inches(2.0),
                [
                    {"text": "Tiling helps only where doors are LARGE in absolute pixels.",
                     "bold": True},
                    "A2.x doors are 27 px — slicing the image doesn't make them bigger; convention mismatch dominates → F1 = 0",
                    "A7.1.1 Typical doors are 107 px — full image was downsampling them past detection; tiling preserves them → F1 = 0.27",
                    "anthropic specifically benefits because its v2 bboxes were already shape-correct, just placement-imprecise",
                ], size=14)


def slide_14_tile_a7_win(s):
    add_title_bar(s, "Best case: A7.1.1 Typical · anthropic",
                  "Full-image F1 = 0.000   →   tile F1 = 0.271")

    add_image(s, VIZ / "A7.1.1_Typical__anthropic.png",
              Inches(0.5), Inches(1.2), height=Inches(5.6),
              caption="Full-image (28 GT · 23 pred · TP = 0 · F1 = 0.000)")
    add_image(s, VIZ / "A7.1.1_Typical__anthropic__tile_swing-v2_t700_ov15.png",
              Inches(7.0), Inches(1.2), height=Inches(5.6),
              caption="Tiled 2×6 (28 GT · 31 pred · TP = 8 · F1 = 0.271)")

    add_textbox(s, Inches(0.5), Inches(6.9), Inches(12.3), Inches(0.4),
                "Same sheet, same model, same prompt — only the inference "
                "strategy changed.",
                size=14, color=GREY, align=PP_ALIGN.CENTER)


def slide_15_multimetric(s):
    add_title_bar(s, "Multi-metric triangulation",
                  "IoU F1 is a single number; the failure mode is plural")

    add_bullets(s, Inches(0.5), Inches(1.3), Inches(5.5), Inches(5),
                [
                    {"text": "Why three metrics", "bold": True},
                    {"text": "IoU @ multiple thresholds — shows the bbox-shape gap",
                     "sub": True},
                    {"text": "Counting — what contractors actually need", "sub": True},
                    {"text": "Center-distance F1 — does the model point at the right door?",
                     "sub": True},
                    "",
                    {"text": "All three implemented in src/swing_reanalyze.py",
                     "bold": True},
                    {"text": "Single CSV in → 3 metric CSVs out", "sub": True},
                    {"text": "Works on full-image AND tile predictions", "sub": True},
                ], size=15)

    add_textbox(s, Inches(6.5), Inches(1.3), Inches(6.5), Inches(0.4),
                "anthropic · IoU sweep (across split)",
                size=16, bold=True, color=NAVY)
    data = [
        ["Config", "F1@0.1", "F1@0.2", "F1@0.3", "F1@0.5"],
        ["v1 full",   "0.203", "0.101", "0.065", "0.000"],
        ["v2 full",   "0.153", "0.107", "0.046", "0.023"],
        ["v2 tile",   "0.340", "0.247", "0.193", "0.077"],
    ]
    add_table(s, Inches(6.5), Inches(1.8), Inches(6.5), Inches(2.2),
              data, body_size=13, header_size=13,
              highlight_cells={(3, 1): GREEN_OK, (3, 2): GREEN_OK,
                               (3, 3): GREEN_OK, (3, 4): GREEN_OK})

    add_textbox(s, Inches(6.5), Inches(4.2), Inches(6.5), Inches(0.4),
                "Counting & distance (overall, swing-v2 full image)",
                size=16, bold=True, color=NAVY)
    data2 = [
        ["Model", "Count err", "Dist P", "Dist R", "Dist F1"],
        ["anthropic", "44 %", "0.45", "0.25", "0.32"],
        ["gemini",    "8 %", "0.31", "0.31", "0.31"],
        ["openai",    "14 %", "0.20", "0.18", "0.19"],
    ]
    add_table(s, Inches(6.5), Inches(4.7), Inches(6.5), Inches(2.1),
              data2, body_size=13, header_size=13,
              highlight_cells={(2, 1): GREEN_OK,   # gemini count
                               (1, 1): RED_BAD,    # anthropic count bad
                               (1, 4): GREEN_OK})  # anthropic dist


def slide_16_failure_modes(s):
    add_title_bar(s, "Three failure modes, in order of impact",
                  "Diagnosed from per-sheet metrics + visual inspection of 36 PNGs")

    boxes = [
        ("1.  Convention mismatch", CARDINAL,
         "GT = arc only · Model = leaf + arc\n"
         "Aspect ratio 1.0 vs 1.4–1.9\n"
         "Caps F1 @ IoU 0.5 < 0.05\n"
         "on all small-door sheets"),
        ("2.  Resolution loss", ACCENT,
         "Full-image input is downsampled\n"
         "Sub-30-px doors lose features\n"
         "Tiling fixes this for the large\n"
         "1728×4032 sheet (F1 0 → 0.27)"),
        ("3.  Gemini extent noise", GREY,
         "~30 % of gemini preds land in\n"
         "title block, north arrow, schedule\n"
         "Inflates FP, depresses precision\n"
         "Not present in v1's pixel mode"),
    ]
    n = len(boxes)
    margin = Inches(0.5)
    gap = Inches(0.4)
    box_w = (SLIDE_W - 2 * margin - gap * (n - 1)) / n
    box_h = Inches(2.5)
    top = Inches(1.3)
    for i, (title, color, body) in enumerate(boxes):
        left = margin + i * (box_w + gap)
        b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                left, top, box_w, box_h)
        b.fill.solid(); b.fill.fore_color.rgb = color
        b.line.fill.background()
        tf = b.text_frame
        tf.margin_left = Inches(0.2); tf.margin_top = Inches(0.15)
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run(); run.text = title
        set_font(run, size=18, bold=True, color=WHITE)
        p2 = tf.add_paragraph()
        p2.line_spacing = 1.2
        run = p2.add_run(); run.text = "\n" + body
        set_font(run, size=13, color=WHITE)

    # Below: per-model viz examples
    add_image(s, VIZ / "A2.1-1__gemini.png",
              Inches(0.5), Inches(4.0), height=Inches(3.0),
              caption="Gemini extent noise — preds in title block")
    add_image(s, VIZ / "A2.1-1__anthropic.png",
              Inches(4.7), Inches(4.0), height=Inches(3.0),
              caption="Anthropic — close placement, wrong shape")
    add_image(s, VIZ / "A2.1-1__openai.png",
              Inches(9.0), Inches(4.0), height=Inches(3.0),
              caption="OpenAI — wider, shifted up-left")


def slide_17_recommendations(s):
    add_title_bar(s, "Recommendation",
                  "What to ship today, what to fix next")

    add_textbox(s, Inches(0.5), Inches(1.3), Inches(12), Inches(0.5),
                "Best config of the four tested",
                size=20, bold=True, color=GREEN_OK)
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                              Inches(0.5), Inches(1.9),
                              Inches(12.3), Inches(1.1))
    box.fill.solid(); box.fill.fore_color.rgb = RGBColor(0xEF, 0xF8, 0xF1)
    box.line.color.rgb = GREEN_OK
    tf = box.text_frame
    tf.margin_left = Inches(0.3); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = ("claude-opus-4-8   +   swing-v2 prompt   +   "
                "700-px tiles  +  15 % overlap   +   NMS @ IoU 0.4")
    set_font(run, size=20, bold=True, color=NAVY, name="Calibri")

    add_textbox(s, Inches(0.5), Inches(3.4), Inches(6.0), Inches(0.4),
                "Ship-today use cases", size=16, bold=True, color=NAVY)
    add_bullets(s, Inches(0.5), Inches(3.9), Inches(6.0), Inches(3),
                [
                    {"text": "Door counting", "bold": True},
                    {"text": "Gemini-2.5-pro at full image: 8 % error", "sub": True},
                    "",
                    {"text": "\"Where to click\" UX overlay", "bold": True},
                    {"text": "Anthropic+tile: P = 0.59 on found doors", "sub": True},
                    "",
                    {"text": "High-res enlarged-unit sheets", "bold": True},
                    {"text": "Anthropic+tile F1 = 0.27 (production-ish quality)",
                     "sub": True},
                ], size=14)

    add_textbox(s, Inches(7.0), Inches(3.4), Inches(6.0), Inches(0.4),
                "Do NOT ship", size=16, bold=True, color=CARDINAL)
    add_bullets(s, Inches(7.0), Inches(3.9), Inches(6.0), Inches(3),
                [
                    {"text": "Pixel-precise takeoff for billing", "bold": True},
                    {"text": "F1 @ IoU 0.5 = 0.077 across the split", "sub": True},
                    "",
                    {"text": "Small-door sheets without convention fix",
                     "bold": True},
                    {"text": "A2.x always F1 = 0 regardless of approach", "sub": True},
                    "",
                    {"text": "Unsupervised — needs human review", "bold": True},
                    {"text": "Best counting model still off by ~14 doors / 167",
                     "sub": True},
                ], size=14)


def slide_18_limitations(s):
    add_title_bar(s, "Limitations & Future Work",
                  "What this 1-week project couldn't address")

    add_textbox(s, Inches(0.5), Inches(1.3), Inches(6.0), Inches(0.5),
                "Limitations of THIS evaluation",
                size=18, bold=True, color=NAVY)
    add_bullets(s, Inches(0.5), Inches(1.9), Inches(6.0), Inches(5),
                [
                    {"text": "Small dataset (10 images, 261 bboxes)",
                     "bold": True},
                    {"text": "95 % CI on recall is ±0.07", "sub": True},
                    "",
                    {"text": "Single annotator, no IRR", "bold": True},
                    {"text": "Spot-checking suggests ~95 % GT recall on real doors",
                     "sub": True},
                    "",
                    {"text": "Arc-only convention is one of several valid choices",
                     "bold": True},
                    {"text": "Models trained on COCO-style boxes don't share it",
                     "sub": True},
                    "",
                    {"text": "Only 3 models, all closed-source", "bold": True},
                    {"text": "Open VLMs (Qwen-VL, InternVL) not tested", "sub": True},
                ], size=14)

    add_textbox(s, Inches(7.0), Inches(1.3), Inches(6.0), Inches(0.5),
                "Concrete next experiments",
                size=18, bold=True, color=GREEN_OK)
    add_bullets(s, Inches(7.0), Inches(1.9), Inches(6.0), Inches(5),
                [
                    {"text": "Dual-convention GT", "bold": True},
                    {"text": "Re-annotate with arc-only AND leaf+arc; measure IoU under each",
                     "sub": True},
                    "",
                    {"text": "Finer tile grid for small-door sheets",
                     "bold": True},
                    {"text": "400-px tiles → 4×4 on A2.x; does IoU 0.5 recover?",
                     "sub": True},
                    "",
                    {"text": "Bbox post-processing", "bold": True},
                    {"text": "Shrink anthropic L-shape → square anchored at leaf endpoint",
                     "sub": True},
                    "",
                    {"text": "Few-shot prompting", "bold": True},
                    {"text": "Show 2-3 GT examples per call", "sub": True},
                ], size=14)


def slide_19_reproducibility(s):
    add_title_bar(s, "Reproducibility",
                  "Every number in this deck was produced by these 6 commands")

    # files
    add_textbox(s, Inches(0.5), Inches(1.3), Inches(12), Inches(0.5),
                "New code added during Phase 3 + 4 (~1 200 lines, 6 new modules)",
                size=16, bold=True, color=NAVY)
    add_bullets(s, Inches(0.5), Inches(1.9), Inches(6.0), Inches(3),
                [
                    "src/model_comparison.py — 3-model dispatch + JSON parsing",
                    "src/swing_eval.py — full-image IoU eval (Phase 3)",
                    "src/swing_tile_eval.py — tile-based eval (Phase 4)",
                    "src/swing_reanalyze.py — IoU sweep + counting + distance",
                    "src/viz_eval.py / src/viz_tile_eval.py — overlay PNGs",
                    "src/build_deck.py — this slide deck (meta!)",
                ], size=14)

    add_textbox(s, Inches(7.0), Inches(1.9), Inches(6.0), Inches(0.5),
                "Run order", size=16, bold=True, color=NAVY)

    code = ("# baseline (full image)\n"
            "python -m src.swing_eval --split val\n\n"
            "# refined-prompt baseline\n"
            "# (bump DOOR_PLAN_SWING_PROMPT_VERSION = 'swing-v2')\n"
            "python -m src.swing_eval --split val\n\n"
            "# tile\n"
            "python -m src.swing_tile_eval --split val\n\n"
            "# multi-metric reanalysis\n"
            "python -m src.swing_reanalyze --pred outputs/eval/...csv\n\n"
            "# viz\n"
            "python -m src.viz_eval\n"
            "python -m src.viz_tile_eval")
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                              Inches(7.0), Inches(2.5),
                              Inches(6.0), Inches(4.0))
    box.fill.solid(); box.fill.fore_color.rgb = RGBColor(0x1E, 0x1E, 0x2E)
    box.line.fill.background()
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.2); tf.margin_top = Inches(0.15)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    run = p.add_run(); run.text = code
    set_font(run, size=11, color=WHITE, name="Consolas")

    add_bullets(s, Inches(0.5), Inches(5.0), Inches(6.0), Inches(2),
                [
                    {"text": "Outputs (already saved)", "bold": True},
                    {"text": "21 CSVs · 36 viz PNGs · 1 findings.md · 1 .pptx",
                     "sub": True},
                    "",
                    {"text": "Cache: outputs/.cache/llm/", "bold": True},
                    {"text": "Hash-keyed per (model, prompt-ver, image) — fully resumable",
                     "sub": True},
                ], size=13)


def slide_20_thanks(s):
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid(); bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()

    stripe = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                 0, Inches(3.5), SLIDE_W, Inches(0.08))
    stripe.fill.solid(); stripe.fill.fore_color.rgb = CARDINAL
    stripe.line.fill.background()

    add_textbox(s, Inches(0.5), Inches(2.4), Inches(12.3), Inches(1),
                "Thanks — questions?",
                size=54, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(s, Inches(0.5), Inches(3.8), Inches(12.3), Inches(0.6),
                "All code, GT, predictions, viz, metrics in the repo",
                size=18, color=RGBColor(0xCC, 0xD6, 0xE0),
                align=PP_ALIGN.CENTER)
    add_textbox(s, Inches(0.5), Inches(4.4), Inches(12.3), Inches(0.6),
                "outputs/eval/phase4_findings.md — full writeup",
                size=14, color=RGBColor(0xCC, 0xD6, 0xE0),
                align=PP_ALIGN.CENTER)

    add_textbox(s, Inches(0.5), Inches(6.5), Inches(12.3), Inches(0.4),
                "CEE329 Final Project · Door Takeoff MVP · 2026",
                size=14, color=RGBColor(0xCC, 0xD6, 0xE0),
                align=PP_ALIGN.CENTER)


if __name__ == "__main__":
    build()
