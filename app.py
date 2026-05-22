"""Door Takeoff MVP — Streamlit app.

Run with::

    streamlit run app.py

Three tabs:

1. **Schedule (A10.2)** — door catalog from the schedule sheet, with a
   side-by-side comparison of Tesseract OCR baseline vs three multimodal
   models (OpenAI gpt-4o, Anthropic Claude Sonnet 4.6, Google Gemini 2.5).
2. **Plans (A2.1–A2.4)** — Tag-OCR detection (Tesseract @300 DPI filtered
   by the schedule vocabulary) on each floor plan, with bbox overlays plus
   an optional VLM comparison per floor.
3. **Takeoff** — schedule × plan reconciliation: each unique mark gets
   its schedule spec + plan-side count + status (matched / orphan / stray /
   mismatch). Exportable to CSV.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image as _PILImage

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import pdf_utils, evaluation, matching, plan_detector, plan_tiling
from src.schedule_extractor import run_schedule_extraction
from src.index_parser import load_index
from src.sheet_selector import (
    SCHEDULE_SHEET,
    plan_pages,
    plan_rows,
    relevant_sheets,
    schedule_page,
    schedule_row,
)
from src import model_comparison, sampling_eval


def _load_env() -> None:
    """Load .env so OPENAI_API_KEY etc. show up in os.environ."""
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(ROOT / ".env", override=False)
    except Exception:
        pass


DATA_DIR = ROOT / "data"
OUTPUT_CSV = ROOT / "outputs" / "takeoff.csv"
CONSTRUCTION_PDF = DATA_DIR / "1522-CONSTRUCTION SET-11.19.18.pdf"

DOOR_CROP = (0.0, 0.0, 0.33, 1.0)

# Tag-OCR detection runs at 300 DPI (plan_detector.PLAN_DPI), but the displayed
# plan image is rendered at this lower DPI to avoid heavy browser downscaling
# (which makes the plan look blurry). bbox coordinates are scaled from
# PLAN_DPI → PLAN_DISPLAY_DPI so the overlays still line up.
PLAN_DISPLAY_DPI = 120


# ============================================================ caches


@st.cache_data(show_spinner=False)
def _load_index_cached(data_dir_str: str):
    return load_index(Path(data_dir_str))


@st.cache_data(show_spinner=False)
def _render_page_cached(pdf_str: str, page_index: int, dpi: int = 200):
    img = pdf_utils.render_page(Path(pdf_str), page_index, dpi=dpi)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), img.size


@st.cache_data(show_spinner=False)
def _ocr_page_cached(pdf_str: str, page_index: int, dpi: int = 200,
                     crop: tuple | None = None):
    return pdf_utils.ocr_page(Path(pdf_str), page_index, dpi=dpi, crop=crop)


# ============================================================ health


def _file_health() -> list[tuple[str, bool, str]]:
    out = []
    out.append(
        (
            "Construction set PDF",
            CONSTRUCTION_PDF.exists(),
            f"{CONSTRUCTION_PDF.name} ({CONSTRUCTION_PDF.stat().st_size//(1024*1024)} MB)"
            if CONSTRUCTION_PDF.exists()
            else f"Missing: place file at {CONSTRUCTION_PDF}",
        )
    )
    xlsx = DATA_DIR / "Drawing Index.xlsx"
    out.append(
        (
            "Drawing Index (xlsx)",
            xlsx.exists(),
            xlsx.name if xlsx.exists() else "Missing — falling back to PDF OCR",
        )
    )
    out.append(("Tesseract OCR", pdf_utils._check_tesseract(), pdf_utils.tesseract_version()))
    return out


def _render_health_screen(health: list[tuple[str, bool, str]]) -> None:
    st.error("Project pre-flight failed. Fix the items below and reload.")
    for label, ok, detail in health:
        prefix = "✅" if ok else "❌"
        st.write(f"{prefix} **{label}** — {detail}")
    st.markdown(
        """
**To install Tesseract**
```bash
brew install tesseract
```

**To place data files**
Copy these into `./data/`:

- `1522-CONSTRUCTION SET-11.19.18.pdf`
- `Drawing Index.xlsx`
        """
    )
    st.stop()


# ============================================================ image helpers


def _door_schedule_image_bytes(index_df: pd.DataFrame) -> bytes | None:
    """Render the A10.2 door schedule (left-third crop) as PNG bytes."""
    page = schedule_page(index_df)
    if page is None:
        return None
    img = pdf_utils.render_page(CONSTRUCTION_PDF, page - 1, dpi=200)
    l, t, r, b = DOOR_CROP
    box = (int(l * img.width), int(t * img.height),
           int(r * img.width), int(b * img.height))
    cropped = img.crop(box)
    buf = io.BytesIO()
    cropped.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _plan_image_bytes_for_vlm(
    page_index: int, max_dim: int = 2048
) -> tuple[bytes, tuple[int, int]]:
    """Render a plan page at PLAN_DPI then downscale to ``max_dim`` px on the
    longer side and return PNG bytes plus the resized dimensions.

    The full 300-DPI plan rasters are very large (≈7000×10000 px) — too big to
    submit comfortably to multimodal endpoints. Returning the resized PNG keeps
    the request payload reasonable while preserving the plan's aspect ratio so
    the model's normalized bboxes still line up when overlaid on this image.
    """
    img = pdf_utils.render_page(
        CONSTRUCTION_PDF, page_index, dpi=plan_detector.PLAN_DPI
    )
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), _PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), img.size


# ============================================================ Tab 1: Schedule


def _render_model_columns(
    runs: dict[str, model_comparison.ModelRun],
    baseline_df: pd.DataFrame,
) -> None:
    columns_data: list[tuple[str, str, pd.DataFrame, str]] = []
    columns_data.append(("Tesseract (baseline)", "OCR + regex", baseline_df, ""))
    for _, run in runs.items():
        columns_data.append((run.spec.label, run.spec.model_id, run.df, run.error))

    cols = st.columns(len(columns_data))
    for col, (label, sublabel, df, err) in zip(cols, columns_data):
        with col:
            st.markdown(f"**{label}**")
            st.caption(f"`{sublabel}` — {len(df)} rows" if not err else f"`{sublabel}`")
            if err:
                st.error(err)
                continue
            view_cols = [
                c for c in
                ["mark", "type", "size", "material", "fire_rating",
                 "hardware_set", "notes"]
                if c in df.columns
            ]
            st.dataframe(
                df[view_cols] if view_cols else df,
                hide_index=True,
                use_container_width=True,
                height=360,
            )


def tab_schedule(index_df: pd.DataFrame) -> pd.DataFrame:
    """Schedule tab — A10.2 extraction + multi-model comparison."""
    st.subheader(f"Door schedule — {SCHEDULE_SHEET}")

    sched_row = schedule_row(index_df)
    if sched_row.empty:
        st.error(f"{SCHEDULE_SHEET} not found in the drawing index.")
        return pd.DataFrame()
    pdf_page = int(sched_row.iloc[0]["pdf_page"])

    cols = st.columns([2, 1])
    with cols[0]:
        with st.spinner("Rendering A10.2 (cached after first time)…"):
            img_bytes = _door_schedule_image_bytes(index_df)
        if not img_bytes:
            st.error("Could not render A10.2.")
            return pd.DataFrame()
        st.image(
            img_bytes,
            caption=f"{SCHEDULE_SHEET} — left-third crop (door schedule only)",
            use_container_width=True,
        )
    with cols[1]:
        with st.spinner("Running OCR…"):
            ocr = _ocr_page_cached(
                str(CONSTRUCTION_PDF), pdf_page - 1, 200, DOOR_CROP
            )
        st.metric("OCR mean confidence", f"{ocr.get('mean_conf', 0.0):.1f} / 100")
        st.markdown(f"**PDF page:** {pdf_page}")
        st.markdown(f"**Title:** {sched_row.iloc[0]['title']}")
        with st.expander("Raw OCR text"):
            st.text(ocr.get("text", ""))

    st.markdown("---")

    st.markdown("### Baseline schedule (Tesseract)")
    state_key = "schedule_df"
    if state_key not in st.session_state:
        with st.spinner("Extracting door schedule rows from OCR…"):
            df = run_schedule_extraction(index_df, CONSTRUCTION_PDF)
            df = matching.add_confidence_flags(df)
        st.session_state[state_key] = df

    schedule_df = st.session_state[state_key]
    st.caption(f"{len(schedule_df)} unique door marks parsed from OCR.")
    st.dataframe(
        schedule_df,
        hide_index=True,
        use_container_width=True,
        height=300,
    )

    st.markdown("---")
    st.markdown("### Multi-model comparison")
    st.markdown(
        "Run **OpenAI gpt-4o**, **Anthropic Claude Sonnet 4.6**, and "
        "**Google Gemini 2.5 Flash** on the same A10.2 crop and compare to the "
        "Tesseract baseline. A random sample of marks can be graded for "
        "per-model accuracy."
    )

    key_status = []
    for spec in model_comparison.MODELS.values():
        ok = bool(os.environ.get(spec.env_var))
        prefix = "✅" if ok else "❌"
        key_status.append(f"{prefix} `{spec.env_var}`")
    st.caption("API keys: " + " · ".join(key_status))
    if not all(os.environ.get(s.env_var) for s in model_comparison.MODELS.values()):
        with st.expander("How to add API keys"):
            st.markdown(
                """
1. Copy `.env.example` to `.env` (in the project root).
2. Paste your keys into `.env`. Never commit this file.
3. Restart Streamlit (`Ctrl+C`, then `streamlit run app.py`).
"""
            )

    runs_key = "model_runs"
    cols = st.columns([2, 1, 1])
    with cols[0]:
        run_clicked = st.button(
            "▶  Run all models on A10.2",
            type="primary",
            use_container_width=True,
        )
    with cols[1]:
        force_refresh = st.checkbox(
            "Skip cache",
            value=False,
            help="Force a fresh API call (otherwise reads from outputs/.cache/llm/).",
        )
    with cols[2]:
        if st.button("Clear comparison", use_container_width=True):
            st.session_state.pop(runs_key, None)
            st.session_state.pop("review_df", None)

    if run_clicked:
        runs: dict[str, model_comparison.ModelRun] = {}
        for key, spec in model_comparison.MODELS.items():
            with st.spinner(f"Calling {spec.label}…"):
                run = model_comparison.run_model(
                    key, img_bytes, use_cache=not force_refresh
                )
                runs[key] = run
                if run.error:
                    st.toast(f"{spec.label}: {run.error}", icon="⚠️")
                else:
                    badge = "cached" if run.cached else f"{run.elapsed_s:.1f}s"
                    st.toast(f"{spec.label}: {len(run.df)} rows ({badge})", icon="✅")
        st.session_state[runs_key] = runs
        st.session_state.pop("review_df", None)

    runs = st.session_state.get(runs_key)
    if not runs:
        st.info("Click **Run all models on A10.2** to populate the comparison.")
        return schedule_df

    baseline_df = model_comparison.tesseract_baseline_as_df(schedule_df)
    st.markdown("#### Side-by-side extraction")
    _render_model_columns(runs, baseline_df)

    st.markdown("---")
    st.markdown("#### Sampling-based human review")

    model_dfs: dict[str, pd.DataFrame] = {"Tesseract (baseline)": baseline_df}
    for _, run in runs.items():
        if not run.error:
            model_dfs[run.spec.label] = run.df

    if all(df.empty for df in model_dfs.values()):
        st.warning("No rows extracted — nothing to grade.")
        return schedule_df

    pool = sampling_eval.collect_marks(model_dfs)
    cols = st.columns([1, 1, 1, 2])
    with cols[0]:
        st.metric("Unique marks", len(pool))
    with cols[1]:
        sample_n = st.number_input(
            "Sample size",
            min_value=1,
            max_value=max(1, len(pool)),
            value=min(10, len(pool)),
            step=1,
        )
    with cols[2]:
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
    with cols[3]:
        expected_count = st.number_input(
            "Expected total doors",
            min_value=0,
            value=43,
            step=1,
            help="From the schedule's own row count (Tesseract finds 43).",
        )

    if st.button("🎲  Sample door marks", use_container_width=True):
        sampled = sampling_eval.sample_marks(model_dfs, int(sample_n), int(seed))
        review = sampling_eval.build_review_table(model_dfs, sampled)
        st.session_state["review_df"] = review
        st.session_state["sampled_marks"] = sampled

    review_df = st.session_state.get("review_df")
    if review_df is None or review_df.empty:
        st.caption("Click **Sample door marks** to draw a random subset.")
        return schedule_df

    st.markdown(
        f"**Sampled marks ({len(st.session_state['sampled_marks'])}):** "
        f"`{', '.join(st.session_state['sampled_marks'])}`"
    )
    st.caption(
        "Set **grade** to *Correct* / *Wrong* / *Partial* / *Skip*. "
        "Rows with `present = False` count against recall but aren't graded."
    )

    edited = st.data_editor(
        review_df,
        column_config={
            "grade": st.column_config.SelectboxColumn(
                "Grade",
                options=sampling_eval.GRADE_OPTIONS,
                required=True,
            ),
            "present": st.column_config.CheckboxColumn("Present?", disabled=True),
            "model": st.column_config.TextColumn("Model", disabled=True),
            "mark": st.column_config.TextColumn("Mark", disabled=True),
        },
        hide_index=True,
        use_container_width=True,
        key="review_editor",
        num_rows="fixed",
    )
    st.session_state["review_df"] = edited

    if st.button("📐  Compute accuracy", type="primary", use_container_width=True):
        model_row_counts = {label: len(df) for label, df in model_dfs.items()}
        n_sampled = len(st.session_state["sampled_marks"])
        summary = sampling_eval.compute_accuracy(
            edited, n_sampled, model_row_counts,
            expected_count=int(expected_count) if expected_count else None,
        )
        st.session_state["accuracy_summary"] = summary

    summary = st.session_state.get("accuracy_summary")
    if summary is not None and not summary.empty:
        st.markdown("#### Accuracy summary")
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.caption(
            "**Row accuracy** = (Correct + 0.5·Partial) / Graded. "
            "**Recall** = sampled-found / N. "
            "**Review rate** = (Wrong + Partial) / Graded — lower = more confident. "
            "**Count vs expected** = 1 − |total rows − expected| / expected."
        )

    return schedule_df


# ============================================================ Tab 2: Plans


def tab_plans(index_df: pd.DataFrame) -> pd.DataFrame:
    """Plans tab — Tag-OCR detection on A2.1–A2.4 + optional VLM comparison."""
    st.subheader("Floor plans — A2.1 to A2.4")
    st.markdown(
        "Find every door tag on each floor plan and visualise the matches as "
        "bounding boxes. The **Tag OCR baseline** runs Tesseract at "
        f"{plan_detector.PLAN_DPI} DPI and keeps every word that matches a "
        "known door mark from the schedule. Multimodal VLMs (gpt-4o, Claude "
        "Sonnet 4.6, Gemini 2.5 Flash) are an optional per-floor comparison."
    )

    # Need the schedule vocabulary to filter plan-side OCR words against.
    schedule_df = st.session_state.get("schedule_df")
    if schedule_df is None or schedule_df.empty:
        st.warning(
            "Open **Tab 1 — Schedule (A10.2)** first so the door vocabulary "
            "is loaded. Plan-side detection compares plan tags against the "
            "schedule's list of marks.",
            icon="📋",
        )
        return pd.DataFrame(columns=["mark", "source_sheet"])

    vocab = plan_detector._build_vocab(schedule_df["mark"].astype(str).tolist())
    st.caption(f"Schedule vocabulary: **{len(vocab)} unique door marks**.")

    plans_df = plan_rows(index_df)
    if plans_df.empty:
        st.error("No plan sheets (A2.1–A2.4 / A7.1.1) found in the drawing index.")
        return pd.DataFrame(columns=["mark", "source_sheet"])

    plan_pages_list = plan_pages(index_df)

    # ---------- Run Tag OCR across all plans
    state_key = "plan_tag_ocr_df"
    cols = st.columns([2, 1, 1])
    with cols[0]:
        run_clicked = st.button(
            "▶  Run Tag OCR across all plans",
            type="primary",
            use_container_width=True,
        )
    with cols[1]:
        if st.button("Clear plan detections", use_container_width=True):
            st.session_state.pop(state_key, None)
            st.session_state.pop("plan_vlm_runs", None)
            for tk in [k for k in st.session_state if k.startswith("plan_tile_runs__")]:
                st.session_state.pop(tk, None)
    with cols[2]:
        st.caption(f"DPI: {plan_detector.PLAN_DPI}")

    if run_clicked:
        with st.spinner(
            f"Tag-OCR scanning {len(plan_pages_list)} plan pages "
            f"(first time is slow; cached after that)…"
        ):
            df = plan_detector.detect_marks_on_all_plans(
                CONSTRUCTION_PDF, plan_pages_list, vocab,
                dpi=plan_detector.PLAN_DPI,
            )
        st.session_state[state_key] = df
        n_dets = len(df)
        n_unique = int(df["mark"].nunique()) if not df.empty else 0
        st.toast(
            f"Tag OCR: {n_dets} detections / {n_unique} unique marks",
            icon="✅",
        )

    plan_df = st.session_state.get(state_key, plan_detector._empty_plan_df())

    if plan_df.empty:
        st.info(
            "Click **Run Tag OCR across all plans** to detect every door tag "
            "across A2.1–A2.4. Detections will then feed Tab 3 (Takeoff)."
        )
        return pd.DataFrame(columns=["mark", "source_sheet"])

    # ---------- Aggregate metrics
    cols = st.columns(4)
    cols[0].metric("Total detections", len(plan_df))
    cols[1].metric("Unique marks", int(plan_df["mark"].nunique()))
    cols[2].metric("Floors covered", int(plan_df["source_sheet"].nunique()))
    cols[3].metric(
        "Mean confidence",
        f"{plan_df['confidence'].mean():.0f}" if not plan_df.empty else "—",
    )

    st.markdown("---")
    st.markdown("### Per-floor visualisation")

    # ---------- Floor selector (use the plans_df ordering)
    options = [
        f"{r['sheet_no']} — {str(r['title'])[:80]} (PDF page {r['pdf_page']})"
        for _, r in plans_df.iterrows()
    ]
    chosen = st.selectbox("Pick a floor", options, index=0, key="plan_view_choice")
    chosen_idx = options.index(chosen)
    chosen_row = plans_df.iloc[chosen_idx]
    chosen_sheet = str(chosen_row["sheet_no"])
    chosen_page = int(chosen_row["pdf_page"])
    page_index = chosen_page - 1

    floor_dets = (
        plan_df[plan_df["source_sheet"] == chosen_sheet].reset_index(drop=True)
    )

    # Scale Tag-OCR bboxes from PLAN_DPI (detection space) to PLAN_DISPLAY_DPI
    # (display space). OCR accuracy stays at 300 DPI; only the displayed image
    # is rendered smaller so the browser doesn't have to downsample 7000×10000
    # px (which is what makes the plan look blurry).
    scale = PLAN_DISPLAY_DPI / plan_detector.PLAN_DPI
    if not floor_dets.empty:
        scaled_dets = floor_dets.copy()
        for c in ("bbox_left", "bbox_top", "bbox_width", "bbox_height"):
            scaled_dets[c] = (
                scaled_dets[c].astype(float) * scale
            ).round().astype(int)
    else:
        scaled_dets = floor_dets

    cols = st.columns([3, 1])
    with cols[0]:
        with st.spinner(f"Rendering {chosen_sheet}…"):
            display_img = pdf_utils.render_page(
                CONSTRUCTION_PDF, page_index, dpi=PLAN_DISPLAY_DPI
            )
        annotated = plan_detector.draw_detections_on_image(
            display_img, scaled_dets,
            line_width=5, font_size=18, pad=20,
        )
        st.image(
            annotated,
            caption=(
                f"{chosen_sheet} — {len(floor_dets)} Tag-OCR bboxes overlaid "
                f"(detection @{plan_detector.PLAN_DPI} DPI, "
                f"display @{PLAN_DISPLAY_DPI} DPI)"
            ),
            use_container_width=True,
        )
    with cols[1]:
        st.markdown(f"**{chosen_sheet}** — *{chosen_row['title']}*")
        st.caption(f"PDF page {chosen_page}")
        st.metric("Detections on this floor", len(floor_dets))
        if not floor_dets.empty:
            this_counts = (
                floor_dets["mark"]
                .value_counts()
                .rename_axis("mark")
                .reset_index(name="count")
            )
            st.markdown("**Per-mark counts (this floor)**")
            st.dataframe(this_counts, hide_index=True, use_container_width=True)

    # ---------- Per-detection zoom (alignment debug)
    if not floor_dets.empty:
        st.markdown("### 🔎  Inspect a single detection")
        st.caption(
            "If the boxes on the overview look misaligned or unclear, pick "
            "any detection here to see a tight high-resolution crop centered "
            "on its OCR bbox. The right-hand panel highlights the exact pixels "
            f"Tesseract matched (rendered @{plan_detector.PLAN_DPI} DPI)."
        )

        det_options = [
            f"#{i} · '{r['mark']}' · ({int(r['bbox_left'])}, {int(r['bbox_top'])}) "
            f"· conf={float(r['confidence']):.0f}"
            for i, r in floor_dets.iterrows()
        ]
        chosen_det = st.selectbox(
            "Pick a detection",
            det_options,
            index=0,
            key=f"det_zoom_{chosen_sheet}",
        )
        det_idx = det_options.index(chosen_det)
        det = floor_dets.iloc[det_idx]

        # Render at full PLAN_DPI so the crop is razor-sharp.
        full_img = pdf_utils.render_page(
            CONSTRUCTION_PDF, page_index, dpi=plan_detector.PLAN_DPI
        )
        zoom_pad = 250  # px around the bbox to show context
        cl = max(0, int(det["bbox_left"]) - zoom_pad)
        ct = max(0, int(det["bbox_top"]) - zoom_pad)
        cr = min(full_img.width, int(det["bbox_left"]) + int(det["bbox_width"]) + zoom_pad)
        cb = min(full_img.height, int(det["bbox_top"]) + int(det["bbox_height"]) + zoom_pad)
        raw_crop = full_img.crop((cl, ct, cr, cb))

        # Bbox relative to the crop's origin.
        relative = pd.DataFrame([{
            "mark": str(det["mark"]),
            "bbox_left": int(det["bbox_left"]) - cl,
            "bbox_top": int(det["bbox_top"]) - ct,
            "bbox_width": int(det["bbox_width"]),
            "bbox_height": int(det["bbox_height"]),
        }])
        annotated_crop = plan_detector.draw_detections_on_image(
            raw_crop, relative,
            line_width=8, font_size=42, pad=10,
        )

        zcols = st.columns(2)
        with zcols[0]:
            st.image(
                raw_crop,
                caption=f"Raw plan crop · {raw_crop.size[0]}×{raw_crop.size[1]} px",
                use_container_width=True,
            )
        with zcols[1]:
            st.image(
                annotated_crop,
                caption=(
                    f"Tesseract bbox for '{det['mark']}' · "
                    f"conf={float(det['confidence']):.0f}"
                ),
                use_container_width=True,
            )
        st.caption(
            "**Sanity check.** If the right-hand box wraps a tag that says "
            f"'{det['mark']}', alignment is correct. If it wraps unrelated "
            "text (a callout, a wall label, a dimension), Tesseract had a "
            "false positive — that's an OCR-quality issue, not a coordinate bug."
        )

    # ---------- Per-mark cross-floor pivot
    st.markdown("### Per-mark counts across all floors")
    pivot = (
        plan_df.groupby(["mark", "source_sheet"], as_index=False)
        .size()
        .pivot(index="mark", columns="source_sheet", values="size")
        .fillna(0)
        .astype(int)
    )
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.reset_index().sort_values("total", ascending=False)
    st.dataframe(pivot, hide_index=True, use_container_width=True)

    # ---------- Optional VLM comparison per floor
    st.markdown("---")
    st.markdown("### 🧩  Tiled multi-model detection (recommended)")
    st.caption(
        "**This is the preferred plan-detection path.** Whole-page VLM "
        "submission fails because tags become illegible after image "
        "downscaling — models then hallucinate plausible marks at made-up "
        "positions. The tiled pipeline crops each plan region into a 2×3 "
        "grid of overlapping tiles and sends each tile to every model at "
        "full resolution, then merges with per-mark NMS and a schedule-vocab "
        "filter. Currently configured for: " + ", ".join(plan_tiling.PLAN_REGIONS)
        + "."
    )

    if chosen_sheet not in plan_tiling.PLAN_REGIONS:
        st.info(
            f"Tiled detection isn't configured for **{chosen_sheet}** yet. "
            f"Pick one of: {', '.join(plan_tiling.PLAN_REGIONS)}."
        )
    else:
        regions = plan_tiling.regions_for_sheet(chosen_sheet)
        n_cols, n_rows = plan_tiling.tile_config_for_sheet(chosen_sheet)
        # Gemini's API blocks "User location is not supported" from this region —
        # skip it for the tiled pipeline. Schedule comparison + whole-page
        # expander still call all three models, so when this user routes around
        # the geographic block (VPN / Vertex AI / proxy) the rest still works.
        TILE_MODELS = {
            k: v for k, v in model_comparison.MODELS.items() if k != "gemini"
        }
        n_tiles_total = len(regions) * n_cols * n_rows
        n_models = len(TILE_MODELS)
        n_calls = n_tiles_total * n_models
        st.caption(
            f"This sheet has **{len(regions)} region(s)** "
            f"({', '.join(r[0] for r in regions)}), tiled "
            f"**{n_cols}×{n_rows}**. Pipeline will run "
            f"{n_tiles_total} tiles × {n_models} models = "
            f"**{n_calls} API calls** total. "
            f"Gemini is **disabled** (geographic API restriction)."
        )

        tile_runs_key = f"plan_tile_runs__{chosen_sheet}"
        cols = st.columns([2, 1, 1])
        with cols[0]:
            tile_clicked = st.button(
                f"▶  Run tiled multi-model detection on {chosen_sheet}",
                type="primary",
                key=f"tile_run_{chosen_sheet}",
                use_container_width=True,
            )
        with cols[1]:
            tile_force = st.checkbox(
                "Skip cache",
                value=False,
                key=f"tile_skip_{chosen_sheet}",
            )
        with cols[2]:
            if st.button(
                "Clear tiled results",
                key=f"tile_clear_{chosen_sheet}",
                use_container_width=True,
            ):
                st.session_state.pop(tile_runs_key, None)

        if tile_clicked:
            # Render the full page once at PLAN_DPI (cached on disk).
            with st.spinner(f"Rendering {chosen_sheet} @{plan_detector.PLAN_DPI} DPI…"):
                full_page = pdf_utils.render_page(
                    CONSTRUCTION_PDF, page_index, dpi=plan_detector.PLAN_DPI
                )

            # results: {region_label: {"region_img": PIL, "model_dfs": {model_key: pixel_df}}}
            results: dict[str, dict] = {}
            total_tile_jobs = len(regions) * n_cols * n_rows * len(TILE_MODELS)
            progress = st.progress(0.0, text="Starting tile pipeline…")
            done = 0
            sorted_vocab = sorted(vocab)

            for region_label, region_bbox in regions:
                region_img = plan_tiling.crop_region(full_page, region_bbox)
                tiles = plan_tiling.tile_image(
                    region_img, n_cols=n_cols, n_rows=n_rows, overlap=0.15
                )
                # accumulator: {model_key: list[pixel_df rows in region coords]}
                accum: dict[str, list[pd.DataFrame]] = {
                    k: [] for k in TILE_MODELS
                }

                for tile in tiles:
                    # Convert tile image to PNG bytes (no resize — full DPI).
                    buf = io.BytesIO()
                    tile.image.save(buf, format="PNG")
                    tile_bytes = buf.getvalue()

                    for k, spec in TILE_MODELS.items():
                        progress.progress(
                            done / max(1, total_tile_jobs),
                            text=(
                                f"{region_label} · tile {tile.label} · "
                                f"{spec.label} ({done}/{total_tile_jobs})"
                            ),
                        )
                        run = model_comparison.run_plan_tile_model(
                            k, tile_bytes,
                            vocab=sorted_vocab if chosen_sheet != "A7.1.1" else None,
                            use_cache=not tile_force,
                            sheet_no=chosen_sheet,
                        )
                        if run.error:
                            st.toast(
                                f"{spec.label} · {region_label} · "
                                f"{tile.label}: {run.error}",
                                icon="⚠️",
                            )
                            done += 1
                            continue
                        # Tile-pixel coords → region-pixel coords.
                        tile_px = model_comparison.tile_detections_as_pixel_df(
                            run.df,
                            source_sheet=chosen_sheet,
                            source_page=chosen_page,
                            tile_width=tile.width,
                            tile_height=tile.height,
                            region_label=region_label,
                        )
                        shifted = plan_tiling.shift_pixel_detections(
                            tile_px, tile.offset_x, tile.offset_y,
                        )
                        accum[k].append(shifted)
                        done += 1

                # Per-model merge + (per-sheet) mark filter + NMS.
                # A2.x uses the schedule vocab. A7.1.1 carries SIGNAGE codes
                # (``1.11`` / ``7.15`` / ...) that don't intersect the door
                # schedule vocab — filter by mark format instead.
                model_dfs: dict[str, pd.DataFrame] = {}
                for k, frames in accum.items():
                    if not frames:
                        model_dfs[k] = plan_detector._empty_plan_df()
                        continue
                    merged = pd.concat(frames, ignore_index=True)
                    if chosen_sheet == "A7.1.1":
                        merged = plan_tiling.filter_by_mark_pattern(
                            merged, r"\d+\.\d+",
                        )
                    else:
                        merged = plan_tiling.filter_by_vocab(merged, vocab)
                    merged = plan_tiling.nms_per_mark(merged, iou_threshold=0.3)
                    model_dfs[k] = merged

                results[region_label] = {
                    "region_img": region_img,
                    "model_dfs": model_dfs,
                }

            progress.progress(1.0, text="Done.")
            st.session_state[tile_runs_key] = results

        tile_results = st.session_state.get(tile_runs_key)
        if tile_results:
            for region_label, payload in tile_results.items():
                region_img: _PILImage.Image = payload["region_img"]
                model_dfs: dict[str, pd.DataFrame] = payload["model_dfs"]

                st.markdown(f"#### {region_label}")
                # Aggregate: union of detections across all models (for the
                # combined view + counting).
                all_models_concat = pd.concat(
                    [df.assign(model_key=k) for k, df in model_dfs.items() if not df.empty],
                    ignore_index=True,
                ) if any(not df.empty for df in model_dfs.values()) else pd.DataFrame()

                n_per_model = {
                    model_comparison.MODELS[k].label: len(df)
                    for k, df in model_dfs.items()
                }
                metric_cols = st.columns(len(n_per_model) + 1)
                for (label, n), mc in zip(n_per_model.items(), metric_cols):
                    mc.metric(label, n)
                metric_cols[-1].metric(
                    "All models combined",
                    len(all_models_concat) if not all_models_concat.empty else 0,
                )

                # Side-by-side annotated region images, one per model.
                # Render at a reduced DPI scale for browser display (boxes are
                # in PLAN_DPI region coords, so scale them down here).
                disp_scale = PLAN_DISPLAY_DPI / plan_detector.PLAN_DPI
                disp_region = region_img.resize(
                    (
                        max(1, int(region_img.width * disp_scale)),
                        max(1, int(region_img.height * disp_scale)),
                    ),
                    _PILImage.LANCZOS,
                )

                model_cols = st.columns(len(model_dfs))
                for col, (k, df) in zip(model_cols, model_dfs.items()):
                    with col:
                        spec = model_comparison.MODELS[k]
                        st.markdown(f"**{spec.label}**")
                        if df.empty:
                            st.info("No vocab-matching detections.")
                            st.image(
                                disp_region,
                                caption=f"{region_label} (no overlay)",
                                use_container_width=True,
                            )
                            continue
                        # Scale region-pixel coords to display coords.
                        disp_df = df.copy()
                        for c in ("bbox_left", "bbox_top", "bbox_width", "bbox_height"):
                            if c in disp_df.columns:
                                disp_df[c] = (
                                    disp_df[c].astype(float) * disp_scale
                                ).round().astype(int)
                        annotated = plan_detector.draw_detections_on_image(
                            disp_region, disp_df,
                            line_width=4, font_size=16, pad=8,
                        )
                        st.image(
                            annotated,
                            caption=(
                                f"{spec.label} · {len(df)} detections "
                                f"on {region_label}"
                            ),
                            use_container_width=True,
                        )
                        with st.expander(f"{spec.label} — detection table"):
                            display_cols = [
                                "mark", "bbox_left", "bbox_top",
                                "bbox_width", "bbox_height", "confidence",
                            ]
                            show_cols = [c for c in display_cols if c in df.columns]
                            st.dataframe(
                                df[show_cols].sort_values("mark").reset_index(drop=True),
                                hide_index=True,
                                use_container_width=True,
                            )

                # Cross-model agreement: which marks did all 3 vs 2 vs 1 model
                # find in this region?
                if not all_models_concat.empty:
                    agreement = (
                        all_models_concat.groupby("mark")["model_key"]
                        .nunique()
                        .reset_index(name="models_agreeing")
                        .merge(
                            all_models_concat["mark"].value_counts()
                            .rename_axis("mark")
                            .reset_index(name="total_detections"),
                            on="mark",
                        )
                        .sort_values(
                            ["models_agreeing", "total_detections"],
                            ascending=[False, False],
                        )
                        .reset_index(drop=True)
                    )
                    st.markdown("**Cross-model agreement (this region)**")
                    st.dataframe(
                        agreement, hide_index=True, use_container_width=True
                    )

                st.markdown("---")
        else:
            st.caption(
                f"Click the button above to run the tiled pipeline on "
                f"{chosen_sheet}."
            )

    # ---------- Old whole-page VLM comparison (kept for reference)
    st.markdown("---")
    with st.expander(
        "🧠  Compare with multimodal models on this floor", expanded=False
    ):
        st.caption(
            "Sends a downscaled version of the plan image to OpenAI gpt-4o, "
            "Anthropic Claude Sonnet 4.6, and Google Gemini 2.5 Flash and asks "
            "each one to enumerate every door tag with a normalized bbox. The "
            "schedule vocabulary is sent along to bias each model's parsing. "
            "Per-floor results are cached on disk under "
            "`outputs/.cache/llm/` — repeated clicks of this button are free."
        )

        runs_key = "plan_vlm_runs"
        all_runs: dict = st.session_state.get(runs_key, {})

        cols = st.columns([2, 1])
        with cols[0]:
            vlm_clicked = st.button(
                f"▶  Run all VLMs on {chosen_sheet}",
                type="primary",
                key=f"vlm_run_{chosen_sheet}",
                use_container_width=True,
            )
        with cols[1]:
            force_refresh = st.checkbox(
                "Skip cache",
                value=False,
                key=f"vlm_skip_{chosen_sheet}",
            )

        if vlm_clicked:
            with st.spinner(f"Rendering {chosen_sheet} for VLM submission…"):
                img_bytes, img_size = _plan_image_bytes_for_vlm(page_index)
            this_runs: dict[str, model_comparison.ModelRun] = {}
            for k, spec in model_comparison.MODELS.items():
                with st.spinner(f"Calling {spec.label}…"):
                    run = model_comparison.run_plan_detection_model(
                        k, img_bytes, vocab=sorted(vocab),
                        use_cache=not force_refresh,
                    )
                    this_runs[k] = run
                    if run.error:
                        st.toast(f"{spec.label}: {run.error}", icon="⚠️")
                    else:
                        badge = "cached" if run.cached else f"{run.elapsed_s:.1f}s"
                        st.toast(
                            f"{spec.label}: {len(run.df)} dets ({badge})",
                            icon="✅",
                        )
            all_runs[chosen_sheet] = {
                "runs": this_runs,
                "img_size": img_size,
                "img_bytes": img_bytes,
            }
            st.session_state[runs_key] = all_runs

        sheet_payload = all_runs.get(chosen_sheet)
        if sheet_payload:
            this_runs = sheet_payload["runs"]
            img_size = sheet_payload["img_size"]
            img_bytes = sheet_payload["img_bytes"]
            base_img = _PILImage.open(io.BytesIO(img_bytes)).convert("RGB")

            cols = st.columns(len(this_runs))
            for col, (k, run) in zip(cols, this_runs.items()):
                with col:
                    st.markdown(f"**{run.spec.label}**")
                    if run.error:
                        st.error(run.error)
                        continue
                    st.caption(
                        f"{len(run.df)} detections · "
                        f"{'cached' if run.cached else f'{run.elapsed_s:.1f}s'}"
                    )
                    if run.df.empty:
                        st.info("No detections.")
                        continue
                    pixel_df = model_comparison.plan_detections_as_pixel_df(
                        run.df,
                        source_sheet=chosen_sheet,
                        source_page=chosen_page,
                        image_width=img_size[0],
                        image_height=img_size[1],
                    )
                    annotated_vlm = plan_detector.draw_detections_on_image(
                        base_img, pixel_df
                    )
                    st.image(
                        annotated_vlm,
                        caption=f"{run.spec.label} on {chosen_sheet}",
                        use_container_width=True,
                    )
                    with st.expander("Detection table"):
                        st.dataframe(
                            run.df, hide_index=True, use_container_width=True
                        )
        else:
            st.caption(
                "Click the button above to run VLM detection on this floor."
            )

    return plan_df


# ============================================================ Tab 3: Takeoff


def tab_takeoff(
    schedule_df: pd.DataFrame, plan_df: pd.DataFrame
) -> None:
    """Takeoff tab — schedule × plan reconciliation + CSV export."""
    st.subheader("Door takeoff — schedule × plan reconciliation")

    if schedule_df is None or schedule_df.empty:
        st.warning("Load the schedule first (Tab 1).")
        return

    reconciled = matching.reconcile(schedule_df, plan_df)
    metrics = evaluation.compute_metrics(schedule_df, plan_df, reconciled)

    cols = st.columns(5)
    cols[0].metric("In schedule", metrics["items_in_schedule"])
    cols[1].metric("On plans", metrics["items_on_plan"])
    cols[2].metric("Matched", metrics["items_matched"])
    cols[3].metric("Orphan", metrics["items_orphan"])
    cols[4].metric("Stray", metrics["items_stray"])

    st.caption(
        f"**Plan detections (total occurrences):** {metrics['plan_detections']} · "
        f"**Needs review:** {metrics['pct_needs_review']:.1f}%"
    )

    if plan_df is None or plan_df.empty:
        st.info(
            "Plan-side detections are empty (Phase 2 work). The reconciliation below "
            "shows every schedule mark as **orphan** until plan detection runs."
        )

    st.markdown("### Reconciliation table")
    st.caption(
        "**status** legend: "
        "*matched* (in schedule + on plan) · "
        "*orphan* (in schedule only) · "
        "*stray* (on plan only) · "
        "*mismatch* (count differs from expected)"
    )
    st.dataframe(
        reconciled,
        hide_index=True,
        use_container_width=True,
        height=420,
    )

    csv_bytes = reconciled.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️  Download takeoff CSV",
        data=csv_bytes,
        file_name="takeoff.csv",
        mime="text/csv",
        type="primary",
    )
    if st.button("Save to outputs/takeoff.csv"):
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        reconciled.to_csv(OUTPUT_CSV, index=False)
        st.success(f"Saved to {OUTPUT_CSV}")


# ============================================================ sidebar / main


def _sidebar(index_df: pd.DataFrame, index_source: str) -> None:
    st.sidebar.title("Door Takeoff MVP")
    st.sidebar.caption("Stanford CEE329 × Strategic Building Innovation")

    health = _file_health()
    for label, ok, detail in health:
        prefix = "✅" if ok else "❌"
        st.sidebar.markdown(f"{prefix} {label}: `{detail}`")
    st.sidebar.caption(
        f"Index loaded from **{index_source}** ({len(index_df)} rows; "
        f"{int(index_df['in_main_set'].sum())} in main set)"
    )

    st.sidebar.divider()
    rel = relevant_sheets(index_df)
    st.sidebar.markdown("**In-scope sheets**")
    for _, r in rel.iterrows():
        st.sidebar.write(f"- `{r['sheet_no']}` — p.{r['pdf_page']}")

    st.sidebar.divider()
    if st.sidebar.button("Clear session state", use_container_width=True):
        for k in [
            "schedule_df", "model_runs", "review_df",
            "sampled_marks", "accuracy_summary",
            "plan_tag_ocr_df", "plan_vlm_runs",
        ]:
            st.session_state.pop(k, None)
        for tk in [k for k in st.session_state if k.startswith("plan_tile_runs__")]:
            st.session_state.pop(tk, None)
        st.toast("Cleared.")


def main() -> None:
    st.set_page_config(
        page_title="Door Takeoff MVP",
        page_icon="🚪",
        layout="wide",
    )
    _load_env()

    health = _file_health()
    if not all(ok for _, ok, _ in health):
        _render_health_screen(health)

    index_df, source = _load_index_cached(str(DATA_DIR))
    if index_df.empty:
        st.error("Could not load Drawing Index. Place it under data/.")
        st.stop()

    _sidebar(index_df, source)

    st.title("AI for 2D Drawing Takeoff — Door scope")
    st.warning(
        "**OCR is approximate — verify all extractions before exporting.** "
        "All schedule pages are image-based; tesseract OCR runs at 200 DPI. "
        "Multimodal model output is also subject to verification.",
        icon="⚠️",
    )

    tabs = st.tabs(
        ["📋  Schedule (A10.2)", "🗺️  Plans (A2.1–A2.4)", "✅  Takeoff"]
    )

    with tabs[0]:
        schedule_df = tab_schedule(index_df)
    with tabs[1]:
        plan_df = tab_plans(index_df)
    with tabs[2]:
        tab_takeoff(schedule_df, plan_df)


if __name__ == "__main__":
    main()
