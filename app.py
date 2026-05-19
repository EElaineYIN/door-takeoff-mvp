"""Door Takeoff MVP — Streamlit app.

Run with::

    streamlit run app.py

Three tabs:

1. **Schedule (A10.2)** — door catalog from the schedule sheet, with a
   side-by-side comparison of Tesseract OCR baseline vs three multimodal
   models (OpenAI gpt-4o, Anthropic Claude Sonnet 4.6, Google Gemini 2.5).
2. **Plans (A2.1–A2.4)** — floor-plan sheets where door tags appear on
   the building. Plan-side detection is the next phase; for now the tab
   renders the plan images and surfaces the OCR confidence per page.
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import pdf_utils, evaluation, matching
from src.schedule_extractor import run_schedule_extraction
from src.index_parser import load_index
from src.sheet_selector import (
    SCHEDULE_SHEET,
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
    """Plans tab — floor plans where door tags appear (Phase 2 placeholder)."""
    st.subheader("Floor plans — A2.1 to A2.4")
    st.warning(
        "**Plan-side door detection is the next milestone.** "
        "This tab currently renders each floor plan and runs a baseline OCR. "
        "Phase 2 will add tag-OCR + multimodal-VLM detectors here so each "
        "plan returns a list of door marks with bounding boxes.",
        icon="🚧",
    )

    plans = plan_rows(index_df)
    if plans.empty:
        st.error("No plan sheets (A2.1–A2.4) found in the drawing index.")
        return pd.DataFrame(columns=["mark", "source_sheet"])

    options = plans.apply(
        lambda r: f"{r['sheet_no']} — {r['title'][:80]} (PDF page {r['pdf_page']})",
        axis=1,
    ).tolist()
    chosen = st.selectbox("Pick a floor plan", options, index=0, key="plan_choice")
    chosen_idx = options.index(chosen)
    plan_row = plans.iloc[chosen_idx]
    page_index = int(plan_row["pdf_page"]) - 1

    cols = st.columns([2, 1])
    with cols[0]:
        with st.spinner(f"Rendering {plan_row['sheet_no']}…"):
            png_bytes, size = _render_page_cached(str(CONSTRUCTION_PDF), page_index)
        st.image(
            png_bytes,
            caption=f"{plan_row['sheet_no']} ({size[0]}×{size[1]}px @200dpi)",
            use_container_width=True,
        )
    with cols[1]:
        st.markdown(f"**Sheet:** `{plan_row['sheet_no']}`")
        st.markdown(f"**Title:** {plan_row['title']}")
        st.markdown(f"**PDF page:** {plan_row['pdf_page']}")
        with st.spinner("Running OCR…"):
            ocr = _ocr_page_cached(str(CONSTRUCTION_PDF), page_index, 200, None)
        st.metric("OCR mean confidence", f"{ocr.get('mean_conf', 0.0):.1f} / 100")
        with st.expander("Raw OCR text (truncated)"):
            st.text(ocr.get("text", "")[:5000])

    # Phase 1 placeholder: empty plan_df. Phase 2 will replace this.
    return pd.DataFrame(columns=["mark", "source_sheet"])


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
        ]:
            st.session_state.pop(k, None)
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
