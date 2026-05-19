# Door Takeoff MVP

A focused Streamlit MVP for the **Stanford CEE329** project, in collaboration
with **Strategic Building Innovation (SBI)**. The app demonstrates AI-assisted
**door takeoff** from 2D construction drawings using a human-in-the-loop
workflow:

- **Schedule side** (sheet A10.2) — extract every door's spec (mark, type,
  size, material, fire rating, hardware, notes) using Tesseract OCR plus
  three multimodal models (OpenAI gpt-4o, Anthropic Claude Sonnet 4.6,
  Google Gemini 2.5 Flash) running in parallel for comparison.
- **Plan side** (sheets A2.1–A2.4) — eventually count occurrences of each
  door tag on the floor plans (Phase 2 work; the current build renders the
  plans and runs baseline OCR).
- **Takeoff** — reconcile schedule vs plan per mark and export the result
  as CSV. Each mark is tagged **matched / orphan / stray / mismatch**.

The dataset is the real construction package for **2067 University Avenue**
(Berkeley), a mixed-use apartment project. PDFs are *image-based* (no
selectable text) — Tesseract is required.

> This project is a focused fork of the broader
> [`cee329-takeoff-mvp`](https://github.com/EElaineYIN/cee329-takeoff-mvp)
> demo. Glazing and area-table scopes have been removed in favour of going
> deep on doors.

---

## What's real vs heuristic vs placeholder

| Capability | Status |
|---|---|
| Index parsing from `Drawing Index.xlsx` | **Real** (deterministic) |
| Schedule sheet selection (A10.2) | **Real** |
| PDF rendering (PyMuPDF @200 DPI) | **Real** |
| Tesseract OCR + row clustering | **Real** (noisy — captioned as approximate) |
| Multimodal extraction (3 models) | **Real** — needs API keys |
| Sampling-based human accuracy grading | **Real** |
| Plan-side door detection (A2.1–A2.4) | **Phase 2 — placeholder for now** |
| Schedule × plan reconciliation | **Real** (over whatever plan_df contains) |
| CSV export | **Real** |

---

## Quickstart

```bash
# 1. Install Tesseract
brew install tesseract

# 2. Create venv and install Python deps
cd ~/door-takeoff-mvp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Make sure source files are in ./data/
ls data/
# expected:
#   1522-CONSTRUCTION SET-11.19.18.pdf
#   Drawing Index.xlsx
#   (Drawing Index.pdf and Exhibit C are optional for this MVP)

# 4. (Optional) pre-warm OCR cache for the in-scope pages (A10.2 + A2.1–A2.4)
python -m src.pdf_utils prewarm

# 5. (Optional) configure multi-model API keys
cp .env.example .env
# edit .env and paste your keys

# 6. Launch
streamlit run app.py
```

The first OCR pass per page takes ~5–15 seconds and writes a disk cache to
`outputs/.cache/`. Subsequent runs are near-instant.

---

## Multi-model comparison

Tab 1 sends the same A10.2 left-third crop to four extractors and lets you
grade a random sample of door marks for per-model accuracy.

| Provider | Env var | Get a key |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| Anthropic | `ANTHROPIC_API_KEY` | https://console.anthropic.com/settings/keys |
| Google | `GOOGLE_API_KEY` | https://aistudio.google.com/app/apikey |

`.env` is in `.gitignore` — never committed. The app reads keys via
`python-dotenv` at startup. Responses are cached on disk under
`outputs/.cache/llm/` keyed by `(model, image_sha256, prompt_version)` so
repeated clicks of *Run all models* are free.

A typical comparison run on A10.2 costs **< $0.10** across all three models.

Workflow:

1. **Run all models on A10.2** → 3 API calls (~5–15 s total).
2. The four side-by-side tables show what each model extracted.
3. **Sample door marks** → randomly draws *N* marks from the union of all
   model outputs and builds a long review table (one row per mark × model).
4. Grade each row *Correct / Wrong / Partial / Skip*. Rows where a model
   didn't extract that mark show `present = False` and count against recall.
5. **Compute accuracy** → summary table with **Recall**, **Row accuracy**
   (Correct + ½ Partial / Graded), **Review rate**, **Count vs expected**.

---

## Project structure

```
~/door-takeoff-mvp/
├── app.py                         # 3-tab Streamlit UI
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── data/
│   ├── 1522-CONSTRUCTION SET-11.19.18.pdf      (~290 MB; gitignored)
│   ├── Drawing Index.xlsx                       (gitignored)
│   └── …
├── outputs/
│   ├── takeoff.csv                             (written on Save)
│   └── .cache/                                  (auto-generated; gitignored)
└── src/
    ├── __init__.py
    ├── pdf_utils.py            # render + Tesseract OCR + disk cache
    ├── index_parser.py         # parse the xlsx → DataFrame
    ├── sheet_selector.py       # SCHEDULE_SHEET + plan-sheet patterns
    ├── schedule_extractor.py   # door-schedule extractor over OCR
    ├── matching.py             # confidence flags + reconcile()
    ├── evaluation.py           # door takeoff metrics
    ├── model_comparison.py     # OpenAI / Anthropic / Gemini wrapper
    └── sampling_eval.py        # per-mark sampling + accuracy
```

---

## Roadmap

- **Phase 2 — plan-side detection (the centerpiece of this fork)**
  - `src/plan_detector.py`: render A2.1–A2.4, run Tesseract OCR, regex-match
    door marks from the schedule, return `(mark, bbox, confidence)` per page.
  - Extend `model_comparison.py` with a plan-detection prompt that asks
    each multimodal model to enumerate door tags with bounding boxes.
  - Tab 2: floor selector + side-by-side annotated images per model +
    per-floor accuracy sampling.
- **Phase 3 — full reconciliation**
  - Tab 3 grows beyond Phase 1 placeholder: per-mark schedule spec +
    plan_count from each detector + status (matched / orphan / stray /
    mismatch) and CSV export with provenance.

---

## Limitations

- **Image-based PDFs** — expect OCR noise on small text and dense tables.
  The app shows OCR confidence per row and flags anything below 60.
- **Plan-side detection is not yet wired up.** Tab 2 renders plan images
  and OCR but does not yet detect door tags. Tab 3's reconciliation uses
  an empty `plan_df` until Phase 2 lands, so every schedule mark currently
  shows as **orphan**.
- **Single project**. Hard-coded against the 2067 University Avenue dataset.

---

## Credits

Built for **Stanford CEE329 (Spring 2026)** in collaboration with
**Strategic Building Innovation (SBI)**. Source drawings from the 2067
University Avenue mixed-use apartment project (Berkeley).
