"""FastAPI backend for the Door Takeoff demo.

Pipeline:  PDF/Image -> render pages -> YOLO detect -> (optional) Claude classify
            -> human review in browser -> export CSV

Run from repo root:
    .venv/bin/uvicorn demo.backend.main:app --reload --port 8000
"""
from __future__ import annotations

import csv
import io
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)

from demo.backend import pdf_render, yolo_infer, vlm_classify, schedule  # noqa: E402

CACHE = ROOT / "demo" / ".cache"
CACHE.mkdir(parents=True, exist_ok=True)
FRONTEND = ROOT / "demo" / "frontend"

app = FastAPI(title="Door Takeoff Demo")

# in-memory job store: job_id -> {dir, pages:[...]}
JOBS: dict[str, dict] = {}


class DetectReq(BaseModel):
    job_id: str
    page: int
    conf: float = 0.25
    region: list[float] | None = None   # [x1,y1,x2,y2] in image px; detect only here


class ClassifyReq(BaseModel):
    job_id: str
    page: int
    boxes: list[dict]


class ExportRow(BaseModel):
    page: str
    count: int


class ExportReq(BaseModel):
    rows: list[dict]


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    job_id = uuid.uuid4().hex[:12]
    job_dir = CACHE / job_id
    name = (file.filename or "upload").lower()
    is_pdf = name.endswith(".pdf")
    if is_pdf:
        pages = pdf_render.render_pdf(data, job_dir)
    elif name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
        pages = pdf_render.render_image(data, job_dir)
    else:
        raise HTTPException(400, "Upload a PDF or an image file.")
    JOBS[job_id] = {
        "dir": job_dir, "pages": pages, "filename": file.filename,
        "pdf_bytes": data if is_pdf else None, "is_pdf": is_pdf,
    }
    return {"job_id": job_id, "filename": file.filename, "pages": pages, "is_pdf": is_pdf}


@app.get("/api/image/{job_id}/{page}")
def image(job_id: str, page: int):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    p = next((x for x in job["pages"] if x["index"] == page), None)
    if not p:
        raise HTTPException(404, "page not found")
    return FileResponse(job["dir"] / p["file"], media_type="image/png")


@app.post("/api/detect")
def detect(req: DetectReq):
    job = JOBS.get(req.job_id)
    if not job:
        raise HTTPException(404, "job not found")
    p = next((x for x in job["pages"] if x["index"] == req.page), None)
    if not p:
        raise HTTPException(404, "page not found")
    boxes = yolo_infer.detect(str(job["dir"] / p["file"]), conf=req.conf, region=req.region)
    return {"page": req.page, "count": len(boxes), "boxes": boxes}


@app.post("/api/classify")
def classify(req: ClassifyReq):
    job = JOBS.get(req.job_id)
    if not job:
        raise HTTPException(404, "job not found")
    p = next((x for x in job["pages"] if x["index"] == req.page), None)
    if not p:
        raise HTTPException(404, "page not found")
    results = vlm_classify.classify(str(job["dir"] / p["file"]), req.boxes)
    return {"results": results}


class ScanReq(BaseModel):
    job_id: str
    page: int | None = None   # optional manual override of the schedule page


@app.post("/api/scan_schedule")
def scan_schedule(req: ScanReq):
    job = JOBS.get(req.job_id)
    if not job:
        raise HTTPException(404, "job not found")
    page_idx = req.page
    if page_idx is None and job.get("pdf_bytes"):
        page_idx = schedule.find_schedule_page(job["pdf_bytes"])
    if page_idx is None:
        return {"found": False, "schedule_page": None, "doors": [], "note": ""}
    p = next((x for x in job["pages"] if x["index"] == page_idx), None)
    if not p:
        return {"found": False, "schedule_page": None, "doors": [], "note": ""}
    result = schedule.read_schedule(str(job["dir"] / p["file"]))
    job["schedule"] = result.get("doors", [])
    return {
        "found": True,
        "schedule_page": page_idx,
        "schedule_page_name": p["name"],
        "doors": result.get("doors", []),
        "note": result.get("note", ""),
        "error": result.get("error"),
    }


class ReconcileReq(BaseModel):
    job_id: str
    detected: list[dict]      # [{type:...}, ...]


@app.post("/api/reconcile")
def reconcile(req: ReconcileReq):
    job = JOBS.get(req.job_id)
    if not job:
        raise HTTPException(404, "job not found")
    sched = job.get("schedule", [])
    return schedule.reconcile(req.detected, sched)


@app.post("/api/export")
def export(req: ExportReq):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["sheet", "door_count", "swing", "double", "sliding", "pocket", "other"])
    total = 0
    for r in req.rows:
        total += int(r.get("door_count", 0))
        w.writerow([
            r.get("sheet", ""), r.get("door_count", 0),
            r.get("swing", 0), r.get("double", 0), r.get("sliding", 0),
            r.get("pocket", 0), r.get("other", 0),
        ])
    w.writerow([])
    w.writerow(["TOTAL", total])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=takeoff.csv"},
    )


# serve frontend at /
app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
