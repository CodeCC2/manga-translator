from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from PIL import UnidentifiedImageError

from .pipeline import MangaPipeline


app = FastAPI(title="Manga Translator", version="0.1.0")


@app.on_event("startup")
async def startup() -> None:
    app.state.pipeline = MangaPipeline()
    # Print clickable link for easy access
    print("\n" + "="*50)
    print("🚀 Manga Translator Server Started!")
    print("="*50)
    print(f"🌐 Open in browser: http://127.0.0.1:8000")
    print(f"💻 Device: {app.state.pipeline.device.upper()}")
    if app.state.pipeline.device == "cpu":
        print("⚠️  Running on CPU - For faster performance, install PyTorch CUDA:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("="*50 + "\n")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = Path(__file__).resolve().parent.parent / "static" / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Front-end not found.")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/api/process")
async def process_image(
    file: UploadFile = File(...),
    source_lang: str = Query("en", description="Source language: en or ja"),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="กรุณาอัพโหลดไฟล์รูป")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="ไฟล์ต้องเป็นภาพเท่านั้น")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="ไม่พบข้อมูลภาพ")

    pipeline: MangaPipeline = app.state.pipeline
    try:
        result = await run_in_threadpool(lambda: pipeline.process_image(image_bytes, source_lang=source_lang))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="ไฟล์ภาพไม่รองรับ")
    except Exception as exc:  # pragma: no cover - FastAPI will log the stack
        raise HTTPException(status_code=500, detail=f"ประมวลผลไม่สำเร็จ: {exc}") from exc
    return result


@app.post("/api/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    source_lang: str = Query("en", description="Source language: en or ja"),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="กรุณาอัพโหลดไฟล์ PDF")
    if file.content_type and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="ไฟล์ต้องเป็น PDF เท่านั้น")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="ไม่พบข้อมูล PDF")

    pipeline: MangaPipeline = app.state.pipeline
    try:
        result = await run_in_threadpool(lambda: pipeline.process_pdf(pdf_bytes, source_lang=source_lang))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ประมวลผล PDF ไม่สำเร็จ: {exc}") from exc
    return result


@app.get("/health")
async def health() -> dict:
    pipeline: MangaPipeline | None = getattr(app.state, "pipeline", None)
    ready = pipeline is not None
    device = pipeline.device if pipeline else None
    return {"ok": ready, "device": device}


static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")