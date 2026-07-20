from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import engine as engine_mod
from .engine import VisionEngine
from .routers import detect_ocr, health, inpaint
from .settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    eng = VisionEngine(settings)
    engine_mod.engine = eng
    await eng.prepare()
    yield
    engine_mod.engine = None


app = FastAPI(
    title="FukiFlow Vision",
    version="0.1.0",
    description="Detect+OCR and inpaint for manga pages (translator ↔ editor bridge).",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(detect_ocr.router)
app.include_router(inpaint.router)
