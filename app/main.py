from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import engine as engine_mod
from .routers import bubbles, clean, health, ocr, pipeline
from .settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    detector = engine_mod.create_bubble_detector(settings)
    ocr_svc = engine_mod.create_ocr_service(settings)
    detector.load()
    ocr_svc.load()
    engine_mod.set_detector(detector)
    engine_mod.set_ocr(ocr_svc)
    yield
    engine_mod.set_detector(None)
    engine_mod.set_ocr(None)


app = FastAPI(
    title="FukiFlow Vision",
    version="0.2.0",
    description="Bubble detect (YOLO-seg) + manga-ocr; clean is a passive no-op stub.",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(bubbles.router)
app.include_router(ocr.router)
app.include_router(clean.router)
app.include_router(pipeline.router)
