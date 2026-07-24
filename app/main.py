from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import engine as engine_mod
from .routers import bubbles, clean, health, pipeline
from .settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    detector = engine_mod.create_bubble_detector(settings)
    cleaner = engine_mod.create_clean_service(settings)
    detector.load()
    cleaner.load()
    engine_mod.set_detector(detector)
    engine_mod.set_cleaner(cleaner)
    yield
    engine_mod.set_detector(None)
    engine_mod.set_cleaner(None)


app = FastAPI(
    title="FukiFlow Vision",
    version="0.3.0",
    description="Bubble detect (YOLO-seg) + inset-fill clean.",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(bubbles.router)
app.include_router(clean.router)
app.include_router(pipeline.router)
