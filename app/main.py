from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import engine as engine_mod
from .routers import bubbles, clean, compose, health, order, pipeline, typeset
from .settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    cleaner = engine_mod.create_clean_service(settings)
    typesetter = engine_mod.create_typeset_service(settings)
    cleaner.load()
    typesetter.load()
    engine_mod.set_cleaner(cleaner)
    engine_mod.set_typeset_service(typesetter)

    if not settings.vision_light:
        detector = engine_mod.create_bubble_detector(settings)
        panel_detector = engine_mod.create_panel_detector(settings)
        order_service = engine_mod.create_order_service(
            settings, panel_detector=panel_detector
        )
        detector.load()
        panel_detector.load()
        order_service.load()
        engine_mod.set_detector(detector)
        engine_mod.set_panel_detector(panel_detector)
        engine_mod.set_order_service(order_service)

    yield

    engine_mod.set_detector(None)
    engine_mod.set_panel_detector(None)
    engine_mod.set_order_service(None)
    engine_mod.set_cleaner(None)
    engine_mod.set_typeset_service(None)


app = FastAPI(
    title="FukiFlow Vision",
    version="0.7.0",
    description=(
        "Bubble detect + panel detect + RSP reading order + inset-fill clean + "
        "typeset + composable layers. Set VISION_LIGHT=1 to skip YOLO."
    ),
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(bubbles.router)
app.include_router(order.router)
app.include_router(clean.router)
app.include_router(pipeline.router)
app.include_router(typeset.router)
app.include_router(compose.router)
