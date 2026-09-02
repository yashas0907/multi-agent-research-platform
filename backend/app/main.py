"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.database import dispose_engine, init_db
from app.core.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger = get_logger("startup")
    settings = get_settings()
    if settings.APP_ENV != "test":
        await init_db()
    logger.info(
        "api_started",
        env=settings.APP_ENV,
        llm=settings.LLM_PROVIDER,
        search=settings.WEB_SEARCH_PROVIDER,
    )
    yield
    await dispose_engine()
    logger.info("api_stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        description="Autonomous multi-agent research pipeline with source verification and citations",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=get_settings().API_HOST,
        port=get_settings().API_PORT,
        reload=False,
    )
