"""Model Advisor — FastAPI application."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.db import init_db
from backend.models.schemas import HealthResponse
from backend.services.refresh import refresh_all


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Always seed canonical models on startup so the DB is queryable from t=0.
    from backend.services.sources import seed
    from backend.services.refresh import purge_inactive_sources
    from backend.db import connect, backfill_param_counts
    await seed.fetch_and_store()
    # Drop stale source_runs rows for sources not currently registered (e.g. AA without key)
    purge_inactive_sources()
    # Heal legacy rows: populate total_params_b for stub models inserted before
    # ensure_model_stub knew how to parse param counts.
    with connect() as conn:
        n = backfill_param_counts(conn)
        if n > 0:
            print(f"[startup] backfilled param counts for {n} models")
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Local AI model advisor for macOS — scan hardware, discover models, get recommendations.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from backend.routers import scan, models, recommend, refresh, meta, images

app.include_router(scan.router, prefix="/api", tags=["scan"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(recommend.router, prefix="/api", tags=["recommend"])
app.include_router(refresh.router, prefix="/api", tags=["refresh"])
app.include_router(meta.router, prefix="/api", tags=["meta"])
app.include_router(images.router, prefix="/api", tags=["images"])


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", app=settings.app_name, version=settings.app_version)


# Serve frontend bundle at / when packaged. In dev, Vite proxies /api to us.
# We mount the asset directory directly and add an SPA-fallback route so
# client-side router paths like /wizard/coding or /results?... resolve correctly
# on direct navigation / refresh.
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        # API routes are matched first by route ordering; this only fires for
        # un-matched paths, which we route to the SPA shell.
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        # Serve specific static files at the root (favicon, etc.) if they exist.
        candidate = STATIC_DIR / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")
