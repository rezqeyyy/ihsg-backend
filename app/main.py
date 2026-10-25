"""
FastAPI backend for the IHSG Opening Estimator.

    uvicorn app.main:app --reload --port 8000
    open http://localhost:8000/docs   (interactive API documentation)
"""
from __future__ import annotations

import logging
import os
import traceback
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .service import ArtifactsMissing, Store

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ihsg")

STATE: dict = {}


def _sync_in_background():
    """On every start (also when Render wakes the server up) bring the forecast log up to date,
    in a background thread so the website does not wait for the price download."""
    import threading

    def job():
        try:
            log.info("history sync: %s", STATE["store"].sync_history())
        except Exception:  # noqa: BLE001 - no internet / Yahoo down: try again next time
            log.warning("history sync failed\n%s", traceback.format_exc())
    if "store" in STATE and os.environ.get("IHSG_SYNC_ON_START", "1") != "0":
        threading.Thread(target=job, daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["store"] = Store()
    except ArtifactsMissing as e:
        STATE["error"] = str(e)
    _sync_in_background()
    yield


app = FastAPI(
    title="IHSG Opening Estimator API",
    description="Estimates the IHSG opening value from pre-market signals of Asian indices and "
                "U.S.-listed Indonesia ETFs (Random Forest & XGBoost ensemble).",
    version="1.0.0",
    lifespan=lifespan,
)


# Unexpected errors -> JSON 500 *inside* the CORS layer. Without this, a crash returns a bare 500
# without CORS headers and the browser only reports "cannot reach the backend", hiding the real error.
# (Registered before CORSMiddleware, so CORS wraps it and still adds its headers.)
@app.middleware("http")
async def catch_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:  # noqa: BLE001
        log.error("Unhandled error on %s\n%s", request.url.path, traceback.format_exc())
        # where it happened: the last frame inside our own code (app/...), plus the very last frame
        frames = traceback.extract_tb(e.__traceback__)
        ours = [f for f in frames if "/app/" in f.filename.replace("\\", "/")]
        where = " <- ".join(f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno} {f.name}" for f in [*ours[-1:], *frames[-1:]])
        return JSONResponse(status_code=500,
                            content={"detail": f"Server error on {request.url.path}: {type(e).__name__}: {e} [{where}]"})


# CORS: which websites may call this API from the browser.
#   IHSG_CORS_ORIGINS      comma-separated origins, default "*" = any website (public read-only API, no cookies)
#   IHSG_CORS_ORIGIN_REGEX optional pattern, e.g. "https://.*\.vercel\.app" for all Vercel preview URLs
_origins = [o.strip().rstrip("/") for o in os.environ.get("IHSG_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=os.environ.get("IHSG_CORS_ORIGIN_REGEX") or None,
    allow_methods=["*"],
    allow_headers=["*"],
)


def store() -> Store:
    if "store" not in STATE:
        raise HTTPException(503, STATE.get("error", "Model artefacts are not loaded."))
    return STATE["store"]


def run(fn, *a, **k):
    try:
        return fn(*a, **k)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except KeyError as e:
        raise HTTPException(404, f"No IHSG session found for {e}") from e


class SimulateRequest(BaseModel):
    date: str | None = Field(None, description="IHSG session (YYYY-MM-DD); default = latest in data")
    model: str | None = None
    overrides: dict[str, float] = Field(
        default_factory=dict,
        description="Pre-market signals as fractions, e.g. {'eido_ret': 0.012, 'nikkei_gap': -0.004}")


@app.get("/", include_in_schema=False)
def root():
    return {"app": "IHSG Opening Estimator API", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health")
def health():
    return {"status": "ok" if "store" in STATE else "no_artifacts", "detail": STATE.get("error")}


@app.get("/api/meta")
def meta():
    return store().info()


@app.get("/api/sessions")
def sessions(limit: int = Query(60, ge=1, le=500)):
    return {"sessions": store().sessions(limit)}


@app.get("/api/estimate")
def estimate(date: str | None = None, model: str | None = None):
    """Replay: estimate for a past (or next) IHSG session using only data known before 09:00 WIB."""
    return run(store().estimate, date, model)


@app.post("/api/estimate/live")
def estimate_live(model: str | None = None, force: bool = False):
    """Fetch the latest prices with yfinance and estimate the next IHSG session (cached 10 minutes)."""
    try:
        return store().live(model, force)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001 - network / data provider problems
        raise HTTPException(502, f"Live data could not be fetched: {e}") from e


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    return run(store().simulate, req.date, req.model, req.overrides)


@app.get("/api/performance")
def performance(config: Literal["A", "B", "C"] = "A"):
    return store().performance(config)


@app.get("/api/data-quality")
def data_quality():
    return store().data_quality()


@app.get("/api/history")
def history():
    return store().history()


@app.post("/api/history/sync")
def history_sync(request: Request, max_days: int = Query(14, ge=1, le=60)):
    """Fill in real opening prices and backfill missed sessions (called by the scheduled job).
    If IHSG_SYNC_TOKEN is set, the request must carry the header  X-Sync-Token: <token>."""
    token = os.environ.get("IHSG_SYNC_TOKEN")
    if token and request.headers.get("x-sync-token") != token:
        raise HTTPException(401, "Invalid or missing X-Sync-Token header.")
    try:
        return store().sync_history(max_days)
    except Exception as e:  # noqa: BLE001 - network / data provider problems
        raise HTTPException(502, f"History sync failed: {e}") from e


@app.post("/api/reload")
def reload():
    """Reload artefacts after copying a newly trained model from Colab."""
    try:
        STATE["store"] = Store()
        STATE.pop("error", None)
    except ArtifactsMissing as e:
        raise HTTPException(503, str(e)) from e
    return {"status": "reloaded", "champion": STATE["store"].champion}