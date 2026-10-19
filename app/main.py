"""
FastAPI backend for the IHSG Opening Estimator.

    uvicorn app.main:app --reload --port 8000
    open http://localhost:8000/docs   (interactive API documentation)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .service import ArtifactsMissing, Store

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["store"] = Store()
    except ArtifactsMissing as e:
        STATE["error"] = str(e)
    yield


app = FastAPI(
    title="IHSG Opening Estimator API",
    description="Estimates the IHSG opening value from pre-market signals of Asian indices and "
                "U.S.-listed Indonesia ETFs (Random Forest & XGBoost).",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("IHSG_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
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


@app.post("/api/reload")
def reload():
    """Reload artefacts after copying a newly trained model from Colab."""
    try:
        STATE["store"] = Store()
        STATE.pop("error", None)
    except ArtifactsMissing as e:
        raise HTTPException(503, str(e)) from e
    return {"status": "reloaded", "champion": STATE["store"].champion}
