"""
Service layer: everything the API does, without any web framework.

It reads the artefacts written by the Colab notebook (section 7):

    ARTIFACTS_DIR/
        metadata.json
        models/deploy_Ensemble.joblib (+ deploy_RandomForest / deploy_XGBoost)
        reports/*.csv
        data/*.csv          (the 7 price files)

and reuses ihsg_core.py, the same module the notebook used for training, so
features are computed by identical code in training and in the website.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import ihsg_core as core

ARTIFACTS_DIR = Path(os.environ.get("IHSG_ARTIFACTS_DIR", Path(__file__).resolve().parents[1] / "artifacts"))
DB_PATH = Path(os.environ.get("IHSG_DB_PATH", ARTIFACTS_DIR / "predictions.db"))
LIVE_CACHE_SECONDS = int(os.environ.get("IHSG_LIVE_CACHE_SECONDS", "600"))


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def clean_json(obj):
    """Convert numpy / pandas objects to plain JSON (NaN and inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if obj is None or obj is pd.NaT:
        return None
    if isinstance(obj, (pd.Timestamp, datetime, date, np.datetime64)):
        ts = pd.Timestamp(obj)
        return ts.strftime("%Y-%m-%d") if ts == ts.normalize() else ts.isoformat(timespec="seconds")
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    return obj


def frame_records(df: pd.DataFrame | None, index_name: str | None = None) -> list[dict]:
    if df is None:
        return []
    d = df.reset_index() if index_name else df
    if index_name and d.columns[0] != index_name:
        d = d.rename(columns={d.columns[0]: index_name})
    return clean_json(d.to_dict(orient="records"))


class ArtifactsMissing(RuntimeError):
    pass


def check_versions(trained: dict) -> list[str]:
    """Models pickled with joblib should be loaded with the library versions used in Colab."""
    import importlib
    out = []
    for name, v in trained.items():
        try:
            here = importlib.import_module(name).__version__
        except Exception:  # noqa: BLE001
            here = "not installed"
        if here.split(".")[:2] != str(v).split(".")[:2]:
            out.append(f"{name}: trained with {v}, backend has {here}")
    return out


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
class Store:
    """Loads models, data and reports once and serves predictions."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self.dir = Path(artifacts_dir)
        self._lock = threading.Lock()
        self._live_cache: dict[str, tuple[float, dict]] = {}
        self.reload()

    # ------------------------------------------------------------------ load
    def reload(self):
        # tolerate one extra folder level, e.g. artifacts/ihsg_opening_estimator/metadata.json
        # (what you get when extracting the Google Drive download as a folder)
        if not (self.dir / "metadata.json").exists():
            nested = sorted(self.dir.glob("*/metadata.json")) + sorted(self.dir.glob("*/*/metadata.json"))
            if nested:
                self.dir = nested[0].parent
        meta_path = self.dir / "metadata.json"
        if not meta_path.exists():
            raise ArtifactsMissing(
                f"{meta_path} not found. Copy the WHOLE folder 'ihsg_opening_estimator' from Google Drive "
                f"(metadata.json, models/, reports/, data/) into {self.dir} (see README).")
        self.meta = json.loads(meta_path.read_text())
        self.models, self.model_errors = {}, {}
        # By default only the main model (the ensemble) is loaded: it already contains RF and XGBoost,
        # and loading the single models too would triple the memory use (Render free = 512 MB).
        # IHSG_MODELS=all loads every model again (e.g. to compare them on a bigger server).
        names = list(self.meta["tree_models"])
        if os.environ.get("IHSG_MODELS", "champion").lower() != "all" and self.meta.get("champion") in names:
            names = [self.meta["champion"]]
        for name in names:
            try:
                path = self.dir / "models" / f"deploy_{name}.joblib"
                if not path.exists():                      # also accept the files directly in artifacts/
                    path = self.dir / f"deploy_{name}.joblib"
                self.models[name] = joblib.load(path)
            except Exception as e:  # noqa: BLE001 - report but keep serving the other model
                self.model_errors[name] = f"{type(e).__name__}: {e}"
        if not self.models:
            raise ArtifactsMissing(f"No model could be loaded: {self.model_errors}")
        self.version_warnings = check_versions(self.meta.get("versions", {}))
        self.champion = self.meta["champion"] if self.meta["champion"] in self.models else next(iter(self.models))
        data_dir = self.dir / "data" if (self.dir / "data").exists() else self.dir
        try:
            self.frames = core.load_from_csv(data_dir)
        except FileNotFoundError as e:
            raise ArtifactsMissing(f"Price CSV files not found in {self.dir / 'data'}. {e}") from e
        self.cleaned, _ = core.clean_all(self.frames)
        self.dataset = core.build_dataset(self.cleaned)
        self._replay_cache: dict[tuple[str, str], dict] = {}

    def report(self, name: str) -> pd.DataFrame | None:
        p = self.dir / "reports" / name
        return pd.read_csv(p) if p.exists() else None

    def _model(self, name: str | None):
        name = name or self.champion
        if name not in self.models:
            raise ValueError(f"Unknown model '{name}'. Available: {list(self.models)}")
        return name, self.models[name]

    # ------------------------------------------------------------------ meta
    def info(self) -> dict:
        ih = self.cleaned["ihsg"].index
        return clean_json({
            "champion": self.champion,
            "models": list(self.models),
            "model_errors": self.model_errors,
            "version_warnings": self.version_warnings,
            "primary_config": self.meta.get("primary_config", "A"),
            "trained_at": self.meta.get("created_at"),
            "deploy_refit": self.meta.get("deploy_refit"),
            "n_trials": self.meta.get("n_trials"),
            "cv_rmse": self.meta.get("cv_rmse"),
            "intervals": self.meta.get("intervals"),
            "best_params": self.meta.get("best_params"),
            "ensemble": self.meta.get("ensemble"),
            "data_first": ih.min(), "data_last": ih.max(),
            "replay_min": ih[40] if len(ih) > 40 else ih.min(),
            "replay_max": ih.max() + pd.Timedelta(days=7),
            "features": [{"key": f, "label": core.FEATURE_LABELS[f],
                          "group": next(g for g, fs in core.FEATURE_GROUPS.items() if f in fs)}
                         for f in core.FEATURES],
            "premarket_features": core.PREMARKET_FEATURES,
            "instruments": {k: {"ticker": v["ticker"], "label": v["label"], "role": v["role"]}
                            for k, v in core.INSTRUMENTS.items()},
            "disclaimer": core.DISCLAIMER,
        })

    def sessions(self, limit: int = 60) -> list[str]:
        idx = self.cleaned["ihsg"].index[-limit:]
        return [d.strftime("%Y-%m-%d") for d in idx[::-1]]

    # ------------------------------------------------------------- estimate
    def _format(self, p: dict, mode: str) -> dict:
        p = dict(p)
        tab = p["analogs"]["table"]
        p["analogs"] = {**{k: v for k, v in p["analogs"].items() if k != "table"},
                        "rows": frame_records(tab, "date")}
        p["mode"] = mode
        p["shap_ranked"] = sorted(
            [{"key": k, "label": core.FEATURE_LABELS[k], "value": v} for k, v in (p.get("shap") or {}).items()],
            key=lambda r: -abs(r["value"]))
        # timeline of the pre-market morning (WIB) for the hero strip
        p["timeline"] = self._timeline(p)
        return clean_json(p)

    def _timeline(self, p: dict) -> list[dict]:
        f = p["features"]
        closed = {w.split(":")[0] for w in p.get("warnings", [])}
        etf_day = p.get("etf_session", {})
        rows = [
            {"time": "04:00", "key": "eido_ret", "label": "EIDO", "venue": "New York close",
             "value": f["eido_ret"], "session": etf_day.get("eido")},
            {"time": "04:00", "key": "idx_ret", "label": "IDX ETF", "venue": "New York close",
             "value": f["idx_ret"], "session": etf_day.get("idx")},
            {"time": "07:00", "key": "nikkei_gap", "label": "Nikkei 225", "venue": "Tokyo open",
             "value": f["nikkei_gap"], "closed": "Nikkei 225" in closed},
            {"time": "07:00", "key": "kospi_gap", "label": "KOSPI", "venue": "Seoul open",
             "value": f["kospi_gap"], "closed": "KOSPI" in closed},
            {"time": "08:30", "key": "hangseng_gap", "label": "Hang Seng", "venue": "Hong Kong open",
             "value": f["hangseng_gap"], "closed": "Hang Seng" in closed},
            {"time": "08:30", "key": "shanghai_gap", "label": "Shanghai", "venue": "Shanghai open",
             "value": f["shanghai_gap"], "closed": "Shanghai Composite" in closed},
        ]
        return rows

    def estimate(self, target_date: str | None = None, model: str | None = None) -> dict:
        name, m = self._model(model)
        d = pd.Timestamp(target_date).normalize() if target_date else self.cleaned["ihsg"].index.max()
        if d.weekday() >= 5:
            raise ValueError(f"{d.date()} is a weekend; the IDX has no session.")
        if d < self.cleaned["ihsg"].index[40]:
            raise ValueError("Choose a date at least 40 sessions after the start of the data.")
        key = (d.strftime("%Y-%m-%d"), name)
        if key not in self._replay_cache:
            p = core.predict_for_date(self.frames, d, m, self.meta, name)
            out = self._format(p, "replay")
            # The dashboard model is refit on 2012-2025 (or 2012-2023 without refit). A replay date inside
            # that period was seen in training, so the estimate is in-sample and must not be read as a test.
            fit_end = core.SPLITS["test" if self.meta.get("deploy_refit") else "val"][1]
            out["fit_end"] = fit_end
            out["in_sample"] = bool(d <= pd.Timestamp(fit_end))
            self._replay_cache[key] = out
        return self._replay_cache[key]

    def live(self, model: str | None = None, force: bool = False) -> dict:
        """Latest yfinance bars merged with the CSV history -> next IHSG session."""
        name, m = self._model(model)
        hit = self._live_cache.get(name)
        if hit and not force and time.time() - hit[0] < LIVE_CACHE_SECONDS:
            return hit[1]
        with self._lock:
            frames = core.refresh_with_yfinance(self.frames)
            p = core.predict_for_date(frames, core.next_session_date(), m, self.meta, name)
            out = self._format(p, "live")
            out["fetched_at"] = datetime.now().isoformat(timespec="seconds")
            self._live_cache[name] = (time.time(), out)
            log_prediction(out)
            update_actuals(core.clean_all(frames)[0]["ihsg"])
        return out

    # ------------------------------------------------------------- simulate
    def simulate(self, target_date: str | None, model: str | None, overrides: dict[str, float]) -> dict:
        """Re-estimate with some pre-market signals replaced (IHSG history fixed).
        EIDO and IDX move together: a change of eido_ret shifts idx_ret by the same amount
        unless idx_ret is given explicitly."""
        name, m = self._model(model)
        base_p = self.estimate(target_date, name)
        base = pd.Series(base_p["features"], dtype=float)
        x = base.copy()
        bad = [k for k in overrides if k not in core.PREMARKET_FEATURES]
        if bad:
            raise ValueError(f"Only pre-market signals can be simulated, not {bad}")
        if "eido_ret" in overrides and "idx_ret" not in overrides:
            x["idx_ret"] = base["idx_ret"] + (overrides["eido_ret"] - base["eido_ret"])
        for k, v in overrides.items():
            x[k] = float(v)
        sim = core.estimate(m, self.meta, name, x, base_p["prev_close"], with_shap=True)
        grid = np.round(np.linspace(-0.03, 0.03, 41), 6)
        curves = []
        for f in ["eido_ret"] + core.ASIA_FEATURES:
            Xg = pd.DataFrame([x[core.FEATURES]] * len(grid)).astype(float)
            Xg[f] = grid
            if f == "eido_ret":
                Xg["idx_ret"] = grid + (x["idx_ret"] - x["eido_ret"])
            curves.append({"key": f, "label": "EIDO & IDX ETF" if f == "eido_ret" else core.FEATURE_LABELS[f],
                           "x": grid.tolist(), "y": m.predict(Xg).tolist(), "current": float(x[f])})
        sim["shap_ranked"] = sorted(
            [{"key": k, "label": core.FEATURE_LABELS[k], "value": v} for k, v in (sim.get("shap") or {}).items()],
            key=lambda r: -abs(r["value"]))
        return clean_json({"target_date": base_p["target_date"], "model": name,
                           "base": {k: base_p[k] for k in ("y_hat", "open_hat", "lower_open", "upper_open", "direction")},
                           "base_features": base_p["features"], "simulated": sim,
                           "delta_points": sim["open_hat"] - base_p["open_hat"],
                           "delta_return": sim["y_hat"] - base_p["y_hat"], "curves": curves})

    # ---------------------------------------------------------- performance
    def performance(self, config: str = "A") -> dict:
        m, dm, iv = (self.report(n) for n in ("metrics_all_configs.csv", "diebold_mariano.csv",
                                             "prediction_interval_coverage.csv"))
        gi, fi = self.report("feature_group_importance.csv"), self.report("feature_importance.csv")
        tp = self.report("test_predictions_configA.csv")
        configs = sorted(m["config"].unique()) if m is not None else []
        sel = lambda df: df[df["config"] == config].drop(columns="config") if df is not None else None  # noqa: E731
        series = []
        if tp is not None:
            cols = [c for c in tp.columns if c not in ("Date",)]
            series = clean_json(tp[["Date"] + cols].rename(columns={"Date": "date"}).to_dict(orient="records"))
        return {"config": config, "configs": configs, "champion": self.champion,
                "metrics": frame_records(sel(m)), "diebold_mariano": frame_records(sel(dm)),
                "interval_coverage": frame_records(sel(iv)),
                "group_importance": frame_records(gi), "feature_importance": frame_records(fi),
                "split_counts": frame_records(self.report("table_3_4_split_counts.csv")),
                "best_hyperparameters": frame_records(self.report("best_hyperparameters.csv")),
                "test_series": series}

    # --------------------------------------------------------- data quality
    def data_quality(self) -> dict:
        names = {"table_3_2": "table_3_2_flagged_share.csv", "raw_quality": "cleaning_0_raw_quality.csv",
                 "cleaning_log": "cleaning_1_log.csv", "quality_flags": "cleaning_2_quality_flags.csv",
                 "missing_after_alignment": "cleaning_3_missing_after_alignment.csv",
                 "leakage_tests": "leakage_tests.csv", "etf_liquidity": "etf_liquidity.csv",
                 "verification_result": "verification_result.csv",
                 "verification_sample": "verification_sample.csv"}
        out = {}
        for k, n in names.items():
            df = self.report(n)
            if df is not None and df.columns[0].startswith("Unnamed"):
                df = df.rename(columns={df.columns[0]: "item"})
            out[k] = frame_records(df)
        return out

    # -------------------------------------------------------------- history
    def history(self) -> dict:
        log = read_log()
        backtest = []
        tp = self.report("test_predictions_configA.csv")
        if tp is not None and self.meta["champion"] in tp.columns:
            c = self.meta["champion"]
            tp = tp.copy()
            tp["abs_err"] = (tp["y"] - tp[c]).abs()
            tp["hit"] = (np.sign(tp["y"]) == np.sign(tp[c])).astype(float).where(tp["y"] != 0)
            tp["mae_20"] = tp["abs_err"].rolling(20).mean()
            tp["da_20"] = tp["hit"].rolling(20, min_periods=10).mean()
            backtest = clean_json(tp[["Date", "y", c, "mae_20", "da_20"]]
                                  .rename(columns={"Date": "date", c: "y_hat"}).to_dict(orient="records"))
        return {"live_log": log, "backtest": backtest, "champion": self.meta["champion"]}


# ---------------------------------------------------------------------------
# Forecast log (SQLite)
# ---------------------------------------------------------------------------
_SCHEMA = """CREATE TABLE IF NOT EXISTS predictions (
    target_date TEXT, model TEXT, created_at TEXT, prev_close REAL, y_hat REAL, open_hat REAL,
    lower_open REAL, upper_open REAL, direction TEXT, sentiment REAL, actual_open REAL, actual_y REAL,
    PRIMARY KEY (target_date, model))"""


def _db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute(_SCHEMA)
    return c


def log_prediction(p: dict):
    with _db() as c:
        c.execute("""INSERT INTO predictions (target_date, model, created_at, prev_close, y_hat, open_hat,
                     lower_open, upper_open, direction, sentiment) VALUES (?,?,?,?,?,?,?,?,?,?)
                     ON CONFLICT(target_date, model) DO UPDATE SET created_at=excluded.created_at,
                     prev_close=excluded.prev_close, y_hat=excluded.y_hat, open_hat=excluded.open_hat,
                     lower_open=excluded.lower_open, upper_open=excluded.upper_open,
                     direction=excluded.direction, sentiment=excluded.sentiment""",
                  (p["target_date"], p["model"], datetime.now().isoformat(timespec="seconds"), p["prev_close"],
                   p["y_hat"], p["open_hat"], p["lower_open"], p["upper_open"], p["direction"],
                   p["sentiment"]["score"]))


def update_actuals(ihsg: pd.DataFrame):
    with _db() as c:
        for td, pc in c.execute("SELECT target_date, prev_close FROM predictions WHERE actual_open IS NULL").fetchall():
            ts = pd.Timestamp(td)
            if ts in ihsg.index:
                o = float(ihsg.at[ts, "Open"])
                c.execute("UPDATE predictions SET actual_open=?, actual_y=? WHERE target_date=?", (o, o / pc - 1, td))


def read_log() -> list[dict]:
    with _db() as c:
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY target_date DESC", c)
    return clean_json(df.to_dict(orient="records"))