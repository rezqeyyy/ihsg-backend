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

import inspect
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
            csv_frames = core.load_from_csv(data_dir)
        except FileNotFoundError as e:
            raise ArtifactsMissing(f"Price CSV files not found in {self.dir / 'data'}. {e}") from e
        # The CSVs end where the dataset was downloaded. Every day after that is downloaded by the
        # sync job and kept in the database (table `prices`), so a restart does not lose those days.
        self.csv_last = {k: (v.index.max() if len(v) else pd.Timestamp("1900-01-01")) for k, v in csv_frames.items()}
        try:
            stored = load_prices()
        except Exception as e:  # noqa: BLE001 - database unreachable: serve the CSV history only
            stored, self.price_db_error = {}, f"{type(e).__name__}: {e}"
        else:
            self.price_db_error = None
        self.frames = {k: merge_prices(v, stored.get(k), self.csv_last[k]) for k, v in csv_frames.items()}
        self._rebuild()

    def _rebuild(self):
        """Recompute everything derived from self.frames (after loading or after new prices)."""
        self.cleaned, _ = core.clean_all(self.frames)
        self.dataset = core.build_dataset(self.cleaned)
        self._replay_cache: dict[tuple[str, str], dict] = {}

    def _absorb(self, frames: dict):
        """Take freshly downloaded prices: keep every day after the CSV end in the database and use the
        merged history from now on (replay dates, analogs, history). Caller holds self._lock."""
        try:
            save_prices(frames, self.csv_last)
            self.price_db_error = None
        except Exception as e:  # noqa: BLE001 - still use the prices in memory
            self.price_db_error = f"{type(e).__name__}: {e}"
        self.frames = frames
        self._rebuild()

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
            "csv_last": self.csv_last.get("ihsg"),
            "price_db_error": self.price_db_error,
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
        asia_day = p.get("asia_session", {})
        rows = [
            {"time": "04:00", "key": "eido_ret", "label": "EIDO", "venue": "New York close",
             "value": f["eido_ret"], "session": etf_day.get("eido")},
            {"time": "04:00", "key": "idx_ret", "label": "IDX ETF", "venue": "New York close",
             "value": f["idx_ret"], "session": etf_day.get("idx")},
            {"time": "07:00", "key": "nikkei_gap", "label": "Nikkei 225", "venue": "Tokyo open",
             "value": f["nikkei_gap"], "session": asia_day.get("nikkei"), "closed": "Nikkei 225" in closed},
            {"time": "07:00", "key": "kospi_gap", "label": "KOSPI", "venue": "Seoul open",
             "value": f["kospi_gap"], "session": asia_day.get("kospi"), "closed": "KOSPI" in closed},
            {"time": "08:30", "key": "hangseng_gap", "label": "Hang Seng", "venue": "Hong Kong open",
             "value": f["hangseng_gap"], "session": asia_day.get("hangseng"), "closed": "Hang Seng" in closed},
            {"time": "08:30", "key": "shanghai_gap", "label": "Shanghai", "venue": "Shanghai open",
             "value": f["shanghai_gap"], "session": asia_day.get("shanghai"), "closed": "Shanghai Composite" in closed},
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
            # live_fallback exists only in newer versions of ihsg_core.py; pass it when supported
            extra = {"live_fallback": True} if "live_fallback" in inspect.signature(core.predict_for_date).parameters else {}
            self._absorb(frames)
            p = core.predict_for_date(frames, core.next_session_date(), m, self.meta, name, **extra)
            out = self._format(p, "live")
            out["fetched_at"] = datetime.now().isoformat(timespec="seconds")
            self._live_cache[name] = (time.time(), out)
            log_prediction(out, source="live")
            update_actuals(self.cleaned["ihsg"])
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

    # --------------------------------------------------------- history sync
    def sync_history(self, max_days: int = 30) -> dict:
        """Keep the forecast log complete without anyone opening the website:
        1. download the latest prices and fill in the real open of every logged session that has traded;
        2. backfill: an IHSG session of the last `max_days` days that has no logged estimate (the server
           was asleep, the morning job did not run) gets one, built ONLY from data before that session
           (the same leak-free builder as replay mode) and marked source='backfill';
        3. sessions inside the dashboard model's training period are skipped (they would be in-sample)."""
        name, m = self._model(None)
        with self._lock:
            frames = core.refresh_with_yfinance(self.frames)
            self._absorb(frames)
            ihsg = self.cleaned["ihsg"]
            fit_end = pd.Timestamp(core.SPLITS["test" if self.meta.get("deploy_refit") else "val"][1])
            since = max(pd.Timestamp.today().normalize() - pd.Timedelta(days=max_days), fit_end + pd.Timedelta(days=1))
            have = logged_dates(name)
            added, errors = [], {}
            for d in ihsg.index[ihsg.index >= since]:
                key = d.strftime("%Y-%m-%d")
                if key in have:
                    continue
                try:
                    p = core.predict_for_date(frames, d, m, self.meta, name)
                    log_prediction(self._format(p, "replay"), source="backfill")
                    added.append(key)
                except Exception as e:  # noqa: BLE001 - one bad day must not stop the others
                    errors[key] = f"{type(e).__name__}: {e}"
            filled = update_actuals(ihsg)
        recent = ihsg[ihsg.index >= since]
        stale = recent.index[core.flag_open_eq_prev_close(ihsg).reindex(recent.index).fillna(False).astype(bool)]
        return {"backfilled": added, "actuals_filled": filled, "errors": errors,
                "latest_ihsg_session": ihsg.index.max().strftime("%Y-%m-%d"),
                "stale_opens": [d.strftime("%Y-%m-%d") for d in stale],
                "price_db_error": self.price_db_error, "synced_at": _now()}

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
        return {"live_log": log, "live_summary": summarise_log(log), "backtest": backtest,
                "champion": self.meta["champion"], "persistent": _PG}


# ---------------------------------------------------------------------------
# Forecast log
#   IHSG_DATABASE_URL (or DATABASE_URL) = postgresql://... -> PostgreSQL (permanent, e.g. a free Neon or
#   Supabase database). Not set -> SQLite file at IHSG_DB_PATH (fine locally; on Render's free plan the
#   disk is wiped on every redeploy/restart, so the history would be lost there).
# ---------------------------------------------------------------------------
DB_URL = os.environ.get("IHSG_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""
_PG = DB_URL.startswith(("postgres://", "postgresql://"))

_SCHEMA = """CREATE TABLE IF NOT EXISTS predictions (
    target_date TEXT, model TEXT, created_at TEXT, prev_close DOUBLE PRECISION, y_hat DOUBLE PRECISION,
    open_hat DOUBLE PRECISION, lower_open DOUBLE PRECISION, upper_open DOUBLE PRECISION, direction TEXT,
    sentiment DOUBLE PRECISION, actual_open DOUBLE PRECISION, actual_y DOUBLE PRECISION,
    PRIMARY KEY (target_date, model))"""
# columns added after the first version (old databases are upgraded in place)
_EXTRA_COLUMNS = {"source": "TEXT", "actual_flagged": "INTEGER", "updated_at": "TEXT"}


class _Conn:
    """Tiny wrapper so the same SQL ('?' placeholders) runs on SQLite and PostgreSQL."""

    def __init__(self):
        if _PG:
            import psycopg
            self.c = psycopg.connect(DB_URL, autocommit=False)
        else:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.c = sqlite3.connect(DB_PATH)

    def run(self, sql: str, args: tuple = ()):
        cur = self.c.cursor()
        cur.execute(sql.replace("?", "%s") if _PG else sql, args)
        return cur

    def rows(self, sql: str, args: tuple = ()) -> list[dict]:
        cur = self.run(sql, args)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.c.commit()
        else:
            self.c.rollback()
        self.c.close()
        return False


_schema_ready = False


def _db() -> _Conn:
    global _schema_ready
    c = _Conn()
    if not _schema_ready:
        c.run(_SCHEMA)
        if _PG:
            for col, typ in _EXTRA_COLUMNS.items():
                c.run(f"ALTER TABLE predictions ADD COLUMN IF NOT EXISTS {col} {typ}")
        else:
            have = {r[1] for r in c.run("PRAGMA table_info(predictions)").fetchall()}
            for col, typ in _EXTRA_COLUMNS.items():
                if col not in have:
                    c.run(f"ALTER TABLE predictions ADD COLUMN {col} {typ}")
        c.c.commit()
        _schema_ready = True
    return c


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log_prediction(p: dict, source: str = "live"):
    """Save (or refresh) the estimate for one session. A 'live' estimate always wins over a
    'backfill' one; a backfill never overwrites an estimate that was made live."""
    with _db() as c:
        old = c.rows("SELECT source FROM predictions WHERE target_date=? AND model=?", (p["target_date"], p["model"]))
        if old and source == "backfill" and old[0]["source"] != "backfill":
            return
        c.run("""INSERT INTO predictions (target_date, model, created_at, prev_close, y_hat, open_hat,
                 lower_open, upper_open, direction, sentiment, source, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                 ON CONFLICT(target_date, model) DO UPDATE SET created_at=excluded.created_at,
                 prev_close=excluded.prev_close, y_hat=excluded.y_hat, open_hat=excluded.open_hat,
                 lower_open=excluded.lower_open, upper_open=excluded.upper_open,
                 direction=excluded.direction, sentiment=excluded.sentiment, source=excluded.source,
                 updated_at=excluded.updated_at""",
              (p["target_date"], p["model"], _now(), float(p["prev_close"]), float(p["y_hat"]), float(p["open_hat"]),
               float(p["lower_open"]), float(p["upper_open"]), p["direction"], float(p["sentiment"]["score"]),
               source, _now()))


def update_actuals(ihsg: pd.DataFrame) -> int:
    """Fill in the real opening price once the session has traded. A reported open that is practically
    equal to the previous close (<= FLAG_TOL, stale Yahoo data) is stored but marked as flagged, and is
    left out of the error statistics, exactly like the flagged days in configuration A."""
    n = 0
    with _db() as c:
        for r in c.rows("SELECT target_date, model, prev_close FROM predictions WHERE actual_open IS NULL"):
            ts = pd.Timestamp(r["target_date"])
            if ts in ihsg.index and pd.notna(ihsg.at[ts, "Open"]):
                o, pc = float(ihsg.at[ts, "Open"]), float(r["prev_close"])
                flagged = int(abs(o - pc) <= core.FLAG_TOL * pc)
                c.run("UPDATE predictions SET actual_open=?, actual_y=?, actual_flagged=?, updated_at=? "
                      "WHERE target_date=? AND model=?", (o, o / pc - 1, flagged, _now(), r["target_date"], r["model"]))
                n += 1
    return n


def logged_dates(model: str) -> set[str]:
    with _db() as c:
        return {r["target_date"] for r in c.rows("SELECT target_date FROM predictions WHERE model=?", (model,))}


def read_log() -> list[dict]:
    with _db() as c:
        rows = c.rows("SELECT * FROM predictions ORDER BY target_date DESC")
    return clean_json(rows)


def summarise_log(rows: list[dict]) -> dict:
    """Error statistics of the logged estimates whose real open is known and not flagged."""
    done = [r for r in rows if r.get("actual_open") is not None and not r.get("actual_flagged")]
    if not done:
        return {"n_done": 0, "n_pending": sum(r.get("actual_open") is None for r in rows),
                "n_flagged": sum(bool(r.get("actual_flagged")) for r in rows)}
    err = np.array([r["actual_open"] - r["open_hat"] for r in done], float)
    nz = [r for r in done if r["actual_y"] != 0]
    return clean_json({
        "n_done": len(done),
        "n_pending": sum(r.get("actual_open") is None for r in rows),
        "n_flagged": sum(bool(r.get("actual_flagged")) for r in rows),
        "mae_points": float(np.mean(np.abs(err))),
        "mae_pp": float(np.mean([abs(r["actual_y"] - r["y_hat"]) for r in done]) * 100),
        "direction_hit": float(np.mean([np.sign(r["actual_y"]) == np.sign(r["y_hat"]) for r in nz])) if nz else None,
        "inside_95": float(np.mean([r["lower_open"] <= r["actual_open"] <= r["upper_open"] for r in done])),
    })


# ---------------------------------------------------------------------------
# Daily prices after the CSV end (same database as the forecast log)
# ---------------------------------------------------------------------------
_PRICE_SCHEMA = """CREATE TABLE IF NOT EXISTS prices (
    instrument TEXT, date TEXT, open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION,
    close DOUBLE PRECISION, adj_close DOUBLE PRECISION, volume DOUBLE PRECISION, updated_at TEXT,
    PRIMARY KEY (instrument, date))"""
_PRICE_COLS = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v) else float(v)


def save_prices(frames: dict, after: dict) -> int:
    """Upsert every row dated after the CSV end. Rows are re-written on each sync, so a bar that
    Yahoo first published while the market was still open is corrected by the next sync."""
    n = 0
    with _db() as c:
        c.run(_PRICE_SCHEMA)
        for k, df in frames.items():
            new = df[df.index > after.get(k, pd.Timestamp("1900-01-01"))]
            for d, r in new.iterrows():
                c.run("""INSERT INTO prices (instrument, date, open, high, low, close, adj_close, volume, updated_at)
                         VALUES (?,?,?,?,?,?,?,?,?)
                         ON CONFLICT(instrument, date) DO UPDATE SET open=excluded.open, high=excluded.high,
                         low=excluded.low, close=excluded.close, adj_close=excluded.adj_close,
                         volume=excluded.volume, updated_at=excluded.updated_at""",
                      (k, d.strftime("%Y-%m-%d"), *(_num(r.get(col)) for col in _PRICE_COLS), _now()))
                n += 1
    return n


def load_prices() -> dict:
    with _db() as c:
        c.run(_PRICE_SCHEMA)
        rows = c.rows("SELECT * FROM prices ORDER BY instrument, date")
    out = {}
    for k in {r["instrument"] for r in rows}:
        sub = [r for r in rows if r["instrument"] == k]
        df = pd.DataFrame({col: [r[low] for r in sub] for col, low in _PRICE_COLS.items()},
                          index=pd.DatetimeIndex([r["date"] for r in sub]), dtype=float)
        df.index.name = "Date"
        out[k] = df
    return out


def merge_prices(csv: pd.DataFrame, stored: pd.DataFrame | None, csv_last: pd.Timestamp) -> pd.DataFrame:
    """CSV history + stored days after the CSV end (the CSV wins for any date it contains)."""
    if stored is None or stored.empty:
        return csv
    extra = stored[stored.index > csv_last].reindex(columns=csv.columns)
    return pd.concat([csv, extra]).sort_index()