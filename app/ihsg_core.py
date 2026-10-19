"""
ihsg_core.py - shared functions for the notebook AND the Streamlit dashboard.

Implements the proposal's Chapter III:
  data loading (Table 3.1), cleaning, calendar alignment (3.4.2),
  feature matrix (Table 3.3), target (3.6), configurations A/B/C (3.3.2),
  metrics (3.9), Diebold-Mariano, prediction service (3.10),
  pre-market sentiment score and historical analog search.
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration (Tables 3.1, 3.3, 3.4)
# ---------------------------------------------------------------------------
INSTRUMENTS = {
    "ihsg":     {"ticker": "^JKSE",     "role": "target", "label": "IHSG",
                 "files": ["IHSG.csv", "JKSE.csv"]},
    "nikkei":   {"ticker": "^N225",     "role": "asia", "label": "Nikkei 225",
                 "files": ["Nikkei225.csv", "Nikkei.csv", "N225.csv"]},
    "kospi":    {"ticker": "^KS11",     "role": "asia", "label": "KOSPI",
                 "files": ["Kospi.csv", "KS11.csv"]},
    "hangseng": {"ticker": "^HSI",      "role": "asia", "label": "Hang Seng",
                 "files": ["HangSeng.csv", "Hang_Seng.csv", "HSI.csv"]},
    "shanghai": {"ticker": "000001.SS", "role": "asia", "label": "Shanghai Composite",
                 "files": ["Shanghai.csv", "SSEC.csv"]},
    "eido":     {"ticker": "EIDO",      "role": "etf", "label": "iShares MSCI Indonesia (EIDO)",
                 "files": ["EIDO.csv"]},
    "idx":      {"ticker": "IDX",       "role": "etf", "label": "VanEck Indonesia (IDX ETF)",
                 "files": ["IDX_VanEck.csv", "IDX.csv"]},
}
ASIA_KEYS = [k for k, v in INSTRUMENTS.items() if v["role"] == "asia"]
ETF_KEYS = [k for k, v in INSTRUMENTS.items() if v["role"] == "etf"]

START_DATE, END_DATE = "2012-01-02", "2025-12-31"
SPLITS = {"train": ("2012-01-01", "2021-12-31"),
          "val":   ("2022-01-01", "2023-12-31"),
          "test":  ("2024-01-01", "2025-12-31")}

ASIA_FEATURES = ["nikkei_gap", "kospi_gap", "hangseng_gap", "shanghai_gap"]
ETF_FEATURES = ["eido_ret", "idx_ret"]
IHSG_FEATURES = ["ihsg_lag1", "ihsg_lag2", "ihsg_lag3", "ihsg_rolling_mean_5", "ihsg_rolling_vol_10"]
FEATURES = ASIA_FEATURES + ETF_FEATURES + IHSG_FEATURES
PREMARKET_FEATURES = ASIA_FEATURES + ETF_FEATURES
FEATURE_GROUPS = {"Asian indices": ASIA_FEATURES,
                  "U.S.-listed Indonesia ETFs": ETF_FEATURES,
                  "IHSG own history": IHSG_FEATURES}
FEATURE_LABELS = {
    "nikkei_gap": "Nikkei 225 opening gap", "kospi_gap": "KOSPI opening gap",
    "hangseng_gap": "Hang Seng opening gap", "shanghai_gap": "Shanghai opening gap",
    "eido_ret": "EIDO overnight return", "idx_ret": "IDX ETF overnight return",
    "ihsg_lag1": "IHSG return (T-1)", "ihsg_lag2": "IHSG return (T-2)",
    "ihsg_lag3": "IHSG return (T-3)", "ihsg_rolling_mean_5": "IHSG 5-day mean return",
    "ihsg_rolling_vol_10": "IHSG 10-day volatility"}
TARGET = "y"

FLAG_TOL = 1e-9                   # Open_T == Close_T-1 (relative tolerance)
CLEAN_YEAR_MAX_FLAG_SHARE = 0.05  # configuration C
ETF_STALE_TO_ZERO = True          # same U.S. session matched twice -> 0 on 2nd day
TIMEZONE = "Asia/Jakarta"
DISCLAIMER = ("Estimates are produced by a statistical model for research purposes. "
              "They are information, not investment advice.")

OHLCV = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

# ---------------------------------------------------------------------------
# Layer 1 - data loading
# ---------------------------------------------------------------------------
_ALIASES = {"date": "Date", "tanggal": "Date", "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "price": "Close", "terakhir": "Close", "adj close": "Adj Close",
            "adj_close": "Adj Close", "volume": "Volume", "vol.": "Volume"}


def _to_number(s: pd.Series, decimal_comma: bool) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    s = s.astype(str).str.strip()
    mult = pd.Series(1.0, index=s.index)
    for suf, m in (("K", 1e3), ("M", 1e6), ("B", 1e9)):
        mask = s.str.upper().str.endswith(suf)
        mult[mask] = m
        s = s.where(~mask, s.str[:-1])
    s = s.str.replace("%", "", regex=False)
    s = (s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
         if decimal_comma else s.str.replace(",", "", regex=False))
    return pd.to_numeric(s.replace({"": np.nan, "-": np.nan, "null": np.nan}), errors="coerce") * mult


def _parse_dates(raw: pd.Series) -> pd.Series:
    raw = raw.astype(str).str.strip().str.replace(r"[ T]\d{2}:\d{2}(:\d{2})?.*$", "", regex=True)
    best = None
    for dayfirst in (False, True):
        d = pd.to_datetime(raw, dayfirst=dayfirst, errors="coerce", format="mixed")
        dd = d.dropna()
        diffs = dd.diff().dt.days.dropna()
        unordered = min(int((diffs < 0).sum()), int((diffs > 0).sum())) if len(diffs) else 0
        score = (int(d.isna().sum()) + unordered, float(diffs.abs().median()) if len(diffs) else 0.0)
        if best is None or score < best[0]:
            best = (score, d)
    return best[1]


def read_price_csv(path) -> pd.DataFrame:
    """Read an OHLCV CSV (yfinance new/old format, Excel ';' format, Investing.com)."""
    path = Path(path)
    raw = path.read_bytes()
    if raw[:4] == b"\x00\x05\x16\x07":
        raise ValueError(f"{path.name} is a macOS '._' metadata file, not the data file.")
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    try:
        delim = csv.Sniffer().sniff("\n".join(lines[:20]), delimiters=",;\t").delimiter
    except csv.Error:
        delim = ","
    skip = []
    if len(lines) > 2 and lines[1].split(delim)[0].strip().lower() == "ticker":
        skip = [1, 2] if lines[2].split(delim)[0].strip().lower() == "date" else [1]
    df = pd.read_csv(io.StringIO(text), sep=delim, skiprows=skip, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    ren = {c: _ALIASES[c.lower()] for c in df.columns if c.lower() in _ALIASES}
    if skip:
        ren[df.columns[0]] = "Date"
    df = df.rename(columns=ren)
    df = df.loc[:, ~df.columns.duplicated()]
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})
    sample = " ".join(df["Close"].dropna().astype(str).head(50))
    dec_comma = delim == ";" and bool(re.search(r"\d,\d", sample))
    out = pd.DataFrame({"Date": _parse_dates(df["Date"])})
    for c in OHLCV:
        out[c] = _to_number(df[c], dec_comma).values if c in df.columns else np.nan
    out["Adj Close"] = out["Adj Close"].fillna(out["Close"])
    # Raw rows are kept as they are (missing values, duplicates, weekends) so the
    # cleaning step can count and report every correction (clean_frame).
    out = out.dropna(subset=["Date"]).sort_values("Date", kind="stable").set_index("Date")
    out.index = pd.DatetimeIndex(out.index).normalize()
    return out[OHLCV]


def find_csv(key: str, data_dir) -> Path | None:
    wanted = {f.lower() for f in INSTRUMENTS[key]["files"]}
    for p in sorted(Path(data_dir).rglob("*.csv")):
        if p.name.startswith("._") or "__MACOSX" in p.parts:
            continue
        name = re.sub(r"^[0-9a-f]{8}-", "", p.name, flags=re.I).lower()
        if name in wanted:
            return p
    return None


def load_from_csv(data_dir) -> dict[str, pd.DataFrame]:
    frames, missing = {}, []
    for k in INSTRUMENTS:
        p = find_csv(k, data_dir)
        if p is None:
            missing.append(f"{k}: {INSTRUMENTS[k]['files']}")
        else:
            frames[k] = read_price_csv(p)
    if missing:
        raise FileNotFoundError("Missing CSV files:\n  " + "\n  ".join(missing))
    return frames


def download_yfinance(key: str, start: str, end: str | None = None) -> pd.DataFrame:
    """auto_adjust=True = yfinance default, identical to the thesis dataset."""
    import yfinance as yf
    h = yf.Ticker(INSTRUMENTS[key]["ticker"]).history(start=start, end=end, interval="1d",
                                                      auto_adjust=True, actions=False)
    if h is None or h.empty:
        raise RuntimeError(f"yfinance returned no data for {INSTRUMENTS[key]['ticker']}")
    h = h.rename(columns=str.title)
    h["Adj Close"] = h["Close"]
    idx = h.index.tz_localize(None) if getattr(h.index, "tz", None) is not None else h.index
    h.index = pd.DatetimeIndex(idx).normalize()
    h = h[~h.index.duplicated(keep="last")].sort_index()
    return h[OHLCV].astype(float)


def refresh_with_yfinance(frames: dict, lookback_days: int = 30) -> dict:
    """CSV history + the latest yfinance bars (used for the live estimate)."""
    start = (pd.Timestamp.today() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    out = {}
    for k, old in frames.items():
        new = download_yfinance(k, start)
        out[k] = pd.concat([old[old.index < new.index.min()], new])
    return out


# ---------------------------------------------------------------------------
# Layer 2 - cleaning, alignment, features, target
# ---------------------------------------------------------------------------
CLEAN_MAX_ABS_MOVE = 0.20   # fixed rule, NOT estimated from the data (no leakage)
STALE_RUN_DAYS = 3          # report-only: close unchanged for >= 3 sessions


def raw_quality_report(frames) -> pd.DataFrame:
    """Missing values and structural problems in the RAW files (before cleaning)."""
    rows = {}
    for k, df in frames.items():
        rows[k] = {"rows": len(df),
                   "duplicate_dates": int(df.index.duplicated().sum()),
                   "weekend_rows": int((df.index.dayofweek >= 5).sum()),
                   **{f"missing_{c}": int(df[c].isna().sum()) for c in ["Open", "High", "Low", "Close", "Volume"]},
                   "nonpositive_price": int((df[["Open", "Close"]] <= 0).any(axis=1).sum()),
                   "first": df.index.min().date(), "last": df.index.max().date()}
    return pd.DataFrame(rows).T


def clean_frame(df):
    """Clean one instrument. Every rule only looks at the row itself or at the
    PREVIOUS row, never at future rows, and thresholds are fixed constants, so
    cleaning cannot leak information from the future.

    1. duplicate dates            -> keep the last record
    2. weekend rows               -> drop (no exchange session)
    3. missing Open or Close      -> drop the row. Prices are NOT forward-filled,
                                     back-filled or interpolated: a filled price
                                     would invent a session that never traded.
                                     Days without a session are handled later by
                                     the alignment rule (gap = 0, Section 3.4.2).
    4. non-positive Open / Close  -> drop
    5. Yahoo 'holiday' rows (O = H = L = C and volume 0) -> drop
    6. price-jump errors: |Close_t/Close_t-1 - 1| or |Open_t/Close_t-1 - 1|
       > 20 % -> drop that row (checked sequentially against the last kept close)
    High / Low / Volume are not used as features, so their missing values are
    left as they are (reported only).
    """
    log = {"rows_in": len(df)}
    d = df[~df.index.duplicated(keep="last")]
    log["removed_duplicates"] = len(df) - len(d)
    wk = d.index.dayofweek >= 5
    miss = d[["Open", "Close"]].isna().any(axis=1)
    nonpos = (d[["Open", "Close"]] <= 0).any(axis=1) & ~miss
    flat = ((d["Open"] == d["High"]) & (d["High"] == d["Low"]) & (d["Low"] == d["Close"])
            & (d["Volume"].fillna(0) == 0)) & ~miss & ~nonpos
    log["removed_weekend"] = int(wk.sum())
    log["removed_missing_open_close"] = int((miss & ~wk).sum())
    log["removed_nonpositive"] = int((nonpos & ~wk).sum())
    log["removed_flat_zero_volume"] = int((flat & ~wk).sum())
    d = d[~(wk | miss | nonpos | flat)]

    # sequential jump filter (only past information is used)
    keep, last_close, jumps = [], None, 0
    for o, c in zip(d["Open"].values, d["Close"].values):
        if last_close is not None and (abs(c / last_close - 1) > CLEAN_MAX_ABS_MOVE
                                       or abs(o / last_close - 1) > CLEAN_MAX_ABS_MOVE):
            keep.append(False); jumps += 1
        else:
            keep.append(True); last_close = c
    d = d[np.array(keep, dtype=bool)] if len(d) else d
    log["removed_price_jump"] = jumps
    log["rows_out"] = len(d)
    return d, log


def quality_flags(df) -> dict:
    """Report-only checks on cleaned data (nothing is changed)."""
    hi = df[["Open", "Close"]].max(axis=1)
    lo = df[["Open", "Close"]].min(axis=1)
    inconsistent = ((df["High"] < hi * (1 - 1e-6)) | (df["Low"] > lo * (1 + 1e-6))).sum()
    same = df["Close"].diff().eq(0)
    runs = same.groupby((~same).cumsum()).sum()
    return {"ohlc_inconsistent": int(inconsistent),
            f"stale_close_runs_>={STALE_RUN_DAYS}": int((runs >= STALE_RUN_DAYS - 1).sum()),
            "zero_volume_days": int((df["Volume"].fillna(0) == 0).sum()),
            "missing_high_low_volume": int(df[["High", "Low", "Volume"]].isna().any(axis=1).sum()),
            "max_abs_close_move_pct": round(float(df["Close"].pct_change().abs().max() * 100), 2)}


def clean_all(frames):
    out, logs = {}, {}
    for k, df in frames.items():
        out[k], logs[k] = clean_frame(df)
    return out, pd.DataFrame(logs).T


def opening_gap(df):
    pc = df["Close"].shift(1)
    return (df["Open"] - pc) / pc


def flag_open_eq_prev_close(df):
    pc = df["Close"].shift(1)
    return ((df["Open"] - pc).abs() <= FLAG_TOL * pc.abs()) & pc.notna()


def etf_return(df):
    return df["Adj Close"].fillna(df["Close"]).pct_change()


def align_asia(df, cal):
    """Day-T opening gap; market closed on T -> 0.0 (Section 3.4.2)."""
    return opening_gap(df).reindex(cal).fillna(0.0)


def align_etf(df, cal):
    """Backward as-of join: U.S. session dated D closes 03:00/04:00 WIB on D+1,
    so it is available before 08:35 WIB on T iff D < T (allow_exact_matches=False)."""
    r = etf_return(df).dropna()
    m = pd.merge_asof(pd.DataFrame({"T": cal}),
                      pd.DataFrame({"us_date": r.index, "ret": r.values}),
                      left_on="T", right_on="us_date", direction="backward",
                      allow_exact_matches=False).set_index("T")
    ret, us = m["ret"], m["us_date"]
    if ETF_STALE_TO_ZERO:
        ret = ret.mask(us.eq(us.shift(1)) & us.notna(), 0.0)
    return ret, us


def assign_split(index):
    s = pd.Series("out_of_range", index=index)
    for name, (a, b) in SPLITS.items():
        s[(index >= a) & (index <= b)] = name
    return s


def build_dataset(frames, target_date=None) -> pd.DataFrame:
    """Aligned dataset on the IHSG calendar. With target_date, a row for that
    session is appended using only information available before 09:00 WIB.
    The same function serves training and inference (no train/serve skew)."""
    ihsg = frames["ihsg"].copy()
    if target_date is not None:
        target_date = pd.Timestamp(target_date).normalize()
        ihsg = ihsg[ihsg.index < target_date]
        ihsg.loc[target_date, :] = np.nan
        ihsg = ihsg.sort_index()
    cal = pd.DatetimeIndex(ihsg.index)
    ds = pd.DataFrame(index=cal)
    ds.index.name = "Date"
    ds["ihsg_open"], ds["ihsg_close"] = ihsg["Open"], ihsg["Close"]
    ds["ihsg_prev_close"] = ihsg["Close"].shift(1)
    ds[TARGET] = (ds["ihsg_open"] - ds["ihsg_prev_close"]) / ds["ihsg_prev_close"]
    ds["flagged"] = flag_open_eq_prev_close(ihsg)
    ret = ihsg["Close"].pct_change()
    for k in (1, 2, 3):
        ds[f"ihsg_lag{k}"] = ret.shift(k)
    ds["ihsg_rolling_mean_5"] = ret.shift(1).rolling(5).mean()
    ds["ihsg_rolling_vol_10"] = ret.shift(1).rolling(10).std()
    for k in ASIA_KEYS:
        df = frames[k] if target_date is None else frames[k][frames[k].index <= target_date]
        ds[f"{k}_closed"] = ~cal.isin(df.index)          # market closed on T -> gap imputed as 0
        ds[f"{k}_gap"] = align_asia(df, cal)
    for k in ETF_KEYS:
        df = frames[k] if target_date is None else frames[k][frames[k].index < target_date]
        ds[f"{k}_ret"], ds[f"{k}_us_date"] = align_etf(df, cal)
        us = ds[f"{k}_us_date"]
        ds[f"{k}_stale"] = us.eq(us.shift(1)) & us.notna()  # no new U.S. session -> 0
    ds["year"] = ds.index.year
    ds["split"] = assign_split(ds.index)
    return ds


def modelling_frame(ds, cfg="A"):
    """A: flagged days removed | B: flagged kept with y=0 | C: A + clean years only."""
    d = ds.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET]).copy()
    d = d[d["split"] != "out_of_range"]
    if cfg == "A":
        d = d[~d["flagged"]]
    elif cfg == "B":
        d.loc[d["flagged"], TARGET] = 0.0
    elif cfg == "C":
        dev = ds[ds["split"].isin(["train", "val"])].dropna(subset=[TARGET])  # test years never inspected
        share = dev.groupby("year")["flagged"].mean()
        clean = share[share < CLEAN_YEAR_MAX_FLAG_SHARE].index
        d = d[~d["flagged"]]
        d = d[(d["split"] == "test") | d["year"].isin(clean)]
    return d


def xy(d, splits):
    splits = [splits] if isinstance(splits, str) else splits
    p = d[d["split"].isin(splits)]
    return p[FEATURES], p[TARGET]


# ---------------------------------------------------------------------------
# Leakage tests
# ---------------------------------------------------------------------------
def perturbation_leakage_test(cleaned, dates, seed=0) -> pd.DataFrame:
    """For each date T, randomly corrupt EVERYTHING that is not yet known at
    08:35 WIB on T and check that the features of T do not change:
      * IHSG: Open/High/Low/Close of T and of every later day
      * Asian indices: High/Low/Close of T (only the open is known) and every later day
      * U.S. ETFs: every session dated T or later (they close after 09:00 WIB on T)
    If any feature of T moves, it depends on future data -> leakage."""
    base = build_dataset(cleaned)
    rng = np.random.default_rng(seed)
    rows = []
    for T in pd.DatetimeIndex(dates):
        f2 = {}
        for k, df in cleaned.items():
            d = df.copy()
            if k == "ihsg" or k in ETF_KEYS:
                m = d.index >= T
                cols = ["Open", "High", "Low", "Close", "Adj Close"]
                d.loc[m, cols] = d.loc[m, cols].values * rng.uniform(0.5, 1.5, (int(m.sum()), len(cols)))
            else:
                later = d.index > T
                cols = ["Open", "High", "Low", "Close", "Adj Close"]
                d.loc[later, cols] = d.loc[later, cols].values * rng.uniform(0.5, 1.5, (int(later.sum()), len(cols)))
                today = d.index == T
                c2 = ["High", "Low", "Close", "Adj Close"]
                d.loc[today, c2] = d.loc[today, c2].values * rng.uniform(0.5, 1.5, (int(today.sum()), len(c2)))
            f2[k] = d
        new = build_dataset(f2)
        a = base.loc[T, FEATURES].astype(float).values
        b = new.loc[T, FEATURES].astype(float).values
        same = np.allclose(a, b, equal_nan=True)
        rows.append({"date": T.date(), "features_unchanged": bool(same),
                     "target_changed": bool(not np.isclose(base.at[T, TARGET], new.at[T, TARGET], equal_nan=True))})
    return pd.DataFrame(rows)


def live_equals_training_test(cleaned, dates) -> pd.DataFrame:
    """The live builder (which physically removes all data from T onward)
    must reproduce the training row exactly."""
    base = build_dataset(cleaned)
    rows = []
    for T in pd.DatetimeIndex(dates):
        live = build_dataset(cleaned, target_date=T).loc[T, FEATURES].astype(float).values
        rows.append({"date": T.date(),
                     "identical": bool(np.allclose(live, base.loc[T, FEATURES].astype(float).values, equal_nan=True))})
    return pd.DataFrame(rows)


def split_integrity(d) -> dict:
    tr, va, te = (d.index[d["split"] == s] for s in ("train", "val", "test"))
    out = {"train_before_val": len(va) == 0 or tr.max() < va.min(),
           "val_before_test": te.min() > (va.max() if len(va) else tr.max()),
           "no_overlapping_dates": len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0,
           "dates_unique": d.index.is_unique, "dates_sorted": d.index.is_monotonic_increasing}
    return out


# ---------------------------------------------------------------------------
# Metrics (Section 3.9)
# ---------------------------------------------------------------------------
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def r2_oos(y, p):
    """R2 against the zero-return benchmark: 1 - SSE/sum(y^2)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum(y ** 2))


def directional_accuracy(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y != 0
    return float(np.mean(np.sign(p[m]) == np.sign(y[m])))


def majority_da(y_train, y_test):
    yt = np.asarray(y_train, float)
    sign = 1.0 if (yt[yt != 0] > 0).mean() >= 0.5 else -1.0
    ys = np.asarray(y_test, float)
    return float(np.mean(np.sign(ys[ys != 0]) == sign)), sign


def diebold_mariano(y, p1, p2, h=1):
    """DM test (squared error) with Harvey-Leybourne-Newbold correction.
    DM < 0 -> forecast 1 more accurate."""
    y, p1, p2 = (np.asarray(a, float) for a in (y, p1, p2))
    d = (y - p1) ** 2 - (y - p2) ** 2
    n, db = len(d), d.mean()
    gam = [np.sum((d[k:] - db) * (d[:n - k] - db)) / n for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    dm = db / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(dm), float(2 * stats.t.sf(abs(dm), df=n - 1))


# ---------------------------------------------------------------------------
# Layer 3 - prediction service
# ---------------------------------------------------------------------------
def explain(model, X: pd.DataFrame):
    """Local TreeSHAP attributions -> (DataFrame, base value)."""
    if type(model).__name__ == "VotingRegressor":
        # ensemble: SHAP values are additive, so the ensemble attribution is the
        # weighted sum of each member's attribution (exact, not an approximation)
        w = np.asarray(model.weights if model.weights is not None else [1.0] * len(model.estimators_), float)
        w = w / w.sum()
        total, base = None, 0.0
        for wi, est in zip(w, model.estimators_):
            c, b = explain(est, X)
            total = c * wi if total is None else total + c * wi
            base += wi * b
        return total, base


def sentiment_score(x, meta):
    """0-100 score: SHAP-weighted standardised pre-market signals, tanh-squashed."""
    w, sd = meta["sentiment_weights"], meta["feature_std_dev"]
    z = {f: float(np.clip(x[f] / sd[f], -3, 3)) for f in PREMARKET_FEATURES}
    s = sum(w[f] * z[f] for f in PREMARKET_FEATURES)
    score = float(50 + 50 * np.tanh(s / 1.5))
    label = ("Strongly bearish" if score < 25 else "Bearish" if score < 40 else
             "Neutral" if score <= 60 else "Bullish" if score <= 75 else "Strongly bullish")
    return {"score": score, "label": label, "z": z,
            "breadth_up": int(sum(x[f] > 0 for f in PREMARKET_FEATURES)),
            "breadth_total": len(PREMARKET_FEATURES)}


def historical_analogs(x, history, meta, k=10):
    """k nearest past sessions (standardised pre-market signals), no look-ahead."""
    h = history.dropna(subset=PREMARKET_FEATURES + [TARGET])
    h = h[~h["flagged"].astype(bool)]
    sd = pd.Series(meta["feature_std_dev"])[PREMARKET_FEATURES]
    dist = np.sqrt((((h[PREMARKET_FEATURES] - x[PREMARKET_FEATURES].astype(float)) / sd) ** 2).sum(axis=1))
    nn = dist.nsmallest(k).index
    tab = h.loc[nn, PREMARKET_FEATURES + [TARGET]].copy()
    tab.insert(0, "distance", dist[nn])
    y = tab[TARGET]
    return {"table": tab, "mean": float(y.mean()), "median": float(y.median()),
            "share_up": float((y > 0).mean()), "k": int(len(tab))}


def estimate(model, meta, model_name, x, prev_close, with_shap=True):
    X = x[FEATURES].astype(float).to_frame().T
    y_hat = float(model.predict(X)[0])
    iv = meta["intervals"][model_name]
    lo, hi = y_hat + iv["q_lower"], y_hat + iv["q_upper"]
    out = {"model": model_name, "y_hat": y_hat, "prev_close": float(prev_close),
           "open_hat": prev_close * (1 + y_hat), "lower_y": lo, "upper_y": hi,
           "lower_open": prev_close * (1 + lo), "upper_open": prev_close * (1 + hi),
           "direction": "UP" if y_hat > 0 else "DOWN" if y_hat < 0 else "FLAT",
           "features": {f: float(x[f]) for f in FEATURES}}
    if with_shap:
        c, base = explain(model, X)
        out["shap"], out["shap_base"] = c.iloc[0].to_dict(), base
    return out


def predict_for_date(frames, target_date, model, meta, model_name, k_analogs=10):
    """Full estimate for one IHSG session using only data available before it."""
    target_date = pd.Timestamp(target_date).normalize()
    cleaned, _ = clean_all(frames)
    ds = build_dataset(cleaned, target_date=target_date)
    row = ds.loc[target_date]
    missing = [f for f in FEATURES if pd.isna(row[f])]
    if missing:
        raise ValueError(f"Cannot build features for {target_date.date()}: missing {missing}")
    closes = ds["ihsg_close"].dropna()
    out = estimate(model, meta, model_name, row, float(closes.iloc[-1]))
    out["target_date"] = str(target_date.date())
    out["prev_close_date"] = str(closes.index[-1].date())
    out["sentiment"] = sentiment_score(row, meta)
    out["analogs"] = historical_analogs(row, ds[ds.index < target_date], meta, k_analogs)
    out["warnings"] = [f"{INSTRUMENTS[k]['label']}: no session on {target_date.date()} "
                       "(holiday / not open yet) -> gap = 0" for k in ASIA_KEYS
                       if target_date not in cleaned[k].index]
    out["etf_session"] = {k: str(pd.Timestamp(row[f"{k}_us_date"]).date())
                          for k in ETF_KEYS if pd.notna(row[f"{k}_us_date"])}
    ih = cleaned["ihsg"]
    if target_date in ih.index:
        o = float(ih.at[target_date, "Open"])
        out["actual_open"], out["actual_y"] = o, o / out["prev_close"] - 1
        out["actual_flagged"] = abs(o - out["prev_close"]) <= FLAG_TOL * out["prev_close"]
    return out


def next_session_date():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    d = pd.Timestamp(datetime.now(ZoneInfo(TIMEZONE)).date())
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def format_report(p):
    s, a = p["sentiment"], p["analogs"]
    lines = [f"IHSG opening estimate for {p['target_date']}  (model: {p['model']})",
             f"  previous close ({p['prev_close_date']}): {p['prev_close']:,.2f}",
             f"  estimated gap  : {p['y_hat'] * 100:+.3f} %   direction: {p['direction']}",
             f"  estimated open : {p['open_hat']:,.2f}",
             f"  95% interval   : {p['lower_open']:,.2f} - {p['upper_open']:,.2f}",
             f"  sentiment      : {s['score']:.0f}/100 ({s['label']}), "
             f"{s['breadth_up']}/{s['breadth_total']} signals up",
             f"  analogs (k={a['k']}): mean gap {a['mean'] * 100:+.3f} %, {a['share_up'] * 100:.0f}% opened up"]
    if "actual_open" in p:
        lines.append(f"  ACTUAL open    : {p['actual_open']:,.2f} ({p['actual_y'] * 100:+.3f} %)")
    lines += [f"  ! {w}" for w in p["warnings"]]
    return "\n".join(lines)
