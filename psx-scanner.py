"""
PSX Market Intelligence Report — Wall Street Edition
KSE-100 · 9-Layer Signal Engine · Institutional-Grade Scanner
─────────────────────────────────────────────────────────────
"""

import html
import json
import math
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(page_title="PSX Market Intelligence", layout="wide")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CFG = {
    "MARKET_OPEN": "09:30",
    "MARKET_CLOSE": "15:30",
    "REFRESH_SEC": 180,
    "MIN_VOLUME": 100_000,
    "MIN_PRICE": 2.0,
    "INST_VOL_X": 2.5,
    "BREADTH_MIN": 0.0,
    "DB_PATH": os.path.join(BASE_DIR, "psx_elite.db"),
    "HIST_DAYS": 365,
    "REGIME_PERIOD": 50,  # MA period for market regime detection (was 200)
    # ── Risk engine ────────────────────────────────────────────────────────
    # ACCOUNT_CAPITAL is a placeholder — edit to your actual trading capital
    # (PKR). Position sizing, risk %, and VaR outputs all scale off this.
    "ACCOUNT_CAPITAL": 100_000,
    "MAX_RISK_PCT": 0.02,          # max capital risked per trade (stop distance)
    "FRACTIONAL_KELLY": 0.25,      # standard practitioner haircut on full Kelly
    "ADV_PARTICIPATION_CAP": 0.08,  # don't size above 8% of 20D avg daily volume
    # ── Signal outcome tracking / calibration ─────────────────────────────
    "VALIDATOR_LOOKBACK_DAYS": 90,
    "VALIDATOR_MIN_SAMPLES": 15,    # below this, a layer's win-rate isn't trusted
    "SIGNAL_HORIZON_DAYS": {"INTRA": 1, "SWING": 15, "LONG": 90, "DIP": 10},
    "SIGNAL_DECAY_TAU_DAYS": {"INTRA": 0.5, "SWING": 4.0, "LONG": 30.0, "DIP": 5.0},
}

# ── Per-strategy layer budgets ────────────────────────────────────────────────
# Intraday, Swing, and Long-Term are different strategies, not the same 9
# layers wearing different thresholds. They used to share one LAYER_BUDGET
# dict, meaning a same-day scalp weighted 30-day OBV trend the same as a
# multi-month hold did. Each budget below still sums to 100 across the same
# 9 layers, but the weighting reflects what actually matters at that horizon:
#   - Intraday: trend/momentum/volume dominate; 30-day "historical quality"
#     and macro regime barely matter for a trade closed same day.
#   - Swing: roughly balanced, closest to the original weights.
#   - Long-term: sector quality/trend, historical stability, institutional
#     flow and macro regime dominate; daily volume spikes barely matter for
#     a position you're building over weeks.
LAYER_BUDGET_INTRA = {
    "trend": 20,
    "momentum": 20,
    "volume": 16,
    "pattern": 14,
    "breadth": 8,
    "volatility": 8,
    "historical": 4,
    "flow": 6,
    "regime": 4,
}
LAYER_BUDGET_SWING = {
    "trend": 18,
    "momentum": 17,
    "volume": 12,
    "pattern": 12,
    "breadth": 7,
    "volatility": 8,
    "historical": 12,
    "flow": 10,
    "regime": 4,
}
LAYER_BUDGET_LONG = {
    "trend": 20,
    "momentum": 14,
    "volume": 8,
    "pattern": 10,
    "breadth": 6,
    "volatility": 6,
    "historical": 16,
    "flow": 12,
    "regime": 8,
}
assert sum(LAYER_BUDGET_INTRA.values()) == 100
assert sum(LAYER_BUDGET_SWING.values()) == 100
assert sum(LAYER_BUDGET_LONG.values()) == 100

THRESH_INTRA = 55
THRESH_SWING = 50
THRESH_LONG = 45
THRESH_DIP = 40

# ── Universe ──────────────────────────────────────────────────────────────────
# Full KSE-100 constituent list, per PSX's official re-composition notice
# effective April 1, 2026 (psx.com.pk) and cross-checked live against the PSX
# Data Portal's KSE-100 constituents table (dps.psx.com.pk/indices/KSE100).
# The previous list here was a stale ~60-symbol subset containing several
# tickers no longer in the index and was missing 40+ current constituents.
KSE100 = [
    "ABL", "ABOT", "AGP", "AHCL", "AICL", "AIRLINK", "AKBL", "APL", "ATLH", "ATRL",
    "BAFL", "BAHL", "BNWM", "BOP", "BWCL", "CHCC", "CNERGY", "COLG", "CPHL", "DCR",
    "DGKC", "EFERT", "ENGROH", "FABL", "FATIMA", "FCCL", "FFC", "FFL", "FHAM", "GADT",
    "GAL", "GHGL", "GHNI", "GLAXO", "HALEON", "HBL", "HCAR", "HGFA", "HINOON", "HMB",
    "HUBC", "HUMNL", "IBFL", "ILP", "INDU", "INIL", "ISL", "JDWS", "JVDC", "KAPCO",
    "KEL", "KOHC", "KTML", "LCI", "LOTCHEM", "LUCK", "MARI", "MCB", "MEBL", "MEHT",
    "MLCF", "MTL", "MUREB", "NATF", "NBP", "NESTLE", "NML", "NPL", "OGDC", "PABC",
    "PAEL", "PAKT", "PGLC", "PIBTL", "PIOC", "PKGS", "POL", "POWER", "PPL", "PSEL",
    "PSO", "PSX", "PTC", "RMPL", "SAZEW", "SCBPL", "SEARL", "SHFA", "SNGP", "SRVI",
    "SSGC", "SSOM", "SYS", "TGL", "THALL", "TPLRF1", "TRG", "UBL", "UPFL", "YOUW",
]
assert len(KSE100) == 100
assert len(set(KSE100)) == 100

SECTORS: Dict[str, Dict] = {
    "Banks": {
        "symbols": [
            "UBL", "BAHL", "BAFL", "BOP", "FABL", "MEBL", "HBL", "HMB",
            "MCB", "NBP", "SCBPL", "AKBL", "ABL",
        ],
        "quality": 9,
    },
    "E&P": {"symbols": ["OGDC", "MARI", "POL", "PPL"], "quality": 9},
    "Fertilizer": {"symbols": ["FFC", "EFERT", "AHCL", "FATIMA"], "quality": 9},
    "Cement": {
        "symbols": ["LUCK", "DGKC", "BWCL", "FCCL", "KOHC", "CHCC", "MLCF", "POWER", "PIOC"],
        "quality": 7,
    },
    "Tech": {"symbols": ["SYS", "AIRLINK", "PTC", "TRG", "HUMNL"], "quality": 7},
    "Power": {"symbols": ["HUBC", "KAPCO", "KEL", "NPL"], "quality": 7},
    "Oil & Gas": {"symbols": ["PSO", "APL", "SNGP", "SSGC", "ATRL", "CNERGY"], "quality": 8},
    "Auto": {
        "symbols": ["MTL", "INDU", "SAZEW", "ATLH", "HCAR", "GAL", "GHNI", "THALL"],
        "quality": 6,
    },
    "Food": {"symbols": ["NESTLE", "FFL", "MUREB", "COLG", "NATF", "RMPL", "UPFL"], "quality": 8},
    "Pharma": {"symbols": ["SEARL", "AGP", "GLAXO", "ABOT", "CPHL", "HALEON", "HINOON"], "quality": 9},
    "Textile": {"symbols": ["GADT", "YOUW", "NML", "MEHT", "KTML", "ILP", "BNWM", "IBFL"], "quality": 5},
    "Chemical": {"symbols": ["LCI", "LOTCHEM"], "quality": 6},
    "Insurance": {"symbols": ["AICL"], "quality": 7},
    "Financial Services": {
        "symbols": ["ENGROH", "PSX", "DCR", "TPLRF1", "FHAM", "PGLC", "HGFA"],
        "quality": 6,
    },
    "Engineering": {"symbols": ["ISL", "INIL", "PAEL"], "quality": 6},
    "Tobacco": {"symbols": ["PAKT"], "quality": 7},
    "Misc": {
        "symbols": [
            "PIBTL", "JDWS", "PKGS", "SSOM", "SRVI", "TGL", "GHGL",
            "PSEL", "PABC", "SHFA", "JVDC",
        ],
        "quality": 5,
    },
}
SYM_SECTOR = {sym: sec for sec, v in SECTORS.items() for sym in v["symbols"]}

# Every KSE100 symbol must resolve to a sector — get_hist_metrics/scoring silently
# falls back to "Misc" quality otherwise, which would misprice long-term scores.
_uncategorized = [s for s in KSE100 if s not in SYM_SECTOR]
assert not _uncategorized, f"KSE100 symbols missing from SECTORS: {_uncategorized}"

TV_COLS = [
    "name",
    "close",
    "change",
    "volume",
    "relative_volume_10d_calc",
    "average_volume_10d_calc",
    "RSI",
    "MACD.macd",
    "MACD.signal",
    "BB.lower",
    "BB.upper",
    "EMA20",
    "EMA50",
    "EMA25",
    "change|1W",
    "High.1M",
    "Low.1M",
    "VWAP",
    "EMA10",
    "ADX",
    "ATR",
    "Stoch.K",
    "Stoch.D",
    "open",
    "high",
    "low",
    "EMA5",
    "change|1M",
    "RSI[1]",
    "low|7D",
    "MACD.hist",
    "Pivot.M.Classic.Middle",
    "BB.basis",
    "High.3M",
    "Low.3M",
    "High.6M",
    "Low.6M",  # fetched for potential range-context use; not yet consumed by any scorer
    # ── 15-min resolution — the Intraday Scalps scorer uses these instead of
    # the daily-resolution columns above; a "scalp" scored off a 5-day/10-day
    # EMA isn't actually an intraday signal. TradingView's scanner accepts
    # a `|<minutes>` suffix per-column, so we can pull both timeframes in
    # one request. Indices 37–45.
    "RSI|15",
    "RSI[1]|15",
    "MACD.macd|15",
    "MACD.signal|15",
    "MACD.hist|15",
    "EMA5|15",
    "EMA10|15",
    "ADX|15",
    "Stoch.K|15",
    "Stoch.D|15",
]

TV_URL = "https://scanner.tradingview.com/pakistan/scan"

# ══════════════════════════════════════════════════════════════════════════════
# CORE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def pkt_now() -> datetime:
    return datetime.now(pytz.timezone("Asia/Karachi"))


def is_market_open() -> bool:
    now = pkt_now()
    if now.weekday() >= 5:
        return False
    t = now.strftime("%H:%M")
    return CFG["MARKET_OPEN"] <= t <= CFG["MARKET_CLOSE"]


def safe(val, default=0.0):
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default


def _rr(price: float, target: float, stop: float) -> float:
    denom = price - stop
    if denom <= 0 or target <= price:
        return 0.0
    return round((target - price) / denom, 2)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _grade(score: int, layers_active: int) -> str:
    if score >= 78 and layers_active >= 7:
        return "A+"
    if score >= 72 and layers_active >= 6:
        return "A"
    if score >= 62 and layers_active >= 5:
        return "B+"
    if score >= 55 and layers_active >= 5:
        return "B"
    if score >= 45 and layers_active >= 4:
        return "C"
    return "D"


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════


def init_db():
    conn = sqlite3.connect(CFG["DB_PATH"])
    try:
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS price_history (
                    date   TEXT,
                    symbol TEXT,
                    open   REAL,
                    high   REAL,
                    low    REAL,
                    close  REAL,
                    volume INTEGER,
                    PRIMARY KEY (date, symbol)
                );
                CREATE INDEX IF NOT EXISTS idx_sym_date ON price_history(symbol, date DESC);

                CREATE TABLE IF NOT EXISTS daily_snapshot (
                    date   TEXT,
                    symbol TEXT,
                    trend  TEXT,
                    rv     REAL,
                    rs_rank REAL DEFAULT 50,
                    PRIMARY KEY (date, symbol)
                );
                CREATE INDEX IF NOT EXISTS idx_snapshot_date ON daily_snapshot(date DESC);

                -- Every signal the scanner fires gets logged here, then
                -- resolved against subsequent price_history bars (did it hit
                -- target, hit stop, or expire unfilled?). This is the actual
                -- data the Bayesian layer-confidence model and the vectorized
                -- validator are calibrated from — without it, any "win
                -- probability" the system reports would be a made-up number.
                CREATE TABLE IF NOT EXISTS signal_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    entry_price REAL,
                    target REAL,
                    stop REAL,
                    score REAL,
                    layers_json TEXT,
                    status TEXT DEFAULT 'OPEN',
                    resolved_date TEXT,
                    resolved_price REAL,
                    r_multiple REAL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_signal_episode
                    ON signal_log(symbol, strategy, first_seen);
                CREATE INDEX IF NOT EXISTS idx_signal_status
                    ON signal_log(symbol, strategy, status);
            """)
            # --- Schema Migration ---
            # Add rs_rank to daily_snapshot if it doesn't exist
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(daily_snapshot)")
            columns = [info[1] for info in cursor.fetchall()]
            if 'rs_rank' not in columns:
                conn.execute("ALTER TABLE daily_snapshot ADD COLUMN rs_rank REAL DEFAULT 50")
                st.toast("Upgraded database schema for `rs_rank`.", icon="🚀")

    finally:
        conn.close()


def sync_historical_data(symbols: List[str]):
    """Batch download historical OHLCV from Yahoo Finance."""
    end = datetime.now()
    start = end - timedelta(days=CFG["HIST_DAYS"] + 30)  # buffer for holidays
    tickers = [f"{s}.KA" for s in symbols]

    ph = st.empty()
    ph.caption(f"Fetching {CFG['HIST_DAYS']}-day market history…")

    try:
        df = yf.download(
            tickers, start=start, end=end, group_by="ticker", progress=False, auto_adjust=True
        )
    except Exception as e:
        ph.error(f"Yahoo download failed: {e}")
        return

    conn = sqlite3.connect(CFG["DB_PATH"])
    saved = 0
    try:
        for sym in symbols:
            try:
                key = f"{sym}.KA"
                if key not in df.columns.get_level_values(0):
                    continue
                ticker_df = df[key].dropna().reset_index()
                if ticker_df.empty:
                    continue
                rows = []
                for _, r in ticker_df.iterrows():
                    try:
                        rows.append(
                            (
                                r["Date"].strftime("%Y-%m-%d"),
                                sym,
                                float(r["Open"]),
                                float(r["High"]),
                                float(r["Low"]),
                                float(r["Close"]),
                                int(r["Volume"]),
                            )
                        )
                    except Exception:
                        continue
                with conn:
                    conn.executemany(
                        "INSERT OR REPLACE INTO price_history VALUES (?,?,?,?,?,?,?)", rows
                    )
                saved += 1
            except Exception:
                continue
    finally:
        conn.close()
        ph.caption(
            f"Synced {saved}/{len(symbols)} symbols — {CFG['HIST_DAYS']}-day history loaded."
        )


def save_snapshot(raw: list):
    if not raw:
        return
    today = pkt_now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(CFG["DB_PATH"])
    try:
        rows = []
        for item in raw:
            d = item.get("d", [])
            if len(d) >= 26 and d[0]:
                rows.append(
                    (
                        today,
                        d[0],
                        safe(d[23]),  # open
                        safe(d[24]),  # high
                        safe(d[25]),  # low
                        safe(d[1]),   # close
                        int(safe(d[3])),  # volume
                    )
                )
        if rows:
            with conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO price_history VALUES (?,?,?,?,?,?,?)", rows
                )
    finally:
        conn.close()


def get_yesterday_snapshot() -> Dict[str, Dict]:
    conn = sqlite3.connect(CFG["DB_PATH"])
    try:
        today = pkt_now().strftime("%Y-%m-%d")
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM daily_snapshot WHERE date < ?", (today,))
        last_date = cursor.fetchone()[0]
        snapshot = {}
        if last_date:
            df = pd.read_sql(
                "SELECT symbol, trend, rv, rs_rank FROM daily_snapshot WHERE date = ?",
                conn,
                params=(last_date,),
            )
            for _, row in df.iterrows():
                snapshot[row["symbol"]] = {
                    "trend": row["trend"],
                    "rv": row["rv"],
                    "rs_rank": row.get("rs_rank", 50),
                }
        return snapshot
    finally:
        conn.close()


def save_daily_snapshot(df: pd.DataFrame):
    if df.empty:
        return
    today = pkt_now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(CFG["DB_PATH"])
    try:
        rows = []
        for _, row in df.iterrows():
            # Removed rs_rank as it's no longer computed
            rows.append((today, row["Symbol"], row["Bias"], row.get("RV", 0.0)))
        with conn:
            # Use explicit column names for robustness
            conn.executemany(
                "INSERT OR REPLACE INTO daily_snapshot (date, symbol, trend, rv) VALUES (?,?,?,?)",
                rows,
            )
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# HISTORICAL ANALYTICS ENGINE  — Wall Street Edition
# ══════════════════════════════════════════════════════════════════════════════


def _calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    delta = close.diff()
    direction = np.where(delta > 0, 1, np.where(delta < 0, -1, 0))
    return (direction * volume).cumsum()


def _calc_cmf(high, low, close, volume, period=20) -> float:
    """Chaikin Money Flow — institutional buying/selling pressure."""
    clv = ((close - low) - (high - close)) / (high - low + 1e-9)
    cmf = (clv * volume).rolling(period).sum() / (volume.rolling(period).sum() + 1e-9)
    return float(cmf.iloc[-1]) if not cmf.empty else 0.0


def _calc_adx(high, low, close, period=14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Wilder's ADX with +DI/-DI.
    ADX alone only measures trend *strength*, not direction — a stock can have
    ADX 40 while grinding lower. We keep +DI/-DI so callers can confirm the
    strong trend is actually bullish before rewarding it.
    """
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr_w + 1e-9))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr_w + 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx, plus_di, minus_di


def _calc_psar(high, low, close, af_step=0.02, af_max=0.2) -> Tuple[pd.Series, pd.Series]:
    """
    Parabolic SAR (Wilder). Returns (sar_series, bullish_series).
    Used two ways downstream: (1) as a trend/flip timing signal, and
    (2) as a trailing-stop level — it hugs price more tightly than a flat
    ATR multiple once a trend is established, which is the whole point of
    using it for swing/long-term stop placement.
    """
    n = len(close)
    idx = close.index
    if n < 5:
        return pd.Series(close.values, index=idx), pd.Series([True] * n, index=idx)

    h, l = high.values, low.values
    sar = np.zeros(n)
    bullish = np.zeros(n, dtype=bool)

    is_bull = close.iloc[1] >= close.iloc[0]
    sar[0] = l[0] if is_bull else h[0]
    ep = h[0] if is_bull else l[0]
    af = af_step
    bullish[0] = is_bull

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if is_bull:
            cur_sar = prev_sar + af * (ep - prev_sar)
            cur_sar = min(cur_sar, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] < cur_sar:
                is_bull = False
                cur_sar = ep
                ep = l[i]
                af = af_step
            elif h[i] > ep:
                ep = h[i]
                af = min(af + af_step, af_max)
        else:
            cur_sar = prev_sar + af * (ep - prev_sar)
            cur_sar = max(cur_sar, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > cur_sar:
                is_bull = True
                cur_sar = ep
                ep = h[i]
                af = af_step
            elif l[i] < ep:
                ep = l[i]
                af = min(af + af_step, af_max)
        sar[i] = cur_sar
        bullish[i] = is_bull

    return pd.Series(sar, index=idx), pd.Series(bullish, index=idx)


def _calc_vwap_bands(high, low, close, volume) -> Tuple[float, float, float]:
    """Rolling VWAP with ±1σ bands over last 20 sessions."""
    tp = (high + low + close) / 3
    n = min(20, len(tp))
    w_sum = (tp * volume).tail(n).sum()
    v_sum = volume.tail(n).sum()
    vwap = w_sum / v_sum if v_sum > 0 else float(close.iloc[-1])
    std = float(((tp.tail(n) - vwap) ** 2).mean() ** 0.5)
    return vwap, vwap + std, vwap - std


def _volume_profile_poc(close: pd.Series, volume: pd.Series, bins=20) -> Tuple[float, float, float]:
    """
    Volume Profile: Point of Control (price with most volume),
    Value Area High, Value Area Low (covers 70% of total volume).
    """
    if len(close) < 10:
        return float(close.iloc[-1]), float(close.max()), float(close.min())
    c_min, c_max = close.min(), close.max()
    if c_max == c_min:
        return float(c_min), float(c_max), float(c_min)
    edges = np.linspace(c_min, c_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    vol_profile = np.zeros(bins)
    for price, vol in zip(close, volume):
        idx = int((price - c_min) / (c_max - c_min + 1e-9) * (bins - 1))
        idx = max(0, min(bins - 1, idx))
        vol_profile[idx] += vol
    poc_idx = int(np.argmax(vol_profile))
    poc = float(centers[poc_idx])
    # Value Area: add bins from POC outward until 70% covered
    total = vol_profile.sum()
    target = total * 0.70
    covered = vol_profile[poc_idx]
    lo_idx, hi_idx = poc_idx, poc_idx
    while covered < target and (lo_idx > 0 or hi_idx < bins - 1):
        lo_add = vol_profile[lo_idx - 1] if lo_idx > 0 else 0
        hi_add = vol_profile[hi_idx + 1] if hi_idx < bins - 1 else 0
        if lo_add >= hi_add and lo_idx > 0:
            lo_idx -= 1
            covered += lo_add
        elif hi_idx < bins - 1:
            hi_idx += 1
            covered += hi_add
        else:
            lo_idx -= 1
            covered += lo_add
    return poc, float(centers[hi_idx]), float(centers[lo_idx])


def _detect_divergence(rsi_series: pd.Series, close: pd.Series) -> Tuple[bool, bool]:
    """
    Bullish divergence: price making lower lows, RSI making higher lows.
    Bearish divergence: price making higher highs, RSI making lower highs.
    """
    if len(rsi_series) < 10 or len(close) < 10:
        return False, False
    # Use last 10 bars
    p = close.tail(10).values
    r = rsi_series.tail(10).values
    bull_div = (p[-1] < p[0]) and (r[-1] > r[0])  # price down, RSI up
    bear_div = (p[-1] > p[0]) and (r[-1] < r[0])  # price up, RSI down
    return bull_div, bear_div


def _mean_reversion_zscore(close: pd.Series, window=20) -> float:
    """Z-score of current price vs rolling mean — how stretched/compressed."""
    if len(close) < window:
        return 0.0
    mu = close.tail(window).mean()
    sig = close.tail(window).std()
    if sig == 0:
        return 0.0
    return float((close.iloc[-1] - mu) / sig)


def _calc_regime(close: pd.Series, period: int) -> str:
    """
    Market regime from 200-day MA slope:
      BULL: price > MA200 and MA200 rising
      BEAR: price < MA200 and MA200 falling
      NEUTRAL: mixed
    """
    if len(close) < period:
        return "NEUTRAL"
    ma = close.rolling(period).mean()
    if ma.iloc[-1] != ma.iloc[-1]:
        return "NEUTRAL"  # NaN check
    slope = (ma.iloc[-1] - ma.iloc[-10]) / ma.iloc[-10] * 100 if len(ma.dropna()) > 10 else 0
    above = float(close.iloc[-1]) > float(ma.iloc[-1])
    if above and slope > 0.1:
        return "BULL"
    if not above and slope < -0.1:
        return "BEAR"
    return "NEUTRAL"


def get_hist_metrics(symbol: str, db_conn: sqlite3.Connection) -> Dict:
    df = pd.read_sql(
        "SELECT open, high, low, close, volume FROM price_history WHERE symbol=? ORDER BY date ASC",
        db_conn,
        params=(symbol,),
    )

    empty = {
        "has_data": False,
        "n": 0,
        "stability": 0.0,
        "avg_vol": 0.0,
        "vol_trend": 0.0,
        "trend_pct_30": 0.0,
        "trend_pct_10": 0.0,
        "trend_pct_60": 0.0,
        "volatility": 0.0,
        "atr_20": 0.0,
        "momentum": 0.0,
        "rsi_hist": 50.0,
        "rsi_slope": 0.0,
        "ema10_slope": 0.0,
        "ema20_slope": 0.0,
        "ema50_slope": 0.0,
        "support_level": 0.0,
        "resistance_level": 0.0,
        "consec_up": 0,
        "consec_down": 0,
        "higher_lows": False,
        "vol_accumulation": False,
        "squeeze": False,
        "triple_bottom": False,
        # New fields
        "obv_slope": 0.0,
        "cmf": 0.0,
        "poc": 0.0,
        "vah": 0.0,
        "val": 0.0,
        "bull_divergence": False,
        "bear_divergence": False,
        "zscore": 0.0,
        "regime": "NEUTRAL",
        "hist_pct": 50.0,
        "ema200": 0.0,
        "return_3m": 0.0,
        # ADX / PSAR
        "adx_hist": 0.0,
        "adx_rising": False,
        "di_bullish": True,
        "psar": 0.0,
        "psar_bullish": True,
        "psar_flip_recent": False,
        "psar_bars_since_flip": 99,
        "psar_dist_pct": 0.0,
        "ret_series": [],
    }

    if len(df) < 20:
        return empty

    n = len(df)
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── 1. Trend metrics ──────────────────────────────────────────────────────
    trend_pct_30 = (c.iloc[-1] / c.iloc[max(0, n - 30)] - 1) * 100 if n >= 30 else 0.0
    trend_pct_10 = (c.iloc[-1] / c.iloc[max(0, n - 10)] - 1) * 100 if n >= 10 else 0.0
    trend_pct_60 = (c.iloc[-1] / c.iloc[max(0, n - 60)] - 1) * 100 if n >= 60 else 0.0
    return_3m = (c.iloc[-1] / c.iloc[max(0, n - 63)] - 1) * 100 if n >= 63 else 0.0

    # ── 2. Stability ──────────────────────────────────────────────────────────
    up_days = (c.diff() >= 0).tail(20).sum()
    stability = (up_days / 20) * 10 if n >= 20 else 0.0

    # ── 3. Volume analytics ───────────────────────────────────────────────────
    avg_vol = v.tail(30).mean() if n >= 30 else v.mean()
    recent_vol = v.tail(5).mean() if n >= 5 else avg_vol
    older_vol = v.iloc[-20:-5].mean() if n >= 20 else avg_vol
    vol_trend = (recent_vol / older_vol - 1) * 100 if older_vol > 0 else 0.0

    # ── 4. ATR (20) ───────────────────────────────────────────────────────────
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_20 = float(tr.rolling(20).mean().iloc[-1]) if len(tr) >= 20 else 0.0

    # ── 5. Historical volatility ──────────────────────────────────────────────
    returns = c.pct_change().dropna()
    volatility = float(returns.tail(20).std()) * math.sqrt(252) * 100 if len(returns) >= 20 else 0.0

    # ── 6. EMAs & slopes ─────────────────────────────────────────────────────
    ema5 = c.ewm(span=5, adjust=False).mean()
    ema10 = c.ewm(span=10, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    # Correctly calculate ema200 only if enough data exists. With HIST_DAYS=90, this will be 0.
    ema200_val = 0.0
    if n >= 200:
        ema200_series = c.ewm(span=200, adjust=False).mean()
        if not ema200_series.empty:
            ema200_val = float(ema200_series.iloc[-1])
    momentum = (float(ema5.iloc[-1]) / float(ema20.iloc[-1]) - 1) * 100 if len(ema5) >= 2 else 0.0
    ema10_slope = (
        (float(ema10.iloc[-1]) / float(ema10.iloc[max(0, len(ema10) - 5)]) - 1) * 100
        if len(ema10) >= 5
        else 0.0
    )
    ema20_slope = (
        (float(ema20.iloc[-1]) / float(ema20.iloc[max(0, len(ema20) - 5)]) - 1) * 100
        if len(ema20) >= 5
        else 0.0
    )
    ema50_slope = (
        (float(ema50.iloc[-1]) / float(ema50.iloc[max(0, len(ema50) - 10)]) - 1) * 100
        if len(ema50) >= 10
        else 0.0
    )

    # ── 7. Historical RSI (14) & slope ───────────────────────────────────────
    delta = c.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi_series = 100 - (100 / (1 + rs))
    rsi_hist = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
    rsi_slope = (
        float(rsi_series.iloc[-1] - rsi_series.iloc[max(0, len(rsi_series) - 4)])
        if len(rsi_series) >= 4
        else 0.0
    )

    # ── 8. Support / Resistance ──────────────────────────────────────────────
    support_level = float(l.tail(20).min()) if n >= 20 else 0.0
    resistance_level = float(h.tail(20).max()) if n >= 20 else 0.0

    # ── 9. Consecutive streak ────────────────────────────────────────────────
    consec_up = consec_down = 0
    if n > 1:
        diffs = c.diff().dropna()
        for val in reversed(diffs):
            if val > 0:
                if consec_down > 0:
                    break
                consec_up += 1
            elif val < 0:
                if consec_up > 0:
                    break
                consec_down += 1
            else:
                break

    # ── 10. Higher-lows ──────────────────────────────────────────────────────
    higher_lows = n >= 3 and l.iloc[-1] > l.iloc[-2] > l.iloc[-3]

    # ── 11. Triple bottom ────────────────────────────────────────────────────
    triple_bottom = False
    if n >= 5:
        rl = l.tail(5)
        if (rl.max() - rl.min()) / (rl.min() + 1e-9) < 0.02:
            triple_bottom = True

    # ── 12. Volume accumulation ──────────────────────────────────────────────
    vol_accumulation = False
    if n >= 10:
        up_mask = c.diff() > 0
        down_mask = c.diff() < 0
        up_vol = v[up_mask].tail(10).sum()
        down_vol = v[down_mask].tail(10).sum()
        if down_vol > 0:
            vol_accumulation = up_vol > down_vol * 1.3

    # ── 13. BB Squeeze ───────────────────────────────────────────────────────
    squeeze = False
    if n >= 20:
        bb_std = c.rolling(20).std()
        bb_mean = c.rolling(20).mean()
        bb_width = ((2 * bb_std) / (bb_mean + 1e-9)) * 100
        squeeze = float(bb_width.iloc[-1]) < 4.0

    # ── NEW 14. OBV slope ────────────────────────────────────────────────────
    obv = _calc_obv(c, v)
    obv_slope = 0.0
    if len(obv) >= 10:
        obv_vals = obv.tail(10).values
        xs = np.arange(len(obv_vals), dtype=float)
        if obv_vals.std() > 0:
            obv_slope = float(np.polyfit(xs, obv_vals / (obv_vals.std() + 1e-9), 1)[0])

    # ── NEW 15. Chaikin Money Flow ────────────────────────────────────────────
    cmf = _calc_cmf(h, l, c, v) if n >= 20 else 0.0

    # ── NEW 16. Volume Profile ────────────────────────────────────────────────
    poc, vah, val_vp = (
        _volume_profile_poc(c.tail(60), v.tail(60))
        if n >= 20
        else (float(c.iloc[-1]), float(c.max()), float(c.min()))
    )

    # ── NEW 17. RSI Divergence ────────────────────────────────────────────────
    bull_div, bear_div = _detect_divergence(rsi_series, c)

    # ── NEW 18. Mean Reversion Z-Score ───────────────────────────────────────
    zscore = _mean_reversion_zscore(c)

    # ── NEW 19. Market Regime ────────────────────────────────────────────────
    regime = _calc_regime(c, CFG["REGIME_PERIOD"])

    # ── NEW 20. Historical price percentile ──────────────────────────────────
    hist_period = min(n, CFG["HIST_DAYS"])
    c_hist = c.tail(hist_period)
    hist_lo, hist_hi = float(c_hist.min()), float(c_hist.max())
    hist_pct = (
        (float(c.iloc[-1]) - hist_lo) / (hist_hi - hist_lo + 1e-9) * 100 if hist_hi > hist_lo else 50.0
    )

    # ── NEW 21. ADX / +DI / -DI (self-computed, direction-aware) ─────────────
    adx_hist = 0.0
    adx_rising = False
    di_bullish = True
    if n >= 20:
        adx_series, plus_di, minus_di = _calc_adx(h, l, c)
        if not adx_series.empty and not pd.isna(adx_series.iloc[-1]):
            adx_hist = float(adx_series.iloc[-1])
            if len(adx_series) >= 6 and not pd.isna(adx_series.iloc[-6]):
                adx_rising = adx_hist - float(adx_series.iloc[-6]) > 1.5
            di_bullish = bool(plus_di.iloc[-1] >= minus_di.iloc[-1])

    # ── NEW 22. Parabolic SAR (trend flip + trailing-stop level) ─────────────
    psar_val = 0.0
    psar_bullish = True
    psar_flip_recent = False
    psar_bars_since_flip = 99
    psar_dist_pct = 0.0
    if n >= 10:
        psar_series, psar_trend = _calc_psar(h, l, c)
        psar_val = float(psar_series.iloc[-1])
        psar_bullish = bool(psar_trend.iloc[-1])
        flips = psar_trend.tail(min(30, n))
        cur = flips.iloc[-1]
        cnt = 0
        for val in reversed(flips.values):
            if val == cur:
                cnt += 1
            else:
                break
        psar_bars_since_flip = cnt - 1
        psar_flip_recent = psar_bars_since_flip <= 2
        last_close = float(c.iloc[-1])
        if last_close > 0:
            psar_dist_pct = (last_close - psar_val) / last_close * 100

    return {
        "has_data": True,
        "n": n,
        "stability": round(stability, 2),
        "avg_vol": float(avg_vol),
        "vol_trend": round(vol_trend, 2),
        "trend_pct_30": round(trend_pct_30, 2),
        "trend_pct_10": round(trend_pct_10, 2),
        "trend_pct_60": round(trend_pct_60, 2),
        "return_3m": round(return_3m, 2),
        "volatility": round(volatility, 2),
        "atr_20": round(atr_20, 4),
        "momentum": round(momentum, 3),
        "rsi_hist": round(rsi_hist, 1),
        "rsi_slope": round(rsi_slope, 2),
        "ema10_slope": round(ema10_slope, 3),
        "ema20_slope": round(ema20_slope, 3),
        "ema50_slope": round(ema50_slope, 3),
        "ema200": round(ema200_val, 2),
        "support_level": round(support_level, 2),
        "resistance_level": round(resistance_level, 2),
        "consec_up": consec_up,
        "consec_down": consec_down,
        "higher_lows": higher_lows,
        "vol_accumulation": vol_accumulation,
        "squeeze": squeeze,
        "triple_bottom": triple_bottom,
        # Wall Street additions
        "obv_slope": round(obv_slope, 4),
        "cmf": round(cmf, 3),
        "poc": round(poc, 2),
        "vah": round(vah, 2),
        "val": round(val_vp, 2),
        "bull_divergence": bull_div,
        "bear_divergence": bear_div,
        "zscore": round(zscore, 2),
        "regime": regime,
        "hist_pct": round(hist_pct, 1),
        # ADX / PSAR
        "adx_hist": round(adx_hist, 1),
        "adx_rising": adx_rising,
        "di_bullish": di_bullish,
        "psar": round(psar_val, 2),
        "psar_bullish": psar_bullish,
        "psar_flip_recent": psar_flip_recent,
        "psar_bars_since_flip": psar_bars_since_flip,
        "psar_dist_pct": round(psar_dist_pct, 2),
        # Trailing daily returns — used for historical VaR in position sizing.
        # Real returns from price_history, not a distributional assumption.
        "ret_series": c.pct_change().dropna().tail(60).tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_resource
def _session():
    s = requests.Session()
    s.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        ),
    )
    s.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    return s


@st.cache_data(ttl=20, show_spinner=False)
def fetch_live() -> list:
    payload = {
        "sort": {"sortBy": "volume", "sortOrder": "desc"},
        "filter": [{"left": "name", "operation": "in_range", "right": KSE100}],
        "markets": ["pakistan"],
        "columns": TV_COLS,
        "range": [0, 500],
    }
    try:
        r = _session().post(TV_URL, json=payload, timeout=15)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return []


def calculate_breadth_from_raw(raw: list) -> Tuple[float, bool, int, int, dict]:
    chgs, prices, highs, lows = [], [], [], []
    total_volume = 0
    adv, dec = 0, 0
    for item in raw:
        d = item.get("d", [])
        if not d or len(d) < 26:
            continue
        sym = d[0]
        chg = safe(d[2])
        price = safe(d[1])
        vol = safe(d[3])
        high = safe(d[24])
        low = safe(d[25])
        if sym in KSE100:
            chgs.append(chg)
            prices.append(price)
            if high > 0:
                highs.append(high)
            if low > 0:
                lows.append(low)
            total_volume += vol
            if chg > 0:
                adv += 1
            elif chg < 0:
                dec += 1

    avg = float(np.median(chgs)) if chgs else 0.0
    avg_price = sum(prices) / len(prices) if prices else 0.0
    max_high = max(highs) if highs else 0.0
    min_low = min(lows) if lows else 0.0
    bull = avg > CFG["BREADTH_MIN"] and adv > dec
    kse = {
        "close": avg_price,
        "change": avg,
        "high": max_high,
        "low": min_low,
        "volume": total_volume,
    }
    return avg, bull, adv, dec, kse


@st.cache_data(ttl=60, show_spinner=False)
def fetch_kse_index() -> Optional[dict]:
    try:
        r = requests.get("https://beta-restapi.sarmaaya.pk/api/indices/overview/KSE100", timeout=8)
        if r.status_code == 200:
            return r.json().get("response", {})
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# RS RANK COMPUTATION — universe-wide
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE — 9-Layer Wall Street Edition
# ══════════════════════════════════════════════════════════════════════════════


def _compute_atr_stop(price, atr, hist_atr, mult):
    eff = atr if atr > price * 0.002 else (hist_atr if hist_atr > 0 else price * 0.015)
    return round(price - mult * eff, 2), eff


def _bb_position(price, bb_low, bb_high, bb_basis):
    span = bb_high - bb_low
    if span <= 0:
        return 0.5
    return max(0.0, min(1.0, (price - bb_low) / span))




# ══════════════════════════════════════════════════════════════════════════════
# INSTITUTIONAL RISK & VALIDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
# Scoped to what price_history / signal_log can actually support: no bid-ask
# spreads (not available from this data source), no stochastic process fitted
# on data too thin to fit honestly, no hand-picked conditional probabilities
# dressed up as calibration. Where there isn't yet enough logged history to
# trust a number, these functions say so explicitly (`calibrated: False`)
# instead of returning a confident-looking guess.

# ── Signal outcome logging & resolution ───────────────────────────────────────

def log_signal(conn: sqlite3.Connection, symbol: str, strategy: str, price: float,
                target: float, stop: float, score: float, layers: Dict[str, float]) -> str:
    """
    Record (or refresh) an open signal episode and return `first_seen` — the
    date this symbol+strategy setup first appeared. A signal re-firing every
    scan isn't a new setup, it's the same one still open; decay and expiry
    are measured against first_seen, not "now".

    entry_price/target/stop are frozen at first detection and never
    overwritten on later scans — they represent the hypothetical entry this
    signal is actually being graded against. Only score/layers/last_seen
    refresh, since those describe "why this setup is still live", not
    "where you would have entered."
    """
    today = datetime.now().strftime("%Y-%m-%d")
    row = conn.execute(
        "SELECT first_seen FROM signal_log WHERE symbol=? AND strategy=? AND status='OPEN' "
        "ORDER BY first_seen DESC LIMIT 1",
        (symbol, strategy),
    ).fetchone()
    layers_json = json.dumps(layers)
    if row:
        first_seen = row[0]
        conn.execute(
            "UPDATE signal_log SET last_seen=?, score=?, layers_json=? "
            "WHERE symbol=? AND strategy=? AND first_seen=?",
            (today, score, layers_json, symbol, strategy, first_seen),
        )
    else:
        first_seen = today
        try:
            conn.execute(
                "INSERT INTO signal_log (symbol, strategy, first_seen, last_seen, entry_price, "
                "target, stop, score, layers_json, status) VALUES (?,?,?,?,?,?,?,?,?,'OPEN')",
                (symbol, strategy, first_seen, today, price, target, stop, score, layers_json),
            )
        except sqlite3.IntegrityError:
            pass  # same-day race with an existing episode row — harmless
    return first_seen


def resolve_pending_signals(conn: sqlite3.Connection) -> int:
    """
    Check every OPEN signal against actual subsequent price action: did it
    hit target first, hit stop first, or run past its strategy's horizon
    unfilled? This is what turns signal_log from a log into real, usable
    backtest data. Runs once per scan — cheap, since it's a handful of open
    rows, not the full KSE-100 universe.
    """
    open_df = pd.read_sql("SELECT * FROM signal_log WHERE status='OPEN'", conn)
    if open_df.empty:
        return 0

    horizons = CFG["SIGNAL_HORIZON_DAYS"]
    resolved = 0

    for _, row in open_df.iterrows():
        horizon = horizons.get(row["strategy"], 15)
        bars = pd.read_sql(
            "SELECT date, high, low, close FROM price_history WHERE symbol=? AND date > ? "
            "ORDER BY date ASC LIMIT ?",
            conn, params=(row["symbol"], row["first_seen"], horizon + 2),
        )
        if bars.empty:
            continue

        hit_target = bars[bars["high"] >= row["target"]]
        hit_stop = bars[bars["low"] <= row["stop"]]
        status = resolved_date = resolved_price = None

        if not hit_target.empty and not hit_stop.empty:
            status = "WIN" if hit_target["date"].iloc[0] <= hit_stop["date"].iloc[0] else "LOSS"
            resolved_date = hit_target["date"].iloc[0] if status == "WIN" else hit_stop["date"].iloc[0]
            resolved_price = row["target"] if status == "WIN" else row["stop"]
        elif not hit_target.empty:
            status, resolved_date, resolved_price = "WIN", hit_target["date"].iloc[0], row["target"]
        elif not hit_stop.empty:
            status, resolved_date, resolved_price = "LOSS", hit_stop["date"].iloc[0], row["stop"]
        elif len(bars) >= horizon:
            status = "EXPIRED"
            resolved_date, resolved_price = bars["date"].iloc[-1], bars["close"].iloc[-1]

        if status:
            risk = row["entry_price"] - row["stop"]
            r_mult = round((resolved_price - row["entry_price"]) / risk, 2) if risk > 0 else 0.0
            conn.execute(
                "UPDATE signal_log SET status=?, resolved_date=?, resolved_price=?, r_multiple=? WHERE id=?",
                (status, resolved_date, resolved_price, r_mult, row["id"]),
            )
            resolved += 1

    if resolved:
        conn.commit()
    return resolved


# ── Vectorized win-rate validator (no per-ticker loops) ───────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _resolve_once_per_hour(_conn: sqlite3.Connection, hour_bucket: str) -> int:
    """
    resolve_pending_signals() only has new work to do after a SYNC — the
    auto-refresh runs every REFRESH_SEC (180s) but price_history doesn't
    change that often. Re-scanning every open signal against unchanged
    data every 3 minutes is pure waste, so this throttles it to once per
    wall-clock hour regardless of refresh cadence.
    """
    return resolve_pending_signals(_conn)


class VectorizedSignalValidator:
    """
    Computes empirical per-layer win rates from resolved signal_log rows.
    Everything past the initial JSON parse is pandas/numpy matrix ops — no
    Python loop over tickers. Layers with too few resolved samples are
    shrunk toward the base rate via Beta-Binomial-style smoothing instead of
    reporting a raw, noisy win rate off a handful of trades.
    """

    def __init__(self, conn: sqlite3.Connection, strategy: str,
                 lookback_days: int = None, min_samples: int = None):
        self.strategy = strategy
        self.lookback_days = lookback_days or CFG["VALIDATOR_LOOKBACK_DAYS"]
        self.min_samples = min_samples or CFG["VALIDATOR_MIN_SAMPLES"]
        self.layer_names: List[str] = []
        self.df = self._load(conn)

    def _load(self, conn: sqlite3.Connection) -> pd.DataFrame:
        cutoff = (datetime.now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        df = pd.read_sql(
            "SELECT * FROM signal_log WHERE strategy=? AND status IN ('WIN','LOSS') "
            "AND resolved_date >= ?",
            conn, params=(self.strategy, cutoff),
        )
        if df.empty:
            return df
        # One unavoidable per-row JSON parse (SQLite has no native JSON
        # column) — everything after this is columnar/matrix ops.
        layer_cols = pd.json_normalize(df["layers_json"].apply(json.loads)).fillna(0.0)
        df = pd.concat([df.reset_index(drop=True), layer_cols.reset_index(drop=True)], axis=1)
        df["win"] = (df["status"] == "WIN").astype(int)
        self.layer_names = list(layer_cols.columns)
        return df

    @property
    def base_rate(self) -> float:
        return 0.5 if self.df.empty else float(self.df["win"].mean())

    @property
    def sample_size(self) -> int:
        return 0 if self.df.empty else len(self.df)

    def layer_stats(self) -> Dict[str, Dict]:
        """Per-layer empirical win rate, Beta-smoothed. Vectorized via boolean masks."""
        stats = {}
        if self.df.empty:
            return stats
        prior_p = self.base_rate
        prior_strength = 4  # pseudo-observations pulling thin samples toward base rate
        for layer in self.layer_names:
            active_mask = self.df[layer] > 0
            n = int(active_mask.sum())
            wins = int(self.df.loc[active_mask, "win"].sum())
            raw_p = wins / n if n > 0 else prior_p
            smoothed_p = (wins + prior_strength * prior_p) / (n + prior_strength)
            stats[layer] = {
                "n": n, "raw_win_rate": round(raw_p, 3),
                "smoothed_p": round(smoothed_p, 3),
                "trusted": n >= self.min_samples,
            }
        return stats

    def correlation_matrix(self) -> pd.DataFrame:
        """Layer-activation correlation — used to discount double-counted evidence."""
        if self.df.empty or len(self.df) < 5:
            return pd.DataFrame()
        active = (self.df[self.layer_names] > 0).astype(int)
        return active.corr().fillna(0.0)


def calculate_posterior_probability(layers_df: pd.DataFrame,
                                     validator: "VectorizedSignalValidator") -> pd.DataFrame:
    """
    Combines each fired signal's active layers into a posterior win
    probability via weight-of-evidence (log-odds) accumulation — the same
    auditable technique used in credit scoring to fuse several imperfect
    binary signals, rather than a black-box model fit on data too thin to
    trust. A correlation-based discount (in place of a full PCA
    orthogonalization, which needs more history than this system has to fit
    safely) prevents redundant, collinear layers from double-counting.

    Operates on the WHOLE current batch of fired signals as one matrix
    operation — no Python loop over tickers.
    """
    n_rows = len(layers_df)
    if n_rows == 0:
        return pd.DataFrame(columns=["posterior", "calibrated", "n_samples"])

    stats = validator.layer_stats()
    base_rate = validator.base_rate
    total_n = validator.sample_size

    if not stats or total_n < CFG["VALIDATOR_MIN_SAMPLES"] * 2:
        # Not enough resolved history yet to calibrate anything for real.
        # Fall back to a mild, explicitly-flagged squash of the additive
        # score rather than fabricate a probability — calibrated=False is
        # the honest signal to the user that this number isn't backed by
        # real outcomes yet.
        posterior = 0.5 + 0.35 * ((layers_df.sum(axis=1) / 100.0).clip(0, 1) - 0.5)
        return pd.DataFrame({
            "posterior": posterior.round(3),
            "calibrated": False,
            "n_samples": total_n,
        }, index=layers_df.index)

    layer_names = [c for c in layers_df.columns if c in stats]
    if not layer_names:
        posterior = pd.Series(base_rate, index=layers_df.index)
        return pd.DataFrame({"posterior": posterior.round(3), "calibrated": False,
                              "n_samples": total_n}, index=layers_df.index)

    active = (layers_df[layer_names] > 0).astype(float)
    corr = validator.correlation_matrix()
    base_logit = math.log(base_rate / (1 - base_rate)) if 0 < base_rate < 1 else 0.0

    woe, discount = {}, {}
    for layer in layer_names:
        s = stats[layer]
        if not s["trusted"]:
            woe[layer] = 0.0  # not enough samples yet — doesn't move the posterior
            discount[layer] = 1.0
            continue
        p = min(max(s["smoothed_p"], 0.02), 0.98)
        woe[layer] = math.log(p / (1 - p)) - base_logit
        if not corr.empty and layer in corr.columns:
            others = [c for c in layer_names if c != layer and c in corr.columns]
            avg_corr = float(corr.loc[layer, others].abs().mean()) if others else 0.0
        else:
            avg_corr = 0.0
        discount[layer] = 1.0 / (1.0 + avg_corr)

    woe_vec = np.array([woe[l] * discount[l] for l in layer_names])
    logit_posterior = base_logit + active[layer_names].values @ woe_vec
    posterior = 1.0 / (1.0 + np.exp(-logit_posterior))
    posterior = np.clip(posterior, 0.05, 0.95)  # never claim false certainty

    return pd.DataFrame({
        "posterior": np.round(posterior, 3),
        "calibrated": True,
        "n_samples": total_n,
    }, index=layers_df.index)


# ── Signal decay ────────────────────────────────────────────────────────────

def apply_signal_decay(score: float, first_seen: str, strategy: str) -> Tuple[float, str]:
    """
    A setup sitting unfilled for days is worth less than a fresh one at the
    same raw score — the conditions that fired it are stale. tau is a
    practical decay constant tied to each strategy's stated holding horizon,
    NOT a fitted Ornstein-Uhlenbeck process — there isn't enough logged
    fill-time data to fit one honestly yet (see VectorizedSignalValidator
    for what IS fitted from real data, once it accumulates).
    """
    tau = CFG["SIGNAL_DECAY_TAU_DAYS"].get(strategy, 4.0)
    try:
        first_dt = datetime.strptime(first_seen, "%Y-%m-%d")
        age_days = max((datetime.now() - first_dt).days, 0)
    except (ValueError, TypeError):
        first_dt = datetime.now()
        age_days = 0
    decay_mult = math.exp(-age_days / tau) if age_days > 0 else 1.0
    decay_mult = max(decay_mult, 0.25)  # floor — stays visible, just deprioritized
    expiry = (first_dt + timedelta(days=tau * 3)).strftime("%Y-%m-%d")
    return round(score * decay_mult, 1), expiry


# ── Liquidity-adjusted, fractional-Kelly position sizing ─────────────────────

def compute_position_sizing(price: float, target: float, stop: float, win_prob: float,
                             adv_shares: float, ret_series: List[float]) -> Dict:
    """
    Fractional-Kelly sizing constrained by (a) a hard per-trade risk cap and
    (b) a participation-rate liquidity cap off average daily volume — the
    standard first-order institutional liquidity guardrail used before any
    spread/impact model. No bid-ask spread or market-impact model is used
    because this data pipeline has no spread data; inventing one would be
    worse than not modeling impact at all. VaR is real historical VaR from
    the stock's own trailing daily returns (which we do have) — not a
    distributional assumption.
    """
    capital = CFG["ACCOUNT_CAPITAL"]
    risk, reward = price - stop, target - price
    if risk <= 0 or reward <= 0 or price <= 0:
        return {"kelly_fraction": 0.0, "position_shares": 0, "position_value": 0.0,
                "max_capacity_shares": 0, "illiquid": False, "var_1d_95": 0.0,
                "risk_pct_capital": 0.0}

    b = reward / risk
    f_star = max(0.0, win_prob - (1 - win_prob) / b)
    f_used = f_star * CFG["FRACTIONAL_KELLY"]

    shares_by_risk = (capital * CFG["MAX_RISK_PCT"]) / risk
    shares_by_kelly = (capital * f_used) / price
    shares = min(shares_by_risk, shares_by_kelly)

    capacity_shares = adv_shares * CFG["ADV_PARTICIPATION_CAP"]
    illiquid = capacity_shares > 0 and shares > capacity_shares
    shares_final = max(0, int(min(shares, capacity_shares) if capacity_shares > 0 else shares))

    position_value = shares_final * price
    var_1d_95 = 0.0
    if ret_series and len(ret_series) >= 20:
        var_pct = -float(np.percentile(ret_series, 5))
        var_1d_95 = round(position_value * max(var_pct, 0), 0)

    return {
        "kelly_fraction": round(f_star, 3),
        "position_shares": shares_final,
        "position_value": round(position_value, 0),
        "max_capacity_shares": int(capacity_shares),
        "illiquid": illiquid,
        "var_1d_95": var_1d_95,
        "risk_pct_capital": round((shares_final * risk) / capital * 100, 2) if capital > 0 else 0.0,
    }


# ── Market regime via rolling Mahalanobis distance ────────────────────────────
# GMM is explicitly the alternative offered alongside Mahalanobis distance —
# we use Mahalanobis because fitting a stable 4-component GMM needs far more
# history than a 1-2 year KSE-100 daily series safely supports without
# overfitting noise into "regimes."

@st.cache_data(ttl=1800, show_spinner=False)
def compute_market_feature_series(_conn: sqlite3.Connection, lookback_days: int = 200) -> pd.DataFrame:
    """
    Daily market-wide features (breadth, average trading range, participation)
    aggregated across the whole KSE-100 universe — a single vectorized
    groupby('date') over price_history, not a per-symbol loop.
    """
    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    df = pd.read_sql(
        "SELECT date, symbol, close, high, low, volume FROM price_history WHERE date >= ?",
        _conn, params=(cutoff,),
    )
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])
    df["ret"] = df.groupby("symbol")["close"].pct_change()
    df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["vol_ma20"] = df.groupby("symbol")["volume"].transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )
    df["rel_vol"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)

    daily = df.groupby("date").agg(
        adv_pct=("ret", lambda s: float((s > 0).mean()) if s.notna().any() else np.nan),
        avg_range=("range_pct", "mean"),
        participation=("rel_vol", "mean"),
    ).dropna()
    return daily


def classify_regime_mahalanobis(feature_df: pd.DataFrame, window: int = 60) -> Dict:
    """
    Classifies the current market state into one of 4 quadrants using a
    rolling Mahalanobis distance from the trailing feature centroid, crossed
    with the trailing volatility level.
    """
    if feature_df.empty or len(feature_df) < window + 10:
        return {"regime": "NEUTRAL (insufficient history)", "distance": 0.0,
                "scalar": 1.0, "confidence": "low"}

    cols = ["adv_pct", "avg_range", "participation"]
    X = feature_df[cols].tail(window + 10).values
    window_data = X[-window:]
    mean = window_data.mean(axis=0)
    cov = np.cov(window_data, rowvar=False)
    try:
        inv_cov = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return {"regime": "NEUTRAL", "distance": 0.0, "scalar": 1.0, "confidence": "low"}

    latest = X[-1]
    diff = latest - mean
    distance = float(np.sqrt(max(diff @ inv_cov @ diff.T, 0)))

    dist_series = []
    for i in range(max(len(window_data) - 10, 0), len(window_data)):
        d = window_data[i] - mean
        dist_series.append(float(np.sqrt(max(d @ inv_cov @ d.T, 0))))
    distance_rising = len(dist_series) >= 2 and dist_series[-1] > np.mean(dist_series[:-1])
    vol_level_high = latest[1] > np.median(window_data[:, 1])

    if not vol_level_high and distance_rising:
        regime, scalar = "Low-Vol Expansion (Bullish Accumulation)", 1.05
    elif vol_level_high and distance_rising:
        regime, scalar = "High-Vol Expansion (Trend Exhaustion)", 0.85
    elif not vol_level_high and not distance_rising:
        regime, scalar = "Low-Vol Compression (Mean-Reverting)", 0.95
    else:
        regime, scalar = "High-Vol Compression (Distribution/Choppy)", 0.75

    confidence = "high" if len(feature_df) >= window * 2 else "moderate"
    return {"regime": regime, "distance": round(distance, 2), "scalar": scalar, "confidence": confidence}


@st.cache_data(ttl=1800, show_spinner=False)
def get_current_regime(_conn: sqlite3.Connection) -> Dict:
    features = compute_market_feature_series(_conn)
    return classify_regime_mahalanobis(features)


def _apply_risk_engine(rows: List[Dict], strategy: str, conn: sqlite3.Connection,
                        regime: Dict) -> List[Dict]:
    """
    Batch post-processing step for one strategy's fired signals: vectorized
    posterior win-probability (all rows at once, one matrix op), then
    per-row decay and Kelly sizing (simple arithmetic over an already-
    filtered handful of signals — not a loop over the full KSE-100 universe).
    """
    if not rows:
        return rows
    layers_df = pd.DataFrame([r["_layers"] for r in rows]).fillna(0.0)
    validator = VectorizedSignalValidator(conn, strategy)
    post_df = calculate_posterior_probability(layers_df, validator)

    for i, r in enumerate(rows):
        posterior = float(post_df["posterior"].iloc[i])
        calibrated = bool(post_df["calibrated"].iloc[i])
        decayed_score, expiry = apply_signal_decay(r["Score"], r["_first_seen"], strategy)
        sizing = compute_position_sizing(
            r["Price"], r["Target"], r["Stop"], posterior, r["_adv_vol"], r["_ret_series"]
        )
        r["Win Prob %"] = round(posterior * 100, 1)
        r["Calibrated"] = calibrated
        r["Decayed Score"] = decayed_score
        r["Confidence Expiry"] = expiry
        r["Kelly Shares"] = sizing["position_shares"]
        r["Position Value"] = sizing["position_value"]
        r["Max Capacity (sh)"] = sizing["max_capacity_shares"]
        r["Illiquid"] = sizing["illiquid"]
        r["VaR 1D 95%"] = sizing["var_1d_95"]
        r["Risk % Capital"] = sizing["risk_pct_capital"]
        r["Regime"] = regime["regime"]
        r["Regime-Adj Score"] = round(r["Score"] * regime["scalar"], 1)
        for k in ("_layers", "_first_seen", "_adv_vol", "_ret_series"):
            r.pop(k, None)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# DIP BUY SCORER (NEW - Refactored)
# ─────────────────────────────────────────────────────────────────────────────
def score_dip(
    price, rsi, stoch_k, bb_low, bb_high, low7d, atr, vol, avg_vol, hist
) -> Tuple[int, List[str], float, float]:
    """Scores stocks in an uptrend that are currently pulling back."""

    reasons = []

    # Conditions for a dip
    near_7d_low = low7d > 0 and (price / low7d - 1) * 100 < 5
    is_oversold = (rsi <= 45) or (stoch_k <= 30)
    is_at_support = (price <= bb_low * 1.02) if bb_low > 0 else False

    if not (is_oversold or is_at_support or near_7d_low):
        return 0, [], 0, 0

    # Scoring logic
    rsi_pts = max(0, min(30, (50 - rsi) * 1.7))
    bb_range = (bb_high - bb_low) if bb_high > bb_low else price * 0.1
    bb_pts = max(0, min(22, 22 * (1 - (price - bb_low) / bb_range))) if bb_low > 0 else 0
    vol_pts = min(13, 13 * (vol / avg_vol)) if avg_vol > 0 else 0
    stab_pts = min(13, hist.get("stability", 5) * 1.3)
    flow_pts = max(0, min(9, hist.get("cmf", 0) * 27 + 4.5))
    # A dip is only worth buying if the underlying trend hasn't actually
    # broken — a pullback that just crossed below PSAR / DI- is a trend
    # change wearing a "dip" costume.
    trend_pts = 0
    if hist.get("psar_bullish"):
        trend_pts += 6
    else:
        trend_pts -= 8
    if hist.get("di_bullish", True):
        trend_pts += 4
    trend_pts = _clamp(trend_pts, -8, 10)

    score = round(rsi_pts + bb_pts + vol_pts + stab_pts + flow_pts + trend_pts)
    if score <= 0:
        return 0, [], 0, 0

    # Reasons
    if rsi <= 35: reasons.append("Oversold RSI")
    elif rsi <= 45: reasons.append("RSI Pullback")
    if stoch_k <= 25: reasons.append("Stoch Oversold")
    if is_at_support: reasons.append("BB Support")
    if near_7d_low: reasons.append("Near 7D Low")
    if hist.get("bull_divergence"): reasons.append("RSI Divergence")
    if hist.get("cmf", 0) > 0.05: reasons.append("CMF Positive")
    if hist.get("psar_bullish"): reasons.append("Trend Intact (PSAR)")
    else: reasons.append("PSAR Broken — caution")

    # Target and Stop
    stop, eff_atr = _compute_atr_stop(price, atr, hist.get("atr_20", 0), 1.0)
    psar_stop = hist.get("psar", 0)
    if hist.get("psar_bullish") and 0 < psar_stop < price:
        stop = max(stop, round(psar_stop, 2))
    target_pct = _clamp((2.0 * eff_atr / price) * 100, 3.0, 8.0) if eff_atr > 0 else 4.0
    target = round(price * (1 + target_pct / 100), 2)

    if not reasons: reasons.append("Pullback")

    return score, reasons, target, stop
# ─────────────────────────────────────────────────────────────────────────────
# INTRADAY SCALP SCORER
# ─────────────────────────────────────────────────────────────────────────────
def score_intraday(
    price,
    change,
    rsi,
    macd,
    macd_sig,
    macd_hist,
    ema5,
    ema10,
    vwap,
    adx,
    stoch_k,
    stoch_d,
    vol,
    avg_vol,
    atr,
    bb_low,
    bb_high,
    bb_basis,
    open_p,
    high_d,
    low_d,
    bullish,
    mkt_chg,
    hist,
    rsi_prev,
) -> Tuple[int, List[str], float, float, str, int, Dict[str, float]]:

    reasons = []
    layers_active = 0

    vol_ratio = vol / avg_vol if avg_vol > 0 else 0
    if price < CFG["MIN_PRICE"]:
        return 0, [], 0, 0, "D", 0, {}
    if vol < CFG["MIN_VOLUME"]:
        return 0, [], 0, 0, "D", 0, {}
    if vol_ratio < 0.7:
        return 0, [], 0, 0, "D", 0, {}
    if price < vwap * 0.98:
        return 0, [], 0, 0, "D", 0, {}

    # ── L1: Multi-TF Trend (18) ──────────────────────────────────────────────
    L1 = 0
    if open_p > 0 and price > open_p and change > 1.0:
        L1 += 4
        reasons.append("Gap Up")
    if ema5 > 0 and ema10 > 0:
        if price > ema5 > ema10:
            L1 += 9
            reasons.append("EMA Alignment")
        elif price > ema5 and price > ema10:
            L1 += 5
            reasons.append("Above EMAs")
    # ADX only means something once direction is confirmed — a strong reading
    # with -DI in charge is a strong downtrend, not a buy signal.
    di_ok = hist.get("di_bullish", True)
    if adx >= 35 and di_ok:
        L1 += 7
        reasons.append(f"ADX {adx:.0f} Strong")
    elif adx >= 25 and di_ok:
        L1 += 4
        reasons.append(f"ADX {adx:.0f}")
    elif adx < 20:
        L1 -= 4  # choppy / no trend to ride
    elif adx >= 25 and not di_ok:
        L1 -= 3  # trending, but against us
    if hist.get("adx_rising"):
        L1 += 2
        reasons.append("ADX Rising")
    # Parabolic SAR — fresh flip = timing edge, price below SAR = trend broken
    if hist.get("psar_bullish"):
        if hist.get("psar_flip_recent"):
            L1 += 5
            reasons.append("PSAR Flip Bullish")
        else:
            L1 += 2
    else:
        L1 -= 4
        reasons.append("Below PSAR")
    # 200-day regime context
    if hist["regime"] == "BULL":
        L1 += 2
    elif hist["regime"] == "BEAR":
        L1 -= 3
    L1 = _clamp(L1, -8, LAYER_BUDGET_INTRA["trend"])
    if L1 > 0:
        layers_active += 1

    # ── L2: Momentum Cascade + RS Rank (17) ──────────────────────────────────
    L2 = 0
    rs_alpha = change - mkt_chg
    if rs_alpha > 1.5:
        L2 += 4
        reasons.append(f"RS +{rs_alpha:.1f}%")
    if macd > macd_sig and macd_hist > 0 and macd > 0:
        L2 += 5
        reasons.append("MACD Bull")
    elif macd < macd_sig:
        L2 -= 3
    rsi_delta = rsi - rsi_prev if rsi_prev > 0 else 0
    if rsi < 40 and rsi_delta > 4:
        L2 += 7
        reasons.append("Oversold Reversal")
    elif 52 < rsi < 75:
        L2 += 4
        reasons.append(f"RSI {rsi:.0f}")
        if rsi_delta > 3:
            L2 += 2
    elif rsi >= 75:
        L2 -= 6
        reasons.append("Overbought")
    # NOTE: daily-RSI divergence deliberately excluded here (unlike swing/
    # long-term) — this scorer runs on 15-min indicators for timing, and a
    # multi-day divergence pattern has no bearing on a same-day scalp entry.
    if stoch_k > stoch_d and 30 < stoch_k < 85:
        L2 += 3
        reasons.append("Stoch Cross")
    elif stoch_k > 85:
        L2 -= 2
    L2 = _clamp(L2, -5, LAYER_BUDGET_INTRA["momentum"])
    if L2 > 0:
        layers_active += 1

    # ── L3: Volume Footprint (13) ─────────────────────────────────────────────
    L3 = 0
    if vol_ratio >= CFG["INST_VOL_X"]:
        L3 += 10
        reasons.append(f"{vol_ratio:.1f}x Vol")
    elif vol_ratio >= 2.0:
        L3 += 7
        reasons.append(f"{vol_ratio:.1f}x Vol")
    elif vol_ratio >= 1.5:
        L3 += 4
    elif vol_ratio >= 1.0:
        L3 += 2
    if hist["vol_accumulation"]:
        L3 += 3
        reasons.append("Accumulation")
    L3 = _clamp(L3, 0, LAYER_BUDGET_INTRA["volume"])
    if L3 > 0:
        layers_active += 1

    # ── L4: Price Pattern + Mean Reversion (12) ──────────────────────────────
    L4 = 0
    vwap_dist = (price - vwap) / vwap * 100 if vwap > 0 else 99
    if -0.5 < vwap_dist < 0.5:
        L4 += 7
        reasons.append("VWAP Bounce")
    elif 0.5 <= vwap_dist < 1.2:
        L4 += 4
        reasons.append("VWAP Edge")
    elif vwap_dist > 3.0:
        L4 -= 3
    # POC magnet
    if hist["poc"] > 0:
        poc_dist = abs(price - hist["poc"]) / hist["poc"] * 100
        if poc_dist < 1.5:
            L4 += 4
            reasons.append("POC Zone")
    # Z-score: prefer moderate compression
    z = hist["zscore"]
    if -0.5 < z < 0.5:
        L4 += 2  # compressed — ready to move
    elif z > 2.5:
        L4 -= 3  # stretched above mean
    if bb_basis > 0:
        dist_basis = (price / bb_basis - 1) * 100
        if 0 < dist_basis < 1.5:
            L4 += 3
            reasons.append("Mean Reversion")
    day_range = high_d - low_d
    if day_range > 0:
        candle_pos = (price - low_d) / day_range
        if candle_pos > 0.7:
            L4 += 2
            reasons.append("Day High Zone")
        elif candle_pos < 0.3:
            L4 -= 2
    if hist["squeeze"] and change > 1.0:
        L4 += 4
        reasons.append("BB Squeeze Break")
    bb_pos = _bb_position(price, bb_low, bb_high, bb_basis)
    if bb_pos > 0.92:
        L4 -= 3
    L4 = _clamp(L4, -5, LAYER_BUDGET_INTRA["pattern"])
    if L4 > 0:
        layers_active += 1

    # ── L5: Breadth (8) ───────────────────────────────────────────────────────
    L5 = 0
    if bullish:
        L5 += 6
    else:
        L5 -= 5
        reasons.append("Bearish Breadth")
    L5 = _clamp(L5, -5, LAYER_BUDGET_INTRA["breadth"])
    if L5 > 0:
        layers_active += 1

    # ── L6: Volatility State (8) ──────────────────────────────────────────────
    L6 = 0
    hvol = hist["volatility"]
    if 20 < hvol < 45:
        L6 += 6
    elif 45 <= hvol < 65:
        L6 += 3
    elif hvol >= 65:
        L6 -= 4
    elif hvol <= 10:
        L6 -= 2
    if hist["consec_up"] >= 3:
        L6 += 2
    L6 = _clamp(L6, -5, LAYER_BUDGET_INTRA["volatility"])
    if L6 > 0:
        layers_active += 1

    # ── L7: Historical Quality (10) ───────────────────────────────────────────
    L7 = 0
    if hist["momentum"] > 0.5:
        L7 += 4
        reasons.append("Hist Momentum")
    if hist["trend_pct_10"] > 2:
        L7 += 3
    if hist["stability"] >= 6:
        L7 += 3
    elif hist["stability"] < 3:
        L7 -= 3
    L7 = _clamp(L7, -3, LAYER_BUDGET_INTRA["historical"])
    if L7 > 0:
        layers_active += 1

    # ── L8: Institutional Flow (10) ── NEW ──────────────────────────────────
    L8 = 0
    if hist["obv_slope"] > 0.1:
        L8 += 5
        reasons.append("OBV Rising")
    elif hist["obv_slope"] < -0.1:
        L8 -= 3
    if hist["cmf"] > 0.1:
        L8 += 5
        reasons.append(f"CMF {hist['cmf']:.2f}")
    elif hist["cmf"] < -0.1:
        L8 -= 3
    L8 = _clamp(L8, -4, LAYER_BUDGET_INTRA["flow"])
    if L8 > 0:
        layers_active += 1

    # ── L9: Market Regime (4) ── NEW ─────────────────────────────────────────
    L9 = 0
    regime = hist["regime"]
    if regime == "BULL":
        L9 += 4
    elif regime == "BEAR":
        L9 -= 3
    L9 = _clamp(L9, -3, LAYER_BUDGET_INTRA["regime"])
    if L9 > 0:
        layers_active += 1

    score = max(0, L1 + L2 + L3 + L4 + L5 + L6 + L7 + L8 + L9)
    if layers_active < 4:
        score = min(score, THRESH_INTRA - 1)

    stop, eff_atr = _compute_atr_stop(price, atr, hist["atr_20"], 0.6)
    psar_stop = hist.get("psar", 0)
    if hist.get("psar_bullish") and 0 < psar_stop < price:
        stop = max(stop, round(psar_stop, 2))  # tighter of ATR floor vs. PSAR level
    raw_target_pct = (2.0 * eff_atr / price) * 100
    target_pct = _clamp(raw_target_pct, 2.0, 6.0)
    target = round(price * (1 + target_pct / 100), 2)

    layers = {
        "trend": L1, "momentum": L2, "volume": L3, "pattern": L4,
        "breadth": L5, "volatility": L6, "historical": L7, "flow": L8, "regime": L9,
    }
    return score, reasons, target, stop, _grade(score, layers_active), layers_active, layers


# ─────────────────────────────────────────────────────────────────────────────
# SWING TRADE SCORER
# ─────────────────────────────────────────────────────────────────────────────
def score_swing(
    price,
    change,
    rsi,
    macd,
    macd_sig,
    macd_hist,
    ema5,
    ema10,
    ema20,
    ema25,
    ema50,
    vwap,
    adx,
    atr,
    stoch_k,
    stoch_d,
    bb_low,
    bb_high,
    bb_basis,
    vol,
    avg_vol,
    chg1w,
    chg1m,
    low1m,
    high1m,
    bullish,
    mkt_chg,
    rsi_prev,
    hist,
) -> Tuple[int, List[str], float, float, str, int, Dict[str, float]]:

    reasons = []
    layers_active = 0

    if price < CFG["MIN_PRICE"]:
        return 0, [], 0, 0, "D", 0, {}
    if vol < CFG["MIN_VOLUME"] * 0.8:
        return 0, [], 0, 0, "D", 0, {}  # No change here, just for context
    if price < ema50 * 0.985:
        return 0, [], 0, 0, "D", 0, {}
    if adx < 12:
        return 0, [], 0, 0, "D", 0, {}

    # ── L1: Multi-TF Trend (18) ──────────────────────────────────────────────
    L1 = 0
    if ema5 > 0 and price > ema20 > ema50:
        L1 += 14
        reasons.append("EMA Trend (20/50)")
    elif price > ema20 and price > ema50:
        L1 += 10
        reasons.append("Above EMAs (20/50)")
    elif price > ema50:
        L1 += 5
    if hist["higher_lows"]:
        L1 += 4
        reasons.append("Higher Lows")
    di_ok = hist.get("di_bullish", True)
    if adx >= 30 and di_ok:
        L1 += 4
        reasons.append(f"ADX {adx:.0f} Strong")
    elif adx >= 25 and di_ok:
        L1 += 2
        reasons.append(f"ADX {adx:.0f}")
    elif adx >= 25 and not di_ok:
        L1 -= 3  # strong trend, wrong direction — avoid the chop-in-disguise trap
    if hist.get("adx_rising") and di_ok:
        L1 += 2
        reasons.append("ADX Rising")
    # Parabolic SAR — a fresh bullish flip is the actual entry-timing signal;
    # being below PSAR means the trend this whole layer is scoring is already broken.
    if hist.get("psar_bullish"):
        if hist.get("psar_flip_recent"):
            L1 += 4
            reasons.append("PSAR Flip Bullish")
        else:
            L1 += 2
    else:
        L1 -= 5
        reasons.append("Below PSAR")
    # 200-day anchoring
    if hist["ema200"] > 0 and price > hist["ema200"]:
        L1 += 2
        reasons.append("Above MA200")
    if hist["regime"] == "BULL":
        L1 += 2
    elif hist["regime"] == "BEAR":
        L1 -= 3
    if ema25 > 0 and price > ema25 > ema50:
        L1 += 5
        reasons.append("Price > EMA25 > EMA50")
    L1 = _clamp(L1, -8, LAYER_BUDGET_SWING["trend"])
    if L1 > 0:
        layers_active += 1

    # ── L2: Momentum + RS Rank (17) ──────────────────────────────────────────
    L2 = 0
    rs_alpha = change - mkt_chg
    if rs_alpha > 1.0:
        L2 += 3
        reasons.append(f"RS +{rs_alpha:.1f}%")
    rsi_delta = rsi - rsi_prev if rsi_prev > 0 else 0
    if rsi < 42 and rsi_delta > 3:
        L2 += 7
        reasons.append("Oversold Reversal")
    elif 45 < rsi < 60:
        L2 += 5
        reasons.append(f"RSI {rsi:.0f} Ideal")
    elif rsi > 72:
        L2 -= 6
        reasons.append("RSI Overbought")
    if hist["bull_divergence"]:
        L2 += 4
        reasons.append("RSI Divergence")
    if macd > macd_sig and macd_hist > 0:
        L2 += 5
        reasons.append("MACD Bullish")
        if macd > 0:
            L2 += 2
            reasons.append("MACD > 0")
    elif macd < macd_sig:
        L2 -= 4
    if stoch_k > stoch_d and stoch_k < 80:
        L2 += 3
        reasons.append("Stoch Cross")
    if hist["ema10_slope"] > 0 and hist["ema20_slope"] > 0:
        L2 += 2
        reasons.append("Slopes Rising")
    if chg1w > 2:
        L2 += 3
        reasons.append("Weekly Momentum")
    elif chg1w > 0:
        L2 += 1
    elif chg1w < -5:
        L2 -= 3
    L2 = _clamp(L2, -5, LAYER_BUDGET_SWING["momentum"])
    if L2 > 0:
        layers_active += 1

    # ── L3: Volume (13) ───────────────────────────────────────────────────────
    L3 = 0
    vol_ratio = vol / avg_vol if avg_vol > 0 else 0
    if vol_ratio >= 2.5:
        L3 += 9
        reasons.append(f"{vol_ratio:.1f}x Vol")
    elif vol_ratio >= 1.7:
        L3 += 6
        reasons.append(f"{vol_ratio:.1f}x Vol")
    elif vol_ratio >= 1.2:
        L3 += 2
    if hist["vol_accumulation"]:
        L3 += 4
        reasons.append("Accumulation")
    L3 = _clamp(L3, 0, LAYER_BUDGET_SWING["volume"])
    if L3 > 0:
        layers_active += 1

    # ── L4: Price Pattern + Mean Reversion (12) ──────────────────────────────
    L4 = 0
    bb_pos = _bb_position(price, bb_low, bb_high, bb_basis)
    if bb_pos < 0.2 and change > 0:
        L4 += 5
        reasons.append("BB Bounce")

    if high1m > low1m > 0:
        range1m = high1m - low1m
        pos1m = (price - low1m) / range1m
        if 0.08 < pos1m < 0.30:
            L4 += 7
            reasons.append("Early Cycle")
    if hist["support_level"] > 0 and price > 0:
        support_gap = (price / hist["support_level"] - 1) * 100
        if 0 < support_gap < 4:
            L4 += 4
            reasons.append("Near Support")
    # Value Area Low from volume profile
    if hist["val"] > 0:
        val_dist = (price - hist["val"]) / hist["val"] * 100
        if 0 < val_dist < 3:
            L4 += 4
            reasons.append("VA Low")
    if bb_basis > 0:
        dist_basis = (price / bb_basis - 1) * 100
        if -1 < dist_basis < 2:
            L4 += 4
            reasons.append("Mean Reversion")
    if hist["squeeze"]:
        L4 += 4
        reasons.append("BB Squeeze")
    z = hist["zscore"]
    if -1.5 < z < 0:
        L4 += 3
        reasons.append("Compressed")
    elif z < -2.0:
        L4 -= 2
    L4 = _clamp(L4, 0, LAYER_BUDGET_SWING["pattern"])
    if L4 > 0:
        layers_active += 1

    # ── L5: Breadth (8) ───────────────────────────────────────────────────────
    L5 = 0
    if bullish:
        L5 += 6
    else:
        L5 -= 5
        reasons.append("Bearish Breadth")
    L5 = _clamp(L5, -5, LAYER_BUDGET_SWING["breadth"])
    if L5 > 0:
        layers_active += 1

    # ── L6: Volatility (8) ────────────────────────────────────────────────────
    L6 = 0
    hvol = hist["volatility"]
    if 15 < hvol < 45:
        L6 += 6
    elif 45 <= hvol < 60:
        L6 += 2
    elif hvol >= 70:
        L6 -= 4
    elif hvol <= 8:
        L6 -= 2
    if hist["consec_up"] >= 2:
        L6 += 2
    elif hist["consec_down"] >= 4:
        L6 -= 3
    if hist["trend_pct_10"] > 2:
        L6 += 1
    L6 = _clamp(L6, -5, LAYER_BUDGET_SWING["volatility"])
    if L6 > 0:
        layers_active += 1

    # ── L7: Historical Quality (10) ───────────────────────────────────────────
    L7 = 0
    if hist["stability"] >= 7:
        L7 += 5
        reasons.append("Stable")
    elif hist["stability"] >= 5:
        L7 += 2
    elif hist["stability"] < 3:
        L7 -= 3
    if hist["momentum"] > 1.0:
        L7 += 3
        reasons.append("Hist Momentum")
    elif hist["momentum"] < -1.0:
        L7 -= 2
    if hist["trend_pct_30"] > 8:
        L7 += 3
        reasons.append("30d Uptrend")
    # Historical position
    hist_p = hist["hist_pct"]
    if 25 < hist_p < 65:
        L7 += 2
        reasons.append(f"Hist@{hist_p:.0f}%")
    elif hist_p > 90:
        L7 -= 2
    L7 = _clamp(L7, -3, LAYER_BUDGET_SWING["historical"])
    if L7 > 0:
        layers_active += 1

    # ── L8: Institutional Flow (10) ── NEW ───────────────────────────────────
    L8 = 0
    if hist["obv_slope"] > 0.15:
        L8 += 5
        reasons.append("OBV Rising")
    elif hist["obv_slope"] < -0.1:
        L8 -= 3
    if hist["cmf"] > 0.1:
        L8 += 5
        reasons.append(f"CMF {hist['cmf']:.2f}")
    elif hist["cmf"] > 0:
        L8 += 2
        reasons.append("CMF Positive")
    elif hist["cmf"] < -0.15:
        L8 -= 3
    L8 = _clamp(L8, -4, LAYER_BUDGET_SWING["flow"])
    if L8 > 0:
        layers_active += 1

    # ── L9: Market Regime (4) ── NEW ─────────────────────────────────────────
    L9 = 0
    if hist["regime"] == "BULL":
        L9 += 4
    elif hist["regime"] == "BEAR":
        L9 -= 3
    L9 = _clamp(L9, -3, LAYER_BUDGET_SWING["regime"])
    if L9 > 0:
        layers_active += 1

    score = max(0, L1 + L2 + L3 + L4 + L5 + L6 + L7 + L8 + L9)
    if layers_active < 4:
        score = min(score, THRESH_SWING - 1)

    stop, eff_atr = _compute_atr_stop(price, atr, hist["atr_20"], 1.5)
    psar_stop = hist.get("psar", 0)
    if hist.get("psar_bullish") and 0 < psar_stop < price:
        stop = max(stop, round(psar_stop, 2))
    raw_target_pct = (3.0 * eff_atr / price) * 100
    target_pct = _clamp(raw_target_pct, 4.0, 12.0)
    target = round(price * (1 + target_pct / 100), 2)

    layers = {
        "trend": L1, "momentum": L2, "volume": L3, "pattern": L4,
        "breadth": L5, "volatility": L6, "historical": L7, "flow": L8, "regime": L9,
    }
    return score, reasons, target, stop, _grade(score, layers_active), layers_active, layers


# ─────────────────────────────────────────────────────────────────────────────
# LONG-TERM INVESTMENT SCORER
# ─────────────────────────────────────────────────────────────────────────────
def score_longterm(
    price,
    rsi,
    macd,
    macd_sig,
    macd_hist,
    ema20,
    ema50,
    stoch_k,
    stoch_d,
    bb_low,
    bb_high,
    bb_basis,
    vol,
    avg_vol,
    chg1w,
    chg1m,
    low1m,
    high1m,
    sector,
    rsi_prev,
    hist,
) -> Tuple[int, List[str], float, float, str, int, Dict[str, float]]:

    reasons = []
    layers_active = 0
    quality = SECTORS.get(sector, {"quality": 5})["quality"]

    if price < CFG["MIN_PRICE"]:
        return 0, [], 0, 0, "D", 0, {}
    if quality < 6:
        return 0, [], 0, 0, "D", 0, {}
    if hist["stability"] < 2.0:
        return 0, [], 0, 0, "D", 0, {}

    # ── L1: Sector Quality + Trend (18) ──────────────────────────────────────
    L1 = {9: 10, 8: 8, 7: 6}.get(quality, 3)
    if hist["ema200"] > 0 and price > hist["ema200"]:
        L1 += 5
        reasons.append("Above EMA200")
    if price > ema50: # Using 50-day MA for trend health
        L1 += 5
        reasons.append("Above EMA50")
    if hist["regime"] == "BULL":
        L1 += 2
    elif hist["regime"] == "BEAR":
        L1 -= 4
    # For long-term entries we care less about a fresh PSAR flip (too noisy at
    # this horizon) and more about whether the multi-month trend is intact.
    if hist.get("psar_bullish"):
        L1 += 2
        reasons.append("Above PSAR")
    elif hist.get("adx_hist", 0) >= 25 and not hist.get("di_bullish", True):
        L1 -= 4  # established downtrend, not just noise — don't average in
        reasons.append("Downtrend (ADX/DI)")
    L1 = _clamp(L1, -4, LAYER_BUDGET_LONG["trend"])
    if L1 > 0:
        layers_active += 1

    # ── L2: Value Zone + RS Rank (17) ────────────────────────────────────────
    L2 = 0
    if high1m > low1m > 0:
        dist_low = (price / low1m - 1) * 100
        # A price sitting AT the monthly low (dist_low near 0) with RSI still
        # falling is a breakdown in progress, not a value entry — the old
        # unbounded "< 10" branch rewarded both cases identically. Require a
        # floor on distance from the low (some basing has happened) and on
        # RSI (downside momentum has stabilized) before crediting it as value.
        if hist["triple_bottom"] and dist_low < 5:
            L2 += 13
            reasons.append("Triple Bottom")
        elif 0.3 < dist_low < 4 and rsi > 28:
            L2 += 8
            reasons.append("Near Monthly Low")
        elif 1.0 < dist_low < 10 and rsi > 32:
            L2 += 5
            reasons.append("Value Zone")
    rsi_delta = rsi - rsi_prev if rsi_prev > 0 else 0
    if 20 < rsi < 35:
        L2 += 5
        reasons.append("Oversold")
    elif 40 <= rsi < 50:
        L2 += 3
        reasons.append(f"RSI {rsi:.0f} Reset")
    elif 50 <= rsi < 60:
        L2 += 1
    elif rsi > 75:
        L2 -= 8
        reasons.append("Expensive")
    if rsi_delta > 3:
        L2 += 3
        reasons.append("Momentum Turn")
    if hist["bull_divergence"]:
        L2 += 4
        reasons.append("RSI Divergence")
    if chg1m < -25:
        L2 -= 5
        reasons.append("Falling Knife")
    L2 = _clamp(L2, -10, LAYER_BUDGET_LONG["momentum"])
    if L2 > 0:
        layers_active += 1

    # ── L3: Volume (13) ───────────────────────────────────────────────────────
    L3 = 0
    vol_ratio = vol / avg_vol if avg_vol > 0 else 0
    if hist["vol_accumulation"]:
        L3 += 7
        reasons.append("Accumulation")
    if vol_ratio >= 1.5 and rsi < 50:
        L3 += 4
        reasons.append("Vol + RSI Reset")
    elif vol_ratio >= 1.2:
        L3 += 2
    # CMF as proxy for institutional buying
    if hist["cmf"] > 0.05:
        L3 += 2
        reasons.append("CMF+")
    L3 = _clamp(L3, 0, LAYER_BUDGET_LONG["volume"])
    if L3 > 0:
        layers_active += 1

    # ── L4: Price Pattern + Value Area (12) ──────────────────────────────────
    L4 = 0
    if bb_basis > 0 and (price / bb_basis - 1) * 100 < 0:
        L4 += 5
        reasons.append("Below Mean")
    if hist["support_level"] > 0 and price > 0:
        gap = (price / hist["support_level"] - 1) * 100
        if 0 < gap < 3:
            L4 += 4
            reasons.append("Near Support")
    bb_pos = _bb_position(price, bb_low, bb_high, bb_basis)
    if bb_pos < 0.15:
        L4 += 5
        reasons.append("Near BB Low")
    elif bb_pos < 0.35:
        L4 += 2
    if hist["squeeze"]:
        L4 += 3
        reasons.append("BB Squeeze")
    if hist["higher_lows"]:
        L4 += 3
        reasons.append("Higher Lows")
    # Volume Profile VAL
    if hist["val"] > 0:
        val_dist = (price - hist["val"]) / hist["val"] * 100
        if -2 < val_dist < 4:
            L4 += 3
            reasons.append("Vol Profile Support")
    L4 = _clamp(L4, 0, LAYER_BUDGET_LONG["pattern"])
    if L4 > 0:
        layers_active += 1

    # ── L5: Breadth / Macro (8) ───────────────────────────────────────────────
    L5 = 0
    if stoch_k > stoch_d and stoch_k < 50:
        L5 += 4
        reasons.append("Stoch Recovery")
    if macd_hist > 0:
        L5 += 3
        reasons.append("MACD Hist+")
    elif macd > macd_sig:
        L5 += 2
    if -3 < chg1w < 3:
        L5 += 1
    elif chg1w < -8:
        L5 -= 3
    L5 = _clamp(L5, -3, LAYER_BUDGET_LONG["breadth"])
    if L5 > 0:
        layers_active += 1

    # ── L6: Volatility (8) ────────────────────────────────────────────────────
    L6 = 0
    hvol = hist["volatility"]
    if hvol < 40:
        L6 += 5
    elif 40 <= hvol < 60:
        L6 += 2
    elif hvol >= 60:
        L6 -= 3
    if hist["ema20_slope"] > 0.2:
        L6 += 3
        reasons.append("EMA20 Rising")
    if chg1m > -5:
        L6 += 1
    elif chg1m < -15:
        L6 -= 2
    L6 = _clamp(L6, -3, LAYER_BUDGET_LONG["volatility"])
    if L6 > 0:
        layers_active += 1

    # ── L7: Historical Quality (10) ───────────────────────────────────────────
    L7 = 0
    if hist["stability"] >= 6:
        L7 += 5
    elif hist["stability"] >= 4:
        L7 += 2
    if hist["trend_pct_30"] > 5:
        L7 += 3
        reasons.append("30d Trend")
    if hist["consec_up"] >= 2:
        L7 += 2
    elif hist["consec_down"] >= 3:
        L7 -= 2
    # Historical position: value buys in lower half
    hist_p = hist["hist_pct"]
    if hist_p < 40:
        L7 += 3
        reasons.append(f"Hist Low@{hist_p:.0f}%")
    elif hist_p > 85:
        L7 -= 1
    L7 = _clamp(L7, -2, LAYER_BUDGET_LONG["historical"])
    if L7 > 0:
        layers_active += 1

    # ── L8: Institutional Flow (10) ── NEW ───────────────────────────────────
    L8 = 0
    if hist["obv_slope"] > 0.1:
        L8 += 5
        reasons.append("OBV Rising")
    elif hist["obv_slope"] < -0.15:
        L8 -= 4
    if hist["cmf"] > 0.1:
        L8 += 5
        reasons.append(f"CMF {hist['cmf']:.2f}")
    elif hist["cmf"] < -0.15:
        L8 -= 4
    L8 = _clamp(L8, -5, LAYER_BUDGET_LONG["flow"])
    if L8 > 0:
        layers_active += 1

    # ── L9: Market Regime (4) ── NEW ─────────────────────────────────────────
    L9 = 0
    if hist["regime"] == "BULL":
        L9 += 4
    elif hist["regime"] == "BEAR":
        L9 -= 4
    L9 = _clamp(L9, -4, LAYER_BUDGET_LONG["regime"])
    if L9 > 0:
        layers_active += 1

    score = max(0, L1 + L2 + L3 + L4 + L5 + L6 + L7 + L8 + L9)
    if layers_active < 4:
        score = min(score, THRESH_LONG - 1)

    if low1m > 0 and (price / low1m - 1) * 100 < 6:
        target = round(price * 1.55, 2)
    elif low1m > 0 and (price / low1m - 1) * 100 < 15:
        target = round(price * 1.30, 2)
    else:
        target = round(price * 1.20, 2)

    stop_floor = round(hist["support_level"] * 0.97, 2) if hist["support_level"] > 0 else 0
    stop = max(round(price * 0.88, 2), stop_floor) if stop_floor > 0 else round(price * 0.88, 2)

    layers = {
        "trend": L1, "momentum": L2, "volume": L3, "pattern": L4,
        "breadth": L5, "volatility": L6, "historical": L7, "flow": L8, "regime": L9,
    }
    return score, reasons, target, stop, _grade(score, layers_active), layers_active, layers


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════


def process_signals(raw: list, bullish: bool, mkt_chg: float):
    if not raw:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "regime": "NEUTRAL", "scalar": 1.0, "thresholds": {
                "INTRA": THRESH_INTRA, "SWING": THRESH_SWING, "LONG": THRESH_LONG, "DIP": THRESH_DIP,
            },
        }

    intra, swing, long_, dips = [], [], [], []

    conn = sqlite3.connect(CFG["DB_PATH"])
    try:
        _resolve_once_per_hour(conn, datetime.now().strftime("%Y-%m-%d-%H"))
        regime = get_current_regime(conn)

        # "Scale downstream signal thresholds mathematically based on the
        # active regime's variance" — a favorable regime (scalar > 1) lowers
        # the bar slightly; a choppy/distribution regime (scalar < 1) raises
        # it, since signals fired in that state are empirically less
        # reliable regardless of their raw additive score. Clamped so a
        # single regime read can't swing the bar wildly.
        regime_factor = _clamp(2 - regime.get("scalar", 1.0), 0.85, 1.2)
        thresholds = {
            "INTRA": round(THRESH_INTRA * regime_factor),
            "SWING": round(THRESH_SWING * regime_factor),
            "LONG": round(THRESH_LONG * regime_factor),
            "DIP": round(THRESH_DIP * regime_factor),
        }
        regime["thresholds"] = thresholds
        regime["regime_factor"] = regime_factor

        for item in raw:
            d = item.get("d", [])
            if len(d) < 28 or not d[0]:
                continue

            sym = d[0]
            price = safe(d[1])
            change = safe(d[2])
            vol = safe(d[3])
            rv = safe(d[4])
            avg_vol = safe(d[5])
            rsi = safe(d[6], 50)
            macd = safe(d[7])
            macd_sig = safe(d[8])
            bb_low = safe(d[9])
            bb_high = safe(d[10])
            ema20 = safe(d[11])
            ema50 = safe(d[12])
            ema25 = safe(d[13])
            chg1w = safe(d[14])
            high1m = safe(d[15])
            low1m = safe(d[16])
            vwap = safe(d[17]) or price
            ema10 = safe(d[18])
            adx = safe(d[19])
            atr = safe(d[20])
            stoch_k = safe(d[21], 50)
            stoch_d = safe(d[22], 50)
            open_p = safe(d[23])
            high_d = safe(d[24])
            low_d = safe(d[25])
            ema5 = safe(d[26])
            chg1m = safe(d[27])
            rsi_prev = safe(d[28], rsi) if len(d) > 28 else rsi
            low7d = safe(d[29]) if len(d) > 29 else low_d
            macd_h = safe(d[30]) if len(d) > 30 else (macd - macd_sig)
            bb_basis = safe(d[32]) if len(d) > 32 else (bb_low + bb_high) / 2

            # ── 15-min resolution fields, for the Intraday scorer only ──────
            # Fall back to the daily value when TradingView doesn't return an
            # intraday reading (illiquid names, off-hours snapshots, etc.) so
            # the scorer never silently sees a zero.
            rsi_15 = safe(d[37]) if len(d) > 37 and safe(d[37]) > 0 else rsi
            rsi_15_prev = safe(d[38], rsi_15) if len(d) > 38 and safe(d[38]) > 0 else rsi_15
            macd_15 = safe(d[39]) if len(d) > 39 else macd
            macd_sig_15 = safe(d[40]) if len(d) > 40 else macd_sig
            macd_h_15 = safe(d[41]) if len(d) > 41 else macd_h
            ema5_15 = safe(d[42]) if len(d) > 42 and safe(d[42]) > 0 else ema5
            ema10_15 = safe(d[43]) if len(d) > 43 and safe(d[43]) > 0 else ema10
            adx_15 = safe(d[44]) if len(d) > 44 and safe(d[44]) > 0 else adx
            stoch_k_15 = safe(d[45], 50) if len(d) > 45 and safe(d[45]) > 0 else stoch_k
            stoch_d_15 = safe(d[46], 50) if len(d) > 46 and safe(d[46]) > 0 else stoch_d

            if price <= 0 or avg_vol <= 0:
                continue

            sector = SYM_SECTOR.get(sym, "Misc")
            hist = get_hist_metrics(sym, conn)

            # Pivot calculation
            h_piv = high_d if high_d > 0 else price
            l_piv = low_d if low_d > 0 else price
            p_piv = (h_piv + l_piv + price) / 3
            r1 = round(2 * p_piv - l_piv, 2)
            s1 = round(2 * p_piv - h_piv, 2)
            r2 = round(p_piv + (h_piv - l_piv), 2)
            s2 = round(p_piv - (h_piv - l_piv), 2)
            best_buy = s1 if price > s1 else s2
            best_sell = r1 if price < r1 else r2

            is_buying = (price > vwap) and (macd > macd_sig) and (rsi > 50)
            trend_label = "BUYING" if is_buying else "SELLING"

            if vol < max(CFG["MIN_VOLUME"], hist["avg_vol"] * 0.30):
                continue

            # ── INTRADAY (scored on 15-min indicators, not daily) ───────────────
            sc, rs, tgt, stp, grd, la, lyr_intra = score_intraday(
                price,
                change,
                rsi_15,
                macd_15,
                macd_sig_15,
                macd_h_15,
                ema5_15,
                ema10_15,
                vwap,
                adx_15,
                stoch_k_15,
                stoch_d_15,
                vol,
                avg_vol,
                atr,
                bb_low,
                bb_high,
                bb_basis,
                open_p,
                high_d,
                low_d,
                bullish,
                mkt_chg,
                hist,
                rsi_15_prev,
            )
            if sc >= thresholds["INTRA"] and tgt > price > stp:
                intra.append(
                    {
                        "Symbol": sym,
                        "Sector": sector,
                        "Price": round(price, 2),
                        "Bias": trend_label,
                        "Chg%": round(change, 2),
                        "Score": sc,
                        "Buy": best_buy,
                        "Sell": best_sell,
                        "R1": r1,
                        "S1": s1,
                        "RV": round(rv, 1),
                        "RSI": round(rsi_15, 0),
                        "Signals": " | ".join(rs[:4]),
                        "Target": tgt,
                        "Stop": stp,
                        "R:R": _rr(price, tgt, stp),
                        "RV_val": rv,  # for sorting
                    }
                )
                fs = log_signal(conn, sym, "INTRA", price, tgt, stp, sc, lyr_intra)
                intra[-1]["_layers"] = lyr_intra
                intra[-1]["_first_seen"] = fs
                intra[-1]["_adv_vol"] = avg_vol
                intra[-1]["_ret_series"] = hist.get("ret_series", [])

            # ── SWING ─────────────────────────────────────────────────────────
            sc, rs, tgt, stp, grd, la, lyr_swing = score_swing(
                price,
                change,
                rsi,
                macd,
                macd_sig,
                macd_h,
                ema5,
                ema10,
                ema20,
                ema25,
                ema50,
                vwap,
                adx,
                atr,
                stoch_k,
                stoch_d,
                bb_low,
                bb_high,
                bb_basis,
                vol,
                avg_vol,
                chg1w,
                chg1m,
                low1m,
                high1m,
                bullish,
                mkt_chg,
                rsi_prev,
                hist,
            )
            if sc >= thresholds["SWING"] and tgt > price > stp:
                swing.append(
                    {
                        "Symbol": sym,
                        "Sector": sector,
                        "Price": round(price, 2),
                        "Bias": trend_label,
                        "Chg%": round(change, 2),
                        "1W%": round(chg1w, 2),
                        "Score": sc,
                        "Buy": best_buy,
                        "Sell": best_sell,
                        "R1": r1,
                        "S1": s1,
                        "RSI": round(rsi, 0),
                        "Signals": " | ".join(rs[:4]),
                        "Target": tgt,
                        "Stop": stp,
                        "R:R": _rr(price, tgt, stp),
                    }
                )
                fs = log_signal(conn, sym, "SWING", price, tgt, stp, sc, lyr_swing)
                swing[-1]["_layers"] = lyr_swing
                swing[-1]["_first_seen"] = fs
                swing[-1]["_adv_vol"] = avg_vol
                swing[-1]["_ret_series"] = hist.get("ret_series", [])

            # ── LONG-TERM ─────────────────────────────────────────────────────
            perf1m = (price / low1m - 1) * 100 if low1m > 0 else 0.0
            sc, rs, tgt, stp, grd, la, lyr_long = score_longterm(
                price,
                rsi,
                macd,
                macd_sig,
                macd_h,
                ema20,
                ema50,
                stoch_k,
                stoch_d,
                bb_low,
                bb_high,
                bb_basis,
                vol,
                avg_vol,
                chg1w,
                chg1m,
                low1m,
                high1m,
                sector,
                rsi_prev,
                hist,
            )
            if sc >= thresholds["LONG"] and tgt > price > stp:
                long_.append(
                    {
                        "Symbol": sym,
                        "Sector": sector,
                        "Price": round(price, 2),
                        "Bias": trend_label,
                        "1W%": round(chg1w, 2),
                        "1M%": round(perf1m, 2),
                        "Score": sc,
                        "Buy": best_buy,
                        "Sell": best_sell,
                        "R1": r1,
                        "S1": s1,
                        "RSI": round(rsi, 0),
                        "Signals": " | ".join(rs[:4]),
                        "Target": tgt,
                        "Stop": stp,
                        "R:R": _rr(price, tgt, stp),
                    }
                )
                fs = log_signal(conn, sym, "LONG", price, tgt, stp, sc, lyr_long)
                long_[-1]["_layers"] = lyr_long
                long_[-1]["_first_seen"] = fs
                long_[-1]["_adv_vol"] = avg_vol
                long_[-1]["_ret_series"] = hist.get("ret_series", [])

            # ── DIP SCANNER ───────────────────────────────────────────────────
            is_uptrend = price > ema50 if ema50 > 0 else (price > ema20 if ema20 > 0 else False)
            if is_uptrend:
                dip_score, dip_reasons, dip_target, dip_stop = score_dip(
                    price, rsi, stoch_k, bb_low, bb_high, low7d, atr, vol, avg_vol, hist
                )

                if dip_score >= thresholds["DIP"] and dip_target > price > dip_stop:
                    dips.append(
                        {
                            "Symbol": sym,
                            "Sector": sector,
                            "Price": round(price, 2),
                            "Bias": trend_label,
                            "Chg%": round(change, 2),
                            "1W%": round(chg1w, 2),
                            "Score": dip_score,
                            "Target": dip_target,
                            "Stop": dip_stop,
                            "R:R": _rr(price, dip_target, dip_stop),
                            "Signals": " | ".join(dip_reasons[:4]),
                            "Buy": best_buy,
                            "Sell": best_sell,
                            "RSI": round(rsi, 0),
                            "R1": r1,
                            "S1": s1,
                        }
                    )
                    fs = log_signal(conn, sym, "DIP", price, dip_target, dip_stop, dip_score,
                                     {"dip_score": dip_score})
                    dips[-1]["_layers"] = {"dip_score": dip_score}
                    dips[-1]["_first_seen"] = fs
                    dips[-1]["_adv_vol"] = avg_vol
                    dips[-1]["_ret_series"] = hist.get("ret_series", [])

        # ── Batch risk-engine pass: vectorized posterior probability across
        # each strategy's whole fired-signal batch, then per-row decay/Kelly
        # sizing on that same (small, already-filtered) batch. Must run
        # before conn closes — the validator reads signal_log.
        intra_ = _apply_risk_engine(intra, "INTRA", conn, regime)
        swing_ = _apply_risk_engine(swing, "SWING", conn, regime)
        long_2 = _apply_risk_engine(long_, "LONG", conn, regime)
        dips_ = _apply_risk_engine(dips, "DIP", conn, regime)
    finally:
        conn.close()

    srt = lambda lst: (
        pd.DataFrame(lst).sort_values("Score", ascending=False).reset_index(drop=True)
        if lst
        else pd.DataFrame()
    )
    return srt(intra_), srt(swing_), srt(long_2), srt(dips_), regime


# ══════════════════════════════════════════════════════════════════════════════
# SECTOR ROTATION ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# CSS — Wall Street Dark Terminal
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"] {
    background: #080c18 !important;
    color: #dde2ec !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}
footer, #MainMenu { visibility: hidden; }
.main .block-container {
    max-width: 1100px !important;
    padding: 2rem 2.5rem !important;
    margin: 0 auto !important;
}

/* ── Masthead ── */
.report-masthead { padding-bottom: 4px; margin-bottom: 0; }
.report-title {
    font-family: 'Libre Baskerville', Georgia, serif;
    font-size: 1.5rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: #dde2ec; margin: 0; line-height: 1.2;
}
.report-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.70rem; color: #5a6170; margin-top: 4px; letter-spacing: 0.05em;
}

/* ── Market Bar ── */
.market-bar {
    display: flex; justify-content: space-between;
    border: 1px solid #1a2030; padding: 10px 18px;
    margin-bottom: 14px; background: #0c1020;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
}
.market-item { text-align: center; }
.market-label {
    font-size: 0.60rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: #5a6170; margin-bottom: 2px;
}
.market-value { font-size: 0.92rem; font-weight: 700; color: #dde2ec; }

/* ── Regime Badge ── */
.regime-bull {
    display: inline-block; padding: 1px 7px;
    background: rgba(52,211,153,0.12); color: #34d399;
    border: 1px solid rgba(52,211,153,0.3);
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace; text-transform: uppercase;
}
.regime-bear {
    display: inline-block; padding: 1px 7px;
    background: rgba(248,113,113,0.12); color: #f87171;
    border: 1px solid rgba(248,113,113,0.3);
    font-size: 0.62rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace; text-transform: uppercase;
}
.regime-neutral {
    display: inline-block; padding: 1px 7px;
    background: rgba(251,191,36,0.12); color: #fbbf24;
    border: 1px solid rgba(251,191,36,0.3);
    font-size: 0.62rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace; text-transform: uppercase;
}

/* ── Sector Row ── */
.sector-row {
    display: flex; flex-wrap: wrap; gap: 0;
    border: 1px solid #1a2030; margin-bottom: 14px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
}
.sector-cell {
    flex: 1 1 auto; min-width: 88px; padding: 5px 10px;
    border-right: 1px solid #1a2030; border-bottom: 1px solid #1a2030;
    display: flex; flex-direction: column; background: #080c18;
}
.sector-cell:hover { background: #0f1322; }
.sector-name { color: #5a6170; font-size: 0.64rem; margin-bottom: 1px; }
.sector-1m   { font-weight: 700; font-size: 0.74rem; }
.sector-phase { font-size: 0.60rem; opacity: 0.7; }
.s-up  { color: #34d399; } .s-dn { color: #f87171; } .s-nt { color: #fbbf24; }
.s-lead { color: #34d399; } .s-weak { color: #fbbf24; }
.s-lag  { color: #f87171; } .s-reco { color: #60a5fa; }

/* ── Section Headers ── */
.section-header {
    font-family: 'Libre Baskerville', Georgia, serif;
    font-size: 0.95rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em;
    border-bottom: 2px solid #2a3040; padding-bottom: 4px;
    margin-top: 26px; margin-bottom: 4px; color: #dde2ec;
}
.section-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem; color: #5a6170;
    margin-bottom: 10px; letter-spacing: 0.03em;
}

/* ── Alerts ── */
.paper-alert {
    border: 1px solid rgba(248,113,113,0.3);
    border-left: 4px solid #f87171;
    padding: 8px 14px; margin-bottom: 12px;
    font-size: 0.76rem; color: #fca5a5;
    background: rgba(239,68,68,0.06);
}
.paper-info {
    border: 1px solid #1a2030; border-left: 4px solid #2a3040;
    padding: 8px 14px; margin-bottom: 12px;
    font-size: 0.76rem; color: #9ca3af;
    background: rgba(12,16,32,0.6);
}

/* ── Tables ── */
[data-testid="stDataFrame"] { background: transparent !important; border-radius: 0 !important; }
[data-testid="stDataFrame"] > div { border: 1px solid #1a2030 !important; }
thead tr th {
    background: #0c1020 !important; color: #5a6170 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important; font-weight: 700 !important;
    letter-spacing: 0.08em !important; text-transform: uppercase;
    padding: 8px 6px !important; border-bottom: 2px solid #222840 !important;
}
tbody tr { border-bottom: 1px solid #161c2e !important; }
tbody tr:nth-child(even) { background: rgba(12,16,32,0.4) !important; }
tbody tr:hover { background: rgba(26,32,48,0.5) !important; }
tbody td {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.70rem !important; color: #dde2ec !important;
    padding: 5px !important;
}

/* ── Buttons ── */
.stButton button {
    background: #0c1020 !important; border: 1px solid #222840 !important;
    border-radius: 0 !important; color: #dde2ec !important;
    padding: 4px 12px !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.70rem !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important;
    min-height: unset !important; line-height: 1.4 !important;
}
.stButton button:hover { background: #161c2e !important; border-color: #3a4050 !important; }

/* ── Footer ── */
.report-footer {
    border-top: 1px solid #1a2030; margin-top: 36px; padding-top: 10px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.60rem;
    color: #3c4455; line-height: 1.7;
}

/* ── Misc ── */
hr { border-color: #1a2030 !important; margin: 1.2rem 0 !important; }
div[data-testid="stExpander"] {
    background: #0c1020 !important; border: 1px solid #1a2030 !important;
    border-radius: 0 !important;
}
@media (max-width: 768px) {
    .main .block-container { padding: 1rem !important; }
    .report-title { font-size: 1.1rem; }
    .market-bar { flex-wrap: wrap; gap: 8px; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# INIT
# ══════════════════════════════════════════════════════════════════════════════

init_db()


def _price_history_symbol_count(db_path: str) -> int:
    """How many distinct symbols already have historical bars saved."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT symbol) FROM price_history")
        row = cur.fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


# Every scoring layer (regime, stability, ADX/PSAR, OBV/CMF, divergence,
# z-score, and the Long-Term strategy's hard stability gate) reads from
# price_history via get_hist_metrics(). That table starts empty and was
# previously only ever populated by a user manually clicking "SYNC" —
# meaning Long-Term signals were hard-rejected (stability < 2.0 gate) and
# Intraday/Swing silently ran with several layers permanently neutral on
# every fresh deploy, which is the main reason the scanner looked broken.
# This one-time, session-cached bootstrap sync fixes that without adding
# any visible UI — the existing SYNC button still works for manual refreshes.
@st.cache_resource
def _ensure_history_bootstrapped():
    have = _price_history_symbol_count(CFG["DB_PATH"])
    if have < len(KSE100) * 0.5:  # most symbols missing → treat as cold start
        sync_historical_data(KSE100)
    return True


_ensure_history_bootstrapped()

is_open = is_market_open()
raw_data = fetch_live()
avg_chg, bullish, adv, dec, kse_fb = calculate_breadth_from_raw(raw_data)
kse_api = fetch_kse_index()
now = pkt_now()


def _kse(key, fb):
    return kse_api.get(key, fb) if kse_api else fb


idx_close = _kse("close", kse_fb["close"])
idx_pct = _kse("changePercent", kse_fb["change"])
idx_vol = _kse("volume", kse_fb["volume"])
vol_cr = idx_vol / 1e7
state_text = "LIVE" if is_open else "CLOSED"
breadth_text = "BULLISH" if bullish else ("BEARISH" if avg_chg < -0.5 else "NEUTRAL")

# ── Masthead + Buttons ────────────────────────────────────────────────────────
head_cols = st.columns([8, 1, 1])
with head_cols[0]:
    st.markdown(
        f"""
    <div class="report-masthead">
        <div class="report-title">PSX Market Intelligence</div>
        <div class="report-subtitle">
            KSE-100 &middot; 9-Layer Signal Engine &middot; Wall Street Edition &middot;
            {now.strftime("%A, %B %d, %Y")} &middot; {now.strftime("%H:%M")} PKT
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with head_cols[1]:
    st.markdown('<div style="padding-top:8px"></div>', unsafe_allow_html=True)
    scan_btn = st.button("SCAN", help="Run scanner now", use_container_width=True)
with head_cols[2]:
    st.markdown('<div style="padding-top:8px"></div>', unsafe_allow_html=True)
    sync_btn = st.button("SYNC", help="Sync 260-day history", use_container_width=True)

st.markdown(
    '<div style="border-bottom:3px double #2a3040;margin-bottom:18px;margin-top:8px"></div>',
    unsafe_allow_html=True,
)

if sync_btn:
    sync_historical_data(KSE100)
    st.rerun()

# ── Market Summary Bar ────────────────────────────────────────────────────────
pct_color = "#34d399" if idx_pct >= 0 else "#f87171"
st.markdown(
    f"""
<div class="market-bar">
    <div class="market-item">
        <div class="market-label">KSE-100</div>
        <div class="market-value">{idx_close:,.0f}</div>
    </div>
    <div class="market-item">
        <div class="market-label">Change</div>
        <div class="market-value" style="color:{pct_color}">{idx_pct:+.2f}%</div>
    </div>
    <div class="market-item">
        <div class="market-label">State</div>
        <div class="market-value">{state_text}</div>
    </div>
    <div class="market-item">
        <div class="market-label">Adv / Dec</div>
        <div class="market-value">{adv} / {dec}</div>
    </div>
    <div class="market-item">
        <div class="market-label">Volume</div>
        <div class="market-value">{vol_cr:.1f} Cr</div>
    </div>
    <div class="market-item">
        <div class="market-label">Breadth</div>
        <div class="market-value" style="color:{"#34d399" if bullish else "#f87171"}">{breadth_text}</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

if not bullish:
    st.markdown(
        f'<div class="paper-alert"><strong>BEARISH BREADTH</strong> — '
        f"Market declining ({adv} advancers vs {dec} decliners). "
        f"Reduce size. Intraday setups need extra confirmation.</div>",
        unsafe_allow_html=True,
    )

scan = is_open or scan_btn

if scan:
    with st.spinner("Scanning KSE-100 (9-layer engine)…"):
        raw = fetch_live()
        save_snapshot(raw)

    if not raw:
        st.warning("No data returned — check network or try again.")
        st.stop()

    # ── Sector Heatmap with Rotation ─────────────────────────────────────────
    sec_perf: Dict[str, list] = {}
    for item in raw:
        d = item.get("d", [])
        if d and len(d) > 2 and d[0]:
            sec = SYM_SECTOR.get(d[0])
            if sec:
                sec_perf.setdefault(sec, []).append(safe(d[2]))

    tiles = ""
    for sec in SECTORS:
        chgs = sec_perf.get(sec, [0])
        avg_c = sum(chgs) / len(chgs)
        cls = "s-up" if avg_c >= 0.5 else ("s-dn" if avg_c < -0.5 else "s-nt")
        arrow = "+" if avg_c >= 0 else ""
        tiles += (
            f'<div class="sector-cell">'
            f'<span class="sector-name">{html.escape(sec)}</span>'
            f'<span class="sector-1m {cls}">{arrow}{avg_c:.1f}%</span>'
            f'<span class="sector-phase"></span>'
            f"</div>"
        )
    st.markdown(f'<div class="sector-row">{tiles}</div>', unsafe_allow_html=True)

    # ── Run Signal Engine ─────────────────────────────────────────────────────
    df_i, df_s, df_l, df_d, regime = process_signals(raw, bullish, avg_chg)
    # process_signals already gates each row against its regime-scaled
    # threshold internally — re-filtering here against the static THRESH_*
    # constants would silently cancel out that scaling, so we don't.
    if not df_d.empty:
        df_d = df_d.reset_index(drop=True)

    all_stocks_df = ( # Used for saving daily snapshot
        pd.concat([df for df in [df_i, df_s, df_l, df_d] if not df.empty]).drop_duplicates(
            subset=["Symbol"]
        )
        if any(not df.empty for df in [df_i, df_s, df_l, df_d])
        else pd.DataFrame()
    )

    def render_table(df: pd.DataFrame, col_order: list, col_cfg: dict, top_n: int = 10):
        if df.empty:
            st.markdown(
                '<div class="paper-info">No setups meet current threshold.</div>',
                unsafe_allow_html=True,
            )
        else:
            available = [c for c in col_order if c in df.columns]
            st.dataframe(
                df.head(top_n)[available].copy(),
                column_config=col_cfg,
                hide_index=True,
                use_container_width=True,
            )

    # ── INTRADAY ──────────────────────────────────────────────────────────────
    n_i = len(df_i)
    st.markdown('<div class="section-header">Intraday Scalps</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-meta">{n_i} setups &middot; Threshold {regime["thresholds"]["INTRA"]}/100 &middot; Breadth: {breadth_text} &middot; 4+/9 layer confluence required</div>',
        unsafe_allow_html=True,
    )
    render_table(
        df_i,
        [
            "Symbol",
            "Price",
            "Chg%",
            "Score",
            "Bias",
            "R:R",
            "Target",
            "Stop",
            "RV",
            "RSI",
            "Signals",
            "Buy",
            "Sell",
            "R1",
            "S1",
        ],
        {
            "Symbol": st.column_config.TextColumn("Symbol"),
            "Price": st.column_config.NumberColumn("Price", format="%.2f"),
            "Chg%": st.column_config.NumberColumn("Chg%", format="%.2f%%"),
            "Score": st.column_config.NumberColumn("Score", format="%d"),
            "Bias": st.column_config.TextColumn("Bias"),
            "R:R": st.column_config.NumberColumn("R:R", format="%.2f"),
            "Target": st.column_config.NumberColumn("Target", format="%.2f"),
            "Stop": st.column_config.NumberColumn("Stop", format="%.2f"),
            "RV": st.column_config.NumberColumn("RV", format="%.1fx"),
            "RSI": st.column_config.NumberColumn("RSI", format="%d"),
            "Signals": st.column_config.TextColumn("Signals"),
            "Buy": st.column_config.NumberColumn("Buy", format="%.2f"),
            "Sell": st.column_config.NumberColumn("Sell", format="%.2f"),
            "R1": st.column_config.NumberColumn("R1", format="%.2f"),
            "S1": st.column_config.NumberColumn("S1", format="%.2f"),
        },
    )

    # ── SWING ─────────────────────────────────────────────────────────────────
    n_s = len(df_s)
    st.markdown('<div class="section-header">Swing Trades</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-meta">{n_s} setups &middot; Threshold {regime["thresholds"]["SWING"]}/100 &middot; 4+/9 layer confluence required</div>',
        unsafe_allow_html=True,
    )
    render_table(
        df_s,
        [
            "Symbol",
            "Price",
            "Chg%",
            "1W%",
            "Score",
            "Bias",
            "R:R",
            "Target",
            "Stop",
            "RSI",
            "Signals",
            "Buy",
            "Sell",
            "R1",
            "S1",
        ],
        {
            "Symbol": st.column_config.TextColumn("Symbol"),
            "Price": st.column_config.NumberColumn("Price", format="%.2f"),
            "Chg%": st.column_config.NumberColumn("Chg%", format="%.2f%%"),
            "1W%": st.column_config.NumberColumn("1W%", format="%.2f%%"),
            "Score": st.column_config.NumberColumn("Score", format="%d"),
            "Bias": st.column_config.TextColumn("Bias"),
            "R:R": st.column_config.NumberColumn("R:R", format="%.2f"),
            "Target": st.column_config.NumberColumn("Target", format="%.2f"),
            "Stop": st.column_config.NumberColumn("Stop", format="%.2f"),
            "RSI": st.column_config.NumberColumn("RSI", format="%d"),
            "Signals": st.column_config.TextColumn("Signals"),
            "Buy": st.column_config.NumberColumn("Buy", format="%.2f"),
            "Sell": st.column_config.NumberColumn("Sell", format="%.2f"),
            "R1": st.column_config.NumberColumn("R1", format="%.2f"),
            "S1": st.column_config.NumberColumn("S1", format="%.2f"),
        },
    )

    # ── LONG-TERM ─────────────────────────────────────────────────────────────
    n_l = len(df_l)
    st.markdown('<div class="section-header">Long-Term Investments</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-meta">{n_l} setups &middot; Threshold {regime["thresholds"]["LONG"]}/100 &middot; Sector quality ≥6 required</div>',
        unsafe_allow_html=True,
    )
    render_table(
        df_l,
        [
            "Symbol",
            "Price",
            "1M%",
            "1W%",
            "Score",
            "Bias",
            "R:R",
            "Target",
            "Stop",
            "RSI",
            "Signals",
            "Buy",
            "Sell",
            "R1",
            "S1",
        ],
        {
            "Symbol": st.column_config.TextColumn("Symbol"),
            "Price": st.column_config.NumberColumn("Price", format="%.2f"),
            "1M%": st.column_config.NumberColumn("1M%", format="%.2f%%"),
            "1W%": st.column_config.NumberColumn("1W%", format="%.2f%%"),
            "Score": st.column_config.NumberColumn("Score", format="%d"),
            "Bias": st.column_config.TextColumn("Bias"),
            "R:R": st.column_config.NumberColumn("R:R", format="%.2f"),
            "Target": st.column_config.NumberColumn("Target", format="%.2f"),
            "Stop": st.column_config.NumberColumn("Stop", format="%.2f"),
            "RSI": st.column_config.NumberColumn("RSI", format="%d"),
            "Signals": st.column_config.TextColumn("Signals"),
            "Buy": st.column_config.NumberColumn("Buy", format="%.2f"),
            "Sell": st.column_config.NumberColumn("Sell", format="%.2f"),
            "R1": st.column_config.NumberColumn("R1", format="%.2f"),
            "S1": st.column_config.NumberColumn("S1", format="%.2f"),
        },
    )

    # ── STOCKS ON DIP ─────────────────────────────────────────────────────────
    n_d = len(df_d)
    st.markdown('<div class="section-header">Stocks on Dip</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-meta">{n_d} setups &middot; Uptrend (above EMA50) + Pullback detected &middot; CMF-enhanced quality scoring</div>',
        unsafe_allow_html=True,
    )
    render_table(
        df_d,
        [
            "Symbol",
            "Price",
            "Chg%",
            "1W%",
            "Score",
            "Bias",
            "R:R",
            "Target",
            "Stop",
            "RSI",
            "Signals",
            "Buy",
            "Sell",
            "R1",
            "S1",
        ],
        {
            "Symbol": st.column_config.TextColumn("Symbol"),
            "Price": st.column_config.NumberColumn("Price", format="%.2f"),
            "Chg%": st.column_config.NumberColumn("Chg%", format="%.2f%%"),
            "1W%": st.column_config.NumberColumn("1W%", format="%.2f%%"),
            "Score": st.column_config.NumberColumn("Score", format="%d", help="Dip quality score"),
            "Bias": st.column_config.TextColumn("Bias"),
            "R:R": st.column_config.NumberColumn("R:R", format="%.2f"),
            "Target": st.column_config.NumberColumn("Target", format="%.2f"),
            "Stop": st.column_config.NumberColumn("Stop", format="%.2f"),
            "RSI": st.column_config.NumberColumn("RSI", format="%d"),
            "Signals": st.column_config.TextColumn("Signals"),
            "Buy": st.column_config.NumberColumn("Buy", format="%.2f"),
            "Sell": st.column_config.NumberColumn("Sell", format="%.2f"),
            "R1": st.column_config.NumberColumn("R1", format="%.2f"),
            "S1": st.column_config.NumberColumn("S1", format="%.2f"),
        },
    )

    # ── TREND REVERSALS ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Trend Reversals</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-meta">Bias changes detected since previous session snapshot</div>',
        unsafe_allow_html=True,
    )

    yesterday_snapshot = get_yesterday_snapshot()
    reversals = []
    if not all_stocks_df.empty:
        for _, row in all_stocks_df.iterrows():
            symbol = row["Symbol"]
            current_trend = row["Bias"]
            yesterday_data = yesterday_snapshot.get(symbol)
            if yesterday_data and yesterday_data["trend"] != current_trend:
                reversals.append(
                    {
                        "Symbol": symbol,
                        "Yesterday Bias": yesterday_data["trend"],
                        "Today Bias": current_trend,
                    }
                )
        save_daily_snapshot(all_stocks_df)

    if reversals:
        st.dataframe(pd.DataFrame(reversals), hide_index=True, use_container_width=True)
    else:
        st.markdown(
            '<div class="paper-info">No trend reversals detected in this scan.</div>',
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="report-footer">
    Generated {now.strftime("%Y-%m-%d %H:%M:%S")} PKT
    &middot; Source: TradingView Scanner, Sarmaaya
    &middot; KSE-100 Universe ({len(KSE100)} symbols)<br>
    9-Layer scoring (Trend · Momentum+RS · Volume · Pattern+MeanRev · Breadth · Volatility · History · InstFlow(OBV+CMF) · Regime).
    Quantitative screening tool only. Not investment advice. Signals require independent verification before execution.
</div>
""",
    unsafe_allow_html=True,
)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if is_open:
    import streamlit.components.v1 as components

    components.html(
        f"""<script>
            setTimeout(function() {{
                window.parent.location.reload();
            }}, {CFG["REFRESH_SEC"] * 1000});
        </script>""",
        height=0,
    )
