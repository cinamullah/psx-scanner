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
import time
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
KSE100 = [
    "CNERGY",
    "BOP",
    "PRL",
    "WTL",
    "KOSM",
    "KEL",
    "UNITY",
    "NCPL",
    "CSIL",
    "PAEL",
    "SSGC",
    "TRG",
    "ATRL",
    "MLCF",
    "SYS",
    "NPL",
    "CLOV",
    "YOUW",
    "TELE",
    "PTC",
    "NBP",
    "LUCK",
    "DGKC",
    "SNGP",
    "PSO",
    "NRL",
    "OGDC",
    "POL",
    "PPL",
    "NETSOL",
    "MEBL",
    "UBL",
    "ABL",
    "BAFL",
    "BAHL",
    "MARI",
    "FFC",
    "ENGROH",
    "EFERT",
    "ATLH",
    "HCAR",
    "MTL",
    "COLG",
    "ABOT",
    "ILP",
    "PIBTL",
    "TGL",
    "CHCC",
    "HUBC",
    "AIRLINK",
    "HBL",
    "MCB",
    "FABL",
    "JSBL",
    "SILK",
    "KAPCO",
    "FCCL",
    "POWER",
    "ACPL",
    "PIOC",
]

SECTORS: Dict[str, Dict] = {
    "Banks": {
        "symbols": [
            "MEBL",
            "UBL",
            "ABL",
            "BAFL",
            "BAHL",
            "BOP",
            "NBP",
            "HBL",
            "MCB",
            "FABL",
            "JSBL",
            "SILK",
        ],
        "quality": 9,
    },
    "E&P": {"symbols": ["OGDC", "PPL", "MARI", "POL"], "quality": 9},
    "Fertilizer": {"symbols": ["FFC", "ENGROH", "EFERT"], "quality": 9},
    "Cement": {"symbols": ["LUCK", "DGKC", "MLCF", "CHCC", "FCCL", "ACPL", "PIOC"], "quality": 7},
    "Tech": {"symbols": ["SYS", "TRG", "PTC", "NETSOL", "AIRLINK"], "quality": 7},
    "Power": {"symbols": ["HUBC", "KEL", "NCPL", "PAEL", "NPL", "KAPCO", "POWER"], "quality": 7},
    "Oil & Gas": {"symbols": ["SNGP", "SSGC", "PSO", "NRL", "ATRL", "PRL", "CNERGY"], "quality": 8},
    "Auto": {"symbols": ["ATLH", "HCAR", "MTL"], "quality": 6},
    "Food": {"symbols": ["UNITY", "COLG"], "quality": 8},
    "Pharma": {"symbols": ["ABOT"], "quality": 9},
    "Textile": {"symbols": ["KOSM", "CLOV", "ILP"], "quality": 5},
    "Misc": {"symbols": ["YOUW", "CSIL", "PIBTL", "TGL", "TELE"], "quality": 5},
}
SYM_SECTOR = {sym: sec for sec, v in SECTORS.items() for sym in v["symbols"]}

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
    "Low.6M",  # ← wider range context
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
            if len(d) >= 25 and d[0]:
                rows.append(
                    (
                        today,
                        d[0],
                        safe(d[22]),
                        safe(d[23]),
                        safe(d[24]),
                        safe(d[1]),
                        int(safe(d[3])),
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
  
