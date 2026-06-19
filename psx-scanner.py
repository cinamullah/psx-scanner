"""
PSX Institutional Scanner — Elite Edition
KSE-100 · 7-Layer Signal Engine · Live + Historical Intelligence
Built with institutional-grade logic: trend, momentum, volume, pattern, breadth, volatility, risk
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import sqlite3
import html
import os
import time
import pytz
import math
import yfinance as yf

from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, List, Optional, Tuple

st.set_page_config(
    page_title="PSX Elite Scanner",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CFG = {
    "MARKET_OPEN":   "09:30",
    "MARKET_CLOSE":  "15:30",
    "REFRESH_SEC":   180,
    "MIN_VOLUME":    100_000,      # Absolute floor — very illiquid stocks filtered
    "MIN_PRICE":     2.0,          # Penny stock filter
    "INST_VOL_X":   2.5,           # Institutional volume threshold
    "BREADTH_MIN":   0.0,
    "DB_PATH":       os.path.join(BASE_DIR, "psx_elite.db"),
    "HIST_DAYS":     90,
}

# ── Universe ──────────────────────────────────────────────────────────────────

KSE100 = [
    "CNERGY","BOP","PRL","WTL","KOSM","KEL","UNITY","NCPL","CSIL","PAEL",
    "SSGC","TRG","ATRL","MLCF","SYS","NPL","CLOV","YOUW","TELE","PTC",
    "NBP","LUCK","DGKC","SNGP","PSO","NRL","OGDC","POL","PPL","NETSOL",
    "MEBL","UBL","ABL","BAFL","BAHL","MARI","FFC","ENGROH","EFERT","ATLH",
    "HCAR","MTL","COLG","ABOT","ILP","PIBTL","TGL","CHCC","HUBC","AIRLINK",
    "HBL","MCB","FABL","JSBL","SILK","KAPCO","FCCL","POWER","ACPL","PIOC",
]

SECTORS: Dict[str, Dict] = {
    "Banks":     {"symbols": ["MEBL","UBL","ABL","BAFL","BAHL","BOP","NBP","HBL","MCB","FABL","JSBL","SILK"], "quality": 9},
    "E&P":       {"symbols": ["OGDC","PPL","MARI","POL"],                                                      "quality": 9},
    "Fertilizer":{"symbols": ["FFC","ENGROH","EFERT"],                                                         "quality": 9},
    "Cement":    {"symbols": ["LUCK","DGKC","MLCF","CHCC","FCCL","ACPL","PIOC"],                              "quality": 7},
    "Tech":      {"symbols": ["SYS","TRG","PTC","NETSOL","AIRLINK"],                                          "quality": 7},
    "Power":     {"symbols": ["HUBC","KEL","NCPL","PAEL","NPL","KAPCO","POWER"],                              "quality": 7},
    "Oil & Gas": {"symbols": ["SNGP","SSGC","PSO","NRL","ATRL","PRL","CNERGY"],                               "quality": 8},
    "Auto":      {"symbols": ["ATLH","HCAR","MTL"],                                                            "quality": 6},
    "Food":      {"symbols": ["UNITY","COLG"],                                                                 "quality": 8},
    "Pharma":    {"symbols": ["ABOT"],                                                                         "quality": 9},
    "Textile":   {"symbols": ["KOSM","CLOV","ILP"],                                                            "quality": 5},
    "Misc":      {"symbols": ["YOUW","CSIL","PIBTL","TGL","TELE"],                                            "quality": 5},
}

SYM_SECTOR = {sym: sec for sec, v in SECTORS.items() for sym in v["symbols"]}

# ── TradingView columns — extended for richer indicators ──────────────────────
TV_COLS = [
    "name",                      # 0
    "close",                     # 1
    "change",                    # 2
    "volume",                    # 3
    "relative_volume_10d_calc",  # 4
    "average_volume_10d_calc",   # 5
    "RSI",                       # 6
    "MACD.macd",                 # 7
    "MACD.signal",               # 8
    "BB.lower",                  # 9
    "BB.upper",                  # 10
    "EMA20",                     # 11
    "EMA50",                     # 12
    "change|1W",                 # 13
    "High.1M",                   # 14
    "Low.1M",                    # 15
    "VWAP",                      # 16
    "EMA10",                     # 17
    "ADX",                       # 18
    "ATR",                       # 19
    "Stoch.K",                   # 20
    "Stoch.D",                   # 21
    "open",                      # 22
    "high",                      # 23
    "low",                       # 24
    "EMA5",                      # 25
    "change|1M",                 # 26
    "RSI[1]",                    # 27  (previous RSI for divergence)
    "MACD.hist",                 # 28  (MACD histogram)
    "Pivot.M.Classic.Middle",    # 29  (monthly pivot)
    "BB.basis",                  # 30  (BB middle/basis)
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
    """Safe value extraction — returns default for None/NaN."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default

def _rr(price: float, target: float, stop: float) -> str:
    denom = price - stop
    if denom <= 0 or target <= price:
        return "—"
    ratio = (target - price) / denom
    return f"1:{ratio:.1f}"

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE — historical OHLCV store
# ══════════════════════════════════════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect(CFG["DB_PATH"])
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
    """)
    conn.commit()
    conn.close()

def sync_historical_data(symbols: List[str]):
    """Optimized batch download from Yahoo Finance."""
    end = datetime.now()
    start = end - timedelta(days=CFG["HIST_DAYS"])
    tickers = [f"{s}.KA" for s in symbols]

    ph = st.empty()
    ph.caption("🔄 Fetching Market History...")

    # Batch download is much faster than sequential
    df = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)

    conn = sqlite3.connect(CFG["DB_PATH"])
    for sym in symbols:
        try:
            ticker_df = df[f"{sym}.KA"].dropna().reset_index()
            if ticker_df.empty: continue

            rows = []
            for _, r in ticker_df.iterrows():
                rows.append((
                    r["Date"].strftime("%Y-%m-%d"), sym,
                    float(r["Open"]), float(r["High"]),
                    float(r["Low"]), float(r["Close"]), int(r["Volume"])
                ))
            conn.executemany("INSERT OR REPLACE INTO price_history VALUES (?,?,?,?,?,?,?)", rows)
        except Exception: continue

    conn.commit()
    conn.close()
    ph.success("✅ Sync Complete")

def save_snapshot(raw: list):
    """Persist today's live prices — uses open/high/low/close from intraday data."""
    if not raw:
        return
    today = pkt_now().strftime("%Y-%m-%d")
    conn  = sqlite3.connect(CFG["DB_PATH"])
    rows  = []
    for item in raw:
        d = item.get("d", [])
        if len(d) >= 25 and d[0]:
            rows.append((
                today, d[0],
                safe(d[22]), safe(d[23]), safe(d[24]), safe(d[1]),
                int(safe(d[3]))
            ))
    if rows:
        conn.executemany("INSERT OR REPLACE INTO price_history VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()

# ══════════════════════════════════════════════════════════════════════════════
# HISTORICAL ANALYTICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def get_hist_metrics(symbol: str, db_conn: sqlite3.Connection) -> Dict:
    """
    Vectorized historical analytics. Uses an existing DB connection for performance.
    """
    df = pd.read_sql(
        "SELECT open, high, low, close, volume FROM price_history WHERE symbol=? ORDER BY date ASC",
        db_conn, params=(symbol,)
    )

    empty = {
        "has_data": False, "n": 0,
        "stability": 0.0, "avg_vol": 0.0, "vol_trend": 0.0,
        "trend_pct_30": 0.0, "trend_pct_10": 0.0,
        "volatility": 0.0, "atr_20": 0.0,
        "momentum": 0.0, "rsi_hist": 50.0, "rsi_slope": 0.0,
        "ema10_slope": 0.0, "ema20_slope": 0.0,
        "support_level": 0.0, "resistance_level": 0.0,
        "consec_up": 0, "consec_down": 0,
        "higher_lows": False, "vol_accumulation": False,
        "squeeze": False,
    }

    if len(df) < 20: # Minimum 20 days for robust calculations (e.g., ATR, BB)
        return empty

    n = len(df)
    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']

    # ── 1. Trend metrics ──────────────────────────────────────────────────────
    trend_pct_30 = (c.iloc[-1] / c.iloc[max(0, n-30)] - 1) * 100 if n >= 30 else 0.0
    trend_pct_10 = (c.iloc[-1] / c.iloc[max(0, n-10)] - 1) * 100 if n >= 10 else 0.0

    # ── 2. Stability (% of up-days, last 15 sessions) ─────────────────────────
    up_days = (c.diff() >= 0).tail(15).sum()
    stability = (up_days / 15) * 10 if n >= 15 else 0.0

    # ── 3. Volume analytics ───────────────────────────────────────────────────
    avg_vol = v.tail(30).mean() if n >= 30 else v.mean()
    recent_vol = v.tail(5).mean() if n >= 5 else avg_vol
    older_vol = v.iloc[-20:-5].mean() if n >= 20 else avg_vol
    vol_trend = (recent_vol / older_vol - 1) * 100 if older_vol > 0 else 0.0

    # ── 4. True-Range ATR (20-period) ─────────────────────────────────────────
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_20 = tr.rolling(20).mean().iloc[-1] if len(tr) >= 20 else 0.0

    # ── 5. Historical volatility (std dev of daily returns, annualised %) ─────
    returns = c.pct_change().dropna()
    volatility = returns.tail(20).std() * math.sqrt(252) * 100 if len(returns) >= 20 else 0.0

    # ── 6. Momentum (EMA5 vs EMA20 on close) ─────────────────────────────────
    ema5 = c.ewm(span=5, adjust=False).mean()
    ema10 = c.ewm(span=10, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()

    momentum = (ema5.iloc[-1] / ema20.iloc[-1] - 1) * 100 if len(ema5) >= 2 and len(ema20) >= 2 else 0.0
    ema10_slope = (ema10.iloc[-1] / ema10.iloc[max(0, len(ema10)-5)] - 1) * 100 if len(ema10) >= 5 else 0.0
    ema20_slope = (ema20.iloc[-1] / ema20.iloc[max(0, len(ema20)-5)] - 1) * 100 if len(ema20) >= 5 else 0.0

    # ── 7. Historical RSI (14-period) & its slope ─────────────────────────────
    delta = c.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # Use Wilder's smoothing for RSI (com = period - 1)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))

    rsi_hist = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50.0
    rsi_slope = (rsi_series.iloc[-1] - rsi_series.iloc[max(0, len(rsi_series)-4)]) if len(rsi_series) >= 4 else 0.0

    # ── 8. Support / resistance (swing lows & highs last 15 bars) ─────────────
    support_level = l.tail(15).min() if n >= 15 else 0.0
    resistance_level = h.tail(15).max() if n >= 15 else 0.0

    # ── 9. Consecutive direction streak ───────────────────────────────────────
    consec_up = consec_down = 0
    if n > 1:
        diffs = c.diff().dropna()
        for val in reversed(diffs):
            if val > 0:
                if consec_down > 0: break # Streak broken if direction changes
                consec_up += 1
            elif val < 0:
                if consec_up > 0: break # Streak broken if direction changes
                consec_down += 1
            else: # Flat day, consider streak broken
                break

    # ── 10. Higher-lows pattern (last 5 swing lows rising) ───────────────────
    higher_lows = False
    if n >= 3:
        # Check if the last 3 daily lows are successively higher
        if l.iloc[-1] > l.iloc[-2] and l.iloc[-2] > l.iloc[-3]:
            higher_lows = True

    # ── 10b. Triple-lows pattern (bottoming) ──────────────────────────────────
    triple_bottom = False
    if n >= 5: # Need enough data to find potential swing lows
        # A simple approximation for triple bottom: look for three recent lows within a small range
        recent_lows = l.tail(5)
        if len(recent_lows) >= 3:
            min_low = recent_lows.min()
            max_low = recent_lows.max()
            # If the range of recent lows is very small, it could indicate a bottoming pattern
            if (max_low - min_low) / min_low < 0.02: # e.g., within 2%
                triple_bottom = True

    # ── 11. Volume accumulation (rising vol on up-days) ──────────────────────
    vol_accumulation = False
    if n >= 10: # Need at least 10 days for this
        up_days_mask = c.diff() > 0
        down_days_mask = c.diff() < 0
        up_vol = v[up_days_mask].tail(10).sum()
        down_vol = v[down_days_mask].tail(10).sum()
        if down_vol > 0:
            vol_accumulation = up_vol > down_vol * 1.3

    # ── 12. Bollinger Band squeeze (volatility contraction) ───────────────────
    squeeze = False
    if n >= 20: # Need at least 20 days for BB
        bb_std = c.rolling(window=20).std()
        bb_mean = c.rolling(window=20).mean()
        bb_width = ((2 * bb_std) / bb_mean) * 100 # Percentage width
        if bb_width.iloc[-1] < 4.0: # A common heuristic for a tight squeeze
            squeeze = True

    return {
        "has_data": True, "n": n,
        "stability":   round(stability, 2),
        "avg_vol":     avg_vol,
        "vol_trend":   round(vol_trend, 2),
        "trend_pct_30": round(trend_pct_30, 2),
        "trend_pct_10": round(trend_pct_10, 2),
        "volatility":  round(volatility, 2),
        "atr_20":      round(atr_20, 4),
        "momentum":    round(momentum, 3),
        "rsi_hist":    round(rsi_hist, 1),
        "rsi_slope":   round(rsi_slope, 2),
        "ema10_slope": round(ema10_slope, 3),
        "ema20_slope": round(ema20_slope, 3),
        "support_level":    round(support_level, 2),
        "resistance_level": round(resistance_level, 2),
        "consec_up":   consec_up,
        "consec_down": consec_down,
        "higher_lows": higher_lows,
        "vol_accumulation": vol_accumulation,
        "squeeze":     squeeze,
        "triple_bottom": triple_bottom,
    }

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _session():
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=1, status_forcelist=[500,502,503,504]
    )))
    s.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    return s

@st.cache_data(ttl=20, show_spinner=False)
def fetch_live() -> list:
    payload = {
        "sort":    {"sortBy": "volume", "sortOrder": "desc"},
        "filter":  [{"left": "name", "operation": "in_range", "right": KSE100}],
        "markets": ["pakistan"],
        "columns": TV_COLS,
        "range":   [0, 500],
    }
    try:
        r = _session().post(TV_URL, json=payload, timeout=15)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        st.error(f"⚡ Live data error: {e}")
        return []

def calculate_breadth_from_raw(raw: list) -> Tuple[float, bool, int, int, dict]:
    """Optimized: Calculates breadth from existing live data instead of second API call."""
    chgs = []
    adv, dec = 0, 0
    kse = {"close": 0.0, "change": 0.0, "high": 0.0, "low": 0.0, "volume": 0}
    for item in raw:
        d = item.get("d", [])
        sym = d[0]
        chg = safe(d[2])
        if sym in KSE100:
            chgs.append(chg)
            if chg > 0: adv += 1
            elif chg < 0: dec += 1

    avg = sum(chgs) / len(chgs) if chgs else 0.0
    bull = avg > CFG["BREADTH_MIN"] and adv > dec
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
# ███████╗ ██████╗ ██████╗ ██████╗ ██╗███╗   ██╗ ██████╗
# ██╔════╝██╔════╝██╔═══██╗██╔══██╗██║████╗  ██║██╔════╝
# ███████╗██║     ██║   ██║██████╔╝██║██╔██╗ ██║██║  ███╗
# ╚════██║██║     ██║   ██║██╔══██╗██║██║╚██╗██║██║   ██║
# ███████║╚██████╗╚██████╔╝██║  ██║██║██║ ╚████║╚██████╔╝
# ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝
# 7-LAYER INSTITUTIONAL SIGNAL ENGINE
# Layer 1: Trend Structure  | Layer 2: Momentum Cascade
# Layer 3: Volume Footprint | Layer 4: Price Pattern
# Layer 5: Breadth/Context  | Layer 6: Volatility State
# Layer 7: Historical Quality
# ══════════════════════════════════════════════════════════════════════════════

def _compute_atr_stop(price, atr, hist_atr, mult):
    """Use live ATR if available, else fall back to historical ATR."""
    effective_atr = atr if atr > price * 0.002 else hist_atr if hist_atr > 0 else price * 0.015
    return round(price - mult * effective_atr, 2), effective_atr

def _bb_position(price, bb_low, bb_high, bb_basis):
    """Returns position within Bollinger Bands (0=lower, 0.5=mid, 1=upper)."""
    span = bb_high - bb_low
    if span <= 0: return 0.5
    return max(0.0, min(1.0, (price - bb_low) / span))

# ─────────────────────────────────────────────────────────────────────────────
# INTRADAY SCALP SCORER  (max possible ≈ 30)
# Philosophy: institutional footprints + VWAP anchoring + trend velocity
# ─────────────────────────────────────────────────────────────────────────────
def score_intraday(
    price, change, rsi, macd, macd_sig, macd_hist,
    ema5, ema10, vwap, adx, stoch_k, stoch_d,
    vol, avg_vol, atr, bb_low, bb_high, bb_basis,
    open_p, high_d, low_d,
    bullish, mkt_chg, hist, rsi_prev
) -> Tuple[int, List[str], float, float, int]:

    score   = 0
    reasons = []

    # ── PRE-FLIGHT GATES (disqualify immediately) ─────────────────────────────
    vol_ratio = vol / avg_vol if avg_vol > 0 else 0
    if price < CFG["MIN_PRICE"]:                return 0, [], 0, 0, 0
    if vol < CFG["MIN_VOLUME"]:                 return 0, [], 0, 0, 0
    if vol_ratio < 0.7:                         return 0, [], 0, 0, 0   # Reduced from 1.0 to catch early moves
    if price < vwap * 0.995:                    return 0, [], 0, 0, 0   # Allow 0.5% wiggle room below VWAP

    # ── LAYER 3: VOLUME FOOTPRINT ─────────────────────────────────────────────
    if vol_ratio >= 1.0: score += 1 # Small bonus for crossing the average
    if vol_ratio >= CFG["INST_VOL_X"]:
        score += 7; reasons.append(f"🐋 {vol_ratio:.1f}x Inst.Vol")
    elif vol_ratio >= 2.0:
        score += 4; reasons.append(f"{vol_ratio:.1f}x Vol Behavior")
    elif vol_ratio >= 1.5:
        score += 2

    # ── LAYER 4: PRICE PATTERN (VWAP ZONE) ───────────────────────────────────
    vwap_dist = (price - vwap) / vwap * 100 if vwap > 0 else 99
    if 0 < vwap_dist < 0.5:
        score += 6; reasons.append("🎯 VWAP Bounce")
    elif 0.5 <= vwap_dist < 1.2:
        score += 3; reasons.append("VWAP Edge")

    if hist["squeeze"] and change > 1.0:
        score += 2; reasons.append("BB Squeeze Break")

    # ── LAYER 1: TREND STRUCTURE ──────────────────────────────────────────────
    if open_p > 0 and price > open_p and change > 1.0:
        score += 2; reasons.append("Gap Bull")

    if ema5 > 0 and ema10 > 0:
        if price > ema5 > ema10:
            score += 4; reasons.append("High Trend Quality")
        elif price > ema5 and price > ema10:
            score += 2; reasons.append("Above EMAs")

    if adx >= 35:
        score += 5; reasons.append(f"ADX {adx:.0f} Strong")
    elif adx >= 25:
        score += 3; reasons.append(f"ADX {adx:.0f}")

    # ── LAYER 2: MOMENTUM CASCADE ─────────────────────────────────────────────
    rs_alpha = change - mkt_chg
    if rs_alpha > 1.5:
        score += 3; reasons.append(f"RS +{rs_alpha:.1f}%")

    if macd > macd_sig:
        if macd_hist > 0 and macd > 0:
            score += 3; reasons.append("MACD↑ Bull")

    rsi_delta = rsi - rsi_prev if rsi_prev > 0 else 0
    if rsi < 40 and rsi_delta > 4:
        score += 5; reasons.append("Oversold Momentum Turn")
    elif 52 < rsi < 75:
        score += 2; reasons.append(f"RSI {rsi:.0f}")
        if rsi_delta > 3:
            score += 1; reasons.append("Momentum Turn")
    elif rsi >= 75:
        score -= 3; reasons.append("⚠️ Overbought") # Penalize instead of killing

    if stoch_k > stoch_d and 30 < stoch_k < 85:
        score += 2; reasons.append("Stoch Turn")

    if bb_basis > 0:
        dist_basis = (price / bb_basis - 1) * 100
        if 0 < dist_basis < 1.5:
            score += 3; reasons.append("Mean-Rev Anchor")

    day_range = high_d - low_d
    if day_range > 0:
        candle_pos = (price - low_d) / day_range
        if candle_pos > 0.7:
            score += 2; reasons.append("Day High Zone")

    if hist["vol_accumulation"]:
        score += 1; reasons.append("Vol Accumulation")

    if hist["momentum"] > 0.5:
        score += 1; reasons.append("Hist Mom↑")

    if hist["volatility"] > 45:
        score += 4; reasons.append("High Beta 🔥")

    # ── PENALTIES & CONTEXT ──────────────────────────────────────────────────
    if adx < 20: score -= 2
    if day_range > 0 and (price - low_d) / day_range < 0.3: score -= 1
    if macd < macd_sig: score -= 1
    if rsi < 50: score -= 1
    if stoch_k > 85: score -= 1
    if vwap_dist > 3.0: score -= 2

    # Bollinger Bands: position within bands
    bb_pos = _bb_position(price, bb_low, bb_high, bb_basis)
    if 0.45 < bb_pos < 0.75:
        score += 1  # mid-upper band — healthy zone
    elif bb_pos > 0.92:
        score -= 2  # upper Bollinger band = stretched

    # Historical squeeze breakout = explosive move coming
    if hist["squeeze"] and change > 1.0:
        score += 2; reasons.append("BB Squeeze Break")

    # ── LAYER 5: BREADTH & CONTEXT ────────────────────────────────────────────
    if bullish:
        score += 2
    else:
        score -= 3  # never fight a bear market intraday

    # ── LAYER 6: VOLATILITY STATE ─────────────────────────────────────────────
    # Annualised vol: prefer moderate (not too calm, not chaotic)
    hvol = hist["volatility"]
    if 20 < hvol < 55:
        score += 1  # tradeable volatility
    elif hvol > 80:
        score -= 2  # too wild — unpredictable

    # ── LAYER 7: HISTORICAL QUALITY ───────────────────────────────────────────
    if hist["momentum"] > 0.5:
        score += 1; reasons.append("Hist Mom↑")
    if hist["consec_up"] >= 3:
        score += 1  # momentum streak

    # ── TARGET & STOP ─────────────────────────────────────────────────────────
    stop, _ = _compute_atr_stop(price, atr, hist["atr_20"], 0.6)
    target = round(price * 1.045, 2) # 4.5% target

    # Simplified prev-score estimation
    prev_score = score - 3 if rsi > rsi_prev else score + 2

    return score, reasons, target, stop, prev_score


# ─────────────────────────────────────────────────────────────────────────────
# SWING TRADE SCORER  (max possible ≈ 36)
# Philosophy: trend alignment + value entry + quality accumulation pattern
# ─────────────────────────────────────────────────────────────────────────────
def score_swing(
    price, change, rsi, macd, macd_sig, macd_hist,
    ema5, ema10, ema20, ema50, vwap,
    adx, atr, stoch_k, stoch_d,
    bb_low, bb_high, bb_basis,
    vol, avg_vol, chg1w, chg1m, low1m, high1m,
    bullish, mkt_chg, rsi_prev, hist
) -> Tuple[int, List[str], float, float, int]:

    score   = 0
    reasons = []

    # ── PRE-FLIGHT GATES ──────────────────────────────────────────────────────
    if price < CFG["MIN_PRICE"]:                return 0, [], 0, 0, 0
    if vol < CFG["MIN_VOLUME"] * 0.8:          return 0, [], 0, 0, 0
    if price < ema50 * 0.985:                   return 0, [], 0, 0, 0  # Allow minor dip below EMA50
    if adx < 12:                                return 0, [], 0, 0, 0  # Catching slightly earlier trends

    if rsi > 72: score -= 5; reasons.append("⚠️ RSI High") # Penalize instead of block

    # ── LAYER 3: VOLUME FOOTPRINT ─────────────────────────────────────────────
    vol_ratio = vol / avg_vol if avg_vol > 0 else 0
    if vol_ratio >= 2.5:
        score += 4; reasons.append(f"🐋 {vol_ratio:.1f}x Vol")
    elif vol_ratio >= 1.7:
        score += 2; reasons.append(f"{vol_ratio:.1f}x Vol Behavior")

    if hist["squeeze"]:
        score += 5; reasons.append("BB Squeeze")

    # ── LAYER 4: PRICE PATTERN ────────────────────────────────────────────────
    if high1m > low1m > 0:
        range1m = high1m - low1m
        pos1m   = (price - low1m) / range1m
        if 0.08 < pos1m < 0.30:
            score += 4; reasons.append("🎯 Early Cycle")

    if hist["support_level"] > 0 and price > 0:
        support_gap = (price / hist["support_level"] - 1) * 100
        if 0 < support_gap < 4:
            score += 2; reasons.append("Mean-Rev Distance ✓")

    if bb_basis > 0:
        dist_basis = (price / bb_basis - 1) * 100
        if -1 < dist_basis < 2:
            score += 3; reasons.append("Mean-Rev Anchor")

    # ── LAYER 1: TREND STRUCTURE ──────────────────────────────────────────────
    if ema5 > 0 and price > ema5 > ema10 > ema20 > ema50:
        score += 10; reasons.append("High Trend Quality")
    elif price > ema10 > ema20 > ema50:
        score += 7; reasons.append("EMA Fan")

    if hist["higher_lows"]:
        score += 3; reasons.append("Higher Lows ✓")

    if adx >= 30:
        score += 5; reasons.append(f"ADX {adx:.0f}")

    # ── LAYER 2: MOMENTUM CASCADE ─────────────────────────────────────────────
    rs_alpha = change - mkt_chg
    if rs_alpha > 1.0:
        score += 2; reasons.append(f"RS +{rs_alpha:.1f}%")

    rsi_delta = rsi - rsi_prev if rsi_prev > 0 else 0
    if rsi < 42 and rsi_delta > 3:
        score += 5; reasons.append("Oversold Momentum Turn")
    elif 45 < rsi < 60:
        score += 4; reasons.append(f"RSI {rsi:.0f} Ideal")

    if macd > macd_sig and macd_hist > 0:
        score += 3; reasons.append("Momentum Turn")

    if hist["ema10_slope"] > 0 and hist["ema20_slope"] > 0:
        score += 3; reasons.append("Trend Quality+")

    if hist["vol_accumulation"]:
        score += 2; reasons.append("Accumulation")

    if hist["momentum"] > 1.0:
        score += 2; reasons.append("Hist Mom↑")

    if hist["volatility"] > 45:
        score += 5
        reasons.append("High Beta 🔥")

    # MACD: full confirmation
    if macd > macd_sig:
        if macd_hist > 0:
            score += 3; reasons.append("MACD Bull Hist")
        else:
            score += 1
    elif macd < macd_sig and macd_hist < 0:
        score -= 2

    # Stochastic %K crossing above %D in non-overbought zone
    if stoch_k > stoch_d and stoch_k < 80:
        score += 2; reasons.append("Stoch Cross")

    # Historical momentum (EMA5 vs EMA20)
    if hist["momentum"] > 1.0:
        score += 2; reasons.append("Hist Mom↑")
    elif hist["momentum"] < -1.0:
        score -= 1

    if hist["trend_pct_10"] > 2:
        score += 2

    # Weekly momentum (not in freefall)
    if chg1w > 2:
        score += 2; reasons.append("Wk Mom↑")
    elif chg1w > 0:
        score += 1
    elif chg1w < -5:
        score -= 2

    # ── LAYER 5: BREADTH & CONTEXT ────────────────────────────────────────────
    rs_alpha = change - mkt_chg
    if rs_alpha > 1.0:
        score += 2; reasons.append(f"RS +{rs_alpha:.1f}%")

    if bullish:
        score += 2
    else:
        score -= 3

    # ── LAYER 6: VOLATILITY STATE ─────────────────────────────────────────────
    hvol = hist["volatility"]
    if 15 < hvol < 50:
        score += 1  # ideal swing volatility
    elif hvol > 70:
        score -= 2  # erratic

    # Consecutive up-days streak
    if hist["consec_up"] >= 2:
        score += 1
    elif hist["consec_down"] >= 4:
        score -= 2

    # ── LAYER 7: HISTORICAL QUALITY ───────────────────────────────────────────
    if hist["stability"] >= 7:
        score += 3; reasons.append("💎 Stable")
    elif hist["stability"] >= 5:
        score += 1
    elif hist["stability"] < 3:
        score -= 2

    if hist["trend_pct_30"] > 8:
        score += 2; reasons.append("30d Uptrend")

    # ── TARGET & STOP ─────────────────────────────────────────────────────────
    stop, _ = _compute_atr_stop(price, atr, hist["atr_20"], 1.5)
    target = round(price * 1.075, 2) # 7.5% target
    prev_score = score - 4 if rsi_delta > 0 else score + 2

    return score, reasons, target, stop, prev_score


# ─────────────────────────────────────────────────────────────────────────────
# LONG-TERM INVESTMENT SCORER  (max possible ≈ 35)
# Philosophy: sector quality + deep value + structural reversal + accumulation
# ─────────────────────────────────────────────────────────────────────────────
def score_longterm(
    price, rsi, macd, macd_sig, macd_hist,
    ema20, ema50, stoch_k, stoch_d,
    bb_low, bb_high, bb_basis,
    vol, avg_vol, chg1w, chg1m, low1m, high1m,
    sector, rsi_prev, hist
) -> Tuple[int, List[str], float, float, int]:

    score   = 0
    reasons = []

    quality = SECTORS.get(sector, {"quality": 5})["quality"]

    # ── PRE-FLIGHT GATES ──────────────────────────────────────────────────────
    if price < CFG["MIN_PRICE"]:                return 0, [], 0, 0, 0
    if quality < 6:                             return 0, [], 0, 0, 0  # low quality sectors excluded
    if hist["stability"] < 2.0:                return 0, [], 0, 0, 0  # structurally broken

    if rsi > 75: score -= 10; reasons.append("⚠️ Expensive")
    if chg1m < -25: score -= 5; reasons.append("⚠️ Falling Knife")

    # ── LAYER 1: SECTOR & FUNDAMENTAL QUALITY ─────────────────────────────────
    if quality == 9:
        score += 5
    elif quality == 8:
        score += 4
    elif quality == 7:
        score += 3
    else:
        score += 1

    # ── LAYER 2: VALUE ZONE DETECTION ─────────────────────────────────────────
    if high1m > low1m > 0:
        dist_low = (price / low1m - 1) * 100
        if hist["triple_bottom"] and dist_low < 5:
            score += 12; reasons.append("🧱 Triple Bottom")
        elif 0.3 < dist_low < 4 and rsi > 28:
            score += 6; reasons.append("🔄 Double Bottom")
        elif dist_low < 10:
            score += 4; reasons.append("💰 Value Zone")

    if bb_basis > 0:
        dist_basis = (price / bb_basis - 1) * 100
        if dist_basis < 0: # Price below mean
            score += 5; reasons.append("Mean-Rev Discount")

    if hist["support_level"] > 0 and price > 0:
        gap = (price / hist["support_level"] - 1) * 100
        if 0 < gap < 3:
            score += 3; reasons.append("Mean-Rev Distance ✓")

    # ── LAYER 5: VOLUME PATTERN ──────────────────────────────────────────────
    if hist["vol_accumulation"]:
        score += 3; reasons.append("Vol Behavior Acc.")

    # ── LAYER 4: TREND INFRASTRUCTURE ────────────────────────────────────────
    if price > ema50:
        score += 3; reasons.append("Above EMA50")
    if hist["ema20_slope"] > 0:
        score += 2; reasons.append("Trend Quality+")
    if hist["higher_lows"]:
        score += 3; reasons.append("Higher Lows ✓")

    # ── LAYER 3: REVERSAL MOMENTUM ────────────────────────────────────────────
    rsi_delta = rsi - rsi_prev if rsi_prev > 0 else 0
    if 20 < rsi < 35:
        score += 4; reasons.append("Oversold Momentum")
    elif 40 <= rsi < 50:
        score += 3; reasons.append(f"RSI {rsi:.0f} Reset")
    elif 50 <= rsi < 60:
        score += 1

    if rsi_delta > 3:
        score += 2; reasons.append("Momentum Turn")

    # Stochastic oversold recovery
    if stoch_k > stoch_d and stoch_k < 50:
        score += 2; reasons.append("Stoch Recovery")

    # MACD turning (histogram improving)
    if macd_hist > 0:
        score += 2; reasons.append("MACD Hist+")
    elif macd > macd_sig:
        score += 1

    # ── LAYER 4: TREND INFRASTRUCTURE ────────────────────────────────────────
    if price > ema50:
        score += 3; reasons.append("Above EMA50")
    elif price > ema50 * 0.97:
        score += 1  # just below — close to reclaim

    if price > ema20:
        score += 2; reasons.append("Above EMA20")

    if hist["ema20_slope"] > 0.2:
        score += 2; reasons.append("EMA20 Rising")

    if hist["higher_lows"]:
        score += 3; reasons.append("Higher Lows ✓")

    # ── LAYER 5: VOLUME PATTERN (Wyckoff accumulation) ───────────────────────
    if hist["vol_accumulation"]:
        score += 3; reasons.append("📊 Wyckoff Acc.")
    vol_ratio = vol / avg_vol if avg_vol > 0 else 0
    if vol_ratio >= 1.5 and rsi < 50:
        score += 2; reasons.append("Vol+RSI Reset")

    # BB: price near lower band = deep value
    bb_pos = _bb_position(price, bb_low, bb_high, bb_basis)
    if bb_pos < 0.15:
        score += 3; reasons.append("Near BB Low")
    elif bb_pos < 0.35:
        score += 1

    # BB Squeeze = big move pending
    if hist["squeeze"]:
        score += 2; reasons.append("BB Squeeze")

    # ── LAYER 6: WEEKLY & MONTHLY CONTEXT ────────────────────────────────────
    if -3 < chg1w < 3:
        score += 1  # stabilising
    elif chg1w < -8:
        score -= 2  # still crashing

    if chg1m > -5:
        score += 1
    elif chg1m < -15:
        score -= 1

    # ── LAYER 7: HISTORICAL QUALITY ───────────────────────────────────────────
    if hist["stability"] >= 6:
        score += 2
    if hist["trend_pct_30"] > 5:
        score += 1; reasons.append("30d Trend")
    if hist["volatility"] < 40:
        score += 1  # controlled, not chaotic

    # ── TARGET & STOP ─────────────────────────────────────────────────────────
    if low1m > 0 and (price / low1m - 1) * 100 < 6:
        target = round(price * 1.55, 2)  # deeper value = bigger target
    elif low1m > 0 and (price / low1m - 1) * 100 < 15:
        target = round(price * 1.30, 2)
    else:
        target = round(price * 1.20, 2)

    prev_score = score - 2 if rsi > rsi_prev else score + 1
    stop = max(round(price * 0.88, 2), round(hist["support_level"] * 0.97, 2)) if hist["support_level"] > 0 else round(price * 0.88, 2)

    return score, reasons, target, stop, prev_score

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSOR — runs all 3 engines per stock
# ══════════════════════════════════════════════════════════════════════════════

THRESH_INTRA = 18
THRESH_SWING = 21
THRESH_LONG  = 18

def process_signals(raw: list, bullish: bool, mkt_chg: float):
    if not raw:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    intra, swing, long_ = [], [], []

    # Open DB connection once for the entire processing loop
    conn = sqlite3.connect(CFG["DB_PATH"])

    for item in raw:
        d = item.get("d", [])
        # Need at least the core 27 cols (rest are optional)
        if len(d) < 27 or not d[0]:
            continue

        sym      = d[0]
        price    = safe(d[1])
        change   = safe(d[2])
        vol      = safe(d[3])
        rv       = safe(d[4])
        avg_vol  = safe(d[5])
        rsi      = safe(d[6], 50)
        macd     = safe(d[7])
        macd_sig = safe(d[8])
        bb_low   = safe(d[9])
        bb_high  = safe(d[10])
        ema20    = safe(d[11])
        ema50    = safe(d[12])
        chg1w    = safe(d[13])
        high1m   = safe(d[14])
        low1m    = safe(d[15])
        vwap     = safe(d[16]) or price
        ema10    = safe(d[17])
        adx      = safe(d[18])
        atr      = safe(d[19])
        stoch_k  = safe(d[20], 50)
        stoch_d  = safe(d[21], 50)
        open_p   = safe(d[22])
        high_d   = safe(d[23])
        low_d    = safe(d[24])
        ema5     = safe(d[25])
        chg1m    = safe(d[26])
        rsi_prev = safe(d[27], rsi) if len(d) > 27 else rsi
        macd_h   = safe(d[28]) if len(d) > 28 else (macd - macd_sig)
        bb_basis = safe(d[30]) if len(d) > 30 else (bb_low + bb_high) / 2

        if price <= 0 or avg_vol <= 0:
            continue

        sector = SYM_SECTOR.get(sym, "Misc")
        hist = get_hist_metrics(sym, conn)

        # Standard Pivot Calculation (Floor Pivots)
        h_piv, l_piv = (high_d if high_d > 0 else price), (low_d if low_d > 0 else price)
        p_piv = (h_piv + l_piv + price) / 3
        r1, s1 = round(2 * p_piv - l_piv, 2), round(2 * p_piv - h_piv, 2)
        r2, s2 = round(p_piv + (h_piv - l_piv), 2), round(p_piv - (h_piv - l_piv), 2)

        # Dynamic Best Buy/Sell based on Pivot Theory
        best_buy = s1 if price > s1 else s2
        best_sell = r1 if price < r1 else r2

        # Trend Bias Logic
        is_buying = (price > vwap) and (macd > macd_sig) and (rsi > 50)
        trend_label = "Buying" if is_buying else "Selling"

        # Minimum liquidity: both absolute floor AND 30% of historical average
        if vol < max(CFG["MIN_VOLUME"], hist["avg_vol"] * 0.30):
            continue

        # ── INTRADAY ──────────────────────────────────────────────────────────
        sc, rs, tgt, stp, p_sc = score_intraday(
            price, change, rsi, macd, macd_sig, macd_h,
            ema5, ema10, vwap, adx, stoch_k, stoch_d,
            vol, avg_vol, atr, bb_low, bb_high, bb_basis,
            open_p, high_d, low_d,
            bullish, mkt_chg, hist, rsi_prev
        )
        if sc >= THRESH_INTRA and tgt > price > stp:
            intra.append({
                "Symbol": sym, "Sector": sector,
                "Price": round(price, 2), "Trend": trend_label,
                "Chg%": round(change, 2),
                "Score": sc, "Prev": p_sc,
                "Best price to buy": best_buy, "Best Price to sell": best_sell,
                "R1": r1, "R2": r2, "S1": s1, "S2": s2,
                "RV": round(rv, 1),
                "RSI": round(rsi, 0),
                "Reasons": " · ".join(rs),
                "Target": tgt, "Stop": stp,
                "R:R": _rr(price, tgt, stp),
            })

        # ── SWING ─────────────────────────────────────────────────────────────
        sc, rs, tgt, stp, p_sc = score_swing(
            price, change, rsi, macd, macd_sig, macd_h,
            ema5, ema10, ema20, ema50, vwap,
            adx, atr, stoch_k, stoch_d,
            bb_low, bb_high, bb_basis,
            vol, avg_vol, chg1w, chg1m, low1m, high1m,
            bullish, mkt_chg, rsi_prev, hist
        )
        if sc >= THRESH_SWING and tgt > price > stp:
            swing.append({
                "Symbol": sym, "Sector": sector,
                "Price": round(price, 2), "Trend": trend_label,
                "Chg%": round(change, 2),
                "Score": sc, "Prev": p_sc,
                "Best price to buy": best_buy, "Best Price to sell": best_sell,
                "R1": r1, "R2": r2, "S1": s1, "S2": s2,
                "1W%": round(chg1w, 2),
                "Reasons": " · ".join(rs[:2]),
                "RSI": round(rsi, 0),
                "Target": tgt, "Stop": stp,
                "R:R": _rr(price, tgt, stp),
            })

        # ── LONG-TERM ─────────────────────────────────────────────────────────
        perf1m = (price / low1m - 1) * 100 if low1m > 0 else 0.0
        sc, rs, tgt, stp, p_sc = score_longterm(
            price, rsi, macd, macd_sig, macd_h,
            ema20, ema50, stoch_k, stoch_d,
            bb_low, bb_high, bb_basis,
            vol, avg_vol, chg1w, chg1m, low1m, high1m,
            sector, rsi_prev, hist
        )
        if sc >= THRESH_LONG and tgt > price > stp:
            long_.append({
                "Symbol": sym, "Sector": sector,
                "Price": round(price, 2), "Trend": trend_label,
                "1W%": round(chg1w, 2),
                "Score": sc, "Prev": p_sc,
                "Best price to buy": best_buy, "Best Price to sell": best_sell,
                "R1": r1, "R2": r2, "S1": s1, "S2": s2,
                "1M%": round(perf1m, 2),
                "Stab": round(hist["stability"], 1),
                "RSI": round(rsi, 0),
                "Reasons": " · ".join(rs[:2]),
                "Target": tgt, "Stop": stp,
                "R:R": _rr(price, tgt, stp),
            })

    conn.close()
    srt = lambda lst: pd.DataFrame(lst).sort_values("Score", ascending=False).reset_index(drop=True) if lst else pd.DataFrame()
    return srt(intra), srt(swing), srt(long_)
# ══════════════════════════════════════════════════════════════════════════════
# UI — STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════

if "positions" not in st.session_state:
    st.session_state.positions = []
if "trend_history" not in st.session_state:
    st.session_state.trend_history = {}


# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', -apple-system, sans-serif; box-sizing: border-box; }
html, body, [class*="css"] { background: #060b18 !important; color: #e2e8f0 !important; }
footer, #MainMenu { visibility: hidden; }

.badge {
    display: inline-block; padding: 0.3rem 0.85rem; border-radius: 8px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; border: 1px solid;
}
.b-open   { border-color:#10b981; color:#10b981; background:rgba(16,185,129,0.07); }
.b-closed { border-color:#ef4444; color:#ef4444; background:rgba(239,68,68,0.07); }
.b-bull   { border-color:#10b981; color:#10b981; background:rgba(16,185,129,0.12); }
.b-bear   { border-color:#ef4444; color:#ef4444; background:rgba(239,68,68,0.12); }
.b-neut   { border-color:#f59e0b; color:#f59e0b; background:rgba(245,158,11,0.10); }

.sec-wrap { display:flex; flex-wrap:wrap; gap:0.35rem; margin-bottom:1rem; }
.sec-tile {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 4px; padding: 2px 8px;
    display:flex; align-items:center; gap:7px;
}
.sec-name { font-size:0.7rem; font-weight:500; color:#64748b; }
.sec-val  { font-family:'JetBrains Mono',monospace; font-size:0.76rem; font-weight:700; }

[data-testid="stDataFrame"] { background:transparent !important; border-radius:10px; overflow:hidden; }
thead tr th {
    background:rgba(10,18,38,0.9) !important; color:#475569 !important;
    font-family:'JetBrains Mono',monospace !important; font-size:0.66rem !important;
    font-weight:700 !important; letter-spacing:0.1em !important; text-transform:uppercase;
    padding: 10px 8px !important;
}
tbody tr:nth-child(even) { background:rgba(10,18,38,0.3) !important; }
tbody tr:hover { background:rgba(30,58,100,0.2) !important; }

.stButton button {
    background: transparent !important; border: none !important;
    box-shadow: none !important; color: #64748b !important;
    padding: 0 !important; font-size: 1.2rem !important;
    min-height: unset !important; line-height: 1 !important;
}
.stButton button:hover { color: #f8fafc !important; }

.stTextInput>div>div>input, .stNumberInput>div>div>input {
    background:rgba(6,11,24,0.8) !important; border:1px solid rgba(51,65,85,0.6) !important;
    border-radius:9px !important; color:#e2e8f0 !important;
    font-family:'JetBrains Mono',monospace !important; font-size:0.88rem !important;
}
.stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
    border-color:#2563eb !important; box-shadow:0 0 0 2px rgba(37,99,235,0.2) !important;
}

div[data-testid="stExpander"] {
    background: rgba(10,18,38,0.5) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
}

hr { border-color:rgba(30,41,59,0.4) !important; margin: 1.5rem 0 !important; }

/* Mobile Fix: Force Desktop-style Horizontal Header */
@media (max-width: 768px) {
    [data-testid="stHorizontalBlock"] {
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        gap: 0 !important;
    }
    [data-testid="stHorizontalBlock"] > div:nth-child(1) {
        flex: 1 1 auto !important;
        min-width: 0 !important;
    }
    [data-testid="stHorizontalBlock"] > div:nth-child(n+2) {
        flex: 0 0 auto !important;
        width: 45px !important;
        min-width: 45px !important;
    }
    .main .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 1rem !important;
    }
}

.alert-box {
    background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2);
    border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 0.75rem;
    font-size: 0.82rem; color: #fca5a5;
}
.info-box {
    background: rgba(59,130,246,0.07); border: 1px solid rgba(59,130,246,0.2);
    border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 0.75rem;
    font-size: 0.82rem; color: #93c5fd;
}
</style>
""", unsafe_allow_html=True)

# ─── Init ────────────────────────────────────────────────────────────────────
init_db()

# ─── Market State ─────────────────────────────────────────────────────────────
is_open = is_market_open()
raw_data = fetch_live()
avg_chg, bullish, adv, dec, kse_fb = calculate_breadth_from_raw(raw_data)
kse_api = fetch_kse_index()

def _kse(key, fb):
    return kse_api.get(key, fb) if kse_api else fb

idx_close = _kse("close",         kse_fb["close"])
idx_pct   = _kse("changePercent", kse_fb["change"])
idx_vol   = _kse("volume",        kse_fb["volume"])

bclass   = "b-bull" if bullish else ("b-bear" if avg_chg < -0.5 else "b-neut")
vol_cr   = idx_vol / 1e7


h_cols = st.columns([10, 1, 1])
with h_cols[0]:
    st.markdown("<h3 style='margin:0; padding:0; font-size:1.4rem;'>PSX Scanner</h3>", unsafe_allow_html=True)
scan_btn = h_cols[1].button("🔍", help="Scan Now")
sync_btn = h_cols[2].button("🔄", help="Sync History")

st.markdown("<div style='border-bottom: 1px solid rgba(255,255,255,0.15); margin: 4px 0 12px 0;'></div>", unsafe_allow_html=True)

if sync_btn:
    sync_historical_data(KSE100)
    st.rerun()

st.markdown(f"""
<div style="display: flex; flex-direction: row; justify-content: space-between; gap: 15px; margin-bottom: 1rem; border-bottom: 1px solid rgba(30,41,59,0.4); padding-bottom: 10px; overflow-x: auto; -webkit-overflow-scrolling: touch; white-space: nowrap;">
    <div style="flex: 0 0 auto; min-width: 80px;">
        <div style="font-size:0.7rem; color:#64748b; font-weight:700; text-transform:uppercase; white-space: nowrap;">KSE-100</div>
        <div style="font-size:1rem; font-weight:700; white-space: nowrap;">{idx_close:,.0f}</div>
        <div style="font-size:0.75rem; color:{'#10b981' if idx_pct >= 0 else '#ef4444'}; font-weight:400; white-space: nowrap;">{idx_pct:+.2f}%</div>
    </div>
    <div style="flex: 0 0 auto; min-width: 70px;">
        <div style="font-size:0.7rem; color:#64748b; font-weight:700; text-transform:uppercase; white-space: nowrap;">State</div>
        <div style="font-size:1rem; font-weight:700; white-space: nowrap;">{'LIVE' if is_open else 'CLOSED'}</div>
    </div>
    <div style="flex: 0 0 auto; min-width: 90px;">
        <div style="font-size:0.7rem; color:#64748b; font-weight:700; text-transform:uppercase; white-space: nowrap;">A/D Ratio</div>
        <div style="font-size:1rem; font-weight:700; white-space: nowrap;">{adv}▲/{dec}▼</div>
    </div>
    <div style="flex: 0 0 auto; min-width: 80px;">
        <div style="font-size:0.7rem; color:#64748b; font-weight:700; text-transform:uppercase; white-space: nowrap;">Volume</div>
        <div style="font-size:1rem; font-weight:700; white-space: nowrap;">{vol_cr:.1f} Cr</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Market breadth warning ───────────────────────────────────────────────────
if not bullish:
    st.markdown(f'<div class="alert-box">⚠️ <strong>Bearish Breadth</strong> — Market declining ({adv} advancers vs {dec} decliners). Reduce position sizes. Intraday setups require extra confirmation.</div>', unsafe_allow_html=True)

scan = is_open or scan_btn

if scan:
    with st.spinner("🔬 Scanning KSE-100 across 7 signal layers…"):
        raw = fetch_live()
        save_snapshot(raw)

    if not raw:
        st.warning("No data returned — check network or try again.")
        st.stop()

    # ── Sector Heatmap ────────────────────────────────────────────────────────
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
        avg  = sum(chgs) / len(chgs)
        col  = "#10b981" if avg >= 0.5 else ("#ef4444" if avg < -0.5 else "#f59e0b")
        q    = SECTORS[sec]["quality"]
        tiles += f'<div class="sec-tile"><span class="sec-name">{html.escape(sec)}</span><span class="sec-val" style="color:{col};">{avg:+.1f}%</span></div>'

    st.markdown(f'<div class="sec-wrap">{tiles}</div>', unsafe_allow_html=True)

    # ── Run Signals ───────────────────────────────────────────────────────────
    df_i, df_s, df_l = process_signals(raw, bullish, avg_chg)
    thresh_i, thresh_s, thresh_l = THRESH_INTRA, THRESH_SWING, THRESH_LONG
    if not df_i.empty: df_i = df_i[df_i["Score"] >= thresh_i].reset_index(drop=True)
    if not df_s.empty: df_s = df_s[df_s["Score"] >= thresh_s].reset_index(drop=True)
    if not df_l.empty: df_l = df_l[df_l["Score"] >= thresh_l].reset_index(drop=True)

    def render_table(df: pd.DataFrame, col_cfg: dict, score_max: int):
        """Renders the main dataframes, applying styling for trend changes."""
        if df.empty:
            st.write("No setups meet current threshold.")
        else:
            df_top = df.head(7).copy()

            def style_trend(row):
                prev_trend = st.session_state.trend_history.get(row['Symbol'])
                current_trend = row['Trend']
                style = ''
                if prev_trend:
                    if prev_trend == 'Selling' and current_trend == 'Buying':
                        style = 'background-color: rgba(16, 185, 129, 0.25);' # Green highlight
                    elif prev_trend == 'Buying' and current_trend == 'Selling':
                        style = 'background-color: rgba(239, 68, 68, 0.2);' # Red highlight
                return [style if col == 'Trend' else '' for col in df_top.columns]

            st.dataframe(
                df_top.style.apply(style_trend, axis=1),
                column_config=col_cfg,
                hide_index=True,
                use_container_width=True
            )

    def render_simple_table(df: pd.DataFrame, col_cfg: dict):
        """Renders a simple table without styling or limits."""
        if df.empty:
            st.write("No setups meet current threshold.")
        else:
            # Add score% column
            df_top = df.head(7).copy()
            st.dataframe(df_top, column_config=col_cfg, hide_index=True, use_container_width=True)

    # ── INTRADAY ──────────────────────────────────────────────────────────────
    st.markdown(f"<div style='margin-top:1rem; margin-bottom:5px;'><b>INTRADAY SCALPS</b> &nbsp;&nbsp; <span style='font-size:0.8rem; color:#64748b;'>Results: {len(df_i)} &nbsp; Threshold: {thresh_i}/30 &nbsp; Breadth: <span style='color:{'#10b981' if bullish else '#ef4444'}'>{'✅ Bullish' if bullish else '⚠️ Bearish'}</span></span></div>", unsafe_allow_html=True)

    render_table(df_i, {
        "Trend":  st.column_config.TextColumn("Trend"),
        "Price":  st.column_config.NumberColumn("Price",  format="%.2f"),
        "Chg%":   st.column_config.NumberColumn("Chg%",   format="%.2f%%"),
        "Score":  st.column_config.NumberColumn("Score",  help="Today's Institutional Score"),
        "Best price to buy": st.column_config.NumberColumn("Best Buy", format="%.2f"),
        "Best Price to sell": st.column_config.NumberColumn("Best Sell", format="%.2f"),
        "R1": st.column_config.NumberColumn("R1", format="%.2f"),
        "R2": st.column_config.NumberColumn("R2", format="%.2f"),
        "S1": st.column_config.NumberColumn("S1", format="%.2f"),
        "S2": st.column_config.NumberColumn("S2", format="%.2f"),
        "Prev":   st.column_config.NumberColumn("Prev",   help="Yesterday's Score"),
        "RV":     st.column_config.NumberColumn("RV",     format="%.1fx"),
        "RSI":    st.column_config.NumberColumn("RSI",    format="%d"),
        "Reasons": st.column_config.TextColumn("Signals"),
        "Target": st.column_config.NumberColumn("Target", format="%.2f"),
        "Stop":   st.column_config.NumberColumn("Stop",   format="%.2f"),
    }, score_max=30)

    # ── SWING ────────────────────────────────────────────────────────────────
    st.markdown(f"<div style='margin-top:1.5rem; margin-bottom:5px;'><b>SWING TRADES</b> &nbsp;&nbsp; <span style='font-size:0.8rem; color:#64748b;'>Results: {len(df_s)} &nbsp; Threshold: {thresh_s}/36 &nbsp; Breadth: <span style='color:{'#10b981' if bullish else '#ef4444'}'>{'✅ Bullish' if bullish else '⚠️ Bearish'}</span></span></div>", unsafe_allow_html=True)

    render_table(df_s, {
        "Trend":  st.column_config.TextColumn("Trend"),
        "Price":  st.column_config.NumberColumn("Price",  format="%.2f"),
        "Chg%":   st.column_config.NumberColumn("Chg%",   format="%.2f%%"),
        "Score":  st.column_config.NumberColumn("Score"),
        "Best price to buy": st.column_config.NumberColumn("Best Buy", format="%.2f"),
        "Best Price to sell": st.column_config.NumberColumn("Best Sell", format="%.2f"),
        "R1": st.column_config.NumberColumn("R1", format="%.2f"),
        "R2": st.column_config.NumberColumn("R2", format="%.2f"),
        "S1": st.column_config.NumberColumn("S1", format="%.2f"),
        "S2": st.column_config.NumberColumn("S2", format="%.2f"),
        "Prev":   st.column_config.NumberColumn("Prev"),
        "1W%":    st.column_config.NumberColumn("1W%",    format="%.2f%%"),
        "RSI":    st.column_config.NumberColumn("RSI",    format="%d"),
        "Reasons": st.column_config.TextColumn("Signals"),
        "Target": st.column_config.NumberColumn("Target", format="%.2f"),
        "Stop":   st.column_config.NumberColumn("Stop",   format="%.2f"),
    }, score_max=36)

    # ── LONG-TERM ────────────────────────────────────────────────────────────
    st.markdown(f"<div style='margin-top:1.5rem; margin-bottom:5px;'><b>LONG-TERM INVESTMENTS</b> &nbsp;&nbsp; <span style='font-size:0.8rem; color:#64748b;'>Results: {len(df_l)} &nbsp; Threshold: {thresh_l}/35</span></div>", unsafe_allow_html=True)

    render_table(df_l, {
        "Trend":  st.column_config.TextColumn("Trend"),
        "Price":  st.column_config.NumberColumn("Price",  format="%.2f"),
        "1W%":    st.column_config.NumberColumn("1W%",    format="%.2f%%"),
        "Score":  st.column_config.NumberColumn("Score"),
        "Best price to buy": st.column_config.NumberColumn("Best Buy", format="%.2f"),
        "Best Price to sell": st.column_config.NumberColumn("Best Sell", format="%.2f"),
        "R1": st.column_config.NumberColumn("R1", format="%.2f"),
        "R2": st.column_config.NumberColumn("R2", format="%.2f"),
        "S1": st.column_config.NumberColumn("S1", format="%.2f"),
        "S2": st.column_config.NumberColumn("S2", format="%.2f"),
        "Prev":   st.column_config.NumberColumn("Prev"),
        "1M%":    st.column_config.NumberColumn("1M%",    format="%.2f%%"),
        "Stab":   st.column_config.NumberColumn("Stab",   format="%.1f"),
        "RSI":    st.column_config.NumberColumn("RSI",    format="%d"),
        "Reasons": st.column_config.TextColumn("Signals"),
        "Target": st.column_config.NumberColumn("Target", format="%.2f"),
        "Stop":   st.column_config.NumberColumn("Stop",   format="%.2f"),
    }, score_max=35)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── TREND REVERSALS ───────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<b>TREND REVERSALS</b> &nbsp;&nbsp; <span style='font-size:0.8rem; color:#64748b;'>Changes since last scan</span>", unsafe_allow_html=True)

    # Combine all valid stocks and get their current trend
    all_stocks_df = pd.concat([df_i, df_s, df_l]).drop_duplicates(subset=['Symbol'])
    current_trends = pd.Series(all_stocks_df.Trend.values, index=all_stocks_df.Symbol).to_dict()

    selling_to_buying = []
    buying_to_selling = []

    for symbol, current_trend in current_trends.items():
        prev_trend = st.session_state.trend_history.get(symbol)
        if prev_trend and prev_trend != current_trend:
            if prev_trend == "Selling" and current_trend == "Buying":
                selling_to_buying.append(f"🟢 {symbol}")
            elif prev_trend == "Buying" and current_trend == "Selling":
                buying_to_selling.append(f"🔴 {symbol}")

    # Update history for the next run
    st.session_state.trend_history.update(current_trends)

    if selling_to_buying or buying_to_selling:
        reversal_html = ""
        if selling_to_buying:
            reversal_html += f"<div style='margin-bottom: 0.5rem;'><b>Bullish:</b> {'&nbsp;·&nbsp;'.join(selling_to_buying)}</div>"
        if buying_to_selling:
            reversal_html += f"<div><b>Bearish:</b> {'&nbsp;·&nbsp;'.join(buying_to_selling)}</div>"
        st.markdown(f"<div class='info-box' style='padding: 0.75rem 1rem;'>{reversal_html}</div>", unsafe_allow_html=True)
    else:
        st.info("No trend reversals detected in this scan.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if is_open:
    time.sleep(CFG["REFRESH_SEC"])
    st.rerun()
