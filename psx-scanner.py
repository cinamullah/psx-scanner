"""
PSX Market Intelligence Report
KSE-100 · 7-Layer Signal Engine · Institutional-Grade Scanner
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
    page_title="PSX Market Intelligence",
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
    "MIN_VOLUME":    100_000,
    "MIN_PRICE":     2.0,
    "INST_VOL_X":   2.5,
    "BREADTH_MIN":   0.0,
    "DB_PATH":       os.path.join(BASE_DIR, "psx_elite.db"),
    "HIST_DAYS":     90,
}

# ── Layer Budgets (total = 100) ───────────────────────────────────────────────
# Each layer has a maximum contribution to the final score.
LAYER_BUDGET = {
    "trend":      20,   # Layer 1: Trend Structure
    "momentum":   20,   # Layer 2: Momentum Cascade
    "volume":     15,   # Layer 3: Volume Footprint
    "pattern":    15,   # Layer 4: Price Pattern
    "breadth":    10,   # Layer 5: Breadth / Context
    "volatility": 10,   # Layer 6: Volatility State
    "historical": 10,   # Layer 7: Historical Quality
}

# ── Thresholds on 0-100 scale ─────────────────────────────────────────────────
THRESH_INTRA = 55
THRESH_SWING = 50
THRESH_LONG  = 45

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

# ── TradingView columns ──────────────────────────────────────────────────────
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
    "RSI[1]",                    # 27
    "low|7D",                    # 28
    "MACD.hist",                 # 29
    "Pivot.M.Classic.Middle",    # 30
    "BB.basis",                  # 31
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

def _rr(price: float, target: float, stop: float) -> float:
    denom = price - stop
    if denom <= 0 or target <= price:
        return 0.0
    return round((target - price) / denom, 2)

def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a value within bounds."""
    return max(low, min(high, value))

def _grade(score: int, layers_active: int) -> str:
    """Assign a letter grade based on score and layer confluence."""
    if score >= 75 and layers_active >= 6:
        return "A"
    elif score >= 60 and layers_active >= 5:
        return "B"
    elif score >= 45 and layers_active >= 4:
        return "C"
    else:
        return "D"

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE — historical OHLCV store
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
                    PRIMARY KEY (date, symbol)
                );
                CREATE INDEX IF NOT EXISTS idx_snapshot_date ON daily_snapshot(date DESC);
            """)
    finally:
        conn.close()

def sync_historical_data(symbols: List[str]):
    """Optimized batch download from Yahoo Finance."""
    end = datetime.now()
    start = end - timedelta(days=CFG["HIST_DAYS"])
    tickers = [f"{s}.KA" for s in symbols]

    ph = st.empty()
    ph.caption("Fetching market history...")

    df = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)

    conn = sqlite3.connect(CFG["DB_PATH"])
    try:
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
                with conn:
                    conn.executemany("INSERT OR REPLACE INTO price_history VALUES (?,?,?,?,?,?,?)", rows)
            except Exception: continue
    finally:
        conn.close()
        ph.empty()

def save_snapshot(raw: list):
    """Persist today's live prices."""
    if not raw:
        return
    today = pkt_now().strftime("%Y-%m-%d")
    conn  = sqlite3.connect(CFG["DB_PATH"])
    try:
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
            with conn:
                conn.executemany("INSERT OR REPLACE INTO price_history VALUES (?,?,?,?,?,?,?)", rows)
    finally:
        conn.close()

def get_yesterday_snapshot() -> Dict[str, Dict]:
    """Fetches the last available daily snapshot from the database."""
    conn = sqlite3.connect(CFG["DB_PATH"])
    try:
        today = pkt_now().strftime("%Y-%m-%d")

        # Find the most recent date in the snapshot table that is not today
        last_date_query = "SELECT MAX(date) FROM daily_snapshot WHERE date < ?"
        cursor = conn.cursor()
        cursor.execute(last_date_query, (today,))
        last_date = cursor.fetchone()[0]

        snapshot = {}
        if last_date:
            df = pd.read_sql(
                "SELECT symbol, trend, rv FROM daily_snapshot WHERE date = ?",
                conn, params=(last_date,)
            )
            for _, row in df.iterrows():
                snapshot[row['symbol']] = {
                    'trend': row['trend'],
                    'rv': row['rv']
                }
        return snapshot
    finally:
        conn.close()

def save_daily_snapshot(df: pd.DataFrame):
    """Saves the current day's trend and RV data to the snapshot table."""
    if df.empty:
        return
    today = pkt_now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(CFG["DB_PATH"])
    try:
        rows = []
        for _, row in df.iterrows():
            rows.append((today, row['Symbol'], row['Bias'], row.get('RV', 0.0)))
        with conn:
            conn.executemany("INSERT OR REPLACE INTO daily_snapshot VALUES (?,?,?,?)", rows)
    finally:
        conn.close()

# ══════════════════════════════════════════════════════════════════════════════
# HISTORICAL ANALYTICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def get_hist_metrics(symbol: str, db_conn: sqlite3.Connection) -> Dict:
    """Vectorized historical analytics."""
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
        "squeeze": False, "triple_bottom": False,
    }

    if len(df) < 20:
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

    # ── 5. Historical volatility (annualised %) ──────────────────────────────
    returns = c.pct_change().dropna()
    volatility = returns.tail(20).std() * math.sqrt(252) * 100 if len(returns) >= 20 else 0.0

    # ── 6. Momentum (EMA5 vs EMA20) ──────────────────────────────────────────
    ema5 = c.ewm(span=5, adjust=False).mean()
    ema10 = c.ewm(span=10, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()

    momentum = (ema5.iloc[-1] / ema20.iloc[-1] - 1) * 100 if len(ema5) >= 2 and len(ema20) >= 2 else 0.0
    ema10_slope = (ema10.iloc[-1] / ema10.iloc[max(0, len(ema10)-5)] - 1) * 100 if len(ema10) >= 5 else 0.0
    ema20_slope = (ema20.iloc[-1] / ema20.iloc[max(0, len(ema20)-5)] - 1) * 100 if len(ema20) >= 5 else 0.0

    # ── 7. Historical RSI (14-period) & slope ────────────────────────────────
    delta = c.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))

    rsi_hist = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50.0
    rsi_slope = (rsi_series.iloc[-1] - rsi_series.iloc[max(0, len(rsi_series)-4)]) if len(rsi_series) >= 4 else 0.0

    # ── 8. Support / resistance (last 15 bars) ──────────────────────────────
    support_level = l.tail(15).min() if n >= 15 else 0.0
    resistance_level = h.tail(15).max() if n >= 15 else 0.0

    # ── 9. Consecutive direction streak ──────────────────────────────────────
    consec_up = consec_down = 0
    if n > 1:
        diffs = c.diff().dropna()
        for val in reversed(diffs):
            if val > 0:
                if consec_down > 0: break
                consec_up += 1
            elif val < 0:
                if consec_up > 0: break
                consec_down += 1
            else:
                break

    # ── 10. Higher-lows pattern ──────────────────────────────────────────────
    higher_lows = False
    if n >= 3:
        if l.iloc[-1] > l.iloc[-2] and l.iloc[-2] > l.iloc[-3]:
            higher_lows = True

    # ── 10b. Triple-bottom pattern ───────────────────────────────────────────
    triple_bottom = False
    if n >= 5:
        recent_lows = l.tail(5)
        if len(recent_lows) >= 3:
            min_low = recent_lows.min()
            max_low = recent_lows.max()
            if (max_low - min_low) / min_low < 0.02:
                triple_bottom = True

    # ── 11. Volume accumulation ──────────────────────────────────────────────
    vol_accumulation = False
    if n >= 10:
        up_days_mask = c.diff() > 0
        down_days_mask = c.diff() < 0
        up_vol = v[up_days_mask].tail(10).sum()
        down_vol = v[down_days_mask].tail(10).sum()
        if down_vol > 0:
            vol_accumulation = up_vol > down_vol * 1.3

    # ── 12. Bollinger Band squeeze ───────────────────────────────────────────
    squeeze = False
    if n >= 20:
        bb_std = c.rolling(window=20).std()
        bb_mean = c.rolling(window=20).mean()
        bb_width = ((2 * bb_std) / bb_mean) * 100
        if bb_width.iloc[-1] < 4.0:
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
        st.error(f"Data fetch error: {e}")
        return []

def calculate_breadth_from_raw(raw: list) -> Tuple[float, bool, int, int, dict]:
    """Calculates breadth from existing live data."""
    chgs = []
    prices = []
    highs = []
    lows = []
    total_volume = 0
    adv, dec = 0, 0
    for item in raw:
        d = item.get("d", [])
        if not d or len(d) < 25:
            continue
        sym = d[0]
        chg = safe(d[2])
        price = safe(d[1])
        vol = safe(d[3])
        high = safe(d[23])
        low = safe(d[24])
        if sym in KSE100:
            chgs.append(chg)
            prices.append(price)
            if high > 0: highs.append(high)
            if low > 0: lows.append(low)
            total_volume += vol
            if chg > 0: adv += 1
            elif chg < 0: dec += 1

    avg = sum(chgs) / len(chgs) if chgs else 0.0
    avg_price = sum(prices) / len(prices) if prices else 0.0
    max_high = max(highs) if highs else 0.0
    min_low = min(lows) if lows else 0.0
    bull = avg > CFG["BREADTH_MIN"] and adv > dec
    
    kse = {
        "close": avg_price,
        "change": avg,
        "high": max_high,
        "low": min_low,
        "volume": total_volume
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
# 7-LAYER NORMALIZED SCORING ENGINE (0–100 scale)
#
# Each layer has a budget. Raw points within a layer are clamped to that
# layer's budget. Final score = sum of all clamped layers.
#
# Layer 1: Trend Structure   (20 pts max)
# Layer 2: Momentum Cascade  (20 pts max)
# Layer 3: Volume Footprint  (15 pts max)
# Layer 4: Price Pattern      (15 pts max)
# Layer 5: Breadth / Context  (10 pts max)
# Layer 6: Volatility State   (10 pts max)
# Layer 7: Historical Quality (10 pts max)
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
# INTRADAY SCALP SCORER  (normalized 0–100)
# ─────────────────────────────────────────────────────────────────────────────
def score_intraday(
    price, change, rsi, macd, macd_sig, macd_hist,
    ema5, ema10, vwap, adx, stoch_k, stoch_d,
    vol, avg_vol, atr, bb_low, bb_high, bb_basis,
    open_p, high_d, low_d,
    bullish, mkt_chg, hist, rsi_prev
) -> Tuple[int, List[str], float, float, str, int]:

    reasons = []
    layers_active = 0  # count how many layers contributed positively

    # ── PRE-FLIGHT GATES ──────────────────────────────────────────────────────
    vol_ratio = vol / avg_vol if avg_vol > 0 else 0
    if price < CFG["MIN_PRICE"]:       return 0, [], 0, 0, "D", 0
    if vol < CFG["MIN_VOLUME"]:        return 0, [], 0, 0, "D", 0
    if vol_ratio < 0.7:               return 0, [], 0, 0, "D", 0
    if price < vwap * 0.98:            return 0, [], 0, 0, "D", 0

    # ── LAYER 1: TREND STRUCTURE (max 20) ─────────────────────────────────────
    L1 = 0
    if open_p > 0 and price > open_p and change > 1.0:
        L1 += 5; reasons.append("Gap Up")
    if ema5 > 0 and ema10 > 0:
        if price > ema5 > ema10:
            L1 += 10; reasons.append("EMA Alignment")
        elif price > ema5 and price > ema10:
            L1 += 5; reasons.append("Above EMAs")
    if adx >= 35:
        L1 += 8; reasons.append(f"ADX {adx:.0f} Strong")
    elif adx >= 25:
        L1 += 5; reasons.append(f"ADX {adx:.0f}")
    elif adx < 20:
        L1 -= 4
    L1 = _clamp(L1, -5, LAYER_BUDGET["trend"])
    if L1 > 0: layers_active += 1

    # ── LAYER 2: MOMENTUM CASCADE (max 20) ────────────────────────────────────
    L2 = 0
    rs_alpha = change - mkt_chg
    if rs_alpha > 1.5:
        L2 += 5; reasons.append(f"RS +{rs_alpha:.1f}%")
    if macd > macd_sig and macd_hist > 0 and macd > 0:
        L2 += 6; reasons.append("MACD Bull")
    elif macd < macd_sig:
        L2 -= 3

    rsi_delta = rsi - rsi_prev if rsi_prev > 0 else 0
    if rsi < 40 and rsi_delta > 4:
        L2 += 8; reasons.append("Oversold Reversal")
    elif 52 < rsi < 75:
        L2 += 4; reasons.append(f"RSI {rsi:.0f}")
        if rsi_delta > 3:
            L2 += 2
    elif rsi >= 75:
        L2 -= 6; reasons.append("Overbought")

    if stoch_k > stoch_d and 30 < stoch_k < 85:
        L2 += 4; reasons.append("Stoch Cross")
    elif stoch_k > 85:
        L2 -= 2
    L2 = _clamp(L2, -5, LAYER_BUDGET["momentum"])
    if L2 > 0: layers_active += 1

    # ── LAYER 3: VOLUME FOOTPRINT (max 15) ────────────────────────────────────
    L3 = 0
    if vol_ratio >= CFG["INST_VOL_X"]:
        L3 += 12; reasons.append(f"{vol_ratio:.1f}x High Volume")
    elif vol_ratio >= 2.0:
        L3 += 8; reasons.append(f"{vol_ratio:.1f}x Volume")
    elif vol_ratio >= 1.5:
        L3 += 4
    elif vol_ratio >= 1.0:
        L3 += 2

    if hist["vol_accumulation"]:
        L3 += 3; reasons.append("Accumulation")
    L3 = _clamp(L3, 0, LAYER_BUDGET["volume"])
    if L3 > 0: layers_active += 1

    # ── LAYER 4: PRICE PATTERN (max 15) ───────────────────────────────────────
    L4 = 0
    vwap_dist = (price - vwap) / vwap * 100 if vwap > 0 else 99
    if -0.5 < vwap_dist < 0.5:
        L4 += 8; reasons.append("VWAP Bounce")
    elif 0.5 <= vwap_dist < 1.2:
        L4 += 4; reasons.append("VWAP Edge")
    elif vwap_dist > 3.0:
        L4 -= 3

    if bb_basis > 0:
        dist_basis = (price / bb_basis - 1) * 100
        if 0 < dist_basis < 1.5:
            L4 += 4; reasons.append("Mean Reversion")

    day_range = high_d - low_d
    if day_range > 0:
        candle_pos = (price - low_d) / day_range
        if candle_pos > 0.7:
            L4 += 3; reasons.append("Day High Zone")
        elif candle_pos < 0.3:
            L4 -= 2

    if hist["squeeze"] and change > 1.0:
        L4 += 4; reasons.append("BB Squeeze Break")

    bb_pos = _bb_position(price, bb_low, bb_high, bb_basis)
    if bb_pos > 0.92:
        L4 -= 3
    L4 = _clamp(L4, -5, LAYER_BUDGET["pattern"])
    if L4 > 0: layers_active += 1

    # ── LAYER 5: BREADTH & CONTEXT (max 10) ───────────────────────────────────
    L5 = 0
    if bullish:
        L5 += 7
    else:
        L5 -= 5; reasons.append("Bearish Breadth")
    L5 = _clamp(L5, -5, LAYER_BUDGET["breadth"])
    if L5 > 0: layers_active += 1

    # ── LAYER 6: VOLATILITY STATE (max 10) ────────────────────────────────────
    L6 = 0
    hvol = hist["volatility"]
    if 20 < hvol < 45:
        L6 += 7  # ideal tradeable range
    elif 45 <= hvol < 65:
        L6 += 3  # elevated but tradeable
    elif hvol >= 65:
        L6 -= 4  # too erratic
    elif hvol <= 10:
        L6 -= 2  # too dead

    if hist["consec_up"] >= 3:
        L6 += 3
    L6 = _clamp(L6, -5, LAYER_BUDGET["volatility"])
    if L6 > 0: layers_active += 1

    # ── LAYER 7: HISTORICAL QUALITY (max 10) ──────────────────────────────────
    L7 = 0
    if hist["momentum"] > 0.5:
        L7 += 5; reasons.append("Hist Momentum")
    if hist["trend_pct_10"] > 2:
        L7 += 3
    if hist["stability"] >= 6:
        L7 += 3
    elif hist["stability"] < 3:
        L7 -= 3
    L7 = _clamp(L7, -3, LAYER_BUDGET["historical"])
    if L7 > 0: layers_active += 1

    # ── FINAL SCORE ──────────────────────────────────────────────────────────
    score = max(0, L1 + L2 + L3 + L4 + L5 + L6 + L7)

    # ── CONFLUENCE GATE: require at least 4 layers contributing ──────────────
    if layers_active < 4:
        score = min(score, THRESH_INTRA - 1)  # cap below threshold

    # ── TARGET & STOP (ATR-based) ─────────────────────────────────────────────
    stop, eff_atr = _compute_atr_stop(price, atr, hist["atr_20"], 0.6)
    # Dynamic target: 2x ATR for intraday, with floor of 2% and cap of 6%
    raw_target_pct = (2.0 * eff_atr / price) * 100
    target_pct = _clamp(raw_target_pct, 2.0, 6.0)
    target = round(price * (1 + target_pct / 100), 2)

    grade = _grade(score, layers_active)

    return score, reasons, target, stop, grade, layers_active


# ─────────────────────────────────────────────────────────────────────────────
# SWING TRADE SCORER  (normalized 0–100)
# ─────────────────────────────────────────────────────────────────────────────
def score_swing(
    price, change, rsi, macd, macd_sig, macd_hist,
    ema5, ema10, ema20, ema50, vwap,
    adx, atr, stoch_k, stoch_d,
    bb_low, bb_high, bb_basis,
    vol, avg_vol, chg1w, chg1m, low1m, high1m,
    bullish, mkt_chg, rsi_prev, hist
) -> Tuple[int, List[str], float, float, str, int]:

    reasons = []
    layers_active = 0

    # ── PRE-FLIGHT GATES ──────────────────────────────────────────────────────
    if price < CFG["MIN_PRICE"]:       return 0, [], 0, 0, "D", 0
    if vol < CFG["MIN_VOLUME"] * 0.8:  return 0, [], 0, 0, "D", 0
    if price < ema50 * 0.985:         return 0, [], 0, 0, "D", 0
    if adx < 12:                      return 0, [], 0, 0, "D", 0

    # ── LAYER 1: TREND STRUCTURE (max 20) ─────────────────────────────────────
    L1 = 0
    if ema5 > 0 and price > ema5 > ema10 > ema20 > ema50:
        L1 += 16; reasons.append("Full EMA Fan")
    elif price > ema10 > ema20 > ema50:
        L1 += 12; reasons.append("EMA Fan")
    elif price > ema20 > ema50:
        L1 += 6

    if hist["higher_lows"]:
        L1 += 5; reasons.append("Higher Lows")

    if adx >= 30:
        L1 += 5; reasons.append(f"ADX {adx:.0f}")
    L1 = _clamp(L1, 0, LAYER_BUDGET["trend"])
    if L1 > 0: layers_active += 1

    # ── LAYER 2: MOMENTUM CASCADE (max 20) ────────────────────────────────────
    L2 = 0
    rs_alpha = change - mkt_chg
    if rs_alpha > 1.0:
        L2 += 4; reasons.append(f"RS +{rs_alpha:.1f}%")

    rsi_delta = rsi - rsi_prev if rsi_prev > 0 else 0
    if rsi < 42 and rsi_delta > 3:
        L2 += 8; reasons.append("Oversold Reversal")
    elif 45 < rsi < 60:
        L2 += 6; reasons.append(f"RSI {rsi:.0f} Ideal")
    elif rsi > 72:
        L2 -= 6; reasons.append("RSI Overbought")

    if macd > macd_sig and macd_hist > 0:
        L2 += 6; reasons.append("MACD Bullish")
    elif macd < macd_sig and macd_hist < 0:
        L2 -= 4

    if stoch_k > stoch_d and stoch_k < 80:
        L2 += 4; reasons.append("Stoch Cross")

    if hist["ema10_slope"] > 0 and hist["ema20_slope"] > 0:
        L2 += 3; reasons.append("Trend Slopes Rising")

    if chg1w > 2:
        L2 += 3; reasons.append("Weekly Momentum")
    elif chg1w > 0:
        L2 += 1
    elif chg1w < -5:
        L2 -= 3
    L2 = _clamp(L2, -5, LAYER_BUDGET["momentum"])
    if L2 > 0: layers_active += 1

    # ── LAYER 3: VOLUME FOOTPRINT (max 15) ────────────────────────────────────
    L3 = 0
    vol_ratio = vol / avg_vol if avg_vol > 0 else 0
    if vol_ratio >= 2.5:
        L3 += 10; reasons.append(f"{vol_ratio:.1f}x High Volume")
    elif vol_ratio >= 1.7:
        L3 += 6; reasons.append(f"{vol_ratio:.1f}x Volume")
    elif vol_ratio >= 1.2:
        L3 += 2

    if hist["vol_accumulation"]:
        L3 += 5; reasons.append("Accumulation")
    L3 = _clamp(L3, 0, LAYER_BUDGET["volume"])
    if L3 > 0: layers_active += 1

    # ── LAYER 4: PRICE PATTERN (max 15) ───────────────────────────────────────
    L4 = 0
    if high1m > low1m > 0:
        range1m = high1m - low1m
        pos1m = (price - low1m) / range1m
        if 0.08 < pos1m < 0.30:
            L4 += 8; reasons.append("Early Cycle")

    if hist["support_level"] > 0 and price > 0:
        support_gap = (price / hist["support_level"] - 1) * 100
        if 0 < support_gap < 4:
            L4 += 4; reasons.append("Near Support")

    if bb_basis > 0:
        dist_basis = (price / bb_basis - 1) * 100
        if -1 < dist_basis < 2:
            L4 += 5; reasons.append("Mean Reversion")

    if hist["squeeze"]:
        L4 += 5; reasons.append("BB Squeeze")
    L4 = _clamp(L4, 0, LAYER_BUDGET["pattern"])
    if L4 > 0: layers_active += 1

    # ── LAYER 5: BREADTH & CONTEXT (max 10) ───────────────────────────────────
    L5 = 0
    if bullish:
        L5 += 7
    else:
        L5 -= 5; reasons.append("Bearish Breadth")
    L5 = _clamp(L5, -5, LAYER_BUDGET["breadth"])
    if L5 > 0: layers_active += 1

    # ── LAYER 6: VOLATILITY STATE (max 10) ────────────────────────────────────
    L6 = 0
    hvol = hist["volatility"]
    if 15 < hvol < 45:
        L6 += 7  # ideal swing range
    elif 45 <= hvol < 60:
        L6 += 2
    elif hvol >= 70:
        L6 -= 4
    elif hvol <= 8:
        L6 -= 2

    if hist["consec_up"] >= 2:
        L6 += 3
    elif hist["consec_down"] >= 4:
        L6 -= 3

    if hist["trend_pct_10"] > 2:
        L6 += 2
    L6 = _clamp(L6, -5, LAYER_BUDGET["volatility"])
    if L6 > 0: layers_active += 1

    # ── LAYER 7: HISTORICAL QUALITY (max 10) ──────────────────────────────────
    L7 = 0
    if hist["stability"] >= 7:
        L7 += 5; reasons.append("Stable")
    elif hist["stability"] >= 5:
        L7 += 2
    elif hist["stability"] < 3:
        L7 -= 3

    if hist["momentum"] > 1.0:
        L7 += 3; reasons.append("Hist Momentum")
    elif hist["momentum"] < -1.0:
        L7 -= 2

    if hist["trend_pct_30"] > 8:
        L7 += 3; reasons.append("30d Uptrend")
    L7 = _clamp(L7, -3, LAYER_BUDGET["historical"])
    if L7 > 0: layers_active += 1

    # ── FINAL SCORE ──────────────────────────────────────────────────────────
    score = max(0, L1 + L2 + L3 + L4 + L5 + L6 + L7)

    if layers_active < 4:
        score = min(score, THRESH_SWING - 1)

    # ── TARGET & STOP (ATR-based) ─────────────────────────────────────────────
    stop, eff_atr = _compute_atr_stop(price, atr, hist["atr_20"], 1.5)
    raw_target_pct = (3.0 * eff_atr / price) * 100
    target_pct = _clamp(raw_target_pct, 4.0, 10.0)
    target = round(price * (1 + target_pct / 100), 2)

    grade = _grade(score, layers_active)

    return score, reasons, target, stop, grade, layers_active


# ─────────────────────────────────────────────────────────────────────────────
# LONG-TERM INVESTMENT SCORER  (normalized 0–100)
# ─────────────────────────────────────────────────────────────────────────────
def score_longterm(
    price, rsi, macd, macd_sig, macd_hist,
    ema20, ema50, stoch_k, stoch_d,
    bb_low, bb_high, bb_basis,
    vol, avg_vol, chg1w, chg1m, low1m, high1m,
    sector, rsi_prev, hist
) -> Tuple[int, List[str], float, float, str, int]:

    reasons = []
    layers_active = 0
    quality = SECTORS.get(sector, {"quality": 5})["quality"]

    # ── PRE-FLIGHT GATES ──────────────────────────────────────────────────────
    if price < CFG["MIN_PRICE"]:       return 0, [], 0, 0, "D", 0
    if quality < 6:                   return 0, [], 0, 0, "D", 0
    if hist["stability"] < 2.0:       return 0, [], 0, 0, "D", 0

    # ── LAYER 1: SECTOR & FUNDAMENTAL QUALITY (max 20) ────────────────────────
    L1 = 0
    if quality == 9:
        L1 += 10
    elif quality == 8:
        L1 += 8
    elif quality == 7:
        L1 += 6
    else:
        L1 += 3

    # Trend infrastructure
    if price > ema50:
        L1 += 6; reasons.append("Above EMA50")
    elif price > ema50 * 0.97:
        L1 += 2
    if price > ema20:
        L1 += 4; reasons.append("Above EMA20")
    L1 = _clamp(L1, 0, LAYER_BUDGET["trend"])
    if L1 > 0: layers_active += 1

    # ── LAYER 2: VALUE ZONE / REVERSAL MOMENTUM (max 20) ─────────────────────
    L2 = 0
    if high1m > low1m > 0:
        dist_low = (price / low1m - 1) * 100
        if hist["triple_bottom"] and dist_low < 5:
            L2 += 14; reasons.append("Triple Bottom")
        elif 0.3 < dist_low < 4 and rsi > 28:
            L2 += 8; reasons.append("Near Monthly Low")
        elif dist_low < 10:
            L2 += 5; reasons.append("Value Zone")

    rsi_delta = rsi - rsi_prev if rsi_prev > 0 else 0
    if 20 < rsi < 35:
        L2 += 5; reasons.append("Oversold")
    elif 40 <= rsi < 50:
        L2 += 3; reasons.append(f"RSI {rsi:.0f} Reset")
    elif 50 <= rsi < 60:
        L2 += 1
    elif rsi > 75:
        L2 -= 8; reasons.append("Expensive")

    if rsi_delta > 3:
        L2 += 3; reasons.append("Momentum Turn")

    if chg1m < -25:
        L2 -= 5; reasons.append("Falling Knife")
    L2 = _clamp(L2, -10, LAYER_BUDGET["momentum"])
    if L2 > 0: layers_active += 1

    # ── LAYER 3: VOLUME PATTERN (max 15) ──────────────────────────────────────
    L3 = 0
    vol_ratio = vol / avg_vol if avg_vol > 0 else 0
    if hist["vol_accumulation"]:
        L3 += 8; reasons.append("Accumulation")
    if vol_ratio >= 1.5 and rsi < 50:
        L3 += 5; reasons.append("Vol + RSI Reset")
    elif vol_ratio >= 1.2:
        L3 += 2
    L3 = _clamp(L3, 0, LAYER_BUDGET["volume"])
    if L3 > 0: layers_active += 1

    # ── LAYER 4: PRICE PATTERN (max 15) ───────────────────────────────────────
    L4 = 0
    if bb_basis > 0:
        dist_basis = (price / bb_basis - 1) * 100
        if dist_basis < 0:
            L4 += 6; reasons.append("Below Mean")

    if hist["support_level"] > 0 and price > 0:
        gap = (price / hist["support_level"] - 1) * 100
        if 0 < gap < 3:
            L4 += 4; reasons.append("Near Support")

    bb_pos = _bb_position(price, bb_low, bb_high, bb_basis)
    if bb_pos < 0.15:
        L4 += 5; reasons.append("Near BB Low")
    elif bb_pos < 0.35:
        L4 += 2

    if hist["squeeze"]:
        L4 += 3; reasons.append("BB Squeeze")

    if hist["higher_lows"]:
        L4 += 4; reasons.append("Higher Lows")
    L4 = _clamp(L4, 0, LAYER_BUDGET["pattern"])
    if L4 > 0: layers_active += 1

    # ── LAYER 5: BREADTH & MACRO CONTEXT (max 10) ────────────────────────────
    L5 = 0
    if stoch_k > stoch_d and stoch_k < 50:
        L5 += 4; reasons.append("Stoch Recovery")
    if macd_hist > 0:
        L5 += 4; reasons.append("MACD Hist+")
    elif macd > macd_sig:
        L5 += 2
    if -3 < chg1w < 3:
        L5 += 2
    elif chg1w < -8:
        L5 -= 3
    L5 = _clamp(L5, -3, LAYER_BUDGET["breadth"])
    if L5 > 0: layers_active += 1

    # ── LAYER 6: VOLATILITY STATE (max 10) ────────────────────────────────────
    L6 = 0
    hvol = hist["volatility"]
    if hvol < 40:
        L6 += 6  # controlled
    elif 40 <= hvol < 60:
        L6 += 2
    elif hvol >= 60:
        L6 -= 3  # chaotic

    if hist["ema20_slope"] > 0.2:
        L6 += 4; reasons.append("EMA20 Rising")

    if chg1m > -5:
        L6 += 2
    elif chg1m < -15:
        L6 -= 2
    L6 = _clamp(L6, -3, LAYER_BUDGET["volatility"])
    if L6 > 0: layers_active += 1

    # ── LAYER 7: HISTORICAL QUALITY (max 10) ──────────────────────────────────
    L7 = 0
    if hist["stability"] >= 6:
        L7 += 5
    elif hist["stability"] >= 4:
        L7 += 2

    if hist["trend_pct_30"] > 5:
        L7 += 3; reasons.append("30d Trend")

    if hist["consec_up"] >= 2:
        L7 += 2
    elif hist["consec_down"] >= 3:
        L7 -= 2
    L7 = _clamp(L7, -2, LAYER_BUDGET["historical"])
    if L7 > 0: layers_active += 1

    # ── FINAL SCORE ──────────────────────────────────────────────────────────
    score = max(0, L1 + L2 + L3 + L4 + L5 + L6 + L7)

    if layers_active < 4:
        score = min(score, THRESH_LONG - 1)

    # ── TARGET & STOP ─────────────────────────────────────────────────────────
    if low1m > 0 and (price / low1m - 1) * 100 < 6:
        target = round(price * 1.55, 2)
    elif low1m > 0 and (price / low1m - 1) * 100 < 15:
        target = round(price * 1.30, 2)
    else:
        target = round(price * 1.20, 2)

    stop = max(round(price * 0.88, 2), round(hist["support_level"] * 0.97, 2)) if hist["support_level"] > 0 else round(price * 0.88, 2)

    grade = _grade(score, layers_active)

    return score, reasons, target, stop, grade, layers_active


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

def process_signals(raw: list, bullish: bool, mkt_chg: float):
    if not raw:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    intra, swing, long_, dips = [], [], [], []

    conn = sqlite3.connect(CFG["DB_PATH"])
    try:
        for item in raw:
            d = item.get("d", [])
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
            chg1m    = safe(d[26]) # This is correct
            low7d    = safe(d[28]) if len(d) > 28 else low_d
            rsi_prev = safe(d[27], rsi) if len(d) > 27 else rsi
            macd_h   = safe(d[29]) if len(d) > 29 else (macd - macd_sig)
            bb_basis = safe(d[31]) if len(d) > 31 else (bb_low + bb_high) / 2

            if price <= 0 or avg_vol <= 0:
                continue

            sector = SYM_SECTOR.get(sym, "Misc")
            hist = get_hist_metrics(sym, conn)

            # Standard Pivot Calculation
            h_piv, l_piv = (high_d if high_d > 0 else price), (low_d if low_d > 0 else price)
            p_piv = (h_piv + l_piv + price) / 3
            r1, s1 = round(2 * p_piv - l_piv, 2), round(2 * p_piv - h_piv, 2)
            r2, s2 = round(p_piv + (h_piv - l_piv), 2), round(p_piv - (h_piv - l_piv), 2)

            best_buy = s1 if price > s1 else s2
            best_sell = r1 if price < r1 else r2

            # Trend Bias
            is_buying = (price > vwap) and (macd > macd_sig) and (rsi > 50)
            trend_label = "BUYING" if is_buying else "SELLING"

            # Minimum liquidity
            if vol < max(CFG["MIN_VOLUME"], hist["avg_vol"] * 0.30):
                continue

            # ── INTRADAY ──────────────────────────────────────────────────────────
            sc, rs, tgt, stp, grd, la = score_intraday(
                price, change, rsi, macd, macd_sig, macd_h,
                ema5, ema10, vwap, adx, stoch_k, stoch_d,
                vol, avg_vol, atr, bb_low, bb_high, bb_basis,
                open_p, high_d, low_d,
                bullish, mkt_chg, hist, rsi_prev
            )
            if sc >= THRESH_INTRA and tgt > price > stp:
                intra.append({
                    "Symbol": sym, "Sector": sector,
                    "Price": round(price, 2), "Bias": trend_label,
                    "Chg%": round(change, 2),
                    "Score": sc, "Grade": grd,
                    "Layers": f"{la}/7",
                    "Buy": best_buy, "Sell": best_sell,
                    "R1": r1, "S1": s1,
                    "RV": round(rv, 1),
                    "RSI": round(rsi, 0),
                    "Signals": " | ".join(rs[:4]),
                    "Target": tgt, "Stop": stp,
                    "R:R": _rr(price, tgt, stp),
                })

            # ── SWING ─────────────────────────────────────────────────────────────
            sc, rs, tgt, stp, grd, la = score_swing(
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
                    "Price": round(price, 2), "Bias": trend_label,
                    "Chg%": round(change, 2),
                    "Score": sc, "Grade": grd,
                    "Layers": f"{la}/7",
                    "Buy": best_buy, "Sell": best_sell,
                    "R1": r1, "S1": s1,
                    "1W%": round(chg1w, 2),
                    "RSI": round(rsi, 0),
                    "Signals": " | ".join(rs[:3]),
                    "Target": tgt, "Stop": stp,
                    "R:R": _rr(price, tgt, stp),
                })

            # ── LONG-TERM ─────────────────────────────────────────────────────────
            perf1m = (price / low1m - 1) * 100 if low1m > 0 else 0.0
            sc, rs, tgt, stp, grd, la = score_longterm(
                price, rsi, macd, macd_sig, macd_h,
                ema20, ema50, stoch_k, stoch_d,
                bb_low, bb_high, bb_basis,
                vol, avg_vol, chg1w, chg1m, low1m, high1m,
                sector, rsi_prev, hist
            )
            if sc >= THRESH_LONG and tgt > price > stp:
                long_.append({
                    "Symbol": sym, "Sector": sector,
                    "Price": round(price, 2), "Bias": trend_label,
                    "1W%": round(chg1w, 2),
                    "Score": sc, "Grade": grd,
                    "Layers": f"{la}/7",
                    "Buy": best_buy, "Sell": best_sell,
                    "R1": r1, "S1": s1,
                    "1M%": round(perf1m, 2),
                    "Stab": round(hist["stability"], 1),
                    "RSI": round(rsi, 0),
                    "Signals": " | ".join(rs[:3]),
                    "Target": tgt, "Stop": stp,
                    "R:R": _rr(price, tgt, stp),
                })

            # ── DIP SCANNER ───────────────────────────────────────────────────────
            is_uptrend = price > ema50 if ema50 > 0 else (price > ema20 if ema20 > 0 else False)

            # Improved dip condition: includes check against 7-day low
            near_7d_low = (low7d > 0 and (price / low7d - 1) * 100 < 5)
            is_dipping = (rsi <= 45) or (stoch_k <= 30) or (price <= bb_low * 1.02) or near_7d_low

            if is_uptrend and is_dipping:
                dip_stop, dip_eff_atr = _compute_atr_stop(price, atr, hist.get("atr_20", 0) if hist else 0, 1.0)
                dip_target = round(price * (1 + _clamp((2.0 * dip_eff_atr / price) * 100, 3.0, 8.0) / 100), 2)

                if dip_target > price > dip_stop:
                    rsi_pts = max(0, min(40, (50 - rsi) * 2))
                    bb_range = (bb_high - bb_low) if bb_high > bb_low else price * 0.1
                    bb_pts = max(0, min(30, 30 * (1 - (price - bb_low) / bb_range)))
                    vol_pts = min(15, 15 * (vol / avg_vol)) if avg_vol > 0 else 0
                    stab_pts = min(15, hist.get("stability", 5) * 1.5) if hist else 7.5

                    dip_score = round(rsi_pts + bb_pts + vol_pts + stab_pts)

                    dip_reasons = []
                    if rsi <= 35: dip_reasons.append("Oversold RSI")
                    elif rsi <= 45: dip_reasons.append("RSI Pullback")
                    if stoch_k <= 25: dip_reasons.append("Stoch Oversold")
                    if price <= bb_low * 1.015: dip_reasons.append("BB Support")
                    if near_7d_low: dip_reasons.append("Near 7D Low")
                    if low1m > 0 and price <= low1m * 1.03: dip_reasons.append("Near 1M Low")

                    dips.append({
                        "Symbol": sym, "Sector": sector,
                        "Price": round(price, 2), "Bias": trend_label,
                        "Chg%": round(change, 2), "1W%": round(chg1w, 2),
                        "Score": dip_score,
                        "Target": dip_target, "Stop": dip_stop,
                        "R:R": _rr(price, dip_target, dip_stop),
                        "Signals": " | ".join(dip_reasons[:3]) if dip_reasons else "Pullback",
                        "Buy": best_buy, "Sell": best_sell,
                        "RSI": round(rsi, 0),
                        "R1": r1, "S1": s1
                    })
    finally:
        conn.close()
    srt = lambda lst: pd.DataFrame(lst).sort_values("Score", ascending=False).reset_index(drop=True) if lst else pd.DataFrame()
    return srt(intra), srt(swing), srt(long_), srt(dips)


# ══════════════════════════════════════════════════════════════════════════════
# UI — DARK REPORT STYLE
# ══════════════════════════════════════════════════════════════════════════════



# ─── CSS: Dark Report ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

/* ── Base ─────────────────────────────────────────────────────────────────── */
* { box-sizing: border-box; }
html, body, [class*="css"] {
    background: #0a0e1a !important;
    color: #e0e4ec !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}
footer, #MainMenu { visibility: hidden; }

.main .block-container {
    max-width: 960px !important;
    padding: 2rem 2.5rem !important;
    margin: 0 auto !important;
}

/* ── Report Masthead ──────────────────────────────────────────────────────── */
.report-masthead {
    padding-bottom: 4px;
    margin-bottom: 0px;
}
.report-title {
    font-family: 'Libre Baskerville', Georgia, serif;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #e0e4ec;
    margin: 0;
    line-height: 1.2;
}
.report-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #6b7280;
    margin-top: 4px;
    letter-spacing: 0.05em;
}

/* ── Market Summary Bar ───────────────────────────────────────────────────── */
.market-bar {
    display: flex;
    justify-content: space-between;
    border: 1px solid #1e2536;
    border-radius: 0;
    padding: 10px 16px;
    margin-bottom: 16px;
    background: #0f1322;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}
.market-item { text-align: center; }
.market-label {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7280;
    margin-bottom: 2px;
}
.market-value {
    font-size: 0.92rem;
    font-weight: 700;
    color: #e0e4ec;
}

/* ── Sector Table ─────────────────────────────────────────────────────────── */
.sector-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0;
    border: 1px solid #1e2536;
    margin-bottom: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
}
.sector-cell {
    flex: 1 1 auto;
    min-width: 90px;
    padding: 4px 10px;
    border-right: 1px solid #1e2536;
    border-bottom: 1px solid #1e2536;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #0a0e1a;
}
.sector-cell:hover { background: #131828; }
.sector-name { color: #6b7280; font-size: 0.68rem; }
.sector-val { font-weight: 700; }
.s-up { color: #34d399; }
.s-dn { color: #f87171; }
.s-nt { color: #fbbf24; }

/* ── Section Headers ──────────────────────────────────────────────────────── */
.section-header {
    font-family: 'Libre Baskerville', Georgia, serif;
    font-size: 1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 2px solid #3a4050;
    padding-bottom: 4px;
    margin-top: 24px;
    margin-bottom: 4px;
    color: #e0e4ec;
}
.section-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #6b7280;
    margin-bottom: 10px;
    letter-spacing: 0.03em;
}

/* ── Alert Boxes ──────────────────────────────────────────────────────────── */
.paper-alert {
    border: 1px solid rgba(248,113,113,0.3);
    border-left: 4px solid #f87171;
    padding: 8px 14px;
    margin-bottom: 12px;
    font-size: 0.78rem;
    color: #fca5a5;
    background: rgba(239,68,68,0.08);
}
.paper-info {
    border: 1px solid #1e2536;
    border-left: 4px solid #3a4050;
    padding: 8px 14px;
    margin-bottom: 12px;
    font-size: 0.78rem;
    color: #9ca3af;
    background: rgba(15,19,34,0.6);
}

/* ── Data Tables ──────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    background: transparent !important;
    border-radius: 0 !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] > div {
    border: 1px solid #1e2536 !important;
}
thead tr th {
    background: #0f1322 !important;
    color: #6b7280 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.64rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase;
    padding: 8px 6px !important;
    border-bottom: 2px solid #2a3040 !important;
}
tbody tr { border-bottom: 1px solid #1a1f2e !important; }
tbody tr:nth-child(even) { background: rgba(15,19,34,0.4) !important; }
tbody tr:hover { background: rgba(30,37,54,0.5) !important; }
tbody td {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #e0e4ec !important;
    padding: 6px !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton button {
    background: #131828 !important;
    border: 1px solid #2a3040 !important;
    border-radius: 0 !important;
    color: #e0e4ec !important;
    padding: 4px 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    min-height: unset !important;
    line-height: 1.4 !important;
}
.stButton button:hover {
    background: #1e2536 !important;
    border-color: #3a4050 !important;
}

/* ── Footer ───────────────────────────────────────────────────────────────── */
.report-footer {
    border-top: 1px solid #1e2536;
    margin-top: 32px;
    padding-top: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #4b5563;
    line-height: 1.6;
}

/* ── Mobile ───────────────────────────────────────────────────────────────── */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem !important;
    }
    .report-title { font-size: 1.1rem; }
    .market-bar { flex-wrap: wrap; gap: 8px; }
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(1) {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    [data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2),
    [data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) {
        min-width: calc(50% - 8px) !important;
        flex: 1 1 calc(50% - 8px) !important;
    }
}

/* ── Misc overrides ───────────────────────────────────────────────────────── */
hr { border-color: #1e2536 !important; margin: 1.2rem 0 !important; }
.stTextInput>div>div>input, .stNumberInput>div>div>input {
    background: #0f1322 !important;
    border: 1px solid #2a3040 !important;
    border-radius: 0 !important;
    color: #e0e4ec !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}
div[data-testid="stExpander"] {
    background: #0f1322 !important;
    border: 1px solid #1e2536 !important;
    border-radius: 0 !important;
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

now = pkt_now()
vol_cr = idx_vol / 1e7
state_text = "LIVE" if is_open else "CLOSED"
breadth_text = "BULLISH" if bullish else ("BEARISH" if avg_chg < -0.5 else "NEUTRAL")

# ─── Report Masthead & Action Buttons ────────────────────────────────────────
head_cols = st.columns([8, 1, 1])

with head_cols[0]:
    st.markdown(f"""
    <div class="report-masthead">
        <div class="report-title">PSX Market Intelligence</div>
        <div class="report-subtitle">KSE-100 Index &middot; 7-Layer Signal Engine &middot; {now.strftime("%A, %B %d, %Y")} &middot; {now.strftime("%H:%M")} PKT</div>
    </div>
    """, unsafe_allow_html=True)

with head_cols[1]:
    st.markdown('<div style="padding-top: 8px;"></div>', unsafe_allow_html=True)
    scan_btn = st.button("SCAN", help="Run scanner now", use_container_width=True)

with head_cols[2]:
    st.markdown('<div style="padding-top: 8px;"></div>', unsafe_allow_html=True)
    sync_btn = st.button("SYNC", help="Sync historical data", use_container_width=True)

# Double border separating the header from the rest of the report
st.markdown('<div style="border-bottom: 3px double #3a4050; margin-bottom: 20px; margin-top: 8px;"></div>', unsafe_allow_html=True)

if sync_btn:
    sync_historical_data(KSE100)
    st.rerun()

# ─── Market Summary ──────────────────────────────────────────────────────────
pct_color = "#34d399" if idx_pct >= 0 else "#f87171"
st.markdown(f"""
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
        <div class="market-value" style="color:{'#34d399' if bullish else '#f87171'}">{breadth_text}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Breadth Warning ─────────────────────────────────────────────────────────
if not bullish:
    st.markdown(f'<div class="paper-alert"><strong>BEARISH BREADTH</strong> — Market declining ({adv} advancers vs {dec} decliners). Reduce position sizes. Intraday setups require extra confirmation.</div>', unsafe_allow_html=True)

scan = is_open or scan_btn

if scan:
    with st.spinner("Scanning KSE-100..."):
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
        cls = "s-up" if avg >= 0.5 else ("s-dn" if avg < -0.5 else "s-nt")
        arrow = "+" if avg >= 0 else ""
        tiles += f'<div class="sector-cell"><span class="sector-name">{html.escape(sec)}</span><span class="sector-val {cls}">{arrow}{avg:.1f}%</span></div>'

    st.markdown(f'<div class="sector-row">{tiles}</div>', unsafe_allow_html=True)

    # ── Run Signals ───────────────────────────────────────────────────────────
    df_i, df_s, df_l, df_d = process_signals(raw, bullish, avg_chg)
    if not df_i.empty: df_i = df_i[df_i["Score"] >= THRESH_INTRA].reset_index(drop=True)
    if not df_s.empty: df_s = df_s[df_s["Score"] >= THRESH_SWING].reset_index(drop=True)
    if not df_l.empty: df_l = df_l[df_l["Score"] >= THRESH_LONG].reset_index(drop=True)
    all_stocks_df = pd.concat([df_i, df_s, df_l, df_d]).drop_duplicates(subset=['Symbol']) if not (df_i.empty and df_s.empty and df_l.empty and df_d.empty) else pd.DataFrame()
    if not df_d.empty: df_d = df_d.reset_index(drop=True)

    def render_table(df: pd.DataFrame, col_order: list, col_cfg: dict):
        """Render a clean paper-style table with ordered columns."""
        if df.empty:
            st.markdown('<div class="paper-info">No setups meet current threshold.</div>', unsafe_allow_html=True)
        else:
            available_cols = [c for c in col_order if c in df.columns]
            df_top = df.head(7)[available_cols].copy()
            st.dataframe(
                df_top,
                column_config=col_cfg,
                hide_index=True,
                use_container_width=True
            )

    # ── INTRADAY ──────────────────────────────────────────────────────────────
    n_i = len(df_i)
    st.markdown(f'<div class="section-header">Intraday Scalps</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-meta">{n_i} setups found &middot; Threshold: {THRESH_INTRA}/100 &middot; Breadth: {breadth_text} &middot; Min. 4/7 layer confluence</div>', unsafe_allow_html=True)

    render_table(df_i,
        ["Symbol", "Price", "Chg%", "Score", "Bias", "R:R", "Target", "Stop", "Signals", "Buy", "Sell", "RV", "RSI", "R1", "S1"],
        {
            "Symbol":  st.column_config.TextColumn("Symbol"),
            "Price":   st.column_config.NumberColumn("Price", format="%.2f"),
            "Chg%":    st.column_config.NumberColumn("Chg%", format="%.2f%%"),
            "Score":   st.column_config.NumberColumn("Score", format="%d", help="Normalized 0-100 score"),
            "Bias":    st.column_config.TextColumn("Bias"),
            "R:R":     st.column_config.NumberColumn("R:R", format="%.2f"),
            "Target":  st.column_config.NumberColumn("Target", format="%.2f"),
            "Stop":    st.column_config.NumberColumn("Stop", format="%.2f"),
            "Signals": st.column_config.TextColumn("Signals"),
            "Buy":     st.column_config.NumberColumn("Buy", format="%.2f"),
            "Sell":    st.column_config.NumberColumn("Sell", format="%.2f"),
            "RV":      st.column_config.NumberColumn("RV", format="%.1fx"),
            "RSI":     st.column_config.NumberColumn("RSI", format="%d"),
            "R1":      st.column_config.NumberColumn("R1", format="%.2f"),
            "S1":      st.column_config.NumberColumn("S1", format="%.2f"),
        }
    )

    # ── SWING ─────────────────────────────────────────────────────────────────
    n_s = len(df_s)
    st.markdown(f'<div class="section-header">Swing Trades</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-meta">{n_s} setups found &middot; Threshold: {THRESH_SWING}/100 &middot; Breadth: {breadth_text} &middot; Min. 4/7 layer confluence</div>', unsafe_allow_html=True)

    render_table(df_s,
        ["Symbol", "Price", "Chg%", "1W%", "Score", "Bias", "R:R", "Target", "Stop", "Signals", "Buy", "Sell", "RSI", "R1", "S1"],
        {
            "Symbol":  st.column_config.TextColumn("Symbol"),
            "Price":   st.column_config.NumberColumn("Price", format="%.2f"),
            "Chg%":    st.column_config.NumberColumn("Chg%", format="%.2f%%"),
            "1W%":     st.column_config.NumberColumn("1W%", format="%.2f%%"),
            "Score":   st.column_config.NumberColumn("Score", format="%d"),
            "Bias":    st.column_config.TextColumn("Bias"),
            "R:R":     st.column_config.NumberColumn("R:R", format="%.2f"),
            "Target":  st.column_config.NumberColumn("Target", format="%.2f"),
            "Stop":    st.column_config.NumberColumn("Stop", format="%.2f"),
            "Signals": st.column_config.TextColumn("Signals"),
            "Buy":     st.column_config.NumberColumn("Buy", format="%.2f"),
            "Sell":    st.column_config.NumberColumn("Sell", format="%.2f"),
            "RSI":     st.column_config.NumberColumn("RSI", format="%d"),
            "R1":      st.column_config.NumberColumn("R1", format="%.2f"),
            "S1":      st.column_config.NumberColumn("S1", format="%.2f"),
        }
    )

    # ── LONG-TERM ─────────────────────────────────────────────────────────────
    n_l = len(df_l)
    st.markdown(f'<div class="section-header">Long-Term Investments</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-meta">{n_l} setups found &middot; Threshold: {THRESH_LONG}/100 &middot; Min. 4/7 layer confluence</div>', unsafe_allow_html=True)

    render_table(df_l,
        ["Symbol", "Price", "1M%", "1W%", "Score", "Bias", "R:R", "Target", "Stop", "Signals", "Buy", "Sell", "RSI", "R1", "S1"],
        {
            "Symbol":  st.column_config.TextColumn("Symbol"),
            "Price":   st.column_config.NumberColumn("Price", format="%.2f"),
            "1M%":     st.column_config.NumberColumn("1M%", format="%.2f%%"),
            "1W%":     st.column_config.NumberColumn("1W%", format="%.2f%%"),
            "Score":   st.column_config.NumberColumn("Score", format="%d"),
            "Bias":    st.column_config.TextColumn("Bias"),
            "R:R":     st.column_config.NumberColumn("R:R", format="%.2f"),
            "Target":  st.column_config.NumberColumn("Target", format="%.2f"),
            "Stop":    st.column_config.NumberColumn("Stop", format="%.2f"),
            "Signals": st.column_config.TextColumn("Signals"),
            "Buy":     st.column_config.NumberColumn("Buy", format="%.2f"),
            "Sell":    st.column_config.NumberColumn("Sell", format="%.2f"),
            "RSI":     st.column_config.NumberColumn("RSI", format="%d"),
            "R1":      st.column_config.NumberColumn("R1", format="%.2f"),
            "S1":      st.column_config.NumberColumn("S1", format="%.2f"),
        }
    )

    # ── STOCKS ON DIP ─────────────────────────────────────────────────────────
    n_d = len(df_d)
    st.markdown(f'<div class="section-header">Stocks on Dip</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-meta">{n_d} setups found &middot; Filter: Price > EMA50 (Uptrend) &middot; Pullback</div>', unsafe_allow_html=True)

    render_table(df_d,
        ["Symbol", "Price", "Chg%", "1W%", "Score", "Bias", "R:R", "Target", "Stop", "Signals", "Buy", "Sell", "RSI", "R1", "S1"],
        {
            "Symbol":  st.column_config.TextColumn("Symbol"),
            "Price":   st.column_config.NumberColumn("Price", format="%.2f"),
            "Chg%":    st.column_config.NumberColumn("Chg%", format="%.2f%%"),
            "1W%":     st.column_config.NumberColumn("1W%", format="%.2f%%"),
            "Score":   st.column_config.NumberColumn("Score", format="%d", help="Dip Score (higher = better quality dip)"),
            "Bias":    st.column_config.TextColumn("Bias"),
            "R:R":     st.column_config.NumberColumn("R:R", format="%.2f"),
            "Target":  st.column_config.NumberColumn("Target", format="%.2f"),
            "Stop":    st.column_config.NumberColumn("Stop", format="%.2f"),
            "Signals": st.column_config.TextColumn("Signals"),
            "Buy":     st.column_config.NumberColumn("Buy", format="%.2f"),
            "Sell":    st.column_config.NumberColumn("Sell", format="%.2f"),
            "RSI":     st.column_config.NumberColumn("RSI", format="%d"),
            "R1":      st.column_config.NumberColumn("R1", format="%.2f"),
            "S1":      st.column_config.NumberColumn("S1", format="%.2f"),
        }
    )

    # ── TREND REVERSALS ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Trend Reversals</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-meta">Changes detected since yesterday</div>', unsafe_allow_html=True)

    # Get yesterday's data for comparison
    yesterday_snapshot = get_yesterday_snapshot()

    reversals = []
    if not all_stocks_df.empty:
        for _, row in all_stocks_df.iterrows():
            symbol = row['Symbol']
            current_trend = row['Bias']
            current_rv = row.get('RV', 'N/A')

            yesterday_data = yesterday_snapshot.get(symbol)
            if yesterday_data:
                prev_trend = yesterday_data['trend']
                prev_rv = yesterday_data.get('rv', 'N/A')
                if prev_trend != current_trend:
                    reversals.append({
                        "Symbol": symbol,
                        "Yesterday Trend": prev_trend,
                        "Today Trend": current_trend,
                        "Yesterday RV": round(prev_rv, 1) if isinstance(prev_rv, float) else 'N/A',
                        "Today RV": current_rv,
                    })
        # Save today's data for the next day's comparison
        save_daily_snapshot(all_stocks_df)

    if reversals:
        reversals_df = pd.DataFrame(reversals)
        st.dataframe(reversals_df, hide_index=True, use_container_width=True)
    else:
        st.markdown('<div class="paper-info">No trend reversals detected in this scan.</div>', unsafe_allow_html=True)

# ── Report Footer ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="report-footer">
    Generated {now.strftime("%Y-%m-%d %H:%M:%S")} PKT &middot; Source: TradingView Scanner API, Sarmaaya &middot; KSE-100 Universe ({len(KSE100)} symbols)<br>
    Scores normalized 0-100 with 7-layer budgets. Grade: A (>=75, 6+ layers) / B (>=60, 5+) / C (>=45, 4+) / D (below).<br>
    This is a quantitative screening tool, not investment advice. All signals require independent verification before execution.
</div>
""", unsafe_allow_html=True)

# ── Auto-refresh (non-blocking) ──────────────────────────────────────────────
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
