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
    page_title="PSX Elite Scanner", page_icon="⚡",
    layout="wide", initial_sidebar_state="expanded"
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

def sync_historical_data():
    """Pull 90 days of full OHLCV from Yahoo Finance for pattern & indicator calculation."""
    end   = datetime.now()
    start = end - timedelta(days=CFG["HIST_DAYS"])
    conn  = sqlite3.connect(CFG["DB_PATH"])

    ph   = st.sidebar.empty()
    prog = st.sidebar.progress(0)
    errors = []

    for i, sym in enumerate(KSE100):
        ph.caption(f"🔄 {sym} ({i+1}/{len(KSE100)})")
        try:
            df = yf.download(f"{sym}.KA", start=start, end=end, progress=False, multi_level_index=False)
            if not df.empty:
                df = df.reset_index()
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
                rows = []
                for _, r in df.iterrows():
                    rows.append((
                        r["date"], sym,
                        safe(r.get("Open")), safe(r.get("High")),
                        safe(r.get("Low")),  safe(r.get("Close")),
                        int(safe(r.get("Volume")))
                    ))
                conn.executemany(
                    "INSERT OR REPLACE INTO price_history VALUES (?,?,?,?,?,?,?)", rows
                )
        except Exception as e:
            errors.append(sym)

    cutoff = (datetime.now() - timedelta(days=85)).strftime("%Y-%m-%d")
    conn.execute("DELETE FROM price_history WHERE date < ?", (cutoff,))
    conn.commit()
    conn.close()

    msg = f"✅ Sync done" + (f" ({len(errors)} failed: {','.join(errors[:5])})" if errors else "")
    ph.success(msg)
    time.sleep(2)
    ph.empty()
    prog.empty()

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

def _ema_series(prices: list, period: int) -> list:
    """Compute EMA for a list of prices (oldest→newest order)."""
    if len(prices) < period:
        return []
    k = 2.0 / (period + 1)
    ema = [sum(prices[:period]) / period]
    for p in prices[period:]:
        ema.append(p * k + ema[-1] * (1 - k))
    return ema

def _rsi_series(closes: list, period: int = 14) -> list:
    """Calculate RSI series."""
    if len(closes) < period + 1:
        return []
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    rsi_vals = []
    for i in range(period, len(gains)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rs = avg_g / avg_l if avg_l > 0 else 100
        rsi_vals.append(100 - 100 / (1 + rs))
    return rsi_vals

def get_hist_metrics(symbol: str) -> Dict:
    """
    Full historical analytics — the backbone of swing and long-term scoring.
    Returns 15+ computed metrics from stored OHLCV data.
    """
    conn = sqlite3.connect(CFG["DB_PATH"])
    rows = conn.execute(
        "SELECT date, open, high, low, close, volume FROM price_history "
        "WHERE symbol=? ORDER BY date ASC LIMIT 90",
        (symbol,)
    ).fetchall()
    conn.close()

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

    if len(rows) < 10:
        return empty

    dates   = [r[0] for r in rows]
    opens   = [r[1] for r in rows]
    highs   = [r[2] for r in rows]
    lows    = [r[3] for r in rows]
    closes  = [r[4] for r in rows]
    volumes = [r[5] for r in rows]

    n = len(closes)

    # ── 1. Trend metrics ──────────────────────────────────────────────────────
    trend_pct_30 = (closes[-1] / closes[max(0, n-30)] - 1) * 100 if n >= 30 else 0.0
    trend_pct_10 = (closes[-1] / closes[max(0, n-10)] - 1) * 100 if n >= 10 else 0.0

    # ── 2. Stability (% of up-days, last 20 sessions) ─────────────────────────
    recent_n = min(20, n - 1)
    up_days  = sum(1 for i in range(n - recent_n, n) if closes[i] >= closes[i-1])
    stability = up_days / recent_n * 10

    # ── 3. Volume analytics ───────────────────────────────────────────────────
    avg_vol     = sum(volumes[-30:]) / min(30, n)
    recent_vol  = sum(volumes[-5:]) / 5 if n >= 5 else avg_vol
    older_vol   = sum(volumes[-20:-5]) / 15 if n >= 20 else avg_vol
    vol_trend   = (recent_vol / older_vol - 1) * 100 if older_vol > 0 else 0.0  # rising volume?

    # ── 4. True-Range ATR (20-period) ─────────────────────────────────────────
    trs = []
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    atr_20 = sum(trs[-20:]) / min(20, len(trs)) if trs else 0.0

    # ── 5. Historical volatility (std dev of daily returns, annualised %) ─────
    if n >= 10:
        returns = [(closes[i] / closes[i-1] - 1) for i in range(1, n)]
        mean_r  = sum(returns) / len(returns)
        var     = sum((r - mean_r)**2 for r in returns) / len(returns)
        volatility = math.sqrt(var) * math.sqrt(252) * 100
    else:
        volatility = 0.0

    # ── 6. Momentum (EMA5 vs EMA20 on close) ─────────────────────────────────
    ema5_s  = _ema_series(closes, 5)
    ema10_s = _ema_series(closes, 10)
    ema20_s = _ema_series(closes, 20)

    momentum = 0.0
    ema10_slope = 0.0
    ema20_slope = 0.0

    if len(ema5_s) >= 2 and len(ema20_s) >= 2:
        momentum = (ema5_s[-1] / ema20_s[-1] - 1) * 100
    if len(ema10_s) >= 5:
        ema10_slope = (ema10_s[-1] / ema10_s[-5] - 1) * 100
    if len(ema20_s) >= 5:
        ema20_slope = (ema20_s[-1] / ema20_s[-5] - 1) * 100

    # ── 7. Historical RSI (14-period) & its slope ─────────────────────────────
    rsi_s = _rsi_series(closes, 14)
    rsi_hist  = rsi_s[-1]  if rsi_s else 50.0
    rsi_slope = (rsi_s[-1] - rsi_s[-4]) if len(rsi_s) >= 4 else 0.0

    # ── 8. Support / resistance (swing lows & highs last 30 bars) ─────────────
    lookback = min(30, n)
    swing_lows  = [lows[i]  for i in range(n - lookback, n) if i > 0 and lows[i]  < lows[i-1]  and (i+1 >= n or lows[i]  < lows[i+1])]
    swing_highs = [highs[i] for i in range(n - lookback, n) if i > 0 and highs[i] > highs[i-1] and (i+1 >= n or highs[i] > highs[i+1])]
    support_level    = max(swing_lows)  if swing_lows  else min(lows[-lookback:])
    resistance_level = min(swing_highs) if swing_highs else max(highs[-lookback:])

    # ── 9. Consecutive direction streak ───────────────────────────────────────
    consec_up = consec_down = 0
    for i in range(n-1, 0, -1):
        if closes[i] >= closes[i-1]:
            if consec_down > 0: break
            consec_up += 1
        else:
            if consec_up > 0: break
            consec_down += 1

    # ── 10. Higher-lows pattern (last 5 swing lows rising) ───────────────────
    if len(swing_lows) >= 3:
        sl3 = swing_lows[-3:]
        higher_lows = sl3[0] < sl3[1] < sl3[2]
    else:
        higher_lows = False

    # ── 11. Volume accumulation (rising vol on up-days) ──────────────────────
    if n >= 10:
        up_vol   = sum(volumes[i] for i in range(n-10, n) if closes[i] >= closes[i-1])
        down_vol = sum(volumes[i] for i in range(n-10, n) if closes[i] <  closes[i-1])
        vol_accumulation = up_vol > down_vol * 1.3
    else:
        vol_accumulation = False

    # ── 12. Bollinger Band squeeze (volatility contraction) ───────────────────
    if len(closes) >= 20:
        bb_window = closes[-20:]
        bb_mean   = sum(bb_window) / 20
        bb_std    = math.sqrt(sum((x - bb_mean)**2 for x in bb_window) / 20)
        bb_width  = (2 * bb_std / bb_mean) * 100 if bb_mean > 0 else 10
        squeeze   = bb_width < 4.0  # tight Bollinger = compression before expansion
    else:
        squeeze = False

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

@st.cache_data(ttl=25, show_spinner=False)
def fetch_breadth() -> Tuple[float, bool, int, int, dict]:
    syms = list(set(KSE100 + ["KSE100","PKS100"]))
    payload = {
        "filter":  [{"left": "name", "operation": "in_range", "right": syms}],
        "markets": ["pakistan"],
        "columns": ["name","change","close","high","low","volume","RSI"],
        "range":   [0, 500],
    }
    kse = {"close": 0.0, "change": 0.0, "high": 0.0, "low": 0.0, "volume": 0}
    try:
        r    = _session().post(TV_URL, json=payload, timeout=10)
        data = r.json().get("data", [])
        chgs, rsis = [], []
        for item in data:
            ticker = item.get("s", "").upper()
            d = item.get("d", [])
            if len(d) < 6: continue
            name = safe(d[0], "")
            chg  = safe(d[1])
            cls  = safe(d[2])
            if "KSE100" in ticker or "PKS100" in ticker or name == "KSE100":
                kse = {"close": cls, "change": chg,
                       "high": safe(d[3]), "low": safe(d[4]), "volume": safe(d[5])}
            if name in KSE100 or ticker.split(":")[-1] in KSE100:
                chgs.append(chg)
                if len(d) > 6: rsis.append(safe(d[6]))

        if not chgs:
            return 0.0, False, 0, 0, kse

        avg   = sum(chgs) / len(chgs)
        adv   = sum(1 for c in chgs if c > 0)
        dec   = sum(1 for c in chgs if c < 0)
        bull  = avg > CFG["BREADTH_MIN"] and adv > dec
        return avg, bull, adv, dec, kse
    except Exception:
        return 0.0, False, 0, 0, kse

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
) -> Tuple[int, List[str], float, float]:

    score   = 0
    reasons = []

    # ── PRE-FLIGHT GATES (disqualify immediately) ─────────────────────────────
    vol_ratio = vol / avg_vol if avg_vol > 0 else 0
    if price < CFG["MIN_PRICE"]:                return 0, [], 0, 0
    if vol < CFG["MIN_VOLUME"]:                 return 0, [], 0, 0
    if vol_ratio < 1.0:                         return 0, [], 0, 0   # below-avg volume = no institutional interest
    if price <= vwap:                           return 0, [], 0, 0   # must be above VWAP
    if rsi > 82:                                return 0, [], 0, 0   # severely overbought — skip

    # ── LAYER 1: TREND STRUCTURE ──────────────────────────────────────────────
    # EMA alignment (price > EMA5 > EMA10 = maximum intraday alignment)
    if ema5 > 0 and ema10 > 0:
        if price > ema5 > ema10:
            score += 4; reasons.append("EMA5>10 Fan")
        elif price > ema5 and price > ema10:
            score += 2; reasons.append("Above EMAs")
        elif price > ema10:
            score += 1

    # ADX: trend strength (only trade strong trends intraday)
    if adx >= 35:
        score += 4; reasons.append(f"ADX {adx:.0f} Strong")
    elif adx >= 25:
        score += 2; reasons.append(f"ADX {adx:.0f}")
    elif adx < 18:
        score -= 2  # choppy market — scalps fail

    # Candle structure: price position within today's range
    day_range = high_d - low_d
    if day_range > 0:
        candle_pos = (price - low_d) / day_range
        if candle_pos > 0.7:
            score += 2; reasons.append("Day High Zone")
        elif candle_pos < 0.3:
            score -= 1  # buying near day low = risky

    # Gap-up continuation
    if open_p > 0 and price > open_p and change > 1.0:
        score += 2; reasons.append("Gap Bull")

    # ── LAYER 2: MOMENTUM CASCADE ─────────────────────────────────────────────
    # MACD: histogram expanding = momentum accelerating
    if macd > macd_sig:
        if macd_hist > 0 and macd > 0:
            score += 3; reasons.append("MACD↑ Bull")
        else:
            score += 1
    elif macd < macd_sig:
        score -= 1

    # RSI: rising momentum in healthy zone
    rsi_delta = rsi - rsi_prev if rsi_prev > 0 else 0
    if 52 < rsi < 72:
        score += 2; reasons.append(f"RSI {rsi:.0f}")
        if rsi_delta > 3:
            score += 1; reasons.append("RSI Rising")
    elif rsi < 50:
        score -= 1
    elif rsi >= 72:
        score -= 2

    # Stochastic: confirm momentum but avoid crossover peaks
    if stoch_k > stoch_d and 40 < stoch_k < 80:
        score += 2; reasons.append("Stoch Bull")
    elif stoch_k > 85:
        score -= 1  # overbought Stoch

    # Relative strength vs market
    rs_alpha = change - mkt_chg
    if rs_alpha > 1.5:
        score += 3; reasons.append(f"RS +{rs_alpha:.1f}%")
    elif rs_alpha > 0.5:
        score += 1

    # ── LAYER 3: VOLUME FOOTPRINT ─────────────────────────────────────────────
    if vol_ratio >= CFG["INST_VOL_X"]:
        score += 5; reasons.append(f"🐋 {vol_ratio:.1f}x Inst.Vol")
    elif vol_ratio >= 2.0:
        score += 3; reasons.append(f"{vol_ratio:.1f}x Vol")
    elif vol_ratio >= 1.5:
        score += 1

    # Volume accumulation from history
    if hist["vol_accumulation"]:
        score += 1; reasons.append("Vol Accumulation")

    # ── LAYER 4: PRICE PATTERN (VWAP ZONE) ───────────────────────────────────
    vwap_dist = (price - vwap) / vwap * 100 if vwap > 0 else 99
    if 0 < vwap_dist < 0.5:
        score += 4; reasons.append("🎯 VWAP Bounce")
    elif 0.5 <= vwap_dist < 1.2:
        score += 2; reasons.append("VWAP Edge")
    elif vwap_dist > 3.0:
        score -= 2  # too extended from VWAP — chasing

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
    stop, e_atr = _compute_atr_stop(price, atr, hist["atr_20"], 0.6)
    target = round(price + 1.0 * e_atr, 2)

    return score, reasons, target, stop


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
) -> Tuple[int, List[str], float, float]:

    score   = 0
    reasons = []

    # ── PRE-FLIGHT GATES ──────────────────────────────────────────────────────
    if price < CFG["MIN_PRICE"]:                return 0, [], 0, 0
    if vol < CFG["MIN_VOLUME"] * 0.8:          return 0, [], 0, 0
    if price <= ema50:                          return 0, [], 0, 0  # must be in long-term uptrend
    if rsi > 78:                                return 0, [], 0, 0  # overbought — bad swing entry
    if adx < 15:                                return 0, [], 0, 0  # no trend = no swing

    # ── LAYER 1: TREND STRUCTURE ──────────────────────────────────────────────
    # Perfect EMA fan: price > EMA5 > EMA10 > EMA20 > EMA50
    if ema5 > 0 and price > ema5 > ema10 > ema20 > ema50:
        score += 7; reasons.append("⭐ Perfect EMA Fan")
    elif price > ema10 > ema20 > ema50:
        score += 5; reasons.append("EMA Fan")
    elif price > ema20 > ema50:
        score += 3; reasons.append("EMA Rising")
    elif price > ema50:
        score += 1; reasons.append("Above EMA50")

    # EMA slopes (are they rising?)
    if hist["ema10_slope"] > 0.3:
        score += 2; reasons.append("EMA10 Rising")
    elif hist["ema10_slope"] < -0.5:
        score -= 2

    # Higher lows pattern (structural uptrend)
    if hist["higher_lows"]:
        score += 3; reasons.append("Higher Lows ✓")

    # ADX: need a real trend for swing
    if adx >= 30:
        score += 4; reasons.append(f"ADX {adx:.0f}")
    elif adx >= 22:
        score += 2
    elif adx < 18:
        score -= 1

    # ── LAYER 2: MOMENTUM CASCADE ─────────────────────────────────────────────
    # RSI: sweet spot 45–65 = room to run without being overbought
    rsi_delta = rsi - rsi_prev if rsi_prev > 0 else 0
    if 45 < rsi < 60:
        score += 4; reasons.append(f"RSI {rsi:.0f} Ideal")
    elif 60 <= rsi < 68:
        score += 2; reasons.append(f"RSI {rsi:.0f}")
    elif 38 < rsi <= 45:
        score += 2; reasons.append(f"RSI Reset {rsi:.0f}")  # pullback entry

    # RSI divergence: price rising, RSI also rising = confirmed
    if rsi_delta > 2 and change > 0:
        score += 2; reasons.append("RSI Divergence+")
    elif rsi_delta < -3 and change > 0:
        score -= 2; reasons.append("⚠️ RSI Div-")  # bearish divergence

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

    # 10-day trend
    if hist["trend_pct_10"] > 2:
        score += 2

    # ── LAYER 3: VOLUME FOOTPRINT ─────────────────────────────────────────────
    vol_ratio = vol / avg_vol if avg_vol > 0 else 0
    if vol_ratio >= 2.5:
        score += 4; reasons.append(f"🐋 {vol_ratio:.1f}x Vol")
    elif vol_ratio >= 1.7:
        score += 2; reasons.append(f"{vol_ratio:.1f}x Vol")
    elif vol_ratio >= 1.2:
        score += 1
    elif vol_ratio < 0.6:
        score -= 2  # suspiciously quiet

    # Historical volume accumulation
    if hist["vol_accumulation"]:
        score += 2; reasons.append("Accumulation")
    if hist["vol_trend"] > 20:
        score += 1; reasons.append("Vol Rising")

    # ── LAYER 4: PRICE PATTERN ────────────────────────────────────────────────
    # Price position in 1-month range (key for cycle timing)
    if high1m > low1m > 0:
        range1m = high1m - low1m
        pos1m   = (price - low1m) / range1m

        if 0.08 < pos1m < 0.30:
            score += 4; reasons.append("🎯 Early Cycle")
        elif 0.30 <= pos1m < 0.55:
            score += 2; reasons.append("Mid Cycle")
        elif pos1m > 0.88:
            score -= 3; reasons.append("⚠️ Near 1M High")

    # Bollinger Band: price in lower-mid zone = room to run
    bb_pos = _bb_position(price, bb_low, bb_high, bb_basis)
    if 0.2 < bb_pos < 0.6:
        score += 2; reasons.append("BB Mid-Low")
    elif bb_pos > 0.90:
        score -= 2

    # BB Squeeze: compression → explosive move
    if hist["squeeze"]:
        score += 3; reasons.append("BB Squeeze")

    # Support proximity
    if hist["support_level"] > 0 and price > 0:
        support_gap = (price / hist["support_level"] - 1) * 100
        if 0 < support_gap < 4:
            score += 2; reasons.append("At Support")
        elif support_gap < 0:
            score -= 2  # below support = danger

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
    stop, e_atr = _compute_atr_stop(price, atr, hist["atr_20"], 1.5)
    # Target: 3×ATR or nearest resistance, whichever is more conservative
    atr_target = round(price + 3.0 * e_atr, 2)
    if hist["resistance_level"] > price:
        target = min(atr_target, round(hist["resistance_level"] * 0.99, 2))
        if target <= price: target = atr_target
    else:
        target = atr_target

    return score, reasons, target, stop


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
) -> Tuple[int, List[str], float, float]:

    score   = 0
    reasons = []

    quality = SECTORS.get(sector, {"quality": 5})["quality"]

    # ── PRE-FLIGHT GATES ──────────────────────────────────────────────────────
    if price < CFG["MIN_PRICE"]:                return 0, [], 0, 0
    if quality < 6:                             return 0, [], 0, 0  # low quality sectors excluded
    if hist["stability"] < 2.5:                return 0, [], 0, 0  # structurally broken
    if rsi > 70:                                return 0, [], 0, 0  # not a value entry
    if chg1m < -20:                             return 0, [], 0, 0  # severe downtrend — wait

    # ── LAYER 1: SECTOR & FUNDAMENTAL QUALITY ─────────────────────────────────
    if quality == 9:
        score += 5; reasons.append(f"⭐⭐ {sector}")
    elif quality == 8:
        score += 4; reasons.append(f"⭐ {sector}")
    elif quality == 7:
        score += 3; reasons.append(f"{sector}")
    else:
        score += 1

    # ── LAYER 2: VALUE ZONE DETECTION ─────────────────────────────────────────
    if high1m > low1m > 0:
        dist_low = (price / low1m - 1) * 100
        pos1m    = (price - low1m) / (high1m - low1m)

        # Double-bottom / accumulation near lows
        if 0.3 < dist_low < 4 and rsi > 28:
            score += 6; reasons.append("🔄 Double Bottom")
        elif dist_low < 10:
            score += 4; reasons.append("💰 Value Zone")
        elif dist_low < 20:
            score += 2; reasons.append("Moderate Value")

        if pos1m < 0.25:
            score += 3; reasons.append("Lower Quartile")
        elif pos1m < 0.40:
            score += 1

    # Historical support level
    if hist["support_level"] > 0 and price > 0:
        gap = (price / hist["support_level"] - 1) * 100
        if 0 < gap < 3:
            score += 3; reasons.append("At Support")
        elif -2 < gap <= 0:
            score -= 2  # just below support — watch

    # ── LAYER 3: REVERSAL MOMENTUM ────────────────────────────────────────────
    # RSI: want oversold recovery, not deep in the hole
    rsi_delta = rsi - rsi_prev if rsi_prev > 0 else 0
    if 25 < rsi < 40:
        score += 4; reasons.append(f"RSI {rsi:.0f} Oversold")
    elif 40 <= rsi < 50:
        score += 3; reasons.append(f"RSI {rsi:.0f} Reset")
    elif 50 <= rsi < 60:
        score += 1

    if rsi_delta > 3:
        score += 2; reasons.append("RSI Turning Up")

    # Stochastic oversold recovery
    if stoch_k > stoch_d and stoch_k < 50:
        score += 2; reasons.append("Stoch Recovery")

    # MACD turning (histogram improving)
    if macd_hist > 0 and macd_hist > 0:
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

    stop = max(round(price * 0.88, 2), round(hist["support_level"] * 0.97, 2)) if hist["support_level"] > 0 else round(price * 0.88, 2)

    return score, reasons, target, stop

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSOR — runs all 3 engines per stock
# ══════════════════════════════════════════════════════════════════════════════

THRESH_INTRA = 13
THRESH_SWING = 16
THRESH_LONG  = 14

def process_signals(raw: list, bullish: bool, mkt_chg: float):
    if not raw:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    intra, swing, long_ = [], [], []

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
        hist   = get_hist_metrics(sym)

        # Minimum liquidity: both absolute floor AND 30% of historical average
        if vol < max(CFG["MIN_VOLUME"], hist["avg_vol"] * 0.30):
            continue

        # ── INTRADAY ──────────────────────────────────────────────────────────
        sc, rs, tgt, stp = score_intraday(
            price, change, rsi, macd, macd_sig, macd_h,
            ema5, ema10, vwap, adx, stoch_k, stoch_d,
            vol, avg_vol, atr, bb_low, bb_high, bb_basis,
            open_p, high_d, low_d,
            bullish, mkt_chg, hist, rsi_prev
        )
        if sc >= THRESH_INTRA and tgt > price > stp:
            intra.append({
                "Symbol": sym, "Sector": sector,
                "Price": round(price, 2), "Chg%": round(change, 2),
                "RV": round(rv, 1), "RSI": round(rsi, 0),
                "ADX": round(adx, 0), "Stoch": round(stoch_k, 0),
                "Score": sc,
                "Signals": " · ".join(rs[:4]),
                "Target": tgt, "Stop": stp,
                "R:R": _rr(price, tgt, stp),
            })

        # ── SWING ─────────────────────────────────────────────────────────────
        sc, rs, tgt, stp = score_swing(
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
                "Price": round(price, 2), "Chg%": round(change, 2),
                "1W%": round(chg1w, 2), "RSI": round(rsi, 0),
                "ADX": round(adx, 0), "Stab": round(hist["stability"], 1),
                "Score": sc,
                "Signals": " · ".join(rs[:4]),
                "Target": tgt, "Stop": stp,
                "R:R": _rr(price, tgt, stp),
            })

        # ── LONG-TERM ─────────────────────────────────────────────────────────
        perf1m = (price / low1m - 1) * 100 if low1m > 0 else 0.0
        sc, rs, tgt, stp = score_longterm(
            price, rsi, macd, macd_sig, macd_h,
            ema20, ema50, stoch_k, stoch_d,
            bb_low, bb_high, bb_basis,
            vol, avg_vol, chg1w, chg1m, low1m, high1m,
            sector, rsi_prev, hist
        )
        if sc >= THRESH_LONG and tgt > price > stp:
            long_.append({
                "Symbol": sym, "Sector": sector,
                "Price": round(price, 2), "1W%": round(chg1w, 2),
                "1M%": round(perf1m, 2), "RSI": round(rsi, 0),
                "Stab": round(hist["stability"], 1), "Score": sc,
                "Signals": " · ".join(rs[:4]),
                "Target": tgt, "Stop": stp,
                "R:R": _rr(price, tgt, stp),
            })

    srt = lambda lst: pd.DataFrame(lst).sort_values("Score", ascending=False).reset_index(drop=True) if lst else pd.DataFrame()
    return srt(intra), srt(swing), srt(long_)

# ══════════════════════════════════════════════════════════════════════════════
# POSITION TRACKER — real-time hold/exit guidance
# ══════════════════════════════════════════════════════════════════════════════

def position_status(symbol: str, entry: float, raw: list) -> Optional[dict]:
    match = next((item for item in raw if item.get("d", [None])[0] == symbol), None)
    if not match:
        return None
    d = match["d"]
    curr      = safe(d[1])
    vol       = safe(d[3])
    avg_vol   = safe(d[5])
    rsi       = safe(d[6], 50)
    macd      = safe(d[7])
    macd_sig  = safe(d[8])
    ema20     = safe(d[11])
    ema50     = safe(d[12])
    ema10     = safe(d[17])
    adx       = safe(d[18])
    atr       = safe(d[19])
    stoch_k   = safe(d[20], 50)
    bb_low    = safe(d[9])
    bb_high   = safe(d[10])
    bb_basis  = safe(d[30]) if len(d) > 30 else (bb_low + bb_high) / 2
    rv        = vol / avg_vol if avg_vol > 0 else 1.0

    pnl_pct = (curr - entry) / entry * 100
    pnl_amt = curr - entry

    signals = []
    action  = "🟢 HOLD"
    conf    = "Medium"

    # Trail stop based on ATR
    trail_stop = round(curr - 1.5 * atr, 2) if atr > 0 else round(curr * 0.95, 2)
    bb_pos     = _bb_position(curr, bb_low, bb_high, bb_basis)

    # ── EXIT SIGNALS (priority order) ─────────────────────────────────────────
    if pnl_pct < -9:
        signals.append("🛑 Hard stop triggered — exit all immediately")
        action = "🔴 EXIT NOW"; conf = "Critical"

    elif curr < ema50 and pnl_pct < -3:
        signals.append("❌ Below EMA50 — major trend broken")
        action = "🔴 EXIT"; conf = "High"

    elif curr < ema20 and curr < ema10 and macd < macd_sig:
        signals.append("❌ EMA stack broken + MACD bear")
        action = "🔴 EXIT"; conf = "High"

    elif rsi > 78 and rv < 0.8 and bb_pos > 0.90:
        signals.append("🚨 RSI extreme + volume drying + BB upper")
        action = "🔴 EXIT"; conf = "High"

    elif pnl_pct > 18:
        signals.append("🎯 Target exceeded — book 80%, trail rest")
        action = "🟡 SCALE OUT"; conf = "High"

    elif pnl_pct > 10:
        signals.append(f"💰 Strong profit — book 50%, move stop to {trail_stop:.2f}")
        action = "🟡 PARTIAL EXIT"; conf = "High"

    elif pnl_pct > 5 and (rsi > 72 or bb_pos > 0.85):
        signals.append("⚠️ Profit + overbought signals — book 30%")
        action = "🟡 TRIM"

    elif macd < macd_sig and adx < 20 and stoch_k < stoch_k:
        signals.append("📉 MACD bear + weakening ADX + Stoch down")
        if pnl_pct > 3:
            action = "🟡 PARTIAL EXIT"

    elif pnl_pct < -5 and adx < 18 and curr < ema20:
        signals.append("⚠️ Loss + no trend + below EMA20")
        action = "🔴 EXIT"

    # ── HOLD SIGNALS ─────────────────────────────────────────────────────────
    if not signals:
        hold_signals = []
        if adx > 25 and curr > ema20:
            hold_signals.append(f"✅ Trend strong (ADX {adx:.0f})")
            conf = "High"
        if macd > macd_sig:
            hold_signals.append("✅ MACD bullish")
        if 45 < rsi < 70:
            hold_signals.append(f"✅ RSI healthy ({rsi:.0f})")
        if not hold_signals:
            hold_signals.append("⏳ Position developing — monitor EMA20")
        signals = hold_signals

    # Add ATR trail stop suggestion
    if action == "🟢 HOLD" and pnl_pct > 2:
        signals.append(f"📌 Trail stop: {trail_stop:.2f}")

    return {
        "symbol": symbol, "entry": entry, "current": curr,
        "pnl_pct": pnl_pct, "pnl_amt": pnl_amt,
        "rsi": rsi, "rv": rv, "adx": adx,
        "action": action, "signals": signals[:4], "confidence": conf,
        "trail_stop": trail_stop,
    }

# ══════════════════════════════════════════════════════════════════════════════
# UI — STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════

if "positions" not in st.session_state:
    st.session_state.positions = []

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', -apple-system, sans-serif; box-sizing: border-box; }
html, body, [class*="css"] { background: #060b18 !important; color: #e2e8f0 !important; }
footer, #MainMenu { visibility: hidden; }

.hdr {
    background: linear-gradient(135deg, rgba(10,18,38,0.98) 0%, rgba(6,11,24,1) 100%);
    border: 1px solid rgba(255,255,255,0.07); border-radius: 20px;
    padding: 1.75rem 2.2rem; margin-bottom: 1.5rem;
    box-shadow: 0 32px 80px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.05);
}
.hdr-title {
    font-size: 2.3rem; font-weight: 900; letter-spacing: -0.04em;
    background: linear-gradient(135deg, #00ff88 0%, #00d4ff 45%, #7c3aed 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.45rem;
}
.hdr-sub {
    font-family: 'JetBrains Mono', monospace; font-size: 0.76rem;
    color: #3d5166; display: flex; gap: 1.2rem; flex-wrap: wrap; align-items: center;
}
.hdr-pill {
    background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2);
    color: #10b981; border-radius: 20px; padding: 2px 10px; font-size: 0.68rem; font-weight: 700;
}

.badge {
    display: inline-block; padding: 0.3rem 0.85rem; border-radius: 8px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; border: 1px solid;
}
.b-open   { border-color:#10b981; color:#10b981; background:rgba(16,185,129,0.07); }
.b-closed { border-color:#ef4444; color:#ef4444; background:rgba(239,68,68,0.07); }
.b-bull   { border-color:#10b981; color:#10b981; background:rgba(16,185,129,0.12); }
.b-bear   { border-color:#ef4444; color:#ef4444; background:rgba(239,68,68,0.12); }
.b-neut   { border-color:#f59e0b; color:#f59e0b; background:rgba(245,158,11,0.10); }

.idx-bar {
    background: rgba(10,18,38,0.7); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 1.3rem 1.75rem; margin-bottom: 1.5rem;
}

.sec-wrap { display:flex; flex-wrap:wrap; gap:0.35rem; margin-bottom:1rem; }
.sec-tile {
    background: rgba(10,18,38,0.8); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px; padding: 4px 11px;
    display:flex; align-items:center; gap:7px;
}
.sec-name { font-size:0.7rem; font-weight:500; color:#64748b; }
.sec-val  { font-family:'JetBrains Mono',monospace; font-size:0.76rem; font-weight:700; }

.sc {
    background: linear-gradient(155deg, rgba(10,18,38,0.75) 0%, rgba(6,11,24,0.9) 100%);
    border: 1px solid rgba(255,255,255,0.06); border-radius: 16px;
    padding: 1.5rem; margin-bottom: 1.6rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.sc-hdr { font-size:1.15rem; font-weight:800; margin-bottom:0.35rem; letter-spacing:-0.01em; }
.sc-sub { font-size:0.76rem; color:#4b5563; margin-bottom:1rem; }
.sc-stats { display:flex; gap:1.5rem; margin-bottom:1rem; }
.sc-stat { font-size:0.72rem; color:#64748b; }
.sc-stat span { color:#94a3b8; font-weight:600; font-family:'JetBrains Mono',monospace; }

.pos-card {
    background: linear-gradient(135deg, rgba(17,24,39,0.95) 0%, rgba(10,14,26,0.98) 100%);
    border: 1px solid rgba(30,41,59,0.8); border-radius: 13px; padding: 1.1rem;
    margin-bottom: 0.5rem; transition: border-color 0.2s, box-shadow 0.2s;
}
.pos-card:hover { border-color:#2d4a6b; box-shadow: 0 4px 20px rgba(59,130,246,0.1); }
.pnl-pos { color:#10b981 !important; font-weight:800; }
.pnl-neg { color:#ef4444 !important; font-weight:800; }

.cb { display:inline-block; padding:0.2rem 0.6rem; border-radius:6px;
      font-size:0.62rem; font-weight:700; letter-spacing:0.06em; border:1px solid; }
.cb-high     { background:rgba(16,185,129,0.1);  color:#10b981; border-color:rgba(16,185,129,0.3); }
.cb-critical { background:rgba(239,68,68,0.1);   color:#ef4444; border-color:rgba(239,68,68,0.3); }
.cb-medium   { background:rgba(245,158,11,0.08); color:#f59e0b; border-color:rgba(245,158,11,0.25); }

[data-testid="stDataFrame"] { background:transparent !important; border-radius:10px; overflow:hidden; }
thead tr th {
    background:rgba(10,18,38,0.9) !important; color:#475569 !important;
    font-family:'JetBrains Mono',monospace !important; font-size:0.66rem !important;
    font-weight:700 !important; letter-spacing:0.1em !important; text-transform:uppercase;
    padding: 10px 8px !important;
}
tbody tr:nth-child(even) { background:rgba(10,18,38,0.3) !important; }
tbody tr:hover { background:rgba(30,58,100,0.2) !important; }

.stButton>button {
    background:linear-gradient(135deg,#2563eb,#1d4ed8); color:#fff;
    border:none; border-radius:9px; font-weight:700; font-size:0.88rem;
    transition:all 0.2s; letter-spacing:0.01em;
}
.stButton>button:hover { transform:translateY(-1px); box-shadow:0 6px 20px rgba(37,99,235,0.45); }

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

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛠️ Data Engine")
    if st.button("🔄 Sync 90-Day OHLCV", use_container_width=True,
                 help="Downloads full OHLCV from Yahoo Finance — run once per session"):
        sync_historical_data()
        st.rerun()
    st.caption("Powers 12 historical metrics: ATR, support/resistance, higher-lows, BB squeeze, volume accumulation, momentum, and more.")

    st.divider()
    st.markdown("### ⚙️ Score Thresholds")
    st.caption("Lower = more results · Higher = only elite setups")
    thresh_i = st.slider("⚡ Intraday",   8,  25, THRESH_INTRA, help="Max ~30")
    thresh_s = st.slider("🚀 Swing",      10, 30, THRESH_SWING, help="Max ~36")
    thresh_l = st.slider("💎 Long-Term",  8,  28, THRESH_LONG,  help="Max ~35")

    st.divider()
    st.markdown("### 📊 Signal Guide")
    st.markdown("""
| Score | Quality |
|---|---|
| 90%+ max | 🔥 Elite |
| 70–89%   | ✅ Strong |
| 50–69%   | ⚡ Good |
| < 50%    | ⚠️ Weak |
""")

# ─── Market State ─────────────────────────────────────────────────────────────
is_open = is_market_open()
now_str = pkt_now().strftime("%H:%M PKT · %a %d %b %Y")

avg_chg, bullish, adv, dec, kse_fb = fetch_breadth()
kse_api = fetch_kse_index()

def _kse(key, fb):
    return kse_api.get(key, fb) if kse_api else fb

idx_close = _kse("close",         kse_fb["close"])
idx_abs   = _kse("change",        0)
idx_pct   = _kse("changePercent", kse_fb["change"])
idx_high  = _kse("high",          kse_fb["high"])
idx_low   = _kse("low",           kse_fb["low"])
idx_vol   = _kse("volume",        kse_fb["volume"])

idx_col  = "#10b981" if idx_pct >= 0 else "#ef4444"
bclass   = "b-bull" if bullish else ("b-bear" if avg_chg < -0.5 else "b-neut")
btext    = f"{'BULLISH' if bullish else 'BEARISH'} {avg_chg:+.2f}%"
mclass   = "b-open" if is_open else "b-closed"
mtext    = "● LIVE" if is_open else "● CLOSED"
vol_cr   = idx_vol / 1e7

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hdr">
  <div class="hdr-title">⚡ PSX Elite Institutional Scanner</div>
  <div class="hdr-sub">
    <span class="hdr-pill">7-Layer Signal Engine</span>
    <span>KSE-100 · Live + Historical Intelligence · Pattern Detection</span>
    <span style="margin-left:auto; color:#2d4a6b;">{now_str}</span>
  </div>
</div>""", unsafe_allow_html=True)

# ─── KSE-100 Index Bar ────────────────────────────────────────────────────────
adv_ratio = adv / (adv + dec) * 100 if (adv + dec) > 0 else 50
st.markdown(f"""
<div class="idx-bar">
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:1rem;">
    <span class="badge {mclass}">{mtext}</span>
    <span class="badge {bclass}">{btext}</span>
    <span style="color:#10b981;font-weight:800;font-family:'JetBrains Mono';font-size:0.82rem;margin-left:6px;">▲ {adv}</span>
    <span style="color:#ef4444;font-weight:800;font-family:'JetBrains Mono';font-size:0.82rem;">▼ {dec}</span>
    <span style="color:#64748b;font-size:0.72rem;margin-left:4px;">A/D {adv_ratio:.0f}%</span>
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:2.5rem;align-items:flex-end;">
    <div>
      <div style="font-size:0.68rem;color:#334155;text-transform:uppercase;font-weight:700;margin-bottom:3px;letter-spacing:0.08em;">KSE-100 Index</div>
      <div style="font-size:2.4rem;font-weight:900;color:#f8fafc;line-height:1;font-family:'JetBrains Mono';">{idx_close:,.0f}</div>
      <div style="font-size:1.05rem;font-weight:700;color:{idx_col};margin-top:5px;">
        {f"{idx_abs:+,.0f} " if idx_abs else ""}({idx_pct:+.2f}%)
      </div>
    </div>
    <div style="display:flex;gap:2rem;padding-left:2rem;border-left:1px solid rgba(255,255,255,0.06);">
      <div><div style="color:#2d4a6b;font-size:0.64rem;text-transform:uppercase;margin-bottom:3px;">High</div>
           <div style="font-weight:700;color:#cbd5e1;font-family:'JetBrains Mono';">{idx_high:,.0f}</div></div>
      <div><div style="color:#2d4a6b;font-size:0.64rem;text-transform:uppercase;margin-bottom:3px;">Low</div>
           <div style="font-weight:700;color:#cbd5e1;font-family:'JetBrains Mono';">{idx_low:,.0f}</div></div>
      <div><div style="color:#2d4a6b;font-size:0.64rem;text-transform:uppercase;margin-bottom:3px;">Volume</div>
           <div style="font-weight:700;color:#cbd5e1;font-family:'JetBrains Mono';">{vol_cr:.2f} Cr</div></div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ─── Market breadth warning ───────────────────────────────────────────────────
if not bullish:
    st.markdown(f'<div class="alert-box">⚠️ <strong>Bearish Breadth</strong> — Market declining ({adv} advancers vs {dec} decliners). Reduce position sizes. Intraday setups require extra confirmation.</div>', unsafe_allow_html=True)

# ─── Scan Trigger ─────────────────────────────────────────────────────────────
scan = is_open
if not is_open:
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        if st.button("🔍 Manual Scan", use_container_width=True):
            scan = True
    with col_info:
        st.markdown('<div class="info-box" style="margin-top:0.25rem;">Market closed — VWAP-based intraday signals use last close. Swing & Long-term remain fully valid.</div>', unsafe_allow_html=True)

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
        star = "⭐" if q == 9 else ("★" if q == 8 else "")
        tiles += f'<div class="sec-tile"><span class="sec-name">{star}{html.escape(sec)}</span><span class="sec-val" style="color:{col};">{avg:+.1f}%</span></div>'

    st.markdown(f'<div style="font-size:0.72rem;color:#334155;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">Sector Heatmap</div><div class="sec-wrap">{tiles}</div>', unsafe_allow_html=True)

    # ── Run Signals ───────────────────────────────────────────────────────────
    df_i, df_s, df_l = process_signals(raw, bullish, avg_chg)

    if not df_i.empty: df_i = df_i[df_i["Score"] >= thresh_i].reset_index(drop=True)
    if not df_s.empty: df_s = df_s[df_s["Score"] >= thresh_s].reset_index(drop=True)
    if not df_l.empty: df_l = df_l[df_l["Score"] >= thresh_l].reset_index(drop=True)

    def render_table(df: pd.DataFrame, col_cfg: dict, score_max: int):
        if df.empty:
            st.markdown('<div style="color:#334155;font-size:0.85rem;padding:0.75rem;">🔍 No setups meet current threshold. Try lowering the slider in the sidebar.</div>', unsafe_allow_html=True)
        else:
            # Add score% column
            df = df.copy()
            df["Score%"] = (df["Score"] / score_max * 100).round(0).astype(int).astype(str) + "%"
            st.dataframe(df, column_config=col_cfg, hide_index=True, use_container_width=True)

    # ── INTRADAY ──────────────────────────────────────────────────────────────
    st.markdown(f"""<div class="sc">
      <div class="sc-hdr">⚡ <span style="color:#fbbf24;">INTRADAY</span> SCALPS</div>
      <div class="sc-sub">Same-day exits · VWAP anchor · Institutional volume · ATR stops · EMA5/10 alignment</div>
      <div class="sc-stats">
        <div class="sc-stat">Results: <span>{len(df_i)}</span></div>
        <div class="sc-stat">Threshold: <span>{thresh_i}/{30}</span></div>
        <div class="sc-stat">Breadth: <span style="color:{'#10b981' if bullish else '#ef4444'}">{'✅ Bullish' if bullish else '⚠️ Bearish'}</span></div>
      </div>
    """, unsafe_allow_html=True)
    render_table(df_i, {
        "Chg%":   st.column_config.NumberColumn("Chg%",   format="%.2f%%"),
        "RV":     st.column_config.NumberColumn("RV",     format="%.1fx"),
        "RSI":    st.column_config.NumberColumn("RSI",    format="%.0f"),
        "ADX":    st.column_config.NumberColumn("ADX",    format="%.0f"),
        "Stoch":  st.column_config.NumberColumn("Stoch",  format="%.0f"),
        "Target": st.column_config.NumberColumn("Target", format="%.2f"),
        "Stop":   st.column_config.NumberColumn("Stop",   format="%.2f"),
    }, score_max=30)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── SWING ────────────────────────────────────────────────────────────────
    st.markdown(f"""<div class="sc">
      <div class="sc-hdr">🚀 <span style="color:#3b82f6;">SWING</span> TRADES</div>
      <div class="sc-sub">3–7 day holds · EMA fan · Cycle position · Volume accumulation · BB squeeze · Support proximity</div>
      <div class="sc-stats">
        <div class="sc-stat">Results: <span>{len(df_s)}</span></div>
        <div class="sc-stat">Threshold: <span>{thresh_s}/{36}</span></div>
        <div class="sc-stat">Breadth: <span style="color:{'#10b981' if bullish else '#ef4444'}">{'✅ Bullish' if bullish else '⚠️ Bearish'}</span></div>
      </div>
    """, unsafe_allow_html=True)
    render_table(df_s, {
        "Chg%":   st.column_config.NumberColumn("Chg%",   format="%.2f%%"),
        "1W%":    st.column_config.NumberColumn("1W%",    format="%.2f%%"),
        "RSI":    st.column_config.NumberColumn("RSI",    format="%.0f"),
        "ADX":    st.column_config.NumberColumn("ADX",    format="%.0f"),
        "Stab":   st.column_config.NumberColumn("Stab",   format="%.1f"),
        "Target": st.column_config.NumberColumn("Target", format="%.2f"),
        "Stop":   st.column_config.NumberColumn("Stop",   format="%.2f"),
    }, score_max=36)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── LONG-TERM ────────────────────────────────────────────────────────────
    st.markdown(f"""<div class="sc">
      <div class="sc-hdr">💎 <span style="color:#10b981;">LONG-TERM</span> INVESTMENTS</div>
      <div class="sc-sub">Multi-week holds · Sector quality · Double bottoms · Wyckoff accumulation · RSI recovery · BB value</div>
      <div class="sc-stats">
        <div class="sc-stat">Results: <span>{len(df_l)}</span></div>
        <div class="sc-stat">Threshold: <span>{thresh_l}/{35}</span></div>
        <div class="sc-stat">Focus: <span>Banks ⭐ · E&P ⭐ · Fertilizer ⭐</span></div>
      </div>
    """, unsafe_allow_html=True)
    render_table(df_l, {
        "1W%":    st.column_config.NumberColumn("1W%",    format="%.2f%%"),
        "1M%":    st.column_config.NumberColumn("1M%",    format="%.2f%%"),
        "RSI":    st.column_config.NumberColumn("RSI",    format="%.0f"),
        "Stab":   st.column_config.NumberColumn("Stab",   format="%.1f"),
        "Target": st.column_config.NumberColumn("Target", format="%.2f"),
        "Stop":   st.column_config.NumberColumn("Stop",   format="%.2f"),
    }, score_max=35)
    st.markdown("</div>", unsafe_allow_html=True)

# ── POSITION TRACKER ──────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📊 Position Tracker")
c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
with c1:
    sym_in = st.text_input("Symbol", placeholder="HBL, LUCK, PPL…", key="ps").strip().upper()
with c2:
    entry_in = st.number_input("Entry Price", min_value=0.0, step=0.01, key="pe")
with c3:
    qty_in = st.number_input("Qty (shares)", min_value=0, step=100, key="pq")
with c4:
    st.write("")
    if st.button("➕ Add", use_container_width=True):
        if sym_in and entry_in > 0:
            st.session_state.positions.append({"symbol": sym_in, "entry": entry_in, "qty": qty_in or 0})
            st.success(f"Added {sym_in} @ {entry_in:.2f}")
            st.rerun()

if st.session_state.positions:
    pos_raw = fetch_live()
    for idx, pos in enumerate(st.session_state.positions):
        status = position_status(pos["symbol"], pos["entry"], pos_raw)
        if not status:
            st.warning(f"{pos['symbol']} — no live data found")
            continue

        qty     = pos.get("qty", 0)
        pnl_cls = "pnl-pos" if status["pnl_pct"] > 0 else "pnl-neg"
        ck      = status["confidence"].lower()
        cb_cls  = "cb-critical" if ck == "critical" else ("cb-high" if ck == "high" else "cb-medium")
        total_pnl = status["pnl_amt"] * qty if qty > 0 else None

        ca, cb_, cc = st.columns([2, 4, 1])
        with ca:
            total_str = f'<div style="font-size:0.75rem;color:#10b981 !important;">PKR {total_pnl:+,.0f} total</div>' if total_pnl else ""
            st.markdown(f"""<div class="pos-card">
              <div style="font-size:1.35rem;font-weight:900;font-family:'JetBrains Mono';">{status['symbol']}</div>
              <div style="font-size:0.74rem;color:#334155;margin:0.25rem 0;">{status['entry']:.2f} → <strong style="color:#94a3b8;">{status['current']:.2f}</strong></div>
              <div style="font-size:1.2rem;font-weight:800;font-family:'JetBrains Mono';" class="{pnl_cls}">{status['pnl_pct']:+.2f}%</div>
              <div style="font-size:0.75rem;color:#475569;">PKR {status['pnl_amt']:+.2f}/share</div>
              {total_str}
            </div>""", unsafe_allow_html=True)
        with cb_:
            bullets = "".join(f'<div style="margin-bottom:4px;font-size:0.8rem;">• {s}</div>' for s in status["signals"])
            st.markdown(f"""<div class="pos-card">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.65rem;">
                <div style="font-size:0.98rem;font-weight:800;">{status['action']}</div>
                <span class="cb {cb_cls}">{status['confidence'].upper()}</span>
              </div>
              <div style="color:#94a3b8;line-height:1.7;">{bullets}</div>
              <div style="margin-top:0.65rem;padding-top:0.65rem;border-top:1px solid rgba(30,41,59,0.5);
                          font-size:0.7rem;color:#334155;font-family:'JetBrains Mono';">
                RSI {status['rsi']:.0f} · Vol {status['rv']:.1f}x · ADX {status['adx']:.0f} · Trail {status['trail_stop']:.2f}
              </div>
            </div>""", unsafe_allow_html=True)
        with cc:
            st.write("")
            if st.button("🗑️ Remove", key=f"del_{idx}", use_container_width=True):
                st.session_state.positions.pop(idx); st.rerun()

# ── METHODOLOGY ───────────────────────────────────────────────────────────────
st.divider()
with st.expander("📖 Signal Methodology & Risk Framework", expanded=False):
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("""
**7-Layer Signal Engine**

Each stock is evaluated across 7 independent layers. All layers must show net positive signals to qualify.

1. **Trend Structure** — EMA alignment (5/10/20/50), ADX strength, candle position, gap-ups. The foundation — no trend = no trade.

2. **Momentum Cascade** — RSI zone + direction, MACD histogram expansion, Stochastic crossovers, relative strength vs market. Is buying pressure accelerating?

3. **Volume Footprint** — Institutional volume (≥2.5× avg = 🐋), volume trend from history, on-balance accumulation (up-day vol > down-day vol). Who is buying?

4. **Price Pattern** — VWAP zone (intraday), 1-month cycle position, Bollinger Band position + squeeze, support/resistance proximity. Where exactly to buy?

5. **Market Breadth** — Advancers vs decliners, average market change, relative alpha. Is the tide helping?

6. **Volatility State** — Historical volatility (annualised), ATR-based stops, BB squeeze detection (pre-breakout compression). Is risk manageable?

7. **Historical Quality** — 90-day stability score, higher-lows pattern, 30-day trend, consecutive streak, Wyckoff accumulation. Has this stock been behaving well?
""")
    with m2:
        st.markdown("""
**Hard Gates (automatic disqualification)**

- **Intraday:** Price below VWAP, volume <1×avg, RSI >82, price <₹2
- **Swing:** Price below EMA50, RSI >78, ADX <15, volume <80% of floor
- **Long-term:** Sector quality <6, stability <2.5, RSI >70, 1-month loss >20%

**Risk Management**

| Strategy | Capital/Trade | Stop Type | Target |
|---|---|---|---|
| Intraday | ≤1% portfolio | 0.6×ATR | 1.0×ATR |
| Swing | ≤3% portfolio | 1.5×ATR or support | 3.0×ATR |
| Long-Term | ≤5% portfolio | −12% hard | +25–55% |

**Golden Rules**
- Never add to a losing position
- Trail stops once +5% in profit
- Exit if A/D ratio drops below 0.4 during intraday
- Breadth bearish = reduce all swing sizes by 50%
- Maximum 2 stocks per sector simultaneously
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid rgba(15,23,42,0.8);
            font-size:0.68rem;color:#1e293b;font-family:'JetBrains Mono',monospace;text-align:center;">
  PSX Elite Scanner · 7-Layer Engine · {len(KSE100)} Symbols · ATR Stops · Live + 90-Day History ·
  {pkt_now().strftime("%H:%M:%S PKT")}
</div>""", unsafe_allow_html=True)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if is_open:
    time.sleep(CFG["REFRESH_SEC"])
    st.rerun()
