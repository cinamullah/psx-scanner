import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import pytz
from requests.adapters import HTTPAdapter
import html
from urllib3.util.retry import Retry
from typing import Dict, List, Tuple, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "START_TIME": "09:17",
    "END_TIME": "15:30",
    "REFRESH_RATE": 120,
    "MIN_LIQUIDITY": 50000,
    "ATR_STOP_MULTIPLIER": 1.5,
    "MARKET_BREADTH_THRESHOLD": 0.0,
}

KSE100_SYMBOLS = ["TRG", "SYS", "NETSOL", "AIRLINK", "ATRL", "NRL", "PRL", "MLCF", "DGKC", "LUCK", "CHCC", "KOHCK", "BWCL", "FCCL", "HCAR", "ATLH", "MTL", "THALL", "HUBC", "PAEL", "KEL", "NCPL", "KAPCO", "MEBL", "UBL", "BAFL", "AKBL", "BAHL", "EFERT", "FFC", "FATIMA", "SFERT", "SNGP", "APL", "MUREB", "RAFHAN", "COLG", "BNWM", "KTML", "ABOT", "SEARL"]

SECTORS = {
    "Banks": {"symbols": ["MEBL","UBL","ABL","BAFL","BAHL","BOP","NBP"], "quality": 9},
    "E&P": {"symbols": ["OGDC","PPL","MARI","POL"], "quality": 8},
    "Fertilizer": {"symbols": ["FFC","ENGROH","EFERT"], "quality": 9},
    "Cement": {"symbols": ["LUCK","DGKC","MLCF","CHCC"], "quality": 7},
    "Tech": {"symbols": ["SYS","TRG","PTC","NETSOL","AIRLINK"], "quality": 6},
    "Power": {"symbols": ["HUBC","KEL","NCPL","PAEL","NPL"], "quality": 7},
    "Oil & Gas": {"symbols": ["SNGP","SSGC","PSO","NRL","ATRL","PRL","CNERGY"], "quality": 8},
    "Auto": {"symbols": ["ATLH","HCAR","MTL"], "quality": 6},
    "Food": {"symbols": ["UNITY","COLG"], "quality": 8},
    "Pharma": {"symbols": ["ABOT"], "quality": 9},
    "Textile": {"symbols": ["KOSM","CLOV","ILP"], "quality": 5},
    "Misc": {"symbols": ["YOUW","CSIL","PIBTL","TGL","TELE"], "quality": 6},
}

SYMBOL_TO_SECTOR = {sym: sec for sec, data in SECTORS.items() for sym in data["symbols"]}

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def pkt_now():
    return datetime.now(pytz.timezone("Asia/Karachi"))

def is_market_open():
    now = pkt_now()
    if now.weekday() >= 5:
        return False
    t = now.strftime("%H:%M")
    return CONFIG["START_TIME"] <= t <= CONFIG["END_TIME"]

def _safe(val, default=0):
    return val if val is not None else default

def calculate_atr_stop(price: float, atr: float, multiplier: float = 1.5) -> float:
    return round(price - (multiplier * atr), 2)

def check_ema_fan(ema5: float, ema10: float, ema20: float, ema50: float) -> bool:
    return ema5 > ema10 > ema20 > ema50

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

TV_URL = "https://scanner.tradingview.com/pakistan/scan"
TV_COLUMNS = [
    "name", "close", "change", "volume", "relative_volume_10d_calc",
    "average_volume_10d_calc", "RSI", "MACD.macd", "MACD.signal",
    "BB.lower", "BB.upper", "EMA20", "EMA50",
    "change|1W", "High.1M", "Low.1M", "VWAP",
    "EMA10", "ADX", "ATR", "Stoch.K"
]

@st.cache_resource
def get_session():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s

@st.cache_data(ttl=25, show_spinner=False)
def fetch_market_data():
    payload = {
        "filter": [{"left": "name", "operation": "in_range", "right": KSE100_SYMBOLS}],
        "markets": ["pakistan"],
        "columns": TV_COLUMNS,
        "range": [0, 400],
    }
    try:
        r = get_session().post(TV_URL, json=payload, timeout=15)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        st.error(f"⚡ Data fetch failed: {e}")
        return []

@st.cache_data(ttl=30, show_spinner=False)
def fetch_breadth():
    symbols = list(set(KSE100_SYMBOLS + ["KSE100", "PKS100"]))
    payload = {
        "filter": [{"left": "name", "operation": "in_range", "right": symbols}],
        "markets": ["pakistan"],
        "columns": ["name", "change", "close", "high", "low", "volume"],
        "range": [0, 500],
    }
    try:
        r = get_session().post(TV_URL, json=payload, timeout=10)
        data = r.json().get("data", [])

        constituent_changes = []
        kse_info = {"close": 0.0, "change": 0.0, "high": 0.0, "low": 0.0, "volume": 0}

        for item in data:
            ticker_full = item.get("s", "").upper()
            d = item.get("d", [])
            if len(d) < 6: continue

            name = _safe(d[0], "")
            change = _safe(d[1])
            close = _safe(d[2])

            if "KSE100" in ticker_full or "PKS100" in ticker_full or name == "KSE100":
                kse_info = {
                    "close": close,
                    "change": change,
                    "high": _safe(d[3]),
                    "low": _safe(d[4]),
                    "volume": _safe(d[5])
                }
            if name in KSE100_SYMBOLS or ticker_full.split(":")[-1] in KSE100_SYMBOLS:
                constituent_changes.append(change)

        if not constituent_changes:
            return 0.0, False, 0, 0, kse_info

        avg = sum(constituent_changes) / len(constituent_changes)
        adv = sum(1 for c in constituent_changes if c > 0)
        dec = sum(1 for c in constituent_changes if c < 0)
        bullish = avg > CONFIG["MARKET_BREADTH_THRESHOLD"] and adv > dec

        return avg, bullish, adv, dec, kse_info
    except Exception:
        return 0.0, False, 0, 0, {"close": 0.0, "change": 0.0, "high": 0.0, "low": 0.0, "volume": 0}

@st.cache_data(ttl=60, show_spinner=False)
def fetch_sarmaaya_index():
    """Fetch KSE100 overview from Sarmaaya API"""
    try:
        r = requests.get("https://beta-restapi.sarmaaya.pk/api/indices/overview/KSE100", timeout=10)
        if r.status_code == 200:
            return r.json().get("response", {})
    except:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_intraday(price, change, rv, rsi, macd, macd_sig, ema10, vwap, adx, stoch, vol, avg_vol, atr, bullish):
    score = 0
    reasons = []

    penalty = 0 if bullish else -2
    vol_ratio = vol / avg_vol if avg_vol > 0 else 0

    # Volume surge
    if vol_ratio > 3.0:
        score += 4
        reasons.append(f"🐋 {vol_ratio:.1f}x Vol")
    elif vol_ratio > 2.0:
        score += 3
        reasons.append(f"{vol_ratio:.1f}x Vol")
    elif rv > 1.5:
        score += 2

    # VWAP edge
    if price > vwap:
        dist = ((price - vwap) / vwap) * 100
        if dist < 1.5:
            score += 2
            reasons.append("VWAP Edge")
        elif dist > 3:
            score -= 1

    # EMA alignment
    if price > ema10:
        score += 2
        reasons.append("EMA+")

    # ADX strength
    if adx > 30:
        score += 3
        reasons.append(f"ADX{adx:.0f}")
    elif adx > 25:
        score += 2

    # RSI
    if 50 < rsi < 65:
        score += 2
        reasons.append(f"RSI{rsi:.0f}")
    elif rsi >= 75:
        score -= 2
        reasons.append("Overbought")

    # MACD
    if macd > macd_sig and macd > 0:
        score += 2
        reasons.append("MACD+")

    # Stochastic
    if 30 < stoch < 80:
        score += 1

    score += penalty

    stop = round(price - (0.5 * atr), 2) if atr > 0 else round(price * 0.99, 2)
    target = round(price + (0.8 * atr), 2) if atr > 0 else round(price * 1.018, 2)

    if price - stop <= 0:
        stop = round(price * 0.985, 2)

    return score, reasons, target, stop

def analyze_swing(price, change, rv, rsi, macd, macd_sig, ema10, ema20, ema50, change1w, low1m, adx, atr, high1m, vol, avg_vol, bullish):
    score = 0
    reasons = []

    penalty = 0 if bullish else -1
    ema5 = ema10 * 1.01

    # EMA fan
    if check_ema_fan(ema5, ema10, ema20, ema50):
        score += 4
        reasons.append("EMA Fan ✓")
    elif price > ema20 and price > ema50:
        score += 2
        reasons.append("EMA20/50+")

    # ADX
    if adx > 25:
        score += 3
        reasons.append(f"ADX{adx:.0f}")
    elif adx > 20:
        score += 2

    # Range position
    if low1m > 0 and high1m > 0:
        pos = (price - low1m) / (high1m - low1m)
        if 0.2 < pos < 0.5:
            score += 3
            reasons.append("Value Zone")
        elif pos > 0.85:
            score -= 2

    # Volume
    vol_ratio = vol / avg_vol if avg_vol > 0 else 0
    if vol_ratio > 1.5:
        score += 2
        reasons.append(f"{vol_ratio:.1f}x RV")
    elif rv > 1.2:
        score += 1

    # RSI
    if 45 < rsi < 60:
        score += 3
        reasons.append(f"RSI{rsi:.0f}")
    elif rsi > 70:
        score -= 2

    # Weekly momentum
    if change1w > 3:
        score += 1
        reasons.append(f"1W+{change1w:.1f}%")

    # MACD
    if macd > macd_sig:
        score += 1

    score += penalty

    stop = round(price - (1.2 * atr), 2) if atr > 0 else round(price * 0.96, 2)
    target = round(price + (2.5 * atr), 2) if atr > 0 else round(price * 1.075, 2)

    if price - stop <= 0:
        stop = round(price * 0.93, 2)

    return score, reasons, target, stop

def analyze_longterm(price, rsi, ema20, ema50, change1w, low1m, high1m, sector, vol, avg_vol):
    score = 0
    reasons = []

    # Sector quality
    quality = SECTORS.get(sector, {"quality": 5})["quality"]
    if quality >= 9:
        score += 3
        reasons.append(f"{sector}")
    elif quality >= 7:
        score += 2

    # Value zone
    if low1m > 0 and high1m > 0:
        dist = (price / low1m - 1) * 100
        if 1 < dist < 5 and rsi > 30:
            score += 4
            reasons.append("Double Bottom")
        elif dist < 15:
            score += 3
            reasons.append("Value Zone")

        pos = (price - low1m) / (high1m - low1m)
        if pos < 0.35:
            score += 2
            reasons.append("Lower 1/3")

    # RSI
    if 30 < rsi < 45:
        score += 3
        reasons.append(f"RSI{rsi:.0f}")
    elif rsi < 50:
        score += 1

    # EMA50
    if price > ema50:
        score += 3
        reasons.append("EMA50+")

    # Weekly stability
    if change1w > -5:
        score += 1

    # Liquidity
    vol_ratio = vol / avg_vol if avg_vol > 0 else 0
    if vol_ratio > 0.8:
        score += 1

    stop = round(price * 0.90, 2)
    target = round(price * 1.25, 2)

    if low1m > 0 and (price / low1m - 1) * 100 < 8:
        target = round(price * 1.50, 2)

    return score, reasons, target, stop

def process_signals(raw_data, bullish):
    if not raw_data:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    intra, swing, long = [], [], []

    for item in raw_data:
        d = item["d"]
        if len(d) < 21 or d[0] is None:
            continue

        sym = d[0]
        price = _safe(d[1])
        change = _safe(d[2])
        vol = _safe(d[3])
        rv = _safe(d[4])
        avg_vol = _safe(d[5])
        rsi = _safe(d[6])
        macd = _safe(d[7])
        macd_sig = _safe(d[8])
        bb_low = _safe(d[9])
        bb_high = _safe(d[10])
        ema20 = _safe(d[11])
        ema50 = _safe(d[12])
        change1w = _safe(d[13])
        high1m = _safe(d[14])
        low1m = _safe(d[15])
        vwap = _safe(d[16]) or price
        ema10 = _safe(d[17])
        adx = _safe(d[18])
        atr = _safe(d[19])
        stoch = _safe(d[20])

        sector = SYMBOL_TO_SECTOR.get(sym, "Other")

        if vol < CONFIG["MIN_LIQUIDITY"]:
            continue

        # INTRADAY
        score, reasons, target, stop = analyze_intraday(
            price, change, rv, rsi, macd, macd_sig, ema10, vwap, adx, stoch, vol, avg_vol, atr, bullish
        )

        if score >= 8:
            denom = price - stop
            rr = (target - price) / denom if denom > 0 else 0

            intra.append({
                "Symbol": sym, "Sector": sector, "Price": round(price, 2),
                "Chg%": round(change, 2), "RV": round(rv, 2), "RSI": round(rsi, 1),
                "Score": score, "Setup": " · ".join(reasons[:3]),
                "Target": target, "Stop": stop,
                "R:R": f"1:{rr:.1f}" if rr > 0 else "—"
            })

        # SWING
        score, reasons, target, stop = analyze_swing(
            price, change, rv, rsi, macd, macd_sig, ema10, ema20, ema50, change1w, low1m, adx, atr, high1m, vol, avg_vol, bullish
        )

        if score >= 9:
            denom = price - stop
            rr = (target - price) / denom if denom > 0 else 0

            swing.append({
                "Symbol": sym, "Sector": sector, "Price": round(price, 2),
                "Chg%": round(change, 2), "1W%": round(change1w, 2), "RSI": round(rsi, 1),
                "Score": score, "Setup": " · ".join(reasons[:3]),
                "Target": target, "Stop": stop,
                "R:R": f"1:{rr:.1f}" if rr > 0 else "—"
            })

        # LONG-TERM
        perf1m = (price / low1m - 1) * 100 if low1m > 0 else 0
        score, reasons, target, stop = analyze_longterm(
            price, rsi, ema20, ema50, change1w, low1m, high1m, sector, vol, avg_vol
        )

        if score >= 8:
            denom = price - stop
            rr = (target - price) / denom if denom > 0 else 0

            long.append({
                "Symbol": sym, "Sector": sector, "Price": round(price, 2),
                "1W%": round(change1w, 2), "1M%": round(perf1m, 2), "RSI": round(rsi, 1),
                "Score": score, "Setup": " · ".join(reasons[:3]),
                "Target": target, "Stop": stop,
                "R:R": f"1:{rr:.1f}" if rr > 0 else "—"
            })

    df_intra = pd.DataFrame(intra).sort_values("Score", ascending=False) if intra else pd.DataFrame()
    df_swing = pd.DataFrame(swing).sort_values("Score", ascending=False) if swing else pd.DataFrame()
    df_long = pd.DataFrame(long).sort_values("Score", ascending=False) if long else pd.DataFrame()

    return df_intra, df_swing, df_long

def get_position_status(symbol, entry, raw_data):
    for item in raw_data:
        if item["d"][0] == symbol:
            d = item["d"]
            break
    else:
        return None

    curr = _safe(d[1])
    rv = _safe(d[4])
    rsi = _safe(d[6])
    macd = _safe(d[7])
    macd_sig = _safe(d[8])
    ema20 = _safe(d[11])
    ema10 = _safe(d[17])
    adx = _safe(d[18])

    pnl_pct = ((curr - entry) / entry) * 100
    pnl_amt = curr - entry

    signals = []
    action = "🟢 HOLD"
    conf = "Medium"

    if pnl_pct > 12:
        signals.append("🎯 Excellent profit - book 75%")
        action = "🟡 SCALE OUT"
        conf = "High"
    elif pnl_pct > 8:
        signals.append("💰 Strong profit - book 50%")
        action = "🟡 PARTIAL EXIT"
        conf = "High"
    elif pnl_pct > 5 and rsi > 70:
        signals.append("⚠️ Profit + overbought")
        action = "🟡 PARTIAL EXIT"

    if curr < ema20 and curr < ema10:
        signals.append("❌ Broke EMAs - trend dead")
        action = "🔴 EXIT"
        conf = "High"
    elif curr < ema20 * 0.98 and pnl_pct < 2:
        signals.append("⚠️ Below EMA20")
        if pnl_pct < 0:
            action = "🔴 EXIT"

    if macd < macd_sig and adx < 20 and pnl_pct > 3:
        signals.append("📉 MACD bear + weak ADX")
        action = "🟡 PARTIAL EXIT"

    if rsi > 75 and rv < 0.8:
        signals.append("🚨 RSI extreme + volume fade")
        action = "🔴 EXIT"
        conf = "High"

    if pnl_pct < -8:
        signals.append("🛑 STOP LOSS - cut now")
        action = "🔴 EXIT NOW"
        conf = "Critical"
    elif pnl_pct < -5 and adx < 18:
        signals.append("⚠️ Loss + no trend")
        action = "🔴 EXIT"

    if not signals or (pnl_pct > 0 and curr > ema20):
        if adx > 25 and rsi < 70:
            signals.append("✅ Strong trend - hold")
            conf = "Good"
        elif pnl_pct > 0:
            signals.append("✅ Profit intact - monitor")
        else:
            signals.append("⏳ Developing position")

    return {
        "symbol": symbol, "entry": entry, "current": curr,
        "pnl_pct": pnl_pct, "pnl_amount": pnl_amt,
        "rsi": rsi, "rv": rv, "adx": adx,
        "action": action, "signals": signals, "confidence": conf
    }

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="PSX Pro Scanner", layout="wide", initial_sidebar_state="collapsed")

if 'positions' not in st.session_state:
    st.session_state.positions = []

# ──────────────────────────────────────────────
# STYLES
# ──────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

* { font-family: 'Inter', -apple-system, sans-serif; }
html, body, [class*="css"] { background: #0a0e1a !important; color: #e2e8f0 !important; }
code, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

.pro-header {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
}

.pro-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #00ff88 0%, #00ccff 50%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem;
}

.pro-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #64748b;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
}

.badge {
    background: #0f172a;
    border: 1px solid #1e293b;
    padding: 0.4rem 0.9rem;
    border-radius: 8px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}

.market-open { border-color: #10b981; color: #10b981; background: rgba(16, 185, 129, 0.05); }
.market-closed { border-color: #ef4444; color: #ef4444; background: rgba(239, 68, 68, 0.05); }
.breadth-bull { border-color: #10b981; color: #10b981; background: rgba(16, 185, 129, 0.1); }
.breadth-bear { border-color: #ef4444; color: #ef4444; background: rgba(239, 68, 68, 0.1); }

.sector-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 1rem;
}

.sector-mini-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    padding: 5px 10px;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: all 0.2s;
}
.sector-mini-card:hover { background: rgba(51, 65, 85, 0.6); border-color: rgba(255, 255, 255, 0.2); }
.sector-mini-name { font-size: 0.75rem; font-weight: 500; color: #94a3b8; }
.sector-mini-val { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 700; }

.strategy-card {
    background: linear-gradient(160deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.7) 100%);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 2rem;
}

.strategy-header {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.strategy-desc {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 1.25rem;
}

.position-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: all 0.2s;
}

.position-card:hover { border-color: #475569; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2); }
.pnl-pos { color: #10b981 !important; font-weight: 700; }
.pnl-neg { color: #ef4444 !important; font-weight: 700; }

.conf-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 6px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}

.conf-high { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
.conf-critical { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }

[data-testid="stDataFrame"] { background: transparent !important; }
thead tr th {
    background: rgba(30, 41, 59, 0.6) !important;
    color: #94a3b8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase;
}

tbody tr { border-bottom: 1px solid rgba(30, 41, 59, 0.3) !important; }
tbody tr:hover { background: rgba(30, 41, 59, 0.2) !important; }

.stButton>button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.7rem 1.5rem;
    font-weight: 700;
    font-size: 0.9rem;
    transition: all 0.2s;
}

.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4); }

.stTextInput>div>div>input, .stNumberInput>div>div>input {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(51, 65, 85, 0.5) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
}

hr { border-color: rgba(30, 41, 59, 0.5) !important; margin: 2rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────

is_open = is_market_open()
now_str = pkt_now().strftime("%H:%M PKT · %a %d %b %Y")

avg_chg, bullish, adv, dec, kse_info = fetch_breadth()
breadth_class = "breadth-bull" if bullish else "breadth-bear"
breadth_text = f"BULLISH {avg_chg:+.2f}%" if bullish else f"BEARISH {avg_chg:+.2f}%"
status_class = "market-open" if is_open else "market-closed"
status_text = "● LIVE" if is_open else "● CLOSED"

st.markdown(f'''<div class="pro-header">
<div class="pro-title">⚡ PSX Professional Scanner</div>
<div class="pro-subtitle">
<span>Institutional Grade Equity Intelligence</span>
<span style="color: #64748b; margin-left: auto; font-family: 'JetBrains Mono', monospace;">{now_str}</span>
</div>
</div>''', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# KSE100 DETAILED SUMMARY (SARMAAYA DATA)
# ──────────────────────────────────────────────

sarmaaya_data = fetch_sarmaaya_index()

# Use Sarmaaya as primary, fallback to TV breadth fetch if Sarmaaya is down
idx_close = sarmaaya_data.get('close', kse_info['close']) if sarmaaya_data else kse_info['close']
idx_change_abs = sarmaaya_data.get('change', 0) if sarmaaya_data else 0
idx_pct = sarmaaya_data.get('changePercent', kse_info['change']) if sarmaaya_data else kse_info['change']
idx_high = sarmaaya_data.get('high', kse_info['high']) if sarmaaya_data else kse_info['high']
idx_low = sarmaaya_data.get('low', kse_info['low']) if sarmaaya_data else kse_info['low']
idx_vol = sarmaaya_data.get('volume', kse_info['volume']) if sarmaaya_data else kse_info['volume']
vol_cr = idx_vol / 10_000_000

idx_color = "#10b981" if idx_pct >= 0 else "#ef4444"

st.markdown(f'''<div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 1.25rem; margin-bottom: 2rem;">
<div style="display: flex; gap: 10px; margin-bottom: 1.25rem; align-items: center; flex-wrap: wrap;">
<span class="badge {status_class}">{status_text}</span>
<span class="badge {breadth_class}">{breadth_text}</span>
<div style="display: flex; gap: 15px; margin-left: 10px; padding-left: 15px; border-left: 1px solid rgba(255,255,255,0.1);">
<span style="color: #10b981; font-weight: 800; font-family: monospace; font-size: 0.85rem;">▲ {adv}</span>
<span style="color: #ef4444; font-weight: 800; font-family: monospace; font-size: 0.85rem;">▼ {dec}</span>
</div>
</div>
<div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 20px;">
<div>
<div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; margin-bottom: 4px;">KSE100 Index</div>
<div style="font-size: 2.2rem; font-weight: 800; color: #fff; line-height: 1;">{idx_close:,.2f}</div>
<div style="font-size: 1.1rem; font-weight: 700; color: {idx_color}; margin-top: 5px;">
{f"{idx_change_abs:+,.2f} " if idx_change_abs != 0 else ""}({idx_pct:+.2f}%)
</div>
</div>
<div style="display: flex; gap: 30px; border-left: 1px solid rgba(255,255,255,0.1); padding-left: 30px;">
<div><div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase; margin-bottom: 2px;">High</div><div style="font-weight: 700; color: #e2e8f0; font-family: 'JetBrains Mono';">{idx_high:,.0f}</div></div>
<div><div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase; margin-bottom: 2px;">Low</div><div style="font-weight: 700; color: #e2e8f0; font-family: 'JetBrains Mono';">{idx_low:,.0f}</div></div>
<div><div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase; margin-bottom: 2px;">Volume</div><div style="font-weight: 700; color: #e2e8f0; font-family: 'JetBrains Mono';">{vol_cr:.2f} Cr</div></div>
</div>
</div>
</div>''', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SCAN
# ──────────────────────────────────────────────

scan = is_open

if not is_open:
    if st.button("🔍 Manual Scan", use_container_width=True):
        scan = True

if scan:
    with st.spinner("Scanning market..."):
        raw = fetch_market_data()

        # Sector performance
        sec_perf = {}
        for item in raw:
            sym = item["d"][0]
            chg = _safe(item["d"][2])
            sec = SYMBOL_TO_SECTOR.get(sym)
            if sec:
                if sec not in sec_perf: sec_perf[sec] = []
                sec_perf[sec].append(chg)

        st.markdown("##### 🏗️ Sector Performance")
        sector_names = list(SECTORS.keys())
        sector_html = '<div class="sector-wrap">'
        for sec in sector_names:
            changes = sec_perf.get(sec, [0])
            avg_chg = sum(changes) / len(changes)
            color = "#10b981" if avg_chg >= 0 else "#ef4444"
            safe_name = html.escape(sec)
            sector_html += f"""
            <div class="sector-mini-card">
                <span class="sector-mini-name">{safe_name}</span>
                <span class="sector-mini-val" style="color: {color};">{avg_chg:+.1f}%</span>
            </div>"""
        sector_html += "</div>"
        st.markdown(sector_html, unsafe_allow_html=True)
        st.write("")

        df_intra, df_swing, df_long = process_signals(raw, bullish)

    # INTRADAY
    st.markdown("""
    <div class="strategy-card">
        <div class="strategy-header">
            ⚡ <span style="color: #fbbf24;">INTRADAY</span> SCALPS
        </div>
        <div class="strategy-desc">
            Same-day exits · ATR stops · VWAP edge · Score ≥8/15
        </div>
    """, unsafe_allow_html=True)

    if df_intra.empty:
        st.info("🔍 No elite setups")
    else:
        st.dataframe(
            df_intra,
            column_config={
                "Chg%": st.column_config.NumberColumn("Chg%", format="%.2f%%"),
                "RV": st.column_config.NumberColumn("RV", format="%.1fx"),
                "RSI": st.column_config.NumberColumn("RSI", format="%.0f"),
                "Target": st.column_config.NumberColumn("Target", format="%.2f"),
                "Stop": st.column_config.NumberColumn("Stop", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # SWING
    st.markdown("""
    <div class="strategy-card">
        <div class="strategy-header">
            🚀 <span style="color: #3b82f6;">SWING</span> TRADES
        </div>
        <div class="strategy-desc">
            3-7 day holds · EMA fan · Value zones · Score ≥9/18
        </div>
    """, unsafe_allow_html=True)

    if df_swing.empty:
        st.info("🔍 No elite setups")
    else:
        st.dataframe(
            df_swing,
            column_config={
                "Chg%": st.column_config.NumberColumn("Chg%", format="%.2f%%"),
                "1W%": st.column_config.NumberColumn("1W%", format="%.2f%%"),
                "RSI": st.column_config.NumberColumn("RSI", format="%.0f"),
                "Target": st.column_config.NumberColumn("Target", format="%.2f"),
                "Stop": st.column_config.NumberColumn("Stop", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # LONG-TERM
    st.markdown("""
    <div class="strategy-card">
        <div class="strategy-header">
            💎 <span style="color: #10b981;">LONG-TERM</span> INVESTMENTS
        </div>
        <div class="strategy-desc">
            Multi-week · Quality sectors · Double bottoms · Score ≥8/16
        </div>
    """, unsafe_allow_html=True)

    if df_long.empty:
        st.info("🔍 No elite setups")
    else:
        st.dataframe(
            df_long,
            column_config={
                "1W%": st.column_config.NumberColumn("1W%", format="%.2f%%"),
                "1M%": st.column_config.NumberColumn("1M%", format="%.2f%%"),
                "RSI": st.column_config.NumberColumn("RSI", format="%.0f"),
                "Target": st.column_config.NumberColumn("Target", format="%.2f"),
                "Stop": st.column_config.NumberColumn("Stop", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# POSITION TRACKER (Moved to bottom)
# ──────────────────────────────────────────────

st.divider()
st.markdown("### 📊 Position Tracker")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    symbol_in = st.text_input("🔍 Symbol", placeholder="e.g., HBL, LUCK, PPL", key="sym").upper()
with col2:
    entry_in = st.number_input("Entry Price", min_value=0.0, step=0.01, key="entry")
with col3:
    st.write("")
    st.write("")
    if st.button("➕ Add", use_container_width=True) and symbol_in and entry_in > 0:
        if symbol_in in KSE100_SYMBOLS:
            st.session_state.positions.append({"symbol": symbol_in, "entry": entry_in})
            st.success(f"✓ Added {symbol_in} @ {entry_in}")
            st.rerun()
        else:
            st.error(f"Symbol {symbol_in} not in KSE100")

if st.session_state.positions:
    raw = fetch_market_data()

    for idx, pos in enumerate(st.session_state.positions):
        status = get_position_status(pos["symbol"], pos["entry"], raw)

        if status:
            pnl_class = "pnl-pos" if status["pnl_pct"] > 0 else "pnl-neg"
            conf_class = f"conf-{status['confidence'].lower()}"

            cola, colb, colc = st.columns([2, 4, 1])

            with cola:
                st.markdown(f"""
                <div class="position-card">
                    <div style="font-size: 1.5rem; font-weight: 800; margin-bottom: 0.4rem; font-family: 'JetBrains Mono', monospace;">
                        {status['symbol']}
                    </div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 0.75rem;">
                        Entry: {status['entry']:.2f} → {status['current']:.2f}
                    </div>
                    <div style="font-size: 1.3rem; font-weight: 800; font-family: 'JetBrains Mono', monospace;" class="{pnl_class}">
                        {status['pnl_pct']:+.2f}%
                    </div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.25rem;">
                        PKR {status['pnl_amount']:+.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with colb:
                st.markdown(f"""
                <div class="position-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <div style="font-size: 1.05rem; font-weight: 700;">
                            {status['action']}
                        </div>
                        <span class="conf-badge {conf_class}">{status['confidence']}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.7;">
                        {'<br>'.join(['• ' + s for s in status['signals'][:3]])}
                    </div>
                    <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid rgba(51, 65, 85, 0.5); font-size: 0.75rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">
                        RSI: {status['rsi']:.0f} · RV: {status['rv']:.1f}x · ADX: {status['adx']:.0f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with colc:
                st.write("")
                st.write("")
                if st.button("🗑️", key=f"del_{idx}", use_container_width=True):
                    st.session_state.positions.pop(idx)
                    st.rerun()

# ──────────────────────────────────────────────
# BEGINNER PLAYBOOK
# ──────────────────────────────────────────────
st.divider()
st.markdown("### 🎓 Professional Trading Playbook")
col_p1, col_p2 = st.columns(2)
with col_p1:
    st.info("""
    **📈 Strategy Specifics & Scores:**

    **1. Intraday Scalps (Score ≥ 8):**
    *   **Goal:** Capture quick, small moves (Target ~1.5%) for same-day exits.
    *   **Entry:** Use 1-minute charts for precise timing.
    *   **Key Indicators:**
        *   **Volume Surge (3x+ RV):** Indicates strong institutional interest.
        *   **VWAP Edge:** Price trading just above VWAP (Volume Weighted Average Price).
        *   **EMA10:** Price above 10-period Exponential Moving Average.
        *   **ADX (25+):** Confirms a strong trend is developing.
        *   **RSI (50-65):** Healthy momentum, not overbought.
        *   **MACD:** Bullish crossover.

    **2. Swing Trades (Score ≥ 9):**
    *   **Goal:** Hold for 3-5 sessions, aiming for ~7% gains.
    *   **Entry:** Look for consolidation after a move, or a pullback to support.
    *   **Key Indicators:**
        *   **EMA Fan (EMA5 > EMA10 > EMA20 > EMA50):** Strong, aligned uptrend.
        *   **ADX (25+):** Sustained trend strength.
        *   **Value Zone:** Stock in the lower to mid-range of its 1-month price.
        *   **Relative Volume (1.5x+):** Confirms buying interest.
        *   **RSI (45-60):** Healthy momentum, not overextended.
        *   **Weekly Momentum:** Positive change over the last week.

    **3. Long-Term Investments (Score ≥ 8):**
    *   **Goal:** Multi-week to multi-month holds, targeting 25%+ returns.
    *   **Focus:** High-quality sectors like Banks and E&P.
    *   **Key Indicators:**
        *   **Sector Quality:** Higher quality sectors (rated 7-9) are preferred.
        *   **Value Zone:** Stock trading near 1-month lows or showing "Double Bottom" patterns.
        *   **RSI (30-45):** Indicates potential for reversal from oversold conditions.
        *   **EMA50:** Price above 50-period EMA for long-term trend confirmation.
        *   **Weekly Stability:** Not in a significant weekly drawdown.

    **Market Breadth:** Always ensure **Advancers > Decliners** before initiating any new trades.
    """)
with col_p2:
    st.info("""
    **🛡️ Risk Management:**
    *   **Intraday Risk:** Max 1% of your trading capital per trade. Use **ATR (Average True Range) stops** to place stops based on volatility, avoiding "noise" and premature exits.
    *   **Swing Risk:** Max 3-4% of your trading capital per trade. **Trail your stop loss** to your entry price once the stock is up 3% to protect capital.
    *   **Portfolio Rule:** Never allocate more than 15% of your total cash to a single stock. Diversification is key.
    *   **The 2% Rule:** The total potential loss from all open trades should not exceed 2% of your total portfolio value. This is your ultimate safety net.
    """)

# Footer
st.markdown(f"""
<div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(30, 41, 59, 0.3);
            font-size: 0.75rem; color: #475569; font-family: 'JetBrains Mono', monospace; text-align: center;">
    Institutional algorithms · ATR stops · Market breadth filter · Last: {pkt_now().strftime("%H:%M:%S")}
</div>
""", unsafe_allow_html=True)

# AUTO REFRESH
if is_open:
    import time
    time.sleep(CONFIG["REFRESH_RATE"])
    st.rerun()
