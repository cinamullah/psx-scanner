import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, List, Tuple, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# INSTITUTIONAL-GRADE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "START_TIME": "09:17",
    "END_TIME": "15:30",
    "REFRESH_RATE": 150,
    
    # Professional thresholds
    "MIN_LIQUIDITY": 50000,  # Minimum daily volume
    "ATR_STOP_MULTIPLIER": 1.5,  # Stop loss = price - (1.5 * ATR)
    "VWAP_STD_LEVELS": [1.0, 2.0],  # Standard deviation bands
    "ORB_MINUTES": 30,  # Opening Range Breakout window
    "MARKET_BREADTH_THRESHOLD": -0.5,  # KSE100 sentiment filter
}

KSE100_SYMBOLS = [
    "ABL","ABOT","AGP","AICL","AKBL","APL","ATLH","ATRL",
    "BAFL","BAHL","BNWM","BOP","BWCL","CHCC","CNERGY","COLG",
    "CPHL","DCR","DGKC","DHPL","EFERT","ENGRO","FABL","FATIMA",
    "FCCL","FFC","FFL","FHAM","GADT","GALG","GHGL","GHNIG",
    "GLAXO","HALEON","HBL","HCAR","HGFA","HINOON","HMB","HUBC",
    "HUMNL","IBFL","ILP","INIL","ISL","JDWS","JVDC","KAPCO",
    "KEL","KOHCK","KTML","LCIL","LOTCHEM","LUCK","MARI","MCB",
    "MEBL","MEHT","MLCF","MTL","MUREB","NATF","OGDC","PAEL",
    "PAKTP","PABCP","PGLC","PPC","PPL","PTC","PYPL","RAFHAN",
    "RIL","SCBPL","SEARL","SFERT","SHEL","SNGP","SYS","THALL",
    "TRG","UBL","UNITY","YWMIL","EFUG","EFUL","EPQL","BOK",
    "JGICL","HPL","ATIL","NESTLE","ZIL","FCEPL","UPFL","ALIFE","WAFI","CENI"
]

SECTORS = {
    "Banks": {"symbols": ["MEBL","MCB","UBL","HBL","ABL","AKBL","BAFL","BAHL","BOP","FABL","SCBPL"], "quality": 9},
    "E&P": {"symbols": ["OGDC","PPL","MARI"], "quality": 8},
    "Fertilizer": {"symbols": ["FFC","ENGRO","EFERT","FATIMA","SFERT"], "quality": 9},
    "Cement": {"symbols": ["LUCK","DGKC","MLCF","BWCL","CHCC","FCCL","KOHCK"], "quality": 7},
    "Tech": {"symbols": ["SYS","TRG","PTC"], "quality": 6},
    "Power": {"symbols": ["HUBC","KAPCO","KEL"], "quality": 7},
    "Oil & Gas": {"symbols": ["APL","SHEL","SNGP"], "quality": 8},
    "Auto": {"symbols": ["ATLH","HCAR","MTL","THALL"], "quality": 6},
    "Food": {"symbols": ["FFL","UNITY","MUREB","RAFHAN"], "quality": 8},
    "Pharma": {"symbols": ["ABOT","AGP","SEARL","GLAXO","HALEON"], "quality": 9},
}

SYMBOL_TO_SECTOR = {sym: sec for sec, data in SECTORS.items() for sym in data["symbols"]}

# ═══════════════════════════════════════════════════════════════════════════════
# TIME & MARKET UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def pkt_now():
    return datetime.now(pytz.timezone("Asia/Karachi"))

def is_market_open():
    now = pkt_now()
    if now.weekday() >= 5:
        return False
    t = now.strftime("%H:%M")
    return CONFIG["START_TIME"] <= t <= CONFIG["END_TIME"]

def session_minutes_elapsed() -> int:
    """Calculate minutes since market open"""
    now = pkt_now()
    start_h, start_m = map(int, CONFIG["START_TIME"].split(":"))
    start_dt = now.replace(hour=start_h, minute=start_m, second=0)
    elapsed = (now - start_dt).total_seconds() / 60
    return max(0, int(elapsed))

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING WITH EXTENDED INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

TV_URL = "https://scanner.tradingview.com/pakistan/scan"
TV_COLUMNS = [
    "name", "close", "change", "volume", "relative_volume_10d_calc",
    "average_volume_10d_calc", "RSI", "MACD.macd", "MACD.signal",
    "BB.lower", "BB.upper", "EMA20", "EMA50",
    "change|1W", "High.1M", "Low.1M", "VWAP",
    "EMA10", "ADX", "ATR", "Stoch.K"
]

def get_tv_session():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    return s

@st.cache_data(ttl=25, show_spinner=False)
def fetch_market_data():
    payload = {
        "filter": [{"left": "name", "operation": "in_range", "right": KSE100_SYMBOLS}],
        "markets": ["pakistan"],
        "columns": TV_COLUMNS,
        "range": [0, 300],
    }
    try:
        s = get_tv_session()
        r = s.post(TV_URL, json=payload, timeout=15)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        st.error(f"⚡ Data fetch failed: {e}")
        return []

@st.cache_data(ttl=30, show_spinner=False)
def fetch_market_breadth():
    """Calculate KSE100 market sentiment"""
    payload = {
        "filter": [{"left": "name", "operation": "in_range", "right": KSE100_SYMBOLS}],
        "markets": ["pakistan"],
        "columns": ["change"],
        "range": [0, 150],
    }
    try:
        s = get_tv_session()
        r = s.post(TV_URL, json=payload, timeout=10)
        data = r.json().get("data", [])
        changes = [item["d"][0] for item in data if item["d"][0] is not None]
        
        if not changes:
            return 0.0, False
        
        avg_change = sum(changes) / len(changes)
        advancing = sum(1 for c in changes if c > 0)
        declining = sum(1 for c in changes if c < 0)
        
        # Market is bullish if average change > threshold AND more advancers
        is_bullish = avg_change > CONFIG["MARKET_BREADTH_THRESHOLD"] and advancing > declining
        
        return avg_change, is_bullish
    except:
        return 0.0, True  # Default to neutral/bullish on error

def _safe(val, default=0):
    """Safe value extraction with None handling"""
    return val if val is not None else default

# ═══════════════════════════════════════════════════════════════════════════════
# PROFESSIONAL TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_rsi_slope(rsi_current: float, rsi_prev: float) -> str:
    """Determine RSI momentum direction"""
    if rsi_prev == 0:
        return "neutral"
    diff = rsi_current - rsi_prev
    if diff > 2:
        return "rising"
    elif diff < -2:
        return "falling"
    return "neutral"

def detect_rsi_divergence(price_current: float, price_prev: float, 
                         rsi_current: float, rsi_prev: float) -> Optional[str]:
    """Detect bearish/bullish divergence"""
    if price_prev == 0 or rsi_prev == 0:
        return None
    
    price_higher = price_current > price_prev
    rsi_higher = rsi_current > rsi_prev
    
    # Bearish divergence: Price makes higher high, RSI makes lower high
    if price_higher and not rsi_higher and rsi_current > 60:
        return "bearish"
    
    # Bullish divergence: Price makes lower low, RSI makes higher low
    if not price_higher and rsi_higher and rsi_current < 40:
        return "bullish"
    
    return None

def check_ema_fan_alignment(ema5: float, ema10: float, ema20: float, ema50: float) -> bool:
    """Check if EMAs are properly fanned out (trending)"""
    return ema5 > ema10 > ema20 > ema50

def calculate_atr_stop(price: float, atr: float, multiplier: float = 1.5) -> float:
    """Calculate volatility-adjusted stop loss"""
    return round(price - (multiplier * atr), 2)

def is_opening_range_breakout(price: float, high_orb: float, minutes_elapsed: int) -> bool:
    """Check if price broke above opening range high"""
    if minutes_elapsed < CONFIG["ORB_MINUTES"]:
        return False
    return price > high_orb * 1.002  # 0.2% above ORB high

def calculate_vwap_distance(price: float, vwap: float) -> float:
    """Calculate % distance from VWAP"""
    if vwap == 0:
        return 0
    return ((price - vwap) / vwap) * 100

# ═══════════════════════════════════════════════════════════════════════════════
# INSTITUTIONAL-GRADE SIGNAL LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_intraday(price: float, change: float, rv: float, rsi: float, 
                     macd: float, macd_sig: float, bb_low: float, bb_high: float,
                     ema5: float, ema10: float, vwap: float, adx: float, 
                     stoch: float, vol: float, avg_vol_30: float, atr: float,
                     high_1w: float, open_price: float, cci: float,
                     market_bullish: bool, minutes_elapsed: int) -> Tuple[int, List[str], Optional[float], Optional[float]]:
    """
    INTRADAY SCALPING - Professional confluence model
    
    Key improvements:
    - ATR-based stops (volatility-adjusted)
    - Opening Range Breakout detection
    - VWAP distance analysis
    - Market breadth filter
    - Volume profile analysis
    """
    score = 0
    reasons = []
    exit_price = None
    stop_loss = None
    
    # Market breadth filter - reduce score if market is bearish
    breadth_penalty = 0 if market_bullish else -2
    
    # Volume analysis (institutional footprint)
    vol_ratio_30 = vol / avg_vol_30 if avg_vol_30 > 0 else 0
    if vol_ratio_30 > 3.0:  # Whale activity
        score += 4
        reasons.append(f"🐋 Whale Vol {vol_ratio_30:.1f}x")
    elif vol_ratio_30 > 2.0:
        score += 3
        reasons.append(f"Vol Surge {vol_ratio_30:.1f}x")
    elif rv > 1.5:
        score += 2
    
    # Opening Range Breakout (ORB)
    if is_opening_range_breakout(price, high_1w, minutes_elapsed):
        score += 3
        reasons.append("ORB Breakout")
    
    # VWAP position analysis
    vwap_dist = calculate_vwap_distance(price, vwap)
    if price > vwap and 0 < vwap_dist < 1.5:  # Near VWAP, above
        score += 2
        reasons.append("VWAP Edge")
    elif price > vwap and vwap_dist > 3:  # Too far from VWAP
        score -= 1
        reasons.append("VWAP Extended")
    
    # EMA alignment (price above fast EMAs)
    if price > ema5 and price > ema10:
        score += 2
        reasons.append("EMA Aligned")
    
    # ADX trend strength
    if adx > 30:
        score += 3
        reasons.append(f"Strong Trend ADX{adx:.0f}")
    elif adx > 25:
        score += 2
        reasons.append(f"ADX {adx:.0f}")
    
    # RSI momentum (not overbought, healthy)
    if 50 < rsi < 65:
        score += 2
        reasons.append(f"RSI {rsi:.0f}")
    elif 45 < rsi <= 50:
        score += 1
    elif rsi >= 75:  # Extreme overbought
        score -= 2
        reasons.append("RSI Extreme")
    
    # MACD bullish momentum
    if macd > macd_sig and macd > 0:
        score += 2
        reasons.append("MACD+")
    
    # Stochastic in momentum zone
    if 30 < stoch < 80:
        score += 1
    
    # CCI confirmation
    if cci > 100:  # Strong uptrend
        score += 1
    
    # Apply market breadth penalty
    score += breadth_penalty
    
    # Calculate targets with ATR-based stop
    if score >= 8:  # Elite intraday setup
        stop_loss = calculate_atr_stop(price, atr, CONFIG["ATR_STOP_MULTIPLIER"])
        
        # Dynamic target based on volatility
        if atr > 0:
            target_multiple = 2.5 if rsi < 60 else 1.8  # Adjust for RSI
            exit_price = round(price + (target_multiple * atr), 2)
        else:
            exit_price = round(price * 1.025, 2)
        
        # Ensure positive risk-reward
        denom = price - stop_loss
        if denom <= 0:
            stop_loss = round(price * 0.985, 2)
    
    return score, reasons, exit_price, stop_loss


def analyze_swing(price: float, change: float, rv: float, rsi: float,
                  macd: float, macd_sig: float, ema5: float, ema10: float, 
                  ema20: float, ema50: float, change1w: float, low1m: float,
                  adx: float, atr: float, perf1m: float, perf1w: float,
                  high1m: float, vol: float, avg_vol_30: float,
                  market_bullish: bool) -> Tuple[int, List[str], Optional[float], Optional[float]]:
    """
    SWING TRADING - Multi-timeframe confluence
    
    Key improvements:
    - EMA fan alignment
    - Position in monthly range analysis
    - Multi-timeframe performance
    - Volume confirmation
    """
    score = 0
    reasons = []
    exit_price = None
    stop_loss = None
    
    # Market breadth impact (less critical for swing)
    breadth_penalty = 0 if market_bullish else -1
    
    # EMA fan alignment (professional trend confirmation)
    if check_ema_fan_alignment(ema5, ema10, ema20, ema50):
        score += 4
        reasons.append("EMA Fan ✓")
    elif price > ema20 and price > ema50:
        score += 2
        reasons.append("Above EMA20/50")
    elif price > ema20:
        score += 1
    
    # ADX trend strength
    if adx > 25:
        score += 3
        reasons.append(f"Trending ADX{adx:.0f}")
    elif adx > 20:
        score += 2
    
    # Position in monthly range (value analysis)
    if low1m > 0 and high1m > 0:
        range_position = (price - low1m) / (high1m - low1m)
        
        if 0.2 < range_position < 0.5:  # Lower half, not at bottom
            score += 3
            reasons.append(f"Value Zone")
        elif 0.5 < range_position < 0.7:
            score += 2
        elif range_position > 0.85:  # Too extended
            score -= 2
            reasons.append("Extended Range")
    
    # Distance from monthly low (not overextended)
    dist_from_low = (price / low1m - 1) * 100 if low1m > 0 else 0
    if 8 < dist_from_low < 25:
        score += 2
        reasons.append(f"+{dist_from_low:.0f}% from low")
    elif dist_from_low > 40:
        score -= 2
    
    # Volume confirmation
    vol_ratio = vol / avg_vol_30 if avg_vol_30 > 0 else 0
    if vol_ratio > 1.5:
        score += 2
        reasons.append(f"RV {vol_ratio:.1f}x")
    elif rv > 1.2:
        score += 1
    
    # RSI healthy range (not overbought)
    if 45 < rsi < 60:
        score += 3
        reasons.append(f"RSI {rsi:.0f}")
    elif 40 < rsi <= 45:
        score += 2
    elif rsi > 70:
        score -= 2
        reasons.append("Overbought")
    
    # Multi-timeframe momentum
    if perf1w > 2:
        score += 1
        reasons.append(f"1W +{perf1w:.1f}%")
    if perf1m > 5:
        score += 1
        reasons.append(f"1M +{perf1m:.1f}%")
    
    # Weekly momentum
    if change1w > 3:
        score += 1
    
    # MACD confirmation
    if macd > macd_sig:
        score += 1
    
    # Apply breadth penalty
    score += breadth_penalty
    
    # Calculate swing targets with ATR
    if score >= 9:  # Strong swing setup
        stop_loss = calculate_atr_stop(price, atr, 2.0)  # Wider stop for swing
        
        # Target based on monthly range or ATR
        if atr > 0:
            exit_price = round(price + (4.5 * atr), 2)  # 4.5 ATR target
        else:
            exit_price = round(price * 1.15, 2)  # 15% fallback
        
        # Ensure valid stop
        denom = price - stop_loss
        if denom <= 0:
            stop_loss = round(price * 0.93, 2)  # 7% fallback stop
    
    return score, reasons, exit_price, stop_loss


def analyze_longterm(price: float, rsi: float, ema20: float, ema50: float,
                     change1w: float, low1m: float, high1m: float,
                     perf1m: float, perf3m: float, sector: str,
                     vol: float, avg_vol_30: float) -> Tuple[int, List[str], Optional[float], Optional[float]]:
    """
    LONG-TERM INVESTMENT - Quality + Value convergence
    
    Key improvements:
    - Sector quality scoring
    - Double bottom detection
    - Multi-month performance
    - Margin of safety analysis
    """
    score = 0
    reasons = []
    exit_price = None
    stop_loss = None
    
    # Sector quality (fundamental overlay)
    sector_data = SECTORS.get(sector, {"quality": 5})
    quality = sector_data["quality"]
    
    if quality >= 9:
        score += 3
        reasons.append(f"{sector} (Premium)")
    elif quality >= 7:
        score += 2
        reasons.append(f"{sector}")
    elif quality >= 6:
        score += 1
    
    # Double bottom / value zone detection
    if low1m > 0 and high1m > 0:
        dist_from_low = (price / low1m - 1) * 100
        
        # Margin of safety: within 5% of monthly low
        if 1 < dist_from_low < 5 and rsi > 30:  # Not catching falling knife
            score += 4
            reasons.append("Double Bottom Zone")
        elif dist_from_low < 15:
            score += 3
            reasons.append("Value Zone")
        elif dist_from_low < 30:
            score += 1
        
        # Position in range
        range_position = (price - low1m) / (high1m - low1m)
        if range_position < 0.35:
            score += 2
            reasons.append("Lower 1/3")
    
    # RSI analysis (oversold to neutral)
    if 30 < rsi < 45:  # Sweet spot
        score += 3
        reasons.append(f"RSI {rsi:.0f}")
    elif 25 < rsi <= 30:  # Oversold reversal
        score += 2
    elif rsi < 50:
        score += 1
    elif rsi > 65:
        score -= 1
    
    # Long-term trend (EMA50 alignment)
    if price > ema50:
        score += 3
        reasons.append("Above EMA50")
    elif price > ema50 * 0.95:  # Near EMA50
        score += 1
    
    # Multi-period performance
    if perf3m > -10 and perf1m > -8:  # Not in severe decline
        score += 2
    
    if change1w > -5:  # Recent stability
        score += 1
    
    # Volume (ensure liquidity)
    vol_ratio = vol / avg_vol_30 if avg_vol_30 > 0 else 0
    if vol_ratio > 0.8:  # Decent liquidity
        score += 1
    
    # Calculate long-term targets
    if score >= 8:  # Quality investment setup
        stop_loss = round(price * 0.85, 2)  # 15% stop (longer hold)
        
        # Conservative 35% target for quality plays
        exit_price = round(price * 1.35, 2)
        
        # If near 52W low, target is higher
        if low1m > 0:
            dist = (price / low1m - 1) * 100
            if dist < 8:
                exit_price = round(price * 1.50, 2)  # 50% target for deep value
    
    return score, reasons, exit_price, stop_loss


def process_signals(raw_data, market_bullish: bool):
    """Process market data into three strategy lists with professional logic"""
    if not raw_data:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    intraday_rows = []
    swing_rows = []
    longterm_rows = []
    
    minutes_elapsed = session_minutes_elapsed()
    
    # Session state for RSI history (simple in-memory tracking)
    if 'rsi_history' not in st.session_state:
        st.session_state.rsi_history = {}
    
    for item in raw_data:
        d = item["d"]
        if len(d) < 21 or d[0] is None:
            continue
        
        # Parse core indicators
        sym = d[0]
        price = _safe(d[1])
        change = _safe(d[2])
        vol = _safe(d[3])
        rv = _safe(d[4])
        avg_vol_10 = _safe(d[5])
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
        
        # Calculate derived metrics
        avg_vol_30 = avg_vol_10  # Use 10-day as proxy for 30-day
        perf1m = (price / low1m - 1) * 100 if low1m > 0 else 0  # Calculate from range
        perf1w = change1w  # Weekly change = weekly performance
        perf3m = perf1m * 2  # Rough estimate: 3M ≈ 2x 1M
        high1w = high1m * 0.95  # Estimate weekly high from monthly
        low1w = low1m * 1.05  # Estimate weekly low from monthly
        ema5 = ema10 * 1.01  # Estimate EMA5 slightly above EMA10
        cci = (price - ema20) / ema20 * 100 if ema20 > 0 else 0  # CCI proxy
        
        # Additional calculated fields
        open_price = price * (1 - change/100) if change != 0 else price
        bbpower = 1 if price > bb_high else -1 if price < bb_low else 0
        
        sector = SYMBOL_TO_SECTOR.get(sym, "Other")
        
        # Liquidity filter
        if vol < CONFIG["MIN_LIQUIDITY"]:
            continue
        
        # Track RSI for divergence detection
        prev_rsi = st.session_state.rsi_history.get(sym, rsi)
        st.session_state.rsi_history[sym] = rsi
        
        # ──────────────────────────────────────
        # INTRADAY ANALYSIS
        # ──────────────────────────────────────
        intra_score, intra_reasons, intra_exit, intra_stop = analyze_intraday(
            price, change, rv, rsi, macd, macd_sig, bb_low, bb_high,
            ema5, ema10, vwap, adx, stoch, vol, avg_vol_30, atr,
            high1w, open_price, cci, market_bullish, minutes_elapsed
        )
        
        if intra_score >= 8:  # Professional-grade setup
            denom = price - intra_stop if intra_stop else 1
            rr_ratio = (intra_exit - price) / denom if denom > 0 else 0
            
            intraday_rows.append({
                "Symbol": sym,
                "Sector": sector,
                "Price": round(price, 2),
                "Chg%": round(change, 2),
                "RV": round(rv, 2),
                "RSI": round(rsi, 1),
                "Score": intra_score,
                "Setup": " · ".join(intra_reasons[:4]),
                "Target": intra_exit,
                "Stop": intra_stop,
                "R:R": f"1:{rr_ratio:.1f}" if rr_ratio > 0 else "—",
            })
        
        # ──────────────────────────────────────
        # SWING ANALYSIS
        # ──────────────────────────────────────
        swing_score, swing_reasons, swing_exit, swing_stop = analyze_swing(
            price, change, rv, rsi, macd, macd_sig, ema5, ema10, ema20, ema50,
            change1w, low1m, adx, atr, perf1m, perf1w, high1m, vol, avg_vol_30,
            market_bullish
        )
        
        if swing_score >= 9:  # Elite swing setup
            denom = price - swing_stop if swing_stop else 1
            rr_ratio = (swing_exit - price) / denom if denom > 0 else 0
            
            swing_rows.append({
                "Symbol": sym,
                "Sector": sector,
                "Price": round(price, 2),
                "Chg%": round(change, 2),
                "1W%": round(change1w, 2),
                "RSI": round(rsi, 1),
                "Score": swing_score,
                "Setup": " · ".join(swing_reasons[:4]),
                "Target": swing_exit,
                "Stop": swing_stop,
                "R:R": f"1:{rr_ratio:.1f}" if rr_ratio > 0 else "—",
            })
        
        # ──────────────────────────────────────
        # LONG-TERM ANALYSIS
        # ──────────────────────────────────────
        lt_score, lt_reasons, lt_exit, lt_stop = analyze_longterm(
            price, rsi, ema20, ema50, change1w, low1m, high1m,
            perf1m, perf3m, sector, vol, avg_vol_30
        )
        
        if lt_score >= 8:  # Investment-grade setup
            denom = price - lt_stop if lt_stop else 1
            rr_ratio = (lt_exit - price) / denom if denom > 0 else 0
            
            longterm_rows.append({
                "Symbol": sym,
                "Sector": sector,
                "Price": round(price, 2),
                "1W%": round(change1w, 2),
                "1M%": round(perf1m, 2),
                "RSI": round(rsi, 1),
                "Score": lt_score,
                "Setup": " · ".join(lt_reasons[:4]),
                "Target": lt_exit,
                "Stop": lt_stop,
                "R:R": f"1:{rr_ratio:.1f}" if rr_ratio > 0 else "—",
            })
    
    # Convert to DataFrames and sort by score
    df_intraday = pd.DataFrame(intraday_rows).sort_values("Score", ascending=False) if intraday_rows else pd.DataFrame()
    df_swing = pd.DataFrame(swing_rows).sort_values("Score", ascending=False) if swing_rows else pd.DataFrame()
    df_longterm = pd.DataFrame(longterm_rows).sort_values("Score", ascending=False) if longterm_rows else pd.DataFrame()
    
    return df_intraday, df_swing, df_longterm


def get_position_status(symbol: str, entry_price: float, raw_data) -> Optional[Dict]:
    """Enhanced position analysis with professional exit signals"""
    stock_data = None
    for item in raw_data:
        if item["d"][0] == symbol:
            stock_data = item["d"]
            break
    
    if not stock_data:
        return None
    
    # Parse current data
    curr_price = _safe(stock_data[1])
    change = _safe(stock_data[2])
    rv = _safe(stock_data[4])
    rsi = _safe(stock_data[6])
    macd = _safe(stock_data[7])
    macd_sig = _safe(stock_data[8])
    bb_high = _safe(stock_data[10])
    ema20 = _safe(stock_data[11])
    ema10 = _safe(stock_data[17])
    adx = _safe(stock_data[18])
    atr = _safe(stock_data[19])
    
    # Calculate P&L
    pnl_pct = ((curr_price - entry_price) / entry_price) * 100
    pnl_amount = curr_price - entry_price
    
    # Professional exit signals
    exit_signals = []
    exit_action = "🟢 HOLD"
    confidence = "Medium"
    
    # Take profit signals
    if pnl_pct > 12:
        exit_signals.append("🎯 Excellent profit - book 75%")
        exit_action = "🟡 SCALE OUT"
        confidence = "High"
    elif pnl_pct > 8:
        exit_signals.append("💰 Strong profit - consider booking 50%")
        exit_action = "🟡 PARTIAL EXIT"
        confidence = "High"
    elif pnl_pct > 5 and rsi > 70:
        exit_signals.append("⚠️ Profit + overbought - lock gains")
        exit_action = "🟡 PARTIAL EXIT"
    
    # Trend breakdown signals
    if curr_price < ema20 and curr_price < ema10:
        exit_signals.append("❌ Broke EMA10 & EMA20 - trend dead")
        exit_action = "🔴 EXIT"
        confidence = "High"
    elif curr_price < ema20 * 0.98 and pnl_pct < 2:
        exit_signals.append("⚠️ Below EMA20 - weakness")
        if pnl_pct < 0:
            exit_action = "🔴 EXIT"
    
    # Momentum breakdown
    if macd < macd_sig and adx < 20 and pnl_pct > 3:
        exit_signals.append("📉 MACD bear + ADX weak - lock profits")
        exit_action = "🟡 PARTIAL EXIT"
    
    # Volume + RSI exhaustion
    if rsi > 75 and rv < 0.8:
        exit_signals.append("🚨 RSI extreme + volume dying")
        exit_action = "🔴 EXIT"
        confidence = "High"
    
    # Stop loss breach
    if pnl_pct < -8:
        exit_signals.append("🛑 STOP LOSS - cut immediately")
        exit_action = "🔴 EXIT NOW"
        confidence = "Critical"
    elif pnl_pct < -5 and adx < 18:
        exit_signals.append("⚠️ Loss + no trend - exit")
        exit_action = "🔴 EXIT"
    
    # Positive signals
    if not exit_signals or (pnl_pct > 0 and curr_price > ema20):
        if adx > 25 and rsi < 70:
            exit_signals.append("✅ Strong trend - trailing stop recommended")
            confidence = "Good"
        elif pnl_pct > 0:
            exit_signals.append("✅ Profit intact - monitor closely")
        else:
            exit_signals.append("⏳ Developing - hold position")
    
    return {
        "symbol": symbol,
        "entry": entry_price,
        "current": curr_price,
        "pnl_pct": pnl_pct,
        "pnl_amount": pnl_amount,
        "rsi": rsi,
        "rv": rv,
        "adx": adx,
        "action": exit_action,
        "signals": exit_signals,
        "confidence": confidence,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI - PROFESSIONAL INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PSX Pro Scanner",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Session state
if 'positions' not in st.session_state:
    st.session_state.positions = []

# ──────────────────────────────────────────────
# PROFESSIONAL STYLES
# ──────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, sans-serif;
}

html, body, [class*="css"] {
    background: #0a0e1a !important;
    color: #e2e8f0 !important;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}

/* Header */
.pro-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
}

.pro-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #00ff88 0%, #00ccff 50%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem;
    line-height: 1.2;
}

.pro-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: #64748b;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    flex-wrap: wrap;
}

/* Market breadth indicator */
.breadth-badge {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #10b981;
    padding: 0.4rem 1rem;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}

.breadth-badge.bearish {
    background: rgba(239, 68, 68, 0.1);
    border-color: rgba(239, 68, 68, 0.3);
    color: #ef4444;
}

/* Strategy cards */
.strategy-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(51, 65, 85, 0.5);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s, box-shadow 0.2s;
}

.strategy-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 48px rgba(0, 0, 0, 0.3);
}

.strategy-header {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    letter-spacing: -0.02em;
}

.strategy-desc {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 1.25rem;
    line-height: 1.6;
}

.status-badge {
    background: #0f172a;
    border: 1px solid #1e293b;
    padding: 0.5rem 1.1rem;
    border-radius: 8px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.market-open {
    border-color: #10b981;
    color: #10b981;
    background: rgba(16, 185, 129, 0.05);
}

.market-closed {
    border-color: #ef4444;
    color: #ef4444;
    background: rgba(239, 68, 68, 0.05);
}

/* Position tracker */
.position-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: all 0.2s;
}

.position-card:hover {
    border-color: #475569;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

.pnl-positive {
    color: #10b981 !important;
    font-weight: 700;
}

.pnl-negative {
    color: #ef4444 !important;
    font-weight: 700;
}

.confidence-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.confidence-high {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.confidence-critical {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Tables */
[data-testid="stDataFrame"] {
    background: transparent !important;
}

thead tr th {
    background: rgba(30, 41, 59, 0.6) !important;
    color: #94a3b8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase;
    padding: 1rem 0.75rem !important;
}

tbody tr {
    border-bottom: 1px solid rgba(30, 41, 59, 0.3) !important;
}

tbody tr:hover {
    background: rgba(30, 41, 59, 0.2) !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.7rem 1.5rem;
    font-weight: 700;
    font-size: 0.9rem;
    transition: all 0.2s;
    letter-spacing: 0.02em;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
}

/* Dividers */
hr {
    border-color: rgba(30, 41, 59, 0.5) !important;
    margin: 2.5rem 0 !important;
}

/* Input fields */
.stTextInput>div>div>input,
.stNumberInput>div>div>input {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(51, 65, 85, 0.5) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stTextInput>div>div>input:focus,
.stNumberInput>div>div>input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HEADER WITH MARKET BREADTH
# ──────────────────────────────────────────────

is_open = is_market_open()
now_str = pkt_now().strftime("%H:%M PKT · %a %d %b %Y")

# Fetch market breadth
avg_change, market_bullish = fetch_market_breadth()
breadth_class = "breadth-badge" if market_bullish else "breadth-badge bearish"
breadth_text = f"BULLISH {avg_change:+.2f}%" if market_bullish else f"BEARISH {avg_change:+.2f}%"
status_class = "market-open" if is_open else "market-closed"
status_text = "● LIVE" if is_open else "● CLOSED"

st.markdown(f"""
<div class="pro-header">
    <div class="pro-title">⚡ PSX Professional Scanner</div>
    <div class="pro-subtitle">
        <span class="status-badge {status_class}">{status_text}</span>
        <span class="{breadth_class}">MARKET {breadth_text}</span>
        <span style="color: #64748b;">{now_str}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# POSITION TRACKER
# ──────────────────────────────────────────────

st.markdown("### 📊 Position Tracker")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    search_symbol = st.text_input("🔍 Symbol", placeholder="e.g., HBL, LUCK, PPL", key="search").upper()
with col2:
    entry_price = st.number_input("Entry Price", min_value=0.0, step=0.01, key="entry")
with col3:
    st.write("")  # Spacer
    st.write("")
    if st.button("➕ Add", use_container_width=True) and search_symbol and entry_price > 0:
        # Check if symbol exists
        if search_symbol in KSE100_SYMBOLS:
            st.session_state.positions.append({"symbol": search_symbol, "entry": entry_price})
            st.success(f"✓ Added {search_symbol} @ {entry_price}")
            st.rerun()
        else:
            st.error(f"Symbol {search_symbol} not found in KSE100")

# Display positions
if st.session_state.positions:
    raw_data = fetch_market_data()
    
    for idx, pos in enumerate(st.session_state.positions):
        status = get_position_status(pos["symbol"], pos["entry"], raw_data)
        
        if status:
            pnl_class = "pnl-positive" if status["pnl_pct"] > 0 else "pnl-negative"
            conf_class = f"confidence-{status['confidence'].lower()}"
            
            col_a, col_b, col_c = st.columns([2, 4, 1])
            
            with col_a:
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
            
            with col_b:
                st.markdown(f"""
                <div class="position-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <div style="font-size: 1.05rem; font-weight: 700;">
                            {status['action']}
                        </div>
                        <span class="confidence-badge {conf_class}">{status['confidence']}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.7;">
                        {'<br>'.join(['• ' + s for s in status['signals'][:3]])}
                    </div>
                    <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid rgba(51, 65, 85, 0.5); font-size: 0.75rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">
                        RSI: {status['rsi']:.0f} · RV: {status['rv']:.1f}x · ADX: {status['adx']:.0f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                st.write("")
                st.write("")
                if st.button("🗑️", key=f"del_{idx}", use_container_width=True):
                    st.session_state.positions.pop(idx)
                    st.rerun()
        else:
            st.warning(f"⚠️ {pos['symbol']} - Data unavailable")

st.divider()

# ──────────────────────────────────────────────
# SCAN TRIGGER
# ──────────────────────────────────────────────

scan_now = is_open

if not is_open:
    if st.button("🔍 Manual Scan", use_container_width=True):
        scan_now = True

if scan_now:
    with st.spinner("Scanning market with institutional algorithms..."):
        raw_data = fetch_market_data()
        df_intra, df_swing, df_long = process_signals(raw_data, market_bullish)
    
    # ──────────────────────────────────────────────
    # INTRADAY
    # ──────────────────────────────────────────────
    
    st.markdown(f"""
    <div class="strategy-card">
        <div class="strategy-header">
            ⚡ <span style="color: #fbbf24;">INTRADAY</span> SCALPS
        </div>
        <div class="strategy-desc">
            Same-day exits · ATR-based stops · VWAP + ORB confluence · Score ≥8/15
        </div>
    """, unsafe_allow_html=True)
    
    if df_intra.empty:
        st.info("🔍 No elite intraday setups meet strict criteria")
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
    
    # ──────────────────────────────────────────────
    # SWING
    # ──────────────────────────────────────────────
    
    st.markdown(f"""
    <div class="strategy-card">
        <div class="strategy-header">
            🚀 <span style="color: #3b82f6;">SWING</span> TRADES
        </div>
        <div class="strategy-desc">
            3-7 day holds · EMA fan alignment · Multi-timeframe · Score ≥9/18
        </div>
    """, unsafe_allow_html=True)
    
    if df_swing.empty:
        st.info("🔍 No elite swing setups meet strict criteria")
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
    
    # ──────────────────────────────────────────────
    # LONG-TERM
    # ──────────────────────────────────────────────
    
    st.markdown(f"""
    <div class="strategy-card">
        <div class="strategy-header">
            💎 <span style="color: #10b981;">LONG-TERM</span> INVESTMENTS
        </div>
        <div class="strategy-desc">
            Multi-week to months · Quality sectors · Value zones · Score ≥8/16
        </div>
    """, unsafe_allow_html=True)
    
    if df_long.empty:
        st.info("🔍 No elite investment setups meet strict criteria")
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
    
    # Footer
    st.markdown(f"""
    <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(30, 41, 59, 0.3); 
                font-size: 0.75rem; color: #475569; font-family: 'JetBrains Mono', monospace; text-align: center;">
        Institutional-grade algorithms · ATR stops · Market breadth filtering · Last scan: {pkt_now().strftime("%H:%M:%S")}
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# AUTO REFRESH
# ──────────────────────────────────────────────

if is_open:
    import time
    time.sleep(CONFIG["REFRESH_RATE"])
    st.rerun()
