import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import pytz
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "START_TIME": "09:17",
    "END_TIME": "15:30",
    "REFRESH_RATE": 30,
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
    "Banks": ["MEBL","MCB","UBL","HBL","ABL","AKBL","BAFL","BAHL","BOP","FABL","SCBPL"],
    "E&P": ["OGDC","PPL","MARI"],
    "Fertilizer": ["FFC","ENGRO","EFERT","FATIMA","SFERT"],
    "Cement": ["LUCK","DGKC","MLCF","BWCL","CHCC","FCCL","KOHCK"],
    "Tech": ["SYS","TRG","PTC"],
    "Power": ["HUBC","KAPCO","KEL"],
    "Oil & Gas": ["APL","SHEL","SNGP"],
    "Auto": ["ATLH","HCAR","MTL","THALL"],
    "Food": ["FFL","UNITY","MUREB","RAFHAN"],
    "Pharma": ["ABOT","AGP","SEARL","GLAXO","HALEON"],
}

SYMBOL_TO_SECTOR = {sym: sec for sec, syms in SECTORS.items() for sym in syms}

# ═══════════════════════════════════════════════════════════════════════════════
# TIME UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def pkt_now():
    return datetime.now(pytz.timezone("Asia/Karachi"))

def is_market_open():
    now = pkt_now()
    if now.weekday() >= 5:
        return False
    t = now.strftime("%H:%M")
    return CONFIG["START_TIME"] <= t <= CONFIG["END_TIME"]

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

TV_URL = "https://scanner.tradingview.com/pakistan/scan"
TV_COLUMNS = [
    "name", "close", "change", "volume", "relative_volume_10d_calc",
    "average_volume_10d_calc", "RSI", "MACD.macd", "MACD.signal",
    "BB.lower", "BB.upper", "EMA20", "EMA50", "change|1W",
    "High.1M", "Low.1M", "VWAP", "EMA10", "ADX", "ATR",
    "Stoch.K", "Perf.1M", "average_volume_30d_calc"
]

def get_tv_session():
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

def _safe(val, default=0):
    return val if val is not None else default

# ═══════════════════════════════════════════════════════════════════════════════
# WALL STREET SIGNAL LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_intraday(price, change, rv, rsi, macd, macd_sig, bb_low, bb_high, 
                     ema10, vwap, adx, stoch, vol, avg_vol_30):
    """
    INTRADAY: High probability scalps with tight stops
    Criteria: Strong momentum + volume + technical alignment
    """
    score = 0
    reasons = []
    exit_price = None
    stop_loss = None
    
    # Volume surge (critical for intraday)
    vol_ratio_30 = vol / avg_vol_30 if avg_vol_30 > 0 else 0
    if vol_ratio_30 > 2.0:
        score += 3
        reasons.append(f"Vol Surge {vol_ratio_30:.1f}x")
    elif rv > 1.5:
        score += 2
    
    # Price above VWAP and EMA10 (bullish alignment)
    if price > vwap and price > ema10:
        score += 2
        reasons.append("Above VWAP+EMA10")
    
    # Strong ADX (trending market)
    if adx > 25:
        score += 2
        reasons.append(f"ADX {adx:.0f}")
    
    # RSI momentum zone (not overbought)
    if 45 < rsi < 70:
        score += 2
        reasons.append(f"RSI {rsi:.0f}")
    elif rsi >= 70:
        score -= 1
        reasons.append("Overbought")
    
    # MACD bullish crossover
    if macd > macd_sig and macd > 0:
        score += 1
        reasons.append("MACD+")
    
    # Stochastic momentum
    if 20 < stoch < 80:
        score += 1
    
    # Calculate targets
    if score >= 6:  # Strong intraday setup
        stop_loss = round(price * 0.985, 2)  # 1.5% stop
        exit_price = round(price * 1.025, 2)  # 2.5% target (1:1.67 RR)
        if rsi > 65 or change > 3:
            exit_price = round(price * 1.015, 2)  # Tighter target if extended
    
    return score, reasons, exit_price, stop_loss


def analyze_swing(price, change, rv, rsi, macd, macd_sig, ema20, ema50, 
                  change1w, low1m, adx, atr, perf1m):
    """
    SWING: 3-7 day momentum plays with trend confirmation
    Criteria: Multi-timeframe alignment + volume + not overextended
    """
    score = 0
    reasons = []
    exit_price = None
    stop_loss = None
    
    # Price above both EMAs (uptrend)
    if price > ema20 and price > ema50:
        score += 3
        reasons.append("Above EMA20/50")
    elif price > ema20:
        score += 1
    
    # Strong ADX (sustained trend)
    if adx > 20:
        score += 2
        reasons.append(f"Trending ADX{adx:.0f}")
    
    # Not overextended from low
    dist_from_low = (price / low1m - 1) * 100 if low1m > 0 else 0
    if 5 < dist_from_low < 25:
        score += 2
        reasons.append(f"{dist_from_low:.0f}% from low")
    elif dist_from_low > 35:
        score -= 2
        reasons.append("Overextended")
    
    # Volume confirmation
    if rv > 1.2:
        score += 2
        reasons.append(f"RV {rv:.1f}x")
    
    # RSI healthy range
    if 40 < rsi < 65:
        score += 2
        reasons.append(f"RSI {rsi:.0f}")
    elif rsi > 70:
        score -= 1
    
    # Weekly momentum
    if change1w > 2:
        score += 1
        reasons.append(f"1W +{change1w:.1f}%")
    
    # Monthly performance
    if perf1m > 5:
        score += 1
    
    # Calculate swing targets
    if score >= 7:
        stop_loss = round(price * 0.95, 2)  # 5% stop
        exit_price = round(price * 1.12, 2)  # 12% target (1:2.4 RR)
        if rsi > 60:
            exit_price = round(price * 1.08, 2)  # Reduce target if RSI high
    
    return score, reasons, exit_price, stop_loss


def analyze_longterm(price, rsi, ema20, ema50, change1w, low1m, high1m, perf1m, sector):
    """
    LONG-TERM: Fundamental + technical value plays
    Criteria: Quality stocks at good prices with long-term trends
    """
    score = 0
    reasons = []
    exit_price = None
    stop_loss = None
    
    # Sector quality (defensive sectors get bonus)
    quality_sectors = ["Banks", "Fertilizer", "Pharma", "Food"]
    if sector in quality_sectors:
        score += 2
        reasons.append(f"{sector} (Quality)")
    
    # Price vs 1M range (buying dips)
    if low1m > 0 and high1m > 0:
        position_in_range = (price - low1m) / (high1m - low1m)
        if position_in_range < 0.35:  # Lower third of range
            score += 3
            reasons.append("Lower 1/3 of range")
        elif position_in_range < 0.5:
            score += 2
            reasons.append("Below mid-range")
    
    # RSI not overbought
    if rsi < 50:
        score += 2
        reasons.append(f"RSI {rsi:.0f}")
    elif rsi < 60:
        score += 1
    
    # Long-term trend (price above EMA50)
    if price > ema50:
        score += 2
        reasons.append("Above EMA50")
    
    # Not in severe drawdown
    if change1w > -8:
        score += 1
    else:
        reasons.append(f"Weak 1W {change1w:.1f}%")
    
    # Monthly performance acceptable
    if perf1m > -5:
        score += 1
    
    # Calculate long-term targets
    if score >= 6:
        stop_loss = round(price * 0.88, 2)  # 12% stop (longer hold)
        exit_price = round(price * 1.30, 2)  # 30% target (1:2.5 RR)
    
    return score, reasons, exit_price, stop_loss


def process_signals(raw_data):
    """Process raw data into three strategy lists"""
    if not raw_data:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    intraday_rows = []
    swing_rows = []
    longterm_rows = []
    
    for item in raw_data:
        d = item["d"]
        if len(d) < 23 or d[0] is None:
            continue
        
        # Parse data
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
        perf1m = _safe(d[21])
        avg_vol_30 = _safe(d[22])
        
        sector = SYMBOL_TO_SECTOR.get(sym, "Other")
        
        # Skip low liquidity stocks
        if vol < 50000:
            continue
        
        # ─── INTRADAY ANALYSIS ───
        intra_score, intra_reasons, intra_exit, intra_stop = analyze_intraday(
            price, change, rv, rsi, macd, macd_sig, bb_low, bb_high,
            ema10, vwap, adx, stoch, vol, avg_vol_30
        )
        
        if intra_score >= 6:  # Strong intraday setup
            rr_ratio = ((intra_exit - price) / (price - intra_stop)) if intra_stop else 0
            intraday_rows.append({
                "Symbol": sym,
                "Sector": sector,
                "Price": round(price, 2),
                "Chg%": round(change, 2),
                "RV": round(rv, 2),
                "RSI": round(rsi, 1),
                "Score": intra_score,
                "Setup": " · ".join(intra_reasons[:3]),
                "Target": intra_exit,
                "Stop": intra_stop,
                "R:R": f"1:{rr_ratio:.1f}" if rr_ratio > 0 else "—",
            })
        
        # ─── SWING ANALYSIS ───
        swing_score, swing_reasons, swing_exit, swing_stop = analyze_swing(
            price, change, rv, rsi, macd, macd_sig, ema20, ema50,
            change1w, low1m, adx, atr, perf1m
        )
        
        if swing_score >= 7:  # Strong swing setup
            rr_ratio = ((swing_exit - price) / (price - swing_stop)) if swing_stop else 0
            swing_rows.append({
                "Symbol": sym,
                "Sector": sector,
                "Price": round(price, 2),
                "Chg%": round(change, 2),
                "1W%": round(change1w, 2),
                "RSI": round(rsi, 1),
                "Score": swing_score,
                "Setup": " · ".join(swing_reasons[:3]),
                "Target": swing_exit,
                "Stop": swing_stop,
                "R:R": f"1:{rr_ratio:.1f}" if rr_ratio > 0 else "—",
            })
        
        # ─── LONG-TERM ANALYSIS ───
        lt_score, lt_reasons, lt_exit, lt_stop = analyze_longterm(
            price, rsi, ema20, ema50, change1w, low1m, high1m, perf1m, sector
        )
        
        if lt_score >= 6:  # Strong long-term setup
            rr_ratio = ((lt_exit - price) / (price - lt_stop)) if lt_stop else 0
            longterm_rows.append({
                "Symbol": sym,
                "Sector": sector,
                "Price": round(price, 2),
                "1W%": round(change1w, 2),
                "1M%": round(perf1m, 2),
                "RSI": round(rsi, 1),
                "Score": lt_score,
                "Setup": " · ".join(lt_reasons[:3]),
                "Target": lt_exit,
                "Stop": lt_stop,
                "R:R": f"1:{rr_ratio:.1f}" if rr_ratio > 0 else "—",
            })
    
    # Convert to DataFrames and sort by score
    df_intraday = pd.DataFrame(intraday_rows).sort_values("Score", ascending=False) if intraday_rows else pd.DataFrame()
    df_swing = pd.DataFrame(swing_rows).sort_values("Score", ascending=False) if swing_rows else pd.DataFrame()
    df_longterm = pd.DataFrame(longterm_rows).sort_values("Score", ascending=False) if longterm_rows else pd.DataFrame()
    
    return df_intraday, df_swing, df_longterm


def get_position_status(symbol, entry_price, raw_data):
    """Analyze current position and provide exit signals"""
    # Find the stock in current data
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
    
    # Calculate P&L
    pnl_pct = ((curr_price - entry_price) / entry_price) * 100
    pnl_amount = curr_price - entry_price
    
    # Exit signals
    exit_signals = []
    exit_action = "🟢 HOLD"
    
    # Profit-taking signals
    if pnl_pct > 8:
        exit_signals.append("🎯 Strong profit - consider booking 50%")
        exit_action = "🟡 PARTIAL EXIT"
    
    # Momentum breakdown
    if curr_price < ema20 * 0.98 and pnl_pct < 0:
        exit_signals.append("⚠️ Broke below EMA20 - trend weakening")
        exit_action = "🔴 EXIT"
    
    # Overbought extremes
    if rsi > 75 and rv < 1.0:
        exit_signals.append("📊 RSI extreme + volume fading")
        exit_action = "🟡 CONSIDER EXIT"
    
    # MACD bearish cross
    if macd < macd_sig and pnl_pct > 3:
        exit_signals.append("💫 MACD bearish - lock profits")
        exit_action = "🟡 PARTIAL EXIT"
    
    # Stop loss breach
    if pnl_pct < -7:
        exit_signals.append("🛑 Stop loss level - cut losses")
        exit_action = "🔴 EXIT NOW"
    
    if not exit_signals:
        if pnl_pct > 0:
            exit_signals.append("✅ Position healthy - maintain stop")
        else:
            exit_signals.append("⏳ Developing - monitor closely")
    
    return {
        "symbol": symbol,
        "entry": entry_price,
        "current": curr_price,
        "pnl_pct": pnl_pct,
        "pnl_amount": pnl_amount,
        "rsi": rsi,
        "rv": rv,
        "action": exit_action,
        "signals": exit_signals
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PSX Pro Scanner",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Session state initialization
if 'positions' not in st.session_state:
    st.session_state.positions = []

# ─── STYLES ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0a0e1a !important;
    color: #e2e8f0 !important;
}

.pro-header {
    background: linear-gradient(135deg, #1a1f35 0%, #0f1219 100%);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.pro-title {
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #00ff88 0%, #00ccff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.pro-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #64748b;
    letter-spacing: 0.05em;
}

.strategy-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1.2rem;
    margin-bottom: 1.5rem;
}

.strategy-header {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.status-badge {
    background: #0f172a;
    border: 1px solid #1e293b;
    padding: 0.4rem 0.9rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}

.market-open {
    border-color: #10b981;
    color: #10b981;
}

.market-closed {
    border-color: #ef4444;
    color: #ef4444;
}

.position-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.pnl-positive { color: #10b981 !important; font-weight: 600; }
.pnl-negative { color: #ef4444 !important; font-weight: 600; }

[data-testid="stDataFrame"] {
    background: transparent !important;
}

thead tr th {
    background: #1e293b !important;
    color: #94a3b8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
}

tbody tr {
    border-bottom: 1px solid #1e293b !important;
}

tbody tr:hover {
    background: rgba(30, 41, 59, 0.3) !important;
}

.stButton>button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    font-size: 0.85rem;
    transition: all 0.2s;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}

hr {
    border-color: #1e293b !important;
    margin: 2rem 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── HEADER ───
is_open = is_market_open()
status_class = "market-open" if is_open else "market-closed"
status_text = "LIVE" if is_open else "CLOSED"
now_str = pkt_now().strftime("%H:%M PKT · %d %b %Y")

st.markdown(f"""
<div class="pro-header">
    <div class="pro-title">⚡ PSX Professional Scanner</div>
    <div class="pro-subtitle">
        <span class="status-badge {status_class}">● {status_text}</span>
        <span style="margin-left: 1rem;">{now_str}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── POSITION TRACKER SECTION ───
st.markdown("### 📊 Position Tracker")

col1, col2 = st.columns([3, 1])
with col1:
    search_symbol = st.text_input("🔍 Enter Symbol to Track", placeholder="e.g., HBL", key="search_symbol").upper()
with col2:
    entry_price_input = st.number_input("Entry Price", min_value=0.0, step=0.01, key="entry_price")

if st.button("➕ Add Position") and search_symbol and entry_price_input > 0:
    st.session_state.positions.append({
        "symbol": search_symbol,
        "entry": entry_price_input
    })
    st.success(f"Added {search_symbol} @ {entry_price_input}")
    st.rerun()

# Display tracked positions
if st.session_state.positions:
    raw_data = fetch_market_data()
    
    for idx, pos in enumerate(st.session_state.positions):
        status = get_position_status(pos["symbol"], pos["entry"], raw_data)
        
        if status:
            pnl_class = "pnl-positive" if status["pnl_pct"] > 0 else "pnl-negative"
            
            col_a, col_b, col_c = st.columns([2, 3, 1])
            
            with col_a:
                st.markdown(f"""
                <div class="position-card">
                    <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.3rem;">{status['symbol']}</div>
                    <div style="font-size: 0.8rem; color: #64748b;">Entry: {status['entry']:.2f} → Current: {status['current']:.2f}</div>
                    <div style="font-size: 1.1rem; margin-top: 0.5rem;" class="{pnl_class}">
                        {status['pnl_pct']:+.2f}% ({status['pnl_amount']:+.2f})
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class="position-card">
                    <div style="font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem;">{status['action']}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">
                        {'<br>'.join(['• ' + s for s in status['signals']])}
                    </div>
                    <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #64748b;">
                        RSI: {status['rsi']:.0f} · RV: {status['rv']:.1f}x
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                if st.button("🗑️ Remove", key=f"remove_{idx}"):
                    st.session_state.positions.pop(idx)
                    st.rerun()

st.divider()

# ─── SCAN TRIGGER ───
scan_now = is_open or st.button("🔍 Manual Scan", use_container_width=True)

if scan_now:
    with st.spinner("Scanning market..."):
        raw_data = fetch_market_data()
        df_intra, df_swing, df_long = process_signals(raw_data)
    
    # ─── INTRADAY ───
    st.markdown("""
    <div class="strategy-card">
        <div class="strategy-header">
            ⚡ <span style="color: #fbbf24;">INTRADAY</span> SCALPS
        </div>
        <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 1rem;">
            High-probability setups for same-day exits · Tight stops · Quick profits
        </div>
    """, unsafe_allow_html=True)
    
    if df_intra.empty:
        st.info("No intraday setups meet criteria")
    else:
        st.dataframe(
            df_intra,
            column_config={
                "Chg%": st.column_config.NumberColumn("Chg%", format="%.2f%%"),
                "RV": st.column_config.NumberColumn("RV", format="%.1fx"),
                "Target": st.column_config.NumberColumn("Target", format="%.2f"),
                "Stop": st.column_config.NumberColumn("Stop", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ─── SWING TRADING ───
    st.markdown("""
    <div class="strategy-card">
        <div class="strategy-header">
            🚀 <span style="color: #3b82f6;">SWING</span> TRADES
        </div>
        <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 1rem;">
            3-7 day momentum plays · Trend-following · Multi-timeframe confirmation
        </div>
    """, unsafe_allow_html=True)
    
    if df_swing.empty:
        st.info("No swing setups meet criteria")
    else:
        st.dataframe(
            df_swing,
            column_config={
                "Chg%": st.column_config.NumberColumn("Chg%", format="%.2f%%"),
                "1W%": st.column_config.NumberColumn("1W%", format="%.2f%%"),
                "Target": st.column_config.NumberColumn("Target", format="%.2f"),
                "Stop": st.column_config.NumberColumn("Stop", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ─── LONG-TERM ───
    st.markdown("""
    <div class="strategy-card">
        <div class="strategy-header">
            💎 <span style="color: #10b981;">LONG-TERM</span> INVESTMENTS
        </div>
        <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 1rem;">
            Quality stocks at value prices · Multi-week to months · Fundamental + technical edge
        </div>
    """, unsafe_allow_html=True)
    
    if df_long.empty:
        st.info("No long-term setups meet criteria")
    else:
        st.dataframe(
            df_long,
            column_config={
                "1W%": st.column_config.NumberColumn("1W%", format="%.2f%%"),
                "1M%": st.column_config.NumberColumn("1M%", format="%.2f%%"),
                "Target": st.column_config.NumberColumn("Target", format="%.2f"),
                "Stop": st.column_config.NumberColumn("Stop", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

# ─── AUTO REFRESH ───
if is_open:
    import time
    time.sleep(CONFIG["REFRESH_RATE"])
    st.rerun()
