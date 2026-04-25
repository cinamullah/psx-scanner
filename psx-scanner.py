import streamlit as st
import pandas as pd
import requests
import sqlite3
import os
from datetime import datetime, timedelta
import pytz
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "MIN_REL_VOL": 0.5,
    "MAX_CHANGE_PCT": 9.0,
    "START_TIME": "09:17",
    "END_TIME": "15:30",
    "DB_PATH": os.path.join(os.path.dirname(__file__), "psx_market_data.db"),
    "REFRESH_RATE": 30,
    "MAX_SCORE": 14,
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
    "TRG","UBL","UNITY","YWMIL",
    # Dividend additions
    "EFUG","EFUL","EPQL","BOK","JGICL","HPL","ATIL","NESTLE",
    "ZIL","FCEPL","UPFL","ALIFE","WAFI","CENI"
]

SECTORS = {
    "Banks":         ["MEBL","MCB","UBL","HBL","ABL","AKBL","BAFL","BAHL","BOP","FABL","SCBPL"],
    "E&P":           ["OGDC","PPL","MARI"],
    "Fertilizer":    ["FFC","ENGRO","EFERT","FATIMA","SFERT"],
    "Cement":        ["LUCK","DGKC","MLCF","BWCL","CHCC","FCCL","KOHCK"],
    "Tech":          ["SYS","TRG","PTC"],
    "Power":         ["HUBC","KAPCO","KEL"],
    "Oil & Gas Mktg":["APL","SHEL","SNGP"],
    "Automobile":    ["ATLH","HCAR","MTL","THALL"],
    "Food":          ["FFL","UNITY","MUREB","RAFHAN"],
    "Pharma":        ["ABOT","AGP","SEARL","GLAXO","HALEON"],
    "Textile":       ["BNWM","KTML","LCIL"],
    "Refinery":      ["ATRL","CNERGY"],
}

SYMBOL_TO_SECTOR = {sym: sec for sec, syms in SECTORS.items() for sym in syms}

SIGNAL_META = {
    "Short-Term Power":  {"color": "#ff00ff", "bg": "#4a044a", "priority": 7},
    "Momentum Run":      {"color": "#00ff9d", "bg": "#001a0d", "priority": 6},
    "Stealth Accum":     {"color": "#f59e0b", "bg": "#1a1200", "priority": 5}, # High volume, stable price
    "Oversold Bounce":   {"color": "#38bdf8", "bg": "#001524", "priority": 4}, # Potential reversal
    "Potential Bargain": {"color": "#a78bfa", "bg": "#110d24", "priority": 3}, # Value hunting
    "Quiet":             {"color": "#94a3b8", "bg": "#111827", "priority": 2},
    "Profit Taking":     {"color": "#f87171", "bg": "#1a0505", "priority": 1}, # Consider booking gains
}

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(CONFIG["DB_PATH"], check_same_thread=False)
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                symbol TEXT NOT NULL,
                price  REAL,
                rel_vol REAL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sym_ts ON history (symbol, timestamp)")
        # Auto-purge records older than 2 days
        cutoff = (datetime.now() - timedelta(days=2)).isoformat()
        conn.execute("DELETE FROM history WHERE timestamp < ?", (cutoff,))
        conn.commit()

# ─────────────────────────────────────────────
# TIME UTILITIES
# ─────────────────────────────────────────────
def pkt_now():
    return datetime.now(pytz.timezone("Asia/Karachi"))

def is_market_open():
    now = pkt_now()
    if now.weekday() >= 5:
        return False
    t = now.strftime("%H:%M")
    return CONFIG["START_TIME"] <= t <= CONFIG["END_TIME"]

def minutes_to_close():
    now = pkt_now()
    close_str = CONFIG["END_TIME"]
    close_h, close_m = map(int, close_str.split(":"))
    close_dt = now.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
    delta = (close_dt - now).total_seconds() / 60
    return max(0, int(delta))

# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────
TV_URL = "https://scanner.tradingview.com/pakistan/scan"
TV_COLUMNS = [
    "name", "close", "change", "volume", "relative_volume_10d_calc",  # 0-4
    "average_volume_10d_calc", "RSI", "MACD.macd", "MACD.signal",      # 5-8
    "BB.lower", "BB.upper", "EMA20", "EMA50",                           # 9-12
    "change|1W", "High.1M", "Low.1M", "VWAP",                          # 13-16
    "dividend_ex_date_upcoming",                                         # 17
    "EMA10", "ADX", "Pivot.M.Classic.Middle", "average_volume_30d_calc"  # 18-21
]

def get_tv_session():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s

@st.cache_data(ttl=25, show_spinner=False)
def fetch_main_data():
    payload = {
        "filter": [
            {"left": "name",                        "operation": "in_range",    "right": KSE100_SYMBOLS},
            # Moved volume/change filters to Python to ensure Dividend stocks aren't filtered out by API
            {"left": "change",                      "operation": "less",        "right": CONFIG["MAX_CHANGE_PCT"]},
        ],
        "markets": ["pakistan"],
        "columns": TV_COLUMNS,
        "range": [0, 250], # Increased range to accommodate extra symbols
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
def fetch_sector_performance():
    all_syms = [s for sec in SECTORS.values() for s in sec]
    payload = {
        "filter": [{"left": "name", "operation": "in_range", "right": all_syms}],
        "markets": ["pakistan"],
        "columns": ["name", "change"],
        "range": [0, 150],
    }
    try:
        s = get_tv_session()
        r = s.post(TV_URL, json=payload, timeout=10)
        data = r.json().get("data", [])
        result = {}
        for sec, syms in SECTORS.items():
            vals = [item["d"][1] for item in data if item["d"][0] in syms and item["d"][1] is not None]
            result[sec] = round(sum(vals) / len(vals), 2) if vals else 0.0
        return result
    except:
        return {}

@st.cache_data(ttl=30, show_spinner=False)
def fetch_market_breadth():
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
        adv = sum(1 for c in changes if c > 0)
        dec = sum(1 for c in changes if c < 0)
        unch = sum(1 for c in changes if c == 0)
        avg_chg = round(sum(changes) / len(changes), 2) if changes else 0
        return adv, dec, unch, avg_chg
    except:
        return 0, 0, 0, 0.0

def _safe(val, default=0):
    return val if val is not None else default

# ─────────────────────────────────────────────
# SIGNAL SCORING ENGINE
# ─────────────────────────────────────────────
def get_signal_and_priority(price, change, rel_vol, rsi, macd, macd_sig,
                            bb_low, bb_high, ema20, ema50, change1w, low1m, vwap,
                            ema_fast, adx, pivot, vol, avg_vol_30,
                            prev_rv, prev_price):
    details = []

    # 0. Short-Term Power (The "Best" Practical Preset)
    # Price > EMA 9 (using EMA10), RSI 45-65, Vol > 1.5x AvgVol30, Close > Pivot, ADX > 25, Chg > 1.5%
    vol_ratio_30 = vol / avg_vol_30 if avg_vol_30 > 0 else 0

    if (price > ema_fast and
        45 <= rsi <= 65 and
        vol_ratio_30 > 1.5 and
        pivot > 0 and price > pivot and
        adx > 25 and
        change > 1.5):
        details.append(f"ADX:{adx:.0f}")
        details.append(f"Vol30:{vol_ratio_30:.1f}x")
        return "Short-Term Power", 7, details

    # 1. Profit Taking Zone (Consider booking gains)
    is_extended = change > 6.5 or (bb_high > 0 and price >= bb_high * 1.01)
    is_overbought = rsi > 72
    vol_fading = rel_vol < prev_rv * 0.8 and prev_rv > 1.8
    if (is_extended and is_overbought) or (is_extended and vol_fading):
        details.append(f"RSI: {rsi:.0f}" if is_overbought else "Extended")
        if vol_fading: details.append("Vol Fading")
        return "Profit Taking", 1, details

    # 2. Stealth Accumulation (High volume, stable price)
    is_stable_price = -1.0 < change < 2.0
    is_high_vol = rel_vol > 1.8
    if is_stable_price and is_high_vol and price > vwap:
        details.append(f"{rel_vol:.1f}x Vol")
        details.append(f"Chg: {change:+.1f}%")
        return "Stealth Accum", 5, details

    # 3. Oversold Bounce (Potential reversal zone)
    is_oversold = rsi < 35 or (bb_low > 0 and price < bb_low * 1.03)
    price_reclaiming = price > prev_price and price > ema20 * 0.995
    if is_oversold and price_reclaiming:
        details.append(f"RSI: {rsi:.0f}")
        if price > ema20: details.append("Crossed EMA20")
        return "Oversold Bounce", 4, details

    # 4. Potential Bargain (Value Hunters)
    is_near_low = low1m > 0 and price < low1m * 1.12
    is_down_week = change1w < -4.0
    if is_near_low and is_down_week and not is_oversold:
         details.append(f"Near 1M Low")
         details.append(f"1W: {change1w:.1f}%")
         return "Potential Bargain", 3, details

    # 5. Momentum Run (Just started raising)
    dist_vwap = (price - vwap) / vwap * 100 if vwap > 0 else 0
    if price > vwap and change > 1.0 and 0 < dist_vwap < 2.5 and rel_vol > 1.2:
        details.append(f"Near VWAP")
        if rel_vol > 2.0: details.append("High Vol")
        return "Momentum Run", 6, details

    return "Quiet", 2, []

def vol_trend_label(rel_vol, prev_rv):
    if prev_rv <= 0:   return "🆕 NEW"
    ratio = rel_vol / prev_rv
    if ratio > 1.25:   return "🔥 SURGING"
    if ratio > 1.05:   return "⬆️ RISING"
    if ratio < 0.75:   return "⬇️ FADING"
    return "➡️ STABLE"

def process_signals(raw_data):
    if not raw_data:
        return pd.DataFrame()

    ts = datetime.now().isoformat()
    today_prefix = datetime.now().strftime("%Y-%m-%d") + "%"
    rows = []

    with get_db() as conn:
        for item in raw_data:
            d = item["d"]
            if len(d) < 22 or d[0] is None:
                continue

            sym    = d[0]
            price  = _safe(d[1])
            change = _safe(d[2])
            vol    = _safe(d[3])
            rv     = _safe(d[4])
            rsi    = _safe(d[6])
            macd   = _safe(d[7])
            macs   = _safe(d[8])
            bb_low = _safe(d[9])
            bb_hi  = _safe(d[10])
            ema20  = _safe(d[11])
            ema50  = _safe(d[12])
            chg1w  = _safe(d[13])
            hi1m   = _safe(d[14])
            lo1m   = _safe(d[15])
            vwap   = _safe(d[16]) or price
            div_raw= d[17]
            ema_fast = _safe(d[18])
            adx      = _safe(d[19])
            pivot    = _safe(d[20])
            avg_vol_30 = _safe(d[21])

            sector = SYMBOL_TO_SECTOR.get(sym, "Other")

            # ── Dividend Ex-Date Parsing ────────────
            ex_date_str = "—"
            has_upcoming_div = False
            if div_raw:
                try:
                    ex_dt = datetime.strptime(str(div_raw), "%Y-%m-%d").date()
                    # Robust parsing: remove 'T' time part if present (e.g., 2026-03-18T00:00:00)
                    ex_dt = datetime.strptime(str(div_raw).split("T")[0], "%Y-%m-%d").date()
                    days_left = (ex_dt - datetime.now().date()).days
                    # Show if upcoming or recently passed (within last 2 days)
                    if days_left >= -1:
                        has_upcoming_div = True
                        base_fmt = ex_dt.strftime("%d-%b")
                        if 0 <= days_left <= 5:
                            ex_date_str = f"⚠️ {base_fmt}"
                        else:
                            ex_date_str = base_fmt
                except: pass

            # ── Filtering Logic ────────────
            # Keep all stocks for initial processing, filter out "Quiet" ones later

            # ── Pull last 3 intraday snapshots ──────
            cur = conn.execute(
                """SELECT rel_vol, price FROM history
                   WHERE symbol=? AND timestamp LIKE ?
                   ORDER BY timestamp DESC LIMIT 3""",
                (sym, today_prefix)
            )
            hist = cur.fetchall()
            prev_rv    = hist[0][0] if len(hist) > 0 else 0
            prev2_rv   = hist[1][0] if len(hist) > 1 else 0
            prev_price = hist[0][1] if len(hist) > 0 else price

            signal, priority, details = get_signal_and_priority(
                price, change, rv, rsi, macd, macs,
                bb_low, bb_hi, ema20, ema50, chg1w, lo1m, vwap,
                ema_fast, adx, pivot, vol, avg_vol_30,
                prev_rv, prev_price
            )

            # Filter out quiet signals unless they have an upcoming dividend
            if signal == "Quiet" and not has_upcoming_div:
                continue

            trend = vol_trend_label(rv, prev_rv)

            # 52W distance (using 1M as proxy for now)
            dist_from_low = round((price / lo1m - 1) * 100, 1) if lo1m > 0 else None

            rows.append({
                "Sym":        sym,
                "Sector":     sector,
                "Price":      round(price, 2),
                "Chg%":       round(change, 2),
                "RV":         round(rv, 2),
                "RSI":        round(rsi, 1),
                "Trend":      trend,
                "Signal":     signal,
                "Details":    " · ".join(details),
                "From1MLow%": dist_from_low,
                "priority":   priority,
                "Ex-Date":    ex_date_str,
            })

            # ── Persist snapshot ────────────────────
            conn.execute("INSERT INTO history VALUES (?,?,?,?)", (sym, price, rv, ts))
        conn.commit()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("priority", ascending=False)
    return df

# ─────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PSX Momentum Scanner",
    layout="wide",
    initial_sidebar_state="collapsed",
)
init_db()

# ── Global Styles ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background: #0d1117 !important;
    color: #e2e8f0 !important;
}

/* ── Hide sidebar ── */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"]        { display: none !important; }
section[data-testid="stSidebarContent"] { display: none !important; }

/* ── Header ── */
.psx-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 0 1rem 0;
    border-bottom: 1px solid #21303f;
    margin-bottom: 1.1rem;
}
.psx-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: #f0f6fc;
}
.psx-title span { color: #3ddc84; }

/* ── Status pill ── */
.status-open {
    background: #0d2e1a; color: #3ddc84;
    border: 1px solid #3ddc8466;
    padding: 3px 12px; border-radius: 20px;
    font-size: 0.74rem; font-family: 'IBM Plex Mono', monospace;
    font-weight: 600; letter-spacing: .06em;
}
.status-closed {
    background: #2d1010; color: #ff6b6b;
    border: 1px solid #ff6b6b55;
    padding: 3px 12px; border-radius: 20px;
    font-size: 0.74rem; font-family: 'IBM Plex Mono', monospace;
    font-weight: 600; letter-spacing: .06em;
}

/* ── KPI cards ── */
.kpi-grid { display: flex; gap: 0.65rem; margin-bottom: 1.1rem; flex-wrap: wrap; }
.kpi-card {
    background: #161b22; border: 1px solid #21303f;
    border-radius: 7px; padding: 0.65rem 1rem; min-width: 130px; flex: 1;
}
.kpi-label {
    font-size: 0.63rem; letter-spacing: .1em; color: #8b949e;
    text-transform: uppercase; font-family: 'IBM Plex Mono', monospace;
}
.kpi-value {
    font-size: 1.45rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace;
    color: #f0f6fc; line-height: 1.2; margin-top: 3px;
}
.kpi-sub { font-size: 0.72rem; color: #8b949e; margin-top: 2px; }
.kpi-pos { color: #3ddc84 !important; }
.kpi-neg { color: #ff6b6b !important; }
.kpi-neu { color: #ffd166 !important; }

/* ── Sector pulse bar ── */
.sector-section {
    margin-bottom: 1.1rem;
}
.sector-label-row {
    font-size: 0.63rem; color: #8b949e; letter-spacing: .1em;
    text-transform: uppercase; font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 0.45rem;
}
.sector-grid { display: flex; flex-wrap: wrap; gap: 0.45rem; }
.sector-tile {
    display: flex; align-items: center; gap: 6px;
    border-radius: 6px; padding: 5px 11px;
    font-size: 0.72rem; font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    border: 1px solid transparent;
    cursor: default; transition: transform 0.12s, opacity 0.12s;
    white-space: nowrap;
}
.sector-tile:hover { transform: translateY(-2px); opacity: 0.9; }
.sector-arrow { font-size: 0.8rem; line-height: 1; }
.sector-name  { font-size: 0.7rem; opacity: 0.85; }
.sector-val   { font-size: 0.78rem; font-weight: 700; }

/* ── Table ── */
[data-testid="stDataFrame"] { background: transparent !important; }
thead tr th {
    background: #161b22 !important; color: #8b949e !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .72rem !important; letter-spacing: .06em;
}
tbody tr { border-bottom: 1px solid #161b22 !important; }

/* ── Dividers ── */
hr { border-color: #21303f !important; }

/* ── Footer ── */
.psx-footer {
    margin-top: 1.5rem;
    padding-top: 0.6rem;
    border-top: 1px solid #161b22;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.63rem;
    color: #3d4f61;
    display: flex;
    justify-content: space-between;
}
.psx-footer span { color: #4e6478; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #21303f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
open_flag = is_market_open()
status_html = (
    '<span class="status-open">● MARKET OPEN</span>'
    if open_flag else
    '<span class="status-closed">● MARKET CLOSED</span>'
)
now_str = pkt_now().strftime("%H:%M:%S PKT  ·  %a %d %b %Y")

st.markdown(f"""
<div class="psx-header">
  <div class="psx-title">⚡ PSX <span>MOMENTUM</span> SCANNER · KSE-100</div>
  <div style="display:flex;align-items:center;gap:14px;">
    {status_html}
    <span style="font-family:'IBM Plex Mono',monospace;font-size:.73rem;color:#8b949e;">{now_str}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SCAN TRIGGER
# ─────────────────────────────────────────────
scan_triggered = open_flag

if not open_flag:
    col_warn, col_btn = st.columns([5, 1])
    with col_warn:
        st.info(f"Market hours: **{CONFIG['START_TIME']}** – **{CONFIG['END_TIME']}** PKT (Mon–Fri). "
                "Use manual scan to view last captured data.")
    with col_btn:
        if st.button("🔎 Manual Scan", use_container_width=True):
            scan_triggered = True

if scan_triggered:
    with st.spinner("Fetching market data…"):
        sec_perf = fetch_sector_performance()
        adv, dec, unch, avg_chg = fetch_market_breadth()
        raw      = fetch_main_data()
        df       = process_signals(raw)

    # ─────────────────────────────────────────
    # KPI ROW
    # ─────────────────────────────────────────
    breadth_color = "kpi-pos" if adv > dec else "kpi-neg" if dec > adv else "kpi-neu"
    avg_color     = "kpi-pos" if avg_chg >= 0 else "kpi-neg"
    top_sym       = df.iloc[0]["Sym"]    if not df.empty else "—"
    top_sig       = df.iloc[0]["Signal"] if not df.empty else "—"
    max_rv_sym    = df.loc[df["RV"].idxmax(), "Sym"] if not df.empty else "—"
    max_rv_val    = f"{df['RV'].max():.1f}x"          if not df.empty else "—"
    mins_left     = minutes_to_close() if open_flag else 0

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Signals</div>
        <div class="kpi-value">{len(df)}</div>
        <div class="kpi-sub">of {len(raw)} scanned</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Top Signal</div>
        <div class="kpi-value">{top_sym}</div>
        <div class="kpi-sub">{top_sig}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Peak Volume</div>
        <div class="kpi-value">{max_rv_sym}</div>
        <div class="kpi-sub">{max_rv_val} rel. vol</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Breadth A/D</div>
        <div class="kpi-value {breadth_color}">{adv}/{dec}</div>
        <div class="kpi-sub">{unch} unchanged</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Avg Change</div>
        <div class="kpi-value {avg_color}">{avg_chg:+.2f}%</div>
        <div class="kpi-sub">KSE-100 avg</div>
      </div>
      {'<div class="kpi-card"><div class="kpi-label">Time to Close</div>'
       f'<div class="kpi-value kpi-neu">{mins_left}m</div>'
       '<div class="kpi-sub">session remaining</div></div>'
       if open_flag else ''}
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # SECTOR PULSE — with direction indicators
    # ─────────────────────────────────────────
    if sec_perf:
        # Pull previous sector snapshot from DB for direction
        today_prefix = datetime.now().strftime("%Y-%m-%d") + "%"
        sec_prev = {}
        with get_db() as conn:
            for sec, syms in SECTORS.items():
                placeholders = ",".join("?" * len(syms))
                cur = conn.execute(
                    f"""SELECT AVG(rel_vol) FROM history
                        WHERE symbol IN ({placeholders})
                        AND timestamp LIKE ?
                        ORDER BY timestamp DESC LIMIT 1""",
                    (*syms, today_prefix)
                )
                row = cur.fetchone()
                sec_prev[sec] = row[0] if row and row[0] else None

        tiles = ""
        for sec, val in sorted(sec_perf.items(), key=lambda x: -x[1]):
            sign    = "+" if val >= 0 else ""
            pos     = val >= 0

            # Direction arrow: compare to previous reading
            prev_val = sec_prev.get(sec)
            if prev_val is None:
                arrow = "●"
                arrow_color = "#8b949e"
            elif val > 0.1:
                arrow = "▲"
                arrow_color = "#3ddc84"
            elif val < -0.1:
                arrow = "▼"
                arrow_color = "#ff6b6b"
            else:
                arrow = "▶"
                arrow_color = "#ffd166"

            # Background and text colors — vivid, readable
            if pos:
                bg_color   = "rgba(61,220,132,0.12)"
                border_col = "rgba(61,220,132,0.35)"
                text_color = "#3ddc84"
            else:
                bg_color   = "rgba(255,107,107,0.12)"
                border_col = "rgba(255,107,107,0.35)"
                text_color = "#ff6b6b"

            tiles += (
                f'<div class="sector-tile" '
                f'style="background:{bg_color};border-color:{border_col};">'
                f'<span class="sector-arrow" style="color:{arrow_color};">{arrow}</span>'
                f'<span class="sector-name" style="color:#cbd5e0;">{sec}</span>'
                f'<span class="sector-val" style="color:{text_color};">{sign}{val:.2f}%</span>'
                f'</div>'
            )

        st.markdown(
            '<div class="sector-section">'
            '<div class="sector-label-row">Sector Pulse</div>'
            f'<div class="sector-grid">{tiles}</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ─────────────────────────────────────────
    # MAIN TABLES (GROUPED BY SIGNAL)
    # ─────────────────────────────────────────
    if df.empty:
        st.warning("⚠️ No signals found. Market may be quiet or data is unavailable.")
    else:
        # Define the order and titles for the tables
        signal_groups = {
            "Short-Term Power":  "⚡ Short-Term Power",
            "Momentum Run":      "🚀 Momentum Run",
            "Stealth Accum":     "🐋 Stealth Accumulation",
            "Oversold Bounce":   "↺ Oversold Bounce",
            "Potential Bargain": "💎 Potential Bargains",
            "Profit Taking":     "💰 Profit Taking Zone"
        }

        # Base columns to display in each table
        display_cols = ["Sym","Sector","Price","Chg%","RV","RSI","Trend",
                        "Details","From1MLow%","Ex-Date"]

        # Base column configuration
        col_cfg = {
            "Chg%":       st.column_config.NumberColumn("Chg%",     format="%.2f%%"),
            "Price":      st.column_config.NumberColumn("Price",    format="%.2f"),
            "RV":         st.column_config.NumberColumn("Rel Vol",  format="%.2fx"),
            "RSI":        st.column_config.NumberColumn("RSI",      format="%.0f"),
            "From1MLow%": st.column_config.NumberColumn("↑ 1M Low", format="%.1f%%"),
            "Details":    st.column_config.TextColumn("Details",    width="large"),
            "Sector":     st.column_config.TextColumn("Sector",     width="small"),
        }

        for signal_name, signal_title in signal_groups.items():
            # Filter the dataframe for the current signal group
            group_df = df[df["Signal"] == signal_name]

            if not group_df.empty:
                st.subheader(signal_title)

                # The styling function needs the 'Signal' column to get the color
                def color_rows(row):
                    meta = SIGNAL_META.get(row["Signal"], {})
                    bg = meta.get("bg", "#111827")
                    fg = meta.get("color", "#e2e8f0")
                    return [f"background-color:{bg};color:{fg};"] * len(row)

                # We pass the columns to display + the 'Signal' column needed for styling
                styled = group_df[display_cols + ["Signal"]].style.apply(color_rows, axis=1)

                st.dataframe(
                    styled,
                    # Hide the 'Signal' column in the final display
                    column_config={**col_cfg, "Signal": None},
                    hide_index=True,
                    use_container_width=True,
                )
                # Add some space between tables for better readability
                st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────
    last_update = datetime.now().strftime("%H:%M:%S")
    st.markdown(
        f'<div class="psx-footer">'
        f'<span>PSX Momentum Scanner · Data: TradingView · KSE-100 Universe</span>'
        f'<span>Last scan: <span>{last_update}</span>'
        f'{"&nbsp;·&nbsp;Next refresh: ~30s" if open_flag else ""}'
        f'</span></div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# AUTO-REFRESH
# ─────────────────────────────────────────────
if open_flag:
    time.sleep(CONFIG["REFRESH_RATE"])
    st.cache_data.clear()
    st.rerun()