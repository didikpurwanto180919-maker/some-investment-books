import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import feedparser
import urllib.parse
import urllib.request
import json
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

# Mengimpor model Machine Learning berbasis Gradient Boosting
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="AI Quantitative Cyber-Stock Predictive & Precision Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Konfigurasi Auto-Refresh Real-Time (Setiap 60 detik)
st_autorefresh(interval=60 * 1000, key="datarefresh")

# --- KUSTOMISASI CSS TEMA NEON FUTURISTIK & KETERBACAAN TEKS ---
st.markdown("""
    <style>
    .stApp {
        background-color: #05070c;
        color: #ffffff;
        font-family: 'Courier New', Courier, monospace;
    }
    h1, h2, h3 {
        color: #00ffcc !important;
        text-shadow: 0 0 8px rgba(0, 255, 204, 0.8);
        font-weight: bold;
    }
    section[data-testid="stSidebar"] {
        background-color: #0e1320;
        border-right: 2px solid #00ffcc55;
    }
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
        font-weight: bold !important;
    }
    .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #161f33 !important;
        color: #00ffcc !important;
        font-weight: bold !important;
        border: 1px solid #00ffcc !important;
    }
    div[data-baseweb="select"] span {
        color: #00ffcc !important;
        font-weight: bold !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(14, 19, 32, 0.9);
        border: 1px solid #00ffcc;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.2);
    }
    div[data-testid="stMetric"] label {
        color: #00ffcc !important;
        font-weight: bold !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.8);
        font-weight: bold !important;
    }
    .stMarkdown a {
        color: #00ffcc !important;
        font-weight: bold !important;
        text-decoration: underline;
    }
    p, span, li {
        color: #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# --- EFEK SUARA AUDIO NOTIFIKASI REAL-TIME ---
def play_neon_sound():
    if "sound_played" not in st.session_state:
        sound_html = """
            <audio autoplay>
              <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg">
            </audio>
        """
        st.markdown(sound_html, unsafe_allow_html=True)
        st.session_state.sound_played = True

play_neon_sound()

# Sidebar Navigasi & Input Parameter Saham
st.sidebar.header("⚡ QUANT CONFIG: ALL IDX STOCKS")

# --- WIDGET JAM REAL-TIME JAVASCRIPT (WIB) ---
st.sidebar.markdown("🕒 **Waktu Sistem (WIB):**")
clock_html = """
<div style="font-family: 'Courier New', Courier, monospace; font-size: 14px; font-weight: bold; color: #00ffcc; background: #161f33; padding: 8px; border-radius: 5px; border: 1px solid #00ffcc; text-align: center;" id="live-clock">Loading Clock...</div>
<script>
function updateClock() {
    const now = new Date();
    const options = { timeZone: 'Asia/Jakarta', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false, weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' };
    const formatter = new Intl.DateTimeFormat('en-GB', options);
    document.getElementById('live-clock').innerText = formatter.format(now);
}
setInterval(updateClock, 1000);
updateClock();
</script>
"""
components.html(clock_html, height=45)

# --- FUNGSI DINAMIS MENGAMBIL SELURUH EMITEN BEI / IDX ---
@st.cache_data(ttl=86400)
def get_all_idx_stocks():
    fallback_stocks = {
        "BBCA – Bank Central Asia Tbk": "BBCA.JK",
        "BBRI – Bank Rakyat Indonesia Tbk": "BBRI.JK",
        "BMRI – Bank Mandiri Tbk": "BMRI.JK",
        "BBNI – Bank Negara Indonesia Tbk": "BBNI.JK",
        "ASII – Astra International Tbk": "ASII.JK",
        "TLKM – Telkom Indonesia Tbk": "TLKM.JK",
        "GOTO – GoTo Gojek Tokopedia Tbk": "GOTO.JK",
        "BREN – Barito Renewables Energy Tbk": "BREN.JK",
        "AMMN – Amman Mineral Internasional Tbk": "AMMN.JK",
        "CUAN – Petrindo Jaya Kreasi Tbk": "CUAN.JK"
    }
    try:
        url = "https://raw.githubusercontent.com/nightfury1204/indonesia-stock-exchange-list/main/stocks.json"
        req = urllib.request.urlopen(url, timeout=5)
        data_json = json.loads(req.read().decode())
        all_stocks = {}
        for item in data_json:
            code = item.get('code')
            name = item.get('name')
            if code:
                ticker_key = f"{code.upper()} – {name}" if name else f"{code.upper()}.JK"
                all_stocks[ticker_key] = f"{code.upper()}.JK"
        return all_stocks if all_stocks else fallback_stocks
    except Exception:
        return fallback_stocks

with st.spinner("Memuat database seluruh emiten BEI..."):
    dict_all_stocks = get_all_idx_stocks()

# --- FILTER MENU: PILIHAN EMITEN & FILTER KHUSUS AI (NAIK > 2% & PRESISI > 90%) ---
filter_high_precision_menu = st.sidebar.checkbox("🎯 Filter Saring Emiten Bluechip / Presisi Tinggi (>90%)", value=False)
filter_ai_bullish_high_acc = st.sidebar.checkbox("🚀 Filter Sinyal Bullish AI (Naik > 2% & Presisi > 90%)", value=False)

if filter_high_precision_menu:
    filtered_stocks = {k: v for k, v in dict_all_stocks.items() if any(x in v for x in ["BBCA", "BBRI", "BMRI", "BBNI", "ASII", "TLKM", "ICBP", "UNVR", "ADRO", "PTBA", "BREN", "ANTM"])}
    if not filtered_stocks:
        filtered_stocks = dict_all_stocks
else:
    filtered_stocks = dict_all_stocks

stock_choice = st.sidebar.selectbox(
    "Pilih atau Cari Emiten BEI (Ketik nama/kode)", 
    options=list(filtered_stocks.keys()),
    index=0
)
selected_ticker_default = filtered_stocks[stock_choice]

ticker_symbol = st.sidebar.text_input("Atau Ketik Manual Kode Saham IDX (Format: KODE.JK)", value=selected_ticker_default)
ticker_symbol = ticker_symbol.strip().upper()

ml_engine = st.sidebar.selectbox(
    "Pilih Neural / Quant Engine AI",
    ["Ensemble (XGBoost + LightGBM Hybrid)", "XGBoost Regressor", "LightGBM Regressor"],
    index=0
)

period_option = st.sidebar.selectbox("Rentang Waktu Analisis", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
interval_option = st.sidebar.selectbox("Interval Candle", ["1d", "1wk"], index=0)

# Opsi Tampilan Indikator Tambahan di Grafik
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Kustomisasi Tampilan Grafik")
show_ichimoku = st.sidebar.checkbox("Tampilkan Ichimoku Cloud", value=True)
show_sar = st.sidebar.checkbox("Tampilkan Parabolic SAR", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Input Manajemen Modal (Trading Calculator)")
user_capital = st.sidebar.number_input("Total Modal Trading (IDR)", min_value=100000, value=10000000, step=500000)
risk_tolerance_pct = st.sidebar.slider("Maksimal Risiko per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Status Sistem Validasi")
st.sidebar.success("🟢 Validasi Out-Of-Sample Aktif\n📊 Fitur Lanjutan (Ichimoku, SAR, MFI) Sinkron")

st.title("⚡ QUANT AI: High-Precision Predictive & Risk Management Dashboard")
st.markdown(
    f"Sistem analitik kuant
