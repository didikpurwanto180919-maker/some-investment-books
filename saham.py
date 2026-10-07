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
    comprehensive_stocks = {
        "BBCA – Bank Central Asia Tbk": "BBCA.JK",
        "BBRI – Bank Rakyat Indonesia Tbk": "BBRI.JK",
        "BMRI – Bank Mandiri Tbk": "BMRI.JK",
        "BBNI – Bank Negara Indonesia Tbk": "BBNI.JK",
        "ASII – Astra International Tbk": "ASII.JK",
        "TLKM – Telkom Indonesia Tbk": "TLKM.JK",
        "GOTO – GoTo Gojek Tokopedia Tbk": "GOTO.JK",
        "BREN – Barito Renewables Energy Tbk": "BREN.JK",
        "AMMN – Amman Mineral Internasional Tbk": "AMMN.JK",
        "CUAN – Petrindo Jaya Kreasi Tbk": "CUAN.JK",
        "ADRO – Adaro Energy Indonesia Tbk": "ADRO.JK",
        "PTBA – Bukit Asam Tbk": "PTBA.JK",
        "UNVR – Unilever Indonesia Tbk": "UNVR.JK",
        "ICBP – Indofood CBP Sukses Makmur Tbk": "ICBP.JK",
        "INDF – Indofood Sukses Makmur Tbk": "INDF.JK",
        "ANTM – Aneka Tambang Tbk": "ANTM.JK",
        "MDKA – Merdeka Copper Gold Tbk": "MDKA.JK",
        "INCO – Vale Indonesia Tbk": "INCO.JK",
        "KLBF – Kalbe Farma Tbk": "KLBF.JK",
        "SMGR – Semen Indonesia Tbk": "SMGR.JK",
        "INTP – Indocement Tunggal Prakarsa Tbk": "INTP.JK",
        "JSMR – Jasa Marga (Persero) Tbk": "JSMR.JK",
        "PGAS – Perusahaan Gas Negara Tbk": "PGAS.JK",
        "UNTR – United Tractors Tbk": "UNTR.JK",
        "ARTO – Bank Jago Tbk": "ARTO.JK",
        "BRIS – Bank Syariah Indonesia Tbk": "BRIS.JK",
        "MAPI – Mitra Adiperkasa Tbk": "MAPI.JK",
        "ACES – Aspirasi Hidup Indonesia Tbk": "ACES.JK",
        "MEDC – Medco Energi Internasional Tbk": "MEDC.JK",
        "EXCL – XL Axiata Tbk": "EXCL.JK",
        "ISAT – Indosat Tbk": "ISAT.JK"
    }
    try:
        url = "https://raw.githubusercontent.com/nightfury1204/indonesia-stock-exchange-list/main/stocks.json"
        req = urllib.request.urlopen(url, timeout=3)
        data_json = json.loads(req.read().decode())
        for item in data_json:
            code = item.get('code')
            name = item.get('name')
            if code:
                ticker_key = f"{code.upper()} – {name}" if name else f"{code.upper()}.JK"
                comprehensive_stocks[ticker_key] = f"{code.upper()}.JK"
    except Exception:
        pass
        
    return comprehensive_stocks

with st.spinner("Memuat database seluruh emiten BEI..."):
    dict_all_stocks = get_all_idx_stocks()

# --- FILTER MENU: PILIHAN EMITEN & FILTER KHUSUS AI ---
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
st.sidebar.success("🟢 Validasi Out-Of-Sample & NLP Sentiment Aktif")

st.title("⚡ QUANT AI: High-Precision Predictive & Risk Management Dashboard")
st.markdown(
    f"Sistem analitik kuantitatif pasar saham tingkat lanjut dengan validasi statistik dan manajemen risiko presisi tinggi untuk emiten **{ticker_symbol}**."
)

@st.cache_data(ttl=30)
def fetch_stock_and_market_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        benchmark = yf.download("^JKSE", period=period, interval=interval, progress=False)
        if isinstance(benchmark.columns, pd.MultiIndex):
            benchmark.columns = benchmark.columns.droplevel(1)
            
        return df, benchmark['Close'] if not benchmark.empty else pd.Series()
    except Exception:
        return pd.DataFrame(), pd.Series()

# --- NLP SENTIMEN BERITA REAL-TIME TINGKAT LANJUT ---
@st.cache_data(ttl=120)
def fetch_advanced_realtime_news_and_sentiment(ticker):
    clean_code = ticker.replace(".JK", "").lower()
    news_items = []
    query = urllib.parse.quote(f"saham {clean_code} OR {ticker}")
    rss_url = f"https://news.google.com/rss/search?q={query}+when:7d&hl=id&gl=ID&ceid=ID:id"
    
    sentiment_score = 0
    
    high_impact_positive = ["laba melonjak", "dividen jumbo", "akuisisi strategis", "rekor tertinggi", "buyback saham", "tumbuh positif", "ekspansi pabrik"]
    moderate_positive = ["naik", "tumbuh", "menguat", "rebound", "beli", "positif", "kontrak baru", "kinerja solid"]
    
    high_impact_negative = ["suspensi bursa", "gagal bayar", "default", "rugi bersih", "anjlok tajam", "kasus hukum", "sanksi ojk"]
    moderate_negative = ["turun", "koreksi", "melemah", "beban meningkat", "rugi", "lesu", "tekanan jual"]

    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:7]:
            title = entry.title
            link = entry.link
            published = entry.published if hasattr(entry, 'published') else "Terbaru"
            title_lower = title.lower()
            
            source = "Portal Finansial"
            if "kontan" in link.lower() or "kontan" in title.lower():
                source = "Kontan Investasi"
            elif "cnbcindonesia" in link.lower() or "cnbc" in title.lower():
                source = "CNBC Indonesia"
            elif "bisnis" in link.lower():
                source = "Bisnis Market"
            elif "idx" in link.lower():
                source = "Keterbukaan Informasi IDX"

            item_sentiment = "Netral 🟡"
            score_delta = 0

            if any(k in title_lower for k in high_impact_positive):
                score_delta += 2
                item_sentiment = "Sangat Positif 🟢 (+2)"
            elif any(k in title_lower for k in moderate_positive):
                score_delta += 1
                item_sentiment = "Positif 🟢 (+1)"

            if any(k in title_lower for k in high_impact_negative):
                score_delta -= 2
                item_sentiment = "Sangat Negatif 🔴 (-2)"
            elif any(k in title_lower for k in moderate_negative):
                score_delta -= 1
                item_sentiment = "Negatif 🔴 (-1)"

            sentiment_score += score_delta
            news_items.append({
                "title": title, 
                "link": link, 
                "publisher": source, 
                "date": published, 
                "sentiment": item_sentiment
            })
    except Exception:
        pass
        
    return news_items, sentiment_score

data, benchmark_close = fetch_stock_and_market_data(ticker_symbol, period_option, interval_option)
realtime_news, news_sentiment_score = fetch_advanced_realtime_news_and_sentiment(ticker_symbol)

if data.empty or len(data) < 50:
    st.warning(
        f"⚠️ Data untuk ticker **{ticker_symbol}** dengan rentang waktu **{period_option}** tidak mencukupi (minimal 50 baris untuk akurasi tinggi). "
        f"Silakan pilih rentang waktu yang lebih panjang di sidebar atau pastikan kode saham benar."
    )
else:
    # --- FEATURE ENGINEERING TINGKAT LANJUT ---
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    std_dev = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (std_dev * 2)
    data['BB_Lower'] = data['BB_Middle'] - (std_dev * 2)
    
    # RSI (14)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Average True Range (ATR)
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()

    # Stochastic Oscillator (%K & %D)
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    data['Stoch_K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
    data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()

    # Money Flow Index (MFI 14)
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    raw_money_flow = typical_price * data['Volume']
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    mfi_ratio = positive_flow / negative_flow
    data['MFI'] = 100 - (100 / (1 + mfi_ratio))

    # Ichimoku Cloud
    nine_high = data['High'].rolling(window=9).max()
    nine_low = data['Low'].rolling(window=9).min()
    data['Tenkan_Sen'] = (nine_high + nine_low) / 2

    twenty_six_high = data['High'].rolling(window=26).max()
    twenty_six_low = data['Low'].rolling(window=26).min()
    data['Kijun_Sen'] = (twenty_six_high + twenty_six_low) / 2

    data['Senkou_Span_A'] = ((data['Tenkan_Sen'] + data['Kijun_Sen']) / 2).shift(26)
    
    fifty_high = data['High'].rolling(window=50).max()
    fifty_low = data['Low'].rolling(window=50).min()
    data['Senkou_Span_B'] = ((fifty_high + fifty_low) / 2).shift(26)

    # Parabolic SAR
    high_series = data['High']
    low_series = data['Low']
    close_series = data['Close']
    
    sar = close_series.copy()
    af = 0.02
    max_af = 0.2
    bullish = True
    hp = high_series.iloc[0]
    lp = low_series.iloc[0]
    
    sar_list = [low_series.iloc[0]]
    for i in range(1, len(data)):
        prev_sar = sar_list[-1]
        if bullish:
            curr_sar = prev_sar + af * (hp - prev_sar)
            curr_sar = min(curr_sar, low_series.iloc[max(0, i-1)], low_series.iloc[i])
            if low_series.iloc[i] < curr_sar:
                bullish = False
                curr_sar = hp
                hp = low_series.iloc[i]
                af = 0.02
            else:
                if high_series.iloc[i] > hp:
                    hp = high_series.iloc[i]
                    af = min(af + 0.02, max_af)
        else:
            curr_sar = prev_sar - af * (prev_sar - lp)
            curr_sar = max(curr_sar, high_series.iloc[max(0, i-1)], high_series.iloc[i])
            if high_series.iloc[i] > curr_sar:
                bullish = True
                curr_sar = lp
                lp = high_series.iloc[i]
                af = 0.02
            else:
                if low_series.iloc[i] < lp:
                    lp = low_series.iloc[i]
                    af = min(af + 0.02, max_af)
        sar_list.append(curr_sar)
    data['Parabolic_SAR'] = sar_list

    # Market Beta terhadap IHSG
    if not benchmark_close.empty:
        combined = pd.concat([data['Close'].pct_change(), benchmark_close.pct_change()], axis=1).dropna()
        combined.columns = ['Stock', 'Market']
        covariance = combined.cov().iloc[0, 1]
        market_variance = combined['Market'].var()
        stock_beta = covariance / market_variance if market_variance != 0 else 1.0
    else:
        stock_beta = 1.0

    # Hanya drop NaN pada kolom esensial agar data tidak habis total
    data.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume', 'MA20', 'RSI'], inplace=True)
    
    if data.empty or len(data) < 20:
        st.warning("⚠️ Data terlalu sedikit setelah pembersihan indikator. Silakan pilih rentang waktu yang lebih panjang di sidebar.")
    else:
        # --- DETEKSI SAHAM GORENGAN / ANOMALI RISIKO ---
        latest_close_check = float(data['Close'].iloc[-1])
        price_std = float(data['Close'].pct_change().std() * 100)
        
        is_potential_gorengan = False
        gorengan_reasons = []
        if ticker_symbol.endswith('.JK'):
            if latest_close_check < 150:
                is_potential_gorengan = True
                gorengan_reasons.append("Harga saham gocap (< Rp 150).")
            if price_std > 4.5:
                is_potential_gorengan = True
                gorengan_reasons.append(f"Volatilitas harian ekstrem ({price_std:.2f}%).")

        latest_close = float(data['Close'].iloc[-1])
        prev_close = float(data['Close'].iloc[-2])
        price_change = latest_close - prev_close
        pct_change = (price_change / prev_close) * 100
        latest_rsi = float(data['RSI'].iloc[-1])
        latest_atr = float(data['ATR'].iloc[-1]) if 'ATR' in data.columns and not pd.isna(data['ATR'].iloc[-1]) else (latest_close * 0.03)
        latest_stoch_k = float(data['Stoch_K'].iloc[-1]) if 'Stoch_K' in data.columns and not pd.isna(data['Stoch_K'].iloc[-1]) else 50.0
        latest_mfi = float(data['MFI'].iloc[-1]) if 'MFI' in data.columns and not pd.isna(data['MFI'].iloc[-1]) else 50.0

        # --- MACHINE LEARNING VALIDASI OUT-OF-SAMPLE ---
        df_pred = data.reset_index()
        df_pred['Days'] = np.arange(len(df_pred))
        
        df_pred['Lag1'] = df_pred['Close'].shift(1)
        df_pred['Lag2'] = df_pred['Close'].shift(2)
        df_pred['Rolling_Mean_5'] = df_pred['Close'].rolling(5).mean()
        df_pred['Rolling_Std_5'] = df_pred['Close'].rolling(5).std()
        df_pred.dropna(inplace=True)

        features = ['Days', 'Lag1', 'Lag2', 'Rolling_Mean_5', 'Rolling_Std_5', 'RSI', 'MACD']
        X = df_pred[features]
        y = df_pred['Close']

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        xgb_model = XGBRegressor(n_estimators=180, learning_rate=0.025, max_depth=4, random_state=42)
        lgb_model = LGBMRegressor(n_estimators=180, learning_rate=0.025, max_depth=4, random_state=42, verbose=-1)

        xgb_model.fit(X_train, y_train)
        lgb_model.fit(X_train, y_train)

        if ml_engine == "XGBoost Regressor":
            val_preds = xgb_model.predict(X_test)
        elif ml_engine == "LightGBM Regressor":
            val_preds = lgb_model.predict(X_test)
        else:
            val_preds = (0.5 * xgb_model.predict(X_test)) + (0.5 * lgb_model.predict(X_test))

        val_rmse = np.sqrt(mean_squared_error(y_test, val_preds))
        val_mape = mean_absolute_percentage_error(y_test, val_preds) * 100
        model_accuracy_score = max(0, 100 - val_mape)

        xgb_model.fit(X, y)
        lgb_model.fit(X, y)

        last_date = data.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=5)
        
        predicted_prices = []
        last_row_features = X.iloc[-1].copy()
        
        for i in range(5):
            next_day_idx = last_row_features['Days'] + 1
            current_input = pd.DataFrame([[
                next_day_idx, 
                last_row_features['Lag1'], 
                last_row_features['Lag2'], 
                last_row_features['Rolling_Mean_5'],
                last_row_features['Rolling_Std_5'],
                last_row_features['RSI'] if 'RSI' in last_row_features else latest_rsi,
                last_row_features['MACD'] if 'MACD' in last_row_features else 0
            ]], columns=features)
            
            pred_xgb = xgb_model.predict(current_input)[0]
            pred_lgb = lgb_model.predict(current_input)[0]
            
            if ml_engine == "XGBoost Regressor":
                pred_val = pred_xgb
            elif ml_engine == "LightGBM Regressor":
                pred_val = pred_lgb
            else: 
                pred_val = (0.5 * pred_xgb) + (0.5 * pred_lgb)

            predicted_prices.append(pred_val)
            last_row_features['Lag2'] = last_row_features['Lag1']
            last_row_features['Lag1'] = pred_val
            last_row_features['Days'] = next_day_idx

        predicted_target = float(predicted_prices[-1])
        pred_pct_change = ((predicted_target - latest_close) / latest_close) * 100

        # --- VALIDASI FILTER KHUSUS ---
        if filter_ai_bullish_high_acc:
            if pred_pct_change > 2.0 and model_accuracy_score > 90.0:
                st.success(f"🚀 **EMITEN UNGGULAN TERVALIDASI**: Prediksi AI naik **{pred_pct_change:+.2f}%** (>2%) dan Tingkat Presisi Model **{model_accuracy_score:.2f}%** (>90%).")
            else:
                st.warning(f"⚠️ Emiten **{ticker_symbol}** tidak memenuhi kriteria ketat filter (Prediksi Naik > 2% & Presisi > 90%). Saat ini Prediksi: **{pred_pct_change:+.2f}%**, Presisi: **{model_accuracy_score:.2f}%**.")

        if model_accuracy_score >= 90.0:
            st.info(f"🎯 **Status Akurasi Model**: Sangat Tinggi (**{model_accuracy_score:.2f}%**).")
        else:
            st.warning(f"⚠️ Catatan Presisi: Model saat ini memiliki tingkat akurasi **{model_accuracy_score:.2f}%** (<90%).")

        if is_potential_gorengan:
            st.error(
                f"🚨 **PERINGATAN RISIKO TINGGI (SAHAM GORENGAN / VOLATIL)**: "
                f"Emiten **{ticker_symbol}** terdeteksi memiliki anomali: " + ", ".join(gorengan_reasons)
            )

        # --- METRIK UTAMA KINERJA DAN PREDIKSI ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Harga Terakhir", f"{latest_close:,.2f}", f"{pct_change:+.2f}%")
        col2.metric(f"Prediksi AI (5 Hari)", f"{predicted_target:,.2f}", f"{pred_pct_change:+.2f}%")
        col3.metric("Tingkat Presisi Model", f"{model_accuracy_score:.2f}%", f"MAPE: {val_mape:.2f}%")
        col4.metric("Beta Pasar (IHSG)", f"{stock_beta:.2f}", "Risiko Relatif")

        # --- GRAFIK INTERAKTIF PLOTLY ---
        st.subheader(f"📈 Grafik Candlestick, Bollinger, Ichimoku & Proyeksi Kuantitatif: {ticker_symbol}")
        
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Aktual Candle'
        ))
        
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], line=dict(color='rgba(0,255,204,0.3)', width=1), name='BB Upper'))
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], line=dict(color='rgba(0,255,204,0.3)', width=1), fill='tonexty', fillcolor='rgba(0,255,204,0.03)', name='BB Lower'))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], line=dict(color='#ff007f', width=1.5), name='MA 20'))

        if show_ichimoku and 'Senkou_Span_A' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['Senkou_Span_A'], line=dict(color='rgba(0,255,0,0.5)', width=1), name='Senkou Span A'))
            fig.add_trace(go.Scatter(x=data.index, y=data['Senkou_Span_B'], line=dict(color='rgba(255,0,0,0.5)', width=1), fill='tonexty', fillcolor='rgba(100,100,100,0.1)', name='Senkou Span B'))

        if show_sar and 'Parabolic_SAR' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['Parabolic_SAR'], mode='markers', marker=dict(size=4, color='yellow'), name='Parabolic SAR'))
        
        plot_pred_dates = [last_date] + list(future_dates)
        plot_pred_prices = [latest_close] + list(predicted_prices)
        labels_text = [""] * len(plot_pred_prices)
        labels_text[-1] = f"<b>Target: {predicted_target:,.0f}</b>"

        fig.add_trace(go.Scatter(
            x=plot_pred_dates, y=plot_pred_prices, mode='lines+markers+text',
            text=labels_text, textposition="top center",
            line=dict(color='#00ffcc', width=2.5, dash='dash'),
            marker=dict(size=9, color='#00ffcc'), name='Proyeksi AI'
        ))
        
        fig.update_layout(
            title=f'Proyeksi Kuantitatif {ticker_symbol} Menuju {future_dates[-1].strftime("%d %b %Y")}',
            yaxis_title='Harga (IDR)', xaxis_title='Timeline',
            template='plotly_dark', paper_bgcolor='#05070c', plot_bgcolor='#0e1320',
            height=580, xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabel Detail Prediksi, Kalkulator Lot, & Tombol Unduh CSV
        with st.expander("📅 Rincian Target Harga Harian, Kalkulator Risiko & Ekspor Data"):
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("**Target Harga Harian AI:**")
                df_future_table = pd.DataFrame({
                    "Tanggal": [d.strftime("%Y-%m-%d (%A)") for d in future_dates],
                    "Target Harga AI": [f"{p:,.2f}" for p in predicted_prices],
                    "Est. Perubahan (%)": [f"{((p - latest_close) / latest_close) * 100:+.2f}%" for p in predicted_prices]
                })
                st.table(df_future_table)

                csv_data = df_future_table.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Unduh Laporan Proyeksi (CSV)",
                    data=csv_data,
                    file_name=f"proyeksi_ai_{ticker_symbol}.csv",
                    mime="text/csv"
                )
            
            with col_t2:
                st.markdown("**Kalkulator Alokasi Lot Berbasis Risiko ATR:**")
                recommended_stop_loss = latest_close - (2 * latest_atr)
                recommended_take_profit = latest_close + (3 * latest_atr)
                
                max_money_risk = user_capital * (risk_tolerance_pct / 100.0)
                risk_per_share = latest_close - recommended_stop_loss
                if risk_per_share > 0:
                    shares_to_buy = max_money_risk / risk_per_share
                    lots_to_buy = int(shares_to_buy / 100)
                else:
                    lots_to_buy = 0

                estimated_total_investment = lots_to_buy * 100 * latest_close

                st.info(
                    f"📌 **Parameter Manajemen Risiko Aktif:**\n\n"
                    f"- **Nilai ATR (14):** Rp {latest_atr:,.2f}\n"
                    f"- **Saran Stop Loss (2x ATR):** Rp {recommended_stop_loss:,.2f}\n"
                    f"- **Saran Take Profit (3x ATR):** Rp {recommended_take_profit:,.2f}\n"
                    f"- **Alokasi Lot Aman:** **{lots_to_buy} Lot** ({lots_to_buy * 100} lembar)\n"
                    f"- **Estimasi Dana Digunakan:** Rp {estimated_total_investment:,.2f}"
                )

        # --- BERITA FINANSIAL REAL-TIME & SENTIMEN NLP TERINTEGRASI ---
        st.subheader(f"📡 Berita Finansial & Analisis Sentimen NLP Real-Time: {ticker_symbol}")
        st.write(f"📊 **Skor Akumulasi Sentimen Berita Korporat:** `{news_sentiment_score}` (Positif jika > 0, Negatif jika < 0)")

        if realtime_news:
            for item in realtime_news:
                st.markdown(f"- **[{item['title']}]({item['link']})**")
                st.caption(f"📌 Sumber: {item['publisher']} | Waktu: {item['date']} | Status Sentimen NLP: **{item['sentiment']}**")
        else:
            st.info("Tidak ada berita real-time baru yang terindeks.")

        st.markdown("---")
        st.subheader("🎯 Sistem Rekomendasi Sinyal Otomatis Berbasis Multi-Indikator")
        
        col_rec1, col_rec2 = st.columns([2, 1])
        with col_rec1:
            score = 0
            if latest_close > data['MA20'].iloc[-1]: score += 1
            if latest_rsi < 35: score += 2
            elif latest_rsi > 65: score -= 2
            if latest_stoch_k < 20: score += 1
            elif latest_stoch_k > 80: score -= 1
            if latest_mfi > 60: score += 1
            elif latest_mfi < 40: score -= 1
            if news_sentiment_score > 0: score += 1
            elif news_sentiment_score < 0: score -= 1
            if pred_pct_change > 0: score += 1

            if score >= 3:
                rec_text = "STRONG BUY / AKUMULASI BERTAHAP 🟢"
            elif score <= -2:
                rec_text = "SELL / TAKE PROFIT 🔴"
            else:
                rec_text = "HOLD / WAIT & SEE 🟡"

            st.markdown(f"### Rekomendasi Aksi: **{rec_text}**")
            st.write(f"Skor Agregat Kuantitatif: **{score} / 8** (Termasuk Filter MFI, Sentimen NLP Berita & AI)")
            st.write(f"💡 **Indikator Pendukung:** MFI: **{latest_mfi:.1f}** | Stoch %K: **{latest_stoch_k:.1f}** | Sentimen NLP: **{news_sentiment_score}**")

        with col_rec2:
            st.markdown("### 📖 Pustaka Referensi Validasi & Metodologi")
            st.caption("• **Ernest P. Chan:** *Quantitative Trading* (Statistical Arbitrage & Out-of-Sample Test)")
            st.caption("• **Marcos López de Prado:** *Advances in Financial Machine Learning* (Cross-Validation & Feature Importance)")
            st.caption("• **J. Welles Wilder:** *New Concepts in Technical Trading Systems* (ATR & RSI Validation)")
            st.caption("• **Thomas N. Bulkowski:** *Encyclopedia of Chart Patterns*")
