import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import feedparser
import urllib.parse
from streamlit_autorefresh import st_autorefresh

# Mengimpor model Machine Learning berbasis Gradient Boosting (XGBoost & LightGBM)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="AI Cyber-Stock Predictive Analysis & News Sentiment Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Konfigurasi Auto-Refresh Real-Time (Setiap 60 detik)
st_autorefresh(interval=60 * 1000, key="datarefresh")

# --- KUSTOMISASI CSS TEMA NEON FUTURISTIK & KETERBACAAN TEKS ---
st.markdown("""
    <style>
    /* Global Background & Font Terang */
    .stApp {
        background-color: #05070c;
        color: #ffffff;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Neon Glow Headers yang Lebih Tajam */
    h1, h2, h3 {
        color: #00ffcc !important;
        text-shadow: 0 0 8px rgba(0, 255, 204, 0.8);
        font-weight: bold;
    }
    
    /* Sidebar Styling agar Kontras */
    section[data-testid="stSidebar"] {
        background-color: #0e1320;
        border-right: 2px solid #00ffcc55;
    }
    
    /* Memperjelas Semua Label & Teks di Sidebar */
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
        font-weight: bold !important;
    }
    
    /* Kotak Input, Selectbox, dan Text Input Lebih Terang */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #161f33 !important;
        color: #00ffcc !important;
        font-weight: bold !important;
        border: 1px solid #00ffcc !important;
    }
    
    /* Placeholder & Teks Dropdown */
    div[data-baseweb="select"] span {
        color: #00ffcc !important;
        font-weight: bold !important;
    }
    
    /* Metric Cards dengan Pendaran Neon Jelas */
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
    
    /* Memperjelas Teks Berita Real-Time di Bawah */
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
    sound_html = """
        <audio autoplay>
          <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg">
        </audio>
    """
    st.markdown(sound_html, unsafe_allow_html=True)

play_neon_sound()

# Data Referensi Buku Literasi Keuangan & Analisis Teknikal (Embedded)
@st.cache_data
def load_book_references():
    return {
        "reference_books": [
            {
                "title": "Encyclopedia of Chart Patterns",
                "author": "Thomas N. Bulkowski",
                "year": 2021,
                "core_principles": [
                    "Validasi breakout harus dikonfirmasi oleh lonjakan volume perdagangan.",
                    "Perhatikan pola throwback dan pullback untuk meminimalkan risiko false breakout."
                ]
            },
            {
                "title": "The Intelligent Investor",
                "author": "Benjamin Graham",
                "year": 1949,
                "core_principles": [
                    "Margin of Safety: Selalu berinvestasi pada perusahaan dengan fundamental kuat dan harga wajar.",
                    "Hindari spekulasi jangka pendek pada saham berisiko tinggi atau tanpa likuiditas memadai."
                ]
            }
        ],
        "patterns_rules": {
            "RSI": "Indikator momentum untuk mendeteksi area Overbought (>70) dan Oversold (<30).",
            "MACD": "Mengukur konvergensi dan divergensi garis rata-rata pergerakan untuk sinyal tren."
        }
    }

book_data = load_book_references()

# Sidebar Navigasi & Input Parameter Saham (Daftar Saham Diperluas)
st.sidebar.header("⚡ NEON CONFIG: SENSOR PASAR")

popular_stocks = {
    "BBCA – Bank Central Asia Tbk": "BBCA.JK",
    "BBRI – Bank Rakyat Indonesia Tbk": "BBRI.JK",
    "BMRI – Bank Mandiri Tbk": "BMRI.JK",
    "BBNI – Bank Negara Indonesia Tbk": "BBNI.JK",
    "BRIS – Bank Syariah Indonesia Tbk": "BRIS.JK",
    "ASII – Astra International Tbk": "ASII.JK",
    "TLKM – Telkom Indonesia Tbk": "TLKM.JK",
    "UNVR – Unilever Indonesia Tbk": "UNVR.JK",
    "ICBP – Indofood CBP Sukses Makmur Tbk": "ICBP.JK",
    "INDF – Indofood Sukses Makmur Tbk": "INDF.JK",
    "KLBF – Kalbe Farma Tbk": "KLBF.JK",
    "GOTO – GoTo Gojek Tokopedia Tbk": "GOTO.JK",
    "ARTO – Bank Jago Tbk": "ARTO.JK",
    "ADRO – Adaro Energy Indonesia Tbk": "ADRO.JK",
    "PTBA – Bukit Asam Tbk": "PTBA.JK",
    "ANTM – Aneka Tambang Tbk": "ANTM.JK",
    "MDKA – Merdeka Copper Gold Tbk": "MDKA.JK",
    "INCO – Vale Indonesia Tbk": "INCO.JK",
    "PGAS – Perusahaan Gas Negara Tbk": "PGAS.JK",
    "JSMR – Jasa Marga Tbk": "JSMR.JK",
    "INKP – Indah Kiat Pulp & Paper Tbk": "INKP.JK",
    "TKIM – Pabrik Kertas Tjiwi Kimia Tbk": "TKIM.JK",
    "SMGR – Semen Indonesia Tbk": "SMGR.JK",
    "INTP – Indocement Tunggal Prakarsa Tbk": "INTP.JK",
    "MYOR – Mayora Indah Tbk": "MYOR.JK",
    "UNTR – United Tractors Tbk": "UNTR.JK",
    "MEDC – Medco Energi Internasional Tbk": "MEDC.JK",
    "HRUM – Harum Energy Tbk": "HRUM.JK",
    "MAPI – Mitra Adiperkasa Tbk": "MAPI.JK",
    "EXCL – XL Axiata Tbk": "EXCL.JK",
    "ISAT – Indosat Tbk": "ISAT.JK",
    "TOWR – Sarana Menara Nusantara Tbk": "TOWR.JK",
    "BREN – Barito Renewables Energy Tbk": "BREN.JK",
    "AMMN – Amman Mineral Internasional Tbk": "AMMN.JK",
    "CUAN – Petrindo Jaya Kreasi Tbk": "CUAN.JK"
}

stock_choice = st.sidebar.selectbox("Pilih Emiten Unggulan BEI", options=list(popular_stocks.keys()))
selected_ticker_default = popular_stocks[stock_choice]

ticker_symbol = st.sidebar.text_input("Atau Ketik Kode Saham IDX (Format: KODE.JK)", value=selected_ticker_default)
ticker_symbol = ticker_symbol.strip().upper()

# Opsi Pemilihan Algoritma Machine Learning
ml_engine = st.sidebar.selectbox(
    "Pilih Neural Engine AI",
    ["Ensemble (XGBoost + LightGBM Hybrid)", "XGBoost Regressor", "LightGBM Regressor"],
    index=0
)

period_option = st.sidebar.selectbox("Rentang Waktu Sensor", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
interval_option = st.sidebar.selectbox("Interval Candle", ["1d", "1wk", "1mo"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Status Sistem Real-Time")
st.sidebar.success("🟢 Auto-Refresh Aktif (60s)\n🔊 Audio Alert Enabled\n⚡ XGBoost/LightGBM Synced")

st.title("⚡ NEON AI: Live Stock Predictive & Sentiment Dashboard")
st.markdown(
    f"Sistem analitik pasar saham real-time dengan pemindaian berita otomatis dan prediksi *Gradient Boosting* untuk emiten **{ticker_symbol}**."
)

@st.cache_data(ttl=60)
def fetch_stock_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
    except Exception as e:
        return pd.DataFrame()

# --- FUNGSI FETCH BERITA REAL-TIME DARI RSS ---
@st.cache_data(ttl=300)
def fetch_realtime_news(ticker):
    clean_code = ticker.replace(".JK", "").lower()
    news_items = []
    
    query = urllib.parse.quote(f"saham {clean_code} OR {ticker}")
    rss_url = f"https://news.google.com/rss/search?q={query}+when:7d&hl=id&gl=ID&ceid=ID:id"
    
    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:5]:
            title = entry.title
            link = entry.link
            published = entry.published if hasattr(entry, 'published') else "Terbaru"
            
            source = "Portal Finansial"
            if "kontan" in link.lower() or "kontan" in title.lower():
                source = "Kontan Investasi"
            elif "cnbcindonesia" in link.lower() or "cnbc" in title.lower():
                source = "CNBC Indonesia"
            elif "bisnis" in link.lower():
                source = "Bisnis Market"
            elif "idx" in link.lower():
                source = "Keterbukaan Informasi IDX"

            news_items.append({
                "title": title,
                "link": link,
                "publisher": source,
                "date": published
            })
    except Exception:
        pass

    if not news_items:
        try:
            yf_obj = yf.Ticker(ticker)
            for item in yf_obj.news[:3]:
                news_items.append({
                    "title": item.get('title', 'Berita Pasar'),
                    "link": item.get('link', '#'),
                    "publisher": item.get('publisher', 'Yahoo Finance / IDX'),
                    "date": "Hari ini"
                })
        except Exception:
            pass

    return news_items

data = fetch_stock_data(ticker_symbol, period_option, interval_option)
realtime_news = fetch_realtime_news(ticker_symbol)

# --- VALIDASI KEAMANAN DATA AWAL ---
if data.empty or len(data) < 30:
    st.warning(
        f"⚠️ Data untuk ticker **{ticker_symbol}** dengan rentang waktu **{period_option}** tidak mencukupi. "
        f"Model Machine Learning memerlukan minimal 30 baris data historis. Silakan pilih rentang waktu yang lebih panjang."
    )
else:
    # Perhitungan Indikator Teknikal
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    data.dropna(inplace=True)
    
    if data.empty or len(data) < 10:
        st.warning("⚠️ Data terlalu sedikit setelah dibersihkan dari nilai kosong (NaN).")
    else:
        # --- DETEKSI SAHAM GORENGAN ---
        latest_close_check = float(data['Close'].iloc[-1])
        price_std = float(data['Close'].pct_change().std() * 100)
        
        is_potential_gorengan = False
        gorengan_reasons = []
        
        if ticker_symbol.endswith('.JK'):
            if latest_close_check < 150:
                is_potential_gorengan = True
                gorengan_reasons.append("Harga saham gocap (< Rp 150).")
            if price_std > 5.0:
                is_potential_gorengan = True
                gorengan_reasons.append(f"Volatilitas ekstrem ({price_std:.2f}%).")

        latest_close = float(data['Close'].iloc[-1])
        prev_close = float(data['Close'].iloc[-2])
        price_change = latest_close - prev_close
        pct_change = (price_change / prev_close) * 100
        latest_rsi = float(data['RSI'].iloc[-1])
        latest_ma20 = float(data['MA20'].iloc[-1])
        latest_ma50 = float(data['MA50'].iloc[-1])
        latest_macd = float(data['MACD'].iloc[-1])
        latest_signal = float(data['Signal_Line'].iloc[-1])

        # --- ANALISIS SENTIMEN BERITA REAL-TIME ---
        news_sentiment_score = 0
        positive_keywords = ["naik", "lonjak", "tumbuh", "laba", "dividen", "positif", "beli", "akuisisi", "ekspansi", "rebound", "menguat"]
        negative_keywords = ["anjlok", "turun", "rugi", "koreksi", "jual", "beban", "sanksi", "melemah", "lesu", "default"]

        for n in realtime_news:
            title_lower = n['title'].lower()
            for pk in positive_keywords:
                if pk in title_lower:
                    news_sentiment_score += 1
            for nk in negative_keywords:
                if nk in title_lower:
                    news_sentiment_score -= 1

        # --- MACHINE LEARNING (XGBOOST / LIGHTGBM) ---
        df_pred = data.reset_index()
        df_pred['Days'] = np.arange(len(df_pred))
        
        df_pred['Lag1'] = df_pred['Close'].shift(1)
        df_pred['Lag2'] = df_pred['Close'].shift(2)
        df_pred['Rolling_Mean_5'] = df_pred['Close'].rolling(5).mean()
        df_pred['Rolling_Std_5'] = df_pred['Close'].rolling(5).std()
        df_pred.dropna(inplace=True)

        features = ['Days', 'Lag1', 'Lag2', 'Rolling_Mean_5', 'Rolling_Std_5']
        X = df_pred[features]
        y = df_pred['Close']

        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        lgb_model = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)

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
                last_row_features['Rolling_Std_5']
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

        if is_potential_gorengan:
            st.error(
                f"🚨 **PERINGATAN RISIKO TINGGI (SAHAM GORENGAN)**: "
                f"Emiten **{ticker_symbol}** terdeteksi memiliki anomali: " + ", ".join(gorengan_reasons)
            )

        # Metrik Utama (Gaya Neon)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Harga Terakhir", f"{latest_close:,.2f}", f"{pct_change:+.2f}%")
        col2.metric(f"Prediksi AI (5 Hari)", f"{predicted_target:,.2f}", f"{pred_pct_change:+.2f}%")
        col3.metric("RSI (14)", f"{latest_rsi:.2f}")
        col4.metric("Sentimen Real-Time", "Bullish 🟢" if news_sentiment_score > 0 else ("Bearish 🔴" if news_sentiment_score < 0 else "Neutral 🟡"))

        # --- GRAFIK PLOTLY TEMA GELAP NEON ---
        st.subheader(f"📈 Grafik Candlestick & Proyeksi Neon AI: {ticker_symbol}")
        
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Aktual Candle'
        ))
        
        fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], line=dict(color='#ff007f', width=1.5), name='MA 20'))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], line=dict(color='#00ffff', width=1.5), name='MA 50'))
        
        plot_pred_dates = [last_date] + list(future_dates)
        plot_pred_prices = [latest_close] + list(predicted_prices)
        
        labels_text = [""] * len(plot_pred_prices)
        labels_text[-1] = f"<b>Target: {predicted_target:,.0f}</b>"

        fig.add_trace(go.Scatter(
            x=plot_pred_dates,
            y=plot_pred_prices,
            mode='lines+markers+text',
            text=labels_text,
            textposition="top center",
            line=dict(color='#00ffcc', width=2.5, dash='dash'),
            marker=dict(size=9, color='#00ffcc'),
            name='Proyeksi AI'
        ))
        
        fig.update_layout(
            title=f'Analisis Proyeksi Harga {ticker_symbol} Menuju {future_dates[-1].strftime("%d %b %Y")}',
            yaxis_title='Harga (IDR)',
            xaxis_title='Timeline',
            template='plotly_dark',
            paper_bgcolor='#05070c',
            plot_bgcolor='#0e1320',
            height=550,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabel Detail Prediksi
        with st.expander("📅 Rincian Target Harga Harian (5 Hari Mendatang)"):
            df_future_table = pd.DataFrame({
                "Tanggal": [d.strftime("%Y-%m-%d (%A)") for d in future_dates],
                "Target Harga AI": [f"{p:,.2f}" for p in predicted_prices],
                "Est. Perubahan (%)": [f"{((p - latest_close) / latest_close) * 100:+.2f}%" for p in predicted_prices]
            })
            st.table(df_future_table)

        # --- BERITA FINANSIAL REAL-TIME ---
        st.subheader(f"📡 Berita Finansial & Sentimen Real-Time: {ticker_symbol}")
        if realtime_news:
            news_cols = st.columns(min(3, len(realtime_news[:3])))
            for idx, item in enumerate(realtime_news[:3]):
                with news_cols[idx]:
                    st.markdown(f"**[{item['title']}]({item['link']})**")
                    st.caption(f"📌 {item['publisher']} | {item['date']}")
        else:
            st.info("Tidak ada berita real-time baru yang terindeks.")

        st.markdown("---")
        st.subheader("🎯 Sistem Rekomendasi Sinyal Otomatis")
        
        col_rec1, col_rec2 = st.columns([2, 1])
        with col_rec1:
            score = 0
            if latest_close > latest_ma20: score += 1
            if latest_rsi < 30: score += 2
            elif latest_rsi > 70: score -= 2
            if latest_macd > latest_signal: score += 1
            if news_sentiment_score > 0: score += 1
            if pred_pct_change > 0: score += 1

            if score >= 2:
                rec_text = "STRONG BUY / AKUMULASI 🟢"
            elif score <= -2:
                rec_text = "SELL / TAKE PROFIT 🔴"
            else:
                rec_text = "HOLD / WAIT & SEE 🟡"

            st.markdown(f"### Rekomendasi Aksi: **{rec_text}**")
            st.write(f"Skor Agregat Sistem: **{score} / 6**")

        with col_rec2:
            st.markdown("### 📖 Pustaka Referensi")
            st.caption("• Benjamin Graham (The Intelligent Investor)")
            st.caption("• Thomas N. Bulkowski (Chart Patterns)")
