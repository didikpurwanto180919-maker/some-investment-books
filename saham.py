import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from sklearn.linear_model import LinearRegression

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="AI Stock Predictive Analysis & Filter Dashboard",
    page_icon="📈",
    layout="wide"
)

# Konfigurasi Auto-Refresh Real-Time (Setiap 60 detik)
st_autorefresh(interval=60 * 1000, key="datarefresh")

# Fungsi untuk memuat referensi buku dari direktori lokal / GitHub
@st.cache_data
def load_book_references():
    path = "books_reference/bulkowski_patterns.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"reference_books": [], "patterns_rules": {}}

book_data = load_book_references()

# Sidebar Navigasi & Input Parameter Saham
st.sidebar.header("⚙️ Konfigurasi & Pemilihan Saham")

popular_stocks = {
    "BBCA.JK (Bank Central Asia - Bluechip)": "BBCA.JK",
    "BBRI.JK (Bank Rakyat Indonesia - Bluechip)": "BBRI.JK",
    "BMRI.JK (Bank Mandiri - Bluechip)": "BMRI.JK",
    "ASII.JK (Astra International)": "ASII.JK",
    "TLKM.JK (Telkom Indonesia)": "TLKM.JK",
    "ADRO.JK (Adaro Energy)": "ADRO.JK",
    "AAPL (Apple Inc.)": "AAPL",
    "TSLA (Tesla Inc.)": "TSLA",
    "NVDA (NVIDIA Corp.)": "NVDA"
}

stock_choice = st.sidebar.selectbox("Pilih Saham Unggulan", options=list(popular_stocks.keys()))
selected_ticker_default = popular_stocks[stock_choice]

ticker_symbol = st.sidebar.text_input("Atau Ketik Kode Saham Lainnya (Yahoo Finance format)", value=selected_ticker_default)
ticker_symbol = ticker_symbol.strip().upper()

period_option = st.sidebar.selectbox("Rentang Waktu", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
interval_option = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("📚 Sumber Dasar Literatur")
st.sidebar.info(
    "Dasbor ini mengintegrasikan kaidah analisis dari buku-buku standar industri seperti "
    "*Visual Guide to Chart Patterns* (Thomas N. Bulkowski)[cite: 6], *The Intelligent Investor* (Benjamin Graham), "
    "dan *Trading in the Zone* (Mark Douglas)."
)

st.title("📊 Dasbor Prediksi & Filter Saham Presisi (Real-Time)")
st.markdown(
    f"Sistem analisis tren harga otomatis dengan proteksi deteksi saham spekulatif / gorengan untuk **{ticker_symbol}**."
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

data = fetch_stock_data(ticker_symbol, period_option, interval_option)

if data.empty or len(data) < 15:
    st.warning(f"⚠️ Data untuk ticker **{ticker_symbol}** tidak ditemukan atau kurang. Pastikan format penulisan benar (Contoh: `BBCA.JK`, `TLKM.JK`).")
else:
    # --- DETEKSI SAHAM GORENGAN / SPEKULATIF ---
    latest_close_check = float(data['Close'].iloc[-1])
    price_std = float(data['Close'].pct_change().std() * 100) # Volatilitas harian
    
    is_potential_gorengan = False
    gorengan_reasons = []
    
    if ticker_symbol.endswith('.JK'):
        if latest_close_check < 150:
            is_potential_gorengan = True
            gorengan_reasons.append("Harga saham gocap / nominal sangat kecil (< Rp 150).")
        if price_std > 5.0:
            is_potential_gorengan = True
            gorengan_reasons.append(f"Volatilitas harga harian sangat ekstrem ({price_std:.2f}%).")

    # Perhitungan Indikator Teknikal
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    # RSI Calculation (14)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD Calculation
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    data.dropna(inplace=True)
    
    latest_close = float(data['Close'].iloc[-1])
    prev_close = float(data['Close'].iloc[-2])
    price_change = latest_close - prev_close
    pct_change = (price_change / prev_close) * 100
    latest_rsi = float(data['RSI'].iloc[-1])
    latest_ma20 = float(data['MA20'].iloc[-1])
    latest_ma50 = float(data['MA50'].iloc[-1])
    latest_macd = float(data['MACD'].iloc[-1])
    latest_signal = float(data['Signal_Line'].iloc[-1])

    # --- SISTEM PREDIKSI HARGA BERBASIS MACHINE LEARNING (REGRESI LINIER 5 HARI KEDEPAN) ---
    df_pred = data.reset_index()
    df_pred['Days'] = np.arange(len(df_pred))
    X = df_pred[['Days']].tail(30)
    y = df_pred['Close'].tail(30)
    
    model = LinearRegression()
    model.fit(X, y)
    
    next_days = np.array([[len(df_pred) + i] for i in range(1, 6)])
    predicted_prices = model.predict(next_days)
    predicted_target = float(predicted_prices[-1])
    pred_pct_change = ((predicted_target - latest_close) / latest_close) * 100

    # Peringatan Jika Terdeteksi Saham Gorengan
    if is_potential_gorengan:
        st.error(
            f"🚨 **PERINGATAN RISIKO TINGGI (INDIKASI SAHAM GORENGAN/SPEKULATIF)**: "
            f"Saham **{ticker_symbol}** terdeteksi memiliki karakteristik berisiko tinggi karena: " + ", ".join(gorengan_reasons) + 
            " Sesuai prinsip *The Intelligent Investor* (Benjamin Graham), hindari saham spekulatif tanpa fundamental yang jelas."
        )

    # Metrik Utama
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Harga Terakhir", f"{latest_close:,.2f}", f"{pct_change:+.2f}%")
    col2.metric("Prediksi Target (5 Hari)", f"{predicted_target:,.2f}", f"{pred_pct_change:+.2f}%")
    col3.metric("RSI (14)", f"{latest_rsi:.2f}")
    col4.metric("Status Volatilitas", "Tinggi / Spekulatif" if is_potential_gorengan else "Normal / Stabil")

    # Visualisasi Grafik Candlestick & Indikator
    st.subheader(f"📉 Grafik Harga & Analisis Prediksi Tren: {ticker_symbol}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    ))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], line=dict(color='orange', width=1.5), name='MA 20'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], line=dict(color='blue', width=1.5), name='MA 50'))
    
    fig.update_layout(
        title=f'Pergerakan & Proyeksi Tren Harga Saham {ticker_symbol}',
        yaxis_title='Harga',
        xaxis_title='Tanggal',
        template='plotly_dark',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # Modul Sistem Prediksi & Rekomendasi Presisi
    st.subheader("🎯 Sistem Prediksi & Validasi Keputusan Otomatis")
    
    col_rec1, col_rec2 = st.columns([2, 1])
    
    with col_rec1:
        st.markdown("### Analisis Probabilitas Arah Harga")
        
        score = 0
        signals = []
        
        if is_potential_gorengan:
            score -= 3
            signals.append("✖ **Peringatan Sistem**: Saham dikategorikan spekulatif tinggi, mengabaikan analisis teknikal murni karena risiko tinggi.")

        if latest_close > latest_ma20:
            score += 1
            signals.append("✔ **Harga di atas MA20**: Momentum jangka pendek positif.")
        else:
            score -= 1
            signals.append("✖ **Harga di bawah MA20**: Tekanan jual jangka pendek.")
            
        if latest_rsi < 30:
            score += 2
            signals.append("✔ **RSI Oversold (< 30)**: Potensi kuat pembalikan arah naik (*rebound*).")
        elif latest_rsi > 70:
            score -= 2
            signals.append("✖ **RSI Overbought (> 70)**: Risiko koreksi tinggi sesuai aturan *throwback*[cite: 6].")
        else:
            signals.append("ℹ **RSI Netral**: Pasar bergerak stabil di koridor normal.")
            
        if latest_macd > latest_signal:
            score += 1
            signals.append("✔ **MACD Bullish Crossover**: Garis MACD di atas garis sinyal.")
        else:
            score -= 1
            signals.append("✖ **MACD Bearish Crossover**: Tekanan tren menurun mendominasi.")

        if pred_pct_change > 0:
            score += 1
            signals.append(f"✔ **Proyeksi Tren AI**: Model regresi memproyeksikan kenaikan sebesar **{pred_pct_change:.2f}%** dalam 5 hari.")
        else:
            score -= 1
            signals.append(f"✖ **Proyeksi Tren AI**: Model regresi memproyeksikan koreksi sebesar **{pred_pct_change:.2f}%** dalam 5 hari.")

        for sig in signals:
            st.markdown(f"- {sig}")
            
        if is_potential_gorengan:
            prediction_direction = "BERISIKO TINGGI (SPEKULATIF)"
            recommendation = "AVOID / JANGAN DIBELI"
            rec_color = "red"
        elif score >= 2:
            prediction_direction = "TREN NAIK (BULLISH)"
            recommendation = "STRONG BUY / AKUMULASI"
            rec_color = "green"
        elif score <= -2:
            prediction_direction = "TREN TURUN (BEARISH)"
            recommendation = "SELL / TAKE PROFIT"
            rec_color = "red"
        else:
            prediction_direction = "KONSOLIDASI / SIDEWAYS"
            recommendation = "HOLD / WAIT & SEE"
            rec_color = "orange"

        st.markdown(f"### Hasil Prediksi Arah: <span style='color:{rec_color}'>{prediction_direction}</span>", unsafe_allow_html=True)
        st.markdown(f"### Rekomendasi Aksi: <span style='color:{rec_color}'>{recommendation}</span>", unsafe_allow_html=True)

    with col_rec2:
        st.markdown("### 📖 Referensi Buku Terhubung")
        if book_data.get("reference_books"):
            for book in book_data["reference_books"][:2]:
                st.markdown(f"**{book['title']}** ({book['year']}) — *{book['author']}*")
                for rule in book['core_principles'][:1]:
                    st.caption(f"• {rule}")
                st.markdown("---")

    with st.expander("🔍 Lihat Detail Literatur Pendukung & Aturan Validasi"):
        st.json(book_data)
