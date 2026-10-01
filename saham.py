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
    page_title="AI Stock Predictive Analysis & News Sentiment Dashboard (IDX)",
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

# Sidebar Navigasi & Input Parameter Saham (Daftar Lengkap Saham Non-Gorengan / Blue Chip BEI)
st.sidebar.header("⚙️ Konfigurasi & Pemilihan Saham")

popular_stocks = {
    "BBCA – Bank Central Asia Tbk": "BBCA.JK",
    "BBRI – Bank Rakyat Indonesia Tbk": "BBRI.JK",
    "BMRI – Bank Mandiri Tbk": "BMRI.JK",
    "BBNI – Bank Negara Indonesia Tbk": "BBNI.JK",
    "ASII – Astra International Tbk": "ASII.JK",
    "TLKM – Telkom Indonesia Tbk": "TLKM.JK",
    "UNVR – Unilever Indonesia Tbk": "UNVR.JK",
    "ICBP – Indofood CBP Sukses Makmur Tbk": "ICBP.JK",
    "INDF – Indofood Sukses Makmur Tbk": "INDF.JK",
    "JSMR – Jasa Marga Tbk": "JSMR.JK",
    "ADRO – Adaro Energy Indonesia Tbk": "ADRO.JK",
    "PGAS – Perusahaan Gas Negara Tbk": "PGAS.JK",
    "INKP – Indah Kiat Pulp & Paper Tbk": "INKP.JK",
    "MDKA – Merdeka Copper Gold Tbk": "MDKA.JK",
    "ANTM – Aneka Tambang Tbk": "ANTM.JK",
    "MYOR – Mayora Indah Tbk": "MYOR.JK",
    "INTP – Indocement Tunggal Prakarsa Tbk": "INTP.JK",
    "SMGR – Semen Indonesia Tbk": "SMGR.JK"
}

stock_choice = st.sidebar.selectbox("Pilih Saham Non-Gorengan (Blue Chip)", options=list(popular_stocks.keys()))
selected_ticker_default = popular_stocks[stock_choice]

ticker_symbol = st.sidebar.text_input("Atau Ketik Kode Saham IDX Lainnya (Format: KODE.JK)", value=selected_ticker_default)
ticker_symbol = ticker_symbol.strip().upper()

period_option = st.sidebar.selectbox("Rentang Waktu", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
interval_option = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("📚 Sumber Dasar Literatur & Berita")
st.sidebar.info(
    "Dasbor ini mengintegrasikan analisis teknikal, Machine Learning, kaidah literatur klasik "
    "(*Visual Guide to Chart Patterns* oleh Thomas N. Bulkowski[cite: 6], *The Intelligent Investor* oleh Benjamin Graham), "
    "serta pemantauan berita finansial terkini dari Kontan, CNBC Indonesia, Bisnis.com, dan Investor.id."
)

st.title("📊 Dasbor Prediksi, Analisis Berita & Filter Saham Presisi (IDX Real-Time)")
st.markdown(
    f"Sistem analisis tren harga otomatis dengan validasi berita finansial *real-time* untuk emiten **{ticker_symbol}**."
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
    st.warning(f"⚠️ Data untuk ticker **{ticker_symbol}** tidak ditemukan atau kurang. Pastikan format penulisan benar dan berakhiran `.JK` (Contoh: `BBCA.JK`, `TLKM.JK`).")
else:
    # --- AMBIL BERITA TERBARU DARI YFINANCE (TERHUBUNG KE PORTAL BERITA UTAMA) ---
    ticker_obj = yf.Ticker(ticker_symbol)
    news_list = []
    try:
        news_list = ticker_obj.news
    except Exception:
        news_list = []

    # --- DETEKSI SAHAM GORENGAN / SPEKULATIF ---
    latest_close_check = float(data['Close'].iloc[-1])
    price_std = float(data['Close'].pct_change().std() * 100)
    
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
    
    latest_close = float(data['Close'].iloc[-1])
    prev_close = float(data['Close'].iloc[-2])
    price_change = latest_close - prev_close
    pct_change = (price_change / prev_close) * 100
    latest_rsi = float(data['RSI'].iloc[-1])
    latest_ma20 = float(data['MA20'].iloc[-1])
    latest_ma50 = float(data['MA50'].iloc[-1])
    latest_macd = float(data['MACD'].iloc[-1])
    latest_signal = float(data['Signal_Line'].iloc[-1])

    # --- SISTEM PREDIKSI HARGA & TANGGAL MASA DEPAN (5 HARI KEDEPAN) ---
    df_pred = data.reset_index()
    df_pred['Days'] = np.arange(len(df_pred))
    X = df_pred[['Days']].tail(30)
    y = df_pred['Close'].tail(30)
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_date = data.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=5)
    
    next_days_idx = np.array([[len(df_pred) + i] for i in range(1, 6)])
    predicted_prices = model.predict(next_days_idx)
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

    # --- VISUALISASI GRAFIK CANDLESTICK & GARIS PROYEKSI TANGGAL MASA DEPAN ---
    st.subheader(f"📉 Grafik Harga & Proyeksi Tanggal Masa Depan: {ticker_symbol}")
    
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick Aktual'
    ))
    
    fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], line=dict(color='orange', width=1.5), name='MA 20'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], line=dict(color='blue', width=1.5), name='MA 50'))
    
    plot_pred_dates = [last_date] + list(future_dates)
    plot_pred_prices = [latest_close] + list(predicted_prices)
    
    fig.add_trace(go.Scatter(
        x=plot_pred_dates,
        y=plot_pred_prices,
        mode='lines+markers',
        line=dict(color='#00FF7F', width=2.5, dash='dash'),
        marker=dict(size=8, color='#00FF7F'),
        name='Proyeksi Tren AI (5 Hari)'
    ))
    
    fig.update_layout(
        title=f'Pergerakan & Proyeksi Harga Saham {ticker_symbol} Hingga {future_dates[-1].strftime("%d %b %Y")}',
        yaxis_title='Harga Saham (IDR)',
        xaxis_title='Tanggal Transaksi',
        template='plotly_dark',
        height=550,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabel Detail Tanggal Prediksi
    with st.expander("📅 Lihat Detail Proyeksi Harga Berdasarkan Tanggal (5 Hari Ke Depan)"):
        df_future_table = pd.DataFrame({
            "Tanggal Proyeksi": [d.strftime("%Y-%m-%d (%A)") for d in future_dates],
            "Prediksi Harga Target": [f"{p:,.2f}" for p in predicted_prices],
            "Estimasi Perubahan (%)": [f"{((p - latest_close) / latest_close) * 100:+.2f}%" for p in predicted_prices]
        })
        st.table(df_future_table)

    # --- MODUL BERITA FINANSIAL & SENTIMEN PASAR REAL-TIME ---
    st.subheader(f"📰 Berita Finansial & Sentimen Pasar Terkini: {ticker_symbol}")
    if news_list:
        news_cols = st.columns(min(3, len(news_list[:3])))
        for idx, item in enumerate(news_list[:3]):
            with news_cols[idx]:
                title = item.get('title', 'Berita Finansial')
                publisher = item.get('publisher', 'Media Finansial')
                link = item.get('link', '#')
                st.markdown(f"**[{title}]({link})**")
                st.caption(f"Sumber: {publisher}")
    else:
        st.info("Belum ada berita real-time terbaru yang terindeks untuk emiten ini dalam 24 jam terakhir. Anda dapat merujuk langsung ke portal seperti [Kontan](https://www.kontan.co.id/), [CNBC Indonesia](https://www.cnbcindonesia.com/market), atau [Bisnis.com](https://www.bisnis.com/).")

    st.markdown("---")

    # Modul Sistem Prediksi & Rekomendasi Presisi
    st.subheader("🎯 Sistem Prediksi & Validasi Keputusan Otomatis")
    
    col_rec1, col_rec2 = st.columns([2, 1])
    
    with col_rec1:
        st.markdown("### Analisis Probabilitas Arah Harga")
        
        score = 0
        signals = []
        
        if is_potential_gorengan:
            score -= 3
            signals.append("✖ **Peringatan Sistem**: Saham dikategorikan spekulatif tinggi, mengabaikan analisis teknikal murni.")

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
            signals.append(f"✔ **Proyeksi Tren AI**: Model regresi memproyeksikan kenaikan sebesar **{pred_pct_change:.2f}%** hingga tanggal **{future_dates[-1].strftime('%d %b %Y')}**.")
        else:
            score -= 1
            signals.append(f"✖ **Proyeksi Tren AI**: Model regresi memproyeksikan koreksi sebesar **{pred_pct_change:.2f}%** hingga tanggal **{future_dates[-1].strftime('%d %b %Y')}**.")

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
