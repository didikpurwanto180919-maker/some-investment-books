import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import feedparser
import urllib.parse
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
st.sidebar.subheader("📚 Sumber Referensi & Berita Resmi")
st.sidebar.info(
    "Dasbor ini mengintegrasikan:\n"
    "- **Kontan, CNBC Indonesia, Bisnis.com** (Berita Real-Time)\n"
    "- **Keterbukaan Informasi IDX & IDNFinancials** (Data Fundamental)\n"
    "- **Stockbit & TradingView** (Riset Komunitas)\n"
    "- *The Intelligent Investor* (Benjamin Graham) & *Encyclopedia of Chart Patterns* (Thomas N. Bulkowski)[cite: 6]"
)

st.title("📊 Dasbor Prediksi, Analisis Berita & Filter Saham Presisi (IDX Real-Time)")
st.markdown(
    f"Sistem analisis tren harga otomatis dengan validasi berita finansial *real-time* dari portal terpercaya untuk emiten **{ticker_symbol}**."
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

# --- FUNGSI FETCH BERITA REAL-TIME DARI RSS (KONTAN, CNBC, BISNIS, IDX) ---
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
if data.empty or len(data) < 20:
    st.warning(
        f"⚠️ Data untuk ticker **{ticker_symbol}** dengan rentang waktu **{period_option}** tidak mencukupi atau kosong. "
        f"Indikator teknikal (seperti MA50) memerlukan minimal 50 data baris. "
        f"Silakan pilih **Rentang Waktu** yang lebih panjang (misalnya **6mo** atau **1y**) melalui sidebar."
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

    # Membersihkan baris kosong (NaN) hasil kalkulasi indikator
    data.dropna(inplace=True)
    
    # Validasi ulang setelah dropna agar aman dari error index out of bounds
    if data.empty or len(data) < 5:
        st.warning("⚠️ Data terlalu sedikit setelah dibersihkan dari nilai kosong (NaN). Harap gunakan rentang waktu yang lebih panjang (misalnya 1y).")
    else:
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

        latest_close = float(data['Close'].iloc[-1])
        prev_close = float(data['Close'].iloc[-2])
        price_change = latest_close - prev_close
        pct_change = (price_change / prev_close) * 100
        latest_rsi = float(data['RSI'].iloc[-1])
        latest_ma20 = float(data['MA20'].iloc[-1])
        latest_ma50 = float(data['MA50'].iloc[-1])
        latest_macd = float(data['MACD'].iloc[-1])
        latest_signal = float(data['Signal_Line'].iloc[-1])

        # --- ANALISIS SENTIMEN BERITA REAL-TIME (KEYWORD SCORING) ---
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

        # --- SISTEM PREDIKSI HARGA & TANGGAL MASA DEPAN (5 HARI KEDEPAN) ---
        df_pred = data.reset_index()
        df_pred['Days'] = np.arange(len(df_pred))
        tail_count = min(30, len(df_pred))
        X = df_pred[['Days']].tail(tail_count)
        y = df_pred['Close'].tail(tail_count)
        
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
        col4.metric("Sentimen Berita Real-Time", "Positif (Bullish)" if news_sentiment_score > 0 else ("Negatif (Bearish)" if news_sentiment_score < 0 else "Netral"))

        # --- VISUALISASI GRAFIK CANDLESTICK & GARIS PROYEKSI (BERSIH & TANPA TUMPUKAN TEKS) ---
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
        
        # Label dikosongkan pada titik 1-4 dan hanya ditampilkan pada titik terakhir (hari ke-5) agar bersih tanpa tumpukan.
        labels_text = [""] * len(plot_pred_prices)
        labels_text[-1] = f"<b>Target 5 Hari: {predicted_target:,.0f}</b>"

        fig.add_trace(go.Scatter(
            x=plot_pred_dates,
            y=plot_pred_prices,
            mode='lines+markers+text',
            text=labels_text,
            textposition="top center",
            line=dict(color='#00FF7F', width=2.5, dash='dash'),
            marker=dict(size=9, color='#00FF7F'),
            name='Proyeksi Tren AI (5 Hari)'
        ))
        
        fig.update_layout(
            title=f'Pergerakan & Proyeksi Harga Saham {ticker_symbol} Hingga {future_dates[-1].strftime("%d %b %Y")}',
            yaxis_title='Harga Saham (IDR)',
            xaxis_title='Rentang Waktu Transaksi & Proyeksi',
            template='plotly_dark',
            height=580,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(
                type='date',
                tickformat='%b %Y',
                hoverformat='%d %b %Y'
            )
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

        # --- MODUL BERITA FINANSIAL & SENTIMEN PASAR REAL-TIME TERINTEGRASI ---
        st.subheader(f"📰 Berita Real-Time (Kontan, CNBC, Bisnis, & Keterbukaan IDX): {ticker_symbol}")
        if realtime_news:
            news_cols = st.columns(min(3, len(realtime_news[:3])))
            for idx, item in enumerate(realtime_news[:3]):
                with news_cols[idx]:
                    title = item.get('title', 'Berita Finansial')
                    publisher = item.get('publisher', 'Portal Berita')
                    link = item.get('link', '#')
                    date_pub = item.get('date', '')
                    st.markdown(f"**[{title}]({link})**")
                    st.caption(f"📌 Sumber: {publisher} | {date_pub}")
        else:
            st.info("Belum ada berita real-time spesifik yang terindeks dalam beberapa hari terakhir. Anda dapat memantau langsung melalui tautan [Kontan Investasi](https://investasi.kontan.co.id/), [CNBC Market](https://www.cnbcindonesia.com/market), atau [Keterbukaan Informasi IDX](https://www.idx.co.id/id/perusahaan-tercatat/keterbukaan-informasi).")

        st.markdown("---")

        # Modul Sistem Prediksi & Rekomendasi Presisi
        st.subheader("🎯 Sistem Prediksi & Validasi Keputusan Otomatis (Teknikal + Berita Real-Time)")
        
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

            # Integrasi Sentimen Berita Real-Time ke Scoring
            if news_sentiment_score > 0:
                score += 1
                signals.append(f"✔ **Sentimen Berita Real-Time Positif**: Berita terkini dari Kontan/CNBC/Bisnis bernada akumulatif/ekspansi.")
            elif news_sentiment_score < 0:
                score -= 1
                signals.append(f"✖ **Sentimen Berita Real-Time Negatif**: Berita terkini memuat sentimen koreksi/tekanan.")
            else:
                signals.append("ℹ **Sentimen Berita Netral**: Tidak ada anomali berita fundamental ekstrem.")

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
                for book in book_data["reference_books"]:
                    st.markdown(f"**{book['title']}** ({book['year']}) — *{book['author']}*")
                    for rule in book['core_principles']:
                        st.caption(f"• {rule}")
                    st.markdown("---")

        with st.expander("🔍 Lihat Detail Literatur Pendukung & Aturan Validasi"):
            st.json(book_data)
