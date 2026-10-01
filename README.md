# AI Stock Analysis & Recommendation Dashboard (Real-Time)

Dasbor interaktif berbasis Python dan Streamlit untuk melakukan analisis teknikal saham secara real-time, dilengkapi dengan basis pengetahuan literatur keuangan klasik (seperti *Visual Guide to Chart Patterns* oleh Thomas N. Bulkowski)[cite: 6].

## Fitur Utama
- **Live & Real-Time Data**: Memuat data harga saham terkini secara otomatis (didukung `yfinance` & `streamlit-autorefresh`).
- **Universal Ticker Support**: Mendukung pemilihan saham populer maupun pencarian kode saham apa saja di seluruh dunia (misal: `BBCA.JK`, `TLKM.JK`, `AAPL`, `TSLA`).
- **Indikator Teknikal Lengkap**: Moving Average (MA20, MA50) dan RSI (Relative Strength Index).
- **Validasi Literatur**: Menggabungkan aturan probabilitas pola grafik dan prinsip manajemen risiko dari literatur investasi terkemuka.
