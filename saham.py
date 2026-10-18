import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import feedparser
import urllib.parse
import json
import os
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

# Mengimpor model Machine Learning berbasis Gradient Boosting
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="AI Quadruple-Threat IDX: 1011 Emiten BEI Predictive Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Konfigurasi Auto-Refresh Real-Time (Setiap 60 detik)
st_autorefresh(interval=60 * 1000, key="datarefresh")

# --- DAFTAR EMBEDDED LENGKAP 1.011 KODE SAHAM BEI & INDEKS UTAMA (FALLBACK OTOMATIS) ---
RAW_1011_IDX_CODES = (
    "AADI,AALI,ABBA,ABDA,ABMM,ACES,ACRO,ACST,ADCP,ADES,ADHI,ADMF,ADMG,ADMR,ADRO,AEGS,AGAR,AGII,AGRO,AGRS,AHAP,AIMS,AISA,AKKU,AKPI,"
    "AKRA,AKSI,ALDO,ALII,ALKA,ALMI,ALTO,AMAG,AMAN,AMAR,AMFG,AMIN,AMMN,AMMS,AMOR,AMRT,ANDI,ANJT,ANTM,APEX,APIC,APII,APLI,APLN,ARCI,"
    "AREA,ARGO,ARII,ARKA,ARKO,ARMY,ARNA,ARTA,ARTI,ARTO,ASBI,ASDM,ASGR,ASHA,ASII,ASJT,ASLC,ASLI,ASMI,ASPI,ASPR,ASRI,ASRM,ASSA,ATAP,"
    "ATIC,ATLA,AUTO,AVIA,AWAN,AXIO,AYAM,AYLS,BABP,BABY,BACA,BACH,BAIK,BAJA,BALI,BANK,BAPA,BAPI,BATA,BATR,BAUT,BAYU,BBCA,BBHI,BBKP,"
    "BBLD,BBMD,BBNI,BBRI,BBRM,BBSI,BBSS,BBTN,BBYB,BCAP,BCIC,BCIP,BDKR,BDMN,BEBS,BEEF,BEER,BEKS,BELI,BELL,BESS,BEST,BFIN,BGTG,BHAT,"
    "BHIT,BIKA,BIKE,BIMA,BINA,BINO,BIPI,BIPP,BIRD,BISI,BJBR,BJTM,BKDP,BKSL,BKSW,BLES,BLOG,BLTA,BLTZ,BLUE,BMAS,BMBL,BMHS,BMRI,BMSR,"
    "BMTR,BNBA,BNBR,BNGA,BNII,BNLI,BOAT,BOBA,BOGA,BOLA,BOLT,BOSS,BPFI,BPII,BPTR,BRAM,BREN,BRIS,BRMS,BRNA,BRPT,BRRC,BSBK,BSDE,BSIM,"
    "BSML,BSSR,BSWD,BTEK,BTEL,BTON,BTPN,BTPS,BUAH,BUDI,BUKA,BUKK,BULL,BUMI,BUVA,BVIC,BWPT,BYAN,CAKK,CAMP,CANI,CARE,CARS,CASA,CASH,"
    "CASS,CBDK,CBMF,CBPE,CBRE,CBUT,CCSI,CDIA,CEKA,CENT,CFIN,CGAS,CHEK,CHEM,CHIP,CINT,CITA,CITY,CLAY,CLEO,CLPI,CMNP,CMNT,CMPP,CMRY,"
    "CNKO,CNMA,CNTX,COAL,COCO,COIN,COWL,CPIN,CPRI,CPRO,CRAB,CRSN,CSAP,CSIS,CSMI,CSRA,CTBN,CTRA,CTTH,CUAN,CYBR,DAAZ,DADA,DART,DATA,"
    "DAYA,DCII,DEAL,DEFI,DEPO,DEWA,DEWI,DFAM,DGIK,DGNS,DGWG,DIGI,DILD,DIVA,DKFT,DKHH,DLTA,DMAS,DMMX,DMND,DNAR,DNET,DOID,DOOH,DOSS,"
    "DPNS,DPUM,DRMA,DSFI,DSNG,DSSA,DUCK,DUTI,DVLA,DWGL,DYAN,EAST,ECII,EDGE,EKAD,ELIT,ELPI,ELSA,ELTY,EMAS,EMDE,EMMI,EMTK,ENAK,ENRG,"
    "ENVY,ENZO,EPAC,EPMT,ERAA,ERAL,ERTX,ESIP,ESSA,ESTA,ESTI,ETWA,EURO,EXCL,FAPA,FAST,FASW,FILM,FIMP,FIRE,FISH,FITT,FLMC,FMII,FOLK,"
    "FOOD,FORE,FORU,FORZ,FPNI,FREN,FUJI,FUTR,FWCT,GAMA,GDST,GDYR,GEMA,GEMS,GGRM,GGRP,GHON,GIAA,GJTL,GLOB,GLVA,GMFI,GMTD,GOLD,GOLF,"
    "GOLL,GOOD,GOTO,GPRA,GPSO,GRIA,GRPH,GRPM,GSMF,GTBO,GTRA,GTSI,GULA,GUNA,GWSA,GZCO,HADE,HAIS,HAJJ,HALO,HATM,HBAT,HDFA,HDIT,HDTX,"
    "HEAL,HELI,HERO,HEXA,HGII,HILL,HITS,HKMU,HMSP,HOKI,HOME,HOMI,HOPE,HOTL,HRME,HRTA,HRUM,HUMI,HYGN,IATA,IBFN,IBOS,IBST,ICBP,ICON,"
    "IDEA,IDPR,IFII,IFSH,IGAR,IIKP,IKAI,IKAN,IKBI,IKPM,IMAS,IMJS,IMPC,INAF,INAI,INCF,INCI,INCO,INDF,INDO,INDR,INDS,INDX,INDY,INET,"
    "INKP,INOV,INPC,INPP,INPS,INRU,INTA,INTD,INTP,IOTF,IPAC,IPCC,IPCM,IPOL,IPPE,IPTV,IRRA,IRSX,ISAP,ISAT,ISEA,ISSP,ITIC,ITMA,ITMG,"
    "JARR,JAST,JATI,JAWA,JAYA,JECC,JECX,JELI,JGLE,JIHD,JKON,JKSW,JMAS,JPFA,JRPT,JSKY,JSMR,JSPT,JTPE,KAEF,KAQI,KARW,KAYU,KBAG,KBLI,"
    "KBLM,KBLV,KBRI,KDSI,KDTN,KEEN,KEJU,KETR,KIAS,KICI,KIJA,KING,KINO,KIOS,KJEN,KKES,KKGI,KLAS,KLBF,KLIN,KMDS,KMTR,KOBX,KOCI,KOIN,"
    "KOKA,KONI,KOPI,KOTA,KPAL,KPAS,KPIG,KRAH,KRAS,KREN,KRYA,KSIX,KUAS,LABA,LABS,LAJU,LAND,LAPD,LCGP,LCKM,LEAD,LFLO,LIFE,LINK,LION,"
    "LIVE,LMAS,LMAX,LMPI,LMSH,LOPI,LPCK,LPGI,LPIN,LPKR,LPLI,LPPF,LPPS,LRNA,LSIP,LTLS,LUCK,LUCY,MABA,MAGP,MAHA,MAIN,MAMI,MANG,MAPA,"
    "MAPB,MAPI,MARI,MARK,MASA,MASB,MAXI,MAYA,MBAP,MBMA,MBSS,MBTO,MCAS,MCOL,MCOR,MDIA,MDIY,MDKA,MDKI,MDLA,MDLN,MDRN,MEDC,MEDS,MEGA,"
    "MEJA,MENN,MERI,MERK,META,MFIN,MFMI,MGLV,MGNA,MGRO,MHKI,MICE,MIDI,MIKA,MINA,MINE,MIRA,MITI,MKAP,MKNT,MKPI,MKTR,MLBI,MLIA,MLPL,"
    "MLPT,MMIX,MMLP,MNCN,MOLI,MORA,MPIX,MPMX,MPOW,MPPA,MPRO,MPXL,MRAT,MREI,MSIE,MSIN,MSJA,MSKY,MSTI,MTDL,MTEL,MTFN,MTLA,MTMH,MTPS,"
    "MTRA,MTSM,MTWI,MUTU,MYOH,MYOR,MYRX,MYTX,NAIK,NANO,NASA,NASI,NATO,NAYZ,NCKL,NELY,NEST,NETV,NFCX,NICE,NICK,NICL,NIKL,NINE,NIPS,"
    "NIRO,NISP,NOBU,NPGF,NRCA,NSSS,NTBK,NUSA,NZIA,OASA,OBAT,OBMD,OCAP,OILS,OKAS,OLIV,OMED,OMRE,OPMS,PACK,PADA,PADI,PALM,PAMG,PANI,"
    "PANR,PANS,PART,PBID,PBRX,PBSA,PCAR,PDES,PDPP,PEGE,PEHA,PEVE,PGAS,PGEO,PGJO,PGLI,PGUN,PICO,PIPA,PJAA,PJHB,PKPK,PLAN,PLAS,PLIN,"
    "PMJS,PMMP,PMUI,PNBN,PNBS,PNGO,PNIN,PNLF,PNSE,POLA,POLI,POLL,POLU,POLY,POOL,PORT,POSA,POWR,PPGL,PPRE,PPRI,PPRO,PRAS,PRAY,PRDA,"
    "PRDL,PRIM,PSAB,PSAT,PSDN,PSGO,PSKT,PSSI,PTBA,PTDU,PTIS,PTMP,PTMR,PTPP,PTPS,PTPW,PTRO,PTSN,PTSP,PUDP,PURA,PURE,PURI,PWON,PYFA,"
    "PZZA,RAAM,RAFI,RAJA,RALS,RANC,RANS,RATU,RBMS,RCCC,RDTX,REAL,RELF,RELI,RGAS,RICY,RIGS,RIMO,RISE,RLCO,RMKE,RMKO,ROCK,RODA,RONY,"
    "ROTI,RSCH,RSGK,RUIS,RUNS,SAFE,SAGE,SAME,SAMF,SAPX,SATU,SBAT,SBMA,SCCO,SCMA,SCNP,SCPI,SDMU,SDPC,SDRA,SEMA,SFAN,SGER,SGRO,SHID,"
    "SHIP,SICO,SIDO,SILO,SIMA,SIMP,SINI,SIPD,SKBM,SKLT,SKRN,SKYB,SLIS,SMAR,SMBR,SMCB,SMDM,SMDR,SMGA,SMGR,SMIL,SMKL,SMKM,SMLE,SMMA,"
    "SMMT,SMRA,SMRU,SMSM,SNLK,SOCI,SOFA,SOHO,SOLA,SONA,SOSS,SOTS,SOUL,SPMA,SPRE,SPTO,SQMI,SRAJ,SRIL,SRSN,SRTG,SSIA,SSMS,SSTM,STAA,"
    "STAR,STRK,STTP,SUGI,SULI,SUNI,SUPA,SUPR,SURE,SURI,SWAT,SWID,TALF,TAMA,TAMU,TAPG,TARA,TAXI,TAYS,TBIG,TBLA,TBMS,TCID,TCPI,TDPM,"
    "TEBE,TECH,TELE,TFAS,TFCO,TGKA,TGRA,TGUK,TIFA,TINS,TIRA,TIRT,TKIM,TLDN,TLKM,TMAS,TMPO,TNCA,TOBA,TOOL,TOPS,TOSK,TOTL,TOTO,TOWR,"
    "TOYS,TPIA,TPMA,TRAM,TRGU,TRIL,TRIM,TRIN,TRIO,TRIS,TRJA,TRON,TRST,TRUE,TRUK,TRUS,TSPC,TUGU,TYRE,UANG,UCID,UDNG,UFOE,ULTJ,UNIC,"
    "UNIQ,UNIT,UNSP,UNTD,UNTR,UNVR,URBN,UVCR,VAST,VERN,VICI,VICO,VINS,VISI,VIVA,VKTR,VOKS,VRNA,VTNY,WAPO,WBSA,WEGE,WEHA,WGSH,WICO,"
    "WIDI,WIFI,WIIM,WIKA,WINE,WINR,WINS,WIRG,WMPP,WMUU,WOMF,WOOD,WOWS,WSBP,WSKT,WTON,YELO,YOII,YPAS,YULE,YUPI,ZATA,ZBRA,ZINC,ZONE,"
    "ZYRX,APOL,ATPK,BAEK,BBNP,BORN,BRAU,CKRA,CNTB,CPGT,CTRP,CTRS,DAJK,DAVO,FINN,GMCW,GREN,INVS,ITTG,JPRS,LAMI,MAMIP,MYRXP,NAGA,RMBA,"
    "SCBD,SIAP,SOBI,SQBB,SQBI,TKGA,TMPI,TRUB,TURI,UNTX,XCID"
).split(",")

LQ45_LIST = [
    "AADI", "ADMR", "ADRO", "AKRA", "AMMN", "AMRT", "ANTM", "ASII", "BBCA", "BBNI", "BBRI", "BBTN", "BMRI", "BRPT", "BUMI",
    "CPIN", "CUAN", "DEWA", "EMTK", "ESSA", "EXCL", "GOTO", "HRTA", "ICBP", "INCO", "INDF", "INDY", "INKP", "ISAT", "ITMG",
    "JPFA", "KLBF", "MAPI", "MBMA", "MDKA", "MEDC", "NCKL", "PGAS", "PGEO", "PTBA", "SCMA", "TLKM", "UNTR", "UNVR", "WIFI"
]

KOMPAS100_LIST = [
    "AADI", "ACES", "ADMR", "ADRO", "AKRA", "AMMN", "AMRT", "ANTM", "ARCI", "ARTO", "ASII", "BBCA", "BBNI", "BBRI", "BBTN",
    "BBYB", "BFIN", "BIPI", "BKSL", "BMRI", "BNBR", "BRIS", "BRMS", "BRPT", "BSDE", "BUKA", "BULL", "BUMI", "BUVA", "CBDK",
    "CMRY", "COIN", "CPIN", "CTRA", "CUAN", "DEWA", "DSNG", "ELSA", "EMAS", "EMTK", "ENRG", "ERAA", "ESSA", "EXCL", "GGRM",
    "GOTO", "HEAL", "HRTA", "HRUM", "ICBP", "IMPC", "INCO", "INDF", "INDY", "INET", "INKP", "ISAT", "ITMG", "JPFA", "JSMR",
    "KIJA", "KLBF", "KPIG", "LSIP", "MAPA", "MAPI", "MBMA", "MDKA", "MEDC", "MIKA", "MINA", "MYOR", "NCKL", "PANI", "PGAS",
    "PGEO", "PNLF", "PSAB", "PTBA", "PTRO", "PWON", "RAJA", "RATU", "RMKE", "SCMA", "SGER", "SMGR", "SMIL", "SMRA", "SSIA",
    "TAPG", "TINS", "TLKM", "TOBA", "TOWR", "TPIA", "UNTR", "UNVR", "WIFI", "WIRG"
]

PEFINDO25_LIST = [
    "ACES", "ASRI", "AUTO", "AVIA", "BFIN", "BRMS", "BTPS", "CFIN", "ELSA", "ENRG", "ERAA", "ESSA", "GJTL", "HEAL", "KIJA",
    "LSIP", "MAPA", "MIDI", "MNCN", "MTDL", "PTRO", "SCMA", "SMDR", "TAPG", "UNVR"
]

DELISTED_LIST = [
    "APOL", "ATPK", "BAEK", "BBNP", "BORN", "BRAU", "CKRA", "CPGT", "CTRP", "CTRS", "DAJK", "DAVO", "FINN", "GMCW", "GREN",
    "INVS", "ITTG", "JPRS", "LAMI", "NAGA", "RMBA", "SCBD", "SIAP", "SOBI", "SQBB", "SQBI", "TKGA", "TMPI", "TRUB", "TURI",
    "UNTX"
]

# --- KUSTOMISASI CSS TEMA NEON FUTURISTIK & BADGE BREAKOUT ---
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
    .breakout-badge {
        background: linear-gradient(90deg, #00ff66, #00b347);
        color: #030508 !important;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 900;
        font-size: 16px;
        text-align: center;
        box-shadow: 0 0 20px rgba(0, 255, 102, 0.85);
        border: 2px solid #ffffff;
        margin-bottom: 15px;
        animation: pulse 1.8s infinite;
    }
    .neutral-badge {
        background: rgba(22, 31, 51, 0.9);
        color: #00ffcc !important;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: bold;
        border: 1px solid #00ffcc;
        margin-bottom: 15px;
        text-align: center;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 10px rgba(0, 255, 102, 0.6); }
        50% { box-shadow: 0 0 25px rgba(0, 255, 102, 1.0); }
        100% { box-shadow: 0 0 10px rgba(0, 255, 102, 0.6); }
    }
    .ticker-tape {
        background: #0e1320;
        border-top: 1px solid #00ffcc;
        border-bottom: 1px solid #00ffcc;
        padding: 8px;
        white-space: nowrap;
        overflow-x: auto;
        margin-bottom: 15px;
        font-size: 13px;
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
st.sidebar.header("⚡ QUANT CONFIG: 1.011 EMITEN BEI")

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

# --- FUNGSI MEMUAT 1.011 SAHAM DARI daftar_lengkap_saham_indonesia_bei.csv ---
@st.cache_data(ttl=86400)
def load_all_1011_idx_stocks():
    csv_candidates = [
        "daftar_lengkap_saham_indonesia_bei.csv",
        "daftar_lengkap_963_saham_indonesia.csv"
    ]
    for csv_file in csv_candidates:
        if os.path.exists(csv_file):
            try:
                df_csv = pd.read_csv(csv_file)
                if 'Kode_Saham' in df_csv.columns:
                    df_csv['Kode_Saham'] = df_csv['Kode_Saham'].astype(str).str.strip().str.upper()
                    if 'Ticker_Yahoo_Finance' not in df_csv.columns:
                        df_csv['Ticker_Yahoo_Finance'] = df_csv['Kode_Saham'] + ".JK"
                    if 'Nama_Perusahaan' not in df_csv.columns:
                        df_csv['Nama_Perusahaan'] = df_csv['Kode_Saham']
                    if 'Sektor_Indonesia' not in df_csv.columns:
                        df_csv['Sektor_Indonesia'] = df_csv.get('Sektor', 'Emiten BEI')
                    if 'Papan_Pencatatan' not in df_csv.columns:
                        df_csv['Papan_Pencatatan'] = "Utama / Pengembangan"
                    if 'Indeks_Utama' not in df_csv.columns:
                        df_csv['Indeks_Utama'] = df_csv['Kode_Saham'].apply(
                            lambda c: "LQ45, KOMPAS100" if c in LQ45_LIST else ("KOMPAS100" if c in KOMPAS100_LIST else "-")
                        )
                    if 'Status_Pencatatan' not in df_csv.columns:
                        df_csv['Status_Pencatatan'] = df_csv['Kode_Saham'].apply(
                            lambda c: "Delisting" if c in DELISTED_LIST else "Aktif"
                        )
                    if 'Jumlah_Saham_Tercatat' not in df_csv.columns:
                        df_csv['Jumlah_Saham_Tercatat'] = 0
                    if 'Tanggal_IPO_Teks' not in df_csv.columns:
                        df_csv['Tanggal_IPO_Teks'] = "-"
                    if 'Ticker_TradingView' not in df_csv.columns:
                        df_csv['Ticker_TradingView'] = "IDX:" + df_csv['Kode_Saham']
                    if 'Ticker_Bloomberg' not in df_csv.columns:
                        df_csv['Ticker_Bloomberg'] = df_csv['Kode_Saham'] + ":IJ"
                    return df_csv
            except Exception:
                pass

    # Fallback Otomatis 1.011 Emiten jika file CSV belum diletakkan di direktori
    rows = []
    for idx, code in enumerate(RAW_1011_IDX_CODES, start=1):
        indeks_list = []
        if code in LQ45_LIST:
            indeks_list.append("LQ45")
        if code in KOMPAS100_LIST:
            indeks_list.append("KOMPAS100")
        if code in PEFINDO25_LIST:
            indeks_list.append("PEFINDO25")
        indeks_str = ", ".join(indeks_list) if indeks_list else "-"
        status_str = "Delisting" if code in DELISTED_LIST else ("Aktif (DIRE/REIT)" if code == "XCID" else "Aktif")
        papan_str = "Delisting" if code in DELISTED_LIST else ("Utama" if code in LQ45_LIST or code in KOMPAS100_LIST else "Pengembangan")
        rows.append({
            "No": idx,
            "Kode_Saham": code,
            "Ticker_BEI": f"BEI:{code}",
            "Ticker_TradingView": f"IDX:{code}",
            "Ticker_Yahoo_Finance": f"{code}.JK",
            "Ticker_Google_Finance": f"IDX:{code}",
            "Ticker_Bloomberg": f"{code}:IJ",
            "Nama_Perusahaan": f"Emiten {code} Tbk.",
            "Sektor_IDX_IC": "IDX Sector",
            "Sektor_Indonesia": "Delisting / Tidak Aktif" if code in DELISTED_LIST else "Semua Sektor BEI",
            "Papan_Pencatatan": papan_str,
            "Tanggal_IPO_ISO": "-",
            "Tanggal_IPO_Teks": "-",
            "Jumlah_Saham_Tercatat": 0,
            "Indeks_Utama": indeks_str,
            "Status_Pencatatan": status_str
        })
    return pd.DataFrame(rows)

with st.spinner("Memuat database lengkap 1.011 emiten BEI..."):
    df_master_bei = load_all_1011_idx_stocks()

# --- FILTER MULTI-DIMENSI BERDASARKAN METADATA BEI DI SIDEBAR ---
st.sidebar.subheader("🔎 Filter Database 1.011 Saham BEI")

hide_delisted = st.sidebar.checkbox("✅ Sembunyikan Saham Delisting (31 Emiten)", value=True)
filter_high_precision_menu = st.sidebar.checkbox("🎯 Filter Emiten Indeks Utama (LQ45 / KOMPAS100 / PEFINDO25)", value=False)
filter_ai_bullish_high_acc = st.sidebar.checkbox("🚀 Filter Sinyal Bullish AI (Naik > 2% & Presisi > 90%)", value=False)

# Filter Sektor Indonesia
available_sectors = ["Semua Sektor"] + sorted(df_master_bei['Sektor_Indonesia'].dropna().unique().tolist())
selected_sector = st.sidebar.selectbox("Filter Sektor IDX-IC", options=available_sectors, index=0)

# Filter Papan Pencatatan
available_boards = ["Semua Papan"] + sorted(df_master_bei['Papan_Pencatatan'].dropna().unique().tolist())
selected_board = st.sidebar.selectbox("Filter Papan Pencatatan BEI", options=available_boards, index=0)

# Filter Indeks Utama
selected_index_filter = st.sidebar.selectbox(
    "Filter Konstituen Indeks",
    options=["Semua Emiten (1.011)", "LQ45 (45 Saham)", "KOMPAS100 (100 Saham)", "PEFINDO25 (25 Saham)"],
    index=0
)

# Terapkan Filter ke DataFrame Master
df_filtered = df_master_bei.copy()
if hide_delisted:
    df_filtered = df_filtered[df_filtered['Status_Pencatatan'] != 'Delisting']

if filter_high_precision_menu:
    df_filtered = df_filtered[df_filtered['Indeks_Utama'] != '-']

if selected_sector != "Semua Sektor":
    df_filtered = df_filtered[df_filtered['Sektor_Indonesia'] == selected_sector]

if selected_board != "Semua Papan":
    df_filtered = df_filtered[df_filtered['Papan_Pencatatan'] == selected_board]

if selected_index_filter == "LQ45 (45 Saham)":
    df_filtered = df_filtered[df_filtered['Indeks_Utama'].astype(str).str.contains("LQ45", na=False)]
elif selected_index_filter == "KOMPAS100 (100 Saham)":
    df_filtered = df_filtered[df_filtered['Indeks_Utama'].astype(str).str.contains("KOMPAS100", na=False)]
elif selected_index_filter == "PEFINDO25 (25 Saham)":
    df_filtered = df_filtered[df_filtered['Indeks_Utama'].astype(str).str.contains("PEFINDO25", na=False)]

if df_filtered.empty:
    st.sidebar.warning("⚠️ Kombinasi filter tidak menemukan saham. Menampilkan seluruh emiten aktif.")
    df_filtered = df_master_bei[df_master_bei['Status_Pencatatan'] != 'Delisting'].copy()

# Buat Dictionary Pilihan Dropdown
filtered_stocks = {}
metadata_lookup = {}
for _, row in df_filtered.iterrows():
    code = str(row['Kode_Saham']).strip()
    yf_tck = str(row['Ticker_Yahoo_Finance']).strip()
    name = str(row['Nama_Perusahaan']).strip()
    sector = str(row['Sektor_Indonesia']).strip()
    board = str(row['Papan_Pencatatan']).strip()
    label = f"{code} – {name} [{sector} | {board}]"
    filtered_stocks[label] = yf_tck
    metadata_lookup[yf_tck] = row.to_dict()

# Pastikan lookup juga mencakup seluruh 1.011 saham jika user mengetik manual
for _, row in df_master_bei.iterrows():
    yf_tck = str(row['Ticker_Yahoo_Finance']).strip()
    if yf_tck not in metadata_lookup:
        metadata_lookup[yf_tck] = row.to_dict()

st.sidebar.caption(f"📋 Menampilkan **{len(filtered_stocks)}** dari **{len(df_master_bei)}** emiten BEI.")

stock_choice = st.sidebar.selectbox(
    "Pilih atau Cari Emiten BEI", 
    options=list(filtered_stocks.keys()),
    index=0
)
selected_ticker_default = filtered_stocks[stock_choice]

ticker_symbol = st.sidebar.text_input("Atau Ketik Manual Kode Saham IDX (Contoh: BBCA / AADI.JK)", value=selected_ticker_default)
ticker_symbol = ticker_symbol.strip().upper()
if not ticker_symbol.endswith(".JK") and not ticker_symbol.startswith("^"):
    ticker_symbol = f"{ticker_symbol}.JK"

current_meta = metadata_lookup.get(ticker_symbol, {
    "Kode_Saham": ticker_symbol.replace(".JK", ""),
    "Ticker_BEI": f"BEI:{ticker_symbol.replace('.JK', '')}",
    "Ticker_TradingView": f"IDX:{ticker_symbol.replace('.JK', '')}",
    "Ticker_Yahoo_Finance": ticker_symbol,
    "Ticker_Bloomberg": f"{ticker_symbol.replace('.JK', '')}:IJ",
    "Nama_Perusahaan": ticker_symbol,
    "Sektor_Indonesia": "IDX Equity",
    "Papan_Pencatatan": "Utama / Pengembangan",
    "Tanggal_IPO_Teks": "-",
    "Jumlah_Saham_Tercatat": 0,
    "Indeks_Utama": "-",
    "Status_Pencatatan": "Aktif"
})

ml_engine = st.sidebar.selectbox(
    "Pilih Neural / Quant Engine AI",
    ["Ensemble (XGBoost + LightGBM Hybrid)", "XGBoost Regressor", "LightGBM Regressor"],
    index=0
)

period_option = st.sidebar.selectbox("Rentang Waktu Analisis", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
interval_option = st.sidebar.selectbox("Interval Candle", ["1d", "1wk"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Kustomisasi Tampilan Grafik")
show_vwap = st.sidebar.checkbox("Tampilkan Garis VWAP", value=True)
show_ma200 = st.sidebar.checkbox("Tampilkan MA 50 & MA 200", value=True)
show_ichimoku = st.sidebar.checkbox("Tampilkan Ichimoku Cloud", value=True)
show_sar = st.sidebar.checkbox("Tampilkan Parabolic SAR", value=False)
show_sr_lines = st.sidebar.checkbox("Tampilkan Dynamic Support/Resistance", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🏦 Parameter Makroekonomi")
bi_rate_input = st.sidebar.number_input("Asumsi Suku Bunga BI Rate (%)", min_value=3.0, max_value=10.0, value=5.75, step=0.25)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Input Manajemen Modal (Trading Calculator)")
user_capital = st.sidebar.number_input("Total Modal Trading (IDR)", min_value=100000, value=10000000, step=500000)
risk_tolerance_pct = st.sidebar.slider("Maksimal Risiko per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Status Database & Validasi")
st.sidebar.success(f"🟢 {len(df_master_bei)} Emiten BEI Terindeks (976 Aktif, 45 LQ45, 100 KOMPAS100)")

# --- FUNGSI PENGAMBILAN DATA MAKRO & LIVE TICKER ---
@st.cache_data(ttl=60)
def fetch_macro_and_watchlist_tape():
    macro_tickers = {
        "IHSG": "^JKSE",
        "USD/IDR": "IDR=X",
        "Crude Oil": "CL=F",
        "Gold": "GC=F",
        "BBCA": "BBCA.JK",
        "BBRI": "BBRI.JK",
        "BMRI": "BMRI.JK",
        "TLKM": "TLKM.JK",
        "GOTO": "GOTO.JK",
        "AADI": "AADI.JK",
        "ADRO": "ADRO.JK"
    }
    results = {}
    for label, tck in macro_tickers.items():
        try:
            df_m = yf.download(tck, period="5d", interval="1d", progress=False)
            if isinstance(df_m.columns, pd.MultiIndex):
                df_m.columns = df_m.columns.droplevel(1)
            if len(df_m) >= 2:
                last_c = float(df_m['Close'].iloc[-1])
                prev_c = float(df_m['Close'].iloc[-2])
                chg = ((last_c - prev_c) / prev_c) * 100
                results[label] = {"price": last_c, "change": chg}
        except Exception:
            continue
    return results

macro_data = fetch_macro_and_watchlist_tape()

if macro_data:
    tape_items = []
    for k, v in macro_data.items():
        color = "#00ff66" if v['change'] >= 0 else "#ff3366"
        sign = "+" if v['change'] >= 0 else ""
        tape_items.append(
            f"<span style='margin-right: 25px;'><b>{k}:</b> {v['price']:,.2f} "
            f"<span style='color:{color};'>({sign}{v['change']:.2f}%)</span></span>"
        )
    st.markdown(f"<div class='ticker-tape'>⚡ <b>LIVE MARKET & MACRO TAPE:</b> {' | '.join(tape_items)}</div>", unsafe_allow_html=True)

st.title("⚡ QUANT AI: Quadruple-Threat IDX (1.011 Emiten BEI) Predictive Dashboard")
st.markdown(
    f"Analisis terpadu **4 Pilar Quadruple-Threat** untuk **{current_meta.get('Nama_Perusahaan', ticker_symbol)} ({ticker_symbol})** | "
    f"Sektor: **{current_meta.get('Sektor_Indonesia', '-')}** | Papan: **{current_meta.get('Papan_Pencatatan', '-')}** | "
    f"Indeks: **{current_meta.get('Indeks_Utama', '-')}** | Tanggal IPO: **{current_meta.get('Tanggal_IPO_Teks', '-')}**"
)

# Peringatan Khusus Status Delisting atau Papan Pemantauan Khusus (FCA) dari CSV
if current_meta.get("Status_Pencatatan") == "Delisting":
    st.error(
        f"⛔ **PERINGATAN DELISTING**: Emiten **{ticker_symbol} ({current_meta.get('Nama_Perusahaan')})** berstatus **Delisting / Tidak Aktif** di BEI. "
        f"Data harga historis mungkin tidak lagi tersedia di server bursa."
    )
elif current_meta.get("Papan_Pencatatan") == "Pemantauan Khusus":
    st.warning(
        f"⚠️ **PERINGATAN PAPAN PEMANTAUAN KHUSUS (FULL CALL AUCTION / FCA)**: "
        f"Emiten **{ticker_symbol}** tercatat berada di **Papan Pemantauan Khusus** BEI. Perhatikan risiko likuiditas dan mekanisme kuotasi berkala."
    )

# --- FUNGSI PENGAMBILAN DATA SAHAM & FUNDAMENTAL ---
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

@st.cache_data(ttl=3600)
def fetch_fundamental_metrics(ticker):
    fund_dict = {
        "market_cap": 0,
        "per": None,
        "pbv": None,
        "roe": None,
        "der": None,
        "npm": None,
        "rev_growth": None,
        "div_yield": 0.0,
        "eps": None,
        "bvps": None
    }
    try:
        t_obj = yf.Ticker(ticker)
        info = t_obj.info
        fund_dict["market_cap"] = info.get("marketCap", 0) or 0
        fund_dict["per"] = info.get("trailingPE") or info.get("forwardPE")
        fund_dict["pbv"] = info.get("priceToBook")
        fund_dict["roe"] = (info.get("returnOnEquity", 0) * 100) if info.get("returnOnEquity") is not None else None
        fund_dict["der"] = info.get("debtToEquity")
        fund_dict["npm"] = (info.get("profitMargins", 0) * 100) if info.get("profitMargins") is not None else None
        fund_dict["rev_growth"] = (info.get("revenueGrowth", 0) * 100) if info.get("revenueGrowth") is not None else None
        raw_div = info.get("dividendYield", 0) or 0
        fund_dict["div_yield"] = raw_div * 100 if raw_div < 1 else raw_div
        fund_dict["eps"] = info.get("trailingEps")
        fund_dict["bvps"] = info.get("bookValue")
    except Exception:
        pass
    return fund_dict

@st.cache_data(ttl=60)
def fetch_advanced_realtime_news_and_sentiment(ticker):
    clean_code = ticker.replace(".JK", "").lower()
    news_items = []
    query = urllib.parse.quote(f"saham {clean_code} OR {ticker}")
    rss_url = f"https://news.google.com/rss/search?q={query}+when:7d&hl=id&gl=ID&ceid=ID:id"
    
    sentiment_score = 0
    high_impact_positive = ["laba melonjak", "dividen jumbo", "akuisisi strategis", "rekor tertinggi", "buyback saham", "tumbuh positif", "ekspansi pabrik", "pendapatan naik", "masuk msci", "net buy asing"]
    moderate_positive = ["naik", "tumbuh", "menguat", "rebound", "beli", "positif", "kontrak baru", "kinerja solid", "optimis", "akumulasi"]
    high_impact_negative = ["suspensi bursa", "gagal bayar", "default", "rugi bersih", "anjlok tajam", "kasus hukum", "sanksi ojk", "korupsi", "keluar msci"]
    moderate_negative = ["turun", "koreksi", "melemah", "beban meningkat", "rugi", "lesu", "tekanan jual", "turun tipis", "net sell"]

    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:8]:
            title = entry.title
            link = entry.link
            published = entry.published if hasattr(entry, 'published') else "Terbaru"
            title_lower = title.lower()
            
            source = "Portal Finansial"
            if "kontan" in link.lower() or "kontan" in title_lower:
                source = "Kontan Investasi"
            elif "cnbcindonesia" in link.lower() or "cnbc" in title_lower:
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
fund_info = fetch_fundamental_metrics(ticker_symbol)
realtime_news, news_sentiment_score = fetch_advanced_realtime_news_and_sentiment(ticker_symbol)

if data.empty or len(data) < 50:
    st.warning(
        f"⚠️ Data historis untuk ticker **{ticker_symbol}** dengan rentang waktu **{period_option}** tidak mencukupi (minimal 50 baris untuk validasi ML). "
        f"Silakan pilih rentang waktu lebih panjang atau pastikan saham berstatus Aktif."
    )
else:
    # --- FEATURE ENGINEERING & INDIKATOR TEKNIKAL PRESISI TINGGI ---
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200, min_periods=30).mean()
    
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['TP_Vol'] = typical_price * data['Volume']
    data['VWAP'] = data['TP_Vol'].rolling(window=20).sum() / data['Volume'].rolling(window=20).sum()

    data['BB_Middle'] = data['MA20']
    std_dev = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (std_dev * 2)
    data['BB_Lower'] = data['BB_Middle'] - (std_dev * 2)
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()

    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    data['Stoch_K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
    data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()

    raw_money_flow = typical_price * data['Volume']
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    mfi_ratio = positive_flow / negative_flow
    data['MFI'] = 100 - (100 / (1 + mfi_ratio))

    mf_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low']).replace(0, np.nan)
    mf_multiplier = mf_multiplier.fillna(0)
    mf_volume = mf_multiplier * data['Volume']
    data['CMF'] = mf_volume.rolling(window=20).sum() / data['Volume'].rolling(window=20).sum()
    data['Smart_Money_Val_IDR'] = mf_volume * data['Close']

    data['Vol_MA20'] = data['Volume'].rolling(window=20).mean()
    data['Vol_Ratio'] = data['Volume'] / data['Vol_MA20'].replace(0, np.nan)
    data['Dynamic_Resistance'] = data['High'].rolling(window=20).max()
    data['Dynamic_Support'] = data['Low'].rolling(window=20).min()

    data['Body_Size'] = (data['Close'] - data['Open']).abs()
    data['Candle_Range'] = data['High'] - data['Low']
    data['Body_Ratio'] = np.where(data['Candle_Range'] > 0, data['Body_Size'] / data['Candle_Range'], 0.5)

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

    high_series = data['High']
    low_series = data['Low']
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

    if not benchmark_close.empty:
        combined = pd.concat([data['Close'].pct_change(), benchmark_close.pct_change()], axis=1).dropna()
        combined.columns = ['Stock', 'Market']
        covariance = combined.cov().iloc[0, 1]
        market_variance = combined['Market'].var()
        stock_beta = covariance / market_variance if market_variance != 0 else 1.0
    else:
        stock_beta = 1.0

    data.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume', 'MA20', 'RSI'], inplace=True)
    
    if data.empty or len(data) < 20:
        st.warning("⚠️ Data terlalu sedikit setelah pembersihan indikator.")
    else:
        latest_close = float(data['Close'].iloc[-1])
        prev_close = float(data['Close'].iloc[-2])
        latest_open = float(data['Open'].iloc[-1])
        prev_open = float(data['Open'].iloc[-2])
        latest_high = float(data['High'].iloc[-1])
        latest_low = float(data['Low'].iloc[-1])
        price_change = latest_close - prev_close
        pct_change = (price_change / prev_close) * 100
        price_std = float(data['Close'].pct_change().std() * 100)

        latest_rsi = float(data['RSI'].iloc[-1])
        latest_macd = float(data['MACD'].iloc[-1])
        latest_signal = float(data['Signal_Line'].iloc[-1])
        latest_atr = float(data['ATR'].iloc[-1]) if not pd.isna(data['ATR'].iloc[-1]) else (latest_close * 0.03)
        latest_stoch_k = float(data['Stoch_K'].iloc[-1]) if not pd.isna(data['Stoch_K'].iloc[-1]) else 50.0
        latest_mfi = float(data['MFI'].iloc[-1]) if not pd.isna(data['MFI'].iloc[-1]) else 50.0
        latest_cmf = float(data['CMF'].iloc[-1]) if not pd.isna(data['CMF'].iloc[-1]) else 0.0
        latest_vol_ratio = float(data['Vol_Ratio'].iloc[-1]) if not pd.isna(data['Vol_Ratio'].iloc[-1]) else 1.0
        latest_body_ratio = float(data['Body_Ratio'].iloc[-1])
        latest_vwap = float(data['VWAP'].iloc[-1]) if not pd.isna(data['VWAP'].iloc[-1]) else latest_close
        latest_ma20 = float(data['MA20'].iloc[-1])
        latest_ma50 = float(data['MA50'].iloc[-1]) if not pd.isna(data['MA50'].iloc[-1]) else latest_ma20
        latest_ma200 = float(data['MA200'].iloc[-1]) if not pd.isna(data['MA200'].iloc[-1]) else latest_ma50
        dyn_res = float(data['Dynamic_Resistance'].iloc[-2]) if len(data) > 2 else latest_high
        dyn_sup = float(data['Dynamic_Support'].iloc[-2]) if len(data) > 2 else latest_low

        # --- DETEKSI POLA CANDLESTICK & BREAKOUT ---
        detected_patterns = []
        if (prev_close < prev_open) and (latest_close > latest_open) and (latest_close >= prev_open) and (latest_open <= prev_close):
            detected_patterns.append("Bullish Engulfing 🟢")

        lower_wick = min(latest_open, latest_close) - latest_low
        upper_wick = latest_high - max(latest_open, latest_close)
        body_abs = abs(latest_close - latest_open)
        if (lower_wick > 2 * max(body_abs, 1)) and (upper_wick <= body_abs):
            detected_patterns.append("Bullish Hammer / Pin Bar 🔨")

        is_volume_spike = latest_vol_ratio >= 1.5
        is_price_breakout = (latest_close >= dyn_res) and (pct_change > 1.5)
        if is_price_breakout:
            detected_patterns.append("20-Day High Breakout 🚀")
        if latest_ma20 > latest_ma50 and float(data['MA20'].iloc[-2]) <= float(data['MA50'].iloc[-2]):
            detected_patterns.append("Golden Cross (MA20 x MA50) ✨")
        if not detected_patterns:
            detected_patterns.append("Struktur Normal / Konsolidasi")

        # --- MACHINE LEARNING VALIDASI OUT-OF-SAMPLE ---
        df_pred = data.reset_index()
        df_pred['Days'] = np.arange(len(df_pred))
        df_pred['Lag1'] = df_pred['Close'].shift(1)
        df_pred['Lag2'] = df_pred['Close'].shift(2)
        df_pred['Rolling_Mean_5'] = df_pred['Close'].rolling(5).mean()
        df_pred['Rolling_Std_5'] = df_pred['Close'].rolling(5).std()
        df_pred.dropna(inplace=True)

        features = ['Days', 'Lag1', 'Lag2', 'Rolling_Mean_5', 'Rolling_Std_5', 'RSI', 'MACD', 'Body_Ratio']
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
        recent_closes_window = list(data['Close'].iloc[-5:].values)
        
        for i in range(5):
            next_day_idx = last_row_features['Days'] + 1
            current_input = pd.DataFrame([[
                next_day_idx, 
                last_row_features['Lag1'], 
                last_row_features['Lag2'], 
                np.mean(recent_closes_window[-5:]),
                np.std(recent_closes_window[-5:]),
                last_row_features['RSI'] if 'RSI' in last_row_features else latest_rsi,
                last_row_features['MACD'] if 'MACD' in last_row_features else 0,
                last_row_features['Body_Ratio'] if 'Body_Ratio' in last_row_features else latest_body_ratio
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
            recent_closes_window.append(pred_val)
            last_row_features['Lag2'] = last_row_features['Lag1']
            last_row_features['Lag1'] = pred_val
            last_row_features['Days'] = next_day_idx

        predicted_target = float(predicted_prices[-1])
        pred_pct_change = ((predicted_target - latest_close) / latest_close) * 100

        # --- PERHITUNGAN KAPITALISASI PASAR PRESISI DARI CSV + YFINANCE ---
        csv_shares = float(current_meta.get("Jumlah_Saham_Tercatat", 0) or 0)
        calculated_mcap_csv = csv_shares * latest_close
        mcap_idr = fund_info["market_cap"] if fund_info["market_cap"] > 0 else calculated_mcap_csv

        # --- PERHITUNGAN 4 PILAR: IDX ALPHA-MATRIX SCORING (0 - 100) ---
        tech_score = 0
        if latest_close > latest_ma20: tech_score += 6
        if latest_close > latest_ma50: tech_score += 5
        if latest_close >= latest_vwap: tech_score += 5
        if 45 <= latest_rsi <= 68: tech_score += 5
        elif latest_rsi < 35: tech_score += 6
        if latest_macd > latest_signal: tech_score += 5
        if is_volume_spike and pct_change > 0: tech_score += 5
        elif latest_vol_ratio > 1.1 and pct_change > 0: tech_score += 3
        if pred_pct_change > 1.0: tech_score += 4
        tech_score = min(35, tech_score)

        flow_score = 0
        five_day_smart_flow_idr = float(data['Smart_Money_Val_IDR'].iloc[-5:].sum())
        adtv_20d_idr = float((data['Close'] * data['Volume']).rolling(20).mean().iloc[-1])

        if latest_cmf > 0.10: flow_score += 10
        elif latest_cmf > 0: flow_score += 6
        if five_day_smart_flow_idr > 0: flow_score += 5
        if latest_mfi > 55: flow_score += 4

        if mcap_idr >= 45_000_000_000_000 and adtv_20d_idr >= 50_000_000_000:
            msci_status = "MSCI Global Standard Candidate / Member 🏆"
            flow_score += 6
        elif mcap_idr >= 8_000_000_000_000 and adtv_20d_idr >= 15_000_000_000:
            msci_status = "MSCI Small Cap Index Candidate 🌟"
            flow_score += 4
        else:
            msci_status = "Non-MSCI / Domestic Driven"
            flow_score += 2
        flow_score = min(25, flow_score)

        fund_score = 0
        eps_val = fund_info["eps"]
        bvps_val = fund_info["bvps"]
        per_val = fund_info["per"]
        pbv_val = fund_info["pbv"]
        roe_val = fund_info["roe"]
        der_val = fund_info["der"]

        if eps_val and bvps_val and eps_val > 0 and bvps_val > 0:
            graham_fv = np.sqrt(22.5 * eps_val * bvps_val)
            pe_fv = eps_val * 15.0
            fair_value_est = (0.5 * graham_fv) + (0.5 * pe_fv)
        elif eps_val and eps_val > 0:
            fair_value_est = eps_val * 15.0
        else:
            fair_value_est = latest_ma200 * 1.08

        margin_of_safety = ((fair_value_est - latest_close) / fair_value_est) * 100 if fair_value_est > 0 else 0.0

        if roe_val is not None:
            if roe_val >= 15: fund_score += 8
            elif roe_val >= 10: fund_score += 5
            elif roe_val > 0: fund_score += 3
        else:
            fund_score += 4

        if per_val is not None and 0 < per_val < 18: fund_score += 6
        elif per_val is None: fund_score += 3
        if margin_of_safety > 15: fund_score += 7
        elif margin_of_safety > 0: fund_score += 4
        if fund_info["div_yield"] >= 3.5: fund_score += 4
        fund_score = min(25, fund_score)

        macro_score = 7
        usd_idr_chg = macro_data.get("USD/IDR", {}).get("change", 0.0)
        if usd_idr_chg <= 0.1:
            macro_score += 3
        else:
            macro_score -= 2
        if news_sentiment_score >= 2: macro_score += 5
        elif news_sentiment_score > 0: macro_score += 3
        elif news_sentiment_score < 0: macro_score -= 3
        macro_score = max(0, min(15, macro_score))

        total_alpha_score = int(tech_score + flow_score + fund_score + macro_score)

        if total_alpha_score >= 80:
            action_signal = "STRONG BUY (Breakout / High-Conviction Accumulation) 🟢"
        elif total_alpha_score >= 65:
            action_signal = "BUY (Trend Following / Bullish Reversal) 🟢"
        elif total_alpha_score >= 45:
            action_signal = "HOLD / WAIT & SEE (Area Konsolidasi) 🟡"
        else:
            action_signal = "TAKE PROFIT / CUT LOSS (Distribusi / Tren Melemah) 🔴"

        # --- BADGE NOTIFIKASI VOLUME SPIKE / BREAKOUT ---
        if (is_volume_spike and pct_change > 0) or is_price_breakout or (pred_pct_change > 2.0 and total_alpha_score >= 75):
            st.markdown(
                f"<div class='breakout-badge'>"
                f"🚀 BULLISH ALERT TERDETEKSI [{ticker_symbol}]: Volume Rasio {latest_vol_ratio:.2f}x MA20 | "
                f"Pola: {', '.join(detected_patterns)} | Skor IDX Alpha-Matrix: {total_alpha_score}/100 ({action_signal})"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='neutral-badge'>"
                f"📊 Status Sinyal [{ticker_symbol}]: {action_signal} | Skor IDX Alpha-Matrix: <b>{total_alpha_score}/100</b> | "
                f"Volume Harian: {latest_vol_ratio:.2f}x MA20 | Pola: {', '.join(detected_patterns)}"
                f"</div>",
                unsafe_allow_html=True
            )

        # --- METRIK UTAMA ---
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Harga Terakhir (IDR)", f"{latest_close:,.0f}", f"{pct_change:+.2f}%")
        col2.metric("Prediksi AI (5 Hari)", f"{predicted_target:,.0f}", f"{pred_pct_change:+.2f}%")
        col3.metric("Skor Alpha-Matrix", f"{total_alpha_score} / 100", f"Presisi ML: {model_accuracy_score:.1f}%")
        col4.metric("Estimasi Fair Value", f"{fair_value_est:,.0f}", f"MoS: {margin_of_safety:+.1f}%")
        col5.metric("Volume Spike Ratio", f"{latest_vol_ratio:.2f}x MA20", f"CMF: {latest_cmf:+.2f}")

        # --- GRAFIK INTERAKTIF PLOTLY ---
        st.subheader(f"📈 Grafik Candlestick, VWAP, Ichimoku, Support/Resistance & Proyeksi AI: {ticker_symbol}")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Aktual Candle'
        ))
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], line=dict(color='rgba(0,255,204,0.3)', width=1), name='BB Upper'))
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], line=dict(color='rgba(0,255,204,0.3)', width=1), fill='tonexty', fillcolor='rgba(0,255,204,0.03)', name='BB Lower'))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], line=dict(color='#ff007f', width=1.5), name='MA 20'))

        if show_vwap and 'VWAP' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], line=dict(color='#ffcc00', width=2, dash='dot'), name='VWAP (20D)'))

        if show_ma200:
            fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], line=dict(color='#3399ff', width=1.5), name='MA 50'))
            fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], line=dict(color='#ff9900', width=2), name='MA 200'))

        if show_ichimoku and 'Senkou_Span_A' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['Senkou_Span_A'], line=dict(color='rgba(0,255,0,0.5)', width=1), name='Senkou Span A'))
            fig.add_trace(go.Scatter(x=data.index, y=data['Senkou_Span_B'], line=dict(color='rgba(255,0,0,0.5)', width=1), fill='tonexty', fillcolor='rgba(100,100,100,0.1)', name='Senkou Span B'))

        if show_sar and 'Parabolic_SAR' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['Parabolic_SAR'], mode='markers', marker=dict(size=4, color='yellow'), name='Parabolic SAR'))

        if show_sr_lines:
            fig.add_hline(y=dyn_res, line_dash="dash", line_color="#ff3366", annotation_text=f"Resistance (20D): {dyn_res:,.0f}", annotation_position="top left")
            fig.add_hline(y=dyn_sup, line_dash="dash", line_color="#00ff66", annotation_text=f"Support (20D): {dyn_sup:,.0f}", annotation_position="bottom left")
        
        plot_pred_dates = [last_date] + list(future_dates)
        plot_pred_prices = [latest_close] + list(predicted_prices)
        labels_text = [""] * len(plot_pred_prices)
        labels_text[-1] = f"<b>Target AI: {predicted_target:,.0f}</b>"

        fig.add_trace(go.Scatter(
            x=plot_pred_dates, y=plot_pred_prices, mode='lines+markers+text',
            text=labels_text, textposition="top center",
            line=dict(color='#00ffcc', width=2.5, dash='dash'),
            marker=dict(size=9, color='#00ffcc'), name='Proyeksi AI (5H)'
        ))
        
        fig.update_layout(
            title=f'Proyeksi Kuantitatif {ticker_symbol} ({current_meta.get("Nama_Perusahaan")}) Menuju {future_dates[-1].strftime("%d %b %Y")}',
            yaxis_title='Harga (IDR)', xaxis_title='Timeline',
            template='plotly_dark', paper_bgcolor='#05070c', plot_bgcolor='#0e1320',
            height=600, xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- PANEL 5 TAB: 4 PILAR QUADRUPLE-THREAT + MASTER DATABASE 1.011 SAHAM ---
        st.markdown("---")
        st.subheader("🏛️ Bedah 4 Pilar Quadruple-Threat IDX & Database Resmi 1.011 Emiten BEI")
        tab_p1, tab_p2, tab_p3, tab_p4, tab_p5 = st.tabs([
            "📊 1. Fundamental & Identitas BEI",
            "🌊 2. Foreign Flow & MSCI Simulator",
            "🌍 3. Makroekonomi & Sentimen NLP",
            "🧮 4. Target Harian AI & Kalkulator Lot ATR",
            "🗂️ 5. Eksplorasi Tabel 1.011 Saham BEI"
        ])

        with tab_p1:
            c_f1, c_f2 = st.columns(2)
            with c_f1:
                st.markdown(f"#### Kinerja Keuangan & Metadata Pencatatan ({ticker_symbol})")
                df_fund_table = pd.DataFrame({
                    "Parameter Fundamental & Pencatatan BEI": [
                        "Nama Resmi Emiten",
                        "Kode Multi-Platform (TV / Bloomberg)",
                        "Sektor IDX-IC / Papan Pencatatan",
                        "Jumlah Saham Tercatat (Database BEI)",
                        "Kapitalisasi Pasar (Market Cap)",
                        "Price to Earnings Ratio (PER)",
                        "Price to Book Value (PBV)",
                        "Return on Equity (ROE)",
                        "Debt to Equity Ratio (DER)",
                        "Estimasi Dividend Yield"
                    ],
                    "Nilai Aktual": [
                        str(current_meta.get("Nama_Perusahaan", ticker_symbol)),
                        f"{current_meta.get('Ticker_TradingView', '-')} | {current_meta.get('Ticker_Bloomberg', '-')}",
                        f"{current_meta.get('Sektor_Indonesia', '-')} ({current_meta.get('Papan_Pencatatan', '-')})",
                        f"{int(csv_shares):,} Lembar" if csv_shares > 0 else "N/A",
                        f"Rp {mcap_idr / 1e12:,.2f} Triliun",
                        f"{per_val:.2f}x" if per_val else "N/A",
                        f"{pbv_val:.2f}x" if pbv_val else "N/A",
                        f"{roe_val:.2f}%" if roe_val is not None else "N/A",
                        f"{der_val:.2f}%" if der_val is not None else "N/A",
                        f"{fund_info['div_yield']:.2f}%"
                    ]
                })
                st.table(df_fund_table)
            with c_f2:
                st.markdown("#### Model Valuasi Wajar (Intrinsic Fair Value)")
                val_status = "UNDERVALUED (Diskon) 🟢" if margin_of_safety > 0 else "OVERVALUED / PREMIUM 🟡"
                st.info(
                    f"📌 **Ringkasan Valuasi Kuantitatif:**\n\n"
                    f"- **Harga Pasar Saat Ini:** Rp {latest_close:,.0f}\n"
                    f"- **Estimasi Nilai Wajar (Graham/Relative):** Rp {fair_value_est:,.0f}\n"
                    f"- **Margin of Safety (MoS):** **{margin_of_safety:+.2f}%**\n"
                    f"- **Status Valuasi:** **{val_status}**\n"
                    f"- **Indeks Utama BEI:** **{current_meta.get('Indeks_Utama', '-')}**\n"
                    f"- **Kontribusi Skor Fundamental:** **{fund_score} / 25 Poin**"
                )

        with tab_p2:
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                st.markdown("#### Pantauan Smart Money / Net Institutional Flow")
                flow_dir = "AKUMULASI BERSIH (Net Buy Proxy) 🟢" if five_day_smart_flow_idr > 0 else "DISTRIBUSI BERSIH (Net Sell Proxy) 🔴"
                st.info(
                    f"📌 **Indikator Arus Dana Institusi & Asing:**\n\n"
                    f"- **Status Arus Dana (5 Hari):** **{flow_dir}**\n"
                    f"- **Estimasi Akumulasi/Distribusi Bersih (5H):** Rp {five_day_smart_flow_idr / 1e9:+,.2f} Miliar\n"
                    f"- **Chaikin Money Flow (CMF 20):** `{latest_cmf:+.3f}`\n"
                    f"- **Money Flow Index (MFI 14):** `{latest_mfi:.1f}`\n"
                    f"- **Rata-Rata Nilai Transaksi Harian (ADTV 20D):** Rp {adtv_20d_idr / 1e9:,.2f} Miliar"
                )
            with c_m2:
                st.markdown("#### Simulator & Analisis Dampak MSCI Rebalancing")
                st.success(
                    f"🏆 **Status Klasifikasi Indeks MSCI:**\n**{msci_status}**\n\n"
                    f"- **Syarat Market Cap MSCI Global Standard:** ~Rp 45 Triliun (Aktual: Rp {mcap_idr / 1e12:,.2f} T)\n"
                    f"- **Syarat Likuiditas Harian (ADTV):** > Rp 50 Miliar/hari (Aktual: Rp {adtv_20d_idr / 1e9:,.2f} M)\n"
                    f"- **Skor Pilar Foreign Flow & MSCI:** **{flow_score} / 25 Poin**"
                )

        with tab_p3:
            c_n1, c_n2 = st.columns([1, 2])
            with c_n1:
                st.markdown("#### Parameter Makroekonomi")
                st.write(f"- **Suku Bunga BI Rate:** `{bi_rate_input:.2f}%`")
                st.write(f"- **Beta Saham terhadap IHSG:** `{stock_beta:.2f}`")
                st.write(f"- **Skor Sentimen Berita NLP:** `{news_sentiment_score}`")
                st.write(f"- **Skor Pilar Makro & Sentimen:** **{macro_score} / 15 Poin**")
            with c_n2:
                st.markdown(f"#### 📡 Berita Finansial Real-Time: {ticker_symbol}")
                if realtime_news:
                    for item in realtime_news:
                        st.markdown(f"- **[{item['title']}]({item['link']})**")
                        st.caption(f"📌 Sumber: {item['publisher']} | Waktu: {item['date']} | Sentimen NLP: **{item['sentiment']}**")
                else:
                    st.info("Tidak ada berita baru dalam 7 hari terakhir yang terindeks.")

        with tab_p4:
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("**Target Harga Harian AI (5 Hari Bursa):**")
                df_future_table = pd.DataFrame({
                    "Tanggal": [d.strftime("%Y-%m-%d (%A)") for d in future_dates],
                    "Target Harga AI": [f"Rp {p:,.0f}" for p in predicted_prices],
                    "Est. Perubahan (%)": [f"{((p - latest_close) / latest_close) * 100:+.2f}%" for p in predicted_prices]
                })
                st.table(df_future_table)

                csv_data = df_future_table.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Unduh Laporan Proyeksi AI (CSV)",
                    data=csv_data,
                    file_name=f"proyeksi_ai_{ticker_symbol}.csv",
                    mime="text/csv"
                )
            with col_t2:
                st.markdown("**Kalkulator Alokasi Lot Berbasis Risiko ATR (Mark Douglas Risk Control):**")
                recommended_stop_loss = latest_close - (2 * latest_atr)
                recommended_tp1 = latest_close + (2 * latest_atr)
                recommended_tp2 = latest_close + (3.5 * latest_atr)
                
                max_money_risk = user_capital * (risk_tolerance_pct / 100.0)
                risk_per_share = latest_close - recommended_stop_loss
                if risk_per_share > 0:
                    shares_to_buy = max_money_risk / risk_per_share
                    lots_to_buy = int(shares_to_buy / 100)
                else:
                    lots_to_buy = 0

                estimated_total_investment = lots_to_buy * 100 * latest_close

                st.info(
                    f"📌 **Trading Plan & Manajemen Risiko Presisi:**\n\n"
                    f"- **Area Entry Ideal:** Rp {latest_close * 0.995:,.0f} – Rp {latest_close:,.0f}\n"
                    f"- **Target Price 1 (Short-Term):** Rp {recommended_tp1:,.0f} (+{((recommended_tp1-latest_close)/latest_close)*100:.2f}%)\n"
                    f"- **Target Price 2 (Swing):** Rp {recommended_tp2:,.0f} (+{((recommended_tp2-latest_close)/latest_close)*100:.2f}%)\n"
                    f"- **Stop Loss Mekanikal (2x ATR):** Rp {recommended_stop_loss:,.0f} (-{((latest_close-recommended_stop_loss)/latest_close)*100:.2f}%)\n"
                    f"- **Ukuran Posisi Aman:** **{lots_to_buy} Lot** (Rp {estimated_total_investment:,.0f})"
                )

        with tab_p5:
            st.markdown(f"#### 🗂️ Tabel Master Emiten BEI Terfilter ({len(df_filtered)} dari {len(df_master_bei)} Emiten)")
            st.dataframe(df_filtered, use_container_width=True, height=420)

        # --- MODUL SCREENER & PAPAN PERINGKAT SAHAM (1.011 EMITEN BEI) ---
        st.markdown("---")
        st.subheader("🚀 IDX Alpha-Matrix Leaderboard & AI Stock Screener (1.011 Emiten BEI)")
        st.write("Memindai daftar emiten berdasarkan filter aktif di Sidebar untuk meranking saham dengan **Skor Alpha-Matrix tertinggi**, **Potensi Return AI**, dan deteksi **Volume Spike / Breakout**.")

        scan_limit = st.slider("Jumlah Sampel Emiten yang Dipindai Sekaligus", min_value=10, max_value=150, value=25, step=5)

        if st.button("🔍 Jalankan Pemindaian Papan Peringkat Alpha-Matrix & AI (Database 1.011 Saham BEI)"):
            with st.spinner(f"Menjalankan pemindaian Machine Learning & Alpha-Matrix pada {scan_limit} emiten... Mohon tunggu."):
                subset_tickers = dict(list(filtered_stocks.items())[:scan_limit])
                
                screener_results = []
                for key_name, tck in subset_tickers.items():
                    try:
                        df_scr = yf.download(tck, period=period_option, interval="1d", progress=False)
                        if isinstance(df_scr.columns, pd.MultiIndex):
                            df_scr.columns = df_scr.columns.droplevel(1)
                            
                        if len(df_scr) > 40:
                            df_scr['MA20'] = df_scr['Close'].rolling(20).mean()
                            df_scr['Vol_MA20'] = df_scr['Volume'].rolling(20).mean()
                            df_scr['Vol_Ratio'] = df_scr['Volume'] / df_scr['Vol_MA20'].replace(0, np.nan)
                            df_scr['Body_Size'] = (df_scr['Close'] - df_scr['Open']).abs()
                            df_scr['Candle_Range'] = df_scr['High'] - df_scr['Low']
                            df_scr['Body_Ratio'] = np.where(df_scr['Candle_Range'] > 0, df_scr['Body_Size'] / df_scr['Candle_Range'], 0.5)
                            
                            df_p_scr = df_scr.reset_index()
                            df_p_scr['Days'] = np.arange(len(df_p_scr))
                            df_p_scr['Lag1'] = df_p_scr['Close'].shift(1)
                            df_p_scr['Lag2'] = df_p_scr['Close'].shift(2)
                            df_p_scr['Rolling_Mean_5'] = df_p_scr['Close'].rolling(5).mean()
                            df_p_scr['Rolling_Std_5'] = df_p_scr['Close'].rolling(5).std()
                            
                            delta_s = df_p_scr['Close'].diff()
                            gain_s = (delta_s.where(delta_s > 0, 0)).rolling(14).mean()
                            loss_s = (-delta_s.where(delta_s < 0, 0)).rolling(14).mean()
                            df_p_scr['RSI'] = 100 - (100 / (1 + (gain_s / loss_s)))
                            df_p_scr['MACD'] = df_p_scr['Close'].ewm(span=12).mean() - df_p_scr['Close'].ewm(span=26).mean()
                            
                            df_p_scr.dropna(inplace=True)
                            
                            if len(df_p_scr) > 20:
                                X_scr = df_p_scr[['Days', 'Lag1', 'Lag2', 'Rolling_Mean_5', 'Rolling_Std_5', 'RSI', 'MACD', 'Body_Ratio']]
                                y_scr = df_p_scr['Close']
                                
                                m_xgb = XGBRegressor(n_estimators=100, learning_rate=0.03, max_depth=3, random_state=42)
                                m_xgb.fit(X_scr, y_scr)
                                
                                last_row_s = X_scr.iloc[-1].copy()
                                next_p_list = []
                                for _ in range(5):
                                    last_row_s['Days'] += 1
                                    inp_s = pd.DataFrame([last_row_s], columns=X_scr.columns)
                                    p_val = m_xgb.predict(inp_s)[0]
                                    next_p_list.append(p_val)
                                    last_row_s['Lag2'] = last_row_s['Lag1']
                                    last_row_s['Lag1'] = p_val
                                    
                                latest_p_scr = float(df_scr['Close'].iloc[-1])
                                prev_p_scr = float(df_scr['Close'].iloc[-2])
                                daily_chg_scr = ((latest_p_scr - prev_p_scr) / prev_p_scr) * 100
                                target_p_scr = float(next_p_list[-1])
                                ml_pct_gain = ((target_p_scr - latest_p_scr) / latest_p_scr) * 100
                                v_ratio_scr = float(df_scr['Vol_Ratio'].iloc[-1]) if not pd.isna(df_scr['Vol_Ratio'].iloc[-1]) else 1.0
                                rsi_scr = float(df_p_scr['RSI'].iloc[-1])
                                
                                if filter_ai_bullish_high_acc and ml_pct_gain <= 2.0:
                                    continue

                                scr_alpha = 50
                                if latest_p_scr > float(df_scr['MA20'].iloc[-1]): scr_alpha += 12
                                if ml_pct_gain > 2.0: scr_alpha += 15
                                elif ml_pct_gain > 0: scr_alpha += 8
                                if v_ratio_scr >= 1.5 and daily_chg_scr > 0: scr_alpha += 15
                                if 45 <= rsi_scr <= 68: scr_alpha += 8
                                scr_alpha = min(99, max(20, scr_alpha))

                                if v_ratio_scr >= 1.5 and daily_chg_scr > 0:
                                    badge_alert = "🟢 VOLUME SPIKE / BREAKOUT"
                                elif ml_pct_gain > 2.0 and scr_alpha >= 75:
                                    badge_alert = "🟢 STRONG BUY (AI Bullish)"
                                elif ml_pct_gain > 0:
                                    badge_alert = "🟡 BUY / ACCUMULATE"
                                else:
                                    badge_alert = "🔴 WAIT / KOREKSI"

                                meta_s = metadata_lookup.get(tck, {})
                                screener_results.append({
                                    "Emiten": key_name,
                                    "Papan": meta_s.get("Papan_Pencatatan", "-"),
                                    "Indeks": meta_s.get("Indeks_Utama", "-"),
                                    "Harga Terakhir": f"Rp {latest_p_scr:,.0f}",
                                    "Target AI (5H)": f"Rp {target_p_scr:,.0f}",
                                    "Potensi Return AI": f"{ml_pct_gain:+.2f}%",
                                    "Volume Ratio": f"{v_ratio_scr:.2f}x MA20",
                                    "Skor Alpha-Matrix": scr_alpha,
                                    "Indikasi Sinyal": badge_alert
                                })
                    except Exception:
                        continue
                
                if screener_results:
                    df_final_scr = pd.DataFrame(screener_results)
                    df_final_scr = df_final_scr.sort_values(by='Skor Alpha-Matrix', ascending=False)
                    
                    st.success("✅ Pemindaian Papan Peringkat IDX Alpha-Matrix (Database 1.011 Saham BEI) Selesai!")
                    st.dataframe(df_final_scr, use_container_width=True)
                else:
                    st.warning("Tidak ada emiten yang memenuhi kriteria filter ketat pada sampel saat ini.")
