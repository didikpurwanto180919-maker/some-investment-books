import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import feedparser
import urllib.parse
import urllib.request
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

# Mengimpor model Machine Learning berbasis Gradient Boosting
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="AI Quadruple-Threat IDX: 1011 Emiten, MSCI Engine & 140+ Omni-News",
    page_icon="⚡",
    layout="wide"
)

# Konfigurasi Auto-Refresh Real-Time (Setiap 60 detik)
st_autorefresh(interval=60 * 1000, key="datarefresh")

# --- DAFTAR EMBEDDED LENGKAP 1.011 KODE SAHAM BEI & INDEKS UTAMA ---
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

# --- DATABASE RESMI KONSTITUEN MSCI INDONESIA & BOBOT INDEKS (FAKTA OKTOBER 2026) ---
MSCI_GLOBAL_STANDARD_WEIGHTS = {
    "BBCA": 31.41,
    "BBRI": 19.96,
    "BMRI": 15.70,
    "TLKM": 10.70,
    "ASII": 7.81,
    "BBNI": 4.64,
    "BRMS": 3.51,
    "UNTR": 3.44,
    "BRPT": 2.83
}

MSCI_SMALL_CAP_WEIGHTS = {
    "INDF": 7.88, "MDKA": 6.80, "BUMI": 5.78, "ENRG": 5.73, "ADRO": 5.71,
    "CPIN": 5.40, "AADI": 4.81, "AMRT": 4.47, "PGAS": 4.01, "PTRO": 3.83,
    "AKRA": 2.15, "AVIA": 1.60, "BBTN": 1.45, "BFIN": 1.50, "CMRY": 1.95,
    "CTRA": 1.75, "EMTK": 2.10, "EXCL": 2.20, "GGRM": 1.40, "INCO": 2.05,
    "INKP": 2.30, "INTP": 1.65, "ITMG": 1.85, "JPFA": 1.55, "JSMR": 1.70,
    "KLBF": 2.85, "MAPA": 1.50, "MAPI": 1.90, "MEDC": 2.10, "MTEL": 1.60,
    "PTBA": 1.80, "PWON": 1.65, "RAJA": 1.35, "TOWR": 1.95, "WIFI": 1.25
}

# Saham dalam pengawasan konsentrasi kepemilikan tinggi (HSC) / pengecualian MSCI
MSCI_HSC_RESTRICTED_WATCHLIST = ["DSSA", "BREN", "CUAN", "AMMN", "PANI", "TPIA", "BYAN", "DCII"]

# --- KATALOG LENGKAP 140+ PORTAL BERITA, REGULATOR, RISET SEKURITAS, BANDARMOLOGY & MSCI ---
OMNI_NEWS_DIRECTORY = {
    "1. Regulator Resmi & Acuan Indeks Global (MSCI, BEI, KSEI, OJK, e-IPO)": [
        ("MSCI Indonesia Index Official Factsheet & Constituents", "https://www.msci.com/indexes/index/105767/msci-indonesia-index"),
        ("MSCI Indonesia Small Cap Index Official Page", "https://www.msci.com/indexes/index/655073/msci-indonesia-small-cap-index"),
        ("Keterbukaan Informasi BEI (Real-Time PDF)", "https://www.idx.co.id/id/perusahaan-tercatat/keterbukaan-informasi/"),
        ("Pengumuman Bursa (Suspensi, Unsuspensi/Buka Gembok, Delisting)", "https://www.idx.co.id/id/berita/pengumuman/"),
        ("Radar Saham UMA (Unusual Market Activity)", "https://www.idx.co.id/id/berita/unusual-market-activity-uma/"),
        ("Daftar Efek Papan Pemantauan Khusus (FCA)", "https://www.idx.co.id/id/data-pasar/data-saham/daftar-efek-pemantauan-khusus/"),
        ("Kalender Aksi Korporasi BEI (Dividen, RUPS, Right Issue, Stock Split)", "https://www.idx.co.id/id/perusahaan-tercatat/kalender-perusahaan-tercatat/"),
        ("Ringkasan Perdagangan Saham (OHLC, Volume & Foreign Flow)", "https://www.idx.co.id/id/data-pasar/ringkasan-perdagangan/ringkasan-saham/"),
        ("Ringkasan Broker Harian (Official Broker Summary)", "https://www.idx.co.id/id/data-pasar/ringkasan-perdagangan/ringkasan-broker/"),
        ("Data Indeks Sektoral BEI (IDX-IC)", "https://www.idx.co.id/id/data-pasar/data-indeks/ringkasan-indeks/"),
        ("Laporan Keuangan Tahunan & Kuartalan Emiten", "https://www.idx.co.id/id/perusahaan-tercatat/laporan-keuangan-dan-tahunan/"),
        ("Daftar Efek Marjin & Short Selling", "https://www.idx.co.id/id/data-pasar/data-saham/daftar-efek-marjin-dan-short-selling/"),
        ("Data Kepemilikan Saham KSEI (>5% & Distribusi)", "https://www.ksei.co.id/publications/shareholder-information"),
        ("Statistik Bulanan KSEI (Scriptless Balance Lokal vs Asing)", "https://www.ksei.co.id/publications/monthly-statistic"),
        ("Jadwal Aksi Korporasi KSEI", "https://www.ksei.co.id/publications/corporate-action-schedules"),
        ("e-IPO (Jadwal & Prospektus Saham IPO)", "https://e-ipo.co.id/id/ipo/index"),
        ("Pengumuman OJK (Pasar Modal)", "https://ojk.go.id/id/berita-dan-kegiatan/pengumuman/Default.aspx"),
    ],
    "2. Media Finansial Tier-1 (CNBC, Kontan, Bisnis, Bloomberg Technoz, EmitenNews, IDX Channel)": [
        ("CNBC Indonesia Market", "https://www.cnbcindonesia.com/market"),
        ("CNBC My Money (Rekomendasi Saham & Trading Ideas)", "https://www.cnbcindonesia.com/mymoney"),
        ("CNBC Research (Bedah Fundamental & Makro)", "https://www.cnbcindonesia.com/research"),
        ("CNBC Indeks Berita Market (Detik-demi-Detik)", "https://www.cnbcindonesia.com/market/indeks"),
        ("Kontan Investasi (Utama Pasar Modal)", "https://investasi.kontan.co.id/"),
        ("Kontan Industri (Tambang, Energi, Batu Bara, Emas, Nikel)", "https://industri.kontan.co.id/"),
        ("Kontan Keuangan (Big Banks, Multifinance, Asuransi)", "https://keuangan.kontan.co.id/"),
        ("Kontan Insight / Riset Eksklusif & Pusat Data", "https://insight.kontan.co.id/"),
        ("Bisnis.com Market & Bursa Saham", "https://market.bisnis.com/bursa-dan-saham"),
        ("Bisnis.com Rekomendasi (Trading Ideas & Target Harga)", "https://market.bisnis.com/rekomendasi"),
        ("Bisnis.com Korporasi & Komoditas", "https://market.bisnis.com/korporasi"),
        ("Bloomberg Technoz Market", "https://www.bloombergtechnoz.com/kanal/market"),
        ("Bloomberg Technoz Finansial, Bisnis & Energi Komoditas", "https://www.bloombergtechnoz.com/kanal/energi"),
        ("EmitenNews — Kanal Emiten (Laba, Aksi Korporasi & Pengendali)", "https://emitennews.com/category/emiten"),
        ("EmitenNews — Radar Borong Saham / Akumulasi Pengendali", "https://emitennews.com/search?q=borong+saham"),
        ("IDX Channel Market News & Live TV", "https://www.idxchannel.com/market-news"),
        ("Investor.id Market & Corporate Action", "https://investor.id/corporate-action"),
        ("IDN Financials (Berita Emiten & Database Kepemilikan)", "https://www.idnfinancials.com/id/news"),
        ("IQPlus, Katadata Bursa, Pasardana, Bareksa, Antara, Detik, Sindonews", "http://www.iqplus.info/"),
    ],
    "3. Sekuritas, Riset Harian & Konsensus Analis": [
        ("Stockbit Snips (Riset Harian, Bedah MSCI & Aksi Korporasi)", "https://snips.stockbit.com/"),
        ("Stockbit Blog & Academy", "https://blog.stockbit.com/"),
        ("Ajaib Berita Saham & Analisis Pasar Modal", "https://ajaib.co.id/belajar/berita-saham/"),
        ("Ajaib Web Trading (Feed Berita, Riset & Forum Chitchat)", "https://invest.ajaib.co.id/"),
        ("IPOTNews (Indo Premier Sekuritas — Berita & Riset Real-Time)", "https://www.ipotnews.com/"),
        ("Mirae Asset Sekuritas (Riset Harian & Market Movers)", "https://sekuritas.miraeasset.co.id/id/research"),
        ("Mandiri Sekuritas (MOST Daily Research)", "https://www.most.co.id/riset/harian"),
        ("BRI Danareksa Sekuritas (BRIGHTS Research)", "https://brights.id/id/research"),
        ("Phillip Sekuritas Indonesia (POEMS Research)", "https://www.poems.co.id/id/research/"),
    ],
    "4. Bandarmology, Whale Tracking, Komunitas Quant & Scalping Lokal": [
        ("Creative Trader (Riset Bandarmology & Foreign Flow — Argha J. Karo Karo)", "https://www.creativetrader.id/"),
        ("Bandarmology.com (Artikel & Riset Akumulasi/Distribusi)", "https://www.bandarmology.com/"),
        ("Stockbit Stream (Trending & Rumor Real-Time)", "https://stockbit.com/stream"),
        ("Stockbit Screener & Chartbit (Bandar Detector & Broker Summary)", "https://stockbit.com/screener"),
        ("Invezgo (Broker Summary, Sankey Flow Bandar & HAKA/HAKI)", "https://app.invezgo.com/"),
        ("Sahamology (Analisa Bandarmology & Teknikal Terpadu)", "https://sahamology.id/"),
        ("Stockwise.id & Andry Hakim (Whale Tracking & Insight)", "https://www.stockwise.id/"),
        ("Emtrade (Riset Harian & Sinyal Trading)", "https://emtrade.id/artikel"),
        ("Big Alpha (Berita Ekonomi & Bedah Emiten)", "https://bigalpha.id/news"),
        ("Dreamstock.id (Screener, Sinyal & Analisa Emiten)", "https://dreamstock.id/"),
        ("SahamGO, PintarSaham & Invesnesia", "https://sahamgo.com/"),
        ("Berduit Official, Michael Yeoh, Bekti Sutikna, Om Ben, Astronacci, Ruang Saham", "https://www.instagram.com/michaelyeoh88/"),
    ],
    "5. Global Macro, Komoditas Futures & Screener Teknikal (TradingView, Investing, Yahoo)": [
        ("TradingView Indonesia — Unusual Volume / Volume Spike", "https://id.tradingview.com/markets/stocks-indonesia/market-movers-unusual-volume/"),
        ("TradingView Indonesia — Top Gainers & Most Active", "https://id.tradingview.com/markets/stocks-indonesia/market-movers-gainers/"),
        ("TradingView Indonesia — Most Volatile & High Beta", "https://id.tradingview.com/markets/stocks-indonesia/market-movers-most-volatile/"),
        ("Investing.com — Berita Saham Indonesia & Harga Komoditas Real-Time", "https://id.investing.com/commodities/real-time-futures"),
        ("TradingEconomics Commodities & Barchart Futures", "https://tradingeconomics.com/commodities"),
        ("Yahoo Finance IHSG (^JKSE) Components & News", "https://finance.yahoo.com/quote/%5EJKSE/news/"),
    ]
}

# --- KUSTOMISASI CSS TEMA NEON FUTURISTIK & BADGE BREAKOUT / MSCI ---
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
    .danger-badge {
        background: linear-gradient(90deg, #ff1e56, #a80038);
        color: #ffffff !important;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 900;
        font-size: 15px;
        text-align: center;
        box-shadow: 0 0 20px rgba(255, 30, 86, 0.85);
        border: 2px solid #ffffff;
        margin-bottom: 15px;
    }
    .msci-banner {
        background: linear-gradient(90deg, #0f172a, #1e293b);
        color: #38bdf8 !important;
        padding: 10px 16px;
        border-radius: 8px;
        border-left: 5px solid #00ffcc;
        border-right: 1px solid #38bdf8;
        font-size: 13.5px;
        margin-bottom: 12px;
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
st.sidebar.header("⚡ QUANT CONFIG: 1.011 EMITEN & MSCI")

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

with st.spinner("Memuat database lengkap 1.011 emiten BEI & Konstituen MSCI..."):
    df_master_bei = load_all_1011_idx_stocks()

# Tambahkan kolom Klasifikasi Resmi MSCI pada Master DataFrame
def classify_msci_segment(code):
    if code in MSCI_GLOBAL_STANDARD_WEIGHTS:
        return f"MSCI Global Standard ({MSCI_GLOBAL_STANDARD_WEIGHTS[code]:.2f}%)"
    elif code in MSCI_SMALL_CAP_WEIGHTS:
        return f"MSCI Small Cap ({MSCI_SMALL_CAP_WEIGHTS[code]:.2f}%)"
    elif code in MSCI_HSC_RESTRICTED_WATCHLIST:
        return "Non-MSCI (HSC / Freeze Watchlist)"
    return "Non-MSCI"

df_master_bei['Status_MSCI'] = df_master_bei['Kode_Saham'].apply(classify_msci_segment)

# --- FILTER MULTI-DIMENSI BERDASARKAN METADATA BEI & MSCI DI SIDEBAR ---
st.sidebar.subheader("🔎 Filter Database 1.011 Saham & MSCI")

hide_delisted = st.sidebar.checkbox("✅ Sembunyikan Saham Delisting (31 Emiten)", value=True)
filter_high_precision_menu = st.sidebar.checkbox("🎯 Filter Emiten Indeks Utama (LQ45 / KOMPAS100 / PEFINDO25)", value=False)
filter_ai_bullish_high_acc = st.sidebar.checkbox("🚀 Filter Sinyal Bullish AI (Naik > 2% & Presisi > 90%)", value=False)

# Filter Segmen MSCI Indonesia
selected_msci_filter = st.sidebar.selectbox(
    "🏆 Filter Konstituen Indeks MSCI",
    options=[
        "Semua Emiten (1.011)",
        "Semua Konstituen MSCI (44 Saham)",
        "MSCI Global Standard Index (9 Saham Utama)",
        "MSCI Indonesia Small Cap Index (35 Saham)",
        "Emiten Non-MSCI (Bebas Tekanan Passive Outflow Asing)"
    ],
    index=0
)

available_sectors = ["Semua Sektor"] + sorted(df_master_bei['Sektor_Indonesia'].dropna().unique().tolist())
selected_sector = st.sidebar.selectbox("Filter Sektor IDX-IC", options=available_sectors, index=0)

available_boards = ["Semua Papan"] + sorted(df_master_bei['Papan_Pencatatan'].dropna().unique().tolist())
selected_board = st.sidebar.selectbox("Filter Papan Pencatatan BEI", options=available_boards, index=0)

selected_index_filter = st.sidebar.selectbox(
    "Filter Konstituen Indeks Domestik",
    options=["Semua Emiten", "LQ45 (45 Saham)", "KOMPAS100 (100 Saham)", "PEFINDO25 (25 Saham)"],
    index=0
)

df_filtered = df_master_bei.copy()
if hide_delisted:
    df_filtered = df_filtered[df_filtered['Status_Pencatatan'] != 'Delisting']
if filter_high_precision_menu:
    df_filtered = df_filtered[df_filtered['Indeks_Utama'] != '-']

if selected_msci_filter == "Semua Konstituen MSCI (44 Saham)":
    df_filtered = df_filtered[df_filtered['Kode_Saham'].isin(list(MSCI_GLOBAL_STANDARD_WEIGHTS.keys()) + list(MSCI_SMALL_CAP_WEIGHTS.keys()))]
elif selected_msci_filter == "MSCI Global Standard Index (9 Saham Utama)":
    df_filtered = df_filtered[df_filtered['Kode_Saham'].isin(list(MSCI_GLOBAL_STANDARD_WEIGHTS.keys()))]
elif selected_msci_filter == "MSCI Indonesia Small Cap Index (35 Saham)":
    df_filtered = df_filtered[df_filtered['Kode_Saham'].isin(list(MSCI_SMALL_CAP_WEIGHTS.keys()))]
elif selected_msci_filter == "Emiten Non-MSCI (Bebas Tekanan Passive Outflow Asing)":
    df_filtered = df_filtered[~df_filtered['Kode_Saham'].isin(list(MSCI_GLOBAL_STANDARD_WEIGHTS.keys()) + list(MSCI_SMALL_CAP_WEIGHTS.keys()))]

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

filtered_stocks = {}
metadata_lookup = {}
for _, row in df_filtered.iterrows():
    code = str(row['Kode_Saham']).strip()
    yf_tck = str(row['Ticker_Yahoo_Finance']).strip()
    name = str(row['Nama_Perusahaan']).strip()
    msci_tag = "🏆MSCI-GS" if code in MSCI_GLOBAL_STANDARD_WEIGHTS else ("🌟MSCI-SC" if code in MSCI_SMALL_CAP_WEIGHTS else row['Papan_Pencatatan'])
    label = f"{code} – {name} [{row['Sektor_Indonesia']} | {msci_tag}]"
    filtered_stocks[label] = yf_tck
    metadata_lookup[yf_tck] = row.to_dict()

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

ticker_symbol = st.sidebar.text_input("Atau Ketik Manual Kode Saham IDX (Contoh: BBCA / BRMS / AADI)", value=selected_ticker_default)
ticker_symbol = ticker_symbol.strip().upper()
if not ticker_symbol.endswith(".JK") and not ticker_symbol.startswith("^"):
    ticker_symbol = f"{ticker_symbol}.JK"

clean_code_global = ticker_symbol.replace(".JK", "")

current_meta = metadata_lookup.get(ticker_symbol, {
    "Kode_Saham": clean_code_global,
    "Ticker_BEI": f"BEI:{clean_code_global}",
    "Ticker_TradingView": f"IDX:{clean_code_global}",
    "Ticker_Yahoo_Finance": ticker_symbol,
    "Ticker_Bloomberg": f"{clean_code_global}:IJ",
    "Nama_Perusahaan": ticker_symbol,
    "Sektor_Indonesia": "IDX Equity",
    "Papan_Pencatatan": "Utama / Pengembangan",
    "Tanggal_IPO_Teks": "-",
    "Jumlah_Saham_Tercatat": 0,
    "Indeks_Utama": "-",
    "Status_Pencatatan": "Aktif",
    "Status_MSCI": classify_msci_segment(clean_code_global)
})

ml_engine = st.sidebar.selectbox(
    "Pilih Neural / Quant Engine AI",
    ["Ensemble (XGBoost + LightGBM Hybrid)", "XGBoost Regressor", "LightGBM Regressor"],
    index=0
)

period_option = st.sidebar.selectbox("Rentang Waktu Analisis", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
interval_option = st.sidebar.selectbox("Interval Candle", ["1d", "1wk"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Parameter Simulasi MSCI & Makro")
est_free_float_pct = st.sidebar.slider("Asumsi Porsi Free-Float (FIF) Emiten (%)", min_value=5, max_value=85, value=40, step=5)
msci_em_passive_aum_bil = st.sidebar.number_input("Total Dana Pasif Global Acuan MSCI EM (US$ Miliar)", min_value=100.0, max_value=3000.0, value=1000.0, step=100.0)
bi_rate_input = st.sidebar.number_input("Asumsi Suku Bunga BI Rate (%)", min_value=3.0, max_value=10.0, value=5.75, step=0.25)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Kustomisasi Tampilan Grafik")
show_vwap = st.sidebar.checkbox("Tampilkan Garis VWAP", value=True)
show_ma200 = st.sidebar.checkbox("Tampilkan MA 50 & MA 200", value=True)
show_ichimoku = st.sidebar.checkbox("Tampilkan Ichimoku Cloud", value=True)
show_sar = st.sidebar.checkbox("Tampilkan Parabolic SAR", value=False)
show_sr_lines = st.sidebar.checkbox("Tampilkan Dynamic Support/Resistance", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Input Manajemen Modal (Trading Calculator)")
user_capital = st.sidebar.number_input("Total Modal Trading (IDR)", min_value=100000, value=10000000, step=500000)
risk_tolerance_pct = st.sidebar.slider("Maksimal Risiko per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Status MSCI & Omni-Channel")
st.sidebar.success("🟢 MSCI Engine Aktif (9 Global Standard, 35 Small Cap, Freeze Policy Radar)")

# --- FUNGSI PENGAMBILAN DATA MAKRO & LIVE TICKER ---
@st.cache_data(ttl=60)
def fetch_macro_and_watchlist_tape():
    macro_tickers = {
        "IHSG": "^JKSE",
        "USD/IDR": "IDR=X",
        "Crude Oil": "CL=F",
        "Gold": "GC=F",
        "BBCA (MSCI 31.4%)": "BBCA.JK",
        "BBRI (MSCI 20.0%)": "BBRI.JK",
        "BMRI (MSCI 15.7%)": "BMRI.JK",
        "TLKM (MSCI 10.7%)": "TLKM.JK",
        "ASII (MSCI 7.8%)": "ASII.JK",
        "BRMS (MSCI 3.5%)": "BRMS.JK"
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
live_usd_idr = macro_data.get("USD/IDR", {}).get("price", 18000.0)
if live_usd_idr < 10000:
    live_usd_idr = 18000.0

if macro_data:
    tape_items = []
    for k, v in macro_data.items():
        color = "#00ff66" if v['change'] >= 0 else "#ff3366"
        sign = "+" if v['change'] >= 0 else ""
        tape_items.append(
            f"<span style='margin-right: 25px;'><b>{k}:</b> {v['price']:,.2f} "
            f"<span style='color:{color};'>({sign}{v['change']:.2f}%)</span></span>"
        )
    st.markdown(f"<div class='ticker-tape'>⚡ <b>LIVE MARKET, MACRO & MSCI HEAVYWEIGHT TAPE:</b> {' | '.join(tape_items)}</div>", unsafe_allow_html=True)

st.title("⚡ QUANT AI: Quadruple-Threat IDX (1.011 Emiten, MSCI Engine & 140+ Omni-News)")
st.markdown(
    f"Analisis terpadu **4 Pilar Quadruple-Threat**, **MSCI Institutional & Rebalancing Freeze Simulator**, serta **6-Layer Omni-Channel NLP** untuk "
    f"**{current_meta.get('Nama_Perusahaan', ticker_symbol)} ({ticker_symbol})** | "
    f"Status MSCI: **{current_meta.get('Status_MSCI', 'Non-MSCI')}** | "
    f"Sektor: **{current_meta.get('Sektor_Indonesia', '-')}** | Papan: **{current_meta.get('Papan_Pencatatan', '-')}**"
)

# Banner Informasi Status Kebijakan MSCI Indonesia Terkini
st.markdown(
    f"<div class='msci-banner'>"
    f"🌐 <b>RADAR INSTITUSIONAL MSCI INDONESIA (UPDATE OKTOBER 2026):</b> Bobot Indonesia di MSCI EM = <b>0,43%</b> (Free-Float MCap <b>US$53,36 Miliar</b> | 9 Global Standard & 35 Small Cap). "
    f"Status Kebijakan: <b>MSCI Rebalancing Freeze Aktif</b> (Pembekuan kenaikan FIF/NOS, penghentian penambahan emiten baru ke IMI & migrasi segmen selama masa konsultasi reklasifikasi hingga Juni 2027)."
    f"</div>",
    unsafe_allow_html=True
)

if current_meta.get("Status_Pencatatan") == "Delisting":
    st.error(
        f"⛔ **PERINGATAN DELISTING**: Emiten **{ticker_symbol} ({current_meta.get('Nama_Perusahaan')})** berstatus **Delisting / Tidak Aktif** di BEI."
    )
elif current_meta.get("Papan_Pencatatan") == "Pemantauan Khusus":
    st.warning(
        f"⚠️ **PERINGATAN PAPAN PEMANTAUAN KHUSUS (FULL CALL AUCTION / FCA)**: "
        f"Emiten **{ticker_symbol}** berada di **Papan Pemantauan Khusus** BEI dan otomatis **dikecualikan/dikeluarkan dari indeks MSCI**."
    )

# --- MESIN AGREGATOR PARALEL MULTI-RSS & WEIGHTED NLP (TERMASUK RADAR MSCI) ---
def _fetch_single_rss(feed_meta):
    layer_name, source_label, url, cred_weight = feed_meta
    entries_out = []
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            raw_bytes = resp.read()
        parsed = feedparser.parse(raw_bytes)
        for entry in parsed.entries[:4]:
            entries_out.append({
                "layer": layer_name,
                "source_group": source_label,
                "title": entry.get("title", ""),
                "link": entry.get("link", url),
                "published": entry.get("published", "Terbaru"),
                "cred_weight": cred_weight
            })
    except Exception:
        pass
    return entries_out

@st.cache_data(ttl=120)
def fetch_omni_channel_news_intelligence(ticker):
    clean_code = ticker.replace(".JK", "").upper()

    multi_rss_targets = [
        # Layer 1: Spesifik Emiten & Radar MSCI
        ("Layer 1: Emiten Spesifik", f"Google News ({clean_code})", f"https://news.google.com/rss/search?q=saham+{clean_code}+when:7d&hl=id&gl=ID&ceid=ID:id", 1.5),
        ("Layer 1: Emiten Spesifik", f"Yahoo Finance ({ticker})", f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker},%5EJKSE&region=US&lang=en-US", 1.3),
        ("Layer 1: Radar MSCI & Asing", "Radar MSCI Indonesia & Rebalancing", 'https://news.google.com/rss/search?q="MSCI"+(Indonesia+OR+saham+OR+IHSG+OR+rebalancing+OR+standalone)+when:7d&hl=id&gl=ID&ceid=ID:id', 1.5),
        # Layer 2: Regulator BEI, IDX Channel & Detektor UMA/Suspensi/Aksi Korporasi
        ("Layer 2: Regulator & Bursa", "BEI & IDX Channel Resmi", "https://news.google.com/rss/search?q=site:idxchannel.com+OR+site:idx.co.id+when:1d&hl=id&gl=ID&ceid=ID:id", 1.5),
        ("Layer 2: Regulator & Bursa", "Radar UMA & Suspensi BEI", 'https://news.google.com/rss/search?q="Unusual+Market+Activity"+OR+"Suspensi+Saham"+BEI+when:2d&hl=id&gl=ID&ceid=ID:id', 1.5),
        ("Layer 2: Regulator & Bursa", "EmitenNews (Borong/Dividen/Akuisisi)", 'https://news.google.com/rss/search?q=site:emitennews.com+(borong+OR+akuisisi+OR+dividen+OR+"rekomendasi+saham"+OR+unsuspensi)+when:2d&hl=id&gl=ID&ceid=ID:id', 1.4),
        ("Layer 2: Regulator & Bursa", "IDN Financials & Investor.id", "https://news.google.com/rss/search?q=site:idnfinancials.com+OR+site:investor.id/market+OR+site:investor.id/corporate-action+when:2d&hl=id&gl=ID&ceid=ID:id", 1.4),
        # Layer 3: Media Finansial Tier-1 (CNBC, Kontan, Bisnis.com, Bloomberg Technoz)
        ("Layer 3: Media Tier-1", "CNBC & Kontan Rekomendasi", 'https://news.google.com/rss/search?q=(site:cnbcindonesia.com+OR+site:kontan.co.id)+("rekomendasi+saham"+OR+"saham+pilihan"+OR+"target+harga")+when:1d&hl=id&gl=ID&ceid=ID:id', 1.3),
        ("Layer 3: Media Tier-1", "Bloomberg Technoz Market & Rekomendasi", 'https://news.google.com/rss/search?q=site:bloombergtechnoz.com+(saham+OR+IHSG+OR+"rekomendasi+saham"+OR+"target+harga")+when:1d&hl=id&gl=ID&ceid=ID:id', 1.3),
        ("Layer 3: Media Tier-1", "Bisnis.com Market & Rekomendasi", 'https://news.google.com/rss/search?q=site:market.bisnis.com+("rekomendasi+saham"+OR+"saham+pilihan"+OR+"target+harga"+OR+IHSG)+when:1d&hl=id&gl=ID&ceid=ID:id', 1.3),
        ("Layer 3: Media Tier-1", "CNBC Market Direct RSS", "https://www.cnbcindonesia.com/market/rss", 1.2),
        ("Layer 3: Media Tier-1", "Kontan Investasi Direct RSS", "https://rss.kontan.co.id/news/investasi", 1.2),
        # Layer 4: Riset Sekuritas (Stockbit Snips, Ajaib, Konsensus Sekuritas)
        ("Layer 4: Riset Sekuritas", "Stockbit Snips & Sekuritas", 'https://news.google.com/rss/search?q=site:snips.stockbit.com+OR+site:blog.stockbit.com+OR+"Stockbit+Sekuritas"+when:7d&hl=id&gl=ID&ceid=ID:id', 1.3),
        ("Layer 4: Riset Sekuritas", "Ajaib Sekuritas & Konsensus", 'https://news.google.com/rss/search?q="Ajaib+Sekuritas"+OR+"rekomendasi+saham+hari+ini"+when:1d&hl=id&gl=ID&ceid=ID:id', 1.3),
        # Layer 5: Bandarmology, Foreign Flow & Komunitas Quant Lokal
        ("Layer 5: Bandarmology & Whale", "Bandarmology & Akumulasi Asing", 'https://news.google.com/rss/search?q="Bandarmology"+OR+"Creative+Trader"+OR+"akumulasi+asing"+OR+"net+buy+asing"+saham+when:3d&hl=id&gl=ID&ceid=ID:id', 1.25),
        ("Layer 5: Bandarmology & Whale", "Stockwise, Big Alpha & Emtrade", 'https://news.google.com/rss/search?q=site:stockwise.id+OR+site:bigalpha.id+OR+site:emtrade.id+OR+site:dreamstock.id+OR+site:sahamgo.com+when:7d&hl=id&gl=ID&ceid=ID:id', 1.2),
        # Layer 6: Komoditas Global & Makro
        ("Layer 6: Makro & Komoditas", "Bloomberg Technoz Komoditas", 'https://news.google.com/rss/search?q=site:bloombergtechnoz.com+(emas+OR+nikel+OR+"batu+bara"+OR+minyak)+when:2d&hl=id&gl=ID&ceid=ID:id', 1.15),
        ("Layer 6: Makro & Komoditas", "Investing.com Saham & Komoditas", "https://id.investing.com/rss/news_25.rss", 1.1)
    ]

    raw_articles = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_fetch_single_rss, target) for target in multi_rss_targets]
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                raw_articles.extend(res)

    kw_insider_whale = ["borong saham", "akumulasi asing", "net buy asing", "buyback", "akuisisi", "pengendali", "insider", "private placement", "akumulasi bandar", "danantara"]
    kw_fundamental_div = ["laba melonjak", "laba naik", "dividen jumbo", "dividen tunai", "dividen interim", "cum date", "kontrak baru", "ekspansi", "pendapatan naik", "rekor tertinggi", "tumbuh positif"]
    kw_analyst_buy = ["rekomendasi saham", "saham pilihan", "target harga", "strong buy", "potensi menguat", "breakout", "top picks", "beli", "menguat", "rebound", "unsuspensi", "buka gembok", "masuk msci"]
    kw_regulatory_danger = ["suspensi", "unusual market activity", "uma", "gagal bayar", "default", "pkpu", "pailit", "delisting", "sanksi ojk", "gembok", "manipulasi pasar"]
    kw_msci_pressure = ["keluar msci", "depak dari msci", "standalone market", "downgrade msci", "outflow asing", "net sell asing", "pembekuan msci"]
    kw_bearish_dist = ["rugi bersih", "laba anjlok", "distribusi", "tekanan jual", "koreksi", "melemah", "anjlok", "divestasi", "trading halt"]

    seen_titles = set()
    processed_news = []
    emiten_specific_score = 0.0
    market_macro_score = 0.0
    detected_catalysts = set()
    uma_or_suspend_alert = False

    for item in raw_articles:
        title_clean = item["title"].strip()
        if not title_clean or title_clean in seen_titles:
            continue
        seen_titles.add(title_clean)

        title_lower = title_clean.lower()
        link_lower = item["link"].lower()
        w = item["cred_weight"]
        is_about_selected_ticker = (clean_code.lower() in title_lower) or (item["source_group"] == f"Google News ({clean_code})")

        publisher = item["source_group"]
        if "kontan" in link_lower or "kontan" in title_lower:
            publisher = "Kontan Investasi/Industri"
        elif "cnbcindonesia" in link_lower or "cnbc" in title_lower:
            publisher = "CNBC Indonesia"
        elif "bloombergtechnoz" in link_lower or "bloomberg" in title_lower:
            publisher = "Bloomberg Technoz"
        elif "bisnis.com" in link_lower or "bisnis" in title_lower:
            publisher = "Bisnis.com Market"
        elif "emitennews" in link_lower or "emitennews" in title_lower:
            publisher = "EmitenNews"
        elif "idxchannel" in link_lower:
            publisher = "IDX Channel"
        elif "idnfinancials" in link_lower:
            publisher = "IDN Financials"
        elif "investor.id" in link_lower:
            publisher = "Investor.id"
        elif "stockbit" in link_lower or "stockbit" in title_lower:
            publisher = "Stockbit Snips / Sekuritas"
        elif "ajaib" in link_lower or "ajaib" in title_lower:
            publisher = "Ajaib Sekuritas"

        raw_delta = 0.0
        tags = []

        if "msci" in title_lower:
            tags.append("🌐 MSCI INDEX RADAR")

        if any(k in title_lower for k in kw_regulatory_danger):
            raw_delta -= 3.0 * w
            tags.append("⚠️ REGULATOR / UMA / SUSPENSI")
            if is_about_selected_ticker:
                uma_or_suspend_alert = True
                detected_catalysts.add("🚨 Peringatan UMA / Suspensi / Risiko Regulasi")

        if any(k in title_lower for k in kw_msci_pressure):
            raw_delta -= 1.5 * w
            tags.append("🌊 TEKANAN REBALANCING MSCI / OUTFLOW")
            if is_about_selected_ticker:
                detected_catalysts.add("🌊 Volatilitas Rebalancing / Outflow Pasif MSCI")

        if any(k in title_lower for k in kw_insider_whale):
            raw_delta += 2.5 * w
            tags.append("🐋 WHALE / INSIDER / AKUMULASI")
            if is_about_selected_ticker:
                detected_catalysts.add("🐋 Akumulasi Institusi / Borong Saham / Buyback")

        if any(k in title_lower for k in kw_fundamental_div):
            raw_delta += 2.0 * w
            tags.append("💰 DIVIDEN / LABA / KORPORASI")
            if is_about_selected_ticker:
                detected_catalysts.add("💰 Katalis Dividen / Pertumbuhan Laba / Ekspansi")

        if any(k in title_lower for k in kw_analyst_buy):
            raw_delta += 1.5 * w
            tags.append("🎯 REKOMENDASI ANALIS / TARGET HARGA")
            if is_about_selected_ticker:
                detected_catalysts.add("🎯 Rekomendasi Beli Analis / Konsensus Sekuritas")

        if any(k in title_lower for k in kw_bearish_dist):
            raw_delta -= 1.5 * w
            tags.append("🔻 TEKANAN JUAL / KOREKSI")

        if raw_delta >= 2.0:
            sent_badge = f"Sangat Bullish 🟢 ({raw_delta:+.1f})"
        elif raw_delta > 0:
            sent_badge = f"Bullish 🟢 ({raw_delta:+.1f})"
        elif raw_delta <= -2.0:
            sent_badge = f"Sangat Bearish / Risiko 🔴 ({raw_delta:+.1f})"
        elif raw_delta < 0:
            sent_badge = f"Bearish 🔴 ({raw_delta:+.1f})"
        else:
            sent_badge = "Netral / Informatif 🟡 (0.0)"

        if is_about_selected_ticker:
            emiten_specific_score += raw_delta
        else:
            market_macro_score += (raw_delta * 0.30)

        processed_news.append({
            "layer": item["layer"],
            "publisher": publisher,
            "title": title_clean,
            "link": item["link"],
            "date": item["published"],
            "sentiment": sent_badge,
            "score": raw_delta,
            "tags": ", ".join(tags) if tags else "Update Pasar",
            "is_emiten": is_about_selected_ticker
        })

    processed_news.sort(key=lambda x: (x["is_emiten"], "MSCI" in x["tags"], abs(x["score"])), reverse=True)
    combined_nlp_score = round(emiten_specific_score + market_macro_score, 1)

    return processed_news, combined_nlp_score, round(emiten_specific_score, 1), list(detected_catalysts), uma_or_suspend_alert

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
        "bvps": None,
        "float_shares": 0
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
        fund_dict["float_shares"] = info.get("floatShares", 0) or 0
    except Exception:
        pass
    return fund_dict

data, benchmark_close = fetch_stock_and_market_data(ticker_symbol, period_option, interval_option)
fund_info = fetch_fundamental_metrics(ticker_symbol)
realtime_news, news_sentiment_score, emiten_nlp_score, active_catalysts, has_uma_risk = fetch_omni_channel_news_intelligence(ticker_symbol)

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

        # --- PERHITUNGAN METRIK KUANTITATIF MSCI (FIF, ATVR, PASSIVE OUTFLOW/INFLOW) ---
        csv_shares = float(current_meta.get("Jumlah_Saham_Tercatat", 0) or 0)
        calculated_mcap_csv = csv_shares * latest_close
        mcap_idr = fund_info["market_cap"] if fund_info["market_cap"] > 0 else calculated_mcap_csv
        mcap_usd_m = (mcap_idr / live_usd_idr) / 1e6 if live_usd_idr > 0 else 0.0

        # Estimasi Free-Float Market Cap (FIF)
        fif_ratio = est_free_float_pct / 100.0
        ff_mcap_idr = mcap_idr * fif_ratio
        ff_mcap_usd_m = mcap_usd_m * fif_ratio

        # Perhitungan ATVR (Annualized Traded Value Ratio = ADTV 20D * 250 hari bursa / Free-Float Market Cap)
        adtv_20d_idr = float((data['Close'] * data['Volume']).rolling(20).mean().iloc[-1])
        annualized_traded_val_idr = adtv_20d_idr * 250.0
        atvr_pct = (annualized_traded_val_idr / ff_mcap_idr) * 100.0 if ff_mcap_idr > 0 else 0.0

        # Status Konstituen & Eksposur Dana Pasif MSCI
        indonesia_em_weight = 0.0043  # 0.43% bobot Indonesia di MSCI EM
        total_indo_passive_pool_usd_m = msci_em_passive_aum_bil * 1000.0 * indonesia_em_weight  # ~US$ 4,300 Juta

        if clean_code_global in MSCI_GLOBAL_STANDARD_WEIGHTS:
            msci_weight_pct = MSCI_GLOBAL_STANDARD_WEIGHTS[clean_code_global]
            msci_tier_label = f"🏆 Anggota Resmi MSCI Global Standard (Bobot: {msci_weight_pct:.2f}%)"
            passive_exposure_usd_m = total_indo_passive_pool_usd_m * (msci_weight_pct / 100.0)
            passive_exposure_idr_t = (passive_exposure_usd_m * 1e6 * live_usd_idr) / 1e12
            days_to_absorb_outflow = (passive_exposure_idr_t * 1e12) / adtv_20d_idr if adtv_20d_idr > 0 else 0.0
            msci_freeze_impact = "Berdampak Langsung: Kenaikan FIF/NOS dibekukan; rentan fluktuasi program buy/sell ETF global."
        elif clean_code_global in MSCI_SMALL_CAP_WEIGHTS:
            msci_weight_pct = MSCI_SMALL_CAP_WEIGHTS[clean_code_global]
            msci_tier_label = f"🌟 Anggota Resmi MSCI Indonesia Small Cap (Bobot Est: {msci_weight_pct:.2f}%)"
            passive_exposure_usd_m = (total_indo_passive_pool_usd_m * 0.16) * (msci_weight_pct / 100.0)
            passive_exposure_idr_t = (passive_exposure_usd_m * 1e6 * live_usd_idr) / 1e12
            days_to_absorb_outflow = (passive_exposure_idr_t * 1e12) / adtv_20d_idr if adtv_20d_idr > 0 else 0.0
            msci_freeze_impact = "Migrasi Naik Kelas Dibekukan: Tidak dapat naik ke Global Standard selama periode MSCI Freeze."
        elif clean_code_global in MSCI_HSC_RESTRICTED_WATCHLIST or current_meta.get("Papan_Pencatatan") == "Pemantauan Khusus":
            msci_weight_pct = 0.0
            msci_tier_label = "⛔ Dikecualikan / Watchlist Konsentrasi Kepemilikan Tinggi (HSC / FCA)"
            passive_exposure_usd_m = 0.0
            passive_exposure_idr_t = 0.0
            days_to_absorb_outflow = 0.0
            msci_freeze_impact = "Restriksi Penuh MSCI: Masuk pengawasan konsentrasi kepemilikan (HSC) atau Papan FCA."
        else:
            msci_weight_pct = 0.0
            passive_exposure_usd_m = 0.0
            passive_exposure_idr_t = 0.0
            days_to_absorb_outflow = 0.0
            if ff_mcap_usd_m >= 1500 and atvr_pct >= 15.0:
                msci_tier_label = "🔒 Kandidat Potensial Global Standard (Tertahan Kebijakan MSCI Freeze)"
                msci_freeze_impact = "Memenuhi syarat Market Cap & ATVR (>15%), namun penambahan konstituen baru sedang dihentikan MSCI."
            elif ff_mcap_usd_m >= 170 and atvr_pct >= 15.0:
                msci_tier_label = "🔒 Kandidat Potensial Small Cap (Tertahan Kebijakan MSCI Freeze)"
                msci_freeze_impact = "Memenuhi syarat Small Cap & ATVR (>15%), namun penambahan konstituen baru sedang dihentikan MSCI."
            else:
                msci_tier_label = "🛡️ Non-MSCI / Domestic Driven (Imun dari Tekanan Jual ETF Asing)"
                msci_freeze_impact = "Bebas dari risiko Forced Outflow dana pasif MSCI; pergerakan murni didorong katalis domestik."

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

        if latest_cmf > 0.10: flow_score += 9
        elif latest_cmf > 0: flow_score += 5
        if five_day_smart_flow_idr > 0: flow_score += 5
        if latest_mfi > 55: flow_score += 4
        if atvr_pct >= 15.0: flow_score += 3

        # Penyesuaian Skor MSCI berdasarkan Arus Dana Aktual
        if clean_code_global in MSCI_GLOBAL_STANDARD_WEIGHTS or clean_code_global in MSCI_SMALL_CAP_WEIGHTS:
            if five_day_smart_flow_idr > 0:
                flow_score += 4  # Konstituen MSCI yang sedang diakumulasi kembali
            elif latest_rsi < 32:
                flow_score += 3  # Oversold ekstrim akibat penjualan mekanikal ETF (peluang Buy on Weakness)
            else:
                flow_score += 1  # Waspada tekanan outflow pasif
        elif clean_code_global not in MSCI_HSC_RESTRICTED_WATCHLIST:
            flow_score += 3  # Keunggulan defensif bebas outflow pasif MSCI
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
        if fund_info["div_yield"] >= 3.5 or any("Dividen" in c for c in active_catalysts): fund_score += 4
        fund_score = min(25, fund_score)

        macro_score = 7
        usd_idr_chg = macro_data.get("USD/IDR", {}).get("change", 0.0)
        if usd_idr_chg <= 0.1:
            macro_score += 2
        else:
            macro_score -= 2
        if news_sentiment_score >= 4: macro_score += 6
        elif news_sentiment_score > 0: macro_score += 4
        elif news_sentiment_score < -2: macro_score -= 4
        if has_uma_risk:
            macro_score = max(0, macro_score - 6)
        macro_score = max(0, min(15, macro_score))

        total_alpha_score = int(tech_score + flow_score + fund_score + macro_score)

        if has_uma_risk:
            action_signal = "HIGH RISK / UMA / SUSPENSI RADAR — HINDARI / WASPADA 🔴"
        elif total_alpha_score >= 80:
            action_signal = "STRONG BUY (Breakout + Institutional/MSCI Alignment) 🟢"
        elif total_alpha_score >= 65:
            action_signal = "BUY / ACCUMULATE ON WEAKNESS (Trend / Reversal) 🟢"
        elif total_alpha_score >= 45:
            action_signal = "HOLD / WAIT & SEE (Area Konsolidasi / Pantau Flow Asing) 🟡"
        else:
            action_signal = "TAKE PROFIT / CUT LOSS (Distribusi Asing / Tren Patah) 🔴"

        # --- BADGE NOTIFIKASI VOLUME SPIKE / BREAKOUT / UMA DANGER ---
        if has_uma_risk:
            st.markdown(
                f"<div class='danger-badge'>"
                f"🚨 RADAR PERINGATAN BURSA [{ticker_symbol}]: Terdeteksi Kata Kunci UMA / Suspensi / Sanksi Regulasi pada Berita Terkini! "
                f"Cek Keterbukaan Informasi BEI Sebelum Entri."
                f"</div>",
                unsafe_allow_html=True
            )
        elif (is_volume_spike and pct_change > 0) or is_price_breakout or (pred_pct_change > 2.0 and total_alpha_score >= 75):
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
                f"Likuiditas ATVR: <b>{atvr_pct:.1f}%</b> | Skor NLP 140+ Portal: <b>{news_sentiment_score:+.1f}</b>"
                f"</div>",
                unsafe_allow_html=True
            )

        # --- TAUTAN CEPAT AUDIT EMITEN ---
        st.markdown(
            f"🔗 **Audit Cepat 1-Klik [{clean_code_global}]:** "
            f"[Stockbit Stream & Bandar Detector](https://stockbit.com/symbol/{clean_code_global}) | "
            f"[Stockbit Insider (>5%)](https://stockbit.com/symbol/{clean_code_global}/insider) | "
            f"[MSCI Indonesia Official](https://www.msci.com/indexes/index/105767/msci-indonesia-index) | "
            f"[EmitenNews ({clean_code_global})](https://emitennews.com/search?q={clean_code_global}) | "
            f"[Keterbukaan Informasi BEI](https://www.idx.co.id/id/perusahaan-tercatat/keterbukaan-informasi/) | "
            f"[TradingView Chart](https://id.tradingview.com/symbols/IDX-{clean_code_global}/)"
        )

        # --- METRIK UTAMA ---
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Harga Terakhir (IDR)", f"{latest_close:,.0f}", f"{pct_change:+.2f}%")
        col2.metric("Prediksi AI (5 Hari)", f"{predicted_target:,.0f}", f"{pred_pct_change:+.2f}%")
        col3.metric("Skor Alpha-Matrix", f"{total_alpha_score} / 100", f"Presisi ML: {model_accuracy_score:.1f}%")
        col4.metric("Estimasi Fair Value", f"{fair_value_est:,.0f}", f"MoS: {margin_of_safety:+.1f}%")
        col5.metric("Bobot MSCI & ATVR", f"{msci_weight_pct:.2f}% MSCI", f"ATVR: {atvr_pct:.1f}%")

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

        # --- PANEL 6 TAB: 4 PILAR QUADRUPLE-THREAT + MSCI ENGINE + TABEL 1.011 SAHAM + 140+ PORTAL BERITA ---
        st.markdown("---")
        st.subheader("🏛️ Bedah 4 Pilar Quadruple-Threat, MSCI Institutional Engine & Direktori 140+ Portal")
        tab_p1, tab_p2, tab_p3, tab_p4, tab_p5, tab_p6 = st.tabs([
            "📊 1. Fundamental & Identitas BEI",
            "🏆 2. MSCI Rebalancing, FIF, ATVR & Foreign Flow",
            "📡 3. Omni-Channel News NLP (Real-Time)",
            "🧮 4. Target Harian AI & Kalkulator Lot ATR",
            "🗂️ 5. Database 1.011 Saham BEI",
            "🌐 6. Command Center 140+ Portal & RSS"
        ])

        with tab_p1:
            c_f1, c_f2 = st.columns(2)
            with c_f1:
                st.markdown(f"#### Kinerja Keuangan & Metadata Pencatatan ({ticker_symbol})")
                df_fund_table = pd.DataFrame({
                    "Parameter Fundamental & Pencatatan BEI": [
                        "Nama Resmi Emiten",
                        "Status Keanggotaan MSCI Indonesia",
                        "Sektor IDX-IC / Papan Pencatatan",
                        "Jumlah Saham Tercatat (Database BEI)",
                        "Full Market Cap (IDR / USD)",
                        "Free-Float Market Cap (FIF Estimasi)",
                        "Price to Earnings Ratio (PER)",
                        "Price to Book Value (PBV)",
                        "Return on Equity (ROE)",
                        "Estimasi Dividend Yield"
                    ],
                    "Nilai Aktual": [
                        str(current_meta.get("Nama_Perusahaan", ticker_symbol)),
                        str(current_meta.get("Status_MSCI", "-")),
                        f"{current_meta.get('Sektor_Indonesia', '-')} ({current_meta.get('Papan_Pencatatan', '-')})",
                        f"{int(csv_shares):,} Lembar" if csv_shares > 0 else "N/A",
                        f"Rp {mcap_idr / 1e12:,.2f} T (US$ {mcap_usd_m:,.1f} Juta)",
                        f"Rp {ff_mcap_idr / 1e12:,.2f} T (US$ {ff_mcap_usd_m:,.1f} Juta)",
                        f"{per_val:.2f}x" if per_val else "N/A",
                        f"{pbv_val:.2f}x" if pbv_val else "N/A",
                        f"{roe_val:.2f}%" if roe_val is not None else "N/A",
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
                    f"- **Indeks Domestik BEI:** **{current_meta.get('Indeks_Utama', '-')}**\n"
                    f"- **Kontribusi Skor Fundamental:** **{fund_score} / 25 Poin**"
                )

        with tab_p2:
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                st.markdown(f"#### 🏆 Analisis Kuantitatif MSCI & Eksposur Dana Pasif ({clean_code_global})")
                atvr_pass = "✅ Memenuhi Syarat MSCI (>=15%)" if atvr_pct >= 15.0 else "⚠️ Di Bawah Ambang Likuiditas MSCI (<15%)"
                st.success(
                    f"**Status Indeks MSCI:**\n{msci_tier_label}\n\n"
                    f"- **Full Market Cap:** US$ {mcap_usd_m:,.2f} Juta (Rp {mcap_idr / 1e12:,.2f} Triliun)\n"
                    f"- **Free-Float Market Cap (FIF {est_free_float_pct}%):** US$ {ff_mcap_usd_m:,.2f} Juta (Rp {ff_mcap_idr / 1e12:,.2f} Triliun)\n"
                    f"- **Rasio Likuiditas Tahunan (ATVR):** **{atvr_pct:.2f}%** ({atvr_pass})\n"
                    f"- **Eksposur Dana Pasif Asing (MSCI Passive Risk):** **US$ {passive_exposure_usd_m:,.2f} Juta (Rp {passive_exposure_idr_t:,.2f} Triliun)**\n"
                    f"- **Estimasi Hari Serapan Transaksi (Outflow Impact):** `{days_to_absorb_outflow:.1f} Hari Bursa ADTV`\n"
                    f"- **Dampak Kebijakan MSCI Freeze (Okt 2026):** {msci_freeze_impact}"
                )
            with c_m2:
                st.markdown("#### 🌊 Pantauan Smart Money, Bandarmology & Tabel 9 MSCI Global Standard")
                flow_dir = "AKUMULASI BERSIH (Net Buy Proxy) 🟢" if five_day_smart_flow_idr > 0 else "DISTRIBUSI BERSIH (Net Sell Proxy) 🔴"
                st.info(
                    f"📌 **Indikator Arus Dana Institusi & Asing ({clean_code_global}):**\n"
                    f"- **Status Arus Dana (5H):** **{flow_dir}** (Rp {five_day_smart_flow_idr / 1e9:+,.2f} Miliar)\n"
                    f"- **Chaikin Money Flow (CMF 20):** `{latest_cmf:+.3f}` | **MFI (14):** `{latest_mfi:.1f}`\n"
                    f"- **Rata-Rata Nilai Transaksi Harian (ADTV 20D):** Rp {adtv_20d_idr / 1e9:,.2f} Miliar"
                )
                df_msci_gs_ref = pd.DataFrame([
                    {"Kode": k, "Bobot MSCI GS (%)": f"{w:.2f}%", "Eksposur Pasif (US$ Juta)": f"${(total_indo_passive_pool_usd_m * w / 100):,.1f}M", "Eksposur (Rp Triliun)": f"Rp {((total_indo_passive_pool_usd_m * w / 100)*1e6*live_usd_idr)/1e12:,.2f} T"}
                    for k, w in MSCI_GLOBAL_STANDARD_WEIGHTS.items()
                ])
                st.caption("📋 **Referensi Bobot & Risiko Outflow 9 Konstituen MSCI Global Standard Indonesia:**")
                st.dataframe(df_msci_gs_ref, use_container_width=True, hide_index=True, height=210)

        with tab_p3:
            c_n1, c_n2 = st.columns([1, 2])
            with c_n1:
                st.markdown("#### Ringkasan Skor 6-Layer Omni-NLP")
                st.write(f"- **Skor Sentimen Spesifik {clean_code_global}:** `{emiten_nlp_score:+.1f}`")
                st.write(f"- **Skor Sentimen Agregat (Emiten + MSCI + Makro):** `{news_sentiment_score:+.1f}`")
                st.write(f"- **Kurs Acuan USD/IDR:** `Rp {live_usd_idr:,.0f}`")
                st.write(f"- **Suku Bunga BI Rate:** `{bi_rate_input:.2f}%`")
                st.write(f"- **Beta Saham terhadap IHSG:** `{stock_beta:.2f}`")
                st.write(f"- **Skor Pilar Makro & Sentimen:** **{macro_score} / 15 Poin**")
                if active_catalysts:
                    st.success("**Katalis Terdeteksi pada Berita:**\n" + "\n".join(f"- {c}" for c in active_catalysts))
                else:
                    st.info("Belum ada katalis ekstrem spesifik emiten dalam 48 jam terakhir.")
            with c_n2:
                st.markdown("#### 📡 Feed Berita Real-Time Multi-Portal (MSCI, BEI, CNBC, Kontan, Bisnis, Bloomberg Technoz, EmitenNews, Stockbit)")
                col_f_n1, col_f_n2 = st.columns(2)
                filter_only_emiten = col_f_n1.checkbox(f"Hanya berita {clean_code_global}", value=False)
                filter_only_msci = col_f_n2.checkbox("Hanya berita terkait MSCI / Foreign Flow", value=False)
                display_list = realtime_news
                if filter_only_emiten:
                    display_list = [n for n in display_list if n["is_emiten"]]
                if filter_only_msci:
                    display_list = [n for n in display_list if "msci" in n["title"].lower() or "asing" in n["title"].lower() or "MSCI" in n["tags"]]
                display_list = display_list[:18]

                if display_list:
                    for item in display_list:
                        prefix_badge = f"🎯 **[{clean_code_global}]** " if item["is_emiten"] else ""
                        st.markdown(f"- {prefix_badge}**[{item['title']}]({item['link']})**")
                        st.caption(
                            f"📌 **{item['publisher']}** ({item['layer']}) | Kategori: `{item['tags']}` | "
                            f"Waktu: {item['date']} | Skor NLP: **{item['sentiment']}**"
                        )
                else:
                    st.info("Tidak ada berita yang cocok dengan saringan spesifik di atas. Hilangkan centang untuk melihat semua berita pasar.")

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
            st.markdown(f"#### 🗂️ Tabel Master Emiten BEI & Status MSCI ({len(df_filtered)} dari {len(df_master_bei)} Emiten)")
            st.dataframe(df_filtered, use_container_width=True, height=420)

        with tab_p6:
            st.markdown("#### 🌐 Direktori Interaktif 140+ Portal Resmi MSCI, BEI, Media Finansial, Sekuritas & Bandarmology")
            for cat_title, links_list in OMNI_NEWS_DIRECTORY.items():
                with st.expander(f"📂 {cat_title} ({len(links_list)} Kanal Resmi)", expanded=False):
                    cols_dir = st.columns(2)
                    for idx_l, (label_l, url_l) in enumerate(links_list):
                        cols_dir[idx_l % 2].markdown(f"- 🔗 [{label_l}]({url_l})")

        # --- MODUL SCREENER & PAPAN PERINGKAT SAHAM (1.011 EMITEN BEI & KONSTITUEN MSCI) ---
        st.markdown("---")
        st.subheader("🚀 IDX Alpha-Matrix Leaderboard & MSCI Screener (1.011 Emiten BEI)")
        st.write("Memindai daftar emiten berdasarkan filter aktif di Sidebar (termasuk filter **MSCI Global Standard / Small Cap**) untuk meranking saham dengan **Skor Alpha-Matrix tertinggi**, **Likuiditas ATVR**, **Potensi Return AI**, dan deteksi **Volume Spike / Breakout**.")

        scan_limit = st.slider("Jumlah Sampel Emiten yang Dipindai Sekaligus", min_value=10, max_value=150, value=25, step=5)

        if st.button("🔍 Jalankan Pemindaian Papan Peringkat Alpha-Matrix, MSCI & AI"):
            with st.spinner(f"Menjalankan pemindaian Machine Learning & MSCI Liquidity pada {scan_limit} emiten... Mohon tunggu."):
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
                                elif rsi_scr < 32: scr_alpha += 6
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
                                    "Status MSCI": meta_s.get("Status_MSCI", "Non-MSCI"),
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
                    
                    st.success("✅ Pemindaian Papan Peringkat IDX Alpha-Matrix & MSCI Selesai!")
                    st.dataframe(df_final_scr, use_container_width=True)
                else:
                    st.warning("Tidak ada emiten yang memenuhi kriteria filter ketat pada sampel saat ini.")
