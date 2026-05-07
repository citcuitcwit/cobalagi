"""
Aplikasi Prediksi Harga Rumah
Model: Orange AdaBoost Regressor
Dataset: Real Estate (Taiwan)
Deploy: Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ──────────────────────────────────────────────
# Konfigurasi halaman
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Rumah",
    page_icon="🏠",
    layout="centered",
)

# ──────────────────────────────────────────────
# Path model — relatif terhadap app.py
# JANGAN gunakan path absolut seperti C:/Users/... atau /content/...
# ──────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "adaboost_rumah.pkcls"

# ──────────────────────────────────────────────
# Konfigurasi fitur
# Nama key HARUS sama persis dengan nama variabel saat training di Orange
# ──────────────────────────────────────────────
FEATURE_CONFIG = {
    "X1 transaction date": {
        "type": "numeric",
        "input": "number",
        "label": "Tanggal Transaksi (tahun desimal, mis. 2013.5)",
        "min": 2012.0,
        "max": 2015.0,
        "default": 2013.5,
        "step": 0.083,        # ~1 bulan
        "help": "Format tahun desimal, mis. 2013.25 ≈ April 2013",
    },
    "X2 house age": {
        "type": "numeric",
        "input": "slider",
        "label": "Usia Rumah (tahun)",
        "min": 0,
        "max": 50,
        "default": 10,
        "step": 1,
        "help": "Umur bangunan dalam tahun",
    },
    "X3 distance to the nearest MRT station": {
        "type": "numeric",
        "input": "number",
        "label": "Jarak ke Stasiun MRT Terdekat (meter)",
        "min": 0.0,
        "max": 7000.0,
        "default": 500.0,
        "step": 10.0,
        "help": "Jarak dalam meter ke stasiun MRT terdekat",
    },
    "X4 number of convenience stores": {
        "type": "numeric",
        "input": "slider",
        "label": "Jumlah Minimarket di Sekitar",
        "min": 0,
        "max": 15,
        "default": 5,
        "step": 1,
        "help": "Jumlah convenience store / minimarket dalam radius tertentu",
    },
    "X5 latitude": {
        "type": "numeric",
        "input": "number",
        "label": "Latitude (lintang)",
        "min": 24.9,
        "max": 25.1,
        "default": 24.97,
        "step": 0.0001,
        "help": "Koordinat latitude properti (kawasan Taiwan ~24.9–25.1)",
    },
    "X6 longitude": {
        "type": "numeric",
        "input": "number",
        "label": "Longitude (bujur)",
        "min": 121.4,
        "max": 121.7,
        "default": 121.54,
        "step": 0.0001,
        "help": "Koordinat longitude properti (kawasan Taiwan ~121.4–121.7)",
    },
}


# ──────────────────────────────────────────────
# Load model dengan cache agar tidak reload tiap interaksi
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model…")
def load_model():
    if not MODEL_PATH.exists():
        return None, (
            f"File model tidak ditemukan: `{MODEL_PATH.name}`\n\n"
            "Pastikan file `adaboost_rumah.pkcls` ada di repository GitHub "
            "yang sama dengan `app.py`."
        )
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, f"Gagal memuat model: {e}"


# ──────────────────────────────────────────────
# Fungsi prediksi dengan dua pendekatan
# ──────────────────────────────────────────────
def predict_sklearn_like(model, input_df: pd.DataFrame):
    """Coba prediksi menggunakan antarmuka scikit-learn (model.predict)."""
    return model.predict(input_df)


def predict_orange_fallback(model, input_df: pd.DataFrame):
    """
    Fallback: konversi DataFrame ke Orange.data.Table lalu panggil model().
    Digunakan bila model.predict() gagal.
    """
    try:
        import Orange.data as odata
    except ImportError:
        raise ImportError(
            "Library `orange3` tidak tersedia. "
            "Tambahkan `orange3` ke requirements.txt."
        )

    attrs = []
    values_list = []

    for col in input_df.columns:
        cfg = FEATURE_CONFIG.get(col, {})
        if cfg.get("type") == "categorical":
            vals = cfg.get("options", [])
            attrs.append(odata.DiscreteVariable(col, values=vals))
        else:
            attrs.append(odata.ContinuousVariable(col))
        values_list.append(input_df[col].values[0])

    domain = odata.Domain(attrs)
    X = np.array([values_list], dtype=float)
    table = odata.Table.from_numpy(domain, X)
    result = model(table)
    return result


def run_prediction(model, input_df: pd.DataFrame):
    """
    Jalankan prediksi.
    Coba scikit-learn style dulu; bila gagal, pakai Orange fallback.
    Kembalikan (nilai_prediksi, metode_yang_dipakai, pesan_error).
    """
    # --- Pendekatan 1: scikit-learn-like ---
    try:
        preds = predict_sklearn_like(model, input_df)
        return float(preds[0]), "scikit-learn interface", None
    except Exception as e1:
        sk_err = str(e1)

    # --- Pendekatan 2: Orange native ---
    try:
        preds = predict_orange_fallback(model, input_df)
        val = float(preds[0]) if hasattr(preds, "__len__") else float(preds)
        return val, "Orange native interface", None
    except Exception as e2:
        return None, None, (
            f"Prediksi gagal dengan kedua metode.\n\n"
            f"• scikit-learn: {sk_err}\n"
            f"• Orange fallback: {e2}"
        )


# ──────────────────────────────────────────────
# UI — Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("📖 Petunjuk Penggunaan")
    st.markdown(
        """
        1. Isi semua nilai pada form di halaman utama.
        2. Klik tombol **Prediksi**.
        3. Hasil prediksi harga rumah per satuan luas akan muncul di bawah form.

        ---
        **Tentang Model**
        - Algoritma: AdaBoost Regressor
        - Training tool: Orange Data Mining
        - File model: `adaboost_rumah.pkcls`  
          *(disimpan di GitHub repository yang sama)*

        ---
        **Dataset**
        - Real Estate Valuation — Taiwan
        - Target: Harga per satuan luas (10.000 TWD / ping)
        """
    )

# ──────────────────────────────────────────────
# UI — Header utama
# ──────────────────────────────────────────────
st.title("🏠 Prediksi Harga Rumah")
st.markdown(
    "Aplikasi ini menggunakan model machine learning hasil training dari "
    "**Orange Data Mining** dan dijalankan melalui **Streamlit Cloud**."
)
st.divider()

# ──────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────
model, load_error = load_model()

if load_error:
    st.error(f"⚠️ {load_error}")
    st.stop()
else:
    st.success(f"✅ Model berhasil dimuat — `{MODEL_PATH.name}`")

st.divider()

# ──────────────────────────────────────────────
# Form input
# ──────────────────────────────────────────────
st.subheader("📋 Data Properti")

with st.form("prediction_form"):
    input_data = {}

    col1, col2 = st.columns(2)
    feature_keys = list(FEATURE_CONFIG.keys())

    for i, feature_name in enumerate(feature_keys):
        cfg = FEATURE_CONFIG[feature_name]
        target_col = col1 if i % 2 == 0 else col2

        with target_col:
            if cfg["type"] == "categorical":
                input_data[feature_name] = st.selectbox(
                    label=cfg.get("label", feature_name),
                    options=cfg["options"],
                    help=cfg.get("help", ""),
                )
            elif cfg["input"] == "slider":
                input_data[feature_name] = st.slider(
                    label=cfg.get("label", feature_name),
                    min_value=cfg["min"],
                    max_value=cfg["max"],
                    value=cfg["default"],
                    step=cfg.get("step", 1),
                    help=cfg.get("help", ""),
                )
            else:  # number_input
                input_data[feature_name] = st.number_input(
                    label=cfg.get("label", feature_name),
                    min_value=float(cfg["min"]),
                    max_value=float(cfg["max"]),
                    value=float(cfg["default"]),
                    step=float(cfg.get("step", 1.0)),
                    help=cfg.get("help", ""),
                )

    submitted = st.form_submit_button("🔍 Prediksi", use_container_width=True)

# ──────────────────────────────────────────────
# Proses prediksi saat tombol ditekan
# ──────────────────────────────────────────────
if submitted:
    st.divider()
    st.subheader("📊 Hasil Prediksi")

    # Buat DataFrame dengan urutan kolom sesuai FEATURE_CONFIG
    input_df = pd.DataFrame([input_data], columns=list(FEATURE_CONFIG.keys()))

    # Tampilkan ringkasan input
    with st.expander("🗂️ Data Input yang Digunakan", expanded=True):
        display_df = pd.DataFrame(
            {
                "Fitur": [cfg.get("label", k) for k, cfg in FEATURE_CONFIG.items()],
                "Nilai": [input_data[k] for k in FEATURE_CONFIG],
            }
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Jalankan prediksi
    with st.spinner("Menghitung prediksi…"):
        prediction, method, pred_error = run_prediction(model, input_df)

    if pred_error:
        st.error(f"❌ {pred_error}")
    else:
        st.success(
            f"### 🏷️ Prediksi Harga: **{prediction:,.2f}** *(10.000 TWD per ping)*"
        )
        st.caption(f"Metode inferensi: {method}")

        # Estimasi kasar dalam USD (opsional, informatif)
        price_twd = prediction * 10_000
        price_usd_approx = price_twd / 32  # kurs kasar
        st.info(
            f"Perkiraan: **{price_twd:,.0f} TWD** per ping  "
            f"≈ **USD {price_usd_approx:,.0f}** per ping *(kurs estimasi)*"
        )

    st.divider()
    st.caption(
        "⚠️ Prediksi bersifat estimasi berdasarkan data historis Taiwan. "
        "Bukan acuan nilai properti aktual."
    )
