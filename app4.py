"""
Aplikasi Prediksi Harga Rumah
Model: Orange AdaBoostRegressor (sklearn wrapper)
Dataset: Real Estate (Taiwan)
Deploy: Streamlit Cloud

Strategi load model:
1. Injeksi dummy stubs untuk Orange DAN sklearn ke sys.modules
   sebelum pickle.load() — agar pickle tidak gagal saat menemukan
   referensi modul yang belum ada.
2. Setelah stubs terpasang, import sklearn yang sesungguhnya
   (dari requirements.txt) menimpa stubs — sehingga prediksi
   berjalan dengan sklearn asli.
3. Ekstrak .skl_model dari wrapper Orange untuk prediksi langsung.
"""

import sys
import types
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


# ──────────────────────────────────────────────
# LANGKAH 1: Injeksi dummy stubs untuk sklearn
# Pickle memanggil modul sklearn saat di-load.
# Kita pasang stub dulu, lalu sklearn asli akan menimpa otomatis.
# ──────────────────────────────────────────────
def _inject_sklearn_stubs():
    """
    Pasang modul sklearn dummy ke sys.modules.
    Ini hanya dibutuhkan jika sklearn belum ter-import sama sekali.
    Setelah `import sklearn` nyata berjalan, stub ini akan diganti.
    """
    sklearn_submodules = [
        "sklearn",
        "sklearn.base",
        "sklearn.utils",
        "sklearn.utils.validation",
        "sklearn.utils._bunch",
        "sklearn.ensemble",
        "sklearn.ensemble._weight_boosting",
        "sklearn.ensemble._base",
        "sklearn.tree",
        "sklearn.tree._classes",
        "sklearn.tree._tree",
        "sklearn.metrics",
        "sklearn.preprocessing",
    ]
    for name in sklearn_submodules:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    # Sekarang import sklearn asli — ini akan menimpa stub di atas
    try:
        import sklearn                              # noqa: F401
        import sklearn.ensemble                     # noqa: F401
        import sklearn.ensemble._weight_boosting    # noqa: F401
        import sklearn.tree                         # noqa: F401
        import sklearn.tree._classes                # noqa: F401
        import sklearn.tree._tree                   # noqa: F401
    except ImportError:
        pass  # sklearn belum tersedia; stub tetap terpasang


_inject_sklearn_stubs()


# ──────────────────────────────────────────────
# LANGKAH 2: Injeksi dummy stubs untuk Orange
# ──────────────────────────────────────────────
def _inject_orange_stubs():
    """Buat modul-modul dummy Orange agar pickle.load tidak gagal."""

    def _make_module(name):
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    orange_submodules = [
        "Orange",
        "Orange.base",
        "Orange.data",
        "Orange.data.domain",
        "Orange.data.variable",
        "Orange.data.instance",
        "Orange.data.storage",
        "Orange.data.table",
        "Orange.preprocess",
        "Orange.preprocess.impute",
        "Orange.classification",
        "Orange.regression",
        "Orange.ensembles",
        "Orange.ensembles.ada_boost",
        "Orange.widgets",
        "Orange.widgets.utils",
        "Orange.widgets.utils.colorpalettes",
    ]

    for name in orange_submodules:
        if name not in sys.modules:
            _make_module(name)

    class _DummyBase:
        """Generic stub: terima argumen apapun, abaikan."""
        def __init__(self, *a, **kw): pass
        def __reduce__(self): return (self.__class__, ())
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

    class ContinuousVariable(_DummyBase): pass
    class DiscreteVariable(_DummyBase): pass
    class StringVariable(_DummyBase): pass
    class TimeVariable(_DummyBase): pass
    class Domain(_DummyBase): pass
    class Table(_DummyBase): pass
    class Instance(_DummyBase): pass
    class ReplaceUnknowns(_DummyBase): pass
    class ContinuousPalette(_DummyBase): pass

    def make_variable(*a, **kw):
        return ContinuousVariable()

    odata = sys.modules["Orange.data"]
    odata.ContinuousVariable = ContinuousVariable
    odata.DiscreteVariable = DiscreteVariable
    odata.StringVariable = StringVariable
    odata.TimeVariable = TimeVariable
    odata.Domain = Domain
    odata.Table = Table
    odata.Instance = Instance
    odata.make_variable = make_variable

    sys.modules["Orange.data.variable"].ContinuousVariable = ContinuousVariable
    sys.modules["Orange.data.variable"].DiscreteVariable = DiscreteVariable
    sys.modules["Orange.data.variable"].TimeVariable = TimeVariable
    sys.modules["Orange.data.variable"].make_variable = make_variable

    sys.modules["Orange.preprocess.impute"].ReplaceUnknowns = ReplaceUnknowns
    sys.modules["Orange.widgets.utils.colorpalettes"].ContinuousPalette = ContinuousPalette

    class SklAdaBoostRegressor(_DummyBase):
        skl_model = None
        domain = None
        original_domain = None
        supports_multiclass = False

        def predict(self, X):
            if self.skl_model is not None:
                return self.skl_model.predict(X)
            raise RuntimeError("skl_model belum diset")

        def __call__(self, data):
            return self.predict(data)

    sys.modules["Orange.ensembles.ada_boost"].SklAdaBoostRegressor = SklAdaBoostRegressor

    orange_root = sys.modules["Orange"]
    orange_root.data = odata


_inject_orange_stubs()

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
            orange_wrapper = pickle.load(f)

        # Orange menyimpan sklearn model di atribut .skl_model
        # Ekstrak langsung agar kita bisa pakai sklearn.predict tanpa orange3
        skl = getattr(orange_wrapper, "skl_model", None)
        if skl is not None:
            return skl, None

        # Fallback: mungkin model sudah sklearn langsung
        if hasattr(orange_wrapper, "predict"):
            return orange_wrapper, None

        return None, (
            "Model berhasil dibaca tapi tidak ditemukan atribut "
            "`skl_model` maupun method `predict`. "
            "Struktur model tidak dikenali."
        )
    except Exception as e:
        return None, f"Gagal memuat model: {e}"


# ──────────────────────────────────────────────
# Fungsi prediksi — langsung sklearn
# ──────────────────────────────────────────────
def run_prediction(model, input_df: pd.DataFrame):
    """
    Jalankan prediksi menggunakan sklearn model yang sudah diekstrak.
    Kembalikan (nilai_prediksi, pesan_error).
    """
    try:
        # Pastikan urutan kolom dan tipe data benar
        X = input_df[list(FEATURE_CONFIG.keys())].astype(float).values
        preds = model.predict(X)
        return float(preds[0]), None
    except Exception as e:
        return None, f"Prediksi gagal: {e}"


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
        prediction, pred_error = run_prediction(model, input_df)

    if pred_error:
        st.error(f"❌ {pred_error}")
    else:
        st.success(
            f"### 🏷️ Prediksi Harga: **{prediction:,.2f}** *(10.000 TWD per ping)*"
        )

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
