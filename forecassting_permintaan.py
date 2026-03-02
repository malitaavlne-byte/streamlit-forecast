import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Forecasting Produksi Harian Otomatis (Multivariat SARIMAX)")

# =============================
# LOAD MODEL
# =============================

with open("model_po.sav", "rb") as f:
    model_po = pickle.load(f)

with open("model_produksi.sav", "rb") as f:
    model_prod = pickle.load(f)

# =============================
# LOAD DATA TERBARU
# =============================

df = pd.read_excel("data_forecast.xlsx")
df.columns = df.columns.str.strip().str.lower()
df['tanggal'] = pd.to_datetime(df['tanggal'])
df = df.sort_values('tanggal')
df.set_index('tanggal', inplace=True)

# =============================
# AMBIL NILAI TERAKHIR OPERASIONAL
# =============================

last_plan = df['plan'].iloc[-1]
last_kapasitas = df['kapasitas'].iloc[-1]
last_bahan = df['bahan'].iloc[-1]
last_pekerja = df['pekerja'].iloc[-1]
last_jam = df['jam'].iloc[-1]

# =============================
# INPUT USER
# =============================

days = st.slider("Jumlah hari yang diprediksi", 1, 14, 7)

if st.button("Prediksi"):

    # =============================
    # STEP 1: FORECAST PO
    # =============================

    fc_po = model_po.get_forecast(steps=days)
    po_pred = fc_po.predicted_mean

    # =============================
    # STEP 2: BUAT FUTURE EXOG PRODUKSI
    # =============================

    future_exog = pd.DataFrame({
        'po': po_pred.values,
        'plan': [last_plan]*days,
        'kapasitas': [last_kapasitas]*days,
        'bahan': [last_bahan]*days,
        'pekerja': [last_pekerja]*days,
        'jam': [last_jam]*days
    })

    # =============================
    # STEP 3: FORECAST PRODUKSI
    # =============================

    fc_prod = model_prod.get_forecast(
        steps=days,
        exog=future_exog
    )

    prod_pred = fc_prod.predicted_mean
    conf_int = fc_prod.conf_int()

    lower = conf_int.iloc[:, 0]
    upper = conf_int.iloc[:, 1]

    # =============================
    # TABEL HASIL
    # =============================

    result_df = pd.DataFrame({
        "Hari ke-": range(1, days+1),
        "Forecast PO": po_pred.values,
        "Forecast Produksi": prod_pred.values,
        "Lower Bound": lower.values,
        "Upper Bound": upper.values
    })

    st.subheader("Hasil Forecast")
    st.dataframe(result_df)

    # =============================
    # GRAFIK HISTORIS + FORECAST
    # =============================

    history = df['hasil'].iloc[-30:]

    future_index = pd.date_range(
        start=history.index[-1] + pd.Timedelta(days=1),
        periods=days,
        freq='D'
    )

    fig, ax = plt.subplots(figsize=(10,5))

    # Historis
    ax.plot(history.index, history.values, label="Historis")

    # Forecast Produksi
    ax.plot(future_index, prod_pred.values, label="Forecast Produksi")

    # Confidence Interval
    ax.fill_between(
        future_index,
        lower.values,
        upper.values,
        alpha=0.3,
        label="Confidence Interval"
    )

    ax.set_title("Forecast Produksi 1–14 Hari ke Depan")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Produksi")
    ax.legend()

    st.pyplot(fig)

    st.success(f"Forecast otomatis {days} hari berhasil dibuat.")