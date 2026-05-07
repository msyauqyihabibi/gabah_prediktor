"""
Model Prediksi Harga Gabah
Menggunakan Facebook Prophet dengan regresor: produksi_padi, harga_beras,
dan flag musiman (panen_raya, pra_panen, pasca_panen).
"""

import pandas as pd
import numpy as np
from prophet import Prophet


# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESSING DATA
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Membaca datamudah.xlsx, transpose, dan memberi nama kolom.
    Mengembalikan DataFrame dengan index datetime mingguan.
    """
    df = pd.read_excel(filepath, header=None)
    df = df.T
    df.columns = ["harga_gabah", "produksi_padi", "harga_beras"]

    # Buat kolom tanggal bulanan mulai Januari 2019
    df["ds"] = pd.date_range(start="2019-01-01", periods=len(df), freq="MS")
    df = df.set_index("ds")

    # Resample ke mingguan (forward-fill)
    df_weekly = df.resample("W").ffill().reset_index()

    return df_weekly


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Menambahkan fitur flag musiman panen ke DataFrame."""
    month = df["ds"].dt.month
    df = df.copy()
    df["panen_raya"]  = month.isin([3, 4]).astype(int)   # Maret–April
    df["pra_panen"]   = month.isin([1, 2]).astype(int)   # Jan–Feb
    df["pasca_panen"] = month.isin([5, 6]).astype(int)   # Mei–Jun
    return df


# ─────────────────────────────────────────────
# 2. TRAINING MODEL
# ─────────────────────────────────────────────

def build_prophet_df(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """Menyiapkan DataFrame dalam format yang dibutuhkan Prophet."""
    df_weekly = df_weekly.rename(columns={"harga_gabah": "y"})
    df_weekly = add_seasonal_features(df_weekly)

    # Isi nilai kosong produksi_padi
    df_weekly["produksi_padi"] = df_weekly["produksi_padi"].ffill().bfill()

    cols = ["ds", "y", "produksi_padi", "harga_beras",
            "panen_raya", "pra_panen", "pasca_panen"]
    return df_weekly[cols]


def train_model(prophet_df: pd.DataFrame) -> Prophet:
    """Melatih model Prophet dengan semua regresor."""
    model = Prophet()
    model.add_regressor("produksi_padi")
    model.add_regressor("harga_beras")
    model.add_regressor("panen_raya")
    model.add_regressor("pra_panen")
    model.add_regressor("pasca_panen")

    model.fit(prophet_df)
    return model


# ─────────────────────────────────────────────
# 3. PREDIKSI / FORECASTING
# ─────────────────────────────────────────────

def make_forecast(
    model: Prophet,
    prophet_df: pd.DataFrame,
    periods: int = 52,
    produksi_padi_override: float = None,
    harga_beras_override: float = None,
) -> pd.DataFrame:
    """
    Membuat prediksi ke depan.

    Parameters
    ----------
    model               : Model Prophet yang sudah dilatih
    prophet_df          : DataFrame training (untuk nilai default regresor)
    periods             : Jumlah minggu ke depan (default 52 = 1 tahun)
    produksi_padi_override : Nilai produksi padi kustom (opsional)
    harga_beras_override   : Nilai harga beras kustom (opsional)

    Returns
    -------
    DataFrame forecast lengkap dari Prophet
    """
    future = model.make_future_dataframe(periods=periods, freq="W")
    future = add_seasonal_features(future)

    last_produksi = produksi_padi_override or prophet_df["produksi_padi"].iloc[-1]
    last_harga_beras = harga_beras_override or prophet_df["harga_beras"].iloc[-1]

    future["produksi_padi"] = last_produksi
    future["harga_beras"]   = last_harga_beras

    forecast = model.predict(future)
    return forecast


def forecast_to_dict(forecast: pd.DataFrame) -> list[dict]:
    """Mengkonversi hasil forecast ke list of dict (JSON-serializable)."""
    cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
    result = forecast[cols].copy()
    result["ds"] = result["ds"].dt.strftime("%Y-%m-%d")
    return result.to_dict(orient="records")


# ─────────────────────────────────────────────
# 3b. REKOMENDASI WAKTU JUAL
# ─────────────────────────────────────────────

STORAGE_COST_PER_KG_PER_WEEK = 60  # Rp 60/kg/minggu


def recommend_sell_time(
    forecast: pd.DataFrame,
    harvest_date: str,
    capacity_kg: float,
    weeks_ahead: int = 26,
) -> dict:
    """
    Menghitung waktu jual optimal dengan mempertimbangkan:
    - Harga gabah prediksi per minggu
    - Biaya penyimpanan Rp 60/kg/minggu
    - Net revenue = (harga_prediksi * kapasitas) - (biaya_simpan * minggu * kapasitas)

    Parameters
    ----------
    forecast     : DataFrame hasil Prophet (kolom ds, yhat, yhat_lower, yhat_upper)
    harvest_date : Tanggal panen, format "YYYY-MM-DD"
    capacity_kg  : Kapasitas produksi dalam kg
    weeks_ahead  : Berapa minggu ke depan yang dievaluasi (default 26 = 6 bulan)

    Returns
    -------
    dict dengan rekomendasi waktu jual, breakdown per minggu, dan perbandingan
    """
    harvest_dt = pd.to_datetime(harvest_date)

    # Filter forecast mulai dari tanggal panen
    fc = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    fc = fc[fc["ds"] >= harvest_dt].head(weeks_ahead).reset_index(drop=True)

    if fc.empty:
        # Jika harvest_date lebih jauh dari forecast, ambil dari ujung
        fc = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(weeks_ahead).reset_index(drop=True)

    results = []
    for i, row in fc.iterrows():
        weeks_stored = i  # Minggu ke-0 = langsung jual saat panen
        storage_cost_total = STORAGE_COST_PER_KG_PER_WEEK * weeks_stored * capacity_kg
        gross_revenue = row["yhat"] * capacity_kg
        net_revenue = gross_revenue - storage_cost_total
        results.append({
            "week": i,
            "date": row["ds"].strftime("%Y-%m-%d"),
            "predicted_price_per_kg": round(row["yhat"], 2),
            "price_lower": round(row["yhat_lower"], 2),
            "price_upper": round(row["yhat_upper"], 2),
            "weeks_stored": weeks_stored,
            "storage_cost_total": round(storage_cost_total, 2),
            "gross_revenue": round(gross_revenue, 2),
            "net_revenue": round(net_revenue, 2),
        })

    # Cari minggu dengan net_revenue tertinggi
    best = max(results, key=lambda x: x["net_revenue"])

    # Harga saat langsung jual (minggu ke-0)
    immediate = results[0]

    return {
        "harvest_date": harvest_date,
        "capacity_kg": capacity_kg,
        "storage_cost_per_kg_per_week": STORAGE_COST_PER_KG_PER_WEEK,
        "recommendation": {
            "sell_date": best["date"],
            "weeks_to_wait": best["weeks_stored"],
            "predicted_price_per_kg": best["predicted_price_per_kg"],
            "net_revenue": best["net_revenue"],
            "storage_cost_total": best["storage_cost_total"],
            "vs_immediate_sell": round(best["net_revenue"] - immediate["net_revenue"], 2),
        },
        "immediate_sell": {
            "date": immediate["date"],
            "predicted_price_per_kg": immediate["predicted_price_per_kg"],
            "net_revenue": immediate["net_revenue"],
        },
        "weekly_breakdown": results,
    }


# ─────────────────────────────────────────────
# 4. EVALUASI MODEL
# ─────────────────────────────────────────────

def evaluate_model(model: Prophet) -> dict:
    """
    Menjalankan cross-validation dan mengembalikan metrik rata-rata.
    Catatan: Proses ini memakan waktu beberapa menit.
    """
    from prophet.diagnostics import cross_validation, performance_metrics

    df_cv = cross_validation(
        model,
        initial="365 days",
        period="90 days",
        horizon="180 days",
    )
    df_p = performance_metrics(df_cv)
    metrics = df_p[["mae", "rmse", "mape"]].mean().to_dict()
    return metrics


# ─────────────────────────────────────────────
# 5. PIPELINE UTAMA (untuk testing lokal)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "datamudah.xlsx"

    print("📥 Loading data...")
    df_weekly = load_data(DATA_PATH)

    print("🔧 Preprocessing...")
    prophet_df = build_prophet_df(df_weekly)

    print("🤖 Training model...")
    model = train_model(prophet_df)

    print("🔮 Forecasting 52 minggu ke depan...")
    forecast = make_forecast(model, prophet_df, periods=52)

    print("✅ Selesai! Contoh hasil forecast:")
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10).to_string(index=False))
