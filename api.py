"""
API Backend — Prediksi Harga Gabah
Jalankan: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

from model import (
    load_data, build_prophet_df, train_model,
    make_forecast, forecast_to_dict, recommend_sell_time
)

app = FastAPI(title="API Prediksi Harga Gabah", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.getenv("DATA_PATH", "datamudah.xlsx")

print("🚀 Loading data dan training model...")
df_weekly   = load_data(DATA_PATH)
prophet_df  = build_prophet_df(df_weekly)
model       = train_model(prophet_df)
print("✅ Model siap!")


class PredictRequest(BaseModel):
    periods: int = 52
    produksi_padi: Optional[float] = None
    harga_beras: Optional[float] = None


class RecommendRequest(BaseModel):
    harvest_date: str
    capacity_kg: float
    weeks_ahead: int = 26


@app.get("/")
def root():
    return {"status": "ok", "message": "API Prediksi Harga Gabah aktif"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        forecast = make_forecast(
            model, prophet_df, periods=req.periods,
            produksi_padi_override=req.produksi_padi,
            harga_beras_override=req.harga_beras,
        )
        result = forecast_to_dict(forecast)
        future_only = result[-req.periods:]
        return {"status": "success", "periods": req.periods, "forecast": future_only}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    Rekomendasi waktu jual optimal.
    - harvest_date : tanggal panen (YYYY-MM-DD)
    - capacity_kg  : kapasitas produksi dalam kg
    - weeks_ahead  : evaluasi berapa minggu ke depan (default 26)
    """
    try:
        periods_needed = req.weeks_ahead + 8
        forecast = make_forecast(model, prophet_df, periods=periods_needed)
        result = recommend_sell_time(
            forecast=forecast,
            harvest_date=req.harvest_date,
            capacity_kg=req.capacity_kg,
            weeks_ahead=req.weeks_ahead,
        )
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/history")
def get_history():
    df = prophet_df.copy()
    df["ds"] = df["ds"].dt.strftime("%Y-%m-%d")
    return {
        "status": "success",
        "history": df[["ds", "y", "produksi_padi", "harga_beras"]].to_dict(orient="records"),
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": "Prophet", "data_rows": len(prophet_df)}
