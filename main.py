from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI(
    title="API de predicción de cancelaciones hoteleras",
    description="Modelo Random Forest optimizado",
    version="1.0"
)

# CARGA DEL MODELO
model = joblib.load("src/models/random_forest_optimized.joblib")

LEAD_TIME_MEDIAN = 80

# =========================
# SCHEMA (esto es clave en FastAPI)
# =========================
class BookingInput(BaseModel):
    hotel: str
    customer_type: str
    market_segment: str
    deposit_type: str
    meal: str
    country: str
    distribution_channel: str
    reserved_room_type: str
    is_repeated_guest: int
    lead_time: float
    previous_cancellations: float
    adults: float
    days_in_waiting_list: float
    adr: float
    previous_bookings_not_canceled: float
    booking_changes: float
    required_car_parking_spaces: float
    total_of_special_requests: float


# =========================
# FEATURE ENGINEERING
# =========================
def feature_engineering(df):
    df = df.copy()

    europe = ["PRT", "GBR", "ESP", "FRA", "DEU", "ITA"]
    america = ["USA", "BRA", "CAN"]
    asia = ["CHN", "IND", "JPN"]

    def group_country(c):
        if c in europe:
            return "Europe"
        if c in america:
            return "America"
        if c in asia:
            return "Asia"
        return "Other"

    df["country_grouped"] = df["country"].apply(group_country)
    df.drop(columns=["country"], inplace=True)

    df["adr"] = df["adr"].clip(lower=0)
    df["adr_log"] = np.log1p(df["adr"])
    df["lead_time_log"] = np.log1p(df["lead_time"])
    df["previous_cancellations"] = np.log1p(df["previous_cancellations"])
    df.drop(columns=["adr"], inplace=True)

    df["had_previous_cancel"] = (df["previous_cancellations"] > 0).astype(int)

    df["cancel_ratio"] = (
        df["previous_cancellations"]
        / (df["previous_cancellations"] + df["previous_bookings_not_canceled"] + 1)
    )

    df["loyal_customer"] = (
        (df["is_repeated_guest"] == 1) & (df["previous_cancellations"] == 0)
    ).astype(int)

    df["lead_time_x_adr"] = df["lead_time_log"] * df["adr_log"]
    df["many_special_requests"] = (df["total_of_special_requests"] >= 2).astype(int)
    df["needs_parking"] = (df["required_car_parking_spaces"] > 0).astype(int)
    df["long_lead"] = (df["lead_time"] > LEAD_TIME_MEDIAN).astype(int)

    df["market_segment"] = df["market_segment"].replace("Undefined", "Other")

    return df


# =========================
# ENDPOINTS
# =========================

@app.get("/")
def home():
    return {
        "mensaje": "API de predicción de cancelaciones hoteleras",
        "documentacion": "/docs"
    }


@app.post("/predict")
def predict(data: BookingInput):
    try:
        df_input = pd.DataFrame([data.dict()])
        df_input = feature_engineering(df_input)

        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        return {
            "prediction": int(prediction),
            "prediction_label": "Cancelada" if int(prediction) == 1 else "No cancelada",
            "probability_cancelation": round(float(probability), 4)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}