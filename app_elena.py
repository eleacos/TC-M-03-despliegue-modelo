from flask import Flask, jsonify, request
import os
import joblib
import pandas as pd
import numpy as np

# Garantiza que las rutas relativas funcionen tanto en local como en Render.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)



# Carga del modelo UNA sola vez al arrancar la app.
# El modelo es un sklearn Pipeline completo que incluye:
#   - ColumnTransformer con imputación + escalado / encoding
#   - RandomForestClassifier optimizado
# Se carga con joblib.load() directamente.

MODEL_PATH = os.path.join("src", "models", "random_forest_optimized.joblib")
model = joblib.load(MODEL_PATH)


# Feature engineering - copia EXACTA de la función del notebook main.ipynb.
# El Pipeline espera recibir datos YA transformados, porque el modelo fue entrenado
# con datos transformados pero el pipeline NO incluye este paso.
# Como la API recibe datos sin transformar, hay que reproducir exactamente esas 
# transformaciones.

def feature_engineering(df_in):
    df = df_in.copy()

    europe  = ["PRT", "GBR", "ESP", "FRA", "DEU", "ITA"]
    america = ["USA", "BRA", "CAN"]
    asia    = ["CHN", "IND", "JPN"]

    def group_country(c):
        if c in europe:  return "Europe"
        if c in america: return "America"
        if c in asia:    return "Asia"
        return "Other"

    df["country_grouped"] = df["country"].apply(group_country)
    df = df.drop(columns=["country"])

    df["adr"] = df["adr"].clip(lower=0)
    df["adr_log"] = np.log1p(df["adr"])
    df["lead_time_log"] = np.log1p(df["lead_time"])
    df["previous_cancellations"] = np.log1p(df["previous_cancellations"])
    df = df.drop(columns=["adr"])

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
    df["long_lead"] = (df["lead_time"] > df["lead_time"].median()).astype(int)
    df["market_segment"] = df["market_segment"].replace("Undefined", "Other")

    return df



# Endopoint 0 /
# Enruta la landing page: describe todos los endpoints disponibles y cómo usarlos.
@app.route("/", methods=["GET"])
def home():
    return "Bienvenido a mi API de — Predicción de cancelaciones hoteleras"


"""
IMPORTANTE: Solo lo básico, pendiente de desarrollar
"""


# Enruta la función al endpoint /api/v1/predict


# 