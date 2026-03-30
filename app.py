from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)


# CARGA DEL MODELO.
model = joblib.load("src/models/random_forest_optimized.joblib")

# Sustituir por la mediana real calculada en el train
LEAD_TIME_MEDIAN = 80



# FEATURE ENGINEERING
def feature_engineering(df):
    df = df.copy()

    # Agrupación de países
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

    # Transformaciones logarítmicas
    df["adr"] = df["adr"].clip(lower=0)
    df["adr_log"] = np.log1p(df["adr"])
    df["lead_time_log"] = np.log1p(df["lead_time"])
    df["previous_cancellations"] = np.log1p(df["previous_cancellations"])
    df.drop(columns=["adr"], inplace=True)

    # Nuevas variables derivadas
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

    # Limpiar categoría poco informativa
    df["market_segment"] = df["market_segment"].replace("Undefined", "Other")

    return df


# LANDING PAGE
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "mensaje": "API de predicción de cancelaciones hoteleras",
        "modelo": "Random Forest optimizado",
        "endpoint_prediccion": "/predict",
        "metodo": "GET",
        "variables_esperadas": [
            "hotel",
            "customer_type",
            "market_segment",
            "deposit_type",
            "meal",
            "country",
            "distribution_channel",
            "reserved_room_type",
            "is_repeated_guest",
            "lead_time",
            "previous_cancellations",
            "adults",
            "days_in_waiting_list",
            "adr",
            "previous_bookings_not_canceled",
            "booking_changes",
            "required_car_parking_spaces",
            "total_of_special_requests"
        ],
        "ejemplo_uso": "/predict?hotel=Resort%20Hotel&customer_type=Transient&market_segment=Online%20TA&deposit_type=No%20Deposit&meal=BB&country=PRT&distribution_channel=TA/TO&reserved_room_type=A&is_repeated_guest=0&lead_time=120&previous_cancellations=0&adults=2&days_in_waiting_list=0&adr=95.5&previous_bookings_not_canceled=0&booking_changes=1&required_car_parking_spaces=0&total_of_special_requests=1"
    })



# FUNCIÓN AUXILIAR
def build_input_from_args(args):
    return {
        "hotel": args.get("hotel"),
        "customer_type": args.get("customer_type"),
        "market_segment": args.get("market_segment"),
        "deposit_type": args.get("deposit_type"),
        "meal": args.get("meal"),
        "country": args.get("country"),
        "distribution_channel": args.get("distribution_channel"),
        "reserved_room_type": args.get("reserved_room_type"),
        "is_repeated_guest": int(args.get("is_repeated_guest")),
        "lead_time": float(args.get("lead_time")),
        "previous_cancellations": float(args.get("previous_cancellations")),
        "adults": float(args.get("adults")),
        "days_in_waiting_list": float(args.get("days_in_waiting_list")),
        "adr": float(args.get("adr")),
        "previous_bookings_not_canceled": float(args.get("previous_bookings_not_canceled")),
        "booking_changes": float(args.get("booking_changes")),
        "required_car_parking_spaces": float(args.get("required_car_parking_spaces")),
        "total_of_special_requests": float(args.get("total_of_special_requests"))
    }



# ENDPOINT DE PREDICCIÓN
@app.route("/predict", methods=["GET"])
def predict():
    try:
        data = build_input_from_args(request.args)

        df_input = pd.DataFrame([data])
        df_input = feature_engineering(df_input)

        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        return jsonify({
            "model": "Random Forest optimizado",
            "prediction": int(prediction),
            "prediction_label": "Cancelada" if int(prediction) == 1 else "No cancelada",
            "probability_cancelation": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


# ENDPOINT EXTRA PARA REDEPLOY
# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({
#         "status": "ok",
#         "mensaje": "Nuevo endpoint desplegado correctamente"
#     })


if __name__ == "__main__":
    app.run(debug=True)