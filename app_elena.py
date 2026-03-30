from flask import Flask, jsonify, request
import os
import joblib
import pandas as pd
import numpy as np

# Garantiza que las rutas relativas funcionen tanto en local como en Render.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)



# Carga del modelo una sola vez al arrancar la app.
# El modelo es un sklearn Pipeline completo que incluye:
#   - ColumnTransformer con imputación + escalado / encoding
#   - RandomForestClassifier optimizado
# Se carga con joblib.load() directamente.

MODEL_PATH = os.path.join("src", "models", "random_forest_optimized.joblib")
model = joblib.load(MODEL_PATH)


# Feature engineering - copia exacta de la función del notebook main.ipynb.
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



# Endopoint 1 /
# Enruta la landing page: describe todos los endpoints disponibles y cómo usarlos.

@app.route("/", methods=["GET"])
def home():
    return """
    <h1>API — Predicción de cancelaciones hoteleras</h1>
    <p>Modelo: <b>Random Forest optimizado</b> &nbsp;|&nbsp;
       Target: <code>is_canceled</code> &nbsp;(0 = no cancela, 1 = cancela)</p>
    <hr>

    <h2>Endpoints disponibles</h2>

    <h3>GET /api/v1/predict</h3>
    <p>Predicción para una reserva individual.
    Los parámetros se pasan como <b>query string</b>.</p>
    <table border="1" cellpadding="5">
      <tr><th>Parámetro</th><th>Tipo</th><th>Descripción / Valores de ejemplo</th></tr>
      <tr><td>hotel</td><td>str</td><td>Resort Hotel | City Hotel</td></tr>
      <tr><td>customer_type</td><td>str</td><td>Transient | Contract | Group | Transient-Party</td></tr>
      <tr><td>market_segment</td><td>str</td><td>Online TA | Offline TA/TO | Direct | Corporate | ...</td></tr>
      <tr><td>deposit_type</td><td>str</td><td>No Deposit | Non Refund | Refundable</td></tr>
      <tr><td>meal</td><td>str</td><td>BB | HB | FB | SC | Undefined</td></tr>
      <tr><td>country</td><td>str</td><td>Código ISO-3166, p.ej. PRT, ESP, GBR</td></tr>
      <tr><td>distribution_channel</td><td>str</td><td>TA/TO | Direct | Corporate | GDS | Undefined</td></tr>
      <tr><td>reserved_room_type</td><td>str</td><td>A | B | C | D | E | F | G | H | L</td></tr>
      <tr><td>is_repeated_guest</td><td>int</td><td>0 = cliente nuevo | 1 = cliente repetido</td></tr>
      <tr><td>lead_time</td><td>int</td><td>Días de antelación de la reserva</td></tr>
      <tr><td>previous_cancellations</td><td>int</td><td>Cancelaciones previas del cliente</td></tr>
      <tr><td>adults</td><td>int</td><td>Número de adultos</td></tr>
      <tr><td>days_in_waiting_list</td><td>int</td><td>Días en lista de espera</td></tr>
      <tr><td>adr</td><td>float</td><td>Precio medio diario (Average Daily Rate)</td></tr>
      <tr><td>previous_bookings_not_canceled</td><td>int</td><td>Reservas previas no canceladas</td></tr>
      <tr><td>booking_changes</td><td>int</td><td>Cambios realizados en la reserva</td></tr>
      <tr><td>required_car_parking_spaces</td><td>int</td><td>Plazas de parking requeridas</td></tr>
      <tr><td>total_of_special_requests</td><td>int</td><td>Número de peticiones especiales</td></tr>
    </table>
    <br>
    <b>Ejemplo:</b><br>
    <code>
    /api/v1/predict?hotel=Resort+Hotel&amp;customer_type=Transient&amp;market_segment=Online+TA
    &amp;deposit_type=No+Deposit&amp;meal=BB&amp;country=PRT&amp;distribution_channel=TA%2FTO
    &amp;reserved_room_type=A&amp;is_repeated_guest=0&amp;lead_time=120
    &amp;previous_cancellations=0&amp;adults=2&amp;days_in_waiting_list=0
    &amp;adr=85&amp;previous_bookings_not_canceled=0&amp;booking_changes=0
    &amp;required_car_parking_spaces=0&amp;total_of_special_requests=1
    </code>

    <h3>GET /api/v1/model_info &nbsp;
        <span style="color:gray">[disponible tras el redespliegue en la exposición]</span>
    </h3>
    <p>Devuelve información del modelo activo: tipo, steps del pipeline, versión.</p>
    """


# Endopoint 0 /api/v1/predict
# Predicción para una reserva.

@app.route("/api/v1/predict", methods=["GET"])
def predict():
    raw = {
        "hotel":request.args.get("hotel",None),
        "customer_type":request.args.get("customer_type",None),
        "market_segment":request.args.get("market_segment",None),
        "deposit_type":request.args.get("deposit_type",None),
        "meal":request.args.get("meal",None),
        "country":request.args.get("country",None),
        "distribution_channel":request.args.get("distribution_channel",None),
        "reserved_room_type":request.args.get("reserved_room_type",None),
        "is_repeated_guest":request.args.get("is_repeated_guest", np.nan, type=float),
        "lead_time":request.args.get("lead_time",  np.nan, type=float),
        "previous_cancellations":request.args.get("previous_cancellations", np.nan, type=float),
        "adults":request.args.get("adults", np.nan, type=float),
        "days_in_waiting_list":request.args.get("days_in_waiting_list", np.nan, type=float),
        "adr":request.args.get("adr", np.nan, type=float),
        "previous_bookings_not_canceled":request.args.get("previous_bookings_not_canceled", np.nan, type=float),
        "booking_changes":request.args.get("booking_changes", np.nan, type=float),
        "required_car_parking_spaces":request.args.get("required_car_parking_spaces", np.nan, type=float),
        "total_of_special_requests":request.args.get("total_of_special_requests", np.nan, type=float),
    }

    missing = [
        k for k, v in raw.items()
        if v is None or (isinstance(v, float) and np.isnan(v))
    ]

    input_df = pd.DataFrame([raw])
    input_fe = feature_engineering(input_df)

    prediction  = int(model.predict(input_fe)[0])
    probability = float(model.predict_proba(input_fe)[0][1])

    response = {
        "prediction":prediction,
        "label":"Cancelada" if prediction == 1 else "No cancelada",
        "probability_canceled":round(probability, 4),
        "probability_not_canceled":round(1 - probability, 4),
    }
    if missing:
        response["warning"] = f"Campos no enviados (imputados como NaN): {', '.join(missing)}"

    return jsonify(response)


# Endpoint 3 RUTA_PTE
# Comentado para el redespliegue en vivo durante la exposición.
# Pasos para activarlo:
#   1. Descomenta las líneas de código
#   2. git add app.py
#   3. git commit -m "add RUTA_PTE endpoint"
#   4. git push origin main
#   5. Render detecta el push y redespliega automáticamente

# @app.route("/api/v1/model_info", methods=["GET"])
# def model_info():
#     return jsonify({
#         "model_type":type(model.named_steps["model"]).__name__,
#         "model_file":MODEL_PATH,
#         "pipeline_steps":list(model.named_steps.keys()),
#         "n_features_in":int(model.named_steps["model"].n_features_in_),
#         "target":"is_canceled (0 = No cancela, 1 = Cancela)",
#         "version":"1.0.0",
#         "description":"Random Forest optimizado — predicción de cancelaciones hoteleras.",
#     })



if __name__ == "__main__":
    app.run(debug=True)