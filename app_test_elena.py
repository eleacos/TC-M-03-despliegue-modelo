from flask import Flask, jsonify, request
import os
import joblib
import pandas as pd
import numpy as np
 
# Asegura que las rutas relativas funcionen tanto en local como en Render
os.chdir(os.path.dirname(os.path.abspath(__file__)))
 
app = Flask(__name__)
 
# Carga del modelo Random Forest (una sola vez al arrancar la app, no dentro de cada endpoint)
MODEL_PATH = os.path.join("src", "models", "random_forest_optimized.joblib")
model = joblib.load(MODEL_PATH)
 

# Features que espera el modelo
FEATURE_NAMES = ['hotel' 'customer_type' 'market_segment' 'deposit_type' 'meal'
 'distribution_channel' 'reserved_room_type' 'is_repeated_guest'
 'lead_time' 'previous_cancellations' 'adults' 'days_in_waiting_list'
 'previous_bookings_not_canceled' 'booking_changes'
 'required_car_parking_spaces' 'total_of_special_requests'
 'country_grouped' 'adr_log' 'lead_time_log' 'had_previous_cancel'
 'cancel_ratio' 'loyal_customer' 'lead_time_x_adr' 'many_special_requests'
 'needs_parking' 'long_lead']
 

# Endpoint 0 — Landing page  /
# Informa al consumidor de todos los endpoints disponibles

@app.route("/", methods=["GET"])
def home():
    return "API de Hotel Booking - Predicción de Cancelaciones de Reservas</h1>"
 
# Endpoint 1 — Predicción individual  /api/v1/predict
 

# Endpoint 2 — Predicción en batch  /api/v1/predict_batch


# Endpoint 3 — Info del modelo  /api/v1/model_info   
# 
# 
if __name__ == "__main__":
    app.run(debug=True)