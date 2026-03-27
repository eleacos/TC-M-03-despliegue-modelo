import requests

url = "http://127.0.0.1:5000/predict"

params = {
    "hotel": "Resort Hotel",
    "customer_type": "Transient",
    "market_segment": "Online TA",
    "deposit_type": "No Deposit",
    "meal": "BB",
    "country": "PRT",
    "distribution_channel": "TA/TO",
    "reserved_room_type": "A",
    "is_repeated_guest": 0,
    "lead_time": 120,
    "previous_cancellations": 0,
    "adults": 2,
    "days_in_waiting_list": 0,
    "adr": 95,
    "previous_bookings_not_canceled": 0,
    "booking_changes": 1,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 1
}

response = requests.get(url, params=params)

print("Status code:", response.status_code)
print("Respuesta JSON:", response.json())