import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "temperature": 120,
    "humidity": 10,
    "smoke": 2500,
    "temp_max": 130,
    "temp_min": 110,
    "pressure": 950,
    "clouds_all": 0.1,
    "wind_speed": 5,
    "wind_deg": 180,
    "temp_local": 115
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response:", response.json())
