import requests

url = "http://127.0.0.1:10000/predict"

data = {
  "temperature": 90.5,
  "humidity": 40.2,
  "smoke": 650.3,
  "temp_max": 95.0,
  "temp_min": 85.0,
  "pressure": 1010.2,
  "clouds_all": 0.5,
  "wind_speed": 3.2,
  "wind_deg": 250.0,
  "wind_gust": 5.4,
  "temp_local": 30.5
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response:", response.json())
