from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained model and scaler
try:
    model = joblib.load("forest_fire_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Model and scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")

@app.route("/")
def home():
    return jsonify({"message": "🌲 Vanrakshak Forest Fire Prediction API is running successfully!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Expected 11 parameters
        features = [
            "temperature", "humidity", "smoke", "temp_max", "temp_min",
            "pressure", "clouds_all", "wind_speed", "wind_deg",
            "wind_gust", "temp_local"
        ]

        # Check all parameters are present
        if not all(f in data for f in features):
            return jsonify({"error": "Missing one or more input parameters."}), 400

        # Extract and convert to array
        input_data = np.array([[data[f] for f in features]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

        response = {
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "message": "🔥 Fire Risk Detected!" if prediction == 1 else "✅ No Fire Risk Detected."
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For Render deployment, use 0.0.0.0 host and port 10000+
    app.run(host="0.0.0.0", port=10000, debug=False)
