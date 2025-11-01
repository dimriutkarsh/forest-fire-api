from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

try:
    model = joblib.load("forest_fire_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {e}")

# Define expected feature order (must match training)
FEATURE_ORDER = ["temperature", "humidity", "smoke", "temp_max", "temp_min",
                "pressure", "clouds_all", "wind_speed", "wind_deg", "temp_local", "wind_gust"]

@app.route("/")
def home():
    return jsonify({"message": "🌲 Forest Fire Prediction API is running!"})

@app.route("/predict", methods=["POST", "OPTIONS"])  # Add OPTIONS for CORS preflight
def predict():
    try:
        # Handle preflight request
        if request.method == "OPTIONS":
            return _build_cors_preflight_response()
        
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Check required fields
        missing_fields = [field for field in FEATURE_ORDER if field not in data and field != "wind_gust"]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400
        
        # Prepare features in correct order
        features = []
        for field in FEATURE_ORDER:
            if field == "wind_gust":  # Optional field
                value = data.get(field, 0)
            else:
                value = data[field]
            features.append(float(value))
        
        features_array = np.array([features])
        
        # Scale and predict
        scaled_features = scaler.transform(features_array)
        prediction = int(model.predict(scaled_features)[0])
        probability = float(model.predict_proba(scaled_features)[0, 1])
        
        response = jsonify({
            "fire_risk": prediction,
            "probability": round(probability, 3),
            "message": "🔥 Forest Fire Detected!" if prediction == 1 else "✅ No Fire Detected."
        })
        
        return _corsify_actual_response(response)
        
    except ValueError as e:
        error_response = jsonify({"error": f"Invalid input format: {str(e)}"}), 400
        return _corsify_actual_response(error_response[0]), error_response[1]
    except Exception as e:
        error_response = jsonify({"error": f"Prediction failed: {str(e)}"}), 500
        return _corsify_actual_response(error_response[0]), error_response[1]

def _build_cors_preflight_response():
    """Handle CORS preflight requests"""
    response = jsonify({"status": "preflight"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    return response

def _corsify_actual_response(response):
    """Add CORS headers to actual responses"""
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)