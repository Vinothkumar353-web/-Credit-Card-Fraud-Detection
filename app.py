from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


# ✅ Load model package ONCE
package = joblib.load("fraud_detection_model.joblib")

model = package["model"]
scaler = package["scaler"]
le = package["label_encoder"]

@app.route("/")
def home():
    return "Fraud Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Invalid input"}), 400

        features = np.array(data["features"]).reshape(1, -1)

        # ✅ Scale before prediction
        scaled_features = scaler.transform(features)

        prediction = model.predict(scaled_features)

        # If needed
        decoded_prediction = le.inverse_transform(prediction)

        return jsonify({
            "prediction": decoded_prediction[0]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    
