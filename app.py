from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model package
package = joblib.load("fraud_detection_model.joblib")

model = package["model"]
scaler = package["scaler"]
le = package["label_encoder"]


# -----------------------------
# HOME PAGE (HTML FORM)
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)


@app.route("/predict-form", methods=["POST"])
def predict_form():
    try:
        # Get 13 inputs from form
        features = [float(request.form[f"feature{i}"]) for i in range(1, 14)]

        features_array = np.array(features).reshape(1, -1)

        # Scale features
        scaled_features = scaler.transform(features_array)

        # Predict
        prediction = model.predict(scaled_features)
        decoded_prediction = int(le.inverse_transform(prediction)[0])

        # Convert result to readable message
        if decoded_prediction == 1:
            result = "⚠️ Fraud Transaction"
        else:
            result = "✅ Legitimate Transaction"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")


# -----------------------------
# JSON API (for curl / Postman)
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "features" not in data:
            return jsonify({"error": "Invalid input"}), 400

        features = np.array(data["features"]).reshape(1, -1)

        scaled_features = scaler.transform(features)

        prediction = model.predict(scaled_features)
        decoded_prediction = int(le.inverse_transform(prediction)[0])

        if decoded_prediction == 1:
            result = "Fraud Transaction"
        else:
            result = "Legitimate Transaction"

        return jsonify({
            "prediction": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
