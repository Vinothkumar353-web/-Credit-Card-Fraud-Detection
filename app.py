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
        # Get 13 inputs from form (feature1 to feature13)
        features = [
            float(request.form[f"feature{i}"]) for i in range(1, 14)
        ]

        features_array = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features_array)

        prediction = model.predict(scaled_features)
        decoded_prediction = le.inverse_transform(prediction)

        return render_template(
            "index.html",
            prediction=str(decoded_prediction[0])
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}"
        )


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
        decoded_prediction = le.inverse_transform(prediction)

        return jsonify({
            "prediction": str(decoded_prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
