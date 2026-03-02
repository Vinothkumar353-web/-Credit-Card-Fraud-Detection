from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model package
package = joblib.load("fraud_detection_model.joblib")

model = package["model"]
scaler = package["scaler"]
le = package["label_encoder"]


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get 13 inputs from form
            features = [
                float(request.form[f"feature{i}"]) for i in range(1, 14)
            ]

            features_array = np.array(features).reshape(1, -1)
            scaled_features = scaler.transform(features_array)
            prediction = model.predict(scaled_features)
            decoded_prediction = le.inverse_transform(prediction)

            return render_template("index.html",
                                   prediction=decoded_prediction[0])

        except Exception as e:
            return render_template("index.html",
                                   prediction=f"Error: {str(e)}")

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        decoded_prediction = le.inverse_transform(prediction)

        return jsonify({"prediction": str(decoded_prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
