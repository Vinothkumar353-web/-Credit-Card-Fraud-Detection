@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Invalid input"}), 400

        if "features" not in data:
            return jsonify({"error": "Missing 'features' key"}), 400

        features = np.array(data["features"]).reshape(1, -1)

        # Scale before prediction
        scaled_features = scaler.transform(features)

        prediction = model.predict(scaled_features)

        decoded_prediction = le.inverse_transform(prediction)

        return jsonify({
            "prediction": str(decoded_prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
