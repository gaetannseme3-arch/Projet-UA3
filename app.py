from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import traceback

app = Flask(__name__)

model = None

def load_model():
    global model
    if model is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "model_fraude.pkl")
        model = joblib.load(model_path)
    return model

@app.route("/")
def home():
    return "API fraude active"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "Aucune donnée JSON reçue"}), 400

        expected_columns = [
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest"
        ]

        for col in expected_columns:
            if col not in data:
                data[col] = 0

        df = pd.DataFrame([data])
        df = df[expected_columns]

        modele = load_model()
        prediction = modele.predict(df)

        result = prediction[0]
        if hasattr(result, "item"):
            result = result.item()

        return jsonify({
            "prediction": int(result)
        })

    except Exception as e:
        return jsonify({
            "error": repr(e),
            "trace": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)