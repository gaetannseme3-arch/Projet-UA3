from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Charger le modèle
model = joblib.load("model.pkl")

# Noms de colonnes par défaut pour creditcard.csv
DEFAULT_FEATURE_NAMES = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Time", "Amount"
]

# Si le modèle connaît déjà les noms des colonnes, on les récupère
if hasattr(model, "feature_names_in_"):
    FEATURE_NAMES = list(model.feature_names_in_)
else:
    FEATURE_NAMES = DEFAULT_FEATURE_NAMES


@app.route("/")
def home():
    return "API fraude active"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Aucune donnée JSON reçue"}), 400

        # Accepte soit "data", soit "features"
        values = data.get("data") or data.get("features")

        if values is None:
            return jsonify({"error": 'La clé "data" ou "features" est requise'}), 400

        if not isinstance(values, list):
            return jsonify({"error": "Les données doivent être une liste"}), 400

        if len(values) != len(FEATURE_NAMES):
            return jsonify({
                "error": f"Il faut exactement {len(FEATURE_NAMES)} valeurs"
            }), 400

        # Essayer d'abord avec DataFrame + noms de colonnes
        try:
            df = pd.DataFrame([values], columns=FEATURE_NAMES)
            prediction = model.predict(df)[0]
        except Exception:
            # Si le modèle attend juste un tableau numpy
            arr = np.array(values).reshape(1, -1)
            prediction = model.predict(arr)[0]

        result = "Fraude" if int(prediction) == 1 else "Non fraude"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)