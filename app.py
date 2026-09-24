# ============================
# FORCE CPU EXECUTION
# ============================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except:
    pass

# ============================
# IMPORTS
# ============================
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

# ⚠️ OPTIONAL: OpenAI (only if you really need it)

# ============================
# FLASK APP
# ============================
app = Flask(__name__)
CORS(app)

# ============================
# LOAD MODEL & ARTIFACTS
# ============================
model = tf.keras.models.load_model("heart_nn_model.keras")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# ============================
# PREDICTION API
# ============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        patient_data = request.json

        # ---------------- CREATE DATAFRAME ----------------
        df = pd.DataFrame([patient_data])

        # ---------------- ONE-HOT ENCODING ----------------
        df = pd.get_dummies(df)

        # ---------------- ALIGN FEATURES ----------------
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_columns]

        # ---------------- SCALE ----------------
        df_scaled = scaler.transform(df)

        # ---------------- PREDICT PROBABILITY ----------------
        probability = float(model.predict(df_scaled)[0][0])

        # ⚠️ Threshold tuned for recall (medical screening)
        THRESHOLD = 0.35
        prediction = 1 if probability >= THRESHOLD else 0

        pred_label = (
            "Has Heart Disease"
            if prediction == 1
            else "No Heart Disease"
        )

        print(pred_label)


        return jsonify({
            "prediction": pred_label,
            "risk_probability": round(probability * 100, 2),
            "threshold_used": THRESHOLD,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ============================
# RUN SERVER
# ============================
if __name__ == "__main__":
    app.run(debug=True)
