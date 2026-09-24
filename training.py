
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# ✅ HARD-disable GPU (important)
try:
    tf.config.set_visible_devices([], "GPU")
except:
    pass

print("Devices:", tf.config.list_physical_devices())

# ============================
# IMPORTS
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ============================
# LOAD DATA
# ============================
data = pd.read_csv("data.csv")

data["HeartDisease"] = data["HeartDisease"].map({
    "No": 0,
    "Yes": 1
})

print("\n" + "=" * 60)
print("DATA EXPLORATION")
print("=" * 60)

target_counts = data["HeartDisease"].value_counts()
target_percent = target_counts / len(data) * 100

for val, count in target_counts.items():
    print(f"Class {val}: {count:,} ({target_percent[val]:.2f}%)")

print(f"\nClass Imbalance Ratio: {target_counts[1] / target_counts[0]:.3f}")

# ============================
# FEATURES & ENCODING
# ============================
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

X = pd.get_dummies(X)

print(f"\nTotal features after encoding: {X.shape[1]}")
print(f"Total samples: {X.shape[0]}")

# ============================
# TRAIN TEST SPLIT
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ============================
# SCALING
# ============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================
# NEURAL NETWORK MODEL
# ============================
print("\nMODEL TRAINING (CPU MODE)")
print("=" * 60)

input_dim = X_train_scaled.shape[1]

model = Sequential([
    Dense(128, activation="relu", input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

early_stop = EarlyStopping(
    monitor="val_auc",
    patience=10,
    mode="max",
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1,
    class_weight={
        0: 1,
        1: target_counts[0] / target_counts[1]
    }
)

print(f"\nBest Validation AUC: {max(history.history['val_auc']):.4f}")

# ============================
# PREDICTIONS
# ============================
y_pred_proba = model.predict(X_test_scaled).ravel()
y_pred = (y_pred_proba >= 0.5).astype(int)

# ============================
# METRICS
# ============================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
average_precision = average_precision_score(y_test, y_pred_proba)

print("\nMODEL PERFORMANCE")
print("=" * 60)
print(f"Accuracy:           {accuracy:.4f}")
print(f"Precision:          {precision:.4f}")
print(f"Recall:             {recall:.4f}")
print(f"F1-Score:           {f1:.4f}")
print(f"ROC-AUC:            {roc_auc:.4f}")
print(f"Average Precision:  {average_precision:.4f}")

# ============================
# CONFUSION MATRIX
# ============================
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)
npv = tn / (tn + fn)
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

print("\nCONFUSION MATRIX")
print("=" * 60)
print(cm)

# ============================
# VISUALIZATIONS
# ============================
def generate_visualizations():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
    axes[0, 0].set_title("Confusion Matrix")

    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_pred_proba)
    axes[0, 1].plot(fpr_curve, tpr_curve, label=f"AUC={roc_auc:.3f}")
    axes[0, 1].plot([0, 1], [0, 1], "--")
    axes[0, 1].legend()
    axes[0, 1].set_title("ROC Curve")

    p, r, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[1, 0].plot(r, p)
    axes[1, 0].set_title("Precision-Recall Curve")

    axes[1, 1].bar(
        ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        [accuracy, precision, recall, f1, roc_auc]
    )
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Performance Metrics")

    plt.tight_layout()
    plt.savefig("model_evaluation.png", dpi=300)
    plt.close()

generate_visualizations()

# ============================
# SAVE ARTIFACTS
# ============================
model.save("heart_nn_model.keras")  # ✅ modern format
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

joblib.dump({
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "average_precision": average_precision,
    "specificity": specificity,
    "npv": npv,
    "fpr": fpr,
    "fnr": fnr,
    "confusion_matrix": cm.tolist()
}, "model_metrics.pkl")

print("\n" + "=" * 60)
print("✅ NEURAL NETWORK TRAINING COMPLETED (CPU)")
print("=" * 60)
