import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# ── Constants ──────────────────────────────────────────────
CROP_CSV   = "data/crop_recommendation.csv"
YIELD_CSV  = "data/yield_df.csv"
MODELS_DIR = "models/"

# ── Create models directory if it doesn't exist ────────────
os.makedirs(MODELS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════
#  PART 1 — CROP RECOMMENDATION DATA
# ══════════════════════════════════════════════════════════

def preprocess_crop_data():
    df = pd.read_csv(CROP_CSV)

    X = df.drop("label", axis=1)
    y = df["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    joblib.dump(scaler,        MODELS_DIR + "crop_scaler.pkl")
    joblib.dump(label_encoder, MODELS_DIR + "crop_label_encoder.pkl")

    print("── Crop Recommendation Data ──")
    print(f"Training samples : {X_train.shape[0]}")
    print(f"Test samples     : {X_test.shape[0]}")
    print(f"Features         : {X_train.shape[1]}")
    print(f"Crop classes     : {list(label_encoder.classes_)}")

    return X_train, X_test, y_train, y_test, label_encoder, scaler


# ══════════════════════════════════════════════════════════
#  PART 2 — YIELD PREDICTION DATA
# ══════════════════════════════════════════════════════════

def preprocess_yield_data():
    df = pd.read_csv(YIELD_CSV)

    df = df.drop_duplicates()
    df = df.dropna()

    area_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df["Area_encoded"] = area_encoder.fit_transform(df["Area"])
    df["Item_encoded"] = item_encoder.fit_transform(df["Item"])

    feature_cols = [
        "Area_encoded",
        "Item_encoded",
        "Year",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp"
    ]

    X = df[feature_cols]
    y = df["hg/ha_yield"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    joblib.dump(scaler,        MODELS_DIR + "yield_scaler.pkl")
    joblib.dump(area_encoder,  MODELS_DIR + "yield_area_encoder.pkl")
    joblib.dump(item_encoder,  MODELS_DIR + "yield_item_encoder.pkl")

    print("\n── Yield Prediction Data ──")
    print(f"Training samples : {X_train.shape[0]}")
    print(f"Test samples     : {X_test.shape[0]}")
    print(f"Features         : {X_train.shape[1]}")
    print(f"Yield range      : {y.min():.0f} – {y.max():.0f} hg/ha")

    return X_train, X_test, y_train, y_test, item_encoder, scaler


# ══════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    preprocess_crop_data()
    preprocess_yield_data()