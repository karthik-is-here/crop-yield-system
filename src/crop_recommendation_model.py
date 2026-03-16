import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import preprocess_crop_data

MODELS_DIR = "models/"

def train_random_forest():
    print("\n" + "=" * 50)
    print("CROP RECOMMENDATION — RANDOM FOREST")
    print("=" * 50)

    X_train, X_test, y_train, y_test, label_encoder, scaler = (
        preprocess_crop_data()
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)
    print("Training complete.")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_
    ))

    feature_names = ["N", "P", "K", "temperature",
                     "humidity", "ph", "rainfall"]
    importances = model.feature_importances_
    print("\nFeature Importances:")
    for name, score in sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    ):
        bar = "█" * int(score * 50)
        print(f"  {name:<12} {score:.4f}  {bar}")

    joblib.dump(model, MODELS_DIR + "crop_rf_model.pkl")
    print(f"\nModel saved to {MODELS_DIR}crop_rf_model.pkl")

    return model, label_encoder


if __name__ == "__main__":
    train_random_forest()