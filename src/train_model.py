import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import uuid
from datetime import datetime

def train_and_save_model(data_path, save_dir="models"):
    # Load dataset
    df = pd.read_excel(data_path, header=1)

    # Clean and prepare data
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop("ID", axis=1, inplace=True)

    # Features and target
    X = df.drop("default", axis=1)
    y = df["default"]

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Save model and scaler
    os.makedirs(save_dir, exist_ok=True)
    model_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = os.path.join(save_dir, f"credit_model_{model_id}_{timestamp}.pkl")
    scaler_path = os.path.join(save_dir, f"scaler_{model_id}_{timestamp}.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # # Save model with unique filename
    # if not os.path.exists("models"):
    #     os.makedirs("models")
    model_filename = f"credit_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    model_path = os.path.join("models", model_filename)
    joblib.dump(model, model_path)

    # Log version info
    version_info = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "accuracy": round(acc * 100, 2)
    }

    with open("models/version_log.json", "a") as f:
        f.write(json.dumps(version_info) + "\n")
    
    # Save scaler
    scaler_filename = f"scaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    scaler_path = os.path.join("models", scaler_filename)
    joblib.dump(scaler, scaler_path)
    # Log the model and scaler paths
    print(f"✅ Model and scaler saved successfully!")

    ARCHIVE_DIR = "models/archive"
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # Archive old models before saving new one
    for file in os.listdir("models"):
        if file.startswith("credit_model") and file.endswith(".pkl"):
            os.rename(os.path.join("models", file), os.path.join(ARCHIVE_DIR, file))


    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Scaler saved to: {scaler_path}")
    print(f"✅ Version info logged: {version_info}")

    print("\n✅ Model training complete!")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(report)
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

    return version_info["accuracy"],report, model_path, scaler_path, 

# Example usage
if __name__ == "__main__":
    data_path = "C:/Users/masra/Desktop/Project/credit-card-prediction/data/default of credit card clients.xls"
    train_and_save_model(data_path)



