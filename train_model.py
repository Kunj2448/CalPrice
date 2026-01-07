import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -----------------------
# File Paths
# -----------------------
DATA_FILE = "data/housing1.csv"
MODEL_FILE = "model/model.pkl"
PIPELINE_FILE = "model/pipeline.pkl"
METRICS_FILE = "metrics.txt"
INPUT_FILE = "data/input.csv"
OUTPUT_FILE = "data/output.csv"

TARGET = "median_house_value"

# -----------------------
# Create Folders If Missing
# -----------------------
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# -----------------------
# Load Data
# -----------------------
def load_data():
    return pd.read_csv(DATA_FILE)

# -----------------------
# Build Pipeline
# -----------------------
def build_pipeline(num_features, cat_features):

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

# -----------------------
# Train Model
# -----------------------
def train_model():

    print("🚀 Training started...")

    df = load_data()

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    # Identify column types
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build preprocessing pipeline
    pipeline = build_pipeline(num_features, cat_features)

    # Transform data
    X_train_prepared = pipeline.fit_transform(X_train)
    X_test_prepared = pipeline.transform(X_test)

    # Train model
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_prepared, y_train)

    # Predictions
    y_pred = model.predict(X_test_prepared)

    # -----------------------
    # Metrics
    # -----------------------
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    accuracy = r2 * 100

    print("\n📊 Model Performance:")
    print(f"R2 Score        : {r2:.4f}")
    print(f"Accuracy (%)    : {accuracy:.2f}%")
    print(f"MAE             : {mae:.2f}")
    print(f"RMSE            : {rmse:.2f}")

    # Save metrics
    with open(METRICS_FILE, "w") as f:
        f.write(f"R2 Score     : {r2:.4f}\n")
        f.write(f"Accuracy (%) : {accuracy:.2f}\n")
        f.write(f"MAE          : {mae:.2f}\n")
        f.write(f"RMSE         : {rmse:.2f}\n")

    # Save model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("\n✅ Model and pipeline saved successfully.")
    print("📁 Metrics saved to metrics.txt")

# -----------------------
# Predict New Data
# -----------------------
def predict_new_data(input_csv=INPUT_FILE, output_csv=OUTPUT_FILE):

    print("\n🔮 Running inference...")

    if not os.path.exists(input_csv):
        print(f"⚠️ Input file not found: {input_csv}")
        return

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv(input_csv)
    transformed = pipeline.transform(input_data)
    predictions = model.predict(transformed)

    input_data["PredictedPrice"] = predictions.round(2)
    input_data.to_csv(output_csv, index=False)

    print(f"✅ Predictions saved to {output_csv}")

# -----------------------
# Main Runner
# -----------------------
if __name__ == "__main__":

    # Train only if model doesn't exist
    if not os.path.exists(MODEL_FILE):
        train_model()
    else:
        print("✅ Model already exists. Skipping training.")

    # Run inference
    predict_new_data()
