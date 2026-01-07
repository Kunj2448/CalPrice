import pandas as pd
import joblib
import numpy as np

MODEL_PATH = "model/model.pkl"
PIPELINE_PATH = "model/pipeline.pkl"

model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)

def predict_single(input_dict):
    df = pd.DataFrame([input_dict])
    transformed = pipeline.transform(df)
    prediction = model.predict(transformed)
    return round(prediction[0], 2)

def predict_bulk(df):
    transformed = pipeline.transform(df)
    predictions = model.predict(transformed)
    df["PredictedPrice"] = predictions.round(2)
    return df

def get_feature_importance():
    try:
        importances = model.feature_importances_
        feature_names = pipeline.get_feature_names_out()
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        return importance_df
    except:
        return None
