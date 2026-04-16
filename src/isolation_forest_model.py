import joblib
import numpy as np

def load_isolation_forest_model(path="models/isolation_forest.pkl"):
    """Load the trained Isolation Forest model from disk."""
    return joblib.load(path)

def predict_anomalies(model, scaled_df):
    """
    Predict anomalies using the Isolation Forest model.
    - model: trained Isolation Forest model
    - scaled_df: DataFrame with scaled features
    Returns: array of binary predictions (0 = normal, 1 = anomaly)
    """
    raw_preds = model.predict(scaled_df)  # 1 = normal, -1 = anomaly
    return np.where(raw_preds == 1, 0, 1)

# Optional standalone test
if __name__ == "__main__":
    import pandas as pd
    from data_preprocessing import load_scaler, preprocess_uploaded_data

    model = load_isolation_forest_model()
    scaler = load_scaler()
    df_scaled = preprocess_uploaded_data("../data/raw/sample_test.csv", scaler)

    preds = predict_anomalies(model, df_scaled)
    print("Predictions:", preds[:10])
