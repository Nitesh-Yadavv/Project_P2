import joblib
import numpy as np
from tensorflow.keras.models import load_model

def load_autoencoder_model(model_path="models/autoencoder_model.h5", threshold_path="models/autoencoder_threshold.pkl"):
    """
    Load the trained autoencoder model and its threshold.
    Returns: (model, threshold)
    """
    model = load_model(model_path)
    threshold = joblib.load(threshold_path)
    return model, threshold

def predict_anomalies(model, threshold, scaled_df):
    """
    Predict anomalies using the autoencoder.
    - model: trained autoencoder
    - threshold: pre-determined anomaly threshold
    - scaled_df: DataFrame with scaled features
    Returns: array of binary predictions (0 = normal, 1 = anomaly)
    """
    reconstructions = model.predict(scaled_df)
    reconstruction_error = np.mean(np.power(scaled_df - reconstructions, 2), axis=1)
    return (reconstruction_error > threshold).astype(int)

# Optional standalone test
if __name__ == "__main__":
    import pandas as pd
    from data_preprocessing import load_scaler, preprocess_uploaded_data

    model, threshold = load_autoencoder_model()
    scaler = load_scaler()
    df_scaled = preprocess_uploaded_data("../data/raw/sample_test.csv", scaler)

    preds = predict_anomalies(model, threshold, df_scaled)
    print("Predictions:", preds[:10])
