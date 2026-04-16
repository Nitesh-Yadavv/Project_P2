import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler



def load_scaler(path="models/scaler.pkl"):
    """Load the saved scaler object."""
    return joblib.load(path)

def preprocess_uploaded_data(df, scaler):
    """
    Preprocess user-uploaded data to match model input.
    - file_path: path to the uploaded CSV file
    - scaler: trained StandardScaler object
    Returns scaled DataFrame with selected features.
    """
    # Top 10 selected features from EDA
    SELECTED_FEATURES = [
    'logged_in', 'count', 'dst_host_count', 'srv_count',
    'dst_host_same_src_port_rate', 'srv_diff_host_rate',
    'same_srv_rate', 'dst_host_srv_serror_rate',
    'serror_rate', 'dst_host_serror_rate'
]

    # Check if all selected features are present
    missing = [col for col in SELECTED_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    df_selected = df[SELECTED_FEATURES].copy()
    df_scaled = scaler.transform(df_selected)
    return pd.DataFrame(df_scaled, columns=SELECTED_FEATURES)

# For standalone testing
if __name__ == "__main__":
    scaler = load_scaler()
    processed = preprocess_uploaded_data("../data/raw/sample_test.csv", scaler)
    print(processed.head())
