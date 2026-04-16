import argparse
import pandas as pd
from src.data_preprocessing import load_scaler, preprocess_uploaded_data
from src.isolation_forest_model import load_isolation_forest_model, predict_anomalies as predict_iso
from src.autoencoder_model import load_autoencoder_model, predict_anomalies as predict_ae

def main():
    parser = argparse.ArgumentParser(description="CyberSentinel - Anomaly Detection")
    parser.add_argument("file", help="Path to the CSV file containing network data")
    parser.add_argument("--model", choices=["isolation_forest", "autoencoder"], default="isolation_forest",
                        help="Which model to use for anomaly detection")
    parser.add_argument("--output", help="Optional path to save the output CSV with predictions")

    args = parser.parse_args()

    try:
        print("Loading scaler and preprocessing data...")
        scaler = load_scaler()
        X = preprocess_uploaded_data(args.file, scaler)

        print(f"Running {args.model} model...")
        if args.model == "isolation_forest":
            model = load_isolation_forest_model()
            preds = predict_iso(model, X)
        else:
            model, threshold = load_autoencoder_model()
            preds = predict_ae(model, threshold, X)

        # Output
        output_df = pd.read_csv(args.file).copy()
        output_df["anomaly"] = preds
        print(output_df[["anomaly"]].value_counts())

        if args.output:
            output_df.to_csv(args.output, index=False)
            print(f"Predictions saved to {args.output}")
        else:
            print("Sample predictions:")
            print(output_df.head())

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
