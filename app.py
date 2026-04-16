import streamlit as st
import pandas as pd
import numpy as np
import io
import seaborn as sns
import matplotlib.pyplot as plt

from src.data_preprocessing import load_scaler, preprocess_uploaded_data
from src.isolation_forest_model import load_isolation_forest_model, predict_anomalies as predict_iso
from src.autoencoder_model import load_autoencoder_model, predict_anomalies as predict_ae

# Configure page
st.set_page_config(page_title="CyberSentinel Dashboard", layout="wide")

st.title(" CyberSentinel – Anomaly Detection in Network Traffic")
st.markdown("Upload your traffic data, choose a model, and detect anomalies with one click.")

uploaded_file = st.file_uploader("Upload a `.csv` file", type=["csv"])

if uploaded_file is not None:
    try:
        content = uploaded_file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")
            st.stop()

        st.success("File uploaded successfully!")
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        if 'label' in df.columns:
            st.markdown("####Label Distribution")
            st.bar_chart(df['label'].value_counts())

        scaler = load_scaler()
        df_scaled = preprocess_uploaded_data(df, scaler)

        st.subheader("Choose Anomaly Detection Model")
        model_choice = st.radio("Select Model:", ["Isolation Forest", "Autoencoder"], horizontal=True)

        if model_choice == "Isolation Forest":
            st.info("Isolation Forest isolates data points randomly. Fewer splits → more likely to be an anomaly.")
        else:
            st.info("Autoencoder learns to reconstruct normal patterns. High reconstruction error → likely an anomaly.")

        run_button_label = f"Run Anomaly Detection with {model_choice}"
        if st.button(run_button_label):
            with st.spinner("Detecting anomalies..."):
                if model_choice == "Isolation Forest":
                    model = load_isolation_forest_model('models/isolation_forest_model.pkl')
                    preds = predict_iso(model, df_scaled)
                else:
                    model, threshold = load_autoencoder_model()
                    preds = predict_ae(model, threshold, df_scaled)

                df['anomaly'] = preds
                st.session_state['predicted_df'] = df.copy()
                st.success("Anomaly detection complete!")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a file to begin.")

# Persistent dashboard using session_state
if 'predicted_df' in st.session_state:
    df = st.session_state['predicted_df']

    st.subheader(" Explore Detected Anomalies")

    # Filter + sort
    anomaly_filter = st.selectbox("Filter by Anomaly:", ["All", "Normal (0)", "Anomaly (1)"])
    sort_column = st.selectbox("Sort by:", df.columns.tolist(), index=df.columns.get_loc("count") if "count" in df.columns else 0)
    num_rows = st.slider("Max rows to display:", min_value=10, max_value=100, value=20)

    filtered_df = df.copy()
    if anomaly_filter == "Normal (0)":
        filtered_df = filtered_df[filtered_df['anomaly'] == 0]
    elif anomaly_filter == "Anomaly (1)":
        filtered_df = filtered_df[filtered_df['anomaly'] == 1]

    if not filtered_df.empty:
        filtered_df = filtered_df.sort_values(by=sort_column, ascending=False).head(num_rows)
        st.dataframe(filtered_df)
    else:
        st.warning(" No records match your filter.")

    # Protocol / Service summary
    st.subheader(" Protocol / Service Anomaly Summary")

    if 'protocol_type' in df.columns:
        st.markdown("#### Protocols with Most Anomalies")
        proto_counts = df[df['anomaly'] == 1]['protocol_type'].value_counts().head(10)
        if not proto_counts.empty:
            fig, ax = plt.subplots(figsize=(5, 3))
            proto_counts.plot(kind='bar', color='orange', ax=ax)
            ax.set_title("Top Protocols (Anomalies Only)")
            ax.set_ylabel("Count")
            st.pyplot(fig)

    if 'service' in df.columns:
        st.markdown("#### Services with Most Anomalies")
        svc_counts = df[df['anomaly'] == 1]['service'].value_counts().head(10)
        if not svc_counts.empty:
            fig, ax = plt.subplots(figsize=(5, 3))
            svc_counts.plot(kind='bar', color='green', ax=ax)
            ax.set_title("Top Services (Anomalies Only)")
            ax.set_ylabel("Count")
            st.pyplot(fig)

    # Summary
    st.subheader(" Threat Summary")
    st.metric("Total Records", len(df))
    st.metric("Anomalies Detected", int(df['anomaly'].sum()))
    st.metric("Model Used", model_choice)

    # Top suspicious activity
    st.subheader(" Top Suspicious Activity")
    top_anomalies = df[df['anomaly'] == 1].copy()
    top_anomalies = top_anomalies.sort_values(by='count', ascending=False).head(10)
    st.dataframe(top_anomalies)

    # Download
    st.markdown("####  Download Labeled Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(" Download as CSV", csv, file_name="anomaly_predictions.csv")

    # Anomaly explanation
    if not top_anomalies.empty:
        st.subheader(" Why This Record Is Suspicious?")
        sample = top_anomalies.iloc[0]
        reasons = []
        if sample['count'] > 50: reasons.append("High packet count")
        if sample['serror_rate'] > 0.5: reasons.append("High SYN error rate")
        if sample['dst_host_serror_rate'] > 0.5: reasons.append("Host-level SYN errors")

        if reasons:
            st.markdown("**Anomaly Reasoning:**")
            for r in reasons:
                st.markdown(f"- {r}")
        else:
            st.markdown("_No clear rules triggered; flagged based on model confidence._")
