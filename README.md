# CyberSentinel

CyberSentinel is an unsupervised anomaly detection system that identifies unusual patterns in network traffic which may indicate cyberattacks or system malfunctions. It uses **Isolation Forest** and **Autoencoder neural networks** trained on the KDD Cup 1999 dataset.

---
#  Streamlit Live Demo

> Want to try it out? Visit the live app here:

**[CyberSentinel Streamlit App](https://cybersentinel-ecymuxdunrynbdpdfexvu4.streamlit.app)**  
*(replace with your actual link after deployment)*

You can upload any `.csv` file with the required features and instantly detect anomalies in your browser — no setup needed.
- for testing you can use this `.csv` file [Download Sample CSV](./data/sample_test.csv)


---

## Features

- Pretrained models (Isolation Forest & Autoencoder)
- Modular codebase for scalable ML pipelines
- EDA & model comparison notebooks
- Clean folder structure for reproducibility
- Streamlit interface for real-time anomaly detection

---
##  Setup Instructions

### 1.  Clone the repository
```bash
git clone https://github.com/yourusername/CyberSentinel.git
cd CyberSentinel
```

### 2. Create and activate virtual environment (recommended)(cann be skipped also)

```bash
python -m venv venv
source venv/bin/activate  # or  venv\\Scripts\\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## How It Works
- The models are trained using normal traffic only

- Anomalies are identified based on how much a new record deviates from this baseline

## How to use

### Command-Line Interface (main.py)

- Run anomaly detection on any user-uploaded CSV file with the required 10 features.

### Required columns:

```bash
logged_in, count, dst_host_count, srv_count,
dst_host_same_src_port_rate, srv_diff_host_rate,
same_srv_rate, dst_host_srv_serror_rate,
serror_rate, dst_host_serror_rate
```

###  Example: Using Isolation Forest (default)

```bash
python main.py data/raw/sample_test.csv
```

###  Example: Using Autoencoder

```bash
python main.py data/raw/sample_test.csv --model autoencoder
```

###  Save predictions to CSV

```bash
python main.py data/raw/sample_test.csv --model autoencoder --output predictions.csv
```
---
# Model Performance Summary

| Metric        | Isolation Forest | Autoencoder |
| ------------- | ---------------- | ----------- |
| **Precision** | 97.33%           |  98.77%    |
| **Recall**    | 98.86%         | 98.45%      |
| **F1-Score**  | 98.09%           |  98.61%    |
| **ROC-AUC**   | 93.90%           |  96.72%    |

--- 

# Dataset
- KDD Cup 1999 (10%)(https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data)

- Preprocessed and used only selected features for better performance