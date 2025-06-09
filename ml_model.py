import pandas as pd
import matplotlib.pyplot as plt

def detect_anomalies(df):
    """
    I use IsolationForest for unsupervised anomaly detection.
    """
    if 'value' in df.columns:
        X = df[['value']]
    else:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns available for anomaly detection.")
        X = df[[numeric_cols[0]]]
    
    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly_label'] = model.fit_predict(X)
    
    anomalies = df[df['anomaly_label'] == -1]
    print("Anomalies detected:", anomalies.shape[0])
    return anomalies

def plot_anomalies(df, anomalies):
    """
    Plots the distribution and highlights the detected anomalies.
    """
    if 'value' in df.columns:
        plt.figure(figsize=(8, 4))
        plt.hist(df['value'], bins=50, alpha=0.7, label="All Data")
        plt.hist(anomalies['value'], bins=50, alpha=0.7, label="Anomalies")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Distribution of 'value' with Detected Anomalies")
        plt.legend()
        plt.show()
    else:
        print("No 'value' column found for plotting.")
