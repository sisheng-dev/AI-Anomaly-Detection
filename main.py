from dataset_loader import extract_data, load_dataset
from preprocess import preprocess_data
from ml_model import detect_anomalies, plot_anomalies
from blockchain import connect_to_blockchain, monitor_transactions

def main():
    # Define paths and directories
    path = "dataset"
    extract_dir = "extracted_data"

    # Simulate blockchain operations
    connect_to_blockchain()
    monitor_transactions()

    # load the dataset
    extract_data(path, extract_dir)
    df = load_dataset(extract_dir)

    # Preprocess the dataset
    df = preprocess_data(df)

    # Run anomaly detection using the ML model
    anomalies = detect_anomalies(df)

    # Plot anomalies if they apply
    plot_anomalies(df, anomalies)

if __name__ == "__main__":
    main()
