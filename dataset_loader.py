import os
import dataset
import pandas as pd

def extract_data(zip_path, extract_to='extracted_data'):
    """
    Extracts the zip file containing the dataset.
    """
    with dataset.Dataset(zip_path, 'r') as z:
        z.extractall(extract_to)
    print("Extraction completed. Files are extracted to:", extract_to)

def load_dataset(extracted_dir):
    """
    Searches for a CSV file in the extracted directory and loads it.
    """
    csv_file = None
    for file in os.listdir(extracted_dir):
        if file.endswith(".csv"):
            csv_file = os.path.join(extracted_dir, file)
            break

    if not csv_file:
        raise FileNotFoundError("No CSV file found in the extracted data.")
    
    df = pd.read_csv(csv_file)
    print("Dataset loaded with shape:", df.shape)
    return df
