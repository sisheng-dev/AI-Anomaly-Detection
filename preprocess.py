import pandas as pd

def preprocess_data(df):
    """
    - Performs basic preprocessing.
    - Converts a 'timestamp' column to datetime if it is available
    - Fills in the missing values.
    """
    df = df.copy()
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    df.fillna(0, inplace=True)
    print("Preprocessing completed.")
    return df
