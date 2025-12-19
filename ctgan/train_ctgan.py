import pandas as pd
import os
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import joblib
import warnings

warnings.filterwarnings('ignore')

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'train.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'ctgan_fraud_model.pkl')

def train_ctgan_model():
    print("Loading training data...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}. Please run utils/helpers.py first.")
    
    df = pd.read_csv(DATA_PATH)
    
    # Isolate Fraud Cases
    fraud_df = df[df['Class'] == 1].drop('Class', axis=1)
    print(f"Training CTGAN on {len(fraud_df)} fraud samples...")
    
    if len(fraud_df) < 50:
        print("WARNING: Very few fraud samples. CTGAN might struggle.")

    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(fraud_df)
    
    # Initialize CTGAN
    # Using fewer epochs for demo speed, usually needs 300+
    ctgan = CTGANSynthesizer(metadata, epochs=200, verbose=True)
    
    # Train
    ctgan.fit(fraud_df)
    print("CTGAN Training Complete.")
    
    # Save
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    ctgan.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_ctgan_model()
