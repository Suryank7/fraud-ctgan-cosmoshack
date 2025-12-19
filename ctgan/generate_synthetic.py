import pandas as pd
import os
from sdv.single_table import CTGANSynthesizer
import argparse

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'ctgan_fraud_model.pkl')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'synthetic_fraud.csv')

def generate_synthetic_data(n_samples=500):
    print(f"Loading CTGAN model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
         raise FileNotFoundError(f"Model not found. Please run ctgan/train_ctgan.py first.")

    ctgan_model = CTGANSynthesizer.load(MODEL_PATH)
    
    print(f"Generating {n_samples} synthetic fraud samples...")
    synthetic_data = ctgan_model.sample(n_samples)
    
    # Add Class label back (since we dropped it during training for pure feature learning)
    synthetic_data['Class'] = 1
    
    # Save
    synthetic_data.to_csv(OUTPUT_PATH, index=False)
    print(f"Synthetic data saved to {OUTPUT_PATH}")
    return synthetic_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=1000, help='Number of synthetic samples to generate')
    args = parser.parse_args()
    
    generate_synthetic_data(args.samples)
