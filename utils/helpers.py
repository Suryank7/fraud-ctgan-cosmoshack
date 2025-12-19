import pandas as pd
import numpy as np
import os
import argparse
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_PATH = os.path.join(DATA_PATH, 'raw', 'creditcard_simulated.csv')
PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')

def generate_credit_card_data(n_rows=50000, fraud_ratio=0.005, random_state=42):
    """
    Generates a synthetic dataset mimicking the Kaggle Credit Card Fraud dataset.
    Features: Time, V1-V28 (PCA), Amount, Class
    """
    np.random.seed(random_state)
    
    n_fraud = int(n_rows * fraud_ratio)
    n_normal = n_rows - n_fraud
    
    # Generate Normal Transactions (Class 0)
    # Normal data tends to be around 0 with unit variance for PCA features
    normal_data = np.random.normal(loc=0, scale=1.0, size=(n_normal, 28))
    
    # Generate Fraud Transactions (Class 1)
    # Fraud data often has higher variance and shifted means in some dimensions
    fraud_data = np.random.normal(loc=1.5, scale=2.0, size=(n_fraud, 28))
    
    # Create DataFrame
    cols = [f'V{i}' for i in range(1, 29)]
    df_normal = pd.DataFrame(normal_data, columns=cols)
    df_normal['Class'] = 0
    
    df_fraud = pd.DataFrame(fraud_data, columns=cols)
    df_fraud['Class'] = 1
    
    # Combine
    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    
    # Add 'Time' (Simulating 2 days of data)
    df['Time'] = np.random.uniform(0, 172800, size=n_rows)
    df = df.sort_values('Time').reset_index(drop=True)
    
    # Add 'Amount'
    # Fraud amounts often higher or specific, but let's make it log-normal for realism
    df['Amount'] = np.random.lognormal(mean=4.0, sigma=1.0, size=n_rows)
    
    print(f"Generated {n_rows} rows with {n_fraud} fraud cases ({fraud_ratio:.2%})")
    return df

def perform_preprocessing(df):
    """
    Scales Amount and Time.
    """
    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    return df

def save_data(df, path):
    df.to_csv(path, index=False)
    print(f"Saved data to {path}")

def load_data():
    if not os.path.exists(RAW_PATH):
        print("Raw data not found. Generating...")
        df = generate_credit_card_data()
        save_data(df, RAW_PATH)
    else:
        df = pd.read_csv(RAW_PATH)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, choices=['generate_data', 'preprocess'], default='generate_data')
    args = parser.parse_args()
    
    if args.action == 'generate_data':
        df = generate_credit_card_data()
        save_data(df, RAW_PATH)
        
        # Also save a preprocessed version split for training
        print("Preprocessing and splitting data...")
        df_processed = perform_preprocessing(df.copy())
        
        # Stratified Split
        X = df_processed.drop('Class', axis=1)
        y = df_processed['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_df.to_csv(os.path.join(PROCESSED_PATH, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(PROCESSED_PATH, 'test.csv'), index=False)
        
        print("Data pipeline completed. Files in data/processed/")
