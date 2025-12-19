import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
import xgboost as xgb
from imblearn.over_sampling import SMOTE, ADASYN

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
SYNTHETIC_PATH = os.path.join(DATA_DIR, 'synthetic_fraud.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
RESULTS_PATH = os.path.join(MODEL_DIR, 'results.json')

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']
    
    return X_train, y_train, X_test, y_test

def train_evaluate(X_train, y_train, X_test, y_test, model_name="baseline"):
    print(f"\n--- Training {model_name} Model ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "recall": float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Calculate ROC curve data
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    metrics['roc_curve'] = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist()
    }
    
    # Calculate PR curve data
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_prob)
    metrics['pr_curve'] = {
        'precision': precision_curve.tolist(),
        'recall': recall_curve.tolist(),
        'thresholds': pr_thresholds.tolist()
    }
    
    # Feature importances
    feature_names = X_train.columns.tolist()
    importances = model.feature_importances_
    metrics['feature_importance'] = {
        'features': feature_names,
        'importances': importances.tolist()
    }
    
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Save Model
    joblib.dump(model, os.path.join(MODEL_DIR, f'rf_{model_name}.pkl'))
    
    # Save predictions for later analysis
    pred_df = pd.DataFrame({
        'y_true': y_test.values,
        'y_pred': y_pred,
        'y_prob': y_prob
    })
    pred_df.to_csv(os.path.join(MODEL_DIR, f'predictions_{model_name}.csv'), index=False)
    
    return metrics

def train_comparison_techniques(X_train, y_train, X_test, y_test):
    """Train models with different resampling techniques for comparison"""
    print("\n=== Training Comparison Techniques ===")
    comparison_results = {}
    
    # SMOTE
    try:
        print("\n--- Training with SMOTE ---")
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        
        model_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_smote.fit(X_smote, y_smote)
        
        y_pred_smote = model_smote.predict(X_test)
        y_prob_smote = model_smote.predict_proba(X_test)[:, 1]
        
        comparison_results['smote'] = {
            "recall": float(recall_score(y_test, y_pred_smote)),
            "precision": float(precision_score(y_test, y_pred_smote, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_smote)),
            "roc_auc": float(roc_auc_score(y_test, y_prob_smote))
        }
        joblib.dump(model_smote, os.path.join(MODEL_DIR, 'rf_smote.pkl'))
        print(f"SMOTE Recall: {comparison_results['smote']['recall']:.4f}")
    except Exception as e:
        print(f"SMOTE failed: {e}")
        comparison_results['smote'] = None
    
    # ADASYN
    try:
        print("\n--- Training with ADASYN ---")
        adasyn = ADASYN(random_state=42)
        X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
        
        model_adasyn = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_adasyn.fit(X_adasyn, y_adasyn)
        
        y_pred_adasyn = model_adasyn.predict(X_test)
        y_prob_adasyn = model_adasyn.predict_proba(X_test)[:, 1]
        
        comparison_results['adasyn'] = {
            "recall": float(recall_score(y_test, y_pred_adasyn)),
            "precision": float(precision_score(y_test, y_pred_adasyn, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_adasyn)),
            "roc_auc": float(roc_auc_score(y_test, y_prob_adasyn))
        }
        joblib.dump(model_adasyn, os.path.join(MODEL_DIR, 'rf_adasyn.pkl'))
        print(f"ADASYN Recall: {comparison_results['adasyn']['recall']:.4f}")
    except Exception as e:
        print(f"ADASYN failed: {e}")
        comparison_results['adasyn'] = None
    
    return comparison_results

def main():
    X_train, y_train, X_test, y_test = load_data()
    results = {}
    
    # 1. Baseline Model (Imbalanced)
    results['baseline'] = train_evaluate(X_train, y_train, X_test, y_test, "baseline")
    
    # 2. Augmented Model (Real + Synthetic)
    if os.path.exists(SYNTHETIC_PATH):
        print("\nFound synthetic data. Creating Augmented Dataset...")
        syn_df = pd.read_csv(SYNTHETIC_PATH)
        X_syn = syn_df.drop('Class', axis=1)
        y_syn = syn_df['Class']
        
        X_train_aug = pd.concat([X_train, X_syn], ignore_index=True)
        y_train_aug = pd.concat([y_train, y_syn], ignore_index=True)
        
        results['augmented'] = train_evaluate(X_train_aug, y_train_aug, X_test, y_test, "augmented")
    else:
        print("\nNo synthetic data found. Skipping augmented model.")
    
    # 3. Comparison Techniques
    comparison = train_comparison_techniques(X_train, y_train, X_test, y_test)
    results['comparison'] = comparison
    
    # Save Results
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
