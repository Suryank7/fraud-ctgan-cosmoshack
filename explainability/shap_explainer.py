import shap
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

class FraudExplainer:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, 'rf_augmented.pkl')
            # Fallback to baseline if augmented doesn't exist
            if not os.path.exists(model_path):
                 model_path = os.path.join(MODEL_DIR, 'rf_baseline.pkl')
        
        self.model = joblib.load(model_path)
        self.explainer = None
        
    def fit_explainer(self, X_background):
        # Using a small background sample for speed
        self.explainer = shap.TreeExplainer(self.model)
        
    def explain_local(self, X_instance):
        if self.explainer is None:
            self.fit_explainer(None) # TreeExplainer often doesn't need background for RF
            
        shap_values = self.explainer.shap_values(X_instance)
        # RF shap_values is a list [class0, class1]. We want class 1 (Fraud)
        if isinstance(shap_values, list):
            return shap_values[1] 
        return shap_values

    def get_summary_plot(self, X_sample):
        if self.explainer is None:
            self.fit_explainer(None)
            
        shap_values = self.explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            vals = shap_values[1]
        else:
            vals = shap_values
            
        return vals

if __name__ == "__main__":
    # Test run
    df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv')).drop('Class', axis=1).iloc[:10]
    exp = FraudExplainer()
    vals = exp.explain_local(df)
    print("SHAP values shape:", vals.shape)
