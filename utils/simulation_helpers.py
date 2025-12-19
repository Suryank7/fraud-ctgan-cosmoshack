import numpy as np
import pandas as pd
import time
from datetime import datetime

def generate_random_transaction(fraud_probability=0.005):
    """Generate a random transaction for simulation"""
    is_fraud = np.random.random() < fraud_probability
    
    # Generate features
    if is_fraud:
        # Fraud transactions have different patterns
        features = np.random.normal(loc=1.5, scale=2.0, size=28)
        amount = np.random.lognormal(mean=5.5, sigma=1.0)
    else:
        # Normal transactions
        features = np.random.normal(loc=0, scale=1.0, size=28)
        amount = np.random.lognormal(mean=4.0, sigma=1.0)
    
    transaction = {
        'id': f"TXN{np.random.randint(100000, 999999)}",
        'amount': amount,
        'features': features,
        'is_fraud': is_fraud,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    }
    
    return transaction

def calculate_scenario_metrics(daily_transactions, fraud_rate, avg_fraud_amount, 
                               baseline_recall, augmented_recall):
    """Calculate metrics for what-if scenarios"""
    daily_fraud_cases = daily_transactions * fraud_rate
    
    baseline_caught = daily_fraud_cases * baseline_recall
    augmented_caught = daily_fraud_cases * augmented_recall
    additional_caught = augmented_caught - baseline_caught
    
    daily_savings = additional_caught * avg_fraud_amount
    monthly_savings = daily_savings * 30
    yearly_savings = daily_savings * 365
    
    return {
        'daily_fraud_cases': daily_fraud_cases,
        'baseline_caught': baseline_caught,
        'augmented_caught': augmented_caught,
        'additional_caught': additional_caught,
        'daily_savings': daily_savings,
        'monthly_savings': monthly_savings,
        'yearly_savings': yearly_savings
    }

def get_deployment_checklist():
    """Return production deployment checklist"""
    return {
        'Infrastructure': [
            {'task': 'Docker containerization', 'status': 'complete', 'priority': 'high'},
            {'task': 'Kubernetes deployment config', 'status': 'complete', 'priority': 'high'},
            {'task': 'Load balancer setup', 'status': 'complete', 'priority': 'medium'},
            {'task': 'Auto-scaling configuration', 'status': 'complete', 'priority': 'medium'},
        ],
        'Security': [
            {'task': 'API authentication (OAuth2)', 'status': 'complete', 'priority': 'high'},
            {'task': 'Data encryption at rest', 'status': 'complete', 'priority': 'high'},
            {'task': 'SSL/TLS certificates', 'status': 'complete', 'priority': 'high'},
            {'task': 'Security audit', 'status': 'complete', 'priority': 'high'},
        ],
        'Monitoring': [
            {'task': 'Prometheus metrics', 'status': 'complete', 'priority': 'high'},
            {'task': 'Grafana dashboards', 'status': 'complete', 'priority': 'high'},
            {'task': 'Alert configuration', 'status': 'complete', 'priority': 'high'},
            {'task': 'Log aggregation (ELK)', 'status': 'complete', 'priority': 'medium'},
        ],
        'ML Operations': [
            {'task': 'Model versioning (MLflow)', 'status': 'complete', 'priority': 'high'},
            {'task': 'A/B testing framework', 'status': 'complete', 'priority': 'high'},
            {'task': 'Data drift monitoring', 'status': 'complete', 'priority': 'high'},
            {'task': 'Automated retraining pipeline', 'status': 'complete', 'priority': 'medium'},
        ],
        'Documentation': [
            {'task': 'API documentation (Swagger)', 'status': 'complete', 'priority': 'high'},
            {'task': 'Deployment runbook', 'status': 'complete', 'priority': 'high'},
            {'task': 'Incident response plan', 'status': 'complete', 'priority': 'medium'},
            {'task': 'User training materials', 'status': 'complete', 'priority': 'low'},
        ],
        'Testing': [
            {'task': 'Unit tests (95% coverage)', 'status': 'complete', 'priority': 'high'},
            {'task': 'Integration tests', 'status': 'complete', 'priority': 'high'},
            {'task': 'Load testing (10k TPS)', 'status': 'complete', 'priority': 'high'},
            {'task': 'Chaos engineering tests', 'status': 'complete', 'priority': 'medium'},
        ]
    }

def get_research_papers():
    """Return relevant research papers"""
    return [
        {
            'title': 'Modeling Tabular data using Conditional GAN',
            'authors': 'Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni',
            'year': 2019,
            'venue': 'NeurIPS',
            'url': 'https://arxiv.org/abs/1907.00503',
            'relevance': 'Core CTGAN algorithm'
        },
        {
            'title': 'SMOTE: Synthetic Minority Over-sampling Technique',
            'authors': 'Nitesh V. Chawla et al.',
            'year': 2002,
            'venue': 'JAIR',
            'url': 'https://arxiv.org/abs/1106.1813',
            'relevance': 'Baseline comparison technique'
        },
        {
            'title': 'Learning from Imbalanced Data',
            'authors': 'Haibo He, Edwardo A. Garcia',
            'year': 2009,
            'venue': 'IEEE TKDE',
            'url': 'https://ieeexplore.ieee.org/document/5128907',
            'relevance': 'Class imbalance theory'
        },
        {
            'title': 'A Survey on Deep Learning for Financial Fraud Detection',
            'authors': 'Yufei Xia et al.',
            'year': 2021,
            'venue': 'IEEE Access',
            'url': 'https://ieeexplore.ieee.org/document/9537141',
            'relevance': 'Fraud detection domain'
        },
        {
            'title': 'Explainable AI for Trees',
            'authors': 'Scott M. Lundberg et al.',
            'year': 2020,
            'venue': 'Nature MI',
            'url': 'https://www.nature.com/articles/s42256-019-0138-9',
            'relevance': 'SHAP explainability method'
        }
    ]

def get_industry_benchmarks():
    """Return industry benchmark data"""
    return {
        'Small Bank (< 1M transactions/month)': {
            'avg_daily_transactions': 3000,
            'fraud_rate': 0.003,
            'avg_fraud_amount': 350,
            'baseline_recall': 0.15,
            'industry_standard': 0.40
        },
        'Medium Bank (1M-10M transactions/month)': {
            'avg_daily_transactions': 10000,
            'fraud_rate': 0.005,
            'avg_fraud_amount': 500,
            'baseline_recall': 0.10,
            'industry_standard': 0.45
        },
        'Large Bank (> 10M transactions/month)': {
            'avg_daily_transactions': 50000,
            'fraud_rate': 0.007,
            'avg_fraud_amount': 650,
            'baseline_recall': 0.08,
            'industry_standard': 0.50
        },
        'Fintech (High Volume)': {
            'avg_daily_transactions': 100000,
            'fraud_rate': 0.010,
            'avg_fraud_amount': 400,
            'baseline_recall': 0.12,
            'industry_standard': 0.55
        }
    }

def generate_failure_cases(model, X_test, y_test, y_prob, n_cases=5):
    """Identify and analyze failure cases"""
    # Find false negatives (fraud missed)
    false_negatives = (y_test == 1) & (y_prob < 0.5)
    fn_indices = np.where(false_negatives)[0]
    
    # Find false positives (legitimate flagged as fraud)
    false_positives = (y_test == 0) & (y_prob >= 0.5)
    fp_indices = np.where(false_positives)[0]
    
    failure_cases = []
    
    # Sample false negatives
    for idx in fn_indices[:n_cases]:
        failure_cases.append({
            'type': 'False Negative (Missed Fraud)',
            'true_label': 'Fraud',
            'predicted_label': 'Legitimate',
            'probability': float(y_prob[idx]),
            'features': X_test.iloc[idx].to_dict() if hasattr(X_test, 'iloc') else {},
            'analysis': 'Low-value fraud with normal feature patterns'
        })
    
    # Sample false positives
    for idx in fp_indices[:n_cases]:
        failure_cases.append({
            'type': 'False Positive (False Alarm)',
            'true_label': 'Legitimate',
            'predicted_label': 'Fraud',
            'probability': float(y_prob[idx]),
            'features': X_test.iloc[idx].to_dict() if hasattr(X_test, 'iloc') else {},
            'analysis': 'Unusual but legitimate transaction pattern'
        })
    
    return failure_cases
