"""Prediction utilities for Employee Attrition Analysis."""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


def load_artifacts(model_path='../models/xgboost_attrition.pkl', scaler_path='../models/scaler.pkl'):
    """Load trained model and scaler."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_attrition_risk(X, model, scaler):
    """Predict attrition risk scores for employees."""
    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = model.predict(X_scaled)
    return predictions, probabilities


def categorize_risk(probabilities, thresholds=[0.3, 0.6]):
    """Categorize employees into risk buckets."""
    categories = []
    for p in probabilities:
        if p < thresholds[0]:
            categories.append('Low Risk')
        elif p < thresholds[1]:
            categories.append('Medium Risk')
        else:
            categories.append('High Risk')
    return categories


def generate_risk_report(employee_df, X_features, model, scaler, top_n=20):
    """Generate a risk report with high-risk employees."""
    predictions, probabilities = predict_attrition_risk(X_features, model, scaler)
    categories = categorize_risk(probabilities)

    report_df = employee_df.copy()
    report_df['PredictedAttrition'] = predictions
    report_df['AttritionRiskScore'] = probabilities
    report_df['RiskCategory'] = categories

    high_risk = report_df[report_df['RiskCategory'] == 'High Risk'].sort_values('AttritionRiskScore', ascending=False)

    return report_df, high_risk.head(top_n)


if __name__ == '__main__':
    # Example usage
    from preprocessing import prepare_modeling_data, engineer_features, load_data

    df = load_data()
    df = engineer_features(df)
    X, _ = prepare_modeling_data(df)

    model, scaler = load_artifacts()
    report, high_risk = generate_risk_report(df, X, model, scaler)

    print(f"Total employees: {len(report)}")
    print(f"High risk employees: {len(report[report['RiskCategory'] == 'High Risk'])}")
    print("\nTop 10 highest risk:")
    print(high_risk[['Age', 'Department', 'JobRole', 'MonthlyIncome', 'AttritionRiskScore']].head(10))
