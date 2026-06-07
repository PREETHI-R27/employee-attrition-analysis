"""Model training utilities for Employee Attrition Analysis."""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)


def train_logistic_regression(X_train, y_train):
    """Train logistic regression."""
    model = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, max_depth=8):
    """Train decision tree."""
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=10, 
                                   random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=200):
    """Train random forest."""
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=12, 
                                   min_samples_split=5, random_state=42, 
                                   class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, n_estimators=150, scale_pos_weight=5):
    """Train XGBoost classifier."""
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name='Model'):
    """Evaluate and return metrics dictionary."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1_Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC_AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        'Confusion_Matrix': cm,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    return metrics


def cross_validate_model(model, X_train, y_train, cv_folds=5):
    """Perform cross-validation and return scores."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    acc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    rec = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall')
    roc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    return {
        'CV_Accuracy_Mean': acc.mean(),
        'CV_Recall_Mean': rec.mean(),
        'CV_ROC_AUC_Mean': roc.mean()
    }


def save_model(model, filepath='../models/xgboost_attrition.pkl'):
    """Save model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath='../models/xgboost_attrition.pkl'):
    """Load model from disk."""
    return joblib.load(filepath)


if __name__ == "__main__":
    print("Testing Employee Attrition Model Training...")
    from preprocessing import full_pipeline
    
    try:
        X_train, X_test, y_train, y_test, scaler, feats = full_pipeline('../data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
        print("Data loaded. Training baseline Logistic Regression...")
        model = train_logistic_regression(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, "Baseline LR")
        print(f"ROC AUC Score: {metrics['ROC_AUC']:.4f}")
        print(f"F1 Score: {metrics['F1_Score']:.4f}")
    except Exception as e:
        print(f"Error: {e}")
