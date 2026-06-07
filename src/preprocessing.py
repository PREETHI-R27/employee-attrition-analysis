"""Preprocessing utilities for Employee Attrition Analysis."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath='../data/WA_Fn-UseC_-HR-Employee-Attrition.csv'):
    """Load raw HR attrition data."""
    return pd.read_csv(filepath)


def engineer_features(df):
    """Create engineered features for attrition modeling."""
    df = df.copy()

    # Age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[17, 25, 35, 45, 55, 65], 
                            labels=['18-25', '26-35', '36-45', '46-55', '56+'])

    # Tenure categories
    df['TenureCategory'] = pd.cut(df['YearsAtCompany'], 
                                  bins=[-1, 2, 5, 10, 20, 50], 
                                  labels=['New (0-2)', 'Junior (3-5)', 'Mid (6-10)', 'Senior (11-20)', 'Veteran (20+)'])

    # Salary levels
    df['SalaryLevel'] = pd.qcut(df['MonthlyIncome'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

    # Satisfaction composite
    satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']
    df['SatisfactionComposite'] = df[satisfaction_cols].mean(axis=1)
    df['SatisfactionRisk'] = pd.cut(df['SatisfactionComposite'], 
                                     bins=[0, 1.5, 2.5, 3.5, 5], 
                                     labels=['Very Low', 'Low', 'Moderate', 'High'])

    # Job hopper indicator
    df['JobHopper'] = np.where((df['NumCompaniesWorked'] >= 4) & (df['TotalWorkingYears'] <= 10), 1, 0)

    # Income efficiency
    df['IncomePerYear'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
    df['IncomeEfficiency'] = pd.qcut(df['IncomePerYear'], q=3, labels=['Low Efficiency', 'Medium Efficiency', 'High Efficiency'])

    return df


def prepare_modeling_data(df):
    """Prepare final feature matrix and target."""
    df = df.copy()

    # Encode target
    df['AttritionEncoded'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in ['Attrition', 'AgeGroup', 'TenureCategory', 
                                                                  'SalaryLevel', 'SatisfactionRisk', 'IncomeEfficiency']]

    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Drop unnecessary columns
    cols_to_drop = ['Attrition', 'EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18']
    cols_to_drop = [c for c in cols_to_drop if c in df_encoded.columns]

    # Also drop engineered categorical columns if they exist as objects
    for col in ['AgeGroup', 'TenureCategory', 'SalaryLevel', 'SatisfactionRisk', 'IncomeEfficiency']:
        if col in df_encoded.columns and df_encoded[col].dtype.name == 'category':
            # Create dummy columns for these engineered features
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)

    X = df_encoded.drop(columns=cols_to_drop + ['AttritionEncoded'])
    y = df_encoded['AttritionEncoded']

    return X, y


def full_pipeline(filepath='../data/WA_Fn-UseC_-HR-Employee-Attrition.csv', test_size=0.2, random_state=42):
    """Complete preprocessing pipeline."""
    df = load_data(filepath)
    df = engineer_features(df)
    X, y = prepare_modeling_data(df)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()


if __name__ == "__main__":
    print("Testing Employee Preprocessing Pipeline...")
    try:
        X_train, X_test, y_train, y_test, scaler, feats = full_pipeline('../data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
        print(f"Success! Features extracted: {len(feats)}")
        print(f"Training shape: {X_train.shape}")
        print(f"Attrition Rate in Test Set: {y_test.mean():.2%}")
    except Exception as e:
        print(f"Error: {e}")
