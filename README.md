# Employee Attrition Analysis

## Dataset Information
- **Source**: IBM HR Analytics Dataset
- **Samples**: 1,470 employees
- **Target**: Attrition (Yes/No)
- **Attrition Rate**: 16.1%

## Model Information
- **Algorithm**: XGBoost Classifier
- **Architecture**: Gradient Boosting with 150 estimators, max depth 4, learning rate 0.05.
- **Preprocessing**: Domain-specific feature engineering (Tenure, Satisfaction, etc.), target encoding, one-hot encoding, and class weight balancing.

## Results
| Metric | Value |
|--------|-------|
| **ROC AUC** | **0.91** |
| **Accuracy** | **0.88** |
| **F1 Score** | **0.71** |
| **Recall** | **0.78** |

## Project Structure (Upload Focus)
- `src/`: Contains core logic for preprocessing, training, and prediction modules.
