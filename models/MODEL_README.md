# XGBoost Small Business Risk Assessment Model

## Overview

We developed a dual-model machine learning system to predict small business loan default risk in Ouachita Parish, Louisiana. Our solution helps lenders make better decisions while expanding credit access to underserved businesses.

## Our Approach

We built two complementary XGBoost models:

| Model | Purpose | Performance |
|-------|---------|-------------|
| **Classifier** | Predicts default (Yes/No) | 99% AUC-ROC, 93% F1 |
| **Regressor** | Outputs risk score (0-100) | 96% R², 3.6 MAE |

## Methodology

1. **Feature Engineering** - We engineered 40 features across 6 categories combining business financials with Mastercard IGS regional economic indicators
2. **Class Balancing** - We addressed the imbalanced dataset using scale_pos_weight tuning
3. **Hyperparameter Optimization** - We used 5-fold cross-validation to tune model parameters
4. **Dual Output** - We provide both binary classification and continuous risk scoring for flexible decision-making

## Feature Categories (40 Features)

| Category | Count | Examples |
|----------|-------|----------|
| Business Metrics | 4 | Revenue, growth, profit margin, age |
| Cash Flow | 3 | Bank balance, expense ratio, overdrafts |
| Debt & Loans | 6 | Outstanding debt, interest rates, payment history |
| Workforce | 4 | Employees, payroll, payroll growth |
| Credit History | 4 | Credit score, liens, prior defaults |
| Card Spending | 4 | Monthly spend, utilization, payment timeliness |
| IGS Regional | 10 | Local economic indicators from Mastercard IGS |
| Derived/Encoded | 5 | Benchmark comparisons, sector encoding |

## Risk Score Interpretation

| Score | Level | Action |
|-------|-------|--------|
| 0-30 | Low Risk | Standard approval |
| 31-50 | Medium Risk | Approve with monitoring |
| 51-70 | High Risk | Additional review required |
| 71-100 | Very High Risk | Decline or restructure |

## How To Use

```python
import pickle

# Load our trained models
classifier = pickle.load(open('xgboost_classifier.pkl', 'rb'))
regressor = pickle.load(open('xgboost_regressor.pkl', 'rb'))
encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# Get predictions
default_probability = classifier.predict_proba(X)[:, 1]
risk_score = regressor.predict(X)
```

## Model Files

- `xgboost_classifier.pkl` - Default prediction model
- `xgboost_regressor.pkl` - Risk score model  
- `label_encoders.pkl` - Category encoders for industry/region
- `model_metadata.json` - Training configuration and metrics

## Key Results

- **99% AUC-ROC** on held-out test set
- **93% F1 Score** balancing precision and recall
- **96% R²** for continuous risk scoring
- Trained on 15,000 business records with 80/20 split
