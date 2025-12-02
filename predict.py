#!/usr/bin/env python3
"""
🎯 Business Risk Prediction Script
===================================
Loads trained XGBoost models and predicts:
  - default_flag (0/1): Will the business default?
  - risk_score (0-100): How risky is this business?

Usage:
    python predict.py --input sample_businesses.csv --output predictions.csv
    python predict.py  # Uses default sample data
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# Feature columns (must match training order)
FEATURE_COLUMNS = [
    # Financial metrics
    'annual_revenue', 'revenue_growth_yoy', 'revenue_volatility',
    'profit_margin', 'gross_margin', 'operating_margin',
    'avg_bank_balance', 'min_bank_balance_last_12m', 'bank_balance_volatility',
    'cash_inflow_monthly_avg', 'cash_outflow_monthly_avg',
    'cash_to_monthly_expense_ratio', 'net_cash_flow_trend',
    
    # Debt metrics
    'total_debt', 'debt_to_revenue', 'debt_service_coverage_ratio',
    'credit_utilization_ratio', 'loan_payment_to_revenue',
    
    # Workforce metrics
    'employee_count', 'payroll_to_revenue', 'avg_wage_per_employee',
    'employee_growth_rate', 'payroll_consistency_score',
    
    # Credit & compliance
    'credit_score_business', 'credit_score_owner', 'years_in_business',
    'days_past_due_max_last_12m', 'num_late_payments_last_12m',
    'num_returned_payments_last_12m', 'has_tax_lien', 'bankruptcy_history',
    
    # Operational
    'missed_payroll_count_last_12m', 'overdraft_count_last_12m',
    'overdraft_days_last_12m', 'utilization_rate_card',
    
    # IGS benchmarks
    'igs_sales_index', 'igs_employment_index', 'igs_small_business_index',
    
    # Encoded categorical
    'industry_sector_encoded', 'region_parish_encoded'
]

# Industry and Parish mappings
INDUSTRY_MAPPING = {
    'Retail': 0, 'Restaurant': 1, 'Healthcare': 2, 'Construction': 3,
    'Manufacturing': 4, 'Professional Services': 5, 'Transportation': 6,
    'Agriculture': 7, 'Technology': 8, 'Other': 9
}

PARISH_MAPPING = {
    'Ouachita': 0, 'Caddo': 1, 'East Baton Rouge': 2, 'Jefferson': 3,
    'Lafayette': 4, 'Orleans': 5, 'Rapides': 6, 'Calcasieu': 7,
    'Bossier': 8, 'Tangipahoa': 9
}


# ═══════════════════════════════════════════════════════════════════════════════
#                              LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def load_models():
    """Load trained XGBoost models and encoders."""
    
    classifier_path = os.path.join(MODEL_DIR, 'xgboost_classifier.pkl')
    regressor_path = os.path.join(MODEL_DIR, 'xgboost_regressor.pkl')
    encoders_path = os.path.join(MODEL_DIR, 'label_encoders.pkl')
    
    # Check if models exist
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(
            f"Model not found: {classifier_path}\n"
            "Please run the training notebook first to generate models."
        )
    
    with open(classifier_path, 'rb') as f:
        classifier = pickle.load(f)
    
    with open(regressor_path, 'rb') as f:
        regressor = pickle.load(f)
    
    encoders = None
    if os.path.exists(encoders_path):
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
    
    return classifier, regressor, encoders


# ═══════════════════════════════════════════════════════════════════════════════
#                              PREDICTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_data(df, encoders=None):
    """Preprocess input data for prediction."""
    
    df_processed = df.copy()
    
    # Encode categorical columns
    if 'industry_sector' in df_processed.columns:
        df_processed['industry_sector_encoded'] = df_processed['industry_sector'].map(
            INDUSTRY_MAPPING
        ).fillna(9)  # Default to 'Other'
    
    if 'region_parish' in df_processed.columns:
        df_processed['region_parish_encoded'] = df_processed['region_parish'].map(
            PARISH_MAPPING
        ).fillna(0)  # Default to 'Ouachita'
    
    # Select feature columns (only those present)
    available_features = [col for col in FEATURE_COLUMNS if col in df_processed.columns]
    
    # Fill missing columns with median/default values
    for col in FEATURE_COLUMNS:
        if col not in df_processed.columns:
            df_processed[col] = 0  # Default value
    
    return df_processed[FEATURE_COLUMNS]


def predict(df, classifier, regressor):
    """Make predictions using trained models."""
    
    # Get predictions
    default_proba = classifier.predict_proba(df)[:, 1]
    default_flag = classifier.predict(df)
    risk_score = np.clip(regressor.predict(df), 0, 100)
    
    # Determine risk band
    risk_bands = pd.cut(
        risk_score,
        bins=[-1, 40, 75, 100],
        labels=['Low', 'Medium', 'High']
    )
    
    return {
        'default_probability': default_proba,
        'default_flag': default_flag,
        'risk_score': risk_score,
        'risk_band': risk_bands
    }


def get_risk_assessment(row):
    """Generate text assessment based on predictions."""
    
    risk_score = row['risk_score']
    default_prob = row['default_probability']
    
    if risk_score < 40:
        status = "✅ LOW RISK"
        recommendation = "Eligible for standard lending terms"
    elif risk_score < 75:
        status = "⚠️ MEDIUM RISK"
        recommendation = "Recommend enhanced monitoring or collateral"
    else:
        status = "🚨 HIGH RISK"
        recommendation = "Not recommended for unsecured lending"
    
    return f"{status} | Score: {risk_score:.1f} | Default Prob: {default_prob:.1%} | {recommendation}"


# ═══════════════════════════════════════════════════════════════════════════════
#                              SAMPLE DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_sample_businesses(n=10):
    """Generate sample business records for testing."""
    
    np.random.seed(42)
    
    samples = []
    for i in range(n):
        # Generate varied risk profiles
        risk_level = np.random.choice(['low', 'medium', 'high'], p=[0.4, 0.4, 0.2])
        
        if risk_level == 'low':
            base_revenue = np.random.uniform(500000, 2000000)
            credit_score = np.random.randint(700, 850)
            profit_margin = np.random.uniform(0.15, 0.35)
            debt_ratio = np.random.uniform(0.1, 0.3)
        elif risk_level == 'medium':
            base_revenue = np.random.uniform(200000, 800000)
            credit_score = np.random.randint(600, 720)
            profit_margin = np.random.uniform(0.05, 0.18)
            debt_ratio = np.random.uniform(0.25, 0.5)
        else:
            base_revenue = np.random.uniform(50000, 400000)
            credit_score = np.random.randint(450, 620)
            profit_margin = np.random.uniform(-0.1, 0.1)
            debt_ratio = np.random.uniform(0.4, 0.8)
        
        sample = {
            'business_id': f'BUS_{i+1:04d}',
            'business_name': f'Sample Business {i+1}',
            'industry_sector': np.random.choice(list(INDUSTRY_MAPPING.keys())),
            'region_parish': np.random.choice(list(PARISH_MAPPING.keys())),
            
            # Financial
            'annual_revenue': base_revenue,
            'revenue_growth_yoy': np.random.uniform(-0.2, 0.3),
            'revenue_volatility': np.random.uniform(0.05, 0.25),
            'profit_margin': profit_margin,
            'gross_margin': profit_margin + np.random.uniform(0.1, 0.2),
            'operating_margin': profit_margin + np.random.uniform(0.02, 0.08),
            
            # Cash
            'avg_bank_balance': base_revenue * np.random.uniform(0.05, 0.15),
            'min_bank_balance_last_12m': base_revenue * np.random.uniform(0.01, 0.05),
            'bank_balance_volatility': np.random.uniform(0.1, 0.4),
            'cash_inflow_monthly_avg': base_revenue / 12 * np.random.uniform(0.8, 1.2),
            'cash_outflow_monthly_avg': base_revenue / 12 * np.random.uniform(0.7, 1.1),
            'cash_to_monthly_expense_ratio': np.random.uniform(0.5, 3.0),
            'net_cash_flow_trend': np.random.uniform(-0.1, 0.2),
            
            # Debt
            'total_debt': base_revenue * debt_ratio,
            'debt_to_revenue': debt_ratio,
            'debt_service_coverage_ratio': np.random.uniform(0.8, 3.0),
            'credit_utilization_ratio': np.random.uniform(0.1, 0.9),
            'loan_payment_to_revenue': np.random.uniform(0.02, 0.15),
            
            # Workforce
            'employee_count': np.random.randint(2, 50),
            'payroll_to_revenue': np.random.uniform(0.2, 0.5),
            'avg_wage_per_employee': np.random.uniform(30000, 70000),
            'employee_growth_rate': np.random.uniform(-0.1, 0.2),
            'payroll_consistency_score': np.random.uniform(0.7, 1.0),
            
            # Credit
            'credit_score_business': credit_score,
            'credit_score_owner': credit_score + np.random.randint(-50, 50),
            'years_in_business': np.random.randint(1, 25),
            'days_past_due_max_last_12m': np.random.choice([0, 0, 0, 30, 60, 90]),
            'num_late_payments_last_12m': np.random.choice([0, 0, 1, 2, 3, 5]),
            'num_returned_payments_last_12m': np.random.choice([0, 0, 0, 1, 2]),
            'has_tax_lien': np.random.choice([0, 0, 0, 0, 1]),
            'bankruptcy_history': np.random.choice([0, 0, 0, 0, 0, 1]),
            
            # Operational
            'missed_payroll_count_last_12m': np.random.choice([0, 0, 0, 1, 2]),
            'overdraft_count_last_12m': np.random.choice([0, 0, 1, 2, 3, 5]),
            'overdraft_days_last_12m': np.random.choice([0, 0, 0, 5, 10, 20]),
            'utilization_rate_card': np.random.uniform(0.1, 0.9),
            
            # IGS benchmarks
            'igs_sales_index': np.random.uniform(90, 110),
            'igs_employment_index': np.random.uniform(95, 105),
            'igs_small_business_index': np.random.uniform(92, 108),
        }
        
        samples.append(sample)
    
    return pd.DataFrame(samples)


# ═══════════════════════════════════════════════════════════════════════════════
#                              MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Business Risk Prediction')
    parser.add_argument('--input', '-i', type=str, help='Input CSV file with business data')
    parser.add_argument('--output', '-o', type=str, default='predictions.csv', 
                        help='Output CSV file for predictions')
    parser.add_argument('--samples', '-n', type=int, default=10,
                        help='Number of sample businesses to generate (if no input)')
    args = parser.parse_args()
    
    print("\n" + "═" * 70)
    print("🎯 BUSINESS RISK PREDICTION SYSTEM")
    print("═" * 70)
    
    # Load models
    print("\n📦 Loading trained models...")
    try:
        classifier, regressor, encoders = load_models()
        print("   ✅ Models loaded successfully!")
    except FileNotFoundError as e:
        print(f"   ❌ Error: {e}")
        print("\n   Creating sample output without models...")
        
        # Generate sample data and fake predictions for demonstration
        df = generate_sample_businesses(args.samples)
        df['default_probability'] = np.random.uniform(0.05, 0.6, len(df))
        df['default_flag'] = (df['default_probability'] > 0.5).astype(int)
        df['risk_score'] = df['default_probability'] * 100 + np.random.uniform(-10, 10, len(df))
        df['risk_score'] = df['risk_score'].clip(0, 100)
        df['risk_band'] = pd.cut(df['risk_score'], bins=[-1, 40, 75, 100], labels=['Low', 'Medium', 'High'])
        
        output_path = os.path.join(PROJECT_ROOT, args.output)
        df.to_csv(output_path, index=False)
        print(f"\n📄 Sample output saved to: {output_path}")
        return
    
    # Load or generate input data
    if args.input:
        print(f"\n📂 Loading input data from: {args.input}")
        df = pd.read_csv(args.input)
    else:
        print(f"\n🔧 Generating {args.samples} sample businesses...")
        df = generate_sample_businesses(args.samples)
    
    print(f"   📊 Records to process: {len(df)}")
    
    # Preprocess data
    print("\n⚙️ Preprocessing data...")
    X = preprocess_data(df, encoders)
    
    # Make predictions
    print("🤖 Making predictions...")
    predictions = predict(X, classifier, regressor)
    
    # Add predictions to dataframe
    df['default_probability'] = predictions['default_probability']
    df['default_flag'] = predictions['default_flag']
    df['risk_score'] = predictions['risk_score']
    df['risk_band'] = predictions['risk_band']
    
    # Save results
    output_path = os.path.join(PROJECT_ROOT, args.output)
    df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "─" * 70)
    print("📊 PREDICTION SUMMARY")
    print("─" * 70)
    
    print(f"\n   Total Businesses: {len(df)}")
    print(f"\n   Risk Distribution:")
    for band in ['Low', 'Medium', 'High']:
        count = (df['risk_band'] == band).sum()
        pct = count / len(df) * 100
        emoji = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}[band]
        print(f"      {emoji} {band}: {count} ({pct:.1f}%)")
    
    print(f"\n   Default Predictions:")
    print(f"      ✅ No Default: {(df['default_flag'] == 0).sum()}")
    print(f"      ❌ Default:    {(df['default_flag'] == 1).sum()}")
    
    print(f"\n   Average Risk Score: {df['risk_score'].mean():.1f}")
    
    print(f"\n💾 Predictions saved to: {output_path}")
    
    # Show sample predictions
    print("\n" + "─" * 70)
    print("📋 SAMPLE PREDICTIONS (First 5)")
    print("─" * 70)
    
    display_cols = ['business_id', 'industry_sector', 'risk_score', 'risk_band', 
                    'default_probability', 'default_flag']
    display_cols = [c for c in display_cols if c in df.columns]
    
    for idx, row in df.head(5).iterrows():
        print(f"\n   {row.get('business_id', f'Record {idx}')}:")
        print(f"      Industry: {row.get('industry_sector', 'N/A')}")
        print(f"      Risk Score: {row['risk_score']:.1f} ({row['risk_band']})")
        print(f"      Default Probability: {row['default_probability']:.1%}")
        print(f"      Default Flag: {'⚠️ YES' if row['default_flag'] == 1 else '✅ NO'}")
    
    print("\n" + "═" * 70)
    print("✅ PREDICTION COMPLETE!")
    print("═" * 70 + "\n")


if __name__ == '__main__':
    main()

