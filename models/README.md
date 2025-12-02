# XGBoost Risk Models

This document explains the two XGBoost models that power the State Business Risk
Scoring Platform. Both models are trained in
`notebooks/xgboost_model_training.ipynb` using the synthetic training set at
`datasets/synthetic_xgboost_training_data.csv` (15,000 businesses).

---

## Targets

| Model | Target Column | Meaning |
|-------|---------------|---------|
| **Classifier** | `default_flag` | Binary label that indicates whether the business is expected to default within the next 12 months (`1`) or remain stable (`0`). |
| **Regressor** | `risk_score` | Continuous score from 0–100 that measures the severity of financial risk (0 = very low risk, 100 = extreme risk). |

Both targets are provided in the dataset and come from the “STATE” layer of the
platform (see `datasets/features.csv` for the data dictionary).

---

## Feature Groups

The models use 40 engineered features that summarize the financial health of a
business. The columns originate from three sources that are blended in the
synthetic dataset:

1. **Business financials (bank, payroll, and accounting data)**
   - Revenue level, growth, and profitability (`annual_revenue`,
     `revenue_growth_yoy`, `profit_margin`)
   - Cash position and liquidity (`avg_bank_balance`,
     `cash_to_monthly_expense_ratio`, `overdraft_days_last_12m`)
   - Loan exposure (`loan_amount_outstanding`, `debt_to_revenue`,
     `interest_rate_weighted_avg`, `term_months_remaining`,
     `days_past_due_max_last_12m`, `num_active_loans`)
   - Workforce and payroll stability (`num_employees`,
     `total_payroll_last_12m`, `missed_payroll_count_last_12m`,
     `payroll_growth_yoy`)

2. **Credit bureau style signals**
   - Business credit score and compliance flags (`credit_score_business`,
     `num_returned_payments_last_12m`, `has_tax_lien`,
     `prior_default_or_bankruptcy`)
   - Card utilization and payment behavior (`avg_monthly_card_spend`,
     `card_spend_growth_yoy`, `utilization_rate_card`,
     `days_late_card_payment_max_last_12m`)

3. **Mastercard IGS (Inclusive Growth Score) tract-level benchmarks**
   - Local spending and lending growth (`igs_spend_growth_tract`,
     `igs_new_businesses_growth_tract`,
     `igs_small_business_loans_growth_tract`)
   - Household context (`igs_personal_income_growth_tract`,
     `igs_spending_per_capita_growth_tract`, `igs_median_income_tract`,
     `igs_tract_population`)
   - Community health indices (`igs_labor_market_engagement_index`,
     `igs_minority_women_owned_pct`, `igs_commercial_diversity_pct`,
     `labor_market_health_score`)
   - Relative performance deltas (`revenue_growth_vs_tract_benchmark`,
     `spending_growth_vs_tract_benchmark`)

Categorical attributes (`industry_sector`, `region_parish`) are encoded with
`LabelEncoder` objects that are persisted in `models/label_encoders.pkl`.

---

## Training Pipeline (Notebook)

1. **Environment setup** – imports, plotting defaults, and reproducible random
   seed.
2. **Data load** – reads `datasets/synthetic_xgboost_training_data.csv`, reports
   shape, memory footprint, and column categories.
3. **Exploratory analysis** – summaries of targets, risk bands, industry and
   parish distributions, financial feature visualizations, and correlation
   checks.
4. **Preprocessing**
   - Split ID, target, categorical, and numerical features.
   - Median imputation for missing numeric values (rare in the synthetic data).
   - Label encoding for `industry_sector` and `region_parish`.
   - Train/test split (80/20) with stratification on `default_flag`.
5. **Model training**
   - **Classifier**: `XGBClassifier` (300 trees, depth 6, learning rate 0.1,
     scale_pos_weight to address class imbalance). Evaluated with stratified
     5-fold ROC-AUC and F1, then fitted on the full training set.
   - **Regressor**: `XGBRegressor` (mirrors classifier hyperparameters, uses
     RMSE objective). Evaluated with 5-fold R² and MAE.
6. **Evaluation** – confusion matrix, ROC, PR curves for classifier; scatter and
   residual plots for regressor; feature importance charts.
7. **Artifacts** – saves models, feature names, encoders, and metadata to the
   `models/` directory. Generates a sample submission (50 in-sample businesses)
   under `submissions/sample_submission.csv`.

---

## Model Metrics (Test Set)

| Metric | Classifier (`default_flag`) | Regressor (`risk_score`) |
|--------|-----------------------------|--------------------------|
| Cross-val | ROC-AUC 0.990 ± 0.001 | R² 0.961 ± 0.002 |
| Test ROC-AUC | 0.9925 | — |
| Test F1 | 0.9267 | — |
| Test Precision | 0.9173 | — |
| Test Recall | 0.9363 | — |
| Test MAE | — | 3.63 |
| Test RMSE | — | 5.87 |
| Test R² | — | 0.965 |

See `models/model_metadata.json` for the full set of metrics and
hyperparameters.

---

## Saved Artifacts

| File | Description |
|------|-------------|
| `models/xgboost_classifier.pkl` | Trained default classifier |
| `models/xgboost_regressor.pkl` | Trained risk-score regressor |
| `models/feature_names.pkl` | Ordered feature list used by both models |
| `models/label_encoders.pkl` | LabelEncoder objects for categorical fields |
| `models/model_metadata.json` | Training summary, hyperparameters, metrics |
| `submissions/sample_submission.csv` | Example predictions on 50 sampled businesses |

---

## How to Reproduce / Answer Questions

1. **Retrain or inspect** – open `notebooks/xgboost_model_training.ipynb`,
   select the `mastercard_env` kernel, and execute all cells.
2. **Explain predictions** – refer to the EDA and feature-importance plots in
   the notebook; they show which drivers correlate most with each target.
3. **Generate submissions** – run the “Generate Sample Submission File” section
   (Cells 48–51) to produce fresh predictions on a sampled slice of the dataset.
4. **Serve predictions** – load both models plus `feature_names.pkl` and
   `label_encoders.pkl`, prepare features in the saved order, and call
   `predict()` (see `predict.py` example).

With this information you can describe the models, data provenance, selected
features, evaluation results, and deployment artifacts to stakeholders without
reading the full notebook.


