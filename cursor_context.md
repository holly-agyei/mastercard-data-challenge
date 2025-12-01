# 🏛️ STATE BUSINESS RISK SCORING PLATFORM - COMPLETE CONTEXT

## **PROJECT OVERVIEW**

We are building a **comprehensive web-based platform** for a US state (Louisiana, specifically Ouachita Parish) to assess the financial health and default risk of small businesses. The system combines **machine learning predictions**, **real-time financial data analysis**, and **state-level economic benchmarking** to help allocate loans, grants, and economic development resources.

---

## **THE BIG PICTURE**

```
┌─────────────────────────────────────────────────────────────────┐
│                     STATE DASHBOARD (Web App)                   │
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                 │
│  │  Business Lookup │    │  Risk Assessment │                 │
│  │  & Registration  │    │  & Scoring       │                 │
│  └──────────────────┘    └──────────────────┘                 │
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                 │
│  │  Grant/Loan      │    │  Economic        │                 │
│  │  Recommendation  │    │  Trends & Data   │                 │
│  └──────────────────┘    └──────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
              ▲
              │ (Sends business data files)
              │
┌─────────────────────────────────────────────────────────────────┐
│              BACKEND PROCESSING ENGINE (Python)                 │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  FILE INGESTION LAYER                                    │  │
│  │  • checking_account.csv (Bank statements)               │  │
│  │  • credit_card_account.csv (Card transactions)          │  │
│  │  • gusto_payroll.csv (Payroll & employees)             │  │
│  │  • Loan.csv / SBA_loan_dataset.csv (Debt data)         │  │
│  │  • Credit_Account_History.csv (Payment history)        │  │
│  └──────────────────────────────────────────────────────────┘  │
│           ▼                                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  FEATURE ENGINEERING LAYER (Calculations)               │  │
│  │  • Revenue trends, growth rates                         │  │
│  │  • Profitability & margins                              │  │
│  │  • Cash flow & liquidity analysis                       │  │
│  │  • Debt burden & leverage ratios                        │  │
│  │  • Workforce stability metrics                          │  │
│  │  • Credit compliance scores                             │  │
│  │  • Economic benchmark comparisons                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│           ▼                                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ML PREDICTION LAYER (XGBoost Models)                   │  │
│  │  • Model 1: default_flag (0/1 binary)                  │  │
│  │  • Model 2: risk_score (0-100 continuous)             │  │
│  └──────────────────────────────────────────────────────────┘  │
│           ▼                                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  STATE METRICS CALCULATION LAYER (Post-Prediction)      │  │
│  │  • Risk band assignment (Low/Medium/High)              │  │
│  │  • Revenue health index (0-100)                        │  │
│  │  • Profitability index (0-100)                         │  │
│  │  • Liquidity index (0-100)                             │  │
│  │  • Debt burden index (0-100)                           │  │
│  │  • Workforce stability index (0-100)                   │  │
│  │  • Credit compliance index (0-100)                     │  │
│  │  • Boolean flags for special conditions                │  │
│  └──────────────────────────────────────────────────────────┘  │
│           ▼                                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  LLM NARRATIVE LAYER (Claude/GPT)                       │  │
│  │  • Generates human-readable explanations               │  │
│  │  • Pillar-based narrative (Financial, Operational...)  │  │
│  │  • Explains risk drivers (top 3-5 factors)            │  │
│  │  • Suggests interventions & support programs          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
              ▲
              │ (Returns complete risk assessment)
              │
         [Back to Dashboard]
```

---

## **WHAT THE SYSTEM DOES (End-to-End Flow)**

### **Step 1: Business Registration**
- State program officer or business owner logs into web dashboard
- Enters basic business info: Name, Industry, Location, Years in Operation
- Uploads financial documents (CSV files from bank, payroll, loan accounts)

### **Step 2: Data Ingestion & Parsing**
- Backend system reads uploaded files:
  - Bank statements (checking, savings accounts)
  - Credit card transaction history
  - Payroll records (Gusto exports)
  - Loan accounts (SBA, commercial)
  - Payment history & credit account data
- Validates file formats and completeness
- Flags missing data but proceeds with available info

### **Step 3: Feature Engineering (Layer 2)**
Python scripts calculate 40+ business health metrics:

**Financial Metrics:**
- Monthly revenue trends & growth rate
- Profit margins & cash flow analysis
- Cash reserves (months of runway)
- Debt-to-revenue ratio
- Credit utilization rate

**Operational Metrics:**
- Number of employees & payroll trends
- Missed payroll incidents
- Industry classification

**Credit & Compliance:**
- Business credit score
- Payment history (late days, returned payments)
- Tax liens or bankruptcies
- Overdraft history

**Economic Context (Mastercard IGS Data):**
- Tract-level business growth metrics
- Personal income trends
- Labor market engagement
- Business diversity index
- Minority/women-owned business percentage
- How this business compares to peers in same area

### **Step 4: ML Model Predictions (Layer 1)**

**Two Trained XGBoost Models:**

1. **Default Classifier**
   - Input: 43 business features
   - Output: `default_flag` (0 or 1) + `default_probability` (e.g., 0.85)
   - Answers: "Will this business default in next 12 months?"
   - Decision: 0 = Likely safe, 1 = At-risk

2. **Risk Regressor**
   - Input: 43 business features
   - Output: `risk_score` (0-100 integer)
   - Answers: "What is the severity of risk on a 0-100 scale?"
   - Scale:
     - 0-30: Very Low Risk (Safe)
     - 31-50: Low-Medium Risk
     - 51-75: Medium-High Risk
     - 76-100: Critical Risk

### **Step 5: State Metrics Calculation (Layer 3)**

After model predicts `risk_score`, Python immediately calculates:

**Pillar Indices** (0-100 scores for dashboard display):
1. `revenue_health_index` - Is revenue growing & stable?
2. `profitability_index` - Is business profitable?
3. `liquidity_index` - Does business have cash reserves?
4. `debt_burden_index` - Is debt manageable?
5. `workforce_stability_index` - Is employment stable?
6. `credit_compliance_index` - Payment history & credit score?

**Risk Band** (Categorical):
- "Low Risk" if risk_score < 40
- "Medium Risk" if 40-75
- "High Risk" if > 75

**Boolean Flags** (Yes/No for special programs):
- `credit_constrained` - High growth but low cash & high card utilization?
- `job_generator_at_risk` - Employs 20+ people AND at high risk?
- `high_growth_low_cash` - Growing revenue but low liquidity?
- `compliant_under_pressure` - Low credit score but paying all obligations?

### **Step 6: LLM Narrative Generation**

Claude/GPT API processes the risk assessment and generates:

**Executive Summary** (2-3 sentences)
- Plain English explanation of business health
- Example: "ABC Plumbing is a 5-year-old construction company with strong revenue growth (+15% YoY) but concerning cash reserves. They have manageable debt levels but missed 2 payroll cycles in the past year."

**Pillar Analysis** (5-7 sentences, one per pillar)
- Detailed breakdown of each health index
- Example: "Liquidity Index (42/100): The business maintains $28K in reserves, providing 1.2 months of operational runway. This is concerning given industry volatility."

**Risk Drivers** (Top 3-5 factors contributing to risk)
- Example:
  1. "Missed 2 payroll cycles (Workforce Stability)"
  2. "Below-average cash reserves for construction sector"
  3. "Debt-to-revenue ratio of 1.85x (above 1.5x threshold)"

**Recommendations**
- Specific state programs or interventions
- Example: "Recommend: Cash Flow Stabilization Loan (6-month terms) + Payroll Reserve Grant"
- Link to state resources

### **Step 7: Dashboard Visualization**

Frontend displays comprehensive report card:

```
┌─────────────────────────────────────────────────────┐
│   ABC Plumbing - Risk Assessment Report             │
│   Business ID: 100042 | Industry: Construction      │
├─────────────────────────────────────────────────────┤
│                                                     │
│   RISK SCORE: 68/100 [MEDIUM-HIGH RISK] ⚠️         │
│   Default Probability: 67%                         │
│                                                     │
│   ┌─ HEALTH INDICES ─────────────────────────────┐ │
│   │ Revenue Health:     ████░░░░░░ 65/100        │ │
│   │ Profitability:      ███░░░░░░░ 42/100        │ │
│   │ Liquidity:          ████░░░░░░ 42/100        │ │
│   │ Debt Burden:        ██░░░░░░░░ 25/100        │ │
│   │ Workforce:          ███░░░░░░░ 35/100        │ │
│   │ Credit Compliance:  █████░░░░░ 65/100        │ │
│   └─────────────────────────────────────────────┘ │
│                                                     │
│   [FULL NARRATIVE] [RECOMMENDATIONS]               │
│   [DOWNLOAD REPORT] [SHARE WITH LENDER]            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## **ARCHITECTURE LAYERS**

### **Layer 1: ML Models (What AI Predicts)**
- **Input:** 43 calculated business features
- **Output:** 2 predictions
  - `default_flag` (binary)
  - `risk_score` (0-100)
- **Technology:** XGBoost (trained on synthetic 15K dataset)
- **Deployment:** Pickle files loaded in Python backend

### **Layer 2: Feature Engineering (Python Calculations)**
- **Input:** Raw CSV files from business
- **Output:** 43 normalized features for model
- **Calculation Type:** Deterministic (same input = same output always)
- **Examples:**
  ```
  profit_margin = (revenue - expenses) / revenue
  cash_runway = avg_bank_balance / (monthly_expenses)
  debt_to_revenue = total_loans / annual_revenue
  missed_payroll_pct = missed_payroll_count / total_payroll_periods
  ```

### **Layer 3: State Metrics (Post-Prediction Calculations)**
- **Input:** Business data + model predictions
- **Output:** 8 pillar indices + risk band + boolean flags
- **Calculation Type:** Rule-based formulas
- **Examples:**
  ```
  risk_band = "Low" if risk_score < 40 else "Medium" if risk_score <= 75 else "High"
  
  revenue_health = (
    normalize(revenue_growth, 0, 1) * 0.5 +
    normalize(profit_margin, 0, 0.25) * 0.3 +
    (1 - normalize(revenue_volatility, 0, 0.3)) * 0.2
  ) * 100
  
  liquidity_index = min(100, (cash_runway / 3) * 100)
  ```

### **Layer 4: LLM Narrative (Claude/GPT)**
- **Input:** Risk assessment results + pillar indices + flags
- **Output:** Human-readable narrative & recommendations
- **Technology:** Claude API or OpenAI GPT-4
- **Purpose:** Explain risk in plain English for non-technical stakeholders

### **Layer 5: Frontend Dashboard (React/Vue.js)**
- **Input:** JSON from backend
- **Output:** Interactive visualizations, charts, downloadable reports
- **Components:**
  - Business lookup & search
  - Risk score visualization (gauge, color-coded)
  - Pillar index charts (bar graphs)
  - Narrative display (formatted text)
  - Action buttons (approve loan, refer to grant program, etc.)

---

## **DATA FLOW SUMMARY**

```
Raw Business Files          Feature Engineering           ML Model
     (CSV)                  (Python Layer 2)            (XGBoost)
       ▼                           ▼                         ▼
[Bank Statements]    →  [Revenue Trends]      →  [43 Features]  →  [Model 1]
[Payroll Data]       →  [Profit Margin]       →       │         →  [Model 2]
[Credit History]     →  [Cash Runway]         →       │         ▼    ▼
[Loan Accounts]      →  [Debt Ratio]          →       │      [default_flag]
                        [Credit Scores]       →       │      [risk_score]
                        [Workforce Metrics]   →       │
                        [IGS Benchmarks]      →       │

         ▼
    State Metrics Layer 3 (Python)
    ▼
[risk_band, indices, flags]
         ▼
    LLM Narrative Layer 4 (Claude)
    ▼
[Executive Summary, Pillar Analysis, Risk Drivers, Recommendations]
         ▼
    Frontend Dashboard Layer 5 (React/Vue)
    ▼
[Interactive Report Card with visualizations]
```

---

## **KEY DESIGN PRINCIPLES**

### **1. Separation of Concerns**
- **AI (Layer 1):** Only predicts default_flag & risk_score
- **Python (Layer 2):** Calculates input features
- **Python (Layer 3):** Transforms predictions into display metrics
- **LLM (Layer 4):** Explains results in human language
- **Frontend (Layer 5):** Visualizes everything

### **2. Explainability**
- No business gets a risk score without explanation
- Every score backed by 5-7 pillar indices
- Top 3-5 risk drivers explained in narrative
- Historical context provided (vs. tract benchmarks)

### **3. Actionability**
- Scores directly map to state programs
- Risk band determines intervention type
- Boolean flags trigger specific resources
- Recommendations include links to support

### **4. Reproducibility**
- Same business data = same assessment always
- Feature calculations are deterministic
- Model is frozen (not retraining continuously)
- Audit trail of all inputs & outputs

---

## **BUSINESS CONTEXT: WHY THIS MATTERS**

**Problem the State is Solving:**
- Ouachita Parish (Louisiana) has limited economic development resources
- Small businesses need credit but traditional banks see them as too risky
- State wants to deploy **targeted** loans/grants to high-impact businesses
- Currently: Manual assessment (slow, inconsistent, subjective)

**Solution:**
- Data-driven scoring (fast, consistent, objective)
- Automated identification of businesses worth supporting
- Clear explanation to business owners WHY they're getting what
- Transparency in state spending decisions

**Business Impact:**
- **For Businesses:** Know their financial health, get access to capital
- **For State:** Deploy funds efficiently, maximize ROI on economic development
- **For Lenders:** Partner with state-backed guarantee programs
- **For Community:** More jobs, more business activity

---

## **TECHNOLOGIES INVOLVED**

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Input** | Python (pandas) | CSV parsing & validation |
| **Feature Engineering** | Python (numpy, pandas, scikit-learn) | Metrics calculation |
| **ML Model Training** | Python (xgboost, sklearn) | Classifier & Regressor |
| **State Metrics** | Python (custom functions) | Rule-based calculations |
| **LLM Integration** | Claude API / OpenAI GPT-4 | Narrative generation |
| **Backend API** | Flask or FastAPI | REST endpoints |
| **Frontend** | React / Vue.js | Interactive dashboard |
| **Database** | PostgreSQL / MongoDB | Store business profiles & assessments |
| **Deployment** | Docker, AWS / Azure | Production hosting |

---

## **FILE STRUCTURE (What Will Be Built)**

```
project-root/
├── backend/
│   ├── ml_models/
│   │   ├── xgboost_classifier.pkl        (Trained model)
│   │   ├── xgboost_regressor.pkl         (Trained model)
│   │   └── feature_names.pkl             (Feature list)
│   │
│   ├── data_processing/
│   │   ├── data_loader.py                (Load & parse CSV)
│   │   ├── feature_preprocessor.py       (Feature engineering)
│   │   └── feature_names_mapping.py      (Feature metadata)
│   │
│   ├── ml_pipeline/
│   │   ├── xgboost_classifier.py         (Classifier training)
│   │   ├── xgboost_regressor.py          (Regressor training)
│   │   └── model_validator.py            (CV & testing)
│   │
│   ├── prediction/
│   │   ├── predictor.py                  (Run inference)
│   │   ├── state_metrics_calculator.py   (Post-prediction calcs)
│   │   └── llm_narrator.py               (Claude/GPT integration)
│   │
│   ├── api/
│   │   ├── app.py                        (Flask/FastAPI app)
│   │   ├── routes.py                     (API endpoints)
│   │   └── middleware.py                 (Auth, logging)
│   │
│   └── utils/
│       ├── constants.py                  (Config values)
│       ├── logger.py                     (Logging setup)
│       └── validators.py                 (Input validation)
│
├── frontend/
│   ├── components/
│   │   ├── RiskCard.jsx                  (Risk score display)
│   │   ├── PillarIndices.jsx             (Health indices charts)
│   │   ├── Narrative.jsx                 (LLM narrative display)
│   │   └── RecommendationPanel.jsx       (Program suggestions)
│   │
│   ├── pages/
│   │   ├── BusinessLookup.jsx            (Search & register)
│   │   ├── RiskAssessment.jsx            (Results display)
│   │   └── Dashboard.jsx                 (Overview)
│   │
│   └── services/
│       └── api.js                        (Backend API calls)
│
├── data/
│   ├── synthetic_xgboost_training_data.csv
│   └── training_results/
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
└── README.md
```

---

## **WHAT WE'RE NOT BUILDING (Out of Scope)**

- ❌ Real-time transaction monitoring (passive assessment only)
- ❌ Loan origination system (integration point, not full system)
- ❌ Multi-currency support (US businesses only)
- ❌ Mobile app (web-only for now)
- ❌ Custom ML model training interface (frozen models)
- ❌ Advanced fraud detection (basic validation only)

---

## **SUMMARY FOR CURSOR AI**

This project is a **complete data pipeline + ML system + web application** designed to:

1. **Ingest** business financial data (bank statements, payroll, loans)
2. **Calculate** 40+ health metrics using Python
3. **Predict** default risk & risk severity using XGBoost
4. **Transform** predictions into 8 pillar indices + flags
5. **Generate** natural language explanation using Claude/GPT
6. **Display** everything on an interactive web dashboard
7. **Enable** state program officers to make funding decisions

**The core intelligence is the ML model** (predicts 2 things: default_flag & risk_score), but it's surrounded by **feature engineering, state metrics, LLM narrative, and a professional dashboard** to make it useful for real stakeholders.

**Nothing is simple. Everything is intentional. Every number has a story. Every story informs a decision.**
