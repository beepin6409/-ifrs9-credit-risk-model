````markdown
# IFRS 9 Expected Credit Loss (ECL) Model

![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

An end-to-end implementation of an IFRS 9-compliant 
Expected Credit Loss (ECL) model for retail credit 
portfolios, built on the Lending Club loan dataset.

Under IFRS 9 (International Financial Reporting 
Standard 9), banks and financial institutions must 
estimate credit losses using a forward-looking 
three-stage classification framework. This project 
implements the Probability of Default (PD) component 
of ECL using machine learning, with full model 
explainability for regulatory compliance.

---

## Business Context

### What is IFRS 9?
IFRS 9 replaced IAS 39 in 2018 and fundamentally 
changed how banks provision for credit losses. 
Instead of recognizing losses only when they occur 
(incurred loss model), banks must now estimate 
**Expected Credit Losses** proactively.

### The Three-Stage Model

| Stage | Definition | ECL Measurement |
|-------|-----------|-----------------|
| Stage 1 | Performing loans — no significant increase in credit risk since origination | 12-month ECL |
| Stage 2 | Significant increase in credit risk since origination | Lifetime ECL |
| Stage 3 | Credit-impaired loans (defaulted) | Lifetime ECL |

### ECL Formula

```
ECL = PD × LGD × EAD × Discount Factor
```

- **PD** — Probability of Default (modeled in this project)
- **LGD** — Loss Given Default
- **EAD** — Exposure at Default

This project focuses on **PD modeling** — the 
foundation of the entire ECL calculation.

---

## Why Machine Learning for PD?

Traditional PD models use logistic regression with 
scorecards. Modern risk teams combine statistical 
baselines with gradient boosting for improved 
discriminatory power, while maintaining 
explainability through SHAP values — a regulatory 
requirement under ECB model risk guidelines.

---

## Dataset

**Source:** Lending Club Loan Data (2007–2018)
**Access:** [Kaggle — Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
**Size:** ~2.2 million loan records
**Target Variable:** `loan_status` → binary
(Default = 1, Non-Default = 0)

### Key Features Used

| Feature | Description |
|---------|-------------|
| `loan_amnt` | Loan amount requested |
| `int_rate` | Interest rate on the loan |
| `grade` / `sub_grade` | Credit grade assigned by Lending Club |
| `dti` | Debt-to-income ratio |
| `annual_inc` | Annual income of borrower |
| `emp_length` | Employment length in years |
| `home_ownership` | Housing status (rent/own/mortgage) |
| `purpose` | Purpose of the loan |
| `fico_range_low/high` | FICO credit score range |
| `open_acc` | Number of open credit lines |
| `revol_util` | Revolving line utilization rate |
| `delinq_2yrs` | Delinquencies in past 2 years |

---

## Project Structure

```
ifrs9-credit-risk-model/
│
├── data/
│   ├── raw/                    # Original dataset
│   │                             (not tracked in git)
│   └── processed/              # Cleaned and
│                                 engineered features
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data
│   │                             analysis
│   ├── 02_feature_engineering  # Feature creation
│   │   .ipynb                    and selection
│   ├── 03_credit_migration     # Markov chain
│   │   .ipynb                    transition matrix
│   ├── 04_pd_model.ipynb       # PD model training
│   │                             and evaluation
│   └── 05_shap_explainability  # SHAP analysis and
│       .ipynb                    regulatory reporting
│
├── src/
│   ├── preprocess.py           # Data cleaning
│   │                             pipeline
│   ├── features.py             # Feature engineering
│   ├── model.py                # Model training
│   │                             and evaluation
│   ├── evaluate.py             # Performance metrics
│   └── explain.py              # SHAP explainability
│
├── outputs/
│   ├── figures/                # All plots and
│   │                             visualizations
│   └── models/                 # Saved model files
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Methodology

### Step 1 — Exploratory Data Analysis
- Default rate distribution across loan grades
- Feature correlation analysis
- Missing value assessment
- Class imbalance analysis
  (defaults are minority class)

### Step 2 — Feature Engineering
- Weight of Evidence (WoE) encoding for
  categorical variables — industry standard
  in credit scoring
- Information Value (IV) for feature selection
- Missing value imputation using
  domain-informed strategies

### Step 3 — Credit Rating Migration Matrix

Using Markov chains to model borrower movement
between credit quality states over time:

```python
import numpy as np

# States: A, B, C, D, E, F, G, Default
# Transition matrix represents 1-year migration
# probabilities between Lending Club grades

transition_matrix = np.array([...])

# Multi-year PD via matrix exponentiation
pd_2yr = np.linalg.matrix_power(
    transition_matrix, 2)[:, -1]

pd_5yr = np.linalg.matrix_power(
    transition_matrix, 5)[:, -1]
```

### Step 4 — PD Modeling

**Baseline:** Logistic Regression (scorecard approach)
- Industry standard, fully interpretable
- Coefficients directly interpretable
  as risk weights

**Primary Model:** XGBoost Classifier
- Gradient boosting for improved discrimination
- Handles non-linear relationships in credit data
- Requires SHAP for regulatory explainability

### Evaluation Metrics (industry standard)

| Metric | Description |
|--------|-------------|
| AUC-ROC | Overall discrimination power |
| Gini Coefficient | 2 × AUC − 1, standard in credit risk |
| KS Statistic | Max separation between default/non-default distributions |
| PSI | Population Stability Index — model drift monitoring |

### Step 5 — SHAP Explainability

Regulatory compliance under ECB guidelines requires
model explainability. SHAP (SHapley Additive
exPlanations) provides:

- **Global explanation:** Which features drive
  default risk across the portfolio
- **Local explanation:** Why a specific borrower
  was classified as high risk
- **Regulatory documentation:** Evidence that
  model decisions are interpretable and auditable

```python
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Summary plot — global feature importance
shap.summary_plot(shap_values, X_test)

# Force plot — individual loan explanation
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0]
)
```

---

## Results

*(To be updated upon project completion — April 2026)*

| Model | AUC-ROC | Gini | KS Statistic |
|-------|---------|------|--------------|
| Logistic Regression | — | — | — |
| XGBoost | — | — | — |

---

## Regulatory Context

This project is designed with the following
regulatory frameworks in mind:

| Framework | Relevance |
|-----------|-----------|
| IFRS 9 | ECL methodology and stage classification |
| Basel IV | Credit risk capital requirements |
| ECB Model Risk Guidelines | Explainability and model validation requirements |
| BaFin | German Federal Financial Supervisory Authority governance expectations |

---

## Technical Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Pandas / NumPy | Data manipulation |
| scikit-learn | Logistic regression, preprocessing, evaluation |
| XGBoost | Gradient boosting PD model |
| SHAP | Model explainability |
| Matplotlib / Seaborn | Visualization |
| SQL (SQLite) | Data querying and validation |
| Git | Version control |

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/beepin6409/ifrs9-credit-risk-model

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
# Create a free Kaggle account and download:
# https://www.kaggle.com/datasets/wordsforthewise/lending-club
# Place the CSV file in data/raw/

# 4. Run notebooks in order
jupyter notebook notebooks/01_eda.ipynb
```

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
jupyter
scorecardpy
imbalanced-learn
```

---

## Key Learning Connections

This project directly applies coursework from
the MSc Data Science program at University of Potsdam:

| Course | Application in This Project |
|--------|---------------------------|
| Foundations of Stochastics | Markov chain credit migration matrix |
| Machine Learning | XGBoost, logistic regression, model evaluation |
| Bayesian Inference | Uncertainty quantification in PD estimates (planned) |

---

## Project Status

- [x] Repository setup and documentation
- [ ] Exploratory data analysis
- [ ] Feature engineering (WoE/IV)
- [ ] Credit migration matrix (Markov chain)
- [ ] Logistic regression baseline
- [ ] XGBoost PD model
- [ ] SHAP explainability
- [ ] Results documentation
- [ ] Clean src/ scripts

---

## Author

**Bipin Ghimire**
MSc Data Science, University of Potsdam, Germany
[LinkedIn](https://www.linkedin.com/in/beepin6409/) |
[GitHub](https://github.com/beepin6409)
bipin.bpn8888@gmail.com

---

## References

- IFRS Foundation. (2014). *IFRS 9 Financial Instruments*
- Basel Committee on Banking Supervision. *Basel IV Framework*
- Lundberg, S. & Lee, S.I. (2017). *A Unified Approach
  to Interpreting Model Predictions* (SHAP original paper)
- Hull, J. (2022). *Options, Futures, and Other
  Derivatives* — Credit risk chapters
- Hand, D. & Henley, W. (1997). *Statistical
  Classification Methods in Consumer Credit Scoring*

---

## License

MIT License — free to use with attribution
````

---
