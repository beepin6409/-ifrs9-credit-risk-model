

---

```markdown
# IFRS 9 Expected Credit Loss (ECL) Model

<p align="center">

![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

# Overview

This project implements an **IFRS 9–compliant Expected Credit Loss (ECL) model** for retail credit portfolios using the **Lending Club loan dataset**.

Under **IFRS 9 (International Financial Reporting Standard 9)**, financial institutions must estimate credit losses using a **forward-looking three-stage framework**.

This repository focuses on building the **Probability of Default (PD)** component of the ECL calculation using **machine learning models with regulatory explainability**.

---

# Business Context

## What is IFRS 9?

**IFRS 9** replaced **IAS 39** in 2018 and fundamentally changed how banks provision for credit losses.

Instead of recognizing losses **only when they occur**, institutions must now estimate **Expected Credit Losses (ECL)** using forward-looking information.

---

## The Three-Stage Model

| Stage | Definition | ECL Measurement |
|------|------------|----------------|
| Stage 1 | Performing loans with no significant increase in credit risk | 12-month ECL |
| Stage 2 | Significant increase in credit risk since origination | Lifetime ECL |
| Stage 3 | Credit-impaired loans (defaulted) | Lifetime ECL |

---

## ECL Formula

```

ECL = PD × LGD × EAD × Discount Factor

```

Where:

- **PD** — Probability of Default  
- **LGD** — Loss Given Default  
- **EAD** — Exposure at Default  

This project focuses on **PD modeling**, the **core driver** of the ECL calculation.

---

# Why Machine Learning for PD?

Traditional credit risk models rely on **logistic regression scorecards**.

Modern risk analytics teams combine:

- statistical baseline models
- gradient boosting algorithms
- explainability frameworks

to improve predictive performance while maintaining **regulatory transparency**.

This project uses:

- **Logistic Regression** → interpretable baseline  
- **XGBoost** → high-performance PD model  
- **SHAP values** → regulatory explainability

---

# Dataset

**Source:** Lending Club Loan Data (2007–2018)  
**Access:** https://www.kaggle.com/datasets/wordsforthewise/lending-club

**Size:** ~2.2 million loan records

**Target Variable**

```

loan_status

```

Binary mapping:

```

Default     → 1
Non-default → 0

```

---

## Key Features

| Feature | Description |
|-------|-------------|
| loan_amnt | Loan amount requested |
| int_rate | Interest rate |
| grade / sub_grade | Credit grade assigned by Lending Club |
| dti | Debt-to-income ratio |
| annual_inc | Borrower annual income |
| emp_length | Employment length |
| home_ownership | Housing status |
| purpose | Loan purpose |
| fico_range_low / high | FICO credit score range |
| open_acc | Number of open credit lines |
| revol_util | Revolving utilization rate |
| delinq_2yrs | Delinquencies in last 2 years |

---

# Project Structure

```

ifrs9-credit-risk-model
│
├── data
│   ├── raw
│   │   └── lending_club.csv
│   └── processed
│
├── notebooks
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_credit_migration.ipynb
│   ├── 04_pd_model.ipynb
│   └── 05_shap_explainability.ipynb
│
├── src
│   ├── preprocess.py
│   ├── features.py
│   ├── model.py
│   ├── evaluate.py
│   └── explain.py
│
├── outputs
│   ├── figures
│   └── models
│
├── requirements.txt
├── .gitignore
└── README.md

````

---

# Methodology

## 1. Exploratory Data Analysis

- Default rate distribution
- Loan grade risk analysis
- Correlation analysis
- Missing value diagnostics
- Class imbalance assessment

---

## 2. Feature Engineering

- **Weight of Evidence (WoE) encoding**
- **Information Value (IV)** feature selection
- Domain-informed **missing value imputation**

These methods follow **industry practices in credit scoring**.

---

## 3. Credit Rating Migration Matrix

Borrower credit transitions are modeled using **Markov chains**.

Example implementation:

```python
import numpy as np

# States: A, B, C, D, E, F, G, Default
transition_matrix = np.array([...])

pd_2yr = np.linalg.matrix_power(
    transition_matrix, 2
)[:, -1]

pd_5yr = np.linalg.matrix_power(
    transition_matrix, 5
)[:, -1]
````

---

## 4. PD Modeling

### Baseline Model

**Logistic Regression**

* interpretable
* standard regulatory benchmark
* coefficients interpretable as risk weights

### Primary Model

**XGBoost Classifier**

Advantages:

* captures nonlinear relationships
* handles large structured datasets
* strong predictive performance

---

## Model Evaluation Metrics

| Metric       | Description                               |
| ------------ | ----------------------------------------- |
| AUC-ROC      | Overall discrimination power              |
| Gini         | 2 × AUC − 1                               |
| KS Statistic | Separation between good/bad distributions |
| PSI          | Population Stability Index                |

---

## 5. SHAP Explainability

Regulatory frameworks require **transparent models**.

**SHAP (SHapley Additive exPlanations)** provides:

### Global Interpretation

Which features drive default risk across the portfolio.

### Local Interpretation

Why a specific borrower is predicted as high risk.

Example:

```python
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0]
)
```

---

# Results

*(To be updated upon project completion — April 2026)*

| Model               | AUC | Gini | KS |
| ------------------- | --- | ---- | -- |
| Logistic Regression | —   | —    | —  |
| XGBoost             | —   | —    | —  |

---

# Regulatory Context

| Framework                 | Relevance                           |
| ------------------------- | ----------------------------------- |
| IFRS 9                    | Expected Credit Loss methodology    |
| Basel IV                  | Credit risk capital requirements    |
| ECB Model Risk Guidelines | Explainability and model validation |
| BaFin                     | German regulatory governance        |

---

# Technical Stack

| Tool                 | Purpose              |
| -------------------- | -------------------- |
| Python 3.10+         | Core language        |
| Pandas / NumPy       | Data manipulation    |
| scikit-learn         | Modeling             |
| XGBoost              | Gradient boosting    |
| SHAP                 | Model explainability |
| Matplotlib / Seaborn | Visualization        |
| SQL (SQLite)         | Data validation      |
| Git                  | Version control      |

---

# How to Run

### Clone repository

```bash
git clone https://github.com/beepin6409/ifrs9-credit-risk-model
cd ifrs9-credit-risk-model
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download dataset

Download from Kaggle:

[https://www.kaggle.com/datasets/wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

Place the dataset in:

```
data/raw/
```

### Run notebooks

```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

# Requirements

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

# Project Status

* [x] Repository setup and documentation
* [ ] Exploratory data analysis
* [ ] Feature engineering
* [ ] Credit migration matrix
* [ ] Logistic regression baseline
* [ ] XGBoost PD model
* [ ] SHAP explainability
* [ ] Results documentation

---

# Author

**Bipin Ghimire**
MSc Data Science — University of Potsdam, Germany

LinkedIn
[https://www.linkedin.com/in/beepin6409/](https://www.linkedin.com/in/beepin6409/)

GitHub
[https://github.com/beepin6409](https://github.com/beepin6409)

Email
[bipin.bpn8888@gmail.com](mailto:bipin.bpn8888@gmail.com)

---

# License

MIT License — free to use with attribution.

```

---

If you want, I can also show you **3 things that would make this README look like a top-1% GitHub ML project (with diagrams, pipeline flow, and results visuals)**.
```
