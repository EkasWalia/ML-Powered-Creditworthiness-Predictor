# Credit Scoring System — ML-Powered Creditworthiness Predictor

## Project Overview

This project builds a full credit scoring pipeline from raw financial data to a deployable prediction system. Given an applicant's income, debt, payment history, and credit profile, the model outputs a credit score (300–850), a grade (Excellent → Very Poor), a loan decision (APPROVED / DECLINED), risk flags, and personalized recommendations.


## Model Performance

| Metric | Score |
|---|---|
| Accuracy | **99.5%** |
| Precision | **99.2%** |
| Recall | **100%** |
| F1-Score | **99.6%** |
| ROC-AUC | **0.9999** |

Cross-validated using **5-Fold Stratified KFold** and **Repeated Stratified KFold (5×10 = 50 folds)**.


## Project Structure

```
credit-scoring-system/
│
├── credit_model.py          # Core ML engine — feature engineering + prediction
├── cli.py                   # Command-line application (single & batch mode)
├── app.py                   # FastAPI REST API for deployment
├── credit_scoring_app.html  # Interactive web application
│
├── model/
│   ├── model.joblib             # Trained & serialized pipeline
│   ├── best_lr_tuned.joblib     # Tuned Logistic Regression
│   ├── best_dt_tuned.joblib     # Tuned Decision Tree
│   ├── best_rf_tuned.joblib     # Tuned Random Forest
│   └── model_metadata.json      # Model version & metrics
│
├── data/
│   ├── credit_data.csv              # Raw synthetic dataset (2,000 records)
│   ├── credit_data_engineered.csv   # Feature-engineered dataset
│   └── sample_batch.csv             # Sample batch input file
│
├── notebooks/
│   └── credit_scoring_full.ipynb    # End-to-end analysis notebook
│
└── requirements.txt
```


## ⚙️ Features

### Feature Engineering
Six new features are derived from raw inputs:

| Feature | Formula | Why It Matters |
|---|---|---|
| `debt_to_income_ratio` | total_debt ÷ annual_income | Core lender metric |
| `savings_to_income_ratio` | savings ÷ annual_income | Financial stability signal |
| `payment_reliability` | 1 ÷ (missed_payments + 1) | Inverse penalty for defaults |
| `loan_burden` | num_loans × debt ÷ income | Multi-debt pressure |
| `credit_age_group` | Binned: New / Fair / Good / Excellent | Categorical credit maturity |
| `income_group` | Binned: Low / Medium / High / Very High | Income tier encoding |

### ML Pipeline
```
Raw Input → Feature Engineering → StandardScaler → Logistic Regression → Decision + Score
```

### Credit Score Bands
| Grade | Score Range | Risk Level |
|---|---|---|
| Excellent | 800–850 | Very Low |
| Good | 740–799 | Low |
| Fair | 670–739 | Medium |
| Poor | 580–669 | High |
| Very Poor | 300–579 | Very High |

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/credit-scoring-system.git
cd credit-scoring-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the CLI (Interactive Mode)
```bash
python3 cli.py
```

### 4. Run Batch Scoring
```bash
python3 cli.py --batch data/sample_batch.csv
```

### 5. Launch the REST API
```bash
uvicorn app:app --reload --port 8000
```
Then open `http://localhost:8000/docs` for the interactive Swagger UI.

### 6. Open the Web App
Simply open `credit_scoring_app.html` in any browser — no server required.

## 🔌 API Usage

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "annual_income": 80000,
    "total_debt": 5000,
    "num_loans": 1,
    "missed_payments": 0,
    "credit_history_years": 12,
    "credit_utilization": 0.15,
    "num_credit_cards": 2,
    "employment_years": 8,
    "savings": 25000
  }'
```

**Response:**
```json
{
  "label": "Creditworthy",
  "class": 1,
  "confidence": 0.9998,
  "risk_level": "Low"
}
```

### Python Usage
```python
from credit_model import predict

applicant = {
    "name": "Rahul Sharma",
    "age": 35,
    "annual_income": 90000,
    "total_debt": 5000,
    "savings": 30000,
    "credit_history_years": 12,
    "credit_utilization": 0.12,
    "num_credit_cards": 2,
    "missed_payments": 0,
    "employment_years": 10,
    "num_loans": 1
}

result = predict(applicant)
print(result)
# {
#   "decision": "APPROVED",
#   "credit_score": 850,
#   "grade": "Excellent",
#   "risk_level": "Very Low",
#   "risk_factors": [...],
#   "recommendations": [...]
# }
```


## Research & Results

The project went through a full ML development lifecycle:

1. **EDA & Feature Engineering** — Explored 10 raw features, engineered 6 new derived features
2. **Model Training** — Compared Logistic Regression, Decision Tree, and Random Forest
3. **Hyperparameter Tuning** — GridSearchCV (LR, DT) and RandomizedSearchCV (RF) with 5-Fold CV
4. **Cross-Validation Deep Dive** — SKF-5, SKF-10, and Repeated SKF (50 folds) for stability analysis
5. **Threshold Optimization** — Business-driven precision/recall tradeoff analysis
6. **Deployment** — joblib export, FastAPI REST service, CLI tool, and web app

---

## 📦 Requirements

```
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
joblib>=1.3
fastapi>=0.100
uvicorn>=0.23
pydantic>=2.0
matplotlib>=3.7
seaborn>=0.12
```

---

# Acknowledgements

Built as part of a machine learning developer role assignment. Dataset is synthetic and generated for research purposes only.


# 📄 License

This project is licensed under the MIT License.
