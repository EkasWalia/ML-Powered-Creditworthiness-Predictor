"""
credit_model.py  —  Core engine for the Credit Scoring Model
All feature engineering + prediction logic lives here.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model.joblib"
_model = None

# ── Credit Score Bands ────────────────────────────────────
BANDS = [
    (0.90, 1.00, "Excellent",  "800-850", "Very Low",  "#27AE60"),
    (0.75, 0.90, "Good",       "740-799", "Low",        "#2ECC71"),
    (0.55, 0.75, "Fair",       "670-739", "Medium",     "#F39C12"),
    (0.35, 0.55, "Poor",       "580-669", "High",       "#E67E22"),
    (0.00, 0.35, "Very Poor",  "300-579", "Very High",  "#E74C3C"),
]

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def engineer_features(raw: dict) -> pd.DataFrame:
    """Convert raw applicant dict → feature DataFrame ready for model."""
    d = raw.copy()
    income = max(d['annual_income'], 1)

    d['debt_to_income_ratio']    = d['total_debt'] / income
    d['savings_to_income_ratio'] = d['savings'] / income
    d['payment_reliability']     = 1 / (d['missed_payments'] + 1)
    d['loan_burden']             = d['num_loans'] * d['total_debt'] / income

    yrs = d['credit_history_years']
    d['credit_age_group'] = 0 if yrs<=5 else 1 if yrs<=10 else 2 if yrs<=20 else 3

    inc = d['annual_income']
    d['income_group'] = 0 if inc<30000 else 1 if inc<60000 else 2 if inc<100000 else 3

    FEATURES = [
        'age','annual_income','total_debt','num_loans','missed_payments',
        'credit_history_years','credit_utilization','num_credit_cards',
        'employment_years','savings','debt_to_income_ratio',
        'savings_to_income_ratio','payment_reliability','loan_burden',
        'credit_age_group','income_group'
    ]
    return pd.DataFrame([d])[FEATURES]

def score_to_points(prob: float) -> int:
    """Map probability (0-1) to a 300-850 credit score."""
    return int(300 + prob * 550)

def get_band(prob: float) -> dict:
    for lo, hi, grade, range_, risk, color in BANDS:
        if lo <= prob <= hi:
            return {"grade": grade, "range": range_, "risk": risk, "color": color}
    return {"grade": "Unknown", "range": "N/A", "risk": "N/A", "color": "#888"}

def get_risk_factors(raw: dict) -> list:
    """Return list of risk flags with explanations."""
    flags = []
    if raw['missed_payments'] >= 3:
        flags.append(("🔴 High Missed Payments",
                       f"{raw['missed_payments']} missed payments detected. "
                       "Payment history is the most critical factor."))
    if raw['credit_utilization'] > 0.7:
        flags.append(("🔴 High Credit Utilization",
                       f"{raw['credit_utilization']*100:.0f}% utilization. "
                       "Keep below 30% for best scores."))
    if raw['total_debt'] / max(raw['annual_income'],1) > 0.5:
        flags.append(("🟠 High Debt-to-Income Ratio",
                       f"DTI = {raw['total_debt']/raw['annual_income']*100:.1f}%. "
                       "Lenders prefer below 36%."))
    if raw['credit_history_years'] < 3:
        flags.append(("🟡 Short Credit History",
                       f"Only {raw['credit_history_years']} years. "
                       "Longer history improves trust."))
    if raw['savings'] < 5000:
        flags.append(("🟡 Low Savings",
                       f"${raw['savings']:,.0f} in savings. "
                       "Emergency fund improves financial stability."))
    if raw['num_loans'] > 5:
        flags.append(("🟠 Too Many Active Loans",
                       f"{raw['num_loans']} loans active. "
                       "Multiple debts increase default risk."))
    if not flags:
        flags.append(("🟢 No Major Risk Factors",
                       "Your financial profile looks healthy!"))
    return flags

def get_recommendations(raw: dict, prob: float) -> list:
    """Actionable advice based on applicant profile."""
    tips = []
    if raw['missed_payments'] > 0:
        tips.append("✅ Set up autopay to avoid missed payments — this is the #1 factor.")
    if raw['credit_utilization'] > 0.3:
        tips.append(f"✅ Reduce credit card balances — target below 30% utilization.")
    if raw['savings'] < 10000:
        tips.append("✅ Build an emergency fund of at least 3 months' expenses.")
    if raw['credit_history_years'] < 5:
        tips.append("✅ Keep old accounts open to grow your credit history.")
    if prob < 0.5:
        tips.append("✅ Consider a secured credit card to rebuild your credit.")
        tips.append("✅ Pay off highest-interest debt first (avalanche method).")
    if not tips:
        tips.append("✅ Maintain your current habits — your profile is excellent!")
    return tips

def predict(raw: dict) -> dict:
    """
    Full prediction pipeline.
    Input : raw applicant dict
    Output: complete credit scoring result
    """
    model    = load_model()
    X        = engineer_features(raw)
    prob     = float(model.predict_proba(X)[0][1])
    decision = "APPROVED" if prob >= 0.50 else "DECLINED"
    score    = score_to_points(prob)
    band     = get_band(prob)
    flags    = get_risk_factors(raw)
    tips     = get_recommendations(raw, prob)

    return {
        "applicant_name":  raw.get("name", "Applicant"),
        "decision":        decision,
        "approved":        decision == "APPROVED",
        "credit_score":    score,
        "probability":     round(prob, 4),
        "grade":           band["grade"],
        "score_range":     band["range"],
        "risk_level":      band["risk"],
        "color":           band["color"],
        "risk_factors":    flags,
        "recommendations": tips,
        "dti_ratio":       round(raw['total_debt'] / max(raw['annual_income'],1) * 100, 1),
        "utilization_pct": round(raw['credit_utilization'] * 100, 1),
    }
