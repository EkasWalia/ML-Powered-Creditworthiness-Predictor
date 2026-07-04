#!/usr/bin/env python3
"""
cli.py  —  Command-Line Credit Scoring Tool
Run:  python3 cli.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from credit_model import predict

BANNER = """
╔══════════════════════════════════════════════════════╗
║          💳  CREDIT SCORING SYSTEM  v1.0            ║
║        ML-Powered Creditworthiness Predictor        ║
╚══════════════════════════════════════════════════════╝
"""

def ask(prompt, cast=float, default=None):
    while True:
        try:
            raw = input(f"  {prompt}: ").strip()
            if raw == "" and default is not None:
                return default
            return cast(raw)
        except ValueError:
            print("  ⚠️  Invalid input, please try again.")

def collect_applicant():
    print("\n📋  APPLICANT INFORMATION")
    print("  " + "─"*45)
    name = input("  Full Name: ").strip() or "Applicant"
    age  = int(ask("Age", int))

    print("\n💰  FINANCIAL DETAILS")
    print("  " + "─"*45)
    annual_income      = ask("Annual Income ($)")
    total_debt         = ask("Total Debt ($)")
    savings            = ask("Savings ($)")

    print("\n📊  CREDIT PROFILE")
    print("  " + "─"*45)
    credit_history_years = int(ask("Credit History (years)", int))
    credit_utilization   = ask("Credit Utilization (0.0 - 1.0)")
    num_credit_cards     = int(ask("Number of Credit Cards", int))
    missed_payments      = int(ask("Missed Payments (last 2 years)", int))

    print("\n🏢  EMPLOYMENT & LOANS")
    print("  " + "─"*45)
    employment_years = int(ask("Years Employed", int))
    num_loans        = int(ask("Number of Active Loans", int))

    return dict(name=name, age=age, annual_income=annual_income,
                total_debt=total_debt, savings=savings,
                credit_history_years=credit_history_years,
                credit_utilization=credit_utilization,
                num_credit_cards=num_credit_cards,
                missed_payments=missed_payments,
                employment_years=employment_years,
                num_loans=num_loans)

def display_result(r):
    verdict = "✅  APPROVED" if r['approved'] else "❌  DECLINED"
    bar_len  = int(r['probability'] * 40)
    bar      = "█" * bar_len + "░" * (40 - bar_len)

    print("\n" + "═"*54)
    print(f"  CREDIT DECISION FOR: {r['applicant_name'].upper()}")
    print("═"*54)
    print(f"\n  DECISION  :  {verdict}")
    print(f"  CREDIT SCORE :  {r['credit_score']}  ({r['score_range']})")
    print(f"  GRADE        :  {r['grade']}")
    print(f"  RISK LEVEL   :  {r['risk_level']}")
    print(f"\n  Confidence   :  [{bar}]  {r['probability']*100:.1f}%")
    print(f"  DTI Ratio    :  {r['dti_ratio']}%")
    print(f"  Utilization  :  {r['utilization_pct']}%")

    print("\n  ── RISK FACTORS ──────────────────────────────")
    for title, detail in r['risk_factors']:
        print(f"\n  {title}")
        print(f"     {detail}")

    print("\n  ── RECOMMENDATIONS ───────────────────────────")
    for tip in r['recommendations']:
        print(f"  {tip}")
    print("\n" + "═"*54)

def run_batch(filepath):
    import csv
    with open(filepath) as f:
        readers = list(csv.DictReader(f))
    print(f"\n  Processing {len(readers)} applicants...\n")
    print(f"  {'Name':<20} {'Score':<8} {'Grade':<12} {'Decision'}")
    print("  " + "─"*55)
    for row in readers:
        r_dict = {
            'name': row.get('name','?'),
            'age': int(row['age']),
            'annual_income': float(row['annual_income']),
            'total_debt': float(row['total_debt']),
            'savings': float(row['savings']),
            'credit_history_years': int(row['credit_history_years']),
            'credit_utilization': float(row['credit_utilization']),
            'num_credit_cards': int(row['num_credit_cards']),
            'missed_payments': int(row['missed_payments']),
            'employment_years': int(row['employment_years']),
            'num_loans': int(row['num_loans']),
        }
        res = predict(r_dict)
        icon = "✅" if res['approved'] else "❌"
        print(f"  {res['applicant_name']:<20} {res['credit_score']:<8} {res['grade']:<12} {icon} {res['decision']}")
    print()

if __name__ == "__main__":
    print(BANNER)
    if len(sys.argv) == 3 and sys.argv[1] == "--batch":
        run_batch(sys.argv[2])
    else:
        while True:
            applicant = collect_applicant()
            result    = predict(applicant)
            display_result(result)
            again = input("\n  Score another applicant? (y/n): ").strip().lower()
            if again != 'y':
                print("\n  👋  Thank you for using the Credit Scoring System!\n")
                break
