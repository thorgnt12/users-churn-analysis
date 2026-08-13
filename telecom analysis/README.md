# Customer Churn Prediction — Telecom Subscriber Retention

Predict which telecom subscribers are likely to churn so a retention team can prioritize proactive outreach. The project uses the **IBM Telco Customer Churn** dataset (7,043 customers) and compares Logistic Regression with Random Forest.

## Business objective

Retention offers cost money, so the goal is not only to classify churn accurately: it is to rank customers by risk and focus interventions where they can prevent the most losses. The pipeline reports ROC-AUC, accuracy, precision, recall, average precision, and the share of churners captured by contacting the highest-risk customers.

## Dataset

The data contains customer tenure, contract, billing, internet/phone services, charges, demographics, and the `Churn` outcome. It is downloaded from IBM's public dataset repository and is not committed to this repository.

## Project structure

```text
.
├── data/
│   └── raw/                         # downloaded source CSV (gitignored)
├── models/                          # trained model (gitignored)
├── reports/                         # metrics, risk scores, and charts (gitignored)
├── scripts/
│   └── download_data.py
├── src/churn_prediction/
│   ├── __init__.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Quick start

Create a virtual environment, activate it, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the source data and train both models:

```powershell
python scripts/download_data.py
$env:PYTHONPATH = "src"
python -m churn_prediction.train
```

To run against a CSV you already have:

```powershell
$env:PYTHONPATH = "src"
python -m churn_prediction.train --data "path\to\WA_Fn-UseC_-Telco-Customer-Churn.csv"
```

## Outputs

The training command creates:

- `models/best_churn_model.joblib` — best candidate by test-set ROC-AUC.
- `reports/model_metrics.csv` and `.json` — comparable model results.
- `reports/test_set_risk_scores.csv` — customer-level held-out risk ranking; `retention_target=True` marks the highest-risk 10%.
- `reports/roc_curve_comparison.png` — ROC curve comparison.
- `reports/precision_recall_comparison.png` — precision-recall curve comparison.

## Modeling choices

- **Leakage control:** `customerID` is kept only for the final ranking export and excluded from model features. All imputing, encoding, and scaling happens within each pipeline after the train/test split.
- **Cleaning:** blank values in `TotalCharges` are converted to missing numeric values and imputed using the training-set median.
- **Class imbalance:** both models use class weighting so the minority churn class receives appropriate attention.
- **Evaluation:** ROC-AUC measures ranking quality independent of a decision threshold. Precision and recall at a 0.50 threshold add operational context, while the top-risk capture rate answers the retention use case directly.

## Resume-ready result statement

After running the project, replace the bracketed values with the generated results:

> Built and compared Logistic Regression and Random Forest churn classifiers on 7,043 telecom customers. The selected model achieved **[ROC-AUC] ROC-AUC** and **[accuracy]% accuracy** on a held-out test set; targeting the top 10% highest-risk subscribers captured **[capture-rate]%** of observed churners.

Your stated benchmark — Logistic Regression at roughly **0.84 ROC-AUC** and **79.5% accuracy**, with the top 10% capturing about **27%** of churners — is plausible for this dataset, but exact values can vary with the split, preprocessing, model configuration, and classification threshold. This project saves the actual reproducible figures from its own held-out evaluation.

## Next improvements

1. Tune classification thresholds using retention-offer cost and customer lifetime value.
2. Use cross-validation and hyperparameter tuning for the final production candidate.
3. Add a fairness review before using demographic variables in outreach decisions.
4. Monitor score distributions, realized churn, and performance drift after deployment.
