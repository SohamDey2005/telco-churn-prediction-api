# Model Card – Telco Customer Churn Classifier

## Model details
- **Developed by**: Internship / portfolio project
- **Model date**: See `trained_at` field inside the joblib artefact
- **Model type**: scikit-learn Pipeline (ColumnTransformer + RandomForestClassifier)
- **Versioning**: SHA-256 hash of the exact training CSV is embedded; library versions recorded

## Intended use
Real-time and batch scoring of telecom customers for churn propensity so that a CRM or retention system can trigger offers, calls or loyalty interventions before the customer leaves.

## Training data
- Public IBM Telco Customer Churn data set (~7 043 rows after cleaning).
- Features: demographic (gender, senior, partner, dependents), account (tenure, contract, payment method, paperless billing), services (phone, internet, add-ons), charges (monthly, total).
- Target: binary churn within the observation window (~26.5 % positive).
- Pre-processing: `TotalCharges` coerced from string to float (11 blank rows dropped); SeniorCitizen treated as categorical; median / mode imputation; standard scaling of numerics; one-hot encoding of categoricals with `handle_unknown="ignore"`.

## Evaluation
- Stratified 80/20 train-test split, 5-fold stratified CV for hyper-parameter search (ROC-AUC).
- Final operating threshold selected by minimising expected cost = 5 × FN + 1 × FP.
- Hold-out metrics (approximate):
  - ROC-AUC ≈ 0.83
  - Precision ≈ 0.41, Recall ≈ 0.93, F1 ≈ 0.57 at the chosen threshold
- Confusion matrix stored in the artefact for full auditability.

## Ethical considerations & limitations
- The public data set may not represent every market or demographic; fairness across gender / senior-citizen status should be monitored in production.
- High-recall threshold will generate more retention offers than a balanced threshold; cost of those offers must stay below the value of retained customers.
- Model is static until retrained; concept drift (new products, pricing changes, competitor actions) will degrade performance. The `/drift` PSI endpoint provides a first-line alert.

## Environmental & computational
- Training completes in seconds on a modern laptop.
- Inference latency for a single record is typically < 20 ms (well under the 100 ms median target).
- Docker image size stays under 600 MB thanks to multi-stage build and slim base image.

## Citation
IBM Telco Customer Churn data set (publicly available). Model implementation follows standard scikit-learn best practices for production pipelines.
