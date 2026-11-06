# Telco Customer Churn Prediction – Production Pipeline

Package, test and containerise a churn model as a secure API that a CRM can call in real time.

## Overview

- **Dataset**: IBM Telco Customer Churn (~7 043 records). Categorical service fields, `TotalCharges` stored as text, modest class imbalance (~26.5 % churn).
- **Pipeline**: scikit-learn `Pipeline` that cleans (numeric conversion + imputation), one-hot encodes categoricals (`handle_unknown="ignore"`), scales numerics and applies a tuned classifier.
- **Tuning**: Randomized search with 5-fold stratified CV on ROC-AUC; final decision threshold chosen by expected cost (FN cost = 5 × FP cost) to prioritise recall of churners.
- **Versioning**: Artefact stores training-data SHA-256 hash, metrics, library versions and training timestamp so every prediction is traceable.
- **API**: FastAPI with Pydantic schemas, single + batch endpoints, health check, API-key auth, structured latency logging, graceful unseen-category handling.
- **Stretch**: `/drift` endpoint computing Population Stability Index (PSI) vs training distribution; flags features with PSI > 0.2.
- **Packaging**: Multi-stage Docker image (Python 3.12-slim), non-root user, health-check. Target size < 600 MB.
- **Tests**: 17 unit & integration tests covering validation, auth, prediction, latency header, drift and artefact integrity. Median single-prediction latency well under 100 ms on typical hardware.

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py                    # produces models/churn_pipeline.joblib
uvicorn main:app --app-dir src --reload # http://127.0.0.1:8000/docs
```

Default API key: `churn-demo-key-change-me` (override with env `API_KEY`).

## Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 -e API_KEY=my-secret churn-api
```

Image size is typically 350–450 MB (well under the 600 MB limit).

## Example requests

### Health
```bash
curl http://localhost:8000/health
```

### Single prediction
```bash
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -H "X-API-Key: churn-demo-key-change-me" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 845.0
  }'
```

### Batch
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: churn-demo-key-change-me" \
  -d '{"records": [ { ... same fields ... } ] }'
```

### Drift check
```bash
curl -X POST http://localhost:8000/drift \
  -H "Content-Type: application/json" \
  -H "X-API-Key: churn-demo-key-change-me" \
  -d '{"records": [ { ... } , { ... } ] }'
```

OpenAPI / Swagger UI is available at `/docs`; ReDoc at `/redoc`. The schema is also exported to `docs/openapi.json` after the server starts.

## Model card (summary)

| Item | Value |
|------|-------|
| Model type | Random Forest (class_weight=balanced, max_depth=8) inside a full preprocessing pipeline |
| Training data hash | (see `models/model_meta.json`) |
| ROC-AUC (hold-out) | ~0.83 |
| Selected threshold | ~0.28 (cost-sensitive) |
| Precision / Recall (at threshold) | ~0.41 / ~0.93 |
| Cost ratio used | FN : FP = 5 : 1 |
| Intended use | Real-time scoring inside a CRM retention workflow |
| Limitations | Trained on a public synthetic-style Telco set; performance may degrade on different customer bases or after major product changes. Monitor PSI and retrain periodically. |

Full model card: `docs/model_card.md`.

## Threshold justification

Missing a true churner (false negative) costs the business far more (lost future revenue, acquisition cost to replace the customer) than sending a retention offer to a customer who would have stayed (false positive – coupon / call centre cost). We therefore set the decision threshold by minimising expected cost = 5 × FN + 1 × FP on the hold-out set. This yields a high-recall operating point (~0.93 recall) at the expense of lower precision – acceptable when the retention offer is inexpensive relative to customer lifetime value.

## Safe retrain & redeploy process

1. **Data**: Append new labelled observations; compute new data hash.
2. **Train**: Re-run `src/train.py` (or a scheduled CI job). The new artefact is written with updated hash, metrics and versions.
3. **Validate**: Run the full test suite + a hold-out evaluation against a frozen validation set; require ROC-AUC and cost metrics within agreed tolerance of the previous model.
4. **Shadow / canary**: Deploy the new image alongside the old one; route a small percentage of traffic or score offline and compare distributions.
5. **Promote**: Once metrics and latency are acceptable, update the production tag / Kubernetes deployment. Keep the previous image for instant rollback.
6. **Monitor**: Continuously log predictions, actual outcomes (when available) and PSI; alert on drift or latency regression.

## Project layout

```
churn-pipeline/
├── data/telco_churn.csv
├── models/churn_pipeline.joblib   # versioned artefact
├── models/model_meta.json
├── src/
│   ├── train.py
│   ├── main.py
│   └── schemas.py
├── tests/test_api.py              # 17 tests
├── docs/
│   ├── model_card.md
│   └── openapi.json               # generated
├── Dockerfile
├── requirements.txt
└── README.md
```

## Licence

This project is supplied for educational / portfolio purposes. The underlying Telco dataset is publicly available under its original terms.
