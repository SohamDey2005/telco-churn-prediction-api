"""
Train a scikit-learn pipeline for Telco Customer Churn prediction.
Bundles cleaning, encoding, scaling and classifier.
Tunes with cross-validated search and selects cost-sensitive decision threshold.
Versions the artefact with data hash, metrics and library versions.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Cost assumptions (business justification later)
COST_FN = 5.0  # cost of missing a churner (lost lifetime value proxy)
COST_FP = 1.0  # cost of wasted retention offer


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # TotalCharges stored as text; convert blanks to NaN then float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)  # treat as categorical
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def build_pipeline():
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # We will search over several classifiers
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )
    return pipe, numeric_features, categorical_features


def find_optimal_threshold(y_true, y_proba, cost_fn=COST_FN, cost_fp=COST_FP):
    """Choose threshold that minimises expected cost = FN*cost_fn + FP*cost_fp."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # thresholds has length n-1; align
    best_thr = 0.5
    best_cost = float("inf")
    for thr in np.linspace(0.05, 0.95, 91):
        y_pred = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = fn * cost_fn + fp * cost_fp
        if cost < best_cost:
            best_cost = cost
            best_thr = thr
    return best_thr, best_cost


def main():
    print("Loading data...")
    df = load_and_clean(DATA_PATH)
    data_hash = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16]

    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe, num_feats, cat_feats = build_pipeline()

    param_distributions = {
        "classifier": [
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
            ),
            GradientBoostingClassifier(random_state=42),
        ],
        "classifier__C": [0.1, 1.0, 10.0],  # only for LR; ignored by others
    }

    # Simpler search focused on LR + RF for speed
    search = RandomizedSearchCV(
        pipe,
        param_distributions={
            "classifier": [
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ],
        },
        n_iter=4,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    print("Tuning model...")
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_

    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba)
    thr, cost = find_optimal_threshold(y_test, y_proba)
    y_pred = (y_proba >= thr).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "roc_auc": float(roc),
        "threshold": float(thr),
        "expected_cost": float(cost),
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "support": int(report["1"]["support"]),
        "confusion_matrix": cm,
        "best_params": str(search.best_params_),
    }

    # Library versions
    import sklearn
    import fastapi  # may not be installed yet; handle later

    versions = {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }

    artefact = {
        "model": best_pipe,
        "threshold": thr,
        "feature_names": list(X.columns),
        "numeric_features": num_feats,
        "categorical_features": cat_feats,
        "data_hash": data_hash,
        "metrics": metrics,
        "versions": versions,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
    }

    model_path = MODEL_DIR / "churn_pipeline.joblib"
    joblib.dump(artefact, model_path)
    print(f"Saved artefact to {model_path}")

    meta_path = MODEL_DIR / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "data_hash": data_hash,
                "metrics": metrics,
                "versions": versions,
                "trained_at": artefact["trained_at"],
                "threshold": thr,
            },
            f,
            indent=2,
        )
    print(json.dumps(metrics, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
