"""
# services/train_utils.py
Low‑level helpers for training and saving models.
Keeps the main train.py script very slim.
"""

from pathlib import Path
from datetime import datetime
import csv
import pandas as pd
import numpy as np
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier          # swap later
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from services.model_defaults import RF, XGB, KNN


# ────────────────────────────────────────────────────────────
def load_processed_csv(csv_path: Path):
    df = pd.read_csv(csv_path)

    # Get ALL features except label3
    feature_cols = [col for col in df.columns if col != 'label3']

    # Drop rows with any NaNs in features or label
    before = len(df)
    df = df.dropna(subset=feature_cols + ['label3'])
    after = len(df)
    if after < before:
        print(f"Dropped {before - after} rows with NaNs in features/label")

    print("Using features:", feature_cols)  # Verify
    print("Label distribution:\n", df['label3'].value_counts())

    return df[feature_cols].values, df['label3'].values

# ────────────────────────────────────────────────────────────

def train_rf_model_over_sampled(X, y,
                                n_trees=RF["TREES"],
                                k_neighbors=5):
    """
    1. Over‑sample minority classes with SMOTE
    2. Train RandomForest on the balanced data
    3. Return (clf, classification_report_string)
    """
    """
    If the training subset has ≥2 classes → SMOTE-balance then fit RF
    else → skip SMOTE and fit RF with class_weight='balanced'.
    """
    if len(np.unique(y)) >= 2:
        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_bal, y_bal = sm.fit_resample(X, y)
    else:
        # keep data as-is; class_weight will handle the imbalance
        X_bal, y_bal = X, y

    # 2 Train / test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_bal, y_bal, test_size=0.25, random_state=42, stratify=y_bal)

    # 3️ Random‑Forest (no class_weight, data already balanced)
    clf = RandomForestClassifier(
            n_estimators=n_trees,
            random_state=42,
            n_jobs=-1)
    clf.fit(X_tr, y_tr)

    report = classification_report(
        y_te, clf.predict(X_te), digits=3, zero_division=0)
    return clf, report

# ────────────────────────────────────────────────────────────

def build_xgb_model(X, y, *,
                    n_estimators=XGB["TREES"],
                    learning_rate=XGB["ETA"],
                    max_depth=XGB["DEPTH"],
                    scale_pos_weight=XGB["SCALE_POS_WEIGHT"],
                    X_val=None, y_val=None,
                    early_stop_rounds=XGB["EARLY_STOP"],
                    explain=False):
    """
    1. SMOTE-balance the classes.
    2. Build an XGBClassifier with supplied hyper-params.
    3. If X_val/y_val given → fit with early stopping,
       else just fit on the balanced data.
    4. Return the *fitted* model.
    """

    X_bal, y_bal = SMOTE(random_state=42).fit_resample(X, y)

    # 2. Configure model
    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,           # Recommended for financial data
        colsample_bytree=0.8,    # Helps prevent overfitting
        eval_metric=["mlogloss", "merror"],  # Multi-class metrics
        early_stopping_rounds=early_stop_rounds if X_val is not None else None,
        random_state=42,
        enable_categorical=False  # Critical for numerical data
    )

    # --- Training ---
    if X_val is not None and y_val is not None:
        clf.fit(
            X_bal, y_bal,
            eval_set=[(X_val, y_val)],
            verbose=explain  # Only show if explain=True
        )
    else:
        clf.fit(X_bal, y_bal, verbose=explain)

    if explain:
        from services.explain import generate_visualizations
        generate_visualizations(clf, X_bal, y_bal)

    return clf

# ────────────────────────────────────────────────────────────
def train_knn_model(X: np.ndarray,
                    y: np.ndarray,
                    n_neighbors=KNN["K"]):
    """Train KNN, return fitted estimator and a performance report."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_tr, y_tr)

    report = classification_report(y_te, clf.predict(X_te), digits=3)
    return clf, report

# ────────────────────────────────────────────────────────────
def save_model(clf,
               timeframe_min: int,
               model_dir: Path,
               model_log: Path):
    """
    Serialize model with timestamped name and append entry to model-list.csv
    """
    model_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%d-%m-%y-%H%M%S")

    prefix = (
        "xgb" if clf.__class__.__name__.startswith("XGB")
        else "rf" if clf.__class__.__name__.startswith("RandomForest")
        else "knn"
    )

    fname = f"{prefix}_{timeframe_min}min_label3_{ts}.pkl"
    out_pkl = model_dir / fname
    dump(clf, out_pkl)

    # append log
    file_exists = model_log.exists()
    with model_log.open("a", newline="") as lf:
        wr = csv.writer(lf)
        if not file_exists:
            wr.writerow(["Filename", "Timeframe", "Created_At"])
        wr.writerow([fname, f"{timeframe_min}min",
                     datetime.now().strftime("%d-%m-%y %H:%M:%S")])
    return out_pkl
