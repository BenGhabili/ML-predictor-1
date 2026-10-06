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
    SMOTE-balance then fit RandomForest on the full data provided.
    The caller (CV loop) is responsible for train/test splitting.
    Returns (clf, None).
    """
    if len(np.unique(y)) >= 2:
        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_bal, y_bal = sm.fit_resample(X, y)
    else:
        X_bal, y_bal = X, y

    clf = RandomForestClassifier(
            n_estimators=n_trees,
            random_state=42,
            n_jobs=-1)
    clf.fit(X_bal, y_bal)

    return clf, None

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
    3. If X_val/y_val given → create an internal holdout from the
       SMOTE'd training data for early stopping (avoids using the
       CV test fold for tuning, which would inflate metrics).
    4. Return the *fitted* model.
    """

    X_bal, y_bal = SMOTE(random_state=42).fit_resample(X, y)

    # Use early stopping only when we have enough data
    use_early_stop = X_val is not None and len(X_bal) >= 100

    if use_early_stop:
        # Split off 15% of the SMOTE'd training data for early stopping
        X_fit, X_es, y_fit, y_es = train_test_split(
            X_bal, y_bal, test_size=0.15, random_state=42, stratify=y_bal)
    else:
        X_fit, y_fit = X_bal, y_bal
        X_es, y_es = None, None

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric=["mlogloss", "merror"],
        early_stopping_rounds=early_stop_rounds if use_early_stop else None,
        random_state=42,
        enable_categorical=False,
    )

    if use_early_stop:
        clf.fit(
            X_fit, y_fit,
            eval_set=[(X_es, y_es)],
            verbose=explain,
        )
    else:
        clf.fit(X_fit, y_fit, verbose=explain)

    if explain:
        from services.explain import generate_visualizations
        generate_visualizations(clf, X_bal, y_bal)

    return clf

# ────────────────────────────────────────────────────────────
def train_knn_model(X: np.ndarray,
                    y: np.ndarray,
                    n_neighbors=KNN["K"]):
    """Train KNN on the full data provided.
    The caller (CV loop) is responsible for train/test splitting.
    Returns (clf, None)."""
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, y)

    return clf, None

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
