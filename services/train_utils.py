"""
train_utils.py
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
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report


# ────────────────────────────────────────────────────────────
def load_processed_csv(csv_path: Path):
    """Read processed_data_* CSV and split into X, y."""
    df = pd.read_csv(csv_path)

    # --- engineered features are already present ---
    X = df[['feature1', 'feature2', 'feature3']].values
    y = df['label3'].values                # 2 / 1 / 0

    return X, y


# ────────────────────────────────────────────────────────────


def train_rf_model_over_sampled(X,
                                y,
                                n_trees: int = 400,
                                k_neighbors: int = 5):
    """
    1. Over‑sample minority classes with SMOTE
    2. Train RandomForest on the balanced data
    3. Return (clf, classification_report_string)
    """
    # 1️ Over‑sample
    sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_bal, y_bal = sm.fit_resample(X, y)

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

def train_rf_model(X, y, n_trees=400):
    """
    Random Forest with class_weight='balanced' so the minority
    classes (1 = down, 2 = up) get extra attention.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=None,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1)
    clf.fit(X_tr, y_tr)
    print("Trained model type:", clf.__class__.__name__)

    report = classification_report(y_te, clf.predict(X_te), digits=3)
    return clf, report

# ────────────────────────────────────────────────────────────
def train_knn_model(X: np.ndarray,
                    y: np.ndarray,
                    n_neighbors: int = 7):
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
    prefix = "rf" if clf.__class__.__name__.startswith("RandomForestClassifie") else "knn"
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
