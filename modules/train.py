"""
# modules/train.py
CLI wrapper that delegates real work to services.train_utils
"""

import argparse, numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import f1_score, classification_report

from services.explain import plot_learning_curve
from services.model_factory import build_model
from services.train_utils import load_processed_csv,save_model

DATA_DIR   = Path("data")
MODEL_DIR  = Path("models")
MODEL_LOG  = MODEL_DIR / "model-list.csv"


# helper to auto‑pick latest processed CSV
def latest_csv(tf_min: int) -> Path:
    pattern = DATA_DIR / f"processed_data_{tf_min}min_*.csv"
    files   = sorted(pattern.parent.glob(pattern.name))
    if not files:
        raise FileNotFoundError(f"No CSV for {tf_min}‑minute timeframe.")
    return files[-1]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["rf", "xgb", "knn"], default="rf",
                   help="Which model to train")
    p.add_argument("--timeframe", "-tf", type=int, default=1,
                   help="Timeframe in minutes (for filename defaults)")
    p.add_argument("--input", "-i", type=str,
                   help="Explicit processed CSV path")
    p.add_argument("--trees", "-t", type=int, default=400,
                   help="#Trees for RF / XGB")
    p.add_argument("--eta", type=float, default=0.05,
                   help="Learning-rate for XGB")
    p.add_argument("--depth", type=int, default=6,
                   help="Max tree depth for RF / XGB")
    p.add_argument("--k", type=int, default=7,
                   help="Neighbours for k-NN")
    p.add_argument("--cv", choices=["walk", "random", "none"], default="walk",
                   help="walk = time-series CV (default); "
                        "random = 25 % hold-out; none = skip CV")
    p.add_argument("--save", action="store_true",
                   help="Save the full-data model (default off when experimenting)")
    p.add_argument("--explain", action="store_true",
                   help="Explain and visualise the model (default off when experimenting)")
    args = p.parse_args()
    
    # ---------- load dataset -------------------------------------------
    csv_path = Path(args.input) if args.input else latest_csv(args.timeframe)
    X, y = load_processed_csv(csv_path)
    
    # ---------- choose splitter ----------------------------------------
    if args.cv == "walk":
        total = len(y)
        n_splits = 5
        test_size = total // (n_splits + 1)
        cv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    elif args.cv == "random":
        tr_idx, te_idx = train_test_split(
            np.arange(len(y)), test_size=0.25, shuffle=True, random_state=42)
        cv = [(tr_idx, te_idx)]
    else:  # none
        cv = []
    
    # ---------- evaluation loop ----------------------------------------
    cv_iter = cv if isinstance(cv, list) else cv.split(X)
    f1_scores = []
    
    for fold, (tr_idx, te_idx) in enumerate(cv_iter, 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
    
        model_fold = build_model(
            args.algo, X_tr, y_tr, args.explain,
            trees=args.trees, depth=args.depth, eta=args.eta, k=args.k,
            X_val=X_te, y_val=y_te,    # XGB early-stop; ignored by RF/k-NN
        )
    
        f1 = f1_score(y_te, model_fold.predict(X_te), average="macro")
        f1_scores.append(f1)
        print(f"Fold {fold} macro-F1: {f1:.3f}")
    
    if f1_scores:
        print(f"CV macro-F1 mean±std: "
              f"{np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    else:
        print("No CV performed (cv=none).")
    
    # ---------- final full-data fit & save -----------------------------
    model_full = build_model(
        args.algo, X, y, args.explain,
        trees=args.trees, depth=args.depth, eta=args.eta, k=args.k,
        X_val=None, y_val=None)       # disables early-stop
    
    print("\n=== in-sample classification report ===")
    print(classification_report(y, model_full.predict(X),
                                digits=3, zero_division=0))
    
    if args.explain:
        plot_learning_curve(model_full, X, y)

    if args.save:
        out = save_model(model_full, args.timeframe,
                     Path("models"), Path("models/model-list.csv"))
        print("Saved ->", out)
    # ----------------------------------------------------------------------
