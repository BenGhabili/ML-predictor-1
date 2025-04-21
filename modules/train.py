"""
train.py
CLI wrapper that delegates real work to services.train_utils
"""

import argparse
from pathlib import Path
from services.train_utils import (
    load_processed_csv,
    train_rf_model_over_sampled as train_model,
    # train_rf_model as train_model,
    # train_knn_model,
    save_model
)

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


# def main(timeframe: int, csv_path: Path, k: int):
def main(timeframe: int, csv_path: Path, n_trees: int):
    X, y = load_processed_csv(csv_path)
    # clf, rpt = train_knn_model(X, y, n_neighbors=k) // KNN neighbour model
    clf, rpt = train_model(X, y, n_trees)  # Random forest model --- uses default n_tree=400

    print("\n=== classification report ===")
    print(rpt)

    out_pkl = save_model(clf, timeframe, MODEL_DIR, MODEL_LOG)
    print("\nModel saved ->", out_pkl)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train KNN (3‑class label3).")
    p.add_argument("--timeframe", "-tf", type=int, default=1,
                   help="Timeframe in minutes (for filename defaults)")
    p.add_argument("--input", "-i", type=str,
                   help="Explicit processed CSV path")
    # p.add_argument("--k", type=int, default=7,
    #                help="n_neighbors for KNN")
    p.add_argument("--trees", "-t", type=int, default=400,
                   help="Number of trees for Random‑Forest")
    args = p.parse_args()

    csv_file = Path(args.input) if args.input else latest_csv(args.timeframe)
    # main(args.timeframe, csv_file, args.k)
    main(args.timeframe, csv_file, args.trees)
