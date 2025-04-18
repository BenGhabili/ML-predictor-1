import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import csv
from tqdm import tqdm
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from services.feature_helpers import compute_features

def main(timeframe: int, input_csv: Path, model_folder: Path, model_log: Path):
    print(f"\n-> Training on: {input_csv}\n")
    
    # Load the data with progress tracking
    print("Preparing to read CSV...")
    with open(input_csv) as f:
        total_chunks = sum(1 for _ in f) // 1000 + 1
    
    print(f"Reading {total_chunks} chunks...")
    chunks = pd.read_csv(input_csv, chunksize=1000)
    df = pd.concat([chunk for chunk in tqdm(chunks, 
                                          total=total_chunks,
                                          desc="Loading",
                                          unit="chunks")])
    
    print(f"\nLoaded {len(df):,} rows from {total_chunks} chunks")
    
    # Validate required columns
    required_columns = {'Close', 'High', 'Low'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")
    
    # Feature computation
    features_list = []
    labels_list = []
    min_period = 20
    
    for i in range(min_period - 1, len(df)):
        close_history = df['Close'].iloc[i - min_period + 1 : i + 1].tolist()
        high_history = df['High'].iloc[i - min_period + 1 : i + 1].tolist()
        low_history = df['Low'].iloc[i - min_period + 1 : i + 1].tolist()
        
        raw_data = {
            "history": {
                "Close": close_history,
                "High": high_history,
                "Low": low_history
            }
        }
        
        try:
            f1, f2, f3 = compute_features(raw_data)
        except Exception as e:
            print(f"Skipping index {i} due to error: {e}")
            continue
        
        features_list.append([f1, f2, f3])
        label = 1 if f1 > 0 else 0
        labels_list.append(label)
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print("\nTotal training examples:", X.shape)
    print("Total labels:", y.shape)
    
    # Train/test split and model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Save model
    now = datetime.now().strftime("%d-%m-%y-%H%M%S")
    model_name = f"knn_model_{timeframe}min_{now}.pkl"
    model_path = model_folder / model_name
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"\nTrained model saved to {model_path}")
    
    # Log to model-list.csv
    header = not model_log.exists()
    with model_log.open('a', newline='') as lf:
        # writer = pd.io.common.csv.writer(lf)
        writer = csv.writer(lf)
        if header:
            writer.writerow(['Filename','Timeframe','Created_At'])
        writer.writerow([model_name, f"{timeframe}min", datetime.now().strftime("%d-%m-%y %H:%M:%S")])
    print(f"Appended to model log: {model_log}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--timeframe', type=int, required=True, help='Timeframe in minutes')
    p.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    args = p.parse_args()
    
    BASE = Path(__file__).parent.parent
    model_folder = BASE / 'models'
    model_folder.mkdir(exist_ok=True)
    model_log = model_folder / 'model-list.csv'
    
    main(args.timeframe, Path(args.input), model_folder, model_log)