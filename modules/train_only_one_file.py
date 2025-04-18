import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from tqdm import tqdm
from services.feature_helpers import compute_features
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Define the path to the processed CSV file relative to the project root.

csv_file = Path(__file__).parent.parent / "data" / "processed_data_1min_15-04-25-200654.csv"

# Load the processed CSV file into a DataFrame.env

print("Preparing to read CSV...")
with open(csv_file) as f:
    total_chunks = sum(1 for _ in f) // 1000 + 1  # Estimate chunks

# Now read with progress
print(f"Reading {total_chunks} chunks...")
chunks = pd.read_csv(csv_file, chunksize=1000)
df = pd.concat([chunk for chunk in tqdm(chunks, 
                                      total=total_chunks,
                                      desc="Loading",
                                      unit="chunks")])

print("\nCSV file read successfully!")  # \n ensures new line
print(f"Loaded {len(df):,} rows from {total_chunks} chunks")

# Ensure the essential columns exist.
required_columns = {'Close', 'High', 'Low'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV file must contain columns: {required_columns}")

# Create lists to store feature vectors and labels.
features_list = []
labels_list = []

# We need at least 20 bars to compute the 20-period SMA.
min_period = 20

# Loop over the DataFrame starting at row index 19 (which gives 20 rows: 0-19)
for i in range(min_period - 1, len(df)):
    # Get the latest 'min_period' rows for this training example.
    close_history = df['Close'].iloc[i - min_period + 1 : i + 1].tolist()
    high_history = df['High'].iloc[i - min_period + 1 : i + 1].tolist()
    low_history = df['Low'].iloc[i - min_period + 1 : i + 1].tolist()
    
    # Build the raw data dictionary expected by our helper functions.
    raw_data = {
        "history": {
            "Close": close_history,
            "High": high_history,
            "Low": low_history
        }
    }
    
    try:
        # Compute features using our helper function.
        f1, f2, f3 = compute_features(raw_data)
    except Exception as e:
        print(f"Skipping index {i} due to error: {e}")
        continue  # Skip this row if features cannot be computed.
    
    features_list.append([f1, f2, f3])
    
    # Create a dummy label: 1 if Feature1 is positive, else 0.
    label = 1 if f1 > 0 else 0
    labels_list.append(label)

# Convert lists to numpy arrays.
X = np.array(features_list)
y = np.array(labels_list)

print("Total training examples:", X.shape)
print("Total labels:", y.shape)

# Split the dataset into training and testing sets (80% training, 20% testing).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a KNeighborsClassifier.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("Classifier created ...")

# Evaluate the model on the test set.
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Define the path where the model will be saved.

# model_save_path_1 = '../models/trained_model_knn.pkl'

print("Saving the trained model into a file ...")

model_save_path = Path(__file__).parent.parent / "models" / "trained_model_knn.pkl"

# Save the trained model using pickle.
with open(model_save_path, 'wb') as file:
    pickle.dump(knn, file)
print(f"Trained model saved to {model_save_path}")
