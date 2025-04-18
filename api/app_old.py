import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify
import pickle
from services.predictor import predict_from_raw_data, set_model

app = Flask(__name__)

# Load the trained model once at startup.

model_file = Path(__file__).parent.parent / "models" / "trained_model_knn.pkl"
with open(model_file, 'rb') as f:
    model = pickle.load(f)
set_model(model)
print("Trained model loaded and set successfully.")

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint."""
    return jsonify({'status': 'ok'}), 200

@app.route('/predict-knn', methods=['POST'])
def predict_knn():
    """
    Prediction endpoint.
    
    Expects a JSON payload with raw data in the format:
    {
      "history": {
         "Close": [list of at least 20 close prices],
         "High": [list of high prices],
         "Low": [list of low prices]
      }
    }
    """
    try:
        raw_data = request.get_json(force=True)
    except Exception as e:
        return jsonify({'error': 'Invalid JSON', 'message': str(e)}), 400

    try:
        prediction = predict_from_raw_data(raw_data)
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'message': str(e)}), 500

    return jsonify({'prediction': prediction}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
