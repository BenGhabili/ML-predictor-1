import sys
from pathlib import Path
import pandas as pd
import pickle
from flask import Flask, request, jsonify
from services.predictor import predict_from_raw_data, set_model

app = Flask(__name__)

# Dictionary to hold all pre-loaded models: {timeframe: model}
loaded_models = {}

def load_all_models():
    """Load all available models at server startup"""
    models_dir = Path(__file__).parent.parent / "models"
    model_log = models_dir / "model-list.csv"
    
    if not model_log.exists():
        raise FileNotFoundError("Model log file not found")
    
    df = pd.read_csv(model_log)
    
    for _, row in df.iterrows():
        timeframe = int(row['Timeframe'].replace('min', ''))
        model_path = models_dir / row['Filename']
        
        try:
            with open(model_path, 'rb') as f:
                loaded_models[timeframe] = pickle.load(f)
            print(f"[OK] Loaded model for {timeframe}min timeframe")
        except Exception as e:
            print(f"[ERROR] Error loading {model_path}: {str(e)}")

# Initialize models when server starts
try:
    load_all_models()
    if not loaded_models:
        print("[WARNING] Warning: No models were loaded!")
except Exception as e:
    print(f"[ERROR] Critical startup error: {str(e)}")
    loaded_models = {}

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    return jsonify({'status': 'ok'}), 200

@app.route('/predict-knn', methods=['POST'])
def predict_knn():
    """
    Prediction endpoint
    
    Request format:
    {
      "timeframe": 1,  # minutes (1, 5, 15 etc)
      "history": {
        "Close": [ ... ],
        "High": [ ... ], 
        "Low": [ ... ]
      }
    }
    """
    try:
        # 1) Parse request
        data = request.get_json(force=True)
        timeframe = int(data.get('timeframe', 1))
        # raw_data = data['history']
        history   = data.get('history', {})
        
        # 2) Validate model exists
        if timeframe not in loaded_models:
            return jsonify({
                'error': 'Model not available',
                'message': f'No model loaded for {timeframe}min timeframe'
            }), 404
        
        # 3) Set model and predict (thread-safe)
        set_model(loaded_models[timeframe])

         # 4) Call your helper—but it expects the full dict with the "history" key,
        #    so wrap it back into that shape:
        wrapped = { "history": history }
        prediction = predict_from_raw_data(wrapped)

        # prediction = predict_from_raw_data(raw_data)
        
        return jsonify({'prediction': prediction}), 200
        
    except ValueError as e:
        return jsonify({'error': 'Invalid data', 'message': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)