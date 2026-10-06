import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from joblib import load
from flask import Flask, request, jsonify
from services.predictor import predict_from_raw_data

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
                loaded_models[timeframe] = load(f)
            print(f"[OK] Loaded model for {timeframe}min timeframe")
        except Exception as e:
            print(f"[ERROR] Error loading {model_path}: {str(e)}")


def create_app():
    app = Flask(__name__)

    try:
        load_all_models()
        if not loaded_models:
            print("[WARNING] Warning: No models were loaded!")
    except Exception as e:
        print(f"[ERROR] Critical startup error: {str(e)}")

    @app.route('/ping', methods=['GET'])
    def ping():
        """Health check endpoint"""
        return jsonify({'status': 'ok'}), 200

    @app.route('/predict', methods=['POST'])
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
            if not isinstance(data, dict):
                return jsonify({'error': 'Invalid JSON', 'message': 'Expected a JSON object'}), 400
            timeframe = int(data.get('timeframe', 1))
            history = data.get('history', {})

            # Validate history arrays contain only numbers
            for key in ('Close', 'High', 'Low'):
                vals = history.get(key, [])
                if not isinstance(vals, list) or not all(isinstance(v, (int, float)) for v in vals):
                    return jsonify({'error': 'Invalid data', 'message': f'history.{key} must be a list of numbers'}), 400

            # 2) Validate model exists
            if timeframe not in loaded_models:
                return jsonify({
                    'error': 'Model not available',
                    'message': f'No model loaded for {timeframe}min timeframe'
                }), 404

            # 3) Predict (pass model directly for thread safety)
            wrapped = {"history": history}
            result = predict_from_raw_data(wrapped, model=loaded_models[timeframe])

            # label mapping: 0=flat/range, 1=down, 2=up
            response = {
                "prediction": result["prediction"],
                "direction": ["range", "down", "up"][result["prediction"]],
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
            }

            if data.get("bar_number") is not None:
                response["bar_number"] = data["bar_number"]
            if data.get("bar_time") is not None:
                response["bar_time"] = data["bar_time"]

            if result["probs"] is not None:
                response["probabilities"] = {
                    "range": result["probs"][0],
                    "down": result["probs"][1],
                    "up": result["probs"][2],
                }
                response["confidence"] = max(result["probs"])

            return jsonify(response), 200

        except ValueError as e:
            return jsonify({'error': 'Invalid data', 'message': str(e)}), 400
        except Exception as e:
            return jsonify({'error': 'Prediction failed', 'message': str(e)}), 500

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
