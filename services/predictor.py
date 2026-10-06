import numpy as np
from services.feature_helpers import compute_features


def predict_from_raw_data(raw_data, model=None):
    """
    Given raw_data of the form:
      {
        "history": {
          "Close": [...],
          "High":  [...],
          "Low":   [...]
        }
      }
    Computes features and returns a prediction dict.
    Raises ValueError if the input arrays are too short.

    Parameters
    ----------
    model : sklearn-compatible estimator (required).
    """
    if model is None:
        raise ValueError("No model provided.")

    history = raw_data.get("history", {})
    closes = history.get("Close", [])
    highs  = history.get("High", [])
    lows   = history.get("Low", [])

    # --- VALIDATION: require at least 20 closes and 1 high, 1 low ---
    REQUIRED_CLOSES = 20
    if len(closes) < REQUIRED_CLOSES or len(highs) < 1 or len(lows) < 1:
        raise ValueError(
            f"Insufficient data: require at least {REQUIRED_CLOSES} closing prices "
            f"and current high/low values."
        )

    # Compute your features (this should raise if anything else is off)
    f1, f2, f3 = compute_features(raw_data)

    # Construct a 2D array for scikit‑learn
    X = np.array([[f1, f2, f3]])

    if not hasattr(model, "predict_proba"):
        return {
            "prediction": int(model.predict(X)[0]),
            "probs": None,
            "warning": "Model doesn't support confidence scores"
        }

    try:
        proba = model.predict_proba(X)[0].tolist()
    except Exception:
        proba = None

    return {
        "prediction": int(model.predict(X)[0]),
        "probs": proba,
    }
