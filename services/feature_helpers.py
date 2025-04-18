# services/feature_helpers.py

import pandas as pd

def compute_feature1(close_prices, period=3):
    """
    Compute the 3-period Rate of Change (ROC) for the provided close prices.
    Returns the percentage change over the specified period.
    """
    series = pd.Series(close_prices)
    # Calculate percent change and take the last value
    roc = series.pct_change(periods=period).iloc[-1]
    return roc

def compute_feature2(high_prices, low_prices):
    """
    Compute a volatility measure using the current bar's high and low.
    Formula: (High - Low) / Low
    """
    current_high = high_prices[-1]
    current_low = low_prices[-1]
    if current_low == 0:
        return 0.0
    return (current_high - current_low) / current_low

def compute_feature3(close_prices, period=20):
    """
    Compute the difference between the current close and the 20-period SMA of close prices.
    Returns the deviation.
    """
    series = pd.Series(close_prices)
    sma = series.rolling(window=period).mean().iloc[-1]
    if pd.isna(sma):
        return 0.0
    return close_prices[-1] - sma

def compute_features(raw_data):
    """
    Compute all three features given a raw data dictionary.
    
    Expected raw_data structure:
    {
      "history": {
         "Close": [list of close prices],
         "High": [list of high prices],
         "Low": [list of low prices]
      }
    }
    
    Returns a tuple: (Feature1, Feature2, Feature3)
    """
    history = raw_data.get("history", {})
    close_prices = history.get("Close", [])
    high_prices = history.get("High", [])
    low_prices = history.get("Low", [])
    
    # Verify there are enough data points: we need at least 20 close prices for Feature3
    if len(close_prices) < 20 or not high_prices or not low_prices:
        raise ValueError("Insufficient data: require at least 20 closing prices and current high/low values.")
    
    f1 = compute_feature1(close_prices, period=3)
    f2 = compute_feature2(high_prices, low_prices)
    f3 = compute_feature3(close_prices, period=20)
    
    return f1, f2, f3
