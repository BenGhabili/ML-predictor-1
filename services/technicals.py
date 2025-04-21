# services/technical.py
import numpy as np
import pandas as pd

# -------------------------------------------------
# True Range  (vectorised for a DataFrame of bars)
# -------------------------------------------------
def true_range(high: pd.Series,
               low:  pd.Series,
               prev_close: pd.Series) -> pd.Series:
    """
    Vectorised True‑Range returned as a pandas Series
    (so we can call .rolling afterwards).
    """
    arr = np.maximum.reduce([
        high.values - low.values,
        np.abs(high.values - prev_close.values),
        np.abs(low.values  - prev_close.values)
    ])
    return pd.Series(arr, index=high.index)   # keep original DateTime index

# -------------------------------------------------
# ATR(n)  on an OHLC DataFrame
# -------------------------------------------------
def add_atr(df_bars: pd.DataFrame,
            period: int = 14,
            col_name: str = "ATR") -> pd.DataFrame:
    """
    Adds an ATR column (in points) and returns the same DataFrame.
    Expects columns High, Low, Close.
    """
    prev_close = df_bars["Close"].shift()
    tr_series = true_range(df_bars["High"], df_bars["Low"], prev_close)
    df_bars[col_name] = tr_series.rolling(period).mean()
    return df_bars

# -------------------------------------------------
# 3‑class label (2=up, 1=down, 0=neutral)
# -------------------------------------------------
def add_label_3class(df_bars: pd.DataFrame,
                     atr_mult: float = 1.0,
                     target_col: str = "label3",
                     atr_col: str = "ATR") -> pd.DataFrame:
    """
    Adds a label column based on next‑bar move vs ATR.
    """
    future_ret = df_bars["Close"].shift(-1) - df_bars["Close"]
    up   =  future_ret >=  atr_mult * df_bars[atr_col]
    down =  future_ret <= -atr_mult * df_bars[atr_col]
    df_bars[target_col] = np.select([up, down], [2, 1], default=0).astype(int)
    df_bars["future_ret"] = future_ret   # optional keep for inspection
    return df_bars
