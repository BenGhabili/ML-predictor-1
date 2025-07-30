# services/features.py
import numpy as np
import pandas as pd
import ta            # pip install ta  (RSI, ATR helpers)

# ------------------------------------------------------------------ #
#  Registry:  every indicator has a key and a function that          #
#  *mutates* the incoming DataFrame (adds 1-N new columns).          #
# ------------------------------------------------------------------ #

FEATURE_REGISTRY = {}


def register(name):
    """Decorator to add a builder to the global registry."""
    def _wrap(func):
        FEATURE_REGISTRY[name] = func
        return func
    return _wrap


# ---------- baseline indicators ------------------------------------ #
@register("legacy")
def add_feature123(df, **_):
    """
    Re-create feature1, feature2, feature3 for every row,
    but vectorised for speed (no Python loops).
    Matches the formulas in feature_helpers.py
    """

    # feature1: 3-bar rate of change
    df["feature1"] = df["Close"].pct_change(3)

    # feature2: (High − Low) / Low for *current* bar
    df["feature2"] = (df["High"] - df["Low"]) / df["Low"]

    # feature3: Close − SMA-20
    df["feature3"] = df["Close"] - df["Close"].rolling(20).mean()


@register("ret")
def add_returns(df: pd.DataFrame, *, windows=(1, 3, 10, 30), **_):
    for n in windows:
        df[f"ret_{n}"] = df["Close"].pct_change(n)

@register("atr")
def add_atr(df: pd.DataFrame, *, window=14, **_):
    atr = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=window
    )
    df["atr_norm"]  = atr / df["Close"]
    df["atr_slope"] = df["atr_norm"].diff(5)

@register("vol")
def add_volatility(df: pd.DataFrame, *, windows=(10, 30), **_):
    ret = df["Close"].pct_change()
    for w in windows:
        df[f"sigma_{w}"] = ret.rolling(w).std()

@register("rsi")
def add_rsi(df: pd.DataFrame, *, window=14, **_):
    df["rsi_14"] = ta.momentum.rsi(df["Close"], window=window)

@register("vol_z")
def add_volume_z(df: pd.DataFrame, *, window=10, **_):
    roll = df["Volume"].rolling(window)
    df["vol_z10"] = (df["Volume"] - roll.mean()) / roll.std()

@register("tod")
def add_time_of_day(df: pd.DataFrame, **_):
    h = df.index.hour + df.index.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)

# ---------- exotic examples (just add more) ------------------------ #

@register("willr")
def add_williams_r(df: pd.DataFrame, *, window=14, **_):
    hh = df["High"].rolling(window).max()
    ll = df["Low"].rolling(window).min()
    df[f"willr_{window}"] = -100 * (hh - df["Close"]) / (hh - ll + 1e-9)
