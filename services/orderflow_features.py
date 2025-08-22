# services/orderflow_features.py
from __future__ import annotations
import numpy as np
import pandas as pd

# Canonical order for orderflow features used across training and runtime
OF_FEATURE_ORDER = [
    "Spread",
    "last_minus_mid",
    "aggressor",
    "signed_size",
    "cdelta_50",
    "cdelta_100",
    "ti_count_500ms",
    "ret_1s",
    "vol_2s",
    "spread_z_60s",
    "sin_time",
    "cos_time",
]

BUY = 1
SELL = -1
NONE = 0


def infer_aggressor_side(df: pd.DataFrame) -> pd.Series:
    last = df["Last"].to_numpy()
    bid = df["Bid"].to_numpy()
    ask = df["Ask"].to_numpy()

    side = np.zeros(len(df), dtype=np.int8)
    side[last >= ask] = BUY
    side[last <= bid] = SELL
    return pd.Series(side, index=df.index, name="aggressor")


def add_trade_intensity_features(df: pd.DataFrame, *, windows_ms=(250, 500, 1000)) -> pd.DataFrame:
    for ms in windows_ms:
        window = f"{int(ms)}ms"
        df[f"ti_count_{ms}ms"] = df["Last"].rolling(window).count()
        df[f"ti_vol_{ms}ms"] = df["Volume"].rolling(window).sum()
    return df


def add_cumulative_delta_features(df: pd.DataFrame, *, windows_trades=(25, 50, 100)) -> pd.DataFrame:
    if "aggressor" not in df.columns:
        df["aggressor"] = infer_aggressor_side(df)
    signed_size = (df["Volume"].astype(float) * df["aggressor"].astype(float)).rename("signed_size")
    df["signed_size"] = signed_size
    for n in windows_trades:
        df[f"cdelta_{n}"] = signed_size.rolling(n, min_periods=1).sum()
    return df


def add_price_position_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Spread" not in df.columns:
        df["Spread"] = df["Ask"] - df["Bid"]
    if "Mid" not in df.columns:
        df["Mid"] = (df["Bid"] + df["Ask"]) / 2

    df["last_minus_mid"] = df["Last"] - df["Mid"]
    return df


def add_time_based_returns_and_vol(df: pd.DataFrame) -> pd.DataFrame:
    # time-based windows using index; assumes DatetimeIndex with tz
    df = df.copy()
    df["last_lag_1s"] = df["Last"].rolling("1s").apply(lambda x: x.iloc[0], raw=False)
    df["last_lag_2s"] = df["Last"].rolling("2s").apply(lambda x: x.iloc[0], raw=False)

    df["ret_1s"] = df["Last"] - df["last_lag_1s"]
    df["ret_2s"] = df["Last"] - df["last_lag_2s"]

    df["vol_2s"] = df["Last"].rolling("2s").std()
    df["vol_5s"] = df["Last"].rolling("5s").std()
    return df


def add_spread_zscore(df: pd.DataFrame, *, window_s: int = 60) -> pd.DataFrame:
    roll = df["Spread"].rolling(f"{int(window_s)}s")
    mean = roll.mean()
    std = roll.std()
    df["spread_z_60s"] = (df["Spread"] - mean) / (std + 1e-9)
    return df


def add_clock_features(df: pd.DataFrame, *, exchange_tz: str = "America/Chicago") -> pd.DataFrame:
    # Compute sin/cos using exchange time
    idx = df.index.tz_convert(exchange_tz)
    hour_float = idx.hour + idx.minute / 60.0 + idx.second / 3600.0
    df["sin_time"] = np.sin(2 * np.pi * hour_float / 24.0)
    df["cos_time"] = np.cos(2 * np.pi * hour_float / 24.0)
    return df


def build_orderflow_features(
    df: pd.DataFrame,
    *,
    windows_trades=(25, 50, 100),
    windows_ms=(250, 500, 1000),
    exchange_tz: str = "America/Chicago",
) -> pd.DataFrame:
    df = add_price_position_features(df)
    df["aggressor"] = infer_aggressor_side(df)
    df = add_cumulative_delta_features(df, windows_trades=windows_trades)
    df = add_trade_intensity_features(df, windows_ms=windows_ms)
    df = add_time_based_returns_and_vol(df)
    df = add_spread_zscore(df, window_s=60)
    df = add_clock_features(df, exchange_tz=exchange_tz)
    return df


def select_compact_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Spread",
        "last_minus_mid",
        "aggressor",
        "signed_size",
        "cdelta_50",
        "cdelta_100",
        "ti_count_500ms",
        "ret_1s",
        "vol_2s",
        "spread_z_60s",
        "sin_time",
        "cos_time",
    ]
    return df[[c for c in cols if c in df.columns]].copy()
