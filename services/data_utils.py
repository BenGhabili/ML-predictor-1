# services/data_utils.py
"""
Generic helpers for converting raw NinjaTrader tick export
into N‑minute bars, adding ATR, three‑class label, and features.
"""

from pathlib import Path
from datetime import datetime
import sys, csv
import numpy as np
import pandas as pd
import ta
from services.technicals import add_atr, add_label_3class, true_range
from services.features import FEATURE_REGISTRY
from services.nt8_loader import (
    detect_data_type,
    load_bars as _load_bars_np,
    _read_ticks,
)


# ──────────────────────────────────────────────────────────
def load_nt8_ticks(raw_file: Path) -> pd.DataFrame:
    """Parse NT8 tick export into a tick-level DataFrame (Last, Volume)."""
    dts, prices, volumes = _read_ticks(str(raw_file))
    if not dts:
        raise ValueError("No valid rows parsed.")
    df = pd.DataFrame(
        {"Last": prices, "Volume": volumes},
        index=pd.DatetimeIndex(dts, name="Datetime"),
    )
    df.sort_index(inplace=True)
    return df


def load_nt8_bars(path: Path) -> pd.DataFrame:
    """Load NT8-style bar file (DATE TIME;O;H;L;C;V) → DataFrame."""
    bars = _load_bars_np(str(path))
    df = pd.DataFrame(
        {
            "Open": bars["open"],
            "High": bars["high"],
            "Low": bars["low"],
            "Close": bars["close"],
            "Volume": bars["volume"],
        },
        index=pd.DatetimeIndex(bars["dt"], name="Datetime"),
    )
    return df


# ──────────────────────────────────────────────────────────
def ticks_to_bars(df_ticks: pd.DataFrame,
                  timeframe_min: int,
                  atr_period: int = 14,
                  atr_mult: float = 1.0) -> pd.DataFrame:
    """
    Resample ticks to OHLCV bars, add ATR, 3‑class label, and
    toy features (feature1/2/3).  Returns ready‑to‑save DataFrame.
    """
    rule = f"{timeframe_min}min"
    bars = df_ticks.resample(rule).agg({
        "Last": ["first", "max", "min", "last"],
        "Volume": "sum",
    }).dropna()
    bars.columns = ["Open", "High", "Low", "Close", "Volume"]

    # indicators + label
    bars = add_atr(bars, period=atr_period, col_name="ATR")
    bars = add_label_3class(bars,
                            atr_mult=atr_mult,
                            target_col="label3",
                            atr_col="ATR")

    bars = bars.dropna(subset=["ATR", "future_ret"])

    # simple engineered features
    bars["feature1"] = bars["Close"].pct_change(3)  # ROC(3)
    bars["feature2"] = bars["Volume"].pct_change(3)  # Vol‑ROC(3)
    bars["feature3"] = (bars["Close"] -
                        bars["Close"].rolling(20).mean())  # dist‑from‑SMA20
    bars = bars.dropna()

    # tidy order & reset index
    cols = ["Datetime", "Open", "High", "Low", "Close", "Volume",
            "ATR", "label3", "feature1", "feature2", "feature3"]
    bars = bars.reset_index()[cols]
    return bars


# ──────────────────────────────────────────────────────────
def write_csv_and_log(bars: pd.DataFrame,
                      timeframe_min: int,
                      output_folder: Path,
                      log_file: Path, atr_mult: float) -> Path:
    """
    Save bars to CSV with unique timestamped name and log entry.
    Returns the Path of the written CSV.
    """
    now = datetime.now()
    ts = now.strftime("%d-%m-%y-%H%M%S")
    fname = f"processed_data_{timeframe_min}min_atr{atr_mult:g}_{ts}.csv"
    out_path = output_folder / fname

    bars.to_csv(out_path, index=False)
    print("Saved:", out_path)

    # append to data‑list CSV
    file_exists = log_file.exists()
    with log_file.open("a", newline="") as lf:
        wr = csv.writer(lf)
        if not file_exists:
            wr.writerow(["Filename", "Timeframe", "ATR_Mult", "Created_At"])
        wr.writerow([fname, f"{timeframe_min}min", atr_mult,
                     now.strftime("%d-%m-%y %H:%M:%S")])
    return out_path

def engineer_bar_features(
        df,
        *,
        feature_list=None,          # ← default becomes None
        atr_period=14,
        k_ahead=1,
        atr_mult=1.0,
        debug=False):
    
    out = df.copy()

    # ---------- feature assembly ------------------------------------
    if feature_list is None:
        feature_list = list(FEATURE_REGISTRY.keys())

    for key in feature_list:
        FEATURE_REGISTRY[key](out)


    # ---------- Proper Label Generation -----------------------------
    # 1. Calculate True ATR first
    out = add_atr(out, period=atr_period)  # Using the true_range function

    # 2. Calculate forward returns and thresholds
    pct_fwd = out["Close"].pct_change(k_ahead).shift(-k_ahead)
    thr = (out["ATR"] * atr_mult) / out["Close"]  # Threshold as % of price

    # 3. Generate labels — encoding: 0=flat, 1=down, 2=up
    #    (must match technicals.add_label_3class and api/app.py mapping)
    out["label3"] = np.select(
        [pct_fwd >= thr, pct_fwd <= -thr],
        [2, 1],  # 2=up, 1=down
        default=0  # 0=flat
    )

    # ---------- Debug Output ----------------------------------------
    if debug:
        print("\n=== LABEL GENERATION DEBUG ===")
        print(f"ATR Stats | Min: {out['ATR'].min():.4f} | "
              f"Median: {out['ATR'].median():.4f} | "
              f"Max: {out['ATR'].max():.4f}")
        print(f"Threshold Stats | Min: {thr.min()*100:.6f}% | "
              f"Median: {thr.median()*100:.6f}% | "
              f"Max: {thr.max()*100:.6f}%")
        print("Label Distribution:")
        print(out["label3"].value_counts(normalize=True).sort_index())

    # ---------- Feature Filtering -----------------------------------
    feature_cols = [
        c for c in out.columns
        if c.startswith((
            "ret_", "atr_", "sigma_", "rsi_", "vol_", "hour_", "feature", "willr_"
        ))
    ]

    essential_features = [
        'ret_3',        # Short-term momentum  
        'ret_10',       # Medium-term momentum  
        'atr_norm',     # Normalized volatility  
        'rsi_14',       # Overbought/oversold  
        'vol_z10',      # Volume spikes  
        'hour_sin',     # Session timing  
        'sigma_10'      # Recent volatility  
    ]

    # ---- NEW: drop rows where rolling indicators are NaN ----
    out = out.dropna(subset=essential_features)

    # keep only the feature columns plus the label
    out = out[essential_features + ["label3"]]

    print("Feature Names:", list(out.columns))

    return out


def resample_ohlcv(df: pd.DataFrame, freq_min: int) -> pd.DataFrame:
    """
    Upscale/aggregate fixed-interval bars to a higher timeframe.

    Parameters
    ----------
    df        : Datetime-indexed DataFrame with Open/High/Low/Close/Volume
    freq_min  : Target bar size in minutes (e.g. 5, 15)

    Returns
    -------
    DataFrame resampled to freq_min with standard OHLCV aggregation.
    """
    rule = f"{freq_min}min"
    return (
        df.resample(rule)
        .agg({"Open": "first",
              "High": "max",
              "Low":  "min",
              "Close":"last",
              "Volume":"sum"})
        .dropna()
    )