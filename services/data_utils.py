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


# ──────────────────────────────────────────────────────────
def load_nt8_ticks(raw_file: Path) -> pd.DataFrame:
    """
    Parse the NT8 text export into a tick‑level DataFrame:
    columns = Datetime, Open, High, Low, Close, Volume
    """
    rows, line_cnt = [], 0
    with raw_file.open() as fh:
        for line in fh:
            line_cnt += 1
            line = line.strip()
            if not line:  # skip blanks
                continue

            parts = line.split(';')
            if len(parts) < 4:
                continue  # malformed

            # split first token into date time volume
            head = parts[0].split()
            if len(head) < 2:  # bad header
                continue
            date_str, time_str = head[:2]
            vol_str = head[2] if len(head) >= 3 else "0"

            try:
                dt = pd.to_datetime(f"{date_str} {time_str}",
                                    format="%Y%m%d %H%M%S")
                op, lo, cl = map(float, parts[1:4])
                hi = max(op, lo, cl)
                vol = int(vol_str)
            except Exception as exc:
                sys.stdout.write(f"\nParse error @ line {line_cnt}: {exc}\n")
                continue

            rows.append((dt, op, hi, lo, cl, vol))
            if line_cnt % 1000 == 0:
                sys.stdout.write(f"Parsed {line_cnt} lines...\r")

    sys.stdout.write(f"\nFinished. Total lines: {line_cnt}\n")
    if not rows:
        raise ValueError("No valid rows parsed.")
    df = pd.DataFrame(rows,
                      columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
    df.sort_values("Datetime", inplace=True)
    df.set_index("Datetime", inplace=True)
    return df


# ❶ Load NT8-style bar file (DATE TIME;O;H;L;C;V)
def load_nt8_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=';',
        header=None,
        names=['date_time', 'open', 'high', 'low', 'close', 'volume'],
        engine='python'
    )
    df['time'] = pd.to_datetime(df['date_time'], format='%Y%m%d %H%M%S')
    return df.drop(columns='date_time').set_index('time')


# ──────────────────────────────────────────────────────────
def ticks_to_bars(df_ticks: pd.DataFrame,
                  timeframe_min: int,
                  atr_period: int = 14,
                  atr_mult: float = 1.0) -> pd.DataFrame:
    """
    Resample ticks to OHLCV bars, add ATR, 3‑class label, and
    toy features (feature1/2/3).  Returns ready‑to‑save DataFrame.
    """
    rule = f"{timeframe_min}T"
    bars = df_ticks.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

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

# ❷ Add ATR column + 3-class label (up / flat / down)
def add_atr_labels(df: pd.DataFrame,
                   atr_mult: float = 1.0,
                   atr_period: int = 14,
                   k_ahead: int = 1) -> pd.DataFrame:
    """
    Add Average True Range (ATR) and a 3-class label column.
    Label: 1 = up, 0 = down, 2 = flat.
    """
    hi, lo, cl = df['high'], df['low'], df['close']

    # --- True-Range as a Series -------------------------------------------
    tr = pd.concat([
        hi - lo,
        (hi - cl.shift()).abs(),
        (cl.shift() - lo).abs()
    ], axis=1).max(axis=1)              # returns a pd.Series

    df['atr'] = tr.rolling(atr_period).mean()

    # --- Forward-return & label -------------------------------------------
    fwd_ret = cl.pct_change(periods=k_ahead).shift(-k_ahead)
    thresh  = (df['atr'] * atr_mult) / cl            # ATR → % move

    df['label'] = np.select(
        [fwd_ret >  thresh,
         fwd_ret < -thresh],
        [1, 0],                                     # up / down
        default=2                                   # flat
    )

    return df.dropna()

def engineer_bar_features2(df: pd.DataFrame,
                          atr_period: int = 14,
                          k_ahead: int = 1,
                          atr_mult: float = 1.0) -> pd.DataFrame:
    """
    Takes a DataFrame with columns:
        Datetime index + Open/High/Low/Close/Volume  (proper-case)
    Adds:
        ATR, label3 (up / down / flat), feature1–3
    Returns a cleaned DataFrame ready for write_csv_and_log().
    """

    # --- ATR ---------------------------------------------------------------
    df["ATR"] = ta.volatility.average_true_range(
        high=df["High"], low=df["Low"], close=df["Close"], window=atr_period
    )

    # --- 3-class label -----------------------------------------------------
    fwd_ret = df["Close"].pct_change(k_ahead).shift(-k_ahead)
    thresh  = (df["ATR"] * atr_mult) / df["Close"]

    df["label3"] = np.select(
        [fwd_ret >  thresh, fwd_ret < -thresh],
        [1,                  0],          # up / down
        default=2                         # flat
    )

    # --- Example placeholder features (match tick version!) ---------------
    df["feature1"] = df["Close"].pct_change()             # 1-bar return
    df["feature2"] = df["Close"].pct_change(5)            # 5-bar return
    df["feature3"] = df["Volume"].pct_change()            # volume delta

    return df.dropna()

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

    # 3. Generate labels
    out["label3"] = np.select(
        [pct_fwd > thr, pct_fwd < -thr],
        [1, 0],  # 1=up, 0=down
        default=2  # 2=flat
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

    # ---- NEW: drop rows where rolling indicators are NaN ----
    out = out.dropna(subset=feature_cols)

    # keep only the feature columns plus the label
    out = out[feature_cols + ["label3"]]
    
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
    rule = f"{freq_min}T"                         # 'T' = minute offset alias
    return (
        df.resample(rule)
        .agg({"Open": "first",
              "High": "max",
              "Low":  "min",
              "Close":"last",
              "Volume":"sum"})
        .dropna()
    )