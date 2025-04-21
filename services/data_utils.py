"""
Generic helpers for converting raw NinjaTrader tick export
into N‑minute bars, adding ATR, three‑class label, and features.
"""

from pathlib import Path
from datetime import datetime
import sys, csv
import numpy as np
import pandas as pd
from services.technicals import add_atr, add_label_3class   # earlier helper file

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
            if not line:                       # skip blanks
                continue

            parts = line.split(';')
            if len(parts) < 4:
                continue                       # malformed

            # split first token into date time volume
            head = parts[0].split()
            if len(head) < 2:                  # bad header
                continue
            date_str, time_str = head[:2]
            vol_str  = head[2] if len(head) >= 3 else "0"

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
                      columns=["Datetime","Open","High","Low","Close","Volume"])
    df.sort_values("Datetime", inplace=True)
    df.set_index("Datetime", inplace=True)
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
    rule = f"{timeframe_min}T"
    bars = df_ticks.resample(rule).agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
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
    bars["feature1"] = bars["Close"].pct_change(3)          # ROC(3)
    bars["feature2"] = bars["Volume"].pct_change(3)         # Vol‑ROC(3)
    bars["feature3"] = (bars["Close"] -
                        bars["Close"].rolling(20).mean())   # dist‑from‑SMA20
    bars = bars.dropna()

    # tidy order & reset index
    cols = ["Datetime","Open","High","Low","Close","Volume",
            "ATR","label3","feature1","feature2","feature3"]
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
    ts  = now.strftime("%d-%m-%y-%H%M%S")
    fname = f"processed_data_{timeframe_min}min_atr{atr_mult:g}_{ts}.csv"
    out_path = output_folder / fname

    bars.to_csv(out_path, index=False)
    print("Saved:", out_path)

    # append to data‑list CSV
    file_exists = log_file.exists()
    with log_file.open("a", newline="") as lf:
        wr = csv.writer(lf)
        if not file_exists:
            wr.writerow(["Filename","Timeframe", "ATR_Mult", "Created_At"])
        wr.writerow([fname, f"{timeframe_min}min", atr_mult,
                     now.strftime("%d-%m-%y %H:%M:%S")])
    return out_path
