# modules/data_preparation_x_min.py
from pathlib import Path
import argparse
import sys
from services.data_utils import (
    load_nt8_ticks,
    ticks_to_bars,
    write_csv_and_log,
    load_nt8_bars,
    engineer_bar_features,
    resample_ohlcv
)

def main(timeframe: int, atr_mult: float, raw_file: str, from_bars: bool):
    root = Path(__file__).parent.parent
    raw_dir  = root / "data_raw"
    # raw_file = root / "data" / "nq_3feb_25mar_last.txt"   # adjust if needed
    out_dir  = root / "data"
    log_csv  = out_dir / "data-list.csv"

    # ── 1) Resolve the raw-file path ────────────────────────────
    file_path = Path(raw_file)
    
    if not file_path.is_absolute():
        file_path = raw_dir / file_path
    if not file_path.exists():
        sys.exit(f" XX File not found: {file_path}")

    print(f"Using raw file: {file_path}")

    # ── A) When the raw file is already fixed-interval OHLCV bars ──────────
    if from_bars:
        print(f"Loading pre-aggregated bars from {file_path.name} …")
        bars_df = load_nt8_bars(file_path)        # ← returns lower-case cols
    
        # 1) Rename to proper case so downstream code is identical
        bars_df = bars_df.rename(columns={
            "open":   "Open",
            "high":   "High",
            "low":    "Low",
            "close":  "Close",
            "volume": "Volume"
        })
    
        # 2) Move the timestamp index into an explicit Datetime column
        bars_df = (
            bars_df
            .reset_index()                 # 'time' column
            .rename(columns={"time": "Datetime"})
            .set_index("Datetime")
        )

        if timeframe > 1:
            print(f"Resampling 1-minute bars to {timeframe}-minute …")
            bars_df = resample_ohlcv(bars_df, timeframe)

        # ── NEW: keep only London + NY core hours ─────────────────────────
        # here the index is already DatetimeIndex
        bars_df = bars_df.between_time("07:00", "20:30")
        # ------------------------------------------------------------------
        
        # 3) Engineer ATR, label3, feature1-3
        print("Engineering features, ATR & labels …")
        bars_df = engineer_bar_features(
            bars_df,
            atr_period=14,
            k_ahead=timeframe,      # horizon matches --timeframe flag
            atr_mult=atr_mult,
            debug=False
        )
         
    else:                   # ← When tick data provided
        print(f"Loading raw ticks from {raw_file} …")
        ticks_df = load_nt8_ticks(file_path)
        print(f"Aggregating to {timeframe}-minute bars …")
        bars_df = ticks_to_bars(ticks_df, timeframe_min=timeframe,atr_mult=atr_mult, atr_period=14)     
    
    write_csv_and_log(bars_df, timeframe, out_dir, log_csv, atr_mult)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate raw NT8 ticks into X‑minute bars with ATR labels."
    )
    parser.add_argument("--timeframe", type=int, default=1,
                        help="Candle size in minutes.")
    parser.add_argument("--atr-mult", type=float, default=1.0,
                        help="ATR multiplier for up/down label (default 1.0)")
    parser.add_argument("--from-bars", action="store_true",
                        help="Skip tick aggregation; input file already contains bars.")
    parser.add_argument("--raw-file",  required=True,     # now required
                        help="Filename *or* full path of the raw data file")
    args = parser.parse_args()

    main(args.timeframe, args.atr_mult, args.raw_file, args.from_bars)
