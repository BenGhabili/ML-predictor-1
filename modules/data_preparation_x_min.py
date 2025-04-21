from pathlib import Path
import argparse
from services.data_utils import (
    load_nt8_ticks,
    ticks_to_bars,
    write_csv_and_log
)

def main(timeframe: int, atr_mult: float):
    root = Path(__file__).parent.parent
    raw_file = root / "data" / "nq_3feb_25mar_last.txt"   # adjust if needed
    out_dir  = root / "data"
    log_csv  = out_dir / "data-list.csv"

    print(f"Loading raw ticks from {raw_file} …")
    ticks_df = load_nt8_ticks(raw_file)

    print(f"Aggregating to {timeframe}-minute bars …")
    bars_df = ticks_to_bars(ticks_df, timeframe_min=timeframe, atr_mult=atr_mult, atr_period=14)

    write_csv_and_log(bars_df, timeframe, out_dir, log_csv, atr_mult)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate raw NT8 ticks into X‑minute bars with ATR labels."
    )
    parser.add_argument("--timeframe", type=int, default=1,
                        help="Candle size in minutes.")
    parser.add_argument("--atr-mult", type=float, default=1.0,   # ← NEW FLAG
                        help="ATR multiplier for up/down label (default 1.0)")
    args = parser.parse_args()

    main(args.timeframe, args.atr_mult)
