#!/usr/bin/env python3
from pathlib import Path
from services.orderflow_data_utils import load_orderflow_ticks, scan_orderflow_tick_file_counts

def main():
    data_file = Path("data_raw/NQ_06-25_tick.txt")
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return

    print(f"Testing orderflow tick loader with first 1000 rows from: {data_file}")
    scan = scan_orderflow_tick_file_counts(data_file, max_rows=1000)
    print("\nRaw scan (pre-filter) on first 1000 rows:")
    for k, v in scan.items():
        print(f"- {k}: {v}")

    df = load_orderflow_ticks(data_file, max_rows=1000)
    print(f"\n✅ Loaded {len(df)} ticks")
    print("Columns:", list(df.columns))
    print("\nTop 5 widest spreads:")
    print(df.nlargest(5, 'Spread')[['Last','Bid','Ask','Spread']])

    wide_spread_count = (df['Spread'] >= 5).sum()
    print(f"\nRows with spread >= 5 ticks: {wide_spread_count}")

if __name__ == "__main__":
    main()
