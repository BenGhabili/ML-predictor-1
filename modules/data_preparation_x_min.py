from pathlib import Path
import pandas as pd
import sys
from datetime import datetime
import csv
import argparse  # For command-line argument parsing

def aggregate_to_x_min(raw_file: Path, output_folder: Path, log_file: Path, timeframe_minutes: int):
    """
    Reads the raw NT8 data from 'raw_file', aggregates the tick data into candles of a specified
    timeframe (in minutes), and writes the aggregated data to a uniquely-named CSV file in 'output_folder'.
    
    Additionally, it logs the generated filename along with the timeframe and creation timestamp 
    into 'log_file' as a CSV. This log file can later be used to reference the processed data files 
    along with additional metadata.
    
    Expected raw_file format (each line):
      20250206 000001 7240000;21803;21803;21804;1
    
    - The part before the first semicolon contains Date, Time, and Volume.
    - The remaining parts (semicolon-separated) are:
         Open, Low, Close, Flag.
    - Tick-level High is derived as the maximum of Open, Low, and Close.
    
    The output CSV will have columns: Datetime, Open, High, Low, Close, Volume.
    """
    data = []
    line_count = 0

    print("Starting to process raw file:", raw_file)
    with raw_file.open("r") as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if not line:
                continue

            parts = line.split(';')
            if len(parts) < 4:
                continue  # Skip invalid lines.

            header = parts[0].split()
            if len(header) < 2:
                continue
            date_str = header[0]       # e.g., "20250206"
            time_str = header[1]       # e.g., "000001"
            volume_str = header[2] if len(header) >= 3 else "0"

            try:
                dt = pd.to_datetime(f"{date_str} {time_str}", format='%Y%m%d %H%M%S')
            except Exception as e:
                sys.stdout.write(f"\nDatetime parse error on line {line_count}: {e}\n")
                continue

            try:
                open_price = float(parts[1].strip())
                low_price  = float(parts[2].strip())
                close_price = float(parts[3].strip())
            except Exception as e:
                sys.stdout.write(f"\nPrice parse error on line {line_count}: {e}\n")
                continue

            high_price = max(open_price, low_price, close_price)

            try:
                volume = int(volume_str)
            except Exception:
                volume = 0

            data.append({
                'Datetime': dt,
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })

            if line_count % 1000 == 0:
                sys.stdout.write(f"Processed {line_count} lines...\r")
                sys.stdout.flush()

    sys.stdout.write(f"\nFinished reading raw file. Total lines processed: {line_count}\n")
    if not data:
        raise ValueError("No valid data was found in the raw file.")

    print("Creating DataFrame from parsed data...")
    df = pd.DataFrame(data)
    df.sort_values('Datetime', inplace=True)
    df.set_index('Datetime', inplace=True)
    print(f"DataFrame created with {len(df)} rows.")

    print(f"Aggregating data into {timeframe_minutes}-minute candles...")
    # Create a resample frequency string using the timeframe_minutes parameter (e.g., '1T' for 1 minute)
    resample_freq = f"{timeframe_minutes}T"
    aggregated = df.resample(resample_freq).agg({
        'Open': 'first',  # Candle open: first open price within the timeframe
        'High': 'max',    # Candle high: maximum high price within the timeframe
        'Low': 'min',     # Candle low: minimum low price within the timeframe
        'Close': 'last',  # Candle close: last close price within the timeframe
        'Volume': 'sum'   # Total volume within the timeframe
    })
    aggregated.dropna(inplace=True)
    print(f"Aggregation complete. Total {timeframe_minutes}-minute bars: {len(aggregated)}")

    # Generate a unique filename using the timeframe and current timestamp.
    now = datetime.now()
    timestamp_str = now.strftime("%d-%m-%y-%H%M%S")
    filename = f"processed_data_{timeframe_minutes}min_{timestamp_str}.csv"
    output_path = output_folder / filename

    aggregated.to_csv(output_path)
    print(f"Aggregated data saved to {output_path}")

    # Append details to a CSV log file with columns: Filename, Timeframe, Created_At.
    file_exists = log_file.exists()
    with log_file.open("a", newline='') as lf:
        writer = csv.writer(lf)
        if not file_exists:
            writer.writerow(["Filename", "Timeframe", "Created_At"])
        writer.writerow([filename, f"{timeframe_minutes}min", now.strftime("%d-%m-%y %H:%M:%S")])
    print(f"Log updated in {log_file}")

if __name__ == "__main__":
    # Parse a command-line argument for the timeframe (default is 1)
    parser = argparse.ArgumentParser(description="Aggregate raw NT8 data into candles.")
    parser.add_argument("--timeframe", type=int, default=1, help="Aggregation timeframe in minutes (e.g., 1 for 1-minute bars)")
    args = parser.parse_args()

    raw_file = Path(__file__).parent.parent / "data" / "nq_3feb_25mar_last.txt"
    output_folder = Path(__file__).parent.parent / "data"
    log_file = output_folder / "data-list.csv"
    
    aggregate_to_x_min(raw_file, output_folder, log_file, args.timeframe)
