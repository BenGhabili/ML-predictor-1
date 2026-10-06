"""Load NinjaTrader text exports and aggregate into bars.

Supports:
  - Tick data  → N-tick bars (e.g. 2000-tick)
  - Tick data  → M-minute bars (e.g. 5-minute)
  - Pre-aggregated time bars (pass-through)

All prepare_* functions write NinjaTrader time-bar format:
    YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
so output files are auto-detected as 'time_bars' by detect_data_type().
"""

import os
import numpy as np
from datetime import datetime, timedelta


# ── Detection ──────────────────────────────────────────────────────────

def detect_data_type(filepath: str) -> str:
    """Detect whether a file contains tick data or pre-aggregated time bars.

    Returns 'tick' or 'time_bars'.

    NinjaTrader tick format (5 semicolon-delimited fields):
        YYYYMMDD HHMMSS FFFFFFF;Last;Bid;Ask;Volume

    NinjaTrader time-bar format (6 semicolon-delimited fields):
        YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
    """
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) == 5:
                return "tick"
            elif len(parts) >= 6:
                return "time_bars"
            break
    raise ValueError(f"Cannot detect data type: unrecognised format in {filepath}")


# ── Reading raw ticks ──────────────────────────────────────────────────

def _read_ticks(filepath: str):
    """Read tick file and return (dts, prices, volumes) lists."""
    tick_dts = []
    tick_prices = []
    tick_volumes = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 5:
                continue
            dt_main = parts[0][:15]  # "YYYYMMDD HHMMSS"
            dt = datetime.strptime(dt_main, "%Y%m%d %H%M%S")
            tick_dts.append(dt)
            tick_prices.append(float(parts[1]))
            tick_volumes.append(float(parts[4]))

    return tick_dts, tick_prices, tick_volumes


# ── Writing bars ───────────────────────────────────────────────────────

def _write_bars(output_path: str, datetimes, opens, highs, lows, closes, volumes) -> int:
    """Write OHLCV bars in NinjaTrader time-bar format. Returns bar count."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    bar_count = 0
    with open(output_path, "w") as out:
        for i in range(len(datetimes)):
            dt_str = datetimes[i].strftime("%Y%m%d %H%M%S")
            out.write(f"{dt_str};{opens[i]};{highs[i]};{lows[i]};{closes[i]};{volumes[i]}\n")
            bar_count += 1
    return bar_count


# ── Loading (into numpy — for backtester consumption) ──────────────────

def load_bars(filepath: str) -> dict:
    """Load pre-aggregated time bars from NT export.

    Format: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
    Returns dict of numpy arrays: dt, open, high, low, close, volume
    """
    datetimes = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 6:
                continue
            dt = datetime.strptime(parts[0], "%Y%m%d %H%M%S")
            datetimes.append(dt)
            opens.append(float(parts[1]))
            highs.append(float(parts[2]))
            lows.append(float(parts[3]))
            closes.append(float(parts[4]))
            volumes.append(float(parts[5]))

    return {
        "dt": np.array(datetimes, dtype="datetime64[s]"),
        "open": np.array(opens, dtype=np.float64),
        "high": np.array(highs, dtype=np.float64),
        "low": np.array(lows, dtype=np.float64),
        "close": np.array(closes, dtype=np.float64),
        "volume": np.array(volumes, dtype=np.float64),
    }


def load_tick_bars(filepath: str, bar_size: int) -> dict:
    """Load tick data and aggregate into N-tick bars on the fly.

    Returns same dict format as load_bars.
    """
    tick_dts, tick_prices, tick_volumes = _read_ticks(filepath)
    n_ticks = len(tick_prices)

    empty = {
        "dt": np.array([], dtype="datetime64[s]"),
        "open": np.array([], dtype=np.float64),
        "high": np.array([], dtype=np.float64),
        "low": np.array([], dtype=np.float64),
        "close": np.array([], dtype=np.float64),
        "volume": np.array([], dtype=np.float64),
    }
    if n_ticks == 0:
        return empty

    datetimes = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    for start in range(0, n_ticks, bar_size):
        end = min(start + bar_size, n_ticks)
        chunk_prices = tick_prices[start:end]
        chunk_vols = tick_volumes[start:end]

        datetimes.append(tick_dts[start])
        opens.append(chunk_prices[0])
        highs.append(max(chunk_prices))
        lows.append(min(chunk_prices))
        closes.append(chunk_prices[-1])
        volumes.append(sum(chunk_vols))

    return {
        "dt": np.array(datetimes, dtype="datetime64[s]"),
        "open": np.array(opens, dtype=np.float64),
        "high": np.array(highs, dtype=np.float64),
        "low": np.array(lows, dtype=np.float64),
        "close": np.array(closes, dtype=np.float64),
        "volume": np.array(volumes, dtype=np.float64),
    }


# ── Preparation (tick → file on disk) ─────────────────────────────────

def prepare_tick_bars_file(source_path: str, bar_size: int, output_path: str) -> int:
    """Aggregate tick data into N-tick bars and write to disk.

    Returns the number of bars written.
    """
    tick_dts, tick_prices, tick_volumes = _read_ticks(source_path)
    n_ticks = len(tick_prices)
    if n_ticks == 0:
        return 0

    datetimes = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    for start in range(0, n_ticks, bar_size):
        end = min(start + bar_size, n_ticks)
        chunk_prices = tick_prices[start:end]
        chunk_vols = tick_volumes[start:end]

        datetimes.append(tick_dts[start])
        opens.append(chunk_prices[0])
        highs.append(max(chunk_prices))
        lows.append(min(chunk_prices))
        closes.append(chunk_prices[-1])
        volumes.append(sum(chunk_vols))

    return _write_bars(output_path, datetimes, opens, highs, lows, closes, volumes)


def prepare_minute_bars_file(source_path: str, minutes: int, output_path: str) -> int:
    """Aggregate tick data into M-minute bars and write to disk.

    Groups ticks into calendar-aligned windows. For example, with minutes=5:
        09:30:00–09:34:59 → bar at 09:30:00
        09:35:00–09:39:59 → bar at 09:35:00

    Returns the number of bars written.
    """
    tick_dts, tick_prices, tick_volumes = _read_ticks(source_path)
    n_ticks = len(tick_prices)
    if n_ticks == 0:
        return 0

    interval = timedelta(minutes=minutes)

    datetimes = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    # Floor the first tick's datetime to the nearest interval boundary
    first_dt = tick_dts[0]
    bar_start = first_dt.replace(second=0) - timedelta(
        minutes=first_dt.minute % minutes
    )
    bar_end = bar_start + interval

    bar_prices = []
    bar_vols = []

    for i in range(n_ticks):
        dt = tick_dts[i]

        # If this tick falls outside the current window, flush the bar
        while dt >= bar_end:
            if bar_prices:
                datetimes.append(bar_start)
                opens.append(bar_prices[0])
                highs.append(max(bar_prices))
                lows.append(min(bar_prices))
                closes.append(bar_prices[-1])
                volumes.append(sum(bar_vols))
                bar_prices = []
                bar_vols = []
            bar_start = bar_end
            bar_end = bar_start + interval

        bar_prices.append(tick_prices[i])
        bar_vols.append(tick_volumes[i])

    # Flush the last bar
    if bar_prices:
        datetimes.append(bar_start)
        opens.append(bar_prices[0])
        highs.append(max(bar_prices))
        lows.append(min(bar_prices))
        closes.append(bar_prices[-1])
        volumes.append(sum(bar_vols))

    return _write_bars(output_path, datetimes, opens, highs, lows, closes, volumes)
