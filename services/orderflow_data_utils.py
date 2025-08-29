# services/orderflow_data_utils.py
"""
Orderflow-specific data utilities for handling tick-level bid/ask data
Format: yyyyMMdd HHmmss fffffff; last; bid; ask; volume
"""

from pathlib import Path
import re
import pandas as pd


def _infer_symbol_and_contract_from_filename(path: Path) -> tuple[str | None, str | None]:
    """Attempt to infer symbol_root and contract from filename like NQ_06-25_tick.txt"""
    stem = path.stem  # e.g., NQ_06-25_tick
    # Try patterns like: NQ_06-25_..., ES_12-24_...
    m = re.match(r"^(?P<sym>[A-Za-z]+)[_-](?P<contract>\d{2}-\d{2})", stem)
    if m:
        return m.group("sym").upper(), m.group("contract")
    # Fallback: try first token as symbol
    parts = re.split(r"[_-]", stem)
    if parts:
        return parts[0].upper(), None
    return None, None


def load_orderflow_ticks(raw_file: Path, max_rows: int = None) -> pd.DataFrame:
    """
    Parse tick data (last; bid; ask; volume) into a DataFrame indexed by Datetime.
    Adds spread, mid, last-minus-mid columns, ts_ms, and optional symbol metadata.
    """
    rows = []

    with raw_file.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            parts = line.split(';')
            if len(parts) < 5:
                continue

            head = parts[0].split()
            if len(head) < 2:
                continue
            date_str, time_str = head[:2]

            try:
                dt = pd.to_datetime(f"{date_str} {time_str}", format="%Y%m%d %H%M%S", utc=True)
                last = float(parts[1])
                bid = float(parts[2])
                ask = float(parts[3])
                volume = int(parts[4])

                # Basic validation
                if bid > ask or last <= 0 or bid <= 0 or ask <= 0:
                    continue

            except Exception:
                continue

            rows.append((dt, last, bid, ask, volume))

            if max_rows and len(rows) >= max_rows:
                break

    if not rows:
        raise ValueError("No valid rows parsed.")

    df = pd.DataFrame(rows, columns=["Datetime", "Last", "Bid", "Ask", "Volume"]).sort_values("Datetime")
    df.set_index("Datetime", inplace=True)

    # Derived
    df["Spread"] = df["Ask"] - df["Bid"]
    df["Mid"] = (df["Bid"] + df["Ask"]) / 2
    df["LastMid"] = df["Last"] - df["Mid"]

    # Schema enrichments
    df["ts_ms"] = (df.index.view("int64") // 1_000_000).astype("int64")
    sym, contract = _infer_symbol_and_contract_from_filename(raw_file)
    if sym is not None:
        df["symbol_root"] = sym
    if contract is not None:
        df["contract"] = contract

    return df


def scan_orderflow_tick_file_counts(raw_file: Path, max_rows: int = None) -> dict:
    """
    Quick pre-filter scan of raw file to count anomalies.
    Returns total_lines, parsed_lines, invalid_bid_gt_ask, invalid_nonpositive_prices.
    """
    total_lines = 0
    parsed_lines = 0
    invalid_bid_gt_ask = 0
    invalid_nonpositive = 0

    with raw_file.open() as fh:
        for line in fh:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            parts = line.split(';')
            if len(parts) < 5:
                continue

            head = parts[0].split()
            if len(head) < 2:
                continue

            try:
                last = float(parts[1])
                bid = float(parts[2])
                ask = float(parts[3])
                parsed_lines += 1

                if bid > ask:
                    invalid_bid_gt_ask += 1
                if last <= 0 or bid <= 0 or ask <= 0:
                    invalid_nonpositive += 1
            except Exception:
                continue

            if max_rows and parsed_lines >= max_rows:
                break

    return {
        "total_lines": total_lines,
        "parsed_lines": parsed_lines,
        "invalid_bid_gt_ask": invalid_bid_gt_ask,
        "invalid_nonpositive_prices": invalid_nonpositive,
    }
