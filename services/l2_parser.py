"""
services/l2_parser.py
Parse nrdtocsv output (L1/L2 records), maintain a 10-level order book,
and emit feature rows at each trade event.

Input format (semicolon-delimited, no header):
  L2;side;timestamp;offset_100ns;operation;position;marketmaker;price;size
  L1;type;timestamp;offset_100ns;price;volume

L2 side: 0=ask, 1=bid
L2 operation: 0=add, 1=update, 2=delete
L1 type: 0=ask, 1=bid, 2=last(trade), 4=open_interest, 5=volume_total
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path

DEPTH = 10  # book levels per side

# ── Order Book ───────────────────────────────────────────────────────

class OrderBook:
    """Maintain a 10-level bid/ask book from L2 events."""

    __slots__ = ("bid_px", "bid_sz", "ask_px", "ask_sz")

    def __init__(self):
        self.bid_px = np.zeros(DEPTH, dtype=np.float64)
        self.bid_sz = np.zeros(DEPTH, dtype=np.float64)
        self.ask_px = np.zeros(DEPTH, dtype=np.float64)
        self.ask_sz = np.zeros(DEPTH, dtype=np.float64)

    def update(self, side: int, op: int, pos: int, price: float, size: float):
        """Apply an L2 event to the book.
        side: 0=ask, 1=bid
        op: 0=add, 1=update, 2=delete
        pos: level index (0=best)
        """
        if pos >= DEPTH:
            return
        px = self.ask_px if side == 0 else self.bid_px
        sz = self.ask_sz if side == 0 else self.bid_sz

        if op == 0:  # add — insert at position, shift rest down
            px[pos + 1:] = px[pos:-1]
            sz[pos + 1:] = sz[pos:-1]
            px[pos] = price
            sz[pos] = size
        elif op == 1:  # update — change size (and price) at position
            px[pos] = price
            sz[pos] = size
        elif op == 2:  # delete — remove at position, shift rest up
            px[pos:-1] = px[pos + 1:]
            sz[pos:-1] = sz[pos + 1:]
            px[-1] = 0.0
            sz[-1] = 0.0

    @property
    def spread(self) -> float:
        if self.ask_px[0] > 0 and self.bid_px[0] > 0:
            return self.ask_px[0] - self.bid_px[0]
        return 0.0

    @property
    def mid(self) -> float:
        if self.ask_px[0] > 0 and self.bid_px[0] > 0:
            return (self.ask_px[0] + self.bid_px[0]) / 2.0
        return 0.0

    @property
    def microprice(self) -> float:
        """Volume-weighted mid price (biased toward thinner side)."""
        b0 = self.bid_sz[0]
        a0 = self.ask_sz[0]
        total = b0 + a0
        if total <= 0 or self.ask_px[0] <= 0 or self.bid_px[0] <= 0:
            return self.mid
        return (self.bid_px[0] * a0 + self.ask_px[0] * b0) / total

    def imbalance(self, levels: int) -> float:
        """(total_bid_size - total_ask_size) / (total_bid_size + total_ask_size) for top N levels."""
        n = min(levels, DEPTH)
        b = self.bid_sz[:n].sum()
        a = self.ask_sz[:n].sum()
        total = b + a
        if total <= 0:
            return 0.0
        return (b - a) / total

    def snapshot(self) -> dict:
        """Return all book-derived features as a dict."""
        return {
            "spread": self.spread,
            "mid": self.mid,
            "microprice": self.microprice,
            "best_bid_sz": self.bid_sz[0],
            "best_ask_sz": self.ask_sz[0],
            "depth_bid_5": self.bid_sz[:5].sum(),
            "depth_ask_5": self.ask_sz[:5].sum(),
            "depth_bid_10": self.bid_sz.sum(),
            "depth_ask_10": self.ask_sz.sum(),
            "imbalance_1": self.imbalance(1),
            "imbalance_5": self.imbalance(5),
            "imbalance_10": self.imbalance(10),
        }


# ── Flow Tracker ─────────────────────────────────────────────────────

class FlowTracker:
    """Track recent trades and book events for flow-based features."""

    def __init__(self, trade_window: int = 200, time_window_ns: int = 1_000_000_000):
        self.trade_window = trade_window
        self.time_window_ns = time_window_ns
        # recent trades: (timestamp_ns, signed_volume)
        self.trades = deque(maxlen=trade_window)
        # recent L2 events: (timestamp_ns, side, operation)
        self.l2_events = deque(maxlen=5000)

    def add_trade(self, ts_ns: int, price: float, volume: float, book: OrderBook):
        """Record a trade with actual aggressor inference from book."""
        if price >= book.ask_px[0] and book.ask_px[0] > 0:
            side = 1  # buy (lifted ask)
        elif price <= book.bid_px[0] and book.bid_px[0] > 0:
            side = -1  # sell (hit bid)
        else:
            side = 0  # between spread
        self.trades.append((ts_ns, side * volume, side, volume))

    def add_l2_event(self, ts_ns: int, side: int, op: int):
        """Record an L2 book change event."""
        self.l2_events.append((ts_ns, side, op))

    def snapshot(self, ts_ns: int) -> dict:
        """Compute flow features at current timestamp."""
        # trade-based features
        trades_list = list(self.trades)
        n_trades = len(trades_list)

        # cumulative delta from last N trades
        if n_trades >= 50:
            delta_50 = sum(t[1] for t in trades_list[-50:])
        else:
            delta_50 = sum(t[1] for t in trades_list)

        if n_trades >= 200:
            delta_200 = sum(t[1] for t in trades_list[-200:])
        else:
            delta_200 = sum(t[1] for t in trades_list)

        # last trade info
        if trades_list:
            last_signed_vol = trades_list[-1][1]
            last_aggressor = trades_list[-1][2]
            last_volume = trades_list[-1][3]
        else:
            last_signed_vol = 0.0
            last_aggressor = 0
            last_volume = 0.0

        # trade intensity in time window
        cutoff = ts_ns - self.time_window_ns
        trades_in_window = sum(1 for t in trades_list if t[0] >= cutoff)
        volume_in_window = sum(t[3] for t in trades_list if t[0] >= cutoff)

        # L2 event features in time window
        l2_list = list(self.l2_events)
        recent_l2 = [(s, op) for ts, s, op in l2_list if ts >= cutoff]

        bid_adds = sum(1 for s, op in recent_l2 if s == 1 and op == 0)
        ask_adds = sum(1 for s, op in recent_l2 if s == 0 and op == 0)
        bid_dels = sum(1 for s, op in recent_l2 if s == 1 and op == 2)
        ask_dels = sum(1 for s, op in recent_l2 if s == 0 and op == 2)

        total_adds = bid_adds + ask_adds
        total_dels = bid_dels + ask_dels

        # pull imbalance: more bid deletes = bearish (bids being pulled)
        pull_imbalance = (bid_dels - ask_dels) / max(total_dels, 1)
        # stack imbalance: more bid adds = bullish (bids being stacked)
        stack_imbalance = (bid_adds - ask_adds) / max(total_adds, 1)

        return {
            "aggressor": last_aggressor,
            "signed_volume": last_signed_vol,
            "trade_volume": last_volume,
            "delta_50": delta_50,
            "delta_200": delta_200,
            "trade_intensity_1s": trades_in_window,
            "volume_1s": volume_in_window,
            "pull_imbalance_1s": pull_imbalance,
            "stack_imbalance_1s": stack_imbalance,
            "book_adds_1s": total_adds,
            "book_dels_1s": total_dels,
        }


# ── Timestamp parsing ────────────────────────────────────────────────

def _parse_ts_ns(ts_str: str, offset_str: str) -> int:
    """Parse timestamp + 100ns offset into nanoseconds since epoch-ish.
    We don't need absolute time, just monotonic ordering + relative deltas.
    """
    # ts_str: "20260223050000" -> date + time
    # offset_str: "240000" -> 100-nanosecond ticks within the second
    # For speed, encode as: (days * 86400 + seconds) * 1e9 + offset * 100
    y = int(ts_str[0:4])
    m = int(ts_str[4:6])
    d = int(ts_str[6:8])
    H = int(ts_str[8:10])
    M = int(ts_str[10:12])
    S = int(ts_str[12:14])

    # simple day count (enough for ordering within a few months)
    day_num = y * 400 + m * 32 + d
    sec = day_num * 86400 + H * 3600 + M * 60 + S
    ns = sec * 1_000_000_000 + int(offset_str) * 100
    return ns


def _parse_ts_str_to_datetime(ts_str: str, offset_str: str) -> pd.Timestamp:
    """Parse timestamp string + offset directly to a pd.Timestamp."""
    us = int(offset_str) * 100 // 1000  # 100ns ticks → microseconds
    return pd.Timestamp(
        year=int(ts_str[0:4]), month=int(ts_str[4:6]), day=int(ts_str[6:8]),
        hour=int(ts_str[8:10]), minute=int(ts_str[10:12]), second=int(ts_str[12:14]),
        microsecond=us,
    )


# ── Main parser ──────────────────────────────────────────────────────

def parse_l2_file(
    csv_path: Path,
    exchange_tz: str = "America/Chicago",
) -> pd.DataFrame:
    """Parse a single nrdtocsv file and return a DataFrame of features at each trade."""

    book = OrderBook()
    flow = FlowTracker(trade_window=200, time_window_ns=1_000_000_000)

    rows = []
    trade_count = 0
    line_count = 0

    with open(csv_path) as fh:
        for line in fh:
            line_count += 1
            parts = line.rstrip('\n').split(';')

            if parts[0] == 'L2' and len(parts) >= 9:
                side = int(parts[1])
                ts_ns = _parse_ts_ns(parts[2], parts[3])
                op = int(parts[4])
                pos = int(parts[5])
                price = float(parts[7])
                size = float(parts[8])

                book.update(side, op, pos, price, size)
                flow.add_l2_event(ts_ns, side, op)

            elif parts[0] == 'L1' and len(parts) >= 6:
                l1_type = int(parts[1])
                ts_ns = _parse_ts_ns(parts[2], parts[3])
                price = float(parts[4])
                volume = float(parts[5])

                if l1_type == 2:  # trade
                    trade_count += 1
                    flow.add_trade(ts_ns, price, volume, book)

                    # emit feature row
                    row = {"ts_ns": ts_ns, "ts_str": parts[2], "ts_offset": parts[3],
                           "last": price, "volume": volume}
                    row.update(book.snapshot())
                    row.update(flow.snapshot(ts_ns))
                    rows.append(row)

            if line_count % 5_000_000 == 0:
                print(f"  parsed {line_count:,} lines, {trade_count:,} trades...")

    print(f"  Done: {line_count:,} lines, {trade_count:,} trades")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # build proper timestamps from stored strings
    dts = [_parse_ts_str_to_datetime(r, o) for r, o in zip(df["ts_str"], df["ts_offset"])]
    dts = pd.DatetimeIndex(dts)

    # time-of-day features
    hour_float = dts.hour + dts.minute / 60.0 + dts.second / 3600.0
    df["sin_time"] = np.sin(2 * np.pi * hour_float / 24.0)
    df["cos_time"] = np.cos(2 * np.pi * hour_float / 24.0)

    # set datetime index for labelling
    df.index = dts
    df.index.name = "Datetime"
    df.drop(columns=["ts_str", "ts_offset"], inplace=True)

    return df


# ── Feature selection ────────────────────────────────────────────────

L2_FEATURE_ORDER = [
    "spread",
    "microprice_vs_mid",
    "imbalance_1",
    "imbalance_5",
    "imbalance_10",
    "best_bid_sz",
    "best_ask_sz",
    "depth_bid_5",
    "depth_ask_5",
    "aggressor",
    "signed_volume",
    "delta_50",
    "delta_200",
    "trade_intensity_1s",
    "volume_1s",
    "pull_imbalance_1s",
    "stack_imbalance_1s",
    "book_adds_1s",
    "book_dels_1s",
    "sin_time",
    "cos_time",
]


def select_l2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and derive the final feature columns."""
    out = df.copy()

    # derived: microprice deviation from mid (directional pressure signal)
    out["microprice_vs_mid"] = out["microprice"] - out["mid"]

    return out[[c for c in L2_FEATURE_ORDER if c in out.columns]].copy()


# ── Bar aggregation ──────────────────────────────────────────────────

def aggregate_to_bars(df: pd.DataFrame, bar_seconds: int) -> pd.DataFrame:
    """Resample per-trade DataFrame into time bars with aggregated features.

    Input df must have a DatetimeIndex and columns from parse_l2_file().
    Returns a DataFrame with one row per bar, columns ready for select_bar_features().
    """
    rule = f"{bar_seconds}s"
    groups = df.resample(rule)

    result = pd.DataFrame()

    # Last price in bar (needed for labelling)
    result["last"] = groups["last"].last()
    result["spread"] = groups["spread"].last()
    result["mid"] = groups["mid"].last()
    result["microprice"] = groups["microprice"].last()

    # Bar OHLC dynamics (microstructure volatility)
    result["bar_range"] = groups["last"].max() - groups["last"].min()
    result["bar_return"] = groups["last"].last() - groups["last"].first()

    # Book state at bar close
    result["imbalance_1"] = groups["imbalance_1"].last()
    result["imbalance_5"] = groups["imbalance_5"].last()
    result["imbalance_10"] = groups["imbalance_10"].last()
    result["best_bid_sz"] = groups["best_bid_sz"].last()
    result["best_ask_sz"] = groups["best_ask_sz"].last()
    result["depth_bid_5"] = groups["depth_bid_5"].last()
    result["depth_ask_5"] = groups["depth_ask_5"].last()

    # Mean book state over bar (smoothed view)
    result["imbalance_1_mean"] = groups["imbalance_1"].mean()
    result["imbalance_5_mean"] = groups["imbalance_5"].mean()
    result["depth_bid_5_mean"] = groups["depth_bid_5"].mean()
    result["depth_ask_5_mean"] = groups["depth_ask_5"].mean()

    # Aggregated trade flow within bar
    result["net_delta"] = groups["signed_volume"].sum()
    result["volume_total"] = groups["trade_volume"].sum()
    result["trade_count"] = groups["last"].count()
    result["net_aggressor"] = groups["aggressor"].sum()

    # Rolling cumulative (last value in bar)
    result["delta_50"] = groups["delta_50"].last()
    result["delta_200"] = groups["delta_200"].last()

    # Event rates over bar
    result["trade_intensity_peak"] = groups["trade_intensity_1s"].max()
    result["volume_peak"] = groups["volume_1s"].max()
    result["pull_imbalance"] = groups["pull_imbalance_1s"].mean()
    result["stack_imbalance"] = groups["stack_imbalance_1s"].mean()
    result["book_adds"] = groups["book_adds_1s"].sum()
    result["book_dels"] = groups["book_dels_1s"].sum()

    # Time features
    result["sin_time"] = groups["sin_time"].last()
    result["cos_time"] = groups["cos_time"].last()

    # Drop empty bars (no trades in that interval)
    result = result.dropna(subset=["last"])

    # ── Rolling / temporal features (how the book is CHANGING) ──
    # These capture dynamics that static snapshots miss.

    # Imbalance momentum: how is order book pressure shifting?
    result["imb1_delta_3"] = result["imbalance_1"] - result["imbalance_1"].shift(3)
    result["imb5_delta_3"] = result["imbalance_5"] - result["imbalance_5"].shift(3)

    # Net delta momentum: is buying/selling pressure accelerating?
    result["delta_roll_6"] = result["net_delta"].rolling(6, min_periods=1).sum()
    result["delta_roll_12"] = result["net_delta"].rolling(12, min_periods=1).sum()

    # Volume/trade momentum: is activity picking up?
    vol_ma_6 = result["volume_total"].rolling(6, min_periods=1).mean()
    vol_ma_12 = result["volume_total"].rolling(12, min_periods=1).mean()
    result["volume_accel"] = vol_ma_6 / vol_ma_12.clip(lower=1)

    tc_ma_6 = result["trade_count"].rolling(6, min_periods=1).mean()
    tc_ma_12 = result["trade_count"].rolling(12, min_periods=1).mean()
    result["trade_accel"] = tc_ma_6 / tc_ma_12.clip(lower=1)

    # Price momentum: short-term trend via rolling bar returns
    result["return_roll_3"] = result["bar_return"].rolling(3, min_periods=1).sum()
    result["return_roll_6"] = result["bar_return"].rolling(6, min_periods=1).sum()

    # Spread regime: is spread widening or tightening?
    result["spread_ma_6"] = result["spread"].rolling(6, min_periods=1).mean()

    # Depth pressure shift: is one side thinning out?
    result["depth_shift_5"] = (
        result["depth_bid_5"].rolling(3, min_periods=1).mean()
        - result["depth_ask_5"].rolling(3, min_periods=1).mean()
    )

    # Book event momentum: are adds accelerating vs deletes?
    adds_roll = result["book_adds"].rolling(6, min_periods=1).sum()
    dels_roll = result["book_dels"].rolling(6, min_periods=1).sum()
    result["add_del_ratio"] = adds_roll / dels_roll.clip(lower=1)

    return result


L2_BAR_FEATURE_ORDER = [
    # Static bar features
    "spread",
    "microprice_vs_mid",
    "bar_range",
    "bar_return",
    "imbalance_1",
    "imbalance_5",
    "imbalance_10",
    "imbalance_1_mean",
    "imbalance_5_mean",
    "best_bid_sz",
    "best_ask_sz",
    "depth_bid_5",
    "depth_ask_5",
    "depth_bid_5_mean",
    "depth_ask_5_mean",
    "net_delta",
    "volume_total",
    "trade_count",
    "net_aggressor",
    "delta_50",
    "delta_200",
    "trade_intensity_peak",
    "volume_peak",
    "pull_imbalance",
    "stack_imbalance",
    "book_adds",
    "book_dels",
    # Temporal / rolling features
    "imb1_delta_3",
    "imb5_delta_3",
    "delta_roll_6",
    "delta_roll_12",
    "volume_accel",
    "trade_accel",
    "return_roll_3",
    "return_roll_6",
    "spread_ma_6",
    "depth_shift_5",
    "add_del_ratio",
    # Time
    "sin_time",
    "cos_time",
]


def select_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and derive the final feature columns for bar-level data."""
    out = df.copy()
    out["microprice_vs_mid"] = out["microprice"] - out["mid"]
    return out[[c for c in L2_BAR_FEATURE_ORDER if c in out.columns]].copy()
