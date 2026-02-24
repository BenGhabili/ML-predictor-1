"""Tests for bar-based labels (technicals.add_label_3class) and
orderflow tick-level labels (orderflow_labels.make_labels_multi)."""

import numpy as np
import pandas as pd
import pytest

from services.technicals import add_label_3class, add_atr
from services.orderflow_labels import make_labels_multi


# ── helpers ──────────────────────────────────────────────────

def _make_bars(closes, atr_values):
    """Build a minimal OHLC DataFrame with a pre-set ATR column."""
    n = len(closes)
    df = pd.DataFrame({
        "Open": closes,
        "High": [c + 1 for c in closes],
        "Low": [c - 1 for c in closes],
        "Close": closes,
    })
    df["ATR"] = atr_values
    return df


def _make_tick_df(prices, ts_ms_list, spreads=None):
    """Build a tick DataFrame with DatetimeIndex (UTC)."""
    idx = pd.to_datetime(ts_ms_list, unit="ms", utc=True)
    df = pd.DataFrame({
        "Last": prices,
        "Bid": [p - 0.125 for p in prices],
        "Ask": [p + 0.125 for p in prices],
        "Spread": spreads if spreads is not None else [0.25] * len(prices),
        "Volume": [1] * len(prices),
    }, index=idx)
    return df


# ── add_label_3class ─────────────────────────────────────────

class TestAddLabel3class:
    def test_units_are_points(self):
        """Threshold must be in points (ATR * mult), not ATR/Close."""
        closes = [100.0, 110.0, 100.0, 90.0, 100.0]
        atr = [5.0] * 5
        df = _make_bars(closes, atr)
        df = add_label_3class(df, atr_mult=1.0)

        # future_ret bar 0 = 110 - 100 = +10  ≥ thresh 5 → up (2)
        assert df["label3"].iloc[0] == 2
        # future_ret bar 1 = 100 - 110 = -10  ≤ -5 → down (1)
        assert df["label3"].iloc[1] == 1

    def test_flat_when_within_threshold(self):
        closes = [100.0, 101.0, 100.0]
        atr = [5.0] * 3
        df = _make_bars(closes, atr)
        df = add_label_3class(df, atr_mult=1.0)
        # future_ret = +1, thresh = 5 → flat (0)
        assert df["label3"].iloc[0] == 0

    def test_atr_mult_scales_threshold(self):
        closes = [100.0, 103.0, 100.0]
        atr = [2.0] * 3
        df = _make_bars(closes, atr)

        # mult=1 → thresh=2, ret=+3 → up
        df1 = add_label_3class(df.copy(), atr_mult=1.0)
        assert df1["label3"].iloc[0] == 2

        # mult=2 → thresh=4, ret=+3 → flat
        df2 = add_label_3class(df.copy(), atr_mult=2.0)
        assert df2["label3"].iloc[0] == 0


# ── make_labels_multi ────────────────────────────────────────

class TestMakeLabelsMulti:
    COST_KWARGS = dict(
        tick_size=0.25,
        tick_value_usd=5.0,
        fees_roundtrip_usd=0.0,
        slippage_ticks_per_side=0.0,
        entry_delay_ms=0,
        spread_cap_ticks=100,
    )

    def test_reference_price_is_post_delay(self):
        """p0 must be the first price after entry delay, not the signal price."""
        # Signal at t=0 price=100, delay=100ms, first post-delay at t=150ms price=102
        prices = [100.0, 102.0, 105.0, 95.0]
        ts = [0, 150, 300, 450]
        ts_ms = ts
        df = _make_tick_df(prices, ts_ms)

        labels = make_labels_multi(
            df,
            targets_net_ticks=[1],
            horizons_s=[1],
            entry_delay_ms=100,
            tick_size=0.25,
            tick_value_usd=5.0,
            fees_roundtrip_usd=0.0,
            slippage_ticks_per_side=0.0,
            spread_cap_ticks=100,
        )
        lbl = list(labels.values())[0]
        # reference price for signal 0 should be 102 (prices[j0]), not 100
        # with target 1 tick gross (no costs), delta=0.25
        # 105 ≥ 102+0.25, so up; 95 ≤ 102-0.25, so down
        # up hit first (at t=300) → label=2 (up)
        assert lbl.iloc[0] == 2

    def test_flat_when_price_stays_still(self):
        """No move → label 0."""
        prices = [100.0] * 5
        ts = list(range(0, 500, 100))
        df = _make_tick_df(prices, ts)

        labels = make_labels_multi(
            df,
            targets_net_ticks=[1],
            horizons_s=[1],
            **self.COST_KWARGS,
        )
        lbl = list(labels.values())[0]
        # no price movement → all flat
        assert (lbl == 0).all()

    def test_spread_cap_filters(self):
        """Ticks with wide spreads get label 0."""
        prices = [100.0, 100.0, 110.0]
        ts = [0, 100, 200]
        spreads = [5.0, 0.25, 0.25]  # first tick has wide spread
        df = _make_tick_df(prices, ts, spreads=spreads)

        labels = make_labels_multi(
            df,
            targets_net_ticks=[1],
            horizons_s=[1],
            tick_size=0.25,
            tick_value_usd=5.0,
            fees_roundtrip_usd=0.0,
            slippage_ticks_per_side=0.0,
            entry_delay_ms=0,
            spread_cap_ticks=2,  # cap = 2 * 0.25 = 0.5
        )
        lbl = list(labels.values())[0]
        assert lbl.iloc[0] == 0  # wide spread → flat
