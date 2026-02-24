"""Tests for orderflow feature engineering."""

import numpy as np
import pandas as pd
import pytest

from services.orderflow_features import (
    OF_FEATURE_ORDER,
    infer_aggressor_side,
    BUY,
    SELL,
    NONE,
    select_compact_features,
    build_orderflow_features,
)
from services.orderflow_runtime import FeatureState, FeatureStateConfig


# ── aggressor inference ──────────────────────────────────────

class TestAggressorInference:
    def test_buy_at_ask(self):
        df = pd.DataFrame({"Last": [100.25], "Bid": [100.0], "Ask": [100.25]})
        side = infer_aggressor_side(df)
        assert side.iloc[0] == BUY

    def test_sell_at_bid(self):
        df = pd.DataFrame({"Last": [100.0], "Bid": [100.0], "Ask": [100.25]})
        side = infer_aggressor_side(df)
        assert side.iloc[0] == SELL

    def test_none_between(self):
        df = pd.DataFrame({"Last": [100.125], "Bid": [100.0], "Ask": [100.25]})
        side = infer_aggressor_side(df)
        assert side.iloc[0] == NONE

    def test_vectorised(self):
        df = pd.DataFrame({
            "Last": [100.25, 100.0, 100.125],
            "Bid":  [100.0,  100.0, 100.0],
            "Ask":  [100.25, 100.25, 100.25],
        })
        sides = infer_aggressor_side(df)
        expected = [BUY, SELL, NONE]
        np.testing.assert_array_equal(sides.values, expected)


# ── OF_FEATURE_ORDER consistency ─────────────────────────────

class TestFeatureOrder:
    def test_select_compact_matches_order(self):
        """select_compact_features should return columns in OF_FEATURE_ORDER."""
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="100ms", tz="UTC")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "Last": 100 + rng.standard_normal(n).cumsum() * 0.25,
            "Bid": 100 + rng.standard_normal(n).cumsum() * 0.25 - 0.125,
            "Ask": 100 + rng.standard_normal(n).cumsum() * 0.25 + 0.125,
            "Volume": rng.integers(1, 5, size=n),
        }, index=idx)
        df["Bid"] = df[["Last", "Bid"]].min(axis=1) - 0.125
        df["Ask"] = df[["Last", "Ask"]].max(axis=1) + 0.125

        out = build_orderflow_features(df)
        compact = select_compact_features(out)
        assert list(compact.columns) == OF_FEATURE_ORDER


class TestFeatureState:
    def _make_cfg(self):
        return FeatureStateConfig(
            tz_exchange="America/Chicago",
            spread_cap_ticks={"NQ": 8},
            tick_size_map={"NQ": 0.25},
        )

    def test_warmup_returns_none(self):
        """First few ticks should return None (warmup)."""
        state = FeatureState(self._make_cfg())
        for i in range(5):
            result = state.update({
                "symbol": "NQ",
                "ts_utc": 1_000_000 + i * 100,
                "last": 20000.0,
                "bid": 19999.75,
                "ask": 20000.25,
                "size": 1,
            })
            assert result is None

    def test_output_column_order(self):
        """After warmup, output columns must match OF_FEATURE_ORDER."""
        state = FeatureState(self._make_cfg())
        result = None
        for i in range(20):
            result = state.update({
                "symbol": "NQ",
                "ts_utc": 1_000_000 + i * 100,
                "last": 20000.0 + i * 0.25,
                "bid": 19999.75 + i * 0.25,
                "ask": 20000.25 + i * 0.25,
                "size": 1,
            })
        assert result is not None
        assert list(result.columns) == OF_FEATURE_ORDER
