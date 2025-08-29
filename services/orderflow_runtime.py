# services/orderflow_runtime.py
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, Tuple
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

from services.orderflow_features import OF_FEATURE_ORDER


@dataclass
class FeatureStateConfig:
    tz_exchange: str = "America/Chicago"
    spread_cap_ticks: dict = None  # e.g., {"NQ": 8, "ES": 4}
    tick_size_map: dict = None     # e.g., {"NQ": 0.25, "ES": 0.25}


class FeatureState:
    def __init__(self, cfg: FeatureStateConfig):
        self.cfg = cfg
        self._deque_ticks: Deque[Tuple[int, float, float, float, float]] = deque()  # (ts_ms, last, bid, ask, size)
        self._deque_spread: Deque[Tuple[int, float]] = deque()
        self._deque_last: Deque[Tuple[int, float]] = deque()
        self._deque_signed: Deque[float] = deque(maxlen=200)
        self._tz = ZoneInfo(cfg.tz_exchange)

    def _prune_time(self, now_ms: int) -> None:
        # keep ~2s for vol_2s and 60s for spread_z_60s; and 1s for ret_1s
        cutoff_60s = now_ms - 60_000
        cutoff_2s = now_ms - 2_000
        cutoff_1s = now_ms - 1_000
        while self._deque_spread and self._deque_spread[0][0] < cutoff_60s:
            self._deque_spread.popleft()
        while self._deque_last and self._deque_last[0][0] < cutoff_1s:
            self._deque_last.popleft()
        # ticks deque used for trade counts in 500ms window
        cutoff_500 = now_ms - 500
        while self._deque_ticks and self._deque_ticks[0][0] < cutoff_500:
            self._deque_ticks.popleft()
        # signed deque is fixed-size by N trades for cdelta windows

    def update(self, sample: dict) -> Optional[pd.DataFrame]:
        # sample: {symbol, contract, ts_utc(ms), last, bid, ask, size}
        symbol = sample.get("symbol", "NQ")
        ts_ms = int(sample["ts_utc"])
        last = float(sample["last"])
        bid = float(sample["bid"])
        ask = float(sample["ask"])
        size = float(sample.get("size", 1))

        if not np.isfinite([last, bid, ask, size]).all():
            return None
        if bid > ask:
            return None

        spread = ask - bid
        # spread cap gate
        if self.cfg.spread_cap_ticks:
            tick_size = self.cfg.tick_size_map.get(symbol, 0.25) if self.cfg.tick_size_map else 0.25
            cap_ticks = self.cfg.spread_cap_ticks.get(symbol, 8) if self.cfg.spread_cap_ticks else 8
            if spread > cap_ticks * tick_size:
                # accept sample but return warmup-like None (bad spread)
                return None

        mid = (bid + ask) / 2.0
        aggressor = 1 if last >= ask else (-1 if last <= bid else 0)
        signed_size = aggressor * size

        # append to buffers
        self._deque_ticks.append((ts_ms, last, bid, ask, size))
        self._deque_spread.append((ts_ms, spread))
        self._deque_last.append((ts_ms, last))
        self._deque_signed.append(signed_size)

        # prune old data
        self._prune_time(ts_ms)

        # Compute by-trade windows
        cdelta_50 = float(np.sum(list(self._deque_signed)[-50:])) if len(self._deque_signed) >= 1 else 0.0
        cdelta_100 = float(np.sum(list(self._deque_signed)[-100:])) if len(self._deque_signed) >= 1 else 0.0

        # Time windows
        trades_per_500ms = float(len(self._deque_ticks))

        # ret_1s: compare to price ~1s ago
        if len(self._deque_last) == 0 or self._deque_last[0][0] > ts_ms - 500:
            ret_1s = 0.0
        else:
            # first element in deque is oldest within ~1s window edge
            last_1s_ago = self._deque_last[0][1]
            ret_1s = last - last_1s_ago

        # vol_2s: std of last in 2s window using ticks deque (reuse last list within 2s)
        last_vals = [v for t, v in self._deque_last if t >= ts_ms - 2_000]
        vol_2s = float(np.std(last_vals, ddof=0)) if len(last_vals) >= 2 else 0.0

        # spread_z_60s
        sp_vals = [v for t, v in self._deque_spread]
        if len(sp_vals) >= 10:
            mu = float(np.mean(sp_vals))
            sd = float(np.std(sp_vals))
            spread_z_60s = (spread - mu) / (sd + 1e-9)
        else:
            spread_z_60s = 0.0

        # time-of-day sin/cos (exchange time)
        dt = pd.to_datetime(ts_ms, unit="ms", utc=True).tz_convert(self._tz)
        hour_float = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        sin_time = float(np.sin(2 * np.pi * hour_float / 24.0))
        cos_time = float(np.cos(2 * np.pi * hour_float / 24.0))

        # warm-up checks: require some history for robust features
        if len(self._deque_signed) < 10:
            return None

        row = {
            "Spread": spread,
            "last_minus_mid": last - mid,
            "aggressor": aggressor,
            "signed_size": signed_size,
            "cdelta_50": cdelta_50,
            "cdelta_100": cdelta_100,
            "ti_count_500ms": trades_per_500ms,
            "ret_1s": ret_1s,
            "vol_2s": vol_2s,
            "spread_z_60s": spread_z_60s,
            "sin_time": sin_time,
            "cos_time": cos_time,
        }
        # return in canonical order
        df = pd.DataFrame([[row[k] for k in OF_FEATURE_ORDER]], columns=OF_FEATURE_ORDER)
        return df

